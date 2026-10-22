"""Ordered native-output binding and one-forward batched Control6 policy.

The training adapter and single-row production policy are intentionally left
unchanged.  This wrapper accepts only real fresh listener rows, performs one
``NativeControl6LoRA`` forward, masks each row with its own legal actions, and
reads the compact selection metadata to the CPU once per batch. The existing
adapter's finite-input guard remains unchanged and may itself synchronize a
CUDA scalar; the one-transfer claim applies only to policy selection metadata.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput
from duplex_kernel.native_control6_phase import NATIVE_CONTROL6_PHASE_T_WIDTH
from duplex_kernel.microbatch_contract import BatchRowDescriptor, ListenerBatchManifest
from duplex_kernel.native_listener_microbatch import (
    LISTENER_STREAM_Q_LEN,
    NativeBatchOutput,
)
from training.duplex_qlora.native_control6 import (
    ACTIONS,
    ACTION_TO_TOKEN_ID,
    HIDDEN_WIDTH,
    NativeControl6LoRA,
    TOKEN_ID_TO_STRING,
)


@dataclass(frozen=True, slots=True)
class ListenerHiddenProvenance:
    """Immutable identity captured with a native listener hidden row."""

    session_id: str
    generation_id: int
    state_epoch_id: str
    state_version: int
    frame_id: int


@dataclass(frozen=True, slots=True)
class NativeControl6RawScoreSnapshot:
    """Immutable raw six-row logits bound to one native listener output row."""

    manifest_id: str
    output_generation: int
    session_id: str
    session_generation: int
    state_epoch_id: str
    listener_state_version: int
    frame_id: int
    sequence_position: int
    cache_position: int
    query_index: int
    checkpoint_path: str
    raw_logits: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        for name in ("manifest_id", "session_id", "state_epoch_id", "checkpoint_path"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"raw Control6 snapshot requires {name}")
        for name in (
            "output_generation", "session_generation", "listener_state_version",
            "frame_id", "sequence_position", "cache_position", "query_index",
        ):
            value = getattr(self, name)
            minimum = 1 if name == "output_generation" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"raw Control6 snapshot {name} is invalid")
        if self.query_index != 0:
            raise ValueError("native Control6 scores must use listener query position zero")
        if not isinstance(self.raw_logits, tuple) or len(self.raw_logits) != len(ACTIONS):
            raise ValueError("raw Control6 snapshot must contain exactly six logits")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.raw_logits
        ):
            raise ValueError("raw Control6 snapshot logits must be finite numbers")
        object.__setattr__(self, "raw_logits", tuple(float(value) for value in self.raw_logits))


@dataclass(frozen=True, slots=True)
class BatchPolicyInput:
    """Batch-only wrapper binding policy state to its hidden-state watermark."""

    state: PolicyInput
    state_epoch_id: str
    listener_state_version: int
    hidden_provenance: ListenerHiddenProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.state, PolicyInput):
            raise TypeError("batch policy input requires a PolicyInput state")
        if not isinstance(self.hidden_provenance, ListenerHiddenProvenance):
            raise TypeError("batch policy input requires hidden provenance")
        if not isinstance(self.state_epoch_id, str) or not self.state_epoch_id.strip():
            raise ValueError("batch policy input requires a state epoch")
        if not isinstance(self.hidden_provenance.session_id, str) or not self.hidden_provenance.session_id.strip():
            raise ValueError("hidden provenance requires a session id")
        if not isinstance(self.hidden_provenance.state_epoch_id, str) or not self.hidden_provenance.state_epoch_id.strip():
            raise ValueError("hidden provenance requires a state epoch")
        for name, value in (
            ("listener_state_version", self.listener_state_version),
            ("generation_id", self.hidden_provenance.generation_id),
            ("state_version", self.hidden_provenance.state_version),
            ("frame_id", self.hidden_provenance.frame_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class DecisionRowSource:
    """One actual native output row selected for one listener decision.

    ``decision_row`` is the canonical listener row that owns the policy
    decision. ``manifest.rows[row_index]`` remains the actual native source,
    which may be a temporary Q2 branch at the same frame boundary.
    """

    native_output: NativeBatchOutput
    manifest: ListenerBatchManifest
    row_index: int
    decision_row: BatchRowDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.native_output, NativeBatchOutput):
            raise TypeError("decision source requires a real NativeBatchOutput")
        if not isinstance(self.manifest, ListenerBatchManifest):
            raise TypeError("decision source requires its native output manifest")
        if isinstance(self.row_index, bool) or not isinstance(self.row_index, int) or self.row_index < 0:
            raise ValueError("decision source row_index must be a non-negative integer")
        if not isinstance(self.decision_row, BatchRowDescriptor):
            raise TypeError("decision source requires its canonical listener row")


def _snapshot_policy_value(value: Any) -> Any:
    """Copy mutable pre-head context while keeping snapshots CPU/GPU local."""
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("policy context cannot snapshot object-dtype arrays")
        copied = np.array(value, copy=True, order="C")
        # Back with immutable bytes so callers cannot re-enable writes on the
        # captured array via setflags(write=True).
        return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(copied.shape)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return MappingProxyType({
            _snapshot_policy_value(key): _snapshot_policy_value(item)
            for key, item in value.items()
        })
    if isinstance(value, tuple):
        return tuple(_snapshot_policy_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_snapshot_policy_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_snapshot_policy_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class PolicyPreHeadContext:
    """Immutable scheduler snapshot paired with its session-owned policy state."""

    policy: Any
    policy_input: PolicyInput
    tick_id: tuple[str, int, int]
    captured_at_ns: int
    listener_generation_id: int

    def __post_init__(self) -> None:
        if not callable(getattr(self.policy, "step_with_logits", None)):
            raise TypeError("policy context requires a step_with_logits session policy")
        if not isinstance(self.policy_input, PolicyInput):
            raise TypeError("policy context requires a pre-head PolicyInput snapshot")
        state = self.policy_input
        if not isinstance(self.tick_id, tuple) or len(self.tick_id) != 3:
            raise ValueError("policy tick ID must be (session_id, generation_id, frame_id)")
        sid, generation, frame_id = self.tick_id
        if (
            not isinstance(sid, str)
            or not sid
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or generation < 0
            or frame_id < 0
            or (sid, generation, frame_id)
            != (state.session_id, state.generation_id, state.frame_id)
        ):
            raise ValueError("policy tick ID must identify session, turn generation, and frame")
        if (
            isinstance(self.listener_generation_id, bool)
            or not isinstance(self.listener_generation_id, int)
            or self.listener_generation_id < 0
        ):
            raise ValueError("listener_generation_id must be a non-negative integer")
        if state.native_listener_generation not in (None, self.listener_generation_id):
            raise ValueError("policy pre-head state has a different listener generation")
        if (
            state.native_control_hidden is not None
            or state.native_control_hidden_frame_id is not None
            or state.native_control_hidden_at_ns is not None
        ):
            raise ValueError("pre-head policy context cannot contain a listener hidden")
        if isinstance(self.captured_at_ns, bool) or not isinstance(self.captured_at_ns, int) or self.captured_at_ns <= 0:
            raise ValueError("policy context capture clock must be a positive monotonic timestamp")
        captured = {
            item.name: _snapshot_policy_value(getattr(state, item.name))
            for item in fields(state)
        }
        captured["legal_actions"] = tuple(state.legal_actions or ())
        captured["native_listener_generation"] = self.listener_generation_id
        frozen_input = replace(state, **captured)
        object.__setattr__(self, "policy_input", frozen_input)
        object.__setattr__(self, "tick_id", (sid, generation, frame_id))


class GemmaNativeControl6BatchPolicy:
    """Evaluate 1, 2, or 4 independent fresh listener states in one forward."""

    policy_id = "gemma_native_control6_batch"

    def __init__(
        self,
        adapter: NativeControl6LoRA,
        *,
        checkpoint_path: str,
        capture_debug_inputs: bool = False,
    ) -> None:
        if not isinstance(adapter, NativeControl6LoRA):
            raise TypeError("batched native Control6 requires NativeControl6LoRA")
        if not checkpoint_path:
            raise ValueError("batched native Control6 requires a loaded adapter checkpoint")
        if type(capture_debug_inputs) is not bool:
            raise TypeError("capture_debug_inputs must be a bool")
        self.adapter = adapter.eval()
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_path = str(checkpoint_path)
        self.capture_debug_inputs = capture_debug_inputs
        self.forward_call_count = 0

    def score_raw_batch(
        self,
        native_output: NativeBatchOutput,
        manifest: ListenerBatchManifest,
    ) -> tuple[NativeControl6RawScoreSnapshot, ...]:
        """Score native query-zero rows once without making policy decisions.

        The returned immutable snapshots preserve raw FP32 logits and complete
        row provenance. The original per-session policy remains responsible
        for legality, confidence, traces, and action mapping.
        """
        if not isinstance(native_output, NativeBatchOutput):
            raise TypeError("raw Control6 scoring requires NativeBatchOutput")
        if not isinstance(manifest, ListenerBatchManifest):
            raise TypeError("raw Control6 scoring requires ListenerBatchManifest")
        rows = manifest.rows
        if not all(isinstance(row, BatchRowDescriptor) for row in rows):
            raise TypeError("manifest must contain complete BatchRowDescriptor rows")
        if native_output.manifest_id != manifest.manifest_id:
            raise RuntimeError("native output belongs to a different manifest")
        if not isinstance(native_output.rows, tuple) or native_output.rows != rows:
            raise RuntimeError("native output ordered manifest rows do not match")
        if (
            isinstance(native_output.generation, bool)
            or not isinstance(native_output.generation, int)
            or native_output.generation <= 0
        ):
            raise RuntimeError("native output lease generation is invalid")
        expected_shape = (manifest.batch_size, LISTENER_STREAM_Q_LEN, HIDDEN_WIDTH)
        tensor = native_output.tensor
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(f"native output tensor must have exact shape {expected_shape}")
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError("native output tensor must retain BF16 listener dtype")
        parameter = self.adapter.lora_A
        if tensor.device != parameter.device:
            raise RuntimeError("native output and Control6 adapter must use the same device")
        if tensor.is_cuda and not isinstance(native_output.completion_event, torch.cuda.Event):
            raise RuntimeError("CUDA native output must carry its completion event")
        if tensor.is_cuda:
            native_output.completion_event.wait(torch.cuda.current_stream(device=tensor.device))

        model_hidden = tensor[:, 0, :].to(
            device=parameter.device, dtype=self.adapter.base_rows.dtype,
        )
        with torch.inference_mode():
            logits = self.adapter(model_hidden).float()
            self.forward_call_count += 1
        if tuple(logits.shape) != (manifest.batch_size, len(ACTIONS)):
            raise RuntimeError("batched native Control6 produced invalid six-way logits")
        host_logits = logits.detach().cpu().tolist()
        if not isinstance(host_logits, list) or len(host_logits) != manifest.batch_size:
            raise RuntimeError("batched native Control6 returned an invalid score packet")
        if any(
            not isinstance(values, list)
            or len(values) != len(ACTIONS)
            or any(not math.isfinite(float(value)) for value in values)
            for values in host_logits
        ):
            raise RuntimeError("batched native Control6 produced invalid six-way logits")
        return tuple(
            NativeControl6RawScoreSnapshot(
                manifest_id=manifest.manifest_id,
                output_generation=native_output.generation,
                session_id=row.session_id,
                session_generation=row.session_generation,
                state_epoch_id=row.state_epoch_id,
                listener_state_version=row.listener_state_version,
                frame_id=row.frame_id,
                sequence_position=row.sequence_position,
                cache_position=row.cache_position,
                query_index=0,
                checkpoint_path=self.checkpoint_path,
                raw_logits=tuple(float(value) for value in values),  # type: ignore[arg-type]
            )
            for row, values in zip(rows, host_logits, strict=True)
        )

    def step_native_output_batch(
        self,
        native_output: NativeBatchOutput,
        manifest: ListenerBatchManifest,
        policy_states: Sequence[PolicyInput],
    ) -> tuple[PolicyDecision, ...]:
        """Bind real native output rows to ordered per-session policy states.

        This is the production entry point. Callers provide policy metadata but
        never a hidden tensor: each hidden vector comes only from row ``i`` of
        the native-owned cloned output for that exact manifest. The selected
        query is index 0 to preserve ``GemmaAudioTowerBackend``'s current
        ``hidden[:, 0, :]`` policy input.
        """
        if not isinstance(native_output, NativeBatchOutput):
            raise TypeError("production batched Control6 requires NativeBatchOutput")
        if not isinstance(manifest, ListenerBatchManifest):
            raise TypeError("production batched Control6 requires ListenerBatchManifest")
        rows = manifest.rows
        if not all(isinstance(row, BatchRowDescriptor) for row in rows):
            raise TypeError("manifest must contain complete BatchRowDescriptor rows")
        if native_output.manifest_id != manifest.manifest_id:
            raise RuntimeError("native output belongs to a different manifest")
        if not isinstance(native_output.rows, tuple) or native_output.rows != rows:
            raise RuntimeError("native output ordered manifest rows do not match")
        if (
            isinstance(native_output.generation, bool)
            or not isinstance(native_output.generation, int)
            or native_output.generation <= 0
        ):
            raise RuntimeError("native output lease generation is invalid")
        expected_shape = (manifest.batch_size, LISTENER_STREAM_Q_LEN, HIDDEN_WIDTH)
        tensor = native_output.tensor
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
            raise RuntimeError(
                f"native output tensor must have exact shape {expected_shape}"
            )
        if tensor.dtype != torch.bfloat16:
            raise RuntimeError("native output tensor must retain BF16 listener dtype")
        if tensor.device != self.adapter.lora_A.device:
            raise RuntimeError("native output and Control6 adapter must use the same device")
        if len(policy_states) != manifest.batch_size:
            raise ValueError("policy inputs must have one ordered row per manifest row")
        if tensor.is_cuda and not isinstance(native_output.completion_event, torch.cuda.Event):
            raise RuntimeError("CUDA native output must carry its completion event")

        batch_inputs: list[BatchPolicyInput] = []
        for index, (row, state) in enumerate(zip(rows, policy_states, strict=True)):
            if not isinstance(state, PolicyInput):
                raise TypeError("production batched Control6 requires PolicyInput rows")
            if (
                state.session_id != row.session_id
                or state.generation_id != row.session_generation
                or state.frame_id != row.frame_id
            ):
                raise RuntimeError(
                    f"policy input order/provenance does not match manifest row {index}"
                )
            if (
                state.native_control_hidden is not None
                or state.native_control_hidden_frame_id is not None
                or state.native_control_hidden_at_ns is not None
            ):
                raise ValueError("production policy inputs must not provide native hidden")
            hidden = tensor[index, 0, :]
            state_with_native_hidden = replace(
                state,
                native_control_hidden=hidden,
                native_control_hidden_frame_id=row.frame_id,
            )
            provenance = ListenerHiddenProvenance(
                session_id=row.session_id,
                generation_id=row.session_generation,
                state_epoch_id=row.state_epoch_id,
                state_version=row.listener_state_version,
                frame_id=row.frame_id,
            )
            batch_inputs.append(BatchPolicyInput(
                state=state_with_native_hidden,
                state_epoch_id=row.state_epoch_id,
                listener_state_version=row.listener_state_version,
                hidden_provenance=provenance,
            ))
        if tensor.is_cuda:
            # The binding returns an owned clone plus its stream completion
            # event. Queue the dependency on this consumer stream without
            # synchronizing the host or device.
            consumer_stream = torch.cuda.current_stream(device=tensor.device)
            native_output.completion_event.wait(consumer_stream)
        return self.step_batch(tuple(batch_inputs))

    def step_native_output_views(
        self,
        output_views: Sequence[DecisionRowSource],
        policy_states: Sequence[PolicyInput | PolicyPreHeadContext],
    ) -> tuple[PolicyDecision, ...]:
        """Evaluate one policy batch from individually proven native outputs.

        A speaking row may use a short-lived Q2 decision view while other rows
        use the canonical batch output. Every hidden still comes from its own
        actual ``NativeBatchOutput`` and manifest. The method performs one
        adapter forward and returns decisions keyed to canonical sessions;
        branch session IDs and output generations remain in per-row
        ``native_output_provenance`` diagnostics.
        """
        sources = tuple(output_views)
        states = tuple(policy_states)
        if not sources or len(sources) != len(states):
            raise ValueError("native output views and policy states need the same non-empty row count")
        if any(isinstance(state, PolicyPreHeadContext) for state in states):
            if not all(isinstance(state, PolicyPreHeadContext) for state in states):
                raise TypeError("mixed-source Control6 cannot mix context and fixture policy rows")
            return self._step_policy_context_views(
                sources,
                tuple(state for state in states if isinstance(state, PolicyPreHeadContext)),
            )
        if len({state.session_id for state in states}) != len(states):
            raise ValueError("mixed-source Control6 requires distinct canonical sessions")

        consumer_stream = None
        batch_inputs: list[BatchPolicyInput] = []
        canonical_rows: list[BatchRowDescriptor] = []
        native_rows: list[BatchRowDescriptor] = []
        for index, (source, state) in enumerate(zip(sources, states, strict=True)):
            if not isinstance(source, DecisionRowSource):
                raise TypeError("mixed-source Control6 requires DecisionRowSource values")
            if not isinstance(state, PolicyInput):
                raise TypeError("mixed-source Control6 requires PolicyInput rows")
            output = source.native_output
            manifest = source.manifest
            if (
                output.manifest_id != manifest.manifest_id
                or not isinstance(output.rows, tuple)
                or output.rows != manifest.rows
            ):
                raise RuntimeError(f"native output view {index} lost its exact manifest provenance")
            if (
                isinstance(output.generation, bool)
                or not isinstance(output.generation, int)
                or output.generation <= 0
            ):
                raise RuntimeError(f"native output view {index} has an invalid generation")
            tensor = output.tensor
            expected_shape = (manifest.batch_size, LISTENER_STREAM_Q_LEN, HIDDEN_WIDTH)
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected_shape:
                raise RuntimeError(f"native output view {index} must have shape {expected_shape}")
            if tensor.dtype != torch.bfloat16 or tensor.device != self.adapter.lora_A.device:
                raise RuntimeError(f"native output view {index} dtype/device differs from Control6")
            if source.row_index >= manifest.batch_size:
                raise IndexError(f"native output view {index} row index is outside its manifest")
            native_row = manifest.rows[source.row_index]
            decision_row = source.decision_row
            if (
                state.session_id != decision_row.session_id
                or state.generation_id != decision_row.session_generation
                or state.frame_id != decision_row.frame_id
            ):
                raise RuntimeError(f"policy state {index} differs from its canonical listener row")
            if (
                native_row.session_generation != decision_row.session_generation
                or native_row.state_epoch_id != decision_row.state_epoch_id
                or native_row.frame_id != decision_row.frame_id
                or native_row.listener_state_version != decision_row.listener_state_version
                or native_row.sequence_position != decision_row.sequence_position
            ):
                raise RuntimeError(f"native output view {index} differs from its decision frame boundary")
            if native_row.session_id == decision_row.session_id:
                if native_row != decision_row:
                    raise RuntimeError(f"canonical output view {index} changed its source row")
            elif (
                not native_row.session_id.startswith("__q2view__")
                or native_row.logical_q_len != LISTENER_STREAM_Q_LEN
                or decision_row.logical_q_len != 0
                or not decision_row.queue_audio_for_next_turn
            ):
                raise RuntimeError(f"native output view {index} is not a valid speaking Q2 branch")
            if (
                state.native_control_hidden is not None
                or state.native_control_hidden_frame_id is not None
                or state.native_control_hidden_at_ns is not None
            ):
                raise ValueError("mixed-source production policy inputs must not provide native hidden")
            if tensor.is_cuda:
                if not isinstance(output.completion_event, torch.cuda.Event):
                    raise RuntimeError(f"CUDA output view {index} lacks its completion event")
                if consumer_stream is None:
                    consumer_stream = torch.cuda.current_stream(device=tensor.device)
                output.completion_event.wait(consumer_stream)

            hidden = tensor[source.row_index, 0, :]
            branch_policy_state = replace(
                state,
                session_id=native_row.session_id,
                native_control_hidden=hidden,
                native_control_hidden_frame_id=native_row.frame_id,
            )
            provenance = ListenerHiddenProvenance(
                session_id=native_row.session_id,
                generation_id=native_row.session_generation,
                state_epoch_id=native_row.state_epoch_id,
                state_version=native_row.listener_state_version,
                frame_id=native_row.frame_id,
            )
            batch_inputs.append(BatchPolicyInput(
                state=branch_policy_state,
                state_epoch_id=native_row.state_epoch_id,
                listener_state_version=native_row.listener_state_version,
                hidden_provenance=provenance,
            ))
            canonical_rows.append(decision_row)
            native_rows.append(native_row)

        # Exactly one call owns the adapter/head forward and all per-row legal
        # selection metadata, so no speaking row advances a second policy path.
        raw_decisions = self.step_batch(tuple(batch_inputs))
        if len(raw_decisions) != len(sources):
            raise RuntimeError("mixed-source Control6 returned an incomplete row set")
        decisions: list[PolicyDecision] = []
        for source, state, canonical_row, native_row, decision in zip(
            sources, states, canonical_rows, native_rows, raw_decisions, strict=True
        ):
            if (
                decision.session != native_row.session_id
                or decision.generation_id != native_row.session_generation
                or decision.frame_id != native_row.frame_id
                or decision.listener_frame_id != native_row.frame_id
            ):
                raise RuntimeError("mixed-source Control6 decision lost its native row provenance")
            diagnostics = dict(decision.diagnostics or {})
            diagnostics["native_output_provenance"] = {
                "manifest_id": source.manifest.manifest_id,
                "output_generation": source.native_output.generation,
                "native_session_id": native_row.session_id,
                "canonical_session_id": canonical_row.session_id,
                "session_generation": native_row.session_generation,
                "state_epoch_id": native_row.state_epoch_id,
                "listener_state_version": native_row.listener_state_version,
                "frame_id": native_row.frame_id,
                "sequence_position": native_row.sequence_position,
                "cache_position": native_row.cache_position,
            }
            decisions.append(replace(
                decision,
                session=state.session_id,
                diagnostics=diagnostics,
            ))
        return tuple(decisions)

    def _step_policy_context_views(
        self,
        sources: tuple[DecisionRowSource, ...],
        contexts: tuple[PolicyPreHeadContext, ...],
    ) -> tuple[PolicyDecision, ...]:
        """Score actual selected hidden rows once, then apply per-session policy state."""
        if len({context.tick_id[0] for context in contexts}) != len(contexts):
            raise ValueError("mixed-source policy contexts need distinct session IDs")
        if len({id(context.policy) for context in contexts}) != len(contexts):
            raise ValueError("each row needs its own session policy and hysteresis state")

        consumer_stream = None
        native_rows: list[BatchRowDescriptor] = []
        hidden_rows: list[torch.Tensor | None] = []
        phase_rows: list[torch.Tensor] = []
        score_indices: list[int] = []
        for index, (source, context) in enumerate(zip(sources, contexts, strict=True)):
            if not isinstance(source, DecisionRowSource):
                raise TypeError("mixed-source Control6 requires DecisionRowSource values")
            if not isinstance(context, PolicyPreHeadContext):
                raise TypeError("mixed-source Control6 requires PolicyPreHeadContext rows")
            state = context.policy_input
            if getattr(context.policy, "adapter", None) is not self.adapter:
                raise RuntimeError("per-session policy does not share the worker's loaded Control6 adapter")
            if (
                context.tick_id != (state.session_id, state.generation_id, state.frame_id)
                or context.tick_id[0] != source.decision_row.session_id
                or state.frame_id != source.decision_row.frame_id
                or state.session_id != source.decision_row.session_id
                or context.listener_generation_id != source.decision_row.session_generation
                or state.native_listener_generation != context.listener_generation_id
            ):
                raise RuntimeError(f"policy pre-head context {index} differs from its canonical listener row")
            if (
                output := source.native_output
            ).manifest_id != source.manifest.manifest_id or output.rows != source.manifest.rows:
                raise RuntimeError(f"native output view {index} lost its exact manifest provenance")
            if not 0 <= source.row_index < source.manifest.batch_size:
                raise IndexError(f"native output view {index} row index is outside its manifest")
            native_row = source.manifest.rows[source.row_index]
            decision_row = source.decision_row
            if (
                native_row.session_generation != decision_row.session_generation
                or native_row.state_epoch_id != decision_row.state_epoch_id
                or native_row.frame_id != decision_row.frame_id
                or native_row.listener_state_version != decision_row.listener_state_version
                or native_row.sequence_position != decision_row.sequence_position
            ):
                raise RuntimeError(f"native output view {index} differs from its decision frame boundary")
            if native_row.session_id == decision_row.session_id:
                if native_row != decision_row:
                    raise RuntimeError(f"canonical output view {index} changed its source row")
            elif (
                not native_row.session_id.startswith("__q2view__")
                or native_row.logical_q_len != LISTENER_STREAM_Q_LEN
                or decision_row.logical_q_len != 0
                or not decision_row.queue_audio_for_next_turn
            ):
                raise RuntimeError(f"native output view {index} is not a valid speaking Q2 branch")
            tensor = output.tensor
            expected_shape = (source.manifest.batch_size, LISTENER_STREAM_Q_LEN, HIDDEN_WIDTH)
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != expected_shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != self.adapter.lora_A.device
            ):
                raise RuntimeError(f"native output view {index} must match {expected_shape} BF16 on adapter device")
            if tensor.is_cuda:
                if not isinstance(output.completion_event, torch.cuda.Event):
                    raise RuntimeError(f"CUDA output view {index} lacks its completion event")
                if consumer_stream is None:
                    consumer_stream = torch.cuda.current_stream(device=tensor.device)
                output.completion_event.wait(consumer_stream)
            native_rows.append(native_row)
            if state.native_control_no_input_tick:
                if state.native_control_hidden is not None or not state.legal_actions:
                    raise RuntimeError("exact-zero no-input context must omit hidden and keep legal actions")
                hidden_rows.append(None)
                continue
            if state.native_control_hidden is not None:
                raise RuntimeError("pre-head policy context contains a hidden from outside this worker")
            hidden = tensor[source.row_index, 0, :]
            hidden_rows.append(hidden)
            phase = state.native_control_phase_t
            phase_tensor = (
                torch.zeros((NATIVE_CONTROL6_PHASE_T_WIDTH,), dtype=torch.float32)
                if phase is None
                else torch.as_tensor(phase, dtype=torch.float32).reshape(-1)
            )
            if tuple(phase_tensor.shape) != (NATIVE_CONTROL6_PHASE_T_WIDTH,) or not bool(
                torch.isfinite(phase_tensor).all()
            ):
                raise RuntimeError("policy pre-head context requires four finite phase_t values")
            if not state.legal_actions:
                raise RuntimeError("policy pre-head context requires a non-empty legal action mask")
            unknown = set(state.legal_actions) - set(ACTIONS)
            if unknown:
                raise ValueError(f"policy pre-head context has unknown legal actions: {sorted(unknown)}")
            phase_rows.append(phase_tensor)
            score_indices.append(index)

        logits_by_index: dict[int, torch.Tensor] = {}
        debug_inputs_by_index: dict[int, dict[str, Any]] = {}
        if score_indices:
            parameter = self.adapter.lora_A
            model_hidden = torch.stack([
                hidden_rows[index].to(
                    device=parameter.device,
                    dtype=self.adapter.base_rows.dtype,
                )
                for index in score_indices
                if hidden_rows[index] is not None
            ])
            model_phase = torch.stack(phase_rows).to(
                device=parameter.device,
                dtype=torch.float32,
            )
            with torch.inference_mode():
                logits = self.adapter(model_hidden, phase_t=model_phase).float()
            self.forward_call_count += 1
            if tuple(logits.shape) != (len(score_indices), len(ACTIONS)):
                raise RuntimeError("mixed-source Control6 produced invalid batched logits")
            for batch_index, row_index in enumerate(score_indices):
                logits_by_index[row_index] = logits[batch_index]
                if self.capture_debug_inputs:
                    context = contexts[row_index]
                    debug_inputs_by_index[row_index] = {
                        "hidden": model_hidden[batch_index].detach().to(
                            device="cpu", dtype=torch.float32,
                        ).numpy().copy(),
                        "phase_t": model_phase[batch_index].detach().to(
                            device="cpu", dtype=torch.float32,
                        ).numpy().copy(),
                        "frame_id": int(native_rows[row_index].frame_id),
                        "generation_id": int(context.tick_id[1]),
                        "phase_profile": str(context.policy.phase_profile),
                    }

        available_at_ns = time.monotonic_ns()
        decisions: list[PolicyDecision] = []
        for index, (source, context, native_row) in enumerate(
            zip(sources, contexts, native_rows, strict=True)
        ):
            state = context.policy_input
            hidden = hidden_rows[index]
            completed_state = replace(
                state,
                native_control_hidden=(
                    None if hidden is None else hidden.reshape(1, 1, -1)
                ),
                native_control_hidden_frame_id=(
                    None if hidden is None else native_row.frame_id
                ),
                native_control_hidden_at_ns=(
                    None if hidden is None else available_at_ns
                ),
                native_listener_generation=native_row.session_generation,
            )
            decision = context.policy.step_with_logits(
                completed_state,
                logits_by_index.get(index),
                tick_id=context.tick_id,
            )
            diagnostics = dict(decision.diagnostics or {})
            diagnostics["native_output_provenance"] = {
                "manifest_id": source.manifest.manifest_id,
                "output_generation": source.native_output.generation,
                "native_session_id": native_row.session_id,
                "canonical_session_id": source.decision_row.session_id,
                "listener_generation_id": native_row.session_generation,
                "turn_generation_id": context.tick_id[1],
                "state_epoch_id": native_row.state_epoch_id,
                "listener_state_version": native_row.listener_state_version,
                "frame_id": native_row.frame_id,
                "sequence_position": native_row.sequence_position,
                "cache_position": native_row.cache_position,
                "policy_tick_id": list(context.tick_id),
                "policy_context_captured_at_ns": context.captured_at_ns,
                "used_for_policy": hidden is not None,
            }
            if index in debug_inputs_by_index:
                # Private binary-capture payload, removed by the server before
                # publishing the ordinary decision trace.
                diagnostics["_native_control6_debug_input"] = debug_inputs_by_index[index]
            decisions.append(replace(decision, diagnostics=diagnostics))
        return tuple(decisions)

    def step_batch(self, states: Sequence[BatchPolicyInput]) -> tuple[PolicyDecision, ...]:
        """Run the low-level policy on explicitly bound fixtures.

        Production callers must use ``step_native_output_batch`` so the hidden
        tensor is sliced from the native output that owns the ordered manifest.
        This helper remains available for fixed-tensor CPU parity tests.
        """
        batch_rows = tuple(states)
        if any(not isinstance(row, BatchPolicyInput) for row in batch_rows):
            raise TypeError("batched Control6 requires provenance-bound BatchPolicyInput rows")
        rows = tuple(row.state for row in batch_rows)
        if not rows:
            raise ValueError("Control6 requires at least one real conversation row")
        if len({state.session_id for state in rows}) != len(rows):
            raise ValueError("Control6 batch requires one real row per distinct session")

        now_ns = time.monotonic_ns()
        hidden_rows: list[torch.Tensor] = []
        phase_rows: list[torch.Tensor] = []
        legal_sets: list[tuple[str, ...]] = []
        for batch_row, state in zip(batch_rows, rows, strict=True):
            provenance = batch_row.hidden_provenance
            if (
                provenance.session_id != state.session_id
                or provenance.generation_id != state.generation_id
                or provenance.state_epoch_id != batch_row.state_epoch_id
                or provenance.state_version != batch_row.listener_state_version
                or provenance.frame_id != state.frame_id
            ):
                raise RuntimeError("batched native Control6 hidden provenance does not match row state")
            hidden = state.native_control_hidden
            if state.native_control_no_input_tick:
                raise RuntimeError("no-input safety ticks are not synthetic Control6 batch rows")
            if hidden is None:
                raise RuntimeError("batched native Control6 requires fresh listener hidden state")
            source_frame = state.native_control_hidden_frame_id
            if isinstance(source_frame, bool) or not isinstance(source_frame, int):
                raise RuntimeError("batched native Control6 requires a listener hidden frame watermark")
            if source_frame != int(state.frame_id):
                raise RuntimeError(
                    "batched native Control6 refuses stale listener hidden: "
                    f"source_frame={source_frame} policy_frame={state.frame_id}"
                )
            if source_frame != provenance.frame_id:
                raise RuntimeError("batched native Control6 hidden provenance frame is stale")
            available_at_ns = state.native_control_hidden_at_ns
            if available_at_ns is not None and (
                isinstance(available_at_ns, bool)
                or not isinstance(available_at_ns, int)
                or available_at_ns <= 0
            ):
                raise RuntimeError("batched native Control6 hidden availability watermark is invalid")
            if available_at_ns is not None and available_at_ns > now_ns:
                raise RuntimeError("batched native Control6 hidden watermark is in the future")
            if not isinstance(hidden, torch.Tensor) or hidden.shape[-1:] != (HIDDEN_WIDTH,):
                raise RuntimeError(
                    f"batched native Control6 requires hidden width {HIDDEN_WIDTH}"
                )
            if hidden.numel() != HIDDEN_WIDTH:
                raise RuntimeError(
                    "batched native Control6 requires exactly one hidden vector per session"
                )
            legal = tuple(state.legal_actions or ())
            unknown = set(legal) - set(ACTIONS)
            if unknown:
                raise ValueError(f"batched native Control6 received unknown legal actions: {sorted(unknown)}")
            if not legal:
                raise RuntimeError("batched native Control6 requires a non-empty legal-action mask")
            hidden_rows.append(hidden.reshape(1, HIDDEN_WIDTH))
            phase = state.native_control_phase_t
            if phase is None:
                phase_tensor = torch.zeros(
                    (NATIVE_CONTROL6_PHASE_T_WIDTH,), dtype=torch.float32
                )
            else:
                phase_tensor = torch.as_tensor(phase, dtype=torch.float32).reshape(-1)
                if tuple(phase_tensor.shape) != (NATIVE_CONTROL6_PHASE_T_WIDTH,) or not bool(
                    torch.isfinite(phase_tensor).all()
                ):
                    raise RuntimeError("batched native Control6 requires four finite phase_t values")
            phase_rows.append(phase_tensor)
            legal_sets.append(legal)

        parameter = self.adapter.lora_A
        model_hidden = torch.cat([
            hidden.to(device=parameter.device, dtype=self.adapter.base_rows.dtype)
            for hidden in hidden_rows
        ], dim=0)
        model_phase_t = torch.stack(phase_rows).to(
            device=parameter.device, dtype=torch.float32
        )
        legal_mask = torch.tensor(
            [[action in legal for action in ACTIONS] for legal in legal_sets],
            dtype=torch.bool,
            device=parameter.device,
        )

        with torch.inference_mode():
            # Keep the adapter's existing finite-input guard intact. On CUDA
            # that guard may synchronize independently of the packed policy
            # metadata transfer below; this wrapper does not bypass it.
            logits = self.adapter(model_hidden, phase_t=model_phase_t).float()
            self.forward_call_count += 1
            if tuple(logits.shape) != (len(rows), len(ACTIONS)):
                raise RuntimeError("batched native Control6 produced invalid six-way logits")
            # Preserve the current policy's fail-closed logits check without a
            # separate host synchronization: the finite bit travels with the
            # same compact row-selection packet below.
            finite_flag = torch.isfinite(logits).all().to(dtype=torch.float32)
            raw_indices = logits.argmax(dim=1)
            selected_logits = logits.masked_fill(~legal_mask, -torch.inf)
            selected_indices = selected_logits.argmax(dim=1)
            legal_probabilities = torch.softmax(selected_logits, dim=1)
            confidence = legal_probabilities.gather(
                1, selected_indices.unsqueeze(1)
            ).squeeze(1)
            packed = torch.stack((
                finite_flag.expand(len(rows)),
                raw_indices.to(dtype=torch.float32),
                selected_indices.to(dtype=torch.float32),
                confidence.to(dtype=torch.float32),
            ), dim=1)
            # One batch-sized policy metadata transfer. No row performs its
            # own Tensor.item() or .cpu(); this does not count the adapter's
            # separate finite-input guard.
            packet = packed.detach().cpu().tolist()

        if not isinstance(packet, list) or len(packet) != len(rows):
            raise RuntimeError("batched native Control6 returned an invalid metadata packet")
        decisions: list[PolicyDecision] = []
        for state, legal, values in zip(rows, legal_sets, packet, strict=True):
            if not isinstance(values, list) or len(values) != 4:
                raise RuntimeError("batched native Control6 returned an invalid row packet")
            try:
                finite_value, raw_value, selected_value, probability = (
                    float(value) for value in values
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("batched native Control6 returned invalid row metadata") from exc
            if finite_value != 1.0:
                raise RuntimeError("batched native Control6 produced invalid six-way logits")
            if (
                not all(math.isfinite(value) for value in (raw_value, selected_value, probability))
                or raw_value != int(raw_value)
                or selected_value != int(selected_value)
                or not 0 <= int(raw_value) < len(ACTIONS)
                or not 0 <= int(selected_value) < len(ACTIONS)
                or not 0.0 <= probability <= 1.0
            ):
                raise RuntimeError("batched native Control6 returned invalid row metadata")

            raw_index = int(raw_value)
            selected_index = int(selected_value)
            selected_action = ControlAction(ACTIONS[selected_index])
            raw_action = ACTIONS[raw_index]
            raw_token_id = ACTION_TO_TOKEN_ID[raw_action]
            selected_token_id = ACTION_TO_TOKEN_ID[selected_action.value]
            source_frame = int(state.native_control_hidden_frame_id)  # validated above
            available_at = state.native_control_hidden_at_ns
            diagnostics = {
                "policy_id": self.policy_id,
                "checkpoint_path": self.checkpoint_path,
                "base_action": raw_action,
                "native_control6_raw_row_index": raw_index,
                "native_control6_raw_token_id": raw_token_id,
                "native_control6_raw_token": TOKEN_ID_TO_STRING[raw_token_id],
                "native_control6_selected_row_index": selected_index,
                "native_control6_selected_token_id": selected_token_id,
                "native_control6_selected_token": TOKEN_ID_TO_STRING[selected_token_id],
                "native_action": selected_action.value,
                "native_action_probability": probability,
                "selected_action_probability": probability,
                "selected_action": selected_action.value,
                "legal_actions": list(legal),
                "listener_hidden_frame_id": source_frame,
                "listener_hidden_at_ns": int(available_at) if available_at is not None else None,
                "listener_hidden_age_frames": 0,
                "hidden_contract": "gemma4_listener_final_norm_frame_v1",
                "response_lora_applied": False,
                "microbatch_policy": True,
            }
            decisions.append(PolicyDecision(
                action=selected_action,
                frame_id=int(state.frame_id),
                confidence=probability,
                reason="gemma_native_control6_microbatch",
                generation_id=int(state.generation_id),
                listener_frame_id=source_frame,
                source="gemma_lm_head_selected_rows+rank16_delta:ListenerOutput.gemma_hidden_t",
                session=str(state.session_id),
                diagnostics=diagnostics,
            ))
        return tuple(decisions)

    def update_hidden(self, hidden: object | None) -> None:
        """Preserve the TurnPolicy lifecycle surface without retaining row state."""
        del hidden

    def reset(self) -> None:
        """The policy has no cross-frame state; per-session caches live elsewhere."""
        return None


__all__ = [
    "BatchPolicyInput",
    "DecisionRowSource",
    "GemmaNativeControl6BatchPolicy",
    "ListenerHiddenProvenance",
    "NativeControl6RawScoreSnapshot",
    "PolicyPreHeadContext",
]
