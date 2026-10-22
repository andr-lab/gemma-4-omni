"""Post-merge native listener feature extraction and six-action head training."""
from __future__ import annotations

import json
import hashlib
import math
import os
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import quote

import numpy as np
import torch
from torch.nn import functional as F

from duplex_kernel.native_control6_phase import (
    NATIVE_CONTROL6_NATIVE1540_PHASE_T_SCHEMA,
    NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES,
    native_control6_native1540_phase_t,
)
from duplex_kernel.acoustic_phase_features import EPSILON as _PCM_RMS_EPSILON

from .control import ControlHead, load_control_head
from .features import (
    FEATURE_CONTRACT,
    last_audio_query,
    listener_feature_contract,
    native_control6_feature_contract,
)
from .prepare import read_jsonl
from .schema import ACTIONS
from .native_control6 import NATIVE_CONTROL6_TRAINING_STATE_CONTEXT


def load_merged_base(
    base_model_path: str,
    adapter_dir: str,
    *,
    device: str = "cuda",
) -> tuple[Any, Any, Any]:
    """Load BF16 Gemma, merge the final PEFT adapter, and return its processor.

    This must run before GemmaListener creates NativeGemmaListenerLM, whose
    C ABI borrows the language model's parameter pointers directly.
    """
    if not Path(adapter_dir).is_dir():
        raise FileNotFoundError(f"adapter LoRA finale mancante: {adapter_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("estrazione listener stage2 richiede il runtime CUDA del kernel")
    try:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("stage2 richiede transformers e peft nel runtime CUDA") from exc
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        processor = AutoProcessor.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model = model.merge_and_unload()
    except Exception as exc:
        raise RuntimeError(
            "BF16 base + PEFT merge fallito prima del binding nativo: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    _attest_merged_model_source(model, base_model_path)
    floating = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != torch.bfloat16 for parameter in floating):
        raise RuntimeError("Gemma merged runtime richiede tutti i parametri floating BF16")
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("merge_and_unload ha lasciato parametri PEFT prima del listener nativo")
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    from duplex_kernel.models.thinker import ThinkerModel

    thinker = ThinkerModel(model, tokenizer, processor, device=device)
    setattr(model, "_duplex_qlora_merged_adapter_dir", str(Path(adapter_dir).resolve()))
    return thinker, processor, model


def load_serving_base(
    *,
    workspace: str | Path,
    base_model_path: str | None = None,
    device: str = "cuda",
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Load the same BF16, base-only Thinker selected by the serving loader."""
    import importlib

    thinker_module = importlib.import_module("duplex_kernel.models.thinker")
    load_thinker = thinker_module.load_thinker
    resolve_thinker_path = thinker_module.resolve_thinker_path

    workspace_path = Path(workspace).resolve()
    resolved_base = Path(resolve_thinker_path(str(workspace_path))).resolve()
    if base_model_path is not None and Path(base_model_path).resolve() != resolved_base:
        raise ValueError(
            "native1540 base path must match the serving resolve_thinker_path result: "
            f"requested={Path(base_model_path).resolve()} serving={resolved_base}"
        )
    thinker = load_thinker(
        str(workspace_path),
        device=device,
        dtype=torch.bfloat16,
    )
    model = getattr(thinker, "model", None)
    if model is None or getattr(thinker, "processor", None) is None:
        raise RuntimeError("serving Thinker loader did not return its Gemma model and audio processor")
    model_path = Path(str(getattr(model, "_gemma4_checkpoint_path", ""))).resolve()
    if model_path != resolved_base:
        raise RuntimeError(
            "serving Thinker model provenance differs from resolve_thinker_path: "
            f"model={model_path} loader={resolved_base}"
        )
    floating = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != torch.bfloat16 for parameter in floating):
        raise RuntimeError("serving native1540 extraction requires the BF16 base-only Thinker")
    if any("lora_" in name.lower() for name, _ in model.named_parameters()):
        raise RuntimeError("serving native1540 extraction refuses a PEFT/LoRA model")
    return thinker, thinker.processor, model, {
        "loader": "duplex_kernel.models.thinker.load_thinker",
        "loader_source": str(Path(thinker_module.__file__).resolve()),
        "workspace": str(workspace_path),
        "base_model_path": str(resolved_base),
        "model_dtype": "bfloat16",
        "response_lora_applied": False,
        "stage1_adapter_applied": False,
    }


class _ServingNativeControl6Worker:
    """Use the serving shared native Q2 worker and unified prefix for Stage2 PCM."""

    def __init__(
        self,
        thinker: Any,
        listener: Any,
        *,
        workspace: str | Path,
        native_library_path: str | Path | None = None,
    ) -> None:
        import importlib
        import sys

        self._workspace = Path(workspace).resolve()
        source_root_value = os.environ.get("DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT")
        source_root: Path | None = None
        if not source_root_value:
            raise RuntimeError("native1540 requires DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT from serving")
        source_root = Path(source_root_value)
        if not source_root.is_absolute():
            source_root = self._workspace / source_root
        source_root = source_root.resolve()
        overlay_package = source_root / "duplex_kernel"
        if not overlay_package.is_dir():
            raise FileNotFoundError(f"serving runtime source overlay missing: {overlay_package}")
        package = importlib.import_module("duplex_kernel")
        package_path = getattr(package, "__path__", None)
        if package_path is None:
            raise RuntimeError("duplex_kernel import is not a package; cannot select the serving mirror")
        overlay_value = str(overlay_package)
        if overlay_value in package_path:
            package_path.remove(overlay_value)
        package_path.insert(0, overlay_value)
        existing_server = sys.modules.get("duplex_kernel.server")
        if existing_server is not None and not Path(str(existing_server.__file__)).resolve().is_relative_to(source_root):
            raise RuntimeError(
                "duplex_kernel.server was imported before activating the serving source mirror: "
                f"{existing_server.__file__}"
            )
        serving_server = importlib.import_module("duplex_kernel.server")
        from duplex_kernel.gemma_representation import AUDIO_FEATURES_PER_FRAME
        from duplex_kernel.listener_microbatch_frontend import MEL_WINDOW_ROWS
        from duplex_kernel.listener_microbatch_serving import (
            initial_listener_session_state,
            make_listener_microbatch_worker,
            make_unified_prefix_import,
        )

        configured_library = (
            native_library_path
            or os.environ.get("DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH")
            or os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH")
        )
        if not configured_library:
            raise RuntimeError(
                "native1540 requires the serving native listener library via "
                "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH"
            )
        resolved_library = Path(configured_library).resolve()
        if not resolved_library.is_file():
            raise FileNotFoundError(f"serving native listener library missing: {resolved_library}")

        self.thinker = thinker
        self.listener = listener
        self.listener_backend = getattr(listener, "backend", None)
        if getattr(listener, "runtime_profile", None) != "production":
            raise RuntimeError("serving native1540 worker requires GemmaListener production profile")
        if self.listener_backend is None or not bool(getattr(self.listener_backend, "_use_native_lm", False)):
            raise RuntimeError("serving native1540 worker requires the production native LM backend")
        effective_rows = int(getattr(self.listener_backend, "mel_window_rows", -1))
        if AUDIO_FEATURES_PER_FRAME != 2 or MEL_WINDOW_ROWS != 240 or effective_rows != 240:
            raise RuntimeError(
                "serving native1540 worker requires effective Q2/W240; "
                f"constants={AUDIO_FEATURES_PER_FRAME}/{MEL_WINDOW_ROWS}, backend={effective_rows}"
            )

        self._initial_listener_session_state = initial_listener_session_state
        self._make_unified_prefix_import = make_unified_prefix_import
        self._prefix_builder = serving_server._build_unified_listener_prefix_token_ids
        self._engine_adapter = SimpleNamespace(
            tokenizer=getattr(thinker, "tokenizer", None),
            _thinker_wrapper=thinker,
        )
        if self._engine_adapter.tokenizer is None:
            raise RuntimeError("serving unified prefix requires the loaded Thinker tokenizer")
        self._native_library_path = resolved_library
        self._prefix_builder_source = str(Path(serving_server.__file__).resolve())
        self._worker_factory_source = str(
            Path(__import__("duplex_kernel.listener_microbatch_serving", fromlist=["make_listener_microbatch_worker"]).__file__).resolve()
        )
        self._expected_runtime_source_root = None
        if source_root is not None:
            self._expected_runtime_source_root = str(source_root)
            try:
                Path(self._prefix_builder_source).relative_to(source_root)
            except ValueError as exc:
                raise RuntimeError(
                    "native1540 prefix builder did not load from the configured serving mirror; "
                    f"source={self._prefix_builder_source} mirror={source_root}"
                ) from exc
            worker_source = Path(self._worker_factory_source).resolve()
            allowed_worker_roots = (source_root, self._workspace / "duplex_kernel")
            if not any(worker_source.is_relative_to(root.resolve()) for root in allowed_worker_roots):
                raise RuntimeError(
                    "native1540 shared worker factory came from outside the serving mirror/workspace; "
                    f"source={worker_source} mirror={source_root}"
                )

        self._worker = make_listener_microbatch_worker(
            listener,
            library_path=resolved_library,
            semantic_prefix_enabled=False,
            unified_kv_enabled=True,
        )
        self._session_index = 0
        self._native_context_index = 0
        self._active_session: dict[str, Any] | None = None
        self._prefix_token_ids: tuple[int, ...] | None = None
        self._closed = False
        try:
            self._worker.start().result()
            self._runtime_signature = self._worker.runtime_signature
            runtime_info = getattr(self.listener_backend, "_listener_microbatch_runtime_info", {})
            if not isinstance(runtime_info, dict) or int(runtime_info.get("stream_q_len", -1)) != 2:
                raise RuntimeError("serving native worker did not attest effective stream_q_len=2")
            effective_library = Path(str(runtime_info.get("library_path", ""))).resolve()
            if effective_library != resolved_library:
                raise RuntimeError(
                    "serving native worker loaded a different listener library: "
                    f"configured={resolved_library} effective={effective_library}"
                )
            self._runtime_info = dict(runtime_info)
        except BaseException:
            try:
                self._worker.stop(drain=True).result()
            except BaseException:
                pass
            self._closed = True
            raise

    def _make_prefix(self, session: Any) -> tuple[int, ...]:
        values = tuple(int(value) for value in self._prefix_builder(self._engine_adapter, session))
        if not values:
            raise RuntimeError("serving unified prefix builder returned no token IDs")
        if self._prefix_token_ids is None:
            self._prefix_token_ids = values
        elif values != self._prefix_token_ids:
            raise RuntimeError("native1540 first-session prefix changed between prepared conversations")
        return values

    def begin_conversation(self, conversation_id: str) -> dict[str, Any]:
        if self._active_session is not None:
            raise RuntimeError("previous native1540 conversation session has not been closed")
        from duplex_kernel.listener import DEFAULT_CONTEXT_FRAMES, GemmaListenerState

        self._session_index += 1
        self._native_context_index += 1
        session_id = f"stage2-native1540-{self._session_index}-{hashlib.sha256(str(conversation_id).encode()).hexdigest()[:12]}"
        state = GemmaListenerState(
            sample_rate=16000,
            frame_duration_ms=80,
            context_frames=DEFAULT_CONTEXT_FRAMES,
        )
        session = SimpleNamespace(
            session_id=session_id,
            listener=self.listener,
            listener_state=state,
            history=[],
            instructions="",
        )
        prefix_ids = self._make_prefix(session)
        row_state = self._initial_listener_session_state(
            session,
            self._runtime_signature,
            native_context_id=self._native_context_index,
            generation_id=1,
            cache_position=len(prefix_ids),
        )
        importer = self._make_unified_prefix_import(self.listener_backend, prefix_ids)
        self._worker.register_session(row_state, prefix_import=importer).result()
        self._active_session = {
            "session_id": session_id,
            "generation_id": 1,
            "state_epoch_id": str(state.state_epoch_id),
            "listener_state": state,
            "prefix_len": len(prefix_ids),
            "prefix_token_ids": prefix_ids,
        }
        return dict(self._active_session)

    def step(
        self,
        frame_id: int,
        user_pcm: np.ndarray,
        assistant_reference_pcm: np.ndarray | None,
        assistant_reference_frame_id: int | None,
    ) -> Any:
        if self._active_session is None:
            raise RuntimeError("native1540 worker session is not open")
        from duplex_kernel.microbatch_contract import ListenerFrame

        active = self._active_session
        frame = ListenerFrame(
            session_id=str(active["session_id"]),
            generation_id=int(active["generation_id"]),
            state_epoch_id=str(active["state_epoch_id"]),
            frame_id=int(frame_id),
            ready_at_ns=time.monotonic_ns(),
            pcm=np.asarray(user_pcm, dtype=np.float32),
            logical_q_len=2,
            reference_pcm=(
                None
                if assistant_reference_frame_id is None
                else np.asarray(assistant_reference_pcm, dtype=np.float32)
            ),
            reference_frame_id=assistant_reference_frame_id,
        )
        result = self._worker.submit_tick((frame,)).result()
        output = result[str(active["session_id"])]
        if int(output.frame_id) != int(frame_id):
            raise RuntimeError(
                f"serving native worker frame misalignment: expected={frame_id}, got={output.frame_id}"
            )
        return output

    def end_conversation(self) -> None:
        active = self._active_session
        if active is None:
            return
        try:
            self._worker.close_session(str(active["session_id"])).result()
        finally:
            state = active.get("listener_state")
            if callable(getattr(state, "close", None)):
                state.close()
            self._active_session = None

    def report(self) -> dict[str, Any]:
        return {
            "worker": "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker",
            "worker_factory_source": self._worker_factory_source,
            "prefix_builder": "duplex_kernel.server._build_unified_listener_prefix_token_ids",
            "prefix_builder_source": self._prefix_builder_source,
            "runtime_source_root": self._expected_runtime_source_root,
            "native_library_path": str(self._native_library_path),
            "runtime_signature": {
                "model_id": self._runtime_signature.model_id,
                "native_abi_version": self._runtime_signature.native_abi_version,
                "batch_api_version": self._runtime_signature.batch_api_version,
                "attention_backend_id": self._runtime_signature.attention_backend_id,
                "rmsnorm_backend_id": self._runtime_signature.rmsnorm_backend_id,
                "max_seq_len": self._runtime_signature.max_seq_len,
                "stream_q_len": self._runtime_signature.stream_q_len,
            },
            "runtime_info": self._runtime_info,
            "prefix": {
                "contract": "serving_unified_chat_prefix_v2",
                "history_messages": 0,
                "session_instructions": "empty_session_default",
                "audio_marker": "final_user_audio_marker_then_boa",
                "token_ids": list(self._prefix_token_ids or ()),
                "length": len(self._prefix_token_ids or ()),
            },
            "conversation_history_policy": (
                "first-session parity: prepared speech history is not appended; "
                "each recording starts with the serving default empty history/instructions"
            ),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.end_conversation()
        finally:
            worker = getattr(self, "_worker", None)
            if worker is not None:
                worker.stop(drain=True).result()


def _attest_merged_model_source(model: Any, base_model_path: str) -> str:
    """Preserve the original BF16 base path on the nested LM borrowed by native CUDA."""
    nested = getattr(model, "model", None)
    language_model = getattr(nested, "language_model", None)
    if nested is None or language_model is None:
        raise RuntimeError("Gemma merged model non espone model.language_model per il listener nativo")
    resolved = str(Path(base_model_path).resolve())
    for module in (model, nested, language_model):
        setattr(module, "_gemma4_checkpoint_path", resolved)
    return resolved


def _channel_index(conversation: Mapping[str, Any], role: str) -> int | None:
    channel_map = conversation.get("channel_map") or {}
    values = [int(channel) for channel, value in channel_map.items() if value == role and str(channel).isdigit()]
    return values[0] if values else None


def _is_synthetic_source(row: Mapping[str, Any]) -> bool:
    """Treat an explicit authored-V5 source/profile as synthetic even if a legacy flag is absent."""
    return bool(row.get("synthetic")) or (
        row.get("source") == "authored_synthetic_audio_candidate"
        or row.get("profile") == "audio_grounded_v5"
    )


def _native1540_pre_action_speaker_state(
    conversation: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    frame_id: int,
) -> tuple[bool | None, str]:
    """Resolve the synthetic assistant state before the decision at input frame t.

    V5 stores post-action audio on the row whose input is t and target is t+1.
    Therefore a contiguous row for t-1 is the only implicit state source for t.
    Explicit, time-bounded action spans are the fallback when the trace has a gap.
    """
    for key, source in (
        ("own_active", "frame_target_own_active"),
        ("active_response", "frame_target_active_response"),
    ):
        value = frame.get(key)
        if type(value) is bool:
            return value, source
    if frame_id == 0:
        return False, "synthetic_stream_initial_idle"

    target_rows = conversation.get("frame_targets")
    if isinstance(target_rows, (list, tuple)):
        previous_rows = [
            row for row in target_rows
            if isinstance(row, Mapping)
            and row.get("input_frame_id") == frame_id - 1
            and row.get("target_frame_id") == frame_id
        ]
        if len(previous_rows) == 1 and type(previous_rows[0].get("assistant_audible")) is bool:
            return previous_rows[0]["assistant_audible"], "prior_target_assistant_audible_t_minus_1"

    explicit_states: set[bool] = set()
    for segment in conversation.get("action_segments", ()) or ():
        if not isinstance(segment, Mapping):
            continue
        start, end = segment.get("target_frame_id_start"), segment.get("target_frame_id_end")
        if (
            isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
            or not start <= frame_id < end
            or segment.get("target_action_mask", True) is not True
        ):
            continue
        action = segment.get("action")
        if action in {"SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL"}:
            explicit_states.add(True)
        elif action == "LISTEN":
            explicit_states.add(False)
    if len(explicit_states) == 1:
        return explicit_states.pop(), "verified_action_segment_covering_current_audio_frame"
    return None, "synthetic_pre_action_state_unavailable"


def _native1540_runtime_target_action(
    conversation: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    frame_id: int,
    speaker_active: bool | None,
) -> dict[str, Any]:
    """Map persisted event labels to the action still legal at this runtime tick.

    Raw labels and their evidence remain untouched. The derived runtime target
    for synthetic SPEAK/BACKCHANNEL spans requires one explicit action segment
    to cover the current and next target frame. A qualified natural YIELD onset
    also carries through its next three observed overlap ticks while both
    channels remain active; direct/reviewed labels retain precedence.
    """
    raw_action = frame.get("target_action", frame.get("action"))
    result: dict[str, Any] = {
        "action": raw_action,
        "derived": False,
        "reason": None,
        "segment_index": None,
        "event_id": None,
        "yield_onset_frame_id": None,
        "yield_local_stop_ms": None,
        "yield_local_segment_id": None,
        "yield_peer_segment_id": None,
    }
    if (
        not _is_synthetic_source(conversation)
        and raw_action == "CONTINUE"
        and frame.get("target_action_source") == "timestamp_derived_local_speech_active"
        and frame.get("target_action_mask", frame.get("action_mask")) is True
        and speaker_active is True
        and frame.get("peer_active_during_frame") is True
    ):
        current_evidence = frame.get("target_action_evidence")
        local_segment_id = (
            current_evidence.get("local_segment_id")
            if isinstance(current_evidence, Mapping)
            else None
        )
        current_frame_id = frame.get("input_frame_id", frame.get("frame_id"))
        if (
            isinstance(local_segment_id, str)
            and local_segment_id
            and isinstance(current_frame_id, int)
            and not isinstance(current_frame_id, bool)
            and current_frame_id == frame_id
        ):
            yield_events: list[tuple[int, str, str, float]] = []
            for candidate in conversation.get("frame_targets", ()) or ():
                if not isinstance(candidate, Mapping):
                    continue
                onset_frame_id = candidate.get("input_frame_id")
                if (
                    isinstance(onset_frame_id, bool)
                    or not isinstance(onset_frame_id, int)
                    or not 1 <= frame_id - onset_frame_id <= 3
                    or candidate.get("action") != "YIELD"
                    or candidate.get("target_action_source")
                    != "timestamp_derived_peer_onset_confirmed_local_stop"
                    or candidate.get("target_action_mask") is not True
                ):
                    continue
                evidence = candidate.get("target_action_evidence")
                if not isinstance(evidence, Mapping) or evidence.get("local_segment_id") != local_segment_id:
                    continue
                peer_segment_id = evidence.get("peer_segment_id")
                local_stop_ms = evidence.get("local_stop_ms")
                if (
                    not isinstance(peer_segment_id, str)
                    or not peer_segment_id
                    or isinstance(local_stop_ms, bool)
                    or not isinstance(local_stop_ms, (int, float))
                    or not math.isfinite(float(local_stop_ms))
                    or float(local_stop_ms) <= 0.0
                ):
                    continue
                yield_events.append((onset_frame_id, local_segment_id, peer_segment_id, float(local_stop_ms)))
            unique_events = list(dict.fromkeys(yield_events))
            if len(unique_events) == 1:
                onset_frame_id, local_segment_id, peer_segment_id, local_stop_ms = unique_events[0]
                result.update({
                    "action": "YIELD",
                    "derived": True,
                    "reason": "qualified_yield_overlap_persistence_until_local_stop",
                    "event_id": peer_segment_id,
                    "yield_onset_frame_id": onset_frame_id,
                    "yield_local_stop_ms": local_stop_ms,
                    "yield_local_segment_id": local_segment_id,
                    "yield_peer_segment_id": peer_segment_id,
                })
                return result
    if (
        not _is_synthetic_source(conversation)
        or raw_action not in {"SPEAK", "BACKCHANNEL"}
        or speaker_active is not True
        or frame.get("assistant_audible") is not True
    ):
        return result

    segments = conversation.get("action_segments", ()) or ()
    covering_current: list[tuple[int, Mapping[str, Any]]] = []
    covering_target: list[tuple[int, Mapping[str, Any]]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or segment.get("action") != raw_action:
            continue
        start, end = segment.get("target_frame_id_start"), segment.get("target_frame_id_end")
        if (
            isinstance(start, bool) or not isinstance(start, int)
            or isinstance(end, bool) or not isinstance(end, int)
            or segment.get("target_action_mask", True) is not True
        ):
            continue
        if start <= frame_id < end:
            covering_current.append((index, segment))
        if start <= frame_id + 1 < end:
            covering_target.append((index, segment))
    shared = [(index, current) for index, current in covering_current
              for target_index, target in covering_target
              if index == target_index and target is current]
    if len(shared) != 1:
        return result

    previous_rows = [
        row for row in conversation.get("frame_targets", ()) or ()
        if isinstance(row, Mapping)
        and row.get("input_frame_id") == frame_id - 1
        and row.get("target_frame_id") == frame_id
    ]
    if len(previous_rows) != 1:
        return result
    previous = previous_rows[0]
    if previous.get("action") != raw_action:
        return result
    previous_evidence = previous.get("target_action_evidence")
    current_evidence = frame.get("target_action_evidence")
    segment_index, segment = shared[0]
    def event_ids(value: Any) -> set[str]:
        if not isinstance(value, Mapping):
            return set()
        return {
            str(value[key])
            for key in ("event_id", "segment_id", "source_event_id")
            if value.get(key) not in (None, "")
        }

    segment_ids = event_ids(segment)
    previous_ids = event_ids(previous_evidence)
    current_ids = event_ids(current_evidence)
    if segment_ids and ((previous_ids and not segment_ids.intersection(previous_ids))
                        or (current_ids and not segment_ids.intersection(current_ids))):
        return result
    if not segment_ids and previous_ids and current_ids and previous_ids != current_ids:
        return result

    result.update({
        "action": "CONTINUE",
        "derived": True,
        "reason": f"same_active_{raw_action.lower()}_event_continues",
        "segment_index": segment_index,
        "event_id": next(iter(segment_ids), next(iter(current_ids), None)),
    })
    return result


def _native1540_supervision_context(
    conversation: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    frame_id: int,
    user_pcm_current: np.ndarray,
    assistant_pcm_prior: np.ndarray | None,
    assistant_prior_frame_id: int | None,
) -> dict[str, Any]:
    """Decode runtime speaker/user context from source state and causal PCM."""
    tick_start_ms = frame_id * 80.0
    tick_end_ms = tick_start_ms + 80.0
    if _is_synthetic_source(conversation):
        own_active, own_active_source = _native1540_pre_action_speaker_state(
            conversation, frame, frame_id=frame_id
        )
        peer_active = frame.get("peer_active_during_frame")
        peer_source = "frame_target_peer_active_during_frame"
        if type(peer_active) is not bool:
            acoustic_state = frame.get("acoustic_state")
            if acoustic_state not in {"user_active", "quiet"}:
                raise ValueError(
                    f"{conversation.get('conversation_id')}:{frame_id}: synthetic target lacks "
                    "peer_active_during_frame or acoustic_state"
                )
            peer_active = acoustic_state == "user_active"
            peer_source = "frame_target_acoustic_state"
        context_provenance = {
            "schema": "native_control6_runtime_context_v1",
            "source_kind": "synthetic_trace",
            "training_state_context": NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
            "speaker_state_source": own_active_source,
            "peer_state_source": peer_source,
            "assistant_audible_proxy_used": False,
            "pre_action_speaker_state_valid": own_active is not None,
        }
    else:
        local_channel = _channel_index(conversation, "assistant")
        remote_channel = _channel_index(conversation, "user")
        if local_channel is None or remote_channel is None:
            raise ValueError(
                f"{conversation.get('conversation_id')}:{frame_id}: natural runtime context requires "
                "explicit assistant/user channel mapping"
            )
        verified_segments = []
        for segment in conversation.get("segments", ()) or ():
            if not isinstance(segment, Mapping):
                continue
            if (
                segment.get("audio_channel_verified") is not True
                or segment.get("time_mask") is not True
                or segment.get("channel_index") not in {local_channel, remote_channel}
            ):
                continue
            try:
                start_ms, end_ms = float(segment["start_ms"]), float(segment["end_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(start_ms) and np.isfinite(end_ms) and start_ms < end_ms:
                verified_segments.append((int(segment["channel_index"]), start_ms, end_ms))
        own_active = any(channel == local_channel and start <= tick_start_ms < end for channel, start, end in verified_segments)
        peer_active = any(channel == remote_channel and start < tick_end_ms and end > tick_start_ms for channel, start, end in verified_segments)
        context_provenance = {
            "schema": "native_control6_runtime_context_v1",
            "source_kind": "verified_role_intervals",
            "training_state_context": NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
            "speaker_state_source": "assistant_interval_active_at_frame_start",
            "peer_state_source": "user_interval_intersects_current_frame",
            "assistant_audible_proxy_used": False,
        }

    user_values = np.asarray(user_pcm_current, dtype=np.float64)
    if user_values.ndim != 1 or not np.isfinite(user_values).all():
        raise ValueError(f"{conversation.get('conversation_id')}:{frame_id}: invalid current user PCM")
    user_rms = float(np.sqrt(np.mean(user_values * user_values)))
    assistant_rms = 0.0
    ratio_valid = False
    if assistant_pcm_prior is not None and assistant_prior_frame_id == frame_id - 1:
        assistant_values = np.asarray(assistant_pcm_prior, dtype=np.float64)
        if assistant_values.ndim == 1 and np.isfinite(assistant_values).all():
            assistant_rms = float(np.sqrt(np.mean(assistant_values * assistant_values)))
            ratio_valid = user_rms > _PCM_RMS_EPSILON and assistant_rms >= 0.001
    user_rms_active = user_rms >= 0.001
    user_log = float(np.clip(np.log10(max(user_rms, _PCM_RMS_EPSILON)), -6.0, 0.0))
    assistant_log = float(np.clip(np.log10(max(assistant_rms, _PCM_RMS_EPSILON)), -6.0, 0.0))
    ratio = float(np.clip(user_log - assistant_log, -6.0, 6.0)) if ratio_valid else 0.0
    user_activity = bool(peer_active or user_rms_active or (ratio_valid and ratio > -0.15))
    return {
        "native_control6_context": context_provenance,
        "training_state_context": NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
        "speaker_active_at_frame_start": own_active,
        "peer_active_during_frame": bool(peer_active),
        "native_control6_user_rms_activity": user_rms_active,
        "native_control6_user_assistant_ratio_valid": ratio_valid,
        "native_control6_user_assistant_log_ratio": ratio,
        "native_control6_user_activity_detected": user_activity,
    }


def _causal_action_targets(
    conversation: Mapping[str, Any],
    *,
    include_masked: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    """Normalize v4 frame labels and v5 input-t/target-t+1 frame targets.

    Legacy ``frames`` keep their existing same-index contract. Presence of
    ``frame_targets`` selects the audio-grounded causal contract, independent
    of source/profile names.
    """
    is_v5 = "frame_targets" in conversation
    key = "frame_targets" if is_v5 else "frames"
    raw_targets = conversation.get(key, ()) or ()
    if not isinstance(raw_targets, (list, tuple)):
        raise ValueError(f"{conversation.get('conversation_id', 'unknown')}: {key} must be a sequence")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    frame_samples = 1280
    v5_onset_action_by_target_frame: dict[int, str] = {}
    if is_v5:
        for event in conversation.get("event_schedule", ()) or ():
            if not isinstance(event, Mapping):
                continue
            action = event.get("action")
            start_sample = event.get("start_sample")
            if action not in ACTIONS or isinstance(start_sample, bool) or not isinstance(start_sample, int):
                continue
            if start_sample < 0 or start_sample % frame_samples:
                raise ValueError("audio-grounded event onset is not aligned to a target frame sample")
            target_start = start_sample // frame_samples
            previous_action = v5_onset_action_by_target_frame.setdefault(target_start, str(action))
            if previous_action != action:
                raise ValueError("audio-grounded events have conflicting event-onset actions")
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{key}[{index}] must be an object")
        frame = dict(raw)
        if is_v5:
            input_id = frame.get("input_frame_id")
            target_id = frame.get("target_frame_id")
            if isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
                raise ValueError("input_frame_id must be a non-negative integer")
            if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id != input_id + 1:
                raise ValueError("target_frame_id must equal input_frame_id + 1")
            declared_frame_id = frame.get("frame_id", input_id)
            if (
                isinstance(declared_frame_id, bool)
                or not isinstance(declared_frame_id, int)
                or declared_frame_id != input_id
            ):
                raise ValueError("frame_id must equal input_frame_id for audio-grounded targets")
            expected_samples = {
                "input_audio_start_sample": input_id * frame_samples,
                "input_audio_end_sample": (input_id + 1) * frame_samples,
                "target_action_start_sample": (input_id + 1) * frame_samples,
                "target_action_end_sample": (input_id + 2) * frame_samples,
            }
            for field, expected in expected_samples.items():
                value = frame.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                    raise ValueError(f"{field} is not aligned to input_frame_id/target_frame_id")
            if frame.get("input_text_visible") is not False:
                raise ValueError("input_text_visible must be false for audio-grounded targets")
            if frame.get("future_audio_excluded") is not True:
                raise ValueError("future_audio_excluded must be true for causal targets")
            if type(frame.get("target_action_mask")) is not bool:
                raise ValueError("target_action_mask must be boolean")
            frame["frame_id"] = input_id
            frame["target_action"] = frame.get("action")
            frame["action_mask"] = frame["target_action_mask"]
            event_action = v5_onset_action_by_target_frame.get(target_id)
            if event_action is not None and frame["target_action"] != event_action:
                raise ValueError("audio-grounded event onset action disagrees with its target frame")
            frame["event_onset"] = event_action is not None
        else:
            input_id = frame.get("frame_id", -1)
            target_id = frame.get("target_frame_id", input_id)
            frame["input_frame_id"] = input_id
            frame.setdefault("target_frame_id", target_id)
            frame["event_onset"] = frame.get("event_onset") is True
        if isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
            raise ValueError("stage2 target input frame id must be a non-negative integer")
        if input_id in seen:
            raise ValueError(f"duplicate stage2 input frame id: {input_id}")
        seen.add(input_id)
        action = frame.get("target_action")
        mask = frame.get("action_mask")
        if is_v5 and action not in ACTIONS:
            raise ValueError(f"frame_targets[{index}] has an action outside the six-class ABI")
        if mask is not True or action not in ACTIONS:
            if include_masked:
                frame["_stage2_is_v5_target"] = is_v5
                result.append(frame)
            continue
        frame["_stage2_is_v5_target"] = is_v5
        result.append(frame)
    result.sort(key=lambda frame: int(frame["frame_id"]))
    return result, is_v5


def _audio_frames(
    conversation: Mapping[str, Any],
    *,
    frame_samples: int,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return 16 kHz channel frames and compact valid-sample prefix masks.

    ``scipy.signal.resample_poly`` upsamples, applies a zero-phase low-pass FIR,
    then downsamples; see its official reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html.
    A count per frame marks real samples; any partial-frame suffix is zero padded.
    """
    import soundfile as sf

    audio_path = conversation.get("audio_path")
    if not audio_path or not Path(str(audio_path)).is_file():
        raise FileNotFoundError(f"audio conversation stage2 mancante: {audio_path}")
    resolved_path = str(Path(str(audio_path)).resolve())
    cached = pcm_cache.get(resolved_path) if pcm_cache is not None else None
    source_sample_rate = None
    if cached is None:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        sample_rate = int(sample_rate)
        source_sample_rate = sample_rate
        if sample_rate <= 0:
            raise ValueError(f"{audio_path}: sample rate audio non valido: {sample_rate}")
        if audio.shape[0] and sample_rate != 16000:
            from math import gcd

            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("stage2 richiede scipy.signal.resample_poly per audio non a 16 kHz") from exc
            common = gcd(sample_rate, 16000)
            audio = resample_poly(
                audio,
                up=16000 // common,
                down=sample_rate // common,
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            audio = np.asarray(audio, dtype=np.float32)
        sample_rate = 16000
        if pcm_cache is not None:
            # The caller scopes this cache to one physical source group so
            # paired role views reuse one decode/resample without corpus caching.
            pcm_cache[resolved_path] = (audio, 16000)
    else:
        audio, sample_rate = cached
    if sample_rate != 16000:
        raise RuntimeError(f"{audio_path}: PCM cache non normalizzata a 16 kHz")
    if audio.ndim != 2 or audio.shape[1] <= 0:
        raise ValueError(f"{audio_path}: PCM deve essere [samples, channels]")
    user_channel = _channel_index(conversation, "user")
    assistant_channel = _channel_index(conversation, "assistant")
    if user_channel is None or user_channel >= audio.shape[1]:
        raise ValueError(f"{audio_path}: channel_map non assegna un canale user esplicito")
    if assistant_channel is not None and assistant_channel >= audio.shape[1]:
        raise ValueError(f"{audio_path}: channel_map assistant fuori range")
    if "frame_targets" in conversation:
        declared_sample_rate = conversation.get("sample_rate_hz", conversation.get("sample_rate", 16000))
        if (
            (source_sample_rate is not None and source_sample_rate != 16000)
            or int(declared_sample_rate) != 16000
            or audio.shape[1] != 2
        ):
            raise ValueError(f"{audio_path}: audio-grounded frames require stereo 16 kHz PCM")
        if user_channel != 0 or assistant_channel != 1:
            raise ValueError(f"{audio_path}: audio-grounded channels require user=0 and assistant=1")
    total_frames = max(1, (audio.shape[0] + frame_samples - 1) // frame_samples)
    user = np.zeros((total_frames, frame_samples), dtype=np.float32)
    assistant = None if assistant_channel is None else np.zeros_like(user)
    valid_samples = np.zeros(total_frames, dtype=np.int32)
    for frame_index in range(total_frames):
        start = frame_index * frame_samples
        end = min(audio.shape[0], start + frame_samples)
        valid_count = max(0, end - start)
        valid_samples[frame_index] = valid_count
        if valid_count:
            user[frame_index, :valid_count] = audio[start:end, user_channel]
        if assistant is not None:
            if valid_count:
                assistant[frame_index, :valid_count] = audio[start:end, assistant_channel]
    return user, assistant, valid_samples


def extract_conversation_features(
    listener: Any,
    conversation: Mapping[str, Any],
    *,
    session_id: str | None = None,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
    timing: dict[str, Any] | None = None,
    control6_profile: str = "historical_q4",
    serving_native_worker: _ServingNativeControl6Worker | Any | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replay one causal conversation stream and keep only supervised frames."""
    from duplex_kernel.listener import DEFAULT_CONTEXT_FRAMES, GemmaListenerState
    from duplex_kernel.reference_buffer import AssistantReferenceFrame

    if getattr(listener, "runtime_profile", None) != "production":
        raise RuntimeError("stage2 usa solo GemmaListener runtime_profile=production")
    if not bool(getattr(listener.backend, "_use_native_lm", False)):
        raise RuntimeError("stage2 richiede il listener LM nativo del kernel")
    if control6_profile not in {"historical_q4", "native1540"}:
        raise ValueError("control6_profile must be 'historical_q4' or 'native1540'")
    if control6_profile == "native1540" and int(getattr(listener.backend, "mel_window_rows", -1)) != 240:
        raise RuntimeError("native1540 extraction requires the production Q2/W240 listener")
    if control6_profile == "native1540" and serving_native_worker is None:
        raise RuntimeError("native1540 extraction requires the serving shared native listener worker")
    tick_ms = int(conversation.get("tick_ms", 80))
    if tick_ms != 80:
        raise ValueError(f"stage2 richiede frame Gemma da 80 ms, ricevuto {tick_ms}")
    frame_samples = 16000 * tick_ms // 1000
    target_rows, is_v5 = _causal_action_targets(conversation)
    sparse_targets: dict[int, Mapping[str, Any]] = {
        int(frame["frame_id"]): frame for frame in target_rows
    }
    if not sparse_targets:
        return np.empty((0, 1536), dtype=np.float32), []
    clusters = _conversation_replay_clusters(list(sparse_targets.values()))
    if timing is not None:
        timing["conversation_replays"] = int(timing.get("conversation_replays", 0)) + len(clusters)
        causal_steps = sum(int(cluster_targets[-1]["frame_id"]) - cluster_start + 1 for cluster_start, cluster_targets in clusters)
        timing["causal_input_frame_steps"] = int(timing.get("causal_input_frame_steps", 0)) + causal_steps
        timing.setdefault("conversation_replay_input_frames", []).extend(
            int(cluster_targets[-1]["frame_id"]) - cluster_start + 1
            for cluster_start, cluster_targets in clusters
        )
    decode_start = time.perf_counter()
    user_frames, assistant_frames, valid_samples = _audio_frames(
        conversation, frame_samples=frame_samples, pcm_cache=pcm_cache
    )
    if timing is not None:
        timing["audio_decode_resample_frame_pack_seconds"] = float(
            timing.get("audio_decode_resample_frame_pack_seconds", 0.0)
        ) + (time.perf_counter() - decode_start)
    if any(frame_id >= len(user_frames) for frame_id in sparse_targets):
        raise ValueError(f"{conversation['conversation_id']}: frame_id supervisionato oltre la durata del PCM")
    if is_v5:
        actual_samples = max(0, (len(valid_samples) - 1) * frame_samples + int(valid_samples[-1]))
        if any(
            int(frame["target_action_end_sample"]) > actual_samples
            for frame in (conversation.get("frame_targets", ()) or ())
        ):
            raise ValueError(f"{conversation['conversation_id']}: target action oltre la durata del PCM")
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for cluster_index, (cluster_start, cluster_targets) in enumerate(clusters):
        target_by_id = {int(frame["frame_id"]): frame for frame in cluster_targets}
        last_target_id = int(cluster_targets[-1]["frame_id"])
        state = None
        native_session: dict[str, Any] | None = None
        if control6_profile == "native1540":
            if cluster_start != 0 or len(clusters) != 1:
                raise RuntimeError("native1540 worker advances one conversation continuously from frame zero")
            native_session = serving_native_worker.begin_conversation(
                str(conversation.get("conversation_id", session_id or "unknown"))
            )
        else:
            state_create_start = time.perf_counter()
            state = GemmaListenerState(
                sample_rate=16000,
                frame_duration_ms=80,
                context_frames=DEFAULT_CONTEXT_FRAMES,
            )
            if timing is not None:
                timing["state_create_host_seconds"] = float(
                    timing.get("state_create_host_seconds", 0.0)
                ) + (time.perf_counter() - state_create_start)
        try:
            if state is not None:
                prime_start = time.perf_counter()
                listener.prime_steady_state(state)
                if timing is not None:
                    timing["prime_host_seconds"] = float(timing.get("prime_host_seconds", 0.0)) + (
                        time.perf_counter() - prime_start
                    )
            for source_frame_id in range(cluster_start, last_target_id + 1):
                local_frame_id = source_frame_id - cluster_start
                user_pcm = user_frames[source_frame_id]
                reference = None
                # At a cluster reset, the first warm-up frame has no visible
                # past reference; later frames use only the previous physical
                # assistant frame, preserving causal delay across the cluster.
                if assistant_frames is not None and source_frame_id > cluster_start:
                    reference = AssistantReferenceFrame(
                        frame_id=local_frame_id - 1,
                        pcm=assistant_frames[source_frame_id - 1],
                        emitted_at_ns=(local_frame_id - 1) * tick_ms * 1_000_000,
                        sample_rate=16000,
                    )
                assistant_reference_frame_id = (
                    source_frame_id - 1
                    if assistant_frames is not None and source_frame_id > 0
                    else None
                )
                if assistant_reference_frame_id is not None and assistant_reference_frame_id > source_frame_id - 1:
                    raise RuntimeError("assistant reference frame is newer than input_frame_id - 1")
                step_start = time.perf_counter()
                if control6_profile == "native1540":
                    output = serving_native_worker.step(
                        source_frame_id,
                        user_pcm,
                        None if assistant_reference_frame_id is None else assistant_frames[assistant_reference_frame_id],
                        assistant_reference_frame_id,
                    )
                else:
                    output = listener.step(user_pcm, reference, state)
                if timing is not None:
                    timing["listener_step_host_seconds"] = float(
                        timing.get("listener_step_host_seconds", 0.0)
                    ) + (time.perf_counter() - step_start)
                expected_listener_frame = source_frame_id if control6_profile == "native1540" else local_frame_id
                if int(output.frame_id) != expected_listener_frame:
                    raise RuntimeError(
                        "listener_frame_id disallineato: "
                        f"window={expected_listener_frame}, output={output.frame_id}"
                    )
                frame = target_by_id.get(source_frame_id)
                if frame is None:
                    continue
                feature_copy_start = time.perf_counter()
                if control6_profile == "native1540":
                    hidden = getattr(output, "gemma_hidden_t", None)
                    queries = getattr(output, "gemma_audio_queries_t", None)
                    tower = getattr(output, "audio_tower_hidden", None)
                    hidden_shape = tuple(getattr(hidden, "shape", ()))
                    query_shape = tuple(getattr(queries, "shape", ()))
                    tower_shape = tuple(getattr(tower, "shape", ()))
                    if hidden_shape != (1, 1536):
                        raise RuntimeError(
                            f"native1540 expects runtime gemma_hidden_t [1,1536], got {hidden_shape}"
                        )
                    if query_shape != (2, 1536) or tower_shape != (2, 1536):
                        raise RuntimeError(
                            "native1540 extraction requires effective Q2 tensors "
                            f"gemma_audio_queries_t/audio_tower_hidden [2,1536], got {query_shape}/{tower_shape}"
                        )
                    if hasattr(hidden, "detach"):
                        import torch

                        if not torch.equal(hidden, queries[0:1]):
                            raise RuntimeError(
                                "runtime gemma_hidden_t must equal the first current-frame Q2 LM hidden"
                            )
                        vector = hidden[0].detach().float().cpu().numpy()
                    else:
                        hidden_array = np.asarray(hidden)
                        query_array = np.asarray(queries)
                        if not np.array_equal(hidden_array, query_array[0:1]):
                            raise RuntimeError(
                                "runtime gemma_hidden_t must equal the first current-frame Q2 LM hidden"
                            )
                        vector = hidden_array[0].astype(np.float32, copy=False)
                else:
                    queries = getattr(output, "gemma_audio_queries_t", None)
                    if queries is None:
                        raise RuntimeError(
                            "listener installato non espone le quattro query Gemma per tick; "
                            "estrazione control6 rifiutata invece di usare la feature V9"
                        )
                    vector = last_audio_query(queries).detach().float().cpu().numpy()
                if vector.shape != (1536,) or not np.isfinite(vector).all():
                    raise RuntimeError("native Gemma listener ABI must yield a finite 1536-vector")
                if timing is not None:
                    timing["target_feature_copy_seconds"] = float(
                        timing.get("target_feature_copy_seconds", 0.0)
                    ) + (time.perf_counter() - feature_copy_start)
                action = frame.get("target_action") if frame.get("action_mask") else None
                context_start = max(cluster_start, source_frame_id - (DEFAULT_CONTEXT_FRAMES - 1))
                runtime_context = (
                    _native1540_supervision_context(
                        conversation,
                        frame,
                        frame_id=source_frame_id,
                        user_pcm_current=user_pcm,
                        assistant_pcm_prior=(
                            assistant_frames[assistant_reference_frame_id]
                            if assistant_frames is not None and assistant_reference_frame_id is not None
                            else None
                        ),
                        assistant_prior_frame_id=assistant_reference_frame_id,
                    )
                    if control6_profile == "native1540"
                    else {}
                )
                features.append(vector)
                metadata.append(
                    {
                        **{
                            key: value
                            for key, value in conversation.items()
                            if key.startswith("source_")
                            or key in {
                                "annotation_path",
                                "control_supervision",
                                "pairing_manifest_path",
                                "duplexconv_provenance",
                            }
                        },
                        "conversation_id": str(frame.get("origin_conversation_id") or conversation["conversation_id"]),
                        "source_group_id": str(conversation.get("source_group_id", conversation["conversation_id"])),
                        "session_id": (
                            str(native_session["session_id"])
                            if native_session is not None
                            else f"{session_id or conversation['conversation_id']}:conversation-{cluster_index}"
                        ),
                        "frame_id": int(source_frame_id),
                        "input_frame_id": int(source_frame_id),
                        "target_frame_id": int(frame.get("target_frame_id", source_frame_id)),
                        "listener_frame_id": int(output.frame_id),
                        "assistant_reference_frame_id": assistant_reference_frame_id,
                        "cluster_start_frame_id": int(cluster_start),
                        "cluster_frame_count": int(source_frame_id - cluster_start + 1),
                        "native_kv_replay_start_frame_id": int(cluster_start),
                        "native_kv_replay_input_frame_count": int(source_frame_id - cluster_start + 1),
                        "context_start_frame_id": int(context_start),
                        "context_frame_count": int(source_frame_id - context_start + 1),
                        "valid_sample_count": int(valid_samples[source_frame_id]),
                        "frame_sample_count": frame_samples,
                        "frame_partial": int(valid_samples[source_frame_id]) < frame_samples,
                        "source": str(conversation.get("source", "unknown")),
                        "synthetic": _is_synthetic_source(conversation),
                        "profile": conversation.get("profile"),
                        "profile_version": conversation.get("profile_version"),
                        "action": action if action in ACTIONS else None,
                        "action_mask": bool(action in ACTIONS),
                        "target_action_mask": bool(frame.get("action_mask")),
                        "event_onset": frame.get("event_onset") is True,
                        "input_audio_start_sample": frame.get("input_audio_start_sample"),
                        "input_audio_end_sample": frame.get("input_audio_end_sample"),
                        "target_action_start_sample": frame.get("target_action_start_sample"),
                        "target_action_end_sample": frame.get("target_action_end_sample"),
                        "input_text_visible": frame.get("input_text_visible"),
                        "future_audio_excluded": frame.get("future_audio_excluded"),
                        "action_masked_reason": frame.get("action_masked_reason"),
                        "target_action_source": frame.get("target_action_source"),
                        "target_action_evidence": frame.get("target_action_evidence"),
                        "native_control6_hidden_source": (
                            "ListenerOutput.gemma_hidden_t"
                            if control6_profile == "native1540"
                            else None
                        ),
                        "native_control6_audio_queries_per_tick": (
                            2 if control6_profile == "native1540" else None
                        ),
                        "native_control6_mel_window_rows": (
                            240 if control6_profile == "native1540" else None
                        ),
                        **runtime_context,
                    }
                )
        finally:
            if native_session is not None:
                serving_native_worker.end_conversation()
            elif state is not None:
                state.close()
    return np.asarray(features, dtype=np.float32), metadata


def _cluster_causal_targets(
    target_frames: Sequence[Mapping[str, Any]],
    *,
    context_frames: int = 64,
    max_target_gap_frames: int = 64,
    max_cluster_frames: int = 128,
) -> list[tuple[int, list[Mapping[str, Any]]]]:
    """Cluster row-local targets without exceeding their causal replay bounds."""
    if context_frames <= 0 or max_target_gap_frames <= 0 or max_cluster_frames < context_frames:
        raise ValueError("limiti cluster stage2 non validi")
    ordered = sorted(target_frames, key=lambda frame: int(frame.get("frame_id", -1)))
    clusters: list[tuple[int, list[Mapping[str, Any]]]] = []
    cluster_start: int | None = None
    cluster_targets: list[Mapping[str, Any]] = []
    previous_target = -1
    for frame in ordered:
        frame_id = int(frame.get("frame_id", -1))
        if frame_id < 0 or frame_id <= previous_target:
            raise ValueError("target frame_id deve essere univoco, crescente e non negativo")
        proposed_start = max(0, frame_id - (context_frames - 1))
        if not cluster_targets:
            cluster_start = proposed_start
            cluster_targets = [frame]
        else:
            assert cluster_start is not None
            joins_cluster = (
                frame_id - previous_target <= max_target_gap_frames
                and frame_id - cluster_start + 1 <= max_cluster_frames
            )
            if joins_cluster:
                cluster_targets.append(frame)
            else:
                clusters.append((cluster_start, cluster_targets))
                cluster_start = proposed_start
                cluster_targets = [frame]
        previous_target = frame_id
    if cluster_targets:
        assert cluster_start is not None
        clusters.append((cluster_start, cluster_targets))
    return clusters


def _conversation_replay_clusters(
    target_frames: Sequence[Mapping[str, Any]],
) -> list[tuple[int, list[Mapping[str, Any]]]]:
    """Advance one native listener state from conversation frame zero to its last target."""
    ordered = sorted(target_frames, key=lambda frame: int(frame.get("frame_id", -1)))
    previous = -1
    for frame in ordered:
        frame_id = int(frame.get("frame_id", -1))
        if frame_id < 0 or frame_id <= previous:
            raise ValueError("conversation replay targets must be unique, increasing, and non-negative")
        previous = frame_id
    return [(0, ordered)] if ordered else []


@contextmanager
def _profile_native_listener_components(listener: Any, timing: dict[str, Any]):
    """Collect optional CPU front-end and CUDA component timings for one benchmark."""
    if not torch.cuda.is_available():
        yield
        return
    backend = getattr(listener, "backend", None)
    if backend is None:
        yield
        return
    events: dict[str, list[tuple[Any, Any]]] = {"audio_tower": [], "lm_forward": []}
    restorers: list[tuple[Any, str, bool, Any]] = []

    def timed_method(owner: Any, attribute: str, phase: str) -> None:
        original = getattr(owner, attribute, None)
        if not callable(original):
            return
        existed = attribute in vars(owner)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            stream = torch.cuda.current_stream()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                end_event.record(stream)
                events[phase].append((start_event, end_event))

        setattr(owner, attribute, wrapped)
        restorers.append((owner, attribute, existed, original))

    frontend = getattr(backend, "_extract_single_frame", None)
    if callable(frontend):
        existed = "_extract_single_frame" in vars(backend)

        def timed_frontend(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return frontend(*args, **kwargs)
            finally:
                timing["mel_frontend_cpu_seconds"] = float(
                    timing.get("mel_frontend_cpu_seconds", 0.0)
                ) + (time.perf_counter() - started)

        setattr(backend, "_extract_single_frame", timed_frontend)
        restorers.append((backend, "_extract_single_frame", existed, frontend))

    timed_method(backend, "_forward_incremental_lm", "lm_forward")
    gemma_model = getattr(backend, "gemma_model", None)
    if gemma_model is not None:
        timed_method(gemma_model, "get_audio_features", "audio_tower")

    graph = getattr(backend, "_cuda_graph", None)
    graph_stream = getattr(backend, "_graph_stream", None)
    if graph is not None and graph_stream is not None:
        existed = "_cuda_graph" in vars(backend)

        class _TimedGraph:
            def replay(self, *args: Any, **kwargs: Any) -> Any:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(graph_stream)
                try:
                    return graph.replay(*args, **kwargs)
                finally:
                    end_event.record(graph_stream)
                    events["audio_tower"].append((start_event, end_event))

        setattr(backend, "_cuda_graph", _TimedGraph())
        restorers.append((backend, "_cuda_graph", existed, graph))

    timing["_cuda_events"] = events
    try:
        yield
    finally:
        for owner, attribute, existed, original in reversed(restorers):
            if existed:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def _resolve_profile_events(timing: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    events = timing.pop("_cuda_events", {})
    for phase, pairs in events.items():
        values[f"{phase}_cuda_event_seconds"] = round(
            sum(float(start.elapsed_time(end)) for start, end in pairs) / 1000.0,
            6,
        )
        values[f"{phase}_cuda_event_calls"] = len(pairs)
    return values


def _public_phase_timings(timing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in timing.items()
        if not key.startswith("_")
    }


def benchmark_native_listener_windows(
    listener: Any,
    conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    windows: int = 8,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
    include_feature_vectors: bool = False,
    control6_profile: str = "historical_q4",
    serving_native_worker: _ServingNativeControl6Worker | Any | None = None,
) -> dict[str, Any]:
    """Measure real merged-model cost while replaying selected conversations from frame zero."""
    if windows <= 0:
        raise ValueError("windows benchmark deve essere positivo")
    conversations = [conversation] if isinstance(conversation, Mapping) else list(conversation)
    candidates: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for row in conversations:
        targets, _ = _causal_action_targets(row)
        if targets:
            candidates.append((row, targets))
    available = sum(len(targets) for _, targets in candidates)
    if not available:
        raise ValueError("nessun target audio supervisionato disponibile nel benchmark")
    target_count = min(int(windows), available)

    # Share the sample budget evenly across natural and synthetic rows when
    # both are supplied, reallocating any unused quota to rows with more labels.
    allocations = [0] * len(candidates)
    remaining = target_count
    while remaining:
        progressed = False
        for index, (_, targets) in enumerate(candidates):
            if allocations[index] >= len(targets):
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("benchmark non è riuscito ad assegnare tutti i target richiesti")

    def select_clustered(targets: list[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
        clusters = _cluster_causal_targets(targets)
        ordered_clusters = sorted(clusters, key=lambda item: (-len(item[1]), item[0]))
        selected: list[Mapping[str, Any]] = []
        for _, cluster_targets in ordered_clusters:
            take = min(count - len(selected), len(cluster_targets))
            if take == len(cluster_targets):
                selected.extend(cluster_targets)
            elif take > 1:
                start_index = max(0, (len(cluster_targets) - take) // 2)
                selected.extend(cluster_targets[start_index:start_index + take])
            elif take == 1:
                selected.append(cluster_targets[len(cluster_targets) // 2])
            if len(selected) >= count:
                break
        return sorted(selected[:count], key=lambda frame: int(frame["frame_id"]))

    selected_rows: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for (row, targets), allocation in zip(candidates, allocations):
        if not allocation:
            continue
        selected_rows.append((row, select_clustered(targets, allocation)))

    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
    else:
        allocated_before = 0
    start = time.perf_counter()
    metadata: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    phase_timings: dict[str, Any] = {}
    cache = pcm_cache if pcm_cache is not None else {}
    with _profile_native_listener_components(listener, phase_timings):
        for row, selected_targets in selected_rows:
            target_key = "frame_targets" if "frame_targets" in row else "frames"
            bounded_row = {**row, target_key: selected_targets}
            row_features, row_metadata = extract_conversation_features(
                listener,
                bounded_row,
                session_id=f"benchmark:{row.get('conversation_id', 'unknown')}",
                pcm_cache=cache,
                timing=phase_timings,
                control6_profile=control6_profile,
                serving_native_worker=serving_native_worker,
            )
            feature_rows.append(row_features)
            metadata.extend(row_metadata)
    if cuda_active:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    causal_steps = int(phase_timings.get("causal_input_frame_steps", 0))
    count = len(metadata)
    if count != target_count or causal_steps <= 0:
        raise RuntimeError("benchmark non ha estratto il numero richiesto di target/context frame")
    features = np.concatenate(feature_rows, axis=0)
    if not np.isfinite(features).all():
        raise RuntimeError("benchmark listener ha prodotto query non finite")
    per_window = elapsed / count
    per_causal_step = elapsed / causal_steps
    projected_windows = {"train": 768, "validation": 192, "test": 128}
    projected_seconds = {
        split: round(per_window * split_count, 3)
        for split, split_count in projected_windows.items()
    }
    allocated_peak = int(torch.cuda.max_memory_allocated()) if cuda_active else 0
    if cuda_active:
        phase_timings["cuda_event_seconds"] = _resolve_profile_events(phase_timings)
    selected_conversations = []
    for row, selected_targets in selected_rows:
        selected_conversations.append(
            {
                "conversation_id": str(row.get("conversation_id", "")),
                "source": str(row.get("source", "unknown")),
                "synthetic": _is_synthetic_source(row),
                "target_frame_ids": [int(frame["frame_id"]) for frame in selected_targets],
            }
        )
    report = {
        "conversations": selected_conversations,
        "windows": count,
        "selected_target_frame_ids": [int(row["frame_id"]) for row in metadata],
        "selected_target_frames": [
            {
                "conversation_id": row["conversation_id"],
                "source": row["source"],
                "synthetic": row["synthetic"],
                "frame_id": row["frame_id"],
                "context_frame_count": row["context_frame_count"],
            }
            for row in metadata
        ],
        "context_frame_counts": [int(row["context_frame_count"]) for row in metadata],
        "causal_input_frame_steps": causal_steps,
        "conversation_replays": int(phase_timings.get("conversation_replays", 0)),
        "tick_ms": 80,
        "input_channel_index": _channel_index(selected_rows[0][0], "user"),
        "input_channel_indices": [
            _channel_index(row, "user") for row, _ in selected_rows
        ],
        "elapsed_seconds": round(elapsed, 6),
        "seconds_per_target_window": round(per_window, 6),
        "seconds_per_causal_input_frame_step": round(per_causal_step, 6),
        "timings": _public_phase_timings(phase_timings),
        "projected_target_windows": projected_windows,
        "projected_seconds_by_split_at_sampled_window_rate": projected_seconds,
        "projected_total_target_window_seconds": round(sum(projected_seconds.values()), 3),
        "cuda_allocated_before_bytes": allocated_before,
        "cuda_peak_allocated_bytes": allocated_peak,
        "cuda_peak_delta_bytes": max(0, allocated_peak - allocated_before),
        "last_query_shape": [int(value) for value in features[-1].shape] if len(features) else None,
        "queries_finite": bool(np.isfinite(features).all()),
    }
    if include_feature_vectors:
        report["_feature_vectors"] = features
        report["_feature_metadata"] = metadata
    return report


def benchmark_serving_native_worker_frames(
    serving_native_worker: _ServingNativeControl6Worker,
    conversation: Mapping[str, Any],
    *,
    frames: int = 8,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> dict[str, Any]:
    """Warm the actual serving worker on a short real-PCM prefix and measure it."""
    if frames < 3 or frames > 8:
        raise ValueError("native worker smoke must cover between three and eight input frames")
    user_frames, assistant_frames, valid_samples = _audio_frames(
        conversation,
        frame_samples=1280,
        pcm_cache=pcm_cache,
    )
    frame_count = min(int(frames), len(user_frames))
    if frame_count < 3:
        raise ValueError("native worker smoke conversation has fewer than three real PCM frames")
    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
    else:
        allocated_before = 0
    step_ms: list[float] = []
    reference_frame_ids: list[int | None] = []
    shapes: list[dict[str, list[int]]] = []
    session_id: str | None = None
    started = time.perf_counter()
    try:
        session = serving_native_worker.begin_conversation(
            str(conversation.get("conversation_id", "native1540-smoke"))
        )
        session_id = str(session["session_id"])
        for frame_id in range(frame_count):
            reference_id = frame_id - 1 if assistant_frames is not None and frame_id > 0 else None
            reference_frame_ids.append(reference_id)
            frame_started = time.perf_counter()
            output = serving_native_worker.step(
                frame_id,
                user_frames[frame_id],
                None if reference_id is None else assistant_frames[reference_id],
                reference_id,
            )
            if int(output.frame_id) != frame_id:
                raise RuntimeError(
                    f"native smoke frame mismatch: expected={frame_id}, got={output.frame_id}"
                )
            hidden = getattr(output, "gemma_hidden_t", None)
            queries = getattr(output, "gemma_audio_queries_t", None)
            tower = getattr(output, "audio_tower_hidden", None)
            hidden_shape = tuple(getattr(hidden, "shape", ()))
            query_shape = tuple(getattr(queries, "shape", ()))
            tower_shape = tuple(getattr(tower, "shape", ()))
            if hidden_shape != (1, 1536) or query_shape != (2, 1536) or tower_shape != (2, 1536):
                raise RuntimeError(
                    "native smoke requires runtime hidden [1,1536] and effective Q2 [2,1536]; "
                    f"got {hidden_shape}/{query_shape}/{tower_shape}"
                )
            if not torch.equal(hidden, queries[0:1]):
                raise RuntimeError("native smoke gemma_hidden_t differs from Q2 output position zero")
            vector = hidden[0].detach().float().cpu().numpy()
            if vector.shape != (1536,) or not np.isfinite(vector).all():
                raise RuntimeError("native smoke produced a non-finite runtime hidden")
            step_ms.append((time.perf_counter() - frame_started) * 1000.0)
            shapes.append({
                "gemma_hidden_t": list(hidden_shape),
                "gemma_audio_queries_t": list(query_shape),
                "audio_tower_hidden": list(tower_shape),
            })
    finally:
        if session_id is not None:
            serving_native_worker.end_conversation()
    if cuda_active:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated()) if cuda_active else 0
    worker_report = serving_native_worker.report()
    return {
        "status": "passed",
        "conversation_id": str(conversation.get("conversation_id", "")),
        "source": str(conversation.get("source", "unknown")),
        "input_frames": frame_count,
        "first_frame_includes_prefix_import": True,
        "first_frame_wall_ms": round(step_ms[0], 3),
        "steady_frame_count": max(0, len(step_ms) - 1),
        "steady_frame_mean_wall_ms": round(float(np.mean(step_ms[1:])), 3) if len(step_ms) > 1 else None,
        "steady_frame_median_wall_ms": round(float(np.median(step_ms[1:])), 3) if len(step_ms) > 1 else None,
        "total_elapsed_seconds": round(elapsed, 6),
        "input_frames_per_second": round(frame_count / elapsed, 3) if elapsed > 0 else None,
        "frame_step_wall_ms": [round(value, 3) for value in step_ms],
        "assistant_reference_frame_ids": reference_frame_ids,
        "valid_samples_per_frame": [int(value) for value in valid_samples[:frame_count]],
        "output_shapes": shapes,
        "queries_finite": True,
        "cuda_allocated_before_bytes": allocated_before,
        "cuda_peak_allocated_bytes": peak,
        "cuda_peak_delta_bytes": max(0, peak - allocated_before),
        "serving_worker": worker_report,
    }


def _write_features(
    path: Path,
    features: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    input_dim: int = 1536,
) -> None:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != input_dim or len(features) != len(metadata):
        raise ValueError(
            f"listener features must align as [N,{input_dim}] with frame metadata"
        )
    if not np.isfinite(features).all():
        raise ValueError("native listener features contain NaN/Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [ACTIONS.index(row["action"]) if row.get("action_mask") else -100 for row in metadata],
        dtype=np.int64,
    )
    frame_ids = np.asarray([int(row["frame_id"]) for row in metadata], dtype=np.int64)
    listener_ids = np.asarray([int(row["listener_frame_id"]) for row in metadata], dtype=np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            labels=labels,
            frame_ids=frame_ids,
            listener_frame_ids=listener_ids,
        )
    temporary.replace(path)
    metadata_path = path.with_suffix(".json")
    temporary_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata_path.write_text(
        json.dumps(list(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary_metadata_path.replace(metadata_path)


def _conversation_partial_path(output_dir: Path, split: str, conversation_id: str) -> Path:
    safe_id = quote(str(conversation_id), safe="-_.")
    return output_dir / "conversation_partials" / split / f"{safe_id}.npz"


def _load_conversation_partial(
    path: Path,
    *,
    conversation: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    control6_profile: str,
) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    metadata_path = path.with_suffix(".json")
    if not path.exists() and not metadata_path.exists():
        return None
    if not path.is_file() or not metadata_path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            features = np.asarray(values["features"], dtype=np.float32).copy()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, list) or features.shape != (len(targets), 1536) or len(metadata) != len(targets):
        return None
    expected_frame_ids = [int(row["frame_id"]) for row in targets]
    observed_frame_ids = [int(row.get("frame_id", -1)) for row in metadata]
    if observed_frame_ids != expected_frame_ids:
        return None
    for expected, observed in zip(targets, metadata):
        expected_action = expected.get("target_action") if expected.get("action_mask") else None
        if (
            bool(observed.get("target_action_mask")) != bool(expected.get("action_mask"))
            or observed.get("action") != expected_action
            or observed.get("target_action_source") != expected.get("target_action_source")
            or observed.get("target_action_evidence") != expected.get("target_action_evidence")
        ):
            return None
    expected_group = str(conversation.get("source_group_id") or conversation["conversation_id"])
    if any(str(row.get("source_group_id")) != expected_group for row in metadata):
        return None
    for row in metadata:
        for key, value in conversation.items():
            if key.startswith("source_") or key in {
                "annotation_path",
                "control_supervision",
                "pairing_manifest_path",
                "duplexconv_provenance",
            }:
                if row.get(key) != value:
                    return None
    if control6_profile == "native1540" and any(
        row.get("native_control6_hidden_source") != "ListenerOutput.gemma_hidden_t"
        or row.get("native_control6_audio_queries_per_tick") != 2
        or row.get("native_control6_mel_window_rows") != 240
        for row in metadata
    ):
        return None
    if not np.isfinite(features).all():
        return None
    return features, [dict(row) for row in metadata]


def augment_control_features_with_acoustic_phase(
    *,
    native_feature_dir: str | Path,
    supplemental_native_feature_dir: str | Path | None = None,
    prepared_dir: str | Path,
    output_dir: str | Path,
    additional_prepared_dir: str | Path | None = None,
    splits: Sequence[str] = ("train", "validation"),
    mask_train_natural_backchannels_without_overlap: bool = True,
    frozen_head_path: str | Path | None = None,
    phase_profile: str = "candidate_c",
) -> dict[str, Any]:
    """Append four causal PCM features to saved native hidden vectors.

    Only the requested splits are read. This function never loads a model or
    recomputes native hidden vectors: Candidate C consumes its post-merge
    extraction, while native1540 consumes the serving base-only extraction.
    The historical ``candidate_c`` phase ABI remains the default; ``native1540``
    uses the shared runtime/training phase helper and carries its exact schema.
    """
    from duplex_kernel.acoustic_phase_features import (
        ACOUSTIC_PHASE_FEATURE_CONTRACT,
        acoustic_phase_features,
    )

    if phase_profile not in {"candidate_c", "native1540"}:
        raise ValueError("phase_profile must be 'candidate_c' or 'native1540'")

    native_dir = Path(native_feature_dir)
    supplemental_native_dir = (
        Path(supplemental_native_feature_dir) if supplemental_native_feature_dir else None
    )
    prepared_root = Path(prepared_dir)
    additional_root = Path(additional_prepared_dir) if additional_prepared_dir else None
    output = Path(output_dir)
    requested_splits = tuple(splits)
    if not requested_splits or any(split not in {"train", "validation", "test"} for split in requested_splits):
        raise ValueError("acoustic phase augmentation accepts train, validation, or held-out test")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("acoustic phase split list contains duplicates")
    if "test" in requested_splits:
        if requested_splits != ("test",):
            raise ValueError("held-out test augmentation must be requested by itself")
        if phase_profile == "candidate_c":
            if frozen_head_path is None:
                raise ValueError("Candidate C test augmentation requires a frozen Candidate C head")
            frozen_path = Path(frozen_head_path)
            if not frozen_path.is_file():
                raise FileNotFoundError(f"frozen Candidate C head is required before test access: {frozen_path}")
            frozen_head = load_control_head(frozen_path, device="cpu")
            if frozen_head.input_dim != 1540:
                raise ValueError("held-out Candidate C evaluation requires the frozen 1540-feature head")
            del frozen_head
    elif frozen_head_path is not None:
        raise ValueError("frozen_head_path is accepted only for held-out test evaluation")
    if output.exists():
        raise FileExistsError(f"Candidate C output must be a fresh directory: {output}")

    source_manifest_path = native_dir / "feature_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"native feature manifest missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if phase_profile == "candidate_c" and source_manifest.get("merged_before_native_listener") is not True:
        raise ValueError("Candidate C acoustic augmentation requires post-merge native listener features")
    source_contract = source_manifest.get("feature_contract")
    if phase_profile == "native1540":
        if source_manifest.get("serving_base_only") is not True:
            raise ValueError("native1540 augmentation requires exact serving base-only model features")
        native_extract = source_manifest.get("native_control6_extraction")
        if not isinstance(native_extract, Mapping) or native_extract.get("profile") != "native1540":
            raise ValueError("native1540 augmentation requires a fresh runtime-hidden Q2/W240 PCM extraction")
        if source_contract != native_control6_feature_contract(1536):
            raise ValueError("native1540 augmentation requires the exact current-frame gemma_hidden_t ABI")
        if (
            native_extract.get("hidden_source") != "ListenerOutput.gemma_hidden_t"
            or native_extract.get("gemma_audio_queries_per_tick") != 2
            or native_extract.get("audio_tower_queries_per_tick") != 2
            or native_extract.get("mel_window_rows_effective") != 240
        ):
            raise ValueError("native1540 source manifest does not attest effective Q2/W240 runtime extraction")
    elif source_contract != listener_feature_contract(1536):
        raise ValueError("Candidate C augmentation accepts only the saved historical native 1536 feature ABI")

    supplemental_manifest_path: Path | None = None
    supplemental_runtime_provenance: dict[str, Any] | None = None
    if supplemental_native_dir is not None:
        if phase_profile != "native1540":
            raise ValueError("supplemental runtime features are accepted only for native1540")
        supplemental_manifest_path = supplemental_native_dir / "feature_manifest.json"
        if not supplemental_manifest_path.is_file():
            raise FileNotFoundError(f"supplemental native feature manifest missing: {supplemental_manifest_path}")
        supplemental_manifest = json.loads(supplemental_manifest_path.read_text(encoding="utf-8"))
        if supplemental_manifest.get("feature_contract") != source_contract:
            raise ValueError("supplemental native features use a different feature schema")
        if (
            supplemental_manifest.get("serving_base_only") is not True
            or supplemental_manifest.get("merged_before_native_listener") is not False
            or supplemental_manifest.get("adapter_dir") is not None
        ):
            raise ValueError("supplemental native features do not use the serving base-only hidden ABI")
        supplemental_extract = supplemental_manifest.get("native_control6_extraction")
        if not isinstance(supplemental_extract, Mapping) or supplemental_extract.get("profile") != "native1540":
            raise ValueError("supplemental native features require the native1540 extraction profile")
        for label, manifest, extraction in (
            ("base", source_manifest, native_extract),
            ("supplement", supplemental_manifest, supplemental_extract),
        ):
            loader = manifest.get("serving_base_loader")
            if (
                manifest.get("serving_base_only") is not True
                or manifest.get("merged_before_native_listener") is not False
                or manifest.get("adapter_dir") is not None
                or not isinstance(loader, Mapping)
                or loader.get("model_dtype") != "bfloat16"
                or loader.get("stage1_adapter_applied") is not False
                or loader.get("base_model_path") != manifest.get("base_model_path")
                or extraction.get("base_model_path") != manifest.get("base_model_path")
                or extraction.get("stage1_adapter_applied") is not False
            ):
                raise ValueError(f"native1540 {label} features must use the same BF16 serving base without Stage1 adapter")
            if (
                extraction.get("hidden_source") != "ListenerOutput.gemma_hidden_t"
                or extraction.get("gemma_audio_queries_per_tick") != 2
                or extraction.get("audio_tower_queries_per_tick") != 2
                or extraction.get("mel_window_rows_effective") != 240
            ):
                raise ValueError(f"native1540 {label} features do not attest the Q2/W240 hidden ABI")
        if source_manifest.get("base_model_path") != supplemental_manifest.get("base_model_path"):
            raise ValueError("supplemental native features use a different serving base model")
        main_worker = (source_manifest.get("native_control6_extraction") or {}).get("serving_worker")
        extra_worker = (supplemental_manifest.get("native_control6_extraction") or {}).get("serving_worker")
        if not isinstance(main_worker, Mapping) or not isinstance(extra_worker, Mapping):
            raise ValueError("both feature manifests must identify the serving native worker")
        for key in ("worker", "prefix_builder", "native_library_path"):
            if main_worker.get(key) != extra_worker.get(key):
                raise ValueError(f"supplemental extraction differs from base runtime at {key}")
        if (
            main_worker.get("worker") != "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker"
            or main_worker.get("prefix_builder") != "duplex_kernel.server._build_unified_listener_prefix_token_ids"
        ):
            raise ValueError("native1540 extraction did not use the serving worker and unified prefix builder")
        main_signature = main_worker.get("runtime_signature")
        extra_signature = extra_worker.get("runtime_signature")
        if not isinstance(main_signature, Mapping) or main_signature != extra_signature:
            raise ValueError("supplemental extraction has a different native runtime signature")
        main_runtime = main_worker.get("runtime_info")
        extra_runtime = extra_worker.get("runtime_info")
        if not isinstance(main_runtime, Mapping) or not isinstance(extra_runtime, Mapping):
            raise ValueError("both feature manifests must identify native worker runtime settings")
        for runtime in (main_runtime, extra_runtime):
            if (
                runtime.get("library_path") != main_worker.get("native_library_path")
                or runtime.get("stream_q_len") != 2
                or runtime.get("unified_kv_enabled") is not True
            ):
                raise ValueError("native1540 extraction must use the same unified-KV Q2 native library")
        if (
            main_runtime.get("library_path") != extra_runtime.get("library_path")
            or main_runtime.get("stream_q_len") != extra_runtime.get("stream_q_len")
            or main_runtime.get("unified_kv_enabled") != extra_runtime.get("unified_kv_enabled")
        ):
            raise ValueError("supplemental extraction has different native runtime settings")
        if main_worker.get("prefix") != extra_worker.get("prefix"):
            raise ValueError("supplemental extraction used different serving prefix token IDs")
        prefix = main_worker.get("prefix")
        if (
            not isinstance(prefix, Mapping)
            or prefix.get("contract") != "serving_unified_chat_prefix_v2"
            or prefix.get("length") != 190
            or len(prefix.get("token_ids", ())) != 190
            or prefix.get("history_messages") != 0
            or prefix.get("session_instructions") != "empty_session_default"
        ):
            raise ValueError("native1540 extraction requires the shared empty-history 190-token serving prefix")
        supplemental_runtime_provenance = {
            "base": {
                key: main_worker.get(key)
                for key in (
                    "runtime_source_root",
                    "worker_factory_source",
                    "prefix_builder_source",
                    "native_library_path",
                )
            },
            "supplement": {
                key: extra_worker.get(key)
                for key in (
                    "runtime_source_root",
                    "worker_factory_source",
                    "prefix_builder_source",
                    "native_library_path",
                )
            },
        }

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def source_rows(root: Path, split: str) -> list[dict[str, Any]]:
        path = root / f"{split}.jsonl"
        return read_jsonl(path) if path.is_file() else []

    report: dict[str, Any] = {
        "status": "completed",
        "feature_contract": (
            native_control6_feature_contract(1540)
            if phase_profile == "native1540"
            else listener_feature_contract(1540)
        ),
        "source_feature_dir": str(native_dir.resolve()),
        "base_model_path": source_manifest.get("base_model_path"),
        "adapter_dir": source_manifest.get("adapter_dir"),
        "merged_before_native_listener": source_manifest.get("merged_before_native_listener") is True,
        "serving_base_only": source_manifest.get("serving_base_only") is True,
        "serving_base_loader": source_manifest.get("serving_base_loader"),
        "prepared_sources": [str(prepared_root.resolve())]
        + ([str(additional_root.resolve())] if additional_root else []),
        "supplemental_native_feature_dir": (
            str(supplemental_native_dir.resolve()) if supplemental_native_dir else None
        ),
        "supplemental_feature_rows_by_split": {},
        "prepared_target_refresh_by_split": {},
        **(
            {"acoustic_phase_contract": ACOUSTIC_PHASE_FEATURE_CONTRACT}
            if phase_profile == "candidate_c"
            else {}
        ),
        "causality": {
            "native_hidden_frame": "input_frame_id t",
            "user_pcm_history": "observed user PCM ending at input frame t; at most 400 ms; left-zero-padded at stream start",
            "assistant_pcm": "only already-emitted assistant reference frame t-1",
            "future_audio_or_supervision_used": False,
            "model_reextraction": False,
        },
        "heldout_evaluation_only": requested_splits == ("test",),
        "frozen_head_path": str(Path(frozen_head_path).resolve()) if frozen_head_path else None,
        "splits": {},
    }
    if phase_profile == "native1540":
        report["phase_profile"] = phase_profile
        report["training_state_context"] = NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
        report["phase_schema"] = {
            "profile": phase_profile,
            "width": len(NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES),
            "names": list(NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES),
        }
        report["phase_definition"] = dict(NATIVE_CONTROL6_NATIVE1540_PHASE_T_SCHEMA)
        report["native_control6_extraction"] = dict(source_manifest["native_control6_extraction"])
        if supplemental_runtime_provenance is not None:
            report["serving_runtime_provenance"] = supplemental_runtime_provenance
        report["causality"] = {
            "native_hidden_frame": "input_frame_id t",
            "user_pcm_current": "observed user PCM frame t",
            "user_pcm_delta_reference": "only user PCM frame t-1; absent at stream start",
            "assistant_pcm": "only already-emitted assistant reference frame t-1; absent at stream start",
            "future_audio_or_supervision_used": False,
            "model_reextraction": False,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_output = output.with_name(output.name + ".staging")
    if staging_output.exists():
        raise FileExistsError(f"stale Candidate C staging directory exists: {staging_output}")
    staging_output.mkdir(parents=False, exist_ok=False)
    total_masked_rows = 0
    total_verified_overlap_rows = 0
    masked_groups: set[str] = set()
    verified_overlap_groups: set[str] = set()
    for split in requested_splits:
        npz_path = native_dir / f"{split}.npz"
        metadata_path = native_dir / f"{split}.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"native feature split is incomplete: {split}")
        with np.load(npz_path, allow_pickle=False) as values:
            native_features = np.asarray(values["features"], dtype=np.float32)
            native_labels = np.asarray(values["labels"], dtype=np.int64)
            frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
            listener_ids = np.asarray(values["listener_frame_ids"], dtype=np.int64)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            native_features.ndim != 2
            or native_features.shape[1] != 1536
            or native_labels.shape != (len(native_features),)
            or frame_ids.shape != native_labels.shape
            or listener_ids.shape != native_labels.shape
            or len(metadata) != len(native_features)
        ):
            raise ValueError(f"native {split} feature arrays/metadata are not aligned at width 1536")
        if not np.isfinite(native_features).all():
            raise ValueError(f"native {split} hidden vectors contain NaN/Inf")
        base_metadata_count = len(metadata)

        prepared_by_id: dict[str, dict[str, Any]] = {}
        for root in (prepared_root, additional_root):
            if root is None:
                continue
            for conversation in source_rows(root, split):
                conversation_id = str(conversation.get("conversation_id", ""))
                if not conversation_id:
                    raise ValueError(f"{root}/{split}.jsonl contains a row without conversation_id")
                previous = prepared_by_id.get(conversation_id)
                if previous is not None:
                    if (
                        previous.get("audio_sha256") != conversation.get("audio_sha256")
                        or previous.get("audio_path") != conversation.get("audio_path")
                        or _is_synthetic_source(previous) != _is_synthetic_source(conversation)
                    ):
                        raise ValueError(f"conflicting prepared source rows for {conversation_id}")
                    # The additional prepared source has the uncapped V5
                    # timeline. For matching V5 audio identity it supersedes
                    # the capped target list in the base prepared directory.
                    if root == additional_root and _is_synthetic_source(conversation):
                        prepared_by_id[conversation_id] = conversation
                    continue
                prepared_by_id[conversation_id] = conversation

        temporal_view_flags = [
            isinstance(row.get("native1540_temporal_supervision_view"), Mapping)
            for row in prepared_by_id.values()
        ]
        temporal_supervision_view = bool(temporal_view_flags) and any(temporal_view_flags)
        if temporal_supervision_view and not all(temporal_view_flags):
            raise ValueError(f"{split} mixes temporal-view and legacy prepared rows")
        temporal_target_keys: list[tuple[str, int]] = []
        if temporal_supervision_view:
            for conversation_id, conversation in prepared_by_id.items():
                view = conversation.get("native1540_temporal_supervision_view")
                if not isinstance(view, Mapping) or view.get("schema") != "native1540_pcm_temporal_supervision_v1":
                    raise ValueError(f"{conversation_id}: unsupported native1540 temporal supervision view")
                targets, _ = _causal_action_targets(conversation, include_masked=True)
                temporal_target_keys.extend(
                    (conversation_id, int(target["frame_id"]))
                    for target in targets
                    if target.get("action_mask") is True and target.get("target_action") in ACTIONS
                )
            if len(temporal_target_keys) != len(set(temporal_target_keys)):
                raise ValueError(f"{split} temporal supervision view contains duplicate current target keys")

        if supplemental_native_dir is not None:
            supplement_npz = supplemental_native_dir / f"{split}.npz"
            supplement_json = supplemental_native_dir / f"{split}.json"
            if not supplement_npz.is_file() or not supplement_json.is_file():
                raise FileNotFoundError(f"supplemental native {split} split is incomplete")
            with np.load(supplement_npz, allow_pickle=False) as values:
                extra_features = np.asarray(values["features"], dtype=np.float32)
                extra_labels = np.asarray(values["labels"], dtype=np.int64)
                extra_frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
                extra_listener_ids = np.asarray(values["listener_frame_ids"], dtype=np.int64)
            extra_metadata = json.loads(supplement_json.read_text(encoding="utf-8"))
            if (
                extra_features.ndim != 2
                or extra_features.shape[1] != 1536
                or extra_labels.shape != (len(extra_features),)
                or extra_frame_ids.shape != extra_labels.shape
                or extra_listener_ids.shape != extra_labels.shape
                or len(extra_metadata) != len(extra_features)
                or not np.isfinite(extra_features).all()
            ):
                raise ValueError(f"supplemental native {split} arrays/metadata are malformed")
            seen_keys = {
                (str(row.get("conversation_id", "")), int(row.get("frame_id", -1)))
                for row in metadata
            }
            for index, row in enumerate(extra_metadata):
                key = (str(row.get("conversation_id", "")), int(row.get("frame_id", -1)))
                if not key[0] or key[1] < 0 or key in seen_keys:
                    raise ValueError(f"supplemental feature key is missing or already cached: {key}")
                if int(extra_frame_ids[index]) != key[1] or int(extra_listener_ids[index]) != int(row.get("listener_frame_id", -1)):
                    raise ValueError(f"supplemental feature frame arrays disagree at {key}")
                if row.get("action_mask") is not True or row.get("action") not in ACTIONS:
                    raise ValueError(f"supplemental feature must contain a supervised endpoint target at {key}")
                if int(extra_labels[index]) != ACTIONS.index(row["action"]):
                    raise ValueError(f"supplemental feature label array disagrees at {key}")
                seen_keys.add(key)
            native_features = np.concatenate((native_features, extra_features), axis=0)
            native_labels = np.concatenate((native_labels, extra_labels), axis=0)
            frame_ids = np.concatenate((frame_ids, extra_frame_ids), axis=0)
            listener_ids = np.concatenate((listener_ids, extra_listener_ids), axis=0)
            metadata = [*metadata, *extra_metadata]
            report["supplemental_feature_rows_by_split"][split] = len(extra_metadata)
        if temporal_supervision_view:
            current_keys = set(temporal_target_keys)
            indexed_keys: dict[tuple[str, int], int] = {}
            for index, row in enumerate(metadata):
                key = (str(row.get("conversation_id", "")), int(row.get("frame_id", -1)))
                if not key[0] or key[1] < 0 or key in indexed_keys:
                    raise ValueError(f"{split} native features contain an invalid or duplicate exact key: {key}")
                indexed_keys[key] = index
            base_keys = {
                (str(row.get("conversation_id", "")), int(row.get("frame_id", -1)))
                for row in metadata[:base_metadata_count]
            }
            supplement_keys = {
                (str(row.get("conversation_id", "")), int(row.get("frame_id", -1)))
                for row in metadata[base_metadata_count:]
            }
            expected_missing = current_keys - base_keys
            if supplement_keys != expected_missing:
                raise ValueError(
                    f"{split} supplement keys must equal missing current temporal targets "
                    f"(missing={len(expected_missing)}, supplement={len(supplement_keys)})"
                )
            if not current_keys.issubset(indexed_keys):
                missing = current_keys - set(indexed_keys)
                raise ValueError(f"{split} merged features are missing current temporal target keys: {len(missing)}")
            ordered_indices = [indexed_keys[key] for key in temporal_target_keys]
            native_features = native_features[ordered_indices]
            native_labels = native_labels[ordered_indices]
            frame_ids = frame_ids[ordered_indices]
            listener_ids = listener_ids[ordered_indices]
            metadata = [metadata[index] for index in ordered_indices]
        target_refresh: Counter[str] = Counter()
        source_supervised_rows = int(sum(bool(row.get("action_mask")) for row in metadata))
        source_per_action_support = {
            action: int(
                sum(bool(row.get("action_mask")) and row.get("action") == action for row in metadata)
            )
            for action in ACTIONS
        }
        by_conversation: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            conversation_id = str(row.get("conversation_id", ""))
            if conversation_id not in prepared_by_id:
                raise ValueError(f"no {split} prepared source row for feature conversation {conversation_id!r}")
            conversation = prepared_by_id[conversation_id]
            if str(row.get("source_group_id")) != str(
                conversation.get("source_group_id") or conversation_id
            ):
                raise ValueError(f"source_group_id mismatch for {conversation_id}")
            if _is_synthetic_source(row) != _is_synthetic_source(conversation):
                raise ValueError(f"synthetic/natural provenance mismatch for {conversation_id}")
            # Normalize convenience metadata from the authoritative source/profile.
            row["synthetic"] = _is_synthetic_source(conversation)
            input_frame_id = row.get("input_frame_id", row.get("frame_id"))
            if (
                isinstance(input_frame_id, bool)
                or not isinstance(input_frame_id, int)
                or int(row.get("frame_id", -1)) != input_frame_id
                or int(frame_ids[index]) != input_frame_id
                or int(listener_ids[index]) != int(row.get("listener_frame_id", -2))
            ):
                raise ValueError(f"saved native frame IDs disagree with their own metadata at {conversation_id}:{index}")
            assistant_reference_id = row.get("assistant_reference_frame_id")
            if input_frame_id == 0:
                if assistant_reference_id not in (None, -1):
                    raise ValueError(f"stream-start assistant reference must be absent at {conversation_id}:0")
            elif (
                isinstance(assistant_reference_id, bool)
                or not isinstance(assistant_reference_id, int)
                or assistant_reference_id != input_frame_id - 1
            ):
                raise ValueError(f"assistant reference is not the causal t-1 frame at {conversation_id}:{input_frame_id}")
            targets, is_v5 = _causal_action_targets(conversation, include_masked=True)
            target_by_input = {int(target["frame_id"]): target for target in targets}
            source_target = target_by_input.get(input_frame_id)
            if source_target is None:
                raise ValueError(f"no source frame target at {conversation_id}:{input_frame_id}")
            expected_action_mask = bool(
                source_target.get("action_mask") is True
                and source_target.get("target_action") in ACTIONS
            )
            expected_action = source_target.get("target_action") if expected_action_mask else None
            expected_label = ACTIONS.index(expected_action) if expected_action_mask else -100
            if (
                row.get("action") != expected_action
                or bool(row.get("action_mask")) != expected_action_mask
                or int(native_labels[index]) != expected_label
                or row.get("target_action_source") != source_target.get("target_action_source")
                or row.get("target_action_evidence") != source_target.get("target_action_evidence")
            ):
                if "native_control6_cached_action_before_target_view" not in row:
                    row["native_control6_cached_action_before_target_view"] = row.get("action")
                    row["native_control6_cached_action_mask_before_target_view"] = bool(row.get("action_mask"))
                    row["native_control6_cached_action_source_before_target_view"] = row.get("target_action_source")
                row["action"] = expected_action
                row["action_mask"] = expected_action_mask
                row["target_action_mask"] = expected_action_mask
                row["target_action_source"] = source_target.get("target_action_source")
                row["target_action_evidence"] = source_target.get("target_action_evidence")
                row["action_masked_reason"] = source_target.get("action_masked_reason")
                target_refresh["source_target_rows_refreshed"] += 1
                if expected_action_mask:
                    target_refresh[f"new_{expected_action}_targets"] += 1
                else:
                    target_refresh["targets_masked_by_current_prepared_view"] += 1
                native_labels[index] = expected_label
            if is_v5:
                if (
                    int(row.get("target_frame_id", -1)) != input_frame_id + 1
                    or int(source_target.get("target_frame_id", -1)) != input_frame_id + 1
                    or row.get("input_text_visible") is not False
                    or row.get("future_audio_excluded") is not True
                ):
                    raise ValueError(f"V5 causal target contract mismatch at {conversation_id}:{input_frame_id}")
            else:
                if int(row.get("target_frame_id", input_frame_id)) != input_frame_id:
                    raise ValueError(f"natural legacy source target is not same-index at {conversation_id}:{input_frame_id}")
            by_conversation.setdefault(conversation_id, []).append(index)

        augmented = np.empty((len(native_features), 1540), dtype=np.float32)
        augmented[:, :1536] = native_features
        result_metadata = [dict(row) for row in metadata]
        pcm_cache: dict[str, tuple[np.ndarray, int]] = {}
        verified_overlap_rows = 0
        masked_rows = 0
        runtime_target_derivations: Counter[str] = Counter()
        unknown_runtime_context_rows: Counter[str] = Counter()
        audio_hashes: dict[str, str] = {}
        cached_source_group: str | None = None
        for conversation_id, indices in by_conversation.items():
            conversation = prepared_by_id[conversation_id]
            current_source_group = str(conversation.get("source_group_id") or conversation_id)
            if current_source_group != cached_source_group:
                # Paired role views reuse one decode within a physical source
                # group; never retain corpus-wide PCM in memory.
                pcm_cache.clear()
                cached_source_group = current_source_group
            source_path_value = conversation.get("audio_path")
            if not source_path_value:
                raise ValueError(f"prepared source lacks audio_path: {conversation_id}")
            source_path = Path(str(source_path_value))
            if not source_path.is_file():
                raise FileNotFoundError(f"prepared source audio is missing: {source_path}")
            resolved_audio_path = str(source_path.resolve())
            if resolved_audio_path not in audio_hashes:
                actual_sha256 = file_sha256(source_path)
                declared_sha256 = conversation.get("audio_sha256")
                if declared_sha256 and actual_sha256 != declared_sha256:
                    raise ValueError(f"prepared source audio SHA-256 mismatch: {source_path}")
                audio_hashes[resolved_audio_path] = actual_sha256
            user_frames, assistant_frames, _ = _audio_frames(
                conversation, frame_samples=1280, pcm_cache=pcm_cache
            )
            user_channel = _channel_index(conversation, "user")
            if user_channel is None:
                raise ValueError(f"prepared source has no explicit user channel: {conversation_id}")
            for index in indices:
                row = result_metadata[index]
                input_frame_id = int(row.get("input_frame_id", row["frame_id"]))
                if input_frame_id >= len(user_frames):
                    raise ValueError(f"input frame extends beyond PCM source: {conversation_id}:{input_frame_id}")
                history_start = max(0, input_frame_id - 4)
                user_history = user_frames[history_start : input_frame_id + 1].reshape(-1)
                if len(user_history) < 5 * 1280:
                    user_history = np.pad(user_history, (5 * 1280 - len(user_history), 0))
                assistant_previous = (
                    assistant_frames[input_frame_id - 1]
                    if assistant_frames is not None and input_frame_id > 0
                    else None
                )
                if phase_profile == "native1540":
                    user_previous = user_frames[input_frame_id - 1] if input_frame_id > 0 else None
                    phase = native_control6_native1540_phase_t(
                        listener_frame_id=input_frame_id,
                        user_pcm_current=user_frames[input_frame_id],
                        user_pcm_previous=user_previous,
                        user_previous_frame_id=input_frame_id - 1 if user_previous is not None else None,
                        assistant_pcm_prior=assistant_previous,
                        assistant_prior_frame_id=input_frame_id - 1 if assistant_previous is not None else None,
                    )
                    row["native_control6_phase_profile"] = phase_profile
                    row["native_control6_phase_schema"] = report["phase_schema"]
                    row["training_state_context"] = NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
                else:
                    phase = acoustic_phase_features(
                        user_history,
                        assistant_previous,
                        sample_rate_hz=16000,
                        tick_ms=80,
                    )
                augmented[index, 1536:] = phase

                if phase_profile == "native1540":
                    source_targets, _ = _causal_action_targets(conversation, include_masked=True)
                    source_target = next(
                        (target for target in source_targets if int(target["frame_id"]) == input_frame_id),
                        None,
                    )
                    if source_target is None:
                        raise ValueError(
                            f"no source target available for native runtime context at "
                            f"{conversation_id}:{input_frame_id}"
                        )
                    runtime_context = _native1540_supervision_context(
                        conversation,
                        source_target,
                        frame_id=input_frame_id,
                        user_pcm_current=user_frames[input_frame_id],
                        assistant_pcm_prior=(
                            assistant_previous
                            if input_frame_id > 0 and assistant_previous is not None
                            else None
                        ),
                        assistant_prior_frame_id=(input_frame_id - 1 if input_frame_id > 0 and assistant_previous is not None else None),
                    )
                    row.update(runtime_context)
                    speaker_active = runtime_context.get("speaker_active_at_frame_start")
                    context_valid = type(speaker_active) is bool
                    row["native_control6_context_valid"] = context_valid
                    if not context_valid:
                        unknown_runtime_context_rows[str(row.get("source", "unknown"))] += 1
                    runtime_target = _native1540_runtime_target_action(
                        conversation,
                        source_target,
                        frame_id=input_frame_id,
                        speaker_active=speaker_active,
                    )
                    row["native_control6_runtime_target_action"] = runtime_target["action"]
                    row["native_control6_runtime_target_derived"] = runtime_target["derived"]
                    row["native_control6_runtime_target_reason"] = runtime_target["reason"]
                    row["native_control6_runtime_target_segment_index"] = runtime_target["segment_index"]
                    row["native_control6_runtime_target_event_id"] = runtime_target["event_id"]
                    if runtime_target["yield_onset_frame_id"] is not None:
                        row["native_control6_runtime_target_yield_onset_frame_id"] = runtime_target["yield_onset_frame_id"]
                        row["native_control6_runtime_target_yield_local_stop_ms"] = runtime_target["yield_local_stop_ms"]
                        row["native_control6_runtime_target_yield_local_segment_id"] = runtime_target["yield_local_segment_id"]
                        row["native_control6_runtime_target_yield_peer_segment_id"] = runtime_target["yield_peer_segment_id"]
                    if runtime_target["derived"]:
                        runtime_target_derivations[str(runtime_target["reason"])] += 1

                if (
                    phase_profile == "candidate_c"
                    and split == "train"
                    and mask_train_natural_backchannels_without_overlap
                    and not _is_synthetic_source(row)
                    and row.get("action") == "BACKCHANNEL"
                ):
                    source_targets, is_v5 = _causal_action_targets(conversation)
                    if is_v5:
                        raise ValueError("natural BACKCHANNEL training mask unexpectedly matched synthetic V5 source")
                    source_frame = next(
                        (target for target in source_targets if int(target["frame_id"]) == input_frame_id),
                        None,
                    )
                    if (
                        source_frame is None
                        or source_frame.get("target_action") != "BACKCHANNEL"
                        or source_frame.get("target_action_source") != "verified_native_assistant_action"
                    ):
                        raise ValueError(f"natural BACKCHANNEL provenance is not verified at {conversation_id}:{input_frame_id}")
                    tick_start = float(source_frame.get("start_ms", input_frame_id * 80))
                    tick_end = float(source_frame.get("end_ms", (input_frame_id + 1) * 80))
                    if not tick_end > tick_start:
                        raise ValueError(f"natural BACKCHANNEL source interval invalid at {conversation_id}:{input_frame_id}")
                    onset_segments = []
                    overlapping_user_segments = []
                    ambiguous_user_segments = []
                    for segment in conversation.get("segments", ()) or ():
                        if not isinstance(segment, Mapping):
                            raise ValueError(f"source segment is not an object in {conversation_id}")
                        try:
                            start_ms = float(segment["start_ms"])
                            end_ms = float(segment["end_ms"])
                        except (KeyError, TypeError, ValueError) as exc:
                            if segment.get("role") in {"assistant", "user"}:
                                raise ValueError(f"role segment lacks a valid time interval in {conversation_id}") from exc
                            continue
                        if not end_ms > start_ms:
                            raise ValueError(f"role segment has an invalid interval in {conversation_id}")
                        if (
                            segment.get("role") == "assistant"
                            and segment.get("native_turn_state") == "backchannel"
                            and int(start_ms // 80) == input_frame_id
                        ):
                            if (
                                segment.get("audio_channel_verified") is not True
                                or segment.get("time_mask") is not True
                                or segment.get("channel_index") != _channel_index(conversation, "assistant")
                            ):
                                raise ValueError(f"assistant BACKCHANNEL onset lacks verified channel/time in {conversation_id}")
                            onset_segments.append(segment)
                        if end_ms > tick_start and start_ms < tick_end:
                            if segment.get("role") == "user":
                                if (
                                    segment.get("audio_channel_verified") is not True
                                    or segment.get("time_mask") is not True
                                    or segment.get("channel_index") != user_channel
                                ):
                                    ambiguous_user_segments.append(segment)
                                else:
                                    overlapping_user_segments.append(segment)
                            elif (
                                segment.get("role") not in {"assistant", None}
                                or segment.get("channel_index") == user_channel
                            ):
                                ambiguous_user_segments.append(segment)
                    if len(onset_segments) != 1:
                        raise ValueError(
                            f"expected one verified native BACKCHANNEL onset at {conversation_id}:{input_frame_id}, found {len(onset_segments)}"
                        )
                    if ambiguous_user_segments:
                        raise ValueError(f"cannot safely classify user overlap at {conversation_id}:{input_frame_id}")
                    source_group = str(row.get("source_group_id") or conversation_id)
                    if overlapping_user_segments:
                        verified_overlap_rows += 1
                        verified_overlap_groups.add(source_group)
                    else:
                        row["action_mask"] = False
                        row["action_masked_reason"] = "natural_backchannel_without_verified_remote_overlap_at_source_onset"
                        masked_rows += 1
                        masked_groups.add(source_group)
        if not np.isfinite(augmented).all():
            raise ValueError(f"augmented {split} feature matrix contains NaN/Inf")
        report["prepared_target_refresh_by_split"][split] = dict(target_refresh)
        if phase_profile == "native1540":
            if not all(
                row.get("native_control6_context", {}).get("schema") == "native_control6_runtime_context_v1"
                and row.get("training_state_context") == NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
                for row in result_metadata
            ):
                raise RuntimeError(f"native1540 context metadata is incomplete in {split}")
            expected_labels = np.asarray(
                [ACTIONS.index(row["action"]) if row.get("action_mask") else -100 for row in result_metadata],
                dtype=np.int64,
            )
            if not np.array_equal(native_labels, expected_labels):
                raise RuntimeError(f"native1540 refreshed labels do not match the current prepared view in {split}")
            report.setdefault("runtime_target_derivations_by_split", {})[split] = {
                "derived_rows_by_reason": dict(sorted(runtime_target_derivations.items())),
                "unknown_pre_action_context_rows_by_source": dict(sorted(unknown_runtime_context_rows.items())),
                "raw_action_labels_preserved": True,
            }
        _write_features(staging_output / f"{split}.npz", augmented, result_metadata, input_dim=1540)
        report["splits"][split] = {
            "rows": int(len(augmented)),
            "cached_metadata_supervised_rows_before_prepared_refresh": source_supervised_rows,
            "cached_metadata_action_support_before_prepared_refresh": source_per_action_support,
            "prepared_supervised_rows": int(sum(bool(row.get("action_mask")) for row in result_metadata)),
            "supervised_rows": int(sum(bool(row.get("action_mask")) for row in result_metadata)),
            "prepared_action_support": {
                action: int(sum(bool(row.get("action_mask")) and row.get("action") == action for row in result_metadata))
                for action in ACTIONS
            },
            "prepared_supervision_aligned": (
                phase_profile == "native1540"
                and int(sum(bool(row.get("action_mask")) for row in result_metadata))
                == int(sum(label >= 0 for label in expected_labels))
            ),
            "audio_source_files": len(audio_hashes),
            "audio_source_sha256": audio_hashes,
        }
        if phase_profile == "candidate_c":
            # Preserve the Candidate C report keys for historical comparisons.
            report["splits"][split]["source_supervised_rows"] = source_supervised_rows
            report["splits"][split]["source_per_action_support"] = source_per_action_support
            report["splits"][split]["per_action_support"] = dict(
                report["splits"][split]["prepared_action_support"]
            )
        if split == "train" and phase_profile == "candidate_c":
            total_masked_rows += masked_rows
            total_verified_overlap_rows += verified_overlap_rows
            report["natural_train_backchannel_masking"] = {
                "enabled": bool(mask_train_natural_backchannels_without_overlap),
                "verified_overlap_rows": int(verified_overlap_rows),
                "masked_rows": int(masked_rows),
                "verified_overlap_groups": len(verified_overlap_groups),
                "masked_groups": len(masked_groups),
                "remaining_supervised_natural_backchannel_rows": int(
                    sum(
                        bool(row.get("action_mask"))
                        and row.get("action") == "BACKCHANNEL"
                        and not _is_synthetic_source(row)
                        for row in result_metadata
                    )
                ),
                "criterion": (
                    "training-only ignore_index for verified natural assistant BACKCHANNEL onset rows "
                    "without a verified user audio/time overlap in the same 80 ms source interval; "
                    "action provenance string is preserved"
                ),
            }

    report["split_scope"] = list(requested_splits)
    report["test_data_read"] = "test" in requested_splits
    report["test_used_for_training_or_calibration"] = False
    manifest_path = staging_output / "feature_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(staging_output, output)
    return report


def merge_native1540_feature_dirs(
    *,
    base_feature_dir: str | Path,
    supplemental_feature_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Append a disjoint native1540 train split and preserve base validation bytes."""
    import shutil

    base = Path(base_feature_dir)
    supplement = Path(supplemental_feature_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"native1540 merged output must be fresh: {output}")
    staging = output.with_name(output.name + ".staging")
    if staging.exists():
        raise FileExistsError(f"stale native1540 merge staging directory exists: {staging}")

    def read_manifest(root: Path) -> dict[str, Any]:
        path = root / "feature_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"native1540 feature manifest missing: {path}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("test_data_read") is True or manifest.get("test_used_for_training_or_calibration") is True:
            raise ValueError(f"native1540 merge refuses test-accessed features: {root}")
        if (
            manifest.get("feature_contract") != native_control6_feature_contract(1540)
            or manifest.get("phase_profile") != "native1540"
            or manifest.get("phase_schema") != {
                "profile": "native1540",
                "width": len(NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES),
                "names": list(NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES),
            }
            or manifest.get("serving_base_only") is not True
            or manifest.get("merged_before_native_listener") is not False
        ):
            raise ValueError(f"native1540 merge requires exact serving-base 1540 features: {root}")
        return manifest

    base_manifest = read_manifest(base)
    supplement_manifest = read_manifest(supplement)
    for key in ("feature_contract", "phase_profile", "phase_schema", "phase_definition", "base_model_path"):
        if base_manifest.get(key) != supplement_manifest.get(key):
            raise ValueError(f"native1540 feature ABI mismatch at manifest field {key}")
    if base_manifest.get("serving_base_loader") != supplement_manifest.get("serving_base_loader"):
        raise ValueError("native1540 serving base loader differs between merge inputs")

    def extraction_signature(manifest: Mapping[str, Any]) -> tuple[Any, ...]:
        extract = manifest.get("native_control6_extraction")
        if not isinstance(extract, Mapping):
            raise ValueError("native1540 merge input lacks runtime extraction provenance")
        worker = extract.get("serving_worker")
        if not isinstance(worker, Mapping):
            raise ValueError("native1540 merge input lacks serving worker provenance")
        prefix = worker.get("prefix")
        signature = worker.get("runtime_signature")
        if not isinstance(prefix, Mapping) or not isinstance(signature, Mapping):
            raise ValueError("native1540 merge input lacks exact runtime prefix/signature")
        return (
            extract.get("profile"),
            extract.get("hidden_source"),
            extract.get("hidden_contract"),
            extract.get("gemma_audio_queries_per_tick"),
            extract.get("audio_tower_queries_per_tick"),
            extract.get("mel_window_rows_effective"),
            extract.get("base_model_path"),
            extract.get("response_lora_applied"),
            extract.get("stage1_adapter_applied"),
            worker.get("worker"),
            worker.get("worker_factory_source"),
            worker.get("prefix_builder"),
            worker.get("prefix_builder_source"),
            worker.get("runtime_source_root"),
            worker.get("native_library_path"),
            dict(signature),
            prefix.get("contract"),
            prefix.get("token_ids"),
            prefix.get("length"),
        )

    if extraction_signature(base_manifest) != extraction_signature(supplement_manifest):
        raise ValueError("native1540 serving worker, DSO, prefix, or runtime signature differs")

    def load_split(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        npz_path = root / f"{split}.npz"
        json_path = root / f"{split}.json"
        if not npz_path.is_file() or not json_path.is_file():
            raise FileNotFoundError(f"native1540 {split} split incomplete in {root}")
        with np.load(npz_path, allow_pickle=False) as values:
            features = np.asarray(values["features"], dtype=np.float32)
            labels = np.asarray(values["labels"], dtype=np.int64)
            frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
            listener_ids = np.asarray(values["listener_frame_ids"], dtype=np.int64)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        n = len(features)
        if (
            features.ndim != 2 or features.shape[1] != 1540
            or labels.shape != (n,) or frame_ids.shape != (n,) or listener_ids.shape != (n,)
            or not isinstance(metadata, list) or len(metadata) != n
            or not np.isfinite(features).all()
        ):
            raise ValueError(f"native1540 {split} arrays/metadata are not aligned at width 1540: {root}")
        expected_labels: list[int] = []
        seen: set[tuple[str, int]] = set()
        group_by_conversation: dict[str, str] = {}
        last_frame_by_conversation: dict[str, int] = {}
        for index, row in enumerate(metadata):
            if not isinstance(row, dict):
                raise ValueError(f"native1540 {split} metadata row {index} is not an object")
            conversation_id = row.get("conversation_id")
            group_id = row.get("source_group_id")
            frame = row.get("frame_id")
            listener_frame = row.get("listener_frame_id")
            if (
                not isinstance(conversation_id, str) or not conversation_id
                or not isinstance(group_id, str) or not group_id
                or isinstance(frame, bool) or not isinstance(frame, int)
                or isinstance(listener_frame, bool) or not isinstance(listener_frame, int)
                or int(frame_ids[index]) != frame or int(listener_ids[index]) != listener_frame
                or row.get("split", split) != split
                or type(row.get("action_mask")) is not bool
            ):
                raise ValueError(f"native1540 {split} metadata provenance/IDs invalid at row {index}")
            key = (conversation_id, frame)
            if key in seen:
                raise ValueError(f"duplicate native1540 {split} feature key {key}")
            seen.add(key)
            previous_group = group_by_conversation.setdefault(conversation_id, group_id)
            if previous_group != group_id:
                raise ValueError(f"mixed source groups within {conversation_id}")
            previous_frame = last_frame_by_conversation.get(conversation_id, -1)
            if frame <= previous_frame:
                raise ValueError(f"native1540 frames are not source/frame ordered in {conversation_id}")
            last_frame_by_conversation[conversation_id] = frame
            action = row.get("action")
            if row["action_mask"]:
                if action not in ACTIONS:
                    raise ValueError(f"native1540 supervised row has invalid action at {key}")
                expected_labels.append(ACTIONS.index(action))
            else:
                expected_labels.append(-100)
        if not np.array_equal(labels, np.asarray(expected_labels, dtype=np.int64)):
            raise ValueError(f"native1540 {split} labels disagree with metadata in {root}")
        return features, labels, frame_ids, listener_ids, metadata

    base_train = load_split(base, "train")
    extra_train = load_split(supplement, "train")
    base_validation = load_split(base, "validation")
    base_keys = {(r["conversation_id"], int(r["frame_id"])) for r in base_train[4]}
    extra_keys = {(r["conversation_id"], int(r["frame_id"])) for r in extra_train[4]}
    duplicate_keys = base_keys & extra_keys
    if duplicate_keys:
        raise ValueError(f"native1540 supplemental train repeats {len(duplicate_keys)} conversation/frame keys")
    base_groups = {r["source_group_id"] for r in base_train[4]}
    validation_groups = {r["source_group_id"] for r in base_validation[4]}
    extra_groups = {r["source_group_id"] for r in extra_train[4]}
    if base_groups & extra_groups or validation_groups & extra_groups:
        raise ValueError("native1540 supplemental train source groups overlap existing train/validation")

    staging.mkdir(parents=True, exist_ok=False)
    merged_features = np.concatenate((base_train[0], extra_train[0]), axis=0)
    merged_metadata = base_train[4] + extra_train[4]
    _write_features(staging / "train.npz", merged_features, merged_metadata, input_dim=1540)
    for name in ("validation.npz", "validation.json"):
        shutil.copyfile(base / name, staging / name)

    manifest = dict(base_manifest)
    manifest["source_feature_dir"] = str(base.resolve())
    manifest["supplemental_native_feature_dir"] = str(supplement.resolve())
    manifest["supplemental_feature_rows_by_split"] = {"train": len(extra_train[4]), "validation": 0}
    manifest["merged_feature_rows_by_split"] = {
        "train": len(merged_metadata),
        "validation": len(base_validation[4]),
    }
    manifest["feature_merge"] = {
        "status": "completed",
        "format": "stage2_native1540",
        "train_base_rows": len(base_train[4]),
        "train_supplement_rows": len(extra_train[4]),
        "train_rows": len(merged_metadata),
        "validation_rows": len(base_validation[4]),
        "validation_source": str(base.resolve()),
        "validation_copied_unchanged": True,
        "source_group_sets_disjoint": True,
        "test_data_read": False,
    }
    manifest["test_data_read"] = False
    manifest["test_used_for_training_or_calibration"] = False
    splits = dict(manifest.get("splits") or {})
    train_summary = dict(splits.get("train") or {})
    action_counts = Counter(
        str(row["action"]) for row in merged_metadata if row.get("action_mask") is True
    )
    source_rows = Counter(str(row.get("source", "unknown")) for row in merged_metadata)
    train_summary.update({
        "rows": len(merged_metadata),
        "supervised_rows": int(sum(action_counts.values())),
        "prepared_supervised_rows": int(sum(action_counts.values())),
        "prepared_action_support": {action: int(action_counts[action]) for action in ACTIONS},
        "source_rows": dict(sorted(source_rows.items())),
        "source_groups": len(base_groups | extra_groups),
    })
    splits["train"] = train_summary
    manifest["splits"] = splits
    (staging / "feature_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    return manifest["feature_merge"]

def _clustered_target_context_workload(conversations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count full causal conversation replay steps before native extraction."""
    windows = 0
    causal_steps = 0
    conversation_replays = 0
    max_conversation_frames = 0
    for conversation in conversations:
        targets, _ = _causal_action_targets(conversation)
        clusters = _conversation_replay_clusters(targets)
        windows += len(targets)
        conversation_replays += len(clusters)
        for cluster_start, cluster_targets in clusters:
            cluster_steps = int(cluster_targets[-1]["frame_id"]) - cluster_start + 1
            causal_steps += cluster_steps
            max_conversation_frames = max(max_conversation_frames, cluster_steps)
    return {
        "target_windows": windows,
        "causal_input_frame_steps": causal_steps,
        "conversation_replays": conversation_replays,
        "max_conversation_replay_input_frames": max_conversation_frames,
    }


def _stage2_extraction_splits(
    control6_profile: str,
    splits: Sequence[str] | None,
) -> tuple[str, ...]:
    if control6_profile not in {"historical_q4", "native1540"}:
        raise ValueError("control6_profile must be 'historical_q4' or 'native1540'")
    requested = tuple(
        splits
        if splits is not None
        else ("train", "validation")
        if control6_profile == "native1540"
        else ("train", "validation", "test")
    )
    if not requested or any(split not in {"train", "validation", "test"} for split in requested):
        raise ValueError("Stage2 extraction splits must be train, validation, or test")
    if len(set(requested)) != len(requested):
        raise ValueError("Stage2 extraction split list contains duplicates")
    return requested


def _v5_event_onset_selection_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score V5 event onsets and false event outputs on true L/C background.

    Event-action interiors remain valid action labels. The legacy rate is
    included for continuity with earlier reports, but is explicitly not a
    false-action metric because it counts correct SPEAK/BACKCHANNEL interiors.
    """
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("V5 selection targets/predictions must be aligned 1D arrays")
    if len(metadata) != len(targets):
        raise ValueError("V5 selection metadata/targets must be aligned")
    if np.any((targets < 0) | (targets >= len(ACTIONS))):
        raise ValueError("V5 selection targets must contain only supervised action indices")
    if np.any((predictions < 0) | (predictions >= len(ACTIONS))):
        raise ValueError("V5 selection predictions contain an action index outside the ABI")

    v5_positions = [
        index
        for index, row in enumerate(metadata)
        if row.get("profile") == "audio_grounded_v5"
        or row.get("source") == "authored_synthetic_audio_candidate"
    ]
    onset_positions = [
        index for index in v5_positions if metadata[index].get("event_onset") is True
    ]
    non_onset_positions = [
        index for index in v5_positions if metadata[index].get("event_onset") is not True
    ]
    background_indices = {ACTIONS.index("LISTEN"), ACTIONS.index("CONTINUE")}
    background_positions = [
        index
        for index in non_onset_positions
        if int(targets[index]) in background_indices
    ]
    event_output_actions = ("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")
    event_output_indices = {ACTIONS.index(action) for action in event_output_actions}

    def summarize(positions: Sequence[int]) -> dict[str, Any]:
        matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
        for position in positions:
            matrix[int(targets[position]), int(predictions[position])] += 1
        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)
        per_class: dict[str, Any] = {}
        f1_values: list[float] = []
        recalls: list[float] = []
        for class_index, action in enumerate(ACTIONS):
            true_positive = int(matrix[class_index, class_index])
            recall = float(true_positive / support[class_index]) if support[class_index] else None
            precision = (
                float(true_positive / predicted_support[class_index])
                if predicted_support[class_index]
                else 0.0
            )
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if recall is not None and precision + recall
                else 0.0
            )
            per_class[action] = {
                "support": int(support[class_index]),
                "precision": precision,
                "recall": recall,
                "f1": f1 if support[class_index] else None,
            }
            if support[class_index]:
                f1_values.append(f1)
                recalls.append(float(recall))
        examples = len(positions)
        correct = int(np.trace(matrix))
        return {
            "examples": int(examples),
            "accuracy": float(correct / examples) if examples else None,
            "macro_f1": float(np.mean(f1_values)) if f1_values else None,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "class_support": {action: int(support[index]) for index, action in enumerate(ACTIONS)},
            "per_class": per_class,
            "confusion_matrix": {
                "true_actions": list(ACTIONS),
                "predicted_actions": list(ACTIONS),
                "values": matrix.tolist(),
            },
        }

    background_predictions = predictions[background_positions]
    background_false_counts = {
        action: int(np.count_nonzero(background_predictions == ACTIONS.index(action)))
        for action in event_output_actions
    }
    legacy_predictions = predictions[non_onset_positions]
    legacy_event_outputs = int(
        np.isin(legacy_predictions, tuple(event_output_indices)).sum()
    )
    return {
        "v5_examples": int(len(v5_positions)),
        "v5_full_six_class": summarize(v5_positions),
        "event_onset_examples": int(len(onset_positions)),
        "event_onset_macro_f1": summarize(onset_positions)["macro_f1"],
        "event_onsets_six_class": summarize(onset_positions),
        "background_definition": (
            "supervised V5 rows outside event_schedule onsets whose target action is LISTEN or CONTINUE; "
            "CONTINUE event-onset rows and SPEAK/BACKCHANNEL interiors are excluded"
        ),
        "background_examples": int(len(background_positions)),
        "background_false_event_predictions": int(sum(background_false_counts.values())),
        "background_false_event_rate": (
            float(sum(background_false_counts.values()) / len(background_positions))
            if background_positions
            else None
        ),
        "background_false_event_predictions_by_class": background_false_counts,
        "event_action_outputs": list(event_output_actions),
        "legacy_off_onset_examples": int(len(non_onset_positions)),
        "legacy_off_onset_event_output_predictions": legacy_event_outputs,
        "legacy_off_onset_event_output_rate_not_false_action_rate": (
            float(legacy_event_outputs / len(non_onset_positions))
            if non_onset_positions
            else None
        ),
        "legacy_metric_note": (
            "counts event-class outputs on every non-onset target, including genuine SPEAK/BACKCHANNEL "
            "interiors; descriptive only and not a false-action rate"
        ),
    }


def _natural_validation_macro_f1(
    head: ControlHead,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    train_class_support: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Score checkpoint selection only on labeled real four-action validation rows."""
    natural_actions = ("LISTEN", "SPEAK", "CONTINUE", "BACKCHANNEL")
    action_indices = [ACTIONS.index(action) for action in natural_actions]
    positions = [
        index
        for index, row in enumerate(metadata)
        if int(labels[index]) in action_indices and not _is_synthetic_source(row)
    ]
    if not positions:
        raise ValueError("checkpoint selection richiede esempi naturali etichettati in validation")
    selected_labels = labels[positions].astype(np.int64, copy=False)
    was_training = bool(head.training)
    head.eval()
    try:
        with torch.inference_mode():
            predictions = head(torch.from_numpy(features[positions])).argmax(dim=-1).cpu().numpy()
    finally:
        if was_training:
            head.train()
    row_index = {action_index: index for index, action_index in enumerate(action_indices)}
    matrix = np.zeros((len(natural_actions), len(ACTIONS)), dtype=np.int64)
    for target, prediction in zip(selected_labels, predictions):
        matrix[row_index[int(target)], int(prediction)] += 1
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    correct_predictions = sum(
        int(matrix[index, ACTIONS.index(action)])
        for index, action in enumerate(natural_actions)
    )
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, action in enumerate(natural_actions):
        global_index = ACTIONS.index(action)
        tp = int(matrix[index, global_index])
        recall = float(tp / support[index]) if support[index] else None
        precision = float(tp / predicted_support[global_index]) if predicted_support[global_index] else 0.0
        f1 = float(
            2 * precision * recall / (precision + recall)
        ) if recall is not None and precision + recall else 0.0
        per_class[action] = {
            "support": int(support[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1 if support[index] else None,
        }
        if support[index]:
            f1_values.append(f1)
            recalls.append(float(recall))
    examples = int(len(selected_labels))
    present_indices = [index for index, value in enumerate(support) if value]
    present_actions = [natural_actions[index] for index in present_indices]
    majority_train_action = max(
        present_actions,
        key=lambda action: int((train_class_support or {}).get(action, 0)),
        default=None,
    )
    majority_accuracy = (
        float(support[natural_actions.index(majority_train_action)] / examples)
        if majority_train_action is not None
        else None
    )
    true_counts = support[present_indices]
    return {
        "examples": examples,
        "accuracy": float(correct_predictions / examples),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "actions": list(natural_actions),
        "class_support": {action: int(support[index]) for index, action in enumerate(natural_actions)},
        "absent_classes": [
            action for index, action in enumerate(natural_actions) if not support[index]
        ],
        "chance_baselines": {
            "uniform_random_over_present_classes_accuracy": float(1.0 / len(present_indices)),
            "empirical_prior_random_expected_accuracy": float(
                np.square(true_counts).sum() / (examples * examples)
            ),
        },
        "majority_baseline_from_train_prior": {
            "action": majority_train_action,
            "accuracy": majority_accuracy,
            "training_class_support": {
                action: int((train_class_support or {}).get(action, 0))
                for action in natural_actions
            },
        },
        "per_class": per_class,
        "confusion_matrix": {
            "true_actions": list(natural_actions),
            "predicted_actions": list(ACTIONS),
            "values": matrix.tolist(),
        },
    }


def extract_stage2_features(
    *,
    prepared_dir: str | Path,
    base_model_path: str | None = None,
    adapter_dir: str | None = None,
    output_dir: str | Path,
    device: str = "cuda",
    benchmark_only: bool = False,
    benchmark_then_extract_if_under_hours: float | None = None,
    verify_prefix_cache: bool = False,
    control6_profile: str = "historical_q4",
    runtime_workspace: str | Path = "/workspace",
    native_microbatch_library_path: str | Path | None = None,
    native_smoke_frames: int = 8,
    splits: Sequence[str] | None = None,
    resume_existing: bool = False,
) -> dict[str, Any]:
    """Extract historical merged features or serving-aligned native1540 features."""
    if control6_profile not in {"historical_q4", "native1540"}:
        raise ValueError("control6_profile must be 'historical_q4' or 'native1540'")
    requested_splits = _stage2_extraction_splits(control6_profile, splits)
    if control6_profile == "historical_q4":
        if base_model_path is None or adapter_dir is None:
            raise ValueError("historical_q4 extraction requires --base-model-path and --adapter-dir")
        if not Path(adapter_dir).is_dir():
            raise FileNotFoundError(f"adapter LoRA finale mancante: {adapter_dir}")
    else:
        if adapter_dir is not None:
            raise ValueError("native1540 extraction is serving base-only; omit --adapter-dir")
        if verify_prefix_cache:
            raise ValueError("native1540 uses the serving unified chat prefix, not canonical prefix-cache verification")
    if benchmark_then_extract_if_under_hours is not None and benchmark_then_extract_if_under_hours <= 0:
        raise ValueError("la soglia auto-extract deve essere positiva")
    serving_base_report: dict[str, Any] | None = None
    serving_native_worker: _ServingNativeControl6Worker | None = None
    if control6_profile == "native1540":
        thinker, processor, model, serving_base_report = load_serving_base(
            workspace=runtime_workspace,
            base_model_path=base_model_path,
            device=device,
        )
    else:
        assert base_model_path is not None and adapter_dir is not None
        thinker, processor, model = load_merged_base(base_model_path, adapter_dir, device=device)
    from duplex_kernel.listener import GemmaListener

    listener = GemmaListener.from_thinker(thinker, runtime_profile="production")
    listener_backend = getattr(listener, "backend", None)
    if control6_profile == "native1540":
        from duplex_kernel.gemma_representation import AUDIO_FEATURES_PER_FRAME
        from duplex_kernel.listener_microbatch_frontend import MEL_WINDOW_ROWS

        effective_rows = int(getattr(listener_backend, "mel_window_rows", -1))
        if AUDIO_FEATURES_PER_FRAME != 2 or MEL_WINDOW_ROWS != 240 or effective_rows != 240:
            raise RuntimeError(
                "native1540 extraction requires effective production Q2/W240; "
                f"constants={AUDIO_FEATURES_PER_FRAME}/{MEL_WINDOW_ROWS}, backend={effective_rows}"
            )
        serving_native_worker = _ServingNativeControl6Worker(
            thinker,
            listener,
            workspace=runtime_workspace,
            native_library_path=native_microbatch_library_path,
        )
    try:
        configure_prefix_cache = getattr(listener_backend, "configure_canonical_prefix_cache", None)
        prefix_cache_stats = getattr(listener_backend, "canonical_prefix_cache_stats", None)
        if verify_prefix_cache and not callable(configure_prefix_cache):
            raise RuntimeError("verifica cache prefisso richiede il backend Gemma nativo cache-aware")
        if control6_profile != "native1540" and callable(configure_prefix_cache):
            configure_prefix_cache(enabled=False, clear=True)
        output = Path(output_dir)
        if output.exists() and any(output.iterdir()) and not (resume_existing or benchmark_only):
            raise FileExistsError(
                f"Stage2 output is non-empty; pass resume_existing=True to reuse conversation partials: {output}"
            )
        report: dict[str, Any] = {
            "feature_contract": (
                native_control6_feature_contract(1536)
                if control6_profile == "native1540"
                else dict(FEATURE_CONTRACT)
            ),
            "base_model_path": str(
                Path(base_model_path).resolve()
                if base_model_path is not None
                else Path(str(serving_base_report["base_model_path"])).resolve()
            ),
            "adapter_dir": str(Path(adapter_dir).resolve()) if adapter_dir is not None else None,
            "merged_before_native_listener": control6_profile != "native1540",
            "serving_base_only": control6_profile == "native1540",
            "serving_base_loader": serving_base_report,
            "extractor": (
                "GemmaListener.production + serving shared native listener worker"
                if control6_profile == "native1540"
                else "GemmaListener.production"
            ),
            "state_reset": (
                "one serving session per conversation; default empty history/instructions; replay continuously from absolute frame 0"
                if control6_profile == "native1540"
                else "one GemmaListenerState per conversation; reset only at conversation boundary, replay from absolute frame 0 through the last supervised input"
            ),
            "cluster_contract": {
                "state_resets_per_conversation": 1,
                "replay_start_frame_id": 0,
                "advance_continuously_through_last_target": True,
                "reset_between_target_groups": False,
                "rolling_frontend_pcm_mel_context_frames": 64,
                "native_lm_cache_capacity_and_rollover": "listener production defaults; derived from the loaded native backend",
            },
            "audio_resampling": {
                "method": "scipy.signal.resample_poly",
                "source_rates_supported": "any positive PCM rate",
                "target_rate_hz": 16000,
                "tick_ms": 80,
                "partial_frame": "zero padded; valid_sample_count is a compact prefix mask",
            },
            "sparse_frame_extraction": "store explicitly supervised targets; for each conversation replay input from absolute frame zero through its last selected target",
            "counts": {},
            "extracted_splits": list(requested_splits),
            "test_data_read": "test" in requested_splits,
            "test_used_for_training_or_calibration": False,
        }
        if control6_profile == "native1540":
            report["training_state_context"] = NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
            report["native_control6_extraction"] = {
                "profile": control6_profile,
                "hidden_source": "ListenerOutput.gemma_hidden_t",
                "hidden_contract": "gemma4_listener_final_norm_frame_v1",
                "hidden_norm": "gemma_model.language_model.norm",
                "feature_vector": "runtime current-frame hidden at Q2 LM output position 0; exactly equal to gemma_audio_queries_t[0]",
                "gemma_audio_queries_per_tick": 2,
                "audio_tower_queries_per_tick": 2,
                "mel_window_rows_effective": int(listener_backend.mel_window_rows),
                "tick_ms": 80,
                "assistant_reference": "only emitted PCM frame t-1",
                "base_model_path": str(Path(str(serving_base_report["base_model_path"])).resolve()),
                "response_lora_applied": False,
                "stage1_adapter_applied": False,
                "hidden_frame_contract": "ListenerOutput.gemma_hidden_t for input audio frame t",
            }
        split_conversations: dict[str, list[dict[str, Any]]] = {}
        for split in requested_splits:
            input_path = Path(prepared_dir) / f"{split}.jsonl"
            if input_path.is_file():
                split_conversations[split] = read_jsonl(input_path)
            elif (
                split == "test"
                and control6_profile == "native1540"
                and splits is not None
                and "test" in splits
            ):
                raise FileNotFoundError(f"explicitly requested held-out test split is missing: {input_path}")
        report["test_data_read"] = "test" in split_conversations
        workload = {
            split: _clustered_target_context_workload(rows)
            for split, rows in split_conversations.items()
        }
        report["planned_target_context_workload"] = workload

        training_conversations = split_conversations.get("train", [])
        benchmark_real = next(
            (
                row for row in training_conversations
                if not _is_synthetic_source(row)
                and row.get("audio_path")
                and Path(str(row["audio_path"])).is_file()
                and (
                    control6_profile == "native1540" and benchmark_only
                    or bool(_causal_action_targets(row)[0])
                )
            ),
            None,
        )
        benchmark_synthetic = next(
            (
                row for row in training_conversations
                if _is_synthetic_source(row)
                and row.get("audio_path")
                and Path(str(row["audio_path"])).is_file()
                and bool(_causal_action_targets(row)[0])
            ),
            None,
        )
        benchmark_conversations = [
            row for row in (benchmark_real, benchmark_synthetic) if row is not None
        ]
        benchmark_groups = {
            str(row.get("source_group_id") or row["conversation_id"])
            for row in benchmark_conversations
        }
        benchmark_pcm_cache: dict[str, tuple[np.ndarray, int]] = {}
        if benchmark_real is not None:
            if control6_profile == "native1540" and benchmark_only:
                benchmark = benchmark_serving_native_worker_frames(
                    serving_native_worker,
                    benchmark_real,
                    frames=native_smoke_frames,
                    pcm_cache=benchmark_pcm_cache,
                )
            elif verify_prefix_cache:
                cold = benchmark_native_listener_windows(
                    listener,
                    benchmark_conversations,
                    windows=8,
                    pcm_cache=benchmark_pcm_cache,
                    include_feature_vectors=True,
                    control6_profile=control6_profile,
                    serving_native_worker=serving_native_worker,
                )
                configure_prefix_cache(enabled=True, clear=True)
                benchmark = benchmark_native_listener_windows(
                    listener,
                    benchmark_conversations,
                    windows=8,
                    pcm_cache=benchmark_pcm_cache,
                    include_feature_vectors=True,
                    control6_profile=control6_profile,
                    serving_native_worker=serving_native_worker,
                )
                cold_features = cold.pop("_feature_vectors")
                cold_metadata = cold.pop("_feature_metadata")
                cached_features = benchmark.pop("_feature_vectors")
                cached_metadata = benchmark.pop("_feature_metadata")
                if len(cold_metadata) != len(cached_metadata) or any(
                    any(
                        cold_row.get(key) != cached_row.get(key)
                        for key in (
                            "conversation_id", "source_group_id", "frame_id", "listener_frame_id",
                            "cluster_start_frame_id", "cluster_frame_count", "context_start_frame_id",
                            "context_frame_count", "valid_sample_count", "action",
                        )
                    )
                    for cold_row, cached_row in zip(cold_metadata, cached_metadata)
                ):
                    raise RuntimeError("cache prefisso ha alterato target, causal context o confini stream")
                if cold_features.shape != cached_features.shape:
                    raise RuntimeError("shape query cached/cold prefisso non coincide")
                max_abs_difference = float(np.max(np.abs(cold_features - cached_features)))
                backend_stats = prefix_cache_stats()
                expected_clusters = int(benchmark["conversation_replays"])
                if (
                    not np.isfinite(cached_features).all()
                    or max_abs_difference > 1e-3
                    or int(backend_stats["prefix_compute_count"]) != 1
                    or int(backend_stats["prefix_reuse_count"]) != expected_clusters - 1
                ):
                    raise RuntimeError(
                        "parità/cache prefisso non valida: "
                        f"max_abs_diff={max_abs_difference}, stats={backend_stats}, clusters={expected_clusters}"
                    )
                benchmark["canonical_prefix_cache_verification"] = {
                    "status": "passed",
                    "cold_target_windows": len(cold_features),
                    "cached_target_windows": len(cached_features),
                    "cold_and_cached_metadata_identical": True,
                    "query_shape": list(cached_features.shape),
                    "queries_finite": True,
                    "max_abs_query_difference": max_abs_difference,
                    "absolute_tolerance": 1e-3,
                    "conversation_replays": expected_clusters,
                    "canonical_prefix_backend_stats": backend_stats,
                    "native_state_reset_between_conversations": True,
                    "causal_targets_and_context_bounds_preserved": True,
                }
            else:
                if control6_profile != "native1540" and callable(configure_prefix_cache):
                    configure_prefix_cache(enabled=True, clear=True)
                benchmark = benchmark_native_listener_windows(
                    listener,
                    benchmark_conversations,
                    windows=8,
                    pcm_cache=benchmark_pcm_cache,
                    control6_profile=control6_profile,
                    serving_native_worker=serving_native_worker,
                )
            if callable(prefix_cache_stats) and control6_profile != "native1540":
                benchmark["canonical_prefix_cache_after_benchmark"] = prefix_cache_stats()
            if not (control6_profile == "native1540" and benchmark_only):
                benchmark_steps_per_second = float(benchmark["seconds_per_causal_input_frame_step"])
                benchmark["projected_seconds_by_split_at_measured_context_rate"] = {
                    split: round(
                        benchmark_steps_per_second * float(split_workload["causal_input_frame_steps"]),
                        3,
                    )
                    for split, split_workload in workload.items()
                }
                benchmark["projected_total_seconds_at_measured_context_rate"] = round(
                    sum(benchmark["projected_seconds_by_split_at_measured_context_rate"].values()),
                    3,
                )
                benchmark["projection_basis"] = (
                    "elapsed benchmark time divided by the exact causal context advances for sampled targets; "
                    + (
                        "uses the serving worker and unified first-session prefix; "
                        if control6_profile == "native1540"
                        else "includes per-target steady-state priming; "
                    )
                    + "includes one PCM decode/resample per sampled conversation"
                )
            report["native_listener_benchmark"] = benchmark
            if serving_native_worker is not None:
                report["native_control6_extraction"]["serving_worker"] = serving_native_worker.report()
        else:
            if control6_profile == "native1540" and benchmark_only:
                raise RuntimeError("native1540 smoke requires a real train conversation with an audio file")
            report["native_listener_benchmark"] = {"status": "no_real_train_conversation_available"}
        if benchmark_only:
            report["status"] = (
                "serving_native_worker_smoke_only"
                if control6_profile == "native1540"
                else "native_listener_benchmark_only"
            )
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "benchmark_manifest.json"
            report["benchmark_manifest_path"] = str(report_path.resolve())
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            if serving_native_worker is not None:
                serving_native_worker.close()
            del processor, model
            return report
        if benchmark_then_extract_if_under_hours is not None:
            benchmark = report["native_listener_benchmark"]
            projected_seconds = benchmark.get("projected_total_seconds_at_measured_context_rate")
            if projected_seconds is None:
                raise RuntimeError("auto-extract richiede un benchmark cluster nativo reale")
            threshold_seconds = float(benchmark_then_extract_if_under_hours) * 3600.0
            report["benchmark_auto_extract_gate"] = {
                "projected_seconds": float(projected_seconds),
                "threshold_seconds": threshold_seconds,
                "proceed": float(projected_seconds) <= threshold_seconds,
                "same_merged_model_reused_for_extraction": True,
            }
            output.mkdir(parents=True, exist_ok=True)
            benchmark_path = output / "benchmark_manifest.json"
            report["benchmark_manifest_path"] = str(benchmark_path.resolve())
            benchmark_snapshot = dict(report)
            benchmark_snapshot["status"] = "native_listener_benchmark_auto_gate"
            benchmark_path.write_text(
                json.dumps(benchmark_snapshot, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if float(projected_seconds) > threshold_seconds:
                report["status"] = "benchmark_projection_over_auto_extract_limit"
                if serving_native_worker is not None:
                    serving_native_worker.close()
                del processor, model
                return report
            report["benchmark_auto_extract_gate"]["proceeded_in_same_process"] = True
        for split in requested_splits:
            conversations = split_conversations.get(split)
            if conversations is None:
                continue
            output.mkdir(parents=True, exist_ok=True)
            processed: dict[int, tuple[np.ndarray, list[dict[str, Any]]]] = {}
            grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
            eligible_rows = 0
            skipped_unlabeled_rows = 0
            for conversation_index, conversation in enumerate(conversations):
                group = str(conversation.get("source_group_id") or conversation["conversation_id"])
                grouped.setdefault(group, []).append((conversation_index, conversation))
            for group_conversations in grouped.values():
                # Retain decoded/resampled PCM only while paired role views of one
                # physical recording are advanced through the causal listener.
                group_key = str(
                    group_conversations[0][1].get("source_group_id")
                    or group_conversations[0][1]["conversation_id"]
                )
                pcm_cache = benchmark_pcm_cache if split == "train" and group_key in benchmark_groups else {}
                for conversation_index, conversation in group_conversations:
                    targets = _causal_action_targets(conversation)[0]
                    if not any(frame.get("action_mask") is True for frame in targets):
                        skipped_unlabeled_rows += 1
                        continue
                    eligible_rows += 1
                    conversation_id = str(conversation["conversation_id"])
                    partial_path = _conversation_partial_path(output, split, conversation_id)
                    if resume_existing:
                        cached = _load_conversation_partial(
                            partial_path,
                            conversation=conversation,
                            targets=targets,
                            control6_profile=control6_profile,
                        )
                        if cached is not None:
                            processed[conversation_index] = cached
                            print(
                                f"[stage2] resume split={split} conversation={conversation_index + 1}/{len(conversations)} "
                                f"id={conversation_id} targets={len(cached[1])}",
                                flush=True,
                            )
                            continue
                    values, metadata = extract_conversation_features(
                        listener,
                        conversation,
                        session_id=f"{split}:{conversation['conversation_id']}",
                        pcm_cache=pcm_cache,
                        control6_profile=control6_profile,
                        serving_native_worker=serving_native_worker,
                    )
                    if len(metadata) != len(targets):
                        raise RuntimeError(
                            f"Stage2 conversation target count changed during extraction: {conversation_id}"
                        )
                    _write_features(partial_path, values, metadata)
                    processed[conversation_index] = (values, metadata)
                    causal_steps = int(metadata[-1]["frame_id"]) + 1 if metadata else 0
                    print(
                        f"[stage2] saved split={split} conversation={conversation_index + 1}/{len(conversations)} "
                        f"id={conversation_id} targets={len(metadata)} causal_steps={causal_steps}",
                        flush=True,
                    )
            ordered_indices = sorted(processed)
            split_features = [processed[index][0] for index in ordered_indices]
            split_metadata = [
                row
                for index in ordered_indices
                for row in processed[index][1]
            ]
            features = np.concatenate(split_features, axis=0) if split_features else np.empty((0, 1536), dtype=np.float32)
            _write_features(output / f"{split}.npz", features, split_metadata)
            report["counts"][split] = {
                "conversations": len(conversations),
                "conversation_partials_saved": len(processed),
                "conversations_with_masked_targets": eligible_rows,
                "conversations_without_masked_targets_skipped": skipped_unlabeled_rows,
                "conversation_partials_dir": str((output / "conversation_partials" / split).resolve()),
                "frames": len(split_metadata),
                "labeled_frames": sum(bool(row["action_mask"]) for row in split_metadata),
                "causal_input_frame_steps": int(
                    workload.get(split, {}).get("causal_input_frame_steps", 0)
                ),
                "conversation_replays": int(workload.get(split, {}).get("conversation_replays", 0)),
                "per_target_context_frame_sum": sum(
                    int(row["context_frame_count"]) for row in split_metadata
                ),
            }
        if callable(prefix_cache_stats):
            report["canonical_prefix_cache_after_extraction"] = prefix_cache_stats()
        report_path = output / "feature_manifest.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if serving_native_worker is not None:
            serving_native_worker.close()
        del processor, model
        return report

    finally:
        if serving_native_worker is not None:
            serving_native_worker.close()


def train_control_head(
    *,
    feature_dir: str | Path,
    output_path: str | Path,
    training_manifest_path: str | Path,
    max_steps: int = 500,
    learning_rate: float = 1e-3,
    architecture: str = "linear",
    class_weighting: str = "none",
    event_onset_loss_multiplier: float = 1.0,
    save_infeasible_diagnostic_checkpoint: bool = False,
    validation_checkpoint_dir: str | Path | None = None,
    selection_metric: str | None = None,
    selection_interval: int = 25,
    sampling_strategy: str = "class_uniform",
    max_validation_v5_background_false_event_rate: float | None = None,
    early_stopping_patience: int | None = None,
    minimum_steps: int = 0,
    seed: int = 20260930,
) -> dict[str, Any]:
    """Fit the six-class head; unknown/unresolved action labels use ignore_index."""
    feature_dir = Path(feature_dir)
    train_path = feature_dir / "train.npz"
    if not train_path.is_file():
        raise FileNotFoundError(f"feature stage2 train mancanti: {train_path}")
    feature_manifest_path = feature_dir / "feature_manifest.json"
    if not feature_manifest_path.is_file():
        raise FileNotFoundError(f"feature stage2 manifest mancante: {feature_manifest_path}")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("merged_before_native_listener") is not True:
        raise ValueError("stage2 rifiuta feature non estratte dopo il merge LoRA")
    feature_contract = feature_manifest.get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise ValueError("feature stage2 privo di un feature_contract valido")
    input_dim = feature_contract.get("input_dim")
    try:
        expected_feature_contract = listener_feature_contract(input_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature stage2 input_dim non supportato: {input_dim!r}") from exc
    if feature_contract != expected_feature_contract:
        raise ValueError("feature stage2 non rispettano il contratto native listener dichiarato")
    train_metadata_path = feature_dir / "train.json"
    if not train_metadata_path.is_file():
        raise FileNotFoundError(f"metadata frame/session stage2 mancanti: {train_metadata_path}")
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    with np.load(train_path, allow_pickle=False) as values:
        train_array = values["features"].copy()
        train_label_array = values["labels"].copy()
        train_frame_ids = values["frame_ids"].copy()
        train_listener_ids = values["listener_frame_ids"].copy()
    if train_array.ndim != 2 or train_array.shape[1] != input_dim:
        raise ValueError(
            f"features stage2 devono essere [N,{input_dim}], ricevuto {train_array.shape}"
        )
    if len(train_metadata) != len(train_label_array) or train_frame_ids.shape != train_label_array.shape or train_listener_ids.shape != train_label_array.shape:
        raise ValueError("feature/label/metadata/frame IDs stage2 non sono allineati")
    if not np.isfinite(train_array).all():
        raise ValueError("features stage2 contengono NaN/Inf")
    if np.any((train_label_array < -100) | (train_label_array >= len(ACTIONS))):
        raise ValueError("stage2 label index fuori range")
    for index, row in enumerate(train_metadata):
        if int(row.get("frame_id", -1)) != int(train_frame_ids[index]):
            raise ValueError("stage2 frame_id metadata non combacia con NPZ")
        if int(row.get("listener_frame_id", -2)) != int(train_listener_ids[index]):
            raise ValueError("stage2 listener_frame_id metadata non combacia con NPZ")
        if int(train_label_array[index]) >= 0:
            expected_action = ACTIONS[int(train_label_array[index])]
            if not bool(row.get("action_mask")) or row.get("action") != expected_action:
                raise ValueError("stage2 label non corrisponde all'azione esplicita/mask metadata")
    labels = torch.from_numpy(train_label_array).long()
    features = torch.from_numpy(train_array)
    valid = labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("nessuna action supervision esplicita nel train split stage2")
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    if architecture not in {
        "linear",
        "mlp_256_gelu_dropout_0.1",
        "native_phase_fusion_mlp",
    }:
        raise ValueError(f"architettura control head non supportata: {architecture!r}")
    if class_weighting not in {"none", "sqrt_inverse_frequency"}:
        raise ValueError(f"class weighting non supportato: {class_weighting!r}")
    if not np.isfinite(float(event_onset_loss_multiplier)) or event_onset_loss_multiplier <= 0:
        raise ValueError("event_onset_loss_multiplier deve essere finito e positivo")
    if save_infeasible_diagnostic_checkpoint and selection_metric != "validation_v5_event_onset_macro_f1":
        raise ValueError("il checkpoint diagnostico infeasible si applica solo alla selezione V5 event-onset")
    if validation_checkpoint_dir is not None and selection_metric != "validation_v5_event_onset_macro_f1":
        raise ValueError("i checkpoint di stato validation richiedono la selezione V5 event-onset")
    if selection_metric not in {
        None,
        "validation_natural_macro_f1",
        "validation_v5_event_onset_macro_f1",
    }:
        raise ValueError(f"metrica di selezione checkpoint non supportata: {selection_metric!r}")
    if selection_metric is not None and selection_interval <= 0:
        raise ValueError("selection_interval deve essere positivo")
    if sampling_strategy not in {"class_uniform", "empirical_rows"}:
        raise ValueError(f"sampling strategy non supportata: {sampling_strategy!r}")
    if selection_metric == "validation_v5_event_onset_macro_f1":
        if (
            max_validation_v5_background_false_event_rate is None
            or not 0 <= float(max_validation_v5_background_false_event_rate) <= 1
        ):
            raise ValueError(
                "selezione V5 event-onset richiede un limite background false-event tra 0 e 1"
            )
    elif max_validation_v5_background_false_event_rate is not None:
        raise ValueError("il limite V5 false-event si applica solo alla selezione V5 event-onset")
    if early_stopping_patience is not None:
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience deve essere positivo")
        if minimum_steps < 0 or minimum_steps > max_steps:
            raise ValueError("minimum_steps deve essere tra zero e max_steps")
    elif minimum_steps:
        raise ValueError("minimum_steps richiede early_stopping_patience")

    # Ensure whole conversation/source groups remain in a single split.
    metadata_by_split: dict[str, list[dict[str, Any]]] = {"train": train_metadata}
    feature_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (train_array, train_label_array)
    }
    for split in ("validation", "test"):
        array_path = feature_dir / f"{split}.npz"
        json_path = feature_dir / f"{split}.json"
        if not array_path.is_file() or not json_path.is_file():
            continue
        split_metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(array_path, allow_pickle=False) as values:
            split_features = values["features"].copy()
            split_labels = values["labels"].copy()
            frame_ids = values["frame_ids"].copy()
            listener_ids = values["listener_frame_ids"].copy()
        if split_features.ndim != 2 or split_features.shape[1] != input_dim or not np.isfinite(split_features).all():
            raise ValueError(f"{split} stage2 features hanno shape o valori non validi")
        if len(split_metadata) != len(split_labels) or frame_ids.shape != split_labels.shape or listener_ids.shape != split_labels.shape:
            raise ValueError(f"{split} stage2 feature/label/metadata non allineati")
        for index, row in enumerate(split_metadata):
            if int(row.get("frame_id", -1)) != int(frame_ids[index]) or int(row.get("listener_frame_id", -2)) != int(listener_ids[index]):
                raise ValueError(f"{split} stage2 frame IDs non allineati")
            if int(split_labels[index]) >= 0:
                expected_action = ACTIONS[int(split_labels[index])]
                if not bool(row.get("action_mask")) or row.get("action") != expected_action:
                    raise ValueError(f"{split} label non corrisponde all'azione esplicita/mask metadata")
        metadata_by_split[split] = split_metadata
        feature_arrays[split] = (split_features, split_labels)
    group_split: dict[str, str] = {}
    for split, rows in metadata_by_split.items():
        for row in rows:
            group = str(row.get("source_group_id") or row.get("conversation_id") or "")
            if not group:
                raise ValueError(f"{split} stage2 metadata privo di conversation/source_group id")
            previous = group_split.setdefault(group, split)
            if previous != split:
                raise ValueError(f"stage2 conversation group leakage: {group} in {previous} e {split}")
    if selection_metric == "validation_v5_event_onset_macro_f1":
        validation_rows = metadata_by_split.get("validation", [])
        validation_labels = feature_arrays.get("validation", (None, None))[1]
        if validation_labels is None or not any(
            int(validation_labels[index]) >= 0
            and (
                row.get("profile") == "audio_grounded_v5"
                or row.get("source") == "authored_synthetic_audio_candidate"
            )
            and row.get("event_onset") is True
            for index, row in enumerate(validation_rows)
        ):
            raise ValueError("selezione V5 event-onset richiede validation V5 uncapped con event_onset")

    validation_checkpoint_path: Path | None = None
    if validation_checkpoint_dir is not None:
        validation_checkpoint_path = Path(validation_checkpoint_dir)
        if validation_checkpoint_path.exists():
            raise FileExistsError(
                f"la directory dei checkpoint validation deve essere nuova: {validation_checkpoint_path}"
            )
        validation_checkpoint_path.mkdir(parents=True)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    head = ControlHead(architecture=architecture, input_dim=input_dim).cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(learning_rate))
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
    class_indices = {
        action_index: [index for index in valid_indices if int(labels[index]) == action_index]
        for action_index in range(len(ACTIONS))
    }
    supported_classes = [index for index, indices in class_indices.items() if indices]
    if not supported_classes:
        raise ValueError("nessuna classe action presente nel train split stage2")
    train_class_support = {
        action: len(class_indices[index]) for index, action in enumerate(ACTIONS)
    }
    class_weight_values = np.zeros(len(ACTIONS), dtype=np.float32)
    if class_weighting == "sqrt_inverse_frequency":
        for class_index in supported_classes:
            class_weight_values[class_index] = 1.0 / np.sqrt(len(class_indices[class_index]))
        class_weight_values /= float(class_weight_values[supported_classes].mean())
        class_weight_tensor: torch.Tensor | None = torch.from_numpy(class_weight_values)
    else:
        class_weight_tensor = None
    training_event_onsets_by_action = Counter(
        ACTIONS[int(labels[index])]
        for index in valid_indices
        if train_metadata[index].get("event_onset") is True
    )
    head.train()
    losses: list[float] = []
    update_exposures = Counter()
    event_onset_update_exposures = Counter()
    selection_history: list[dict[str, Any]] = []
    best_selection_score: float | None = None
    best_selection_step: int | None = None
    best_selection_state: dict[str, torch.Tensor] | None = None
    best_selection_key: tuple[float, float, float] | None = None
    best_diagnostic_score: float | None = None
    best_diagnostic_step: int | None = None
    best_diagnostic_state: dict[str, torch.Tensor] | None = None
    best_diagnostic_metrics: dict[str, Any] | None = None
    last_selection_improvement_step = 0
    stopped_early = False
    for step in range(int(max_steps)):
        if sampling_strategy == "class_uniform":
            selected_class = random.choice(supported_classes)
            index = random.choice(class_indices[selected_class])
        else:
            index = random.choice(valid_indices)
            selected_class = int(labels[index])
        logits = head(features[index:index + 1])
        per_example_loss = F.cross_entropy(
            logits,
            labels[index:index + 1],
            weight=class_weight_tensor,
            reduction="none",
        )
        is_event_onset = train_metadata[index].get("event_onset") is True
        onset_multiplier = float(event_onset_loss_multiplier) if is_event_onset else 1.0
        loss = (per_example_loss * onset_multiplier).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        update_exposures[ACTIONS[selected_class]] += 1
        if is_event_onset:
            event_onset_update_exposures[ACTIONS[selected_class]] += 1
        completed_steps = step + 1
        if selection_metric is not None and (
            completed_steps % int(selection_interval) == 0 or completed_steps == int(max_steps)
        ):
            if "validation" not in feature_arrays:
                raise ValueError("checkpoint selection richiede lo split validation")
            val_features, val_labels = feature_arrays["validation"]
            val_metadata = metadata_by_split["validation"]
            if selection_metric == "validation_natural_macro_f1":
                validation_score = _natural_validation_macro_f1(
                    head, val_features, val_labels, val_metadata
                )
                record = {"step": completed_steps, **validation_score}
                score = validation_score.get("macro_f1")
                selection_key = (
                    float(score),
                    0.0,
                    0.0,
                ) if score is not None else None
                selection_is_feasible = score is not None
                selection_value = float(score) if score is not None else None
            else:
                valid_positions = np.flatnonzero(val_labels >= 0)
                head.eval()
                with torch.inference_mode():
                    val_predictions = head(
                        torch.from_numpy(val_features[valid_positions])
                    ).argmax(dim=-1).cpu().numpy()
                head.train()
                validation_score = _v5_event_onset_selection_metrics(
                    val_labels[valid_positions],
                    val_predictions,
                    [val_metadata[int(index)] for index in valid_positions],
                )
                event_per_class = validation_score["event_onsets_six_class"]["per_class"]
                supported_event_actions = [
                    action
                    for action in validation_score["event_action_outputs"]
                    if event_per_class[action]["support"] > 0
                ]
                meaningful_onset_recall = bool(supported_event_actions) and all(
                    float(event_per_class[action]["recall"] or 0.0) > 0.0
                    for action in supported_event_actions
                )
                background_rate = validation_score["background_false_event_rate"]
                budget_met = (
                    background_rate is not None
                    and float(background_rate) <= float(max_validation_v5_background_false_event_rate)
                )
                score = validation_score.get("event_onset_macro_f1")
                full_macro_f1 = validation_score["v5_full_six_class"].get("macro_f1")
                selection_value = float(score) if score is not None else None
                selection_key = (
                    float(score),
                    float(full_macro_f1 or 0.0),
                    -float(background_rate if background_rate is not None else 1.0),
                ) if score is not None else None
                selection_is_feasible = bool(
                    budget_met and meaningful_onset_recall and score is not None
                )
                record = {
                    "step": completed_steps,
                    **validation_score,
                    "false_event_budget": float(max_validation_v5_background_false_event_rate),
                    "background_false_event_budget_met": bool(budget_met),
                    "meaningful_event_onset_recall": meaningful_onset_recall,
                    "checkpoint_feasible": selection_is_feasible,
                }
            selection_history.append(record)
            if validation_checkpoint_path is not None:
                state_path = validation_checkpoint_path / f"step_{completed_steps:04d}.pt"
                head.save(state_path)
                record["cpu_validation_state_checkpoint"] = str(state_path.resolve())
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            if selection_value is not None and (
                best_diagnostic_score is None or selection_value > best_diagnostic_score
            ):
                best_diagnostic_score = selection_value
                best_diagnostic_step = completed_steps
                best_diagnostic_state = current_state
                if selection_metric == "validation_v5_event_onset_macro_f1":
                    best_diagnostic_metrics = validation_score
            if selection_value is not None and selection_is_feasible and (
                selection_key is not None
                and (best_selection_key is None or selection_key > best_selection_key)
            ):
                best_selection_key = selection_key
                best_selection_score = selection_value
                best_selection_step = completed_steps
                best_selection_state = current_state
                last_selection_improvement_step = completed_steps
            elif selection_metric == "validation_natural_macro_f1" and selection_value is not None and (
                best_selection_score is None or selection_value > best_selection_score
            ):
                best_selection_score = selection_value
                best_selection_step = completed_steps
                best_selection_state = current_state
                last_selection_improvement_step = completed_steps
            if (
                early_stopping_patience is not None
                and completed_steps >= max(int(minimum_steps), int(selection_interval))
                and last_selection_improvement_step > 0
                and completed_steps - last_selection_improvement_step >= int(early_stopping_patience)
            ):
                stopped_early = True
                break
    if selection_metric is not None:
        if selection_metric == "validation_v5_event_onset_macro_f1" and best_selection_state is None:
            if best_diagnostic_state is not None:
                head.load_state_dict(best_diagnostic_state, strict=True)
        elif best_selection_state is None or best_selection_step is None or best_selection_score is None:
            raise RuntimeError("checkpoint selection non ha prodotto alcuno stato validation valido")
        else:
            head.load_state_dict(best_selection_state, strict=True)
    head.eval()

    def _split_metrics(split_name: str) -> dict[str, Any] | None:
        if split_name not in feature_arrays:
            return None
        split_features, split_labels_array = feature_arrays[split_name]
        valid_positions = np.flatnonzero(split_labels_array >= 0)
        if not len(valid_positions):
            return {"examples": 0, "reason": "no_explicit_action_labels"}
        with torch.inference_mode():
            predictions = head(torch.from_numpy(split_features[valid_positions])).argmax(dim=-1).cpu().numpy()
        targets = split_labels_array[valid_positions].astype(np.int64, copy=False)

        train_prior = {
            action: len(class_indices[index]) for index, action in enumerate(ACTIONS)
        }

        def summarize(
            selected_targets: np.ndarray,
            selected_predictions: np.ndarray,
            actions: Sequence[str],
        ) -> dict[str, Any]:
            action_indices = [ACTIONS.index(action) for action in actions]
            keep = np.isin(selected_targets, action_indices)
            selected_targets = selected_targets[keep]
            selected_predictions = selected_predictions[keep]
            if not len(selected_targets):
                return {
                    "examples": 0,
                    "actions": list(actions),
                    "absent_classes": list(actions),
                    "reason": "no_examples_for_selected_actions",
                }
            local_index = {action_index: index for index, action_index in enumerate(action_indices)}
            # Keep every model output column even for the natural four-class
            # subset; YIELD/REPLAN predictions must count as false positives.
            matrix = np.zeros((len(actions), len(ACTIONS)), dtype=np.int64)
            for target, prediction in zip(selected_targets, selected_predictions):
                if int(target) in local_index:
                    matrix[local_index[int(target)], int(prediction)] += 1
            support = matrix.sum(axis=1)
            predicted_support = matrix.sum(axis=0)
            correct_predictions = sum(
                int(matrix[index, ACTIONS.index(action)])
                for index, action in enumerate(actions)
            )
            per_class: dict[str, Any] = {}
            f1_values = []
            recalls = []
            for index, action in enumerate(actions):
                global_index = ACTIONS.index(action)
                tp = int(matrix[index, global_index])
                recall = float(tp / support[index]) if support[index] else None
                precision = float(tp / predicted_support[global_index]) if predicted_support[global_index] else 0.0
                f1 = float(2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
                per_class[action] = {
                    "support": int(support[index]),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1 if support[index] else None,
                }
                if support[index]:
                    f1_values.append(f1)
                    recalls.append(recall)
            present_indices = [index for index, count in enumerate(support) if count]
            present_actions = [actions[index] for index in present_indices]
            majority_train_action = max(
                present_actions,
                key=lambda action: train_prior.get(action, 0),
                default=None,
            )
            examples = int(len(selected_targets))
            true_counts = support[present_indices]
            majority_accuracy = (
                float(support[actions.index(majority_train_action)] / examples)
                if majority_train_action is not None
                else None
            )
            prior_random_accuracy = float(np.square(true_counts).sum() / (examples * examples))
            return {
                "examples": examples,
                "actions": list(actions),
                "accuracy": float(correct_predictions / examples),
                "macro_f1": float(np.mean(f1_values)) if f1_values else None,
                "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
                "class_support": {action: int(support[index]) for index, action in enumerate(actions)},
                "absent_classes": [action for index, action in enumerate(actions) if not support[index]],
                "per_class": per_class,
                "confusion_matrix": {
                    "true_actions": list(actions),
                    "predicted_actions": list(ACTIONS),
                    "values": matrix.tolist(),
                },
                "chance_baselines": {
                    "uniform_random_over_present_classes_accuracy": float(1.0 / len(present_indices)),
                    "empirical_prior_random_expected_accuracy": prior_random_accuracy,
                },
                "majority_baseline_from_train_prior": {
                    "action": majority_train_action,
                    "accuracy": majority_accuracy,
                    "training_class_support": {
                        action: int(train_prior.get(action, 0)) for action in actions
                    },
                },
            }

        overall = summarize(targets, predictions, ACTIONS)
        row_metadata = metadata_by_split[split_name]
        if len(row_metadata) != len(split_labels_array):
            raise ValueError(f"{split_name} metadata non allineati durante il report per-source")
        valid_metadata = [row_metadata[int(index)] for index in valid_positions]
        source_indices: dict[str, list[int]] = {}
        natural_actions = ("LISTEN", "SPEAK", "CONTINUE", "BACKCHANNEL")
        natural_action_indices = {ACTIONS.index(action) for action in natural_actions}
        natural_positions: list[int] = []
        synthetic_positions: list[int] = []
        audio_grounded_v5_positions: list[int] = []
        audio_grounded_v5_event_onset_positions: list[int] = []
        audio_grounded_v5_non_onset_positions: list[int] = []
        for position, row in enumerate(valid_metadata):
            source = str(row.get("source", "unknown"))
            source_indices.setdefault(source, []).append(position)
            if _is_synthetic_source(row):
                synthetic_positions.append(position)
            if (
                row.get("profile") == "audio_grounded_v5"
                or source == "authored_synthetic_audio_candidate"
            ):
                audio_grounded_v5_positions.append(position)
                if row.get("event_onset") is True:
                    audio_grounded_v5_event_onset_positions.append(position)
                else:
                    audio_grounded_v5_non_onset_positions.append(position)
            if not _is_synthetic_source(row) and int(targets[position]) in natural_action_indices:
                natural_positions.append(position)
        by_source = {
            source: summarize(targets[indices], predictions[indices], ACTIONS)
            for source, indices in sorted(source_indices.items())
        }
        subgroup_metrics = {
            "natural_four_class": (
                _natural_validation_macro_f1(
                    head,
                    split_features,
                    split_labels_array,
                    row_metadata,
                    train_class_support=train_prior,
                )
                if natural_positions
                else summarize(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), natural_actions)
            ),
            "synthetic_six_class": summarize(
                targets[synthetic_positions], predictions[synthetic_positions], ACTIONS
            ),
            "audio_grounded_v5_six_class": summarize(
                targets[audio_grounded_v5_positions],
                predictions[audio_grounded_v5_positions],
                ACTIONS,
            ),
            "audio_grounded_v5_event_onsets_six_class": summarize(
                targets[audio_grounded_v5_event_onset_positions],
                predictions[audio_grounded_v5_event_onset_positions],
                ACTIONS,
            ),
        }
        v5_selection_metrics = _v5_event_onset_selection_metrics(
            targets, predictions, valid_metadata
        )
        subgroup_metrics["audio_grounded_v5_event_onset_background_false_event_rate"] = {
            "examples": v5_selection_metrics["background_examples"],
            "event_action_outputs": v5_selection_metrics["event_action_outputs"],
            "false_event_action_predictions": v5_selection_metrics[
                "background_false_event_predictions"
            ],
            "false_event_rate": v5_selection_metrics["background_false_event_rate"],
            "false_event_predictions_by_class": v5_selection_metrics[
                "background_false_event_predictions_by_class"
            ],
            "denominator": v5_selection_metrics["background_definition"],
        }
        subgroup_metrics[
            "audio_grounded_v5_event_onset_legacy_off_onset_event_output_rate_not_false_action_rate"
        ] = {
            "examples": v5_selection_metrics["legacy_off_onset_examples"],
            "event_action_outputs": v5_selection_metrics["event_action_outputs"],
            "event_output_predictions": v5_selection_metrics[
                "legacy_off_onset_event_output_predictions"
            ],
            "event_output_rate": v5_selection_metrics[
                "legacy_off_onset_event_output_rate_not_false_action_rate"
            ],
            "note": v5_selection_metrics["legacy_metric_note"],
        }
        source_support = {}
        for split, rows in metadata_by_split.items():
            counts: dict[str, Counter[str]] = {}
            source_groups: dict[str, set[str]] = {}
            for row in rows:
                if not bool(row.get("action_mask")) or row.get("action") not in ACTIONS:
                    continue
                source = str(row.get("source", "unknown"))
                counts.setdefault(source, Counter())[str(row["action"])] += 1
                source_groups.setdefault(source, set()).add(
                    str(row.get("source_group_id") or row.get("conversation_id") or "")
                )
            source_support[split] = {
                source: {
                    "examples": int(sum(action_counts.values())),
                    "source_groups": len(source_groups.get(source, set())),
                    "class_support": {
                        action: int(action_counts.get(action, 0)) for action in ACTIONS
                    },
                    "absent_classes": [
                        action for action in ACTIONS if not action_counts.get(action, 0)
                    ],
                }
                for source, action_counts in sorted(counts.items())
            }
        overall["by_source"] = by_source
        overall["subgroups"] = subgroup_metrics
        overall["source_class_support_by_split"] = source_support
        overall["source_groups_by_split"] = {
            split: {
                "count": len({
                    str(row.get("source_group_id") or row.get("conversation_id") or "")
                    for row in rows
                }),
                "disjoint_with_other_splits": True,
            }
            for split, rows in metadata_by_split.items()
        }
        return overall

    output = Path(output_path)
    candidate_selection_infeasible = (
        selection_metric == "validation_v5_event_onset_macro_f1"
        and best_selection_state is None
    )
    diagnostic_checkpoint_saved = bool(
        candidate_selection_infeasible
        and save_infeasible_diagnostic_checkpoint
        and best_diagnostic_state is not None
    )
    head_saved = not candidate_selection_infeasible or diagnostic_checkpoint_saved
    if head_saved:
        head.save(output)
    if sampling_strategy == "class_uniform":
        sampling_probabilities = {
            action: (1.0 / len(supported_classes) if class_indices[index] else 0.0)
            for index, action in enumerate(ACTIONS)
        }
        sampling_description = "uniform_over_observed_action_classes_then_uniform_within_class"
    else:
        sampling_probabilities = {
            action: (len(class_indices[index]) / len(valid_indices))
            for index, action in enumerate(ACTIONS)
        }
        sampling_description = "uniform_over_all_supervised_training_rows"
    if selection_metric == "validation_v5_event_onset_macro_f1":
        selected_history = next(
            (row for row in selection_history if row["step"] == best_selection_step),
            None,
        )
        checkpoint_selection: dict[str, Any] | None = {
            "metric": selection_metric,
            "validation_population": "all_explicitly_supervised_uncapped_V5_validation_event_schedule_onsets",
            "interval_updates": int(selection_interval),
            "evaluated_steps": [int(row["step"]) for row in selection_history],
            "selected_step": best_selection_step,
            "selected_event_onset_macro_f1": best_selection_score,
            "selected_validation_event_onset_examples": (
                int(selected_history["event_onset_examples"]) if selected_history else None
            ),
            "background_false_event_rate_budget": float(
                max_validation_v5_background_false_event_rate
            ),
            "background_definition": (
                "off-onset supervised V5 targets whose true action is LISTEN or CONTINUE; "
                "CONTINUE event onsets and event-action interiors are excluded"
            ),
            "background_false_event_budget_met": bool(selected_history is not None),
            "meaningful_event_onset_recall_required": True,
            "selection_status": "feasible_checkpoint_selected" if selected_history else "infeasible_no_checkpoint_met_budget_and_recall_gate",
            "promotion_eligible": bool(selected_history is not None),
            "diagnostic_checkpoint_saved_for_validation_calibration_only": diagnostic_checkpoint_saved,
            "diagnostic_best_onset_f1_step": best_diagnostic_step,
            "diagnostic_best_onset_f1_metrics": best_diagnostic_metrics,
            "history": selection_history,
            "validation_state_checkpoint_dir": (
                str(validation_checkpoint_path.resolve())
                if validation_checkpoint_path is not None
                else None
            ),
            "test_used_for_selection": False,
            "heldout_test_input_present": "test" in feature_arrays,
        }
    elif selection_metric is None:
        checkpoint_selection = None
    else:
        selected_history = next(
            row for row in selection_history if row["step"] == best_selection_step
        )
        checkpoint_selection = {
            "metric": selection_metric,
            "validation_population": "natural_labeled_LISTEN_SPEAK_CONTINUE_BACKCHANNEL_only",
            "interval_updates": int(selection_interval),
            "evaluated_steps": [int(row["step"]) for row in selection_history],
            "selected_step": best_selection_step,
            "selected_macro_f1": best_selection_score,
            "selected_validation_examples": int(selected_history["examples"]),
            "history": selection_history,
            "test_used_for_selection": False,
        }
    metadata = {
        "control_head_path": str(output.resolve()) if head_saved else None,
        "control_head": {
            "input_dim": input_dim,
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(ACTIONS),
            "architecture": architecture,
            "feature_contract": feature_contract,
        },
        "stage2_training": {
            "examples": int(len(valid_indices)),
            "masked_frames": int(len(labels) - len(valid_indices)),
            "steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "architecture": architecture,
            "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "trainable_head_parameters": sum(
                parameter.numel() for parameter in head.parameters() if parameter.requires_grad
            ),
            "precomputed_feature_encoder_frozen": True,
            "class_support": {
                action: int(train_class_support[action]) for index, action in enumerate(ACTIONS)
            },
            "sampling_strategy": sampling_strategy,
            "class_sampling_probabilities": sampling_probabilities,
            "loss": (
                {"function": "unweighted_cross_entropy", "class_weights": None}
                if class_weighting == "none"
                else {
                    "function": "weighted_cross_entropy_reduction_none_then_example_scale",
                    "class_weighting": class_weighting,
                    "class_weights": {
                        action: float(class_weight_values[index])
                        for index, action in enumerate(ACTIONS)
                    },
                    "class_weight_normalization": "arithmetic_mean_1_over_observed_training_classes",
                }
            ),
            "event_onset_loss_weighting": {
                "multiplier": float(event_onset_loss_multiplier),
                "applied_when": "training_metadata.event_onset is true for any supervised action, including CONTINUE transitions",
                "training_event_onset_examples_by_action": {
                    action: int(training_event_onsets_by_action.get(action, 0))
                    for action in ACTIONS
                },
                "optimizer_exposures_to_event_onsets_by_action": {
                    action: int(event_onset_update_exposures.get(action, 0))
                    for action in ACTIONS
                },
            },
            "optimizer_update_exposures_by_action": {
                action: int(update_exposures.get(action, 0)) for action in ACTIONS
            },
            "expected_updates_per_observed_class": {
                action: float(len(losses) * sampling_probabilities[action])
                for action in ACTIONS
            },
            "checkpoint_selection": checkpoint_selection,
            "early_stopping": {
                "patience_updates": early_stopping_patience,
                "minimum_steps": int(minimum_steps),
                "stopped_early": stopped_early,
                "completed_steps": len(losses),
            },
            "checkpoint_saved": head_saved,
            "diagnostic_checkpoint_role": (
                "best_infeasible_event_onset_checkpoint_saved_for_validation_calibration_only"
                if diagnostic_checkpoint_saved
                else "best_infeasible_event_onset_checkpoint_not_saved"
                if candidate_selection_infeasible
                else "selected_checkpoint"
            ),
            "candidate_promotion_eligible": not candidate_selection_infeasible,
            "absent_classes": [ACTIONS[index] for index in range(len(ACTIONS)) if not class_indices[index]],
            "sampling": sampling_description,
            "seed": int(seed),
            "train_diagnostics": _split_metrics("train"),
            "validation": _split_metrics("validation"),
            "heldout_test": _split_metrics("test"),
            "feature_manifest": str(feature_manifest_path.resolve()),
        },
    }
    manifest_path = Path(training_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest stage1 mancante: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("training manifest schema_version non supportato")
    manifest.update(metadata)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "extract_conversation_features",
    "extract_stage2_features",
    "load_merged_base",
    "train_control_head",
]
