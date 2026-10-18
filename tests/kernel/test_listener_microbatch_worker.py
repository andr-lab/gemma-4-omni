from __future__ import annotations

import asyncio
from concurrent.futures import Future
from dataclasses import dataclass, replace
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from duplex_kernel.gemma_native_control6_batch_policy import GemmaNativeControl6BatchPolicy
from duplex_kernel.listener import FairModelLock
from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator
from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor,
    ListenerFrame,
    ListenerSessionState,
    NativeBatchCapability,
    PCM_TAIL_SAMPLES,
    RuntimeSignature,
)
from duplex_kernel.native_listener_microbatch import NativeBatchOutput, NativeBatchOutputHealth
from duplex_kernel.turn_policy import PolicyInput
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA

from duplex_kernel.listener_microbatch_worker import (
    PreparedInitialPrefix,
    ListenerMicrobatchRuntime,
    ListenerMicrobatchWorker,
    PrefixCacheImport,
    SemanticPrefixShadowRuntime,
    SemanticPrefixShadowWorker,
    ShadowQ2Frame,
    WorkerStopReport,
)


def _signature() -> RuntimeSignature:
    return RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        topology_sha256="c" * 64,
        native_library_sha256="d" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )


def _capability() -> NativeBatchCapability:
    return NativeBatchCapability(
        api_version=2,
        native_abi_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
        supports_arbitrary_batch_size=True,
        stream_priority=-1,
        stream_identity="cuda:0:stream:0x17",
    )


def _state(signature: RuntimeSignature, sid: str, *, generation: int = 7) -> ListenerSessionState:
    return ListenerSessionState(
        session_id=sid,
        generation_id=generation,
        state_epoch_id=f"epoch-{sid}-{generation}",
        runtime_signature=signature,
        cache_position=34,
        valid_history_tokens=34,
        native_context_id=10_000 + sum(ord(char) for char in sid) + generation,
    )


def _frame(
    state: ListenerSessionState,
    *,
    frame_id: int = 0,
    value: float = 0.25,
    ready_at_ns: int | None = None,
) -> ListenerFrame:
    return ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id,
        ready_at_ns=time.monotonic_ns() if ready_at_ns is None else ready_at_ns,
        pcm=np.full(1280, value, dtype=np.float32),
    )


class _FakeStream:
    device = torch.device("cpu")
    cuda_stream = 23

    def __init__(self) -> None:
        self.waited_events: list[Any] = []

    def wait_event(self, event: Any) -> None:
        self.waited_events.append(event)


class _FakeCudaEvent:
    def __init__(self, **_kwargs: Any) -> None:
        self.recorded_stream: Any | None = None

    def record(self, stream: Any) -> None:
        self.recorded_stream = stream


@pytest.fixture(autouse=True)
def _fake_cuda_event_for_worker_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU-only owner tests independent from CUDA initialization."""
    monkeypatch.setattr(torch.cuda, "Event", _FakeCudaEvent)


class _FakeMelExtractor:
    def __call__(self, pcm: np.ndarray, previous: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        mel = np.full((8, 128), float(np.mean(pcm)), dtype=np.float32)
        prefix = np.zeros(160, dtype=np.float32) if previous is None else previous
        tail = np.concatenate((prefix, np.asarray(pcm, dtype=np.float32)))[-160:]
        return mel, tail.copy()


class _FakeAudioModel:
    def __init__(self) -> None:
        self.calls = 0
        self.call_shapes: list[tuple[int, ...]] = []
        self.call_inputs: list[torch.Tensor] = []
        self.thread_ids: list[int] = []

    def get_audio_features(self, *, input_features, input_features_mask, return_dict):
        self.calls += 1
        self.thread_ids.append(threading.get_ident())
        self.call_shapes.append(tuple(input_features.shape))
        self.call_inputs.append(input_features.detach().clone())
        assert return_dict is True
        n = input_features.shape[0]
        output_rows = input_features.shape[1] // 4
        pooled = torch.empty((n, output_rows, HIDDEN_WIDTH), dtype=torch.bfloat16, device=input_features.device)
        tokens = input_features.float().reshape(n, output_rows, 4, 128).mean(dim=(2, 3))
        for row in range(n):
            for token in range(output_rows):
                pooled[row, token].fill_(float(tokens[row, token] + token))
        mask = torch.ones((n, output_rows), dtype=torch.bool, device=input_features.device)
        return SimpleNamespace(pooler_output=pooled, attention_mask=mask)


class _FakeCompletionEvent:
    def __init__(self) -> None:
        self.waited = False

    def wait(self, _stream: Any) -> None:
        self.waited = True


class _FakeNativeBatch:
    def __init__(self, signature: RuntimeSignature, capability: NativeBatchCapability, stream: _FakeStream) -> None:
        self.creation_thread_id = threading.get_ident()
        self.runtime_signature = signature
        self.capability = capability
        self.listener_stream = stream
        self.max_batch_size = 4
        self.capacity_requests: list[int] = []
        self.event_order: list[tuple[str, object]] = []
        self.fail_capacity_at: int | None = None
        self.imports: list[tuple[int, str, int]] = []
        self.steps: list[tuple[tuple[str, ...], int]] = []
        self.releases: list[tuple[str, int, int]] = []
        self.release_waited_event_counts: list[int] = []
        self.close_thread_id: int | None = None
        self.owner_thread_id: int | None = None
        self.fail_step: BaseException | None = None
        self.nonfinite_health = False
        self.health_generations: list[int] = []
        self.model_lock: Any | None = None
        self.lock_owners_at_import: list[dict[str, Any] | None] = []
        self.lock_held_at_import: list[bool | None] = []
        self.lock_held_at_step: list[bool | None] = []
        self.import_rows: list[BatchRowDescriptor] = []
        self.manifests: list[Any] = []
        self.audio_rows: list[torch.Tensor] = []
        self.cache_copy_source_rows: list[BatchRowDescriptor] = []
        self.fork_calls: list[tuple[BatchRowDescriptor, BatchRowDescriptor]] = []
        self.slot_fork_calls: list[tuple[BatchRowDescriptor, BatchRowDescriptor]] = []
        self.fail_fork: BaseException | None = None
        self.fork_entered: threading.Event | None = None
        self.fork_continue: threading.Event | None = None
        self.step_entered: threading.Event | None = None
        self.step_continue: threading.Event | None = None
        self.step_block_at_count = 1

    def _claim(self) -> None:
        current = threading.get_ident()
        if self.owner_thread_id is None:
            self.owner_thread_id = current
        assert self.owner_thread_id == current

    def import_session(self, row, past_key_values, prefix_attention, *, prefix_len, producer_stream=None):
        self._claim()
        assert past_key_values is not None
        assert tuple(prefix_attention.shape) == (1, prefix_len)
        assert producer_stream is self.listener_stream
        snapshot = getattr(self.model_lock, "host_timeline_snapshot", None)
        self.lock_owners_at_import.append(
            None if not callable(snapshot) else snapshot().get("current_owner")
        )
        self.lock_held_at_import.append(
            None if self.model_lock is None else bool(getattr(self.model_lock, "held", False))
        )
        self.imports.append((row.native_context_id, row.session_id, row.frame_id))
        self.import_rows.append(row)
        self.event_order.append(("import", row.session_id))

    def ensure_capacity(self, requested_capacity: int) -> int:
        self._claim()
        self.capacity_requests.append(requested_capacity)
        self.event_order.append(("ensure_capacity", requested_capacity))
        if self.fail_capacity_at == requested_capacity:
            raise RuntimeError("native listener capacity allocation failed: CUDA out of memory")
        self.max_batch_size = max(self.max_batch_size, requested_capacity)
        return self.max_batch_size

    def step_async(self, manifest, audio_soft, *, per_layer_inputs=None, producer_stream=None):
        self._claim()
        if self.fail_step is not None:
            raise self.fail_step
        assert producer_stream is self.listener_stream
        assert tuple(audio_soft.shape) == (manifest.batch_size, 2, HIDDEN_WIDTH)
        assert audio_soft.dtype == torch.bfloat16
        self.manifests.append(manifest)
        self.audio_rows.append(audio_soft.detach().clone())
        self.lock_held_at_step.append(
            None if self.model_lock is None else bool(getattr(self.model_lock, "held", False))
        )
        self.steps.append((tuple(row.session_id for row in manifest.rows), threading.get_ident()))
        self.event_order.append(("step", manifest.batch_size))
        if self.step_entered is not None and len(self.steps) == self.step_block_at_count:
            self.step_entered.set()
        if (
            self.step_continue is not None
            and len(self.steps) == self.step_block_at_count
        ):
            assert self.step_continue.wait(3)
        output_tensor = audio_soft.detach().clone()
        if manifest.rows and all(row.session_id.startswith("__q2view__") for row in manifest.rows):
            output_tensor = output_tensor + 7
        return NativeBatchOutput(
            manifest_id=manifest.manifest_id,
            generation=len(self.steps),
            rows=manifest.rows,
            tensor=output_tensor,
            completion_event=_FakeCompletionEvent(),
            logical_q_lens=tuple(row.logical_q_len for row in manifest.rows),
            embedded_text_input=per_layer_inputs is not None,
        )

    def step_queued_audio_async(self, manifest, audio_soft, *, producer_stream=None):
        output = self.step_async(
            manifest, audio_soft, producer_stream=producer_stream,
        )
        return replace(output, preserves_listener_cursors=True)

    def fork_session_slot(
        self, source_row, *, fork_session_id, fork_native_context_id,
    ):
        self._claim()
        if self.fail_fork is not None:
            raise self.fail_fork
        destination = replace(
            source_row,
            session_id=fork_session_id,
            native_context_id=fork_native_context_id,
        )
        self.slot_fork_calls.append((source_row, destination))
        self.event_order.append(("slot_fork", fork_session_id))
        return destination

    def rewind_session_async(self, row, *, target_position):
        self._claim()
        self.event_order.append(("rewind", row.session_id))
        return SimpleNamespace(
            cache_position=int(target_position),
            valid_history_tokens=int(target_position),
        )

    def verify_output_health(self, output):
        self._claim()
        self.health_generations.append(output.generation)
        self.event_order.append(("health", output.generation))
        status = "nonfinite_guard" if self.nonfinite_health else "healthy"
        return tuple(
            NativeBatchOutputHealth(
                generation=output.generation,
                row=row,
                checked=True,
                nonfinite=self.nonfinite_health,
                status=status,
            )
            for row in output.rows
        )

    def release_session_async(self, row) -> None:
        self._claim()
        self.releases.append((row.session_id, row.frame_id, row.cache_position))
        self.release_waited_event_counts.append(len(self.listener_stream.waited_events))

    def prepare_cache_copy_source(self, row):
        self._claim()
        self.cache_copy_source_rows.append(row)
        return SimpleNamespace(row=row)

    def fork_session_cache_within_owner(self, source_row, destination_row):
        self._claim()
        self.fork_calls.append((source_row, destination_row))
        if self.fork_entered is not None:
            self.fork_entered.set()
        if self.fork_continue is not None:
            assert self.fork_continue.wait(3)
        if self.fail_fork is not None:
            raise self.fail_fork
        return destination_row

    def close(self) -> None:
        self._claim()
        self.close_thread_id = threading.get_ident()


class _RecordingAdapter(NativeControl6LoRA):
    def __init__(self) -> None:
        super().__init__(torch.zeros((len(ACTIONS), HIDDEN_WIDTH), dtype=torch.float32))
        self.creation_thread_id = threading.get_ident()
        self.thread_ids: list[int] = []
        self.hidden_rows: list[torch.Tensor] = []

    def forward(self, hidden: torch.Tensor, *, phase_t=None) -> torch.Tensor:
        self.thread_ids.append(threading.get_ident())
        self.hidden_rows.append(hidden.detach().clone())
        return super().forward(hidden, phase_t=phase_t)


class _FakeTokenAdapter:
    def prepare(self, token_ids, *, logical_q_lens):
        count = len(tuple(token_ids))
        assert logical_q_lens == count
        return SimpleNamespace(
            inputs_embeds=torch.full((1, 2, HIDDEN_WIDTH), float(count), dtype=torch.bfloat16),
            per_layer_inputs=torch.zeros((1, 2, 35, 256), dtype=torch.bfloat16),
            logical_q_lens=(count,),
        )

    def logits_from_final_norm_hidden(self, hidden):
        return hidden.float().mean(dim=-1, keepdim=True)


class _CudaProvenanceTensor:
    """CPU fixture tensor carrying the device metadata required by queue admission."""

    def __init__(self, value: torch.Tensor) -> None:
        self.value = value
        self.shape = value.shape
        self.dtype = value.dtype
        self.device = SimpleNamespace(type="cuda")

    def record_stream(self, _stream: Any) -> None:
        pass

    def unsqueeze(self, dim: int) -> torch.Tensor:
        return self.value.unsqueeze(dim)


def _assistant_q0(
    state: ListenerSessionState,
    frame_id: int = 0,
    *,
    queue_audio_for_next_turn: bool = True,
) -> ListenerFrame:
    return ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=frame_id,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.full(1280, 0.25, dtype=np.float32),
        logical_q_len=0,
        ephemeral_reason="assistant_turn_open",
        queue_audio_for_next_turn=queue_audio_for_next_turn,
    )


def _mark_queued_soft_cuda_for_cpu_fixture(rig: _Rig, session_id: str) -> None:
    queued = rig.worker._queued_audio_soft[session_id]
    assert len(queued) == 1
    queued[0] = replace(
        queued[0],
        audio_soft_tokens=_CudaProvenanceTensor(queued[0].audio_soft_tokens),
    )


class _OwnerRecordingCoordinator(ListenerMicrobatchCoordinator):
    def __init__(self, signature: RuntimeSignature, capability: NativeBatchCapability) -> None:
        super().__init__(signature, capability)
        self.operation_threads: list[tuple[str, int]] = []

    def register_session(self, state: ListenerSessionState) -> None:
        self.operation_threads.append(("register", threading.get_ident()))
        return super().register_session(state)

    def enqueue(self, frame: ListenerFrame) -> None:
        self.operation_threads.append(("enqueue", threading.get_ident()))
        return super().enqueue(frame)

    def close_session(self, session_id: str):
        self.operation_threads.append(("close", threading.get_ident()))
        return super().close_session(session_id)

    def session_state(self, session_id: str):
        self.operation_threads.append(("snapshot", threading.get_ident()))
        return super().session_state(session_id)


@dataclass
class _Rig:
    worker: ListenerMicrobatchWorker
    coordinator: ListenerMicrobatchCoordinator
    model: _FakeAudioModel
    native: _FakeNativeBatch | None
    adapter: _RecordingAdapter | None
    runtime_factory_threads: list[int]
    prefix_import_threads: list[int]
    prefix_import: Any
    completion_barrier: Any
    results: list[tuple[Any, ...]]
    errors: list[tuple[Any, BaseException]]
    result_event: threading.Event
    error_event: threading.Event


def _rig(
    *,
    completion_barrier=None,
    model_lock: Any | None = None,
    fail_step=None,
    nonfinite_health: bool = False,
    auto_start=True,
    runtime_factory_entered=None,
    runtime_factory_continue=None,
) -> _Rig:
    signature = _signature()
    capability = _capability()
    coordinator = _OwnerRecordingCoordinator(signature, capability)
    model = _FakeAudioModel()
    native_holder: list[_FakeNativeBatch] = []
    adapter_holder: list[_RecordingAdapter] = []
    factory_threads: list[int] = []
    prefix_threads: list[int] = []

    def runtime_factory() -> ListenerMicrobatchRuntime:
        factory_threads.append(threading.get_ident())
        if runtime_factory_entered is not None:
            runtime_factory_entered.set()
        if runtime_factory_continue is not None:
            assert runtime_factory_continue.wait(4)
        stream = _FakeStream()
        native = _FakeNativeBatch(signature, capability, stream)
        native.fail_step = fail_step
        native.nonfinite_health = nonfinite_health
        native.model_lock = model_lock
        adapter = _RecordingAdapter()
        policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="fixture://control6")
        native_holder.append(native)
        adapter_holder.append(adapter)
        frontend = ListenerMicrobatchAudioFrontend(
            model,
            _FakeMelExtractor(),
            device="cpu",
            dtype=torch.bfloat16,
        )
        return ListenerMicrobatchRuntime(
            listener_stream=stream,
            frontend=frontend,
            native=native,
            policy=policy,
            policy_inputs=lambda batch, states: tuple(
                PolicyInput(
                    session_id=row.session_id,
                    frame_id=row.frame_id,
                    user_pcm=frame.pcm,
                    speaker_active=False,
                    generation_id=row.session_generation,
                    legal_actions=tuple(ACTIONS),
                )
                for row, frame in zip(batch.manifest.rows, batch.frames, strict=True)
            ),
            completion_barrier=completion_barrier,
            listener_model_lock=model_lock,
            token_adapter=_FakeTokenAdapter(),
        )

    def prefix_import(state, row) -> PrefixCacheImport:
        del state, row
        prefix_threads.append(threading.get_ident())
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 34), dtype=torch.int64),
            prefix_len=34,
        )

    results: list[tuple[Any, ...]] = []
    errors: list[tuple[Any, BaseException]] = []
    result_event = threading.Event()
    error_event = threading.Event()

    def on_result(batch, decisions) -> None:
        results.append((batch, decisions))
        result_event.set()

    def on_error(batch, error) -> None:
        errors.append((batch, error))
        error_event.set()

    worker = ListenerMicrobatchWorker(
        coordinator,
        runtime_factory=runtime_factory,
        on_result=on_result,
        on_error=on_error,
    )
    if auto_start:
        worker.start().result(timeout=3)
    return _Rig(
        worker, coordinator,
        model,
        native_holder[0] if native_holder else None,
        adapter_holder[0] if adapter_holder else None,
        factory_threads,
        prefix_threads, prefix_import, completion_barrier, results, errors, result_event, error_event,
    )


def _register(rig: _Rig, *states: ListenerSessionState) -> None:
    for state in states:
        rig.worker.register_session(state, prefix_import=rig.prefix_import).result(timeout=3)


def _prime_native_prefix(rig: _Rig, state: ListenerSessionState) -> None:
    prepared = PreparedInitialPrefix(
        prefix=PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 34), dtype=torch.int64),
            prefix_len=34,
            cache_position=34,
        ),
        ready_event=None,
        session_id=state.session_id,
        listener_generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        semantic_context_revision=1,
        expected_cache_position=34,
    )
    assert rig.worker.prime_initial_prefix(state.session_id, prepared).result(timeout=3).status == "primed"


def _set_fixture_capacity(rig: _Rig, max_seq_len: int) -> None:
    assert rig.native is not None
    assert rig.native.capability is rig.coordinator.capability
    # The production contract fixes max_seq_len to 16K. These ownership tests
    # lower the fixture's admission bound after construction to exercise the
    # near-capacity reservation arithmetic without weakening that ABI contract.
    object.__setattr__(rig.coordinator.capability, "max_seq_len", max_seq_len)


def _shadow_request(session_id: str, revision: int, frame_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        state_epoch_id=f"epoch-{session_id}-7",
        current_frame_id=frame_id,
        listener_state_version=frame_id,
        semantic_context_revision=revision,
        prefix_token_ids=(1, 2),
        prefix_len=50,
        replay_q2=(),
        replay_frame_ids=(),
        cache_position=50,
        fallback_reason=None,
    )


def _shadow_history_request(
    session_id: str,
    revision: int,
    *,
    target_frame_id: int,
    target_state_version: int,
    target_sequence_position: int,
    replay_frame_ids: tuple[int, ...],
    replay_count: int | None = None,
) -> SimpleNamespace:
    count = len(replay_frame_ids) if replay_count is None else replay_count
    return SimpleNamespace(
        session_id=session_id,
        session_generation_id=7,
        state_epoch_id=f"epoch-{session_id}-7",
        current_frame_id=target_frame_id,
        listener_state_version=target_state_version,
        semantic_context_revision=revision,
        prefix_token_ids=(1, 2),
        prefix_len=50,
        replay_q2=tuple(
            torch.full((2, HIDDEN_WIDTH), float(index), dtype=torch.bfloat16)
            for index in range(count)
        ),
        replay_frame_ids=replay_frame_ids,
        cache_position=50 + 2 * count,
        fallback_reason=None,
    )


def _shadow_row(session_id: str, frame_id: int, *, cache_position: int = 50) -> BatchRowDescriptor:
    return BatchRowDescriptor(
        session_id=session_id,
        session_generation=7,
        state_epoch_id=f"epoch-{session_id}-7",
        frame_id=frame_id,
        listener_state_version=frame_id,
        sequence_position=frame_id,
        cache_position=cache_position,
        valid_history_tokens=cache_position,
        native_context_id=90_000 + frame_id,
        input_frame_sha256="a" * 64,
        reference_pcm_sha256=None,
        frame_ready_at_ns=1,
        reference_frame_id=None,
    )


def _start_shadow_worker(builder=None, native_factory=None):
    signature = _signature()
    stream = _FakeStream()
    capability = replace(
        _capability(),
        stream_priority=0,
        stream_identity="cuda:0:shadow-stream",
    )
    native = (
        _FakeNativeBatch(signature, capability, stream)
        if native_factory is None
        else native_factory(signature, capability, stream)
    )

    def prefix_builder(request, _stream):
        if builder is not None:
            return builder(request)
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, request.prefix_len), dtype=torch.int64),
            prefix_len=request.prefix_len,
            cache_position=request.cache_position,
            semantic_context_revision=request.semantic_context_revision,
        )

    shadow = SemanticPrefixShadowWorker(
        lambda: SemanticPrefixShadowRuntime(native=native, listener_stream=stream),
        prefix_builder,
        runtime_signature=signature,
    )
    shadow.start().result(timeout=3)
    return shadow, native, stream


def _prefix_only_import(request) -> PrefixCacheImport:
    return PrefixCacheImport(
        past_key_values=object(),
        prefix_attention=torch.ones((1, request.prefix_len), dtype=torch.int64),
        prefix_len=request.prefix_len,
        cache_position=request.prefix_len,
        semantic_context_revision=request.semantic_context_revision,
        replay_frame_ids=tuple(request.replay_frame_ids),
    )


def _wait_for_shadow_candidate(shadow, *, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidate = shadow._candidate
        if candidate is not None:
            return candidate
        time.sleep(0.005)
    return None


def test_shadow_prefix_only_r0_imports_and_accepts_exact_target_cursor() -> None:
    shadow, native, _stream = _start_shadow_worker(
        builder=lambda request: _prefix_only_import(request)
    )
    request = _shadow_history_request(
        "shadow-prefix-only-r0",
        12,
        target_frame_id=9,
        target_state_version=17,
        target_sequence_position=21,
        replay_frame_ids=(),
    )
    # The request carries the websocket/session generation; the stage receives
    # the independent listener-state generation from its owner.
    request.session_generation_id = 0
    try:
        assert shadow.stage(request, listener_generation_id=7, sequence_position=21)
        candidate = _wait_for_shadow_candidate(shadow)
        assert candidate is not None
        assert native.import_rows[0].frame_id == 9
        assert native.import_rows[0].listener_state_version == 17
        assert native.import_rows[0].sequence_position == 21
        assert native.import_rows[0].cache_position == request.prefix_len
        assert native.manifests == []
        assert candidate.row.frame_id == request.current_frame_id
        assert candidate.row.listener_state_version == request.listener_state_version
        assert candidate.row.sequence_position == 21
        assert candidate.row.cache_position == request.cache_position == request.prefix_len
        assert candidate.stage.request is request
        stale_listener_row = replace(candidate.row, session_generation=0)
        assert shadow.candidate_for(stale_listener_row) is None
        assert (
            shadow.status["diagnostics"]["candidate_last_rejection_reason"]
            == "generation_mismatch"
        )
        assert shadow.candidate_for(candidate.row) is candidate
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_native_history_replay_runs_after_prefix_builder_releases_model_lock() -> None:
    class PrefixLock:
        def __init__(self) -> None:
            self.held = False
            self.prefix_lock_observations: list[bool] = []

        def acquire_fair(self, priority: str) -> bool:
            assert priority == "listener"
            assert not self.held
            self.held = True
            return True

        def release(self) -> None:
            assert self.held
            self.held = False

    model_lock = PrefixLock()

    def builder(request):
        assert model_lock.acquire_fair("listener")
        try:
            model_lock.prefix_lock_observations.append(model_lock.held)
            return _prefix_only_import(request)
        finally:
            model_lock.release()

    shadow, native, _stream = _start_shadow_worker(builder=builder)
    native.model_lock = model_lock
    request = _shadow_history_request(
        "shadow-native-lock-scope",
        3,
        target_frame_id=1,
        target_state_version=2,
        target_sequence_position=4,
        replay_frame_ids=(0,),
    )
    try:
        assert shadow.stage(request, listener_generation_id=7, sequence_position=4)
        candidate = _wait_for_shadow_candidate(shadow)
        assert candidate is not None
        assert model_lock.prefix_lock_observations == [True]
        assert native.lock_held_at_import == [False]
        assert native.lock_held_at_step == [False]
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_prefix_replays_all_64_frozen_q2_rows_before_candidate_acceptance() -> None:
    shadow, native, _stream = _start_shadow_worker(
        builder=lambda request: _prefix_only_import(request)
    )
    request = _shadow_history_request(
        "shadow-prefix-r64",
        13,
        target_frame_id=64,
        target_state_version=69,
        target_sequence_position=77,
        replay_frame_ids=tuple(range(64)),
    )
    native.step_entered = threading.Event()
    native.step_continue = threading.Event()
    native.step_block_at_count = 64
    try:
        assert shadow.stage(request, listener_generation_id=7, sequence_position=77)
        assert native.step_entered.wait(5)
        assert len(native.manifests) == 64
        assert shadow._candidate is None
        assert native.cache_copy_source_rows == []
        native.step_continue.set()
        candidate = _wait_for_shadow_candidate(shadow, timeout=5.0)
        assert candidate is not None
        imported = native.import_rows[0]
        assert (
            imported.frame_id,
            imported.listener_state_version,
            imported.sequence_position,
            imported.cache_position,
        ) == (0, 5, 13, request.prefix_len)
        assert len(native.manifests) == 64
        assert [manifest.rows[0].frame_id for manifest in native.manifests] == list(range(64))
        assert [manifest.rows[0].cache_position for manifest in native.manifests] == [
            request.prefix_len + 2 * index for index in range(64)
        ]
        assert [manifest.rows[0].listener_state_version for manifest in native.manifests] == [
            5 + index for index in range(64)
        ]
        assert [manifest.rows[0].sequence_position for manifest in native.manifests] == [
            13 + index for index in range(64)
        ]
        assert [float(audio[0, 0, 0]) for audio in native.audio_rows] == list(range(64))
        assert len(native.cache_copy_source_rows) == 1
        assert (
            candidate.row.frame_id,
            candidate.row.listener_state_version,
            candidate.row.sequence_position,
            candidate.row.cache_position,
        ) == (64, 69, 77, request.cache_position)
        assert candidate.stage.request is request
        assert shadow.candidate_for(candidate.row) is candidate
    finally:
        shadow.stop().result(timeout=3)


@pytest.mark.parametrize(
    ("target_frame_id", "target_state_version", "target_sequence_position", "frame_ids", "message"),
    (
        (0, 1, 1, (0,), "frame"),
        (1, 0, 1, (0,), "state_version"),
        (1, 1, 0, (0,), "sequence_position"),
    ),
)
def test_shadow_prefix_rejects_replay_cursor_underflow(
    target_frame_id, target_state_version, target_sequence_position, frame_ids, message,
) -> None:
    shadow, _native, _stream = _start_shadow_worker()
    request = _shadow_history_request(
        "shadow-underflow",
        1,
        target_frame_id=target_frame_id,
        target_state_version=target_state_version,
        target_sequence_position=target_sequence_position,
        replay_frame_ids=frame_ids,
    )
    try:
        with pytest.raises(ValueError, match=message):
            shadow.stage(
                request,
                listener_generation_id=7,
                sequence_position=target_sequence_position,
            )
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_prefix_rejects_noncontiguous_frozen_frame_ids() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    request = _shadow_history_request(
        "shadow-noncontiguous",
        1,
        target_frame_id=5,
        target_state_version=5,
        target_sequence_position=5,
        replay_frame_ids=(2, 4),
    )
    try:
        with pytest.raises(ValueError, match="contiguous"):
            shadow.stage(request, listener_generation_id=7, sequence_position=5)
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_prefix_rejects_target_cache_cursor_mismatch() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    request = _shadow_history_request(
        "shadow-cache-mismatch",
        1,
        target_frame_id=2,
        target_state_version=2,
        target_sequence_position=2,
        replay_frame_ids=(0, 1),
    )
    request.cache_position += 1
    try:
        with pytest.raises(ValueError, match="cache cursor"):
            shadow.stage(request, listener_generation_id=7, sequence_position=2)
    finally:
        shadow.stop().result(timeout=3)


@pytest.mark.parametrize("invalid_value", (True, 1.5, "1"))
def test_shadow_prefix_rejects_invalid_frozen_cursor_types(invalid_value) -> None:
    shadow, _native, _stream = _start_shadow_worker()
    request = _shadow_history_request(
        "shadow-invalid-cursor-type",
        1,
        target_frame_id=1,
        target_state_version=1,
        target_sequence_position=1,
        replay_frame_ids=(0,),
    )
    request.listener_state_version = invalid_value
    try:
        with pytest.raises(ValueError, match="listener_state_version"):
            shadow.stage(request, listener_generation_id=7, sequence_position=1)
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_prefix_rejects_boolean_historical_frame_id() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    request = _shadow_history_request(
        "shadow-invalid-frame-type",
        1,
        target_frame_id=1,
        target_state_version=1,
        target_sequence_position=1,
        replay_frame_ids=(True,),
    )
    try:
        with pytest.raises(ValueError, match="replay_frame_id"):
            shadow.stage(request, listener_generation_id=7, sequence_position=1)
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_candidate_is_not_visible_until_full_replay_and_invalidation_retires_partial_owner() -> None:
    shadow, native, _stream = _start_shadow_worker(
        builder=lambda request: _prefix_only_import(request)
    )
    native.step_entered = threading.Event()
    native.step_continue = threading.Event()
    request = _shadow_history_request(
        "shadow-reset-during-replay",
        8,
        target_frame_id=3,
        target_state_version=7,
        target_sequence_position=9,
        replay_frame_ids=(1, 2),
    )
    try:
        assert shadow.stage(request, listener_generation_id=7, sequence_position=9)
        assert native.step_entered.wait(2)
        assert shadow._candidate is None
        assert native.cache_copy_source_rows == []
        shadow.invalidate_session(request.session_id)
        native.step_continue.set()
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and not native.releases:
            time.sleep(0.005)
        assert shadow._candidate is None
        assert native.releases
        assert native.releases[0][0] == request.session_id
        assert native.releases[0][1:] == (2, request.prefix_len + 2)
        assert native.cache_copy_source_rows == []
        assert shadow.status["active_session_id"] is None
    finally:
        native.step_continue.set()
        shadow.stop().result(timeout=3)


def test_shadow_stage_replacement_during_replay_retires_old_cursor_before_new_candidate() -> None:
    shadow, native, _stream = _start_shadow_worker(
        builder=lambda request: _prefix_only_import(request)
    )
    native.step_entered = threading.Event()
    native.step_continue = threading.Event()
    request_r1 = _shadow_history_request(
        "shadow-replaced-during-replay",
        1,
        target_frame_id=2,
        target_state_version=7,
        target_sequence_position=9,
        replay_frame_ids=(0, 1),
    )
    request_r2_fields = vars(request_r1).copy()
    request_r2_fields["semantic_context_revision"] = 2
    request_r2 = SimpleNamespace(**request_r2_fields)
    try:
        assert shadow.stage(request_r1, listener_generation_id=7, sequence_position=9)
        assert native.step_entered.wait(2)
        assert shadow._candidate is None
        assert native.cache_copy_source_rows == []

        assert shadow.stage(request_r2, listener_generation_id=7, sequence_position=9)
        native.step_continue.set()
        candidate = _wait_for_shadow_candidate(shadow, timeout=4.0)

        assert candidate is not None
        assert candidate.stage.request is request_r2
        assert candidate.stage.request.semantic_context_revision == 2
        assert candidate.row.frame_id == request_r2.current_frame_id == 2
        assert candidate.row.listener_state_version == request_r2.listener_state_version == 7
        assert candidate.row.sequence_position == 9
        assert len(native.cache_copy_source_rows) == 1
        assert native.cache_copy_source_rows[0] == candidate.row
        assert native.releases[0][1:] == (1, request_r1.prefix_len + 2)
        assert shadow.status["latest_error"] is None
    finally:
        native.step_continue.set()
        shadow.stop().result(timeout=3)


def test_shadow_q2_published_during_history_replay_runs_after_exact_target_cursor() -> None:
    shadow, native, _stream = _start_shadow_worker(
        builder=lambda request: _prefix_only_import(request)
    )
    native.step_entered = threading.Event()
    native.step_continue = threading.Event()
    session_id = "shadow-q2-during-history-replay"
    request = _shadow_history_request(
        session_id,
        3,
        target_frame_id=2,
        target_state_version=7,
        target_sequence_position=9,
        replay_frame_ids=(0, 1),
    )
    state = _state(_signature(), session_id)
    live_row = replace(
        _shadow_row(session_id, 2, cache_position=request.cache_position),
        listener_state_version=request.listener_state_version,
        sequence_position=9,
        valid_history_tokens=request.cache_position,
    )
    try:
        assert shadow.stage(request, listener_generation_id=7, sequence_position=9)
        assert native.step_entered.wait(2)
        assert shadow._candidate is None
        assert shadow.publish_q2(ShadowQ2Frame(
            frame=_frame(state, frame_id=2),
            row=live_row,
            audio_soft=torch.full((2, HIDDEN_WIDTH), 9.0, dtype=torch.bfloat16),
            ready_event=None,
        ))
        native.step_continue.set()

        deadline = time.monotonic() + 4.0
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            q2 = shadow.status["diagnostics"]["q2"]
            if candidate is not None and q2["completed_count"] == 1:
                break
            time.sleep(0.005)

        assert candidate is not None
        assert candidate.stage.request is request
        assert [manifest.rows[0].frame_id for manifest in native.manifests] == [0, 1, 2]
        assert [float(audio[0, 0, 0]) for audio in native.audio_rows] == [0.0, 1.0, 9.0]
        assert candidate.row.frame_id == 3
        assert candidate.row.listener_state_version == 8
        assert candidate.row.sequence_position == 10
        assert candidate.row.cache_position == request.cache_position + 2
        assert candidate.catchup_frames == 1
        assert native.cache_copy_source_rows[-1] == candidate.row
        assert shadow.status["diagnostics"]["q2"]["submitted_count"] == 1
        assert shadow.status["diagnostics"]["q2"]["completed_count"] == 1
        assert shadow.status["latest_error"] is None
    finally:
        native.step_continue.set()
        shadow.stop().result(timeout=3)


def test_shadow_diagnostic_snapshot_tracks_stage_q2_and_cursor_rejection() -> None:
    def builder(request):
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, request.prefix_len), dtype=torch.int64),
            prefix_len=request.prefix_len,
            cache_position=request.cache_position,
            semantic_context_revision=request.semantic_context_revision,
            prefix_build_ms=1.25,
            replay_ms=2.5,
            fair_lock_wait_ms=0.75,
            teacher_prefix_fenced_wall_ms=5.0,
        )

    shadow, _native, _stream = _start_shadow_worker(builder=builder)
    session_id = "shadow-diagnostic-contract"
    request = _shadow_request(session_id, 12, 0)
    try:
        assert shadow.stage(
            request,
            listener_generation_id=7,
            sequence_position=0,
        )
        deadline = time.monotonic() + 2
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            diagnostics = shadow.status["diagnostics"]
            if candidate is not None and diagnostics["current_job"]["phase"] == "candidate_ready":
                break
            time.sleep(0.005)
        assert candidate is not None
        diagnostics = shadow.status["diagnostics"]
        current_job = diagnostics["current_job"]
        assert current_job["session_id"] == session_id
        assert current_job["listener_generation_id"] == 7
        assert current_job["state_epoch_id"] == request.state_epoch_id
        assert current_job["semantic_context_revision"] == 12
        assert current_job["request_base_frame_id"] == 0
        assert current_job["phase"] == "candidate_ready"
        assert current_job["stage_requested_at_ns"] <= current_job["stage_queued_at_ns"]
        assert current_job["stage_queued_at_ns"] <= current_job["stage_execution_started_at_ns"]
        assert current_job["fair_lock_wait_ms"] == 0.75
        assert current_job["teacher_prefix_build_wall_ms"] == 1.25
        assert current_job["teacher_prefix_replay_wall_ms"] == 2.5
        assert current_job["teacher_prefix_fenced_wall_ms"] == 5.0
        assert current_job["shadow_native_prefix_import_ms"] is not None

        assert shadow.candidate_for(replace(candidate.row, frame_id=1)) is None
        diagnostics = shadow.status["diagnostics"]
        assert diagnostics["candidate_cursor_rejections"] == 1
        assert diagnostics["candidate_last_rejection_reason"] == "listener_cursor_mismatch"

        state = _state(_signature(), session_id, generation=7)
        assert shadow.publish_q2(ShadowQ2Frame(
            frame=_frame(state, frame_id=0),
            row=candidate.row,
            audio_soft=torch.zeros((2, HIDDEN_WIDTH), dtype=torch.bfloat16),
            ready_event=None,
        ))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            q2 = shadow.status["diagnostics"]["q2"]
            if q2["completed_count"] == 1:
                break
            time.sleep(0.005)
        q2 = shadow.status["diagnostics"]["q2"]
        assert q2["submitted_count"] == 1
        assert q2["completed_count"] == 1
        assert q2["last_submitted_frame_id"] == 0
        assert q2["last_completed_frame_id"] == 0
        assert q2["pending_depth"] == 0
        assert q2["max_pending_depth"] == 1
        assert q2["timings"]["fence"]["count"] == 1
        assert q2["timings"]["output_health_validation"]["count"] == 1
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_discards_stale_queued_watermark_then_accepts_fresh_q2_suffix() -> None:
    shadow, native, _stream = _start_shadow_worker()
    holder_id = "stale-shadow-holder"
    session_id = "stale-shadow-watermark"
    try:
        # The shadow worker has one native slot. Keep it occupied so the
        # second SID's snapshot remains queued while its next real Q2 arrives.
        assert shadow.stage(
            _shadow_request(holder_id, 1, 0),
            listener_generation_id=7,
            sequence_position=0,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None and candidate.row.session_id == holder_id:
                break
            time.sleep(0.005)
        assert candidate is not None and candidate.row.session_id == holder_id

        assert shadow.stage(
            _shadow_request(session_id, 3, 3),
            listener_generation_id=7,
            sequence_position=3,
        )
        state = ListenerSessionState(
            session_id=session_id,
            generation_id=7,
            state_epoch_id=f"epoch-{session_id}-7",
            runtime_signature=_signature(),
            native_context_id=77,
        )
        assert shadow.publish_q2(ShadowQ2Frame(
            frame=_frame(state, frame_id=4),
            row=_shadow_row(session_id, 4),
            audio_soft=torch.zeros((2, HIDDEN_WIDTH), dtype=torch.bfloat16),
            ready_event=None,
        ))
        shadow.invalidate_session(holder_id)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and "watermark stale" not in str(shadow.status["latest_error"]):
            time.sleep(0.005)
        assert "watermark stale" in str(shadow.status["latest_error"])
        assert shadow._thread is not None and shadow._thread.is_alive()
        assert [sid for _context_id, sid, _frame_id in native.imports] == [holder_id]

        assert shadow.stage(
            _shadow_request(session_id, 3, 4),
            listener_generation_id=7,
            sequence_position=4,
        )
        assert shadow.publish_q2(ShadowQ2Frame(
            frame=_frame(state, frame_id=4),
            row=_shadow_row(session_id, 4),
            audio_soft=torch.zeros((2, HIDDEN_WIDTH), dtype=torch.bfloat16),
            ready_event=None,
        ))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not native.steps:
            time.sleep(0.005)
        if not native.steps:
            debug_state = {
                "status": dict(shadow.status),
                "active_row": shadow._active_row,
                "candidate_row": None if shadow._candidate is None else shadow._candidate.row,
                "q2_processing": shadow._q2_processing,
                "q2_rows": {
                    sid: [row.frame.frame_id for row in rows]
                    for sid, rows in shadow._q2.items()
                },
                "native_imports": native.imports,
            }
        assert native.steps, repr(debug_state) if not native.steps else None
        assert native.imports[-1][1:] == (session_id, 4)
        assert native.steps == [((session_id,), native.owner_thread_id)]
    finally:
        shadow.stop().result(timeout=3)


def test_shadow_claim_pins_source_across_new_revision_until_copy_fence() -> None:
    shadow, native, stream = _start_shadow_worker()
    session_id = "claimed-shadow-copy"
    try:
        assert shadow.stage(
            _shadow_request(session_id, 1, 0),
            listener_generation_id=7,
            sequence_position=0,
        )
        deadline = time.monotonic() + 2
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None:
                break
            time.sleep(0.005)
        assert candidate is not None
        claimed = shadow.candidate_for(candidate.row)
        assert claimed is candidate

        # A new semantic revision makes the source obsolete while the primary
        # copy is in flight. The shadow owner must retain its native slot.
        assert shadow.stage(
            _shadow_request(session_id, 2, 0),
            listener_generation_id=7,
            sequence_position=0,
        )
        with shadow._condition:
            assert shadow._stage_obsolete_locked(candidate.stage)
        with pytest.raises(RuntimeError, match="claimed shadow source"):
            shadow._release_active_row(candidate.row)
        time.sleep(0.04)
        assert not native.releases
        assert shadow._candidate is candidate

        copy_event = object()
        shadow.finish_promotion(
            candidate,
            SimpleNamespace(destination_row=candidate.row, completion_event=copy_event),
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not native.releases:
            time.sleep(0.005)
        assert native.releases
        assert copy_event in stream.waited_events
        assert native.release_waited_event_counts[0] >= 1

        # Promotion applies revision 1 at the matched cursor, then the queued
        # revision 2 is allowed to build in the same private slot.
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and (
            shadow._candidate is None
            or shadow._candidate.stage.request.semantic_context_revision != 2
        ):
            time.sleep(0.005)
        assert shadow._candidate is not None
        assert shadow._candidate.stage.request.semantic_context_revision == 2
        shadow.invalidate_session(session_id)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and len(native.releases) < 2:
            time.sleep(0.005)
        assert len(native.releases) == 2
    finally:
        shadow.stop().result(timeout=3)


def test_primary_applied_initial_revision_cancels_queued_stage_and_q2_backlog() -> None:
    shadow, native, _stream = _start_shadow_worker()
    holder_id = "primary-applied-shadow-holder"
    session_id = "primary-applied-initial"
    try:
        assert shadow.stage(
            _shadow_request(holder_id, 1, 0),
            listener_generation_id=7,
            sequence_position=0,
        )
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None and candidate.row.session_id == holder_id:
                break
            time.sleep(0.005)
        assert candidate is not None and candidate.row.session_id == holder_id

        request = _shadow_request(session_id, 4, 0)
        assert shadow.stage(
            request,
            listener_generation_id=7,
            sequence_position=0,
        )
        shadow.mark_primary_applied(request, listener_generation_id=7)

        state = ListenerSessionState(
            session_id=session_id,
            generation_id=7,
            state_epoch_id=request.state_epoch_id,
            runtime_signature=_signature(),
            native_context_id=78,
        )
        for frame_id in range(65):
            assert not shadow.publish_q2(ShadowQ2Frame(
                frame=_frame(state, frame_id=frame_id),
                row=_shadow_row(session_id, frame_id),
                audio_soft=torch.zeros((2, HIDDEN_WIDTH), dtype=torch.bfloat16),
                ready_event=None,
            ))
        status = shadow.status
        assert status["latest_error"] is None
        assert status["pending_q2"] == 0
        assert status["queued_sessions"] == 0
        assert session_id not in shadow._desired
        assert all(imported_session != session_id for _ctx, imported_session, _frame in native.imports)
        candidate_status = status["candidate_stage"]
        assert candidate_status["session_id"] == holder_id
        assert candidate_status["listener_generation_id"] == 7
        assert candidate_status["state_epoch_id"] == f"epoch-{holder_id}-7"
        assert candidate_status["semantic_context_revision"] == 1
        assert candidate_status["request_current_frame_id"] == 0
        assert candidate_status["current_frame_id"] == 0
        assert candidate_status["listener_state_version"] == 0
        assert candidate_status["prefix_len"] == 50
        assert candidate_status["replay_count"] == 0
        assert candidate_status["replay_first_q2_frame_id"] is None
        assert candidate_status["replay_last_q2_frame_id"] is None
        applied = status["primary_applied"]
        assert applied[0]["session_id"] == session_id
        assert applied[0]["semantic_context_revision"] == 4
        assert applied[0]["request_current_frame_id"] == 0
        assert applied[0]["current_frame_id"] == 0
        assert applied[0]["listener_state_version"] == 0
        assert applied[0]["prefix_len"] == 50
        assert applied[0]["replay_count"] == 0
        assert applied[0]["replay_first_q2_frame_id"] is None
        assert applied[0]["replay_last_q2_frame_id"] is None
    finally:
        shadow.invalidate_session(holder_id)
        shadow.stop().result(timeout=3)


def test_primary_applied_revision_does_not_suppress_next_revision() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    session_id = "primary-applied-next-revision"
    try:
        applied = _shadow_request(session_id, 8, 0)
        assert shadow.stage(applied, listener_generation_id=7, sequence_position=0)
        shadow.mark_primary_applied(applied, listener_generation_id=7)

        next_request = _shadow_request(session_id, 9, 0)
        assert shadow.stage(next_request, listener_generation_id=7, sequence_position=0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None and candidate.stage.request.semantic_context_revision == 9:
                break
            time.sleep(0.005)
        assert candidate is not None
        assert candidate.stage.request.semantic_context_revision == 9
    finally:
        shadow.invalidate_session(session_id)
        shadow.stop().result(timeout=3)


def test_committing_old_prefix_does_not_cancel_newer_staged_revision() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    session_id = "primary-applied-old-row-keeps-new-stage"
    try:
        newer = _shadow_request(session_id, 2, 4)
        assert shadow.stage(newer, listener_generation_id=7, sequence_position=4)
        old_row = SimpleNamespace(
            session_id=session_id,
            state_epoch_id=newer.state_epoch_id,
            semantic_context_revision=1,
            semantic_prefix_len=50,
            semantic_prefix_replay_frame_ids=(),
            frame_id=9,
        )
        shadow.mark_primary_applied(
            old_row,
            listener_generation_id=7,
            current_frame_id=10,
            listener_state_version=10,
        )

        deadline = time.monotonic() + 2
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None and candidate.stage.request.semantic_context_revision == 2:
                break
            time.sleep(0.005)
        assert candidate is not None
        assert candidate.stage.request.semantic_context_revision == 2
        assert shadow.status["primary_applied"][0]["semantic_context_revision"] == 1
    finally:
        shadow.invalidate_session(session_id)
        shadow.stop().result(timeout=3)


def test_primary_applied_row_keeps_unknown_prefix_base_separate_from_commit_frame() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    try:
        row_output = SimpleNamespace(
            session_id="primary-applied-row-metadata",
            state_epoch_id="epoch-primary-applied-row-metadata-7",
            semantic_context_revision=4,
            semantic_prefix_len=50,
            semantic_prefix_replay_frame_ids=(8, 9),
            frame_id=9,
        )
        shadow.mark_primary_applied(
            row_output,
            listener_generation_id=7,
            current_frame_id=10,
            listener_state_version=11,
        )

        applied = shadow.status["primary_applied"][0]
        assert applied["request_current_frame_id"] is None
        assert applied["request_current_frame_source"] == "unknown"
        assert applied["current_frame_id"] == 10
        assert applied["listener_state_version"] == 11
        assert applied["replay_first_q2_frame_id"] == 8
        assert applied["replay_last_q2_frame_id"] == 9
    finally:
        shadow.stop().result(timeout=3)


def test_primary_applied_revision_is_scoped_to_generation_and_epoch() -> None:
    shadow, _native, _stream = _start_shadow_worker()
    session_id = "primary-applied-new-identity"
    try:
        old_request = _shadow_request(session_id, 5, 0)
        shadow.mark_primary_applied(old_request, listener_generation_id=7)

        new_request = SimpleNamespace(
            **{**vars(old_request), "state_epoch_id": "epoch-after-reset"}
        )
        assert shadow.stage(new_request, listener_generation_id=8, sequence_position=0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None:
                break
            time.sleep(0.005)
        assert candidate is not None
        assert candidate.stage.listener_generation_id == 8
        assert candidate.stage.request.state_epoch_id == "epoch-after-reset"
        assert candidate.stage.request.semantic_context_revision == 5
    finally:
        shadow.invalidate_session(session_id)
        shadow.stop().result(timeout=3)


def test_primary_applied_notification_preserves_claimed_copy_until_fence() -> None:
    shadow, native, stream = _start_shadow_worker()
    session_id = "primary-applied-claimed-copy"
    release_fence = threading.Event()
    try:
        request = _shadow_request(session_id, 6, 0)
        assert shadow.stage(request, listener_generation_id=7, sequence_position=0)
        deadline = time.monotonic() + 2
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None:
                break
            time.sleep(0.005)
        assert candidate is not None
        assert shadow.candidate_for(candidate.row) is candidate

        copy_event = object()
        fence_entered = threading.Event()
        wait_event = stream.wait_event

        def wait_for_copy_fence(event: Any) -> None:
            if event is copy_event:
                fence_entered.set()
                assert release_fence.wait(2)
            wait_event(event)

        stream.wait_event = wait_for_copy_fence
        shadow.finish_promotion(
            candidate,
            SimpleNamespace(destination_row=candidate.row, completion_event=copy_event),
        )
        assert fence_entered.wait(2)
        shadow.mark_primary_applied(request, listener_generation_id=7)
        assert shadow._candidate is candidate
        assert not native.releases

        release_fence.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not native.releases:
            time.sleep(0.005)
        assert copy_event in stream.waited_events
        assert native.releases
        assert native.release_waited_event_counts[0] >= 1
        assert shadow._candidate is None
    finally:
        release_fence.set()
        shadow.invalidate_session(session_id)
        shadow.stop().result(timeout=3)


def test_shadow_candidate_cannot_be_claimed_while_a_real_q2_step_is_inflight() -> None:
    class BlockingNative(_FakeNativeBatch):
        def __init__(self, signature, capability, stream):
            super().__init__(signature, capability, stream)
            self.step_entered = threading.Event()
            self.release_step = threading.Event()

        def step_async(self, manifest, audio_soft, *, producer_stream=None):
            self._claim()
            self.step_entered.set()
            assert self.release_step.wait(3)
            return super().step_async(manifest, audio_soft, producer_stream=producer_stream)

    shadow, native, _stream = _start_shadow_worker(native_factory=BlockingNative)
    session_id = "q2-inflight-shadow"
    try:
        assert shadow.stage(
            _shadow_request(session_id, 1, 0),
            listener_generation_id=7,
            sequence_position=0,
        )
        deadline = time.monotonic() + 2
        candidate = None
        while time.monotonic() < deadline:
            candidate = shadow._candidate
            if candidate is not None:
                break
            time.sleep(0.005)
        assert candidate is not None
        state = ListenerSessionState(
            session_id=session_id,
            generation_id=7,
            state_epoch_id=f"epoch-{session_id}-7",
            runtime_signature=_signature(),
            native_context_id=78,
        )
        assert shadow.publish_q2(ShadowQ2Frame(
            frame=_frame(state, frame_id=0),
            row=candidate.row,
            audio_soft=torch.zeros((2, HIDDEN_WIDTH), dtype=torch.bfloat16),
            ready_event=None,
        ))
        entered = native.step_entered.wait(2)
        if not entered:
            with shadow._condition:
                debug_state = {
                    "status": dict(shadow.status),
                    "active_row": shadow._active_row,
                    "q2_processing": shadow._q2_processing,
                    "q2_rows": {
                        sid: [row.frame.frame_id for row in rows]
                        for sid, rows in shadow._q2.items()
                    },
                    "candidate_claimed": shadow._candidate_claimed,
                    "native_steps": native.steps,
                    "native_imports": native.imports,
                }
        assert entered, repr(debug_state) if not entered else None
        assert shadow.candidate_for(candidate.row) is None
        native.release_step.set()

        deadline = time.monotonic() + 2
        advanced = None
        while time.monotonic() < deadline:
            advanced = shadow._candidate
            if advanced is not None and advanced.row.frame_id == 1:
                break
            time.sleep(0.005)
        assert advanced is not None and advanced.row.frame_id == 1
        claimed = shadow.candidate_for(advanced.row)
        assert claimed is advanced
        shadow.discard_candidate(advanced, "test teardown", copy_not_enqueued=True)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and not native.releases:
            time.sleep(0.005)
        assert native.releases
    finally:
        native.release_step.set()
        shadow.stop().result(timeout=3)


def test_private_listener_pipeline_runs_while_compute_owns_shared_model_lock() -> None:
    lock = FairModelLock(host_timeline_enabled=True)
    compute_acquired = threading.Event()
    release_compute = threading.Event()
    compute_thread_id: list[int] = []
    completion_while_compute_owned = threading.Event()

    def hold_compute_lock() -> None:
        assert lock.acquire_fair("compute")
        compute_thread_id.append(threading.get_ident())
        compute_acquired.set()
        assert release_compute.wait(4)
        lock.release()

    def completion_barrier(_stream) -> None:
        owner = lock.host_timeline_snapshot()["current_owner"]
        assert owner is not None and owner["priority"] == "compute"
        assert owner["thread_id"] == compute_thread_id[0]
        completion_while_compute_owned.set()

    rig = _rig(model_lock=lock, completion_barrier=completion_barrier)
    state = _state(rig.coordinator.runtime_signature, "private-lock-scope")
    _register(rig, state)
    compute_thread = threading.Thread(target=hold_compute_lock, daemon=True)
    compute_thread.start()
    try:
        assert compute_acquired.wait(2)
        output = rig.worker.submit_tick((_frame(state),)).result(timeout=4)
        assert completion_while_compute_owned.is_set()
        assert tuple(output) == (state.session_id,)
        row_output = output[state.session_id]
        assert row_output.listener_model_lock_wait_ms is None
        assert row_output.listener_model_lock_hold_ms is None
        assert row_output.capacity_growth_ms is not None
        assert row_output.frontend_encode_ms is not None
        assert row_output.mel_prepare_ms is not None and row_output.mel_prepare_ms >= 0.0
        assert row_output.frontend_copy_ms is not None and row_output.frontend_copy_ms >= 0.0
        assert row_output.frontend_copy_cuda_ms is None
        assert row_output.audio_tower_lock_wait_ms is not None
        assert row_output.audio_tower_lock_hold_ms is not None
        assert row_output.audio_tower_cuda_ms is None
        assert row_output.native_submit_ms is not None
        assert row_output.native_output_health_ms is not None
        assert row_output.native_output_health_ms >= 0.0
        assert row_output.control6_policy_ms is not None
        assert row_output.cuda_completion_wait_ms is not None
        assert row_output.native_output_health is not None
        assert row_output.native_output_health.status == "healthy"
        assert row_output.native_output_health.checked
        assert rig.native is not None
        assert rig.native.lock_owners_at_import == [
            {"thread_id": compute_thread_id[0], "priority": "compute", "ticket": 0, "depth": 1}
        ]
    finally:
        release_compute.set()
        compute_thread.join(timeout=3)
        rig.worker.stop(drain=True).result(timeout=3)
    assert not compute_thread.is_alive()


def test_shared_model_lock_wraps_only_prefix_importer_that_requires_it() -> None:
    lock = FairModelLock(host_timeline_enabled=True)
    rig = _rig(model_lock=lock)
    state = _state(rig.coordinator.runtime_signature, "semantic-lock-scope")
    _register(rig, state)
    first = rig.worker.submit_tick((_frame(state),)).result(timeout=4)[state.session_id]
    assert first.listener_model_lock_wait_ms is None
    assert first.listener_model_lock_hold_ms is None

    importer_owner: list[dict[str, Any] | None] = []

    def semantic_import(_state, _row):
        importer_owner.append(lock.host_timeline_snapshot()["current_owner"])
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 50), dtype=torch.int64),
            prefix_len=50,
            cache_position=50,
        )

    setattr(semantic_import, "requires_shared_lm_lock", True)
    rig.worker.refresh_session_prefix(
        state.session_id,
        expected_frame_id=1,
        expected_state_version=1,
        cache_position=50,
        prefix_import=semantic_import,
    ).result(timeout=3)
    second = rig.worker.submit_tick((_frame(state, frame_id=1),)).result(timeout=4)[state.session_id]

    assert importer_owner and importer_owner[0] is not None
    assert importer_owner[0]["priority"] == "listener"
    assert importer_owner[0]["thread_id"] == rig.native.owner_thread_id
    assert rig.native.lock_owners_at_import[-1] is None
    assert second.listener_model_lock_wait_ms is not None
    assert second.listener_model_lock_hold_ms is not None
    assert second.semantic_prefix_import_ms is not None
    events = lock.host_timeline_snapshot()["events"]
    assert [event["event"] for event in events].count("request") == 1
    assert [event["priority"] for event in events if event["event"] == "request"] == ["listener"]
    rig.worker.stop(drain=True).result(timeout=3)


def test_dynamic_capacity_growth_failure_rejects_new_session_without_fixed_limit() -> None:
    rig = _rig()
    states = tuple(
        _state(rig.coordinator.runtime_signature, f"capacity-{index}")
        for index in range(4)
    )
    _register(rig, *states)
    assert rig.native is not None
    assert rig.native.max_batch_size == 4
    rig.native.fail_capacity_at = 5

    fifth = _state(rig.coordinator.runtime_signature, "capacity-4")
    with pytest.raises(RuntimeError, match="capacity allocation failed: CUDA out of memory"):
        rig.worker.register_session(fifth, prefix_import=rig.prefix_import).result(timeout=3)
    with pytest.raises(KeyError, match="unknown listener session"):
        rig.worker.session_state(fifth.session_id).result(timeout=3)
    assert rig.native.max_batch_size == 4
    assert rig.native.capacity_requests[-1] == 5
    rig.worker.stop(drain=True).result(timeout=3)


@pytest.mark.parametrize("batch_size", (1, 2, 3, 4, 5, 8, 12))
def test_worker_uses_one_frontend_native_step_and_control6_forward_per_batch(batch_size: int) -> None:
    rig = _rig()
    signature = rig.coordinator.runtime_signature
    states = tuple(_state(signature, f"sid-{row}") for row in range(batch_size))
    _register(rig, *states)
    aligned_ready_at_ns = time.monotonic_ns() + 25_000_000
    tick_output = rig.worker.submit_tick(tuple(
        _frame(state, value=0.125 + row, ready_at_ns=aligned_ready_at_ns)
        for row, state in enumerate(states)
    )).result(timeout=4)
    assert rig.result_event.wait(4)

    batch, decisions = rig.results[0]
    assert tuple(tick_output) == tuple(state.session_id for state in states)
    for row_output in tick_output.values():
        assert row_output.capacity_growth_ms is not None
        assert row_output.frontend_encode_ms is not None
        assert row_output.mel_prepare_ms is not None
        assert row_output.frontend_copy_ms is not None
        assert row_output.frontend_copy_cuda_ms is None
        assert row_output.audio_tower_lock_wait_ms is not None
        assert row_output.audio_tower_lock_hold_ms is not None
        assert row_output.audio_tower_cuda_ms is None
        assert row_output.native_submit_ms is not None
        assert row_output.control6_policy_ms is not None
        assert row_output.cuda_completion_wait_ms is not None
    assert batch.manifest.batch_size == batch_size
    assert len(rig.model.call_shapes) == 1
    valid_mel_rows = rig.model.call_inputs[0].shape[1]
    assert rig.model.call_shapes == [(batch_size, valid_mel_rows, 128)]
    assert valid_mel_rows == 8  # first tick has no prior user Mel; startup is unpadded
    assert len(rig.native.steps) == 1
    assert rig.native.health_generations == [1]
    assert rig.native.event_order.index(("health", 1)) < rig.native.event_order.index(("step", batch_size)) + 2
    assert rig.native.steps[0][0] == tuple(state.session_id for state in states)
    assert len(rig.adapter.hidden_rows) == 1
    assert len(decisions) == batch_size
    assert len(rig.native.imports) == batch_size
    assert rig.native.max_batch_size >= batch_size
    assert rig.native.capacity_requests[-1] == batch_size
    capacity_index = max(
        index for index, event in enumerate(rig.native.event_order)
        if event == ("ensure_capacity", batch_size)
    )
    first_import_index = next(
        index for index, event in enumerate(rig.native.event_order)
        if event[0] == "import"
    )
    assert capacity_index < first_import_index
    expected_first_queries = torch.tensor(
        [
            float(
                rig.model.call_inputs[0][row]
                .float()
                .reshape(valid_mel_rows // 4, 4, 128)[valid_mel_rows // 4 - 2]
                .mean()
            ) + (valid_mel_rows // 4 - 2)
            for row in range(batch_size)
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        rig.adapter.hidden_rows[0][:, 0],
        expected_first_queries,
        rtol=0,
        atol=0,
    )

    worker_thread = rig.runtime_factory_threads[0]
    assert worker_thread == rig.native.owner_thread_id
    assert worker_thread == rig.native.creation_thread_id
    assert worker_thread == rig.adapter.creation_thread_id
    assert worker_thread == rig.model.thread_ids[0]
    assert worker_thread == rig.adapter.thread_ids[0]
    assert rig.prefix_import_threads == [worker_thread] * batch_size
    assert {thread_id for _, thread_id in rig.coordinator.operation_threads} == {worker_thread}
    assert all(decision.session == state.session_id for decision, state in zip(decisions, states, strict=True))
    for state in states:
        snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
        assert snapshot.next_frame_id == 1
        assert snapshot.state_version == 1
        assert snapshot.cache_position == 36
        assert snapshot.valid_history_tokens == 36
        assert state.user_pcm_tail is not None
        assert len(state.user_mel_history) == 1

    report = rig.worker.stop(drain=True).result(timeout=3)
    assert isinstance(report, WorkerStopReport)
    assert report.stopped_cleanly
    assert report.processed_batches == 1
    assert rig.native.close_thread_id == worker_thread


def test_worker_serializes_text_appends_around_ephemeral_audio_without_cursor_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "unified-alternating")
    _register(rig, state)

    first = rig.worker.submit_tick((_frame(state, frame_id=0),)).result(timeout=4)
    assert first[state.session_id].cache_position == 36

    event_order: list[str] = []
    runtime = rig.worker._runtime
    assert runtime is not None
    adapter = runtime.token_adapter
    assert adapter is not None
    original_lm_head = adapter.logits_from_final_norm_hidden

    def record_lm_head(hidden: torch.Tensor) -> torch.Tensor:
        event_order.append("lm_head")
        return original_lm_head(hidden)

    monkeypatch.setattr(adapter, "logits_from_final_norm_hidden", record_lm_head)

    class ReadyEvent:
        recorded_stream: Any | None = None

        def record(self, stream: Any) -> None:
            self.recorded_stream = stream
            event_order.append("ready_event")

    ready_events: list[ReadyEvent] = []

    def make_event(**_kwargs: Any) -> ReadyEvent:
        event = ReadyEvent()
        ready_events.append(event)
        return event

    monkeypatch.setattr(torch.cuda, "Event", make_event)

    append = rig.worker.append_text_tokens(
        state.session_id, (101,), expected_generation_id=state.generation_id,
        expected_cache_position=36,
    ).result(timeout=4)
    assert tuple(append.logits.shape) == (1, 1)
    assert tuple(append.final_norm_hidden.shape) == (1, 1, HIDDEN_WIDTH)
    assert append.ready_event is ready_events[0]
    assert append.ready_event.recorded_stream is runtime.listener_stream
    assert event_order == ["lm_head", "ready_event"]
    after_text = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (after_text.next_frame_id, after_text.state_version, after_text.sequence_position) == (1, 1, 1)
    assert after_text.cache_position == after_text.valid_history_tokens == 37

    ephemeral = ListenerFrame(
        session_id=state.session_id,
        generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        frame_id=1,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.zeros(1280, dtype=np.float32),
        logical_q_len=0,
        ephemeral_reason="assistant_turn_open",
    )
    during_assistant = rig.worker.submit_tick((ephemeral,)).result(timeout=4)
    assert during_assistant[state.session_id].logical_q_len == 0
    assert during_assistant[state.session_id].ephemeral_reason == "assistant_turn_open"
    assert during_assistant[state.session_id].queue_audio_for_next_turn is False
    after_ephemeral = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (after_ephemeral.next_frame_id, after_ephemeral.state_version,
            after_ephemeral.sequence_position) == (2, 2, 2)
    assert after_ephemeral.cache_position == after_ephemeral.valid_history_tokens == 37

    rig.worker.append_text_tokens(
        state.session_id, (202,), expected_generation_id=state.generation_id,
        expected_cache_position=37,
    ).result(timeout=4)
    final = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (final.next_frame_id, final.state_version, final.sequence_position) == (2, 2, 2)
    assert final.cache_position == final.valid_history_tokens == 38
    assert rig.native is not None
    assert [manifest.rows[0].logical_q_len for manifest in rig.native.manifests] == [2, 1, 0, 1]
    assert [manifest.rows[0].frame_id for manifest in rig.native.manifests] == [0, 1, 1, 2]
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_speaking_q2_decision_view_replays_onset_queue_on_private_slot_once() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "speaking-q2-view")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=0,
    ).result(timeout=3)

    first = rig.worker.submit_tick((_assistant_q0(state, 0),)).result(timeout=4)[state.session_id]
    first_view = getattr(first, "decision_view", None)
    assert first_view is not None
    assert first.policy_decision is not None
    assert first.policy_decision.session == state.session_id
    assert first.policy_decision.diagnostics["native_output_provenance"]["native_session_id"].startswith(
        "__q2view__"
    )
    assert rig.adapter is not None and len(rig.adapter.hidden_rows) == 1
    assert first_view.source_q2_frame_ids == (0,)
    assert first_view.source_row.session_id == state.session_id
    assert first_view.fork_row.session_id != state.session_id
    assert first_view.fork_row.logical_q_len == 2
    assert first.logical_q_len == 0
    assert first_view.output_generation == first_view.native_output_health.generation
    assert first_view.native_output_health.row == first_view.fork_row
    assert first.native_output_health.row.session_id == state.session_id
    assert first.native_output_health.row.logical_q_len == 0
    assert first_view.output_generation != first.native_output_health.generation
    assert rig.adapter is not None
    torch.testing.assert_close(
        rig.adapter.hidden_rows[-1][0], first_view.gemma_hidden_t[0, 0].float()
    )
    assert torch.all(first.gemma_audio_queries_t == first_view.gemma_audio_queries_t - 7)

    second = rig.worker.submit_tick((_assistant_q0(state, 1),)).result(timeout=4)[state.session_id]
    second_view = getattr(second, "decision_view", None)
    assert second_view is not None
    assert second.policy_decision is not None
    assert rig.adapter is not None and len(rig.adapter.hidden_rows) == 2
    assert second_view.source_q2_frame_ids == (0, 1)
    assert second_view.source_row.frame_id == 1
    assert second_view.fork_row.frame_id == 1
    torch.testing.assert_close(
        rig.adapter.hidden_rows[-1][0], second_view.gemma_hidden_t[0, 0].float()
    )

    # The old onset Q2 and current Q2 are copied from the queue/frontend output;
    # the audio tower is run exactly once per submitted 80 ms frame.
    assert len(rig.model.call_shapes) == 2
    assert len(rig.native.audio_rows) == 5
    torch.testing.assert_close(rig.native.audio_rows[2], rig.native.audio_rows[1])
    torch.testing.assert_close(rig.native.audio_rows[3], rig.native.audio_rows[4])
    assert [len(rig.worker._queued_audio_soft[state.session_id])] == [2]
    assert all(
        session_id.startswith("__q2view__")
        for _source, destination in rig.native.slot_fork_calls
        for session_id in (destination.session_id,)
    )
    assert len(rig.native.slot_fork_calls) == 2
    assert len([item for item in rig.native.releases if item[0].startswith("__q2view__")]) == 2

    canonical = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (canonical.next_frame_id, canonical.state_version, canonical.sequence_position) == (2, 2, 2)
    assert canonical.cache_position == canonical.valid_history_tokens == 34
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_backchannel_private_decode_fork_isolated_budgeted_and_cancel_cleanup() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "private-backchannel")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    before = rig.worker.session_state(state.session_id).result(timeout=3)

    begin = getattr(rig.worker, "begin_ephemeral_decode", None)
    append = getattr(rig.worker, "append_private_tokens", None)
    append_transition = getattr(rig.worker, "append_private_transition_tokens", None)
    end = getattr(rig.worker, "end_ephemeral_decode", None)
    assert callable(begin) and callable(append) and callable(append_transition) and callable(end)
    handle = begin(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=before.cache_position,
    ).result(timeout=3)
    assert handle.parent_session_id == state.session_id
    assert handle.parent_generation_id == state.generation_id
    assert handle.cache_position == before.cache_position
    assert handle.fork_session_id != state.session_id
    assert state.session_id not in rig.worker._assistant_audio_queue_open

    # A canonical listener tick remains runnable while the private decode
    # branch is alive. Its KV cursor advances independently of the branch.
    tick = rig.worker.submit_tick((_frame(state, frame_id=0),)).result(timeout=4)
    main_after_tick = rig.worker.session_state(state.session_id).result(timeout=3)
    assert tick[state.session_id].cache_position == before.cache_position + 2
    assert main_after_tick.cache_position == before.cache_position + 2

    transitioned = append_transition(
        handle, (899, 900, 901), expected_cache_position=handle.cache_position,
    ).result(timeout=4)
    assert transitioned.cache_position == handle.cache_position + 3
    assert transitioned.token_count == 0
    with pytest.raises(RuntimeError, match="private transition is allowed once"):
        append_transition(
            handle, (902,), expected_cache_position=transitioned.cache_position,
        ).result(timeout=3)
    appended = append(
        handle, (903,), expected_cache_position=transitioned.cache_position,
    ).result(timeout=4)
    assert appended.cache_position == transitioned.cache_position + 1
    assert appended.token_count == 1
    assert tuple(appended.logits.shape) == (1, 1)
    assert tuple(appended.final_norm_hidden.shape) == (1, 1, HIDDEN_WIDTH)
    with pytest.raises(RuntimeError, match="private decode budget"):
        append(
            handle, (904, 905), expected_cache_position=appended.cache_position,
        ).result(timeout=3)

    end(handle).result(timeout=3)
    after = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (
        after.generation_id,
        after.state_epoch_id,
        after.cache_position,
        after.valid_history_tokens,
        after.next_frame_id,
    ) == (
        main_after_tick.generation_id,
        main_after_tick.state_epoch_id,
        main_after_tick.cache_position,
        main_after_tick.valid_history_tokens,
        main_after_tick.next_frame_id,
    )
    assert not rig.worker._assistant_audio_queue_open
    assert not rig.worker._queued_audio_soft[state.session_id]
    assert any(item[0] == handle.fork_session_id for item in rig.native.releases)
    with pytest.raises(RuntimeError, match="unknown private decode fork"):
        append(
            handle, (906,), expected_cache_position=appended.cache_position,
        ).result(timeout=3)

    # Once sampled-token decoding begins, transition IDs are no longer valid.
    second_handle = begin(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=main_after_tick.cache_position,
    ).result(timeout=3)
    sampled = append(
        second_handle, (907,), expected_cache_position=second_handle.cache_position,
    ).result(timeout=4)
    with pytest.raises(RuntimeError, match="private transition is allowed once"):
        append_transition(
            second_handle, (908,), expected_cache_position=sampled.cache_position,
        ).result(timeout=3)
    end(second_handle).result(timeout=3)

    even_handle = begin(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=main_after_tick.cache_position,
    ).result(timeout=3)
    even_transition = append_transition(
        even_handle, (909, 910, 911, 912),
        expected_cache_position=even_handle.cache_position,
    ).result(timeout=4)
    assert even_transition.cache_position == even_handle.cache_position + 4
    assert even_transition.token_count == 0
    end(even_handle).result(timeout=3)
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_parent_generation_reset_releases_live_private_decode_fork() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "private-backchannel-reset")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    handle = rig.worker.begin_ephemeral_decode(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=3)

    replacement = _state(rig.coordinator.runtime_signature, state.session_id, generation=8)
    rig.worker.reset_session(
        state.session_id, replacement, prefix_import=rig.prefix_import,
    ).result(timeout=3)
    assert any(item[0] == handle.fork_session_id for item in rig.native.releases)
    with pytest.raises(RuntimeError, match="unknown private decode fork"):
        rig.worker.append_private_tokens(
            handle, (905,), expected_cache_position=34,
        ).result(timeout=3)
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_assistant_queue_reservation_survives_text_append_then_replays_before_next_user_turn() -> None:
    rig = _rig()
    _set_fixture_capacity(rig, 40)
    state = _state(rig.coordinator.runtime_signature, "unified-reserve-success")
    _register(rig, state)
    _prime_native_prefix(rig, state)

    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=4,
    ).result(timeout=3)
    output = rig.worker.submit_tick((_assistant_q0(state),)).result(timeout=4)
    assert output[state.session_id].logical_q_len == 0
    assert output[state.session_id].ephemeral_reason == "assistant_turn_open"
    assert output[state.session_id].queue_audio_for_next_turn is True
    after_q0 = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (after_q0.next_frame_id, after_q0.state_version, after_q0.sequence_position) == (1, 1, 1)
    assert after_q0.cache_position == after_q0.valid_history_tokens == 34

    rig.worker.append_text_tokens(
        state.session_id,
        (201, 202),
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        consume_reserved_tokens=2,
    ).result(timeout=4)
    after_close = rig.worker.session_state(state.session_id).result(timeout=3)
    assert after_close.cache_position == after_close.valid_history_tokens == 36
    assert rig.worker._reserved_tail_tokens[state.session_id] == 2

    _mark_queued_soft_cuda_for_cpu_fixture(rig, state.session_id)
    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=36,
    ).result(timeout=4)
    assert drained.cache_position == drained.valid_history_tokens == 38
    assert (drained.next_frame_id, drained.state_version, drained.sequence_position) == (1, 1, 1)
    assert rig.worker._reserved_tail_tokens[state.session_id] == 0
    assert not rig.worker._queued_audio_soft[state.session_id]
    assert rig.native is not None
    assert rig.native.manifests[-1].rows[0].logical_q_len == 2
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_assistant_q2_drain_rejects_a_missing_frame_without_replaying_or_dropping() -> None:
    rig = _rig()
    _set_fixture_capacity(rig, 48)
    state = _state(rig.coordinator.runtime_signature, "unified-q2-gap")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=2,
    ).result(timeout=3)
    for frame_id, queue_for_next_turn in ((0, True), (1, True), (2, False), (3, True)):
        output = rig.worker.submit_tick((
            _assistant_q0(
                state,
                frame_id,
                queue_audio_for_next_turn=queue_for_next_turn,
            ),
        )).result(timeout=4)[state.session_id]
        if frame_id == 3:
            assert output.decision_view is None
            assert output.decision_view_unavailable_reason is not None
            assert output.policy_decision is not None
            assert output.policy_decision.diagnostics["decision_view_fallback_reason"]

    queued = rig.worker._queued_audio_soft[state.session_id]
    assert [item.source_frame_id for item in queued] == [0, 1, 3]
    for index, item in enumerate(tuple(queued)):
        queued[index] = replace(
            item,
            audio_soft_tokens=_CudaProvenanceTensor(item.audio_soft_tokens),
        )
    try:
        with pytest.raises(RuntimeError, match="non-contiguous Q2 frame IDs"):
            rig.worker.drain_queued_audio(
                state.session_id,
                expected_generation_id=state.generation_id,
                expected_cache_position=34,
            ).result(timeout=4)
        assert [item.source_frame_id for item in queued] == [0, 1, 3]
        assert rig.native is not None
        canonical_manifests = [
            manifest for manifest in rig.native.manifests
            if not manifest.rows[0].session_id.startswith("__q2view__")
        ]
        scratch_manifests = [
            manifest for manifest in rig.native.manifests
            if manifest.rows[0].session_id.startswith("__q2view__")
        ]
        assert len(canonical_manifests) == 4  # four listener frames
        assert len(scratch_manifests) == 3  # decision views only; drain did not run
    finally:
        rig.worker.stop(drain=False).result(timeout=3)


def test_backchannel_native_slot_fork_copies_generation_state_without_mutating_primary() -> None:
    rig = _rig()
    main = _state(rig.coordinator.runtime_signature, "backchannel-fork-main")
    main.user_pcm_tail = np.full(PCM_TAIL_SAMPLES, 0.1, dtype=np.float32)
    main.assistant_pcm_tail = np.full(PCM_TAIL_SAMPLES, 0.2, dtype=np.float32)
    main.user_mel_history = [torch.full((2, 4), 0.3)]
    main.assistant_mel_history = [torch.full((2, 4), 0.4)]
    _register(rig, main)
    _prime_native_prefix(rig, main)
    before = rig.worker.session_state(main.session_id).result(timeout=3)
    forked_session_id = rig.worker.fork_native_session(
        main.session_id, "backchannel-fork-secondary"
    ).result(timeout=4)
    forked = rig.worker._session_states[forked_session_id]
    after = rig.worker.session_state(main.session_id).result(timeout=3)

    assert (
        after.session_id,
        after.generation_id,
        after.state_epoch_id,
        after.next_frame_id,
        after.state_version,
        after.sequence_position,
        after.cache_position,
        after.valid_history_tokens,
        after.native_context_id,
    ) == (
        before.session_id,
        before.generation_id,
        before.state_epoch_id,
        before.next_frame_id,
        before.state_version,
        before.sequence_position,
        before.cache_position,
        before.valid_history_tokens,
        before.native_context_id,
    )
    np.testing.assert_array_equal(after.user_pcm_tail, before.user_pcm_tail)
    np.testing.assert_array_equal(after.assistant_pcm_tail, before.assistant_pcm_tail)
    assert after.user_mel_history == before.user_mel_history
    assert after.assistant_mel_history == before.assistant_mel_history
    assert forked_session_id == "backchannel-fork-secondary"
    assert forked.session_id == "backchannel-fork-secondary"
    assert forked.native_context_id != before.native_context_id
    assert forked.cache_position == before.cache_position
    assert forked.valid_history_tokens == before.valid_history_tokens
    assert rig.native is not None and len(rig.native.fork_calls) == 1
    source_row, destination_row = rig.native.fork_calls[0]
    assert source_row.session_id == main.session_id
    assert destination_row.session_id == "backchannel-fork-secondary"
    assert (source_row.cache_position, source_row.frame_id, source_row.sequence_position) == (
        destination_row.cache_position, destination_row.frame_id, destination_row.sequence_position
    )
    assert forked.user_pcm_tail is not main.user_pcm_tail
    assert np.array_equal(forked.user_pcm_tail, main.user_pcm_tail)
    assert forked.assistant_pcm_tail is not main.assistant_pcm_tail
    assert np.array_equal(forked.assistant_pcm_tail, main.assistant_pcm_tail)
    assert forked.user_mel_history[0] is not main.user_mel_history[0]
    assert torch.equal(forked.user_mel_history[0], main.user_mel_history[0])
    assert forked.assistant_mel_history[0] is not main.assistant_mel_history[0]
    assert torch.equal(forked.assistant_mel_history[0], main.assistant_mel_history[0])
    rig.worker.close_session("backchannel-fork-secondary").result(timeout=4)
    after_close = rig.worker.session_state(main.session_id).result(timeout=3)
    assert (
        after_close.generation_id,
        after_close.state_epoch_id,
        after_close.cache_position,
        after_close.valid_history_tokens,
        after_close.native_context_id,
    ) == (
        before.generation_id,
        before.state_epoch_id,
        before.cache_position,
        before.valid_history_tokens,
        before.native_context_id,
    )
    np.testing.assert_array_equal(after_close.user_pcm_tail, before.user_pcm_tail)
    np.testing.assert_array_equal(after_close.assistant_pcm_tail, before.assistant_pcm_tail)
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_backchannel_native_slot_fork_error_and_cancel_leave_primary_unchanged() -> None:
    rig = _rig()
    main = _state(rig.coordinator.runtime_signature, "backchannel-fork-fault-main")
    _register(rig, main)
    _prime_native_prefix(rig, main)
    before = rig.worker.session_state(main.session_id).result(timeout=3)
    assert rig.native is not None

    rig.native.fail_fork = RuntimeError("injected secondary copy failure")
    with pytest.raises(RuntimeError, match="injected secondary copy failure"):
        rig.worker.fork_native_session(
            main.session_id, "backchannel-fork-error"
        ).result(timeout=4)
    with pytest.raises(KeyError, match="backchannel-fork-error"):
        rig.worker.session_state("backchannel-fork-error").result(timeout=3)
    assert rig.worker.session_state(main.session_id).result(timeout=3) == before

    rig.native.fail_fork = None
    rig.native.fork_entered = threading.Event()
    rig.native.fork_continue = threading.Event()
    cancelled = rig.worker.fork_native_session(
        main.session_id, "backchannel-fork-cancelled"
    )
    assert rig.native.fork_entered.wait(3)
    assert cancelled.cancel()
    rig.native.fork_continue.set()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        state_future = rig.worker.session_state("backchannel-fork-cancelled")
        try:
            state_future.result(timeout=0.2)
        except KeyError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("cancelled secondary native slot was not retired")
    assert rig.worker.session_state(main.session_id).result(timeout=3) == before
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_cancel_releases_close_reservation_and_drains_queued_q2_into_open_user_turn() -> None:
    rig = _rig()
    _set_fixture_capacity(rig, 40)
    state = _state(rig.coordinator.runtime_signature, "unified-reserve-cancel")
    _register(rig, state)
    _prime_native_prefix(rig, state)

    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=4,
    ).result(timeout=3)
    rig.worker.submit_tick((_assistant_q0(state),)).result(timeout=4)
    rig.worker.append_text_tokens(
        state.session_id,
        (201,),
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        consume_reserved_tokens=1,
    ).result(timeout=4)
    rewound = rig.worker.rollback_assistant_branch(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=35,
        target_cache_position=34,
    ).result(timeout=4)
    assert rewound.cache_position == rewound.valid_history_tokens == 34
    assert rig.worker._reserved_tail_tokens[state.session_id] == 0

    _mark_queued_soft_cuda_for_cpu_fixture(rig, state.session_id)
    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        user_turn_already_open=True,
    ).result(timeout=4)
    assert drained.cache_position == drained.valid_history_tokens == 36
    assert (drained.next_frame_id, drained.state_version, drained.sequence_position) == (1, 1, 1)
    assert rig.native is not None
    assert rig.native.manifests[-1].rows[0].frame_id == 1
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_empty_assistant_queue_drain_releases_reserved_tail() -> None:
    rig = _rig()
    _set_fixture_capacity(rig, 40)
    state = _state(rig.coordinator.runtime_signature, "unified-reserve-empty")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=2,
    ).result(timeout=3)

    drained = rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
    ).result(timeout=3)

    assert drained.cache_position == drained.valid_history_tokens == 34
    assert rig.worker._reserved_tail_tokens[state.session_id] == 0
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_text_append_and_reservation_reassert_wait_for_accepted_q0_watermark() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "unified-watermark")
    _register(rig, state)
    _prime_native_prefix(rig, state)
    rig.worker.begin_assistant_audio_queue(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        reserve_tail_tokens=4,
    ).result(timeout=3)

    delayed = _assistant_q0(
        state,
        queue_audio_for_next_turn=False,
    )
    object.__setattr__(delayed, "ready_at_ns", time.monotonic_ns() + 10_000_000_000)
    rig.worker.submit_frame(delayed).result(timeout=3)
    with pytest.raises(RuntimeError, match="tail reservation requires a completed listener frame watermark"):
        rig.worker.reserve_tail_tokens(
            state.session_id,
            expected_generation_id=state.generation_id,
            expected_cache_position=34,
            token_count=4,
        ).result(timeout=3)
    with pytest.raises(RuntimeError, match="text append requires completed listener frame watermark"):
        rig.worker.append_text_tokens(
            state.session_id,
            (501, 502),
            expected_generation_id=state.generation_id,
            expected_cache_position=34,
        ).result(timeout=3)

    with rig.coordinator._condition:
        pending = rig.coordinator._queues[state.session_id][0][1]
        object.__setattr__(pending, "ready_at_ns", time.monotonic_ns())
        rig.coordinator._condition.notify_all()
    with rig.worker._wake:
        rig.worker._wake.notify_all()
    assert rig.result_event.wait(4)
    snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
    assert snapshot.cache_position == snapshot.valid_history_tokens == 34
    assert rig.worker.reserve_tail_tokens(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        token_count=4,
    ).result(timeout=3) == 4
    rig.worker.append_text_tokens(
        state.session_id,
        (501, 502),
        expected_generation_id=state.generation_id,
        expected_cache_position=34,
        consume_reserved_tokens=2,
    ).result(timeout=3)
    rig.worker.drain_queued_audio(
        state.session_id,
        expected_generation_id=state.generation_id,
        expected_cache_position=36,
    ).result(timeout=3)
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_failed_pipeline_aborts_frontend_and_poison_commits_no_cursor() -> None:
    rig = _rig(fail_step=RuntimeError("synthetic native batch failure"))
    states = tuple(
        _state(rig.coordinator.runtime_signature, f"faulted-{index}")
        for index in range(2)
    )
    _register(rig, *states)
    ready_at_ns = time.monotonic_ns() + 25_000_000
    accepted = [
        rig.worker.submit_frame(_frame(state, ready_at_ns=ready_at_ns))
        for state in states
    ]
    for future in accepted:
        future.result(timeout=3)

    assert rig.error_event.wait(4)
    for state in states:
        snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
        assert snapshot.poisoned_reason is not None
        assert "synthetic native batch failure" in snapshot.poisoned_reason
        assert snapshot.next_frame_id == 0
        assert snapshot.state_version == 0
        assert snapshot.cache_position == 34
        assert snapshot.user_pcm_tail is None
        assert snapshot.user_mel_history == ()
    assert rig.model.calls == 1
    assert rig.native.steps == []
    assert len(rig.errors) == 1

    report = rig.worker.stop(drain=True).result(timeout=3)
    assert not report.stopped_cleanly
    assert report.fatal_error is not None


def test_native_nonfinite_guard_quarantines_all_rows_before_policy_or_cursor_commit() -> None:
    rig = _rig(nonfinite_health=True)
    state = _state(rig.coordinator.runtime_signature, "nonfinite")
    _register(rig, state)
    with pytest.raises(RuntimeError, match="all rows quarantined"):
        rig.worker.submit_tick((_frame(state),)).result(timeout=4)
    assert rig.error_event.wait(3)
    assert rig.native is not None
    assert rig.native.health_generations == [1]
    assert rig.adapter is not None
    assert rig.adapter.hidden_rows == []
    snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
    assert snapshot.next_frame_id == 0
    assert snapshot.state_version == 0
    assert snapshot.poisoned_reason is not None
    assert "finite guard" in snapshot.poisoned_reason
    report = rig.worker.stop(drain=False).result(timeout=3)
    assert not report.stopped_cleanly


def test_result_notification_failure_is_reported_after_committed_batch() -> None:
    rig = _rig()
    notified = threading.Event()

    def fail_notification(_batch, _decisions) -> None:
        notified.set()
        raise RuntimeError("synthetic event-loop dispatch failure")

    rig.worker._on_result = fail_notification
    state = _state(rig.coordinator.runtime_signature, "dispatch-failure")
    _register(rig, state)
    ready_at_ns = time.monotonic_ns() + 50_000_000
    rig.worker.submit_frame(_frame(state, ready_at_ns=ready_at_ns)).result(timeout=3)
    rig.worker.submit_frame(
        _frame(state, frame_id=1, ready_at_ns=ready_at_ns)
    ).result(timeout=3)

    report = rig.worker.stop(drain=True).result(timeout=4)
    assert notified.is_set()
    assert state.next_frame_id == 1
    assert report.status == "failed"
    assert not report.stopped_cleanly
    assert report.pending_frames == 1
    assert report.processed_batches == 1
    assert "result dispatch failed after state commit" in (report.fatal_error or "")
    assert "synthetic event-loop dispatch failure" in (report.notification_error or "")


def test_startup_done_callback_cannot_block_on_queued_owner_command() -> None:
    runtime_factory_entered = threading.Event()
    runtime_factory_continue = threading.Event()
    callback_finished = threading.Event()
    callback_errors: list[BaseException] = []
    rig = _rig(
        auto_start=False,
        runtime_factory_entered=runtime_factory_entered,
        runtime_factory_continue=runtime_factory_continue,
    )
    startup = rig.worker.start()
    try:
        assert runtime_factory_entered.wait(3)
        state = _state(rig.coordinator.runtime_signature, "queued-before-startup")
        queued_command = rig.worker.register_session(state, prefix_import=rig.prefix_import)

        def wait_for_queued_command(_future: Future[Any]) -> None:
            assert _future.result() is None
            assert _future.exception() is None
            try:
                queued_command.result(timeout=1)
            except RuntimeError as exc:
                callback_errors.append(exc)
            try:
                queued_command.exception(timeout=1)
            except RuntimeError as exc:
                callback_errors.append(exc)
            try:
                rig.worker.session_state(state.session_id)
            except RuntimeError as exc:
                callback_errors.append(exc)
            try:
                rig.worker.stop(drain=True)
            except RuntimeError as exc:
                callback_errors.append(exc)
            finally:
                callback_finished.set()

        startup.add_done_callback(wait_for_queued_command)
        runtime_factory_continue.set()
        startup.result(timeout=3)
        assert callback_finished.wait(3)
        assert len(callback_errors) == 4
        assert "owner thread cannot wait on its own pending Future.result()" in str(
            callback_errors[0]
        )
        assert "owner thread cannot wait on its own pending Future.exception()" in str(
            callback_errors[1]
        )
        assert "owner thread cannot submit 'snapshot'" in str(callback_errors[2])
        assert "owner thread cannot submit 'stop'" in str(callback_errors[3])
        queued_command.result(timeout=3)
        assert rig.worker.session_state(state.session_id).result(timeout=3).next_frame_id == 0
        assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly
    finally:
        runtime_factory_continue.set()
        if not rig.worker._stopped.done():
            rig.worker.stop(drain=False)
            rig.worker._stopped.result(timeout=3)


def test_reset_retires_old_native_generation_and_rejects_stale_frame() -> None:
    rig = _rig()
    signature = rig.coordinator.runtime_signature
    old = _state(signature, "reset-me", generation=7)
    _register(rig, old)
    rig.worker.submit_frame(_frame(old)).result(timeout=3)
    assert rig.result_event.wait(4)

    new = _state(signature, "reset-me", generation=8)
    rig.worker.reset_session(old.session_id, new, prefix_import=rig.prefix_import).result(timeout=3)
    stale = ListenerFrame(
        session_id=old.session_id,
        generation_id=old.generation_id,
        state_epoch_id=old.state_epoch_id,
        frame_id=1,
        ready_at_ns=time.monotonic_ns(),
        pcm=np.zeros(1280, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="generation or epoch"):
        rig.worker.submit_frame(stale).result(timeout=3)

    rig.worker.submit_frame(_frame(new)).result(timeout=3)
    deadline = time.monotonic() + 4
    while len(rig.results) < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(rig.results) == 2
    assert len(rig.native.imports) == 2
    assert rig.native.releases == [(old.session_id, 1, 36)]
    current = rig.worker.session_state(new.session_id).result(timeout=3)
    assert current.generation_id == 8
    assert current.next_frame_id == 1
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_semantic_prefix_refresh_rebases_only_kv_cursor_and_imports_before_next_row() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "semantic-refresh")
    _register(rig, state)
    first = rig.worker.submit_tick((_frame(state),)).result(timeout=4)[state.session_id]
    assert first.frame_id == 0
    before = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (before.next_frame_id, before.state_version, before.sequence_position) == (1, 1, 1)
    assert before.cache_position == 36

    imported_rows = []

    def revised_prefix(_state, row):
        imported_rows.append(row)
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 50), dtype=torch.int64),
            prefix_len=50,
            cache_position=50,
            semantic_context_revision=9,
            prefix_build_ms=2.0,
            replay_ms=3.0,
        )

    rig.worker.refresh_session_prefix(
        state.session_id,
        expected_frame_id=1,
        expected_state_version=1,
        cache_position=50,
        prefix_import=revised_prefix,
    ).result(timeout=3)
    rebased = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (rebased.next_frame_id, rebased.state_version, rebased.sequence_position) == (1, 1, 1)
    assert rebased.generation_id == state.generation_id
    assert rebased.state_epoch_id == state.state_epoch_id
    assert rebased.cache_position == rebased.valid_history_tokens == 50

    second = rig.worker.submit_tick((_frame(state, frame_id=1, value=0.75),)).result(timeout=4)
    row_output = second[state.session_id]
    assert row_output.frame_id == 1
    assert row_output.semantic_context_revision == 9
    assert row_output.semantic_prefix_len == 50
    assert row_output.semantic_prefix_build_ms == 2.0
    assert row_output.semantic_prefix_replay_ms == 3.0
    assert row_output.semantic_prefix_import_ms is not None
    assert row_output.semantic_prefix_refresh_event is True
    assert row_output.semantic_prefix_refresh_event_id
    assert row_output.semantic_prefix_refresh_revision == 9
    assert imported_rows[0].frame_id == 1
    assert imported_rows[0].cache_position == 50
    assert rig.native is not None
    assert rig.native.releases == [(state.session_id, 1, 36)]
    assert len(rig.native.imports) == 2
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_initial_semantic_prefix_import_waits_for_prepare_and_keeps_frame_zero_unadvanced() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "initial-semantic")
    _register(rig, state)
    event = _FakeCompletionEvent()
    prepared = PreparedInitialPrefix(
        prefix=PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 47), dtype=torch.int64),
            prefix_len=47,
            cache_position=47,
            semantic_context_revision=12,
            prefix_build_ms=81.5,
            replay_ms=0.0,
        ),
        ready_event=event,
        session_id=state.session_id,
        listener_generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        semantic_context_revision=12,
        expected_cache_position=34,
    )

    admission = rig.worker.prime_initial_prefix(state.session_id, prepared).result(timeout=3)

    assert admission.status == "primed"
    assert admission.prefix_len == 47
    assert admission.semantic_context_revision == 12
    assert len(rig.native.imports) == 1
    assert rig.native.imports[0][2] == 0  # import identity, not an input frame
    assert rig.native.steps == []
    assert event in rig.native.listener_stream.waited_events
    snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (snapshot.next_frame_id, snapshot.state_version, snapshot.sequence_position) == (0, 0, 0)
    assert snapshot.cache_position == snapshot.valid_history_tokens == 47

    output = rig.worker.submit_tick((_frame(state),)).result(timeout=4)[state.session_id]
    assert output.frame_id == 0
    assert output.semantic_context_revision == 12
    assert len(rig.native.imports) == 1
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_initial_semantic_prefix_replacement_retires_only_frame_zero_native_slot() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "initial-semantic-replace")
    _register(rig, state)

    def prepared(length: int, revision: int, expected_cache_position: int) -> PreparedInitialPrefix:
        return PreparedInitialPrefix(
            prefix=PrefixCacheImport(
                past_key_values=object(),
                prefix_attention=torch.ones((1, length), dtype=torch.int64),
                prefix_len=length,
                cache_position=length,
                semantic_context_revision=revision,
            ),
            ready_event=_FakeCompletionEvent(),
            session_id=state.session_id,
            listener_generation_id=state.generation_id,
            state_epoch_id=state.state_epoch_id,
            semantic_context_revision=revision,
            expected_cache_position=expected_cache_position,
        )

    first = rig.worker.prime_initial_prefix(
        state.session_id, prepared(41, 3, 34)
    ).result(timeout=3)
    second = rig.worker.prime_initial_prefix(
        state.session_id, prepared(53, 4, 41)
    ).result(timeout=3)

    assert first.status == second.status == "primed"
    assert rig.native.releases == [(state.session_id, 0, 41)]
    assert [item[2] for item in rig.native.imports] == [0, 0]
    snapshot = rig.worker.session_state(state.session_id).result(timeout=3)
    assert (snapshot.next_frame_id, snapshot.state_version, snapshot.sequence_position) == (0, 0, 0)
    assert snapshot.cache_position == 53
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_initial_semantic_prefix_cas_rejects_a_frame_that_arrived_during_preparation() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "initial-semantic-stale")
    _register(rig, state)
    state.next_frame_id = 1
    prepared = PreparedInitialPrefix(
        prefix=PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 47), dtype=torch.int64),
            prefix_len=47,
            cache_position=47,
            semantic_context_revision=12,
        ),
        ready_event=_FakeCompletionEvent(),
        session_id=state.session_id,
        listener_generation_id=state.generation_id,
        state_epoch_id=state.state_epoch_id,
        semantic_context_revision=12,
        expected_cache_position=34,
    )

    admission = rig.worker.prime_initial_prefix(state.session_id, prepared).result(timeout=3)

    assert admission.status == "ineligible"
    assert admission.reason == "listener_state_changed_before_initial_prefix_import"
    assert rig.native.imports == []
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_native_tick_progresses_while_initial_prefix_preparation_holds_fair_model_lock(monkeypatch) -> None:
    from duplex_kernel.listener_microbatch_serving import (
        prepare_initial_semantic_prefix_async,
    )

    model_lock = FairModelLock()
    preparation_started = threading.Event()
    finish_preparation = threading.Event()

    def blocked_preparation(*_args, **_kwargs):
        assert model_lock.acquire_fair("listener")
        preparation_started.set()
        try:
            assert finish_preparation.wait(4)
            return "prepared"
        finally:
            model_lock.release()

    monkeypatch.setattr(
        "duplex_kernel.listener_microbatch_serving.prepare_initial_semantic_prefix",
        blocked_preparation,
    )
    rig = _rig(model_lock=model_lock)
    state = _state(rig.coordinator.runtime_signature, "tick-during-initial-prefix")
    _register(rig, state)

    async def exercise() -> None:
        preparation = asyncio.create_task(prepare_initial_semantic_prefix_async())
        assert await asyncio.to_thread(preparation_started.wait, 2)
        try:
            outputs = await asyncio.wait_for(
                asyncio.wrap_future(rig.worker.submit_tick((_frame(state),))),
                timeout=3,
            )
        finally:
            finish_preparation.set()
        assert outputs[state.session_id].frame_id == 0
        assert await preparation == "prepared"

    asyncio.run(exercise())
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_prefix_phase_timings_are_batch_local_for_mixed_rows_and_refresh_provenance_persists() -> None:
    rig = _rig()
    state_a = _state(rig.coordinator.runtime_signature, "prefix-steady")
    state_b = _state(rig.coordinator.runtime_signature, "prefix-refresh")

    def initial_prefix(_state, _row):
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 34), dtype=torch.int64),
            prefix_len=34,
            semantic_context_revision=4,
            prefix_build_ms=7.0,
            replay_ms=11.0,
        )

    rig.worker.register_session(state_a, prefix_import=rig.prefix_import).result(timeout=3)
    rig.worker.register_session(state_b, prefix_import=initial_prefix).result(timeout=3)
    rig.worker.submit_tick((_frame(state_a),)).result(timeout=4)
    initial_b = rig.worker.submit_tick((_frame(state_b),)).result(timeout=4)[state_b.session_id]
    assert initial_b.semantic_prefix_build_ms == 7.0
    assert initial_b.semantic_prefix_replay_ms == 11.0
    assert initial_b.semantic_prefix_import_ms is not None
    assert initial_b.semantic_prefix_refresh_event is False

    def revised_prefix(_state, _row):
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, 50), dtype=torch.int64),
            prefix_len=50,
            cache_position=50,
            semantic_context_revision=9,
            prefix_build_ms=2.0,
            replay_ms=3.0,
        )

    rig.worker.refresh_session_prefix(
        state_b.session_id,
        expected_frame_id=1,
        expected_state_version=1,
        cache_position=50,
        prefix_import=revised_prefix,
    ).result(timeout=3)
    mixed = rig.worker.submit_tick((
        _frame(state_a, frame_id=1),
        _frame(state_b, frame_id=1, value=0.75),
    )).result(timeout=4)

    steady_row = mixed[state_a.session_id]
    refreshed_row = mixed[state_b.session_id]
    assert steady_row.semantic_prefix_build_ms is None
    assert steady_row.semantic_prefix_replay_ms is None
    assert steady_row.semantic_prefix_import_ms is None
    assert steady_row.semantic_prefix_refresh_event is False
    assert steady_row.semantic_prefix_refresh_event_id is None
    assert refreshed_row.semantic_prefix_build_ms == 2.0
    assert refreshed_row.semantic_prefix_replay_ms == 3.0
    assert refreshed_row.semantic_prefix_import_ms is not None
    assert refreshed_row.semantic_prefix_refresh_event is True
    assert refreshed_row.semantic_prefix_refresh_event_id
    assert refreshed_row.semantic_prefix_refresh_revision == 9

    later = rig.worker.submit_tick((
        _frame(state_a, frame_id=2),
        _frame(state_b, frame_id=2, value=0.5),
    )).result(timeout=4)
    for output in later.values():
        assert output.semantic_prefix_build_ms is None
        assert output.semantic_prefix_replay_ms is None
        assert output.semantic_prefix_import_ms is None
        assert output.semantic_prefix_refresh_event is False
        assert output.semantic_prefix_refresh_event_id is None
    assert later[state_b.session_id].semantic_context_revision == 9
    assert later[state_b.session_id].semantic_prefix_len == 50
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_unified_instruction_prefix_refresh_rebases_pristine_frame_zero_before_audio() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "unified-prefix-refresh")
    _register(rig, state)
    refreshed_ids = (101, 102, 103, 104, 105, 106, 107)
    imported_rows: list[BatchRowDescriptor] = []
    importer_threads: list[int] = []

    def refreshed_prefix(_state, row):
        imported_rows.append(row)
        importer_threads.append(threading.get_ident())
        return PrefixCacheImport(
            past_key_values=object(),
            prefix_attention=torch.ones((1, len(refreshed_ids)), dtype=torch.int64),
            prefix_len=len(refreshed_ids),
            cache_position=len(refreshed_ids),
        )

    rig.worker.refresh_unified_prefix_before_audio(
        state.session_id,
        refreshed_ids,
        new_revision=1,
        expected_generation_id=state.generation_id,
        expected_cache_position=state.cache_position,
        prefix_import=refreshed_prefix,
    ).result(timeout=3)

    output = rig.worker.submit_tick((_frame(state),)).result(timeout=4)[state.session_id]

    assert imported_rows and imported_rows[0].frame_id == 0
    assert imported_rows[0].cache_position == len(refreshed_ids)
    assert imported_rows[0].valid_history_tokens == len(refreshed_ids)
    assert importer_threads == rig.runtime_factory_threads
    assert rig.native is not None
    assert rig.native.import_rows[0].cache_position == len(refreshed_ids)
    assert output.frame_id == 0
    assert output.cache_position == len(refreshed_ids) + 2
    assert output.valid_history_tokens == len(refreshed_ids) + 2
    assert output.semantic_prefix_len == len(refreshed_ids)
    assert output.semantic_context_revision == 1
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_unified_instruction_prefix_refresh_rejects_nonpristine_cursor_without_mutation() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "unified-prefix-stale")
    initial_cache_position = state.cache_position
    _register(rig, state)
    first = rig.worker.submit_tick((_frame(state),)).result(timeout=4)[state.session_id]
    assert first.frame_id == 0
    assert first.cache_position == initial_cache_position + 2
    assert rig.native is not None
    old_importer_count = len(rig.native.import_rows)
    old_prefix_position = rig.native.import_rows[0].cache_position

    failure = rig.worker.refresh_unified_prefix_before_audio(
        state.session_id,
        (201, 202, 203, 204, 205),
        new_revision=1,
        expected_generation_id=state.generation_id,
        expected_cache_position=first.cache_position,
        prefix_import=lambda _state, _row: pytest.fail("stale refresh must not replace the importer"),
    )
    with pytest.raises(RuntimeError, match="pristine frame-zero"):
        failure.result(timeout=3)

    second = rig.worker.submit_tick((_frame(state, frame_id=1),)).result(timeout=4)[state.session_id]
    assert second.frame_id == 1
    assert second.cache_position == first.cache_position + 2
    assert len(rig.native.import_rows) == old_importer_count
    assert rig.native.import_rows[0].cache_position == old_prefix_position == initial_cache_position
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_close_command_waits_for_inflight_completion_event_before_release() -> None:
    completion_started = threading.Event()
    completion_continue = threading.Event()

    def completion_barrier(_stream) -> None:
        completion_started.set()
        assert completion_continue.wait(3)

    rig = _rig(completion_barrier=completion_barrier)
    state = _state(rig.coordinator.runtime_signature, "close-me")
    _register(rig, state)
    rig.worker.submit_frame(_frame(state)).result(timeout=3)
    assert completion_started.wait(4)

    close_future: Future[Any] = rig.worker.close_session(state.session_id)
    assert not close_future.done()
    assert rig.native.releases == []
    completion_continue.set()
    assert rig.result_event.wait(4)
    retired = close_future.result(timeout=4)
    assert retired.state.session_id == state.session_id
    assert rig.native.releases == [(state.session_id, 1, 36)]
    assert rig.worker.stop(drain=True).result(timeout=3).stopped_cleanly


def test_bounded_stop_reports_accepted_but_not_yet_ready_frames() -> None:
    rig = _rig()
    state = _state(rig.coordinator.runtime_signature, "future-frame")
    _register(rig, state)
    rig.worker.submit_frame(
        _frame(state, ready_at_ns=time.monotonic_ns() + 2_000_000_000)
    ).result(timeout=3)

    report = rig.worker.stop(drain=True, drain_timeout_s=0.02).result(timeout=3)
    assert report.status == "drain_timeout"
    assert not report.stopped_cleanly
    assert not report.drained
    assert report.pending_frames == 1
