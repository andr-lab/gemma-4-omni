from __future__ import annotations

import threading
import time
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from duplex_kernel.gemma_native_control6_batch_policy import (
    GemmaNativeControl6BatchPolicy,
    ListenerHiddenProvenance,
    BatchPolicyInput,
    NativeControl6RawScoreSnapshot,
)
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator
from duplex_kernel.microbatch_contract import (
    BatchRowDescriptor,
    BatchOutputSlot,
    ListenerBatchManifest,
    ListenerFrame,
    ListenerSessionState,
    NativeBatchCapability,
    OutputSlotBusy,
    RowStateUpdate,
    RuntimeSignature,
)
from duplex_kernel.native_listener_microbatch import NativeBatchOutput
from duplex_kernel.turn_policy import ControlAction, PolicyInput
from training.duplex_qlora.native_control6 import ACTIONS, HIDDEN_WIDTH, NativeControl6LoRA


@pytest.fixture
def runtime_signature() -> RuntimeSignature:
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
        stream_q_len=4,
    )


@pytest.fixture
def capability() -> NativeBatchCapability:
    return NativeBatchCapability(
        api_version=2,
        native_abi_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=4,
        supports_arbitrary_batch_size=True,
        stream_priority=-1,
        stream_identity="listener-cuda-stream-17",
    )


@pytest.mark.parametrize("priority", [-1, 0])
def test_listener_priority_contract_accepts_primary_and_shadow_owners(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    priority: int,
) -> None:
    row = BatchRowDescriptor(
        session_id="priority-row", session_generation=1, state_epoch_id="epoch",
        frame_id=0, listener_state_version=0, sequence_position=0,
        cache_position=34, valid_history_tokens=34, native_context_id=11,
        input_frame_sha256="e" * 64, reference_pcm_sha256=None,
        frame_ready_at_ns=10, reference_frame_id=None,
    )
    attested = replace(capability, stream_priority=priority)
    attested.validate_for(runtime_signature)
    manifest = ListenerBatchManifest(
        manifest_id=f"priority-{priority}", runtime_signature=runtime_signature,
        stream_identity=attested.stream_identity, stream_priority=priority,
        dispatch_at_ns=10, max_coalesce_wait_ns=0, dispatch_reason="multiple_ready",
        rows=(row,),
    )
    assert manifest.stream_priority == attested.stream_priority

    with pytest.raises(ValueError, match="priority must be -1 or 0"):
        replace(capability, stream_priority=-5)
    with pytest.raises(ValueError, match="stream_priority must be an integer"):
        replace(capability, stream_priority=True)
    with pytest.raises(ValueError, match="priority must be -1 or 0"):
        replace(manifest, stream_priority=False)


def test_production_coordinator_rejects_priority_zero_shadow_capability(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    with pytest.raises(ValueError, match="production listener dispatcher requires"):
        ListenerMicrobatchCoordinator(
            runtime_signature,
            replace(capability, stream_priority=0),
        )


def _state(
    runtime_signature: RuntimeSignature,
    session_id: str,
    *,
    next_frame_id: int = 0,
    cache_position: int = 34,
) -> ListenerSessionState:
    return ListenerSessionState(
        session_id=session_id,
        generation_id=7,
        state_epoch_id=f"epoch-{session_id}",
        runtime_signature=runtime_signature,
        next_frame_id=next_frame_id,
        state_version=next_frame_id,
        sequence_position=next_frame_id,
        cache_position=cache_position,
        valid_history_tokens=cache_position,
        native_context_id=10_000 + sum(map(ord, session_id)),
    )


def _frame(
    session_id: str,
    frame_id: int,
    ready_at_ns: int,
    *,
    generation_id: int = 7,
    epoch: str | None = None,
    value: float | None = None,
) -> ListenerFrame:
    return ListenerFrame(
        session_id=session_id,
        generation_id=generation_id,
        state_epoch_id=epoch or f"epoch-{session_id}",
        frame_id=frame_id,
        ready_at_ns=ready_at_ns,
        pcm=np.full(1280, frame_id if value is None else value, dtype=np.float32),
    )


def _coordinator(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    session_count: int = 4,
) -> ListenerMicrobatchCoordinator:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    for index in range(session_count):
        session_id = f"s{index}"
        coordinator.register_session(_state(runtime_signature, session_id))
    return coordinator


def test_frame_and_causal_reference_are_immutable_hashed_snapshots() -> None:
    current = np.arange(1280, dtype=np.float32)
    reference = np.full(1280, 0.25, dtype=np.float32)
    frame = ListenerFrame(
        session_id="s0",
        generation_id=7,
        state_epoch_id="epoch-s0",
        frame_id=9,
        ready_at_ns=1_000,
        pcm=current,
        reference_pcm=reference,
        reference_frame_id=8,
    )
    input_hash = frame.pcm_sha256
    reference_hash = frame.reference_pcm_sha256

    current[0] = -100.0
    reference[0] = -100.0

    assert frame.pcm[0] == 0.0
    assert frame.reference_pcm is not None and frame.reference_pcm[0] == 0.25
    assert frame.pcm_sha256 == input_hash
    assert frame.reference_pcm_sha256 == reference_hash
    with pytest.raises(ValueError):
        frame.pcm[0] = 4.0
    with pytest.raises(ValueError):
        frame.reference_pcm[0] = 4.0  # type: ignore[index]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 8, 12])
def test_phase_aligned_ready_frames_make_one_real_batch_call(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    batch_size: int,
) -> None:
    coordinator = _coordinator(
        runtime_signature, capability, session_count=batch_size
    )
    calls: list[tuple[str, ...]] = []
    for index in range(batch_size):
        coordinator.enqueue(_frame(f"s{index}", 0, 1_000_000))

    class OneCallProcessor:
        def process_batch(self, batch: object, states: tuple[ListenerSessionState, ...]):
            calls.append(tuple(state.session_id for state in states))
            return tuple(
                RowStateUpdate(
                    session_id=row.session_id,
                    frame_id=row.frame_id,
                    cache_position=row.cache_position + 4,
                    valid_history_tokens=row.valid_history_tokens + 4,
                )
                for row in batch.manifest.rows  # type: ignore[attr-defined]
            )

    now = 1_000_000 + (1 if batch_size == 1 else 0)
    if batch_size == 1:
        assert coordinator.take_ready(now) is None
        now += 2_000_000
    result = coordinator.run_one(now, OneCallProcessor())

    assert result is not None
    assert result.manifest.batch_size == batch_size
    assert result.manifest.padded_rows == 0
    assert result.pcm_batch().shape == (batch_size, 1280)
    assert len(calls) == 1
    assert len(calls[0]) == batch_size
    assert all(coordinator.session_state(sid).state_version == 1 for sid in calls[0])


def test_one_ready_frame_dispatches_at_two_ms_without_waiting_for_missing_sessions(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability)
    coordinator.enqueue(_frame("s0", 0, 100))

    assert coordinator.take_ready(2_000_099) is None
    batch = coordinator.take_ready(2_000_100)

    assert batch is not None
    assert batch.manifest.batch_size == 1
    assert batch.manifest.dispatch_reason == "coalesce_deadline"
    assert batch.manifest.max_coalesce_wait_ns == 2_000_000
    assert batch.frames[0].pcm.shape == (1280,)


def test_future_ready_timestamp_never_enters_a_batch_early(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(
        runtime_signature, capability, coalesce_window_ns=0
    )
    coordinator.register_session(_state(runtime_signature, "s0"))
    coordinator.register_session(_state(runtime_signature, "s1"))
    coordinator.enqueue(_frame("s0", 0, 100))
    coordinator.enqueue(_frame("s1", 0, 200))

    assert coordinator.take_ready(99) is None
    assert coordinator.next_wakeup_ns(99) == 100
    first = coordinator.take_ready(100)
    assert first is not None and [row.session_id for row in first.manifest.rows] == ["s0"]
    coordinator.complete(first, _updates(first))
    assert coordinator.take_ready(199) is None
    second = coordinator.take_ready(200)
    assert second is not None and [row.session_id for row in second.manifest.rows] == ["s1"]


def test_worker_waits_for_future_ready_time_instead_of_spinning_on_two_queued_rows(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    ready_at = time.monotonic_ns() + 1_000_000_000
    coordinator.enqueue(_frame("s0", 0, ready_at))
    coordinator.enqueue(_frame("s1", 0, ready_at))
    stop = threading.Event()
    observed_timeouts: list[float | None] = []

    def observe_wait(timeout: float | None = None) -> bool:
        observed_timeouts.append(timeout)
        stop.set()
        return False

    monkeypatch.setattr(coordinator._condition, "wait", observe_wait)

    assert coordinator.wait_and_take(stop) is None
    assert len(observed_timeouts) == 1
    assert observed_timeouts[0] is not None and observed_timeouts[0] > 0.0


def test_phase_skew_emits_real_subgroups_without_padding_or_idle_rows(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability)
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 500_000))
    first = coordinator.take_ready(1_000_000)
    assert first is not None and first.manifest.batch_size == 2
    coordinator.complete(first, _updates(first))

    coordinator.enqueue(_frame("s2", 0, 9_000_000))
    coordinator.enqueue(_frame("s3", 0, 10_000_000))
    second = coordinator.take_ready(10_500_000)

    assert second is not None and second.manifest.batch_size == 2
    assert {row.session_id for row in second.manifest.rows} == {"s2", "s3"}
    assert second.manifest.padded_rows == 0


def _updates(batch: object) -> tuple[RowStateUpdate, ...]:
    return tuple(
        RowStateUpdate(
            session_id=row.session_id,
            frame_id=row.frame_id,
            cache_position=row.cache_position + 4,
            valid_history_tokens=row.valid_history_tokens + 4,
        )
        for row in batch.manifest.rows  # type: ignore[attr-defined]
    )


def test_one_session_is_serialized_and_next_manifest_uses_committed_cache_position(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s0", 1, 100_000))
    coordinator.enqueue(_frame("s1", 0, 100_000))

    first = coordinator.take_ready(200_000)
    assert first is not None
    assert [(row.session_id, row.frame_id) for row in first.manifest.rows] == [("s0", 0), ("s1", 0)]
    coordinator.complete(first, _updates(first))

    assert coordinator.take_ready(2_099_999) is None
    second = coordinator.take_ready(2_100_000)
    assert second is not None
    assert [(row.session_id, row.frame_id) for row in second.manifest.rows] == [("s0", 1)]
    row = second.manifest.rows[0]
    assert row.listener_state_version == 1
    assert row.sequence_position == 1
    assert row.cache_position == 38


def test_stale_frame_wrong_epoch_and_out_of_order_frame_are_rejected(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=1)

    with pytest.raises(ValueError, match="next expected frame"):
        coordinator.enqueue(_frame("s0", 1, 0))
    with pytest.raises(ValueError, match="generation or epoch"):
        coordinator.enqueue(_frame("s0", 0, 0, epoch="wrong"))
    coordinator.enqueue(_frame("s0", 0, 0))
    with pytest.raises(ValueError, match="next expected frame"):
        coordinator.enqueue(_frame("s0", 0, 1))


def test_runtime_signature_mismatch_and_missing_batch_capability_fail_closed(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    with pytest.raises(ValueError, match="capability is missing"):
        ListenerMicrobatchCoordinator(
            runtime_signature,
            replace(capability, api_version=0),
        )

    coordinator = _coordinator(runtime_signature, capability, session_count=1)
    other_signature = replace(runtime_signature, checkpoint_sha256="e" * 64)
    with pytest.raises(ValueError, match="runtime signature"):
        coordinator.register_session(_state(other_signature, "other"))

    with pytest.raises(ValueError, match="native ABI"):
        ListenerMicrobatchCoordinator(
            replace(runtime_signature, native_abi_version=4), capability
        )
    with pytest.raises(ValueError, match="backend"):
        ListenerMicrobatchCoordinator(
            replace(runtime_signature, attention_backend_id=4), capability
        )
    with pytest.raises(ValueError, match="RMSNorm backend"):
        replace(runtime_signature, rmsnorm_backend_id=0)
    with pytest.raises(ValueError, match="max_seq_len"):
        replace(runtime_signature, max_seq_len=4_096)
    with pytest.raises(ValueError, match="stream_q_len"):
        replace(capability, stream_q_len=2)


def test_shared_mutable_listener_state_is_rejected_between_sessions(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    first = _state(runtime_signature, "s0")
    coordinator.register_session(first)
    second = _state(runtime_signature, "s1")
    second.user_mel_history = first.user_mel_history

    with pytest.raises(ValueError, match="private state alias"):
        coordinator.register_session(second)


def test_private_storage_aliases_inside_distinct_wrappers_are_rejected(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    first = _state(runtime_signature, "s0")
    shared = np.arange(256, dtype=np.float32)
    first.user_mel_history = [[shared[:64]]]
    coordinator.register_session(first)

    second = _state(runtime_signature, "s1")
    second.assistant_mel_history = [[shared[192:256]]]
    with pytest.raises(ValueError, match="private state alias"):
        coordinator.register_session(second)


def test_torch_storage_aliases_nested_in_cache_are_rejected(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    first = _state(runtime_signature, "s0")
    shared = torch.arange(32, dtype=torch.float32)
    first.native_kv_cache = {"layers": [(shared[:16],)]}
    coordinator.register_session(first)

    second = _state(runtime_signature, "s1")
    second.attention_mask = shared[8:24]
    with pytest.raises(ValueError, match="private state alias"):
        coordinator.register_session(second)


def test_unverifiable_tensor_storage_rejects_registration_fail_closed(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    class BrokenTensorMetadata:
        device = "cuda:0"

        def numel(self) -> int:
            return 1

        def untyped_storage(self) -> object:
            raise RuntimeError("storage unavailable")

    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    state = _state(runtime_signature, "s0")
    state.attention_mask = BrokenTensorMetadata()
    with pytest.raises(ValueError, match="cannot verify private tensor backing storage"):
        coordinator.register_session(state)


def test_session_state_returns_deep_readonly_observer_snapshot(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    state = _state(runtime_signature, "s0")
    state.user_pcm_tail = np.arange(160, dtype=np.float32)
    mel = np.arange(12, dtype=np.float32).reshape(3, 4)
    state.user_mel_history = [mel]
    state.attention_mask = torch.ones((1, 8), dtype=torch.bool)
    coordinator.register_session(state)

    snapshot = coordinator.session_state("s0")
    assert snapshot is not state
    assert snapshot.user_pcm_tail is not None
    with pytest.raises(ValueError):
        snapshot.user_pcm_tail.setflags(write=True)
    with pytest.raises((AttributeError, TypeError)):
        snapshot.state_version = 99  # type: ignore[misc]
    state.state_version = 2
    mel[0, 0] = 999
    assert snapshot.state_version == 0
    assert snapshot.user_mel_history[0][0, 0] == 0
    assert snapshot.attention_mask.type_name.endswith("Tensor")


def test_only_one_worker_thread_can_own_batch_state(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=1)
    coordinator.enqueue(_frame("s0", 0, 0))
    batch = coordinator.take_ready(2_000_000)
    assert batch is not None
    errors: list[BaseException] = []

    def finish_from_other_thread() -> None:
        try:
            coordinator.complete(batch, _updates(batch))
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=finish_from_other_thread)
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    coordinator.complete(batch, _updates(batch))


def test_session_state_rejects_owner_reads_during_batch_and_foreign_observers(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=1)
    coordinator.enqueue(_frame("s0", 0, 0))
    batch = coordinator.take_ready(2_000_000)
    assert batch is not None

    with pytest.raises(RuntimeError, match="batch is active"):
        coordinator.session_state("s0")

    observer_started = threading.Event()
    observer_finished = threading.Event()
    errors: list[BaseException] = []

    def observe_from_foreign_thread() -> None:
        observer_started.set()
        try:
            coordinator.session_state("s0")
        except BaseException as exc:
            errors.append(exc)
        finally:
            observer_finished.set()

    observer = threading.Thread(target=observe_from_foreign_thread)
    observer.start()
    assert observer_started.wait(timeout=2.0)
    assert observer_finished.wait(timeout=2.0)
    observer.join(timeout=2.0)
    assert not observer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "owned by another worker thread" in str(errors[0])

    coordinator.complete(batch, _updates(batch))
    assert coordinator.session_state("s0").state_version == 1

    errors.clear()
    observer = threading.Thread(target=observe_from_foreign_thread)
    observer.start()
    observer.join(timeout=2.0)
    assert not observer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_native_context_handles_must_exist_and_be_unique_per_session(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=1)
    missing = _state(runtime_signature, "missing")
    missing.native_context_id = None
    with pytest.raises(ValueError, match="native context before"):
        coordinator.register_session(missing)

    reused = _state(
        runtime_signature,
        "reused",
        next_frame_id=0,
    )
    reused.native_context_id = coordinator.session_state("s0").native_context_id
    with pytest.raises(ValueError, match="native context handle is shared"):
        coordinator.register_session(reused)


def test_session_close_retires_private_context_without_disposing_it_in_coordinator(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    native_context = object()
    state = _state(runtime_signature, "s0")
    state.native_context = native_context
    coordinator.register_session(state)
    coordinator.enqueue(_frame("s0", 0, 0))

    retired = coordinator.close_session("s0")

    assert retired.state is state
    assert retired.state.native_context is native_context
    assert retired.queued_frame_count == 1
    assert retired.state.closed is True
    assert coordinator.ready_depth == 0


def test_malformed_multirow_completion_poison_is_atomic_and_cannot_partially_advance(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    before = {
        session_id: (coordinator.session_state(session_id).next_frame_id,
                     coordinator.session_state(session_id).state_version)
        for session_id in ("s0", "s1")
    }
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 0))
    batch = coordinator.take_ready(1)
    assert batch is not None

    with pytest.raises(ValueError, match="exactly one update"):
        coordinator.complete(batch, _updates(batch)[:1])

    after = {
        session_id: (coordinator.session_state(session_id).next_frame_id,
                     coordinator.session_state(session_id).state_version)
        for session_id in ("s0", "s1")
    }
    assert after == before
    assert coordinator.active_manifest_id is None
    assert all(coordinator.session_state(sid).poisoned_reason for sid in ("s0", "s1"))


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda update: replace(update, cache_position=update.cache_position + 4), "exactly Q=4"),
        (lambda update: replace(update, valid_history_tokens=update.cache_position - 1), "valid_history_tokens"),
        (lambda update: replace(update, cache_position=16_385, valid_history_tokens=16_385), "max_seq_len=16384"),
    ],
)
def test_multirow_completion_prevalidates_q4_and_history_before_any_commit(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    mutate: object,
    message: str,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    before = {
        sid: (coordinator.session_state(sid).cache_position,
              coordinator.session_state(sid).next_frame_id)
        for sid in ("s0", "s1")
    }
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 0))
    batch = coordinator.take_ready(1)
    assert batch is not None
    valid_updates = list(_updates(batch))
    valid_updates[1] = mutate(valid_updates[1])  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        coordinator.complete(batch, valid_updates)

    after = {
        sid: (coordinator.session_state(sid).cache_position,
              coordinator.session_state(sid).next_frame_id)
        for sid in ("s0", "s1")
    }
    assert after == before
    assert all(coordinator.session_state(sid).poisoned_reason for sid in ("s0", "s1"))


def test_completion_staged_state_hook_runs_after_prevalidation_before_cursor_commit(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 0))
    batch = coordinator.take_ready(1)
    assert batch is not None

    observed: list[tuple[int, int]] = []

    def commit_staged_state() -> None:
        observed.extend(
            (coordinator._states[sid].next_frame_id, coordinator._states[sid].cache_position)
            for sid in ("s0", "s1")
        )

    coordinator.complete(batch, _updates(batch), before_cursor_commit=commit_staged_state)

    assert observed == [(0, 34), (0, 34)]
    assert [coordinator.session_state(sid).cache_position for sid in ("s0", "s1")] == [38, 38]


def test_staged_state_commit_failure_poisons_all_rows_without_cursor_advance(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 0))
    batch = coordinator.take_ready(1)
    assert batch is not None

    def fail_staged_state() -> None:
        raise RuntimeError("synthetic staged frontend commit failure")

    with pytest.raises(RuntimeError, match="synthetic staged frontend commit failure"):
        coordinator.complete(batch, _updates(batch), before_cursor_commit=fail_staged_state)

    assert coordinator.active_manifest_id is None
    assert all(coordinator._states[sid].next_frame_id == 0 for sid in ("s0", "s1"))
    assert all(coordinator._states[sid].cache_position == 34 for sid in ("s0", "s1"))
    assert all(coordinator._states[sid].poisoned_reason for sid in ("s0", "s1"))


def test_fail_batch_poisons_rows_and_never_requeues_for_implicit_fallback(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    coordinator.enqueue(_frame("s0", 0, 0))
    coordinator.enqueue(_frame("s1", 0, 0))
    batch = coordinator.take_ready(1)
    assert batch is not None

    coordinator.fail(batch, RuntimeError("synthetic batch failure"))

    assert coordinator.ready_depth == 0
    for session_id in ("s0", "s1"):
        with pytest.raises(RuntimeError, match="poisoned"):
            coordinator.enqueue(_frame(session_id, 1, 2))


def test_borrowed_output_slot_waits_for_producer_and_consumer_fences() -> None:
    class Fence:
        def __init__(self, done: bool = False) -> None:
            self.done = done

        def query(self) -> bool:
            return self.done

    slot = BatchOutputSlot()
    producer = Fence()
    first = slot.publish(("borrowed",), producer_fence=producer)
    consumer = Fence()
    retained = first.retain_consumer(
        lambda borrowed, producer_fence: (tuple(borrowed), producer_fence),
        completion_fence=consumer,
    )
    assert retained[0] == ("borrowed",)

    with pytest.raises(OutputSlotBusy):
        slot.publish(("next",), producer_fence=Fence(True))
    producer.done = True
    with pytest.raises(OutputSlotBusy):
        slot.publish(("next",), producer_fence=Fence(True))
    consumer.done = True
    second = slot.publish(("next",), producer_fence=Fence(True))
    assert second.generation == first.generation + 1


def test_borrowed_output_slot_retains_consumer_fence_if_enqueue_raises() -> None:
    class Fence:
        def __init__(self, done: bool = False) -> None:
            self.done = done

        def query(self) -> bool:
            return self.done

    slot = BatchOutputSlot()
    producer = Fence(done=True)
    lease = slot.publish(("borrowed",), producer_fence=producer)
    consumer = Fence()

    def partially_enqueue(_borrowed: object, _producer: object) -> None:
        # A consumer can enqueue work and then raise while preparing metadata.
        # The output remains leased until that work is explicitly complete.
        raise RuntimeError("consumer failed after enqueue")

    with pytest.raises(RuntimeError, match="after enqueue"):
        lease.retain_consumer(partially_enqueue, completion_fence=consumer)
    with pytest.raises(OutputSlotBusy):
        slot.publish(("next",), producer_fence=Fence(done=True))

    consumer.done = True
    next_lease = slot.publish(("next",), producer_fence=Fence(done=True))
    assert next_lease.generation == lease.generation + 1


def test_borrowed_output_consumer_registration_is_atomic_with_slot_retirement() -> None:
    class Fence:
        def __init__(self, done: bool = False) -> None:
            self.done = done

        def query(self) -> bool:
            return self.done

    slot = BatchOutputSlot()
    lease = slot.publish(("borrowed",), producer_fence=Fence(done=True))
    consumer_fence = Fence(done=True)
    callback_entered = threading.Event()
    allow_enqueue = threading.Event()
    publish_started = threading.Event()
    publish_finished = threading.Event()
    publish_result: list[object] = []

    def delayed_enqueue(borrowed: object, producer: object) -> object:
        assert borrowed == ("borrowed",)
        assert producer is not None
        callback_entered.set()
        assert allow_enqueue.wait(timeout=2.0)
        # An unrecorded event can still query complete until its consumer work
        # has actually been enqueued and the event recorded.
        consumer_fence.done = False
        return borrowed

    consumer_thread = threading.Thread(
        target=lambda: lease.retain_consumer(
            delayed_enqueue,
            completion_fence=consumer_fence,
        )
    )
    consumer_thread.start()
    assert callback_entered.wait(timeout=2.0)

    def publish_next() -> None:
        publish_started.set()
        try:
            publish_result.append(slot.publish(("next",), producer_fence=Fence(done=True)))
        except BaseException as exc:
            publish_result.append(exc)
        finally:
            publish_finished.set()

    publisher_thread = threading.Thread(target=publish_next)
    publisher_thread.start()
    assert publish_started.wait(timeout=2.0)
    # The publisher must wait until consumer enqueue/registration is atomic.
    assert not publish_finished.wait(timeout=0.05)
    allow_enqueue.set()
    consumer_thread.join(timeout=2.0)
    publisher_thread.join(timeout=2.0)

    assert not consumer_thread.is_alive()
    assert not publisher_thread.is_alive()
    assert len(publish_result) == 1
    assert isinstance(publish_result[0], OutputSlotBusy)
    assert slot._current is lease


class _FixedBatchAdapter(NativeControl6LoRA):
    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__(torch.zeros((len(ACTIONS), HIDDEN_WIDTH), dtype=torch.float32))
        self.fixed_logits = torch.tensor(logits, dtype=torch.float32)
        self.forward_calls = 0

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(normalized_hidden).all():
            raise ValueError("normalized_hidden contains non-finite values")
        self.forward_calls += 1
        rows = normalized_hidden.shape[0]
        return self.fixed_logits[:rows].to(normalized_hidden.device)


class _RecordingBatchAdapter(_FixedBatchAdapter):
    def __init__(self, logits: list[list[float]]) -> None:
        super().__init__(logits)
        self.seen_hidden: torch.Tensor | None = None

    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        self.seen_hidden = normalized_hidden.detach().clone()
        return super().forward(normalized_hidden)


class _NonfiniteLogitAdapter(_FixedBatchAdapter):
    def forward(self, normalized_hidden: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        rows = normalized_hidden.shape[0]
        return torch.full((rows, len(ACTIONS)), float("nan"), device=normalized_hidden.device)


def _policy_input(
    session_id: str,
    frame_id: int,
    legal_actions: tuple[str, ...],
    hidden: torch.Tensor | None = None,
    *,
    generation_id: int = 3,
) -> PolicyInput:
    return PolicyInput(
        session_id=session_id,
        frame_id=frame_id,
        user_pcm=None,
        speaker_active=False,
        generation_id=generation_id,
        native_control_hidden=torch.zeros((1, HIDDEN_WIDTH)) if hidden is None else hidden,
        native_control_hidden_frame_id=frame_id,
        native_control_hidden_at_ns=1,
        legal_actions=legal_actions,
    )


def _batch_policy_input(
    session_id: str,
    frame_id: int,
    legal_actions: tuple[str, ...],
    *,
    generation_id: int = 3,
    state_epoch_id: str | None = None,
    state_version: int | None = None,
    hidden_generation_id: int | None = None,
    hidden_state_epoch_id: str | None = None,
    hidden_state_version: int | None = None,
    hidden_frame_id: int | None = None,
    hidden: torch.Tensor | None = None,
) -> BatchPolicyInput:
    epoch = state_epoch_id or f"epoch-{session_id}"
    version = frame_id if state_version is None else state_version
    return BatchPolicyInput(
        state=_policy_input(
            session_id, frame_id, legal_actions, hidden, generation_id=generation_id
        ),
        state_epoch_id=epoch,
        listener_state_version=version,
        hidden_provenance=ListenerHiddenProvenance(
            session_id=session_id,
            generation_id=generation_id if hidden_generation_id is None else hidden_generation_id,
            state_epoch_id=epoch if hidden_state_epoch_id is None else hidden_state_epoch_id,
            state_version=version if hidden_state_version is None else hidden_state_version,
            frame_id=frame_id if hidden_frame_id is None else hidden_frame_id,
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 8, 12])
def test_batched_control6_matches_single_row_policy_for_each_real_row(batch_size: int) -> None:
    base_logits = [
        [0.0, 1.0, 7.0, 12.0, -1.0, -2.0],
        [9.0, 0.0, 8.0, -1.0, 3.0, 2.0],
        [-2.0, 0.0, 1.0, 2.0, 3.0, 10.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    ]
    base_legal_sets = [
        ("CONTINUE", "YIELD"),
        ("LISTEN", "BACKCHANNEL"),
        ("SPEAK", "REPLAN"),
        tuple(ACTIONS),
    ]
    logits = [base_logits[index % len(base_logits)] for index in range(batch_size)]
    legal_sets = [base_legal_sets[index % len(base_legal_sets)] for index in range(batch_size)]
    batch_adapter = _FixedBatchAdapter(logits)
    batch_policy = GemmaNativeControl6BatchPolicy(batch_adapter, checkpoint_path="cpu-test.pt")
    states = tuple(
        _batch_policy_input(f"s{index}", 12 + index, legal_sets[index])
        for index in range(batch_size)
    )

    batch_decisions = batch_policy.step_batch(states)

    assert len(batch_decisions) == batch_size
    assert batch_adapter.forward_calls == 1
    for index, (batch_state, decision) in enumerate(zip(states, batch_decisions, strict=True)):
        state = batch_state.state
        single_policy = GemmaNativeControl6Policy(
            _FixedBatchAdapter([logits[index]]), checkpoint_path="cpu-test.pt"
        )
        single = single_policy.step(state)
        assert decision.action is single.action
        assert decision.confidence == single.confidence
        assert decision.diagnostics["native_control6_raw_row_index"] == single.diagnostics[
            "native_control6_raw_row_index"
        ]
        assert decision.diagnostics["native_control6_selected_row_index"] == single.diagnostics[
            "native_control6_selected_row_index"
        ]


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 8, 12])
def test_native_output_entry_binds_exact_first_query_rows_to_ordered_manifest(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    batch_size: int,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=batch_size)
    for index in range(batch_size):
        coordinator.enqueue(_frame(f"s{index}", 0, 10 + index))
    batch = coordinator.take_ready(2_000_100)
    assert batch is not None
    manifest = batch.manifest

    output_tensor = torch.zeros((batch_size, 4, HIDDEN_WIDTH), dtype=torch.bfloat16)
    for index in range(batch_size):
        output_tensor[index, 0].fill_(index + 1)
        output_tensor[index, 1].fill_(101 + index)
        output_tensor[index, 2].fill_(201 + index)
        output_tensor[index, 3].fill_(301 + index)
    native_output = NativeBatchOutput(
        manifest_id=manifest.manifest_id,
        generation=9,
        rows=manifest.rows,
        tensor=output_tensor,
        completion_event=object(),
    )
    policy_states = tuple(
        PolicyInput(
            session_id=row.session_id,
            frame_id=row.frame_id,
            user_pcm=None,
            speaker_active=False,
            generation_id=row.session_generation,
            legal_actions=tuple(ACTIONS),
        )
        for row in manifest.rows
    )
    adapter = _RecordingBatchAdapter(
        [[0.0] * len(ACTIONS) for _ in range(batch_size)]
    )
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")

    policy.step_native_output_batch(native_output, manifest, policy_states)

    assert adapter.forward_calls == 1
    assert adapter.seen_hidden is not None
    # Production uses the first hidden row from each Q=4 frame, matching
    # GemmaAudioTowerBackend._run_incremental_lm's hidden[:, 0, :] return.
    assert torch.equal(adapter.seen_hidden, output_tensor[:, 0, :].float())


def test_native_output_entry_rejects_caller_hidden_and_reordered_output_rows(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=2)
    coordinator.enqueue(_frame("s0", 0, 10))
    coordinator.enqueue(_frame("s1", 0, 11))
    batch = coordinator.take_ready(2_000_100)
    assert batch is not None
    manifest = batch.manifest
    output = NativeBatchOutput(
        manifest_id=manifest.manifest_id,
        generation=9,
        rows=manifest.rows,
        tensor=torch.zeros((2, 4, HIDDEN_WIDTH), dtype=torch.bfloat16),
        completion_event=object(),
    )
    policy_states = tuple(
        PolicyInput(
            session_id=row.session_id,
            frame_id=row.frame_id,
            user_pcm=None,
            speaker_active=False,
            generation_id=row.session_generation,
            legal_actions=tuple(ACTIONS),
        )
        for row in manifest.rows
    )
    adapter = _RecordingBatchAdapter([[0.0] * len(ACTIONS) for _ in range(2)])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")

    caller_hidden = (
        replace(policy_states[0], native_control_hidden=torch.zeros(HIDDEN_WIDTH)),
        policy_states[1],
    )
    with pytest.raises(ValueError, match="must not provide native hidden"):
        policy.step_native_output_batch(output, manifest, caller_hidden)
    assert adapter.forward_calls == 0
    with pytest.raises(RuntimeError, match="policy input order/provenance"):
        policy.step_native_output_batch(output, manifest, tuple(reversed(policy_states)))
    assert adapter.forward_calls == 0

    wrong_manifest_output = replace(output, manifest_id="different-manifest")
    with pytest.raises(RuntimeError, match="different manifest"):
        policy.step_native_output_batch(wrong_manifest_output, manifest, policy_states)
    assert adapter.forward_calls == 0

    reordered_output = replace(output, rows=tuple(reversed(manifest.rows)))
    with pytest.raises(RuntimeError, match="ordered manifest rows"):
        policy.step_native_output_batch(reordered_output, manifest, policy_states)
    assert adapter.forward_calls == 0


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_raw_control6_scores_use_one_real_native_q0_batch_and_preserve_single_policy(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    batch_size: int,
) -> None:
    coordinator = _coordinator(runtime_signature, capability, session_count=batch_size)
    for index in range(batch_size):
        coordinator.enqueue(_frame(f"s{index}", 0, 10 + index))
    batch = coordinator.take_ready(2_000_100)
    assert batch is not None and batch.manifest.batch_size == batch_size
    manifest = batch.manifest

    generator = torch.Generator().manual_seed(441 + batch_size)
    output_tensor = torch.randn(
        (batch_size, 4, HIDDEN_WIDTH), generator=generator, dtype=torch.float32
    ).to(torch.bfloat16)
    output_tensor[:, 1:, :].fill_(20.0)  # Query zero is the trained policy input.
    native_output = NativeBatchOutput(
        manifest_id=manifest.manifest_id,
        generation=17,
        rows=manifest.rows,
        tensor=output_tensor,
        completion_event=object(),
    )

    class CountingAdapter(NativeControl6LoRA):
        def __init__(self, rows: torch.Tensor) -> None:
            super().__init__(rows, rank=4)
            self.forward_calls = 0

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            self.forward_calls += 1
            return super().forward(hidden)

    torch.manual_seed(442)
    base_rows = torch.randn((len(ACTIONS), HIDDEN_WIDTH), dtype=torch.float32) * 0.01
    batch_adapter = CountingAdapter(base_rows)
    with torch.no_grad():
        batch_adapter.lora_B.normal_(mean=0.0, std=0.02)
    single_adapter = CountingAdapter(base_rows)
    single_adapter.load_state_dict(copy.deepcopy(batch_adapter.state_dict()))
    batch_policy = GemmaNativeControl6BatchPolicy(
        batch_adapter, checkpoint_path="cpu-native-control6.pt"
    )

    snapshots = batch_policy.score_raw_batch(native_output, manifest)

    assert len(snapshots) == batch_size
    assert batch_adapter.forward_calls == 1
    assert batch_policy.forward_call_count == 1
    assert all(isinstance(item, NativeControl6RawScoreSnapshot) for item in snapshots)
    for index, (row, snapshot) in enumerate(zip(manifest.rows, snapshots, strict=True)):
        assert (
            snapshot.manifest_id,
            snapshot.output_generation,
            snapshot.session_id,
            snapshot.session_generation,
            snapshot.state_epoch_id,
            snapshot.listener_state_version,
            snapshot.frame_id,
            snapshot.sequence_position,
            snapshot.cache_position,
            snapshot.query_index,
        ) == (
            manifest.manifest_id,
            native_output.generation,
            row.session_id,
            row.session_generation,
            row.state_epoch_id,
            row.listener_state_version,
            row.frame_id,
            row.sequence_position,
            row.cache_position,
            0,
        )
        reference_hidden = output_tensor[index, 0, :].reshape(1, -1).float()
        with torch.inference_mode():
            reference_logits = single_adapter(reference_hidden).reshape(-1)
        assert torch.allclose(
            torch.tensor(snapshot.raw_logits), reference_logits, atol=1e-6, rtol=1e-6
        )

    single_policy = GemmaNativeControl6Policy(
        single_adapter, checkpoint_path="cpu-native-control6.pt"
    )
    legal_sets = [tuple(ACTIONS), (ACTIONS[0], ACTIONS[2]), (ACTIONS[1], ACTIONS[4]), (ACTIONS[2],)]
    for index, (row, snapshot) in enumerate(zip(manifest.rows, snapshots, strict=True)):
        state = _policy_input(
            row.session_id,
            row.frame_id,
            legal_sets[index % len(legal_sets)],
            output_tensor[index, 0, :].reshape(1, -1),
            generation_id=row.session_generation,
        )
        expected = single_policy.step(state)
        calls_before = single_adapter.forward_calls
        actual = single_policy.step(replace(
            state,
            native_control6_raw_scores=snapshot,
            native_listener_generation=row.session_generation,
        ))
        assert single_adapter.forward_calls == calls_before
        assert actual.action is expected.action
        assert actual.confidence == pytest.approx(expected.confidence, rel=1e-6, abs=1e-7)
        assert actual.diagnostics["native_control6_raw_row_index"] == expected.diagnostics[
            "native_control6_raw_row_index"
        ]
        assert actual.diagnostics["native_control6_selected_row_index"] == expected.diagnostics[
            "native_control6_selected_row_index"
        ]


def test_raw_control6_snapshot_rejects_nonfinite_scores_and_wrong_native_output() -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        NativeControl6RawScoreSnapshot(
            manifest_id="manifest",
            output_generation=1,
            session_id="session",
            session_generation=1,
            state_epoch_id="epoch",
            listener_state_version=0,
            frame_id=0,
            sequence_position=0,
            cache_position=34,
            query_index=0,
            checkpoint_path="cpu.pt",
            raw_logits=(0.0, 1.0, float("nan"), 3.0, 4.0, 5.0),
        )

    adapter = _FixedBatchAdapter([[0.0] * len(ACTIONS)])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu.pt")
    output = NativeBatchOutput(
        manifest_id="wrong-manifest", generation=1, rows=(),
        tensor=torch.zeros((1, 4, HIDDEN_WIDTH), dtype=torch.bfloat16),
        completion_event=object(),
    )
    manifest = ListenerBatchManifest(
            manifest_id="expected-manifest",
            runtime_signature=RuntimeSignature(
                model_id="model", model_config_sha256="a" * 64,
                checkpoint_sha256="b" * 64, topology_sha256="c" * 64,
                native_library_sha256="d" * 64, native_abi_version=3,
                batch_api_version=2, attention_backend_id=3, rmsnorm_backend_id=2,
                max_seq_len=16_384, stream_q_len=4,
            ),
            stream_identity="listener-cuda-stream-17", stream_priority=-1,
            dispatch_at_ns=1, max_coalesce_wait_ns=0, dispatch_reason="batch_full",
            rows=(BatchRowDescriptor(
                session_id="s0", session_generation=1, state_epoch_id="epoch",
                frame_id=0, listener_state_version=0, sequence_position=0,
                cache_position=34, valid_history_tokens=34, native_context_id=1,
                input_frame_sha256="e" * 64, reference_pcm_sha256=None,
                frame_ready_at_ns=1, reference_frame_id=None,
            ),),
        )
    with pytest.raises(RuntimeError, match="different manifest"):
        policy.score_raw_batch(output, manifest)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 8, 12])
def test_batched_control6_actual_lora_cpu_forward_matches_single_row_calls(
    batch_size: int,
) -> None:
    torch.manual_seed(91)
    base_rows = torch.randn((len(ACTIONS), HIDDEN_WIDTH), dtype=torch.float32) * 0.01
    batch_adapter = NativeControl6LoRA(base_rows, rank=4)
    with torch.no_grad():
        batch_adapter.lora_B.normal_(mean=0.0, std=0.02)
    single_adapter = NativeControl6LoRA(base_rows, rank=4)
    single_adapter.load_state_dict(batch_adapter.state_dict())

    generator = torch.Generator().manual_seed(1729)
    states = tuple(
        _batch_policy_input(
            f"actual-{index}",
            20 + index,
            tuple(ACTIONS) if index % 2 == 0 else ("LISTEN", "CONTINUE", "REPLAN"),
            hidden=torch.randn((1, HIDDEN_WIDTH), generator=generator),
        )
        for index in range(batch_size)
    )
    batched = GemmaNativeControl6BatchPolicy(
        batch_adapter, checkpoint_path="cpu-actual-lora.pt"
    ).step_batch(states)

    singles = tuple(
        GemmaNativeControl6Policy(
            single_adapter, checkpoint_path="cpu-actual-lora.pt"
        ).step(state.state)
        for state in states
    )
    for row, (actual, reference) in enumerate(zip(batched, singles, strict=True)):
        assert actual.action is reference.action, row
        assert actual.diagnostics["native_control6_raw_token_id"] == reference.diagnostics[
            "native_control6_raw_token_id"
        ], row
        assert actual.diagnostics["native_control6_selected_token_id"] == reference.diagnostics[
            "native_control6_selected_token_id"
        ], row
        assert actual.confidence == pytest.approx(reference.confidence, rel=1e-6, abs=1e-7), row


def test_batched_control6_requires_unique_sessions_and_fresh_hidden() -> None:
    policy = GemmaNativeControl6BatchPolicy(
        _FixedBatchAdapter([[1.0] * len(ACTIONS)]), checkpoint_path="cpu-test.pt"
    )
    with pytest.raises(ValueError, match="distinct session"):
        policy.step_batch((_batch_policy_input("s0", 0, tuple(ACTIONS)),) * 3)
    with pytest.raises(RuntimeError, match="hidden provenance|stale listener hidden"):
        policy.step_batch((_batch_policy_input(
            "s0", 8, tuple(ACTIONS), hidden_frame_id=7
        ),))


@pytest.mark.parametrize(
    "overrides",
    [
        {"hidden_generation_id": 2},
        {"hidden_state_epoch_id": "old-epoch"},
        {"hidden_state_version": 6},
        {"hidden_frame_id": 6},
    ],
)
def test_batched_control6_rejects_hidden_from_old_row_provenance(
    overrides: dict[str, object],
) -> None:
    adapter = _FixedBatchAdapter([[1.0] * len(ACTIONS)])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")
    row = _batch_policy_input(
        "same-session",
        7,
        tuple(ACTIONS),
        state_epoch_id="epoch-new",
        state_version=7,
        **overrides,
    )

    with pytest.raises(RuntimeError, match="hidden provenance"):
        policy.step_batch((row,))
    assert adapter.forward_calls == 0


def test_batched_control6_fails_closed_for_nonfinite_rows_without_per_row_fallback() -> None:
    adapter = _FixedBatchAdapter([[1.0] * len(ACTIONS), [1.0] * len(ACTIONS)])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")
    states = (
        _batch_policy_input("s0", 0, tuple(ACTIONS)),
        _batch_policy_input(
            "s1", 0, tuple(ACTIONS), hidden=torch.full((1, HIDDEN_WIDTH), float("nan"))
        ),
    )

    with pytest.raises(ValueError, match="non-finite"):
        policy.step_batch(states)
    assert adapter.forward_calls == 0


def test_batched_control6_fails_closed_for_nonfinite_logits_in_the_metadata_packet() -> None:
    adapter = _NonfiniteLogitAdapter([[0.0] * len(ACTIONS)])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")

    with pytest.raises(RuntimeError, match="invalid six-way logits"):
        policy.step_batch((_batch_policy_input("s0", 0, tuple(ACTIONS)),))
    assert adapter.forward_calls == 1


def test_batched_control6_uses_one_policy_metadata_readback_and_no_scalar_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FixedBatchAdapter([
        [0.0, 1.0, 7.0, 12.0, -1.0, -2.0],
        [9.0, 0.0, 8.0, -1.0, 3.0, 2.0],
    ])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")
    cpu_calls = 0
    item_calls = 0
    original_cpu = torch.Tensor.cpu
    original_item = torch.Tensor.item

    def counted_cpu(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal cpu_calls
        cpu_calls += 1
        return original_cpu(self, *args, **kwargs)

    def counted_item(self: torch.Tensor, *args: object, **kwargs: object) -> object:
        nonlocal item_calls
        item_calls += 1
        return original_item(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    monkeypatch.setattr(torch.Tensor, "item", counted_item)

    policy.step_batch((
        _batch_policy_input("s0", 0, ("SPEAK", "CONTINUE")),
        _batch_policy_input("s1", 0, ("LISTEN", "BACKCHANNEL")),
    ))

    # Counts only the packed policy-result transfer. The production adapter's
    # unchanged finite-input guard can independently synchronize on CUDA.
    assert cpu_calls == 1
    assert item_calls == 0
    assert adapter.forward_calls == 1


def test_batched_control6_reset_is_stateless_across_rows_and_calls() -> None:
    adapter = _FixedBatchAdapter([
        [0.0, 1.0, 7.0, 12.0, -1.0, -2.0],
        [9.0, 0.0, 8.0, -1.0, 3.0, 2.0],
    ])
    policy = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path="cpu-test.pt")
    first = policy.step_batch((
        _batch_policy_input("s0", 0, ("SPEAK", "CONTINUE")),
        _batch_policy_input("s1", 0, ("LISTEN", "BACKCHANNEL")),
    ))

    policy.reset()

    second = policy.step_batch((
        _batch_policy_input("s1", 1, ("LISTEN", "BACKCHANNEL")),
        _batch_policy_input("s0", 1, ("SPEAK", "CONTINUE")),
    ))
    assert [decision.action for decision in first] == [ControlAction.CONTINUE, ControlAction.LISTEN]
    assert [decision.action for decision in second] == [ControlAction.LISTEN, ControlAction.CONTINUE]
    assert [decision.session for decision in second] == ["s1", "s0"]
    assert adapter.forward_calls == 2
