import asyncio
import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from duplex_kernel import listener as listener_module
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.server_metrics import _send_unified_native_finalizer_event
from duplex_kernel.server_realtime_scheduler import _record_listener_commit_timing


class _ObservedLock:
    def __init__(self):
        self._lock = threading.Lock()
        self.held = False

    def acquire(self):
        self._lock.acquire()
        self.held = True

    def release(self):
        self.held = False
        self._lock.release()


class _ArrayProbe:
    def __init__(self, lock, value):
        self.lock = lock
        self.value = value

    def __array__(self, dtype=None, copy=None):
        assert not self.lock.held
        return np.array(self.value, dtype=dtype, copy=True)


class _TensorProbe:
    def __init__(self, lock, shape):
        self.lock = lock
        self.shape = shape
        self.detached = SimpleNamespace(shape=shape)

    def detach(self):
        assert not self.lock.held
        return self.detached


def _row_output(state, *, hidden, queries, tower, snapshot=None):
    if snapshot is None:
        snapshot = SimpleNamespace(
            user_mel_history=(np.ones((8, 128), dtype=np.float32),),
            assistant_mel_history=(np.zeros((8, 128), dtype=np.float32),),
            user_pcm_tail=np.ones(160, dtype=np.float32),
            assistant_pcm_tail=np.zeros(160, dtype=np.float32),
        )
    return SimpleNamespace(
        generation_id=7,
        frame_id=state.next_frame_id,
        state_epoch_id=state.state_epoch_id,
        listener_state_version=state.listener_state_version + 1,
        sequence_position=1,
        cache_position=36,
        valid_history_tokens=36,
        logical_q_len=2,
        ephemeral_reason=None,
        queue_audio_for_next_turn=False,
        kv_rollback_tokens=0,
        gemma_hidden_t=hidden,
        gemma_audio_queries_t=queries,
        audio_tower_hidden=tower,
        state_snapshot=snapshot,
        native_state=SimpleNamespace(
            current_position=36,
            prefix_len=34,
            valid_kv_len=35,
        ),
        native_output_health=SimpleNamespace(status="healthy", checked=True),
        semantic_context_revision=5,
        cache_epoch_id=2,
        applied_history_revision=3,
        semantic_replay_fallback_reason=None,
        semantic_prefix_len=44,
        listener_model_lock_wait_ms=1.25,
        listener_model_lock_hold_ms=4.5,
    )


def test_commit_microbatch_output_prepare_error_leaves_state_unmodified():
    state = GemmaListenerState()
    listener = GemmaListener(backend=SimpleNamespace(hidden_size=1536))
    original_user = np.full(listener.frame_samples, 0.1, dtype=np.float32)
    original_assistant = np.full(listener.frame_samples, 0.2, dtype=np.float32)
    state.user_pcm_history.append(original_user)
    state.assistant_pcm_history.append(original_assistant)
    state.user_mel_history.append(np.ones((2, 4), dtype=np.float32))
    state.assistant_mel_history.append(np.zeros((2, 4), dtype=np.float32))
    state.last_frontend_wall_ms = 12.0
    state.last_audio_tower_call_wall_ms = 13.0
    user_history_before = tuple(state.user_pcm_history)
    assistant_history_before = tuple(state.assistant_pcm_history)
    user_mel_before = tuple(state.user_mel_history)
    assistant_mel_before = tuple(state.assistant_mel_history)
    row = _row_output(
        state,
        hidden=SimpleNamespace(shape=(1, 1536), detach=lambda: None),
        queries=SimpleNamespace(shape=(2, 1536), detach=lambda: None),
        tower=SimpleNamespace(shape=(2, 1536)),
        snapshot=None,
    )
    row.state_snapshot = None

    with pytest.raises(RuntimeError, match="lacks its committed frontend-state snapshot"):
        listener.commit_microbatch_output(
            np.zeros(listener.frame_samples, dtype=np.float32),
            None,
            state,
            row,
            source_available_at_ns=123,
        )

    assert len(state.user_pcm_history) == len(user_history_before)
    assert len(state.assistant_pcm_history) == len(assistant_history_before)
    assert len(state.user_mel_history) == len(user_mel_before)
    assert len(state.assistant_mel_history) == len(assistant_mel_before)
    assert all(actual is before for actual, before in zip(state.user_pcm_history, user_history_before))
    assert all(
        actual is before
        for actual, before in zip(state.assistant_pcm_history, assistant_history_before)
    )
    assert all(actual is before for actual, before in zip(state.user_mel_history, user_mel_before))
    assert all(
        actual is before
        for actual, before in zip(state.assistant_mel_history, assistant_mel_before)
    )
    assert state.next_frame_id == 0
    assert state.listener_state_version == 0
    assert state.last_frontend_wall_ms == 12.0
    assert state.last_audio_tower_call_wall_ms == 13.0


def test_commit_microbatch_output_minimal_lock_hold_and_valid_output(monkeypatch):
    state = GemmaListenerState()
    lock = _ObservedLock()
    state._lock = lock
    frame_done = threading.Event()
    tick_started = threading.Event()
    backend = SimpleNamespace(
        hidden_size=1536,
        _frame_done_event=frame_done,
        _tick_started_event=tick_started,
        _frame_counter=0,
    )
    listener = GemmaListener(backend=backend)
    hidden = _TensorProbe(lock, (1, 1536))
    queries = _TensorProbe(lock, (2, 1536))
    tower = SimpleNamespace(shape=(2, 1536))
    mel_values = tuple(
        _ArrayProbe(lock, np.ones((80, 128), dtype=np.float32))
        for _ in range(8)
    )
    assistant_mel_values = tuple(
        _ArrayProbe(lock, np.zeros((80, 128), dtype=np.float32))
        for _ in range(8)
    )
    snapshot = SimpleNamespace(
        user_mel_history=mel_values,
        assistant_mel_history=assistant_mel_values,
        user_pcm_tail=_ArrayProbe(lock, np.ones(160, dtype=np.float32)),
        assistant_pcm_tail=_ArrayProbe(lock, np.zeros(160, dtype=np.float32)),
    )
    row = _row_output(state, hidden=hidden, queries=queries, tower=tower, snapshot=snapshot)
    state.last_frontend_wall_ms = 12.0
    state.last_audio_tower_call_wall_ms = 13.0
    state.gemma_lm_cache = object()
    state.gemma_lm_shared_kv_states = object()
    state.gemma_lm_native = object()
    state.gemma_lm_text_context = (11, 12)
    state.cache_epoch_id = 1
    original_sha256 = listener_module.hashlib.sha256

    def checked_sha256(data=b"", *args, **kwargs):
        if lock.held:
            assert len(data) < listener.frame_samples * np.dtype("<f4").itemsize
        return original_sha256(data, *args, **kwargs)

    monkeypatch.setattr(listener_module.hashlib, "sha256", checked_sha256)
    timing = {}

    output = listener.commit_microbatch_output(
        _ArrayProbe(lock, np.full(listener.frame_samples, 0.25, dtype=np.float32)),
        None,
        state,
        row,
        source_available_at_ns=123_456,
        timing_sink=timing,
    )

    assert output.gemma_hidden_t is hidden.detached
    assert output.gemma_audio_queries_t is queries.detached
    assert output.audio_tower_hidden is tower
    assert output.audio_timestamp == 80_000_000
    assert output.listener_state_version == 1
    assert output.frame_id == output.source_frame_id == 0
    assert output.reference_frame_id is None
    assert output.source_available_at_ns == 123_456
    assert output.finite and output.fresh and not output.silence_skipped
    assert output.listener_generation_id == 7
    assert output.listener_state_epoch_id == state.state_epoch_id
    assert output.listener_sequence_position == 1
    assert output.listener_cache_position == 36
    assert output.listener_valid_history_tokens == 36
    assert output.native_cache_current_position_after == 36
    assert output.native_cache_valid_kv_len_after == 35
    assert state.next_frame_id == state.listener_state_version == state.listener_input_frames == 1
    assert state.total_samples == listener.frame_samples
    assert state.state_lineage_digest
    assert state.last_hidden is state.native_last_hidden is hidden.detached
    assert state.last_audio_queries is queries.detached
    assert state.last_hidden_source_frame_id == 0
    assert state.last_hidden_at_ns == 123_456
    assert state.gemma_lm_audio_frames == 1
    assert state.gemma_lm_source_frame_ids == [0]
    assert state.gemma_lm_context_start_frame == state.gemma_lm_last_source_frame_id == 0
    assert state.gemma_lm_text_context == ()
    assert state.gemma_lm_cache is state.gemma_lm_shared_kv_states is state.gemma_lm_native is None
    assert state.native_semantic_context_revision == 5
    assert state.semantic_context_revision == state.applied_history_revision == 3
    assert state.cache_epoch_id == 2
    assert state.gemma_lm_prefix_len == 44
    assert state.last_lock_wait_ms == 1.25
    assert state.last_lock_hold_ms == 4.5
    assert state.last_frontend_wall_ms is None
    assert state.last_audio_tower_call_wall_ms is None
    assert len(state.user_pcm_history) == len(state.assistant_pcm_history) == 1
    np.testing.assert_array_equal(
        state.user_pcm_history[0], np.full(listener.frame_samples, 0.25, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        state.assistant_pcm_history[0], np.zeros(listener.frame_samples, dtype=np.float32)
    )
    assert len(state.user_mel_history) == len(state.assistant_mel_history) == 8
    np.testing.assert_array_equal(state.user_pcm_tail, np.ones(160, dtype=np.float32))
    np.testing.assert_array_equal(state.assistant_pcm_tail, np.zeros(160, dtype=np.float32))
    assert tick_started.is_set() and frame_done.is_set()
    assert backend._frame_counter == 1
    assert timing["state_lock_acquired_ns"] <= timing["state_lock_released_ns"]
    assert timing["state_lock_wait_ns"] >= 0
    assert 0 <= timing["state_lock_hold_ns"] < timing["prepare_ns"]

    tick = SimpleNamespace()
    _record_listener_commit_timing(tick, timing)
    assert tick.microbatch_commit_state_lock_wait_ms is not None
    assert tick.microbatch_commit_state_lock_hold_ms is not None
    assert tick.microbatch_commit_state_lock_acquired_ns == timing["state_lock_acquired_ns"]
    assert tick.microbatch_commit_state_lock_released_ns == timing["state_lock_released_ns"]
    assert tick.microbatch_commit_prepare_ms == round(timing["prepare_ns"] / 1_000_000.0, 3)


def test_commit_microbatch_output_preserves_first_string_history_revision():
    state = GemmaListenerState(semantic_context_revision=4)
    listener = GemmaListener(backend=SimpleNamespace(hidden_size=1536))
    row = _row_output(
        state,
        hidden=SimpleNamespace(shape=(1, 1536), detach=lambda: object()),
        queries=SimpleNamespace(shape=(2, 1536), detach=lambda: object()),
        tower=SimpleNamespace(shape=(2, 1536)),
    )
    row.applied_history_revision = "history-r3"

    listener.commit_microbatch_output(
        np.zeros(listener.frame_samples, dtype=np.float32),
        None,
        state,
        row,
        source_available_at_ns=123,
    )

    assert state.applied_history_revision == "history-r3"
    assert state.semantic_context_revision == 4
    assert state.native_semantic_context_revision == 5


def test_send_unified_native_finalizer_event_payload_is_compact():
    sent = []

    async def send(payload):
        sent.append(payload)
        return True

    runtime_metrics = {
        "unified_native_rollback": {
            "physical_rewind_tokens": 5,
            "acknowledged_source_token_ids_replayed": list(range(5_000)),
            "failure_reason": "z" * 80_000,
            "trace": "x" * 80_000,
        },
        "unified_native_finalizer": {
            "queued_audio_drain_ms": 12.5,
            "queued_audio_drain_frames": 3,
            "queued_audio_source_frame_ids": list(range(5_000)),
            "lock_timeline": [
                {"acquired_ns": index, "released_ns": index + 1}
                for index in range(5_000)
            ],
        },
        "generation_trace": [{"token": index} for index in range(5_000)],
        "scheduler_lock_timeline": "y" * 80_000,
    }

    accepted = asyncio.run(
        _send_unified_native_finalizer_event(
            send,
            "response-final",
            status="completed",
            rollback_metrics={"physical_rewind_tokens": 5},
            drain_metrics={"queued_audio_drain_ms": 12.5, "queued_audio_drain_frames": 3},
            runtime_metrics=runtime_metrics,
        )
    )

    assert accepted is True
    assert sent[0]["response_id"] == "response-final"
    assert sent[0]["status"] == "completed"
    compact_metrics = sent[0]["runtime_metrics"]
    assert set(compact_metrics) == {
        "unified_native_rollback",
        "unified_native_finalizer",
    }
    assert compact_metrics["unified_native_rollback"]["physical_rewind_tokens"] == 5
    assert len(
        compact_metrics["unified_native_rollback"][
            "acknowledged_source_token_ids_replayed"
        ]
    ) <= 32
    assert len(compact_metrics["unified_native_rollback"]["failure_reason"]) <= 240
    assert compact_metrics["unified_native_finalizer"]["queued_audio_drain_ms"] == 12.5
    assert len(
        compact_metrics["unified_native_finalizer"]["queued_audio_source_frame_ids"]
    ) <= 32
    assert len(json.dumps(sent[0]).encode("utf-8")) < 2_048
