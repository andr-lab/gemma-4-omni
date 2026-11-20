from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from duplex_kernel.listener_microbatch_worker import CompactionBuilderCapacityError
from duplex_kernel.scheduler import AudioFrame
from duplex_kernel.server_listener_compaction import SessionListenerCompactionAdapter
from duplex_kernel.session import SessionHistorySnapshot, UserHistoryItemProvenance
from duplex_kernel.server_realtime_policy import _prepare_policy_tick_state_impl
import duplex_kernel.server as server_module


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        del add_generation_prompt
        rendered = "\n".join(
            f"{message['role']} {message['content']}" for message in messages
        )
        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return [900 if word == "<|audio|>" else 1 for word in str(text).split()]


def _engine():
    return SimpleNamespace(
        tokenizer=_Tokenizer(),
        _thinker_wrapper=SimpleNamespace(
            processor=SimpleNamespace(audio_token="<|audio|>")
        ),
    )


def _record(
    *,
    session_id,
    archive_id,
    item_id,
    message_index,
    first,
    last,
    revision,
    sequence,
    epoch="epoch-a",
    timeline="timeline-a",
    source="post_response_asr",
    coverage=True,
    asr=True,
):
    return UserHistoryItemProvenance(
        session_id=session_id,
        archive_id=archive_id,
        history_item_id=item_id,
        message_index=message_index,
        timeline_id=timeline,
        state_epoch_id=epoch,
        first_frame_id=first,
        last_frame_id=last,
        coverage_complete=coverage,
        transcript_status="committed",
        source=source,
        asr_provenance=(
            (
                ("model", "whisper-small"),
                ("device", "cpu"),
                ("compute_type", "int8"),
                ("cpu_threads", 4),
            )
            if asr else ()
        ),
        history_revision=revision,
        history_mutation_sequence=sequence,
    )


def _snapshot(
    session_id,
    history,
    *,
    ids=None,
    provenance=(),
    timeline="timeline-a",
    revision=1,
    sequence=1,
    instructions="Answer the user.",
    language="en",
):
    return SessionHistorySnapshot(
        session_id=session_id,
        history=tuple({"role": role, "content": content} for role, content in history),
        history_item_ids=tuple(ids if ids is not None else range(1, len(history) + 1)),
        provenance=tuple(provenance),
        history_revision=revision,
        history_mutation_sequence=sequence,
        instructions=instructions,
        language=language,
        timeline_id=timeline,
    )


class _Session:
    def __init__(self, snapshot):
        self.session_id = snapshot.session_id
        self.timeline_id = snapshot.timeline_id
        self._snapshot = snapshot
        self.listener = SimpleNamespace(
            backend=SimpleNamespace(boa_token_id=901)
        )

    def snapshot_history_with_provenance(self):
        return self._snapshot


def _listener_snapshot(session_id, *, next_frame_id, epoch="epoch-a", generation=1):
    return SimpleNamespace(
        session_id=session_id,
        generation_id=generation,
        state_epoch_id=epoch,
        next_frame_id=next_frame_id,
    )


def _ring(session_id, first, end_exclusive, *, epoch="epoch-a", generation=1):
    return tuple(
        SimpleNamespace(
            frame=SimpleNamespace(
                session_id=session_id,
                generation_id=generation,
                state_epoch_id=epoch,
                frame_id=frame_id,
            )
        )
        for frame_id in range(first, end_exclusive)
    )


def _adapter_for(session, *, unified=False):
    adapter = SessionListenerCompactionAdapter(
        _engine(),
        unified_kv_enabled=unified,
    )
    adapter.register_session(session, generation_id=1)
    return adapter


def _classify(adapter, session_id, frame_ids, *, voiced, epoch="epoch-a", timeline="timeline-a"):
    for frame_id in frame_ids:
        assert adapter.record_input_frame_classification(
            session_id,
            epoch,
            frame_id,
            voiced,
            timeline_id=timeline,
        )


def test_audio_frame_classification_uses_session_identity_not_scheduler_frame_fields():
    session = _Session(_snapshot("session-a", []))
    session.listener_state = SimpleNamespace(state_epoch_id="epoch-a")
    adapter = _adapter_for(session)
    frame = AudioFrame(
        frame_id=0,
        pcm=np.zeros(1280, dtype=np.float32),
        capture_start_ns=1,
        capture_end_ns=2,
        enqueue_ns=3,
    )

    assert adapter.record_session_input_frame_classification(session, frame, True)
    assert adapter._sessions["session-a"].last_input_by_timeline == {
        "timeline-a": (0, True, "epoch-a")
    }


def test_policy_tick_exports_raw_current_frame_voice_class():
    class _Tracker:
        last_voiced_frame_id = None

        def observe(self, frame_id, pcm):
            if pcm == "voiced":
                self.last_voiced_frame_id = int(frame_id)
            return False, False

    class _SessionState:
        session_id = "vad-session"
        _generation_lock = threading.RLock()
        _policy_revision = 0
        session_generation_id = 1
        turn_policy = None

        def observe_user_onset(self, _frame_id):
            return False

    class _Admission:
        def update_input_frame_locked(self, _frame_id):
            pass

    tracker = _Tracker()
    common = dict(
        sess=_SessionState(),
        speech_tracker=tracker,
        policy_cooldown_lock=threading.Lock(),
        response_audio_admission=_Admission(),
        policy_causality_lock=threading.Lock(),
        pending_response_speak_handoffs=queue.Queue(),
        begin_transcript_utterance=lambda _frame_id: None,
        pending_speak=None,
        peer_audio_event_key=None,
        latest_policy_frame_id=-1,
        partial_last_snapshot_frame=None,
    )
    voiced = _prepare_policy_tick_state_impl(
        SimpleNamespace(frame_id=0, pcm="voiced"),
        SimpleNamespace(),
        **common,
    )[0]
    quiet = _prepare_policy_tick_state_impl(
        SimpleNamespace(frame_id=1, pcm="quiet"),
        SimpleNamespace(),
        **{
            **common,
            "latest_policy_frame_id": 0,
            "pending_speak": None,
            "peer_audio_event_key": None,
        },
    )[0]

    assert voiced["input_frame_voiced"] is True
    assert quiet["input_frame_voiced"] is False


def test_builder_uses_cpu_asr_cutoff_and_exact_contiguous_q2_suffix():
    session_id = "session-a"
    snapshot = _snapshot(
        session_id,
        [
            ("user", "first transcript"),
            ("assistant", "first reply"),
            ("user", "second transcript"),
            ("assistant", "second reply"),
        ],
        ids=(11, 12, 13, 14),
        revision=2,
        sequence=2,
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=11,
                message_index=0,
                first=0,
                last=1,
                revision=1,
                sequence=1,
            ),
            _record(
                session_id=session_id,
                archive_id=2,
                item_id=13,
                message_index=2,
                first=2,
                last=4,
                revision=2,
                sequence=2,
            ),
        ),
    )
    session = _Session(snapshot)
    adapter = _adapter_for(session)

    _classify(adapter, session_id, range(0, 2), voiced=True)
    _classify(adapter, session_id, range(2, 5), voiced=True)
    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=205),
        _ring(session_id, 5, 205),
        "2",
    )

    assert context is not None
    assert context.replay_after_frame_id == 4
    assert context.history_revision == 2
    assert context.history_context[3] == ("replay_after_frame_id", 4)
    assert context.history_context[5] == (
        "included_history",
        (
            ("user", "first transcript"),
            ("assistant", "first reply"),
            ("user", "second transcript"),
            ("assistant", "second reply"),
        ),
    )


def test_missing_covered_frames_is_typed_capacity_and_not_pending():
    session_id = "session-gap"
    snapshot = _snapshot(
        session_id,
        [("user", "covered one"), ("assistant", "reply one"),
         ("user", "covered two"), ("assistant", "reply two")],
        ids=(21, 22, 23, 24),
        revision=2,
        sequence=2,
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=21,
                message_index=0,
                first=0,
                last=1,
                revision=1,
                sequence=1,
            ),
            _record(
                session_id=session_id,
                archive_id=2,
                item_id=23,
                message_index=2,
                first=3,
                last=4,
                revision=2,
                sequence=2,
            ),
        ),
    )
    adapter = _adapter_for(_Session(snapshot))
    for frame_id in range(5):
        # Frame 2 is voiced but no complete ASR interval covers it.
        assert adapter.record_input_frame_classification(
            session_id, "epoch-a", frame_id, True, timeline_id="timeline-a"
        )

    with pytest.raises(CompactionBuilderCapacityError) as error:
        adapter.build_context(
            _listener_snapshot(session_id, next_frame_id=205),
            _ring(session_id, 5, 205),
            "2",
        )

    assert error.value.reason == "evicted_voiced_input_lacks_complete_cpu_asr_coverage"


def test_missing_vad_classification_is_typed_capacity_not_inferred_silence():
    session_id = "session-unclassified"
    snapshot = _snapshot(session_id, [])
    adapter = _adapter_for(_Session(snapshot))
    assert adapter.record_input_frame_classification(
        session_id, "epoch-a", 0, False, timeline_id="timeline-a"
    )
    assert not adapter.record_input_frame_classification(
        session_id, "epoch-a", 2, False, timeline_id="timeline-a"
    )

    with pytest.raises(CompactionBuilderCapacityError) as error:
        adapter.build_context(
            _listener_snapshot(session_id, next_frame_id=205),
            _ring(session_id, 5, 205),
            "1",
        )

    assert error.value.reason == "listener_input_classification_has_unclassified_frames"


def test_new_input_timeline_does_not_reuse_old_vad_coverage():
    session_id = "session-reset-timeline"
    old = _snapshot(session_id, [], timeline="timeline-before-reset")
    session = _Session(old)
    adapter = _adapter_for(session)
    _classify(
        adapter, session_id, range(205), voiced=False,
        timeline="timeline-before-reset",
    )

    new = _snapshot(
        session_id,
        [],
        timeline="timeline-after-reset",
        revision=old.history_revision,
        sequence=old.history_mutation_sequence,
    )
    assert adapter.mark_pending(session_id, new)
    assert adapter.publish_snapshot(session, new)
    with pytest.raises(CompactionBuilderCapacityError) as error:
        adapter.build_context(
            _listener_snapshot(session_id, next_frame_id=205),
            _ring(session_id, 5, 205),
            str(new.history_revision),
        )
    assert error.value.reason == "listener_input_classification_missing_before_q2_ring"

    _classify(
        adapter, session_id, range(205), voiced=False,
        timeline="timeline-after-reset",
    )
    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=205),
        _ring(session_id, 5, 205),
        str(new.history_revision),
    )
    assert context is not None
    assert context.replay_after_frame_id == 4


def test_text_without_provenance_stays_context_and_does_not_create_cutoff():
    session_id = "session-text"
    snapshot = _snapshot(
        session_id,
        [("user", "typed text"), ("assistant", "typed reply")],
        ids=(31, 32),
    )
    adapter = _adapter_for(_Session(snapshot))

    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=3),
        _ring(session_id, 0, 3),
        "1",
    )

    assert context is not None
    assert context.replay_after_frame_id is None
    assert context.history_context[5] == (
        "included_history",
        (("user", "typed text"), ("assistant", "typed reply")),
    )


def test_known_audio_message_overlapping_q2_replay_is_not_in_prefix():
    session_id = "session-overlap"
    snapshot = _snapshot(
        session_id,
        [("user", "current audio transcript"), ("assistant", "reply")],
        ids=(41, 42),
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=41,
                message_index=0,
                first=0,
                last=4,
                revision=1,
                sequence=1,
            ),
        ),
    )
    adapter = _adapter_for(_Session(snapshot))

    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=5),
        _ring(session_id, 0, 5),
        "1",
    )

    assert context is not None
    assert context.replay_after_frame_id is None
    assert context.history_context[5] == (
        "included_history",
        (("assistant", "reply"),),
    )


def test_prefix_budget_trims_only_complete_oldest_turns():
    session_id = "session-trim"
    long_user = " ".join(["word"] * 520)
    snapshot = _snapshot(
        session_id,
        [("user", long_user), ("assistant", "old reply"),
         ("user", "keep this turn"), ("assistant", "keep reply")],
        ids=(51, 52, 53, 54),
    )
    adapter = _adapter_for(_Session(snapshot))

    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=0),
        (),
        "1",
    )

    assert context is not None
    assert len(context.prefix_token_ids) <= 512
    assert context.history_context[5] == (
        "included_history",
        (("user", "keep this turn"), ("assistant", "keep reply")),
    )


def test_history_revision_pending_then_published_and_session_isolation():
    first = _Session(_snapshot("session-first", [("user", "first only")], ids=(61,)))
    second = _Session(_snapshot("session-second", [("user", "second only")], ids=(71,)))
    adapter = SessionListenerCompactionAdapter(_engine(), unified_kv_enabled=False)
    adapter.register_session(first, generation_id=1)
    adapter.register_session(second, generation_id=1)

    updated = _snapshot(
        "session-first",
        [("user", "first revised")],
        ids=(61,),
        revision=2,
        sequence=1,
        instructions="New instructions.",
    )
    first._snapshot = updated
    assert adapter.mark_pending(first.session_id, updated)
    assert adapter.build_context(
        _listener_snapshot(first.session_id, next_frame_id=0), (), "2"
    ) is None
    assert adapter.publish_snapshot(first, updated)
    first_context = adapter.build_context(
        _listener_snapshot(first.session_id, next_frame_id=0), (), "2"
    )
    second_context = adapter.build_context(
        _listener_snapshot(second.session_id, next_frame_id=0), (), "1"
    )

    assert first_context is not None and second_context is not None
    assert first_context.history_revision == 2
    assert first_context.history_context[5] == (
        "included_history",
        (("user", "first revised"),),
    )
    assert second_context.history_context[5] == (
        "included_history",
        (("user", "second only"),),
    )


def test_message_identity_survives_history_index_shift():
    session_id = "session-pruned"
    snapshot = _snapshot(
        session_id,
        [("user", "retained transcript"), ("assistant", "reply")],
        ids=(82, 83),
        revision=2,
        sequence=2,
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=82,
                message_index=7,
                first=0,
                last=2,
                revision=1,
                sequence=1,
            ),
        ),
    )
    adapter = _adapter_for(_Session(snapshot))

    _classify(adapter, session_id, range(3), voiced=True)
    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=203),
        _ring(session_id, 3, 203),
        "2",
    )

    assert context is not None
    assert context.replay_after_frame_id == 2
    assert context.history_context[5] == (
        "included_history",
        (("user", "retained transcript"), ("assistant", "reply")),
    )


def test_non_cpu_or_incomplete_provenance_cannot_advance_cutoff():
    session_id = "session-no-cpu"
    snapshot = _snapshot(
        session_id,
        [("user", "unverified transcript"), ("assistant", "reply")],
        ids=(91, 92),
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=91,
                message_index=0,
                first=0,
                last=2,
                revision=1,
                sequence=1,
                asr=False,
            ),
        ),
    )
    adapter = _adapter_for(_Session(snapshot))

    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=3),
        _ring(session_id, 0, 3),
        "1",
    )

    assert context is not None
    assert context.replay_after_frame_id is None
    assert context.history_context[5] == (("included_history", (("assistant", "reply"),)))


def test_prior_epoch_transcript_remains_text_history_without_cutting_current_epoch():
    session_id = "session-new-epoch"
    snapshot = _snapshot(
        session_id,
        [("user", "prior epoch transcript"), ("assistant", "prior reply")],
        ids=(96, 97),
        provenance=(
            _record(
                session_id=session_id,
                archive_id=1,
                item_id=96,
                message_index=0,
                first=0,
                last=3,
                revision=1,
                sequence=1,
                epoch="epoch-old",
                timeline="timeline-old",
            ),
        ),
        timeline="timeline-current",
    )
    adapter = _adapter_for(_Session(snapshot))

    context = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=3, epoch="epoch-current"),
        _ring(session_id, 0, 3, epoch="epoch-current"),
        "1",
    )

    assert context is not None
    assert context.replay_after_frame_id is None
    assert context.history_context[5] == (
        "included_history",
        (("user", "prior epoch transcript"), ("assistant", "prior reply")),
    )


def test_long_input_compaction_uses_nonvoice_and_asr_across_physical_epochs_twice():
    session_id = "session-long-timeline"
    timeline = "timeline-long"
    snapshot_v1 = _snapshot(
        session_id,
        [
            ("user", "utterance one"),
            ("assistant", "reply one"),
            ("user", "utterance two"),
            ("assistant", "reply two"),
        ],
        ids=(111, 112, 113, 114),
        timeline=timeline,
        revision=2,
        sequence=2,
        provenance=(
            _record(
                session_id=session_id, archive_id=1, item_id=111, message_index=0,
                first=30, last=199, revision=1, sequence=1, epoch="epoch-a",
                timeline=timeline,
            ),
            _record(
                session_id=session_id, archive_id=2, item_id=113, message_index=2,
                first=5000, last=5160, revision=2, sequence=2, epoch="epoch-b",
                timeline=timeline,
            ),
        ),
    )
    session = _Session(snapshot_v1)
    adapter = _adapter_for(session)
    _classify(adapter, session_id, range(0, 30), voiced=False, timeline=timeline)
    _classify(adapter, session_id, range(30, 200), voiced=True, timeline=timeline)
    _classify(adapter, session_id, range(200, 5000), voiced=False, timeline=timeline)
    _classify(
        adapter, session_id, range(5000, 5161), voiced=True,
        epoch="epoch-b", timeline=timeline,
    )
    _classify(
        adapter, session_id, range(5161, 16385), voiced=False,
        epoch="epoch-b", timeline=timeline,
    )

    first = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=16385, epoch="epoch-b"),
        _ring(session_id, 16185, 16385, epoch="epoch-b"),
        "2",
    )
    assert first is not None
    assert first.replay_after_frame_id == 16184
    assert first.history_context[5] == (
        "included_history",
        (
            ("user", "utterance one"),
            ("assistant", "reply one"),
            ("user", "utterance two"),
            ("assistant", "reply two"),
        ),
    )

    # The second compaction follows another physical epoch change. A complete
    # ASR span reserved in epoch-b remains valid on the same logical timeline.
    _classify(
        adapter, session_id, range(16385, 16390), voiced=False,
        epoch="epoch-c", timeline=timeline,
    )
    _classify(
        adapter, session_id, range(16390, 16411), voiced=True,
        epoch="epoch-c", timeline=timeline,
    )
    _classify(
        adapter, session_id, range(16411, 16620), voiced=False,
        epoch="epoch-c", timeline=timeline,
    )
    snapshot_v2 = _snapshot(
        session_id,
        [
            ("user", "utterance one"),
            ("assistant", "reply one"),
            ("user", "utterance two"),
            ("assistant", "reply two"),
            ("user", "utterance three"),
            ("assistant", "reply three"),
        ],
        ids=(111, 112, 113, 114, 115, 116),
        timeline=timeline,
        revision=3,
        sequence=3,
        provenance=(
            *snapshot_v1.provenance,
            _record(
                session_id=session_id, archive_id=3, item_id=115, message_index=4,
                first=16390, last=16410, revision=3, sequence=3, epoch="epoch-b",
                timeline=timeline,
            ),
        ),
    )
    assert adapter.mark_pending(session_id, snapshot_v2)
    assert adapter.publish_snapshot(session, snapshot_v2)

    second = adapter.build_context(
        _listener_snapshot(session_id, next_frame_id=16620, epoch="epoch-c"),
        _ring(session_id, 16420, 16620, epoch="epoch-c"),
        "3",
    )
    assert second is not None
    assert second.replay_after_frame_id == 16419
    assert second.history_context[5][1][-2:] == (
        ("user", "utterance three"),
        ("assistant", "reply three"),
    )


def test_incomplete_last_turn_cannot_be_trimmed_to_fake_a_valid_prefix():
    session_id = "session-open-turn"
    snapshot = _snapshot(
        session_id,
        [("user", " ".join(["word"] * 520))],
        ids=(101,),
    )
    adapter = _adapter_for(_Session(snapshot))

    with pytest.raises(CompactionBuilderCapacityError) as error:
        adapter.build_context(
            _listener_snapshot(session_id, next_frame_id=0), (), "1"
        )

    assert error.value.reason == "confirmed_history_and_instructions_exceed_prefix_capacity"


def test_server_factory_enables_real_compaction_builder_on_worker_startup(monkeypatch):
    captured = {}

    class _Worker:
        runtime_signature = object()
        fatal_error = None
        owner_thread_id = 1

        def start(self):
            future = Future()
            future.set_result(None)
            return future

        def stop(self, **_kwargs):
            future = Future()
            future.set_result(None)
            return future

    class _Dispatcher:
        is_running = False
        fatal_error = None

        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            self.is_running = True

        def stop(self, **_kwargs):
            self.is_running = False

    def _make_worker(listener, **kwargs):
        captured["listener"] = listener
        captured.update(kwargs)
        return _Worker()

    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH", "1")
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "0")
    monkeypatch.setattr(server_module, "make_listener_microbatch_worker", _make_worker)
    monkeypatch.setattr(server_module, "SharedListenerTickDispatcher", _Dispatcher)
    engine = SimpleNamespace(
        listener=SimpleNamespace(backend=object()),
        enforce_production_listener=False,
        native_control6_semantic_prefix_enabled=False,
        enable_gemma_native_control6_policy=False,
        enable_listener_silence_skip=False,
    )

    app = server_module.create_app(engine)

    async def run_lifecycle():
        for handler in app.router.on_startup:
            await handler()
        try:
            builder = captured.get("compaction_request_builder")
            assert callable(builder)
            assert isinstance(builder.__self__, SessionListenerCompactionAdapter)
        finally:
            for handler in app.router.on_shutdown:
                await handler()

    asyncio.run(run_lifecycle())
