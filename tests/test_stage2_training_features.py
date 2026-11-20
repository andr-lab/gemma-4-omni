from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import importlib

import numpy as np
import pytest
import torch

from training.duplex_qlora import stage2
from training.duplex_qlora.features import native_control6_feature_contract
from training.duplex_qlora.schema import ACTIONS
from duplex_kernel.session import MemoryRevision, UserAudioSpan


def test_context_replays_real_audio_and_serializes_memory():
    span = UserAudioSpan(1, 0, 2, 3, 4, None, np.ones((4, 1536), np.float32), "Q2", True)
    memory = MemoryRevision("m1", None, (), (), "budget 80 euro", {"turn_id": 1})
    context = stage2._native_session_context(history=[span], instructions="Italiano", memory_revision=memory, turn_open=True)
    assert len(context.audio_chunks) == 2
    assert np.array_equal(context.audio_chunks[1].audio_soft, span.soft_tokens_reference[2:])
    assert context.turn_open is True
    assert "budget 80 euro" in context.instructions
    assert "<memory_data>" in context.instructions
    assert stage2._native_context_fingerprint(context) != stage2._native_context_fingerprint(stage2._native_session_context())


def test_audio_context_rejects_silent_loss_or_reordering():
    missing = UserAudioSpan(1, 0, 1, 0, 0, None, None, "Q2", False)
    with pytest.raises(ValueError, match="soft-token"):
        stage2._native_session_context(history=[missing])
    span = UserAudioSpan(1, 0, 1, 0, 0, None, np.ones((2, 1536), np.float32), "Q2", False)
    with pytest.raises(ValueError, match="interleav"):
        stage2._native_session_context(history=[span, {"role": "assistant", "content": "ciao"}])


def test_worker_without_mirror_uses_workspace_and_operates(monkeypatch, tmp_path):
    import duplex_kernel.listener_microbatch_serving as serving
    library = tmp_path / "listener.so"
    library.touch()
    def done(value=None):
        future = Future()
        future.set_result(value)
        return future
    backend = SimpleNamespace(_use_native_lm=True, mel_window_rows=240,
        _listener_microbatch_runtime_info={"stream_q_len": 2, "library_path": str(library)})
    worker = SimpleNamespace(start=lambda: done(), stop=lambda **kw: done(), runtime_signature=SimpleNamespace(),
        register_session=lambda *a, **kw: done(), close_session=lambda *a: done())
    monkeypatch.setattr(serving, "make_listener_microbatch_worker", lambda *a, **kw: worker)
    monkeypatch.setattr(serving, "initial_listener_session_state", lambda *a, **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(serving, "make_unified_prefix_import", lambda *a, **kw: object())
    original_import = importlib.import_module
    fake_server = SimpleNamespace(__file__=str(Path(stage2.__file__).parents[2] / "duplex_kernel/server.py"),
        _build_unified_listener_prefix_token_ids=lambda engine, session: (1, 10 + len(session.instructions), 2))
    monkeypatch.setattr(importlib, "import_module", lambda name: fake_server if name == "duplex_kernel.server" else original_import(name))
    monkeypatch.delenv("DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT", raising=False)
    listener = SimpleNamespace(backend=backend, runtime_profile="production")
    instance = stage2._ServingNativeControl6Worker(SimpleNamespace(tokenizer=object()), listener,
        workspace=Path(stage2.__file__).parents[2], native_library_path=library)
    first = instance.begin_conversation("a")
    instance.end_conversation()
    second = instance.begin_conversation("b", instructions="new context", turn_open=True)
    assert first["prefix_token_ids"] != second["prefix_token_ids"]
    instance.close()


def test_six_actions_and_q2_native1540_contract():
    assert ACTIONS == ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
    contract = native_control6_feature_contract(1540)
    assert contract["queries_per_tick"] == 2
    assert contract["tick_ms"] == 80
    assert contract["input_dim"] == 1536 + contract["phase_schema"]["width"]
    logits = torch.tensor([[1., 5., 2., 4., 3., 6.]])
    mask = stage2._control6_legal_mask(("LISTEN", "REPLAN"))
    assert logits.masked_fill(~mask, -torch.inf).argmax().item() == ACTIONS.index("REPLAN")
    assert mask.tolist() == [True, False, False, False, False, True]
    with pytest.raises(ValueError):
        stage2._control6_legal_mask(("HOLD",))


def test_audio_prefix_import_replays_payload_and_keeps_real_cursor():
    span = UserAudioSpan(1, 0, 1, 3, 4, None, np.arange(4 * 1536, dtype=np.float32).reshape(4, 1536), "Q2", True)
    chunks = stage2._native_session_context(history=[span]).audio_chunks
    received = []
    def build(ids, *, replay_q2, replay_frame_ids):
        received.extend(replay_q2)
        return dict(prefix_len=3, cache_position=7, replay_frame_ids=replay_frame_ids,
            past_key_values=object(), prefix_attention=torch.ones((1, 7)), prefix_build_ms=0., replay_ms=0.)
    worker = object.__new__(stage2._ServingNativeControl6Worker)
    worker.listener_backend = SimpleNamespace(device="cpu", build_semantic_prefix_cache=build)
    importer = worker._context_prefix_import((1, 9, 2), chunks)
    state = SimpleNamespace(session_id="s", next_frame_id=0, generation_id=1, state_epoch_id="e",
        sequence_position=0, cache_position=7)
    row = SimpleNamespace(session_id="s", frame_id=0, cache_position=7, session_generation=1,
        state_epoch_id="e", sequence_position=0, valid_history_tokens=7)
    imported = importer(state, row)
    assert imported.prefix_len == 3
    assert imported.cache_position == 7
    assert importer.requires_shared_lm_lock is True
    assert torch.equal(torch.cat(received), torch.from_numpy(span.soft_tokens_reference))
    assert importer(state, row) is imported


def test_old_empty_context_partial_is_rejected_for_a_new_context(tmp_path):
    conversation = {"conversation_id": "c1", "source_group_id": "g1"}
    targets = [{"frame_id": 0, "action_mask": True, "target_action": "LISTEN",
        "target_action_source": "annotation", "target_action_evidence": {"source": "fixture"}}]
    metadata = [{"conversation_id": "c1", "source_group_id": "g1", "frame_id": 0,
        "listener_frame_id": 0, "action": "LISTEN", "action_mask": True,
        "target_action_mask": True, "target_action_source": "annotation",
        "target_action_evidence": {"source": "fixture"},
        "native_control6_hidden_source": "ListenerOutput.gemma_hidden_t",
        "native_control6_audio_queries_per_tick": 2, "native_control6_mel_window_rows": 240}]
    partial = stage2._conversation_partial_path(tmp_path, "train", "c1")
    stage2._write_features(partial, np.zeros((1, 1536), np.float32), metadata)
    assert stage2._load_conversation_partial(partial, conversation=conversation,
        targets=targets, control6_profile="native1540") is not None
    conversation["instructions"] = "new memory-conditioned run"
    assert stage2._load_conversation_partial(partial, conversation=conversation,
        targets=targets, control6_profile="native1540") is None


def test_context_feature_profile_allows_conversation_specific_prefixes():
    profile = "compact_memory_recent_q2_open_turn_v1"
    base = {"context_profile": profile, "prefix": {"contract": "serving_unified_chat_prefix_v2",
        "token_ids": [1, 2], "length": 2}}
    supplement = {"context_profile": profile, "prefix": {"contract": "serving_unified_chat_prefix_v2",
        "token_ids": [1, 3, 2], "length": 3}}
    assert stage2._native1540_prefix_signature(base) == stage2._native1540_prefix_signature(supplement)
    legacy = {"prefix": {"contract": "serving_unified_chat_prefix_v2", "token_ids": [1, 3, 2], "length": 3}}
    assert stage2._native1540_prefix_signature(base) != stage2._native1540_prefix_signature(legacy)


def test_false_events_exclude_valid_speech_interiors():
    rows = [{"profile": "audio_grounded_v5", "event_onset": False} for _ in range(4)]
    result = stage2._v5_event_onset_selection_metrics(np.array([0, 2, 1, 4]), np.array([1, 2, 1, 4]), rows)
    assert result["background_examples"] == 2
    assert result["background_false_event_rate"] == .5


def test_latency_is_bounded_by_contiguous_event_and_conversation():
    targets = np.array([1, 1, 1, 3, 3, 1])
    predictions = np.array([0, 1, 1, 2, 3, 0])
    rows = [dict(conversation_id="a", source_group_id="g", target_frame_id=i, event_onset=i in (0, 3)) for i in range(5)]
    rows.append(dict(conversation_id="b", source_group_id="h", target_frame_id=0, event_onset=True))
    result = stage2._control6_temporal_metrics(targets, predictions, rows)
    assert result["transitions"]["onsets"] == 3
    assert result["transitions"]["detected"] == 2
    assert result["transitions"]["missed"] == 1
    assert result["transitions"]["mean_latency_ms"] == 80
    rows[1]["target_frame_id"] = 7
    assert stage2._control6_temporal_metrics(targets, predictions, rows)["transitions"]["detected"] == 1


def test_acoustic_metrics_require_explicit_echo_and_count_double_talk():
    rows = [dict(conversation_id="a", target_frame_id=i, profile="audio_grounded_v5", event_onset=False) for i in range(3)]
    rows[0]["echo_event"] = True
    rows[1].update(speaker_active_at_frame_start=True, peer_active_during_frame=True,
        native_control6_context={"source_kind": "verified_role_intervals"})
    result = stage2._control6_temporal_metrics(np.array([0, 3, 2]), np.array([1, 3, 2]), rows)
    assert result["echo"]["examples"] == 1
    assert result["echo"]["background_false_event_rate"] == 1
    assert result["double_talk"]["accuracy"] == 1
    assert result["double_talk"]["examples"] == 1
    absent = stage2._control6_temporal_metrics(np.array([0]), np.array([0]), [{}])
    assert absent["echo"]["examples"] == 0
    assert absent["echo"]["accuracy"] is None


@pytest.mark.parametrize("field", ["source_group_id", "conversation_id"])
def test_split_leakage_is_blocked_independently(field):
    first = {"source_group_id": "g1", "conversation_id": "c1"}
    second = {"source_group_id": "g2", "conversation_id": "c2", field: first[field]}
    with pytest.raises(ValueError, match="leakage"):
        stage2._validate_control6_split_disjointness({"train": [first], "test": [second]})
