from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
import torch

from training.duplex_qlora.stage2 import (
    _audio_frames,
    _native1540_supervision_context,
    _native1540_runtime_target_action,
    _native1540_pre_action_speaker_state,
    _attest_merged_model_source,
    _causal_action_targets,
    _clustered_target_context_workload,
    _conversation_partial_path,
    _load_conversation_partial,
    _stage2_extraction_splits,
    benchmark_serving_native_worker_frames,
    benchmark_native_listener_windows,
    extract_conversation_features,
    extract_stage2_features,
    load_serving_base,
    _write_features,
)
from training.duplex_qlora.cli import build_parser


def _conversation(audio_path: Path, sample_rate: int, channel_map: dict[str, str]):
    return {
        "conversation_id": "stage2-audio-test",
        "source_group_id": "stage2-audio-group",
        "source": "SmoothConv",
        "audio_path": str(audio_path),
        "sample_rate_hz": sample_rate,
        "duration_ms": 130,
        "tick_ms": 80,
        "channel_map": channel_map,
    }


def test_native1540_context_uses_verified_intervals_and_synthetic_trace_fields():
    natural = {
        "conversation_id": "natural-context",
        "source": "DuplexConv",
        "channel_map": {"0": "assistant", "1": "user"},
        "segments": [
            {
                "role": "assistant", "channel_index": 0, "start_ms": 80, "end_ms": 280,
                "audio_channel_verified": True, "time_mask": True,
            },
            {
                "role": "user", "channel_index": 1, "start_ms": 190, "end_ms": 250,
                "audio_channel_verified": True, "time_mask": True,
            },
        ],
    }
    natural_context = _native1540_supervision_context(
        natural,
        {},
        frame_id=2,
        user_pcm_current=np.zeros(1280, dtype=np.float32),
        assistant_pcm_prior=np.full(1280, 0.01, dtype=np.float32),
        assistant_prior_frame_id=1,
    )
    assert natural_context["speaker_active_at_frame_start"] is True
    assert natural_context["peer_active_during_frame"] is True
    assert natural_context["native_control6_user_activity_detected"] is True
    assert natural_context["native_control6_context"]["source_kind"] == "verified_role_intervals"
    assert natural_context["training_state_context"] == "user_stream_q2_only"
    assert natural_context["native_control6_context"]["training_state_context"] == "user_stream_q2_only"

    synthetic = {"conversation_id": "synthetic-context", "synthetic": True}
    synthetic_context = _native1540_supervision_context(
        synthetic,
        {"own_active": True, "assistant_audible": False, "acoustic_state": "quiet"},
        frame_id=2,
        user_pcm_current=np.full(1280, 0.01, dtype=np.float32),
        assistant_pcm_prior=np.full(1280, 0.02, dtype=np.float32),
        assistant_prior_frame_id=1,
    )
    assert synthetic_context["speaker_active_at_frame_start"] is True
    assert synthetic_context["peer_active_during_frame"] is False
    assert synthetic_context["native_control6_user_activity_detected"] is True
    assert synthetic_context["native_control6_context"]["speaker_state_source"] == "frame_target_own_active"
    assert synthetic_context["training_state_context"] == "user_stream_q2_only"
    assert synthetic_context["native_control6_user_assistant_ratio_valid"] is True


def test_native1540_synthetic_prestate_and_runtime_targets_follow_causal_trace():
    frames = [
        {
            "input_frame_id": 0,
            "target_frame_id": 1,
            "assistant_audible": False,
            "action": "LISTEN",
            "target_action_mask": True,
        },
        {
            "input_frame_id": 1,
            "target_frame_id": 2,
            "assistant_audible": True,
            "action": "SPEAK",
            "target_action_mask": True,
            "target_action_source": "synthetic_action_segment",
            "target_action_evidence": {"event_id": "speak-1"},
        },
        {
            "input_frame_id": 2,
            "target_frame_id": 3,
            "assistant_audible": True,
            "action": "SPEAK",
            "target_action_mask": True,
            "target_action_source": "synthetic_action_segment",
            "target_action_evidence": {"event_id": "speak-1"},
        },
        {
            "input_frame_id": 3,
            "target_frame_id": 4,
            "assistant_audible": False,
            "action": "YIELD",
            "target_action_mask": True,
            "target_action_source": "synthetic_action_segment",
            "target_action_evidence": {"event_id": "yield-1"},
        },
        {
            "input_frame_id": 4,
            "target_frame_id": 5,
            "assistant_audible": True,
            "action": "BACKCHANNEL",
            "target_action_mask": True,
            "target_action_source": "synthetic_action_segment",
            "target_action_evidence": {"event_id": "bc-1"},
        },
        {
            "input_frame_id": 5,
            "target_frame_id": 6,
            "assistant_audible": True,
            "action": "BACKCHANNEL",
            "target_action_mask": True,
            "target_action_source": "synthetic_action_segment",
            "target_action_evidence": {"event_id": "bc-1"},
        },
    ]
    conversation = {
        "conversation_id": "synthetic-causal-state",
        "source": "authored_synthetic_audio_candidate",
        "frame_targets": frames,
        "action_segments": [
            {
                "action": "SPEAK", "event_id": "speak-1",
                "target_frame_id_start": 2, "target_frame_id_end": 4,
                "input_frame_id_start": 1, "input_frame_id_end": 3,
            },
            {
                "action": "YIELD", "event_id": "yield-1",
                "target_frame_id_start": 4, "target_frame_id_end": 5,
                "input_frame_id_start": 3, "input_frame_id_end": 4,
            },
            {
                "action": "BACKCHANNEL", "event_id": "bc-1",
                "target_frame_id_start": 5, "target_frame_id_end": 7,
                "input_frame_id_start": 4, "input_frame_id_end": 6,
            },
        ],
    }

    speak_onset = frames[1]
    active, source = _native1540_pre_action_speaker_state(
        conversation, speak_onset, frame_id=1
    )
    assert active is False
    assert source == "prior_target_assistant_audible_t_minus_1"
    onset_target = _native1540_runtime_target_action(
        conversation, speak_onset, frame_id=1, speaker_active=active
    )
    assert onset_target["action"] == "SPEAK"
    assert onset_target["derived"] is False

    speak_continuation = frames[2]
    active, source = _native1540_pre_action_speaker_state(
        conversation, speak_continuation, frame_id=2
    )
    assert active is True
    assert source == "prior_target_assistant_audible_t_minus_1"
    continuation_target = _native1540_runtime_target_action(
        conversation, speak_continuation, frame_id=2, speaker_active=active
    )
    assert continuation_target["action"] == "CONTINUE"
    assert continuation_target["derived"] is True
    assert continuation_target["reason"] == "same_active_speak_event_continues"
    assert continuation_target["event_id"] == "speak-1"
    assert speak_continuation["action"] == "SPEAK"
    assert speak_continuation["target_action_source"] == "synthetic_action_segment"

    yield_target = frames[3]
    active, _ = _native1540_pre_action_speaker_state(conversation, yield_target, frame_id=3)
    assert active is True
    assert yield_target["assistant_audible"] is False
    assert _native1540_runtime_target_action(
        conversation, yield_target, frame_id=3, speaker_active=active
    )["action"] == "YIELD"

    backchannel_onset = frames[4]
    active, _ = _native1540_pre_action_speaker_state(
        conversation, backchannel_onset, frame_id=4
    )
    assert active is False
    assert _native1540_runtime_target_action(
        conversation, backchannel_onset, frame_id=4, speaker_active=active
    )["action"] == "BACKCHANNEL"
    backchannel_continuation = frames[5]
    active, _ = _native1540_pre_action_speaker_state(
        conversation, backchannel_continuation, frame_id=5
    )
    assert active is True
    assert _native1540_runtime_target_action(
        conversation, backchannel_continuation, frame_id=5, speaker_active=active
    )["action"] == "CONTINUE"


def test_native1540_synthetic_prestate_uses_t0_idle_and_explicit_span_on_gap():
    first = {
        "input_frame_id": 0,
        "target_frame_id": 1,
        "assistant_audible": True,
        "own_active": True,
        "action": "SPEAK",
        "target_action_mask": True,
    }
    conversation = {
        "conversation_id": "synthetic-causal-boundaries",
        "source": "authored_synthetic_audio_candidate",
        "frame_targets": [first],
    }
    assert _native1540_pre_action_speaker_state(conversation, first, frame_id=0) == (
        True, "frame_target_own_active"
    )
    no_explicit_prestate = {**first}
    no_explicit_prestate.pop("own_active")
    assert _native1540_pre_action_speaker_state(
        conversation, no_explicit_prestate, frame_id=0
    ) == (False, "synthetic_stream_initial_idle")

    gap_frame = {
        "input_frame_id": 2,
        "target_frame_id": 3,
        "assistant_audible": True,
        "action": "CONTINUE",
        "target_action_mask": True,
    }
    assert _native1540_pre_action_speaker_state(
        {**conversation, "frame_targets": [first, gap_frame]}, gap_frame, frame_id=2
    ) == (None, "synthetic_pre_action_state_unavailable")

    explicit_span = {
        **conversation,
        "frame_targets": [first, gap_frame],
        "action_segments": [{
            "action": "CONTINUE",
            "target_frame_id_start": 2,
            "target_frame_id_end": 4,
            "target_action_mask": True,
        }],
    }
    assert _native1540_pre_action_speaker_state(
        explicit_span, gap_frame, frame_id=2
    ) == (True, "verified_action_segment_covering_current_audio_frame")


def test_native1540_exact_v5_trace_segment_bounds_map_onset_and_held_frames():
    def target(frame_id: int, action: str, assistant_audible: bool, mask: bool = False):
        return {
            "frame_id": frame_id,
            "input_frame_id": frame_id,
            "target_frame_id": frame_id + 1,
            "input_audio_start_sample": frame_id * 1280,
            "input_audio_end_sample": (frame_id + 1) * 1280,
            "target_action_start_sample": (frame_id + 1) * 1280,
            "target_action_end_sample": (frame_id + 2) * 1280,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": action,
            "target_action_mask": mask,
            "assistant_audible": assistant_audible,
        }

    # Bounds and keys match the prepared V5 trace: an onset SPEAK span followed
    # by two held SPEAK targets, and one complete BACKCHANNEL span.
    speak_rows = [
        target(91, "LISTEN", False),
        target(92, "SPEAK", True),
        target(93, "SPEAK", True),
        target(94, "SPEAK", True),
    ]
    backchannel_rows = [
        target(195, "LISTEN", False),
        *[target(frame_id, "BACKCHANNEL", True) for frame_id in range(196, 203)],
        target(203, "BACKCHANNEL", False),
    ]
    conversation = {
        "conversation_id": "verified-v5-trace-segment-geometry",
        "source": "authored_synthetic_audio_candidate",
        "frame_targets": speak_rows + backchannel_rows,
        "action_segments": [
            {
                "action": "SPEAK", "start_ms": 7440, "end_ms": 7680,
                "input_frame_id_start": 92, "input_frame_id_end": 95,
                "target_frame_id_start": 93, "target_frame_id_end": 96,
                "target_action_mask": True,
            },
            {
                "action": "BACKCHANNEL", "start_ms": 15760, "end_ms": 16320,
                "input_frame_id_start": 196, "input_frame_id_end": 203,
                "target_frame_id_start": 197, "target_frame_id_end": 204,
                "target_action_mask": True,
            },
        ],
    }

    speak_targets = []
    for frame in speak_rows[1:]:
        state, _ = _native1540_pre_action_speaker_state(
            conversation, frame, frame_id=frame["input_frame_id"]
        )
        speak_targets.append(_native1540_runtime_target_action(
            conversation, frame, frame_id=frame["input_frame_id"], speaker_active=state
        )["action"])
    assert speak_targets == ["SPEAK", "CONTINUE", "CONTINUE"]

    bc_targets = []
    for frame in backchannel_rows[1:]:
        state, _ = _native1540_pre_action_speaker_state(
            conversation, frame, frame_id=frame["input_frame_id"]
        )
        bc_targets.append(_native1540_runtime_target_action(
            conversation, frame, frame_id=frame["input_frame_id"], speaker_active=state
        )["action"])
    assert bc_targets == ["BACKCHANNEL", *(["CONTINUE"] * 6), "BACKCHANNEL"]


def test_native1540_qualified_yield_continuations_remain_yield_until_local_stop():
    local_id = "track-0-event-9"
    peer_id = "track-1-event-12"
    onset = {
        "input_frame_id": 1073,
        "target_frame_id": 1074,
        "action": "YIELD",
        "target_action_mask": True,
        "target_action_source": "timestamp_derived_peer_onset_confirmed_local_stop",
        "target_action_evidence": {
            "local_segment_id": local_id,
            "peer_segment_id": peer_id,
            "peer_onset_ms": 85840.0,
            "local_stop_ms": 86100.0,
            "peer_end_ms": 86400.0,
        },
    }
    held = [
        {
            "input_frame_id": frame_id,
            "target_frame_id": frame_id + 1,
            "action": "CONTINUE",
            "target_action_mask": True,
            "target_action_source": "timestamp_derived_local_speech_active",
            "target_action_evidence": {"local_segment_id": local_id, "tick_start_ms": frame_id * 80},
            "speaker_active_at_frame_start": True,
            "peer_active_during_frame": True,
        }
        for frame_id in (1074, 1075, 1076)
    ]
    conversation = {
        "conversation_id": "DuplexConv:Edu--003466:local-channel-0",
        "source": "DuplexConv",
        "frame_targets": [onset, *held],
    }

    mapped = [
        _native1540_runtime_target_action(
            conversation, frame, frame_id=frame["input_frame_id"], speaker_active=True
        )
        for frame in held
    ]
    assert [row["action"] for row in mapped] == ["YIELD", "YIELD", "YIELD"]
    assert all(row["derived"] for row in mapped)
    assert all(row["reason"] == "qualified_yield_overlap_persistence_until_local_stop" for row in mapped)
    assert [row["yield_onset_frame_id"] for row in mapped] == [1073, 1073, 1073]
    assert [row["yield_local_stop_ms"] for row in mapped] == [86100.0] * 3
    assert [row["event_id"] for row in mapped] == [peer_id] * 3
    assert [frame["action"] for frame in held] == ["CONTINUE"] * 3

    after_stop = {
        **held[-1],
        "input_frame_id": 1077,
        "target_frame_id": 1078,
        "speaker_active_at_frame_start": False,
        "target_action_evidence": {"local_segment_id": local_id, "tick_start_ms": 1077 * 80},
    }
    assert _native1540_runtime_target_action(
        conversation, after_stop, frame_id=1077, speaker_active=False
    )["action"] == "CONTINUE"

    explicit_continue = {
        **held[0],
        "target_action_source": "verified_native_assistant_action",
    }
    assert _native1540_runtime_target_action(
        conversation, explicit_continue, frame_id=1074, speaker_active=True
    )["action"] == "CONTINUE"


def test_48khz_resampling_filters_alias_and_preserves_partial_frame_mask():
    rate = 48000
    count = int(rate * 0.13)
    time = np.arange(count, dtype=np.float64) / rate
    user_high = 0.5 * np.sin(2 * np.pi * 12000 * time)
    assistant_low = 0.5 * np.sin(2 * np.pi * 1000 * time)
    audio = np.column_stack([user_high, assistant_low]).astype(np.float32)
    with TemporaryDirectory(prefix="stage2-resample-48k-") as directory:
        path = Path(directory) / "two-channel-48k.wav"
        sf.write(path, audio, rate, subtype="FLOAT")
        user, assistant, valid_samples = _audio_frames(
            _conversation(path, rate, {"0": "user", "1": "assistant"}),
            frame_samples=1280,
        )

    assert user.shape == assistant.shape == (2, 1280)
    assert valid_samples.tolist() == [1280, 800]
    assert np.sqrt(np.mean(user[0] ** 2)) < 0.02
    assert np.sqrt(np.mean(assistant[0] ** 2)) > 0.3
    assert np.count_nonzero(user[1, 800:]) == 0
    assert np.count_nonzero(assistant[1, 800:]) == 0


def test_24khz_resampling_preserves_duration_and_marks_final_partial_frame():
    rate = 24000
    count = int(rate * 0.17)
    time = np.arange(count, dtype=np.float64) / rate
    signal = (0.4 * np.sin(2 * np.pi * 1000 * time)).astype(np.float32)
    with TemporaryDirectory(prefix="stage2-resample-24k-") as directory:
        path = Path(directory) / "mono-24k.wav"
        sf.write(path, signal, rate, subtype="FLOAT")
        user, assistant, valid_samples = _audio_frames(
            _conversation(path, rate, {"0": "user"}), frame_samples=1280
        )

    assert user.shape == (3, 1280)
    assert assistant is None
    assert valid_samples.tolist() == [1280, 1280, 160]
    assert np.count_nonzero(user[2, 160:]) == 0
    assert 0.25 < np.sqrt(np.mean(user[0] ** 2)) < 0.35


def test_sparse_targets_still_advance_listener_over_every_causal_frame():
    rate = 16000
    signal = (np.ones(3200, dtype=np.float32) * 0.1)[:, None]
    with TemporaryDirectory(prefix="stage2-sparse-frames-") as directory:
        path = Path(directory) / "mono-16k.wav"
        sf.write(path, signal, rate, subtype="FLOAT")
        conversation = _conversation(path, rate, {"0": "user"})
        conversation["frames"] = [
            {"frame_id": 0, "target_action": "LISTEN", "action_mask": True},
            {"frame_id": 2, "target_action": "CONTINUE", "action_mask": True},
        ]

        class _State:
            def __init__(self, **_kwargs):
                self.frames_seen = 0

            def close(self):
                pass

        class _Listener:
            runtime_profile = "production"

            def __init__(self):
                self.backend = SimpleNamespace(_use_native_lm=True)
                self.calls = 0

            def prime_steady_state(self, _state):
                pass

            def step(self, _pcm, _reference, _state):
                frame_id = _state.frames_seen
                _state.frames_seen += 1
                self.calls += 1
                queries = torch.full((4, 1536), float(frame_id))
                return SimpleNamespace(frame_id=frame_id, gemma_audio_queries_t=queries)

        listener = _Listener()
        with patch("duplex_kernel.listener.GemmaListenerState", _State):
            features, metadata = extract_conversation_features(listener, conversation)

    assert listener.calls == 3
    assert features.shape == (2, 1536)
    assert [row["frame_id"] for row in metadata] == [0, 2]
    assert [row["listener_frame_id"] for row in metadata] == [0, 2]
    assert [row["valid_sample_count"] for row in metadata] == [1280, 640]
    assert [row["action"] for row in metadata] == ["LISTEN", "CONTINUE"]


def test_native1540_extraction_uses_runtime_hidden_and_requires_q2_w240(tmp_path):
    signal = np.column_stack((
        np.full(1280 * 3, 0.1, dtype=np.float32),
        np.full(1280 * 3, 0.2, dtype=np.float32),
    ))
    audio_path = tmp_path / "native-q2.wav"
    sf.write(audio_path, signal, 16000, subtype="FLOAT")
    conversation = _conversation(audio_path, 16000, {"0": "user", "1": "assistant"})
    conversation["duration_ms"] = 240
    conversation["frames"] = [{
        "frame_id": 1,
        "target_action": "YIELD",
        "action_mask": True,
        "target_action_source": "timestamp_derived_peer_onset_confirmed_local_stop",
        "target_action_evidence": {"peer_onset_ms": 80.0},
    }]

    class _State:
        def __init__(self, **_kwargs):
            self.frames_seen = 0

        def close(self):
            pass

    class _Listener:
        runtime_profile = "production"

        def __init__(self, mel_rows=240, query_count=2):
            self.backend = SimpleNamespace(_use_native_lm=True, mel_window_rows=mel_rows)
            self.query_count = query_count

    class _ServingWorker:
        def __init__(self, query_count=2):
            self.query_count = query_count
            self.steps = []

        def begin_conversation(self, conversation_id):
            return {"session_id": f"fake-serving:{conversation_id}"}

        def step(self, frame_id, user_pcm, assistant_pcm, assistant_frame_id):
            self.steps.append((frame_id, assistant_frame_id, assistant_pcm))
            current_value = 10.0 + float(np.mean(user_pcm))
            queries = torch.stack([
                torch.full((1536,), current_value),
                torch.full((1536,), 100.0 + current_value),
            ])[:self.query_count]
            hidden = queries[0:1].clone()
            tower = torch.ones((self.query_count, 1536))
            return SimpleNamespace(
                frame_id=frame_id,
                gemma_hidden_t=hidden,
                gemma_audio_queries_t=queries,
                audio_tower_hidden=tower,
            )

        def end_conversation(self):
            pass

    with patch("duplex_kernel.listener.GemmaListenerState", _State):
        worker = _ServingWorker()
        features, metadata = extract_conversation_features(
            _Listener(), conversation, control6_profile="native1540", serving_native_worker=worker
        )
        np.testing.assert_array_equal(features, np.full((1, 1536), 10.1, dtype=np.float32))
        assert [step[0] for step in worker.steps] == [0, 1]
        assert [step[1] for step in worker.steps] == [None, 0]
        assert worker.steps[1][2].shape == (1280,)
        assert metadata[0]["native_control6_hidden_source"] == "ListenerOutput.gemma_hidden_t"
        assert metadata[0]["native_control6_audio_queries_per_tick"] == 2
        assert metadata[0]["native_control6_mel_window_rows"] == 240
        assert metadata[0]["target_action_source"] == "timestamp_derived_peer_onset_confirmed_local_stop"
        assert metadata[0]["target_action_evidence"] == {"peer_onset_ms": 80.0}

        future_changed = signal.copy()
        future_changed[2 * 1280:] = 0.8
        future_audio_path = tmp_path / "native-q2-future-changed.wav"
        sf.write(future_audio_path, future_changed, 16000, subtype="FLOAT")
        future_conversation = dict(conversation, audio_path=str(future_audio_path))
        future_features, _ = extract_conversation_features(
            _Listener(), future_conversation, control6_profile="native1540", serving_native_worker=_ServingWorker()
        )
        np.testing.assert_array_equal(future_features, features)

        with pytest.raises(RuntimeError, match="Q2/W240"):
            extract_conversation_features(
                _Listener(mel_rows=239), conversation, control6_profile="native1540", serving_native_worker=_ServingWorker()
            )
        with pytest.raises(RuntimeError, match="effective Q2 tensors"):
            extract_conversation_features(
                _Listener(query_count=4), conversation, control6_profile="native1540", serving_native_worker=_ServingWorker(query_count=4)
            )


def test_native1540_loads_only_the_serving_base_and_requires_resolved_path(tmp_path):
    base = tmp_path / "serving-base"
    base.mkdir()
    model = torch.nn.Linear(2, 2, bias=False).to(dtype=torch.bfloat16)
    model._gemma4_checkpoint_path = str(base.resolve())
    thinker = SimpleNamespace(model=model, processor=object())
    with (
        patch("duplex_kernel.models.thinker.resolve_thinker_path", return_value=str(base)),
        patch("duplex_kernel.models.thinker.load_thinker", return_value=thinker) as load,
    ):
        loaded_thinker, processor, loaded_model, report = load_serving_base(
            workspace=tmp_path,
            base_model_path=str(base),
            device="cpu",
        )
        assert loaded_thinker is thinker
        assert processor is thinker.processor
        assert loaded_model is model
        assert report["response_lora_applied"] is False
        assert report["stage1_adapter_applied"] is False
        load.assert_called_once()
        with pytest.raises(ValueError, match="must match the serving resolve_thinker_path"):
            load_serving_base(workspace=tmp_path, base_model_path=str(tmp_path / "other"), device="cpu")
        load.assert_called_once()


def test_native1540_cli_omits_adapter_and_selects_runtime_workspace():
    args = build_parser().parse_args([
        "extract-control",
        "--prepared-dir", "prepared",
        "--output-dir", "features",
        "--control6-profile", "native1540",
    ])
    assert args.adapter_dir is None
    assert args.base_model_path is None
    assert args.runtime_workspace == Path("/workspace")
    assert args.native_microbatch_library_path is None

    augment_args = build_parser().parse_args([
        "augment-control-phase",
        "--native-feature-dir", "native-features",
        "--prepared-dir", "prepared",
        "--output-dir", "features1540",
        "--phase-profile", "native1540",
        "--no-mask-train-natural-backchannels-without-overlap",
    ])
    assert augment_args.phase_profile == "native1540"
    assert augment_args.mask_train_natural_backchannels_without_overlap is False
    assert _stage2_extraction_splits("native1540", None) == ("train", "validation")
    assert _stage2_extraction_splits("historical_q4", None) == ("train", "validation", "test")
    assert _stage2_extraction_splits("native1540", ("test",)) == ("test",)
    assert _stage2_extraction_splits("native1540", ("train", "validation", "test")) == (
        "train", "validation", "test"
    )


def test_native1540_benchmark_smoke_is_bounded_to_real_pcm_frames(tmp_path):
    audio_path = tmp_path / "native-smoke.wav"
    stereo = np.column_stack((
        np.full(1280 * 5, 0.1, dtype=np.float32),
        np.full(1280 * 5, 0.2, dtype=np.float32),
    ))
    sf.write(audio_path, stereo, 16000, subtype="FLOAT")
    conversation = _conversation(audio_path, 16000, {"0": "user", "1": "assistant"})

    class _Worker:
        def __init__(self):
            self.calls = []
            self.closed = 0

        def begin_conversation(self, _conversation_id):
            return {"session_id": "smoke-session"}

        def step(self, frame_id, user_pcm, assistant_pcm, assistant_frame_id):
            self.calls.append((frame_id, assistant_frame_id, assistant_pcm))
            hidden = torch.full((1, 1536), 1.0 + frame_id)
            return SimpleNamespace(
                frame_id=frame_id,
                gemma_hidden_t=hidden,
                gemma_audio_queries_t=torch.cat((hidden, hidden + 1.0), dim=0),
                audio_tower_hidden=torch.ones((2, 1536)),
            )

        def end_conversation(self):
            self.closed += 1

        def report(self):
            return {"prefix": {"length": 4}}

    worker = _Worker()
    report = benchmark_serving_native_worker_frames(worker, conversation, frames=4)
    assert report["status"] == "passed"
    assert report["input_frames"] == 4
    assert report["first_frame_includes_prefix_import"] is True
    assert report["assistant_reference_frame_ids"] == [None, 0, 1, 2]
    assert [call[0] for call in worker.calls] == [0, 1, 2, 3]
    assert worker.calls[1][2].shape == (1280,)
    assert report["steady_frame_count"] == 3
    assert all(shape["gemma_audio_queries_t"] == [2, 1536] for shape in report["output_shapes"])
    assert worker.closed == 1
    with pytest.raises(ValueError, match="between three and eight"):
        benchmark_serving_native_worker_frames(worker, conversation, frames=9)


def test_v5_listener_features_at_t_predict_target_t_plus_one_and_use_assistant_t_minus_one():
    rate = 16000
    signal = np.column_stack([
        np.full(1280 * 3, 0.1, dtype=np.float32),
        np.full(1280 * 3, 0.2, dtype=np.float32),
    ])
    with TemporaryDirectory(prefix="stage2-v5-offset-") as directory:
        path = Path(directory) / "stereo-16k.wav"
        sf.write(path, signal, rate, subtype="PCM_16")
        conversation = _conversation(path, rate, {"0": "user", "1": "assistant"})
        conversation["frames"] = []
        conversation["frame_targets"] = [{
            "frame_id": 1,
            "input_frame_id": 1,
            "target_frame_id": 2,
            "input_audio_start_sample": 1280,
            "input_audio_end_sample": 2560,
            "target_action_start_sample": 2560,
            "target_action_end_sample": 3840,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": "BACKCHANNEL",
            "target_action_mask": True,
        }]
        conversation["event_schedule"] = [{"action": "BACKCHANNEL", "start_sample": 2560}]

        class _State:
            def __init__(self, **_kwargs):
                self.frames_seen = 0

            def close(self):
                pass

        class _Listener:
            runtime_profile = "production"

            def __init__(self):
                self.backend = SimpleNamespace(_use_native_lm=True)
                self.references = []

            def prime_steady_state(self, _state):
                pass

            def step(self, _pcm, reference, state):
                frame_id = state.frames_seen
                state.frames_seen += 1
                self.references.append(None if reference is None else reference.frame_id)
                return SimpleNamespace(
                    frame_id=frame_id,
                    gemma_audio_queries_t=torch.full((4, 1536), float(frame_id)),
                )

        listener = _Listener()
        with patch("duplex_kernel.listener.GemmaListenerState", _State):
            features, metadata = extract_conversation_features(listener, conversation)

    assert listener.references[1] == 0
    assert features.shape == (1, 1536)
    assert metadata[0]["frame_id"] == metadata[0]["input_frame_id"] == 1
    assert metadata[0]["listener_frame_id"] == 1
    assert metadata[0]["target_frame_id"] == 2
    assert metadata[0]["assistant_reference_frame_id"] == 0
    assert metadata[0]["action"] == "BACKCHANNEL"
    assert metadata[0]["event_onset"] is True


def test_v5_listener_targets_reject_future_or_misaligned_frame_contract():
    conversation = _conversation(Path("unused.wav"), 16000, {"0": "user", "1": "assistant"})
    conversation["frame_targets"] = [{
        "input_frame_id": 1,
        "target_frame_id": 3,
        "input_audio_start_sample": 1280,
        "input_audio_end_sample": 2560,
        "target_action_start_sample": 2560,
        "target_action_end_sample": 3840,
        "input_text_visible": False,
        "future_audio_excluded": True,
        "action": "REPLAN",
        "target_action_mask": True,
    }]
    with pytest.raises(ValueError, match="target_frame_id"):
        extract_conversation_features(
            SimpleNamespace(runtime_profile="production", backend=SimpleNamespace(_use_native_lm=True)),
            conversation,
        )


def test_v5_targets_mark_explicit_event_schedule_onset_frame():
    conversation = _conversation(Path("unused.wav"), 16000, {"0": "user", "1": "assistant"})
    actions = ["LISTEN", "CONTINUE", "CONTINUE", "SPEAK"]
    conversation["frame_targets"] = [
        {
            "input_frame_id": frame_id,
            "target_frame_id": frame_id + 1,
            "input_audio_start_sample": frame_id * 1280,
            "input_audio_end_sample": (frame_id + 1) * 1280,
            "target_action_start_sample": (frame_id + 1) * 1280,
            "target_action_end_sample": (frame_id + 2) * 1280,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": action,
            "target_action_mask": True,
        }
        for frame_id, action in enumerate(actions)
    ]
    conversation["event_schedule"] = [
        {"action": "CONTINUE", "start_sample": 2560},
        {"action": "SPEAK", "start_sample": 5120},
    ]

    targets, is_v5 = _causal_action_targets(conversation)

    assert is_v5 is True
    assert [frame["event_onset"] for frame in targets] == [False, True, False, True]


def test_sparse_targets_replay_one_full_causal_conversation_state():
    rate = 16000
    signal = (np.ones(1280 * 194, dtype=np.float32) * 0.1)[:, None]
    with TemporaryDirectory(prefix="stage2-clustered-frames-") as directory:
        path = Path(directory) / "mono-16k.wav"
        sf.write(path, signal, rate, subtype="FLOAT")
        conversation = _conversation(path, rate, {"0": "user"})
        target_frame_ids = [63, 64, 128, 129, 193]
        conversation["frames"] = [
            {"frame_id": frame_id, "target_action": "LISTEN", "action_mask": True}
            for frame_id in target_frame_ids
        ]

        class _State:
            constructed = 0

            def __init__(self, **_kwargs):
                self.frames_seen = 0
                _State.constructed += 1

            def close(self):
                pass

        class _Listener:
            runtime_profile = "production"

            def __init__(self):
                self.backend = SimpleNamespace(_use_native_lm=True)
                self.primes = 0
                self.calls = 0

            def prime_steady_state(self, _state):
                self.primes += 1

            def step(self, _pcm, _reference, state):
                frame_id = state.frames_seen
                state.frames_seen += 1
                self.calls += 1
                return SimpleNamespace(
                    frame_id=frame_id,
                    gemma_audio_queries_t=torch.full((4, 1536), float(frame_id)),
                )

        listener = _Listener()
        with patch("duplex_kernel.listener.GemmaListenerState", _State):
            features, metadata = extract_conversation_features(listener, conversation)

    assert _State.constructed == listener.primes == 1
    assert listener.calls == 194
    assert features[:, 0].tolist() == [63.0, 64.0, 128.0, 129.0, 193.0]
    assert [row["frame_id"] for row in metadata] == target_frame_ids
    assert [row["cluster_start_frame_id"] for row in metadata] == [0, 0, 0, 0, 0]
    assert [row["cluster_frame_count"] for row in metadata] == [64, 65, 129, 130, 194]
    assert _clustered_target_context_workload([conversation]) == {
        "target_windows": 5,
        "causal_input_frame_steps": 194,
        "conversation_replays": 1,
        "max_conversation_replay_input_frames": 194,
    }


def test_native_benchmark_replays_eight_sparse_target_contexts():
    rate = 16000
    signal = (np.ones(1280 * 133, dtype=np.float32) * 0.1)[:, None]
    with TemporaryDirectory(prefix="stage2-benchmark-frames-") as directory:
        path = Path(directory) / "mono-16k.wav"
        sf.write(path, signal, rate, subtype="FLOAT")
        conversation = _conversation(path, rate, {"0": "user"})
        natural_target_frames = [0, 1, 20, 63, 64, 70, 127, 129]
        conversation["frames"] = [
            {"frame_id": frame_id, "target_action": "LISTEN", "action_mask": True}
            for frame_id in natural_target_frames
        ]
        synthetic = _conversation(path, rate, {"0": "user"})
        synthetic.update(conversation_id="stage2-synthetic-benchmark", source="synthetic_it_t6", synthetic=True)
        synthetic_target_frames = [4, 63, 64, 128, 129, 130, 131, 132]
        synthetic["frames"] = [
            {"frame_id": frame_id, "target_action": "SPEAK", "action_mask": True}
            for frame_id in synthetic_target_frames
        ]

        class _State:
            def __init__(self, **_kwargs):
                self.frames_seen = 0

            def close(self):
                pass

        class _Listener:
            def __init__(self):
                self.backend = SimpleNamespace(_use_native_lm=True)
                self.runtime_profile = "production"
                self.calls = 0

            def prime_steady_state(self, _state):
                pass

            def step(self, _pcm, _reference, _state):
                frame_id = _state.frames_seen
                _state.frames_seen += 1
                self.calls += 1
                return SimpleNamespace(
                    frame_id=frame_id,
                    gemma_audio_queries_t=torch.ones((4, 1536)),
                )

        listener = _Listener()
        with patch("duplex_kernel.listener.GemmaListenerState", _State), patch(
            "torch.cuda.is_available", return_value=False
        ):
            report = benchmark_native_listener_windows(listener, [conversation, synthetic], windows=8)

    assert listener.calls == 197
    assert report["windows"] == 8
    assert report["input_channel_index"] == 0
    assert report["last_query_shape"] == [1536]
    assert report["queries_finite"] is True
    assert report["selected_target_frame_ids"] == [1, 20, 63, 64, 128, 129, 130, 131]
    assert report["context_frame_counts"] == [2, 21, 64, 64, 64, 64, 64, 64]
    assert report["causal_input_frame_steps"] == 197
    assert report["conversation_replays"] == 2
    assert [row["synthetic"] for row in report["conversations"]] == [False, True]
    assert report["projected_target_windows"]["train"] == 768
    assert report["projected_target_windows"]["validation"] == 192
    assert report["projected_target_windows"]["test"] == 128


def test_benchmark_only_writes_report_without_extracting_split_features():
    rate = 16000
    signal = (np.ones(1280 * 67, dtype=np.float32) * 0.1)[:, None]
    with TemporaryDirectory(prefix="stage2-benchmark-only-") as directory:
        root = Path(directory)
        audio_path = root / "mono-16k.wav"
        sf.write(audio_path, signal, rate, subtype="FLOAT")
        natural = _conversation(audio_path, rate, {"0": "user"})
        natural["frames"] = [
            {"frame_id": frame_id, "target_action": "LISTEN", "action_mask": True}
            for frame_id in (0, 63, 64, 66)
        ]
        synthetic = _conversation(audio_path, rate, {"0": "user"})
        synthetic.update(conversation_id="stage2-synthetic-benchmark", source="synthetic_it_t6", synthetic=True)
        synthetic["frames"] = [
            {"frame_id": frame_id, "target_action": "SPEAK", "action_mask": True}
            for frame_id in (0, 63, 64, 66)
        ]
        prepared = root / "prepared"
        prepared.mkdir()
        (prepared / "train.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (natural, synthetic)) + "\n",
            encoding="utf-8",
        )
        adapter = root / "adapter"
        adapter.mkdir()
        output = root / "benchmark"

        class _State:
            def __init__(self, **_kwargs):
                self.frames_seen = 0

            def close(self):
                pass

        class _Listener:
            runtime_profile = "production"

            def __init__(self):
                self.backend = SimpleNamespace(_use_native_lm=True)

            def prime_steady_state(self, _state):
                pass

            def step(self, _pcm, _reference, state):
                frame_id = state.frames_seen
                state.frames_seen += 1
                return SimpleNamespace(
                    frame_id=frame_id,
                    gemma_audio_queries_t=torch.ones((4, 1536)),
                )

        listener = _Listener()
        with patch("training.duplex_qlora.stage2.load_merged_base", return_value=(None, None, None)), \
            patch("duplex_kernel.listener.GemmaListener.from_thinker", return_value=listener), \
            patch("duplex_kernel.listener.GemmaListenerState", _State), \
            patch("torch.cuda.is_available", return_value=False):
            result = extract_stage2_features(
                prepared_dir=prepared,
                base_model_path="unused-base",
                adapter_dir=str(adapter),
                output_dir=output,
                benchmark_only=True,
            )
            assert result["status"] == "native_listener_benchmark_only"
            assert result["native_listener_benchmark"]["windows"] == 8
            assert (output / "benchmark_manifest.json").is_file()
            assert not (output / "train.npz").exists()


def test_native1540_conversation_partials_resume_with_targets_and_provenance():
    with TemporaryDirectory(prefix="stage2-partial-resume-") as directory:
        root = Path(directory)
        conversation = {
            "conversation_id": "smooth/a",
            "source_group_id": "group-1",
            "source": "SmoothConv",
            "source_key": "smooth-key",
            "source_audio_sha256": "abc123",
            "source_split": "train",
            "annotation_path": None,
            "control_supervision": "native_turn_state_and_observed_channel_timing",
        }
        targets = [
            {
                "frame_id": 2,
                "target_action": "LISTEN",
                "action_mask": True,
                "target_action_source": "verified_remote_user_active",
                "target_action_evidence": {"source": "annotation", "event": 3},
            },
            {
                "frame_id": 5,
                "target_action": "YIELD",
                "action_mask": True,
                "target_action_source": "derived_peer_onset",
                "target_action_evidence": {"source": "timestamp", "event": 8},
            },
        ]
        metadata = [
            {
                **{key: value for key, value in conversation.items() if key.startswith("source_") or key in {"annotation_path", "control_supervision"}},
                "frame_id": target["frame_id"],
                    "listener_frame_id": target["frame_id"],
                "source_group_id": conversation["source_group_id"],
                "action": target["target_action"],
                "action_mask": True,
                "target_action_mask": True,
                "target_action_source": target["target_action_source"],
                "target_action_evidence": target["target_action_evidence"],
                "native_control6_hidden_source": "ListenerOutput.gemma_hidden_t",
                "native_control6_audio_queries_per_tick": 2,
                "native_control6_mel_window_rows": 240,
            }
            for target in targets
        ]
        partial = _conversation_partial_path(root, "train", conversation["conversation_id"])
        assert "%2F" in partial.name
        expected_features = np.arange(2 * 1536, dtype=np.float32).reshape(2, 1536)
        _write_features(partial, expected_features, metadata)

        resumed = _load_conversation_partial(
            partial,
            conversation=conversation,
            targets=targets,
            control6_profile="native1540",
        )

        assert resumed is not None
        features, resumed_metadata = resumed
        np.testing.assert_array_equal(features, expected_features)
        assert [row["frame_id"] for row in resumed_metadata] == [2, 5]
        assert [row["target_action_source"] for row in resumed_metadata] == [
            "verified_remote_user_active",
            "derived_peer_onset",
        ]
        assert all(row["source_audio_sha256"] == "abc123" for row in resumed_metadata)

        partial.with_suffix(".json").unlink()
        assert _load_conversation_partial(
            partial,
            conversation=conversation,
            targets=targets,
            control6_profile="native1540",
        ) is None


def test_extract_control_cli_exposes_resume_existing():
    args = build_parser().parse_args(
        [
            "extract-control",
            "--prepared-dir",
            "prepared",
            "--output-dir",
            "output",
            "--control6-profile",
            "native1540",
            "--resume-existing",
        ]
    )
    assert args.resume_existing is True


def test_merged_checkpoint_provenance_reaches_nested_language_model():
    with TemporaryDirectory(prefix="stage2-source-attestation-") as directory:
        language_model = SimpleNamespace()
        nested = SimpleNamespace(language_model=language_model)
        model = SimpleNamespace(model=nested)

        resolved = _attest_merged_model_source(model, directory)

    assert resolved == str(Path(directory).resolve())
    assert model._gemma4_checkpoint_path == resolved
    assert nested._gemma4_checkpoint_path == resolved
    assert language_model._gemma4_checkpoint_path == resolved


def test_canonical_prefix_snapshot_owns_immutable_kv_tensors():
    from transformers.cache_utils import DynamicCache
    from duplex_kernel.listener import GemmaAudioTowerBackend

    source = DynamicCache()
    keys = torch.arange(24, dtype=torch.float32).view(1, 2, 3, 4)
    values = keys + 100
    source.update(keys, values, 0)
    source.shared_layers = {13: (keys + 200, values + 200)}

    snapshot = GemmaAudioTowerBackend._clone_canonical_prefix_snapshot(source)

    assert snapshot is not source
    assert torch.equal(snapshot.layers[0].keys, keys)
    assert torch.equal(snapshot.layers[0].values, values)
    assert snapshot.layers[0].keys.data_ptr() != source.layers[0].keys.data_ptr()
    assert snapshot.layers[0].values.data_ptr() != source.layers[0].values.data_ptr()
    assert torch.equal(snapshot.shared_layers[13][0], source.shared_layers[13][0])
    assert snapshot.shared_layers[13][0].data_ptr() != source.shared_layers[13][0].data_ptr()
    source.layers[0].keys.fill_(999)
    source.shared_layers[13][0].fill_(999)
    assert not bool(torch.any(snapshot.layers[0].keys == 999))
    assert not bool(torch.any(snapshot.shared_layers[13][0] == 999))


if __name__ == "__main__":
    test_48khz_resampling_filters_alias_and_preserves_partial_frame_mask()
    test_24khz_resampling_preserves_duration_and_marks_final_partial_frame()
    test_sparse_targets_still_advance_listener_over_every_causal_frame()
    test_native_benchmark_replays_eight_sparse_target_contexts()
    test_benchmark_only_writes_report_without_extracting_split_features()
    test_merged_checkpoint_provenance_reaches_nested_language_model()
    test_canonical_prefix_snapshot_owns_immutable_kv_tensors()
    print("Stage2 resampling/sparse-frame CPU smoke: 8 tests PASS")
