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
    _attest_merged_model_source,
    _causal_action_targets,
    _cluster_causal_targets,
    _clustered_target_context_workload,
    benchmark_native_listener_windows,
    extract_conversation_features,
    extract_stage2_features,
)


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
    test_clustered_sparse_targets_share_causal_state_and_never_advance_future_frames()
    test_native_benchmark_replays_eight_sparse_target_contexts()
    test_benchmark_only_writes_report_without_extracting_split_features()
    test_merged_checkpoint_provenance_reaches_nested_language_model()
    test_canonical_prefix_snapshot_owns_immutable_kv_tensors()
    print("Stage2 resampling/sparse-frame CPU smoke: 8 tests PASS")
