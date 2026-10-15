from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from training.duplex_qlora.features import FEATURE_CONTRACT
from training.duplex_qlora.schema import ACTIONS
from training.duplex_qlora.stage2 import augment_control_features_with_acoustic_phase


class Stage2AcousticPhaseAugmentationTests(unittest.TestCase):
    def test_test_split_is_gated_until_a_frozen_candidate_head_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "must_not_be_created"
            with self.assertRaisesRegex(ValueError, "frozen Candidate C head"):
                augment_control_features_with_acoustic_phase(
                    native_feature_dir=root / "no_test_features",
                    prepared_dir=root / "no_test_source",
                    output_dir=output,
                    splits=("test",),
                )
            self.assertFalse(output.exists())

    def test_appends_causal_features_and_masks_only_verified_nonoverlap_train_bc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.wav"
            feature_dir = root / "native_features"
            prepared = root / "prepared"
            output = root / "candidate_c"
            feature_dir.mkdir()
            prepared.mkdir()

            pcm = np.zeros((6 * 1280, 2), dtype=np.float32)
            pcm[1280:2560, 1] = 0.2
            pcm[2560:3840, 0] = 0.4
            sf.write(source, pcm, 16_000, subtype="FLOAT")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = [
                {
                    "conversation_id": "conversation-1",
                    "source_group_id": "group-1",
                    "session_id": "train:conversation-1:cluster-0",
                    "frame_id": tick,
                    "input_frame_id": tick,
                    "target_frame_id": tick,
                    "listener_frame_id": tick,
                    "assistant_reference_frame_id": tick - 1,
                    "source": "SmoothConv",
                    "synthetic": False,
                    "action": "BACKCHANNEL",
                    "action_mask": True,
                    "target_action_mask": True,
                    "event_onset": False,
                }
                for tick in (1, 3)
            ]
            np.savez_compressed(
                feature_dir / "train.npz",
                features=np.zeros((2, 1536), dtype=np.float32),
                labels=np.full((2,), ACTIONS.index("BACKCHANNEL"), dtype=np.int64),
                frame_ids=np.asarray([1, 3], dtype=np.int64),
                listener_frame_ids=np.asarray([1, 3], dtype=np.int64),
            )
            (feature_dir / "train.json").write_text(json.dumps(metadata), encoding="utf-8")
            (feature_dir / "feature_manifest.json").write_text(
                json.dumps({
                    "feature_contract": FEATURE_CONTRACT,
                    "merged_before_native_listener": True,
                    "adapter_dir": "/models/final-adapter",
                    "base_model_path": "/models/base",
                }),
                encoding="utf-8",
            )
            source_row = {
                "conversation_id": "conversation-1",
                "source_group_id": "group-1",
                "split": "train",
                "synthetic": False,
                "audio_path": str(source),
                "audio_sha256": digest,
                "sample_rate_hz": 16_000,
                "channels": 2,
                "tick_ms": 80,
                "channel_map": {"0": "assistant", "1": "user"},
                "frames": [
                    {"frame_id": 1, "start_ms": 80, "end_ms": 160, "target_action": "BACKCHANNEL", "action_mask": True, "target_action_source": "verified_native_assistant_action"},
                    {"frame_id": 3, "start_ms": 240, "end_ms": 320, "target_action": "BACKCHANNEL", "action_mask": True, "target_action_source": "verified_native_assistant_action"},
                ],
                "segments": [
                    {"role": "assistant", "channel_index": 0, "start_ms": 85, "end_ms": 120, "native_turn_state": "backchannel", "audio_channel_verified": True, "time_mask": True, "target_mask": False},
                    {"role": "user", "channel_index": 1, "start_ms": 90, "end_ms": 150, "audio_channel_verified": True, "time_mask": True, "target_mask": False},
                    {"role": "assistant", "channel_index": 0, "start_ms": 245, "end_ms": 275, "native_turn_state": "backchannel", "audio_channel_verified": True, "time_mask": True, "target_mask": True},
                ],
            }
            (prepared / "train.jsonl").write_text(
                json.dumps(source_row) + "\n", encoding="utf-8"
            )

            report = augment_control_features_with_acoustic_phase(
                native_feature_dir=feature_dir,
                prepared_dir=prepared,
                output_dir=output,
                splits=("train",),
            )

            with np.load(output / "train.npz", allow_pickle=False) as values:
                augmented = values["features"]
                labels = values["labels"]
            rows = json.loads((output / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(augmented.shape, (2, 1540))
            self.assertEqual(labels.tolist(), [ACTIONS.index("BACKCHANNEL"), -100])
            np.testing.assert_array_equal(augmented[:, :1536], np.zeros((2, 1536), dtype=np.float32))
            self.assertTrue(rows[0]["action_mask"])
            self.assertFalse(rows[1]["action_mask"])
            self.assertEqual(rows[1]["action"], "BACKCHANNEL")
            self.assertIn("without_verified_remote_overlap", rows[1]["action_masked_reason"])
            self.assertFalse((output / "test.npz").exists())
            self.assertEqual(report["natural_train_backchannel_masking"]["masked_rows"], 1)
            self.assertEqual(report["natural_train_backchannel_masking"]["verified_overlap_rows"], 1)

    def test_authored_v5_backchannel_is_not_misclassified_as_natural(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "v5.wav"
            feature_dir = root / "native_features"
            prepared = root / "prepared"
            output = root / "candidate_e"
            feature_dir.mkdir()
            prepared.mkdir()

            sf.write(source, np.zeros((4 * 1280, 2), dtype=np.float32), 16_000, subtype="FLOAT")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            metadata = [{
                "conversation_id": "v5-conversation",
                "source_group_id": "v5-group",
                "session_id": "train:v5-conversation:conversation-0",
                "frame_id": 1,
                "input_frame_id": 1,
                "target_frame_id": 2,
                "listener_frame_id": 1,
                "assistant_reference_frame_id": 0,
                "cluster_start_frame_id": 0,
                "source": "authored_synthetic_audio_candidate",
                # Older prepared remix rows may omit this convenience flag;
                # source/profile remains the authoritative provenance.
                "synthetic": False,
                "action": "BACKCHANNEL",
                "action_mask": True,
                "target_action_mask": True,
                "event_onset": False,
                "input_text_visible": False,
                "future_audio_excluded": True,
            }]
            np.savez_compressed(
                feature_dir / "train.npz",
                features=np.zeros((1, 1536), dtype=np.float32),
                labels=np.asarray([ACTIONS.index("BACKCHANNEL")], dtype=np.int64),
                frame_ids=np.asarray([1], dtype=np.int64),
                listener_frame_ids=np.asarray([1], dtype=np.int64),
            )
            (feature_dir / "train.json").write_text(json.dumps(metadata), encoding="utf-8")
            (feature_dir / "feature_manifest.json").write_text(
                json.dumps({
                    "feature_contract": FEATURE_CONTRACT,
                    "merged_before_native_listener": True,
                    "adapter_dir": "/models/final-adapter",
                    "base_model_path": "/models/base",
                }),
                encoding="utf-8",
            )
            frame = {
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
            }
            source_row = {
                "conversation_id": "v5-conversation",
                "source_group_id": "v5-group",
                "split": "train",
                "source": "authored_synthetic_audio_candidate",
                "synthetic": False,
                "audio_path": str(source),
                "audio_sha256": digest,
                "sample_rate": 16_000,
                "channels": 2,
                "channel_map": {"0": "user", "1": "assistant"},
                "frame_targets": [frame],
                "event_schedule": [],
                "segments": [],
            }
            (prepared / "train.jsonl").write_text(json.dumps(source_row) + "\n", encoding="utf-8")

            report = augment_control_features_with_acoustic_phase(
                native_feature_dir=feature_dir,
                prepared_dir=prepared,
                output_dir=output,
                splits=("train",),
            )

            rows = json.loads((output / "train.json").read_text(encoding="utf-8"))
            self.assertTrue(rows[0]["synthetic"])
            self.assertTrue(rows[0]["action_mask"])
            self.assertEqual(report["natural_train_backchannel_masking"]["masked_rows"], 0)
            self.assertEqual(report["natural_train_backchannel_masking"]["remaining_supervised_natural_backchannel_rows"], 0)


if __name__ == "__main__":
    unittest.main()
