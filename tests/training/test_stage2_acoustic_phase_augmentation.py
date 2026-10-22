from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from training.duplex_qlora.features import FEATURE_CONTRACT, native_control6_feature_contract
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

    def test_native1540_preserves_prepared_targets_and_provenance_without_endpoint_masking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "duplexconv-source.wav"
            feature_dir = root / "native1540_features"
            prepared = root / "prepared_timestamp_derived"
            output = root / "native1540_phase"
            feature_dir.mkdir()
            prepared.mkdir()

            pcm = np.zeros((3 * 1280, 2), dtype=np.float32)
            pcm[:, 1] = 0.1
            sf.write(source, pcm, 16_000, subtype="FLOAT")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target_frames = [
                {
                    "frame_id": 0,
                    "target_action": "BACKCHANNEL",
                    "action_mask": True,
                    "target_action_source": "verified_native_assistant_action",
                    "target_action_evidence": {"native_turn_state": "backchannel"},
                },
                {
                    "frame_id": 1,
                    "target_action": "YIELD",
                    "action_mask": True,
                    "target_action_source": "timestamp_derived_peer_onset_confirmed_local_stop",
                    "target_action_evidence": {"peer_onset_ms": 80.0, "local_stop_ms": 250.0},
                },
            ]
            feature_metadata = [
                {
                    "conversation_id": "duplexconv-paired-01-user",
                    "source_group_id": "duplexconv-paired-01",
                    "session_id": f"train:duplexconv-paired-01-user:cluster-{frame['frame_id']}",
                    "frame_id": frame["frame_id"],
                    "input_frame_id": frame["frame_id"],
                    "target_frame_id": frame["frame_id"],
                    "listener_frame_id": frame["frame_id"],
                    "assistant_reference_frame_id": frame["frame_id"] - 1,
                    "source": "DuplexConv",
                    "synthetic": False,
                    "action": frame["target_action"],
                    "action_mask": frame["action_mask"],
                    "target_action_mask": frame["action_mask"],
                    "target_action_source": frame["target_action_source"],
                    "target_action_evidence": frame["target_action_evidence"],
                }
                for frame in target_frames
            ]
            source_targets = [
                {
                    **frame,
                    "start_ms": frame["frame_id"] * 80,
                    "end_ms": (frame["frame_id"] + 1) * 80,
                }
                for frame in target_frames
            ]
            np.savez_compressed(
                feature_dir / "train.npz",
                features=np.zeros((2, 1536), dtype=np.float32),
                labels=np.asarray(
                    [ACTIONS.index(frame["target_action"]) for frame in target_frames],
                    dtype=np.int64,
                ),
                frame_ids=np.asarray([0, 1], dtype=np.int64),
                listener_frame_ids=np.asarray([0, 1], dtype=np.int64),
            )
            (feature_dir / "train.json").write_text(json.dumps(feature_metadata), encoding="utf-8")
            (feature_dir / "feature_manifest.json").write_text(
                json.dumps({
                    "feature_contract": native_control6_feature_contract(1536),
                    "serving_base_only": True,
                    "merged_before_native_listener": False,
                    "native_control6_extraction": {
                        "profile": "native1540",
                        "hidden_source": "ListenerOutput.gemma_hidden_t",
                        "gemma_audio_queries_per_tick": 2,
                        "audio_tower_queries_per_tick": 2,
                        "mel_window_rows_effective": 240,
                    },
                }),
                encoding="utf-8",
            )
            source_row = {
                "conversation_id": "duplexconv-paired-01-user",
                "source_group_id": "duplexconv-paired-01",
                "split": "train",
                "source": "DuplexConv",
                "synthetic": False,
                "audio_path": str(source),
                "audio_sha256": digest,
                "sample_rate_hz": 16_000,
                "channels": 2,
                "tick_ms": 80,
                "channel_map": {"0": "assistant", "1": "user"},
                "frames": source_targets,
                "segments": [
                    {
                        "role": "assistant",
                        "channel_index": 0,
                        "start_ms": 10.0,
                        "end_ms": 50.0,
                        "native_turn_state": "backchannel",
                        "audio_channel_verified": True,
                        "time_mask": True,
                    }
                ],
            }
            (prepared / "train.jsonl").write_text(json.dumps(source_row) + "\n", encoding="utf-8")

            report = augment_control_features_with_acoustic_phase(
                native_feature_dir=feature_dir,
                prepared_dir=prepared,
                output_dir=output,
                splits=("train",),
                phase_profile="native1540",
            )

            with np.load(output / "train.npz", allow_pickle=False) as values:
                self.assertEqual(values["labels"].tolist(), [ACTIONS.index("BACKCHANNEL"), ACTIONS.index("YIELD")])
                self.assertEqual(values["features"].shape, (2, 1540))
            result_rows = json.loads((output / "train.json").read_text(encoding="utf-8"))
            for source_row, result_row in zip(feature_metadata, result_rows):
                for field in ("action", "action_mask", "target_action_mask", "target_action_source", "target_action_evidence"):
                    self.assertEqual(result_row[field], source_row[field])
            self.assertTrue(report["splits"]["train"]["prepared_supervision_aligned"])
            self.assertEqual(report["splits"]["train"]["prepared_action_support"], report["splits"]["train"]["cached_metadata_action_support_before_prepared_refresh"])
            self.assertEqual(report["splits"]["train"]["prepared_supervised_rows"], 2)
            self.assertNotIn("natural_train_backchannel_masking", report)
            self.assertEqual(report["training_state_context"], "user_stream_q2_only")
            for row in result_rows:
                self.assertEqual(row["training_state_context"], "user_stream_q2_only")

    def test_native1540_supplement_merge_uses_prepared_targets_and_preserves_cached_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "qualified-source.wav"
            base = root / "base-features"
            supplement = root / "supplement-features"
            prepared = root / "prepared-qualified-seed"
            output = root / "native1540-merged"
            for path in (base, supplement, prepared):
                path.mkdir()

            pcm = np.zeros((6 * 1280, 2), dtype=np.float32)
            pcm[:, 1] = 0.02
            sf.write(source, pcm, 16_000, subtype="FLOAT")
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            worker = {
                "worker": "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker",
                "prefix_builder": "duplex_kernel.server._build_unified_listener_prefix_token_ids",
                "native_library_path": "/runtime/libnative-cap512.so",
                "runtime_source_root": "/runtime/listener_microbatch_source",
                "worker_factory_source": "/runtime/listener_microbatch_source/duplex_kernel/listener_microbatch_serving.py",
                "prefix_builder_source": "/runtime/listener_microbatch_source/duplex_kernel/server.py",
                "runtime_info": {
                    "library_path": "/runtime/libnative-cap512.so",
                    "stream_q_len": 2,
                    "unified_kv_enabled": True,
                },
                "runtime_signature": {"model_id": "google/gemma-4-E2B-it", "stream_q_len": 2},
                "prefix": {
                    "contract": "serving_unified_chat_prefix_v2",
                    "token_ids": [1] * 189 + [256000],
                    "length": 190,
                    "history_messages": 0,
                    "session_instructions": "empty_session_default",
                },
            }
            supplement_worker = {
                **worker,
                "runtime_source_root": "/runtime/control6_training_source_supplement",
                "worker_factory_source": "/runtime/control6_training_source_supplement/duplex_kernel/listener_microbatch_serving.py",
                "prefix_builder_source": "/runtime/control6_training_source_supplement/duplex_kernel/server.py",
            }

            def write_native_feature_dir(
                directory: Path,
                rows: list[dict],
                ids: list[int],
                values: np.ndarray,
                worker_info: dict | None = None,
            ) -> None:
                np.savez_compressed(
                    directory / "train.npz",
                    features=values,
                    labels=np.asarray(
                        [ACTIONS.index(row["action"]) if row["action_mask"] else -100 for row in rows],
                        dtype=np.int64,
                    ),
                    frame_ids=np.asarray(ids, dtype=np.int64),
                    listener_frame_ids=np.asarray(ids, dtype=np.int64),
                )
                (directory / "train.json").write_text(json.dumps(rows), encoding="utf-8")
                (directory / "feature_manifest.json").write_text(
                    json.dumps({
                        "feature_contract": native_control6_feature_contract(1536),
                        "serving_base_only": True,
                        "merged_before_native_listener": False,
                        "adapter_dir": None,
                        "base_model_path": "/models/gemma4-base",
                        "serving_base_loader": {
                            "base_model_path": "/models/gemma4-base",
                            "model_dtype": "bfloat16",
                            "stage1_adapter_applied": False,
                        },
                        "native_control6_extraction": {
                            "profile": "native1540",
                            "hidden_source": "ListenerOutput.gemma_hidden_t",
                            "gemma_audio_queries_per_tick": 2,
                            "audio_tower_queries_per_tick": 2,
                            "mel_window_rows_effective": 240,
                            "base_model_path": "/models/gemma4-base",
                            "stage1_adapter_applied": False,
                            "serving_worker": worker_info or worker,
                        },
                    }),
                    encoding="utf-8",
                )

            prepared_frames = [
                {"frame_id": 1, "start_ms": 80, "end_ms": 160, "target_action": "SPEAK", "action_mask": True,
                 "target_action_source": "timestamp_derived_last_vocal_complete_user_turn",
                 "target_action_evidence": {"seed_onset_frame_id": 5, "deduplicated_seed_count": 2}},
                {"frame_id": 2, "start_ms": 160, "end_ms": 240, "target_action": None, "action_mask": False,
                 "target_action_source": None, "action_masked_reason": "superseded_onset_speak_by_endpoint_view"},
                {"frame_id": 3, "start_ms": 240, "end_ms": 320, "target_action": "BACKCHANNEL", "action_mask": True,
                 "target_action_source": "verified_native_assistant_action",
                 "target_action_evidence": {"native_turn_state": "backchannel", "reviewed": True}},
                {"frame_id": 4, "start_ms": 320, "end_ms": 400, "target_action": "SPEAK", "action_mask": True,
                 "target_action_source": "timestamp_derived_last_vocal_complete_user_turn",
                 "target_action_evidence": {"seed_onset_frame_id": 6, "deduplicated_seed_count": 2}},
            ]

            def metadata(frame_id: int, action: str, evidence: dict) -> dict:
                return {
                    "conversation_id": "qualified-conversation",
                    "source_group_id": "qualified-group",
                    "session_id": f"train:qualified-conversation:cluster-{frame_id}",
                    "frame_id": frame_id,
                    "input_frame_id": frame_id,
                    "target_frame_id": frame_id,
                    "listener_frame_id": frame_id,
                    "assistant_reference_frame_id": frame_id - 1,
                    "source": "DuplexConv",
                    "synthetic": False,
                    "action": action,
                    "action_mask": True,
                    "target_action_mask": True,
                    "target_action_source": "cached-stale-label",
                    "target_action_evidence": evidence,
                }

            base_rows = [
                metadata(1, "LISTEN", {"cached": "old-listen"}),
                metadata(2, "SPEAK", {"cached": "old-onset"}),
                metadata(3, "BACKCHANNEL", {"native_turn_state": "backchannel", "reviewed": True}),
            ]
            supplement_rows = [
                metadata(4, "SPEAK", {"cached": "exact-supplement-target"}),
            ]
            base_hidden = np.stack([
                np.full(1536, 0.11, dtype=np.float32),
                np.full(1536, 0.22, dtype=np.float32),
                np.full(1536, 0.33, dtype=np.float32),
            ])
            supplement_hidden = np.full((1, 1536), 0.44, dtype=np.float32)
            write_native_feature_dir(base, base_rows, [1, 2, 3], base_hidden)
            write_native_feature_dir(supplement, supplement_rows, [4], supplement_hidden, supplement_worker)
            source_row = {
                "conversation_id": "qualified-conversation",
                "source_group_id": "qualified-group",
                "split": "train",
                "source": "DuplexConv",
                "synthetic": False,
                "audio_path": str(source),
                "audio_sha256": digest,
                "sample_rate_hz": 16_000,
                "channels": 2,
                "tick_ms": 80,
                "native1540_temporal_supervision_view": {
                    "schema": "native1540_pcm_temporal_supervision_v1",
                },
                "channel_map": {"0": "assistant", "1": "user"},
                "frames": prepared_frames,
                "segments": [
                    {"role": "assistant", "channel_index": 0, "start_ms": 245, "end_ms": 275,
                     "native_turn_state": "backchannel", "audio_channel_verified": True, "time_mask": True}
                ],
            }
            (prepared / "train.jsonl").write_text(json.dumps(source_row) + "\n", encoding="utf-8")

            report = augment_control_features_with_acoustic_phase(
                native_feature_dir=base,
                supplemental_native_feature_dir=supplement,
                prepared_dir=prepared,
                output_dir=output,
                splits=("train",),
                phase_profile="native1540",
            )

            with np.load(output / "train.npz", allow_pickle=False) as values:
                result_features = values["features"]
                self.assertEqual(values["frame_ids"].tolist(), [1, 3, 4])
                self.assertEqual(values["labels"].tolist(), [ACTIONS.index("SPEAK"),
                                                               ACTIONS.index("BACKCHANNEL"), ACTIONS.index("SPEAK")])
                np.testing.assert_array_equal(result_features[0, :1536], base_hidden[0])
                np.testing.assert_array_equal(result_features[1, :1536], base_hidden[2])
                np.testing.assert_array_equal(result_features[2, :1536], supplement_hidden[0])
            rows = json.loads((output / "train.json").read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["native_control6_cached_action_before_target_view"], "LISTEN")
            self.assertEqual(rows[0]["action"], "SPEAK")
            self.assertEqual(rows[0]["target_action_source"], "timestamp_derived_last_vocal_complete_user_turn")
            self.assertEqual(rows[1]["frame_id"], 3)
            self.assertEqual(rows[1]["action"], "BACKCHANNEL")
            self.assertEqual(rows[1]["target_action_source"], "verified_native_assistant_action")
            self.assertEqual(rows[2]["target_action_evidence"]["deduplicated_seed_count"], 2)
            self.assertNotIn(2, [row["frame_id"] for row in rows])
            self.assertEqual(report["supplemental_feature_rows_by_split"]["train"], 1)
            self.assertEqual(report["splits"]["train"]["prepared_supervised_rows"], 3)
            self.assertTrue(report["splits"]["train"]["prepared_supervision_aligned"])
            provenance = report["serving_runtime_provenance"]
            self.assertEqual(provenance["base"]["runtime_source_root"], "/runtime/listener_microbatch_source")
            self.assertEqual(
                provenance["supplement"]["runtime_source_root"],
                "/runtime/control6_training_source_supplement",
            )
            self.assertNotIn("source_feature_sha256", report)
            self.assertNotIn("source_feature_manifest_sha256", report)
            saved_manifest = json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["serving_runtime_provenance"], provenance)
            self.assertNotIn("source_feature_sha256", saved_manifest)
            self.assertNotIn("source_feature_manifest_sha256", saved_manifest)

            supplement_manifest_path = supplement / "feature_manifest.json"
            valid_supplement_manifest = json.loads(supplement_manifest_path.read_text(encoding="utf-8"))
            invalid_cases = (
                (
                    "q4",
                    lambda manifest: manifest["native_control6_extraction"].__setitem__(
                        "gemma_audio_queries_per_tick", 4
                    ),
                    "Q2/W240",
                ),
                (
                    "adapter",
                    lambda manifest: manifest.__setitem__("adapter_dir", "/models/stage1-adapter"),
                    "serving base-only",
                ),
                (
                    "feature-schema",
                    lambda manifest: manifest["feature_contract"].__setitem__("input_dim", 1535),
                    "different feature schema",
                ),
            )
            for label, change, expected_error in invalid_cases:
                invalid = json.loads(json.dumps(valid_supplement_manifest))
                change(invalid)
                supplement_manifest_path.write_text(json.dumps(invalid), encoding="utf-8")
                with self.subTest(invalid_contract=label), self.assertRaisesRegex(ValueError, expected_error):
                    augment_control_features_with_acoustic_phase(
                        native_feature_dir=base,
                        supplemental_native_feature_dir=supplement,
                        prepared_dir=prepared,
                        output_dir=root / f"native1540-reject-{label}-supplement",
                        splits=("train",),
                        phase_profile="native1540",
                    )
            supplement_manifest_path.write_text(json.dumps(valid_supplement_manifest), encoding="utf-8")

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
