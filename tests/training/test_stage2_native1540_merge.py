import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from training.duplex_qlora.features import native_control6_feature_contract
from training.duplex_qlora.stage2 import merge_native1540_feature_dirs


class Native1540MergeTest(unittest.TestCase):
    def test_appends_train_and_preserves_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            supplement = root / "supplement"
            output = root / "merged"
            base.mkdir()
            supplement.mkdir()

            worker = {
                "worker": "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker",
                "worker_factory_source": "/runtime/duplex_kernel/listener_microbatch_serving.py",
                "prefix_builder": "duplex_kernel.server._build_unified_listener_prefix_token_ids",
                "prefix_builder_source": "/runtime/duplex_kernel/server.py",
                "runtime_source_root": "/runtime",
                "native_library_path": "/runtime/build5.so",
                "runtime_signature": {"batch_api_version": 5, "stream_q_len": 2},
                "prefix": {"contract": "serving_unified_chat_prefix_v2", "token_ids": [2, 256000], "length": 2},
            }
            manifest = {
                "feature_contract": native_control6_feature_contract(1540),
                "phase_profile": "native1540",
                "phase_schema": {"profile": "native1540", "width": 4,
                                 "names": ["user_rms_log", "user_rms_delta", "assistant_rms_log",
                                           "user_minus_assistant_log_ratio"]},
                "phase_definition": {"profile": "native1540"},
                "base_model_path": "/model",
                "serving_base_only": True,
                "merged_before_native_listener": False,
                "serving_base_loader": {"loader": "duplex_kernel.models.thinker.load_thinker"},
                "native_control6_extraction": {
                    "profile": "native1540", "hidden_source": "ListenerOutput.gemma_hidden_t",
                    "hidden_contract": "gemma4_listener_final_norm_frame_v1",
                    "gemma_audio_queries_per_tick": 2, "audio_tower_queries_per_tick": 2,
                    "mel_window_rows_effective": 240, "base_model_path": "/model",
                    "response_lora_applied": False, "stage1_adapter_applied": False,
                    "serving_worker": worker,
                },
                "test_data_read": False,
                "test_used_for_training_or_calibration": False,
                "splits": {"train": {"rows": 1}, "validation": {"rows": 1}},
            }
            for directory in (base, supplement):
                (directory / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            def write_split(directory, split, rows, values):
                metadata = rows
                features = np.asarray(values, dtype=np.float32).reshape(len(rows), 1540)
                labels = np.asarray([0 if row["action_mask"] else -100 for row in rows], dtype=np.int64)
                frames = np.asarray([row["frame_id"] for row in rows], dtype=np.int64)
                listener = np.asarray([row["listener_frame_id"] for row in rows], dtype=np.int64)
                np.savez_compressed(directory / f"{split}.npz", features=features, labels=labels,
                                    frame_ids=frames, listener_frame_ids=listener)
                (directory / f"{split}.json").write_text(json.dumps(metadata), encoding="utf-8")

            write_split(base, "train", [{
                "conversation_id": "base-a", "source_group_id": "group-a", "frame_id": 0,
                "listener_frame_id": 0, "action": "LISTEN", "action_mask": True, "split": "train",
            }], [np.ones(1540)])
            write_split(base, "validation", [{
                "conversation_id": "val-a", "source_group_id": "val-group", "frame_id": 0,
                "listener_frame_id": 0, "action": "LISTEN", "action_mask": True, "split": "validation",
            }], [np.full(1540, 3.0)])
            write_split(supplement, "train", [{
                "conversation_id": "new-a", "source_group_id": "new-group", "frame_id": 1,
                "listener_frame_id": 1, "action": "LISTEN", "action_mask": True, "split": "train",
            }], [np.full(1540, 2.0)])

            original_validation_npz = (base / "validation.npz").read_bytes()
            original_validation_json = (base / "validation.json").read_bytes()
            report = merge_native1540_feature_dirs(
                base_feature_dir=base, supplemental_feature_dir=supplement, output_dir=output,
            )
            with np.load(output / "train.npz", allow_pickle=False) as merged:
                self.assertEqual(merged["features"].shape, (2, 1540))
                np.testing.assert_array_equal(merged["labels"], np.asarray([0, 0]))
            self.assertEqual((output / "validation.npz").read_bytes(), original_validation_npz)
            self.assertEqual((output / "validation.json").read_bytes(), original_validation_json)
            self.assertEqual(report["train_rows"], 2)
            self.assertEqual(report["validation_rows"], 1)


if __name__ == "__main__":
    unittest.main()
