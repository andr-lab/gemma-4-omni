from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from duplex_kernel.action_conditioning import action_conditioning_instruction
from training.duplex_qlora.action_conditioned_data import (
    build_action_conditioned_records,
    build_backchannel_lexicon,
    prepare_action_conditioned_dataset,
    prepare_audio_asr_mix_dataset,
    prepare_audio_grounded_v5_dataset,
)
from training.duplex_qlora.stage1 import _training_records


def _row() -> dict:
    return {
        "conversation_id": "v4-train-001",
        "source_group_id": "shared_space",
        "source": "synthetic_it_t6",
        "synthetic": True,
        "script_version": "synthetic-it-t6-v4.0.0",
        "split": "train",
        "language": "Italian",
        "audio_path": "/data/v4.wav",
        "audio_sha256": "a" * 64,
        "sample_rate_hz": 24000,
        "duration_ms": 5000,
        "channel_map": {"0": "user", "1": "assistant"},
        "source_action_segments": [
            {"action": "SPEAK", "start_ms": 1020, "end_ms": 1260},
            {"action": "BACKCHANNEL", "start_ms": 2220, "end_ms": 2780},
            {"action": "REPLAN", "start_ms": 3220, "end_ms": 3460},
        ],
        "segments": [
            {"role": "user", "script_role": "ordinary", "segment_id": "u0", "start_ms": 0, "end_ms": 800, "text": "Prima richiesta completa.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_start", "segment_id": "a0", "start_ms": 1000, "end_ms": 1800, "text": "Risposta iniziale.", "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "script_role": "ordinary", "segment_id": "u1", "start_ms": 1900, "end_ms": 2500, "text": "Trascrizione futura da non mostrare.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_backchannel", "segment_id": "a1", "start_ms": 2220, "end_ms": 2780, "text": "Certo.", "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "script_role": "ordinary", "segment_id": "u2", "start_ms": 2800, "end_ms": 3200, "text": "Correzione completa.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_replan", "segment_id": "a2", "start_ms": 3220, "end_ms": 4000, "text": "Rivedo il piano.", "time_mask": True, "text_mask": True, "target_mask": True},
            {"role": "user", "script_role": "ordinary", "segment_id": "u3", "start_ms": 3000, "end_ms": 3600, "text": "Incomplete e sovrapposta.", "native_turn_state": "incomplete", "time_mask": True, "text_mask": True},
        ],
    }


def _v5_row() -> dict:
    def utterance(utterance_id, speaker, role, start_ms, end_ms, text, target_action=None,
                  measured_duration_s=None):
        row = {
            "utterance_id": utterance_id,
            "speaker": speaker,
            "role": role,
            "start_sample": start_ms * 16,
            "end_sample": end_ms * 16,
            "text": text,
            "target_action": target_action,
        }
        if measured_duration_s is not None:
            row["measured_duration_s"] = measured_duration_s
        return row

    return {
        "conversation_id": "v5-train-001",
        "topic_group_id": "topic-garden",
        "split": "train",
        "profile": "audio_grounded_v5",
        "profile_version": "synthetic-it-t6-audio-grounded-v5-plan.1",
        "source": "authored_synthetic_audio_candidate",
        "human_annotated": False,
        "language": "it-IT",
        "audio_path": "/data/v5.wav",
        "audio_sha256": "b" * 64,
        "sample_rate": 16000,
        "channels": 2,
        "channel_map": {"0": "user", "1": "assistant"},
        "duration_s": 9.0,
        "frame_ms": 80,
        "causal_frame_cutoff": "input_frame_id=t contains audio through its end; target_frame_id=t+1 labels the following 80 ms interval",
        "frame_targets": [{
            "frame_id": 0,
            "input_frame_id": 0,
            "target_frame_id": 1,
            "input_audio_start_sample": 0,
            "input_audio_end_sample": 1280,
            "target_action_start_sample": 1280,
            "target_action_end_sample": 2560,
            "input_text_visible": False,
            "future_audio_excluded": True,
            "action": "LISTEN",
            "target_action_mask": True,
        }],
        "utterances": [
            utterance("user_open", "user", "ordinary", 0, 1040, "Richiesta iniziale."),
            utterance("assistant_reply", "assistant", "assistant_start", 1200, 3200,
                      "La risposta iniziale.", "SPEAK", 2.0),
            utterance("user_ack", "user", "non_interrupting_ack", 1840, 2160,
                      "Sì, certo."),
            utterance("user_elaboration", "user", "ordinary", 3360, 4960,
                      "Un dettaglio che resta segreto nel prompt."),
            utterance("assistant_backchannel", "assistant", "assistant_backchannel", 4160, 4400,
                      "Capito.", "BACKCHANNEL", 0.24),
            utterance("assistant_long", "assistant", "assistant_start_interruptible", 5120, 6240,
                      "La risposta interrotta.", "SPEAK", 2.0),
            utterance("user_correction", "user", "fresh_correction", 6080, 6880,
                      "Correzione da non mostrare.") ,
            utterance("assistant_replan", "assistant", "assistant_replan", 7680, 8480,
                      "Rivediamo il piano.", "REPLAN", 0.8),
        ],
        "event_schedule": [
            {"action": "SPEAK", "start_s": 1.2, "end_s": 1.44,
             "response_utterance_id": "assistant_reply", "context_utterance_id": "user_open"},
            {"action": "BACKCHANNEL", "start_s": 4.16, "end_s": 4.4,
             "response_utterance_id": "assistant_backchannel", "context_utterance_id": "user_elaboration",
             "user_elapsed_before_backchannel_ms": 800.0},
            {"action": "SPEAK", "start_s": 5.12, "end_s": 5.36,
             "response_utterance_id": "assistant_long", "context_utterance_id": "user_elaboration"},
            {"action": "YIELD", "start_s": 6.24, "end_s": 6.32,
             "response_utterance_id": None, "context_utterance_id": "assistant_long",
             "trigger_utterance_id": "user_correction"},
            {"action": "REPLAN", "start_s": 7.68, "end_s": 7.76,
             "response_utterance_id": "assistant_replan", "context_utterance_id": "user_correction",
             "wait_after_correction_ms": 800.0},
        ],
    }


class ActionConditionedDataTests(unittest.TestCase):
    def test_v5_candidate_is_pcm_verified_and_merged_with_clean_asr_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate = root / "synthetic_it_t6_audio_grounded_v5_smoke_seed260928"
            candidate.mkdir()
            wav_root = candidate / "audio_grounded_v5" / "audio"
            counts = {"train": 1, "validation": 1, "test": 1}
            wav_rows = []
            for split_index, split in enumerate(counts):
                audio_path = wav_root / split / f"v5-{split}.wav"
                audio_path.parent.mkdir(parents=True)
                audio = np.zeros((9 * 16000, 2), dtype=np.float32)
                audio[split_index, 0] = 0.25
                sf.write(audio_path, audio, 16000, subtype="PCM_16")
                row = _v5_row()
                row.update({
                    "conversation_id": f"v5-{split}",
                    "topic_group_id": f"topic-{split}",
                    "split": split,
                    "audio_path": str(audio_path),
                    "audio_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                })
                wav_rows.append(row)
            plan_dialogues = [
                {
                    "conversation_id": f"v5-{split}" if index == 0 else f"planned-{split}-{index:02d}",
                    "split": split,
                    "topic_group_id": f"topic-{split}",
                }
                for split, total in (("train", 32), ("validation", 8), ("test", 8))
                for index in range(total)
            ]
            (candidate / "dialogue_plan.json").write_text(json.dumps({
                "profile": "audio_grounded_v5",
                "dialogues": plan_dialogues,
                "split_metadata": {"dialogue_counts": {
                    "train": 32, "validation": 8, "test": 8,
                }},
            }), encoding="utf-8")
            (candidate / "manifest_audio_grounded_v5.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in wav_rows), encoding="utf-8"
            )

            base = root / "base_prepared"
            base.mkdir()
            for split in counts:
                asr_rows = []
                row_count = 32 if split == "validation" else 1
                for index in range(row_count):
                    clean_asr = {
                        "conversation_id": f"clean-asr-{split}-{index:02d}",
                        # The 32 validation utterances are sampled inside two source groups,
                        # preserving the original group-held-out split instead of claiming
                        # 32 independent source groups.
                        "source_group_id": (
                            f"clean-asr-group-validation-{index % 2}"
                            if split == "validation" else f"clean-asr-group-{split}"
                        ),
                        "source": "synthetic_v3_clean_asr",
                        "split": split,
                        "audio_path": str(wav_root / split / f"v5-{split}.wav"),
                        "synthetic": False,
                        "sample_rate_hz": 16000,
                        "duration_ms": 9000,
                        "language": "it-IT",
                        "channel_map": {"0": "user", "1": "assistant"},
                        "segments": [{
                            "segment_id": f"asr-utterance-{split}-{index:02d}",
                            "task_kind": "asr",
                            "role": None,
                            "channel_index": 0,
                            "audio_channel_verified": True,
                            "text": f"Testo ASR {split} {index}.",
                            "start_ms": 0,
                            "end_ms": 500,
                            "time_mask": True,
                            "text_mask": True,
                        }],
                    }
                    asr_rows.append(json.dumps(clean_asr))
                (base / f"{split}.jsonl").write_text("\n".join(asr_rows) + "\n", encoding="utf-8")

            output = root / "prepared_v5_clean_asr_smoke_20261002"
            report = prepare_audio_grounded_v5_dataset(candidate, base, output)
            self.assertTrue(report["group_split_disjoint"])
            self.assertEqual(report["purpose"], "smoke")
            self.assertTrue(report["v5_smoke_rows_included"])
            for split in counts:
                rows = [json.loads(line) for line in (output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(rows), 33 if split == "validation" else 2)
                self.assertEqual({row["source"] for row in rows}, {
                    "authored_synthetic_audio_candidate", "synthetic_v3_clean_asr",
                })
            response_dir = root / "audio_responses_smoke_20261002"
            response_manifest = prepare_action_conditioned_dataset(output, response_dir)
            self.assertEqual(response_manifest["dataset_purpose"], "smoke")
            mix_dir = root / "audio_asr_mix_smoke_20261002"
            mix_manifest = prepare_audio_asr_mix_dataset(
                output, response_dir, mix_dir, seed=4,
            )
            self.assertEqual(mix_manifest["purpose"], "smoke")
            self.assertEqual(mix_manifest["splits"]["validation"]["asr_validation_examples"], 32)
            self.assertEqual(mix_manifest["splits"]["validation"]["asr_validation_unique_conversations"], 32)
            self.assertEqual(mix_manifest["splits"]["validation"]["asr_validation_source_groups"], 2)
            train_rows = [json.loads(line) for line in (mix_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(row["task_kind"] == "asr" for row in train_rows), 1)
            self.assertEqual({row["action"] for row in train_rows if row["task_kind"] == "conversation_response"}, {
                "SPEAK", "BACKCHANNEL", "REPLAN",
            })

            v2_response_dir = root / "audio_responses_v2_smoke_20261002"
            v2_response_manifest = prepare_action_conditioned_dataset(
                output, v2_response_dir, response_protocol="heard_action_say_v2",
            )
            self.assertEqual(v2_response_manifest["audio_response_protocol"], "heard_action_say_v2")
            v2_train_responses = [
                json.loads(line) for line in (v2_response_dir / "train.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(v2_train_responses)
            self.assertTrue(all(row["audio_response_protocol"] == "heard_action_say_v2" for row in v2_train_responses))
            self.assertTrue(all("current_user_transcript" not in row for row in v2_train_responses))
            for row in v2_train_responses:
                target = json.loads(row["target_text"])
                self.assertEqual(list(target), ["heard", "action", "say"])
                self.assertEqual(target["action"], row["action"])
                self.assertEqual(target["say"], row["target_say_text"])
                self.assertEqual(target["heard"], row["target_heard_text"])

            v2_mix_dir = root / "audio_asr_mix_v2_smoke_20261002"
            v2_mix_manifest = prepare_audio_asr_mix_dataset(
                output, v2_response_dir, v2_mix_dir, seed=4, response_protocol="v2",
            )
            self.assertEqual(v2_mix_manifest["task_mix_contract"], "audio_asr_mix_v2")
            v1_asr_targets = {
                split: [row["target_text"] for row in map(json.loads, (mix_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
                        if row.get("task_kind") == "asr"]
                for split in ("train", "validation", "test")
            }
            v2_asr_targets = {
                split: [row["target_text"] for row in map(json.loads, (v2_mix_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines())
                        if row.get("task_kind") == "asr"]
                for split in ("train", "validation", "test")
            }
            self.assertEqual(v2_asr_targets, v1_asr_targets)

    def test_audio_grounded_v5_builds_causal_speak_backchannel_replan_targets(self) -> None:
        records, report = build_action_conditioned_records([_v5_row()], split="train")

        self.assertEqual([row["action"] for row in records], ["SPEAK", "BACKCHANNEL", "SPEAK", "REPLAN"])
        self.assertEqual(report["records_by_action"], {
            "BACKCHANNEL": 1, "REPLAN": 1, "SPEAK": 2, "UNKNOWN": 0,
        })
        by_action = {row["target_utterance_id"]: row for row in records}
        backchannel = by_action["assistant_backchannel"]
        self.assertEqual(backchannel["source"], "authored_synthetic_audio_candidate")
        self.assertEqual(backchannel["audio_end_ms"] - backchannel["audio_start_ms"], 800.0)
        self.assertNotIn("Un dettaglio che resta segreto nel prompt.", str(backchannel["history_messages"]))
        self.assertNotIn("Correzione da non mostrare.", str(backchannel["history_messages"]))
        self.assertEqual(backchannel["target_text"], "Capito.")
        replan = by_action["assistant_replan"]
        self.assertEqual(replan["audio_end_ms"], replan["target_start_ms"])
        self.assertNotIn("Correzione da non mostrare.", str(replan["history_messages"]))
        self.assertIn("La risposta iniziale.", str(replan["history_messages"]))
        self.assertNotIn("La risposta interrotta.", str(replan["history_messages"]))

    def test_action_conditioning_is_plain_text_and_closed_to_three_actions(self) -> None:
        for action in ("BACKCHANNEL", "SPEAK", "REPLAN"):
            prompt = action_conditioning_instruction(action.lower())
            self.assertTrue(prompt)
            self.assertNotIn("<|", prompt)
            self.assertNotIn("|>", prompt)
        with self.assertRaises(ValueError):
            action_conditioning_instruction("LISTEN")

    def test_builder_aligns_actions_and_excludes_unfinished_user_transcript_from_backchannel(self) -> None:
        records, report = build_action_conditioned_records([_row()], split="train")

        self.assertEqual([record["action"] for record in records], ["SPEAK", "BACKCHANNEL"])
        self.assertEqual(report["records_by_action"], {"BACKCHANNEL": 1, "REPLAN": 0, "SPEAK": 1, "UNKNOWN": 0})
        self.assertTrue(all(record["source_audio_sha256"] == "a" * 64 for record in records))
        backchannel = next(record for record in records if record["action"] == "BACKCHANNEL")
        self.assertNotIn("Trascrizione futura da non mostrare.", str(backchannel["history_messages"]))
        self.assertNotIn("Correzione completa.", str(backchannel["history_messages"]))
        self.assertTrue(any(message["content"] == "Risposta iniziale." for message in backchannel["history_messages"]))
        self.assertTrue(all("Incomplete e sovrapposta." not in str(record["history_messages"]) for record in records))

    def test_builder_keeps_action_unknown_when_event_evidence_is_missing(self) -> None:
        row = _row()
        row["source_action_segments"] = [
            event for event in row["source_action_segments"] if event["action"] != "BACKCHANNEL"
        ]
        row["segments"][2]["end_ms"] = 2100
        records, report = build_action_conditioned_records([row], split="train")
        backchannel_target = next(record for record in records if record["target_utterance_id"] == "a1")
        self.assertEqual(backchannel_target["action"], "UNKNOWN")
        self.assertEqual(backchannel_target["action_label"], "unknown")
        self.assertEqual(backchannel_target["sampling_stratum"], "conversational_regular")
        self.assertIsNone(backchannel_target["action_event_start_ms"])
        self.assertEqual(report["records_by_action"]["UNKNOWN"], 1)

    def test_overlapping_backchannel_precedes_user_turn_in_completion_order(self) -> None:
        row = _row()
        row["source_action_segments"] = [
            {"action": "SPEAK", "start_ms": 1200, "end_ms": 1400},
        ]
        row["segments"] = [
            {"role": "user", "segment_id": "u0", "start_ms": 0, "end_ms": 1000,
             "text": "La frase completa.", "time_mask": True, "text_mask": True},
            {"role": "assistant", "script_role": "assistant_backchannel", "segment_id": "bc0",
             "start_ms": 500, "end_ms": 700, "text": "Certo.", "time_mask": True,
             "text_mask": True, "target_mask": False},
            {"role": "assistant", "script_role": "assistant_start", "segment_id": "s0",
             "start_ms": 1200, "end_ms": 1500, "text": "Risposta sostanziale.", "time_mask": True,
             "text_mask": True, "target_mask": True},
        ]
        row["channel_map"] = {"0": "user", "1": "assistant"}

        action_records, _ = build_action_conditioned_records([row], split="train")
        action_history = action_records[0]["history_messages"]
        self.assertEqual(
            action_history,
            [{"role": "assistant", "content": "Certo."}],
        )  # completed prior assistant BC is context; current user transcript stays hidden
        expected_runtime_text_history = [
            {"role": "assistant", "content": "Certo."},
            {"role": "user", "content": "La frase completa."},
        ]

        stage1_row = {
            "conversation_id": "stage1-overlap",
            "source": "synthetic_it_t6",
            "synthetic": True,
            "channel_map": {"0": "user", "1": "assistant"},
            "segments": [
                {"speaker": "user", "synthetic_role": True, "channel": "0", "start_ms": 0,
                 "end_ms": 1000, "text": "La frase completa.", "time_mask": True, "text_mask": True},
                {"speaker": "assistant", "synthetic_role": True, "channel": "1", "start_ms": 500,
                 "end_ms": 700, "text": "Certo.", "time_mask": True, "text_mask": True},
                {"speaker": "assistant", "synthetic_role": True, "channel": "1", "start_ms": 1200,
                 "end_ms": 1500, "text": "Risposta sostanziale.", "time_mask": True, "text_mask": True},
            ],
        }
        stage1_records = _training_records([stage1_row], conditioning_mode="runtime_asr_text")
        stage1_history = stage1_records[0]["history_messages"]
        self.assertEqual(stage1_history, expected_runtime_text_history)

    def test_backchannel_lexicon_uses_only_short_unique_training_targets(self) -> None:
        records = [
            {"split": "train", "action": "BACKCHANNEL", "target_text": "Certo."},
            {"split": "train", "action": "BACKCHANNEL", "target_text": "certo."},
            {"split": "train", "action": "BACKCHANNEL", "target_text": "Sì, va bene."},
            {"split": "train", "action": "BACKCHANNEL", "target_text": "Perché va bene?"},
            {"split": "train", "action": "BACKCHANNEL", "target_text": ""},
            {"split": "train", "action": "BACKCHANNEL", "target_text": "Uno due tre quattro."},
            {"split": "train", "action": "BACKCHANNEL", "target_text": "<|turn|>"},
            {"split": "validation", "action": "BACKCHANNEL", "target_text": "Okay."},
            {"split": "train", "action": "SPEAK", "target_text": "Questa va ignorata."},
        ]

        lexicon = build_backchannel_lexicon(records)

        self.assertEqual(lexicon["schema_version"], 1)
        self.assertEqual(lexicon["language"], "it")
        self.assertEqual(lexicon["phrases"], ["Certo.", "Sì, va bene."])
        self.assertEqual(lexicon["counts"]["target_records"], 7)
        self.assertEqual(lexicon["counts"]["candidate_count"], 2)
        self.assertEqual(lexicon["counts"]["covered_target_records"], 3)
        self.assertEqual(lexicon["counts"]["excluded_target_rows_by_reason"], {
            "empty": 1,
            "internal_marker": 1,
            "question": 1,
            "over_3_words": 1,
        })


if __name__ == "__main__":
    unittest.main()
