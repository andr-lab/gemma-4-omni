from __future__ import annotations

import os
import unittest
from pathlib import Path

from duplex_kernel.action_conditioning import action_conditioned_messages, action_control_event


class ActionConditionedMessageTests(unittest.TestCase):
    def test_appends_empty_ephemeral_user_turn_after_any_causal_history_without_mutating_it(self):
        histories = [
            [],
            [{"role": "user", "content": "Richiesta completata."}],
            [{"role": "assistant", "content": "Risposta completata."}],
            [
                {"role": "user", "content": "Richiesta."},
                {"role": "assistant", "content": "Prima frase."},
                {"role": "assistant", "content": "Seconda frase."},
            ],
        ]
        for history in histories:
            original = [dict(item) for item in history]
            for action in ("BACKCHANNEL", "SPEAK", "REPLAN"):
                with self.subTest(history_roles=[item["role"] for item in history], action=action):
                    messages = action_conditioned_messages("Rispondi in italiano.", action, history)
                    self.assertEqual([item["role"] for item in messages[1:-1]], [item["role"] for item in history])
                    self.assertEqual(action_control_event(action), "")
                    self.assertEqual(messages[-1], {"role": "user", "content": ""})
                    self.assertEqual(history, original)
                    self.assertEqual(messages[0]["role"], "system")
                    self.assertIn(action, messages[0]["content"])
                    self.assertEqual(messages[-1]["content"], "")

    def test_rejects_non_conversational_or_malformed_history_messages(self):
        with self.assertRaises(ValueError):
            action_conditioned_messages("base", "SPEAK", [{"role": "system", "content": "stale"}])
        with self.assertRaises(ValueError):
            action_conditioned_messages("base", "SPEAK", [{"role": "user", "content": None}])


class GemmaActionPromptPrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_path = Path(
            os.environ.get(
                "GEMMA4_TEST_MODEL_PATH",
                "/workspace/.hf_cache/models--google--gemma-4-E2B-it/full",
            )
        )
        if not model_path.is_dir():
            raise unittest.SkipTest(f"Gemma tokenizer locale non disponibile: {model_path}")
        from transformers import AutoProcessor

        cls.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

    def test_prompt_ids_are_exact_prefix_of_full_chat_for_all_history_endings_and_actions(self):
        from training.duplex_qlora.stage1 import _format_chat

        histories = [
            [],
            [{"role": "user", "content": "Messaggio utente completato."}],
            [{"role": "assistant", "content": "Messaggio assistant completato."}],
            [
                {"role": "user", "content": "Richiesta completata."},
                {"role": "assistant", "content": "Prima risposta completata."},
                {"role": "assistant", "content": "Seconda risposta completata."},
            ],
        ]
        for history in histories:
            for action in ("BACKCHANNEL", "SPEAK", "REPLAN"):
                record = {
                    "task_kind": "conversation_response",
                    "conditioning_mode": "action_conditioned_text",
                    "audio_mask": False,
                    "action": action,
                    "language": "Italian",
                    "history_messages": history,
                    "target_text": "Risposta di prova.",
                }
                prompt = _format_chat(self.processor, record, add_target=False)
                full = _format_chat(self.processor, record, add_target=True)
                prompt_ids = self.processor(text=[prompt], return_tensors="pt", padding=True)["input_ids"][0].tolist()
                full_ids = self.processor(text=[full], return_tensors="pt", padding=True)["input_ids"][0].tolist()
                with self.subTest(history_roles=[item["role"] for item in history], action=action):
                    self.assertLess(len(prompt_ids), len(full_ids))
                    self.assertEqual(full_ids[: len(prompt_ids)], prompt_ids)

    def test_empty_internal_user_event_is_also_a_prefix_for_the_ablation(self):
        histories = [
            [],
            [{"role": "user", "content": "Messaggio utente completato."}],
            [{"role": "assistant", "content": "Messaggio assistant completato."}],
            [
                {"role": "user", "content": "Richiesta completata."},
                {"role": "assistant", "content": "Prima risposta completata."},
                {"role": "assistant", "content": "Seconda risposta completata."},
            ],
        ]
        tokenizer = self.processor.tokenizer
        for history in histories:
            for action in ("BACKCHANNEL", "SPEAK", "REPLAN"):
                messages = action_conditioned_messages("Rispondi in italiano.", action, history)
                messages[-1]["content"] = ""
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                full_messages = [*messages, {"role": "assistant", "content": "Risposta di prova."}]
                full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
                prompt_ids = self.processor(text=[prompt], return_tensors="pt", padding=True)["input_ids"][0].tolist()
                full_ids = self.processor(text=[full], return_tensors="pt", padding=True)["input_ids"][0].tolist()
                with self.subTest(history_roles=[item["role"] for item in history], action=action):
                    self.assertLess(len(prompt_ids), len(full_ids))
                    self.assertEqual(full_ids[: len(prompt_ids)], prompt_ids)


if __name__ == "__main__":
    unittest.main()
