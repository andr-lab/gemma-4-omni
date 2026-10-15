import unittest

from training.duplex_qlora.action_delta import _generation_observation, _summarize_generations


class ActionDeltaGenerationTests(unittest.TestCase):
    def test_backchannel_short_candidate_requires_eos_before_budget(self):
        eos = _generation_observation(
            action="BACKCHANNEL",
            token_ids=[101, 102, 2],
            eos_token_ids={2},
            token_budget=16,
            generated_text="Capisco, certo.",
        )
        self.assertTrue(eos["eos_terminated"])
        self.assertFalse(eos["reached_budget_without_eos"])
        self.assertTrue(eos["backchannel_1_to_4_word_candidate"])
        self.assertFalse(eos["backchannel_full_answer_length_risk"])

    def test_backchannel_budget_exhaustion_and_long_answer_are_flagged(self):
        truncated = _generation_observation(
            action="BACKCHANNEL",
            token_ids=list(range(101, 117)),
            eos_token_ids={2},
            token_budget=16,
            generated_text="Va bene, capisco, allora possiamo rivedere insieme tutti i dettagli.",
        )
        self.assertFalse(truncated["eos_terminated"])
        self.assertTrue(truncated["reached_budget_without_eos"])
        self.assertFalse(truncated["backchannel_1_to_4_word_candidate"])
        self.assertTrue(truncated["backchannel_full_answer_length_risk"])

    def test_non_backchannel_generations_are_not_scored_as_acknowledgments(self):
        speak = _generation_observation(
            action="SPEAK",
            token_ids=[21, 22, 2],
            eos_token_ids={2},
            token_budget=96,
            generated_text="Possiamo spostare la riunione.",
        )
        summary = _summarize_generations([{"action": "SPEAK", **speak}])
        self.assertFalse(speak["backchannel_1_to_4_word_candidate"])
        self.assertEqual(summary["SPEAK"]["eos_terminated"], 1)
        self.assertEqual(summary["BACKCHANNEL"]["examples"], 0)


if __name__ == "__main__":
    unittest.main()
