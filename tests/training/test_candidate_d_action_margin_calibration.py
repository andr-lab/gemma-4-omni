import unittest

import numpy as np

from tools.calibrate_candidate_d_action_margins import (
    apply_action_margins,
    exhaustive_onset_threshold_search,
)
from training.duplex_qlora.schema import ACTIONS


class ActionMarginPolicyTests(unittest.TestCase):
    def test_exact_threshold_keeps_raw_event_and_strictly_higher_suppresses(self):
        actions = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
        logits = np.asarray(
            [
                [1.0, 1.5, 0.8, 0.0, 0.0, 0.0],
                [1.0, 1.4999, 0.8, 0.0, 0.0, 0.0],
                [1.0, 0.4, 1.4, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        thresholds = {name: 0.5 for name in ("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")}

        predicted = apply_action_margins(logits, actions, thresholds)

        self.assertEqual([actions[index] for index in predicted], ["SPEAK", "LISTEN", "CONTINUE"])

    def test_threshold_contract_requires_exact_eligible_action_keys(self):
        actions = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
        logits = np.zeros((1, len(actions)), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "exactly"):
            apply_action_margins(logits, actions, {"SPEAK": 0.0})

    def test_nonfinite_threshold_is_rejected(self):
        actions = ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
        logits = np.zeros((1, len(actions)), dtype=np.float32)
        thresholds = {name: 0.0 for name in ("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")}
        thresholds["YIELD"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            apply_action_margins(logits, actions, thresholds)

    def test_exhaustive_search_keeps_each_event_and_suppresses_background_false_events(self):
        actions = tuple(ACTIONS)
        index = {action: i for i, action in enumerate(actions)}
        truth_names = ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN", "CONTINUE"] + [
            "LISTEN", "CONTINUE", "LISTEN", "CONTINUE", "LISTEN",
            "CONTINUE", "LISTEN", "CONTINUE", "LISTEN", "CONTINUE",
        ]
        truth = np.asarray([index[action] for action in truth_names], dtype=np.int64)
        logits = np.zeros((len(truth), len(actions)), dtype=np.float32)
        logits[:, index["LISTEN"]] = 0.1
        logits[:, index["CONTINUE"]] = 0.0
        for row, action in enumerate(("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")):
            logits[row, index[action]] = 0.6
        logits[4, index["CONTINUE"]] = 0.2
        # Two off-onset rows look like SPEAK but fall below the onset margin.
        logits[5, index["SPEAK"]] = 0.3
        logits[7, index["SPEAK"]] = 0.3
        onset_mask = np.zeros(len(truth), dtype=bool)
        onset_mask[:5] = True
        background_mask = ~onset_mask

        thresholds, result, scope = exhaustive_onset_threshold_search(
            logits, truth, actions, onset_mask, background_mask
        )

        self.assertEqual(set(thresholds), {"SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"})
        self.assertEqual(result["event_onset_metrics"]["macro_f1_present_classes"], 1.0)
        self.assertEqual(result["background_false_event_count"], 0)
        self.assertTrue(scope["global_optimum_within_scope"])


if __name__ == "__main__":
    unittest.main()
