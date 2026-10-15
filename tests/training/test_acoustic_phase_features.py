from __future__ import annotations

import numpy as np
import unittest

from duplex_kernel.acoustic_phase_features import (
    ACOUSTIC_PHASE_FEATURE_NAMES,
    acoustic_phase_features,
)


class AcousticPhaseFeaturesTests(unittest.TestCase):
    def test_use_only_causal_pcm_and_named_order(self) -> None:
        user_history = np.full(5 * 1280, 0.1, dtype=np.float32)
        assistant_previous = np.full(1280, 0.25, dtype=np.float32)

        values = acoustic_phase_features(user_history, assistant_previous)

        self.assertEqual(
            ACOUSTIC_PHASE_FEATURE_NAMES,
            (
                "user_log10_rms_current_80ms",
                "user_log10_rms_past_400ms_inclusive",
                "assistant_log10_rms_prior_80ms",
                "user_log_rms_delta_clip6_div6",
            ),
        )
        self.assertEqual(values.shape, (4,))
        self.assertEqual(values.dtype, np.float32)
        np.testing.assert_allclose(values, [-1.0, -1.0, np.log10(0.25), 0.0], atol=1e-6)


    def test_left_pad_stream_start_and_bound_delta(self) -> None:
        current_user = np.full(1280, 0.1, dtype=np.float32)
        assistant_missing = None

        values = acoustic_phase_features(current_user, assistant_missing)

        self.assertAlmostEqual(values[0], -1.0, delta=1e-6)
        self.assertAlmostEqual(values[1], -1.0 - 0.5 * np.log10(5.0), delta=1e-6)
        self.assertAlmostEqual(values[2], -6.0, delta=1e-6)
        self.assertAlmostEqual(values[3], 5.0 / 6.0, delta=1e-6)


    def test_compare_current_to_previous_user_tick(self) -> None:
        previous_user = np.full(1280, 0.01, dtype=np.float32)
        current_user = np.full(1280, 0.1, dtype=np.float32)
        history = np.concatenate((previous_user, current_user))

        values = acoustic_phase_features(history, np.zeros(1280, dtype=np.float32))

        self.assertAlmostEqual(values[0], -1.0, delta=1e-6)
        self.assertAlmostEqual(values[3], 1.0 / 6.0, delta=1e-6)


    def test_reject_invalid_pcm_contract(self) -> None:
        invalid = (
            (np.zeros(1279, dtype=np.float32), None),
            (np.zeros(6401, dtype=np.float32), None),
            (np.zeros(1280, dtype=np.float32), np.zeros(1279, dtype=np.float32)),
            (np.full(1280, np.nan, dtype=np.float32), None),
        )
        for history, assistant in invalid:
            with self.subTest(history_size=len(history), assistant_size=None if assistant is None else len(assistant)):
                with self.assertRaises(ValueError):
                    acoustic_phase_features(history, assistant)
