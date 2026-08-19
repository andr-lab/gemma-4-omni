# -*- coding: utf-8 -*-
"""
_test_sampling2.py -- TDD suite for per-level temperature sampling in Qwen3TTSSubTalker.
Tests per-level temperature configuration, length validation, reproducibility,
greedy equivalence, and ASCII encoding integrity.
"""
import sys
import torch
import qwen_subtalker as QS

def run_tests():
    print("=== RUNNING SAMPLING2 TDD SUITE (PER-LEVEL TEMPERATURE) ===")
    torch.manual_seed(42)
    cfg = dict(QS.DEFAULT_CONFIG)
    cfg["num_hidden_layers"] = 2
    model = QS.Qwen3TTSSubTalker(cfg).eval()

    B, T = 2, 4
    h_t = torch.randn(B, T, cfg["talker_hidden_size"])
    cb0 = torch.randint(0, cfg["vocab_size"], (B, T))

    failures = []
    def check(name, cond, extra=""):
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}" + (f" | {extra}" if extra else ""))
        if not cond:
            failures.append((name, extra))

    # Baseline greedy output
    out_greedy = model.sample_acoustic_codes(h_t, cb0, temperature=0.0)

    # Test A: temperatures_per_level with 15 values > 0 -> shape (B, T, 15) and range [0, vocab_size)
    try:
        temps_15 = [0.8] * 15
        out_15 = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_15, seed=123
        )
        check("Test A1: Shape with 15 temperatures is (B, T, 15)", out_15.shape == (B, T, 15), f"got {out_15.shape}")
        check(
            "Test A2: Output tokens in vocab range [0, 2048)",
            bool((out_15 >= 0).all() and (out_15 < cfg["vocab_size"]).all()),
            f"min={int(out_15.min())}, max={int(out_15.max())}"
        )
    except Exception as e:
        check("Test A: 15 positive temperatures", False, str(e))

    # Test B: temperatures_per_level=[0.0]*15 -> EXACT match with greedy (temperature=0.0)
    try:
        temps_zero = [0.0] * 15
        out_zero = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_zero
        )
        check(
            "Test B: [0.0]*15 matches greedy exactly",
            torch.equal(out_zero, out_greedy),
            f"equal={torch.equal(out_zero, out_greedy)}"
        )
    except Exception as e:
        check("Test B: [0.0]*15 equivalence", False, str(e))

    # Test C: Single level > 0 and others 0.0
    # C1: Last level (level 15, index 14) > 0, levels 1..14 (indices 0..13) == 0.0
    # Levels 1..14 must match greedy EXACTLY; level 15 differs stochastically.
    try:
        temps_last_only = [0.0] * 14 + [0.9]
        out_last_only = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_last_only, seed=999
        )
        levels_1_to_14_match = torch.equal(out_last_only[:, :, :14], out_greedy[:, :, :14])
        level_15_diff = (out_last_only[:, :, 14] != out_greedy[:, :, 14]).sum().item()
        check(
            "Test C1a: Levels 1..14 match greedy when only level 15 has temp > 0",
            levels_1_to_14_match,
            f"levels 1..14 match={levels_1_to_14_match}"
        )
        check(
            "Test C1b: Level 15 differs from greedy when level 15 has temp > 0",
            level_15_diff > 0,
            f"level 15 differing tokens: {level_15_diff}/{B*T}"
        )
    except Exception as e:
        check("Test C1: Single level > 0 (last level)", False, str(e))

    # C2: First level (level 1, index 0) > 0, levels 2..15 (indices 1..14) == 0.0
    try:
        temps_first_only = [0.9] + [0.0] * 14
        out_first_only = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_first_only, seed=999
        )
        level_1_diff = (out_first_only[:, :, 0] != out_greedy[:, :, 0]).sum().item()
        check(
            "Test C2: Level 1 differs from greedy when level 1 has temp > 0",
            level_1_diff > 0,
            f"level 1 differing tokens: {level_1_diff}/{B*T}"
        )
    except Exception as e:
        check("Test C2: Single level > 0 (first level)", False, str(e))

    # Test D: temperatures_per_level wrong length (14 or 16) -> ValueError
    try:
        raised_14 = False
        try:
            model.sample_acoustic_codes(h_t, cb0, temperatures_per_level=[0.8] * 14)
        except ValueError:
            raised_14 = True
        check("Test D1: Length 14 raises ValueError", raised_14)

        raised_16 = False
        try:
            model.sample_acoustic_codes(h_t, cb0, temperatures_per_level=[0.8] * 16)
        except ValueError:
            raised_16 = True
        check("Test D2: Length 16 raises ValueError", raised_16)
    except Exception as e:
        check("Test D: Invalid length ValueError", False, str(e))

    # Test E: Reproducibility with same seed and temperatures_per_level
    try:
        temps_mixed = [0.5 + 0.03 * i for i in range(15)]
        out_s1 = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_mixed, seed=777
        )
        out_s2 = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_mixed, seed=777
        )
        check(
            "Test E: Reproducibility with same seed and temperatures_per_level",
            torch.equal(out_s1, out_s2),
            f"equal={torch.equal(out_s1, out_s2)}"
        )
    except Exception as e:
        check("Test E: Reproducibility", False, str(e))

    # Test F: Retrocompatibility: temperatures_per_level=None uses scalar temperature
    try:
        out_none = model.sample_acoustic_codes(
            h_t, cb0, temperature=0.9, temperatures_per_level=None, seed=42
        )
        out_scalar = model.sample_acoustic_codes(
            h_t, cb0, temperature=0.9, seed=42
        )
        check(
            "Test F: temperatures_per_level=None matches scalar temperature",
            torch.equal(out_none, out_scalar),
            f"equal={torch.equal(out_none, out_scalar)}"
        )
    except Exception as e:
        check("Test F: Retrocompatibility None", False, str(e))

    # Test G: Tuple input support for Sequence[float]
    try:
        temps_tuple = tuple([0.5] * 15)
        out_tuple = model.sample_acoustic_codes(
            h_t, cb0, temperatures_per_level=temps_tuple, seed=123
        )
        check(
            "Test G: Sequence (tuple) input support",
            out_tuple.shape == (B, T, 15)
        )
    except Exception as e:
        check("Test G: Tuple input", False, str(e))

    # Test H: ASCII pure check
    try:
        max_b_qs = max(open("qwen_subtalker.py", "rb").read())
        max_b_t2 = max(open("_test_sampling2.py", "rb").read())
        check(
            "Test H: ASCII byte check (< 128)",
            max_b_qs < 128 and max_b_t2 < 128,
            f"qwen_subtalker.py max byte={max_b_qs}, _test_sampling2.py max byte={max_b_t2}"
        )
    except Exception as e:
        check("Test H: ASCII byte check", False, str(e))

    print("\n" + "=" * 50)
    if failures:
        print(f"RESULT: RED ({len(failures)} failures)")
        for f, extra in failures:
            print(f"  - {f}: {extra}")
        return 1
    else:
        print("RESULT: GREEN (All tests passed)")
        return 0

if __name__ == "__main__":
    sys.exit(run_tests())
