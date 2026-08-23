# -*- coding: utf-8 -*-
"""
Validation and regression test suite for TASK AGY 11B (Adapter v2 with Real Voices).
Checks script syntax, report structure, numerical quality gates, and checkpoint files.
"""
import ast
import json
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.isfile(os.path.join(cur_dir, "train_adapter_it_3090.py")):
    ROOT = cur_dir
else:
    ROOT = os.path.abspath(os.path.join(cur_dir, "..", ".."))

def test_script_syntax_and_ast():
    """Verify train_adapter_it_3090.py compiles and has clean AST structure."""
    script_path = os.path.join(ROOT, "train_adapter_it_3090.py")
    assert os.path.isfile(script_path), f"Missing script: {script_path}"
    with open(script_path, "r", encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, filename="train_adapter_it_3090.py")
    assert tree is not None, "Failed to parse AST of train_adapter_it_3090.py"
    print("[PASS] train_adapter_it_3090.py AST parsed successfully.")

def test_report_json_quality_gates():
    """Validate logs/adapter_it_v2_training_report.json against strict quality gates."""
    report_path = os.path.join(ROOT, "logs", "adapter_it_v2_training_report.json")
    assert os.path.isfile(report_path), f"Missing report JSON: {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    assert data.get("status") == "COMPLETED", f"Status is not COMPLETED: {data.get('status')}"
    assert data.get("total_steps") == 2374, f"Unexpected step count: {data.get('total_steps')}"
    
    # Initial loss check (healthy, non-collapsed)
    init_loss = data.get("initial_loss")
    assert init_loss is not None and 5.0 <= init_loss <= 35.0, f"Initial loss abnormal: {init_loss}"
    
    # Val loss & convergence
    val_loss = data.get("val_loss")
    final_loss = data.get("final_train_loss")
    assert val_loss is not None and val_loss < init_loss, f"No convergence in val loss: {val_loss} vs {init_loss}"
    assert final_loss is not None and final_loss < init_loss, f"No convergence in train loss: {final_loss} vs {init_loss}"
    
    # Cosine Similarity check
    cos_sim = data.get("cosine_sim_val")
    assert cos_sim is not None, "Missing cosine_sim_val"
    assert cos_sim >= 0.80, f"Cosine similarity below threshold (0.80): {cos_sim}"
    assert abs(cos_sim - 0.9366) < 0.005, f"Cosine similarity mismatch: expected ~0.9366, got {cos_sim}"
    
    # SVD Metrics check
    svd = data.get("svd_metrics", {})
    eff_rank = svd.get("effective_rank", 0)
    dim = svd.get("dim", 2048)
    cum_energy = svd.get("cum_energy_1000", 0.0)
    assert eff_rank >= 1000, f"SVD rank collapse detected: {eff_rank} / {dim} < 1000"
    assert cum_energy >= 0.95, f"SVD cum energy below 0.95: {cum_energy}"
    
    print(f"[PASS] Quality Gates in JSON: cos_sim={cos_sim:.4f}, SVD={eff_rank}/{dim}, cum_energy={cum_energy}")

def test_markdown_report_presence():
    """Verify docs/REPORT_TASK11B_ADAPTER_V2.md exists and contains all required sections."""
    doc_path = os.path.join(ROOT, "docs", "REPORT_TASK11B_ADAPTER_V2.md")
    assert os.path.isfile(doc_path), f"Missing report MD: {doc_path}"
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    required_keywords = [
        "TASK 11B",
        "0.9366",
        "1510",
        "0.96%",
        "gemma4_duplex_sota_v2",
        "gemma4_duplex_sota",
        "check_voice_quality.py",
        "100% PASS",
    ]
    for kw in required_keywords:
        assert kw in content, f"Keyword '{kw}' missing from REPORT_TASK11B_ADAPTER_V2.md"
    print("[PASS] docs/REPORT_TASK11B_ADAPTER_V2.md verified.")

def test_synthesis_metrics():
    """Verify samples/synthesis_metrics.txt contains correct Mel-L1 and MCD values."""
    metrics_path = os.path.join(ROOT, "samples", "synthesis_metrics.txt")
    assert os.path.isfile(metrics_path), f"Missing synthesis metrics: {metrics_path}"
    with open(metrics_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    medie_line = [l for l in lines if l.startswith("MEDIE")]
    assert len(medie_line) == 1, "MEDIE line missing in synthesis_metrics.txt"
    parts = medie_line[0].strip().split("\t")
    mel_l1 = float(parts[1])
    mcd = float(parts[2])
    
    assert mel_l1 < 0.80, f"Mel-L1 too high: {mel_l1}"
    assert mcd < 5.0, f"MCD too high: {mcd} dB"
    print(f"[PASS] Synthesis metrics verified: Mel-L1={mel_l1}, MCD={mcd} dB")

def main():
    test_script_syntax_and_ast()
    test_report_json_quality_gates()
    test_markdown_report_presence()
    test_synthesis_metrics()
    print("\nALL TEST GATES PASSED (GREEN)!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
