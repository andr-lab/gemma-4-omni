# -*- coding: utf-8 -*-
import json, ast, os

def verify():
    nb_path = "kernel_window128_r1/gemma4-window128-r1.ipynb"
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) == 3, f"Expected 3 cells, got {len(nb['cells'])}"
    for i, cell in enumerate(nb["cells"]):
        assert isinstance(cell["source"], list), f"Cell {i} source is not list"
        code = "".join(cell["source"])
        ast.parse(code)
        print(f"Cell {i} ({cell['id']}) AST parsed OK, lines: {len(cell['source'])}")

    full_code = "".join(nb["cells"][2]["source"])

    probes = [
        "WINDOW128_ACTIVE",
        "T=128",
        "layers=",
        "warm-start ht_adapter_stage2r2_final",
        "riuso Gemma E2B",
        "_tdev",
        "PATCH sample_acoustic_codes",
        "EVAL_WINDOW128"
    ]

    print("\n--- VERIFICA PROBES WINDOW 128 ---")
    for p in probes:
        assert p in full_code, f"PROBE MANCANTE: {p}"
        print(f"Probe '{p}': PRESENTE")

    with open("kernel_window128_r1/kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n--- METADATA ---")
    print(json.dumps(meta, indent=2))
    assert meta["id"] == "andrlab/gemma4-window128-r1"
    assert meta["code_file"] == "gemma4-window128-r1.ipynb"
    assert "andrlab/gemma-4-omni-module" in meta["dataset_sources"]
    assert "andrlab/gemma4-ht-adapter-r2" in meta["dataset_sources"]

    print("\n=== TUTTI I CONTROLLI WINDOW 128 SUPERATI CON SUCCESSO! ===")

if __name__ == "__main__":
    verify()
