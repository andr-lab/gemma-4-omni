# -*- coding: utf-8 -*-
import json, ast, os

def verify():
    nb_path = "kernel_turnhead_v2_r1/gemma4-turnhead-v2-r1.ipynb"
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
        "TURNHEAD_V2_ACTIVE",
        "N_TURN_CLASSES",
        "TURN_LABELS_SOURCE",
        "TURN_CLASS_DIST_TRAIN",
        "TURN_CLASS_DIST_BATCH",
        "TURNDATA_VAL_v2",
        "riuso Gemma",
        "head_grad",
        "backbone_grad0",
        "SPEAK",
        "YIELD",
        "BARGE_IN",
        "USER_BACKCHANNEL",
        "EMIT_BACKCHANNEL",
    ]

    print("\n--- VERIFICA PROBES TURNHEAD v2 STAGE 3 ---")
    for p in probes:
        assert p in full_code, f"PROBE MANCANTE: {p}"
        print(f"Probe '{p}': PRESENTE")

    with open("kernel_turnhead_v2_r1/kernel-metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n--- METADATA ---")
    print(json.dumps(meta, indent=2))
    assert meta["id"] == "andrlab/gemma4-turnhead-v2-r1"
    assert meta["code_file"] == "gemma4-turnhead-v2-r1.ipynb"
    assert "andrlab/gemma-4-omni-module" in meta["dataset_sources"]
    assert "andrlab/gemma4-ht-adapter-r2" in meta["dataset_sources"]

    # Byte non-ASCII check
    with open(nb_path, "rb") as f:
        data = f.read()
    non_ascii = [b for b in data if b > 127]
    assert len(non_ascii) == 0, f"Trovati {len(non_ascii)} byte non-ASCII nel notebook!"

    print("\n=== TUTTI I CONTROLLI TURNHEAD v2 SUPERATI CON SUCCESSO! ===")

if __name__ == "__main__":
    verify()
