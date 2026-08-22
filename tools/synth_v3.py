#!/usr/bin/env python3
"""
synth_v3.py - Sintesi wav dal checkpoint adapter v3 (Task 11H)
Teacher-free: usa duplex_model.generate() nativo (16 codebook) come synthesize_examples.
Output: samples_v3/generated_XXX.wav + oracle_XXX.wav + README con testi attesi.
"""
import os, sys

os.environ.setdefault("PIPELINE_ENV", "local3090")
sys.path.insert(0, "/workspace")

import numpy as np
import kaggle_pipeline as KP


def main():
    n_windows = int(os.environ.get("SYNTH_WINDOWS", "8"))
    ckpt = os.environ.get("V3_CKPT", "/workspace/checkpoints/gemma4_duplex_sota_v3/final")
    out_dir = "/workspace/samples_v3"

    print("[SYNTH-V3] 1/4 Carico sistema Gemma4 Duplex...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    assert duplex_model is not None, "modello non caricato"
    base_dev = next(duplex_model.base_model.parameters()).device

    print(f"[SYNTH-V3] 2/4 Carico checkpoint {ckpt} ...", flush=True)
    KP.load_gemma4_duplex_checkpoint(duplex_model, ckpt)
    duplex_model.eval()

    print("[SYNTH-V3] 3/4 Carico dataset CV26 (finestre di validazione, ultimo 5%)...", flush=True)
    prefix = "/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"
    user_mel, agent_mel, agent_codes, text_by_window = KP.load_tokens_from_parquet(prefix)
    t_agent = agent_codes.shape[1]
    import json
    seg_ends = []
    sidecar = prefix.replace(".parquet", "_hyperparams.json")
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            seg_ends = json.load(f).get("SEGMENT_ENDS", [])
    val_start = int(0.95 * t_agent)
    for e in seg_ends:
        if e >= val_start:
            val_start = e
            break
    n_val_w = max(1, (t_agent - val_start) // 64)
    val_ds = KP.FullDuplexDataset(
        user_mel[val_start * 8:],
        agent_mel[val_start * 8:],
        agent_codes[:, val_start:],
        seq_len=64,
        text_by_window=text_by_window[len(text_by_window) - n_val_w:] if text_by_window else None,
    )
    print(f"[SYNTH-V3] Finestre val: {len(val_ds)}", flush=True)

    print("[SYNTH-V3] 4/4 Sintesi (decoder ufficiale via subprocess)...", flush=True)
    decoder = KP.Code2WavDecoder()
    paths = KP.synthesize_examples(
        duplex_model, val_ds, decoder=decoder, n_windows=n_windows, output_dir=out_dir
    )
    print(f"[SYNTH-V3] FATTO: {len(paths)} wav generated in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
