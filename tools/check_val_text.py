#!/usr/bin/env python3
"""check_val_text.py — testo GT delle prime finestre val (per confronto post-ASR)."""
import os, sys, json

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
import kaggle_pipeline as KP

prefix = "/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet".replace(".parquet", "")
user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(prefix + ".parquet")
t_agent = agent_codes.shape[1]
sidecar = prefix + "_hyperparams.json"
seg_ends = json.load(open(sidecar)).get("SEGMENT_ENDS", [])
val_start = int(0.90 * t_agent)
for e in seg_ends:
    if e >= val_start:
        val_start = e
        break
print(f"val_start={val_start} t_agent={t_agent}")
for i in range(4):
    ws = val_start + i * 64
    we = ws + 64
    txt = tbw[ws:we]
    # testo mostrato: dedup delle ripetizioni del blocco testo (il blocco si ripete per frame)
    toks = [t for t in txt if t != 0]
    print(f"win{i} frames[{ws}:{we}] text_len={len(toks)}")
    print(f"   GT text: {''.join([str(t) if isinstance(t,str) else t for t in toks[:200]])}")
