#!/usr/bin/env python3
"""merge_thv5.py — Unisce le parti THv5 nel formato parquet standard atteso dal
training TurnHead (mel/ameld/codes/turnlabels/hyperparams con un prefisso).

- _mel.parquet   = concat user mel parts (128 colonne, float16)
- _ameld.parquet = concat agent mel parts
- _codes.parquet = placeholder zeros (16, T) int16 — il TurnHead usa SOLO mel+label
- _turnlabels.parquet = già esistente (train_tokens_thv5_turnlabels.parquet)
- _hyperparams.json = esteso con SEGMENT_ENDS
"""
import glob
import json
import os

import numpy as np
import pandas as pd

DATA = "/workspace/data"
PREFIX = f"{DATA}/train_tokens_thv5.parquet"

def merge_parts(pattern, out_path):
    parts = sorted(glob.glob(pattern))
    assert parts, f"nessuna parte per {pattern}"
    dfs = [pd.read_parquet(p) for p in parts]
    merged = pd.concat(dfs, ignore_index=True)
    merged.to_parquet(out_path, index=False)
    print(f"{out_path}: {merged.shape}")
    return len(merged)

def main():
    n_mel_rows = merge_parts(f"{DATA}/thv5_user_mel_part*.parquet", f"{PREFIX}_mel.parquet")
    merge_parts(f"{DATA}/thv5_agent_mel_part*.parquet", f"{PREFIX}_ameld.parquet")

    # turnlabels
    tl = pd.read_parquet(f"{DATA}/train_tokens_thv5_turnlabels.parquet")
    tl.to_parquet(f"{PREFIX}_turnlabels.parquet", index=False)
    n_frames = len(tl)

    # codes placeholder: (16, T) zeros int16, T = n frame agente = n_frames
    codes = pd.DataFrame(np.zeros((16, n_frames), dtype=np.int16))
    codes.to_parquet(f"{PREFIX}_codes.parquet", index=False)
    print(f"{PREFIX}_codes.parquet: placeholder (16, {n_frames})")

    with open(f"{DATA}/train_tokens_thv5_hyperparams.json") as f:
        hyp = json.load(f)
    hyp["TOTAL_FRAMES"] = n_frames
    hyp["N_CODEBOOKS"] = 16
    hyp["SEMANTIC_VOCAB_SIZE"] = 4096
    hyp["ACOUSTIC_VOCAB_SIZE"] = 2048
    hyp["NOTE_CODES"] = "placeholder zeros: il TurnHead usa solo mel + turnlabels native"
    hyp["CLASS_NAMES"] = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]
    with open(f"{PREFIX}_hyperparams.json", "w") as f:
        json.dump(hyp, f, indent=2)
    print("hyperparams scritti. MERGE COMPLETATO.")

if __name__ == "__main__":
    main()
