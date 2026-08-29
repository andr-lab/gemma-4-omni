#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
import pandas as pd
import numpy as np
from transformers import AutoTokenizer

DATA_DIR = "/workspace/data"

def decode_texts():
    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    if len(gemma_dirs) == 0:
        print("Tokenizer not found")
        return
    tok = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True)

    f = "train_tokens_cv26_it_40.8h_31916s_text.parquet"
    p = os.path.join(DATA_DIR, f)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        print(f"=== {f} (Total rows: {len(df)}) ===")
        for i in range(10):
            val = df.iloc[i]['text']
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                print(f"[{i}]: {tok.decode(val, skip_special_tokens=True)}")

    # Check MDC texts
    f_mdc = "train_tokens_mdc_16.0h_9185s_text.parquet"
    p_mdc = os.path.join(DATA_DIR, f_mdc)
    if os.path.exists(p_mdc):
        df_mdc = pd.read_parquet(p_mdc)
        print(f"\n=== {f_mdc} ===")
        valid_rows = df_mdc[df_mdc['text'].notnull()]
        print(f"Valid text rows: {len(valid_rows)}")
        for i in range(min(5, len(valid_rows))):
            val = valid_rows.iloc[i]['text']
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                print(f"MDC [{i}]: {tok.decode(val, skip_special_tokens=True)}")

if __name__ == "__main__":
    decode_texts()
