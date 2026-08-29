#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

DATA_DIR = "/workspace/data"

def inspect():
    lbl_p = os.path.join(DATA_DIR, "turnhead_20h_labels.npy")
    feat_p = os.path.join(DATA_DIR, "turnhead_20h_z_feats.npy")

    if os.path.exists(lbl_p) and os.path.exists(feat_p):
        labels = np.load(lbl_p, mmap_mode='r')
        feats = np.load(feat_p, mmap_mode='r')
        print(f"turnhead_20h_labels: shape={labels.shape}, dtype={labels.dtype}")
        print(f"turnhead_20h_z_feats: shape={feats.shape}, dtype={feats.dtype}")

        unq, cnts = np.unique(labels[:100000], return_counts=True)
        print("Class distribution in first 100k frames:")
        for u, c in zip(unq, cnts):
            print(f"  Class {u}: {c} frames ({c/1000:.1f}%)")

        # Trova una sequenza dialogica continua con cambi di turno (User -> Yield -> Agent / Barge-in)
        # Cerchiamo transizioni tra classi diverse
        changes = np.where(labels[1:5000] != labels[:4999])[0]
        print(f"\nFound {len(changes)} turn transitions in first 5k frames:")
        for ch in changes[:10]:
            print(f"  Frame {ch} ({ch*0.08:.2f}s): {labels[ch]} -> {labels[ch+1]}")

if __name__ == "__main__":
    inspect()
