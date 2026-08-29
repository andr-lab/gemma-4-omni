#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import glob
from faster_whisper import WhisperModel

def main():
    whisper_dirs = glob.glob('/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*')
    whisper_path = whisper_dirs[0] if len(whisper_dirs) > 0 else "small"
    model = WhisperModel(whisper_path, device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")

    for f_name in ["synth_0000_ch0.wav", "synth_0000_ch1.wav"]:
        p = f"/workspace/data/synth_it/out/{f_name}"
        if os.path.exists(p):
            print(f"=== {f_name} ===")
            segs, _ = model.transcribe(p, language="it")
            for s in segs:
                print(f"  [{s.start:05.2f}s - {s.end:05.2f}s]: {s.text}")

if __name__ == "__main__":
    main()
