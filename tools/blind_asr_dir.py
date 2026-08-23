#!/usr/bin/env python3
"""blind_asr_dir.py — Trascrive cieca tutti i wav di una directory.
Usage: python3 blind_asr_dir.py <dir> <language>"""
import glob
import os
import sys

from faster_whisper import WhisperModel

d = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else None
model = WhisperModel("small", device="cuda", compute_type="float16")
for f in sorted(glob.glob(os.path.join(d, "*.wav"))):
    segs, info = model.transcribe(f, language=lang, beam_size=5)
    text = " ".join(s.text.strip() for s in segs)
    print(f"[BLIND] {os.path.basename(f)} ({info.language}): {text[:150]}", flush=True)
