#!/usr/bin/env python3
"""blind_asr2.py — Trascrizione CIECA via faster-whisper (niente torchaudio)."""
from faster_whisper import WhisperModel
import soundfile as sf

FILES = ["oracle_003", "generated_003", "oracle_001", "generated_001", "generated_005"]

model = WhisperModel("small", device="cuda", compute_type="float16")
print("[BLIND-ASR2] modello caricato", flush=True)
for f in FILES:
    try:
        segments, info = model.transcribe(f"/workspace/samples_v3/{f}.wav", language="it", beam_size=5)
        text = " ".join(s.text.strip() for s in segments)
        print(f"[BLIND] {f}: {text}", flush=True)
    except Exception as e:
        print(f"[BLIND] {f}: ERRORE {e}", flush=True)
