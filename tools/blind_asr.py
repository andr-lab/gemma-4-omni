#!/usr/bin/env python3
"""blind_asr.py — Trascrizione CIECA dei wav v3 (nessun testo atteso in input).
Whisper-small via HF transformers sul 3090. L'ASR non sa quale testo ci si aspetta:
il confronto col testo atteso avviene SOLO dopo, lato coordinatore."""
import sys
from transformers import pipeline
import soundfile as sf

FILES = [
    "oracle_003", "generated_003",   # la coppia contestata dall'utente
    "oracle_001", "generated_001",
    "oracle_005" if False else "generated_005",
]

def main():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)
    print("[BLIND-ASR] modello caricato, inizio trascrizioni (nessun testo atteso fornito)")
    for f in FILES:
        try:
            d, sr = sf.read(f"/workspace/samples_v3/{f}.wav")
            out = asr({"array": d, "sampling_rate": sr}, generate_kwargs={"language": "italian"})
            print(f"[BLIND] {f}: {out['text'].strip()}", flush=True)
        except Exception as e:
            print(f"[BLIND] {f}: ERRORE {e}", flush=True)

if __name__ == "__main__":
    main()
