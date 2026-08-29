#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duplex_kernel/launch.py — Avvio del custom duplex kernel (ADR-029).

Uso (su gpu3090, dentro il container o venv con GPU):
    python duplex_kernel/launch.py --port 8092
    python duplex_kernel/launch.py --smoke   # solo test caricamento + 1 sintesi

Il kernel è narrow (ispirato a DwarfStar/antirez-ds4) adattato alla nostra
architettura: Thinker Gemma4 E2B → V9 → Talker Qwen3-TTS → Code2Wav, con
TurnDetectorHead nativa e KV persistente per sessione.
"""

import argparse
import logging
import os
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--smoke", action="store_true", help="solo test: load + 1 sintesi + exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    sys.path.insert(0, args.workspace)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # radice repo (duplex_kernel/)
    sys.path.insert(0, args.workspace + "/qwen3tts_src")

    from duplex_kernel.engine import PipelineEngine

    print("=" * 68)
    print("  GEMMA-4-OMNI DUPLEX KERNEL (ADR-029, narrow engine)")
    print("=" * 68)

    engine = PipelineEngine()
    t0 = time.time()
    ok = engine.load(workspace=args.workspace)
    print(f"[launch] load engine: {'OK' if ok else 'FAIL'} in {time.time()-t0:.1f}s")
    if not ok:
        return 1

    if args.smoke:
        # Smoke: 1 domanda → testo + audio, metriche fisiche
        import numpy as np

        t1 = time.time()
        prompt = "Quali sono i colori primari? Rispondi in una sola breve frase."
        chat = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        think = engine.think(ids, max_new_tokens=48)
        n_words = len(think["text"].split())
        h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])
        wav = engine.talk(think["text"], h_pfx, n_words=n_words)
        dt = time.time() - t1
        peak = float(np.max(np.abs(wav)))
        rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
        print(f"[smoke] testo: {think['text']}")
        print(f"[smoke] audio: {len(wav)/24000:.2f}s ({n_words} parole, {len(wav)/24000*12.5:.0f} frame)")
        print(f"[smoke] tempo: {dt:.1f}s | RTF={dt/(len(wav)/24000):.3f}")
        print(f"[smoke] crest={peak/rms:.2f} peak={peak:.3f} rms={rms:.4f}")

        os.makedirs(os.path.join(args.workspace, "samples_kernel"), exist_ok=True)
        import soundfile as sf

        out = os.path.join(args.workspace, "samples_kernel", "kernel_smoke.wav")
        sf.write(out, wav, 24000)
        print(f"[smoke] WAV: {out}")
        return 0

    from duplex_kernel.server import run

    run(engine, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
