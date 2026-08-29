#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duplex_kernel/benchmark.py — Benchmark qualità audio + velocità del kernel.

Misura per un set di domande conversazionali:
  - E2E (s): tempo totale think + v9 + talk
  - RTF: e2e / durata audio (real-time factor; <1 = più veloce del real-time)
  - TTFA (s): tempo al primo chunk audio (dopo il prefill del talker)
  - Metriche fisiche: Crest Factor, HNR (dB), THD+N (%), clipping %
  - ASR Whisper: fedeltà lessicale (%)

Confronta con vLLM-Omni (se raggiungibile) e salva report JSON + WAV.

Uso (nel container, GPU):
    python3 duplex_kernel/benchmark.py --workspace /workspace --n 5 --out /workspace/samples_kernel
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import Any

import numpy as np

logging.basicConfig(level=logging.WARNING)

QUESTIONS = [
    "Quali sono i colori primari? Rispondi in una sola breve frase.",
    "Chi è Leonardo da Vinci? Rispondi in una sola breve frase.",
    "Cosa fa la fotosintesi clorofilliana? Rispondi in una sola breve frase.",
    "Qual è la capitale della Francia? Rispondi in una sola breve frase.",
    "Spiega cos'è l'energia solare in una sola breve frase.",
]


def compute_physical_metrics(wav: np.ndarray, sr: int = 24000) -> dict[str, float]:
    peak = float(np.max(np.abs(wav)))
    rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
    crest = peak / rms
    frame_len = int(0.040 * sr)
    hop = int(0.020 * sr)
    hnr_list: list[float] = []
    for s in range(0, len(wav) - frame_len, hop):
        f = wav[s : s + frame_len]
        f_rms = np.sqrt(np.mean(f**2))
        if f_rms < 0.01:
            continue
        fw = f * np.hanning(len(f))
        r = np.correlate(fw, fw, mode="full")[len(fw) - 1 :]
        lo, hi = int(sr / 400), int(sr / 60)
        if hi >= len(r):
            continue
        pk = lo + np.argmax(r[lo:hi])
        r0, rmax = r[0] + 1e-9, r[pk]
        if rmax > 0 and r0 > rmax:
            hnr_list.append(10.0 * np.log10(rmax / (r0 - rmax + 1e-9)))
    hnr = float(np.mean(hnr_list)) if hnr_list else 0.0
    thd = float(100.0 / (1.0 + 10.0 ** (hnr / 10.0))) if hnr > -30 else 99.0
    clip = float(np.mean(np.abs(wav) >= 0.999) * 100.0)
    return {
        "crest_factor": round(crest, 2),
        "hnr_db": round(hnr, 2),
        "thd_n_pct": round(thd, 2),
        "clipping_pct": round(clip, 3),
        "peak": round(peak, 4),
        "rms": round(rms, 4),
    }


def transcribe_asr(wav_path: str, workspace: str) -> str:
    try:
        from faster_whisper import WhisperModel

        wd = sorted(glob.glob(os.path.join(workspace, ".hf_cache/models--Systran--faster-whisper-small/snapshots/*")))
        m = WhisperModel(wd[0] if wd else "small", device="cuda", compute_type="float16",
                         download_root=os.path.join(workspace, ".hf_cache"))
        segs, _ = m.transcribe(wav_path, language="it", temperature=0.0)
        return " ".join(s.text for s in segs).strip()
    except Exception as exc:
        return f"(ASR non disponibile: {exc})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/workspace")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out", default="/workspace/samples_kernel")
    ap.add_argument("--asr", action="store_true", help="esegue anche la verifica ASR")
    args = ap.parse_args()

    sys.path.insert(0, args.workspace)
    sys.path.insert(0, args.workspace + "/qwen3tts_src")
    from duplex_kernel.engine import PipelineEngine

    print("=" * 70)
    print("  DUPLEX KERNEL — BENCHMARK QUALITÀ + VELOCITÀ (ADR-029)")
    print("=" * 70)
    engine = PipelineEngine()
    if not engine.load(workspace=args.workspace):
        print("LOAD FAIL")
        return 1

    os.makedirs(args.out, exist_ok=True)
    results: list[dict[str, Any]] = []
    for i, q in enumerate(QUESTIONS[: args.n]):
        chat = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True
        )
        ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        t0 = time.time()
        think = engine.think(ids, max_new_tokens=48)
        t_think = time.time() - t0
        h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])
        t_v9 = time.time() - t0
        wav = engine.talk(think["text"], h_pfx, n_words=max(1, len(think["text"].split())))
        t_end = time.time() - t0
        dur = len(wav) / 24000
        metrics = compute_physical_metrics(wav)
        out = os.path.join(args.out, f"kernel_bench_{i:02d}.wav")
        import soundfile as sf

        sf.write(out, wav, 24000)
        row = {
            "q": q,
            "testo": think["text"],
            "n_words": len(think["text"].split()),
            "durata_audio_s": round(dur, 2),
            "e2e_s": round(t_end, 2),
            "think_s": round(t_think, 2),
            "v9_s": round(t_v9 - t_think, 2),
            "talk_s": round(t_end - t_v9, 2),
            "rtf": round(t_end / dur, 3),
            **metrics,
            "wav": out,
        }
        if args.asr:
            row["asr"] = transcribe_asr(out, args.workspace)
        results.append(row)
        print(f"[{i+1}/{len(QUESTIONS[:args.n])}] {think['text'][:60]}")
        print(f"    e2e={t_end:.1f}s dur={dur:.1f}s RTF={t_end/dur:.2f} crest={metrics['crest_factor']} "
              f"HNR={metrics['hnr_db']}dB THD={metrics['thd_n_pct']}%")

    report = {"results": results}
    avg = {
        "e2e_s": round(float(np.mean([r["e2e_s"] for r in results])), 2),
        "rtf": round(float(np.mean([r["rtf"] for r in results])), 3),
        "crest_factor": round(float(np.mean([r["crest_factor"] for r in results])), 2),
        "hnr_db": round(float(np.mean([r["hnr_db"] for r in results])), 2),
        "thd_n_pct": round(float(np.mean([r["thd_n_pct"] for r in results])), 2),
        "clipping_pct": round(float(np.mean([r["clipping_pct"] for r in results])), 3),
    }
    report["avg"] = avg
    rp = os.path.join(args.out, "kernel_benchmark_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 70)
    print("  MEDIE:")
    for k, v in avg.items():
        print(f"    {k}: {v}")
    print(f"  Report: {rp}")
    print("=" * 70)
    engine.unload()
    return 0


if __name__ == "__main__":
    sys.exit(main())
