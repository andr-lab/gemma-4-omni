#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/run_contrastive_gemma_synthesis.py
Standalone contrastive prosody & acoustic evaluation suite.
Generates speech directly through PipelineEngine (Gemma 4 Thinker -> FrameResidualQueryFiLMAdapter -> Qwen Talker -> Code2Wav)
for contrastive affective/prosodic prompts.

Measures:
1. Acoustic Quality: Crest factor, THD+N (%), HNR (dB), Rumble (<80Hz) %, Clipping %
2. Intelligibility: Faster-Whisper ASR transcription and Word Error Rate (WER)
3. Prosody: F0 mean (Hz), F0 std (Hz), Pitch range (semitones), Dynamic range (dB)
4. Comparative A/B Deltas
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/qwen3tts_src")

import numpy as np
import soundfile as sf
import torch

from duplex_kernel.engine import PipelineEngine
from tools.analyze_and_transcribe_sample import analyze_distortion, calculate_wer
from tools.measure_prosodic_modulation import extract_prosodic_features, compare_prosody


import re

def clean_speech_text(text: str) -> str:
    # Remove markdown bold/italics/headings
    text = re.sub(r'[*_#~`]', '', text)
    # Remove emojis and characters outside standard Italian orthography
    text = re.sub(r'[^\w\s.,!?:;\'"àèéìòùÀÈÉÌÒÙ\-]', '', text)
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


CONTRASTIVE_PROMPTS = [
    {
        "id": "A_entusiasta",
        "label": "Tono Entusiasta / Gioioso",
        "user_prompt": "Sei estremamente felice ed entusiasta! Esclama con grande gioia ed energia (solo testo semplice, senza emoji né asterischi): Che splendida vittoria, abbiamo vinto la finale!",
    },
    {
        "id": "B_calmo_triste",
        "label": "Tono Calmo / Malinconico",
        "user_prompt": "Sei calmo, cupo e addolorato. Rispondi con voce sommessa e triste (solo testo semplice, senza emoji né asterischi): Purtroppo la situazione è grave e non ci sono speranze.",
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run contrastive Gemma speech synthesis and acoustic evaluation.")
    parser.add_argument("--workspace", type=str, default="/workspace", help="Workspace root path")
    parser.add_argument("--out-dir", type=str, default="samples/contrastive_eval", help="Output directory for WAVs")
    parser.add_argument("--report", type=str, default="reports/contrastive_eval_report.json", help="Output report path")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("  CONTRASTIVE PROSODIC & ACOUSTIC EVALUATION (GEMMA 4 -> QWEN)", flush=True)
    print("=" * 70, flush=True)

    # 1. Initialize PipelineEngine
    print("\n[1/3] Inizializzazione PipelineEngine...", flush=True)
    engine = PipelineEngine()
    if not engine.load(workspace=args.workspace):
        print("Errore: engine.load() fallito!", file=sys.stderr)
        sys.exit(1)
    print("  -> Engine caricato con successo.", flush=True)

    # 2. Inizializza Faster-Whisper per verifica intelligibilita esterna
    print("\n[2/3] Caricamento Faster-Whisper per validazione ASR esterna...", flush=True)
    from faster_whisper import WhisperModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = WhisperModel("small", device=device, compute_type="float16" if device == "cuda" else "int8")
    print(f"  -> Faster-Whisper 'small' attivo su {device}.", flush=True)

    results = {}
    wav_files = {}

    # 3. Esecuzione sintesi contrastiva
    print("\n[3/3] Generazione campioni contrastivi...", flush=True)
    for item in CONTRASTIVE_PROMPTS:
        pid = item["id"]
        label = item["label"]
        prompt = item["user_prompt"]
        print(f"\n---> [{pid}] {label}", flush=True)
        print(f"     Prompt: \"{prompt}\"", flush=True)

        chat = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()

        t0 = time.time()
        # Generazione risposta testuale e stati hidden Gemma
        think = engine.think(input_ids, max_new_tokens=48)
        text_generated = think["text"].strip()
        t_think = time.time() - t0
        print(f"     Gemma 4 Text ({t_think:.2f}s): \"{text_generated}\"", flush=True)

        clean_text = clean_speech_text(text_generated)
        print(f"     Clean Speech Text: \"{clean_text}\"", flush=True)

        # Prefisso V9 (se attivo)
        h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])

        # Sintesi audio con Talker Qwen e FrameResidualQueryFiLMAdapter
        t_talk_start = time.time()
        wav = engine.talk(
            clean_text,
            h_pfx,
            n_words=max(1, len(clean_text.split())),
        )
        t_talk = time.time() - t_talk_start
        dur_s = float(len(wav) / 24000)
        print(f"     Audio generato ({dur_s:.2f}s in {t_talk:.2f}s, RTF={t_talk/dur_s:.2f})", flush=True)

        out_wav = out_dir / f"contrastive_{pid}.wav"
        sf.write(str(out_wav), wav, 24000)
        wav_files[pid] = out_wav

        # Analisi distorsione fisica
        dist = analyze_distortion(wav, 24000)
        # Analisi prosodica (F0, intonazione, dinamica)
        pros = extract_prosodic_features(wav, 24000)

        # Trascrizione ASR Whisper
        segments, _ = asr_model.transcribe(str(out_wav), language="it")
        asr_transcript = " ".join(s.text.strip() for s in segments).strip()
        wer = calculate_wer(clean_text, asr_transcript)

        results[pid] = {
            "prompt": prompt,
            "text": text_generated,
            "wav_path": str(out_wav),
            "durata_s": round(dur_s, 2),
            "think_s": round(t_think, 2),
            "talk_s": round(t_talk, 2),
            "rtf": round(t_talk / dur_s, 3),
            "distortion": dist,
            "prosody": pros,
            "asr_transcript": asr_transcript,
            "wer": round(wer, 4),
        }

        print(f"     [Acoustic Quality]")
        print(f"       Crest Factor : {dist['crest_factor']} (healthy: 4.0 - 8.5)")
        print(f"       THD+N        : {dist['thd_n_pct']}% (target: < 40%)")
        print(f"       HNR          : {dist['hnr_db']} dB (target: > 2.0 dB)")
        print(f"     [Prosody]")
        print(f"       Mean F0      : {pros['f0_mean_hz']} Hz")
        print(f"       Pitch Range  : {pros['pitch_range_semitones']} semitoni")
        print(f"       Dynamic Range: {pros['dynamic_range_db']} dB")
        print(f"     [Intelligibilità Whisper]")
        print(f"       Transcript   : \"{asr_transcript}\"")
        print(f"       WER          : {wer:.2%}")

    # Confronto contrastivo diretto A vs B
    if "A_entusiasta" in wav_files and "B_calmo_triste" in wav_files:
        print("\n" + "=" * 70, flush=True)
        print("  CONFRONTO DELLE TRAIETTORIE PROSODICHE (A vs B)", flush=True)
        print("=" * 70, flush=True)
        comp = compare_prosody(wav_files["B_calmo_triste"], wav_files["A_entusiasta"])
        results["comparison_A_vs_B"] = comp

        print(f"  Delta Mean F0 (Entusiasta - Calmo) : {comp['deltas']['delta_f0_mean_hz']:+} Hz")
        print(f"  Delta F0 Std (Espressività)        : {comp['deltas']['delta_f0_std_hz']:+} Hz")
        print(f"  Delta Pitch Range (semitoni)       : {comp['deltas']['delta_pitch_range_semitones']:+} semitoni")
        print(f"  Delta Dynamic Range (dB)           : {comp['deltas']['delta_dynamic_range_db']:+} dB")
        print(f"  Delta Crest Factor                 : {comp['deltas']['delta_crest_factor']:+}")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport completo salvato in {report_path}", flush=True)


if __name__ == "__main__":
    main()
