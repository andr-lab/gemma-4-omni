#!/usr/bin/env python3
"""Genera e analizza campioni audio di verifica per chiusura P4D:
- Causal V9 a L=0
- FuturePrefixQwen con _SlotKVCache
- SubTalker (15 residui CUDA Graph)
- Code2Wav (vocoder 24kHz)
- Calcolo metriche audio (RMS, clip, click, silenzio degenerato)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io.wavfile as wavfile
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    IncrementalV9Projector,
)
from tools.kernel_gates.p4_checks import _stack_cache


def analyze_audio(wav: np.ndarray, sample_rate: int = 24000) -> dict[str, Any]:
    """Analisi acustica del segnale generato."""
    if len(wav) == 0:
        return {"error": "empty_audio"}

    wav_f = wav.astype(np.float32)
    if wav_f.max() > 1.0 or wav_f.min() < -1.0:
        wav_f = wav_f / max(abs(wav_f.max()), abs(wav_f.min()), 1e-6)

    # 1. RMS Energy
    rms = float(np.sqrt(np.mean(wav_f ** 2)))
    has_speech = bool(rms > 0.01)

    # 2. Silenzio degenerato (frazione di frame a -60dB)
    frame_len = int(sample_rate * 0.02)  # 20ms
    frames = [wav_f[i:i+frame_len] for i in range(0, len(wav_f) - frame_len, frame_len)]
    frame_rms = [float(np.sqrt(np.mean(f ** 2))) for f in frames]
    silent_frames = sum(1 for r in frame_rms if r < 0.001)
    silence_ratio = silent_frames / len(frames) if frames else 1.0

    # 3. Discontinuità e click (differenze prime estreme)
    diff = np.abs(np.diff(wav_f))
    click_threshold = 0.5
    click_count = int(np.sum(diff > click_threshold))

    # 4. Zero crossing rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(wav_f)))) / 2.0)

    # 5. Spettro e centroid
    spectrum = np.abs(np.fft.rfft(wav_f))
    freqs = np.fft.rfftfreq(len(wav_f), 1.0 / sample_rate)
    spec_sum = np.sum(spectrum)
    spectral_centroid = float(np.sum(freqs * spectrum) / spec_sum) if spec_sum > 0 else 0.0

    # Gate audio
    pass_speech = has_speech and (silence_ratio < 0.60)
    pass_clicks = (click_count < 5)

    return {
        "duration_s": float(len(wav) / sample_rate),
        "rms_energy": rms,
        "peak_amplitude": float(np.max(np.abs(wav_f))),
        "silence_ratio": silence_ratio,
        "click_count": click_count,
        "zero_crossing_rate": zcr,
        "spectral_centroid_hz": spectral_centroid,
        "has_speech": has_speech,
        "pass_audio_quality": bool(pass_speech and pass_clicks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P4D Audio Generation & Verification")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_p4d_r_512.pt"))
    parser.add_argument("--ckpt", type=Path, default=Path("/workspace/checkpoints/v9_causal_l0_final.pt"))
    parser.add_argument("--heads", type=Path, default=Path("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/samples/p4d_causal_l0"))
    parser.add_argument("--num-samples", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        ckpt_path = Path("/workspace/checkpoints/v9_causal_l0_unknown.pt")

    print(f"[AUDIO] Inizializzazione pipeline con {ckpt_path}...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=True)
    talker.talker.to(dtype=dtype)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    projector = DeepCausalTransformerProjectorV9().to(device=device, dtype=dtype).eval()
    projector.load_state_dict(sd.get("proj", sd), strict=False)

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=dtype).eval()
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"])
    else:
        hsd = torch.load(args.heads, map_location="cpu", weights_only=True)
        causal_conv.conv.load_state_dict(hsd["causal_conv"], strict=False)

    from duplex_kernel.subtalker_graph import SubtalkerGraph
    cp = talker.talker.code_predictor
    subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

    raw, anchors, codes, _ = _stack_cache(args.cache)
    raw = raw[:args.num_samples].to(device=device, dtype=dtype)
    anchors = anchors[:args.num_samples].to(device=device, dtype=dtype)
    frames_per_seq = raw.shape[1]

    samples_results = []
    all_audio_pass = True

    print(f"[AUDIO] Generazione di {args.num_samples} campioni a L=0 (durata ~{frames_per_seq/12.5:.1f}s ciascuno)...", flush=True)

    for idx in range(args.num_samples):
        seq_raw = raw[idx:idx+1]
        seq_anchor = anchors[idx:idx+1]

        # V9 incrementale
        inc_v9 = IncrementalV9Projector(projector, causal_conv)
        v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=frames_per_seq)

        speaker = FuturePrefixQwen(
            talker,
            lookahead_frames=0,
            max_prefix_frames=frames_per_seq,
            device=device,
            dtype=dtype,
            max_audio_frames=frames_per_seq,
        )
        speaker.begin("Verifica causale del kernel.")

        generated_codes: list[list[int]] = []

        for t in range(frames_per_seq):
            # 1. Step V9 frame t
            h_pfx = inc_v9.step(seq_raw[:, t, :], seq_anchor, v9_state)
            # 2. Push V9 frame t nello slot
            speaker.push_v9_frame(t, h_pfx)
            # 3. Forward Talker (CB0)
            step_out = speaker.step_frame()
            c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
            # 4. SubTalker (15 residui)
            c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
            resid = subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
            if isinstance(resid, torch.Tensor):
                resid = resid.reshape(-1).tolist()
            frame_c = [c0] + [int(x) for x in resid]
            generated_codes.append(frame_c)
            speaker.commit_codes(frame_c)

        # 5. Decode Code2Wav
        codes_tensor = torch.tensor(generated_codes, device=device, dtype=torch.long).unsqueeze(0)
        audio_tensor = talker.decode(codes_tensor)
        wav = audio_tensor.numpy().squeeze()

        # Salvataggio WAV
        wav_path = args.output_dir / f"sample_{idx+1:03d}_causal_l0.wav"
        wav_norm = np.clip(wav, -1.0, 1.0)
        wav_int16 = (wav_norm * 32767.0).astype(np.int16)
        wavfile.write(str(wav_path), 24000, wav_int16)

        # Analisi acustica
        metrics = analyze_audio(wav_norm, sample_rate=24000)
        metrics["sample_id"] = idx + 1
        metrics["wav_path"] = str(wav_path)
        metrics["num_frames"] = frames_per_seq
        samples_results.append(metrics)

        print(f"  Sample {idx+1}: RMS={metrics['rms_energy']:.4f}, Sil={metrics['silence_ratio']*100:.1f}%, Clicks={metrics['click_count']}, Quality={'PASS' if metrics['pass_audio_quality'] else 'FAIL'}", flush=True)
        if not metrics["pass_audio_quality"]:
            all_audio_pass = False

    report = {
        "num_samples": args.num_samples,
        "sample_rate": 24000,
        "lookahead_frames": 0,
        "checkpoint": str(ckpt_path),
        "all_audio_pass": all_audio_pass,
        "samples": samples_results,
    }

    report_path = Path("/workspace/validation/kernel_gates/P4D_R/P4D_AUDIO_VERIFICATION.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[AUDIO] Report salvato in {report_path}", flush=True)
    print(f"[AUDIO] Esito acustico globale: {'PASS' if all_audio_pass else 'FAIL'}\n", flush=True)

    return 0 if all_audio_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
