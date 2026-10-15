#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/build_voice_profiles.py — Estrae profili voce (x-vector) dal talker
Qwen3-TTS per il serving vLLM-Omni (voci di default).

Per ogni wav di riferimento:
  1. Carica il talker Qwen3-TTS (HF reference) su GPU.
  2. Estrae lo speaker embedding (x-vector 2048D) con lo speaker encoder.
  3. Salva custom_voices/<name>.safetensors (speaker_embedding [+ ref_code se --icl])
     + custom_voice_manifest.json (formato ufficiale vLLM-Omni).

Uso:
  python tools/build_voice_profiles.py --voices "gemma=path/to/gemma.wav,leonardo=path/to/leo.wav"
  python tools/build_voice_profiles.py --voices gemma=... --out /workspace/custom_voices --mode xvec
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/qwen3tts_src")

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import types, importlib.machinery

try:
    import torchaudio
except Exception:
    _m = types.ModuleType("torchaudio")
    _m.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = _m
    sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
    sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
    sys.modules["torchaudio.compliance.kaldi"] = types.ModuleType("torchaudio.compliance.kaldi")
sys.modules["sox"] = types.ModuleType("sox")

import transformers.modeling_rope_utils as _ropeu


def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0


_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

import safetensors.torch as st
import qwen_tts.core.models as _M


def load_talker() -> "tuple":
    """Carica il talker Qwen3-TTS (HF reference) in bf16 su cuda."""
    snaps = sorted(glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*"))
    if not snaps:
        raise FileNotFoundError("Qwen3-TTS repo non trovato in /workspace/.hf_cache")
    snap = snaps[-1]
    cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = cfg.talker_config
    if not hasattr(_tc, "pad_token_id"):
        object.__setattr__(_tc, "pad_token_id", None)
    teacher = _M.Qwen3TTSForConditionalGeneration(cfg).to(torch.bfloat16)
    sd = st.load_file(snap + "/model.safetensors", device="cpu")
    teacher.load_state_dict(sd, strict=False)
    del sd
    torch.cuda.empty_cache()
    teacher = teacher.to(device="cuda", dtype=torch.bfloat16).eval()
    return teacher


def _mel_filter_bank_slaney(n_fft: int, n_mels: int, sr: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """Replica torchaudio melscale_fbanks (mel_scale='slaney', norm='slaney')."""
    import numpy as _np

    if fmax is None:
        fmax = float(sr) / 2.0
    fftfreqs = _np.linspace(0.0, float(sr) / 2.0, n_fft // 2 + 1)

    def hz_to_mel(hz):
        f_min, f_sp = 0.0, 200.0 / 3
        mels = (hz - f_min) / f_sp
        min_log_hz, min_log_mel = 1000.0, (1000.0 - f_min) / f_sp
        logstep = _np.log(6.4) / 27.0
        mels = _np.where(
            hz >= min_log_hz,
            min_log_mel + _np.log(_np.maximum(hz, 1e-9) / min_log_hz) / logstep,
            mels,
        )
        return mels

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_bins = _np.linspace(mel_min, mel_max, n_mels + 2)
    mel_f = hz_to_mel(fftfreqs)
    fbanks = _np.zeros((n_mels, len(fftfreqs)))
    for i in range(n_mels):
        left, center, right = mel_bins[i], mel_bins[i + 1], mel_bins[i + 2]
        up = (mel_f - left) / (center - left + 1e-9)
        down = (right - mel_f) / (right - center + 1e-9)
        fbanks[i] = _np.maximum(0.0, _np.minimum(up, down))
    enorm = 2.0 / (mel_bins[2:] - mel_bins[:-2])
    fbanks *= enorm[:, None]
    return torch.from_numpy(fbanks.astype(_np.float32))


def _mel_spectrogram(wav: torch.Tensor, sr: int = 24000) -> torch.Tensor:
    """Mel 128 bin ESATTA del prompt_embeds_builder (n_fft=1024, hop=256,
    reflect pad, center=False, matmul filterbank, log compression)."""
    n_fft, hop, n_mels, fmin, fmax = 1024, 256, 128, 0, 12000
    device = wav.device
    mel_basis = _mel_filter_bank_slaney(n_fft, n_mels, sr, fmin, fmax).to(device)
    hann = torch.hann_window(n_fft, device=device)
    padding = (n_fft - hop) // 2
    y = torch.nn.functional.pad(wav.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)
    spec = torch.stft(
        y, n_fft, hop_length=hop, win_length=n_fft, window=hann,
        center=False, pad_mode="reflect", normalized=False, onesided=True,
        return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    return torch.log(torch.clamp(mel_spec, min=1e-5))  # dynamic range compression


def extract_xvector(talker, wav_path: str) -> torch.Tensor:
    """Estrae lo speaker embedding (x-vector 2048D) da un wav di riferimento.

    Ricetta ufficiale (prompt_embeds_builder.extract_speaker_embedding):
    resample 24kHz → mel 128 (slaney) → speaker_encoder(mels)[0].
    """
    wav, sr = sf.read(wav_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 24000:
        n = int(len(wav) * 24000 / sr)
        x = torch.from_numpy(wav).float().unsqueeze(0)
        x = F.interpolate(x.view(1, 1, -1), size=n, mode="linear", align_corners=False).view(-1)
        wav = x.numpy()
    wav_t = torch.from_numpy(wav).float().unsqueeze(0).to("cuda")
    mels = _mel_spectrogram(wav_t).transpose(1, 2).to(torch.bfloat16)  # (1, T, 128)
    with torch.no_grad():
        spk = talker.speaker_encoder(mels)[0]
    return spk.detach().cpu()


def main() -> None:
    ap = argparse.ArgumentParser(description="Estrazione profili voce Qwen3-TTS")
    ap.add_argument("--voices", required=True, help='"nome=path.wav,nome2=path2.wav"')
    ap.add_argument("--out", default="/workspace/custom_voices")
    ap.add_argument("--mode", default="xvec", choices=["xvec", "icl"])
    ap.add_argument("--ref-text", default="", help="trascrizione del ref (per icl)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    talker = load_talker()

    manifest: dict = {"model_type": "qwen3_tts", "voices": {}}
    for pair in args.voices.split(","):
        name, path = pair.split("=", 1)
        name = name.strip()
        path = path.strip()
        print(f"[voices] Estrazione x-vector per '{name}' da {path} ...", flush=True)
        spk = extract_xvector(talker, path)
        print(f"  -> shape: {tuple(spk.shape)}", flush=True)
        tensors = {"speaker_embedding": spk.squeeze(0).detach().cpu()}
        fname = f"{name.lower()}.safetensors"
        st.save_file(tensors, os.path.join(args.out, fname))
        entry = {"name": name, "file": fname, "mode": args.mode}
        if args.mode == "icl" and args.ref_text:
            entry["ref_text"] = args.ref_text
        manifest["voices"][name.lower()] = entry
        print(f"  -> salvato {os.path.join(args.out, fname)}", flush=True)

    with open(os.path.join(args.out, "custom_voice_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[voices] Manifest salvato in {args.out}/custom_voice_manifest.json", flush=True)


if __name__ == "__main__":
    main()
