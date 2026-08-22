#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_official_worker.py - Official Qwen3-TTS Tokenizer 12Hz Audio Decoder Worker
Operates inside /opt/venv_phase1 (transformers 4.49/4.57 + qwen-tts) to decode RVQ codes
into 24kHz float32 audio without polluting Phase 2 (transformers 5.5.0).
"""

import sys
import os
import types
import importlib.machinery
import json
import argparse
import traceback
import numpy as np

# 1. Mock torchaudio with full module spec to satisfy qwen_tts dependencies
if "torchaudio" not in sys.modules:
    mock_ta = types.ModuleType("torchaudio")
    mock_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = mock_ta
    sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
    sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
    mock_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    sys.modules["torchaudio.compliance.kaldi"] = mock_kaldi

# 2. Exclude .pylibs from sys.path to avoid corrupted accelerate/transformers
sys.path = [p for p in sys.path if ".pylibs" not in p]
for local_p in ["/home/appuser/.local/lib/python3.11/site-packages", os.path.expanduser("~/.local/lib/python3.11/site-packages")]:
    if os.path.exists(local_p) and local_p not in sys.path:
        sys.path.insert(0, local_p)

import torch

_CACHED_MODEL = None
_CACHED_MODEL_NAME = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_official_model(model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
    """Loads and caches the official Qwen3TTSTokenizerV2Model on GPU."""
    global _CACHED_MODEL, _CACHED_MODEL_NAME
    if _CACHED_MODEL is None or _CACHED_MODEL_NAME != model_name:
        from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
        token = os.environ.get("HF_TOKEN")
        model = Qwen3TTSTokenizerV2Model.from_pretrained(model_name, token=token)
        model = model.to(_DEVICE).eval()
        _CACHED_MODEL = model
        _CACHED_MODEL_NAME = model_name
        sys.stderr.write(f"[decode_official_worker] Model '{model_name}' loaded on {_DEVICE}\n")
        sys.stderr.flush()
    return _CACHED_MODEL


def decode_codes_array(
    codes_arr: np.ndarray,
    model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    normalize: bool = True,
) -> np.ndarray:
    """
    Decodes an RVQ codes array into 24kHz float32 audio waveform.
    Accepts shapes: (T, 16), (16, T), (1, T, 16), (1, 16, T), (B, T, 16).
    """
    model = get_official_model(model_name)

    # Standardize shape to (B, T, 16)
    if codes_arr.ndim == 2:
        if codes_arr.shape[0] == 16 and codes_arr.shape[1] != 16:
            codes_arr = codes_arr.T  # (T, 16)
        codes_arr = np.expand_dims(codes_arr, 0)  # (1, T, 16)
    elif codes_arr.ndim == 3:
        if codes_arr.shape[1] == 16 and codes_arr.shape[2] != 16:
            codes_arr = np.transpose(codes_arr, (0, 2, 1))  # (B, T, 16)

    # Clamp to valid codebook vocabulary [0, 2047]
    codes_arr = np.clip(codes_arr, 0, 2047).astype(np.int64)
    codes_t = torch.tensor(codes_arr, dtype=torch.long, device=_DEVICE)

    with torch.no_grad():
        out = model.decode(codes_t)
        if hasattr(out, "audio_values"):
            wav = out.audio_values
        elif hasattr(out, "audio"):
            wav = out.audio
        elif isinstance(out, (list, tuple)):
            wav = out[0]
        else:
            wav = out

        if isinstance(wav, (list, tuple)):
            wav = wav[0]
        if isinstance(wav, torch.Tensor):
            wav_arr = wav.detach().cpu().numpy().squeeze()
        else:
            wav_arr = np.asarray(wav).squeeze()

    wav_arr = wav_arr.astype(np.float32)

    # Peak normalization: percentile-based to reject transient single-sample spikes
    if normalize:
        p_peak = float(np.percentile(np.abs(wav_arr), 99.7))
        if p_peak > 1e-8:
            wav_arr = np.clip(wav_arr / (p_peak * 1.05), -1.0, 1.0).astype(np.float32)
        else:
            max_val = float(np.max(np.abs(wav_arr)))
            if max_val > 1e-8:
                wav_arr = (wav_arr / max_val).astype(np.float32)

    return wav_arr


def run_daemon(model_name: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"):
    """Runs interactive line-by-line daemon service over stdin/stdout."""
    # Pre-warm model on GPU
    try:
        get_official_model(model_name)
    except Exception as e:
        sys.stderr.write(f"[decode_official_worker] Prewarm failed: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

    # Handshake ready signal
    print("READY", flush=True)

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
            cmd = req.get("command")
            if cmd == "exit":
                print(json.dumps({"status": "ok", "message": "exiting"}), flush=True)
                break
            elif cmd == "ping":
                print(json.dumps({"status": "pong"}), flush=True)
                continue

            input_file = req.get("input_file")
            output_file = req.get("output_file")
            normalize = req.get("normalize", True)
            req_model = req.get("model_name", model_name)

            if input_file and os.path.exists(input_file):
                codes_arr = np.load(input_file)
            elif "codes" in req:
                codes_arr = np.array(req["codes"], dtype=np.int64)
            else:
                raise ValueError("Missing 'input_file' or 'codes' in request")

            wav_arr = decode_codes_array(codes_arr, model_name=req_model, normalize=normalize)

            if output_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                np.save(output_file, wav_arr)
                resp = {
                    "status": "ok",
                    "output_file": output_file,
                    "samples": len(wav_arr),
                    "duration_sec": round(len(wav_arr) / 24000.0, 3),
                    "peak": float(np.max(np.abs(wav_arr))),
                    "rms": float(np.sqrt(np.mean(wav_arr**2))),
                }
            else:
                resp = {
                    "status": "ok",
                    "samples": len(wav_arr),
                    "duration_sec": round(len(wav_arr) / 24000.0, 3),
                    "peak": float(np.max(np.abs(wav_arr))),
                    "rms": float(np.sqrt(np.mean(wav_arr**2))),
                }

            print(json.dumps(resp), flush=True)

        except Exception as err:
            err_msg = str(err)
            sys.stderr.write(f"[decode_official_worker] Error processing request: {err_msg}\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            print(json.dumps({"status": "error", "error": err_msg}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Official Qwen3-TTS Tokenizer V2 Audio Decoder Worker")
    parser.add_argument("--daemon", action="store_true", help="Run in persistent JSON line daemon mode")
    parser.add_argument("--input", type=str, help="Path to input .npy file containing RVQ codes")
    parser.add_argument("--output", type=str, help="Path to output .npy file to write float32 waveform")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz", help="HF model name")
    parser.add_argument("--no_normalize", action="store_true", help="Disable peak normalization")

    args = parser.parse_args()

    if args.daemon:
        run_daemon(model_name=args.model_name)
    elif args.input:
        if not os.path.exists(args.input):
            sys.stderr.write(f"Error: input file {args.input} does not exist\n")
            sys.exit(1)
        codes_arr = np.load(args.input)
        wav_arr = decode_codes_array(codes_arr, model_name=args.model_name, normalize=not args.no_normalize)
        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            np.save(args.output, wav_arr)
            print(f"SUCCESS: {args.output}", flush=True)
        else:
            print(f"SUCCESS: decoded {len(wav_arr)} samples ({len(wav_arr)/24000.0:.2f}s)", flush=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
