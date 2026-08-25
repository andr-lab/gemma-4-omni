#!/usr/bin/env python3
"""test_codec_head_transfer.py — Il codec_head preaddestrato (predice cb0 nel Qwen3-TTS
originale) trasferisce su h_sub (Gemma->adapter)? Misura TF top-1/top-5 su val cache.
Varianti: raw, RMSNorm finale del backbone, z-norm."""
import os, sys, glob
sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
import torch
import numpy as np
from safetensors.torch import load_file

device = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load("/workspace/checkpoints/ar_semantic_final.pt", weights_only=False)
blob = torch.load("/workspace/cache_ars_full.pt", weights_only=False)
tr_h, tr_c, va_h, va_c = blob["tr_h"], blob["tr_c"], blob["va_h"], blob["va_c"]
print(f"cache: h dim={tr_h[0].shape[-1]}, val windows={len(va_h)}", flush=True)

st = glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*/model.safetensors")
st = sorted(st)[0]
sd = load_file(st)
W = sd.get("talker.codec_head.weight")
print(f"codec_head.weight: {tuple(W.shape)}", flush=True)
norm_w = sd.get("talker.model.norm.weight")
if norm_w is None:
    norm_w = sd.get("model.norm.weight")
print(f"norm finale backbone: {None if norm_w is None else tuple(norm_w.shape)}", flush=True)
# elenco chiavi talker.model (per capire lo spazio)
k0 = sorted(k for k in sd if k.startswith("talker.model."))[:5]
print("prime chiavi talker.model:", k0, flush=True)

V = int(ck.get("vocab", 2048))
W = W.to(device).float()  # (3072, 2048)


def rms_norm(x, w):
    # x: (T,2048) fp32, w: (2048,) fp16 -> fp32
    rms = x.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
    return (x / rms) * w.float()


@torch.inference_mode()
def eval_transfer(mode):
    hits = t5 = tot = 0
    for hs, cs in zip(va_h, va_c):
        T = min(cs.shape[0], hs.shape[0])
        if T < 8:
            continue
        tgt = cs[:T].to(device).long()
        h = hs[:T].to(device).float()
        if mode == "znorm":
            h = (h - h.mean(0, keepdim=True)) / (h.std(0, keepdim=True) + 1e-5)
        elif mode == "rmsnorm" and norm_w is not None:
            h = rms_norm(h, norm_w.to(device).float())
        logits = h @ W.t()  # (T,3072)
        pred = torch.argmax(logits, dim=-1)
        top5 = torch.topk(logits, 5, dim=-1).indices
        m = tgt != 0
        hits += int((pred[m] == tgt[m]).sum())
        t5 += int(((top5 == tgt.unsqueeze(-1)).any(-1) & m).sum())
        tot += int(m.sum())
    return hits / max(1, tot), t5 / max(1, tot)


for mode in ["raw", "znorm", "rmsnorm"]:
    a, b = eval_transfer(mode)
    print(f"codec_head on h_sub [{mode:8s}]: top1={a:.4f} top5={b:.4f}   (ARS trained: top1=0.1201)", flush=True)
print("FATTO", flush=True)
