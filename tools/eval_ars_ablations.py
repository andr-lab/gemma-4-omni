#!/usr/bin/env python3
"""eval_ars_ablations.py — diagnostico dal cache, senza retraining:
1) TF top-1/top-5 su vocabolario pieno, con shift temporale h_t in [-2..+2] (test allineamento)
2) ceiling con vocabolario ristretto ai codici cb0 visti nel training (240-ish)
3) top-1 restricted a varie temperature di generazione? (no: solo analisi logits qui)
"""
import sys, os, json
sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
import torch
import numpy as np

sys.path.insert(0, "/workspace/tools")
from train_ar_semantic import ARSemanticDecoder

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("/workspace/checkpoints/ar_semantic_final.pt", weights_only=False)
blob = torch.load("/workspace/cache_ars_full.pt", weights_only=False)
tr_h, tr_c, va_h, va_c = blob["tr_h"], blob["tr_c"], blob["va_h"], blob["va_c"]
V = int(ck["vocab"])
print(f"checkpoint steps={ck.get('steps')} tf_acc={ck.get('tf_acc'):.4f} vocab={V}", flush=True)

model = ARSemanticDecoder(h_dim=tr_h[0].shape[-1], vocab=V).to(device)
model.load_state_dict(ck["model"])
model.eval()

# codici visti nel train (escludendo BOS=0)
seen = set()
for c in tr_c[:8000]:
    seen.update(c.tolist())
seen.discard(0)
seen = sorted(seen)
seen_t = torch.tensor(seen, device=device)
mask_full = torch.full((V,), float("-inf"), device=device)
mask_full[seen_t] = 0.0
print(f"codici cb0 visti nel train: {len(seen)} (max={max(seen)})", flush=True)


@torch.inference_mode()
def eval_split(shift, restrict):
    hits = tot = 0
    t5 = 0
    for hs, cs in zip(va_h, va_c):
        T = min(cs.shape[0], hs.shape[0])
        if T < 8:
            continue
        tgt = cs[:T].to(device)
        idx = torch.clamp(torch.arange(T, device=device) + shift, 0, T - 1)
        hs_s = hs.to(device)[idx].unsqueeze(0)
        cb0_in = torch.cat([torch.tensor([[0]], device=device), tgt[:-1].unsqueeze(0)], dim=1)
        logits = model(hs_s, cb0_in)[0]
        if restrict:
            logits = logits + mask_full
        pred = torch.argmax(logits, dim=-1)
        m = tgt != 0
        hits += int((pred[m] == tgt[m]).sum())
        # top-5
        top5 = torch.topk(logits, 5, dim=-1).indices
        t5 += int(((top5 == tgt.unsqueeze(-1)).any(dim=-1) & m).sum())
        tot += int(m.sum())
    return hits / max(1, tot), t5 / max(1, tot)


print("shift | full-vocab top1 | restrict top1 | restrict top5", flush=True)
for s in [-2, -1, 0, 1, 2]:
    a = eval_split(s, restrict=False)[0]
    b, c5 = eval_split(s, restrict=True)
    print(f"  {s:+d}  |   {a:.4f}        |   {b:.4f}      |   {c5:.4f}", flush=True)
print("FATTO", flush=True)
