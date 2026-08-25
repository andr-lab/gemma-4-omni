#!/usr/bin/env python3
"""probe_cb0_ceiling.py — Ceiling informativo di h_sub per cb0 (stessa cache).
1) probe lineare (logistic) su h_sub per-frame
2) MLP 2 layer per-frame
3) transformer bidirezionale (senza causalita') su finestra -> ceiling di capacita'
Se (3) ~ (12-15%): h_sub NON contiene abbastanza info -> serve ritrainare l'adapter
con pressione cb0. Se (3) >> 12%: il problema e' capacita'/causalita' -> opzione B."""
import os, sys
sys.path.insert(0, "/workspace")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
blob = torch.load("/workspace/cache_ars_full.pt", weights_only=False)
tr_h, tr_c, va_h, va_c = blob["tr_h"], blob["tr_c"], blob["va_h"], blob["va_c"]
V = 2048
print(f"cache: h dim={tr_h[0].shape[-1]}", flush=True)

# ---- flatten per-frame per i probe locali ----
def flat(hs, cs):
    X, Y = [], []
    for h, c in zip(hs, cs):
        T = min(h.shape[0], c.shape[0])
        X.append(h[:T]); Y.append(c[:T])
    return torch.cat(X).float(), torch.cat(Y).long()

Xtr, Ytr = flat(tr_h, tr_c)
Xva, Yva = flat(va_h, va_c)
m_tr = Ytr != 0
m_va = Yva != 0
Xtr, Ytr, Xva, Yva = Xtr[m_tr], Ytr[m_tr], Xva[m_va], Yva[m_va]
Xtr, Ytr = Xtr.to(device), Ytr.to(device)
Xva, Yva = Xva.to(device), Yva.to(device)
# sottocampiona per i probe (memoria)
rng = np.random.default_rng(0)
i_tr = rng.choice(Xtr.shape[0], min(120000, Xtr.shape[0]), replace=False)
i_va = rng.choice(Xva.shape[0], min(20000, Xva.shape[0]), replace=False)
Xtr, Ytr = Xtr[i_tr], Ytr[i_tr]
Xva, Yva = Xva[i_va], Yva[i_va]
print(f"frame train={Xtr.shape[0]} val={Xva.shape[0]} (sottocampionati)", flush=True)

def acc(pred, y):
    return float((pred == y).float().mean())

# ---- 1) probe lineare ----
lin = nn.Linear(Xtr.shape[1], V).to(device)
opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
for it in range(600):
    opt.zero_grad()
    loss = F.cross_entropy(lin(Xtr), Ytr)
    loss.backward(); opt.step()
with torch.inference_mode():
    print(f"probe lineare: top1={acc(lin(Xva).argmax(-1), Yva):.4f}", flush=True)

# ---- 2) MLP 2 layer ----
mlp = nn.Sequential(nn.Linear(Xtr.shape[1], 1024), nn.GELU(), nn.Linear(1024, V)).to(device)
opt = torch.optim.AdamW(mlp.parameters(), lr=1e-3)
for it in range(1200):
    opt.zero_grad()
    loss = F.cross_entropy(mlp(Xtr), Ytr)
    loss.backward(); opt.step()
with torch.inference_mode():
    print(f"probe MLP:     top1={acc(mlp(Xva).argmax(-1), Yva):.4f}", flush=True)
class BiXf(nn.Module):
    def __init__(self, d=1024, L=6, H=8, maxlen=64):
        super().__init__()
        self.proj = nn.Linear(Xtr.shape[1], d)
        self.pos = nn.Parameter(torch.zeros(1, maxlen, d))
        layer = nn.TransformerEncoderLayer(d, H, 4 * d, batch_first=True, dropout=0.1, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L)
        self.head = nn.Linear(d, V)
    def forward(self, x):
        B, T, _ = x.shape
        z = self.proj(x) + self.pos[:, :T]
        return self.head(self.enc(z))

def windows(hs, cs, T=64):
    H, C = [], []
    for h, c in zip(hs, cs):
        t = min(T, h.shape[0], c.shape[0])
        if t < 16:
            continue
        H.append(h[:t].float()); C.append(c[:t].long())
    return H, C

Wtr_h, Wtr_c = windows(tr_h, tr_c)
Wva_h, Wva_c = windows(va_h, va_c)
print(f"window train={len(Wtr_h)} val={len(Wva_h)}", flush=True)
bix = BiXf().to(device)
opt = torch.optim.AdamW(bix.parameters(), lr=3e-4)
import random
random.seed(0)
bix.train()
for it in range(6000):
    idx = random.sample(range(len(Wtr_h)), 32)
    hs = torch.stack([Wtr_h[i] for i in idx]).to(device)
    cs = torch.stack([Wtr_c[i] for i in idx]).to(device)
    logits = bix(hs)
    loss = F.cross_entropy(logits.reshape(-1, V), cs.reshape(-1), ignore_index=0)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(bix.parameters(), 1.0)
    opt.step()
    if it % 1500 == 0:
        print(f"  bix it{it} loss={loss.item():.3f}", flush=True)
bix.eval()
with torch.inference_mode():
    hits = tot = 0
    for h, c in zip(Wva_h, Wva_c):
        logits = bix(h.unsqueeze(0).to(device))[0]
        pred = logits.argmax(-1)
        m = c != 0
        hits += int((pred[m] == c[m]).sum()); tot += int(m.sum())
    print(f"transformer bidirezionale: top1={hits/max(1,tot):.4f}", flush=True)
print("FATTO", flush=True)
