#!/usr/bin/env python3
"""
train_ar_semantic_v4.py — AR-SemanticDecoder con ANCORA TESTUALE (P3).

Basato su v3 (P1 input corruption + P2 hybrid TF/free-running) + ancoraggio
testuale (docs/DESIGN_FULLDUPLEX_LETTERATURA.md §2.0, Moshi inner monologue
2410.00037, Mini-Omni text-instructed 2408.16725):

  P3. ANCORA TESTUALE: il contenuto semantico viene da Gemma (testo della
      risposta agente, gia' nel prompt). h_text = mean-pooling degli hidden di
      Gemma sul blocco testo (cache v2, extract_ars_cache_v2.py). Il predittore
      cb0 riceve h_text come condizione GLOBALE (proiettata e sommata all'input
      a ogni passo): il testo fissa il contenuto, cb0 lo realizza foneticamente.
      -> gli errori cb0 diventano errori acustici tollerati (mel cos 0.92-0.99),
         non semantici (chiude il gap "free vs ancorato").

Cache richiesta: cache_ars_full_v2.pt (tr_h= h_sub, tr_htx= h_text, tr_c= cb0).
"""
import os, sys, json, time, argparse

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modello: come v3 + condizionamento testuale globale
# ---------------------------------------------------------------------------
class AdaLNLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout, h_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model))
        self.ada = nn.Sequential(
            nn.Linear(h_dim, 4 * d_model), nn.SiLU(),
            nn.Linear(4 * d_model, 4 * d_model))

    def forward(self, x, h_t, mask):
        gb = self.ada(h_t)                       # (B,T,4d)
        g1, b1, g2, b2 = gb.chunk(4, dim=-1)
        x1 = self.norm1(x) * (1 + g1) + b1
        x = x + self.attn(x1, x1, x1, attn_mask=mask)[0]
        x2 = self.norm2(x) * (1 + g2) + b2
        x = x + self.ffn(x2)
        return x


class ARSemanticDecoderV4(nn.Module):
    """v3 + text_cond: h_text (pooling del blocco testo) sommato all'input.

    h_t: (B,T,h_dim) condizione per-frame (AdaLN)
    h_text: (B,h_text_dim) condizione globale (contenuto, inner monologue)
    """

    def __init__(self, h_dim=2048, text_dim=1536, vocab=2048, d_model=1024,
                 n_layers=4, n_heads=8, dropout=0.1, cond_dropout=0.1,
                 text_dropout=0.15, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.cond_dropout = cond_dropout
        self.text_dropout = text_dropout
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.h_proj = nn.Linear(h_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.pos_emb = nn.Embedding(max_len + 1, d_model)
        self.layers = nn.ModuleList([
            AdaLNLayer(d_model, n_heads, 4 * d_model, dropout, h_dim)
            for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)
        self.head_bias_init = False

    def init_head_bias(self, log_prior):
        with torch.no_grad():
            self.head.bias.copy_(log_prior)
        self.head_bias_init = True

    def forward(self, h_t, h_text, cb0_in):
        """h_t: (B,T,h_dim); h_text: (B,text_dim); cb0_in: (B,T) shiftati."""
        B, T, _ = h_t.shape
        pos = torch.arange(T, device=h_t.device).unsqueeze(0)
        x = self.tok_emb(cb0_in) + self.h_proj(h_t) + self.pos_emb(pos)
        # ancora testuale globale: sommata a ogni posizione
        t_cond = self.text_proj(h_text).unsqueeze(1)          # (B,1,d)
        if self.training and self.text_dropout > 0:
            drop = (torch.rand(B, 1, 1, device=h_t.device) < self.text_dropout).float()
            t_cond = t_cond * (1 - drop)
        x = x + t_cond
        if self.training and self.cond_dropout > 0:
            drop = (torch.rand(B, 1, 1, device=h_t.device) < self.cond_dropout).float()
            x = x - drop * self.h_proj(h_t)
        mask = torch.triu(torch.ones(T, T, device=h_t.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, h_t, mask)
        return self.head(self.ln(x))


@torch.no_grad()
def generate_ar(model, h_t, h_text, max_len=None, temperature=0.9, top_k=50, top_p=0.9):
    """Rolling autoregressivo (v3) + ancora testuale globale fissa."""
    model.eval()
    T = h_t.shape[1]
    BOS = 0
    emitted = [BOS]
    for t in range(min(max_len or T, T)):
        seq = torch.tensor(emitted, device=h_t.device).unsqueeze(0)
        h_ctx = h_t[:, : t + 1, :]
        logits = model(h_ctx, h_text, seq)[0, -1].float()

        if t >= 6:
            last = emitted[-1]
            run = 1
            for k in range(len(emitted) - 2, -1, -1):
                if emitted[k] == last:
                    run += 1
                else:
                    break
            if run > 5:
                logits[last] /= 1.9 if logits[last] > 0 else -1.9

        loop_break = False
        if t >= 8:
            pat = tuple(emitted[-3:])
            cnt = sum(1 for i in range(len(emitted) - 3) if tuple(emitted[i:i + 3]) == pat)
            if cnt >= 3:
                loop_break = True

        if loop_break:
            logits = logits / max(temperature * 1.4, 1e-5)
            probs = torch.softmax(logits, dim=-1)
            sorted_p, idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_p, dim=0)
            keep = cum <= top_p
            keep[0] = True
            probs = probs.clone()
            probs[idx[~keep]] = 0.0
            probs = probs / probs.sum()
            nxt = int(torch.multinomial(probs, 1))
        else:
            logits = logits / max(temperature, 1e-5)
            if top_k and 0 < top_k < logits.numel():
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[-1]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            nxt = int(torch.multinomial(probs, 1))
        emitted.append(nxt)
    return np.array(emitted[1:], dtype=np.int64)


# ---------------------------------------------------------------------------
# P1: input corruption (arXiv:2607.29363)
# ---------------------------------------------------------------------------
def corrupt_prefix(cb0_in, eps, V, BOS=0):
    if eps <= 0:
        return cb0_in
    mask = (torch.rand(cb0_in.shape, device=cb0_in.device) < eps) & (cb0_in != BOS)
    rnd = torch.randint(1, V, cb0_in.shape, device=cb0_in.device)
    return torch.where(mask, rnd, cb0_in)


# ---------------------------------------------------------------------------
# P2: free-running per hybrid training (arXiv:2509.17021)
# ---------------------------------------------------------------------------
@torch.no_grad()
def free_run(model, h_seq, h_text, cb0_in_gt, K, V, temperature=1.0, top_k=50):
    seq = cb0_in_gt.clone()
    T = seq.shape[1]
    start = max(T - K, 1)
    confs = []
    for i in range(start, T):
        logits = model(h_seq[:, :i, :], h_text, seq[:, :i])[0, -1].float() / max(temperature, 1e-5)
        if top_k and 0 < top_k < logits.numel():
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        if i - start < 4:
            confs.append(float(probs.max().item()))
        seq[0, i] = torch.multinomial(probs, 1)
    return seq, (sum(confs) / len(confs) if confs else 1.0)


def val_accuracy(model, va_h, va_tx, va_c, V, device):
    model.eval()
    hits, tot = 0, 0
    with torch.inference_mode():
        for hs, tx, cs in zip(va_h, va_tx, va_c):
            T = min(cs.shape[0], hs.shape[0])
            if T < 4:
                continue
            tgt = cs[:T].to(device)
            cb0_in = torch.cat([torch.tensor([[0]], device=device), tgt[:-1].unsqueeze(0)], dim=1)
            logits = model(hs[:T].unsqueeze(0).to(device),
                           tx.unsqueeze(0).to(device), cb0_in)
            pred = torch.argmax(logits, dim=-1).squeeze(0)
            m = tgt != 0
            hits += int((pred[m] == tgt[m]).sum())
            tot += int(m.sum())
    return hits / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/cache_ars_full_v2.pt",
                    help="cache v2: tr_h/tr_sub, tr_htx, tr_c (extract_ars_cache_v2.py)")
    ap.add_argument("--output", default="/workspace/checkpoints/ar_semantic_v4.pt")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("ARS_STEPS", "100000")))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--corrupt-eps", type=float, default=0.15)
    ap.add_argument("--hybrid-k", type=int, default=8)
    ap.add_argument("--hybrid-w", type=float, default=0.5)
    ap.add_argument("--hybrid-every", type=int, default=2)
    ap.add_argument("--hybrid-start", type=int, default=1000)
    ap.add_argument("--eos-conf", type=float, default=0.02)
    ap.add_argument("--text-dropout", type=float, default=0.15,
                    help="dropout sull'ancora testuale (robustezza a testo mancante)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 6
        args.hybrid_start = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[ARS4] 1/3 Carico cache v2: {args.cache}", flush=True)
    if not os.path.isfile(args.cache):
        raise SystemExit(f"[ARS4] cache v2 assente: {args.cache}\n"
                         "Esegui prima tools/extract_ars_cache_v2.py (h_raw+h_text)")
    blob = torch.load(args.cache, weights_only=False)
    tr_h = blob.get("tr_sub", blob.get("tr_h"))
    va_h = blob.get("va_sub", blob.get("va_h"))
    tr_tx = blob.get("tr_htx")
    va_tx = blob.get("va_htx")
    tr_c, va_c = blob["tr_c"], blob["va_c"]
    if tr_tx is None or va_tx is None:
        raise SystemExit("[ARS4] cache senza h_text (tr_htx/va_htx): rigenerare con extract_ars_cache_v2.py")
    print(f"[ARS4] cache: {len(tr_h)} train / {len(va_h)} val "
          f"(h_sub {tr_h[0].shape}, h_text {tr_tx[0].shape})", flush=True)

    V = max(int(max(max(c.max().item() for c in tr_c), max(c.max().item() for c in va_c))) + 1, 2048)
    V = int(V)
    model = ARSemanticDecoderV4(h_dim=tr_h[0].shape[-1], text_dim=tr_tx[0].shape[-1],
                                vocab=V, text_dropout=args.text_dropout).to(device)

    from collections import Counter
    cnt = Counter()
    for c in tr_c[:2000]:
        cnt.update(c.tolist())
    total = sum(cnt.values())
    log_prior = torch.full((V,), -20.0)
    for k, v in cnt.items():
        if 0 <= k < V:
            log_prior[k] = np.log((v + 1) / (total + V))
    model.init_head_bias(log_prior.to(device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    BOS = 0
    n_free = 0
    n_skipped = 0

    print(f"[ARS4] 2/3 Training {args.steps} step "
          f"(corrupt={args.corrupt_eps}, hybrid k={args.hybrid_k} w={args.hybrid_w}, "
          f"text_dropout={args.text_dropout})...", flush=True)
    model.train()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for hs, tx, cs in zip(tr_h, tr_tx, tr_c):
            if step >= args.steps:
                break
            T = min(cs.shape[0], hs.shape[0])
            if T < 8:
                continue
            tgt = cs[:T].to(device)
            h_seq = hs[:T].unsqueeze(0).to(device)
            h_text = tx.unsqueeze(0).to(device)
            cb0_in_gt = torch.cat([torch.tensor([[BOS]], device=device), tgt[:-1].unsqueeze(0)], dim=1)

            # P1: input corruption
            cb0_in = corrupt_prefix(cb0_in_gt, args.corrupt_eps, V, BOS=BOS)

            logits = model(h_seq, h_text, cb0_in)
            loss = F.cross_entropy(logits.view(-1, V), tgt.unsqueeze(0).expand(1, -1).reshape(-1),
                                   ignore_index=BOS, label_smoothing=args.label_smoothing)

            # P2: hybrid free-running
            do_hybrid = (args.hybrid_k > 0 and step >= args.hybrid_start
                         and step % args.hybrid_every == 0 and T > args.hybrid_k + 8)
            if do_hybrid:
                seq_free, conf = free_run(model, h_seq, h_text, cb0_in_gt, args.hybrid_k, V)
                if conf >= args.eos_conf:
                    logits_h = model(h_seq, h_text, seq_free)
                    loss_h = F.cross_entropy(
                        logits_h.view(-1, V), tgt.unsqueeze(0).expand(1, -1).reshape(-1),
                        ignore_index=BOS, label_smoothing=args.label_smoothing)
                    loss = loss + args.hybrid_w * loss_h
                    n_free += 1
                else:
                    n_skipped += 1

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1

            if step % 500 == 0 or step == 1:
                val_acc = val_accuracy(model, va_h, va_tx, va_c, V, device)
                model.train()
                print(f"[ARS4] step {step}/{args.steps} loss={loss.item():.4f} "
                      f"val_tf={val_acc:.4f} free={n_free} skip={n_skipped} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    val_acc = val_accuracy(model, va_h, va_tx, va_c, V, device)
    print(f"[ARS4] Finale Val TF-accuracy: {val_acc:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "vocab": V, "tf_acc": val_acc,
                "steps": args.steps, "corrupt_eps": args.corrupt_eps,
                "hybrid_k": args.hybrid_k, "hybrid_w": args.hybrid_w,
                "text_dropout": args.text_dropout}, args.output)
    print(f"[ARS4] 3/3 Salvato {args.output}", flush=True)
    print("[ARS4] FATTO.", flush=True)


if __name__ == "__main__":
    main()
