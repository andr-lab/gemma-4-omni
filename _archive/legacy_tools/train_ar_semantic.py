#!/usr/bin/env python3
"""
train_ar_semantic.py — Task 12: AR-SemanticDecoder (pattern VALL-E) per cb0 libero.
Revisione agy 24/08 (RECONCILE): AdaLN per-layer, label smoothing, bias log-prior,
scheduled sampling gated su val-acc, repetition-aware sampling solo su run/n-gram.
"""
import os, sys, json, time, argparse

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Modello (revisione agy: condizionamento AdaLN per-layer)
# ---------------------------------------------------------------------------
class AdaLNLayer(nn.Module):
    """Layer transformer causale con Adaptive LayerNorm modulata da h_t."""

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


class ARSemanticDecoder(nn.Module):
    def __init__(self, h_dim=2048, vocab=2048, d_model=1024, n_layers=4, n_heads=8,
                 dropout=0.1, cond_dropout=0.1, max_len=64):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.cond_dropout = cond_dropout
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.h_proj = nn.Linear(h_dim, d_model)
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

    def forward(self, h_t, cb0_in, pad_mask=None):
        """h_t: (B,T,h_dim); cb0_in: (B,T) shiftati a destra (pos0=BOS).
        Ritorna logits (B,T,vocab)."""
        B, T, _ = h_t.shape
        pos = torch.arange(T, device=h_t.device).unsqueeze(0)
        x = self.tok_emb(cb0_in) + self.h_proj(h_t) + self.pos_emb(pos)
        if self.training and self.cond_dropout > 0:
            drop = (torch.rand(B, 1, 1, device=h_t.device) < self.cond_dropout).float()
            x = x - drop * self.h_proj(h_t)
        mask = torch.triu(torch.ones(T, T, device=h_t.device, dtype=torch.bool), diagonal=1)
        for layer in self.layers:
            x = layer(x, h_t, mask)
        return self.head(self.ln(x))


@torch.no_grad()
def generate_ar(model, h_t, max_len=None, temperature=0.9, top_k=50, top_p=0.9):
    """Rolling autoregressivo con repetition-aware sampling stile VALL-E 2:
    - run identici >5 frame → forte penalità sul token ripetuto;
    - n-gram di lunghezza 3 ripetuto ≥3 volte → fallback nucleus (top_p) per rompere il loop.
    h_t: (1,T,h_dim). Ritorna (T,) cb0 predetti."""
    model.eval()
    T = h_t.shape[1]
    BOS = 0
    emitted = [BOS]
    for t in range(min(max_len or T, T)):
        seq = torch.tensor(emitted, device=h_t.device).unsqueeze(0)
        h_ctx = h_t[:, : t + 1, :]
        logits = model(h_ctx, seq)[0, -1].float()

        # 1) run identico >5 frame → penalità sul token in loop
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

        # 2) n-gram len3 ripetuto ≥3 volte → fallback nucleus (rompe il loop stocastico)
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


def val_accuracy(model, va_h, va_c, V, device):
    """TF top-1 accuracy su val (finestre val precompute)."""
    model.eval()
    hits, tot = 0, 0
    with torch.inference_mode():
        for hs, cs in zip(va_h, va_c):
            T = min(cs.shape[0], hs.shape[0])
            if T < 4:
                continue
            tgt = cs[:T].to(device)
            cb0_in = torch.cat([torch.tensor([[0]], device=device), tgt[:-1].unsqueeze(0)], dim=1)
            logits = model(hs[:T].unsqueeze(0).to(device), cb0_in)
            pred = torch.argmax(logits, dim=-1).squeeze(0)
            m = tgt != 0
            hits += int((pred[m] == tgt[m]).sum())
            tot += int(m.sum())
    return hits / max(1, tot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet")
    ap.add_argument("--adapter-ckpt", default="/workspace/checkpoints/ht_adapter_temporal2_final.pt")
    ap.add_argument("--output", default="/workspace/checkpoints/ar_semantic_final.pt")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("ARS_STEPS", "100000")))
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--ss-gate", type=float, default=0.25,
                    help="val-acc minima per attivare scheduled sampling (revisione agy)")
    ap.add_argument("--ss-max", type=float, default=0.15,
                    help="frazione massima di token predetti sostituiti")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 6

    print("[ARS] 1/5 Carico sistema + adapter temporale CONGELATO...", flush=True)
    import kaggle_pipeline as KP
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(args.adapter_ckpt))
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)
    if hasattr(duplex_model, "semantic_head"):
        duplex_model.semantic_head.to(device)

    print("[ARS] 2/5 Dataset CV26...", flush=True)
    prefix = args.dataset.replace(".parquet", "")
    user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(prefix + ".parquet")
    t_agent = agent_codes.shape[1]
    seg_ends = []
    sidecar = prefix + "_hyperparams.json"
    if os.path.isfile(sidecar):
        seg_ends = json.load(open(sidecar)).get("SEGMENT_ENDS", [])
    val_start = int(0.90 * t_agent)
    for e in seg_ends:
        if e >= val_start:
            val_start = e
            break
    MPF = KP.MEL_PER_AGENT_FRAME
    ds_train = KP.FullDuplexDataset(user_mel[:val_start*MPF], agent_mel[:val_start*MPF],
                                    agent_codes[:, :val_start], seq_len=64)
    ds_val = KP.FullDuplexDataset(user_mel[val_start*MPF:], agent_mel[val_start*MPF:],
                                  agent_codes[:, val_start:], seq_len=64)

    print("[ARS] 3/5 Estrazione h_t/cb0 (cache su disco)...", flush=True)
    cache_f = "/workspace/cache_ars_smoke.pt" if args.smoke else "/workspace/cache_ars_full.pt"
    if os.path.isfile(cache_f):
        blob = torch.load(cache_f, weights_only=False)
        tr_h, tr_c, va_h, va_c = blob["tr_h"], blob["tr_c"], blob["va_h"], blob["va_c"]
        print(f"[ARS] cache caricata: {len(tr_h)} train / {len(va_h)} val", flush=True)
    else:
        def extract(ds, limit):
            hs, cs = [], []
            fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
            afs = 2 + KP.TEXT_BLOCK_LEN
            with torch.inference_mode():
                for i in range(min(limit, len(ds))):
                    item = ds[i]
                    T = item["audio_targets"].shape[0]
                    inp = {"input_ids": item["input_ids"].unsqueeze(0).to(device),
                           "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
                           "input_features": item["input_features"].unsqueeze(0).to(device),
                           "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device)}
                    go = duplex_model.base_model(**inp, output_hidden_states=True, return_dict=True)
                    lh = go.hidden_states[-1]
                    ah = lh[:, afs+fpf-1 : afs+fpf*T : fpf, :].float()
                    ad = duplex_model._apply_causal_conv(
                        ah.to(next(duplex_model.causal_conv.parameters()).dtype)).float()
                    h_t = duplex_model.ht_adapter(
                        ad.to(next(duplex_model.ht_adapter.parameters()).dtype)).float()
                    hs.append(h_t.squeeze(0).cpu())
                    cs.append(item["audio_targets"][:, 0].clone())
            return hs, cs
        n_tr = min(len(ds_train), 300) if args.smoke else len(ds_train)
        n_va = min(len(ds_val), 40) if args.smoke else min(len(ds_val), 100)
        tr_h, tr_c = extract(ds_train, n_tr)
        va_h, va_c = extract(ds_val, n_va)
        torch.save({"tr_h": tr_h, "tr_c": tr_c, "va_h": va_h, "va_c": va_c}, cache_f)
        print(f"[ARS] cache scritta ({cache_f}): {len(tr_h)} train / {len(va_h)} val", flush=True)

    V = max(int(max(max(c.max().item() for c in tr_c), max(c.max().item() for c in va_c))) + 1, 2048)
    V = int(V)
    model = ARSemanticDecoder(h_dim=tr_h[0].shape[-1], vocab=V).to(device)

    # bias init con log-prior della distribuzione empirica cb0 (revisione agy)
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
    ss_active = False

    print(f"[ARS] 4/5 Training {args.steps} step (sched sampling gate={args.ss_gate})...", flush=True)
    model.train()
    t0 = time.time()
    step = 0
    last_val = 0.0
    while step < args.steps:
        for hs, cs in zip(tr_h, tr_c):
            if step >= args.steps:
                break
            T = min(cs.shape[0], hs.shape[0])
            if T < 8:
                continue
            tgt = cs[:T].to(device)
            h_seq = hs[:T].unsqueeze(0).to(device)
            cb0_in = torch.cat([torch.tensor([[BOS]], device=device), tgt[:-1].unsqueeze(0)], dim=1)

            # scheduled sampling: solo se val-acc ha superato il gate (revisione agy)
            if ss_active:
                with torch.no_grad():
                    logits_full = model(h_seq, cb0_in)
                    pred = torch.argmax(logits_full, dim=-1)
                frac = args.ss_max
                mix_mask = torch.rand_like(tgt.unsqueeze(0), dtype=torch.float32) < frac
                cb0_in = torch.where(mix_mask, pred[:, :T], cb0_in)

            logits = model(h_seq, cb0_in)
            loss = F.cross_entropy(logits.view(-1, V), tgt.unsqueeze(0).expand(1, -1).reshape(-1),
                                   ignore_index=BOS, label_smoothing=args.label_smoothing)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1

            if step % 500 == 0 or step == 1:
                val_acc = val_accuracy(model, va_h, va_c, V, device)
                model.train()
                if not ss_active and val_acc >= args.ss_gate:
                    ss_active = True
                    print(f"[ARS] step {step}: val-acc {val_acc:.3f} >= gate → scheduled sampling ATTIVO",
                          flush=True)
                print(f"[ARS] step {step}/{args.steps} loss={loss.item():.4f} "
                      f"val_tf={val_acc:.4f} ss={ss_active} ({time.time()-t0:.0f}s)", flush=True)

    val_acc = val_accuracy(model, va_h, va_c, V, device)
    print(f"[ARS] Finale Val TF-accuracy: {val_acc:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "vocab": V, "tf_acc": val_acc,
                "steps": args.steps}, args.output)
    print(f"[ARS] 5/5 Salvato {args.output}", flush=True)
    print("[ARS] FATTO.", flush=True)


if __name__ == "__main__":
    main()
