#!/usr/bin/env python3
"""
train_ar_semantic.py — Task 12: AR-SemanticDecoder (pattern VALL-E) per cb0 libero.

Architettura (sopra l'adapter temporale CONGELATO validato nel Task 11I-bis):
  input passo t: [emb(cb0_{t-1}) ; proj(h_t_{≤t})] → transformer causale → logits cb0_t
Training: teacher forcing CE → scheduled sampling (mischia input GT e predetti).
Inferenza: rolling step-by-step con repetition-aware sampling (VALL-E 2).

Output: checkpoints/ar_semantic_final.pt + report.
"""
import os, sys, json, time, argparse

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggle_pipeline as KP


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------
class ARSemanticDecoder(nn.Module):
    def __init__(self, h_dim=2048, vocab=2048, d_model=1024, n_layers=4, n_heads=8,
                 dropout=0.1, cond_dropout=0.1, max_len=512):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.cond_dropout = cond_dropout
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.h_proj = nn.Linear(h_dim, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, h_t, cb0_in, pad_mask=None):
        """h_t: (B,T,h_dim) condizioni; cb0_in: (B,T) token input shiftati a destra
        (posizione 0 = token BOS=0). Ritorna logits (B,T,vocab)."""
        B, T, _ = h_t.shape
        pos = torch.arange(T, device=h_t.device).unsqueeze(0)
        x = self.tok_emb(cb0_in) + self.h_proj(h_t) + self.pos_emb(pos)

        # condition dropout (abilita CFG futuro): a volte togli h_t
        if self.training and self.cond_dropout > 0:
            drop = (torch.rand(B, 1, 1, device=h_t.device) < self.cond_dropout).float()
            x = x - drop * self.h_proj(h_t)

        mask = torch.triu(torch.ones(T, T, device=h_t.device, dtype=torch.bool), diagonal=1)
        x = self.encoder(x, mask=mask, src_key_padding_mask=pad_mask)
        return self.head(self.ln(x))


@torch.no_grad()
def generate_ar(model, h_t, max_len=None, temperature=0.9, top_k=50,
                repetition_window=12, repetition_penalty=1.4):
    """Rolling autoregressivo con repetition-aware sampling (stile VALL-E 2).
    h_t: (1,T,h_dim). Ritorna (T,) cb0 predetti."""
    model.eval()
    T = h_t.shape[1]
    BOS = 0
    emitted = [BOS]
    for t in range(min(max_len or T, T)):
        seq = torch.tensor(emitted, device=h_t.device).unsqueeze(0)  # (1,t+1)
        h_ctx = h_t[:, : t + 1, :]
        logits = model(h_ctx, seq)[0, -1]  # (vocab,)
        # repetition-aware penalty sul contesto recente
        recent = emitted[-repetition_window:]
        for tok in set(recent):
            if logits[tok] > 0:
                logits[tok] /= repetition_penalty
            else:
                logits[tok] *= repetition_penalty
        logits = logits / max(temperature, 1e-5)
        if top_k and 0 < top_k < logits.numel():
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, 1))
        emitted.append(nxt)
    return np.array(emitted[1:], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet")
    ap.add_argument("--adapter-ckpt", default="/workspace/checkpoints/ht_adapter_temporal2_final.pt")
    ap.add_argument("--output", default="/workspace/checkpoints/ar_semantic_final.pt")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("ARS_STEPS", "100000")))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sched-sampling-start", type=int, default=1200,
                    help="step da cui inizia lo scheduled sampling")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 6

    print("[ARS] 1/5 Carico sistema + adapter temporale CONGELATO...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(args.adapter_ckpt))
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()

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
    from torch.utils.data import DataLoader
    loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = next(duplex_model.base_model.parameters()).device
    # Allinea esplicitamente i moduli al device del backbone
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)
    if hasattr(duplex_model, "semantic_head"):
        duplex_model.semantic_head.to(device)
    fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
    afs = 2 + KP.TEXT_BLOCK_LEN

    # precomputa h_t_pred e cb0 GT per tutte le finestre (forward congelato, una volta sola)
    print(f"[ARS] 3/5 Estrazione h_t/cb0 su {len(ds_train)} finestre train "
          f"+ {len(ds_val)} val (una sola passata)...", flush=True)
    def extract(ds, limit):
        hs, cs = [], []
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
                ad = duplex_model._apply_causal_conv(ah.to(next(duplex_model.causal_conv.parameters()).dtype))
                ad = ad.float()
                h_t = duplex_model.ht_adapter(ad.to(next(duplex_model.ht_adapter.parameters()).dtype)).float()
                hs.append(h_t.squeeze(0).cpu())
                cs.append(item["audio_targets"][:, 0].clone())  # cb0 GT
        return hs, cs
    cache_f = "/workspace/cache_ars_smoke.pt" if args.smoke else "/workspace/cache_ars_full.pt"
    if os.path.isfile(cache_f):
        blob = torch.load(cache_f, weights_only=False)
        tr_h, tr_c, va_h, va_c = blob["tr_h"], blob["tr_c"], blob["va_h"], blob["va_c"]
        print(f"[ARS] cache caricata: {len(tr_h)} train / {len(va_h)} val", flush=True)
    else:
        n_tr = min(len(ds_train), 300) if args.smoke else len(ds_train)
        n_va = min(len(ds_val), 40) if args.smoke else min(len(ds_val), 100)
        tr_h, tr_c = extract(ds_train, n_tr)
        va_h, va_c = extract(ds_val, n_va)
        torch.save({"tr_h": tr_h, "tr_c": tr_c, "va_h": va_h, "va_c": va_c}, cache_f)
        print(f"[ARS] cache scritta ({cache_f}): {len(tr_h)} train / {len(va_h)} val", flush=True)

    V = int(tr_c[0].max().item()) + 1
    V = max(V, 2048)
    model = ARSemanticDecoder(h_dim=tr_h[0].shape[-1], vocab=V).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    BOS = 0

    print(f"[ARS] 4/5 Training {args.steps} step (sched sampling da {args.sched_sampling_start})...", flush=True)
    model.train()
    t0 = time.time()
    step = 0
    while step < args.steps:
        for hs, cs in zip(tr_h, tr_c):
            if step >= args.steps:
                break
            T = min(cs.shape[0], hs.shape[0])
            if T < 8:
                continue
            tgt = cs[:T].to(device)
            h_seq = hs[:T].unsqueeze(0).to(device)
            # teacher forcing base
            cb0_in = torch.cat([torch.tensor([[BOS]], device=device), tgt[:-1].unsqueeze(0)], dim=1)
            # scheduled sampling: sostituisci alcuni input con predetti dal modello corrente
            if step >= args.sched_sampling_start:
                with torch.no_grad():
                    logits_full = model(h_seq, cb0_in)
                    pred = torch.argmax(logits_full, dim=-1)
                mix_mask = (torch.rand_like(tgt.unsqueeze(0), dtype=torch.float32) < 0.35)
                cb0_in = torch.where(mix_mask, pred[:, :T], cb0_in)
            logits = model(h_seq, cb0_in)
            loss = F.cross_entropy(logits.view(-1, V), tgt.unsqueeze(0).expand(1, -1).reshape(-1),
                                   ignore_index=BOS)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); step += 1
            if step % 200 == 0 or step == 1:
                print(f"[ARS] step {step}/{args.steps} loss={loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)

    # eval veloce: accuracy teacher-forced su val
    model.eval()
    hits, tot = 0, 0
    with torch.inference_mode():
        for hs, cs in zip(va_h, va_c):
            T = min(cs.shape[0], hs.shape[0])
            if T < 4: continue
            tgt = cs[:T].to(device)
            cb0_in = torch.cat([torch.tensor([[BOS]], device=device), tgt[:-1].unsqueeze(0)], dim=1)
            logits = model(hs[:T].unsqueeze(0).to(device), cb0_in)
            pred = torch.argmax(logits, dim=-1).squeeze(0)
            m = tgt != BOS
            hits += int((pred[m] == tgt[m]).sum()); tot += int(m.sum())
    tf_acc = hits / max(1, tot)
    print(f"[ARS] Val TF-accuracy: {tf_acc:.4f}", flush=True)

    torch.save({"model": model.state_dict(), "vocab": V, "tf_acc": tf_acc}, args.output)
    print(f"[ARS] 5/5 Salvato {args.output}", flush=True)
    print("[ARS] FATTO.", flush=True)


if __name__ == "__main__":
    main()
