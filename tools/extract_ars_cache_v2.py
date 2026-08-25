#!/usr/bin/env python3
"""
extract_ars_cache_v2.py — Cache estesa per il design evidence-based (P3 + test h_raw).

Motivazione (docs/DESIGN_FULLDUPLEX_LETTERATURA.md §2.0):
  - il predittore cb0 attuale usa h_sub (output adapter ACUSTICO, addestrato per i
    residui cb1..15) -> probe per-frame 3.7%: h_sub non contiene l'info cb0.
  - Il contenuto semantico sta in Gemma. Due canali diretti da testare:
      h_raw: hidden di Gemma PRE-adapter (dopo causal_conv) sulle posizioni audio
             (pattern Thinker->Talker, Qwen2.5-Omni arXiv:2503.20215)
      h_text: hidden di Gemma sul blocco TESTO della risposta agente
              ([bos]+48 token testo, left-pad) -> pooling -> ancora testuale
              (inner monologue, Moshi arXiv:2410.00037; text-instructed, Mini-Omni)
  - Se il probe lineare/MLP su h_raw >> 3.7% su h_sub, il collo e' l'adapter:
    il v4 usera' h_raw+h_text come condizionamento del predittore cb0.

Cache di output (cache_ars_full_v2.pt):
  tr_raw/va_raw: hidden Gemma pre-adapter, posizioni audio (B,T,1536)
  tr_sub/va_sub: h_sub post-adapter (identico a cache_ars_full.pt, per confronto)
  tr_htx/va_htx: h_text pooled sul blocco testo (B,1536) o (B,T,1536) se per-frame
  tr_c/va_c: cb0 target (invariato)
"""
import os, sys, json, time, argparse

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet")
ap.add_argument("--adapter-ckpt", default="/workspace/checkpoints/ht_adapter_temporal2_final.pt")
ap.add_argument("--output", default="/workspace/cache_ars_full_v2.pt")
ap.add_argument("--subsample", type=int, default=1,
                help="estrai 1 finestra ogni N (default 1 = tutte; 4 = ~6.4k train, ~65min in parallelo)")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

print("[CACHE2] 1/4 Carico sistema + adapter...", flush=True)
import kaggle_pipeline as KP
duplex_model, tokenizer = KP.load_gemma4_duplex_system()
KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(args.adapter_ckpt))
for p in duplex_model.parameters():
    p.requires_grad = False
duplex_model.eval()
device = next(duplex_model.base_model.parameters()).device
duplex_model.ht_adapter.to(device)
duplex_model.causal_conv.to(device)

print("[CACHE2] 2/4 Dataset + split...", flush=True)
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
                                agent_codes[:, :val_start], seq_len=64, text_by_window=tbw)
ds_val = KP.FullDuplexDataset(user_mel[val_start*MPF:], agent_mel[val_start*MPF:],
                              agent_codes[:, val_start:], seq_len=64, text_by_window=tbw)

print("[CACHE2] 3/4 Estrazione h_raw / h_sub / h_text...", flush=True)
fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
afs = 2 + KP.TEXT_BLOCK_LEN          # posizione della prima feature audio nel seq LLM
TEXT_BLOCK_LEN = KP.TEXT_BLOCK_LEN


@torch.inference_mode()
def extract(ds, limit, subsample=1):
    raw, sub, htx, cs = [], [], [], []
    for i in range(min(limit, len(ds))):
        if i % subsample != 0:
            continue
        item = ds[i]
        T = item["audio_targets"].shape[0]
        inp = {"input_ids": item["input_ids"].unsqueeze(0).to(device),
               "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
               "input_features": item["input_features"].unsqueeze(0).to(device),
               "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device)}
        go = duplex_model.base_model(**inp, output_hidden_states=True, return_dict=True)
        lh = go.hidden_states[-1]                              # (1, L, 1536)
        # blocco testo: posizioni [1, 1+TEXT_BLOCK_LEN) (dopo [bos], prima di [boa])
        tblock = lh[:, 1:1+TEXT_BLOCK_LEN, :]                  # (1, 48, 1536)
        # maschera: token non-pad del testo (input_ids != pad id 0?) -> mean pooling
        tid = inp["input_ids"][0, 1:1+TEXT_BLOCK_LEN]
        m = (tid != 0).unsqueeze(-1).float()                   # (48,1)
        denom = m.sum().clamp(min=1.0)
        htx.append((tblock.squeeze(0) * m).sum(0) / denom)     # (1536,)
        # posizioni audio: [afs, afs+fpf*T) step fpf
        ah = lh[:, afs: afs + fpf*T: fpf, :].float()           # (1, T, 1536)
        ad = duplex_model._apply_causal_conv(
            ah.to(next(duplex_model.causal_conv.parameters()).dtype)).float()
        raw.append(ad.squeeze(0).cpu())
        h_s = duplex_model.ht_adapter(
            ad.to(next(duplex_model.ht_adapter.parameters()).dtype)).float()
        sub.append(h_s.squeeze(0).cpu())
        cs.append(item["audio_targets"][:, 0].clone())
        if (i + 1) % 200 == 0:
            print(f"[CACHE2]   ... {i+1}/{min(limit, len(ds))}", flush=True)
    return raw, sub, htx, cs


n_tr = min(len(ds_train), 300) if args.smoke else len(ds_train)
n_va = min(len(ds_val), 40) if args.smoke else min(len(ds_val), 100)
tr_raw, tr_sub, tr_htx, tr_c = extract(ds_train, n_tr, args.subsample)
va_raw, va_sub, va_htx, va_c = extract(ds_val, n_va, args.subsample)

print("[CACHE2] 4/4 Salvataggio...", flush=True)
torch.save({"tr_raw": tr_raw, "tr_sub": tr_sub, "tr_htx": tr_htx, "tr_c": tr_c,
            "va_raw": va_raw, "va_sub": va_sub, "va_htx": va_htx, "va_c": va_c},
           args.output)
print(f"[CACHE2] Salvata {args.output}: {len(tr_raw)} train / {len(va_raw)} val "
      f"(h_raw {tr_raw[0].shape}, h_sub {tr_sub[0].shape}, h_text {tr_htx[0].shape})", flush=True)
print("[CACHE2] FATTO.", flush=True)
