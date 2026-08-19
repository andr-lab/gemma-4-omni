# -*- coding: utf-8 -*-
"""
TEST DECISIVO: il nostro loop sample_acoustic_codes (forward completa con junk
nelle posizioni future) e' EQUIVALENTE alla generazione ufficiale passo-passo
del teacher Qwen3-TTS (code_predictor.generate con KV cache)?

Ipotesi di bug dell'utente: gli oracle (h_t GT) suonano robotici/incomprensibili
mentre nei run passati erano coerenti. Possibile causa: il nostro wrapper mette
junk (0) nelle posizioni future del tensore `full` a ogni passo g, mentre
l'ufficiale genera un token per volta con KV cache e non ha mai posizioni future.
Se la causal mask non isola perfettamente, i logits cambiano -> residui sbagliati.

Test: con gli stessi h_t e cb0, confrontiamo per ogni livello g i logits di:
  A) nostro sample loop (wrapper): forward completa con codici gia' campionati
     nelle posizioni 1..g-1 e 0 nelle posizioni future
  B) forward causale del teacher con SOLO le posizioni fino a g (niente junk):
     inputs = [h_t, cb0, cb1..cb_{g-1}]  (g+1 posizioni reali, niente future)

Se i logits di B per il livello g (lm_head[g-1](hidden[:, g])) == logits di A
per il livello g -> il junk e' innocuo, nessun bug. Se differiscono -> BUG.

Nota: questo e' verificabile SENZA il teacher completo: usiamo il nostro wrapper
con i pesi reali del sub-talker (347MB, gia' in locale). La forward a g+1
posizioni con la mask causale del wrapper e' il ground truth "senza junk".
"""
import os
import sys
import json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_subtalker import load_qwen_subtalker

CKPT_DIR = "eval_r2_out_final/qwen_subtalker"

def load_subtalker(device="cpu"):
    st, _ = load_qwen_subtalker(CKPT_DIR, device=device, dtype=torch.float32)
    return st

def reference_step_forward(st, h_t, codes_g):
    """Forward causale del teacher con SOLO g+1 posizioni reali (niente junk).
    codes_g: (B, T, g) con cb0..cb_{g-1} gia' noti.
    Ritorna i logits per il livello g: lm_head[g-1](hidden[:, g])."""
    B, T, g = codes_g.shape
    h_flat = h_t.reshape(B * T, 1, st.cfg["talker_hidden_size"])
    # costruisco inputs come _forward_flat ma solo fino a g posizioni
    inputs = [h_flat]
    inputs.append(st.cb0_embedding(codes_g[:, :, :1].reshape(B * T, 1).clamp(min=0)))
    for k in range(g - 1):  # posizioni 2..g: cb1..cb_{g-1} con embedding gruppo k
        inputs.append(st.group_embeddings[k](codes_g[:, :, k + 1 : k + 2].reshape(B * T, 1).clamp(min=0)))
    inputs_embeds = torch.cat(inputs, dim=1)  # (B*T, g+1, talker_hidden)
    S = inputs_embeds.shape[1]
    hidden_states = st.small_to_mtp_projection(inputs_embeds)
    position_ids = torch.arange(0, S, device=h_t.device).unsqueeze(0)
    position_embeddings = st.rotary_emb(hidden_states, position_ids)
    causal_mask = st._causal_mask[:, :, :S, :S].to(h_t.device).to(h_t.dtype)
    for layer in st.layers:
        hidden_states = layer(hidden_states, position_embeddings, causal_mask)
    hidden_states = st.norm(hidden_states)
    # livello g (1-based): logits = lm_head[g-1](hidden[:, g])  (posizione g, 0-based = g)
    logits_g = st.lm_head[g - 1](hidden_states[:, g])  # (B*T, V)
    return logits_g

def wrapper_logits_at_g(st, h_t, cb0, codes_sampled, g):
    """Logits del livello g come li calcola il nostro sample loop:
    full con cb0 in col 0, codici campionati in 1..g-1, 0 (junk) nelle future."""
    B, T = cb0.shape
    h_flat = h_t.reshape(B * T, 1, st.cfg["talker_hidden_size"])
    full = torch.zeros(B, T, st.cfg["num_code_groups"], dtype=torch.long, device=h_t.device)
    full[:, :, 0] = cb0
    if g > 1:
        # replica esatta del wrapper: seq = [cb0, cb1..cb_{g-1}] -> full[:, :, 1:g] = seq[:, :, 1:]
        full[:, :, 1:g] = codes_sampled[:, :, 1:g]
    logits = st._forward_flat(h_flat, full.reshape(B * T, st.cfg["num_code_groups"]))
    return logits[:, g - 1, :]  # (B*T, V)

def main():
    torch.manual_seed(42)
    dev = "cpu"
    st = load_subtalker(dev)
    cfg = st.cfg
    V = cfg["vocab_size"]
    T = 3
    B = 1
    h_t = torch.randn(B, T, cfg["talker_hidden_size"], dtype=torch.float32) * 0.5
    cb0 = torch.randint(0, V - 1, (B, T), dtype=torch.long)

    max_delta = 0.0
    print(f"sub-talker caricato: {sum(p.numel() for p in st.parameters())/1e6:.1f}M params")
    print(f"confronto logits per livello g: wrapper (junk future) vs reference (solo g+1 posizioni)")
    print("-" * 70)
    ok = True
    for g in range(1, cfg["num_code_groups"]):
        # codici campionati fino a g-1 (simulati: casuali nel range)
        codes_sampled = torch.randint(0, V - 1, (B, T, g), dtype=torch.long)
        codes_sampled[:, :, 0] = cb0
        # riferimento: forward con solo g+1 posizioni
        ref = reference_step_forward(st, h_t, codes_sampled)
        # wrapper: full con junk
        wr = wrapper_logits_at_g(st, h_t, cb0, codes_sampled, g)
        d = (ref - wr).abs().max().item()
        max_delta = max(max_delta, d)
        status = "OK" if d < 1e-4 else "*** DIFFERISCE ***"
        if d >= 1e-4:
            ok = False
        print(f"  g={g:2d}: max|ref - wrapper| = {d:.6f}  {status}")

    print("-" * 70)
    if ok:
        print(f"VERDETTO: il junk nelle posizioni future e' INNOCUO (max delta {max_delta:.2e}).")
        print("Il loop del wrapper produce gli stessi logits della forward causale senza junk.")
        print("=> Nessun bug di causalita' nel wrapper.")
    else:
        print(f"VERDETTO: BUG TROVATO! Il junk nelle posizioni future cambia i logits (max delta {max_delta:.4f}).")
        print("=> Il loop del wrapper NON e' equivalente alla generazione passo-passo del teacher.")

if __name__ == "__main__":
    main()
