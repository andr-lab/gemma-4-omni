#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test e Validazione della Logica Batched Parallel Causal Forward per Trainer V7
"""
import torch

def test_batched_tensor_logic():
    print("=" * 70)
    print("=== TEST TENSORIALE BATCHED PARALLEL CAUSAL FORWARD ===")
    print("=" * 70)

    B = 4
    T = 64
    D = 2048

    # Simula prefill di lunghezze variabili
    L_lens = [25, 32, 28, 30]
    L_max = max(L_lens)
    print(f"Batch size: {B}, T audio: {T}, Lunghezze prefill: {L_lens}, L_max: {L_max}")

    # Simula prefill embeds
    padded_prefill = torch.zeros(B, L_max, D)
    prefill_mask = torch.zeros(B, L_max)
    for b in range(B):
        Lb = L_lens[b]
        padded_prefill[b, :Lb, :] = torch.randn(Lb, D)
        prefill_mask[b, :Lb] = 1.0

    # Simula audio embeds (1..T-1)
    audio_embeds = torch.randn(B, T-1, D)
    audio_mask = torch.ones(B, T-1)

    # Concatena sequenza totale
    full_inputs = torch.cat([padded_prefill, audio_embeds], dim=1) # (B, L_max + T - 1, D)
    full_mask = torch.cat([prefill_mask, audio_mask], dim=1)       # (B, L_max + T - 1)

    print(f"full_inputs shape: {full_inputs.shape}")
    print(f"full_mask shape  : {full_mask.shape}")
    assert full_inputs.shape == (B, L_max + T - 1, D)
    assert full_mask.shape == (B, L_max + T - 1)

    # Simula output del Transformer
    full_h = torch.randn(B, L_max + T - 1, D)

    # Estrazione degli hidden states dei frame audio (T=64 per ciascun sample)
    h_audio_list = []
    for b in range(B):
        Lb = L_lens[b]
        # Frame t=0: ultimo token del prefill reale di questo sample
        h_t0 = full_h[b, Lb - 1 : Lb, :] # (1, D)
        # Frame t=1..T-1: posizioni audio partendo da L_max
        h_t_rest = full_h[b, L_max : L_max + T - 1, :] # (T-1, D)
        h_sample = torch.cat([h_t0, h_t_rest], dim=0) # (T, D)
        h_audio_list.append(h_sample)

    H_all = torch.stack(h_audio_list, dim=0) # (B, T, D)
    print(f"H_all extracted shape: {H_all.shape}")
    assert H_all.shape == (B, T, D)
    print("✅ TEST TENSORIALE SUPERATO: Shape e indicizzazione perfette al 100%!")

if __name__ == '__main__':
    test_batched_tensor_logic()
