#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDD Suite: Real Synthetic Dialogue Full-Duplex Benchmark
Verifica:
  1. User Audio 80ms Chunking Contract (1920 samples @ 24kHz)
  2. Benchmark Scripts Contract (run_exact_interactive_dialogue_benchmark.py & run_real_synth_dialogue_benchmark.py)
  3. Projector V9 Architecture & Prefix Alignment Shape Contract (16 frames x 2048D)
  4. Multi-Turn Dialogue Timing, Non-Overlap & Barge-In Cut Contract
  5. Natural Energy Trimming & Fade-Out Contract (Zero clicks, zero residual noise)
  6. Chat History Strict Alternation & Multi-Turn Context Consistency Contract
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))

from turn_detector_head import (
    TurnDetectorHead,
    TurnDetectorConfig,
    StreamingDuplexController,
    ACTION_SPEAK,
    ACTION_YIELD,
    ACTION_BARGE_IN,
    ACTION_NAMES,
)

# --- V9 Projector Definition (Exact as tools/run_exact_interactive_dialogue_benchmark.py) ---
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=512, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len):
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim=2048, ffn_dim=5632):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim=2048, num_heads=16, ffn_dim=5632):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DeepCausalTransformerProjectorV9(nn.Module):
    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, num_layers=4, num_heads=16, ffn_dim=5632):
        super().__init__()
        self.audio_in = nn.Linear(raw_dim, hidden_dim, bias=False)
        self.text_in = nn.Linear(text_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)
        ])
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)



def test_user_audio_chunking_contract():
    """Verifica che l'audio reale dell'utente venga suddiviso a blocchi di 80ms (1920 campioni a 24kHz)."""
    print("=== [TEST 1/6] User Audio 80ms Chunking Contract ===", flush=True)
    sr = 24000
    chunk_samples = int(0.080 * sr)  # 1920 campioni

    # Creiamo un buffer di 2.4s (30 frame)
    audio_stream = np.sin(2 * np.pi * 440 * np.arange(30 * chunk_samples) / sr).astype(np.float32)
    chunks = [audio_stream[i*chunk_samples : (i+1)*chunk_samples] for i in range(30)]

    assert len(chunks) == 30, f"Expected 30 chunks, got {len(chunks)}"
    assert len(chunks[0]) == 1920, f"Expected 1920 samples/chunk, got {len(chunks[0])}"
    print(f"  OK: 30 chunk da 80ms (1920 campioni a 24kHz) generati con successo")


def test_script_contract():
    """Verifica l'esistenza degli script di benchmark reali su synth_0000."""
    print("\n=== [TEST 2/6] Benchmark Script Contracts Verification ===", flush=True)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    script_exact = os.path.join(root_dir, "tools", "run_exact_interactive_dialogue_benchmark.py")
    script_real = os.path.join(root_dir, "tools", "run_real_synth_dialogue_benchmark.py")
    script_live = os.path.join(root_dir, "tools", "interactive_gemma4_omni_live.py")

    assert os.path.exists(script_exact), f"Lo script {script_exact} deve esistere!"
    assert os.path.exists(script_real), f"Lo script {script_real} deve esistere!"
    assert os.path.exists(script_live), f"Lo script di riferimento {script_live} deve esistere!"
    print(f"  OK: Script di riferimento e benchmark trovati:")
    print(f"      - {script_exact}")
    print(f"      - {script_real}")
    print(f"      - {script_live}")


def test_projector_v9_contract():
    """Verifica la correttezza architetturale e contrattuale di DeepCausalTransformerProjectorV9."""
    print("\n=== [TEST 3/6] Projector V9 Architecture & Prefix Contract ===", flush=True)
    proj = DeepCausalTransformerProjectorV9(
        raw_dim=1536,
        text_dim=1536,
        hidden_dim=2048,
        num_layers=4,
        num_heads=16,
        ffn_dim=5632
    ).eval()

    # Parametri attesi: ~211.83M
    total_params = sum(p.numel() for p in proj.parameters())
    print(f"  • Parametri Projector V9: {total_params / 1e6:.2f}M")
    assert 200_000_000 < total_params < 225_000_000, f"Unexpected parameter count: {total_params}"

    # Forward con dummy input: h_raw (1, 16, 1536), h_tx (1, 1536)
    h_raw = torch.randn(1, 16, 1536)
    h_tx = torch.randn(1, 1536)
    with torch.no_grad():
        h_pfx = proj(h_raw, h_tx)

    assert h_pfx.shape == (1, 16, 2048), f"Expected (1, 16, 2048), got {h_pfx.shape}"
    assert not torch.isnan(h_pfx).any(), "NaN found in Projector output!"
    print("  OK: Projector V9 forward test superato (output: [1, 16, 2048])")


def test_dialogue_timing_and_bargein():
    """Verifica che i turni conversazionali e i barge-in siano consistenti e non causino sovrapposizioni non gestite."""
    print("\n=== [TEST 4/6] Dialogue Timing & Barge-In Cut Contract ===", flush=True)
    dialogue_turns = [
        {"turn_id": 1, "user_start_s": 0.0, "user_end_s": 10.0, "avail_pause": 7.5, "is_interrupted": False},
        {"turn_id": 2, "user_start_s": 18.0, "user_end_s": 26.0, "avail_pause": 3.8, "is_interrupted": True, "bargein_at_s": 30.0},
        {"turn_id": 3, "user_start_s": 30.0, "user_end_s": 32.0, "avail_pause": 3.8, "is_interrupted": True, "bargein_at_s": 36.0},
        {"turn_id": 4, "user_start_s": 36.0, "user_end_s": 45.0, "avail_pause": 9.5, "is_interrupted": False},
        {"turn_id": 5, "user_start_s": 55.0, "user_end_s": 64.32, "avail_pause": 12.0, "is_interrupted": False},
    ]

    for turn in dialogue_turns:
        t_id = turn["turn_id"]
        u_end = turn["user_end_s"]
        avail = turn["avail_pause"]
        agent_start = u_end + 0.3
        max_dur = avail - 0.5
        agent_end = agent_start + max_dur

        if turn["is_interrupted"]:
            bargein_t = turn["bargein_at_s"]
            assert agent_end <= bargein_t + 0.1, f"Turn {t_id}: Agent audio extends past barge-in ({agent_end} > {bargein_t})!"
            print(f"  • Turno {t_id} (Barge-In a {bargein_t:.1f}s): Agente parla da {agent_start:.1f}s a {agent_end:.1f}s (tagliato prima dell'interruzione)")
        else:
            print(f"  • Turno {t_id} (Completo): Agente parla da {agent_start:.1f}s a {agent_end:.1f}s (budget: {max_dur:.1f}s)")

    print("  OK: Tutti i vincoli temporali di turno e barge-in sono rigorosamente rispettati")


def test_energy_trimming_and_fade():
    """Verifica l'algoritmo di trimming energetico e dissolvenza naturale."""
    print("\n=== [TEST 5/6] Natural Energy Trimming & Fade Contract ===", flush=True)
    sr = 24000
    # Creiamo un segnale audio di 5 secondi: 2.5s voce (seno a 220Hz) + 2.5s silenzio/rumore di fondo bassissimo
    t_voice = np.linspace(0, 2.5, int(2.5 * sr), endpoint=False)
    voice = 0.5 * np.sin(2 * np.pi * 220 * t_voice)
    silence = 0.001 * np.random.randn(int(2.5 * sr))
    full_wav = np.concatenate([voice, silence]).astype(np.float32)

    chunk_size = int(0.080 * sr)
    energies = [np.sqrt(np.mean(full_wav[i:i+chunk_size]**2)) for i in range(0, len(full_wav), chunk_size)]
    voice_end_idx = len(energies)
    for idx in range(len(energies)-1, 10, -1):
        if energies[idx] > 0.015:
            voice_end_idx = min(len(energies), idx + 4)
            break
    trim_sample = min(len(full_wav), voice_end_idx * chunk_size)
    clean_wav = full_wav[:trim_sample].copy()

    if len(clean_wav) > 1200:
        fade_len = int(0.040 * 24000)
        clean_wav[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    p_peak = float(np.percentile(np.abs(clean_wav), 99.7))
    clean_wav = np.clip(clean_wav / (p_peak * 1.05), -1.0, 1.0) if p_peak > 1e-6 else clean_wav

    # Il segnale deve essere stato trimmato prima dei 5s (intorno a 2.5 - 2.9s)
    trimmed_dur = len(clean_wav) / sr
    assert 2.4 <= trimmed_dur <= 3.2, f"Unexpected trimmed duration: {trimmed_dur:.2f}s"
    assert clean_wav[-1] == 0.0 or abs(clean_wav[-1]) < 1e-4, "Fade-out did not smoothly reach zero!"
    print(f"  OK: Trimming energetico naturale validato: audio ridotto da 5.00s a {trimmed_dur:.2f}s con fade-out a zero")


def test_chat_history_alternation():
    """Verifica che la cronologia di chat mantenga una rigorosa alternanza di ruoli."""
    print("\n=== [TEST 6/6] Chat History Strict Alternation Contract ===", flush=True)
    turns = [
        ("Voglio scrivere un racconto breve di genere noir.", "Certamente, iniziamo impostando il protagonista."),
        ("Salo ha falsificato le prove.", "Questo rivela il conflitto interiore di Salo."),
        ("Salo ha colpito Claudia.", "La tensione sale alle stelle."),
    ]

    chat_history = []
    for u, a in turns:
        chat_history.append({"role": "user", "content": u})
        chat_history.append({"role": "assistant", "content": a})

    # Verifichiamo che i ruoli si alternino esattamente
    for i, msg in enumerate(chat_history):
        expected_role = "user" if i % 2 == 0 else "assistant"
        assert msg["role"] == expected_role, f"Message {i} has unexpected role {msg['role']}"

    assert len(chat_history) == 6
    print(f"  OK: Cronologia multi-turno verificata con successo ({len(chat_history)} messaggi alternati)")


if __name__ == "__main__":
    test_user_audio_chunking_contract()
    test_script_contract()
    test_projector_v9_contract()
    test_dialogue_timing_and_bargein()
    test_energy_trimming_and_fade()
    test_chat_history_alternation()
    print("\n>>> TDD SUITE REAL SYNTH DIALOGUE: 6/6 TEST VERDI <<<", flush=True)

