# -*- coding: utf-8 -*-
"""
duplex_kernel/models/v9.py — DeepCausalTransformerProjectorV9 (ADR-024).

Architettura V9 (211.83M) + CausalConv1D addestrata (Task 11i), ricetta
ground-truth da tools/interactive_gemma4_omni_live.py:

    lh = hidden_states[-1] della sequenza [prompt + generati]   (1, L, 1536)
    h_tx        = mean(lh[:, prompt_len:])                      (1, 1536)
    h_raw_style = interpolate(generati, 16, linear) → causal_conv
    h_pfx       = proj(h_raw_style, h_tx)                       (1, 16, 2048)

Checkpoint: /workspace/checkpoints/v9_prefix_final.pt (key 'proj') +
            custom_duplex_heads.pt (key 'causal_conv').
Modulo autocontenuto del kernel (ADR-029): nessuna dipendenza dal plugin.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_V9_CKPT = "/workspace/checkpoints/v9_prefix_final.pt"
DEFAULT_HEADS_CKPT = "/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt"


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int):
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(0).unsqueeze(1)
    sin = sin.unsqueeze(0).unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_heads: int = 16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary(x, T)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out_proj(out.transpose(1, 2).contiguous().view(B, T, C))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_dim: int = 2048, ffn_dim: int = 5632):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 2048, num_heads: int = 16, ffn_dim: int = 5632):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attn = CausalSelfAttention(hidden_dim, num_heads)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.ffn = SwiGLUFFN(hidden_dim, ffn_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class DeepCausalTransformerProjectorV9(nn.Module):
    """Proiettore causale profondo V9 — 211.83M parametri (ADR-024)."""

    def __init__(self, raw_dim: int = 1536, text_dim: int = 1536, hidden_dim: int = 2048,
                 num_layers: int = 4, num_heads: int = 16, ffn_dim: int = 5632):
        super().__init__()
        self.audio_in = nn.Linear(raw_dim, hidden_dim, bias=False)
        self.text_in = nn.Linear(text_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def forward(self, h_raw: torch.Tensor, h_tx: torch.Tensor) -> torch.Tensor:
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class CausalConv1D(nn.Module):
    """Convoluzione causale 1D con left-padding esplicito (F.pad(x, (2, 0)))."""

    def __init__(self, hidden_size: int = 1536, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_conv = hidden_states.transpose(1, 2)
        hidden_padded = F.pad(hidden_conv, (2, 0))
        return self.conv(hidden_padded.to(self.conv.weight.dtype)).transpose(1, 2)


def _resolve_ckpt_path(primary: str, fallbacks: tuple[str, ...]) -> str:
    if primary and os.path.isfile(primary):
        return primary
    for p in fallbacks:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Checkpoint V9 non trovato: {primary}")


def load_v9_conditioning(
    v9_ckpt: Optional[str] = None,
    heads_ckpt: Optional[str] = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[DeepCausalTransformerProjectorV9, CausalConv1D]:
    """Costruisce projector V9 + causal conv e carica i pesi addestrati."""
    v9_path = v9_ckpt or os.environ.get("GEMMA4_OMNI_V9_CKPT") or _resolve_ckpt_path(DEFAULT_V9_CKPT, ())
    heads_path = heads_ckpt or os.environ.get("GEMMA4_OMNI_HEADS_CKPT") or DEFAULT_HEADS_CKPT

    proj = DeepCausalTransformerProjectorV9()
    sd = torch.load(v9_path, map_location="cpu", weights_only=False)
    proj.load_state_dict(sd.get("proj", sd), strict=False)

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
    heads = torch.load(heads_path, map_location="cpu", weights_only=True)
    cc = heads.get("causal_conv")
    if cc is None:
        raise RuntimeError(f"Chiave 'causal_conv' assente in {heads_path}")
    causal_conv.conv.load_state_dict(cc, strict=False)

    proj = proj.to(device=device, dtype=dtype).eval()
    causal_conv = causal_conv.to(device=device, dtype=dtype).eval()
    for p in proj.parameters():
        p.requires_grad = False
    for p in causal_conv.parameters():
        p.requires_grad = False
    return proj, causal_conv


def compute_v9_prefix(
    v9_thinker_hidden: torch.Tensor,
    proj: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    device: torch.device,
    dtype: torch.dtype,
    n_prefix_frames: int = 16,
) -> torch.Tensor:
    """Calcola il prefisso V9 [1, 16, 2048] dagli hidden dei token generati."""
    h = v9_thinker_hidden.to(device=device, dtype=dtype)
    if h.dim() != 2 or h.shape[0] == 0:
        raise ValueError(f"v9_thinker_hidden deve essere (N, 1536) non vuoto, got {tuple(h.shape)}")
    proj = proj.to(device=device, dtype=dtype)
    causal_conv = causal_conv.to(device=device, dtype=dtype)
    h_tx = h.mean(dim=0, keepdim=True)  # (1, 1536)
    seq_trans = h.unsqueeze(0).transpose(1, 2)  # (1, 1536, N)
    h_raw_style = F.interpolate(seq_trans, size=n_prefix_frames, mode="linear", align_corners=False)
    h_raw_style = h_raw_style.transpose(1, 2).contiguous()  # (1, 16, 1536)
    with torch.no_grad():
        h_raw_style = causal_conv(h_raw_style.to(dtype=causal_conv.conv.weight.dtype)).to(dtype)
        h_pfx = proj(h_raw_style, h_tx)  # (1, 16, 2048)
    return h_pfx.contiguous()


__all__ = ["CausalConv1D", "DeepCausalTransformerProjectorV9", "compute_v9_prefix", "load_v9_conditioning"]
