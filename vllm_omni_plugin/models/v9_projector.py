# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/models/v9_projector.py — DeepCausalTransformerProjectorV9 (ADR-024)
+ Causal Conv1D addestrata (Task 11i SOTA heads).

Copia FEDELE dell'architettura validata in:
- tools/train_v9_prefix_kv_engine.py        (definizione architettura V9, 211.83M)
- tools/interactive_gemma4_omni_live.py     (ricetta di generazione più precisa)
- kaggle_pipeline.py                        (causal_conv del wrapper full-duplex)

Percorso feature (ground truth, interactive_gemma4_omni_live.py):
    lh = hidden_states[-1] della sequenza [prompt + generati]         (1, L, 1536)
    h_tx        = mean(lh[:, prompt_len:])                            (1, 1536)
    h_raw_style = interpolate(generati, size=16, linear)              (1, 16, 1536)
                  -> CAUSAL CONV 1D addestrata (k=3, left-pad 2)
    h_pfx       = proj(h_raw_style, h_tx)                             (1, 16, 2048)

Checkpoint sorgenti (sulla 3090):
- /workspace/checkpoints/v9_prefix_final.pt            -> key 'proj' (step 10000)
- /workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt -> key 'causal_conv'
  (md5 ae556e20..., identico a task11i_cv26_v2recipe/custom_duplex_heads.pt)
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Percorsi di default (overridabili via env per riproducibilità).
DEFAULT_V9_CKPT = "/workspace/checkpoints/v9_prefix_final.pt"
DEFAULT_V9_CKPT_FALLBACKS = (
    "/workspace/checkpoints/v9_prefix_step10000.pt",
    "/workspace/checkpoints/v9_prefix_10k_final.pt",
)
DEFAULT_HEADS_CKPT = "/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt"


# ---------------------------------------------------------------------------
# Architettura V9 (identica a train_v9_prefix_kv_engine.py)
# ---------------------------------------------------------------------------
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
    """Proiettore causale profondo V9 — 211.83M parametri (ADR-024)."""

    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, num_layers=4, num_heads=16, ffn_dim=5632):
        super().__init__()
        self.audio_in = nn.Linear(raw_dim, hidden_dim, bias=False)
        self.text_in = nn.Linear(text_dim, hidden_dim, bias=False)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim, num_heads, ffn_dim) for _ in range(num_layers)])
        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        x = self.audio_in(h_raw) + self.text_in(h_tx)
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Causal Conv 1D addestrata (Task 11i SOTA heads) — ADR-002/global notes
# ---------------------------------------------------------------------------
class CausalConv1D(nn.Module):
    """Convoluzione causale 1D con left-padding esplicito (F.pad(x, (2, 0))).

    Identica a kaggle_pipeline.Gemma4FullDuplexModel.causal_conv/_apply_causal_conv.
    """

    def __init__(self, hidden_size: int = 1536, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_conv = hidden_states.transpose(1, 2)
        hidden_padded = F.pad(hidden_conv, (2, 0))
        return self.conv(hidden_padded.to(self.conv.weight.dtype)).transpose(1, 2)


# ---------------------------------------------------------------------------
# Loader pesi (projector V9 + causal conv addestrata)
# ---------------------------------------------------------------------------
def _resolve_ckpt_path(primary: str, fallbacks: tuple[str, ...]) -> str:
    if os.path.isfile(primary):
        return primary
    for p in fallbacks:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Checkpoint non trovato: {primary} (fallback: {fallbacks}). "
        "Impostare GEMMA4_OMNI_V9_CKPT / GEMMA4_OMNI_HEADS_CKPT."
    )


def load_v9_conditioning(
    v9_ckpt: Optional[str] = None,
    heads_ckpt: Optional[str] = None,
) -> tuple[DeepCausalTransformerProjectorV9, CausalConv1D]:
    """Costruisce projector V9 + causal conv e ne carica i pesi addestrati.

    Chiamata da load_weights() del talker (il modello è ancora su meta):
    load_state_dict con assign=True gestisce il placement su qualsiasi device.
    """
    v9_path = v9_ckpt or os.environ.get("GEMMA4_OMNI_V9_CKPT") or _resolve_ckpt_path(
        DEFAULT_V9_CKPT, DEFAULT_V9_CKPT_FALLBACKS
    )
    heads_path = heads_ckpt or os.environ.get("GEMMA4_OMNI_HEADS_CKPT") or _resolve_ckpt_path(
        DEFAULT_HEADS_CKPT, ()
    )

    proj = DeepCausalTransformerProjectorV9()
    sd = torch.load(v9_path, map_location="cpu", weights_only=False)
    proj.load_state_dict(sd.get("proj", sd), strict=False, assign=True)

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
    heads = torch.load(heads_path, map_location="cpu", weights_only=True)
    cc = heads.get("causal_conv")
    if cc is None:
        raise RuntimeError(f"Chiave 'causal_conv' assente in {heads_path}")
    causal_conv.conv.load_state_dict(cc, strict=False, assign=True)

    proj = proj.eval()
    causal_conv = causal_conv.eval()
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
    """Calcola il prefisso V9 [1, 16, 2048] dagli hidden states dei token GENERATI.

    Ricetta esatta di tools/interactive_gemma4_omni_live.py (righe 254-263):
        h_tx        = mean(lh[:, prompt_len:])                     # (1, 1536)
        h_raw_style = interpolate(seq_embeds, 16, linear) -> conv  # (1, 16, 1536)
        h_pfx       = proj(h_raw_style, h_tx)                      # (1, 16, 2048)

    Args:
        v9_thinker_hidden: hidden states last-layer dei token generati
            dal thinker, shape (N, 1536), CPU o GPU.
    """
    h = v9_thinker_hidden.to(device=device, dtype=dtype)
    if h.dim() != 2:
        raise ValueError(f"v9_thinker_hidden deve essere (N, 1536), got {tuple(h.shape)}")
    if h.shape[0] == 0:
        raise ValueError("v9_thinker_hidden vuoto: nessun token generato dal thinker.")

    # I buffer NON-persistenti (cos/sin RoPE) non viaggiano in state_dict e
    # possono restare su CPU dopo load_state_dict(assign=True): forza il
    # placement di moduli e buffer sul device di destinazione.
    proj = proj.to(device=device, dtype=dtype)
    causal_conv = causal_conv.to(device=device, dtype=dtype)

    # h_tx: inner monologue text embedding (mean sui token generati).
    h_tx = h.mean(dim=0, keepdim=True)  # (1, 1536)

    # h_raw_style: interpolazione lineare a 16 frame + causal conv addestrata.
    seq_trans = h.unsqueeze(0).transpose(1, 2)  # (1, 1536, N)
    h_raw_style = F.interpolate(seq_trans, size=n_prefix_frames, mode="linear", align_corners=False)
    h_raw_style = h_raw_style.transpose(1, 2).contiguous()  # (1, 16, 1536)
    with torch.no_grad():
        h_raw_style = causal_conv(h_raw_style.to(dtype=causal_conv.conv.weight.dtype)).to(dtype)
        h_pfx = proj(h_raw_style, h_tx)  # (1, 16, 2048)
    return h_pfx.contiguous()
