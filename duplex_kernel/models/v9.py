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
import threading
from dataclasses import dataclass
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
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self.base = float(base)
        self._cache(max_seq_len)

    def _cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()

    def _ensure_capacity(self, seq_len: int) -> None:
        if seq_len > self.cos_cached.shape[0]:
            self._cache(max(seq_len, self.cos_cached.shape[0] * 2))

    def forward(self, x: torch.Tensor, seq_len: int):
        self._ensure_capacity(seq_len)
        return self.cos_cached[:seq_len].to(device=x.device, dtype=x.dtype), self.sin_cached[:seq_len].to(device=x.device, dtype=x.dtype)

    def at(self, x: torch.Tensor, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if position < 0:
            raise ValueError("RoPE position non può essere negativa")
        self._ensure_capacity(position + 1)
        return (
            self.cos_cached[position:position + 1].to(device=x.device, dtype=x.dtype),
            self.sin_cached[position:position + 1].to(device=x.device, dtype=x.dtype),
        )


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


@dataclass
class IncrementalV9State:
    """State preallocato per una sequenza V9 causale frame-by-frame."""

    batch_size: int
    max_seq_len: int
    device: torch.device
    dtype: torch.dtype
    conv_history: torch.Tensor
    key_cache: list[torch.Tensor]
    value_cache: list[torch.Tensor]
    rope_position: int = 0
    frames_processed: int = 0
    text_anchor: torch.Tensor | None = None

    @torch.inference_mode()
    def reset(self) -> None:
        """Riporta la posizione a zero, mantenendo le allocazioni KV."""
        self.conv_history.zero_()
        self.rope_position = 0
        self.frames_processed = 0
        self.text_anchor = None


class IncrementalV9Projector:
    """Projector V9 stateful equivalente al forward batch causale.

    Ogni ``step`` consuma esattamente una raw frame, aggiorna la finestra della
    CausalConv e appende una sola K/V position per ciascun layer. Le KV sono
    preallocate per evitare ``cat`` O(T^2) nel percorso realtime.
    """

    def __init__(
        self,
        projector: DeepCausalTransformerProjectorV9,
        causal_conv: CausalConv1D,
        *,
        max_seq_len: int = 512,
    ) -> None:
        if max_seq_len <= 0:
            raise ValueError("max_seq_len deve essere positivo")
        raw_dim = int(projector.audio_in.in_features)
        if int(causal_conv.conv.in_channels) != raw_dim or int(causal_conv.conv.out_channels) != raw_dim:
            raise ValueError("CausalConv e projector devono condividere raw_dim")
        if causal_conv.conv.kernel_size != (3,) or causal_conv.conv.stride != (1,):
            raise ValueError("IncrementalV9 richiede CausalConv kernel=3/stride=1")
        self.projector = projector.eval()
        self.causal_conv = causal_conv.eval()
        self.max_seq_len = int(max_seq_len)
        self.raw_dim = raw_dim
        self.hidden_dim = int(projector.audio_in.out_features)
        self.num_layers = len(projector.layers)
        self.num_heads = int(projector.layers[0].attn.num_heads)
        if not self.num_layers:
            raise ValueError("projector V9 deve avere almeno un layer")
        if any(int(layer.attn.num_heads) != self.num_heads for layer in projector.layers):
            raise ValueError("tutti i layer V9 devono avere lo stesso numero di head")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim deve essere divisibile per num_heads")
        self.head_dim = self.hidden_dim // self.num_heads
        self._parameter = next(self.projector.parameters())
        self._lock = threading.Lock()
        if self._parameter.device != next(self.causal_conv.parameters()).device:
            raise ValueError("projector e CausalConv devono stare sullo stesso device")
        if self._parameter.dtype != next(self.causal_conv.parameters()).dtype:
            raise ValueError("projector e CausalConv devono usare lo stesso dtype")

    def new_state(
        self,
        *,
        batch_size: int = 1,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        max_seq_len: int | None = None,
    ) -> IncrementalV9State:
        if batch_size <= 0:
            raise ValueError("batch_size deve essere positivo")
        state_device = self._parameter.device if device is None else torch.device(device)
        state_dtype = self._parameter.dtype if dtype is None else dtype
        if state_device != self._parameter.device:
            raise ValueError("state e projector devono stare sullo stesso device")
        if state_dtype != self._parameter.dtype:
            raise ValueError("state e projector devono usare lo stesso dtype")
        length = self.max_seq_len if max_seq_len is None else int(max_seq_len)
        if length <= 0 or length > self.max_seq_len:
            raise ValueError(f"max_seq_len deve essere tra 1 e {self.max_seq_len}")
        shape = (int(batch_size), 2, self.raw_dim)
        conv_history = torch.zeros(shape, device=state_device, dtype=state_dtype)
        key_cache = [
            torch.zeros(
                (int(batch_size), self.num_heads, length, self.head_dim),
                device=state_device,
                dtype=state_dtype,
            )
            for _ in range(self.num_layers)
        ]
        value_cache = [cache.clone() for cache in key_cache]
        return IncrementalV9State(
            batch_size=int(batch_size),
            max_seq_len=length,
            device=state_device,
            dtype=state_dtype,
            conv_history=conv_history,
            key_cache=key_cache,
            value_cache=value_cache,
        )

    def _step_tensor(self, value: torch.Tensor, *, name: str, batch_size: int) -> torch.Tensor:
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.dim() != 2 or value.shape != (batch_size, self.raw_dim):
            raise ValueError(f"{name} deve avere shape ({batch_size}, {self.raw_dim}), got {tuple(value.shape)}")
        return value.to(device=self._parameter.device, dtype=self._parameter.dtype)

    @torch.inference_mode()
    def step(
        self,
        h_raw_t: torch.Tensor,
        h_tx_anchor: torch.Tensor,
        state: IncrementalV9State,
    ) -> torch.Tensor:
        """Serializza l'accesso al modello condiviso tra sessioni scheduler."""
        with self._lock:
            return self._step_unlocked(h_raw_t, h_tx_anchor, state)

    @torch.inference_mode()
    def _step_unlocked(
        self,
        h_raw_t: torch.Tensor,
        h_tx_anchor: torch.Tensor,
        state: IncrementalV9State,
    ) -> torch.Tensor:
        """Consuma una raw frame e restituisce ``h_pfx_t`` di shape ``(B, 2048)``."""
        if not isinstance(state, IncrementalV9State):
            raise TypeError("state deve essere IncrementalV9State")
        if state.rope_position >= state.max_seq_len:
            raise OverflowError("IncrementalV9 max_seq_len raggiunto")
        if state.batch_size <= 0 or state.conv_history.shape != (state.batch_size, 2, self.raw_dim):
            raise ValueError("state conv_history non compatibile con IncrementalV9")
        raw = self._step_tensor(h_raw_t, name="h_raw_t", batch_size=state.batch_size)
        anchor = self._step_tensor(h_tx_anchor, name="h_tx_anchor", batch_size=state.batch_size)
        if state.text_anchor is not None and not torch.equal(state.text_anchor, anchor):
            raise ValueError("h_tx_anchor deve restare fisso lungo la sequenza")
        position = int(state.rope_position)

        # La finestra [t-2, t-1, t] è la stessa del F.pad(left=2) batch.
        window = torch.cat((state.conv_history, raw.unsqueeze(1)), dim=1)
        conv_weight = self.causal_conv.conv.weight
        conv_t = self.causal_conv.conv(window.transpose(1, 2).to(conv_weight.dtype))
        x = self.projector.audio_in(conv_t[:, :, 0]) + self.projector.text_in(anchor)

        written: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        try:
            for index, layer in enumerate(self.projector.layers):
                normalized = layer.norm1(x)
                attention = layer.attn
                q = attention.q_proj(normalized).view(state.batch_size, self.num_heads, self.head_dim).unsqueeze(2)
                k = attention.k_proj(normalized).view(state.batch_size, self.num_heads, self.head_dim).unsqueeze(2)
                v = attention.v_proj(normalized).view(state.batch_size, self.num_heads, self.head_dim).unsqueeze(2)
                cos, sin = attention.rotary.at(x, position)
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

                key_slot = state.key_cache[index][:, :, position:position + 1, :]
                value_slot = state.value_cache[index][:, :, position:position + 1, :]
                written.append((key_slot, value_slot, torch.cat((key_slot, value_slot), dim=-1).clone()))
                key_slot.copy_(k)
                value_slot.copy_(v)
                keys = state.key_cache[index][:, :, :position + 1, :]
                values = state.value_cache[index][:, :, :position + 1, :]
                attended = F.scaled_dot_product_attention(q, keys, values, is_causal=False)
                attended = attended.transpose(1, 2).contiguous().view(state.batch_size, self.hidden_dim)
                x = x + attention.out_proj(attended)
                x = x + layer.ffn(layer.norm2(x))
            output = self.projector.final_norm(x)
            if not torch.isfinite(output).all():
                raise RuntimeError("IncrementalV9 ha prodotto output non finito")
        except BaseException:
            for key_slot, value_slot, old in written:
                split = self.head_dim
                key_slot.copy_(old[..., :split])
                value_slot.copy_(old[..., split:])
            raise

        state.conv_history = window[:, 1:, :].detach()
        if state.text_anchor is None:
            state.text_anchor = anchor.detach().clone()
        state.rope_position += 1
        state.frames_processed += 1
        return output.detach()


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
