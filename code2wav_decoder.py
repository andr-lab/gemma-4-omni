# -*- coding: utf-8 -*-
"""
code2wav_decoder.py -- Standalone Pure-PyTorch Neural Audio Decoder for Qwen3-TTS 12Hz RVQ Tokens.
Loads code2wav_weights/code2wav_decoder.safetensors and decodes (B, 16, T) RVQ tokens to 24kHz PCM audio waveform.
Zero transformers/qwen-tts external dependency.
"""

import os
import json
import math
import numpy as np
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        alpha = self.alpha.view(1, -1, 1)
        beta = self.beta.view(1, -1, 1)
        alpha = torch.exp(alpha)
        beta = torch.exp(beta)
        return x + (1.0 / (beta + 1e-9)) * torch.sin(alpha * x) ** 2


class ResidualUnit(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.act1 = Snake1d(channels)
        self.conv1 = nn.Sequential()
        pad = 3 * dilation
        conv1 = nn.Conv1d(channels, channels, kernel_size=7, dilation=dilation, padding=pad)
        self.conv1.add_module("conv", conv1)
        self.act2 = Snake1d(channels)
        self.conv2 = nn.Sequential()
        conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.conv2.add_module("conv", conv2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        return res + x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int):
        super().__init__()
        pad = (kernel_size - stride) // 2
        block = nn.Sequential()
        block.add_module("0", Snake1d(in_channels))
        conv_t = nn.Sequential()
        conv_t.add_module("conv", nn.ConvTranspose1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad))
        block.add_module("1", conv_t)
        block.add_module("2", ResidualUnit(out_channels, dilation=1))
        block.add_module("3", ResidualUnit(out_channels, dilation=3))
        block.add_module("4", ResidualUnit(out_channels, dilation=9))
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvNeXtBlock1D(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int = 4096):
        super().__init__()
        self.dwconv = nn.Sequential()
        self.dwconv.add_module("conv", nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.transpose(1, 2)  # (B, C, T)
        return res + x


class UpsampleStage(nn.Module):
    def __init__(self, dim: int = 1024):
        super().__init__()
        conv_t = nn.Sequential()
        conv_t.add_module("conv", nn.ConvTranspose1d(dim, dim, kernel_size=2, stride=2))
        self.add_module("0", conv_t)
        self.add_module("1", ConvNeXtBlock1D(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._modules["0"](x)
        x = self._modules["1"](x)
        return x


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_scale: float = 0.01):
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class PreTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int = 512, intermediate_size: int = 1024, num_heads: int = 16, head_dim: int = 64):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.self_attn.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.self_attn.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.self_attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.self_attn_layer_scale = LayerScale(hidden_size)

        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = nn.Module()
        self.mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.mlp_layer_scale = LayerScale(hidden_size)

        self.num_heads = num_heads
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        norm_x = self.input_layernorm(x)
        B, T, C = norm_x.shape
        q = self.self_attn.q_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.self_attn.k_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.self_attn.v_proj(norm_x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        attn_out = self.self_attn.o_proj(attn_out)
        x = x + self.self_attn_layer_scale(attn_out)

        norm_x = self.post_attention_layernorm(x)
        gate = self.mlp.gate_proj(norm_x)
        up = self.mlp.up_proj(norm_x)
        mlp_out = self.mlp.down_proj(F.silu(gate) * up)
        x = x + self.mlp_layer_scale(mlp_out)
        return x


class PreTransformer(nn.Module):
    def __init__(self, hidden_size: int = 512, num_layers: int = 8, intermediate_size: int = 1024, max_pos: int = 8000, rope_theta: float = 10000.0):
        super().__init__()
        self.input_proj = nn.Linear(1024, hidden_size, bias=True)
        self.layers = nn.ModuleList([PreTransformerLayer(hidden_size, intermediate_size) for _ in range(num_layers)])
        self.norm = RMSNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, 1024, bias=True)
        self.head_dim = 64
        self.rope_theta = rope_theta

    def _get_rotary_emb(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        dim = self.head_dim
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype).unsqueeze(0).unsqueeze(0)
        sin = emb.sin().to(dtype).unsqueeze(0).unsqueeze(0)
        return cos, sin

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, 1024)
        x = self.input_proj(x)
        B, T, C = x.shape
        cos, sin = self._get_rotary_emb(T, x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        norm_x = self.norm(x)
        proj_out = self.output_proj(norm_x)
        return norm_x, proj_out


class EuclideanCodebook(nn.Module):
    def __init__(self, num_embeddings: int = 2048, embedding_dim: int = 256):
        super().__init__()
        self.cluster_usage = nn.Parameter(torch.zeros(num_embeddings), requires_grad=False)
        self.embedding_sum = nn.Parameter(torch.zeros(num_embeddings, embedding_dim), requires_grad=False)

    def get_embedding(self) -> torch.Tensor:
        usage = self.cluster_usage.unsqueeze(-1).clamp(min=1.0)
        return self.embedding_sum / usage

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, T)
        embed = self.get_embedding()
        return F.embedding(indices, embed)


class VQLayer(nn.Module):
    def __init__(self, num_embeddings: int = 2048, embedding_dim: int = 256):
        super().__init__()
        self._codebook = EuclideanCodebook(num_embeddings, embedding_dim)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return self._codebook.decode(indices)


class RVQGroup(nn.Module):
    def __init__(self, num_layers: int, in_dim: int = 512, codebook_dim: int = 256, out_dim: int = 512, num_embeddings: int = 2048):
        super().__init__()
        self.input_proj = nn.Conv1d(in_dim, codebook_dim, kernel_size=1, bias=False)
        self.vq = nn.Module()
        self.vq.layers = nn.ModuleList([VQLayer(num_embeddings, codebook_dim) for _ in range(num_layers)])
        self.output_proj = nn.Conv1d(codebook_dim, out_dim, kernel_size=1, bias=False)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        # codes: (B, num_layers, T)
        emb_sum = 0
        for i, layer in enumerate(self.vq.layers):
            c_i = codes[:, i, :]  # (B, T)
            emb = layer.decode(c_i)  # (B, T, codebook_dim)
            emb = emb.transpose(1, 2)  # (B, codebook_dim, T)
            emb_sum = emb_sum + emb
        out = self.output_proj(emb_sum)  # (B, out_dim, T)
        return out.transpose(1, 2)  # (B, T, out_dim)


class Quantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.rvq_first = RVQGroup(num_layers=1, in_dim=512, codebook_dim=256, out_dim=512, num_embeddings=2048)
        self.rvq_rest = RVQGroup(num_layers=15, in_dim=512, codebook_dim=256, out_dim=512, num_embeddings=2048)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # audio_codes: (B, 16, T)
        first_codes = audio_codes[:, :1, :]  # (B, 1, T)
        rest_codes = audio_codes[:, 1:, :]   # (B, 15, T)

        first_emb = self.rvq_first.decode(first_codes)  # (B, T, 512)
        rest_emb = self.rvq_rest.decode(rest_codes)    # (B, T, 512)

        return torch.cat([first_emb, rest_emb], dim=-1)  # (B, T, 1024)


class Code2WavDecoderModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.total_upsample = 1920
        self.quantizer = Quantizer()
        self.pre_transformer = PreTransformer(hidden_size=512, num_layers=8, intermediate_size=1024)
        
        self.pre_conv = nn.Sequential()
        self.pre_conv.add_module("conv", nn.Conv1d(512, 1024, kernel_size=3, padding=1))

        self.upsample = nn.ModuleList([
            UpsampleStage(1024),
            UpsampleStage(1024),
        ])

        decoder = nn.Sequential()
        c0 = nn.Sequential()
        c0.add_module("conv", nn.Conv1d(1024, 1536, kernel_size=7, padding=3))
        decoder.add_module("0", c0)
        decoder.add_module("1", DecoderBlock(1536, 768, stride=8, kernel_size=16))
        decoder.add_module("2", DecoderBlock(768, 384, stride=5, kernel_size=10))
        decoder.add_module("3", DecoderBlock(384, 192, stride=4, kernel_size=8))
        decoder.add_module("4", DecoderBlock(192, 96, stride=3, kernel_size=6))
        decoder.add_module("5", Snake1d(96))
        c6 = nn.Sequential()
        c6.add_module("conv", nn.Conv1d(96, 1, kernel_size=7, padding=3))
        decoder.add_module("6", c6)
        self.decoder = decoder

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # audio_codes: (B, 16, T) or (B, T, 16) or (16, T)
        if audio_codes.dim() == 2:
            if audio_codes.shape[0] == 16:
                audio_codes = audio_codes.unsqueeze(0)
            elif audio_codes.shape[1] == 16:
                audio_codes = audio_codes.transpose(0, 1).unsqueeze(0)
            else:
                raise ValueError(f"Unexpected 2D shape: {audio_codes.shape}")
        elif audio_codes.dim() == 3:
            if audio_codes.shape[1] != 16 and audio_codes.shape[2] == 16:
                audio_codes = audio_codes.transpose(1, 2)

        # 1. Quantizer Embedding lookup
        z = self.quantizer(audio_codes)  # (B, T, 1024)

        # 2. PreTransformer
        norm_h, proj_h = self.pre_transformer(z)  # norm_h: (B, T, 512), proj_h: (B, T, 1024)

        # 3. PreConv + OutputProj
        conv_h = self.pre_conv(norm_h.transpose(1, 2))  # (B, 1024, T)
        h = conv_h + proj_h.transpose(1, 2)             # (B, 1024, T)

        # 4. Upsample stages (4x)
        for stage in self.upsample:
            h = stage(h)                 # (B, 1024, 4*T)

        # 5. Decoder backbone (480x)
        wav = self.decoder(h)            # (B, 1, 1920*T)
        return wav.squeeze(1)            # (B, 1920*T)

    def decode(self, audio_codes: torch.Tensor, normalize: bool = True) -> np.ndarray:
        """Decodifica codici RVQ in forma d'onda float32 con normalizzazione di picco in [-1, 1]."""
        with torch.no_grad():
            wav = self.forward(audio_codes)
            wav_arr = wav.cpu().numpy().squeeze()
            if normalize:
                max_val = float(np.max(np.abs(wav_arr)))
                if max_val > 1e-8:
                    wav_arr = wav_arr / max_val
            return wav_arr.astype(np.float32)


def load_code2wav_decoder(weights_dir: str = "code2wav_weights", device: str = "cpu") -> Tuple[Code2WavDecoderModule, Dict[str, Any]]:
    safetensors_path = os.path.join(weights_dir, "code2wav_decoder.safetensors")
    config_path = os.path.join(weights_dir, "code2wav_decoder_config.json")
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"Missing weights file: {safetensors_path}")

    cfg = {}
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    model = Code2WavDecoderModule()
    state_dict = load_file(safetensors_path)

    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        print(f"[Code2WavDecoder] Warning: missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device).eval()
    return model, cfg

