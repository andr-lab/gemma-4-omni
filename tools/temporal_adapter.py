#!/usr/bin/env python3
"""
temporal_adapter.py — TemporalResidualAdapter v2 (estratta da
train_temporal2_cv26_faithful.py, architettura canonica Task 11F/11I-bis).

BUG RISOLTO (2026-08-25): il runtime kaggle_pipeline costruiva ht_adapter come
ResidualAdapter semplice (PIPELINE_HEAD_ARCH=v3) ma il checkpoint canonico
checkpoints/ht_adapter_temporal2_final.pt è la TemporalResidualAdapter v2
(adapter + CausalTemporalModule, 63 chiavi). load_gemma4_duplex_checkpoint
cerca custom_duplex_heads.pt (assente) -> adapter mai caricato -> h_sub
spazzatura -> residui sub-talker random -> "elica".

Questo modulo espone le classi in modo riutilizzabile dal runtime e dai tool.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttentionBlock(nn.Module):
    """Blocco Causal Multi-Head Self-Attention Pre-LN con SDPA nativo PyTorch."""

    def __init__(self, d_model=1024, n_heads=8, d_ff=2048, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"

        self.ln1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = dropout

    def forward(self, x):
        norm_x = self.ln1(x)
        B, T, D = norm_x.shape
        q = self.q_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = x + self.out_proj(attn_out)
        x = x + self.mlp(self.ln2(x))
        return x


class CausalTemporalModule(nn.Module):
    """Modulo Temporale Causale v2 (4 blocchi Transformer causali Pre-LN).
    Mappa h_adapt (2048D) -> h_t (2048D), finestra T=64 (5.12s a 12.5Hz).
    Identity-Init su out_proj (ReZero): delta iniziale nullo.
    """

    def __init__(self, in_dim=2048, hidden_dim=1024, n_layers=4, n_heads=8,
                 d_ff=2048, max_len=512):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([
            CausalSelfAttentionBlock(d_model=hidden_dim, n_heads=n_heads, d_ff=d_ff)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)

    def forward(self, x):
        B, T, D = x.shape
        h = self.in_proj(x) + self.pos_emb[:, :T, :]
        for layer in self.layers:
            h = layer(h)
        h = self.out_norm(h)
        delta = self.out_proj(h)
        return x + delta


class TemporalResidualAdapter(nn.Module):
    """Wrapper composito: ResidualAdapter (1536->2048) + CausalTemporalModule.
    Architettura CANONICA del progetto (checkpoint ht_adapter_temporal2_final.pt).
    """

    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072,
                 temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048,
                 use_causal_conv=True):
        super().__init__()
        from kaggle_pipeline import ResidualAdapter
        self.adapter = ResidualAdapter(in_dim=in_dim, out_dim=out_dim,
                                       hidden_dim=hidden_dim,
                                       use_causal_conv=use_causal_conv)
        self.temporal = CausalTemporalModule(in_dim=out_dim, hidden_dim=temp_dim,
                                             n_layers=temp_layers,
                                             n_heads=temp_heads, d_ff=temp_ff)

    def forward(self, x):
        h_adapt = self.adapter(x)
        return self.temporal(h_adapt)


def build_ht_temporal_adapter(hid, temp_dim=1024, temp_layers=4, temp_heads=8,
                              temp_ff=2048):
    return TemporalResidualAdapter(
        in_dim=hid, out_dim=2048, hidden_dim=3072,
        temp_dim=temp_dim, temp_layers=temp_layers,
        temp_heads=temp_heads, temp_ff=temp_ff, use_causal_conv=True,
    )


def load_temporal_adapter_checkpoint(ckpt_path, device="cuda", dtype=torch.float32):
    """Carica ht_adapter_temporal2_final.pt in un TemporalResidualAdapter v2."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # determina dims dalle chiavi
    in_dim = ck["adapter.skip_proj.weight"].shape[1] if "adapter.skip_proj.weight" in ck else 1536
    out_dim = ck["adapter.skip_proj.weight"].shape[0] if "adapter.skip_proj.weight" in ck else 2048
    temp_dim = ck["temporal.in_proj.weight"].shape[0]
    n_layers = sum(1 for k in ck if k.startswith("temporal.layers.") and k.endswith(".ln1.weight"))
    n_heads = ck["temporal.layers.0.q_proj.weight"].shape[0] // (
        ck["temporal.layers.0.q_proj.weight"].shape[1] // 1 or 1)
    d_ff = ck["temporal.layers.0.mlp.0.weight"].shape[0]
    n_heads = ck["temporal.layers.0.q_proj.weight"].shape[0] // (
        ck["temporal.layers.0.q_proj.weight"].shape[1] // (
            ck["temporal.layers.0.q_proj.weight"].shape[1] // 128 or 1) or 1)
    # fallback: head_dim 128 (config sub-talker) -> n_heads = d_model // 128
    d_model = ck["temporal.in_proj.weight"].shape[0]
    n_heads = d_model // 128
    model = TemporalResidualAdapter(
        in_dim=in_dim, out_dim=out_dim, hidden_dim=3072,
        temp_dim=d_model, temp_layers=n_layers, temp_heads=n_heads, temp_ff=d_ff,
        use_causal_conv=True,
    )
    # rimappa: ck 'adapter.X'/'temporal.X' -> model 'adapter.X'/'temporal.X' (stesso nome)
    missing, unexpected = model.load_state_dict(ck, strict=False)
    print(f"[TEMPORAL] ckpt {ckpt_path}: {len(ck)} chiavi caricate, "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[TEMPORAL] missing: {list(missing)[:8]}", flush=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
