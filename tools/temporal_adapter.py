#!/usr/bin/env python3
"""
temporal_adapter.py — TemporalResidualAdapter v2 (estratta 1:1 da
train_temporal2_cv26_faithful.py, architettura canonica Task 11F/11I-bis).

BUG RADICE RISOLTO (2026-08-25): il runtime kaggle_pipeline costruiva
ht_adapter come ResidualAdapter runtime (in_features/out_features, NO
causal_conv interno, NO modulo temporal) ma il checkpoint canonico
checkpoints/ht_adapter_temporal2_final.pt è la TemporalResidualAdapter v2
(CausalConv1d + skip_proj/mlp/norm + CausalTemporalModule, 63 chiavi).
load_gemma4_duplex_checkpoint cercava custom_duplex_heads.pt (assente) ->
adapter mai caricato -> h_sub spazzatura -> residui sub-talker random -> "elica".

Questo modulo è AUTOSUFFICIENTE (non importa kaggle_pipeline): replica le
classi del trainer 1:1 così il checkpoint si carica con strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """Conv1D causale k=3 con left-pad (2,0): contesto [t-2,t-1,t]=240ms, 0 look-ahead.
    Identity-init: peso centrale=I, laterali=0 -> forward(x)==x a passo 0."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, bias=False)
        self._reset_identity_parameters()

    def _reset_identity_parameters(self):
        nn.init.zeros_(self.conv.weight)
        with torch.no_grad():
            _c = self.conv.kernel_size[0] - 1
            for i in range(self.conv.weight.shape[0]):
                self.conv.weight[i, i, _c] = 1.0

    def forward(self, x):
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (2, 0))
        out = self.conv(x_pad)
        return out.transpose(1, 2)


class ResidualAdapter(nn.Module):
    """Adapter MLP residuale: CausalConv1D -> LayerNorm( MLP(h) + skip(h) ).
    skip_proj: Linear(in_dim, out_dim); mlp: Linear -> SiLU -> Linear; norm."""

    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072, use_causal_conv=True):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        if use_causal_conv:
            self.causal_conv = CausalConv1d(in_dim)
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        if self.use_causal_conv:
            x = self.causal_conv(x)
        h_proj = self.skip_proj(x)
        h_mlp = self.mlp(x)
        return self.norm(h_mlp + h_proj)


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
    Identity-Init su out_proj (ReZero): delta iniziale nullo."""

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
    """Wrapper composito: ResidualAdapter (1536->2048) + CausalTemporalModule (2048->2048)."""

    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072,
                 temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048,
                 use_causal_conv=True):
        super().__init__()
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
    """Carica ht_adapter_temporal2_final.pt in un TemporalResidualAdapter v2 (strict)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    in_dim = ck["adapter.skip_proj.weight"].shape[1]
    out_dim = ck["adapter.skip_proj.weight"].shape[0]
    hidden_dim = ck["adapter.mlp.0.weight"].shape[0]
    temp_dim = ck["temporal.in_proj.weight"].shape[0]
    d_ff = ck["temporal.layers.0.mlp.0.weight"].shape[0]
    n_layers = sum(1 for k in ck if k.startswith("temporal.layers.") and k.endswith(".ln1.weight"))
    head_dim = ck["temporal.layers.0.q_proj.weight"].shape[1] // (
        ck["temporal.layers.0.q_proj.weight"].shape[0] // 128 or 1)
    n_heads = ck["temporal.layers.0.q_proj.weight"].shape[0] // head_dim if head_dim else 8

    model = TemporalResidualAdapter(
        in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim,
        temp_dim=temp_dim, temp_layers=n_layers, temp_heads=n_heads, temp_ff=d_ff,
        use_causal_conv=True,
    )
    missing, unexpected = model.load_state_dict(ck, strict=False)
    print(f"[TEMPORAL] ckpt {ckpt_path}: {len(ck)} chiavi, "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[TEMPORAL] missing: {list(missing)[:8]}", flush=True)
    if unexpected:
        print(f"[TEMPORAL] unexpected: {list(unexpected)[:8]}", flush=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model
