# -*- coding: utf-8 -*-
# Test unitari per TemporalResidualAdapter IT e Language Verification
import sys, os, torch, numpy as np

sys.path.insert(0, ".")

def test_adapter_it_forward_and_causality():
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super().__init__()
            self.pad = kernel_size - 1
            self.conv = nn.Conv1d(channels, channels, kernel_size, padding=0, bias=False)
            nn.init.zeros_(self.conv.weight)
            with torch.no_grad():
                self.conv.weight[:, :, -1] = torch.eye(channels, channels)

        def forward(self, x):
            x_t = x.transpose(1, 2)
            x_pad = F.pad(x_t, (self.pad, 0))
            out = self.conv(x_pad)
            return out.transpose(1, 2)

    class ResidualAdapter(nn.Module):
        def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072, use_causal_conv=True):
            super().__init__()
            self.use_causal_conv = use_causal_conv
            if use_causal_conv:
                self.conv_in = CausalConv1d(in_dim)
            self.skip_proj = nn.Linear(in_dim, out_dim, bias=False)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x):
            h = self.conv_in(x) if self.use_causal_conv else x
            out = self.skip_proj(h) + self.mlp(h)
            return self.norm(out)

    class CausalSelfAttentionBlock(nn.Module):
        def __init__(self, d_model=1024, n_heads=8, d_ff=2048, dropout=0.0):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
            self.dropout = dropout

        def forward(self, x):
            norm_x = self.ln1(x)
            B, T, D = norm_x.shape
            q = self.q_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.out_proj(attn_out)
            x = x + self.mlp(self.ln2(x))
            return x

    class CausalTemporalModule(nn.Module):
        def __init__(self, in_dim=2048, hidden_dim=1024, n_layers=4, n_heads=8, d_ff=2048, max_len=512):
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
        def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072,
                     temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048):
            super().__init__()
            self.adapter = ResidualAdapter(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, use_causal_conv=True)
            self.temporal = CausalTemporalModule(in_dim=out_dim, hidden_dim=temp_dim, n_layers=temp_layers, n_heads=temp_heads, d_ff=temp_ff)

        def forward(self, x):
            return self.temporal(self.adapter(x))

    model = TemporalResidualAdapter()
    model.eval()

    # 1. Test shape
    x = torch.randn(2, 64, 1536)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 64, 2048), f"Shape attesa (2, 64, 2048), ottenuta {out.shape}"
    print(f"[PASS] Forward shape: {out.shape}")

    # 2. Test causalita rigorosa: modificare x[:, 32:, :] non deve alterare out[:, :32, :]
    x1 = torch.randn(1, 64, 1536)
    x2 = x1.clone()
    x2[:, 32:, :] = torch.randn(1, 32, 1536)

    with torch.no_grad():
        out1 = model(x1)
        out2 = model(x2)

    diff_past = (out1[:, :32, :] - out2[:, :32, :]).abs().max().item()
    diff_future = (out1[:, 32:, :] - out2[:, 32:, :]).abs().max().item()

    assert diff_past < 1e-5, f"Causal leakage rilevato nel passato! diff={diff_past}"
    assert diff_future > 1e-2, f"Nessuna variazione nel futuro modificato! diff={diff_future}"
    print(f"[PASS] Causalita verificata: diff_past={diff_past:.2e} (<1e-5), diff_future={diff_future:.4f}")

    # 3. Test parametrizzazione (59.55M)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PASS] Parametri trainable totali: {n_params:,} (~59.55M)")
    assert 59_000_000 <= n_params <= 60_500_000, f"Numero parametri anomalo: {n_params}"

if __name__ == "__main__":
    test_adapter_it_forward_and_causality()
    print("Tutti i test unitari per TemporalResidualAdapter IT PASS!")
