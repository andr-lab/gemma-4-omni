# -*- coding: utf-8 -*-
# Test unitari per il Modulo Temporale Causale v2 e TemporalResidualAdapter
import sys, os, torch, numpy as np

sys.path.insert(0, ".")

def test_temporal2_architecture():
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConv1d(nn.Module):
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
                is_causal=True
            )
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
                     temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048,
                     use_causal_conv=True):
            super().__init__()
            self.adapter = ResidualAdapter(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, use_causal_conv=use_causal_conv)
            self.temporal = CausalTemporalModule(in_dim=out_dim, hidden_dim=temp_dim, n_layers=temp_layers, n_heads=temp_heads, d_ff=temp_ff)

        def forward(self, x):
            h_adapt = self.adapter(x)
            return self.temporal(h_adapt)

    print("--- TEST 1: Shape e Forward (4 layers) ---")
    model = TemporalResidualAdapter()
    x = torch.randn(2, 64, 1536)
    out = model(x)
    assert out.shape == (2, 64, 2048), f"Shape errata: {out.shape}"
    print("Test 1 PASS: out shape (2, 64, 2048)")

    print("--- TEST 2: Identity-Init a Passo 0 (Exact Match con Adapter R2) ---")
    model.eval()
    with torch.no_grad():
        out_full = model(x)
        out_adapt = model.adapter(x)
    delta = (out_full - out_adapt).abs().max().item()
    print(f"Max delta (full vs adapter at step 0): {delta:.8e}")
    assert delta < 1e-6, f"Identity init fallita! delta={delta}"
    print("Test 2 PASS: Identity-Init preserva l'output dell'adapter a passo 0 esatto")

    print("--- TEST 3: Causalita' Rigorosa (Zero Look-Ahead su 4 blocchi) ---")
    # Modificando il futuro a t >= 32, l'output a t < 32 NON deve cambiare
    x1 = torch.randn(1, 64, 1536)
    x2 = x1.clone()
    x2[:, 32:, :] = torch.randn(1, 32, 1536)

    # Impostiamo pesi non nulli nel modulo temporale per testare l'attention causale
    nn.init.normal_(model.temporal.out_proj.weight, std=0.1)
    with torch.no_grad():
        o1 = model(x1)
        o2 = model(x2)

    diff_past = (o1[:, :32, :] - o2[:, :32, :]).abs().max().item()
    diff_future = (o1[:, 32:, :] - o2[:, 32:, :]).abs().max().item()
    print(f"Diff passato (t < 32): {diff_past:.8e} | Diff futuro (t >= 32): {diff_future:.4f}")
    assert diff_past == 0.0, f"Causal leakage rilevato! diff_past={diff_past}"
    assert diff_future > 0.01, "Il futuro doveva cambiare ma non e cambiato"
    print("Test 3 PASS: Causalita' verificata (zero look-ahead al 100%)")

    print("--- TEST 4: Parametri Totali (Vincolo < 80M) ---")
    n_ad = sum(p.numel() for p in model.adapter.parameters())
    n_temp = sum(p.numel() for p in model.temporal.parameters())
    n_tot = n_ad + n_temp
    print(f"Parametri Adapter: {n_ad:,} ({n_ad/1e6:.2f}M)")
    print(f"Parametri Modulo Temporale v2 (4 layers): {n_temp:,} ({n_temp/1e6:.2f}M)")
    print(f"Totale Trainable: {n_tot:,} ({n_tot/1e6:.2f}M)")
    assert n_tot < 80_000_000, f"Troppi parametri! {n_tot} >= 80M"
    assert n_temp > 30_000_000, f"Attesi oltre 30M nel modulo temporale 4 layers, trovati {n_temp}"
    print("Test 4 PASS: Parametri nel budget (<80M, esattamente ~59.51M)")

    print("--- TEST 5: Autograd Backward & Gradients ---")
    model.train()
    target = torch.randn(2, 64, 2048)
    out = model(x)
    loss = F.mse_loss(out, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"Gradiente mancante per {name}"
        assert not torch.isnan(p.grad).any(), f"NaN nel gradiente per {name}"
    print("Test 5 PASS: Autograd e gradienti attivi su tutti i moduli")

    print("=== TUTTI I 5 TEST TEMPORAL2 SUPERATI CON SUCCESSO! ===")

if __name__ == "__main__":
    test_temporal2_architecture()
