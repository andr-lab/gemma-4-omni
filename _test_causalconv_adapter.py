"""TDD: CausalConv1d dentro ResidualAdapter (fix mismatch training/inferenza, R3).

Estrae le classi CausalConv1d e ResidualAdapter dal SORGENTE REALE
(kernel_src/syntest_v3_cell2.py) — non copie — e verifica il contratto:

1. Identity-init = no-op esatto (delta 0) -> non rompe il checkpoint R2.
2. Warm-start: checkpoint R2 (dataset_r2/ht_adapter_stage2r2_final.pt) caricato
   con strict=False -> output IDENTICO all'adapter senza conv (delta < 1e-4).
3. La conv e' addestrabile (grad step riduce la loss).
4. Causalita': out[t] non dipende da x[t+1:] (zero look-ahead, left-pad (2,0)).
5. Kernel center = j = k-1 = 2 (frame corrente), laterali zero.
6. Shape (B,T,C) -> (B,T,out_dim) corretto.

Per lanciare: python _test_causalconv_adapter.py  (no GPU richiesta)
"""
import ast
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "kernel_src", "syntest_v3_cell2.py")
CKPT = os.path.join(ROOT, "dataset_r2", "ht_adapter_stage2r2_final.pt")


def _load_classes_from_source():
    src = open(SRC, encoding="utf-8").read()
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in ("CausalConv1d", "ResidualAdapter"):
            found[node.name] = ast.get_source_segment(src, node)
    assert set(found) == {"CausalConv1d", "ResidualAdapter"}, f"classi mancanti: {found.keys()}"
    ns = {"torch": torch, "nn": nn, "F": F}
    exec(found["CausalConv1d"], ns)
    exec(found["ResidualAdapter"], ns)
    return ns["CausalConv1d"], ns["ResidualAdapter"]


def main():
    CausalConv1d, ResidualAdapter = _load_classes_from_source()
    ok = True

    # [1] identity no-op
    conv = CausalConv1d(64)
    x = torch.randn(2, 10, 64)
    d = (conv(x) - x).abs().max().item()
    assert d == 0.0, f"[1] identity no-op FAIL: {d}"
    print(f"[1] identity-init no-op: OK (delta 0)")

    # [2] warm-start R2 preserva output
    assert os.path.exists(CKPT), f"[2] checkpoint R2 mancante: {CKPT}"
    sd = torch.load(CKPT, map_location="cpu")
    ad = ResidualAdapter(use_causal_conv=True)
    missing = ad.load_state_dict(sd, strict=False)
    assert set(missing.missing_keys) == {"causal_conv.conv.weight"}, missing
    plain = ResidualAdapter(use_causal_conv=False)
    plain.load_state_dict(sd, strict=False)
    xb = torch.randn(2, 8, 1536)
    with torch.no_grad():
        d2 = (ad(xb) - plain(xb)).abs().max().item()
    assert d2 < 1e-4, f"[2] warm-start FAIL: {d2}"
    print(f"[2] warm-start R2 preserva output: OK (delta {d2:.2e}, missing={list(missing.missing_keys)})")

    # [3] conv addestrabile
    ad2 = ResidualAdapter(use_causal_conv=True)
    opt = torch.optim.Adam(ad2.causal_conv.parameters(), lr=1e-3)
    xin = torch.randn(4, 16, 1536)
    tgt = torch.randn(4, 16, 2048)
    lb = nn.functional.mse_loss(ad2(xin), tgt).item()
    opt.zero_grad()
    nn.functional.mse_loss(ad2(xin), tgt).backward()
    opt.step()
    la = nn.functional.mse_loss(ad2(xin), tgt).item()
    assert la < lb, f"[3] conv non apprende: {lb:.4f} -> {la:.4f}"
    print(f"[3] conv addestrabile: OK ({lb:.4f} -> {la:.4f})")

    # [4] causalita' zero look-ahead
    conv32 = CausalConv1d(32)
    xa = torch.randn(1, 6, 32)
    xb2 = xa.clone()
    xb2[0, 3:, :] = torch.randn(1, 3, 32)
    with torch.no_grad():
        d3 = (conv32(xa)[0, :3, :] - conv32(xb2)[0, :3, :]).abs().max().item()
    assert d3 == 0.0, f"[4] look-ahead FAIL: {d3}"
    print("[4] causalita' zero look-ahead: OK (delta 0)")

    # [5] kernel center j=k-1=2
    w = conv.conv.weight.detach()
    assert w[:, :, 2].abs().sum() == 64.0
    assert w[:, :, 0].abs().sum() == 0.0 and w[:, :, 1].abs().sum() == 0.0
    print("[5] kernel center j=2 (frame corrente), laterali zero: OK")

    # [6] shape forward
    xt = torch.randn(2, 16, 1536)
    with torch.no_grad():
        y = ad(xt)
    assert y.shape == (2, 16, 2048), y.shape
    print("[6] forward (B,T,1536)->(B,T,2048): OK")

    print("\nTUTTI I TEST PASSATI (GREEN)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
