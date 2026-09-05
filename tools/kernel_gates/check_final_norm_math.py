#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from test_progressive_layers import cosine_sim, mae, max_diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)

norm = lm.norm
print("norm type:", type(norm))
print("norm weight:", norm.weight.shape, norm.weight.dtype, norm.weight.min().item(), norm.weight.max().item())
print("norm eps:", norm.eps)

x = torch.randn(1, 4, 1536, dtype=dtype, device=device)

# 1. Official PyTorch norm module
y1 = norm(x)

# 2. ShapeStable _padded_rmsnorm
y2 = ss_lm._padded_rmsnorm(norm, x)
print(f"y1 vs y2 (Official vs ShapeStable): MAE={mae(y1, y2):.4e}, Cos={cosine_sim(y1, y2):.6f}")

# 3. Native RMSNorm via step_debug(num_layers=0, apply_final_norm=True)
nat_y = native_lm.step_debug(x, num_layers=0, apply_final_norm=True)
torch.cuda.synchronize()
print(f"y1 vs nat_y (Official PyTorch vs Native Kernel): MAE={mae(y1, nat_y):.4e}, Cos={cosine_sim(y1, nat_y):.6f}, MaxDiff={max_diff(y1, nat_y):.4f}")
assert cosine_sim(y1, nat_y) >= 0.999, f"Final norm kernel mismatch! Cosine={cosine_sim(y1, nat_y)}"
print("[PASS] Final RMSNorm kernel is 100% bit-exact with PyTorch!")
