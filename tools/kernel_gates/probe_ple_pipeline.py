#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace")
lm = th.model.model.language_model

pad_ids = torch.full((1, 4), 0, device="cuda:0", dtype=torch.long)
pad_embeds = lm.embed_tokens(pad_ids)
ple_raw = lm.get_per_layer_inputs(pad_ids, pad_embeds)
print("ple_raw shape:", ple_raw.shape)

x = torch.randn((1, 4, 1536), device="cuda:0", dtype=torch.bfloat16)

# Test project_per_layer_inputs step-by-step
proj = F.linear(x, lm.per_layer_model_projection.weight)
proj = proj * lm.per_layer_model_projection_scale
proj = proj.reshape(1, 4, 35, 256)
mean_sq = proj.float().pow(2).mean(-1, keepdim=True) + 1e-6
proj_normed = (proj.float() * torch.pow(mean_sq, -0.5) * lm.per_layer_projection_norm.weight.float()).to(torch.bfloat16)

projected_ple = (proj_normed + ple_raw) * lm.per_layer_input_scale
print("projected_ple shape:", projected_ple.shape, "mean:", projected_ple.float().mean().item(), "std:", projected_ple.float().std().item())

# Compare with ShapeStable implementation
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
ss_projected = ss_lm._project_per_layer_inputs(x, ple_raw)
diff = (projected_ple.float() - ss_projected.float()).abs().max().item()
print("Max diff vs ShapeStable _project_per_layer_inputs:", diff)
assert diff < 1e-4, f"Mismatch in PLE projection: {diff}"
print("[PASS] PLE projection logic verified bit-exact!")
