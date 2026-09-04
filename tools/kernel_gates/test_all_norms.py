#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker

th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

layer0 = lm.layers[0]
norms = [
    ("input_layernorm", layer0.input_layernorm, (1, 4, 1536)),
    ("q_norm", layer0.self_attn.q_norm, (1, 4, 16, 256)),
    ("k_norm", layer0.self_attn.k_norm, (1, 4, 16, 256)),
    ("v_norm", layer0.self_attn.v_norm, (1, 4, 16, 256)),
    ("post_attention_layernorm", layer0.post_attention_layernorm, (1, 4, 1536)),
    ("pre_feedforward_layernorm", layer0.pre_feedforward_layernorm, (1, 4, 1536)),
    ("post_feedforward_layernorm", layer0.post_feedforward_layernorm, (1, 4, 1536)),
    ("post_per_layer_input_norm", layer0.post_per_layer_input_norm, (1, 4, 1536)),
    ("per_layer_projection_norm", lm.per_layer_projection_norm, (1, 4, 28, 256)),
]

for name, norm, shape in norms:
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda:0")
    weight = getattr(norm, "weight", None)
    eps = float(norm.eps)
    
    # Method 1: ShapeStable
    mean_sq = x.float().pow(2).mean(-1, keepdim=True) + eps
    out1 = x.float() * torch.pow(mean_sq, -0.5)
    if weight is not None:
        out1 = out1 * weight.float()
    out1 = out1.type_as(x)
    
    # Method 2: F.rms_norm
    norm_shape = (shape[-1],)
    out2 = F.rms_norm(x, norm_shape, weight, eps)
    
    diff = (out1.float() - out2.float()).abs().max().item()
    print(f"Norm {name:28s}: type={type(norm).__name__:20s} weight={'Tensor' if weight is not None else 'None':6s} | diff={diff:.8e}")
