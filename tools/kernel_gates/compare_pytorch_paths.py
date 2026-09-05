#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
from test_progressive_layers import run_pytorch_reference_layers, cosine_sim, mae, max_diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

print("[1] Loading Thinker...")
th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

backend = GemmaAudioTowerBackend(th, device=str(device))
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)

# Path A: run_pytorch_reference_layers (which native matches at 0.9995)
ref_out = run_pytorch_reference_layers(lm, ss_lm, audio_soft, num_layers=35, apply_final_norm=True)

# Path B: backend._forward_incremental_lm (unprimed)
state_py_unprimed = GemmaListenerState()
backend_out = backend._forward_incremental_lm(
    embeds=audio_soft,
    per_layer=backend._frame_pad_per_layer_inputs(lm),
    attention=torch.ones((1, 4), device=device, dtype=torch.long),
    positions=torch.arange(0, 4, device=device, dtype=torch.long),
    state=state_py_unprimed,
)

print(f"Path A (ref) vs Path B (backend._forward_incremental_lm):")
print(f"MAE: {mae(ref_out, backend_out):.4e}")
print(f"Cosine: {cosine_sim(ref_out, backend_out):.6f}")
print(f"Max Diff: {max_diff(ref_out, backend_out):.4f}")

# Path C: ShapeStable forward directly
pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping = create_masks_for_generate(
    lm.config,
    audio_soft,
    torch.ones((1, 4), device=device, dtype=torch.long),
    past_key_values=None,
    position_ids=pos_ids,
)
with torch.inference_mode():
    ss_out = ss_lm.forward(
        inputs_embeds=audio_soft,
        per_layer_inputs=backend._frame_pad_per_layer_inputs(lm),
        attention_mask=mask_mapping,
        position_ids=pos_ids,
        past_key_values=None,
        shared_kv_states=None,
    )

print(f"\nPath B vs Path C (ShapeStable directly):")
print(f"MAE: {mae(backend_out, ss_out.last_hidden_state):.4e}")
print(f"Cosine: {cosine_sim(backend_out, ss_out.last_hidden_state):.6f}")

print(f"\nPath A (ref) vs Path C (ShapeStable directly):")
print(f"MAE: {mae(ref_out, ss_out.last_hidden_state):.4e}")
print(f"Cosine: {cosine_sim(ref_out, ss_out.last_hidden_state):.6f}")
