#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import torch
import numpy as np
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
from test_progressive_layers import run_pytorch_reference_layers, cosine_sim, mae, max_diff

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model
backend = GemmaAudioTowerBackend(th, device=str(device))
ss_lm = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

rng = np.random.default_rng(2026)
audio_soft = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)

pos_ids = torch.arange(0, 4, device=device, dtype=torch.long).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate
mask_mapping = create_masks_for_generate(
    lm.config,
    audio_soft,
    torch.ones((1, 4), device=device, dtype=torch.long),
    past_key_values=None,
    position_ids=pos_ids,
)

print(f"{'Layer':<6} | {'MAE':<10} | {'Cosine':<10} | {'MaxDiff':<10}")
print("-" * 45)

for num_layers in range(1, 10):
    # Run Path A for num_layers
    out_a = run_pytorch_reference_layers(lm, ss_lm, audio_soft, num_layers=num_layers, apply_final_norm=False)

    # Run ShapeStable for num_layers manually
    projected_ple = ss_lm._project_per_layer_inputs(audio_soft, backend._frame_pad_per_layer_inputs(lm))
    position_embeddings = {
        layer_type: lm.rotary_emb(audio_soft, pos_ids, layer_type)
        for layer_type in lm.unique_layer_types
    }
    from collections import UserDict
    shared = UserDict()
    from transformers.cache_utils import DynamicCache
    past_key_values = DynamicCache(config=lm.config)
    hidden_states = audio_soft.clone()
    for index in range(num_layers):
        layer = lm.layers[index]
        per_layer_input = projected_ple[:, :, index, :]
        layer_type = lm.config.layer_types[index]
        hidden_states = ss_lm._decoder_layer(
            layer,
            hidden_states,
            per_layer_input,
            position_embeddings[layer_type],
            mask_mapping[layer_type],
            pos_ids,
            shared,
            past_key_values,
        )

    m = mae(out_a, hidden_states)
    c = cosine_sim(out_a, hidden_states)
    mx = max_diff(out_a, hidden_states)
    print(f"{num_layers:<6} | {m:<10.2e} | {c:<10.6f} | {mx:<10.4f}")
