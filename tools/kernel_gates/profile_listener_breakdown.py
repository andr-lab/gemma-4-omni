import time
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

device = "cuda:0"
dtype = torch.bfloat16
th = load_thinker(workspace="/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

ss = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

embeds = torch.randn((1, 4, 1536), device=device, dtype=dtype)
positions = torch.arange(4, device=device).unsqueeze(0)
ple = torch.randn((1, 4, 1536), device=device, dtype=dtype)
cache = DynamicCache(config=lm.config)
mask = create_masks_for_generate(lm.config, embeds, torch.ones((1, 4), device=device), past_key_values=cache, position_ids=positions)

# Warmup
for _ in range(5):
    out = ss.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask, position_ids=positions, past_key_values=cache, shared_kv_states=None)
torch.cuda.synchronize()

# Time forward components
t0 = time.perf_counter()
for _ in range(20):
    out = ss.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask, position_ids=positions, past_key_values=cache, shared_kv_states=None)
torch.cuda.synchronize()
total_ms = (time.perf_counter() - t0) * 1000.0 / 20

print(f"ShapeStableGemmaListenerLM total forward: {total_ms:.2f} ms")
