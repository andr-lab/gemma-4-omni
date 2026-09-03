import time
import torch
from pathlib import Path
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers.cache_utils import DynamicCache

workspace = Path("/workspace")
device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker(workspace, device=device, dtype=dtype)
lm = th.model.model.language_model

ss_291 = ShapeStableGemmaListenerLM(lm, target_length=291)
ss_4 = ShapeStableGemmaListenerLM(lm, target_length=4)

embeds = torch.randn((1, 4, 1536), device=device, dtype=dtype)
positions = torch.arange(4, device=device).unsqueeze(0)
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

cache_291 = DynamicCache(config=lm.config)
mask_291 = create_masks_for_generate(lm.config, embeds, torch.ones((1, 4), device=device), past_key_values=cache_291, position_ids=positions)

pad_ids = torch.full((1, 4), 0, dtype=torch.long, device=device)
pad_embeds = lm.embed_tokens(pad_ids)
ple = lm.get_per_layer_inputs(pad_ids, pad_embeds).detach()

# Warmup 291
for _ in range(5):
    out = ss_291.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_291, position_ids=positions, past_key_values=cache_291, shared_kv_states=None)
torch.cuda.synchronize()

times_291 = []
for _ in range(20):
    t0 = time.perf_counter()
    out = ss_291.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_291, position_ids=positions, past_key_values=cache_291, shared_kv_states=None)
    torch.cuda.synchronize()
    times_291.append((time.perf_counter() - t0) * 1000.0)

cache_4 = DynamicCache(config=lm.config)
mask_4 = create_masks_for_generate(lm.config, embeds, torch.ones((1, 4), device=device), past_key_values=cache_4, position_ids=positions)

# Warmup 4
for _ in range(5):
    out = ss_4.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_4, position_ids=positions, past_key_values=cache_4, shared_kv_states=None)
torch.cuda.synchronize()

times_4 = []
for _ in range(20):
    t0 = time.perf_counter()
    out = ss_4.forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_4, position_ids=positions, past_key_values=cache_4, shared_kv_states=None)
    torch.cuda.synchronize()
    times_4.append((time.perf_counter() - t0) * 1000.0)

print(f"LM Forward (target_length=291): mean={sum(times_291)/len(times_291):.2f} ms")
print(f"LM Forward (target_length=4):   mean={sum(times_4)/len(times_4):.2f} ms")

try:
    print("Testing torch.compile on ss_4.forward...", flush=True)
    compiled_forward = torch.compile(ss_4.forward, mode="default")
    # Warmup compiled
    for _ in range(5):
        out = compiled_forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_4, position_ids=positions, past_key_values=cache_4, shared_kv_states=None)
    torch.cuda.synchronize()
    times_compiled = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = compiled_forward(inputs_embeds=embeds, per_layer_inputs=ple, attention_mask=mask_4, position_ids=positions, past_key_values=cache_4, shared_kv_states=None)
        torch.cuda.synchronize()
        times_compiled.append((time.perf_counter() - t0) * 1000.0)
    print(f"LM Forward (compiled target_length=4): mean={sum(times_compiled)/len(times_compiled):.2f} ms")
except Exception as e:
    print(f"torch.compile failed on ss_4: {e}", flush=True)
