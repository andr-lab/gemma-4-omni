import time
import torch
from pathlib import Path
from duplex_kernel.models.thinker import load_thinker
from transformers import StaticCache

workspace = Path("/workspace")
device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker(workspace, device=device, dtype=dtype)
thinker = th.model
tokenizer = th.tokenizer

prompt = "Spiega l'architettura di un sistema conversazionale full-duplex"
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

compiled_thinker = torch.compile(thinker, mode="default")
cache = StaticCache(config=thinker.config, max_batch_size=1, max_cache_len=256, device=device, dtype=dtype)

print("--- WARMUP RUN ---", flush=True)
# Warmup prefill
out = thinker(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True)
tok = torch.tensor([[100]], device=device, dtype=torch.long)
# Warmup 5 steps to compile both step 0 and loop body
for i in range(5):
    t0 = time.perf_counter()
    out = compiled_thinker(input_ids=tok, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True)
    torch.cuda.synchronize()
    print(f"Warmup step {i}: {(time.perf_counter() - t0)*1000:.2f} ms", flush=True)

print("--- PRODUCTION RUN (reset cache) ---", flush=True)
cache.reset()
t0 = time.perf_counter()
out = thinker(input_ids=prompt_ids, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True)
torch.cuda.synchronize()
print(f"Production prefill: {(time.perf_counter() - t0)*1000:.2f} ms", flush=True)

prod_times = []
for i in range(10):
    t0 = time.perf_counter()
    out = compiled_thinker(input_ids=tok, past_key_values=cache, use_cache=True, output_hidden_states=True, return_dict=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) * 1000.0
    prod_times.append(dt)
    print(f"Production step {i}: {dt:.2f} ms", flush=True)

print(f"Production run max step latency: {max(prod_times):.2f} ms")
