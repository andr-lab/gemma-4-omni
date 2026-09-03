import time
import torch
from pathlib import Path
from duplex_kernel.models.thinker import load_thinker
from transformers.cache_utils import DynamicCache

workspace = Path("/workspace")
device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker(workspace, device=device, dtype=dtype)
thinker = th.model
tokenizer = th.tokenizer

prompt = "Ciao, come stai?"
ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

with torch.inference_mode():
    # Warmup prefill
    out = thinker(input_ids=ids, use_cache=True, return_dict=True)
    past = out.past_key_values
    tok = torch.tensor([[100]], device=device)
    mask = torch.ones((1, ids.shape[1] + 1), device=device)
    
    # Measure 20 decode steps without output_hidden_states
    times_no_hidden = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = thinker(input_ids=tok, past_key_values=past, use_cache=True, return_dict=True)
        torch.cuda.synchronize()
        times_no_hidden.append((time.perf_counter() - t0) * 1000.0)
        past = out.past_key_values
        
    # Measure 20 decode steps WITH output_hidden_states
    past = DynamicCache(config=thinker.config)
    out = thinker(input_ids=ids, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
    times_with_hidden = []
    for _ in range(20):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = thinker(input_ids=tok, past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        torch.cuda.synchronize()
        times_with_hidden.append((time.perf_counter() - t0) * 1000.0)
        past = out.past_key_values

print(f"Decode without hidden_states: mean={sum(times_no_hidden)/len(times_no_hidden):.2f} ms")
print(f"Decode WITH hidden_states:    mean={sum(times_with_hidden)/len(times_with_hidden):.2f} ms")
