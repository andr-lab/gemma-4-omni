import time
import torch
import torch.nn.functional as F
import numpy as np
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph

device = "cuda:0"
dtype = torch.bfloat16
talker = load_talker("/workspace", device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)
cp = talker.talker.code_predictor
embed = talker.talker.get_input_embeddings()

subtalker = SubtalkerGraph(cp, talker_embed=embed).capture()

last_h = torch.randn(1, 1, 2048, device=device, dtype=dtype)
c0 = 100
c0_embed = embed(torch.tensor([[c0]], device=device))

# Preallocated GPU buffers
static_tokens = torch.zeros(1, 15, dtype=torch.long, device=device)
temp = 0.80
top_k = 20
top_p = 0.90

def fast_sample_gpu(logits: torch.Tensor, out_tok: torch.Tensor) -> None:
    # Top-K (20)
    lg = logits / temp
    topk_vals, topk_idx = torch.topk(lg, k=top_k, dim=-1)
    # Top-P on the 20 candidates
    probs = F.softmax(topk_vals, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = (cum - probs) > top_p
    probs[mask] = 0.0
    # Sample from the 20 candidates directly on GPU
    sample_idx = torch.multinomial(probs, num_samples=1)
    # Gather token id directly into out_tok
    torch.gather(topk_idx, -1, sample_idx, out=out_tok)

def fast_greedy_gpu(logits: torch.Tensor, out_tok: torch.Tensor) -> None:
    torch.argmax(logits, dim=-1, keepdim=True, out=out_tok)

def run_fast_subtalker(do_sample: bool = True):
    for _l in subtalker.static_cache.layers:
        _l.cumulative_length.zero_()
    prefill_inputs = torch.cat([last_h, c0_embed], dim=1)
    subtalker.static_prefill_inputs.copy_(prefill_inputs)
    subtalker.prefill_graph.replay()
    
    if do_sample:
        fast_sample_gpu(subtalker.static_logits, subtalker.static_tok)
    else:
        fast_greedy_gpu(subtalker.static_logits, subtalker.static_tok)
    static_tokens[0, 0] = subtalker.static_tok[0, 0]

    for g in range(1, subtalker.max_code_groups):
        subtalker.static_pos.fill_(g + 1)
        subtalker.step_graphs[g - 1].replay()
        if do_sample:
            fast_sample_gpu(subtalker.static_logits, subtalker.static_tok)
        else:
            fast_greedy_gpu(subtalker.static_logits, subtalker.static_tok)
        static_tokens[0, g] = subtalker.static_tok[0, 0]
    
    return static_tokens[0].tolist()

# Warmup
for _ in range(10):
    _ = run_fast_subtalker(do_sample=False)
    _ = run_fast_subtalker(do_sample=True)
torch.cuda.synchronize()

# Benchmark Fast Greedy
times_g = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = run_fast_subtalker(do_sample=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_g.append((t1 - t0) * 1000.0)

p50_g = float(np.percentile(times_g, 50))
p95_g = float(np.percentile(times_g, 95))

# Benchmark Fast Sampling
times_s = []
for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens = run_fast_subtalker(do_sample=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_s.append((t1 - t0) * 1000.0)

p50_s = float(np.percentile(times_s, 50))
p95_s = float(np.percentile(times_s, 95))

print(f"[FAST SUBTALKER RESULTS]")
print(f"  Greedy:   p50={p50_g:.2f}ms, p95={p95_g:.2f}ms")
print(f"  Sampling: p50={p50_s:.2f}ms, p95={p95_s:.2f}ms")
