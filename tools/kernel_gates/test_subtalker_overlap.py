#!/usr/bin/env python3
"""Test if running SubTalker concurrently with Listener Audio Tower/LM causes contention."""
import sys
sys.path.insert(0, "/workspace")
import time
import numpy as np
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph

device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker("/workspace", device=device, dtype=dtype)
listener = GemmaListener.from_thinker(th)
state = GemmaListenerState(context_frames=64)
listener.prime_steady_state(state)

talker = load_talker("/workspace", device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)
cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

pcm = (0.05 * np.random.default_rng(42).standard_normal(1280)).astype(np.float32)
last_h = torch.randn(1, 1, 2048, device=device, dtype=dtype)
c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[100]], device=device))

# Streams
s_listener = torch.cuda.Stream(priority=-1)
s_sub = torch.cuda.Stream(priority=0)

# Warmup both
with torch.cuda.stream(s_listener):
    for _ in range(5):
        listener.step(pcm, None, state)
with torch.cuda.stream(s_sub):
    for _ in range(5):
        subtalker_graph.generate(last_h, 100, c0_embed=c0_embed)
torch.cuda.synchronize()

# Benchmark Isolated Listener
times_l_iso = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s_listener):
        listener.step(pcm, None, state)
        s_listener.synchronize()
    times_l_iso.append((time.perf_counter() - t0) * 1000)

# Benchmark Isolated SubTalker
times_sub_iso = []
for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.cuda.stream(s_sub):
        subtalker_graph.generate(last_h, 100, c0_embed=c0_embed)
        s_sub.synchronize()
    times_sub_iso.append((time.perf_counter() - t0) * 1000)

# Benchmark OVERLAPPED: Launch SubTalker and Listener simultaneously
times_l_ovlp = []
times_sub_ovlp = []
times_wall = []

for _ in range(20):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # Launch SubTalker on stream 1
    with torch.cuda.stream(s_sub):
        subtalker_graph.generate(last_h, 100, c0_embed=c0_embed)
    
    # Launch Listener on stream 0 (priority -1)
    with torch.cuda.stream(s_listener):
        listener.step(pcm, None, state)
    
    s_sub.synchronize()
    t_sub_done = time.perf_counter()
    
    s_listener.synchronize()
    t_l_done = time.perf_counter()
    
    times_sub_ovlp.append((t_sub_done - t0) * 1000)
    times_l_ovlp.append((t_l_done - t0) * 1000)
    times_wall.append((max(t_sub_done, t_l_done) - t0) * 1000)

print(f"Isolated Listener:  p50={np.percentile(times_l_iso, 50):.2f} ms, p95={np.percentile(times_l_iso, 95):.2f} ms")
print(f"Isolated SubTalker: p50={np.percentile(times_sub_iso, 50):.2f} ms, p95={np.percentile(times_sub_iso, 95):.2f} ms")
print(f"Overlapped Listener:  p50={np.percentile(times_l_ovlp, 50):.2f} ms, p95={np.percentile(times_l_ovlp, 95):.2f} ms")
print(f"Overlapped SubTalker: p50={np.percentile(times_sub_ovlp, 50):.2f} ms, p95={np.percentile(times_sub_ovlp, 95):.2f} ms")
print(f"WALL TIME (MAX):      p50={np.percentile(times_wall, 50):.2f} ms, p95={np.percentile(times_wall, 95):.2f} ms")
print(f"Serialized Sum would be: {np.percentile(times_l_iso, 50) + np.percentile(times_sub_iso, 50):.2f} ms")
