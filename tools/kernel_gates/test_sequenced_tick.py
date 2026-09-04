#!/usr/bin/env python3
"""Test exact sequenced tick latency on RTX 3090."""
import sys
sys.path.insert(0, "/workspace")
import time
import numpy as np
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState, AssistantReferenceBuffer
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph
from duplex_kernel.future_prefix import FuturePrefixQwen

device = "cuda:0"
dtype = torch.bfloat16

print("Loading models...")
th = load_thinker("/workspace", device=device, dtype=dtype)
listener = GemmaListener.from_thinker(th)

projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)
v9_stream = IncrementalV9Projector(projector, causal_conv)

talker = load_talker("/workspace", device=device, load_code2wav=True)
talker.talker.to(dtype=dtype)
talker.talker.model = torch.compile(talker.talker.model, mode="default")

cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

turn_len = 64
speaker = FuturePrefixQwen(
    talker,
    lookahead_frames=0,
    max_prefix_frames=turn_len,
    device=device,
    dtype=dtype,
    max_audio_frames=turn_len,
)

listener_state = GemmaListenerState(context_frames=64)
v9_anchor = torch.zeros((1, 1536), device=device, dtype=dtype)
v9_state = v9_stream.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

print("Warming up listener...")
warmup_pcm = (0.05 * np.random.default_rng(42).standard_normal(1280)).astype(np.float32)
for _ in range(64):
    w_out = listener.step(warmup_pcm, None, listener_state)
    w_h = w_out.gemma_hidden_t
    if w_h.dim() == 1:
        w_h = w_h.unsqueeze(0)
    v9_stream.step(w_h, v9_anchor, v9_state)
listener.prime_steady_state(listener_state)
torch.cuda.synchronize()

print("Warming up compiled speaker & SubTalker...")
speaker.begin("Warmup compiled speaker.")
dummy_cond = torch.randn((1, 2048), device=device, dtype=dtype)
dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)
for t in range(8):
    print(f"  Warmup speaker frame {t} starting...", flush=True)
    t0_w = time.perf_counter()
    speaker.push_v9_frame(t, dummy_cond)
    step_out = speaker.step_frame()
    c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
    c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
    subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
    speaker.commit_codes(dummy_codes)
    torch.cuda.synchronize()
    print(f"  Warmup speaker frame {t} done in {time.perf_counter() - t0_w:.2f}s", flush=True)
torch.cuda.synchronize()

print("Measuring 30 sequenced ticks...")
speaker.begin("Continuous conversational speech.")
v9_state = v9_stream.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

times_l = []
times_v9 = []
times_qwen = []
times_sub = []
times_tot = []

for t in range(30):
    pcm = (0.05 * np.random.default_rng(100 + t).standard_normal(1280)).astype(np.float32)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # 1. Listener
    l_out = listener.step(pcm, None, listener_state)
    h_gemma = l_out.gemma_hidden_t
    if h_gemma.dim() == 1:
        h_gemma = h_gemma.unsqueeze(0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    # 2. V9 Projector
    cond = v9_stream.step(h_gemma, v9_anchor, v9_state)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    # 3. Speaker (Qwen Joint Step)
    speaker.push_v9_frame(t, cond)
    step_out = speaker.step_frame()
    c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
    c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    # 4. SubTalker (15 codebooks GPU resident)
    sub_codes = subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
    codes_16 = torch.tensor([c0] + (sub_codes if isinstance(sub_codes, list) else sub_codes.tolist()), device=device, dtype=torch.long)
    speaker.commit_codes(codes_16)
    torch.cuda.synchronize()
    t4 = time.perf_counter()
    
    times_l.append((t1 - t0) * 1000)
    times_v9.append((t2 - t1) * 1000)
    times_qwen.append((t3 - t2) * 1000)
    times_sub.append((t4 - t3) * 1000)
    times_tot.append((t4 - t0) * 1000)
    print(f"  Tick {t:2d}: L={times_l[-1]:.1f}ms, V9={times_v9[-1]:.1f}ms, Qwen={times_qwen[-1]:.1f}ms, Sub={times_sub[-1]:.1f}ms | TOT={times_tot[-1]:.1f}ms", flush=True)

print("\n--- Latency Breakdown (30 ticks) ---")
print(f"Listener:    p50={np.percentile(times_l, 50):.2f} ms, p95={np.percentile(times_l, 95):.2f} ms")
print(f"V9:          p50={np.percentile(times_v9, 50):.2f} ms, p95={np.percentile(times_v9, 95):.2f} ms")
print(f"Qwen Joint:  p50={np.percentile(times_qwen, 50):.2f} ms, p95={np.percentile(times_qwen, 95):.2f} ms")
print(f"SubTalker:   p50={np.percentile(times_sub, 50):.2f} ms, p95={np.percentile(times_sub, 95):.2f} ms")
print(f"TOTAL TICK:  p50={np.percentile(times_tot, 50):.2f} ms, p95={np.percentile(times_tot, 95):.2f} ms")
