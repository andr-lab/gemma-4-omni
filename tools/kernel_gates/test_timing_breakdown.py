import time
import torch
import numpy as np
from pathlib import Path
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning

workspace = Path("/workspace")
device = "cuda:0"
dtype = torch.bfloat16

th = load_thinker(workspace, device=device, dtype=dtype)
listener = GemmaListener.from_thinker(th)
v9, causal_conv = load_v9_conditioning(device=device, dtype=dtype)
v9_stream = IncrementalV9Projector(v9, causal_conv)
v9_state = v9_stream.new_state(batch_size=1, max_seq_len=512, device=device, dtype=dtype)
anchor = torch.zeros((1, 1536), device=device, dtype=dtype)

state = GemmaListenerState(context_frames=64)
pcm = np.random.randn(1280).astype(np.float32)

# Use default ShapeStableGemmaListenerLM (target_length=291, stream_length=4)

# Fill context to 64 frames to achieve steady state
for _ in range(70):
    out = listener.step(pcm, None, state)
    v9_stream.step(out.gemma_hidden_t, anchor, v9_state)
torch.cuda.synchronize()

times_step = []
times_v9 = []
times_total = []

for i in range(30):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = listener.step(pcm, None, state)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    v9_out = v9_stream.step(out.gemma_hidden_t, anchor, v9_state)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    times_step.append((t1 - t0) * 1000.0)
    times_v9.append((t2 - t1) * 1000.0)
    times_total.append((t2 - t0) * 1000.0)

print(f"Listener.step steady-state: {np.mean(times_step):.2f} ms (p95: {np.percentile(times_step, 95):.2f} ms)")
print(f"V9.step steady-state:       {np.mean(times_v9):.2f} ms")
print(f"Total Complete Step:        {np.mean(times_total):.2f} ms (p95: {np.percentile(times_total, 95):.2f} ms)")
