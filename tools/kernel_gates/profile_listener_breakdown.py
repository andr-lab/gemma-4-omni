import sys
from pathlib import Path
sys.path.insert(0, "/workspace")
import time
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaListener, GemmaListenerState

device = "cuda:0"
dtype = torch.bfloat16
th = load_thinker("/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model

l_base = GemmaListener.from_thinker(th)
s_base = GemmaListenerState(context_frames=64)
l_base.prime_steady_state(s_base)

pcm = (0.05 * torch.randn(1280).numpy()).astype("float32")

# Warmup
for _ in range(5):
    l_base.step(pcm, None, s_base)
torch.cuda.synchronize()

# Time step breakdown
times_fe = []
times_tower = []
times_lm = []
times_step = []

for _ in range(50):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    # 1. FE
    backend = l_base.backend
    mel, past = backend._extract_single_frame(pcm, None)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    # 2. Tower
    audio_soft = backend._encode_audio_frame(mel)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    
    # 3. LM
    h = backend._forward_incremental_lm(audio_soft, s_base)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    
    times_fe.append((t1 - t0) * 1000)
    times_tower.append((t2 - t1) * 1000)
    times_lm.append((t3 - t2) * 1000)
    times_step.append((t3 - t0) * 1000)

import numpy as np
print(f"Incremental FE:   p50={np.percentile(times_fe, 50):.2f} ms, p95={np.percentile(times_fe, 95):.2f} ms")
print(f"Audio Tower:      p50={np.percentile(times_tower, 50):.2f} ms, p95={np.percentile(times_tower, 95):.2f} ms")
print(f"Incremental LM:   p50={np.percentile(times_lm, 50):.2f} ms, p95={np.percentile(times_lm, 95):.2f} ms")
print(f"Total Listener:   p50={np.percentile(times_step, 50):.2f} ms, p95={np.percentile(times_step, 95):.2f} ms")
