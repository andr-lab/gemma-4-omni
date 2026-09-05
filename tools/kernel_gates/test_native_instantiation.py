#!/usr/bin/env python3
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM

print("[TEST] Loading thinker...")
th = load_thinker("/workspace", device="cuda:0", dtype=torch.bfloat16)
lm = th.model.model.language_model

print("[TEST] Instantiating NativeGemmaListenerLM...")
native_lm = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
print("[TEST] Instantiated! Current position:", native_lm.current_position)
print("[TEST] Output tensor shape:", native_lm.output_tensor.shape, "device:", native_lm.output_tensor.device)

# Test unprimed diagnostic step (production step_async requires prime).
dummy_input = torch.randn(1, 4, 1536, device="cuda:0", dtype=torch.bfloat16)
out = native_lm.step_debug(dummy_input)
torch.cuda.synchronize()
print("[TEST] Step executed successfully! Out shape:", out.shape, "Out mean:", out.float().mean().item())
print("[TEST] New position:", native_lm.current_position)
print("[SUCCESS] Native executor initialized and executed 1 step cleanly!")
