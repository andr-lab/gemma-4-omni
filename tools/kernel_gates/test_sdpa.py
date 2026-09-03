import sys
sys.path.insert(0, "/workspace")
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from duplex_kernel.models.talker import load_talker
from duplex_kernel.future_prefix import FuturePrefixQwen

device = "cuda"
dtype = torch.bfloat16
talker = load_talker("/workspace", device=device, load_code2wav=False)
speaker = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=64, device=device, dtype=dtype, max_audio_frames=64)
speaker.begin("Verifica causale del kernel.")

pfx = torch.randn((1, 2048), device=device, dtype=dtype)
codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

# Test baseline eager step time
for t in range(5):
    speaker.push_v9_frame(t, pfx)
    speaker.step_frame()
    speaker.commit_codes(codes)

ev_start = torch.cuda.Event(enable_timing=True)
ev_end = torch.cuda.Event(enable_timing=True)

torch.cuda.synchronize()
ev_start.record()
for t in range(5, 15):
    speaker.push_v9_frame(t, pfx)
    speaker.step_frame()
    speaker.commit_codes(codes)
ev_end.record()
torch.cuda.synchronize()
eager_ms = ev_start.elapsed_time(ev_end) / 10.0
print(f"Baseline Eager step latency: {eager_ms:.2f} ms")

# Now monkey patch eager_attention_forward to use SDPA
import qwen_tts.core.models.modeling_qwen3_tts as M

def sdpa_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key_states = M.repeat_kv(key, module.num_key_value_groups)
    value_states = M.repeat_kv(value, module.num_key_value_groups)
    causal_mask = attention_mask[:, :, :, :key_states.shape[-2]] if attention_mask is not None else None
    out = F.scaled_dot_product_attention(query, key_states, value_states, attn_mask=causal_mask, scale=scaling)
    return out.transpose(1, 2).contiguous(), None

M.eager_attention_forward = sdpa_attention_forward

speaker.begin("Verifica causale del kernel.")
for t in range(5):
    speaker.push_v9_frame(t, pfx)
    speaker.step_frame()
    speaker.commit_codes(codes)

torch.cuda.synchronize()
ev_start.record()
for t in range(5, 15):
    speaker.push_v9_frame(t, pfx)
    speaker.step_frame()
    speaker.commit_codes(codes)
ev_end.record()
torch.cuda.synchronize()
sdpa_ms = ev_start.elapsed_time(ev_end) / 10.0
print(f"SDPA step latency: {sdpa_ms:.2f} ms")
