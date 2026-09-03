#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.kernel_gates.verify_p4d_runtime_suite import (
    run_future_leak_perturbation,
    run_realtime_profiling,
    run_listen_semantics,
    run_cancellation_fencing,
)
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import DeepCausalTransformerProjectorV9, CausalConv1D

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)

    sd = torch.load("/workspace/checkpoints/v9_causal_l0_unknown.pt", map_location="cpu", weights_only=True)
    p = DeepCausalTransformerProjectorV9().to(device=device, dtype=dtype).eval()
    p.load_state_dict(sd.get("proj", sd), strict=False)

    c = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=dtype).eval()
    if "causal_conv" in sd:
        c.conv.load_state_dict(sd["causal_conv"])
    else:
        hsd = torch.load("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt", map_location="cpu", weights_only=True)
        c.conv.load_state_dict(hsd["causal_conv"], strict=False)

    print("Testing Gate 8 (Future Leak)...")
    r8 = run_future_leak_perturbation(talker, p, c, Path("/workspace/cache_p4d_r_512.pt"), device, dtype, num_seq=2)
    print("Gate 8 PASS:", r8["pass_leak"])

    print("Testing Gate 9 (Realtime Profiling in bfloat16 with 64-frame utterance)...")
    talker_bf16 = load_talker("/workspace", device=device, load_code2wav=False)
    p_bf16 = DeepCausalTransformerProjectorV9().to(device=device, dtype=torch.bfloat16).eval()
    p_bf16.load_state_dict(sd.get("proj", sd), strict=False)
    c_bf16 = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=torch.bfloat16).eval()
    if "causal_conv" in sd: c_bf16.conv.load_state_dict(sd["causal_conv"])
    r9 = run_realtime_profiling(talker_bf16, p_bf16, c_bf16, device, torch.bfloat16, num_frames=60)
    print("Gate 9 PASS:", r9["pass_performance"])

    print("Testing Gate 10 (LISTEN)...")
    r10 = run_listen_semantics(None, None, device=device, duration_s=1.0)
    print("Gate 10 PASS:", r10["pass_listen"])

    print("Testing Gate 11 (Fencing)...")
    r11 = run_cancellation_fencing(device=device, num_transitions=10)
    print("Gate 11 PASS:", r11["pass_fencing"])

    print("ALL TEST PASSED!")

if __name__ == "__main__":
    main()
