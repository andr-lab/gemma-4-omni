#!/usr/bin/env python3
"""Verify numerical parity of compiled Qwen vs eager Qwen in FuturePrefixQwen."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import CausalConv1D, DeepCausalTransformerProjectorV9
from tools.kernel_gates.p4_checks import _stack_cache


def main():
    device = "cuda:0"
    dtype = torch.float32
    print("=" * 70)
    print("[TEST] Verifying Compiled Qwen Slot-KV Parity & Latency")
    print("=" * 70, flush=True)

    # 1. Load Talker
    print("[TEST] Loading Talker Qwen3-TTS...", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)

    # 2. Load Cache
    cache_path = Path("/workspace/cache_ars_runtime_aligned_v1.pt")
    print(f"[TEST] Loading cache from {cache_path}...", flush=True)
    raw, anchors, codes, _ = _stack_cache(cache_path)
    raw = raw.to(device=device, dtype=dtype)
    code_seqs = codes.to(device=device)
    if code_seqs.dim() == 2:
        code_seqs = code_seqs.unsqueeze(-1)
    if code_seqs.shape[-1] == 1:
        pad_cbs = torch.zeros((*code_seqs.shape[:-1], 15), dtype=code_seqs.dtype, device=code_seqs.device)
        code_seqs = torch.cat((code_seqs, pad_cbs), dim=-1)

    # 3. Load V9
    ckpt_path = Path("/workspace/checkpoints/v9_causal_l0_final.pt")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    projector = DeepCausalTransformerProjectorV9()
    projector.load_state_dict(sd.get("proj", sd))
    projector = projector.to(device=device, dtype=dtype).eval()

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"])
    causal_conv = causal_conv.to(device=device, dtype=dtype).eval()

    # Pre-compute V9 prefix for 5 sequences
    num_seq = 5
    frames_per_seq = 64
    all_pfx = []
    with torch.no_grad():
        for s in range(num_seq):
            seq_raw = raw[s : s + 1]
            h_tx = seq_raw[:, 0, :]
            c = causal_conv(seq_raw)
            pfx = projector(c, h_tx)
            all_pfx.append(pfx)

    # --- Run Eager Mode ---
    print("\n[TEST] Running Eager FuturePrefixQwen on 5 sequences...", flush=True)
    eager_logits_all = []
    for s in range(num_seq):
        pfx = all_pfx[s]
        seq_codes = code_seqs[s]
        spk = FuturePrefixQwen(
            talker,
            lookahead_frames=0,
            max_prefix_frames=frames_per_seq,
            device=device,
            dtype=dtype,
            max_audio_frames=frames_per_seq,
        )
        spk.begin("Verifica causale del kernel.")
        seq_l = []
        for t in range(frames_per_seq):
            spk.push_v9_frame(t, pfx[:, t, :])
            step_out = spk.step_frame()
            seq_l.append(step_out.logits[:, :1, :].clone())
            spk.commit_codes(seq_codes[t])
        eager_logits_all.append(torch.cat(seq_l, dim=1))

    # --- Compile Model ---
    print("\n[TEST] Compiling Qwen backbone with torch.compile...", flush=True)
    talker.talker.model = torch.compile(talker.talker.model, mode="default")

    # Warmup compiled model
    print("[TEST] Warming up compiled Qwen...", flush=True)
    spk_warm = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=frames_per_seq,
        device=device,
        dtype=dtype,
        max_audio_frames=frames_per_seq,
    )
    spk_warm.begin("Verifica causale del kernel.")
    for t in range(16):
        spk_warm.push_v9_frame(t, all_pfx[0][:, t, :])
        step_out = spk_warm.step_frame()
        spk_warm.commit_codes(code_seqs[0][t])
    torch.cuda.synchronize()
    print("[TEST] Warmup completed.", flush=True)

    # --- Run Compiled Mode ---
    print("\n[TEST] Running Compiled FuturePrefixQwen on 5 sequences...", flush=True)
    compiled_logits_all = []
    latencies_ms = []

    for s in range(num_seq):
        pfx = all_pfx[s]
        seq_codes = code_seqs[s]
        spk = FuturePrefixQwen(
            talker,
            lookahead_frames=0,
            max_prefix_frames=frames_per_seq,
            device=device,
            dtype=dtype,
            max_audio_frames=frames_per_seq,
        )
        spk.begin("Verifica causale del kernel.")
        seq_l = []
        for t in range(frames_per_seq):
            spk.push_v9_frame(t, pfx[:, t, :])
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step_out = spk.step_frame()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
            seq_l.append(step_out.logits[:, :1, :].clone())
            spk.commit_codes(seq_codes[t])
        compiled_logits_all.append(torch.cat(seq_l, dim=1))

    # --- Compare Eager vs Compiled ---
    total_frames = num_seq * frames_per_seq
    maes = []
    cosines = []
    top1_matches = 0

    for s in range(num_seq):
        e_l = eager_logits_all[s].float()
        c_l = compiled_logits_all[s].float()
        for t in range(frames_per_seq):
            e_t = e_l[0, t]
            c_t = c_l[0, t]
            diff = (e_t - c_t).abs()
            mae = float(diff.mean().item())
            cos = float(F.cosine_similarity(e_t.unsqueeze(0), c_t.unsqueeze(0)).item())
            top1_e = int(e_t.argmax().item())
            top1_c = int(c_t.argmax().item())

            maes.append(mae)
            cosines.append(cos)
            if top1_e == top1_c:
                top1_matches += 1

    top1_rate = top1_matches / total_frames * 100.0
    mean_mae = sum(maes) / len(maes)
    min_cos = min(cosines)

    import numpy as np
    lats = np.array(latencies_ms)
    p50 = float(np.percentile(lats, 50))
    p95 = float(np.percentile(lats, 95))
    p99 = float(np.percentile(lats, 99))

    print("\n" + "=" * 70)
    print("[COMPILED QWEN NUMERICAL PARITY RESULTS]")
    print(f"  • Total Frames Tested: {total_frames}")
    print(f"  • Top-1 Agreement:     {top1_rate:.2f}% (Target: 100.00%)")
    print(f"  • Mean Logit MAE:      {mean_mae:.8f} (Target: <= 0.02)")
    print(f"  • Min Cosine:          {min_cos:.8f} (Target: >= 0.999)")
    print(f"  • Compiled Latency:    p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")
    print("=" * 70, flush=True)

    pass_parity = (top1_rate == 100.0) and (mean_mae <= 0.02) and (min_cos >= 0.999)
    if pass_parity:
        print(">>> RESULT: COMPILED QWEN PARITY PASS <<<")
        return 0
    else:
        print(">>> RESULT: COMPILED QWEN PARITY FAIL <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
