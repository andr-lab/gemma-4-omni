#!/usr/bin/env python3
"""Benchmark Code2Wav latency for 1, 2, 4 frames on RTX 3090."""
from __future__ import annotations

import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.talker import load_talker


def main():
    device = "cuda:0"
    print("=" * 70)
    print("[BENCHMARK] Code2Wav Latency on RTX 3090")
    print("=" * 70, flush=True)

    talker = load_talker("/workspace", device=device, load_code2wav=True)
    c2w = talker.code2wav

    for num_frames in [1, 2, 4]:
        # Shape expected by c2w.decode: (batch, seq_len, 16)
        codes = torch.randint(0, 2048, (1, num_frames, 16), device=device, dtype=torch.long)
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = c2w.decode(codes)
        torch.cuda.synchronize()

        times = []
        for _ in range(30):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = c2w.decode(codes)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

        import numpy as np
        t = np.array(times)
        print(f"Code2Wav {num_frames} frames ({num_frames * 80} ms audio):")
        print(f"  p50:  {np.percentile(t, 50):.2f} ms")
        print(f"  p95:  {np.percentile(t, 95):.2f} ms")
        print(f"  mean: {np.mean(t):.2f} ms")


if __name__ == "__main__":
    main()
