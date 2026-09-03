#!/usr/bin/env python3
"""HARD GATE P4D: Offline causal reference vs vero FuturePrefixQwen slot-KV runtime (P4D §5-§6).

Confronta sugli STESSI input:
A. Reference causale offline: _qwen_logits(causal_prefix=True, prefix_lookahead=0)
B. Vero runtime: FuturePrefixQwen con _SlotKVCache, absolute cache positions, L=0, frame-by-frame.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.v9 import CausalConv1D, DeepCausalTransformerProjectorV9
from tools.kernel_gates.p4_checks import _qwen_logits


def load_checkpoint(ckpt_path: Path, heads_path: Path, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    print(f"[PARITY] Caricamento V9 da {ckpt_path}...", flush=True)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    proj_state = sd.get("proj", sd)

    projector = DeepCausalTransformerProjectorV9()
    projector.load_state_dict(proj_state)
    projector = projector.to(device=device, dtype=dtype).eval()

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"])
    elif heads_path.exists():
        hsd = torch.load(heads_path, map_location="cpu", weights_only=True)
        conv_sd = {k.replace("causal_conv.conv.", "").replace("conv.", ""): v for k, v in hsd.items() if "causal_conv" in k or "conv." in k}
        causal_conv.conv.load_state_dict(conv_sd)
    causal_conv = causal_conv.to(device=device, dtype=dtype).eval()
    return projector, causal_conv


def main() -> int:
    parser = argparse.ArgumentParser(description="P4D Runtime Slot-KV Parity Gate")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_ars_runtime_aligned_v1.pt"))
    parser.add_argument("--ckpt", type=Path, default=Path("/workspace/checkpoints/v9_causal_l0_final.pt"))
    parser.add_argument("--heads", type=Path, default=Path("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    parser.add_argument("--num-seq", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "float32" else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    # Risoluzione fallback checkpoint
    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        fallback = Path("/workspace/checkpoints/v9_causal_l0_unknown.pt")
        if fallback.exists():
            ckpt_path = fallback
            print(f"[PARITY] Fallback al checkpoint: {ckpt_path}", flush=True)
        else:
            print(f"[PARITY] ERRORE: Checkpoint {ckpt_path} non trovato!", flush=True)
            return 1

    projector, causal_conv = load_checkpoint(ckpt_path, args.heads, device=device, dtype=dtype)

    # Caricamento Qwen Talker
    print("[PARITY] Caricamento Qwen3-TTS Talker...", flush=True)
    from duplex_kernel.models.talker import load_talker
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    if args.dtype == "float32":
        talker.talker.to(dtype=torch.float32)

    # Caricamento Cache
    print(f"[PARITY] Caricamento dataset/cache da {args.cache}...", flush=True)
    from tools.kernel_gates.p4_checks import _stack_cache
    raw, anchors, codes, _ = _stack_cache(args.cache)
    raw = raw.to(device=device, dtype=dtype)
    code_seqs = codes.to(device=device)
    if code_seqs.dim() == 2:
        code_seqs = code_seqs.unsqueeze(-1)
    if code_seqs.shape[-1] == 1:
        pad_cbs = torch.zeros((*code_seqs.shape[:-1], 15), dtype=code_seqs.dtype, device=code_seqs.device)
        code_seqs = torch.cat((code_seqs, pad_cbs), dim=-1)

    target_cb0_all = code_seqs[:, :, 0].contiguous()

    num_seq = min(args.num_seq, raw.shape[0])
    frames_per_seq = raw.shape[1]
    total_frames = num_seq * frames_per_seq

    print(f"[PARITY] Esecuzione verifica parita' su {num_seq} sequenze ({total_frames} frame totali, L=0)...", flush=True)

    frame_traces: list[dict[str, Any]] = []
    all_maes: list[float] = []
    all_max_diffs: list[float] = []
    all_cosines: list[float] = []
    top1_agreements = 0
    top5_agreements = 0
    nonfinite_count = 0

    ref_ce_sum = 0.0
    run_ce_sum = 0.0

    t_start = time.time()

    for s_idx in range(num_seq):
        seq_raw = raw[s_idx:s_idx + 1]
        seq_codes = code_seqs[s_idx:s_idx + 1]
        target_cb0 = target_cb0_all[s_idx]

        # 1. Calcolo prefisso V9
        with torch.no_grad():
            h_tx = seq_raw[:, 0, :]
            c = causal_conv(seq_raw)
            pfx = projector(c, h_tx)

        # 2. Reference A: Offline full-sequence causal forward (4D causal mask, no cache)
        with torch.no_grad():
            ref_logits = _qwen_logits(talker, pfx, seq_codes, causal_prefix=True, prefix_lookahead=0)
            # shape: (1, 64, 3072)

        # 3. Candidate B: Vero runtime FuturePrefixQwen con _SlotKVCache (frame-by-frame)
        speaker = FuturePrefixQwen(
            talker,
            lookahead_frames=0,
            max_prefix_frames=frames_per_seq,
            device=device,
            dtype=dtype,
            max_audio_frames=frames_per_seq,
        )
        speaker.begin("Verifica causale del kernel.")

        runtime_logits_list: list[torch.Tensor] = []
        for t in range(frames_per_seq):
            speaker.push_v9_frame(t, pfx[:, t, :])
            step_out = speaker.step_frame()
            runtime_logits_list.append(step_out.logits[:, :1, :])
            speaker.commit_codes(seq_codes[0, t])

        run_logits = torch.cat(runtime_logits_list, dim=1)
        # shape: (1, 64, 3072)

        # 4. Confronto frame-by-frame
        for t in range(frames_per_seq):
            ref_l = ref_logits[0, t].float()
            run_l = run_logits[0, t].float()
            target_y = target_cb0[t]

            if not torch.isfinite(run_l).all() or not torch.isfinite(ref_l).all():
                nonfinite_count += 1

            abs_diff = (ref_l - run_l).abs()
            mae = float(abs_diff.mean().item())
            max_diff = float(abs_diff.max().item())
            cos = float(F.cosine_similarity(ref_l.unsqueeze(0), run_l.unsqueeze(0)).item())

            ref_top1 = int(ref_l.argmax().item())
            run_top1 = int(run_l.argmax().item())
            top1_match = (ref_top1 == run_top1)
            if top1_match:
                top1_agreements += 1

            ref_top5 = set(ref_l.topk(5).indices.tolist())
            run_top5 = set(run_l.topk(5).indices.tolist())
            top5_match = (ref_top5 == run_top5)
            if top5_match:
                top5_agreements += 1

            ref_ce = float(F.cross_entropy(ref_l.unsqueeze(0), target_y.unsqueeze(0)).item())
            run_ce = float(F.cross_entropy(run_l.unsqueeze(0), target_y.unsqueeze(0)).item())

            ref_ce_sum += ref_ce
            run_ce_sum += run_ce

            all_maes.append(mae)
            all_max_diffs.append(max_diff)
            all_cosines.append(cos)

            frame_traces.append({
                "sequence_id": s_idx,
                "frame_id": t,
                "visible_prefix_end": t,
                "target_cb0": int(target_y.item()),
                "ref_top1": ref_top1,
                "run_top1": run_top1,
                "top1_match": top1_match,
                "top5_match": top5_match,
                "mae": mae,
                "max_diff": max_diff,
                "cosine": cos,
                "ref_ce": ref_ce,
                "run_ce": run_ce,
            })

        if (s_idx + 1) % 20 == 0 or (s_idx + 1) == num_seq:
            elapsed = time.time() - t_start
            cur_mae = sum(all_maes) / len(all_maes)
            cur_cos = min(all_cosines)
            cur_top1 = (top1_agreements / len(all_maes)) * 100.0
            print(f"  [PARITY] Sequenze {s_idx + 1}/{num_seq} ({elapsed:.1f}s) | MAE: {cur_mae:.6f} | MinCos: {cur_cos:.6f} | Top-1 Agr: {cur_top1:.2f}%", flush=True)

    # Aggregazione finale
    mean_mae = float(sum(all_maes) / len(all_maes))
    max_diff_overall = float(max(all_max_diffs))
    min_cos_overall = float(min(all_cosines))
    top1_agreement_pct = float((top1_agreements / total_frames) * 100.0)
    top5_agreement_pct = float((top5_agreements / total_frames) * 100.0)
    avg_ref_ce = float(ref_ce_sum / total_frames)
    avg_run_ce = float(run_ce_sum / total_frames)
    ce_ratio = float(avg_run_ce / avg_ref_ce) if avg_ref_ce > 0 else 1.0

    # Criteri di Gate (§6)
    gate_top1 = (top1_agreement_pct == 100.0)
    gate_mae = (mean_mae <= 0.02)
    gate_cos = (min_cos_overall >= 0.999)
    gate_finite = (nonfinite_count == 0)
    all_pass = gate_top1 and gate_mae and gate_cos and gate_finite

    print("=" * 60, flush=True)
    print("RISULTATI RUNTIME PARITY (Offline Causal Ref vs FuturePrefixQwen Slot-KV)", flush=True)
    print(f"  Sequenze valutate: {num_seq} ({total_frames} frame)", flush=True)
    print(f"  Mean Logit MAE:        {mean_mae:.6f} (gate <= 0.02) -> {'PASS' if gate_mae else 'FAIL'}", flush=True)
    print(f"  Max Logit Diff:        {max_diff_overall:.6f}", flush=True)
    print(f"  Min Cosine Similarity: {min_cos_overall:.6f} (gate >= 0.999) -> {'PASS' if gate_cos else 'FAIL'}", flush=True)
    print(f"  Top-1 Agreement:       {top1_agreement_pct:.2f}% ({top1_agreements}/{total_frames}) (gate = 100%) -> {'PASS' if gate_top1 else 'FAIL'}", flush=True)
    print(f"  Top-5 Agreement:       {top5_agreement_pct:.2f}% ({top5_agreements}/{total_frames})", flush=True)
    print(f"  Ref Avg CE:            {avg_ref_ce:.4f}", flush=True)
    print(f"  Run Avg CE:            {avg_run_ce:.4f}", flush=True)
    print(f"  CE Ratio (Run / Ref):  {ce_ratio:.6f}", flush=True)
    print(f"  Nonfinite count:       {nonfinite_count} (gate = 0) -> {'PASS' if gate_finite else 'FAIL'}", flush=True)
    print("=" * 60, flush=True)

    verdict = "P4D RUNTIME SLOT-KV PARITY PASS" if all_pass else "P4D RUNTIME SLOT-KV PARITY FAIL"
    print(f"VERDETTO: {verdict}", flush=True)

    # Salvataggio Report
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "checkpoint": str(ckpt_path),
        "total_sequences": num_seq,
        "total_frames": total_frames,
        "lookahead_frames": 0,
        "mean_mae": mean_mae,
        "max_diff": max_diff_overall,
        "min_cosine": min_cos_overall,
        "top1_agreement_pct": top1_agreement_pct,
        "top5_agreement_pct": top5_agreement_pct,
        "avg_ref_ce": avg_ref_ce,
        "avg_run_ce": avg_run_ce,
        "ce_ratio_run_to_ref": ce_ratio,
        "nonfinite_count": nonfinite_count,
        "gates": {
            "top1_agreement_is_100": gate_top1,
            "mean_mae_le_0_02": gate_mae,
            "min_cosine_ge_0_999": gate_cos,
            "nonfinite_is_0": gate_finite,
            "all_pass": all_pass,
        },
        "verdict": verdict,
    }

    report_path = out_dir / "P4D_RUNTIME_SLOT_KV_PARITY.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[PARITY] Report salvato in {report_path}", flush=True)

    trace_path = out_dir / "P4D_RUNTIME_SLOT_KV_PARITY_TRACE.jsonl"
    with open(trace_path, "w", encoding="utf-8") as f:
        for item in frame_traces:
            f.write(json.dumps(item) + "\n")
    print(f"[PARITY] Trace per-frame salvato in {trace_path} ({len(frame_traces)} record)", flush=True)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
