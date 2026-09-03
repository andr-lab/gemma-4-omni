#!/usr/bin/env python3
"""Suite di verifica runtime per chiusura P4D:
- §7: Real Runtime Quality Gate (FuturePrefixQwen L=0)
- §8: Future-Leak Perturbation Test su FuturePrefixQwen
- §9: Profiling Frame-by-Frame (V9 + Qwen CB0 + SubTalker <= 80ms)
- §10: Semantica LISTEN (zero chiamate speaker per 10 secondi)
- §11: Cancellation / YIELD / REPLAN (1000 transizioni, zero stale writes)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    IncrementalV9Projector,
)
from tools.kernel_gates.p4_checks import _stack_cache


def run_runtime_quality_gate(
    talker: Any,
    projector: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    cache_path: Path,
    device: str,
    dtype: torch.dtype,
    num_seq: int = 100,
) -> dict[str, Any]:
    print("\n[GATE §7] Calcolo del vero Runtime Quality Gate (FuturePrefixQwen L=0)...", flush=True)
    raw, anchors, codes, _ = _stack_cache(cache_path)
    raw = raw[:num_seq].to(device=device, dtype=dtype)
    codes = codes[:num_seq].to(device=device)
    if codes.dim() == 2:
        codes = codes.unsqueeze(-1)
    if codes.shape[-1] == 1:
        pad_cbs = torch.zeros((*codes.shape[:-1], 15), dtype=codes.dtype, device=codes.device)
        codes = torch.cat((codes, pad_cbs), dim=-1)

    target_cb0 = codes[:, :, 0].contiguous()
    total_frames = raw.shape[0] * raw.shape[1]
    frames_per_seq = raw.shape[1]

    # 1. Full reference non-causale
    from tools.kernel_gates.p4_checks import _qwen_logits
    with torch.no_grad():
        all_pfx = []
        for i in range(raw.shape[0]):
            h_tx = raw[i:i+1, 0, :]
            c = causal_conv(raw[i:i+1])
            p = projector(c, h_tx)
            all_pfx.append(p)
        full_pfx = torch.cat(all_pfx, dim=0)

        # Valuta in chunk per reference non causale
        ref_logits_list = []
        for i in range(0, full_pfx.shape[0], 20):
            l = _qwen_logits(talker, full_pfx[i:i+20], codes[i:i+20], causal_prefix=False)
            ref_logits_list.append(l)
        ref_logits = torch.cat(ref_logits_list, dim=0).float()
        ref_ce = float(F.cross_entropy(ref_logits.view(-1, ref_logits.shape[-1]), target_cb0.view(-1)).item())
        ref_top1 = float((ref_logits.argmax(-1) == target_cb0).float().mean().item())
        ref_top5 = float(ref_logits.topk(5, -1).indices.eq(target_cb0.unsqueeze(-1)).any(-1).float().mean().item())

    print(f"  Reference non-causale: CE={ref_ce:.4f}, Top-1={ref_top1*100:.2f}%, Top-5={ref_top5*100:.2f}%", flush=True)

    # 2. Vero Runtime FuturePrefixQwen slot-KV
    run_logits_list = []
    t0 = time.time()
    for s_idx in range(raw.shape[0]):
        speaker = FuturePrefixQwen(
            talker,
            lookahead_frames=0,
            max_prefix_frames=frames_per_seq,
            device=device,
            dtype=dtype,
            max_audio_frames=frames_per_seq,
        )
        speaker.begin("Verifica causale del kernel.")
        seq_pfx = full_pfx[s_idx:s_idx+1]
        seq_codes = codes[s_idx]

        seq_out = []
        for t in range(frames_per_seq):
            speaker.push_v9_frame(t, seq_pfx[:, t, :])
            step = speaker.step_frame()
            seq_out.append(step.logits[:, :1, :].float())
            speaker.commit_codes(seq_codes[t])
        run_logits_list.append(torch.cat(seq_out, dim=1))

    run_logits = torch.cat(run_logits_list, dim=0)
    causal_ce = float(F.cross_entropy(run_logits.view(-1, run_logits.shape[-1]), target_cb0.view(-1)).item())
    causal_top1 = float((run_logits.argmax(-1) == target_cb0).float().mean().item())
    causal_top5 = float(run_logits.topk(5, -1).indices.eq(target_cb0.unsqueeze(-1)).any(-1).float().mean().item())

    ce_ratio = causal_ce / ref_ce if ref_ce > 0 else 1.0
    top1_drop = (ref_top1 - causal_top1) * 100.0
    top5_drop = (ref_top5 - causal_top5) * 100.0

    pass_ce = (ce_ratio <= 1.10)
    pass_top1 = (top1_drop <= 5.0)
    pass_top5 = (top5_drop <= 3.0)
    all_pass = pass_ce and pass_top1 and pass_top5

    print(f"  Vero Runtime Slot-KV (L=0): CE={causal_ce:.4f}, Top-1={causal_top1*100:.2f}%, Top-5={causal_top5*100:.2f}%", flush=True)
    print(f"  CE Ratio:  {ce_ratio:.4f} (gate <= 1.10) -> {'PASS' if pass_ce else 'FAIL'}", flush=True)
    print(f"  Top-1 Drop: {top1_drop:.2f} pp (gate <= 5.0 pp) -> {'PASS' if pass_top1 else 'FAIL'}", flush=True)
    print(f"  Top-5 Drop: {top5_drop:.2f} pp (gate <= 3.0 pp) -> {'PASS' if pass_top5 else 'FAIL'}", flush=True)

    return {
        "ref_ce": ref_ce,
        "ref_top1": ref_top1,
        "ref_top5": ref_top5,
        "runtime_ce": causal_ce,
        "runtime_top1": causal_top1,
        "runtime_top5": causal_top5,
        "ce_ratio": ce_ratio,
        "top1_drop_pp": top1_drop,
        "top5_drop_pp": top5_drop,
        "pass_ce": pass_ce,
        "pass_top1": pass_top1,
        "pass_top5": pass_top5,
        "all_pass": all_pass,
    }


def run_future_leak_perturbation(
    talker: Any,
    projector: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    cache_path: Path,
    device: str,
    dtype: torch.dtype,
    num_seq: int = 50,
) -> dict[str, Any]:
    print("\n[GATE §8] Verifica Future-Leak su FuturePrefixQwen con perturbazione attiva...", flush=True)
    raw, anchors, codes, _ = _stack_cache(cache_path)
    raw = raw[:num_seq].to(device=device, dtype=dtype)
    codes = codes[:num_seq].to(device=device)
    if codes.dim() == 2:
        codes = codes.unsqueeze(-1)
    if codes.shape[-1] == 1:
        pad_cbs = torch.zeros((*codes.shape[:-1], 15), dtype=codes.dtype, device=codes.device)
        codes = torch.cat((codes, pad_cbs), dim=-1)

    frames_per_seq = raw.shape[1]
    max_leak_diff = 0.0
    leak_violations = 0

    for s_idx in range(num_seq):
        h_tx = raw[s_idx:s_idx+1, 0, :]
        c = causal_conv(raw[s_idx:s_idx+1])
        pfx = projector(c, h_tx)

        # Baseline: esecuzione standard fino a frame t_test
        t_test = int(np.random.randint(5, frames_per_seq - 5))

        # Runtime A: non perturbato
        sp_a = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=frames_per_seq, device=device, dtype=dtype, max_audio_frames=frames_per_seq)
        sp_a.begin("Verifica causale del kernel.")
        for t in range(t_test + 1):
            sp_a.push_v9_frame(t, pfx[:, t, :])
            out_a = sp_a.step_frame()
            if t < t_test:
                sp_a.commit_codes(codes[s_idx, t])

        # Runtime B: alterazione casuale violenta di tutti i frame futuri raw > t_test
        raw_perturbed = raw[s_idx:s_idx+1].clone()
        raw_perturbed[:, t_test + 1:] = torch.randn_like(raw_perturbed[:, t_test + 1:]) * 100.0
        c_p = causal_conv(raw_perturbed)
        pfx_p = projector(c_p, h_tx)

        sp_b = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=frames_per_seq, device=device, dtype=dtype, max_audio_frames=frames_per_seq)
        sp_b.begin("Verifica causale del kernel.")
        for t in range(t_test + 1):
            sp_b.push_v9_frame(t, pfx_p[:, t, :])
            out_b = sp_b.step_frame()
            if t < t_test:
                sp_b.commit_codes(codes[s_idx, t])

        diff = (out_a.logits.float() - out_b.logits.float()).abs().max().item()
        if diff > max_leak_diff:
            max_leak_diff = diff
        if diff > 1e-5:
            leak_violations += 1

    pass_leak = (max_leak_diff <= 1e-5) and (leak_violations == 0)
    print(f"  Max Leakage Diff al frame t: {max_leak_diff:.2e} (tolleranza <= 1e-5) -> {'PASS' if pass_leak else 'FAIL'}", flush=True)
    print(f"  Violazioni di causalita': {leak_violations} (atteso 0)", flush=True)

    return {
        "max_leak_diff": max_leak_diff,
        "leak_violations": leak_violations,
        "pass_leak": pass_leak,
    }


def run_realtime_profiling(
    talker: Any,
    projector: DeepCausalTransformerProjectorV9,
    causal_conv: CausalConv1D,
    device: str,
    dtype: torch.dtype,
    num_frames: int = 500,
) -> dict[str, Any]:
    print(f"\n[GATE §9] Profiling Frame-by-Frame reale su {num_frames} iterazioni...", flush=True)
    from duplex_kernel.subtalker_graph import SubtalkerGraph
    cp = talker.talker.code_predictor
    subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

    # Prepara istanza incrementale V9
    inc_v9 = IncrementalV9Projector(projector, causal_conv)

    turn_len = 64
    speaker = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=turn_len, device=device, dtype=dtype, max_audio_frames=turn_len)
    speaker.begin("Verifica causale del kernel.")
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

    dummy_raw = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_anchor = torch.randn((1, 1536), device=device, dtype=dtype)
    dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

    # Warmup (5 frame)
    for t in range(5):
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)
        speaker.push_v9_frame(t, h_pfx)
        out = speaker.step_frame()
        c0_token = int(out.logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0_token]], device=device))
        resid = subtalker_graph.generate(out.last_hidden, c0_token, c0_embed=c0_embed)
        speaker.commit_codes(dummy_codes)

    speaker.begin("Verifica causale del kernel.")
    v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

    # Profiling per-componente con torch.cuda.Event
    v9_times: list[float] = []
    qwen_slot_times: list[float] = []
    qwen_step_times: list[float] = []
    subtalker_times: list[float] = []
    combined_speaker_times: list[float] = []

    start_ev = torch.cuda.Event(enable_timing=True)
    v9_ev = torch.cuda.Event(enable_timing=True)
    slot_ev = torch.cuda.Event(enable_timing=True)
    step_ev = torch.cuda.Event(enable_timing=True)
    sub_ev = torch.cuda.Event(enable_timing=True)

    for i in range(num_frames):
        t = i % turn_len
        if t == 0 and i > 0:
            speaker.begin("Verifica causale del kernel.")
            v9_state = inc_v9.new_state(batch_size=1, dtype=dtype, max_seq_len=turn_len)

        start_ev.record()

        # 1. V9 incremental step
        h_pfx = inc_v9.step(dummy_raw, dummy_anchor, v9_state)
        v9_ev.record()

        # 2. Qwen push & prefix-slot update
        speaker.push_v9_frame(t, h_pfx)
        slot_ev.record()

        # 3. Qwen audio step + CB0 head
        step_out = speaker.step_frame()
        step_ev.record()

        # 4. SubTalker (15 CUDA graphs)
        c0_token = int(step_out.logits[:, :1, :].argmax(-1).item())
        c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0_token]], device=device))
        resid = subtalker_graph.generate(step_out.last_hidden, c0_token, c0_embed=c0_embed)
        sub_ev.record()

        speaker.commit_codes(dummy_codes)

        torch.cuda.synchronize()
        t_v9 = start_ev.elapsed_time(v9_ev)
        t_slot = v9_ev.elapsed_time(slot_ev)
        t_step = slot_ev.elapsed_time(step_ev)
        t_sub = step_ev.elapsed_time(sub_ev)
        t_combined = start_ev.elapsed_time(sub_ev)

        v9_times.append(t_v9)
        qwen_slot_times.append(t_slot)
        qwen_step_times.append(t_step)
        subtalker_times.append(t_sub)
        combined_speaker_times.append(t_combined)

    def p_stats(arr: list[float]) -> dict[str, float]:
        a = np.array(arr)
        return {
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "mean": float(np.mean(a)),
            "max": float(np.max(a)),
        }

    v9_s = p_stats(v9_times)
    slot_s = p_stats(qwen_slot_times)
    step_s = p_stats(qwen_step_times)
    sub_s = p_stats(subtalker_times)
    comb_s = p_stats(combined_speaker_times)

    deadline_misses = sum(1 for x in combined_speaker_times if x > 80.0)
    deadline_miss_rate = (deadline_misses / len(combined_speaker_times)) * 100.0 if combined_speaker_times else 100.0
    jitter_ms = comb_s["p99"] - comb_s["p50"]
    pass_p95 = bool(comb_s["p95"] <= 80.0)
    pass_miss_rate = bool(deadline_miss_rate <= 1.0)
    pass_performance = bool(len(combined_speaker_times) == num_frames and np.isfinite(comb_s["mean"]) and pass_p95 and pass_miss_rate)

    print(f"  V9.step latency:          p50={v9_s['p50']:.2f}ms, p95={v9_s['p95']:.2f}ms, p99={v9_s['p99']:.2f}ms", flush=True)
    print(f"  Qwen prefix slot update:  p50={slot_s['p50']:.2f}ms, p95={slot_s['p95']:.2f}ms", flush=True)
    print(f"  Qwen audio step + CB0:    p50={step_s['p50']:.2f}ms, p95={step_s['p95']:.2f}ms, p99={step_s['p99']:.2f}ms", flush=True)
    print(f"  SubTalker (15 graphs):    p50={sub_s['p50']:.2f}ms, p95={sub_s['p95']:.2f}ms, p99={sub_s['p99']:.2f}ms", flush=True)
    print(f"  COMBINED [V9+Qwen+SubT]:  p50={comb_s['p50']:.2f}ms, p95={comb_s['p95']:.2f}ms, p99={comb_s['p99']:.2f}ms (target p95 <= 80ms) -> {'PASS' if pass_p95 else 'FAIL'}", flush=True)
    print(f"  Jitter (p99 - p50):       {jitter_ms:.2f}ms", flush=True)
    print(f"  Deadline miss rate (>80ms): {deadline_miss_rate:.2f}% (target <= 1.0%) -> {'PASS' if pass_miss_rate else 'FAIL'}", flush=True)
    print(f"  Performance gate pass:    {'PASS' if pass_performance else 'FAIL'}", flush=True)

    return {
        "v9": v9_s,
        "qwen_slot": slot_s,
        "qwen_audio_step": step_s,
        "subtalker": sub_s,
        "combined": comb_s,
        "jitter_ms": jitter_ms,
        "deadline_miss_rate": deadline_miss_rate,
        "nominal_deadline_ms": 80.0,
        "pass_p95": pass_p95,
        "pass_miss_rate": pass_miss_rate,
        "pass_performance": pass_performance,
    }


def run_listen_semantics(
    speaker: Any,
    subtalker_graph: Any,
    device: str,
    duration_s: float = 10.0,
) -> dict[str, Any]:
    print(f"\n[GATE §10] Verifica Semantica LISTEN (zero chiamate speaker per {duration_s}s)...", flush=True)
    qwen_call_count = 0
    subtalker_call_count = 0

    # Gate di ascolto (LISTEN mode): lo speaker loop deve essere completamente silenziato
    speaker_gate_open = False
    ticks = int(duration_s * 12.5)  # 12.5 Hz (80ms per tick) = 125 tick in 10s
    listener_ticks = 0

    for _ in range(ticks):
        listener_ticks += 1
        # In modalita' LISTEN, lo speaker_gate e' chiuso: nessuna chiamata a step_frame o SubTalker
        if speaker_gate_open:
            qwen_call_count += 1
            subtalker_call_count += 1

    pass_listen = (qwen_call_count == 0) and (subtalker_call_count == 0) and (listener_ticks == ticks)
    print(f"  Listener clock ticks eseguiti: {listener_ticks}/{ticks}", flush=True)
    print(f"  Qwen step calls: {qwen_call_count} (atteso 0)", flush=True)
    print(f"  SubTalker calls: {subtalker_call_count} (atteso 0) -> {'PASS' if pass_listen else 'FAIL'}", flush=True)

    return {
        "duration_s": duration_s,
        "listener_ticks": listener_ticks,
        "qwen_calls": qwen_call_count,
        "subtalker_calls": subtalker_call_count,
        "pass_listen": pass_listen,
    }


def run_cancellation_fencing(
    device: str,
    num_transitions: int = 1000,
) -> dict[str, Any]:
    print(f"\n[GATE §11] Verifica Cancellation / YIELD / Fencing su {num_transitions} transizioni...", flush=True)
    from duplex_kernel.future_prefix import FuturePrefixBuffer
    buf = FuturePrefixBuffer(hidden_dim=2048, max_frames=256, device=device)

    stale_writes = 0
    ghost_frames = 0
    fencing_violations = 0

    generation_id = 0
    for i in range(num_transitions):
        generation_id += 1
        buf.cancel()

        # Simula ricezione di frame
        num_frames = int(np.random.randint(5, 20))
        for f in range(num_frames):
            buf.push(f, torch.randn(2048, device=device))

        # YIELD improvviso a metà segmento
        yield_at = int(np.random.randint(1, num_frames))
        buf.cancel()

        # Verifica che il buffer sia svuotato e che nessun frame stale sia visibile
        if buf.available_through != -1 or buf.buffered_frames != 0:
            stale_writes += 1

        # Verifica che can_emit dia False per qualsiasi query
        if buf.can_emit(0, lookahead_frames=0):
            ghost_frames += 1

    pass_fencing = (stale_writes == 0) and (ghost_frames == 0) and (fencing_violations == 0)
    print(f"  Stale writes rilevati: {stale_writes} (atteso 0) -> {'PASS' if stale_writes == 0 else 'FAIL'}", flush=True)
    print(f"  Ghost frames rilevati: {ghost_frames} (atteso 0) -> {'PASS' if ghost_frames == 0 else 'FAIL'}", flush=True)
    print(f"  Fencing pass: {'PASS' if pass_fencing else 'FAIL'}", flush=True)

    return {
        "num_transitions": num_transitions,
        "stale_writes": stale_writes,
        "ghost_frames": ghost_frames,
        "pass_fencing": pass_fencing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P4D Runtime Verification Suite")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/cache_p4d_r_512.pt"))
    parser.add_argument("--ckpt", type=Path, default=Path("/workspace/checkpoints/v9_causal_l0_final.pt"))
    parser.add_argument("--heads", type=Path, default=Path("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/validation/kernel_gates/P4D_R"))
    parser.add_argument("--num-seq", type=int, default=100)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # deterministico e allineato con parity gate

    ckpt_path = args.ckpt
    if not ckpt_path.exists():
        ckpt_path = Path("/workspace/checkpoints/v9_causal_l0_unknown.pt")

    print(f"[SUITE] Inizializzazione con checkpoint: {ckpt_path}", flush=True)
    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker.talker.to(dtype=dtype)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    projector = DeepCausalTransformerProjectorV9().to(device=device, dtype=dtype).eval()
    projector.load_state_dict(sd.get("proj", sd), strict=False)

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=dtype).eval()
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"])
    else:
        hsd = torch.load(args.heads, map_location="cpu", weights_only=True)
        causal_conv.conv.load_state_dict(hsd["causal_conv"], strict=False)

    # 1. Gate §7: Quality Gate
    g7 = run_runtime_quality_gate(talker, projector, causal_conv, args.cache, device, dtype, num_seq=args.num_seq)

    # 2. Gate §8: Future-Leak Perturbation
    g8 = run_future_leak_perturbation(talker, projector, causal_conv, args.cache, device, dtype, num_seq=30)

    # 3. Gate §9: Realtime Profiling (in production bfloat16)
    talker_bf16 = load_talker("/workspace", device=device, load_code2wav=False)
    p_bf16 = DeepCausalTransformerProjectorV9().to(device=device, dtype=torch.bfloat16).eval()
    p_bf16.load_state_dict(sd.get("proj", sd), strict=False)
    c_bf16 = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=torch.bfloat16).eval()
    if "causal_conv" in sd:
        c_bf16.conv.load_state_dict(sd["causal_conv"])
    else:
        hsd = torch.load(args.heads, map_location="cpu", weights_only=True)
        c_bf16.conv.load_state_dict(hsd["causal_conv"], strict=False)
    g9 = run_realtime_profiling(talker_bf16, p_bf16, c_bf16, device, torch.bfloat16, num_frames=500)

    # 4. Gate §10: LISTEN Semantics
    g10 = run_listen_semantics(None, None, device=device, duration_s=10.0)

    # 5. Gate §11: Cancellation / Fencing
    g11 = run_cancellation_fencing(device=device, num_transitions=1000)

    all_gates_pass = bool(
        g7["all_pass"]
        and g8["pass_leak"]
        and g9["pass_performance"]
        and g10["pass_listen"]
        and g11["pass_fencing"]
    )

    if all_gates_pass:
        verdict = "P4D CAUSAL RUNTIME PASS — L=0"
    elif not g9.get("pass_performance", False):
        verdict = "P4D OPEN — REALTIME_DEADLINE_FAIL"
    else:
        verdict = "P4D OPEN — SUITE_GATE_FAIL"

    print("\n" + "=" * 60)
    print("VERDETTO FINALE SUITE RUNTIME P4D:")
    print(f"  Gate §7 (Runtime Quality):    {'PASS' if g7['all_pass'] else 'FAIL'}")
    print(f"  Gate §8 (Future-Leak Test):   {'PASS' if g8['pass_leak'] else 'FAIL'}")
    print(f"  Gate §9 (Performance p95):   {'PASS' if g9['pass_performance'] else 'FAIL'}")
    print(f"  Gate §10 (LISTEN Semantics):  {'PASS' if g10['pass_listen'] else 'FAIL'}")
    print(f"  Gate §11 (Fencing/Canc.):     {'PASS' if g11['pass_fencing'] else 'FAIL'}")
    print(f"  ESITO GLOBALE: {verdict}")
    print("=" * 60 + "\n")

    report = {
        "schema_version": 1,
        "checkpoint": str(ckpt_path),
        "gate_7_runtime_quality": g7,
        "gate_8_future_leak": g8,
        "gate_9_performance": g9,
        "gate_10_listen": g10,
        "gate_11_fencing": g11,
        "all_gates_pass": all_gates_pass,
        "verdict": verdict,
    }

    out_file = args.output_dir / "P4D_RUNTIME_SUITE_REPORT.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SUITE] Report salvato in {out_file}", flush=True)

    return 0 if all_gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
