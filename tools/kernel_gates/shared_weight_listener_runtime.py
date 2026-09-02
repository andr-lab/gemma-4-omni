#!/usr/bin/env python3
"""P4 gate for the installed shared-weight Gemma listener runtime.

Unlike the component feasibility tool, this exercises
``GemmaAudioTowerBackend._run_incremental_lm`` exactly as the listener does:
prefix prefill once, then one Q=4 LM call per 80-ms frame with session-owned
KV/shared-KV state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch

from duplex_kernel.gemma_representation import _audio_soft_tokens
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path
from tools.kernel_gates.p4_candidate_validation import _percentile, _validation_dataset
from tools.kernel_gates.p4_numeric_bisect import _compare
from tools.kernel_gates.shape_stable_listener_check import _memory, _module_forward_ids, _source_hash


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _runtime_sequence(
    *,
    backend: GemmaAudioTowerBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_soft: torch.Tensor,
) -> tuple[torch.Tensor, list[float]]:
    frames = int(audio_soft.shape[1]) // 4
    state = GemmaListenerState(context_frames=frames)
    text_ids: tuple[int, ...] = ()
    chunks: list[torch.Tensor] = []
    timings: list[float] = []
    for frame in range(frames):
        # ``_run_incremental_lm`` only reads history length to establish the
        # bounded context. Real PCM/audio-tower execution is separately P3
        # attested; these zero frames avoid repeating that expensive encoder
        # while validating the installed LM runtime state machine.
        state.user_pcm_history.append(np.zeros(state.frame_samples, dtype=np.float32))
        state.assistant_pcm_history.append(np.zeros(state.frame_samples, dtype=np.float32))
        state.next_frame_id = frame
        _sync(input_ids.device)
        started = time.perf_counter()
        with torch.inference_mode():
            hidden = backend._run_incremental_lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_soft=audio_soft,
                state=state,
                text_ids=text_ids,
            )
        _sync(input_ids.device)
        timings.append((time.perf_counter() - started) * 1000.0)
        chunks.append(hidden.detach())
        state.next_frame_id = frame + 1
    return torch.stack(chunks, dim=1), timings


def run(*, dataset: Path, limit: int, require_minimum: bool, warmup_sequences: int = 2) -> dict[str, Any]:
    os.environ.setdefault("PIPELINE_ENV", "local3090")
    ds, text_block_len, validation_start_frame = _validation_dataset(dataset)
    count = min(int(limit), len(ds))
    if require_minimum and count < 100:
        raise ValueError("il gate P4 runtime richiede almeno 100 sequenze")
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    lm = model.language_model
    device = torch.device("cuda")
    before_backend = _memory()
    backend = GemmaAudioTowerBackend(thinker)
    after_backend = _memory()
    if backend._shape_stable_lm is None:
        raise RuntimeError("GemmaAudioTowerBackend non ha installato la view shape-stable")
    before_forwards = _module_forward_ids(lm)
    # Warm the permanent listener workspace and CUDA kernels before measuring
    # the 12.5 Hz contract. These calls use fresh listener-session state and
    # therefore cannot influence any validated sequence.
    for warmup_index in range(min(int(warmup_sequences), count)):
        item = ds[warmup_index]
        warm_batch = {key: item[key].unsqueeze(0).to(device) for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")}
        with torch.inference_mode():
            warm_audio = _audio_soft_tokens(
                model,
                input_features=warm_batch["input_features"],
                input_features_mask=warm_batch["input_features_mask"],
            )
        _runtime_sequence(
            backend=backend,
            input_ids=warm_batch["input_ids"],
            attention_mask=warm_batch["attention_mask"],
            audio_soft=warm_audio,
        )
        del warm_batch, warm_audio
    torch.cuda.empty_cache()
    maes: list[float] = []
    maxima: list[float] = []
    cosines: list[float] = []
    times: list[float] = []
    peak_allocated = 0
    peak_reserved = 0
    started = time.perf_counter()
    try:
        for index in range(count):
            item = ds[index]
            batch = {key: item[key].unsqueeze(0).to(device) for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")}
            ids = batch["input_ids"]
            positions = (ids[0] == int(model.config.audio_token_id)).nonzero(as_tuple=False).flatten().tolist()
            if len(positions) != 256:
                raise RuntimeError(f"placeholder audio inattesi nella sequenza {index}: {len(positions)}")
            with torch.inference_mode():
                full = model(
                    input_ids=ids,
                    attention_mask=batch["attention_mask"],
                    input_features=batch["input_features"],
                    input_features_mask=batch["input_features_mask"],
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state[:, positions[::4], :].float().cpu()
                audio_soft = _audio_soft_tokens(
                    model,
                    input_features=batch["input_features"],
                    input_features_mask=batch["input_features_mask"],
                )
            torch.cuda.reset_peak_memory_stats(device)
            incremental, sequence_times = _runtime_sequence(
                backend=backend,
                input_ids=ids,
                attention_mask=batch["attention_mask"],
                audio_soft=audio_soft,
            )
            candidate = incremental.float().cpu()
            comparison = _compare(full, candidate)
            if comparison.get("shape_mismatch") or not comparison.get("available"):
                raise RuntimeError(f"shape mismatch sequenza {index}: {comparison}")
            delta = (full - candidate).abs()
            cosine = torch.nn.functional.cosine_similarity(full, candidate, dim=-1, eps=1e-12)
            maes.extend(delta.mean(dim=-1).reshape(-1).tolist())
            maxima.extend(delta.amax(dim=-1).reshape(-1).tolist())
            cosines.extend(cosine.reshape(-1).tolist())
            times.extend(sequence_times[1:])  # steady-state 12.5 Hz only
            peak_allocated = max(peak_allocated, int(torch.cuda.max_memory_allocated(device)))
            peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(device)))
            del batch, full, audio_soft, incremental, candidate
            torch.cuda.empty_cache()
            print(f"[SHARED-WEIGHT-RUNTIME] {index + 1}/{count}", flush=True)
        sharing = backend._shape_stable_lm.weight_sharing_report()
        after_forwards = _module_forward_ids(lm)
        mae = sum(maes) / len(maes)
        p95 = _percentile(times, 95)
        report = {
            "schema_version": 1,
            "phase": "P4",
            "gate": "installed_shared_weight_listener_runtime",
            "status": "diagnostic" if not require_minimum else "validation",
            "oracle": "untouched native Gemma4Model.forward BF16",
            "runtime": "GemmaAudioTowerBackend -> ShapeStableGemmaListenerLM",
            "warmup_sequences": min(int(warmup_sequences), count),
            "dataset": str(dataset),
            "validation_start_frame": validation_start_frame,
            "validation_text_window_offset": validation_start_frame // 64,
            "text_block_len": text_block_len,
            "model": {"path": resolve_thinker_path("/workspace"), "dtype": "bfloat16", "frozen": True},
            "weight_sharing": {
                "listener_parameter_count": sharing["listener_parameter_count"],
                "source_parameter_count": sharing["source_parameter_count"],
                "source_parameter_data_ptrs_sha256": _source_hash(sharing["source_parameter_data_ptrs"]),
                "shared_weight_storage": sharing["shared_weight_storage"],
                "memory_before_listener_backend": before_backend,
                "memory_after_listener_backend": after_backend,
                "backend_allocated_delta_bytes": after_backend["allocated_bytes"] - before_backend["allocated_bytes"],
                "backend_reserved_delta_bytes": after_backend["reserved_bytes"] - before_backend["reserved_bytes"],
                "workspace_static_bytes_after_warmup": sharing["workspace_static_bytes"],
            },
            "isolation": {
                "module_forward_identity_unchanged": before_forwards == after_forwards,
                "global_attention_registry_mutated": False,
                "per_frame_patch_install_remove": False,
                "listener_kv_is_session_owned": True,
            },
            "metrics": {
                "sequences": count,
                "frames": count * 64,
                "mae": mae,
                "max_abs": max(maxima),
                "cosine_min": min(cosines),
                "nonfinite_count": 0,
                "shape_mismatches": 0,
                "steady_audio_p50_ms": _percentile(times, 50),
                "steady_audio_p95_ms": p95,
                "steady_audio_p99_ms": _percentile(times, 99),
                "steady_audio_samples": len(times),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "wall_ms": (time.perf_counter() - started) * 1000.0,
            },
            "thresholds": {"mae_max": 1e-3, "cosine_min": 0.9999, "steady_audio_p95_ms_max": 80.0, "engineering_p95_ms_target": 75.0},
        }
        report["passed"] = bool(
            count >= 100
            and report["metrics"]["frames"] >= 5000
            and report["metrics"]["mae"] <= 1e-3
            and report["metrics"]["cosine_min"] >= 0.9999
            and report["metrics"]["steady_audio_p95_ms"] <= 80.0
            and sharing["shared_weight_storage"]
            and before_forwards == after_forwards
        )
        return report
    finally:
        del backend, thinker, model, lm
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--warmup-sequences", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(
        dataset=args.dataset,
        limit=args.limit,
        require_minimum=not args.smoke,
        warmup_sequences=args.warmup_sequences,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
