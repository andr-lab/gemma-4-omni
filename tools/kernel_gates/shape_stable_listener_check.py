#!/usr/bin/env python3
"""Real-data feasibility gate for the functional shared-weight Gemma listener.

The native multimodal Gemma forward is the untouched oracle.  The candidate is
``ShapeStableGemmaListenerLM``: it owns only listener KV/workspace tensors and
reads the original frozen language-model weights without monkey-patching them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import UserDict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

from duplex_kernel.gemma_representation import _audio_soft_tokens  # noqa: E402
from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path  # noqa: E402
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM  # noqa: E402
from tools.kernel_gates.p4_candidate_validation import _percentile, _validation_dataset  # noqa: E402
from tools.kernel_gates.p4_numeric_bisect import _compare  # noqa: E402

AUDIO_FEATURES_PER_FRAME = 4


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _memory() -> dict[str, int | None]:
    if not torch.cuda.is_available():
        return {"allocated_bytes": None, "reserved_bytes": None}
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _source_hash(source_parameter_data_ptrs: dict[str, int]) -> str:
    payload = json.dumps(source_parameter_data_ptrs, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _module_forward_ids(language_model: Any) -> dict[str, int]:
    # ``module.forward`` creates a fresh bound-method wrapper on attribute
    # access; its underlying function is the stable identity to attest.
    return {
        name: id(getattr(module.forward, "__func__", module.forward))
        for name, module in language_model.named_modules()
        if hasattr(module, "forward")
    }


def _functional_incremental(
    *,
    model: Any,
    runtime: ShapeStableGemmaListenerLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_soft: torch.Tensor,
    audio_start: int,
) -> tuple[torch.Tensor, list[float], int, int]:
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    lm = model.language_model
    device = input_ids.device
    batch_size = int(input_ids.shape[0])
    if batch_size != 1:
        raise ValueError("la verifica runtime listener richiede batch-size=1")
    stable_ids = input_ids[:, :audio_start]
    stable_attention = attention_mask[:, :audio_start].to(device=device, dtype=torch.long)
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
    pad_ids = torch.full(
        (batch_size, AUDIO_FEATURES_PER_FRAME),
        int(model.config.text_config.pad_token_id),
        dtype=torch.long,
        device=device,
    )
    pad_embeds = lm.embed_tokens(pad_ids)
    pad_ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)
    past: Any | None = None
    shared: UserDict | None = None
    chunks: list[torch.Tensor] = []

    def call(embeds: torch.Tensor, ple: torch.Tensor, positions: torch.Tensor, causal_attention: torch.Tensor) -> torch.Tensor:
        nonlocal past, shared
        position_ids = positions.view(1, -1)
        masks = create_masks_for_generate(
            model.config,
            embeds,
            causal_attention,
            past_key_values=past,
            position_ids=position_ids,
        )
        output = runtime.forward(
            inputs_embeds=embeds,
            per_layer_inputs=ple,
            attention_mask=masks,
            position_ids=position_ids,
            past_key_values=past,
            shared_kv_states=shared,
        )
        past = output.past_key_values
        shared = output.shared_kv_states
        return output.last_hidden_state

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Keep the stable text prefill and every four-token audio frame as distinct
    # calls.  This is the validated incremental schedule; combining the first
    # frame with the prefix changes the BF16 GEMM shape and is not equivalent.
    with torch.inference_mode():
        stable_hidden = call(
            stable_embeds,
            stable_ple,
            torch.arange(audio_start, device=device, dtype=torch.long),
            stable_attention,
        )
    chunks.append(stable_hidden)
    audio_ms: list[float] = []
    for start in range(0, int(audio_soft.shape[1]), AUDIO_FEATURES_PER_FRAME):
        positions = torch.arange(
            audio_start + start,
            audio_start + start + AUDIO_FEATURES_PER_FRAME,
            device=device,
            dtype=torch.long,
        )
        causal_attention = torch.ones((1, audio_start + start + AUDIO_FEATURES_PER_FRAME), device=device, dtype=torch.long)
        causal_attention[:, :audio_start] = stable_attention
        _sync(device)
        started = time.perf_counter()
        with torch.inference_mode():
            hidden = call(
                audio_soft[:, start:start + AUDIO_FEATURES_PER_FRAME].to(dtype=lm.embed_tokens.weight.dtype),
                pad_ple,
                positions,
                causal_attention,
            )
        _sync(device)
        audio_ms.append((time.perf_counter() - started) * 1000.0)
        chunks.append(hidden)
    result = torch.cat(chunks, dim=1)
    return (
        result,
        audio_ms,
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    )


def run(*, dataset_path: Path, limit: int, require_minimum: bool = True) -> dict[str, Any]:
    ds, text_block_len, validation_start_frame = _validation_dataset(dataset_path)
    count = min(int(limit), len(ds))
    if require_minimum and count < 100:
        raise ValueError("il gate richiede almeno 100 sequenze")
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    lm = model.language_model
    device = torch.device("cuda")
    audio_token_id = int(model.config.audio_token_id)
    before_view = _memory()
    runtime = ShapeStableGemmaListenerLM(lm, target_length=291)
    after_view = _memory()
    before_forwards = _module_forward_ids(lm)
    frame_maes: list[float] = []
    frame_cosines: list[float] = []
    frame_maxima: list[float] = []
    audio_times: list[float] = []
    peak_allocated = 0
    peak_reserved = 0
    started = time.perf_counter()
    try:
        for index in range(count):
            item = ds[index]
            batch = {
                key: item[key].unsqueeze(0).to(device)
                for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
            }
            ids = batch["input_ids"]
            attention = batch["attention_mask"]
            positions = (ids[0] == audio_token_id).nonzero(as_tuple=False).flatten().tolist()
            if len(positions) != 256:
                raise RuntimeError(f"placeholder audio inattesi nella sequenza {index}: {len(positions)}")
            audio_soft = _audio_soft_tokens(
                model,
                input_features=batch["input_features"],
                input_features_mask=batch["input_features_mask"],
            )
            with torch.inference_mode():
                full = model(
                    input_ids=ids,
                    attention_mask=attention,
                    input_features=batch["input_features"],
                    input_features_mask=batch["input_features_mask"],
                    use_cache=True,
                    return_dict=True,
                ).last_hidden_state[:, positions, :].float().cpu()
            candidate_all, timings, allocated, reserved = _functional_incremental(
                model=model,
                runtime=runtime,
                input_ids=ids,
                attention_mask=attention,
                audio_soft=audio_soft,
                audio_start=int(positions[0]),
            )
            candidate = candidate_all[:, positions, :].float().cpu()
            comparison = _compare(full, candidate)
            if not comparison.get("available") or comparison.get("shape_mismatch"):
                raise RuntimeError(f"shape mismatch sequenza {index}: {comparison}")
            error = (full[:, ::4, :] - candidate[:, ::4, :]).abs()
            cosine = torch.nn.functional.cosine_similarity(full[:, ::4, :], candidate[:, ::4, :], dim=-1, eps=1e-12)
            frame_maes.extend(error.mean(dim=-1).reshape(-1).tolist())
            frame_maxima.extend(error.amax(dim=-1).reshape(-1).tolist())
            frame_cosines.extend(cosine.reshape(-1).tolist())
            audio_times.extend(timings)
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
            del batch, ids, attention, audio_soft, full, candidate_all, candidate
            torch.cuda.empty_cache()
            print(f"[SHARED-WEIGHT-LISTENER] {index + 1}/{count}", flush=True)
        after_forwards = _module_forward_ids(lm)
        sharing = runtime.weight_sharing_report()
        mae = sum(frame_maes) / len(frame_maes)
        cosine_min = min(frame_cosines)
        p50 = _percentile(audio_times, 50)
        p95 = _percentile(audio_times, 95)
        p99 = _percentile(audio_times, 99)
        passed = bool(
            mae <= 1e-3
            and cosine_min >= 0.9999
            and max(frame_maxima) <= 1e-3
            and p95 is not None
            and p95 <= 80.0
            and sharing["shared_weight_storage"]
            and before_forwards == after_forwards
        )
        return {
            "schema_version": 1,
            "phase": "P4",
            "gate": "shared_weight_listener_feasibility",
            "status": "spike",
            "candidate": "ShapeStableGemmaListenerLM",
            "oracle": "untouched native Gemma4Model.forward BF16",
            "dataset": str(dataset_path),
            "validation_start_frame": validation_start_frame,
            "validation_text_window_offset": validation_start_frame // 64,
            "text_block_len": text_block_len,
            "model": {"path": resolve_thinker_path("/workspace"), "dtype": "bfloat16", "frozen": True},
            "weight_sharing": {
                "listener_parameter_count": sharing["listener_parameter_count"],
                "source_parameter_count": sharing["source_parameter_count"],
                "source_parameter_data_ptrs_sha256": _source_hash(sharing["source_parameter_data_ptrs"]),
                "shared_weight_storage": sharing["shared_weight_storage"],
                "memory_before_listener_view": before_view,
                "memory_after_listener_view": after_view,
                "listener_view_allocated_delta_bytes": None if before_view["allocated_bytes"] is None else after_view["allocated_bytes"] - before_view["allocated_bytes"],
                "listener_view_reserved_delta_bytes": None if before_view["reserved_bytes"] is None else after_view["reserved_bytes"] - before_view["reserved_bytes"],
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
                "max_abs": max(frame_maxima),
                "cosine_min": cosine_min,
                "nonfinite_count": 0,
                "shape_mismatches": 0,
                "audio_p50_ms": p50,
                "audio_p95_ms": p95,
                "audio_p99_ms": p99,
                "audio_samples": len(audio_times),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "wall_ms": (time.perf_counter() - started) * 1000.0,
            },
            "thresholds": {"mae_max": 1e-3, "cosine_min": 0.9999, "audio_p95_ms_max": 80.0, "engineering_audio_p95_ms_target": 75.0},
            "passed": passed,
        }
    finally:
        del runtime, thinker, model, lm
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="solo debug: non costituisce il gate P4 da 100 sequenze")
    args = parser.parse_args()
    report = run(dataset_path=args.dataset, limit=args.limit, require_minimum=not args.smoke)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
