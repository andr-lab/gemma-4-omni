#!/usr/bin/env python3
"""Measure native-audio Thinker parity for 4096 versus the selected cache bucket.

Run only in the project's CUDA container after the engine's normal graph warmup.
The report contains model/cache statistics, never transcripts or generated text.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.engine import PipelineEngine, select_audio_thinker_cache_bucket
from duplex_kernel.server import _prepare_direct_audio_processor_inputs


def _read_user_pcm(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frames = wav.getnframes()
        raw = wav.readframes(frames)
    if sample_rate != 16_000 or sample_width != 2 or channels not in (1, 2):
        raise ValueError("input WAV must be PCM16 at 16 kHz with one or two channels")
    interleaved = np.frombuffer(raw, dtype="<i2")
    if channels == 2:
        # The V5 source format stores the user on physical channel 0 and the
        # assistant on channel 1. Runtime Thinker receives only the user PCM.
        interleaved = interleaved.reshape(-1, 2)[:, 0]
    pcm = interleaved.astype(np.float32) / 32768.0
    if not pcm.size or not np.isfinite(pcm).all():
        raise ValueError("input WAV contains no finite user PCM")
    return pcm, {
        "sample_rate_hz": sample_rate,
        "source_channels": channels,
        "selected_channel": 0,
        "samples": int(pcm.size),
        "duration_s": round(pcm.size / sample_rate, 4),
    }


def _cache_length(cache: Any) -> int | None:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        return None
    try:
        return int(getter())
    except TypeError:
        return int(getter(0))


def _cache_capacity(cache: Any) -> int | None:
    value = getattr(cache, "max_cache_len", None)
    if isinstance(value, int):
        return value
    getter = getattr(cache, "get_max_cache_shape", None)
    if callable(getter):
        shape = getter()
        if isinstance(shape, int):
            return shape
        if isinstance(shape, (tuple, list, torch.Size)) and shape:
            return int(shape[-2] if len(shape) > 1 else shape[-1])
    layers = getattr(cache, "layers", None)
    if layers:
        keys = getattr(layers[0], "keys", None)
        if isinstance(keys, torch.Tensor) and keys.ndim >= 2:
            return int(keys.shape[-2])
    return None


def _shared_cache_keys(cache: Any) -> list[int] | None:
    shared = getattr(cache, "shared_layers", None)
    if not isinstance(shared, dict):
        return None
    return sorted(int(key) for key in shared)


def _top_ids(logits: torch.Tensor, count: int = 5) -> list[int]:
    return [int(value) for value in torch.topk(logits.float(), k=count, dim=-1).indices[0].tolist()]


def _shape_info(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, dict):
        return {str(key): _shape_info(item) for key, item in value.items()}
    return None if value is None else type(value).__name__


def _run_capacity(
    engine: PipelineEngine,
    *,
    prompt_ids: list[int],
    audio_inputs: dict[str, torch.Tensor],
    protocol_enabled: bool,
    max_new_tokens: int,
    generated_steps: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    original_compiled = engine._compiled_thinker
    original_prefill = engine._run_thinker_audio_prefill
    prior_protocol = getattr(engine, "audio_response_protocol", None)
    prior_reuse = getattr(engine, "audio_text_prefix_cache_reuse_enabled", False)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2 if protocol_enabled else None
    engine.audio_text_prefix_cache_reuse_enabled = False

    def record(kind: str, output: Any, kwargs: dict[str, Any], elapsed_ms: float) -> None:
        logits = getattr(output, "logits", None)
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(logits, torch.Tensor) or not hidden_states:
            raise RuntimeError(f"{kind} omitted logits or hidden states")
        cache = kwargs.get("past_key_values")
        positions = kwargs.get("cache_position")
        records.append({
            "kind": kind,
            "elapsed_ms": elapsed_ms,
            "_logits": logits[:, -1, :].detach().float().cpu(),
            "_hidden": hidden_states[-1][:, -1, :].detach().float().cpu(),
            "cache_len_before": (
                int(positions.reshape(-1)[0].item())
                if isinstance(positions, torch.Tensor) and positions.numel()
                else _cache_length(cache)
            ),
            "cache_len_after": _cache_length(cache),
            "cache_capacity": _cache_capacity(cache),
            "cache_position": None if positions is None else [int(x) for x in positions.detach().cpu().tolist()],
            "shared_cache_keys": _shared_cache_keys(cache),
            "input_ids_shape": list(kwargs["input_ids"].shape) if isinstance(kwargs.get("input_ids"), torch.Tensor) else None,
            "attention_mask_shape": _shape_info(kwargs.get("attention_mask")),
        })

    def prefill_call(prepared_audio: dict[str, torch.Tensor], **kwargs: Any) -> Any:
        started = time.perf_counter()
        result = original_prefill(prepared_audio, **kwargs)
        torch.cuda.synchronize(engine.device)
        elapsed = (time.perf_counter() - started) * 1000.0
        output, _hidden, cache, mask = result
        prefix_len = int(prepared_audio["input_ids"].shape[1])
        position = torch.arange(prefix_len, device=prepared_audio["input_ids"].device)
        record("audio_prefill", output, {
            "input_ids": prepared_audio["input_ids"],
            "attention_mask": mask,
            "past_key_values": cache,
            "cache_position": position,
        }, elapsed)
        records[-1]["cache_len_before"] = int(kwargs.get("past_length", 0))
        return result

    def compiled_call(**kwargs: Any) -> Any:
        cache = kwargs.get("past_key_values")
        before = _cache_length(cache)
        started = time.perf_counter()
        output = original_compiled(**kwargs)
        torch.cuda.synchronize(engine.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        record("q1", output, kwargs, elapsed_ms)
        records[-1]["cache_len_before"] = before
        records[-1]["cache_len_after"] = _cache_length(cache)
        return output

    engine._run_thinker_audio_prefill = prefill_call
    engine._compiled_thinker = compiled_call
    generated = 0

    def should_stop() -> bool:
        return generated >= generated_steps

    def observe_token(_index: int, _token_id: int, _text: str, _hidden: torch.Tensor) -> None:
        nonlocal generated
        generated += 1

    allocated_before = torch.cuda.memory_allocated(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    started = time.perf_counter()
    try:
        result = engine.think(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            min_new_tokens=generated_steps,
            temperature=0.0,
            top_p=1.0,
            stop_check=should_stop,
            input_audio=audio_inputs,
            on_generated_token=observe_token,
        )
    finally:
        engine._run_thinker_audio_prefill = original_prefill
        engine._compiled_thinker = original_compiled
        engine.audio_response_protocol = prior_protocol
        engine.audio_text_prefix_cache_reuse_enabled = prior_reuse
    torch.cuda.synchronize(engine.device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    generated_ids = [int(token) for token in result.get("gen_ids", [])]
    if len(records) != generated_steps + 1:
        raise RuntimeError(
            f"expected one multimodal prefill plus {generated_steps} Q1 forwards; "
            f"observed {len(records)} records and {len(generated_ids)} generated tokens"
        )
    for row in records:
        row["top5_ids"] = _top_ids(row["_logits"])
        row["hidden_rms"] = round(float(torch.mean(row["_hidden"].square()).sqrt().item()), 7)
    return {
        "cache_mode": "v2_bucket" if protocol_enabled else "4096_reference",
        "selected_capacity": result.get("runtime_metrics", {}).get("thinker_cache_bucket_capacity"),
        "generated_token_count": len(generated_ids),
        "generated_token_ids": generated_ids,
        "terminal_reason": result.get("terminal_reason"),
        "runtime_metrics": result.get("runtime_metrics", {}),
        "calls": records,
        "wall_ms": round(wall_ms, 3),
        "memory_allocated_before_bytes": allocated_before,
        "memory_allocated_after_bytes": torch.cuda.memory_allocated(engine.device),
        "memory_reserved_after_bytes": torch.cuda.memory_reserved(engine.device),
        "memory_peak_delta_bytes": max(0, torch.cuda.max_memory_allocated(engine.device) - allocated_before),
    }


def _compare(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparisons = []
    for left, right in zip(reference["calls"], candidate["calls"], strict=True):
        logits_left, logits_right = left.pop("_logits"), right.pop("_logits")
        hidden_left, hidden_right = left.pop("_hidden"), right.pop("_hidden")
        logits_delta = (logits_left - logits_right).abs()
        hidden_delta = (hidden_left - hidden_right).abs()
        comparisons.append({
            "kind": left["kind"],
            "cache_position_equal": left["cache_position"] == right["cache_position"],
            "cache_length_before_equal": left["cache_len_before"] == right["cache_len_before"],
            "cache_length_after_equal": left["cache_len_after"] == right["cache_len_after"],
            "reference_capacity": left["cache_capacity"],
            "candidate_capacity": right["cache_capacity"],
            "reference_shared_cache_keys": left["shared_cache_keys"],
            "candidate_shared_cache_keys": right["shared_cache_keys"],
            "logits_shape_equal": list(logits_left.shape) == list(logits_right.shape),
            "logits_max_abs_delta": round(float(logits_delta.max().item()), 7),
            "logits_mean_abs_delta": round(float(logits_delta.mean().item()), 7),
            "hidden_shape_equal": list(hidden_left.shape) == list(hidden_right.shape),
            "hidden_max_abs_delta": round(float(hidden_delta.max().item()), 7),
            "hidden_mean_abs_delta": round(float(hidden_delta.mean().item()), 7),
            "top5_exact": left["top5_ids"] == right["top5_ids"],
            "reference_top1": left["top5_ids"][0],
            "candidate_top1": right["top5_ids"][0],
            "reference_hidden_rms": left["hidden_rms"],
            "candidate_hidden_rms": right["hidden_rms"],
            "reference_q1_ms": left["elapsed_ms"],
            "candidate_q1_ms": right["elapsed_ms"],
        })
    return {
        "calls": comparisons,
        "generated_token_ids_equal": reference["generated_token_ids"] == candidate["generated_token_ids"],
        "reference_generated_token_ids": reference["generated_token_ids"],
        "candidate_generated_token_ids": candidate["generated_token_ids"],
        "reference_wall_ms": reference["wall_ms"],
        "candidate_wall_ms": candidate["wall_ms"],
        "speedup_wall": round(reference["wall_ms"] / candidate["wall_ms"], 4) if candidate["wall_ms"] else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("cache-bucket parity requires CUDA")
    if args.decode_steps < 1 or args.decode_steps >= args.max_new_tokens:
        raise ValueError("decode-steps must be positive and lower than max-new-tokens")

    pcm, audio_meta = _read_user_pcm(args.audio)
    started = time.perf_counter()
    engine = PipelineEngine(
        device="cuda", dtype=torch.bfloat16, runtime_backend="pytorch",
        duplex_response_bundle_dir=str(args.bundle_dir.resolve()),
        thinker_compile_mode="default", thinker_static_cache_max_len=4096,
    )
    if not engine.load(workspace=str(args.workspace.resolve())):
        raise RuntimeError("PipelineEngine failed to load the selected response bundle")
    load_wall_s = time.perf_counter() - started
    if engine._compiled_thinker is None:
        raise RuntimeError("compiled Thinker was not initialized")
    ids, audio_inputs, prepare_ms = _prepare_direct_audio_processor_inputs(
        engine,
        language="it",
        history_messages=[],
        pcm=pcm,
        response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
    )
    if len(ids) + args.max_new_tokens > 4096:
        raise ValueError("actual audio request exceeds the supported 4096-token cap")
    chosen = select_audio_thinker_cache_bucket(
        len(ids), args.max_new_tokens, max_cache_len=4096
    )
    reference = _run_capacity(
        engine, prompt_ids=ids, audio_inputs=audio_inputs,
        protocol_enabled=False, max_new_tokens=args.max_new_tokens,
        generated_steps=args.decode_steps,
    )
    candidate = _run_capacity(
        engine, prompt_ids=ids, audio_inputs=audio_inputs,
        protocol_enabled=True, max_new_tokens=args.max_new_tokens,
        generated_steps=args.decode_steps,
    )
    result = {
        "status": "observed",
        "bundle_id": getattr(engine, "duplex_control6_runtime", {}).get("bundle_id"),
        "base_model_path": getattr(engine._thinker_wrapper.model, "_gemma4_checkpoint_path", None),
        "dtype": str(engine.dtype),
        "compile_mode": engine.thinker_compile_mode,
        "audio": audio_meta,
        "prompt_tokens": len(ids),
        "max_new_tokens_budget": args.max_new_tokens,
        "selected_bucket": chosen,
        "processor_prepare_ms": round(float(prepare_ms), 3),
        "engine_load_and_graph_warmup_s": round(load_wall_s, 3),
        "memory_allocated_after_warmup_bytes": torch.cuda.memory_allocated(engine.device),
        "memory_reserved_after_warmup_bytes": torch.cuda.memory_reserved(engine.device),
        "reference": reference,
        "candidate": candidate,
        "comparison": _compare(reference, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "bundle_id": result["bundle_id"],
        "prompt_tokens": len(ids),
        "selected_bucket": chosen,
        "audio_duration_s": audio_meta["duration_s"],
        "load_and_warmup_s": round(load_wall_s, 3),
        "comparison": result["comparison"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
