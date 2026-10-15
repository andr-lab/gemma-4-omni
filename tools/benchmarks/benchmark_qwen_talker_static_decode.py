"""Manual Qwen Talker DynamicCache/StaticCache parity and Q1 compile probe.

Run only in the repository's offline 3090 container after stopping the active
serving container.  This is an isolated proof tool; it does not alter the
production engine or enable StaticCache in Hugging Face ``generate``.

The probe loads the pinned, cached Qwen3-TTS Base Talker, rebuilds the entire
``[assistant/codec ICL | SAY text | V9 prefix | codec BOS]`` input eagerly into
each fresh cache, then compares three one-frame decode calls.  Its historical
V9 prefix fixture is a real saved projection, but is not claimed to be the
current F run's SAY hidden; the result proves only the manual cache path.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_CAPACITIES = (512, 1024)
DEFAULT_ATOL = 2e-2
DEFAULT_RTOL = 2e-2


def select_cache_capacity(
    *,
    prefill_tokens: int,
    decode_steps: int,
    capacities: Iterable[int] = DEFAULT_CAPACITIES,
) -> int:
    """Select the smallest bucket covering eager prefix plus every decode."""
    if isinstance(prefill_tokens, bool) or int(prefill_tokens) <= 0:
        raise ValueError("prefill_tokens must be a positive integer")
    if isinstance(decode_steps, bool) or int(decode_steps) < 0:
        raise ValueError("decode_steps must be a non-negative integer")
    buckets = tuple(sorted({int(item) for item in capacities}))
    if not buckets or any(item <= 0 for item in buckets):
        raise ValueError("static cache buckets must be positive integers")
    required = int(prefill_tokens) + int(decode_steps)
    for capacity in buckets:
        if capacity >= required:
            return capacity
    raise ValueError(
        "required Qwen Talker cache length "
        f"{required} exceeds largest static cache bucket {buckets[-1]}"
    )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def compare_tensors(
    reference: Any,
    candidate: Any,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> dict[str, Any]:
    """Return shape, BF16-tolerance, and argmax parity without hiding ties."""
    ref = _as_numpy(reference)
    cand = _as_numpy(candidate)
    shape_equal = tuple(ref.shape) == tuple(cand.shape)
    result: dict[str, Any] = {
        "shape_equal": shape_equal,
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "atol": float(atol),
        "rtol": float(rtol),
        "allclose": False,
        "max_abs_diff": None,
        "reference_argmax": None,
        "candidate_argmax": None,
        "top_id_equal": False,
    }
    if not shape_equal or ref.size == 0:
        return result
    ref64 = np.asarray(ref, dtype=np.float64)
    cand64 = np.asarray(cand, dtype=np.float64)
    finite = bool(np.isfinite(ref64).all() and np.isfinite(cand64).all())
    result["finite"] = finite
    if not finite:
        return result
    result["max_abs_diff"] = float(np.max(np.abs(ref64 - cand64)))
    result["allclose"] = bool(np.allclose(ref64, cand64, atol=atol, rtol=rtol))
    result["reference_argmax"] = int(np.argmax(ref64))
    result["candidate_argmax"] = int(np.argmax(cand64))
    result["top_id_equal"] = result["reference_argmax"] == result["candidate_argmax"]
    return result


def _sync(torch: Any) -> None:
    torch.cuda.synchronize()


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache_len(cache: Any, layer: int = 0) -> int:
    value = cache.get_seq_length(layer)
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def _cache_lengths(cache: Any, layer_count: int) -> dict[str, Any]:
    values = [_cache_len(cache, layer) for layer in range(layer_count)]
    return {
        "per_layer": values,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "all_equal": len(set(values)) <= 1,
    }


def _new_dynamic_cache(torch: Any, config: Any) -> Any:
    from transformers import DynamicCache

    # Match Qwen's own no-past fallback, which is DynamicCache().
    del config
    return DynamicCache()


def _new_static_cache(
    torch: Any,
    config: Any,
    *,
    capacity: int,
    device: Any,
    dtype: Any,
) -> Any:
    from transformers import StaticCache

    return StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=int(capacity),
        device=device,
        dtype=dtype,
    )


def _forward(
    model: Any,
    embeds: Any,
    cache: Any | None,
    *,
    cache_position: Any,
) -> Any:
    return model(
        inputs_embeds=embeds,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
    )


def _prefill(
    torch: Any,
    model: Any,
    prefix: Any,
    cache: Any | None,
    *,
    layer_count: int,
) -> tuple[Any, dict[str, Any]]:
    position = torch.arange(prefix.shape[1], device=prefix.device, dtype=torch.long)
    output = _forward(model, prefix, cache, cache_position=position)
    resulting_cache = output.past_key_values
    lengths = _cache_lengths(resulting_cache, layer_count)
    expected = int(prefix.shape[1])
    if not lengths["all_equal"] or lengths["min"] != expected:
        raise RuntimeError(
            f"prefill cache positions differ from full input: expected {expected}, got {lengths}"
        )
    return output, lengths


def _decode_one(
    torch: Any,
    model: Any,
    cache: Any,
    embeds: Any,
    *,
    position: int,
) -> tuple[Any, Any, dict[str, Any]]:
    cache_position = torch.tensor([int(position)], device=embeds.device, dtype=torch.long)
    output = _forward(model, embeds, cache, cache_position=cache_position)
    return output.last_hidden_state[:, -1:, :], output.past_key_values, {
        "position_input": int(position),
        "cache_len_after": _cache_len(output.past_key_values),
    }


def _load_prefix(npz_path: Path, *, frames: int, device: Any, dtype: Any, torch: Any):
    if npz_path.suffix.lower() == ".npy":
        array = np.asarray(np.load(npz_path, allow_pickle=False), dtype=np.float32)
    else:
        with np.load(npz_path, allow_pickle=False) as source:
            if "h_pfx" not in source.files:
                raise ValueError(f"V9 fixture lacks h_pfx: {npz_path}")
            array = np.asarray(source["h_pfx"], dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[-1] != 2048:
        raise ValueError(f"expected h_pfx shape (1,T,2048), got {array.shape}")
    if frames <= 0 or frames > array.shape[1]:
        raise ValueError(f"prefix frames {frames} outside 1..{array.shape[1]}")
    selected = np.ascontiguousarray(array[:, :frames, :])
    if not np.isfinite(selected).all() or float(np.max(np.abs(selected))) == 0.0:
        raise ValueError("V9 prefix fixture must be finite and nonzero")
    digest = hashlib.sha256(selected.tobytes()).hexdigest()
    tensor = torch.from_numpy(selected).to(device=device, dtype=dtype)
    return tensor, digest


def _deterministic_codes(torch: Any, *, cb0: int, frame: int, device: Any):
    codes = torch.zeros((1, 1, 16), device=device, dtype=torch.long)
    codes[0, 0, 0] = int(cb0) % 2048
    for codebook in range(1, 16):
        codes[0, 0, codebook] = (int(frame) * 37 + codebook * 19) % 2048
    return codes


def _audio_query(torch: Any, qwen: Any, codes: Any, pad_embed: Any):
    from duplex_kernel.qwen_conditioning import build_qwen_audio_conditioning

    return build_qwen_audio_conditioning(
        qwen,
        codes,
        tts_pad_embed=pad_embed,
    )


def _codec_logits(codec_head: Any, hidden: Any):
    return codec_head(hidden)[:, -1, :]


def _record_cache_topology(cache: Any, config: Any) -> dict[str, Any]:
    layers = int(config.num_hidden_layers)
    return {
        "class": type(cache).__name__,
        "capacity": int(getattr(cache, "max_cache_len", -1)),
        "cache_lengths": _cache_lengths(cache, layers),
    }


def _run_parity_bucket(
    torch: Any,
    qwen: Any,
    model: Any,
    codec_head: Any,
    prefix: Any,
    pad_embed: Any,
    *,
    capacity: int,
    decode_steps: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    config = model.config
    layer_count = int(config.num_hidden_layers)
    device = prefix.device
    dtype = prefix.dtype
    required = int(prefix.shape[1]) + int(decode_steps)
    if capacity < required:
        raise ValueError(f"capacity {capacity} cannot hold {required} tokens")

    dynamic_prefill, dynamic_lengths = _prefill(
        torch, model, prefix, None, layer_count=layer_count
    )
    dynamic_cache = dynamic_prefill.past_key_values
    static_cache = _new_static_cache(
        torch,
        config,
        capacity=capacity,
        device=device,
        dtype=dtype,
    )
    static_prefill, static_lengths = _prefill(
        torch, model, prefix, static_cache, layer_count=layer_count
    )
    dynamic_hidden = dynamic_prefill.last_hidden_state[:, -1:, :]
    static_hidden = static_prefill.last_hidden_state[:, -1:, :]
    dynamic_logits = _codec_logits(codec_head, dynamic_hidden)
    static_logits = _codec_logits(codec_head, static_hidden)
    prefill_hidden = compare_tensors(dynamic_hidden, static_hidden, atol=atol, rtol=rtol)
    prefill_logits = compare_tensors(dynamic_logits, static_logits, atol=atol, rtol=rtol)
    prefill_logits["top_id_equal"] = bool(
        int(torch.argmax(dynamic_logits[0, :2048]).item())
        == int(torch.argmax(static_logits[0, :2048]).item())
    )
    prefill_logits["reference_argmax"] = int(torch.argmax(dynamic_logits[0, :2048]).item())
    prefill_logits["candidate_argmax"] = int(torch.argmax(static_logits[0, :2048]).item())

    checks: list[dict[str, Any]] = []
    prev_cb0 = prefill_logits["reference_argmax"]
    pos = int(prefix.shape[1])
    for step in range(int(decode_steps)):
        codes = _deterministic_codes(torch, cb0=prev_cb0, frame=step, device=device)
        query = _audio_query(torch, qwen, codes, pad_embed)
        if _cache_len(dynamic_cache) != pos or _cache_len(static_cache) != pos:
            raise RuntimeError(
                f"cache position mismatch before step {step}: dynamic={_cache_len(dynamic_cache)}, "
                f"static={_cache_len(static_cache)}, expected={pos}"
            )
        dynamic_hidden, dynamic_cache, dynamic_position = _decode_one(
            torch, model, dynamic_cache, query, position=pos
        )
        static_hidden, static_cache, static_position = _decode_one(
            torch, model, static_cache, query, position=pos
        )
        dynamic_logits = _codec_logits(codec_head, dynamic_hidden)
        static_logits = _codec_logits(codec_head, static_hidden)
        hidden_comparison = compare_tensors(dynamic_hidden, static_hidden, atol=atol, rtol=rtol)
        logits_comparison = compare_tensors(dynamic_logits, static_logits, atol=atol, rtol=rtol)
        dynamic_top = int(torch.argmax(dynamic_logits[0, :2048]).item())
        static_top = int(torch.argmax(static_logits[0, :2048]).item())
        logits_comparison["reference_argmax"] = dynamic_top
        logits_comparison["candidate_argmax"] = static_top
        logits_comparison["top_id_equal"] = dynamic_top == static_top
        checks.append({
            "step": step,
            "position_before": pos,
            "dynamic_position": dynamic_position,
            "static_position": static_position,
            "hidden": hidden_comparison,
            "codec_logits": logits_comparison,
        })
        prev_cb0 = dynamic_top
        pos += 1

    return {
        "capacity": int(capacity),
        "required_tokens": required,
        "dynamic_prefill_cache_lengths": dynamic_lengths,
        "static_prefill_cache_lengths": static_lengths,
        "prefill_last_hidden": prefill_hidden,
        "prefill_codec_logits": prefill_logits,
        "decode_steps": checks,
        "parity_pass": bool(
            prefill_hidden["allclose"]
            and prefill_logits["allclose"]
            and prefill_logits["top_id_equal"]
            and all(
                check["hidden"]["allclose"]
                and check["codec_logits"]["allclose"]
                and check["codec_logits"]["top_id_equal"]
                and check["dynamic_position"]["cache_len_after"]
                == check["static_position"]["cache_len_after"]
                == check["position_before"] + 1
                for check in checks
            )
        ),
        "static_cache_topology": _record_cache_topology(static_cache, config),
    }


def _query_inputs(torch: Any, qwen: Any, pad_embed: Any, *, first_cb0: int, count: int):
    queries = []
    for step in range(int(count)):
        cb0 = int(first_cb0) if step == 0 else (int(first_cb0) + 97 * step) % 2048
        codes = _deterministic_codes(torch, cb0=cb0, frame=step + 101, device=pad_embed.device)
        queries.append(_audio_query(torch, qwen, codes, pad_embed))
    return queries


def _timed_eager_decode(
    torch: Any,
    model: Any,
    cache: Any,
    prefix: Any,
    queries: list[Any],
    *,
    layer_count: int,
    warm_steps: int,
) -> dict[str, Any]:
    _prefill(torch, model, prefix, cache, layer_count=layer_count)
    start_position = int(prefix.shape[1])
    for step in range(int(warm_steps)):
        _decode_one(
            torch, model, cache, queries[step], position=start_position + step
        )
    timings: list[tuple[Any, Any]] = []
    for step in range(int(warm_steps), len(queries)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _decode_one(
            torch, model, cache, queries[step], position=start_position + step
        )
        end.record()
        timings.append((start, end))
    _sync(torch)
    elapsed = [float(start.elapsed_time(end)) for start, end in timings]
    return {
        "warm_steps": int(warm_steps),
        "measured_steps": len(elapsed),
        "step_ms": elapsed,
        "p50_ms": float(statistics.median(elapsed)) if elapsed else None,
        "p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        "mean_ms": float(statistics.mean(elapsed)) if elapsed else None,
    }


def _dynamo_snapshot(torch: Any) -> dict[str, Any]:
    try:
        counters = torch._dynamo.utils.counters
        graph_breaks = counters.get("graph_break", {})
        stats = counters.get("stats", {})
        frames = counters.get("frames", {})
        return {
            "graph_break_events": int(sum(int(value) for value in graph_breaks.values())),
            "graph_break_counts": {str(key): int(value) for key, value in graph_breaks.items()},
            "unique_graphs": int(stats.get("unique_graphs", 0)),
            "calls_captured": int(stats.get("calls_captured", 0)),
            "frames_total": int(frames.get("total", 0)),
        }
    except Exception as exc:  # diagnostic only; Torch internals vary by minor release
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _make_static_q1(torch: Any, model: Any, codec_head: Any, cache: Any) -> Any:
    class StaticQ1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.codec_head = codec_head
            self.cache = cache

        def forward(self, inputs_embeds: Any, cache_position: Any):
            output = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=self.cache,
                use_cache=True,
                cache_position=cache_position,
            )
            hidden = output.last_hidden_state[:, -1:, :]
            return hidden, self.codec_head(hidden)[:, -1, :]

    return StaticQ1().eval()


def _compile_static_q1(
    torch: Any,
    model: Any,
    codec_head: Any,
    config: Any,
    prefix: Any,
    queries: list[Any],
    *,
    capacity: int,
    layer_count: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    from transformers import StaticCache

    cache = _new_static_cache(
        torch,
        config,
        capacity=capacity,
        device=prefix.device,
        dtype=prefix.dtype,
    )
    _prefill(torch, model, prefix, cache, layer_count=layer_count)
    compile_module = _make_static_q1(torch, model, codec_head, cache)
    torch._dynamo.reset()
    # reset() clears counters too, so the post-run snapshot is local to this
    # bucket and does not accumulate noise from parity/eager measurements.
    before = _dynamo_snapshot(torch)
    compiled = torch.compile(
        compile_module,
        fullgraph=False,
        dynamic=False,
        mode="default",
    )
    prefix_len = int(prefix.shape[1])
    first_position = torch.tensor([prefix_len], device=prefix.device, dtype=torch.long)
    torch.cuda.reset_peak_memory_stats()
    compile_started = time.perf_counter()
    with torch.inference_mode():
        eager_cache = _new_static_cache(
            torch,
            config,
            capacity=capacity,
            device=prefix.device,
            dtype=prefix.dtype,
        )
        _prefill(torch, model, prefix, eager_cache, layer_count=layer_count)
        eager_hidden, eager_cache, _ = _decode_one(
            torch, model, eager_cache, queries[0], position=prefix_len
        )
        eager_logits = _codec_logits(codec_head, eager_hidden)
        compiled_hidden, compiled_logits = compiled(queries[0], first_position)
        _sync(torch)
    first_call_wall_ms = (time.perf_counter() - compile_started) * 1000.0
    first_hidden_parity = compare_tensors(eager_hidden, compiled_hidden, atol=atol, rtol=rtol)
    first_logits_parity = compare_tensors(eager_logits, compiled_logits, atol=atol, rtol=rtol)
    first_hidden_parity["top_id_equal"] = True
    eager_top = int(torch.argmax(eager_logits[0, :2048]).item())
    compiled_top = int(torch.argmax(compiled_logits[0, :2048]).item())
    first_logits_parity["reference_argmax"] = eager_top
    first_logits_parity["candidate_argmax"] = compiled_top
    first_logits_parity["top_id_equal"] = eager_top == compiled_top

    warm_steps = 16
    for step in range(1, warm_steps):
        position = torch.tensor([prefix_len + step], device=prefix.device, dtype=torch.long)
        with torch.inference_mode():
            compiled(queries[step], position)
    _sync(torch)

    measured_pairs: list[tuple[Any, Any]] = []
    for step in range(warm_steps, len(queries)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        position = torch.tensor([prefix_len + step], device=prefix.device, dtype=torch.long)
        with torch.inference_mode():
            compiled(queries[step], position)
        end.record()
        measured_pairs.append((start, end))
    _sync(torch)
    elapsed = [float(start.elapsed_time(end)) for start, end in measured_pairs]
    after = _dynamo_snapshot(torch)
    result = {
        "status": "pass" if (
            first_hidden_parity["allclose"]
            and first_logits_parity["allclose"]
            and first_logits_parity["top_id_equal"]
        ) else "compiled_parity_fail",
        "mode": "default",
        "fullgraph": False,
        "dynamic": False,
        "capacity": int(capacity),
        "warm_steps": warm_steps,
        "measured_steps": len(elapsed),
        "first_compile_and_call_wall_ms": float(first_call_wall_ms),
        "step_ms": elapsed,
        "p50_ms": float(statistics.median(elapsed)) if elapsed else None,
        "p95_ms": float(np.percentile(elapsed, 95)) if elapsed else None,
        "mean_ms": float(statistics.mean(elapsed)) if elapsed else None,
        "first_output_hidden": first_hidden_parity,
        "first_output_codec_logits": first_logits_parity,
        "dynamo_before": before,
        "dynamo_after": after,
        "dynamo_delta": {
            key: after.get(key) for key in ("graph_break_events", "unique_graphs", "calls_captured", "frames_total")
        },
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    del compiled, compile_module, cache, eager_cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen StaticCache proof requires the authorized CUDA probe window")
    if not hasattr(torch, "compile"):
        raise RuntimeError("installed Torch has no torch.compile")

    from duplex_kernel.models.talker import load_talker

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    started = time.perf_counter()
    qwen = load_talker(
        workspace=str(args.workspace),
        device=str(device),
        load_code2wav=False,
    )
    model = qwen.talker.model.eval()
    codec_head = qwen.talker.codec_head.eval()
    config = model.config
    dtype = next(model.parameters()).dtype
    if dtype != torch.bfloat16:
        raise RuntimeError(f"cached Qwen Talker must load BF16, got {dtype}")
    configured_layer_count = int(config.num_hidden_layers)
    model_layers = getattr(model, "layers", None)
    if model_layers is None or len(model_layers) != configured_layer_count:
        raise RuntimeError(
            "Qwen Talker model layer modules do not match config.num_hidden_layers: "
            f"modules={None if model_layers is None else len(model_layers)}, "
            f"config={configured_layer_count}"
        )
    # The Talker config has no layer_types in this checkpoint. A nested
    # code_predictor_config does, but those entries describe its separate
    # five-layer predictor and must not be mistaken for Talker layers.
    layer_types = list(getattr(config, "layer_types", []))

    text = str(args.say_text)
    icl_part, text_part, bos_part, pad_embed = qwen.build_prefill_parts(text)
    v9_prefix, v9_digest = _load_prefix(
        args.v9_prefix_npz,
        frames=int(args.prefix_frames),
        device=device,
        dtype=dtype,
        torch=torch,
    )
    prefix = torch.cat((icl_part, text_part, v9_prefix, bos_part), dim=1).contiguous()
    prefix_tokens = int(prefix.shape[1])
    steps = int(args.parity_steps)
    compile_steps = int(args.compile_warm_steps) + int(args.compile_measure_steps)
    max_required = prefix_tokens + max(steps, compile_steps)
    capacities = tuple(int(value) for value in args.capacities)
    selected = [capacity for capacity in capacities if capacity >= max_required]
    if not selected:
        raise ValueError(
            f"full prompt ({prefix_tokens}) plus decode ({max(steps, compile_steps)}) "
            f"needs {max_required}; no requested cache bucket fits: {capacities}"
        )
    # Validate deterministic bucket selection independently of the CLI list.
    chosen_capacity = select_cache_capacity(
        prefill_tokens=prefix_tokens,
        decode_steps=max(steps, compile_steps),
        capacities=capacities,
    )
    if chosen_capacity != min(selected):
        raise RuntimeError("capacity selector did not return the smallest fitting bucket")

    report: dict[str, Any] = {
        "schema": "qwen_talker_static_decode_probe_v1",
        "status": "running",
        "scope": "manual_low_level_cache_only_not_hf_generate_or_production_engine",
        "runtime_versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
        },
        "model": {
            "qwen_talker_revision": getattr(qwen, "qwen_talker_revision", None),
            "qwen_snapshot_path": getattr(qwen, "qwen_snapshot_path", None),
            "qwen_source_path": getattr(qwen, "qwen_source_path", None),
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "actual_model_layer_modules": len(model_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "sliding_window": getattr(config, "sliding_window", None),
            "use_sliding_window": bool(getattr(config, "use_sliding_window", False)),
            "layer_types": layer_types,
            "attn_implementation": getattr(config, "_attn_implementation", None),
            "static_cache_declared_supported": bool(getattr(type(model), "_supports_static_cache", False)),
            "dtype": str(dtype),
        },
        "conditioning": {
            "say_text_source": args.say_text_source,
            "say_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "say_text_chars": len(text),
            "v9_prefix_source": str(args.v9_prefix_npz),
            "v9_prefix_key": "h_pfx",
            "v9_prefix_frames_used": int(args.prefix_frames),
            "v9_prefix_sha256": v9_digest,
        "v9_prefix_provenance": "first 19 h_pfx frames copied from canonical P7 teacher-forced V9 fixture p0-00; not current F SAY hidden; mechanical cache parity only",
            "full_prefix_tokens": prefix_tokens,
            "prefix_components": {
                "icl_tokens": int(icl_part.shape[1]),
                "say_text_tokens": int(text_part.shape[1]),
                "v9_frames": int(v9_prefix.shape[1]),
                "codec_bos_tokens": int(bos_part.shape[1]),
            },
            "voice_conditioning": "none",
        },
        "parity_tolerance": {"atol": float(args.atol), "rtol": float(args.rtol), "greedy_top_id_required": True},
        "cache_policy": {
            "fresh_cache_per_run": True,
            "cross_response_prefix_reuse": False,
            "prefix_prefill": "full eager replay [ICL|SAY|V9|BOS] into each cache",
            "requested_capacities": list(capacities),
            "max_decode_steps_per_timing_path": compile_steps,
            "capacity_required_tokens": max_required,
            "selected_minimum_capacity": chosen_capacity,
        },
        "parity": {},
        "timing": {},
        "model_load_seconds": float(time.perf_counter() - started),
    }
    load_finished = time.perf_counter()

    def persist() -> None:
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["gpu_memory"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        _write_json_atomic(Path(args.output), report)

    persist()

    with torch.inference_mode():
        for capacity in selected:
            torch.cuda.reset_peak_memory_stats()
            bucket_started = time.perf_counter()
            try:
                parity = _run_parity_bucket(
                    torch,
                    qwen.talker,
                    model,
                    codec_head,
                    prefix,
                    pad_embed,
                    capacity=capacity,
                    decode_steps=steps,
                    atol=float(args.atol),
                    rtol=float(args.rtol),
                )
            except Exception as exc:
                report["parity"][str(capacity)] = {
                    "parity_pass": False,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                report["timing"][str(capacity)] = {"status": "skipped_parity_error"}
                persist()
                continue
            report["parity"][str(capacity)] = parity
            persist()
            if not parity["parity_pass"]:
                report["timing"][str(capacity)] = {"status": "skipped_parity_failed"}
                persist()
                continue

            first_cb0 = int(parity["prefill_codec_logits"]["reference_argmax"])
            queries = _query_inputs(
                torch,
                qwen.talker,
                pad_embed,
                first_cb0=first_cb0,
                count=compile_steps,
            )
            dynamic_cache = _new_dynamic_cache(torch, config)
            dynamic_eager = _timed_eager_decode(
                torch,
                model,
                dynamic_cache,
                prefix,
                queries,
                layer_count=int(config.num_hidden_layers),
                warm_steps=int(args.compile_warm_steps),
            )
            del dynamic_cache
            static_cache = _new_static_cache(
                torch,
                config,
                capacity=capacity,
                device=device,
                dtype=dtype,
            )
            static_eager = _timed_eager_decode(
                torch,
                model,
                static_cache,
                prefix,
                queries,
                layer_count=int(config.num_hidden_layers),
                warm_steps=int(args.compile_warm_steps),
            )
            del static_cache
            gc.collect()
            torch.cuda.empty_cache()
            try:
                compiled = _compile_static_q1(
                    torch,
                    model,
                    codec_head,
                    config,
                    prefix,
                    queries,
                    capacity=capacity,
                    layer_count=int(config.num_hidden_layers),
                    atol=float(args.atol),
                    rtol=float(args.rtol),
                )
            except Exception as exc:
                compiled = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report["timing"][str(capacity)] = {
                "status": compiled["status"],
                "dynamic_eager": dynamic_eager,
                "static_eager": static_eager,
                "static_compiled": compiled,
            }
            report["timing"][str(capacity)]["wall_seconds"] = float(time.perf_counter() - bucket_started)
            report["timing"][str(capacity)]["compile_peak_memory_allocated_bytes"] = compiled[
                "peak_memory_allocated_bytes"
            ]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            persist()
            if bool(args.stop_after_first_bucket):
                break

    report["model_load_seconds"] = float(load_finished - started)
    parity_pass = bool(report["parity"]) and all(
        entry.get("parity_pass") is True for entry in report["parity"].values()
    )
    compile_pass = bool(report["timing"]) and all(
        entry.get("status") == "pass" for entry in report["timing"].values()
    )
    report["status"] = "pass" if parity_pass and compile_pass else "failed_or_incomplete"
    report["elapsed_seconds"] = float(time.perf_counter() - started)
    report["gpu_memory"] = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    persist()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument(
        "--v9-prefix-npz",
        type=Path,
        default=Path("/workspace/.tmp/qwen_talker_static_probe_v9_h_pfx_20261003.npy"),
    )
    parser.add_argument("--prefix-frames", type=int, default=19)
    parser.add_argument("--say-text", default="Perfetto, Marco. Ho registrato il tuo codice ordine: 4829.")
    parser.add_argument("--say-text-source", default="existing_v2_response_smoke_report_20261003")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--capacities", type=int, nargs="+", default=list(DEFAULT_CAPACITIES))
    parser.add_argument("--parity-steps", type=int, default=3)
    parser.add_argument("--compile-warm-steps", type=int, default=16)
    parser.add_argument("--compile-measure-steps", type=int, default=16)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--stop-after-first-bucket", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/reports/qwen_talker_static_decode_probe_20261003.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = run(args)
    except Exception as exc:
        try:
            existing = json.loads(report_path.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or existing.get("schema") != "qwen_talker_static_decode_probe_v1":
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        report = existing or {"schema": "qwen_talker_static_decode_probe_v1"}
        report["status"] = "error_after_partial" if report.get("parity") or report.get("timing") else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return 2
    _write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
