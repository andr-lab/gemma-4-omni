"""Probe torch.compile around the production Qwen Talker DynamicCache path.

This standalone experiment keeps Qwen's eager prefill and DynamicCache
semantics. It compiles only the one-token Talker forward, with
``dynamic=True, fullgraph=False``. It does not edit or enable compilation in
the production engine. Run in the offline 3090 container after coordinating a
GPU window with the serving owner.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def exact_array_equal(reference: Any, candidate: Any) -> bool:
    """Compare CPU arrays/tensors exactly without passing NumPy arrays to torch.equal."""
    ref = _numpy(reference)
    cand = _numpy(candidate)
    return bool(ref.shape == cand.shape and np.array_equal(ref, cand))


def tensor_metrics(reference: Any, candidate: Any, *, pointwise_atol: float = 0.02,
                   pointwise_rtol: float = 0.02) -> dict[str, Any]:
    """Report scale-aware BF16 metrics without hiding changed token IDs."""
    ref = _numpy(reference).astype(np.float64, copy=False)
    cand = _numpy(candidate).astype(np.float64, copy=False)
    result: dict[str, Any] = {
        "shape_equal": tuple(ref.shape) == tuple(cand.shape),
        "reference_shape": list(ref.shape),
        "candidate_shape": list(cand.shape),
        "pointwise_atol": float(pointwise_atol),
        "pointwise_rtol": float(pointwise_rtol),
        "all_finite": False,
        "relative_l2": None,
        "cosine_similarity": None,
        "max_abs_diff": None,
        "pointwise_failed_count": None,
        "pointwise_element_count": int(ref.size),
        "pointwise_failed_fraction": None,
    }
    if not result["shape_equal"] or ref.size == 0:
        return result
    if not (np.isfinite(ref).all() and np.isfinite(cand).all()):
        return result
    delta = cand - ref
    ref_norm = float(np.linalg.norm(ref.reshape(-1)))
    cand_norm = float(np.linalg.norm(cand.reshape(-1)))
    denom = max(ref_norm, np.finfo(np.float64).tiny)
    cos_denom = max(ref_norm * cand_norm, np.finfo(np.float64).tiny)
    failed = np.abs(delta) > (pointwise_atol + pointwise_rtol * np.abs(ref))
    failed_count = int(np.count_nonzero(failed))
    result.update({
        "all_finite": True,
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / denom),
        "cosine_similarity": float(np.dot(ref.reshape(-1), cand.reshape(-1)) / cos_denom),
        "max_abs_diff": float(np.max(np.abs(delta))),
        "pointwise_failed_count": failed_count,
        "pointwise_failed_fraction": float(failed_count / ref.size),
    })
    return result


def parity_decision(*, metrics: dict[str, Any], reference_ids: list[int],
                    candidate_ids: list[int], max_relative_l2: float = 0.02,
                    min_cosine_similarity: float = 0.999) -> dict[str, Any]:
    ids_equal = list(reference_ids) == list(candidate_ids)
    numerical_ok = (
        metrics.get("shape_equal") is True
        and metrics.get("all_finite") is True
        and metrics.get("relative_l2") is not None
        and metrics["relative_l2"] <= max_relative_l2
        and metrics.get("cosine_similarity") is not None
        and metrics["cosine_similarity"] >= min_cosine_similarity
    )
    return {
        "numerical_ok": bool(numerical_ok),
        "greedy_ids_equal": ids_equal,
        "max_relative_l2": float(max_relative_l2),
        "min_cosine_similarity": float(min_cosine_similarity),
        "pass": bool(numerical_ok and ids_equal),
    }


def step_parity_pass(record: dict[str, Any]) -> bool:
    """Require close hidden/logits, greedy CB0, and correct DynamicCache mutation.

    Lower-ranked CB0 and residual-code differences are recorded as numerical
    diagnostics; they are not a mechanical cache/compile parity gate.
    """
    return bool(
        record.get("hidden_decision", {}).get("pass") is True
        and record.get("cb0_decision", {}).get("pass") is True
        and record.get("cache_mutation_one_token_each") is True
        and record.get("eager_returned_same_cache_object") is True
        and record.get("compiled_returned_same_cache_object") is True
    )


def compiler_capture_summary(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Summarize exact Dynamo counters; do not infer a recompilation count."""
    unique_delta = int(after.get("unique_graphs", 0)) - int(before.get("unique_graphs", 0))
    calls_delta = int(after.get("calls_captured", 0)) - int(before.get("calls_captured", 0))
    breaks_delta = int(after.get("graph_break_count", 0)) - int(before.get("graph_break_count", 0))
    captured = unique_delta > 0 and calls_delta > 0
    return {
        "q1_graph_captured": captured,
        "capture_status": "captured" if captured else "no_graph_captured",
        "unique_graphs_delta": unique_delta,
        "calls_captured_delta": calls_delta,
        "graph_breaks_delta": breaks_delta,
        "partial_capture_with_fallback": bool(captured and breaks_delta > 0),
        "recompile_count": None,
        "recompile_count_note": "Transformers-container Torch 2.12 Dynamo counters expose unique graphs/captured calls, not an explicit recompile count",
    }


def _topk(logits: Any, *, k: int = 5) -> dict[str, Any]:
    import torch

    values, ids = torch.topk(logits[0, :2048].float(), k=min(int(k), int(logits.shape[-1])))
    values_list = [float(value) for value in values.detach().cpu().tolist()]
    ids_list = [int(value) for value in ids.detach().cpu().tolist()]
    return {
        "ids": ids_list,
        "values": values_list,
        "top1_to_top2_margin": values_list[0] - values_list[1] if len(values_list) > 1 else None,
    }


def _atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cache_len(cache: Any) -> int:
    value = cache.get_seq_length()
    return int(value.item()) if hasattr(value, "item") else int(value)


def _dynamo_stats(torch: Any) -> dict[str, Any]:
    try:
        counters = torch._dynamo.utils.counters
        breaks = counters.get("graph_break", {})
        stats = counters.get("stats", {})
        frames = counters.get("frames", {})
        return {
            "graph_break_count": int(sum(int(n) for n in breaks.values())),
            "graph_break_reasons": {str(k): int(v) for k, v in breaks.items()},
            "unique_graphs": int(stats.get("unique_graphs", 0)),
            "calls_captured": int(stats.get("calls_captured", 0)),
            "frames_total": int(frames.get("total", 0)),
            "frames_ok": int(frames.get("ok", 0)),
        }
    except Exception as exc:  # Compiler internals vary by Torch minor release.
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def _make_q1_module(torch: Any, model: Any) -> Any:
    class QwenDynamicCacheQ1(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = model

        def forward(self, inputs_embeds: Any, past_key_values: Any):
            # Deliberately mirror engine.py's current Q1 call: no static cache,
            # no explicit cache_position, no mask rewrite, and no cache clone.
            output = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
            )
            return output.last_hidden_state[:, -1:, :], output.past_key_values

    return QwenDynamicCacheQ1().eval()


def _prefill(torch: Any, model: Any, prefix: Any) -> tuple[Any, Any]:
    output = model(inputs_embeds=prefix, use_cache=True)
    cache = output.past_key_values
    if cache is None:
        raise RuntimeError("eager Qwen prefill did not return its DynamicCache")
    return output.last_hidden_state[:, -1:, :], cache


def _run_subtalker_greedy(torch: Any, graph: Any, talker: Any, hidden: Any,
                          cb0: int) -> list[int]:
    c0 = torch.tensor([[int(cb0)]], device=hidden.device, dtype=torch.long)
    c0_embed = talker.get_input_embeddings()(c0)
    residual = graph.generate(
        hidden,
        int(cb0),
        do_sample=False,
        c0_embed=c0_embed,
        return_tensor=True,
    )
    values = residual.detach().to(device="cpu", dtype=torch.long).reshape(-1).tolist()
    if len(values) != 15:
        raise RuntimeError(f"SubTalker emitted {len(values)} residual IDs, expected 15")
    return [int(item) for item in values]


def _make_audio_query(torch: Any, talker: Any, codes: list[int], pad_embed: Any,
                      *, generation_step: int) -> Any:
    from duplex_kernel.qwen_conditioning import build_qwen_audio_conditioning

    tensor = torch.tensor(codes, device=pad_embed.device, dtype=torch.long).view(1, 1, 16)
    return build_qwen_audio_conditioning(
        talker,
        tensor,
        tts_pad_embed=pad_embed,
        generation_step=int(generation_step),
    )


def _compile_parity(
    torch: Any,
    talker: Any,
    model: Any,
    codec_head: Any,
    prefix: Any,
    pad_embed: Any,
    *,
    steps: int,
    max_relative_l2: float,
    min_cosine_similarity: float,
    warm_steps: int,
    measure_steps: int,
    seed: int,
    on_progress: Any,
) -> dict[str, Any]:
    from transformers import DynamicCache
    from duplex_kernel.subtalker_graph import SubtalkerGraph

    eager_hidden, eager_cache = _prefill(torch, model, prefix)
    compiled_prefill_hidden, compiled_cache = _prefill(torch, model, prefix)
    prefill_metrics = tensor_metrics(eager_hidden, compiled_prefill_hidden)
    if not torch.equal(eager_hidden, compiled_prefill_hidden):
        raise RuntimeError("two eager Qwen prefill replays did not match exactly")
    if not isinstance(eager_cache, DynamicCache) or not isinstance(compiled_cache, DynamicCache):
        raise RuntimeError(
            "current Qwen use_cache=True path did not return DynamicCache: "
            f"{type(eager_cache).__name__}, {type(compiled_cache).__name__}"
        )
    if _cache_len(eager_cache) != _cache_len(compiled_cache):
        raise RuntimeError("eager prefill caches have different sequence lengths")
    on_progress({
        "status": "running",
        "eager_prefill_exact_equal": True,
        "prefill_metrics": prefill_metrics,
        "prefill_cache_type": type(eager_cache).__name__,
        "prefill_cache_length": _cache_len(eager_cache),
        "decode_steps": [],
    })

    compile_module = _make_q1_module(torch, model)
    torch._dynamo.reset()
    before = _dynamo_stats(torch)
    compiled = torch.compile(
        compile_module,
        dynamic=True,
        fullgraph=False,
        mode="default",
    )
    graph = SubtalkerGraph(
        talker.code_predictor,
        talker_embed=talker.get_input_embeddings(),
        device=str(prefix.device),
    )
    decode_records: list[dict[str, Any]] = []
    eager_last = eager_hidden
    compiled_last = compiled_prefill_hidden
    eager_logits = codec_head(eager_last)[:, -1, :]
    compiled_logits = codec_head(compiled_last)[:, -1, :]
    initial_eager_cb0 = int(torch.argmax(eager_logits[0, :2048]).item())
    initial_compiled_cb0 = int(torch.argmax(compiled_logits[0, :2048]).item())
    common_codes = _run_subtalker_greedy(torch, graph, talker, eager_last, initial_eager_cb0)
    first_query = _make_audio_query(
        torch, talker, [initial_eager_cb0] + common_codes, pad_embed, generation_step=0
    )

    for step in range(int(steps)):
        if step > 0:
            first_query = _make_audio_query(
                torch, talker, [initial_eager_cb0] + common_codes, pad_embed,
                generation_step=step,
            )
        eager_before = _cache_len(eager_cache)
        compiled_before = _cache_len(compiled_cache)
        if eager_before != compiled_before:
            raise RuntimeError(f"paired DynamicCache lengths differ before Q1 {step}")
        started = time.perf_counter()
        eager_out = model(
            inputs_embeds=first_query,
            past_key_values=eager_cache,
            use_cache=True,
        )
        torch.cuda.synchronize()
        eager_wall_ms = (time.perf_counter() - started) * 1000.0
        eager_last = eager_out.last_hidden_state[:, -1:, :]
        eager_returned_cache = eager_out.past_key_values

        started = time.perf_counter()
        compiled_last, compiled_returned_cache = compiled(first_query, compiled_cache)
        torch.cuda.synchronize()
        compiled_wall_ms = (time.perf_counter() - started) * 1000.0
        compiled_same_cache_object = compiled_returned_cache is compiled_cache
        eager_same_cache_object = eager_returned_cache is eager_cache
        eager_cache = eager_returned_cache
        compiled_cache = compiled_returned_cache
        eager_logits = codec_head(eager_last)[:, -1, :]
        compiled_logits = codec_head(compiled_last)[:, -1, :]
        eager_cb0 = int(torch.argmax(eager_logits[0, :2048]).item())
        compiled_cb0 = int(torch.argmax(compiled_logits[0, :2048]).item())
        eager_residual = _run_subtalker_greedy(torch, graph, talker, eager_last, eager_cb0)
        compiled_residual = _run_subtalker_greedy(torch, graph, talker, compiled_last, eager_cb0)
        eager_ids = [eager_cb0] + eager_residual
        compiled_ids = [compiled_cb0] + compiled_residual
        eager_top5 = _topk(eager_logits)
        compiled_top5 = _topk(compiled_logits)
        h_metrics = tensor_metrics(eager_last, compiled_last)
        logits_metrics = tensor_metrics(eager_logits[:, :2048], compiled_logits[:, :2048])
        h_ids = parity_decision(
            metrics=h_metrics,
            reference_ids=[eager_cb0],
            candidate_ids=[compiled_cb0],
            max_relative_l2=max_relative_l2,
            min_cosine_similarity=min_cosine_similarity,
        )
        logits_ids = parity_decision(
            metrics=logits_metrics,
            reference_ids=[eager_cb0],
            candidate_ids=[compiled_cb0],
            max_relative_l2=max_relative_l2,
            min_cosine_similarity=min_cosine_similarity,
        )
        residual_equal = eager_residual == compiled_residual
        decode_records.append({
            "step": step,
            "eager_cache_len_before": eager_before,
            "compiled_cache_len_before": compiled_before,
            "eager_cache_len_after": _cache_len(eager_cache),
            "compiled_cache_len_after": _cache_len(compiled_cache),
            "cache_mutation_one_token_each": (
                _cache_len(eager_cache) == eager_before + 1
                and _cache_len(compiled_cache) == compiled_before + 1
            ),
            "eager_returned_same_cache_object": eager_same_cache_object,
            "compiled_returned_same_cache_object": compiled_same_cache_object,
            "eager_wall_ms_including_sync": eager_wall_ms,
            "compiled_wall_ms_including_sync": compiled_wall_ms,
            "hidden": h_metrics,
            "hidden_decision": h_ids,
            "cb0_logits": logits_metrics,
            "cb0_decision": logits_ids,
            "greedy_cb0_ids": {"eager": eager_cb0, "compiled": compiled_cb0},
            "cb0_top5": {
                "eager": eager_top5,
                "compiled": compiled_top5,
                "ids_equal": eager_top5["ids"] == compiled_top5["ids"],
            },
            "subtalker_residual_ids": {
                "eager": eager_residual,
                "compiled": compiled_residual,
                "all_15_equal": residual_equal,
            },
            "all_16_codebook_ids_equal": eager_ids == compiled_ids,
            "next_query_uses_common_eager_frame": True,
        })
        # Keep the next model input identical on the paired paths so each Q1
        # comparison isolates the cache/compiled-forward difference.
        initial_eager_cb0 = eager_cb0
        common_codes = eager_residual
        on_progress({
            "status": "parity_running",
            "eager_prefill_exact_equal": True,
            "prefill_metrics": prefill_metrics,
            "prefill_cache_type": type(eager_cache).__name__,
            "prefill_cache_length": _cache_len(eager_cache) - len(decode_records),
            "decode_steps": decode_records,
            "torch_compile": {
                "dynamic": True,
                "fullgraph": False,
                "mode": "default",
                "before": before,
                "after": _dynamo_stats(torch),
            },
        })

    compile_after_parity = _dynamo_stats(torch)
    compile_capture = compiler_capture_summary(before, compile_after_parity)
    parity_pass = bool(decode_records) and all(step_parity_pass(row) for row in decode_records)
    timing: dict[str, Any] = {"status": "skipped_parity_failed"}
    if parity_pass and not compile_capture["q1_graph_captured"]:
        timing = {"status": "inconclusive_no_q1_graph_captured"}
    elif parity_pass:
        timing = {
            "status": "running",
            "warm_steps": int(warm_steps),
            "measure_steps": int(measure_steps),
        }
        on_progress({
            "status": "timing_running",
            "eager_prefill_exact_equal": True,
            "prefill_metrics": prefill_metrics,
            "prefill_cache_type": type(eager_cache).__name__,
            "prefill_cache_length": _cache_len(eager_cache) - len(decode_records),
            "decode_steps": decode_records,
            "torch_compile": {"dynamic": True, "fullgraph": False, "mode": "default", "before": before,
                              "after_parity": compile_after_parity, "capture": compile_capture},
            "timing": timing,
            "torch_compile": {
                "dynamic": True,
                "fullgraph": False,
                "mode": "default",
                "before": before,
                "after": _dynamo_stats(torch),
                "capture": compile_capture,
            },
        })
        queries = _make_queries(
            torch, talker, pad_embed, count=int(warm_steps) + int(measure_steps), seed=int(seed)
        )
        eager_event_pairs: list[tuple[Any, Any]] = []
        compiled_event_pairs: list[tuple[Any, Any]] = []
        eager_wall_ms: list[float] = []
        compiled_wall_ms: list[float] = []
        eager_cache_len_start = _cache_len(eager_cache)
        compiled_cache_len_start = _cache_len(compiled_cache)
        torch.cuda.synchronize()
        for index, query in enumerate(queries):
            eager_start_event = torch.cuda.Event(enable_timing=True)
            eager_end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            eager_start_event.record()
            eager_out = model(
                inputs_embeds=query,
                past_key_values=eager_cache,
                use_cache=True,
            )
            eager_end_event.record()
            eager_wall = (time.perf_counter() - started) * 1000.0
            eager_cache = eager_out.past_key_values
            compiled_start_event = torch.cuda.Event(enable_timing=True)
            compiled_end_event = torch.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            compiled_start_event.record()
            _, compiled_cache = compiled(query, compiled_cache)
            compiled_end_event.record()
            compiled_wall = (time.perf_counter() - started) * 1000.0
            if index >= int(warm_steps):
                eager_event_pairs.append((eager_start_event, eager_end_event))
                compiled_event_pairs.append((compiled_start_event, compiled_end_event))
                eager_wall_ms.append(float(eager_wall))
                compiled_wall_ms.append(float(compiled_wall))
        torch.cuda.synchronize()

        def summarize(events: list[tuple[Any, Any]], walls: list[float]) -> dict[str, Any]:
            gpu_ms = [float(start.elapsed_time(end)) for start, end in events]
            return {
                "measured_steps": len(gpu_ms),
                "gpu_event_ms": gpu_ms,
                "gpu_p50_ms": float(statistics.median(gpu_ms)) if gpu_ms else None,
                "gpu_p95_ms": float(np.percentile(gpu_ms, 95)) if gpu_ms else None,
                "host_call_wall_ms_includes_python_dispatch": walls,
                "host_wall_p50_ms": float(statistics.median(walls)) if walls else None,
                "host_wall_p95_ms": float(np.percentile(walls, 95)) if walls else None,
            }

        timing = {
            "status": "complete",
            "warm_steps": int(warm_steps),
            "measured_steps": int(measure_steps),
            "input_source": "fixed actual 16-codebook Qwen audio embeddings; input tokens are not predicted in timing segment",
            "eager_dynamic_cache": summarize(eager_event_pairs, eager_wall_ms),
            "compiled_dynamic_cache": summarize(compiled_event_pairs, compiled_wall_ms),
            "cache_lengths": {
                "eager_start": eager_cache_len_start,
                "compiled_start": compiled_cache_len_start,
                "eager_end": _cache_len(eager_cache),
                "compiled_end": _cache_len(compiled_cache),
            },
            "dynamo_after_timing": _dynamo_stats(torch),
        }
        compile_capture_after_timing = compiler_capture_summary(before, timing["dynamo_after_timing"])
        timing["capture_after_timing"] = compile_capture_after_timing
        on_progress({
            "status": "timing_complete",
            "eager_prefill_exact_equal": True,
            "prefill_metrics": prefill_metrics,
            "prefill_cache_type": type(eager_cache).__name__,
            "prefill_cache_length": _cache_len(eager_cache) - len(decode_records) - int(warm_steps) - int(measure_steps),
            "decode_steps": decode_records,
            "torch_compile": {"dynamic": True, "fullgraph": False, "mode": "default", "before": before,
                              "after": timing["dynamo_after_timing"], "capture": compile_capture,
                              "capture_after_timing": compile_capture_after_timing},
            "timing": timing,
        })

    return {
        "parity_pass": parity_pass,
        "eager_prefill_exact_equal": exact_array_equal(eager_hidden, compiled_prefill_hidden),
        "prefill_metrics": prefill_metrics,
        "prefill_cache_type": "DynamicCache",
        "prefill_cache_length": _cache_len(eager_cache) - int(steps) - (int(warm_steps) + int(measure_steps) if timing.get("status") == "complete" else 0),
        "decode_steps": decode_records,
        "parity_pass": parity_pass,
        "timing": timing,
        "torch_compile": {
            "dynamic": True,
            "fullgraph": False,
            "mode": "default",
            "before": before,
            "after": timing.get("dynamo_after_timing", compile_after_parity),
            "capture": compile_capture,
            "capture_after_timing": timing.get("capture_after_timing"),
        },
    }


def _make_queries(torch: Any, talker: Any, pad_embed: Any, *, count: int,
                  seed: int) -> list[Any]:
    generator = np.random.default_rng(int(seed))
    queries = []
    from duplex_kernel.qwen_conditioning import build_qwen_audio_conditioning

    for step in range(int(count)):
        codes = generator.integers(0, 2048, size=(1, 1, 16), dtype=np.int64)
        codes[0, 0, 0] = (37 + step * 97) % 2048
        tensor = torch.as_tensor(codes, device=pad_embed.device, dtype=torch.long)
        queries.append(build_qwen_audio_conditioning(talker, tensor, tts_pad_embed=pad_embed))
    return queries


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from duplex_kernel.models.talker import load_talker

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen DynamicCache compile proof requires the coordinated CUDA window")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    started = time.perf_counter()
    qwen = load_talker(workspace=str(args.workspace), device=str(device), load_code2wav=False)
    talker = qwen.talker
    model = talker.model.eval()
    codec_head = talker.codec_head.eval()
    dtype = next(model.parameters()).dtype
    if dtype != torch.bfloat16:
        raise RuntimeError(f"cached Qwen Talker must load BF16, got {dtype}")
    config = model.config
    say = str(args.say_text)
    icl, say_part, bos, pad_embed = qwen.build_prefill_parts(say)
    v9_array = np.asarray(np.load(args.v9_prefix_npy, allow_pickle=False), dtype=np.float32)
    if v9_array.ndim != 3 or v9_array.shape[0] != 1 or v9_array.shape[-1] != int(config.hidden_size):
        raise ValueError(f"expected V9 prefix shape (1,T,{config.hidden_size}), got {v9_array.shape}")
    frames = int(args.prefix_frames)
    if frames <= 0 or frames > v9_array.shape[1]:
        raise ValueError(f"prefix_frames {frames} outside 1..{v9_array.shape[1]}")
    v9 = torch.from_numpy(np.ascontiguousarray(v9_array[:, :frames])).to(device=device, dtype=dtype)
    if not bool(torch.isfinite(v9).all()) or float(v9.abs().max()) == 0.0:
        raise ValueError("V9 prefix must be finite and nonzero")
    prefix = torch.cat((icl, say_part, v9, bos), dim=1).contiguous()
    torch.cuda.reset_peak_memory_stats()

    report: dict[str, Any] = {
        "schema": "qwen_talker_dynamic_cache_compile_probe_v1",
        "status": "running",
        "scope": "compile_q1_only_dynamiccache_prefill_eager_no_production_engine_edit",
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
        },
        "model": {
            "qwen_revision": getattr(qwen, "qwen_talker_revision", None),
            "snapshot_path": getattr(qwen, "qwen_snapshot_path", None),
            "source_path": getattr(qwen, "qwen_source_path", None),
            "dtype": str(dtype),
            "hidden_size": int(config.hidden_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "attention_implementation": getattr(config, "_attn_implementation", None),
            "static_cache_declared_supported": bool(getattr(type(model), "_supports_static_cache", False)),
        },
        "input": {
            "say_source": args.say_text_source,
            "say_sha256": hashlib.sha256(say.encode("utf-8")).hexdigest(),
            "say_chars": len(say),
            "v9_prefix_path": str(args.v9_prefix_npy),
            "v9_prefix_sha256": hashlib.sha256(v9.detach().float().cpu().numpy().tobytes()).hexdigest(),
            "v9_prefix_frames": frames,
            "v9_provenance": "historical P7 teacher-forced V9 projection; mechanical compile/cache parity only",
            "prefix_tokens": int(prefix.shape[1]),
            "components": {
                "icl_tokens": int(icl.shape[1]),
                "say_text_tokens": int(say_part.shape[1]),
                "v9_frames": int(v9.shape[1]),
                "codec_bos_tokens": int(bos.shape[1]),
            },
        },
        "compile": {
            "dynamic": True,
            "fullgraph": False,
            "mode": "default",
            "expected_cache": "DynamicCache; exact eager prefill before each path",
            "parity_steps": int(args.parity_steps),
        },
        "load_seconds": float(time.perf_counter() - started),
        "parity": None,
        "timing": None,
    }
    _atomic_write(args.output, report)

    def persist_parity_progress(value: dict[str, Any]) -> None:
        report["parity"] = value
        report["timing"] = value.get("timing")
        report["progress_status"] = value.get("status")
        _atomic_write(args.output, report)

    with torch.inference_mode():
        report["parity"] = _compile_parity(
            torch,
            talker,
            model,
            codec_head,
            prefix,
            pad_embed,
            steps=int(args.parity_steps),
            max_relative_l2=float(args.max_relative_l2),
            min_cosine_similarity=float(args.min_cosine_similarity),
            warm_steps=int(args.warm_steps),
            measure_steps=int(args.measure_steps),
            seed=int(args.seed),
            on_progress=persist_parity_progress,
        )
        report["timing"] = report["parity"].get("timing")
        parity_ok = report["parity"].get("parity_pass") is True
        timing_ok = isinstance(report["timing"], dict) and report["timing"].get("status") == "complete"
        report["status"] = "pass" if parity_ok and timing_ok else "parity_failed_or_incomplete"
        report["elapsed_seconds"] = float(time.perf_counter() - started)
        report["gpu_memory"] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        _atomic_write(args.output, report)
    del qwen, talker, model, codec_head
    gc.collect()
    torch.cuda.empty_cache()
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--v9-prefix-npy", type=Path,
                        default=Path("/workspace/.tmp/qwen_talker_static_probe_v9_h_pfx_20261003.npy"))
    parser.add_argument("--prefix-frames", type=int, default=19)
    parser.add_argument("--say-text", default="Perfetto, Marco. Ho registrato il tuo codice ordine: 4829.")
    parser.add_argument("--say-text-source", default="existing_v2_response_smoke_report_20261003")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--parity-steps", type=int, default=3)
    parser.add_argument("--warm-steps", type=int, default=16)
    parser.add_argument("--measure-steps", type=int, default=16)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--output", type=Path,
                        default=Path("/workspace/reports/qwen_talker_dynamic_compile_probe_20261003.json"))
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        report = run(args)
    except Exception as exc:
        try:
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if not isinstance(existing, dict) or existing.get("schema") != "qwen_talker_dynamic_cache_compile_probe_v1":
                existing = {}
        except (OSError, json.JSONDecodeError):
            existing = {}
        report = existing or {"schema": "qwen_talker_dynamic_cache_compile_probe_v1"}
        report["status"] = "error_after_partial" if report.get("parity") or report.get("timing") else "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        _atomic_write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
