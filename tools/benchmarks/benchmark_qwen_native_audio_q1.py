"""Compare recurrent native audio-only Q1 against eager HF DynamicCache Q1.

Run only in the coordinated CUDA window, after the native library has been
built and Qwen weights are already cached. Prefill is eager on both paths;
then each path consumes the same three Q1 audio embeddings for parity. If
parity passes, both caches continue through a separate 16-step warmup and
16-step timing window; those timings cover Qwen Q1 only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _metrics(reference: Any, candidate: Any) -> dict[str, float]:
    import torch.nn.functional as F

    ref = reference.detach().float().reshape(-1)
    got = candidate.detach().float().reshape(-1)
    denom = max(float(ref.norm().item()), 1e-12)
    return {
        "relative_l2": float((got - ref).norm().item() / denom),
        "cosine_similarity": float(F.cosine_similarity(ref[None], got[None]).item()),
    }


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from duplex_kernel.models.talker import load_talker
    from duplex_kernel.native_qwen_talker import NativeQwenTalker
    from duplex_kernel.qwen_conditioning import build_qwen_audio_conditioning
    from duplex_kernel.subtalker_graph import SubtalkerGraph

    if not torch.cuda.is_available():
        raise RuntimeError("native Qwen audio-Q1 benchmark requires the coordinated CUDA window")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    device = torch.device(args.device)
    qwen = load_talker(workspace=str(args.workspace), device=str(device), load_code2wav=False)
    talker = qwen.talker
    model = talker.model.eval()
    codec_head = talker.codec_head.eval()
    dtype = next(model.parameters()).dtype
    if dtype != torch.bfloat16:
        raise RuntimeError(f"expected resident BF16 Qwen base weights, got {dtype}")

    icl, text_part, bos, pad_embed = qwen.build_prefill_parts(args.say_text)
    v9_array = np.asarray(np.load(args.v9_prefix_npy, allow_pickle=False), dtype=np.float32)
    hidden = int(model.config.hidden_size)
    if v9_array.ndim != 3 or v9_array.shape[0] != 1 or v9_array.shape[-1] != hidden:
        raise ValueError(f"expected V9 prefix (1,T,{hidden}), got {v9_array.shape}")
    if not 0 < int(args.prefix_frames) <= v9_array.shape[1]:
        raise ValueError("prefix_frames must select a nonempty in-range V9 sequence")
    v9 = torch.from_numpy(np.ascontiguousarray(v9_array[:, : int(args.prefix_frames)])).to(
        device=device, dtype=dtype
    )
    if not bool(torch.isfinite(v9).all()) or float(v9.abs().max().item()) == 0.0:
        raise ValueError("V9 prefix must be finite and nonzero")
    prefill = torch.cat((icl, text_part, v9, bos), dim=1).contiguous()

    with torch.inference_mode():
        reference_prefill = model(inputs_embeds=prefill, use_cache=True)
        native_prefill = model(inputs_embeds=prefill, use_cache=True)
        reference_cache = reference_prefill.past_key_values
        native_cache = native_prefill.past_key_values
        if reference_cache is None or native_cache is None:
            raise RuntimeError("eager Qwen prefill did not return a DynamicCache")
        if not torch.equal(reference_prefill.last_hidden_state, native_prefill.last_hidden_state):
            raise RuntimeError("paired eager Qwen prefills diverged before native cache import")
        sequence_length = int(native_cache.get_seq_length())
        timing_frames = int(args.warmup_steps) + int(args.timed_steps)
        max_audio_frames = int(args.steps) + timing_frames
        native = NativeQwenTalker(talker, max_cache_len=sequence_length + max_audio_frames)
        native.import_dynamic_cache_async(native_cache, max_audio_frames=max_audio_frames)
        torch.cuda.synchronize(device)

        generator = np.random.default_rng(int(args.seed))
        queries = []
        for step in range(max_audio_frames):
            codes = generator.integers(0, 2048, size=(1, 1, 16), dtype=np.int64)
            codes[0, 0, 0] = (37 + step * 97) % 2048
            code_tensor = torch.as_tensor(codes, device=device, dtype=torch.long)
            queries.append(build_qwen_audio_conditioning(
                talker,
                code_tensor,
                tts_pad_embed=pad_embed,
                generation_step=step,
            ).contiguous())

        subtalker = SubtalkerGraph(
            talker.code_predictor,
            talker_embed=talker.get_input_embeddings(),
            device=str(device),
        )
        records: list[dict[str, Any]] = []
        for index, query in enumerate(queries[: int(args.steps)]):
            if native._audio_q1_next_position != sequence_length + index:
                raise RuntimeError("native Q1 cache position was not aligned before parity step")
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            eager_output = model(
                inputs_embeds=query,
                past_key_values=reference_cache,
                use_cache=True,
            )
            reference_hidden = eager_output.last_hidden_state[:, -1:, :]
            reference_cache = eager_output.past_key_values
            end.record()
            end.synchronize()
            eager_gpu_ms = float(start.elapsed_time(end))

            start_native = torch.cuda.Event(enable_timing=True)
            end_native = torch.cuda.Event(enable_timing=True)
            start_native.record()
            native_hidden_view = native.step_audio_async(query)
            end_native.record()
            end_native.synchronize()
            native_gpu_ms = float(start_native.elapsed_time(end_native))
            native_hidden = native_hidden_view.clone()
            if native._audio_q1_next_position != sequence_length + index + 1:
                raise RuntimeError("native Q1 cache position did not advance by exactly one")

            reference_logits = codec_head(reference_hidden)[:, -1, :]
            native_logits = codec_head(native_hidden)[:, -1, :]
            cb0_reference = int(reference_logits[0, :2048].argmax().item())
            cb0_native = int(native_logits[0, :2048].argmax().item())
            residual_reference = subtalker.generate(
                reference_hidden,
                cb0_reference,
                do_sample=False,
                return_tensor=True,
            )
            residual_native = subtalker.generate(
                native_hidden,
                cb0_native,
                do_sample=False,
                return_tensor=True,
            )
            residual_reference_ids = residual_reference.detach().cpu().reshape(-1).tolist()
            residual_native_ids = residual_native.detach().cpu().reshape(-1).tolist()
            records.append({
                "step": index,
                "position": sequence_length + index,
                "hidden": _metrics(reference_hidden, native_hidden),
                "codec_logits": _metrics(reference_logits, native_logits),
                "cb0_logits": _metrics(reference_logits[:, :2048], native_logits[:, :2048]),
                "cb0_reference": cb0_reference,
                "cb0_native": cb0_native,
                "cb0_equal": cb0_reference == cb0_native,
                "cb0_top5_reference": [int(v) for v in reference_logits[0, :2048].topk(5).indices.cpu().tolist()],
                "cb0_top5_native": [int(v) for v in native_logits[0, :2048].topk(5).indices.cpu().tolist()],
                "cb0_top5_equal_informational_only": reference_logits[0, :2048].topk(5).indices.tolist()
                == native_logits[0, :2048].topk(5).indices.tolist(),
                "residual_ids_reference": [int(value) for value in residual_reference_ids],
                "residual_ids_native": [int(value) for value in residual_native_ids],
                "residual_ids_equal": residual_reference_ids == residual_native_ids,
                "residual_difference_informational_only": True,
                "eager_gpu_ms": eager_gpu_ms,
                "native_gpu_ms": native_gpu_ms,
            })

        tolerance_ok = all(
            record[metric]["relative_l2"] <= float(args.max_relative_l2)
            and record[metric]["cosine_similarity"] >= float(args.min_cosine_similarity)
            for record in records
            for metric in ("hidden", "codec_logits")
        ) and all(record["cb0_equal"] for record in records)

        timing: dict[str, Any] | None = None
        if tolerance_ok:
            eager_times: list[float] = []
            native_times: list[float] = []
            timing_queries = queries[int(args.steps):]
            for timing_index, query in enumerate(timing_queries):
                expected_position = sequence_length + int(args.steps) + timing_index
                if native._audio_q1_next_position != expected_position:
                    raise RuntimeError("native Q1 cache position was not aligned before timing step")
                torch.cuda.synchronize(device)
                eager_start = torch.cuda.Event(enable_timing=True)
                eager_end = torch.cuda.Event(enable_timing=True)
                eager_start.record()
                eager_output = model(
                    inputs_embeds=query,
                    past_key_values=reference_cache,
                    use_cache=True,
                )
                reference_cache = eager_output.past_key_values
                eager_end.record()
                eager_end.synchronize()
                eager_ms = float(eager_start.elapsed_time(eager_end))

                native_start = torch.cuda.Event(enable_timing=True)
                native_end = torch.cuda.Event(enable_timing=True)
                native_start.record()
                native.step_audio_async(query)
                native_end.record()
                native_end.synchronize()
                native_ms = float(native_start.elapsed_time(native_end))
                if native._audio_q1_next_position != expected_position + 1:
                    raise RuntimeError("native Q1 cache position did not advance by exactly one")
                if timing_index >= int(args.warmup_steps):
                    eager_times.append(eager_ms)
                    native_times.append(native_ms)
            timing = {
                "scope": "qwen_q1_forward_only; excludes codec_head, CB0 selection, and Subtalker",
                "warmup_steps": int(args.warmup_steps),
                "measured_steps": int(args.timed_steps),
                "native_positions_start": sequence_length + int(args.steps),
                "native_positions_end_exclusive": sequence_length + max_audio_frames,
                "eager_hf_cuda_event": _timing_summary(eager_times),
                "native_cuda_event": _timing_summary(native_times),
            }

    report = {
        "schema": "qwen_native_audio_q1_parity_v1",
        "status": "pass" if tolerance_ok else "parity_failed",
        "scope": "eager_hf_v2_prefill_then_three_parity_steps_and_optional_recurrent_q1_timing",
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
        },
        "model": {
            "qwen_revision": getattr(qwen, "qwen_talker_revision", None),
            "source_path": getattr(qwen, "qwen_source_path", None),
            "dtype": str(dtype),
            "hidden_size": hidden,
            "layers": int(model.config.num_hidden_layers),
        },
        "input": {
            "say_text_sha256": hashlib.sha256(args.say_text.encode("utf-8")).hexdigest(),
            "v9_prefix_path": str(args.v9_prefix_npy),
            "v9_prefix_frames": int(args.prefix_frames),
            "prefill_tokens": int(prefill.shape[1]),
            "cache_type": type(native_cache).__name__,
            "cache_sequence_length": sequence_length,
            "steps": max_audio_frames,
            "parity_steps": int(args.steps),
            "timing_warmup_steps": int(args.warmup_steps),
            "timing_measured_steps": int(args.timed_steps),
            "seed": int(args.seed),
            "audio_query_contract": "each recurrent call receives only the current [1,1,2048] audio embedding; cache position advances by one",
            "fixture_provenance": "historical V9 prefix fixture plus deterministic synthetic codec-code queries; mechanical parity/performance probe only, not conversational evidence",
        },
        "criteria": {
            "max_relative_l2": float(args.max_relative_l2),
            "min_cosine_similarity": float(args.min_cosine_similarity),
            "codec_logits_use_same_thresholds": True,
            "greedy_cb0_must_match": True,
            "top5_order": "informational_only",
            "residual_difference": "informational_only",
        },
        "prefill_hidden_equal": True,
        "steps": records,
        "timing": timing,
    }
    native.close()
    _write(args.output, report)
    del qwen, talker, model, codec_head
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    parser.add_argument("--v9-prefix-npy", type=Path, required=True)
    parser.add_argument("--prefix-frames", type=int, default=19)
    parser.add_argument("--say-text", default="Perfetto, Marco. Ho registrato il tuo codice ordine: 4829.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--timed-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    parser.add_argument("--min-cosine-similarity", type=float, default=0.999)
    parser.add_argument("--output", type=Path, default=Path("/workspace/.tmp/qwen_native_audio_q1_parity.json"))
    args = parser.parse_args()
    if args.steps != 3:
        parser.error("the parity smoke is intentionally fixed to three Q1 steps")
    if args.warmup_steps != 16 or args.timed_steps != 16:
        parser.error("the coordinated benchmark is fixed to 16 warmup and 16 measured steps")
    return args


if __name__ == "__main__":
    result = run(_parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
