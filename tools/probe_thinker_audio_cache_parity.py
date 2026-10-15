#!/usr/bin/env python3
"""Compare the production compiled Thinker path with and without text-prefix KV reuse.

This is a GPU diagnostic intended for the 3090 container. It runs the same
BF16 Gemma audio prompt through a fresh and committed-prefix cache, then repeats
the comparison with a second recording while reusing only the identical hard
text prefix. It never prints generated ``heard`` content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.audio_response_protocol import AudioResponseV2Collector
from duplex_kernel.engine import PipelineEngine
from duplex_kernel.server import _prepare_direct_audio_processor_inputs
from duplex_kernel.session import DuplexSession
from tools.eval_duplex_qlora_e2e import _decode_input_wav


def _tensor_sha256(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def _compare_tensors(left: torch.Tensor, right: torch.Tensor, *, atol: float, rtol: float) -> dict[str, Any]:
    same_shape = tuple(left.shape) == tuple(right.shape)
    left_float = left.float()
    right_float = right.float()
    difference = left_float - right_float if same_shape else None
    l2_error = float(torch.linalg.vector_norm(difference).item()) if difference is not None else None
    reference_l2 = float(torch.linalg.vector_norm(left_float).item()) if same_shape else None
    candidate_l2 = float(torch.linalg.vector_norm(right_float).item()) if same_shape else None
    relative_l2 = l2_error / reference_l2 if l2_error is not None and reference_l2 and reference_l2 > 0.0 else None
    if same_shape and left_float.numel() and reference_l2 > 0.0 and candidate_l2 > 0.0:
        cosine = float(torch.nn.functional.cosine_similarity(
            left_float.reshape(1, -1), right_float.reshape(1, -1), dim=1,
        ).item())
    elif same_shape and left_float.numel() and reference_l2 == candidate_l2 == 0.0:
        cosine = 1.0
    else:
        cosine = None
    if same_shape and difference is not None and difference.ndim >= 1 and difference.shape[-1] > 0:
        max_vector_l2_error = float(torch.linalg.vector_norm(difference, dim=-1).max().item())
        reference_max_vector_l2 = float(torch.linalg.vector_norm(left_float, dim=-1).max().item())
        candidate_max_vector_l2 = float(torch.linalg.vector_norm(right_float, dim=-1).max().item())
    else:
        max_vector_l2_error = reference_max_vector_l2 = candidate_max_vector_l2 = None
    within_tolerance = bool(
        same_shape and torch.allclose(left_float, right_float, atol=atol, rtol=rtol)
    )
    return {
        "shape": list(left.shape),
        "other_shape": list(right.shape),
        "max_abs_error": _max_abs(left, right),
        "l2_error": l2_error,
        "reference_l2_norm": reference_l2,
        "candidate_l2_norm": candidate_l2,
        "relative_l2_error": relative_l2,
        "cosine_similarity": cosine,
        "max_vector_l2_error": max_vector_l2_error,
        "reference_max_vector_l2_norm": reference_max_vector_l2,
        "candidate_max_vector_l2_norm": candidate_max_vector_l2,
        "within_tolerance": within_tolerance,
    }


def _history(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("--history-json must contain an array of completed user/assistant messages")
    result: list[dict[str, str]] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ValueError(f"history item {index} must be a user/assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"history item {index} must contain non-empty text")
        result.append({"role": str(message["role"]), "content": content})
    return result


def _word_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    def words(value: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return re.findall(r"\w+(?:['’]\w+)*", normalized, flags=re.UNICODE)

    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for ref_index, ref_word in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hyp, start=1):
            current.append(min(
                current[-1] + 1,
                previous[hyp_index] + 1,
                previous[hyp_index - 1] + int(ref_word != hyp_word),
            ))
        previous = current
    return previous[-1], len(ref)


def _say_spans_cover_text(say: str, tokens: list[dict[str, Any]]) -> bool:
    spans = sorted((int(item["char_span"][0]), int(item["char_span"][1])) for item in tokens)
    cursor = 0
    for start, end in spans:
        if start != cursor or end < start or end > len(say):
            return False
        if end == start:
            continue
        cursor = end
    return cursor == len(say) and bool(say)


def _manifest_transcript(path: Path, audio_path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    clips = data.get("clips") if isinstance(data, dict) else None
    if not isinstance(clips, list):
        raise ValueError("scenario manifest must contain clips[]")
    matches = [row for row in clips if isinstance(row, dict) and row.get("filename") == audio_path.name]
    if len(matches) != 1 or not isinstance(matches[0].get("asr_transcript"), str):
        raise ValueError(f"manifest must contain one ASR reference for {audio_path.name}")
    return matches[0]["asr_transcript"]


def _make_audio_inputs(
    engine: PipelineEngine,
    pcm: Any,
    history_messages: list[dict[str, str]],
) -> tuple[list[int], dict[str, Any], float]:
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    ids, inputs, prepare_ms = _prepare_direct_audio_processor_inputs(
        engine,
        language="it",
        history_messages=history_messages,
        pcm=pcm,
        response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
    )
    return ids, inputs, float(prepare_ms)


def _run_pass(
    engine: PipelineEngine,
    *,
    label: str,
    prompt_ids: list[int],
    audio_inputs: dict[str, Any],
    state: DuplexSession | None,
    max_new_tokens: int,
    request_started: float | None = None,
) -> dict[str, Any]:
    first_forward: dict[str, torch.Tensor] = {}
    generated: list[tuple[int, int, str, torch.Tensor]] = []
    token_observed_at: dict[int, float] = {}
    request_started = time.perf_counter() if request_started is None else request_started

    def capture_first_forward(_module: Any, _inputs: Any, output: Any) -> None:
        if first_forward:
            return
        logits = getattr(output, "logits", None)
        hidden_states = getattr(output, "hidden_states", None)
        if not isinstance(logits, torch.Tensor) or not hidden_states:
            raise RuntimeError("Thinker first forward omitted logits or hidden_states")
        first_forward["logits"] = logits[:, -1, :].detach().float().cpu()
        first_forward["hidden"] = hidden_states[-1].detach().float().cpu()

    def capture_token(index: int, token_id: int, text: str, hidden: torch.Tensor) -> None:
        generated.append((int(index), int(token_id), str(text), hidden.detach().float().cpu()))
        token_observed_at[int(index)] = time.perf_counter()
        return None

    # PyTorch 2.12 documents (module, args, output) for this hook and returns a
    # removable handle; it is observation-only and is removed after each pass.
    # https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html
    hook = engine.thinker.register_forward_hook(capture_first_forward)
    think_started = time.perf_counter()
    try:
        result = engine.think(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            state=state,
            generation_id=0 if state is not None else None,
            action=None,
            input_audio=audio_inputs,
            on_generated_token=capture_token,
        )
    finally:
        hook.remove()
    think_finished = time.perf_counter()
    if not first_forward:
        raise RuntimeError(f"{label}: production Thinker forward hook observed no prefill")

    protocol: dict[str, Any] = {"status": "invalid_or_incomplete", "reason": "no_response"}
    say_tokens: list[dict[str, Any]] = []
    private_heard = ""
    collector = AudioResponseV2Collector()
    try:
        for index, token_id, text, hidden in generated:
            collector.feed_token(token_index=index, token_id=token_id, text=text, hidden=hidden)
        parsed = collector.finish(expected_text=str(result.get("text", "")))
        # Only allow the public say text length and the typed class into this report.
        protocol = {
            "status": "valid",
            "action": parsed["action"],
            "say_char_count": len(parsed["say"]),
        }
        private_heard = parsed["heard"]
        say_text = parsed["say"]
        say_tokens = [
            {"token_index": token.token_index, "token_id": token.token_id, "char_span": list(token.char_span),
             "hidden": token.hidden.detach().float().cpu()}
            for token in collector.say_tokens
        ]
    except Exception as exc:
        say_text = ""
        protocol = {
            "status": "invalid_or_incomplete",
            "reason": getattr(exc, "reason", type(exc).__name__),
        }
    protocol_finished = time.perf_counter()

    output_hidden = torch.cat([entry[3] for entry in generated], dim=1) if generated else torch.empty((1, 0, 0))
    first_say_index = min((int(token["token_index"]) for token in say_tokens), default=None)
    first_token_at = token_observed_at.get(min(token_observed_at)) if token_observed_at else None
    first_say_at = token_observed_at.get(first_say_index) if first_say_index is not None else None
    return {
        "label": label,
        "cache_metrics": dict(result.get("runtime_metrics", {})),
        "logits": first_forward["logits"],
        "prefill_hidden": first_forward["hidden"],
        "generated_ids": [entry[1] for entry in generated],
        "generated_hidden": output_hidden,
        "say_tokens": say_tokens,
        "say_layout": [
            {key: token[key] for key in ("token_index", "token_id", "char_span")}
            for token in say_tokens
        ],
        "protocol": protocol,
        "say_text": say_text,
        "private_heard": private_heard,
        "timings": {
            "first_generated_token_from_request_ms": (
                round((first_token_at - request_started) * 1000.0, 3) if first_token_at is not None else None
            ),
            "first_say_token_from_request_ms": (
                round((first_say_at - request_started) * 1000.0, 3) if first_say_at is not None else None
            ),
            "valid_json_and_say_ready_from_request_ms": round((protocol_finished - request_started) * 1000.0, 3),
            "thinker_generation_ms": round((think_finished - think_started) * 1000.0, 3),
        },
        "response_char_count": len(str(result.get("text", ""))),
        "prompt_len": len(prompt_ids),
    }


def _compare_passes(
    fresh: dict[str, Any],
    cached: dict[str, Any],
    *,
    prefix_len: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    fresh_suffix_hidden = fresh["prefill_hidden"][:, prefix_len:, :]
    logits = _compare_tensors(fresh["logits"], cached["logits"], atol=atol, rtol=rtol)
    hidden = _compare_tensors(fresh_suffix_hidden, cached["prefill_hidden"], atol=atol, rtol=rtol)
    decode_hidden = _compare_tensors(fresh["generated_hidden"], cached["generated_hidden"], atol=atol, rtol=rtol)
    fresh_say_hidden = (
        torch.cat([token["hidden"] for token in fresh["say_tokens"]], dim=1)
        if fresh["say_tokens"] else torch.empty((1, 0, 0))
    )
    cached_say_hidden = (
        torch.cat([token["hidden"] for token in cached["say_tokens"]], dim=1)
        if cached["say_tokens"] else torch.empty((1, 0, 0))
    )
    say_hidden = _compare_tensors(fresh_say_hidden, cached_say_hidden, atol=atol, rtol=rtol)
    if fresh["logits"].shape[-1] == cached["logits"].shape[-1]:
        k = min(5, int(fresh["logits"].shape[-1]))
        top_a = torch.topk(fresh["logits"], k=k, dim=-1).indices.tolist()
        top_b = torch.topk(cached["logits"], k=k, dim=-1).indices.tolist()
    else:
        top_a, top_b = [], []
    return {
        "status": "pass" if (
            logits["within_tolerance"] and hidden["within_tolerance"]
            and decode_hidden["within_tolerance"] and say_hidden["within_tolerance"]
            and fresh["generated_ids"] == cached["generated_ids"]
            and fresh["say_layout"] == cached["say_layout"]
            and fresh["protocol"]["status"] == cached["protocol"]["status"] == "valid"
        ) else "fail",
        "logits": logits,
        "prefill_hidden_suffix": hidden,
        "generated_hidden": decode_hidden,
        "say_hidden": say_hidden,
        "top5_token_ids_equal": top_a == top_b,
        "top5_token_ids_fresh": top_a,
        "top5_token_ids_cached": top_b,
        "generated_token_ids_equal": fresh["generated_ids"] == cached["generated_ids"],
        "say_layout_equal": fresh["say_layout"] == cached["say_layout"],
        "say_token_count": len(fresh["say_layout"]),
        "protocol_valid_both": fresh["protocol"]["status"] == cached["protocol"]["status"] == "valid",
        "prefix_tokens_reused": int(cached["cache_metrics"].get("thinker_cache_reused_tokens", 0)),
        "expected_text_prefix_tokens": int(prefix_len),
    }


def _public_pass_summary(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["cache_metrics"]
    return {
        "label": result["label"],
        "protocol": result["protocol"],
        "response_char_count": result["response_char_count"],
        "generated_token_count": len(result["generated_ids"]),
        "cache_hit": bool(metrics.get("thinker_cache_hit", False)),
        "cache_reason": str(metrics.get("thinker_cache_reason", "")),
        "reused_tokens": int(metrics.get("thinker_cache_reused_tokens", 0)),
        "prefill_ms": metrics.get("thinker_prefill_ms"),
        "decode_ms": metrics.get("thinker_decode_ms"),
    }


def _run_fresh_only(args: argparse.Namespace, engine: PipelineEngine, history_messages: list[dict[str, str]]) -> dict[str, Any]:
    if bool(getattr(engine, "audio_text_prefix_cache_reuse_enabled", False)):
        raise RuntimeError("fresh-only smoke requires DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=0")
    audio_path = args.audio_a.resolve()
    expected_heard = _manifest_transcript(args.scenario_manifest.resolve(), audio_path)
    pcm = _decode_input_wav(audio_path)
    request_started = time.perf_counter()
    prompt_ids, audio_inputs, processor_ms = _make_audio_inputs(engine, pcm, history_messages)
    fresh = _run_pass(
        engine,
        label="fresh_unconstrained_v2",
        prompt_ids=prompt_ids,
        audio_inputs=audio_inputs,
        state=None,
        max_new_tokens=args.max_new_tokens,
        request_started=request_started,
    )
    cache = fresh["cache_metrics"]
    word_errors, reference_words = _word_error_counts(expected_heard, fresh["private_heard"])
    say_hidden = (
        torch.cat([token["hidden"] for token in fresh["say_tokens"]], dim=1)
        if fresh["say_tokens"] else torch.empty((1, 0, 0))
    )
    spans_complete = _say_spans_cover_text(fresh["say_text"], fresh["say_tokens"])
    valid_fresh_cache = (
        not bool(cache.get("thinker_cache_hit", False))
        and int(cache.get("thinker_cache_reused_tokens", 0)) == 0
    )
    result: dict[str, Any] = {
        "status": "pass" if (
            fresh["protocol"]["status"] == "valid" and spans_complete and valid_fresh_cache
        ) else "fail",
        "checkpoint_mode": "response_bundle" if args.bundle_dir is not None else "adapter_only_diagnostic",
        "bundle_id": getattr(args, "_probe_bundle_id", None),
        "response_protocol": "heard_action_say_v2",
        "action_source": "native_v2_generation_unconstrained",
        "runtime_backend": engine.runtime_backend,
        "dtype": str(engine.dtype),
        "compile_mode": engine.thinker_compile_mode,
        "cross_response_audio_text_prefix_cache_reuse": False,
        "history_message_count": len(history_messages),
        "audio_input": {
            "filename": audio_path.name,
            "pcm_samples": int(np.asarray(pcm).size),
            "sample_rate_hz": 16_000,
            "duration_s": round(float(np.asarray(pcm).size) / 16_000.0, 3),
            "input_features_shape": list(audio_inputs["input_features"].shape),
            "input_features_sha256": _tensor_sha256(
                audio_inputs["input_features"].detach().to(dtype=torch.bfloat16)
            ),
            "prompt_token_count": len(prompt_ids),
        },
        "native_response": {
            "protocol_status": fresh["protocol"]["status"],
            "action": fresh["protocol"].get("action"),
            "say_text": fresh["say_text"],
            "say_char_count": len(fresh["say_text"]),
            "say_hidden_shape": list(say_hidden.shape),
            "say_hidden_token_count": len(fresh["say_tokens"]),
            "say_char_spans": fresh["say_layout"],
            "say_spans_cover_text_exactly": spans_complete,
        },
        "input_asr_quality": {
            "word_errors": word_errors,
            "reference_words": reference_words,
            "wer": round(word_errors / reference_words, 6) if reference_words else None,
        },
        "timings": {
            "processor_ms": round(processor_ms, 3),
            "first_generated_token_from_processor_start_ms": fresh["timings"]["first_generated_token_from_request_ms"],
            "first_say_token_from_processor_start_ms": fresh["timings"]["first_say_token_from_request_ms"],
            "validated_public_say_ready_ms": fresh["timings"]["valid_json_and_say_ready_from_request_ms"],
            "thinker_generation_ms": fresh["timings"]["thinker_generation_ms"],
            "thinker_prefill_ms": cache.get("thinker_prefill_ms"),
            "thinker_decode_ms": cache.get("thinker_decode_ms"),
        },
        "cache_contract": {
            "status": "pass" if valid_fresh_cache else "fail",
            "cache_hit": bool(cache.get("thinker_cache_hit", False)),
            "reused_tokens": int(cache.get("thinker_cache_reused_tokens", 0)),
            "reason": str(cache.get("thinker_cache_reason", "")),
            "decode_steps": int(cache.get("thinker_decode_steps", 0)),
        },
    }
    if result["status"] == "pass":
        try:
            result["v2_say_to_audio"] = _synthesize_say(
                engine, fresh, args.wav_output.resolve(), request_started=request_started,
            )
        except Exception as exc:
            result["status"] = "fail"
            result["v2_say_to_audio"] = {
                "status": "failed",
                "reason": type(exc).__name__,
            }
    else:
        result["v2_say_to_audio"] = {"status": "skipped", "reason": "response_or_cache_contract_invalid"}
    return result


def _synthesize_say(
    engine: PipelineEngine,
    result: dict[str, Any],
    wav_path: Path,
    *,
    request_started: float | None = None,
) -> dict[str, Any]:
    if result["protocol"]["status"] != "valid" or not result["say_tokens"]:
        raise RuntimeError("cannot synthesize: complete v2 JSON and say-only hidden tokens are required")
    say_hidden = torch.cat([token["hidden"] for token in result["say_tokens"]], dim=1).to(
        device=engine.device, dtype=engine.dtype,
    )
    v9_started = torch.cuda.Event(enable_timing=True)
    v9_finished = torch.cuda.Event(enable_timing=True)
    v9_started.record(torch.cuda.current_stream(engine.device))
    prefix = engine.v9_prefix(say_hidden, 0)
    v9_finished.record(torch.cuda.current_stream(engine.device))
    torch.cuda.synchronize(engine.device)

    first_pcm_at: list[float] = []
    talk_started = time.perf_counter()

    def on_audio_chunk(chunk: Any) -> None:
        if not first_pcm_at and np.asarray(chunk).size:
            first_pcm_at.append(time.perf_counter())

    talk_metrics: dict[str, Any] = {}
    from tools.eval_duplex_qlora_e2e import _write_wav

    wav = engine.talk(
        result["say_text"], prefix, n_words=len(result["say_text"].split()),
        min_frames=48, on_audio_chunk=on_audio_chunk, metrics=talk_metrics,
    )
    talk_finished = time.perf_counter()
    pcm = np.asarray(wav, dtype=np.float32).reshape(-1)
    if not pcm.size or not np.isfinite(pcm).all() or float(np.max(np.abs(pcm))) <= 1e-5:
        raise RuntimeError("V9/Talker/Code2Wav did not produce finite non-silent PCM")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    _write_wav(wav_path, pcm, 24_000)
    return {
        "status": "observed",
        "say_hidden_tokens": int(say_hidden.shape[1]),
        "v9_input_shape": list(say_hidden.shape),
        "v9_prefix_shape": list(prefix.shape),
        "v9_ms": round(float(v9_started.elapsed_time(v9_finished)), 3),
        "first_pcm_from_talker_ms": (
            round((first_pcm_at[0] - talk_started) * 1000.0, 3) if first_pcm_at else None
        ),
        "first_pcm_from_request_ms": (
            round((first_pcm_at[0] - request_started) * 1000.0, 3)
            if first_pcm_at and request_started is not None else None
        ),
        "talk_total_ms": round((talk_finished - talk_started) * 1000.0, 3),
        "pcm_complete_from_request_ms": (
            round((talk_finished - request_started) * 1000.0, 3)
            if request_started is not None else None
        ),
        "output_samples": int(pcm.size),
        "output_seconds": round(pcm.size / 24_000.0, 3),
        "peak": round(float(np.max(np.abs(pcm))), 6),
        "wav": str(wav_path),
        "talk_metrics": talk_metrics,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("this probe requires the 3090 CUDA runtime")
    workspace = args.workspace.resolve()
    bundle_id: str | None = None
    if args.bundle_dir is not None:
        bundle = args.bundle_dir.resolve()
        if not bundle.is_dir():
            raise FileNotFoundError(bundle)
        bundle_manifest = json.loads((bundle / "bundle_manifest.json").read_text(encoding="utf-8"))
        if bundle_manifest.get("response_protocol") != "heard_action_say_v2":
            raise RuntimeError("selected response bundle is not heard_action_say_v2")
        bundle_id = str(bundle_manifest.get("bundle_id") or bundle.name)
        engine_kwargs = {"duplex_response_bundle_dir": str(bundle)}
    else:
        adapter_dir = args.adapter_dir.resolve()
        if not adapter_dir.is_dir() or not (adapter_dir / "adapter_model.safetensors").is_file():
            raise FileNotFoundError(f"adapter-only candidate is incomplete: {adapter_dir}")
        # The current repair candidate is an adapter-only checkpoint, before
        # its aligned six-class head/bundle exists. Load through the production
        # adapter-only path; the v2 prompt and strict parser below remain fixed.
        bundle_id = f"adapter-only:{adapter_dir.parent.name}"
        engine_kwargs = {"gemma_qlora_adapter_dir": str(adapter_dir)}
    engine = PipelineEngine(
        device="cuda", dtype=torch.bfloat16, runtime_backend="pytorch",
        **engine_kwargs, thinker_compile_mode="default",
        thinker_static_cache_max_len=args.static_cache_len,
    )
    if not engine.load(workspace=str(workspace)):
        raise RuntimeError("PipelineEngine failed to load the selected response bundle")
    if engine._compiled_thinker is None:
        raise RuntimeError("compiled Thinker was not initialized; run with THINKER_COMPILE_MODE=default")

    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2

    history_messages = _history(args.history_json)
    args._probe_bundle_id = bundle_id
    if args.fresh_only:
        return _run_fresh_only(args, engine, history_messages)
    if args.audio_b is None:
        raise ValueError("cache parity mode requires --audio-b")
    audio_a = _decode_input_wav(args.audio_a)
    audio_b = _decode_input_wav(args.audio_b)
    ids_a, inputs_a, _prepare_a_ms = _make_audio_inputs(engine, audio_a, history_messages)
    ids_b, inputs_b, _prepare_b_ms = _make_audio_inputs(engine, audio_b, history_messages)
    prepared_a = engine._prepare_thinker_audio_inputs(inputs_a, ids_a)
    prepared_b = engine._prepare_thinker_audio_inputs(inputs_b, ids_b)
    prefix_a, reason_a = engine._audio_text_cache_prefix(prepared_a)
    prefix_b, reason_b = engine._audio_text_cache_prefix(prepared_b)
    if reason_a or reason_b or prefix_a is None or prefix_b is None:
        raise RuntimeError(f"audio text prefix unavailable: audio_a={reason_a}, audio_b={reason_b}")
    if prefix_a != prefix_b:
        raise RuntimeError("same response history produced different hard-text cache prefixes")
    if len(prefix_a) >= len(ids_a) or len(prefix_b) >= len(ids_b):
        raise RuntimeError("audio-prefix cut must leave the current audio suffix uncached")
    features_a = inputs_a["input_features"].detach().to(dtype=torch.bfloat16)
    features_b = inputs_b["input_features"].detach().to(dtype=torch.bfloat16)
    hash_a, hash_b = _tensor_sha256(features_a), _tensor_sha256(features_b)
    if hash_a == hash_b:
        raise RuntimeError("audio_a and audio_b produced identical native input_features; choose distinct clips")

    state = DuplexSession(session_id="thinker-cache-parity")
    a_fresh = _run_pass(
        engine, label="audio_a_fresh", prompt_ids=ids_a, audio_inputs=inputs_a,
        state=state, max_new_tokens=args.max_new_tokens,
    )
    if state.thinker_static_cache is None:
        raise RuntimeError("fresh native-audio pass did not publish a hard-text prompt cache")
    a_cached = _run_pass(
        engine, label="audio_a_cached", prompt_ids=ids_a, audio_inputs=inputs_a,
        state=state, max_new_tokens=args.max_new_tokens,
    )
    b_cached = _run_pass(
        engine, label="audio_b_after_audio_a_cache", prompt_ids=ids_b, audio_inputs=inputs_b,
        state=state, max_new_tokens=args.max_new_tokens,
    )
    b_fresh = _run_pass(
        engine, label="audio_b_fresh", prompt_ids=ids_b, audio_inputs=inputs_b,
        state=None, max_new_tokens=args.max_new_tokens,
    )

    comparison_a = _compare_passes(a_fresh, a_cached, prefix_len=len(prefix_a), atol=args.atol, rtol=args.rtol)
    comparison_b = _compare_passes(b_fresh, b_cached, prefix_len=len(prefix_b), atol=args.atol, rtol=args.rtol)
    b_cache_hit = bool(b_cached["cache_metrics"].get("thinker_cache_hit"))
    b_prefix_only = int(b_cached["cache_metrics"].get("thinker_cache_reused_tokens", -1)) == len(prefix_b)
    result = {
        "status": "pass" if comparison_a["status"] == comparison_b["status"] == "pass" and b_cache_hit and b_prefix_only else "fail",
        "bundle_id": bundle_id,
        "checkpoint_mode": "response_bundle" if args.bundle_dir is not None else "adapter_only_diagnostic",
        "response_protocol": AUDIO_RESPONSE_PROTOCOL_V2,
        "runtime_backend": engine.runtime_backend,
        "dtype": str(engine.dtype),
        "compile_mode": engine.thinker_compile_mode,
        "audio": {
            "audio_a_features_sha256": hash_a,
            "audio_b_features_sha256": hash_b,
            "features_differ": hash_a != hash_b,
            "audio_a_prompt_tokens": len(ids_a),
            "audio_b_prompt_tokens": len(ids_b),
            "hard_text_prefix_tokens": len(prefix_a),
            "audio_a_uncached_suffix_tokens": len(ids_a) - len(prefix_a),
            "audio_b_uncached_suffix_tokens": len(ids_b) - len(prefix_b),
        },
        "audio_a_fresh": _public_pass_summary(a_fresh),
        "audio_a_cached": _public_pass_summary(a_cached),
        "audio_b_cached_after_a": _public_pass_summary(b_cached),
        "audio_b_fresh": _public_pass_summary(b_fresh),
        "same_audio_fresh_vs_cached": comparison_a,
        "different_audio_cached_vs_fresh": comparison_b,
        "different_audio_cache_contract": {
            "status": "pass" if b_cache_hit and b_prefix_only and hash_a != hash_b else "fail",
            "reused_only_hard_text_prefix": b_prefix_only,
            "new_audio_features_used": hash_a != hash_b,
        },
    }
    if args.synthesize:
        if result["status"] != "pass":
            result["v2_say_to_audio"] = {
                "status": "skipped",
                "reason": "cache_parity_or_changed_audio_contract_failed",
            }
            return result
        wav_path = args.output.resolve() if args.output else workspace / ".tmp" / "thinker_audio_cache_parity_say.wav"
        if wav_path.suffix.lower() != ".wav":
            wav_path = wav_path.with_name("thinker_audio_cache_parity_say.wav")
        result["v2_say_to_audio"] = _synthesize_say(engine, a_fresh, wav_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/workspace"))
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--bundle-dir", type=Path,
                                  help="v2 response bundle containing adapter and aligned head")
    checkpoint_group.add_argument("--adapter-dir", type=Path,
                                  help="adapter-only v2 candidate, for Thinker/TTS diagnostics before aligned head")
    parser.add_argument("--audio-a", type=Path, required=True, help="real mono 16 kHz input WAV")
    parser.add_argument("--audio-b", type=Path, help="different real mono 16 kHz input WAV for cache parity mode")
    parser.add_argument("--history-json", type=Path, help="optional completed user/assistant history JSON array")
    parser.add_argument("--scenario-manifest", type=Path,
                        help="manifest with the private ASR reference; required in fresh-only mode")
    parser.add_argument("--fresh-only", action="store_true",
                        help="run one unconstrained v2 audio response with cross-response text-prefix cache disabled, then synthesize only validated say hidden states")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--static-cache-len", type=int, default=4096)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wav-output", type=Path,
                        help="output WAV for fresh-only v2 say synthesis")
    parser.add_argument("--synthesize", action="store_true",
                        help="After parity succeeds, pass only validated say-token hidden states through V9/Talker/Code2Wav and save real PCM.")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.static_cache_len < 8:
        parser.error("max-new-tokens must be positive and static-cache-len must be at least 8")
    if args.fresh_only:
        if args.scenario_manifest is None or args.wav_output is None:
            parser.error("--fresh-only requires --scenario-manifest and --wav-output")
        if not args.audio_a.is_file() or not args.scenario_manifest.is_file():
            parser.error("--audio-a and --scenario-manifest must exist")
    elif args.audio_b is None:
        parser.error("cache parity mode requires --audio-b")
    result = run(args)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = args.output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as temporary:
            temporary.write(encoded + "\n")
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    print(encoded)
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
