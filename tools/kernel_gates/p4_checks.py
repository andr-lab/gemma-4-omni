"""GPU checks for P4: stateful V9 equivalence, causality, and Qwen ablation."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


MIN_SEQUENCES = 100
MIN_FRAMES = 5_000
PERF_STEPS = 10_000
WARMUP_STEPS = 100
MAE_MAX = 0.01
COSINE_MIN = 0.999
FUTURE_DIFF_MAX = 1e-5
P95_MAX_MS = 10.0


def _stack_cache(cache_path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    try:
        raw = torch.stack([torch.as_tensor(item) for item in cache["va_raw"]])
        anchor = torch.stack([torch.as_tensor(item) for item in cache["va_htx"]])
    except KeyError as exc:
        raise KeyError(f"cache V9 incompleta: manca {exc.args[0]}") from exc
    codes = None
    if "va_c" in cache:
        codes = torch.stack([torch.as_tensor(item) for item in cache["va_c"]])
    if raw.ndim != 3 or raw.shape[0] < MIN_SEQUENCES or raw.shape[2] != 1536:
        raise ValueError(f"va_raw deve contenere almeno 100 sequenze (N,T,1536), got {tuple(raw.shape)}")
    if anchor.shape != (raw.shape[0], 1536):
        raise ValueError(f"va_htx non allineato a va_raw: {tuple(anchor.shape)}")
    if codes is not None and codes.shape[:2] != raw.shape[:2]:
        raise ValueError(f"va_c non allineato a va_raw: {tuple(codes.shape)}")
    return raw, anchor, codes


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _append(trace: list[dict[str, Any]], event: dict[str, Any]) -> None:
    trace.append(event)


def _run_equivalence(
    stream: Any,
    raw: torch.Tensor,
    anchors: torch.Tensor,
    *,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    sequence_mae: list[float] = []
    sequence_cosine: list[float] = []
    frame_count = 0
    for sequence_id in range(raw.shape[0]):
        sequence = raw[sequence_id].to(device=stream.projector.audio_in.weight.device, dtype=stream.projector.audio_in.weight.dtype)
        anchor = anchors[sequence_id].to(device=sequence.device, dtype=sequence.dtype).unsqueeze(0)
        batch_output = stream.projector(stream.causal_conv(sequence.unsqueeze(0)), anchor).squeeze(0)
        state = stream.new_state(batch_size=1, dtype=sequence.dtype, max_seq_len=sequence.shape[0])
        incremental = torch.stack([stream.step(frame, anchor, state).squeeze(0) for frame in sequence])
        errors = (incremental.float() - batch_output.float()).abs()
        mae = float(errors.mean().item())
        cosine = float(F.cosine_similarity(incremental.float(), batch_output.float(), dim=-1).min().item())
        max_error = float(errors.max().item())
        sequence_mae.append(mae)
        sequence_cosine.append(cosine)
        for frame_id in range(sequence.shape[0]):
            frame_error = (incremental[frame_id].float() - batch_output[frame_id].float()).abs()
            frame_cosine = float(F.cosine_similarity(
                incremental[frame_id].float().unsqueeze(0), batch_output[frame_id].float().unsqueeze(0), dim=-1
            ).item())
            _append(trace, {
                "event": "p4_equivalence_frame",
                "sequence_id": sequence_id,
                "frame_id": frame_id,
                "mae": float(frame_error.mean().item()),
                "max_abs_error": float(frame_error.max().item()),
                "cosine": frame_cosine,
                "finite": bool(torch.isfinite(incremental[frame_id]).all() and torch.isfinite(batch_output[frame_id]).all()),
            })
            frame_count += 1
        _append(trace, {
            "event": "p4_equivalence_sequence",
            "sequence_id": sequence_id,
            "frames": int(sequence.shape[0]),
            "mae": mae,
            "max_abs_error": max_error,
            "cosine_min": cosine,
            "finite": bool(torch.isfinite(incremental).all() and torch.isfinite(batch_output).all()),
        })
    return {
        "sequences": len(sequence_mae),
        "frames": frame_count,
        "mae": float(sum(sequence_mae) / len(sequence_mae)),
        "mae_max_sequence": max(sequence_mae),
        "cosine_min": min(sequence_cosine),
        "finite": all(_finite(value) for value in sequence_mae + sequence_cosine),
    }


def _run_future_prefix(
    stream: Any,
    raw: torch.Tensor,
    anchors: torch.Tensor,
    *,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    differences: list[float] = []
    sequences = min(int(raw.shape[0]), MIN_SEQUENCES)
    prefix_len = max(1, int(raw.shape[1]) // 2)
    for sequence_id in range(sequences):
        prefix = raw[sequence_id, :prefix_len].to(
            device=stream.projector.audio_in.weight.device, dtype=stream.projector.audio_in.weight.dtype
        )
        sequence_a = raw[sequence_id].to(device=prefix.device, dtype=prefix.dtype)
        sequence_b = torch.cat((prefix, torch.flip(sequence_a[prefix_len:], dims=(0,))), dim=0)
        anchor = anchors[sequence_id].to(device=prefix.device, dtype=prefix.dtype).unsqueeze(0)
        state_a = stream.new_state(batch_size=1, dtype=prefix.dtype, max_seq_len=raw.shape[1])
        state_b = stream.new_state(batch_size=1, dtype=prefix.dtype, max_seq_len=raw.shape[1])
        outputs_a = []
        outputs_b = []
        for frame_id in range(sequence_a.shape[0]):
            output_a = stream.step(sequence_a[frame_id], anchor, state_a).squeeze(0).float()
            output_b = stream.step(sequence_b[frame_id], anchor, state_b).squeeze(0).float()
            if frame_id < prefix_len:
                outputs_a.append(output_a)
                outputs_b.append(output_b)
        step_difference = float(torch.stack(outputs_a).sub(torch.stack(outputs_b)).abs().max().item())
        batch_a = stream.projector(stream.causal_conv(sequence_a.unsqueeze(0)), anchor).squeeze(0)[:prefix_len]
        batch_b = stream.projector(stream.causal_conv(sequence_b.unsqueeze(0)), anchor).squeeze(0)[:prefix_len]
        batch_difference = float(batch_a.float().sub(batch_b.float()).abs().max().item())
        max_difference = max(step_difference, batch_difference)
        differences.append(max_difference)
        _append(trace, {
            "event": "p4_future_prefix_sequence",
            "sequence_id": sequence_id,
            "prefix_frames": prefix_len,
            "step_max_difference": step_difference,
            "batch_max_difference": batch_difference,
            "max_future_prefix_difference": max_difference,
            "finite": all(_finite(value) for value in (step_difference, batch_difference, max_difference)),
        })
    return {
        "sequences": sequences,
        "prefix_frames": prefix_len,
        "max_difference": max(differences),
        "mean_difference": sum(differences) / len(differences),
        "finite": all(_finite(value) for value in differences),
    }


def _run_performance(stream: Any, raw: torch.Tensor, anchors: torch.Tensor, *, trace: list[dict[str, Any]]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("P4-T3 richiede CUDA reale")
    device = stream.projector.audio_in.weight.device
    dtype = stream.projector.audio_in.weight.dtype
    sequence = raw[0].to(device=device, dtype=dtype)
    anchor = anchors[0].to(device=device, dtype=dtype).unsqueeze(0)
    state = stream.new_state(batch_size=1, dtype=dtype, max_seq_len=sequence.shape[0])
    for index in range(WARMUP_STEPS):
        stream.step(sequence[index % sequence.shape[0]], anchor, state)
        if (index + 1) % sequence.shape[0] == 0:
            state.reset()
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies: list[float] = []
    for index in range(PERF_STEPS):
        if index % sequence.shape[0] == 0:
            state.reset()
        start.record()
        stream.step(sequence[index % sequence.shape[0]], anchor, state)
        end.record()
        end.synchronize()
        latency = float(start.elapsed_time(end))
        latencies.append(latency)
        _append(trace, {
            "event": "p4_v9_step_timing",
            "step_id": index,
            "latency_ms": latency,
            "finite": _finite(latency),
        })
    values = torch.tensor(latencies, dtype=torch.float64)
    return {
        "steps": len(latencies),
        "warmup_steps": WARMUP_STEPS,
        "p50_ms": float(torch.quantile(values, 0.50).item()),
        "p95_ms": float(torch.quantile(values, 0.95).item()),
        "max_ms": float(values.max().item()),
        "finite": bool(torch.isfinite(values).all()),
    }


def _load_codes(codes_path: Path, metadata_path: Path | None, cache_codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow richiesto per riallineare i 16 codebook Qwen") from exc
    names = [f"a_cb_{index}" for index in range(16)]
    table = pq.read_table(codes_path, columns=names, use_threads=False)
    codes = torch.from_numpy(table.to_pandas().to_numpy(copy=True)).to(torch.long)
    fraction = 0.1
    segment_ends: list[int] = []
    if metadata_path is not None and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        fraction = float(metadata.get("VAL_FRACTION", fraction))
        segment_ends = [int(item) for item in metadata.get("SEGMENT_ENDS", [])]
    start = int((1.0 - fraction) * codes.shape[0])
    if segment_ends:
        start = next((item for item in segment_ends if item >= start), start)
    expected = codes[start:start + cache_codes.shape[0] * cache_codes.shape[1], 0].reshape(cache_codes.shape)
    if not torch.equal(expected, cache_codes.to(torch.long)):
        raise ValueError("va_c non coincide con a_cb_0 della validation split; allineamento rifiutato")
    windows = codes[start:start + cache_codes.shape[0] * cache_codes.shape[1]]
    if windows.shape[0] != cache_codes.shape[0] * cache_codes.shape[1]:
        raise ValueError("codici validation insufficienti per le sequenze cache")
    return windows.reshape(cache_codes.shape[0], cache_codes.shape[1], 16), start


def _aggregate_code_embeddings(model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Somma embedding CB0..CB15 come nella ricetta Qwen ufficiale."""
    parts = [model.get_input_embeddings()(codes[..., :1])]
    for codebook in range(1, 16):
        parts.append(model.code_predictor.get_input_embeddings()[codebook - 1](codes[..., codebook:codebook + 1]))
    return torch.cat(parts, dim=-2).sum(dim=-2)


def _qwen_logits(talker: Any, prefix: torch.Tensor, codes: torch.Tensor, *, causal_prefix: bool) -> torch.Tensor:
    talker_model = talker.talker
    model = talker_model.model
    batch_size, prefix_frames, _ = prefix.shape
    icl, text, bos, _ = talker.build_prefill_parts("Verifica causale del kernel.")
    icl = icl.expand(batch_size, -1, -1)
    text = text.expand(batch_size, -1, -1)
    bos = bos.expand(batch_size, -1, -1)
    embeddings = torch.cat((icl, text, prefix, bos), dim=1)
    prefix_start = icl.shape[1] + text.shape[1]
    prefix_end = prefix_start + prefix_frames
    audio_inputs = _aggregate_code_embeddings(talker_model, codes[:, :-1].to(device=prefix.device))
    embeddings = torch.cat((embeddings, audio_inputs), dim=1)
    length = embeddings.shape[1]
    mask_value = torch.finfo(embeddings.dtype).min
    attention_mask = torch.zeros((batch_size, 1, length, length), device=prefix.device, dtype=embeddings.dtype)
    causal = torch.triu(torch.ones((length, length), device=prefix.device, dtype=torch.bool), diagonal=1)
    attention_mask[:, 0].masked_fill_(causal, mask_value)
    if causal_prefix:
        for audio_index in range(prefix_frames):
            query = prefix_end + audio_index
            attention_mask[:, 0, query, prefix_start + audio_index + 1:prefix_end] = mask_value
    position_ids = torch.arange(length, device=prefix.device, dtype=torch.long).view(1, 1, length).expand(3, batch_size, length)
    with torch.inference_mode():
        hidden = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state
        # prefix_end è l'indice del BOS: il BOS predice il primo CB0;
        # seguono le T-1 audio input che predicono i frame successivi.
        audio_hidden = hidden[:, prefix_end:prefix_end + prefix_frames]
        return talker_model.codec_head(audio_hidden)


def _run_qwen_ablation(
    talker: Any,
    stream: Any,
    raw: torch.Tensor,
    anchors: torch.Tensor,
    codes: torch.Tensor,
    *,
    trace: list[dict[str, Any]],
    batch_size: int = 4,
) -> dict[str, Any]:
    # The decoder model owns codec_head, while build_prefill_parts is exposed by TalkerModel.
    device = stream.projector.audio_in.weight.device
    dtype = stream.projector.audio_in.weight.dtype
    total_frames = 0
    full_losses: list[float] = []
    causal_losses: list[float] = []
    full_top1: list[float] = []
    causal_top1: list[float] = []
    full_top5: list[float] = []
    causal_top5: list[float] = []
    kl_values: list[float] = []
    logit_deltas: list[float] = []
    quartiles: dict[str, list[float]] = {f"q{index}": [] for index in range(4)}
    for start in range(0, raw.shape[0], batch_size):
        end = min(start + batch_size, raw.shape[0])
        raw_batch = raw[start:end].to(device=device, dtype=dtype)
        anchor_batch = anchors[start:end].to(device=device, dtype=dtype)
        prefix = stream.projector(stream.causal_conv(raw_batch), anchor_batch)
        code_batch = codes[start:end].to(device=device)
        full = _qwen_logits(talker, prefix, code_batch, causal_prefix=False)
        causal = _qwen_logits(talker, prefix, code_batch, causal_prefix=True)
        target = code_batch[:, :, 0]
        full_log_probs = F.log_softmax(full.float(), dim=-1)
        causal_log_probs = F.log_softmax(causal.float(), dim=-1)
        full_nll = -full_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        causal_nll = -causal_log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        full_prob = full_log_probs.exp()
        kl = (full_prob * (full_log_probs - causal_log_probs)).sum(dim=-1)
        delta = (full.float() - causal.float()).abs().amax(dim=-1)
        full_top1_values = full.argmax(dim=-1).eq(target)
        causal_top1_values = causal.argmax(dim=-1).eq(target)
        full_top5_values = full.topk(5, dim=-1).indices.eq(target.unsqueeze(-1)).any(dim=-1)
        causal_top5_values = causal.topk(5, dim=-1).indices.eq(target.unsqueeze(-1)).any(dim=-1)
        for local_seq in range(end - start):
            for frame in range(target.shape[1]):
                values = {
                    "event": "p4_qwen_ablation_frame",
                    "sequence_id": start + local_seq,
                    "frame_id": frame,
                    "full_ce": float(full_nll[local_seq, frame].item()),
                    "causal_ce": float(causal_nll[local_seq, frame].item()),
                    "full_top1": bool(full_top1_values[local_seq, frame]),
                    "causal_top1": bool(causal_top1_values[local_seq, frame]),
                    "full_top5": bool(full_top5_values[local_seq, frame]),
                    "causal_top5": bool(causal_top5_values[local_seq, frame]),
                    "kl_full_to_causal": float(kl[local_seq, frame].item()),
                    "max_logit_delta": float(delta[local_seq, frame].item()),
                }
                _append(trace, values)
                bucket = min(3, int(frame * 4 / target.shape[1]))
                quartiles[f"q{bucket}"].append(values["causal_ce"] - values["full_ce"])
        full_losses.extend(full_nll.flatten().tolist())
        causal_losses.extend(causal_nll.flatten().tolist())
        full_top1.extend(full_top1_values.float().flatten().tolist())
        causal_top1.extend(causal_top1_values.float().flatten().tolist())
        full_top5.extend(full_top5_values.float().flatten().tolist())
        causal_top5.extend(causal_top5_values.float().flatten().tolist())
        kl_values.extend(kl.flatten().tolist())
        logit_deltas.extend(delta.flatten().tolist())
        total_frames += int(target.numel())
    quartile_delta = {key: float(sum(values) / len(values)) for key, values in quartiles.items() if values}
    return {
        "sequences": int(raw.shape[0]),
        "frames": total_frames,
        "full_ce": sum(full_losses) / len(full_losses),
        "causal_ce": sum(causal_losses) / len(causal_losses),
        "delta_ce": (sum(causal_losses) - sum(full_losses)) / len(full_losses),
        "full_top1": sum(full_top1) / len(full_top1),
        "causal_top1": sum(causal_top1) / len(causal_top1),
        "full_top5": sum(full_top5) / len(full_top5),
        "causal_top5": sum(causal_top5) / len(causal_top5),
        "kl_full_to_causal": sum(kl_values) / len(kl_values),
        "max_logit_delta": max(logit_deltas),
        "quartile_delta_ce": quartile_delta,
        "finite": all(_finite(value) for value in full_losses + causal_losses + kl_values + logit_deltas),
        "mask_future_prefix": True,
    }


def run_checks(
    workspace: Path,
    *,
    cache_path: Path,
    codes_path: Path | None,
    codes_metadata_path: Path | None,
    stream: Any,
    future_stream: Any | None,
    talker: Any | None,
    trace: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    del workspace
    raw, anchors, cache_codes = _stack_cache(cache_path)
    if cache_codes is None:
        raise ValueError("cache validation priva di va_c: T4 non può verificare l'allineamento codebook")
    if stream.max_seq_len < raw.shape[1]:
        raise ValueError(f"V9 max_seq_len={stream.max_seq_len} inferiore a T={raw.shape[1]}")
    equivalence = _run_equivalence(stream, raw, anchors, trace=trace)
    future = _run_future_prefix(future_stream or stream, raw, anchors, trace=trace)
    performance = _run_performance(stream, raw, anchors, trace=trace)
    ablation: dict[str, Any]
    codes_start: int | None = None
    if talker is None or codes_path is None:
        ablation = {"passed": False, "blocked": True, "reason": "Qwen/tokens validation non forniti"}
    else:
        codes, codes_start = _load_codes(codes_path, codes_metadata_path, cache_codes)
        ablation = _run_qwen_ablation(talker, stream, raw, anchors, codes, trace=trace)
        ablation["passed"] = bool(
            ablation["frames"] >= MIN_FRAMES
            and ablation["finite"]
            and ablation["mask_future_prefix"]
            and ablation["max_logit_delta"] > 1e-6
        )
        ablation["codes_validation_start"] = codes_start
    thresholds = {
        "equivalence_sequences_min": MIN_SEQUENCES,
        "equivalence_frames_min": MIN_FRAMES,
        "equivalence_mae_max": MAE_MAX,
        "equivalence_cosine_min": COSINE_MIN,
        "future_prefix_sequences_min": MIN_SEQUENCES,
        "future_prefix_max_difference": FUTURE_DIFF_MAX,
        "performance_steps_min": PERF_STEPS,
        "v9_step_p95_ms_max": P95_MAX_MS,
        "ablation_sequences_min": MIN_SEQUENCES,
        "ablation_frames_min": MIN_FRAMES,
    }
    metrics = {
        "real_model_required": True,
        "cache_path": str(cache_path),
        "equivalence": equivalence,
        "future_prefix": future,
        "performance": performance,
        "qwen_ablation": ablation,
        "passed": bool(
            equivalence["sequences"] >= MIN_SEQUENCES
            and equivalence["frames"] >= MIN_FRAMES
            and equivalence["mae"] <= MAE_MAX
            and equivalence["cosine_min"] >= COSINE_MIN
            and future["sequences"] >= MIN_SEQUENCES
            and future["max_difference"] <= FUTURE_DIFF_MAX
            and performance["steps"] >= PERF_STEPS
            and performance["p95_ms"] <= P95_MAX_MS
            and ablation.get("passed", False)
        ),
    }
    thresholds["qwen_ablation_required"] = True
    return metrics, thresholds
