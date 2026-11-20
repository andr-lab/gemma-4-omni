from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.engine_prefill import _cache_sequence_length
from duplex_kernel.engine_types import (
    _CompiledTalkerDecodeError,
    _NativeQ1GraphResponseLease,
    _NativeTalkerDecodeError,
)
from duplex_kernel.engine_warmup import (
    _warmup_diag_backends,
    _warmup_diag_profile,
    _warmup_diag_rotary_state,
    _warmup_diag_tensor,
)

logger = logging.getLogger("duplex_kernel.engine")
_TALKER_NATIVE_AUDIO_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1"


def _new_native_talker_q1_impl(
    engine: Any,
    talker: Any,
    past_key_values: Any,
    *,
    max_audio_frames: int,
    talk_context: Any | None = None,
) -> Any:
    """Import the current response KV into its persistent Q1 slot context."""
    if getattr(engine, "audio_response_protocol", None) != AUDIO_RESPONSE_PROTOCOL_V2:
        raise _NativeTalkerDecodeError(
            f"{_TALKER_NATIVE_AUDIO_Q1_ENV}=1 requires {AUDIO_RESPONSE_PROTOCOL_V2}"
        )
    max_audio_frames = int(max_audio_frames)
    if max_audio_frames <= 0:
        raise ValueError("max_audio_frames native Q1 deve essere positivo")
    if engine._talker_native_audio_q1_graph_requested:
        if talk_context is None:
            if not engine._native_q1_startup_warmup_active:
                raise _NativeTalkerDecodeError(
                    "native Q1 graph serving requires an exclusive response talk-context slot"
                )
            slot_id = 0
        else:
            if not talk_context.is_current():
                raise _NativeTalkerDecodeError(
                    "stale response talk-context before Q1 graph import"
                )
            slot_id = int(talk_context.slot_id)
        executor = engine._native_q1_slot_executors.get(slot_id)
        if executor is None:
            raise _NativeTalkerDecodeError(
                f"native Q1 graph slot {slot_id} was not captured before readiness"
            )
        try:
            executor.fence_q1_graph()
            executor.import_dynamic_cache_async(
                past_key_values, max_audio_frames=max_audio_frames
            )
        except Exception as exc:
            raise _NativeTalkerDecodeError(
                f"native Q1 graph slot {slot_id} import failed: {type(exc).__name__}: {exc}"
            ) from exc
        return _NativeQ1GraphResponseLease(executor, slot_id)
    from duplex_kernel.native_qwen_talker import NativeQwenTalker

    sequence_length = _cache_sequence_length(past_key_values)
    executor = NativeQwenTalker(
        talker, max_cache_len=sequence_length + max_audio_frames
    )
    try:
        executor.import_dynamic_cache_async(
            past_key_values, max_audio_frames=max_audio_frames
        )
    except Exception:
        executor.close()
        raise
    return executor


def _native_talker_q1_step_impl(
    engine: Any,
    executor: Any,
    inputs_embeds: torch.Tensor,
    *,
    metrics: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Advance one audio-only Q1 step; the returned hidden is borrowed until the next step."""
    warmup_probe = bool(getattr(engine, "_warmup_diagnostic_active", False)) and (
        not bool(getattr(engine, "_warmup_debug_first_q1_checked", False))
    )
    decoder = getattr(getattr(engine, "talker", None), "code2wav", None)
    rotary_before: list[dict[str, Any]] = []
    if warmup_probe:
        rotary_before = _warmup_diag_rotary_state(decoder)
    graph_step_used = False
    try:
        graph_step = getattr(executor, "step_audio_graph_async", None)
        graph_requested = bool(
            getattr(engine, "_talker_native_audio_q1_graph_requested", False)
        )
        if graph_requested:
            if not callable(graph_step):
                raise RuntimeError("native Q1 graph executor missing graph-step method")
            hidden = graph_step(inputs_embeds)
            graph_step_used = True
        else:
            hidden = executor.step_audio_async(inputs_embeds)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != inputs_embeds.shape[0]
            or (hidden.shape[1] != 1)
            or (hidden.shape[-1] != 2048)
        ):
            raise RuntimeError("native Qwen Q1 returned an invalid hidden state")
        if warmup_probe:
            try:
                graph_info: dict[str, Any] = {}
                info_fn = getattr(executor, "q1_graph_info", None)
                if callable(info_fn):
                    info = info_fn()
                    if isinstance(info, dict):
                        graph_info = {
                            str(key): value
                            for key, value in info.items()
                            if value is None
                            or isinstance(value, (str, bool, int, float))
                        }
                output_snapshot = hidden.detach().clone()
                engine._warmup_debug_first_q1 = {
                    "backend": (
                        "native_audio_q1_graph"
                        if graph_step_used
                        else "native_audio_q1_eager"
                    ),
                    "input": _warmup_diag_tensor(inputs_embeds),
                    "output": _warmup_diag_tensor(output_snapshot),
                    "nonfinite": not bool(torch.isfinite(output_snapshot).all().item()),
                    "rotary_buffers_before_q1": rotary_before,
                    "rotary_buffers_after_q1": _warmup_diag_rotary_state(decoder),
                    "q1_graph_info": graph_info,
                }
            except Exception as diag_exc:
                engine._warmup_debug_first_q1 = {
                    "backend": (
                        "native_audio_q1_graph"
                        if graph_step_used
                        else "native_audio_q1_eager"
                    ),
                    "diagnostic_error": type(diag_exc).__name__,
                    "rotary_buffers_before_q1": rotary_before,
                }
            finally:
                engine._warmup_debug_first_q1_checked = True
        if metrics is not None:
            metrics["talker_q1_backend"] = "native_audio_q1"
            metrics["native_talker_q1_calls"] = (
                int(metrics.get("native_talker_q1_calls", 0)) + 1
            )
            if graph_step_used:
                metrics["native_talker_q1_graph_calls"] = (
                    int(metrics.get("native_talker_q1_graph_calls", 0)) + 1
                )
        return hidden
    except Exception as exc:
        if warmup_probe:
            engine._warmup_debug_first_q1 = {
                "backend": (
                    "native_audio_q1_graph"
                    if graph_step_used
                    else "native_audio_q1_eager"
                ),
                "input": _warmup_diag_tensor(inputs_embeds),
                "error": f"{type(exc).__name__}: {exc}",
                "rotary_buffers_before_q1": rotary_before,
            }
            engine._warmup_debug_first_q1_checked = True
        raise _NativeTalkerDecodeError(
            f"native Qwen audio-Q1 failed: {type(exc).__name__}: {exc}"
        ) from exc


def _warmup_check_code2wav_weights_impl(
    engine: Any, stage: str
) -> dict[str, Any] | None:
    if not getattr(engine, "_warmup_diagnostic_active", False):
        return None
    decoder = getattr(engine.talker, "code2wav", None)
    if decoder is None:
        result = {"stage": stage, "available": False}
        engine._warmup_debug_code2wav_weights[stage] = result
        return result
    flags: list[tuple[str, torch.Tensor]] = []
    specs: set[tuple[str, str]] = set()
    trainable = 0
    for name, parameter in decoder.named_parameters():
        if parameter.is_floating_point():
            flags.append((name, torch.isfinite(parameter.detach()).all()))
            specs.add((str(parameter.dtype), str(parameter.device)))
            trainable += int(bool(parameter.requires_grad))
    finite_values = (
        torch.stack([flag for _name, flag in flags]).detach().cpu().tolist()
        if flags
        else []
    )
    bad = [name for (name, _flag), finite in zip(flags, finite_values) if not finite]
    result = {
        "stage": stage,
        "available": True,
        "parameter_tensor_count": len(flags),
        "trainable_parameter_tensor_count": trainable,
        "parameter_specs": sorted(specs),
        "finite": not bad,
        "finite_parameter_tensor_count": len(flags) - len(bad),
        "first_nonfinite_parameter_names": bad[:8],
    }
    engine._warmup_debug_code2wav_weights[stage] = result
    logger.info(
        "[warmup_code2wav_parameter_finiteness] %s",
        json.dumps(result, sort_keys=True, allow_nan=False),
    )
    if bad:
        raise RuntimeError(f"Code2Wav parameters non-finite at {stage}: {bad[:4]}")
    return result


def _qwen_q1_forward_impl(
    engine: Any,
    model: Any,
    inputs_embeds: torch.Tensor,
    past_key_values: Any,
    *,
    metrics: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, Any]:
    """Run one decode query; only v2 may select the optional compiled path."""
    compiled = (
        getattr(engine, "_compiled_talker_q1", None)
        if getattr(engine, "audio_response_protocol", None)
        == AUDIO_RESPONSE_PROTOCOL_V2
        else None
    )
    if compiled is None:
        if metrics is not None:
            metrics.setdefault("talker_q1_backend", "eager")
        warmup_probe = getattr(engine, "_warmup_diagnostic_active", False) and (
            not getattr(engine, "_warmup_debug_first_q1_checked", False)
        )
        if warmup_probe:
            engine._warmup_debug_first_q1 = {
                "backend": "eager_torch_q1",
                "input": _warmup_diag_tensor(inputs_embeds),
                "runtime_backends": _warmup_diag_backends(),
            }
            engine._warmup_check_code2wav_weights("before_first_q1")
        profiler = None
        if warmup_probe and os.environ.get("DUPLEX_WARMUP_CODE2WAV_TRACE") == "1":
            try:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if inputs_embeds.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                profiler = torch.profiler.profile(
                    activities=activities,
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                )
            except Exception as exc:
                engine._warmup_debug_first_q1["profiler_init_error"] = type(
                    exc
                ).__name__
        try:
            if profiler is None:
                output = model(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            else:
                with profiler:
                    output = model(
                        inputs_embeds=inputs_embeds,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
        finally:
            if warmup_probe:
                engine._warmup_check_code2wav_weights("after_first_q1")
                engine._warmup_debug_first_q1_checked = True
        hidden = output.last_hidden_state[:, -1:, :]
        if warmup_probe:
            engine._warmup_debug_first_q1.update(
                {
                    "output": _warmup_diag_tensor(hidden),
                    "profiler_backends": _warmup_diag_profile(profiler),
                    "output_cache_length": _cache_sequence_length(
                        output.past_key_values
                    ),
                }
            )
            engine._warmup_debug_first_q1["nonfinite"] = (
                not engine._warmup_debug_first_q1["output"]["finite"]
            )
            if engine._warmup_debug_first_q1["nonfinite"]:
                logger.error(
                    "[warmup_q1_nonfinite] %s",
                    json.dumps(
                        engine._warmup_debug_first_q1, sort_keys=True, allow_nan=False
                    ),
                )
        return (hidden, output.past_key_values)
    try:
        cache_len_before = _cache_sequence_length(past_key_values)
        hidden, returned_cache = compiled(inputs_embeds, past_key_values)
        cache_len_after = _cache_sequence_length(past_key_values)
        if returned_cache is not past_key_values:
            raise RuntimeError("compiled Qwen Q1 replaced its DynamicCache object")
        if cache_len_after != cache_len_before + int(inputs_embeds.shape[1]):
            raise RuntimeError(
                f"compiled Qwen Q1 did not append exactly its input token count (before={cache_len_before}, after={cache_len_after}, input={int(inputs_embeds.shape[1])})"
            )
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[0] != inputs_embeds.shape[0]
            or (hidden.shape[1] != 1)
        ):
            raise RuntimeError("compiled Qwen Q1 returned an invalid last hidden state")
        if engine.talker_q1_compile_status == "warming":
            engine._talker_q1_compile_warmup_calls += 1
        if metrics is not None:
            metrics["talker_q1_backend"] = "torch_compile_dynamic"
            metrics["compiled_talker_q1_calls"] = (
                int(metrics.get("compiled_talker_q1_calls", 0)) + 1
            )
        return (hidden, returned_cache)
    except Exception as exc:
        raise _CompiledTalkerDecodeError(
            f"compiled Qwen Q1 failed its DynamicCache contract: {type(exc).__name__}: {exc}"
        ) from exc
