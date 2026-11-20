"""Unified native listener and response helpers used by the duplex server."""
from __future__ import annotations

import time
from typing import Any

import torch

from duplex_kernel.server_unified_templates import (
    _strict_env_switch,
    _listener_frame_is_voiced,
    _unified_listener_frame_contract,
    _unified_response_owns_shared_audio_queue,
    _shared_listener_response_branch,
    _shared_listener_frame_queue_contract,
    _shared_listener_queue_voice_activity,
    _shared_listener_queue_retention_reason,
    _shared_listener_queue_onset_frame_id,
    _clear_shared_listener_response_branches,
    _attach_shared_listener_response_branches,
    _validate_private_transition_chunk,
    _configured_audio_response_protocol,
    _require_unified_stable_audio_protocol,
    _tokenize_template_text,
    _unified_audio_template_ids,
    _unified_turn_end_token_id,
    _validate_unified_audio_turn_end_suffix,
    _build_unified_audio_response_suffix,
    _prewarm_unified_v1_transition,
    _unified_v1_transition_prewarm_key,
    _prewarm_unified_v1_transition_after_import,
    _build_unified_backchannel_response_suffix,
    _build_unified_next_user_header,
    _unified_end_of_turn_token_id,
)
from duplex_kernel.server_unified_cache import _finalize_unified_native_cache_branch
from duplex_kernel.server_unified_thinker import _run_unified_native_thinker

def _consume_unified_native_text_append_result(
    result: Any,
    *,
    timings_ns: dict[str, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Join the native LM-head stream before exposing its tensors to sampling.

    The worker records ``ready_event`` after the LM head on its listener
    stream. A stream wait keeps the dependency asynchronous; record_stream
    keeps the allocator from recycling either output while this consumer
    stream is still reading it.
    """
    try:
        logits = result.logits
        hidden = result.final_norm_hidden
        ready_event = result.ready_event
    except AttributeError as exc:
        raise RuntimeError(
            "unified native text append must return logits, final_norm_hidden, and ready_event"
        ) from exc

    logits_device = getattr(logits, "device", None)
    logits_is_cuda = bool(
        getattr(logits, "is_cuda", getattr(logits_device, "type", None) == "cuda")
    )
    hidden_device = getattr(hidden, "device", logits_device)
    hidden_is_cuda = bool(
        getattr(hidden, "is_cuda", getattr(hidden_device, "type", None) == "cuda")
    )
    if logits_is_cuda != hidden_is_cuda or (
        logits_is_cuda and hidden_device != logits_device
    ):
        raise RuntimeError("unified native text outputs must share one CUDA device")
    if not logits_is_cuda:
        if ready_event is not None:
            raise RuntimeError("CPU unified native text outputs cannot carry a CUDA ready_event")
        return logits, hidden
    if ready_event is None:
        raise RuntimeError("CUDA unified native text append is missing its ready_event")

    consumer_stream = torch.cuda.current_stream(device=logits_device)
    if timings_ns is not None:
        timings_ns["consumer_wait_event_started_ns"] = time.monotonic_ns()
    consumer_stream.wait_event(ready_event)
    if timings_ns is not None:
        timings_ns["consumer_wait_event_enqueued_ns"] = time.monotonic_ns()
    logits.record_stream(consumer_stream)
    hidden.record_stream(consumer_stream)
    if timings_ns is not None:
        timings_ns["consumer_record_stream_returned_ns"] = time.monotonic_ns()
    return logits, hidden
