"""Native listener validation and response queue policy helpers."""
from __future__ import annotations

import queue
import time
from typing import Any

import numpy as np

from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.turn_policy import ControlAction

RESPONSE_QUEUE_BACKPRESSURE_TIMEOUT_S = 2.0
RESPONSE_QUEUE_BACKPRESSURE_RETRY_S = 0.005

def _native_control6_endpoint_bypass_enabled(policy: Any) -> bool:
    """Only native-row Control6 owns the early endpoint bypass path."""
    return isinstance(policy, GemmaNativeControl6Policy)

def _native_control6_decision_has_valid_input(
    *,
    fresh_hidden: bool,
    no_input_tick: bool,
    speaker_active: bool,
    action: Any,
    legal_actions: Any,
) -> bool:
    """Every native policy decision requires a genuine current-frame hidden."""
    del no_input_tick, speaker_active, action, legal_actions
    return bool(fresh_hidden)


def _native_control6_listener_source_frame_id(listener_output: Any | None) -> int | None:
    """Return a validated listener source frame without trusting stale output fields."""
    if listener_output is None:
        return None
    source_frame_id = getattr(listener_output, "source_frame_id", None)
    if source_frame_id is None:
        source_frame_id = getattr(listener_output, "frame_id", None)
    if isinstance(source_frame_id, bool) or not isinstance(source_frame_id, (int, np.integer)):
        return None
    source_frame_id = int(source_frame_id)
    return source_frame_id if source_frame_id >= 0 else None

def _native_control6_listener_output_is_fresh_for_frame(
    listener_output: Any | None,
    frame_id: int,
    *,
    source_frame_id: int | None = None,
) -> bool:
    """Require both output and source frame identities to match the policy tick."""
    if listener_output is None or source_frame_id is None:
        return False
    output_frame_id = getattr(listener_output, "frame_id", None)
    if isinstance(output_frame_id, bool) or not isinstance(output_frame_id, (int, np.integer)):
        return False
    return bool(
        int(output_frame_id) == int(frame_id)
        and source_frame_id == int(frame_id)
        and getattr(listener_output, "fresh", False)
    )

def _normalize_chat_modalities(value: Any) -> list[str]:
    """Return the API default and reject malformed modality arrays safely."""
    modalities = ["text", "audio"] if value is None else value
    if (
        not isinstance(modalities, list)
        or not modalities
        or any(not isinstance(item, str) or item not in {"text", "audio"} for item in modalities)
    ):
        raise ValueError("modalities deve contenere text e/o audio")
    return list(dict.fromkeys(modalities))

class ResponseOutputQueueFull(queue.Full):
    """A bounded response queue remained full until its worker deadline."""

    def __init__(self, *, queue_kind, burst_items, wait_ms, attempts):
        super().__init__(f"{queue_kind} response output queue full after {wait_ms:.1f} ms")
        self.queue_kind = queue_kind
        self.burst_items = int(burst_items)
        self.wait_ms = float(wait_ms)
        self.attempts = int(attempts)

def _put_response_queue_bounded(
    try_put, *, queue_kind, burst_items,
    timeout_s=RESPONSE_QUEUE_BACKPRESSURE_TIMEOUT_S,
    retry_s=RESPONSE_QUEUE_BACKPRESSURE_RETRY_S,
    cancel_reason="generation_fenced",
):
    """Retry a nonblocking producer while the event loop drains the bounded queue."""
    if queue_kind not in {"audio", "text"}:
        raise ValueError("queue_kind must be audio or text")
    if burst_items < 0 or timeout_s < 0 or retry_s <= 0:
        raise ValueError("invalid bounded response queue limits")
    started = time.monotonic()
    deadline = started + float(timeout_s)
    attempts = 0
    while True:
        attempts += 1
        outcome = try_put()
        elapsed_ms = (time.monotonic() - started) * 1000.0
        if outcome is True:
            return {
                "status": "enqueued", "queue_kind": queue_kind,
                "burst_items": int(burst_items), "wait_ms": round(elapsed_ms, 3),
                "attempts": attempts,
            }
        if outcome is False:
            return {
                "status": "cancelled", "queue_kind": queue_kind,
                "burst_items": int(burst_items), "wait_ms": round(elapsed_ms, 3),
                "attempts": attempts, "cancel_reason": str(cancel_reason),
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ResponseOutputQueueFull(
                queue_kind=queue_kind, burst_items=burst_items,
                wait_ms=(time.monotonic() - started) * 1000.0, attempts=attempts,
            )
        time.sleep(min(float(retry_s), remaining))

def _response_projector_mode(engine: Any) -> str:
    if bool(getattr(engine, "ablate_response_projector", False)):
        return "plain_qwen_text_only_no_prefix_kv"
    adapter = getattr(engine, "frame_residual_adapter", None)
    if adapter is not None:
        return str(
            getattr(
                engine,
                "frame_residual_adapter_architecture",
                getattr(adapter, "architecture_id", "frame_residual_adapter_loaded"),
            )
        )
    return "live_v9_projector"

def _module_inference_freeze_status(module: Any) -> dict[str, Any] | None:
    if module is None:
        return None
    parameters = getattr(module, "parameters", None)
    if not callable(parameters):
        return None
    values = list(parameters())
    return {
        "training": bool(getattr(module, "training", False)),
        "parameter_count": len(values),
        "requires_grad_count": sum(bool(parameter.requires_grad) for parameter in values),
    }

def _realtime_inference_failure_done_event(
    response_id: str,
    runtime_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Build the terminal event paired with a realtime INFERENCE_ERROR."""
    return {
        "type": "response.done",
        "response_id": str(response_id),
        "status": "failed",
        "reason": "inference_error",
        "runtime_metrics": runtime_metrics,
    }

def _record_turn_local_vocalized_prefix(
    branch: dict[str, Any] | None,
    generation_id: int,
    kv_position: int | None,
    text: str,
) -> None:
    """Publish an ACKed text prefix only to its owning response generation.

    A missing cursor is the frozen empty FIFO snapshot before any Qwen ACK.
    It must not manufacture a KV position or an assistant prefix.
    """
    if not isinstance(branch, dict):
        return
    lock = branch.get("spoken_prefix_lock")
    if lock is None:
        return
    with lock:
        if int(branch.get("session_generation_id", -1)) != int(generation_id):
            return
        normalized_text = str(text)
        if kv_position is None:
            if normalized_text:
                raise RuntimeError("spoken text has no acknowledged KV cursor")
            return
        previous = branch.get("spoken_prefix_kv_position")
        if previous is not None and int(kv_position) < int(previous):
            raise RuntimeError("turn-local spoken-prefix KV cursor moved backwards")
        branch["spoken_prefix_kv_position"] = int(kv_position)
        branch["spoken_prefix_text"] = normalized_text
