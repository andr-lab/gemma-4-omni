"""Opt-in diagnostics for the native Control6 listener output path."""
from __future__ import annotations

from typing import Any

import torch


def native_control6_row_statistics(value: Any) -> dict[str, Any] | None:
    """Summarize a Q4 tensor without copying its payload to the host.

    Returned statistics describe the tensor exactly as supplied. In particular,
    query-output statistics are post-native-output-guard; the separate native
    health check is what detects values sanitized by that guard.
    """
    if not isinstance(value, torch.Tensor):
        return None
    rows = value.detach()
    if rows.ndim == 3 and rows.shape[0] == 1:
        rows = rows[0]
    if rows.ndim != 2:
        return {"shape": [int(dim) for dim in rows.shape], "rows": None}

    work = rows.float()
    finite = torch.isfinite(work)
    clean = torch.where(finite, work, torch.zeros((), device=work.device, dtype=work.dtype))
    finite_count = finite.sum(dim=-1).clamp_min(1)
    measurements = torch.stack(
        (
            (clean.square().sum(dim=-1) / finite_count).sqrt(),
            clean.abs().amax(dim=-1),
            finite.float().mean(dim=-1),
        ),
        dim=-1,
    ).detach().cpu().tolist()
    return {
        "shape": [int(dim) for dim in rows.shape],
        "rows": [
            {
                "rms": float(row[0]),
                "maxabs": float(row[1]),
                "isfinite_fraction": float(row[2]),
            }
            for row in measurements
        ],
    }


def native_control6_verify_output_health_if_zero(
    native_listener: Any,
    *,
    hidden_rms: Any,
    hidden_maxabs: Any,
) -> dict[str, Any]:
    """Synchronously read the existing native finite-guard flag only on zeros."""
    try:
        exact_zero = float(hidden_rms) == 0.0 and float(hidden_maxabs) == 0.0
    except (TypeError, ValueError, OverflowError):
        exact_zero = False
    if not exact_zero:
        return {"checked": False, "status": "not_zero_hidden"}

    verify = getattr(native_listener, "verify_output_health", None)
    if not callable(verify):
        return {"checked": False, "status": "health_check_unavailable"}
    try:
        verify()
    except Exception as exc:
        message = str(exc)[:240]
        return {
            "checked": True,
            "status": "nonfinite_guard" if "output nativo non finito" in message else "health_check_error",
            "error_type": type(exc).__name__,
            "error": message,
        }
    return {"checked": True, "status": "healthy"}


def native_control6_verify_scored_output_health(
    native_listener: Any,
    *,
    native_policy: bool,
    fresh_hidden: bool,
) -> dict[str, Any]:
    """Check the native finite-guard result before a fresh action is accepted."""
    if not native_policy or not fresh_hidden:
        return {"checked": False, "status": "not_scored"}
    verify = getattr(native_listener, "verify_output_health", None)
    if not callable(verify):
        return {"checked": False, "status": "health_check_unavailable"}
    try:
        verify()
    except Exception as exc:
        message = str(exc)[:240]
        return {
            "checked": True,
            "status": "nonfinite_guard" if "output nativo non finito" in message else "health_check_error",
            "error_type": type(exc).__name__,
            "error": message,
        }
    return {"checked": True, "status": "healthy"}


def native_control6_validate_batch_output_health(
    snapshot: Any,
    *,
    session_id: str,
    generation_id: int,
    state_epoch_id: str,
    frame_id: int,
    state_version: int,
) -> dict[str, Any]:
    """Validate the worker's immutable, native-checked batch-row snapshot.

    The underlying native finite guard is batch-wide. The snapshot is row-bound
    for identity and cursor validation, but a nonfinite flag fails every row in
    that native batch and is never attributed to one session alone.
    """
    if snapshot is None:
        return {"checked": False, "status": "health_check_unavailable"}
    row = getattr(snapshot, "row", None)
    actual = (
        getattr(row, "session_id", None),
        getattr(row, "session_generation", None),
        getattr(row, "state_epoch_id", None),
        getattr(row, "frame_id", None),
        getattr(row, "listener_state_version", None),
    )
    expected = (
        session_id,
        generation_id,
        state_epoch_id,
        frame_id,
        state_version,
    )
    batch_generation = getattr(snapshot, "generation", None)
    if (
        actual != expected
        or isinstance(batch_generation, bool)
        or not isinstance(batch_generation, int)
        or batch_generation <= 0
    ):
        return {
            "checked": False,
            "status": "health_provenance_mismatch",
            "error": "native batch health snapshot does not match the current listener row",
        }
    if getattr(snapshot, "checked", False) is not True:
        return {"checked": False, "status": "health_check_unavailable"}
    status = getattr(snapshot, "status", None)
    nonfinite = getattr(snapshot, "nonfinite", None)
    if status not in {"healthy", "nonfinite_guard"} or type(nonfinite) is not bool:
        return {
            "checked": False,
            "status": "health_check_error",
            "error": "native batch health snapshot has an invalid result",
        }
    if nonfinite != (status == "nonfinite_guard"):
        return {
            "checked": False,
            "status": "health_check_error",
            "error": "native batch health status disagrees with its finite-guard flag",
        }
    return {
        "checked": True,
        "status": status,
        "batch_health_scope": "all_rows",
        "listener_batch_output_generation": int(batch_generation),
        "listener_generation_id": int(generation_id),
        "listener_state_epoch_id": state_epoch_id,
        "listener_frame_id": int(frame_id),
        "listener_state_version": int(state_version),
    }


def native_control6_validate_listener_output_health(
    listener_output: Any,
    *,
    session_id: str,
    generation_id: int,
    state_epoch_id: str,
    frame_id: int,
) -> dict[str, Any]:
    """Validate the immutable health snapshot carried by this listener row.

    ``ListenerOutput.listener_state_version`` is post-commit while the native
    batch health manifest records the pre-step version. Require the exact row
    before deriving that predecessor so a stale session output cannot satisfy
    another frame's policy callback.
    """
    output_frame_id = getattr(listener_output, "frame_id", None)
    output_state_version = getattr(listener_output, "listener_state_version", None)
    if (
        isinstance(output_frame_id, bool)
        or not isinstance(output_frame_id, int)
        or output_frame_id != frame_id
        or isinstance(output_state_version, bool)
        or not isinstance(output_state_version, int)
        or output_state_version <= 0
    ):
        return {
            "checked": False,
            "status": "health_provenance_mismatch",
            "error": "listener output does not match the current native Control6 frame",
        }
    return native_control6_validate_batch_output_health(
        getattr(listener_output, "native_output_health", None),
        session_id=session_id,
        generation_id=generation_id,
        state_epoch_id=state_epoch_id,
        frame_id=frame_id,
        state_version=output_state_version - 1,
    )


def attach_native_control6_zero_output_diagnostic(
    trace: dict[str, Any],
    *,
    enabled: bool,
    native_policy: bool,
    head_scored: bool,
    native_listener: Any,
    audio_soft: Any,
    q4_output: Any,
    prior_health_result: dict[str, Any] | None = None,
) -> None:
    """Augment the actual decision trace, probing native health only on zeros."""
    if not enabled or not native_policy or not head_scored:
        return
    hidden_rms = trace.get("native_hidden_rms")
    hidden_maxabs = trace.get("native_hidden_maxabs")
    exact_zero_hidden = hidden_rms == 0.0 and hidden_maxabs == 0.0
    trace["native_control6_output_guard_checked"] = False
    trace["native_control6_output_guard_status"] = (
        "not_zero_hidden" if not exact_zero_hidden else "not_checked"
    )
    if not exact_zero_hidden:
        return

    # Match the canonical client event schema and capture the Q4 rows before
    # the health check synchronizes/poisons a context with a finite-guard fault.
    trace["native_control6_input_rows"] = native_control6_row_statistics(audio_soft)
    trace["native_control6_output_rows"] = native_control6_row_statistics(q4_output)

    guard_result = prior_health_result or native_control6_verify_output_health_if_zero(
        native_listener, hidden_rms=hidden_rms, hidden_maxabs=hidden_maxabs
    )
    trace["native_control6_output_guard_checked"] = bool(
        guard_result.get("checked", False)
    )
    trace["native_control6_output_guard_status"] = str(
        guard_result.get("status", "unknown")
    )
    if "error_type" in guard_result:
        trace["native_control6_output_guard_error_type"] = str(
            guard_result["error_type"]
        )
        trace["native_control6_output_guard_error"] = str(
            guard_result.get("error", "")
        )


__all__ = [
    "attach_native_control6_zero_output_diagnostic",
    "native_control6_row_statistics",
    "native_control6_validate_batch_output_health",
    "native_control6_validate_listener_output_health",
    "native_control6_verify_output_health_if_zero",
    "native_control6_verify_scored_output_health",
]
