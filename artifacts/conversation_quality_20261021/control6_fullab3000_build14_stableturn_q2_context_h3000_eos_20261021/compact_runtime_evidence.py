from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

root = Path(__file__).parent
rows = [json.loads(line) for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def payload(row):
    nested = row.get("event")
    return nested if isinstance(nested, dict) else row


def percentile(values, fraction):
    ordered = sorted(float(x) for x in values if isinstance(x, (int, float)))
    if not ordered:
        return None
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


fields = (
    "frame_id", "listener_batch_decision_view_context_mode", "listener_batch_decision_view_fork_count",
    "listener_batch_decision_view_replayed_q2_count", "listener_batch_decision_view_cache_hit",
    "listener_batch_decision_view_ms", "listener_batch_queue_wait_ms", "listener_batch_compute_ms",
    "listener_batch_frontend_encode_ms", "listener_batch_native_submit_ms",
    "listener_batch_native_output_health_ms", "listener_batch_control6_policy_ms",
    "scheduler_enqueue_to_deadline_ms", "scheduler_deadline_lateness_ms",
    "scheduler_listener_callback_ms", "scheduler_deadline_to_policy_decision_ms",
    "scheduler_dispatch_end_ns", "action", "selected_action", "confidence", "accepted",
    "rejection_reason", "queue_audio_for_next_turn", "listener_cache_position_before",
    "listener_cache_position_after", "listener_batch_error",
)


def compact(row):
    p = payload(row)
    return {k: p[k] for k in fields if k in p and p[k] is not None}


def p95_max(key):
    values = [payload(row).get(key) for row in rows]
    return {"count": sum(isinstance(x, (int, float)) for x in values), "p95": percentile(values, .95), "max": max((x for x in values if isinstance(x, (int, float))), default=None)}


interesting = []
for row in rows:
    if row.get("type") not in {"input_audio_buffer.consumed", "duplex_control6.decision"}:
        continue
    p = payload(row)
    interesting.append({"type": row["type"], "timestamp_ns": row.get("timestamp_ns", row.get("received_ns")), **compact(row)})

keys = sorted({key for row in rows for key in payload(row) if "drain" in key.lower() or "eos" in key.lower() or "budget" in key.lower()})
drain_eos_values = []
for row in rows:
    p = payload(row)
    values = {k: p[k] for k in keys if k in p and p[k] is not None}
    if values:
        drain_eos_values.append({"type": row.get("type"), "timestamp_ns": row.get("timestamp_ns", row.get("received_ns")), **values})

terminal = []
for row in rows:
    if row.get("type") not in {"response.audio.segment_done", "response.done", "response.cancelled"}:
        continue
    p = payload(row)
    rm = p.get("runtime_metrics", {}) if isinstance(p.get("runtime_metrics"), dict) else {}
    talker = rm.get("talker", {}) if isinstance(rm.get("talker"), dict) else {}
    terminal.append({
        "type": row.get("type"), "timestamp_ns": row.get("timestamp_ns", row.get("received_ns")),
        "response_id": p.get("response_id"), "generation_id": p.get("generation_id"),
        "start_sample": p.get("start_sample"), "end_sample": p.get("end_sample"),
        "pcm_samples": p.get("pcm_samples", rm.get("direct_audio", {}).get("pcm_samples")),
        "talker_evidence": {k: talker[k] for k in ("generated_cb0_frames", "effective_max_frames", "terminated_by", "eos_detected", "eos_frame_id", "terminal_queries") if k in talker},
    })

errors = []
for row in rows:
    if row.get("type") not in {"error", "server_error"}:
        continue
    p = payload(row)
    errors.append({"type": row.get("type"), "timestamp_ns": row.get("timestamp_ns", row.get("received_ns")), "response_id": p.get("response_id"), "error": p.get("error"), "rejected_frames": p.get("rejected_frames")})

selected = {
    "events": len(rows),
    "input_frames_sent": sum(row.get("type") == "input_audio_frame_sent" for row in rows),
    "input_frames_received": sum(row.get("type") == "input_audio_buffer.consumed" for row in rows),
    "decision_count": sum(row.get("type") == "duplex_control6.decision" for row in rows),
    "last_20_input_ticks": [x for x in interesting if x["type"] == "input_audio_buffer.consumed"][-20:],
    "last_20_decisions": [x for x in interesting if x["type"] == "duplex_control6.decision"][-20:],
    "latency_p95_max_ms": {k: p95_max(k) for k in (
        "listener_batch_decision_view_ms", "listener_batch_queue_wait_ms", "listener_batch_compute_ms",
        "listener_batch_frontend_encode_ms", "listener_batch_native_submit_ms",
        "listener_batch_native_output_health_ms", "listener_batch_control6_policy_ms",
        "scheduler_deadline_lateness_ms", "scheduler_listener_callback_ms",
        "scheduler_deadline_to_policy_decision_ms",
    )},
    "drain_eos_budget_rows": drain_eos_values,
    "terminal_rows": terminal,
    "errors": errors,
    "response_active_tick_context": [x for x in interesting if x.get("frame_id") in {78, 79, 80, 81, 82}],
    "textual_eos_or_drain_keys_in_events": keys,
}
(root / "compact_runtime_evidence.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(selected, indent=2, ensure_ascii=False))
