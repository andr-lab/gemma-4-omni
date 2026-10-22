from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


root = Path(__file__).parent
rows = [
    json.loads(line)
    for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
counts = Counter(str(row.get("type")) for row in rows)


def payload(row):
    nested = row.get("event")
    return nested if isinstance(nested, dict) else row


def percentiles(values):
    ordered = sorted(float(value) for value in values if isinstance(value, (int, float)))
    if not ordered:
        return {"count": 0, "p95": None, "max": None}
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {"count": len(ordered), "p95": ordered[index], "max": ordered[-1]}


sent = [row for row in rows if row.get("type") == "input_audio_frame_sent"]
correction = [row for row in sent if row.get("step_id") == "correction_bargein"]
voiced_correction = [row for row in correction if float(row.get("input_rms", 0.0)) >= 0.001]
decisions = [payload(row) for row in rows if row.get("type") == "duplex_control6.decision"]
decision_fields = [
    {
        key: decision.get(key)
        for key in (
            "frame_id", "action", "native_action", "selected_action", "confidence",
            "accepted", "rejection_reason", "speaker_active", "native_control6_head_scored",
        )
        if key in decision
    }
    for decision in decisions
    if decision.get("action") in {"SPEAK", "YIELD", "REPLAN"}
    or decision.get("accepted") is False
]
metric_fields = (
    "listener_batch_frontend_encode_ms",
    "listener_batch_mel_prepare_ms",
    "listener_batch_compute_ms",
    "listener_batch_decision_view_ms",
    "listener_batch_control6_policy_ms",
    "listener_batch_native_output_health_ms",
    "listener_batch_native_submit_ms",
)
latency = {
    field: percentiles(
        payload(row).get(field)
        for row in rows
        if row.get("type") in {"input_audio_buffer.consumed", "duplex_control6.decision"}
    )
    for field in metric_fields
}
server_errors = [payload(row) for row in rows if row.get("type") in {"server_error", "error"}]
response_events = []
for row in rows:
    kind = str(row.get("type", ""))
    if not (kind.startswith("response.") or kind in {"server_error", "error"}):
        continue
    nested = payload(row)
    response_events.append({
        "type": kind,
        "timestamp_ns": row.get("timestamp_ns", row.get("received_ns", row.get("received_monotonic_ns"))),
        "response_id": row.get("response_id", nested.get("response_id")),
        "generation_id": nested.get("generation_id"),
        "status": nested.get("status"),
        "action": nested.get("action"),
        "error": nested.get("error"),
        "start_sample": nested.get("start_sample"),
        "end_sample": nested.get("end_sample"),
        "output_text": nested.get("text"),
    })

summary = {
    "event_count": len(rows),
    "event_types": dict(sorted(counts.items())),
    "input_frames_sent": len(sent),
    "input_frames_sent_by_step": dict(sorted(Counter(str(row.get("step_id")) for row in sent).items())),
    "last_ingress_frame_id": sent[-1].get("ingress_frame_id") if sent else None,
    "correction_frames_sent": len(correction),
    "first_voiced_correction_frame": (
        {key: voiced_correction[0].get(key) for key in ("frame_index", "ingress_frame_id", "sent_ns", "input_rms")}
        if voiced_correction else None
    ),
    "last_voiced_correction_frame": (
        {key: voiced_correction[-1].get(key) for key in ("frame_index", "ingress_frame_id", "sent_ns", "input_rms")}
        if voiced_correction else None
    ),
    "decisions": decision_fields,
    "latency_ms": latency,
    "server_errors": server_errors,
    "response_event_count": len(response_events),
    "response_events": response_events,
}
(root / "partial_replay_summary.json").write_text(
    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(summary, indent=2, ensure_ascii=False))
