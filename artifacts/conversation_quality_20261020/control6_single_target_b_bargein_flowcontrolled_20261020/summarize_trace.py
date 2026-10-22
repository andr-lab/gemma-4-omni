import json
from pathlib import Path

root = Path("/workspace/artifacts/conversation_quality_20261020/control6_single_target_b_bargein_flowcontrolled_20261020")
report = json.loads((root / "report.json").read_text(encoding="utf-8"))
events = [json.loads(line) for line in (root / "events.jsonl").open(encoding="utf-8")]
injection = report["timing"]["correction_first_sample_sent_ns"]
commit = report["timing"]["correction_commit_ns"]
sent = [
    e for e in events
    if e.get("type") == "input_audio_frame_sent" and e.get("step_id") == "correction_bargein"
]
frames = [e.get("event", e) for e in sent]
sent_ns = [int(e.get("sent_ns", e.get("timestamp_ns", 0))) for e in sent]
gaps = [(b - a) / 1e6 for a, b in zip(sent_ns, sent_ns[1:])]
decisions = []
for row in events:
    if row.get("type") != "duplex_control6.decision":
        continue
    e = row["event"]
    t = int(e.get("policy_decision_at_ns", row.get("received_ns", 0)))
    if injection - 1_000_000_000 <= t <= commit + 2_000_000_000:
        context = e.get("control6_context") or {}
        decisions.append({
            "frame_id": e.get("frame_id"),
            "ms_from_injection": round((t - injection) / 1e6, 1),
            "raw": e.get("native_action"),
            "p_raw": e.get("native_action_probability"),
            "effective": e.get("native_control6_effective_candidate"),
            "p_effective": e.get("native_control6_selected_effective_probability"),
            "count": e.get("native_control6_hysteresis_count"),
            "applied": e.get("gate_applied_action"),
            "accepted": e.get("accepted"),
            "reject": e.get("rejection_reason"),
            "speaker_active": context.get("active_response"),
            "peer_turn_active": context.get("peer_turn_active"),
            "user_active": context.get("user_active"),
        })
print(json.dumps({
    "correction_sent_frames": len(sent),
    "first_sent_ns": sent_ns[0] if sent_ns else None,
    "last_sent_ns": sent_ns[-1] if sent_ns else None,
    "correction_send_gap_ms": {
        "min": min(gaps) if gaps else None,
        "median": sorted(gaps)[len(gaps)//2] if gaps else None,
        "max": max(gaps) if gaps else None,
        "over_100ms": sum(x > 100 for x in gaps),
    },
    "decision_rows_near_injection": decisions,
}, ensure_ascii=False, indent=2))
