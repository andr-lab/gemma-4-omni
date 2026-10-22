from __future__ import annotations

import json
from pathlib import Path


path = Path(__file__).with_name("eos_smoke_events.jsonl")
events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_talker(value):
    if isinstance(value, dict):
        talker = value.get("talker")
        if isinstance(talker, dict) and "generated_cb0_frames" in talker:
            yield talker
        for child in value.values():
            yield from find_talker(child)
    elif isinstance(value, list):
        for child in value:
            yield from find_talker(child)


summaries = []
for event in events:
    if event.get("type") == "response.done":
        for metrics in find_talker(event):
            summaries.append({
                key: metrics.get(key)
                for key in (
                    "requested_acoustic_frame_budget",
                    "text_derived_acoustic_frame_budget",
                    "acoustic_frame_budget",
                    "effective_min_frames",
                    "effective_max_frames",
                    "live_text_eos_drained_frame_cap",
                    "generated_cb0_frames",
                    "eos_detected",
                    "eos_frame_id",
                    "terminal_queries",
                    "terminated_by",
                    "generated_pcm_samples",
                    "active_pcm_duration_ms",
                )
            })

print(json.dumps({
    "event_count": len(events),
    "types": {kind: sum(1 for event in events if event.get("type") == kind)
              for kind in sorted({event.get("type") for event in events})},
    "talker_units": summaries,
}, indent=2, ensure_ascii=False))
