from __future__ import annotations

import json
from pathlib import Path


path = Path(__file__).with_name("eos_smoke_runtime_metrics.log")
found = []


def walk(value):
    if isinstance(value, dict):
        if "generated_cb0_frames" in value:
            found.append(value)
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)


for line in path.read_text(encoding="utf-8").splitlines():
    start = line.find("{")
    if start < 0:
        continue
    try:
        payload = json.loads(line[start:])
    except json.JSONDecodeError:
        continue
    walk(payload)

keys = (
    "requested_acoustic_frame_budget",
    "text_derived_acoustic_frame_budget",
    "acoustic_frame_budget",
    "effective_max_frames",
    "live_text_eos_drained_frame_cap",
    "generated_cb0_frames",
    "eos_detected",
    "eos_frame_id",
    "terminal_queries",
    "terminated_by",
)
print(json.dumps([{key: row.get(key) for key in keys if key in row} for row in found], indent=2))
