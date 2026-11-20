import json
import sys

run_dir = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T221138714355Z/run_02"
events_file = f"{run_dir}/events.jsonl"

with open(events_file, "r") as f:
    for line in f:
        e = json.loads(line)
        p = e.get("payload", {})
        if e.get("type") == "input_audio_buffer.consumed" and p.get("frame_id") == 1128:
            print(json.dumps(p, indent=2))
