import json

events_path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261103T082725622045Z/run_02/events.jsonl"
with open(events_path, "r", encoding="utf-8") as f:
    for line in f:
        ev = json.loads(line)
        payload = ev.get("payload", {})
        if isinstance(payload, dict) and payload.get("frame_id") == 865:
            print(f"Server event for frame 865: {ev.get('type')}")
            for k, v in payload.items():
                if "ms" in k or "ns" in k or "lock" in k:
                    print(f"  {k}: {v}")
