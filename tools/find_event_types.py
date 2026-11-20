import json

events_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/events.jsonl"
types = set()
with open(events_path, "r", encoding="utf-8") as f:
    for line in f:
        ev = json.loads(line)
        t = ev.get("type", "")
        if "ack" in t.lower() or "input" in t.lower():
            types.add(t)

print("Matching event types:", types)
