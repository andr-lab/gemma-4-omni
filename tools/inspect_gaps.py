import json

r1 = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261104T001943152237Z/run_01"

with open(f"{r1}/events.jsonl") as f:
    for line in f:
        ev = json.loads(line)
        p = ev.get("payload", {})
        if isinstance(p, dict) and p.get("type") == "duplex_control6.decision" and (p.get("frame_id") in (2700, 2701) or p.get("local_frame_id") in (2700, 2701)):
            print(json.dumps(p, indent=2))
