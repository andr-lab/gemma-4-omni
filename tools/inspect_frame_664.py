import json

base = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T205147331301Z/run_01"
t_start = 6436679390000000
t_end =   6436679555000000

print("--- EVENTS in window ---")
with open(f"{base}/events.jsonl") as f:
    for line in f:
        ev = json.loads(line)
        ts = ev.get("timestamp_ns") or ev.get("monotonic_ns") or ev.get("payload", {}).get("timestamp_ns")
        if ts and t_start <= ts <= t_end:
            p = ev.get("payload", {})
            print(f"[{ts}] {ev.get('type')}: {p.get('frame_id') or p.get('sequence') or ''}")

print("\n--- SERVER LOG in window ---")
try:
    with open(f"{base}/server.log") as f:
        for line in f:
            if "664" in line or "1175" in line:
                print(line.strip())
except Exception as e:
    print(e)
