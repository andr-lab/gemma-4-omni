import json

path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T133946017662Z/run_01/report.finalized.json"
with open(path) as f:
    d = json.load(f)

print("--- INPUT METRICS ---")
print(d.get("input", {}))

print("\n--- PLAYOUT METRICS ---")
print(d.get("playout", {}))

print("\n--- SERVER LOG ASSESSMENT ---")
print(d.get("runtime_evidence", {}).get("server_log", {}).get("status"))
print(d.get("runtime_evidence", {}).get("server_log", {}).get("errors"))
