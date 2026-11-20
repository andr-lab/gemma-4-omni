import json

path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T133946017662Z/run_01/report.finalized.json"
with open(path) as f:
    d = json.load(f)

for k, v in d.items():
    if isinstance(v, dict):
        print(f"KEY: {k} -> dict keys: {list(v.keys())[:10]}")
    elif isinstance(v, list):
        print(f"KEY: {k} -> list len {len(v)}")
    else:
        print(f"KEY: {k} -> {v}")
