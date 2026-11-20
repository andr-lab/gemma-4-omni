import json
import sys

for r in [1, 2, 3]:
    p = f"/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T150828520438Z/run_{r:02d}/runtime_telemetry.json"
    with open(p) as f:
        d = json.load(f)
    print(f"RUN {r} sample_errors:", d.get("collection", {}).get("sample_errors"))
