import json
p = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261104T025917413344Z/run_01/report.finalized.json"
d = json.load(open(p))
print("acceptance_parameters:", json.dumps(d.get("acceptance_parameters"), indent=2))
