import json
import sys

path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T133946017662Z/run_01/report.finalized.json"
if len(sys.argv) > 1:
    path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

tel = d.get("runtime_evidence", {}).get("telemetry", {})
print("TELEMETRY STATUS:", tel.get("status"))
print("TELEMETRY ERRORS:", tel.get("errors"))
print("TELEMETRY EVIDENCE GAPS:", tel.get("evidence_gaps"))
for k, v in tel.items():
    if k not in ["errors", "evidence_gaps"]:
        print(f"  {k}: {v}")
