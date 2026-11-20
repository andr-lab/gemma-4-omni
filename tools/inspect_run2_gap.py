import json
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T234523994806Z/run_02"
report_file = f"{run_dir}/report.finalized.json"

with open(report_file, "r") as f:
    report = json.load(f)

print("long_response_input_window:")
print(json.dumps(report.get("scenario", {}).get("long_response_input_window"), indent=2))
