import json

telemetry_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/report.json"
with open(telemetry_path, "r", encoding="utf-8") as f:
    data = json.load(f)

p = data.get("acceptance_parameters", {})
print("pacing tolerance:", p.get("pacing_tolerance_ms"))
print("max_input_gap_ms:", p.get("max_input_gap_ms"))

# Let's check max_input_ack_gap in report
for k, v in data.items():
    if isinstance(v, dict):
        for subk, subv in v.items():
            if "gap" in subk or "ack" in subk:
                print(f"{k}.{subk} = {subv}")
