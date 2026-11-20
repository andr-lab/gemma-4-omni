import json

report_path = "/workspace/artifacts/acceptance_suite_verify_12/run_01/report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

print("Errors in report:")
for err in report.get("errors", []):
    print("  ", err)

fixtures = report.get("fixtures", {})
for fkey, fval in fixtures.items():
    print(f"\nFixture: {fkey}")
    if isinstance(fval, dict):
        for subk in ["response_id", "scenario_tag", "status", "max_input_ack_gap_ms", "validation_status", "interruption_latency_ms", "action"]:
            if subk in fval:
                print(f"  {subk}: {fval[subk]}")

events_path = "/workspace/artifacts/acceptance_suite_verify_12/run_01/events.jsonl"
print("\nScanning events for responses...")
with open(events_path, "r", encoding="utf-8") as f:
    for lidx, line in enumerate(f, 1):
        if any(w in line for w in ["response.created", "response.done", "response.cancelled", "scenario_response_assigned", "EvidenceError"]):
            ev = json.loads(line)
            payload = ev.get("payload", {})
            print(f"line {lidx} [{ev.get('utc')}] {ev.get('type')}: {payload.get('response_id', '')} {payload.get('action', '')} {payload.get('scenario_tag', '')} {payload.get('message', '')}")
