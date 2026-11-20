import json

report_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/report.json"
with open(report_path, "r", encoding="utf-8") as f:
    report = json.load(f)

print("Errors in report:")
for err in report.get("errors", []):
    print("  ", err)

# Find max gap details from continuity metrics in report
for k, v in report.items():
    if "input" in k or "ack" in k or "continuity" in k:
        print(f"Key {k}: {v}")

# Let's check responses in report
fixtures = report.get("fixtures", {})
for fkey, fval in fixtures.items():
    print(f"\nFixture: {fkey}")
    if isinstance(fval, dict):
        for subk in ["response_id", "scenario_tag", "status", "max_input_ack_gap_ms", "max_gap_frame_pair"]:
            if subk in fval:
                print(f"  {subk}: {fval[subk]}")
            
# Also check events for response IDs
events_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/events.jsonl"
print("\nScanning events for responses...")
with open(events_path, "r", encoding="utf-8") as f:
    for lidx, line in enumerate(f, 1):
        if any(w in line for w in ["response.created", "response.done", "response.cancelled", "EvidenceError"]):
            ev = json.loads(line)
            payload = ev.get("payload", {})
            print(f"line {lidx} [{ev.get('utc')}] {ev.get('type')}: {payload.get('response_id', '')} {payload.get('action', '')} {payload.get('reason', '')} {payload.get('message', '')}")
