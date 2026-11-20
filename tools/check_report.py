import json
import sys

report_path = sys.argv[1]
with open(report_path) as f:
    r = json.load(f)

print("Overall status:", r.get("overall_status"))
print("Errors:", r.get("errors"))
print("Dialogue turns:", r.get("dialogue_turns"))
for k in ["simulated_playback", "long_response_summary", "scenario"]:
    if k in r:
        print(f"{k}:", r[k])

responses = r.get("responses") or {}
print(f"Total responses in dict: {len(responses)}")
for resp_id, resp in responses.items():
    print(f"\n--- Response {resp_id} ---")
    print("Keys:", list(resp.keys()))
    print("Action:", resp.get("action") or resp.get("generation_trace", {}).get("conditioned_action"))
    print("Terminal reason:", resp.get("terminal_reason"))
    print("Completed:", resp.get("completed"))
    print("Cancelled:", resp.get("cancelled"))

