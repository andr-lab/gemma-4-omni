import json
import sys

path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T133946017662Z/run_01/report.finalized.json"
if len(sys.argv) > 1:
    path = sys.argv[1]

with open(path) as f:
    d = json.load(f)

print("OVERALL_STATUS:", d.get("overall_status"))
print("ERRORS:", d.get("errors"))
print("EVIDENCE GAPS:", d.get("evidence_gaps"))

# Check telemetry & VRAM
rt = d.get("runtime_telemetry", {})
print("\n--- RUNTIME TELEMETRY / VRAM ---")
print("telemetry status:", rt.get("status"))
print("telemetry errors:", rt.get("errors"))
print("telemetry evidence gaps:", rt.get("evidence_gaps"))
for k in ["baseline_allocated_bytes", "baseline_reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes",
          "peak_allocated_growth_mib", "peak_reserved_growth_mib", "post_drain_allocated_growth_mib",
          "post_drain_reserved_growth_mib", "vram_peak_over_baseline_mib", "vram_post_drain_growth_mib",
          "vram_reserved_peak_growth_mib", "vram_reserved_post_drain_growth_mib"]:
    if k in rt:
        print(f"  {k}: {rt[k]}")

# Check pacing
pm = d.get("pacing", {})
print("\n--- PACING ---")
print("pacing status:", pm.get("status"))
print("max_lateness_ms:", pm.get("max_lateness_ms"))
print("p95_lateness_ms:", pm.get("p95_lateness_ms"))
print("max_interframe_gap_ms:", pm.get("max_interframe_gap_ms"))
print("max_net_interframe_gap_ms:", pm.get("max_net_interframe_gap_ms"))

# Check scenario responses
sc = d.get("scenario", {})
print("\n--- SCENARIO RESPONSES ---")
print("normal_responses:", sc.get("normal_response_count"))
print("cancelled_responses:", sc.get("cancelled_response_count"))
print("long_completed:", sc.get("long_completed_response_count"))
for r in sc.get("responses", []):
    print(f"  resp {r.get('response_id')}: tag={r.get('scenario_tag')} status={r.get('status')} duration={r.get('duration_s', r.get('audio_duration_seconds'))}")
