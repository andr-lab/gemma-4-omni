import json
import sys

path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

tel = d.get("runtime_evidence", {}).get("telemetry", {})
print("path:", path)
print("status:", tel.get("status"))
print("within_budget:", tel.get("within_budget"))
print("allocated_peak_growth:", tel.get("allocated_peak_growth_over_post_warmup_mib"))
print("max_allocated_post_drain_growth:", tel.get("max_allocated_post_drain_growth_over_post_warmup_mib"))
print("reserved_peak_growth:", tel.get("reserved_peak_growth_over_post_warmup_mib"))
print("max_reserved_post_drain_growth:", tel.get("max_reserved_post_drain_growth_over_post_warmup_mib"))
