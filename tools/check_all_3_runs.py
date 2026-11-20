import glob
import json
import os
import sys

suite_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261103T232958973451Z"

for r in [1, 2, 3]:
    p = os.path.join(suite_dir, f"run_{r:02d}", "report.finalized.json")
    if not os.path.exists(p):
        print(f"=== RUN {r} NOT FOUND: {p} ===")
        continue
    with open(p) as f:
        d = json.load(f)
    print(f"=== RUN {r} ===")
    print("  overall_status:", d.get("overall_status"))
    print("  errors:", d.get("errors"))
    print("  evidence_gaps:", d.get("evidence_gaps"))
    lr = d.get("scenario", {}).get("long_response_input_window", {})
    print("  max_input_ack_gap_ms:", lr.get("max_input_ack_gap_ms"))
    print("  ack_gap_validation_status:", lr.get("validation_status"))
    tel = d.get("runtime_evidence", {}).get("telemetry", {})
    print("  telemetry_status:", tel.get("status"))
    print("  within_budget:", tel.get("within_budget"))
    print("  vram_post_drain_growth_mib:", tel.get("max_allocated_post_drain_growth_over_post_warmup_mib"))
    print("  vram_reserved_growth_mib:", tel.get("max_reserved_post_drain_growth_over_post_warmup_mib"))

