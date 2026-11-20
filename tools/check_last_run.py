import json
import sys

path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T133946017662Z/run_01/report.finalized.json"
if len(sys.argv) > 1:
    path = sys.argv[1]

with open(path) as f:
    d = json.load(f)

print("=== FINAL REPORT SUMMARY ===")
print("overall_status:", d.get("overall_status"))
print("errors:", d.get("errors"))
print("evidence_gaps:", d.get("evidence_gaps"))

for section in ["gates", "scenario", "dialogue", "vram", "pacing", "telemetry"]:
    if section in d:
        print(f"\n--- {section} ---")
        val = d[section]
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, (dict, list)):
                    print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:120]}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {val}")

if "turns" in d:
    print("\n--- turns ---")
    for k, v in d["turns"].items():
        print(f"  turn {k}: status={v.get('status')} vocal_frames={v.get('vocal_frames')} errors={v.get('errors')}")

print("\n--- Top-level keys ---")
for k in d.keys():
    v = d[k]
    if isinstance(v, dict) and "status" in v:
        print(f"  key {k}: status={v.get('status')}")


