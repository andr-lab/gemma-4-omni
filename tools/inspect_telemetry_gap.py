import json

telemetry_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/runtime_telemetry.json"
with open(telemetry_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Telemetry keys:", data.keys())
samples = data.get("samples") or data.get("telemetry_samples") or []
if isinstance(samples, list):
    print(f"Total samples: {len(samples)}")
    for s in samples:
        utc = s.get("utc", "")
        if any(ts in utc for ts in ["20:19:03", "20:19:04", "20:19:05"]):
            print(f"[{utc}] {s}")
