import json

events_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/events.jsonl"
print("Thinker/Talker events between 20:19:03.5 and 20:19:05.5:")
with open(events_path, "r", encoding="utf-8") as f:
    for line in f:
        if any(ts in line for ts in ["20:19:03.7", "20:19:03.8", "20:19:03.9", "20:19:04", "20:19:05.0", "20:19:05.1", "20:19:05.2"]):
            ev = json.loads(line)
            t = ev.get("type")
            if t in {"response.output_text.delta", "response.audio.delta", "response.thinker_metrics"}:
                print(f"[{ev.get('utc')}] {t}")
