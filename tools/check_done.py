import json

events_path = "/workspace/artifacts/acceptance_suite_verify_11/run_01/events.jsonl"
with open(events_path, "r", encoding="utf-8") as f:
    for line in f:
        if '"response.done"' in line:
            ev = json.loads(line)
            p = ev.get("payload", {})
            print("response.done tokens:", p.get("runtime_metrics", {}).get("response_completion"))
            print("finish_reason:", p.get("finish_reason"))
            print("audio_samples:", p.get("runtime_metrics", {}).get("response_completion", {}).get("codec_frames_produced"))
