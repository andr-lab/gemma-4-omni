import json
import wave
import sys

wav_path = "/workspace/work/gpu_forensics/fixtures_16k_final/long_backpressure_prompt.wav"
with wave.open(wav_path, "rb") as w:
    frames = w.getnframes()
    rate = w.getframerate()
    print(f"WAV frames: {frames}, rate: {rate}, sec: {frames/rate:.2f}, 80ms_frames: {frames/(rate*0.08):.2f}")

events_path = "/workspace/artifacts/acceptance_suite_verify_10/run_01/events.jsonl"
responses = {}
with open(events_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        t = ev.get("type")
        payload = ev.get("payload", {})
        rid = payload.get("response_id") or ev.get("response_id")
        if rid:
            responses.setdefault(rid, []).append((line_idx, ev.get("utc"), t, payload))

print(f"\nTotal distinct response IDs: {len(responses)}")
for rid, ev_list in responses.items():
    types = [x[2] for x in ev_list]
    first_time = ev_list[0][1]
    last_time = ev_list[-1][1]
    print(f"\nResponse {rid}: {len(ev_list)} events, first={first_time}, last={last_time}")
    print(f"  Event types summary: {set(types)}")
    for lidx, utc, etype, payload in ev_list:
        if etype in {"response.created", "response.done", "response.cancelled", "response.unified_native_finalized", "scenario_response_assigned"}:
            extra = {}
            if etype == "response.created":
                extra = {"action": payload.get("action"), "frame_id": payload.get("listener_hidden_frame_id")}
            elif etype == "response.unified_native_finalized":
                extra = {"status": payload.get("status"), "term_reason": payload.get("runtime_metrics", {}).get("generation_trace", {}).get("terminal_reason")}
            elif etype == "response.done":
                extra = {"finish_reason": payload.get("finish_reason")}
            elif etype == "scenario_response_assigned":
                extra = {"scenario": payload.get("scenario_tag")}
            print(f"    line {lidx} [{utc}] {etype}: {extra}")
