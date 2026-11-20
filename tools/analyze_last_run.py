import glob
import json
import os

def main():
    dirs = glob.glob("/workspace/artifacts/acceptance_suite_*/run_01") or glob.glob("/home/administrator/gemma-4-omni/artifacts/acceptance_suite_*/run_01")
    dirs.sort(key=lambda d: os.path.getmtime(d))
    run_dir = dirs[-1]
    print(f"Inspecting run_dir: {run_dir}")
    
    rep_path = os.path.join(run_dir, "report.finalized.json")
    if not os.path.exists(rep_path):
        rep_path = os.path.join(run_dir, "report.json")
    
    with open(rep_path, "r", encoding="utf-8") as f:
        r = json.load(f)
    
    with open(os.path.join(run_dir, "report.json"), "r", encoding="utf-8") as f:
        rep_prelim = json.load(f)
    with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    
    with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    created_rows = [row for row in rows if row.get("type") == "response.created"]
    created_by_id = {}
    for row in created_rows:
        event = row.get("payload", {})
        rid = event.get("response_id")
        if rid:
            created_by_id[rid] = event
            
    assignment_rows = [row for row in rows if row.get("type") == "scenario_response_assigned"]
    assignments = {}
    for row in assignment_rows:
        payload = row.get("payload", {})
        rid = payload.get("response_id")
        if rid:
            assignments[rid] = payload
            
    with open(os.path.join(run_dir, "report.json"), "r", encoding="utf-8") as f:
        rep = json.load(f)
    rep_responses = rep.get("scenario", {}).get("responses", [])
    response_reports = {r.get("response_id"): r for r in rep_responses if r.get("response_id")}
    
    print("created_by_id keys:", list(created_by_id.keys()))
    print("assignments keys:", list(assignments.keys()))
    for rid, a in assignments.items():
        print(f"  Assignment {rid}: tag={a.get('scenario_tag')}")
    print("response_reports keys:", list(response_reports.keys()))
    import sys
    sys.path.insert(0, "/home/administrator/gemma-4-omni")
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/tools")

    from tools.real_dialogue_simulator import _reconcile_saved_run, _extract_kv_evidence_from_rows
    
    with open(os.path.join(run_dir, "report.json"), "r", encoding="utf-8") as f:
        rep_test = json.load(f)
    with open(os.path.join(run_dir, "events.jsonl"), "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
        
    recomputed_kv = _extract_kv_evidence_from_rows(rows)
    # Simulate what the new live run will produce in report.json:
    rep_test["runtime_evidence"]["kv_events"] = recomputed_kv
    
    import wave
    import numpy as np
    
    wav_path = os.path.join(run_dir, "response_resp_6d211852a220_received.wav")
    if not os.path.exists(wav_path):
        wavs = glob.glob(os.path.join(run_dir, "*_received.wav"))
        wav_path = wavs[0] if wavs else None
    print(f"Reading wav: {wav_path}")
    if wav_path and os.path.exists(wav_path):
        with wave.open(wav_path, "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
        print("Samples:", len(pcm))
        print("Max abs:", float(np.max(np.abs(pcm))))
        print("Overall RMS:", float(np.sqrt(np.mean(pcm**2))))
        # Check 80ms framed RMS (at 24000 Hz, 80ms is 1920 samples)
        frame_len = 1920
        frame_rms = [float(np.sqrt(np.mean(pcm[i:i+frame_len]**2))) for i in range(0, len(pcm)-frame_len, frame_len)]
        vocal = sum(r >= 0.015 for r in frame_rms)
        print("80ms framed RMS vocal count (>=0.015):", vocal, "out of", len(frame_rms))
        print("Max frame RMS:", max(frame_rms) if frame_rms else 0)
        print("First 10 frame RMS:", frame_rms[:10])
    
    old_events_path = "/workspace/artifacts/acceptance_suite_20261029T150705916984Z/run_01/events.jsonl"
    with open(old_events_path, "r", encoding="utf-8") as f:
        old_rows = [json.loads(line) for line in f if line.strip()]
    new_done = [r for r in rows if r.get("type") == "response.done"]
    if new_done:
        pl = new_done[0].get("payload", {})
        gen_trace = pl.get("runtime_metrics", {}).get("generation_trace", {})
        print("generation_trace keys:", list(gen_trace.keys()))
        for k, v in gen_trace.items():
            if k not in ["thinker_token_ids", "native_q1_steps"]:
                print(f"  {k}: {v}")





















if __name__ == "__main__":
    main()

