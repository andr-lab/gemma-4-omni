import json
import sys

log_path = sys.argv[1]
with open(log_path, encoding="utf-8", errors="replace") as f:
    for line in f:
        if "[duplex_control6.trace]" in line:
            idx = line.find("{")
            if idx >= 0:
                try:
                    data = json.loads(line[idx:])
                except Exception:
                    continue
                fid = data.get("frame_id")
                if fid and fid >= 2400:
                    rms = data.get("native_control6_user_rms", 0)
                    act = data.get("action")
                    cand = data.get("native_control6_effective_candidate")
                    prob = data.get("native_control6_selected_effective_probability", 0)
                    act_det = data.get("native_control6_user_activity_detected")
                    fallback = data.get("policy_fallback_reason")
                    if rms > 0.005 or act != "CONTINUE" or cand != "CONTINUE":
                        print(f"Frame {fid}: act={act}, cand={cand}, prob={prob:.3f}, rms={rms:.4f}, user_act={act_det}, fallback={fallback}")
