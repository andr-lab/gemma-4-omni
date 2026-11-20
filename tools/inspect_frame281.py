import subprocess, json

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
for line in proc.stdout.splitlines() + proc.stderr.splitlines():
    if "duplex_control6.trace" in line:
        try:
            # find JSON in line
            idx = line.find("{")
            if idx != -1:
                data = json.loads(line[idx:])
                fid = data.get("frame_id")
                if fid in (279, 280, 281, 282):
                    print(f"Frame {fid}: action={data.get('action')}, selected_action={data.get('selected_action')}, native_action={data.get('native_action')}, prob={data.get('native_action_probability')}, early_gated={data.get('native_control6_early_speak_gated')}, early_conf={data.get('native_control6_early_speak_confidence')}, hyst_applied={data.get('native_control6_hysteresis_applied_action')}, hyst_count={data.get('native_control6_hysteresis_count')}, endpoint_ready={data.get('native_control6_endpoint_speak_ready')}, endpoint_released={data.get('native_control6_endpoint_speak_released')}")
        except Exception:
            pass
