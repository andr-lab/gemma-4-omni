import subprocess, json

proc = subprocess.run(["docker", "logs", "gemma4-listener-attempt9-implementation-fixes-20261024"], capture_output=True, text=True)
print("Trace for frame 1111 and 1112:")
for line in proc.stdout.splitlines() + proc.stderr.splitlines():
    if "duplex_control6.trace" in line and ('"frame_id": 1111' in line or '"frame_id": 1112' in line):
        idx = line.find("{")
        if idx != -1:
            data = json.loads(line[idx:])
            for k in [
                "frame_id", "listener_batch_compute_ms", "listener_batch_queue_wait_ms",
                "listener_batch_native_cuda_ms", "listener_batch_control6_policy_ms",
                "listener_batch_native_submit_ms", "listener_batch_decision_view_ms",
                "listener_batch_audio_tower_cuda_ms", "scheduler_listener_callback_ms",
                "scheduler_commit_state_lock_hold_ms", "scheduler_commit_state_lock_wait_ms"
            ]:
                print(f"  {k}: {data.get(k)}")
