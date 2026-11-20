import json
import sys

run_dir = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T234523994806Z/run_03"
events_file = f"{run_dir}/events.jsonl"

with open(events_file, "r") as f:
    for line in f:
        e = json.loads(line)
        p = e.get("payload", {})
        fid = p.get("local_frame_id") if p.get("local_frame_id") is not None else p.get("frame_id")
        if fid in [1173, 1174, 1175]:
            t = e.get("type")
            ns = e.get("monotonic_ns", p.get("sent_ns"))
            print(f"{e.get('utc')} | {t:30s} | fid={fid} | monotonic_ns={ns}")
            if t == "input_audio_buffer.consumed":
                for k in ["server_listener_queue_ms", "server_listener_step_ms", "listener_batch_compute_ms", "listener_batch_native_cuda_ms", "scheduler_commit_state_lock_hold_ms", "server_ack_queue_ms", "server_ack_lock_wait_ms"]:
                    print(f"   {k}: {p.get(k)}")
