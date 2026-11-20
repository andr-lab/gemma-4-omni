import json
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T234523994806Z/run_02"
report_path = f"{run_dir}/report.finalized.json"
events_path = f"{run_dir}/events.jsonl"

with open(report_path, "r") as f:
    rep = json.load(f)
long_win = rep.get("scenario", {}).get("long_response_input_window", {})
start_ns = long_win.get("microphone_continuity_start_ns", 0)
end_ns = long_win.get("microphone_continuity_end_ns", 0)
cutoff_ns = long_win.get("input_ack_cutoff_ns", 0)

sends = {}
acks = []
with open(events_path, "r") as f:
    for line in f:
        e = json.loads(line)
        t = e.get("type")
        p = e.get("payload", {})
        fid = p.get("local_frame_id")
        ns = e.get("monotonic_ns", p.get("sent_ns"))
        if t == "input_frame_sent" and start_ns <= ns <= end_ns:
            sends[fid] = ns
        elif t == "input_frame_ack_received":
            acks.append((ns, fid))

window_acks = [a for a in acks if a[1] in sends and start_ns <= a[0] <= cutoff_ns]
window_acks.sort(key=lambda x: x[0])

print(f"Total sends in window: {len(sends)}, total acks: {len(window_acks)}")
print("Last 10 acks in window:")
for i in range(max(1, len(window_acks)-10), len(window_acks)):
    prev_a = window_acks[i-1]
    curr_a = window_acks[i]
    dt_ms = (curr_a[0] - prev_a[0]) / 1e6
    print(f"  Frame {prev_a[1]} -> {curr_a[1]}: dt = {dt_ms:.2f} ms")
