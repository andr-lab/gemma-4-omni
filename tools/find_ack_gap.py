import json
import sys

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T221138714355Z/run_02"
events_file = f"{run_dir}/events.jsonl"

acks = []

with open(events_file, "r") as f:
    for line in f:
        e = json.loads(line)
        if e.get("type") == "input_frame_ack_received":
            p = e.get("payload", {})
            fid = p.get("local_frame_id")
            acks.append((e.get("monotonic_ns"), fid))

acks.sort(key=lambda x: x[0])

gaps = []
for i in range(1, len(acks)):
    dt_ms = (acks[i][0] - acks[i-1][0]) / 1e6
    gaps.append((dt_ms, acks[i-1][1], acks[i][1], acks[i-1][0], acks[i][0]))

gaps.sort(reverse=True)
print("Top 10 consecutive ACK arrival gaps (ms):")
for dt_ms, f_prev, f_curr, ns_prev, ns_curr in gaps[:10]:
    print(f"Gap {dt_ms:.2f}ms between Frame {f_prev} and Frame {f_curr}")
