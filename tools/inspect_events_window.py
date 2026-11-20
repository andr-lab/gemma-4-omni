import json
import sys

run_dir = sys.argv[1]
ts_start = int(sys.argv[2])
ts_end = int(sys.argv[3])

with open(f"{run_dir}/events.jsonl") as f:
    for line in f:
        e = json.loads(line)
        ts = e.get("monotonic_ns") or e.get("t_monotonic_ns") or 0
        if ts_start - 20_000_000 <= ts <= ts_end + 20_000_000:
            print(f"EVENT: {e.get('type')} ts={ts} dt_ms={(ts - ts_start)/1e6:.2f}")
            p = e.get("payload", {})
            for k in ["local_frame_id", "server_frame_id", "response_id", "status", "action", "effective_action", "item_type"]:
                if k in p:
                    print(f"    {k}: {p[k]}")
