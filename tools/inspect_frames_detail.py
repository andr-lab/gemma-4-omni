import json
import sys

run_dir = sys.argv[1]
fids = [int(x) for x in sys.argv[2:]]

events_file = f"{run_dir}/events.jsonl"
with open(events_file) as f:
    for line in f:
        e = json.loads(line)
        p = e.get("payload", {})
        fid = p.get("local_frame_id") or p.get("frame_id") or e.get("frame_id")
        if fid in fids:
            print(f"EVENT: {e.get('type')} fid={fid} ts={e.get('monotonic_ns') or e.get('t_monotonic_ns')}")
            print(json.dumps(e, indent=2))


