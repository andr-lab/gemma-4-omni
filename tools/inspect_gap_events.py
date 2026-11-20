import sys, json

run_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T234523994806Z/run_01"
events_path = f"{run_dir}/events.jsonl"
print(f"Events around frame 1135-1136 in {events_path}:")
with open(events_path, "r", encoding="utf-8") as f:
    for line in f:
        ev = json.loads(line)
        utc = ev.get("utc", "")
        if "2026-11-02T23:46:57.6" <= utc <= "2026-11-02T23:46:58.2":
            t = ev.get("type")
            p = ev.get("payload", {})
            fid = p.get("local_frame_id") or p.get("frame_id")
            rid = p.get("response_id")
            print(f"[{utc}] {t:35s} fid={str(fid):5s} rid={str(rid):20s}")
