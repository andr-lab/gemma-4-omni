import json

base = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T205147331301Z/run_01"
with open(f"{base}/events.jsonl") as f:
    for line in f:
        ev = json.loads(line)
        if ev.get("type") == "input_audio_buffer.consumed" and ev.get("payload", {}).get("frame_id") == 664:
            p = ev["payload"]
            tl = p.get("listener_batch_host_lock_timeline", {})
            events = tl.get("events", [])
            print(f"Total events in timeline: {len(events)}")
            for e in events:
                print(f"  ticket={e.get('ticket')} seq={e.get('trace_seq')} ev={e.get('event')} prio={e.get('priority')} wait_ms={e.get('wait_ns', 0)/1e6 if e.get('wait_ns') else 0:.3f} hold_ms={e.get('hold_ns', 0)/1e6 if e.get('hold_ns') else 0:.3f} pend={e.get('pending_counts')}")
