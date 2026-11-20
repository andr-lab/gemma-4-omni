import json

events_path = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T135715955563Z/run_02/events.jsonl"

prev_ack_mono = None

with open(events_path) as f:
    for line in f:
        row = json.loads(line)
        payload = row.get("payload", {})
        et = row.get("type")
        mono = row.get("monotonic_ns")
        
        if et == "input_audio_buffer.consumed":
            cfid = payload.get("frame_id")
            gap_prev = (mono - prev_ack_mono) / 1e6 if prev_ack_mono else 0
            if cfid is not None and 775 <= cfid <= 785:
                print(f"ACK frame_id={cfid} mono={mono} gap_from_prev_ack_ms={gap_prev:.2f} listener_q_ms={payload.get('server_listener_queue_ms')} step_ms={payload.get('server_listener_step_ms')} ack_q_ms={payload.get('server_ack_queue_ms')}")
            prev_ack_mono = mono
        elif et == "input_audio_buffer.append":
            afid = payload.get("frame_id")
            if afid is not None and 775 <= afid <= 785:
                print(f"SEND frame_id={afid} mono={mono}")
        elif et not in {"input_audio_buffer.append", "response.audio.delta"}:
            if mono and 6153007100000000 <= mono <= 6153007450000000:
                print(f"OTHER EVENT: type={et} dir={row.get('direction')} mono={mono}")
