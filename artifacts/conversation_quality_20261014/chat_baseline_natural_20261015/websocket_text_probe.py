"""Three-turn manual text probe for the live realtime WebSocket service.

Run inside the serving container. It uses only WebSocket conversation state:
each user message is appended once, then response.create asks the Thinker to
answer from that session history. The server currently synthesizes audio even
for this actionless text comparison, so audio deltas are counted but omitted.
"""

from __future__ import annotations

import asyncio
import json
import time

import websockets


URL = "ws://127.0.0.1:8000/v1/realtime?duplex=1"
SYSTEM = (
    "Rispondi in italiano naturale e conciso, con al massimo due frasi. "
    "Usa solo le informazioni dette nella conversazione e rispetta i vincoli espliciti."
)
TURNS = [
    "Ciao, mi chiamo Elena. Sto organizzando un fine settimana a Torino con un budget iniziale di 120 euro. Voglio considerare soltanto attività gratuite. Che cosa potrei fare sabato?",
    "Preferisco passeggiare e mi interessa la storia. Ricorda queste preferenze: passeggiate e storia. Per sabato, come posso abbinarle mantenendo soltanto attività gratuite?",
    "Correzione importante: il mio budget massimo ora è 80 euro, non 120. Riepiloga il mio nome, la città, il budget aggiornato, il vincolo sulle attività e la preferenza che ti ho detto; poi dammi una proposta gratuita adatta.",
]


async def main() -> None:
    result: dict = {
        "url": URL,
        "session_config": {"auto_response": False, "instructions": SYSTEM},
        "turns": [],
        "events": [],
    }
    async with websockets.connect(
        URL, max_size=2_000_000, ping_interval=20, ping_timeout=20
    ) as ws:

        async def recv(timeout: float = 180) -> dict:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
            record = {"t": round(time.monotonic(), 3), "type": event.get("type")}
            for key in (
                "response_id", "action", "effective_action", "status",
                "output_index", "content_index", "delta", "text", "error", "details",
            ):
                if key in event:
                    value = event[key]
                    if isinstance(value, str) and len(value) > 3000:
                        value = value[:3000]
                    record[key] = value
            if event.get("type") == "response.audio.delta":
                record["delta_chars"] = len(event.get("delta", ""))
                record["sample_rate"] = event.get("sample_rate")
                record["packet_id"] = event.get("packet_id")
                record.pop("delta", None)
            result["events"].append(record)
            return event

        result["session_created"] = await recv()
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {"instructions": SYSTEM, "auto_response": False},
        }))
        updated = await recv()
        if updated.get("type") != "session.updated":
            result["setup_failure"] = updated
            print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
            return
        result["session_updated"] = {
            key: value for key, value in updated.items() if key != "runtime_metrics"
        }

        for index, user_text in enumerate(TURNS, 1):
            started = time.monotonic()
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": user_text},
            }, ensure_ascii=False))
            ack = await recv(timeout=30)
            if ack.get("type") != "conversation.item.created":
                result["turns"].append({
                    "turn": index, "user": user_text,
                    "failure": "no conversation.item.created acknowledgement",
                    "event": ack,
                })
                break

            # No duplicate history is sent here; the session already owns it.
            await ws.send(json.dumps({
                "type": "response.create", "response": {"modalities": ["text"]}
            }))
            output_text = ""
            audio_transcript = ""
            audio_chunks = 0
            response_id = None
            event_types: list[str] = []
            done = False
            try:
                while True:
                    event = await recv(timeout=180)
                    event_type = event.get("type")
                    event_types.append(event_type)
                    if event_type == "response.created":
                        response_id = event.get("response_id")
                    elif event_type == "response.output_text.delta":
                        output_text += event.get("delta", "")
                    elif event_type == "response.audio_transcript.delta":
                        audio_transcript += event.get("delta", "")
                    elif event_type == "response.audio.delta":
                        audio_chunks += 1
                    elif event_type == "response.done":
                        done = True
                        break
                    elif event_type == "error":
                        break
            except asyncio.TimeoutError:
                pass

            turn = {
                "turn": index,
                "user": user_text,
                "assistant_output_text": output_text,
                "audio_transcript": audio_transcript,
                "response_id": response_id,
                "audio_chunks": audio_chunks,
                "response_done": done,
                "elapsed_s": round(time.monotonic() - started, 3),
                "event_types": event_types,
            }
            result["turns"].append(turn)
            if not done:
                turn["client_wait_or_server_generation"] = (
                    "timeout waiting for response.done; inspect event_types"
                )
                break

        await ws.send(json.dumps({"type": "session.close"}))
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    asyncio.run(main())
