from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import wave
from pathlib import Path
from math import gcd

import numpy as np
import scipy.signal
import websockets


ROOT = Path("/workspace")
INPUT_PATH = ROOT / "artifacts/conversation_quality_20261014/inputs/pair_seed_it.wav"
OUTPUT = ROOT / "artifacts/conversation_quality_20261021/control6_fullab3000_build14_stableturn_q2_context_h3000_eos_20261021"
WS_URL = "ws://127.0.0.1:8000/v1/realtime?duplex=1"
RATE_IN = 16_000
RATE_OUT = 24_000
FRAME_IN = 1_280
PROMPT = (
    "Sei un assistente conversazionale italiano. Ascolta attentamente e rispondi in modo naturale e conciso. "
    "Se non hai capito, chiedi chiarimenti. Non inventare informazioni."
)


def read_audio() -> np.ndarray:
    with wave.open(str(INPUT_PATH), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        if wav.getsampwidth() != 2 or channels not in (1, 2):
            raise ValueError(f"expected mono/stereo PCM16 WAV: {INPUT_PATH}")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if rate != RATE_IN:
        divisor = gcd(rate, RATE_IN)
        pcm = scipy.signal.resample_poly(pcm, RATE_IN // divisor, rate // divisor).astype(np.float32)
    return np.clip(pcm, -1.0, 1.0)


def encode_pcm16(pcm: np.ndarray) -> str:
    packed = np.rint(np.clip(pcm, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def decode_delta(event: dict) -> np.ndarray:
    encoded = event.get("delta")
    if not isinstance(encoded, str):
        raise ValueError("audio delta is missing base64 payload")
    with wave.open(io.BytesIO(base64.b64decode(encoded, validate=True)), "rb") as wav:
        if wav.getframerate() != RATE_OUT or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("unexpected PCM delta WAV format")
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").copy()


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUT / "eos_smoke_events.jsonl"
    summary_path = OUTPUT / "eos_smoke_summary.json"
    out_wav = OUTPUT / "eos_smoke_pcm.wav"
    events = []
    segments: dict[tuple[str, int, str], dict] = {}
    pcm_chunks: list[np.ndarray] = []
    ack_sequence = 0
    completed = False
    errors = []
    started = time.monotonic()
    with events_path.open("w", encoding="utf-8") as event_file:
        async with websockets.connect(WS_URL, max_size=1_000_000, max_queue=8) as ws:
            created = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if created.get("type") != "session.created":
                raise RuntimeError(f"expected session.created, got {created!r}")
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": "gemma",
                    "instructions": PROMPT,
                    "auto_response": True,
                    "turn_policy": "gemma_native_control6",
                    "input_audio_buffer_acknowledgements": True,
                    "input_audio_buffer_conditioning_acknowledgements": True,
                },
            }, separators=(",", ":")))
            updated = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            if updated.get("type") != "session.updated":
                raise RuntimeError(f"expected session.updated, got {updated!r}")

            pcm = read_audio()
            input_start = time.monotonic()
            for start in range(0, pcm.size, FRAME_IN):
                frame_index = start // FRAME_IN
                due = input_start + frame_index * 0.080
                await asyncio.sleep(max(0.0, due - time.monotonic()))
                frame = np.zeros(FRAME_IN, dtype=np.float32)
                part = pcm[start : start + FRAME_IN]
                frame[: part.size] = part
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": encode_pcm16(frame),
                }, separators=(",", ":")))
            await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            deadline = time.monotonic() + 180
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    errors.append({"kind": "timeout", "seconds": 180})
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    errors.append({"kind": "timeout", "seconds": 180})
                    break
                event = json.loads(raw)
                event["received_monotonic_ns"] = time.monotonic_ns()
                events.append(event)
                event_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                event_file.flush()
                kind = event.get("type")
                if kind == "response.audio.delta":
                    key = (
                        str(event.get("response_id", "")),
                        int(event.get("generation_id", -1)),
                        str(event.get("speaker_segment_id", "")),
                    )
                    chunk = decode_delta(event)
                    segments.setdefault(key, {"samples": 0, "done": None, "acked": False})
                    segments[key]["samples"] += int(chunk.size)
                    pcm_chunks.append(chunk)
                elif kind == "response.audio.segment_done":
                    key = (
                        str(event.get("response_id", "")),
                        int(event.get("generation_id", -1)),
                        str(event.get("speaker_segment_id", "")),
                    )
                    segment = segments.setdefault(key, {"samples": 0, "done": None, "acked": False})
                    segment["done"] = event
                    expected = int(event.get("end_sample", -1)) - int(event.get("start_sample", -1))
                    if expected <= 0 or segment["samples"] != expected:
                        errors.append({"kind": "segment_pcm_range_mismatch", "key": key,
                                       "expected": expected, "actual": segment["samples"]})
                        break
                    ack_sequence += 1
                    await ws.send(json.dumps({
                        "type": "client_consumed_pcm",
                        "response_id": key[0],
                        "generation_id": key[1],
                        "speaker_segment_id": key[2],
                        "start_sample": int(event["start_sample"]),
                        "end_sample": int(event["end_sample"]),
                        "sample_rate": RATE_OUT,
                        "sequence": ack_sequence,
                        "client_monotonic_ns": time.monotonic_ns(),
                        "consumption_semantics": "logical_client_device_buffer",
                    }, separators=(",", ":")))
                    segment["acked"] = True
                elif kind == "response.done":
                    completed = True
                    break
                elif kind == "error":
                    errors.append(event)
                    break

    combined = np.concatenate(pcm_chunks) if pcm_chunks else np.empty(0, dtype="<i2")
    if combined.size:
        with wave.open(str(out_wav), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE_OUT)
            wav.writeframes(combined.tobytes())
    summary = {
        "status": "passed" if completed and not errors and pcm_chunks and all(v["acked"] for v in segments.values()) else "failed",
        "elapsed_seconds": time.monotonic() - started,
        "input_samples": int(read_audio().size),
        "event_count": len(events),
        "response_done": completed,
        "errors": errors,
        "audio_delta_count": sum(1 for e in events if e.get("type") == "response.audio.delta"),
        "segment_done_count": sum(1 for e in events if e.get("type") == "response.audio.segment_done"),
        "acked_segment_count": sum(1 for v in segments.values() if v["acked"]),
        "pcm_sample_count": int(combined.size),
        "pcm_seconds": float(combined.size / RATE_OUT),
        "events_path": str(events_path),
        "pcm_path": str(out_wav) if combined.size else None,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    asyncio.run(main())
