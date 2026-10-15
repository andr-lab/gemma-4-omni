"""Repeatable two-turn direct-audio Thinker probe for the Mondaini service.

Manual mode commits each complete WAV with auto_response disabled, so the
controller's action timing is separated from full-utterance Thinker context.
Automatic mode adds two 80 ms silence frames after each WAV, then streams
silence at 80 ms cadence until a SPEAK response completes or the limit is
reached. It also reconstructs each received response.audio.delta WAV packet
into a per-response output file. Both modes use one WebSocket session and one
serving model.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import math
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import websockets


REPO_ROOT = next(
    parent for parent in Path(__file__).resolve().parents
    if (parent / "artifacts" / "conversation_quality_20261014" / "inputs").is_dir()
)
INPUT_DIR = REPO_ROOT / "artifacts" / "conversation_quality_20261014" / "inputs"
DEFAULT_FIRST = INPUT_DIR / "clean_initial.wav"
DEFAULT_FOLLOWUP = INPUT_DIR / "clean_followup.wav"
URL = "ws://127.0.0.1:8000/v1/realtime?duplex=1"
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
FRAME_MS = 80
SYSTEM = (
    "Rispondi in italiano naturale e conciso, con al massimo due frasi. "
    "Usa solo le informazioni dette nella conversazione e rispetta i vincoli espliciti."
)


def read_audio_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels, rate, width = wav.getnchannels(), wav.getframerate(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2 or channels not in (1, 2):
        raise ValueError(f"expected mono/stereo PCM16 WAV: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        pcm = scipy.signal.resample_poly(
            pcm, SAMPLE_RATE // divisor, rate // divisor
        ).astype(np.float32)
    return np.clip(pcm, -1.0, 1.0)


def to_frames(pcm: np.ndarray, silence_frames: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for start in range(0, pcm.size, FRAME_SAMPLES):
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        chunk = pcm[start:start + FRAME_SAMPLES]
        frame[:chunk.size] = chunk
        frames.append(frame)
    frames.extend(np.zeros(FRAME_SAMPLES, dtype=np.float32) for _ in range(silence_frames))
    return frames


def decode_audio_delta_wav(event: dict[str, Any]) -> tuple[tuple[int, int, int], bytes]:
    """Decode one response.audio.delta WAV packet into raw PCM frames."""
    payload = base64.b64decode(str(event.get("delta", "")), validate=True)
    with wave.open(io.BytesIO(payload), "rb") as packet:
        params = (packet.getnchannels(), packet.getsampwidth(), packet.getframerate())
        if packet.getcomptype() != "NONE":
            raise ValueError(f"unsupported response audio compression: {packet.getcomptype()}")
        if event.get("sample_rate") is not None and int(event["sample_rate"]) != params[2]:
            raise ValueError(
                f"response audio sample-rate mismatch: event={event['sample_rate']} wav={params[2]}"
            )
        return params, packet.readframes(packet.getnframes())


def write_response_audio_wav(path: Path, params: tuple[int, int, int], chunks: list[bytes]) -> None:
    channels, sample_width, sample_rate = params
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(chunks))


def pcm16_bytes(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("manual", "auto"), required=True)
    parser.add_argument("--url", default=URL)
    parser.add_argument("--first", type=Path, default=DEFAULT_FIRST)
    parser.add_argument("--followup", type=Path, default=DEFAULT_FOLLOWUP)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--silence-frames", type=int, default=2)
    parser.add_argument("--max-silence-frames", type=int, default=375)
    parser.add_argument("--turn-delay-s", type=float, default=10.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "events.jsonl"
    report_path = args.output_dir / "report.json"
    result: dict[str, Any] = {
        "url": args.url,
        "mode": args.mode,
        "model": "gemma-4-omni-duplex",
        "response_protocol_expected": "controller_guided_say_v1",
        "system_prompt": SYSTEM,
        "session_config": {
            "turn_policy": "gemma_native_control6",
            "auto_response": args.mode == "auto",
            "input_audio_buffer_acknowledgements": True,
        },
        "source_transcripts": {
            "clean_initial.wav": (
                "Mi chiamo Elena. Il mio codice è 7-3-1. Vorrei visitare Torino con 120 euro. "
                "Rispondi in italiano, in non più di due frasi."
            ),
            "clean_followup.wav": (
                "Il baggettore è 80 euro. Quali sono i miei nome, codice, città e baggetta attuale?"
            ),
        },
        "inputs": {},
        "turns": [],
        "response_audio": {},
        "audio_capture_errors": [],
    }
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    response_created_event = asyncio.Event()
    audio_packets: dict[str, list[bytes]] = {}
    audio_params: dict[str, tuple[int, int, int]] = {}

    with events_path.open("w", encoding="utf-8") as event_file:
        def record(direction: str, event: dict[str, Any]) -> None:
            row = {"t_monotonic": round(time.monotonic(), 6), "direction": direction, **event}
            if event.get("type") == "response.audio.delta":
                row["delta_chars"] = len(str(row.pop("delta", "")))
            event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            event_file.flush()

        async with websockets.connect(
            args.url, max_size=2_000_000, ping_interval=20, ping_timeout=20
        ) as ws:
            first_event = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
            record("server", first_event)
            result["session_created"] = first_event

            async def reader() -> None:
                while True:
                    event = json.loads(await ws.recv())
                    record("server", event)
                    if event.get("type") == "response.created":
                        response_created_event.set()
                    await queue.put(event)

            reader_task = asyncio.create_task(reader())

            async def next_event(timeout_s: float = 30.0) -> dict[str, Any]:
                return await asyncio.wait_for(queue.get(), timeout=timeout_s)

            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "instructions": SYSTEM,
                    "voice": "gemma",
                    "turn_policy": "gemma_native_control6",
                    "auto_response": args.mode == "auto",
                    "input_audio_buffer_acknowledgements": True,
                    "input_audio_transcription": False,
                },
            }, ensure_ascii=False))
            record("client", {"type": "session.update", **result["session_config"]})
            while True:
                updated = await next_event()
                if updated.get("type") in {"session.updated", "error"}:
                    result["session_updated"] = updated
                    if updated.get("type") != "session.updated":
                        raise RuntimeError(f"session.update failed: {updated}")
                    break

            wavs = [("initial", args.first), ("followup", args.followup)]
            for turn_index, (name, path) in enumerate(wavs, 1):
                response_created_event = asyncio.Event()
                pcm = read_audio_16k(path)
                frames = to_frames(pcm, args.silence_frames if args.mode == "auto" else 0)
                result["inputs"][name] = {
                    "wav": str(path),
                    "samples_16khz": int(pcm.size),
                    "wav_duration_ms": round(pcm.size * 1000.0 / SAMPLE_RATE, 3),
                    "source_audio_frames_80ms": int(math.ceil(pcm.size / FRAME_SAMPLES)),
                    "silence_frames_appended": args.silence_frames if args.mode == "auto" else 0,
                    "total_frames_sent": len(frames),
                }
                t0 = asyncio.get_running_loop().time()
                first_sent_ns = None
                last_sent_ns = None
                for frame_index, frame in enumerate(frames):
                    due = t0 + frame_index * FRAME_MS / 1000.0
                    delay = due - asyncio.get_running_loop().time()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    sent_ns = time.monotonic_ns()
                    first_sent_ns = first_sent_ns or sent_ns
                    last_sent_ns = sent_ns
                    await ws.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(pcm16_bytes(frame)).decode("ascii"),
                    }))
                    record("client", {
                        "type": "input_audio_buffer.append",
                        "turn": name,
                        "frame_index": frame_index,
                        "sample_count": FRAME_SAMPLES,
                        "source_frame_kind": "wav" if frame_index < result["inputs"][name]["source_audio_frames_80ms"] else "silence",
                        "sent_at_ns": sent_ns,
                    })
                silence_tail_frames = 0
                if args.mode == "manual":
                    commit = {"type": "input_audio_buffer.commit"}
                    await ws.send(json.dumps(commit))
                    record("client", {**commit, "turn": name, "after_all_audio_frames": True})
                result["inputs"][name]["silence_tail_frames_until_response_created"] = silence_tail_frames
                result["inputs"][name]["total_frames_sent"] = len(frames) + silence_tail_frames

                text_by_response: dict[str, str] = {}
                transcript_by_response: dict[str, str] = {}
                responses: dict[str, dict[str, Any]] = {}
                response_actions: dict[str, str] = {}
                pending_response_ids: set[str] = set()
                silence_stop = asyncio.Event()
                silence_sender_task: asyncio.Task[Any] | None = None
                if args.mode == "auto":
                    async def send_silence_until_speak_done() -> None:
                        nonlocal silence_tail_frames
                        # The first two quiet frames are already included in
                        # `frames`; keep feeding the listener at 80 ms while
                        # waiting for the actual SPEAK response to finish.
                        max_tail = max(0, args.max_silence_frames - args.silence_frames)
                        for tail_index in range(max_tail):
                            if silence_stop.is_set():
                                break
                            due = t0 + (len(frames) + tail_index) * FRAME_MS / 1000.0
                            delay = due - asyncio.get_running_loop().time()
                            if delay > 0:
                                await asyncio.sleep(delay)
                            if silence_stop.is_set():
                                break
                            sent_ns = time.monotonic_ns()
                            await ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": base64.b64encode(
                                    pcm16_bytes(np.zeros(FRAME_SAMPLES, dtype=np.float32))
                                ).decode("ascii"),
                            }))
                            silence_tail_frames += 1
                            record("client", {
                                "type": "input_audio_buffer.append",
                                "turn": name,
                                "frame_index": len(frames) + tail_index,
                                "sample_count": FRAME_SAMPLES,
                                "source_frame_kind": "silence_until_speak_done",
                                "sent_at_ns": sent_ns,
                            })

                    silence_sender_task = asyncio.create_task(send_silence_until_speak_done())
                deadline = asyncio.get_running_loop().time() + (
                    args.max_silence_frames * FRAME_MS / 1000.0
                    if args.mode == "auto" else 90.0
                )
                speak_response_done = False
                while asyncio.get_running_loop().time() < deadline:
                    try:
                        event = await next_event(timeout_s=max(0.05, deadline - asyncio.get_running_loop().time()))
                    except asyncio.TimeoutError:
                        break
                    kind = event.get("type")
                    response_id = str(event.get("response_id", ""))
                    if kind == "response.created":
                        pending_response_ids.add(response_id)
                        responses.setdefault(response_id, {"response_created": event})
                        response_actions[response_id] = str(
                            event.get("effective_action") or event.get("action") or ""
                        ).strip().upper()
                    elif kind == "response.action_selected":
                        selected = str(
                            event.get("response_action") or event.get("wake_action") or ""
                        ).strip().upper()
                        if selected:
                            response_actions[response_id] = selected
                    elif kind == "response.output_text.delta":
                        text_by_response[response_id] = text_by_response.get(response_id, "") + str(event.get("delta", ""))
                    elif kind == "response.audio_transcript.delta":
                        transcript_by_response[response_id] = transcript_by_response.get(response_id, "") + str(event.get("delta", ""))
                    elif kind == "response.audio.delta":
                        try:
                            params, pcm_bytes = decode_audio_delta_wav(event)
                            previous = audio_params.setdefault(response_id, params)
                            if previous != params:
                                raise ValueError(
                                    f"response audio format changed: {previous} -> {params}"
                                )
                            audio_packets.setdefault(response_id, []).append(pcm_bytes)
                        except Exception as exc:
                            result["audio_capture_errors"].append({
                                "response_id": response_id,
                                "packet_id": event.get("packet_id"),
                                "error": f"{type(exc).__name__}: {exc}",
                            })
                    elif kind == "response.done":
                        entry = responses.setdefault(response_id, {})
                        entry["response_done"] = event
                        entry["assistant_text"] = text_by_response.get(response_id, "")
                        entry["audio_transcript"] = transcript_by_response.get(response_id, "")
                        pending_response_ids.discard(response_id)
                        done_status = str(event.get("status", "")).strip().lower()
                        if args.mode == "auto" and response_actions.get(response_id) == "SPEAK":
                            speak_response_done = done_status not in {"failed", "cancelled"}
                            silence_stop.set()
                            break
                        if args.mode == "auto":
                            # BACKCHANNEL/CONTINUE is not the requested turn
                            # completion; keep the listener and silence stream
                            # alive until a SPEAK response finishes or 30 s pass.
                            continue
                        if not pending_response_ids and (responses or kind == "response.done"):
                            break
                    elif kind == "error":
                        result.setdefault("errors", []).append(event)
                        silence_stop.set()
                        break

                if silence_sender_task is not None:
                    silence_stop.set()
                    await asyncio.gather(silence_sender_task, return_exceptions=True)
                result["inputs"][name]["silence_tail_frames_until_response_created"] = silence_tail_frames
                result["inputs"][name]["silence_tail_frames_until_speak_done"] = silence_tail_frames
                result["inputs"][name]["total_frames_sent"] = len(frames) + silence_tail_frames

                # Persist audio before session.close; overload may already have
                # closed the socket even though response audio was received.
                for rid, chunks in audio_packets.items():
                    # audio_packets spans the session; only current-turn IDs
                    # belong in this turn's completion aggregate.
                    if rid not in responses or not chunks:
                        continue
                    wav_path = args.output_dir / f"{rid}.wav"
                    write_response_audio_wav(wav_path, audio_params[rid], chunks)
                    samples = sum(len(chunk) for chunk in chunks) // (
                        audio_params[rid][0] * audio_params[rid][1]
                    )
                    result["response_audio"][rid] = {
                        "wav": str(wav_path),
                        "sample_rate": audio_params[rid][2],
                        "channels": audio_params[rid][0],
                        "sample_width_bytes": audio_params[rid][1],
                        "samples": samples,
                        "duration_ms": round(samples * 1000.0 / audio_params[rid][2], 3),
                        "packets": len(chunks),
                    }
                    responses.setdefault(rid, {})["audio_wav"] = str(wav_path)
                    responses[rid]["audio_packets"] = len(chunks)

                turn = {
                    "turn": turn_index,
                    "name": name,
                    "waveform": str(path),
                    "frames_sent": len(frames) + silence_tail_frames,
                    "silence_tail_frames_until_response_created": silence_tail_frames,
                    "first_frame_sent_at_ns": first_sent_ns,
                    "last_frame_sent_at_ns": last_sent_ns,
                    "manual_commit": args.mode == "manual",
                    "responses": responses,
                    "response_done": bool(responses) and all("response_done" in item for item in responses.values()),
                    "speak_response_done": speak_response_done if args.mode == "auto" else None,
                    "pending_response_ids": sorted(pending_response_ids),
                }
                result["turns"].append(turn)
                for rid, entry in responses.items():
                    entry.setdefault("assistant_text", text_by_response.get(rid, ""))
                    entry.setdefault("audio_transcript", transcript_by_response.get(rid, ""))
                result["partial_report_path"] = str(args.output_dir / "partial_report.json")
                (args.output_dir / "partial_report.json").write_text(
                    json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                terminal_statuses = {
                    str(item.get("response_done", {}).get("status", "")).lower()
                    for item in responses.values()
                    if isinstance(item.get("response_done"), dict)
                }
                if (
                    not turn["response_done"]
                    or terminal_statuses.intersection({"failed", "cancelled"})
                    or (args.mode == "auto" and not speak_response_done)
                ):
                    result["stopped_after_incomplete_or_failed_response"] = name
                    (args.output_dir / "partial_report.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    break
                if turn_index == 1 and turn.get("response_done"):
                    await asyncio.sleep(max(0.0, args.turn_delay_s))
                    record("client", {"type": "probe.turn_gap", "seconds": args.turn_delay_s, "after_turn": name})

            try:
                await ws.send(json.dumps({"type": "session.close"}))
                record("client", {"type": "session.close"})
            except websockets.exceptions.ConnectionClosed as exc:
                result.setdefault("transport_errors", []).append(
                    f"session.close after peer disconnect: {type(exc).__name__}: {exc}"
                )
            reader_task.cancel()
            await asyncio.gather(reader_task, return_exceptions=True)

    result["server_events_path"] = str(events_path)
    result["server_events_count"] = sum(1 for _ in events_path.open(encoding="utf-8"))
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "events": str(events_path),
        "mode": args.mode,
        "turns": [
            {"name": t["name"], "response_done": t["response_done"],
             "responses": [{"response_id": rid, "text": r.get("assistant_text"),
                            "history_roles": r.get("response_done", {}).get("runtime_metrics", {}).get("generation_trace", {}).get("completed_history_roles")}
                           for rid, r in t["responses"].items()]}
            for t in result["turns"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
