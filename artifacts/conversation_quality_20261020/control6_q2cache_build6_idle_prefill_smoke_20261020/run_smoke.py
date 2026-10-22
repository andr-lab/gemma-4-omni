from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import time
import wave
from collections import deque
from math import gcd
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal
import websockets


ROOT = Path("/workspace")
INPUTS = ROOT / "artifacts/conversation_quality_20261014/inputs"
OUTPUT = ROOT / "artifacts/conversation_quality_20261020/control6_q2cache_build6_idle_prefill_smoke_20261020"
WS_URL = "ws://127.0.0.1:8000/v1/realtime?duplex=1"
TARGET_B_PROMPT = (
    "Sei un assistente conversazionale italiano. Ascolta con attenzione e rispondi in modo naturale e conciso, "
    "al massimo due frasi. Mantieni i dati effettivamente comunicati e applica le correzioni più recenti. "
    "Se non hai capito, chiedi chiarimenti. Non inventare informazioni."
)
RATE_IN = 16_000
RATE_OUT = 24_000
FRAME_MS = 80
FRAME_IN = RATE_IN * FRAME_MS // 1000
FRAME_OUT = RATE_OUT * FRAME_MS // 1000
PERIOD_NS = FRAME_MS * 1_000_000
PLAYBACK_CAPACITY = RATE_OUT * 3


def read_audio_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        if wav.getsampwidth() != 2 or channels not in (1, 2):
            raise ValueError(f"expected mono/stereo PCM16 WAV: {path}")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if channels == 2:
        pcm = pcm.reshape(-1, 2).mean(axis=1)
    if rate != RATE_IN:
        divisor = gcd(rate, RATE_IN)
        pcm = scipy.signal.resample_poly(pcm, RATE_IN // divisor, rate // divisor).astype(np.float32)
    return np.clip(pcm, -1.0, 1.0)


def encode_pcm16(frame: np.ndarray) -> str:
    pcm = np.clip(np.asarray(frame, dtype=np.float32), -1.0, 1.0)
    return base64.b64encode(np.rint(pcm * 32767.0).astype("<i2").tobytes()).decode("ascii")


def decode_response_audio(event: dict[str, Any]) -> dict[str, Any]:
    if event.get("type") != "response.audio.delta":
        raise ValueError("expected response.audio.delta")
    encoded = event.get("delta")
    if not isinstance(encoded, str):
        raise ValueError("audio delta is missing its base64 payload")
    with wave.open(io.BytesIO(base64.b64decode(encoded, validate=True)), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("response audio must be mono PCM16 WAV")
        rate = wav.getframerate()
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if rate != RATE_OUT or rate != int(event.get("sample_rate", -1)):
        raise ValueError(f"unexpected response sample rate: wav={rate}, event={event.get('sample_rate')}")
    response_id = str(event.get("response_id", ""))
    segment_id = str(event.get("speaker_segment_id", ""))
    generation = int(event.get("generation_id", -1))
    start = int(event.get("start_sample", -1))
    end = int(event.get("end_sample", -1))
    if not response_id or not segment_id or generation < 0 or end - start != pcm.size:
        raise ValueError("response audio identity or sample range is invalid")
    return {
        "response_id": response_id,
        "speaker_segment_id": segment_id,
        "generation_id": generation,
        "start_sample": start,
        "end_sample": end,
        "sample_rate": rate,
        "pcm": pcm,
    }


async def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    correction_asr = json.loads((INPUTS / "correction_bargein_asr.json").read_text(encoding="utf-8"))
    seed = read_audio_16k(INPUTS / "pair_seed_it.wav")
    correction = read_audio_16k(INPUTS / "correction_bargein.wav")
    correction = correction[: 3 * FRAME_IN]
    correction_text = str(correction_asr["results"][0]["text"]).strip()
    correction_frames = math.ceil(correction.size / FRAME_IN)
    if correction_frames != 3:
        raise RuntimeError(f"expected exactly 3 correction frames; got {correction_frames}")

    events_path = OUTPUT / "events.jsonl"
    report_path = OUTPUT / "report.json"
    event_file = events_path.open("w", encoding="utf-8")
    events: list[dict[str, Any]] = []
    input_pcm = np.empty(0, dtype=np.float32)
    consumed_pcm: list[np.ndarray] = []
    chunks: deque[dict[str, Any]] = deque()
    segments: dict[tuple[str, int, str], dict[str, Any]] = {}
    completed_responses: set[str] = set()
    cancelled: set[tuple[str, int | None]] = set()
    responses: dict[str, dict[str, Any]] = {}
    first_audio_delta_ns: int | None = None
    first_pcm_consumed_ns: int | None = None
    correction_first_sample_ns: int | None = None
    correction_commit_ns: int | None = None
    seed_commit_ns: int | None = None
    input_frames_sent = 0
    input_frames_consumed = 0
    pending_samples = 0
    ack_sequence = 0
    last_event_ns = time.monotonic_ns()
    producer_done = asyncio.Event()
    first_pcm_consumed = asyncio.Event()
    input_frame_consumed_event = asyncio.Event()
    send_lock = asyncio.Lock()
    stop_playback = asyncio.Event()
    producer_error: str | None = None
    stop_reason = "running"
    playback_acknowledgements: list[dict[str, Any]] = []

    def log(event_type: str, details: dict[str, Any]) -> None:
        row = {"type": event_type, "timestamp_ns": time.monotonic_ns(), **details}
        event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        event_file.flush()
        events.append(row)

    def remove_stale(predicate) -> None:
        nonlocal pending_samples
        kept: deque[dict[str, Any]] = deque()
        for chunk in chunks:
            if predicate(chunk):
                pending_samples -= int(chunk["pcm"].size - chunk["offset"])
            else:
                kept.append(chunk)
        chunks.clear()
        chunks.extend(kept)

    async with websockets.connect(
        WS_URL,
        max_size=1_000_000,
        max_queue=8,
        ping_interval=20,
        ping_timeout=20,
    ) as ws:
        created = json.loads(await ws.recv())
        if created.get("type") != "session.created":
            raise RuntimeError(f"expected session.created, got {created!r}")
        log("session.created", {"session": created})
        async with send_lock:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "voice": "gemma",
                    "instructions": TARGET_B_PROMPT,
                    "auto_response": True,
                    "turn_policy": "gemma_native_control6",
                    "input_audio_buffer_acknowledgements": True,
                    "input_audio_buffer_conditioning_acknowledgements": True,
                },
            }, ensure_ascii=False, separators=(",", ":")))
        updated = json.loads(await ws.recv())
        if updated.get("type") != "session.updated":
            raise RuntimeError(f"session.update rejected: {updated!r}")
        log("session.updated", {"session": updated})

        async def send_frame(frame: np.ndarray, *, step: str, index: int, source_start: int) -> None:
            nonlocal input_frames_sent, correction_first_sample_ns, input_pcm
            if frame.shape != (FRAME_IN,):
                raise ValueError(f"input frame must have {FRAME_IN} samples, got {frame.shape}")
            while input_frames_sent - input_frames_consumed >= 3:
                input_frame_consumed_event.clear()
                if input_frames_sent - input_frames_consumed < 3:
                    break
                try:
                    await asyncio.wait_for(input_frame_consumed_event.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    raise TimeoutError(
                        f"input consumed-ack stalled with {input_frames_sent - input_frames_consumed} frames outstanding"
                    )
            message = {
                "type": "input_audio_buffer.append",
                "audio": encode_pcm16(frame),
            }
            async with send_lock:
                await ws.send(json.dumps(message, separators=(",", ":")))
                sent_ns = time.monotonic_ns()
            if step == "correction_bargein" and index == 0:
                correction_first_sample_ns = sent_ns
            input_pcm = np.concatenate((input_pcm, frame.astype(np.float32, copy=False)))
            input_frames_sent += 1
            log("input_audio_frame_sent", {
                "step_id": step,
                "frame_index": index,
                "ingress_frame_id": input_frames_sent - 1,
                "sent_ns": sent_ns,
                "source_start_sample": source_start,
                "source_end_sample": min(source_start + FRAME_IN, seed.size if step == "pair_seed_it" else correction.size),
                "sample_rate": RATE_IN,
                "sample_count": FRAME_IN,
            })

        async def send_phrase(pcm: np.ndarray, *, step: str) -> None:
            count = math.ceil(pcm.size / FRAME_IN)
            start_ns = time.monotonic_ns()
            prev_sent_ns: int | None = None
            for index in range(count):
                due_ns = start_ns + index * PERIOD_NS
                if prev_sent_ns is not None:
                    due_ns = max(due_ns, prev_sent_ns + PERIOD_NS)
                await asyncio.sleep(max(0.0, (due_ns - time.monotonic_ns()) / 1e9))
                start = index * FRAME_IN
                end = min(start + FRAME_IN, pcm.size)
                frame = np.zeros(FRAME_IN, dtype=np.float32)
                frame[: end - start] = pcm[start:end]
                await send_frame(frame, step=step, index=index, source_start=start)
                prev_sent_ns = time.monotonic_ns()
            async with send_lock:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                committed_ns = time.monotonic_ns()
            log("input_audio_buffer_commit_sent", {
                "step_id": step,
                "timestamp_ns": committed_ns,
                "source_sample_count": int(pcm.size),
                "frame_count": count,
            })
            return committed_ns

        async def input_producer() -> None:
            nonlocal seed_commit_ns, correction_commit_ns, producer_error
            try:
                log("input_step_started", {
                    "step_id": "pair_seed_it",
                    "source": str(INPUTS / "pair_seed_it.wav"),
                    "source_sample_count": int(seed.size),
                    "expected_transcript": "original seed audio; transcript not reasserted",
                    "frame_pacing_ms": FRAME_MS,
                })
                seed_commit_ns = await send_phrase(seed, step="pair_seed_it")
                try:
                    await asyncio.wait_for(first_pcm_consumed.wait(), timeout=70.0)
                except asyncio.TimeoutError:
                    raise TimeoutError("no assistant PCM was consumed within 70 seconds after seed")
                target_ns = int(first_pcm_consumed_ns or time.monotonic_ns()) + 500_000_000
                await asyncio.sleep(max(0.0, (target_ns - time.monotonic_ns()) / 1e9))
                log("input_step_started", {
                    "step_id": "correction_bargein",
                    "source": str(INPUTS / "correction_bargein.wav"),
                    "expected_transcript": correction_text,
                    "source_sample_count": int(correction.size),
                    "frame_count": correction_frames,
                    "frame_pacing_ms": FRAME_MS,
                    "anchor": "first logical client PCM consumption",
                    "anchor_ns": first_pcm_consumed_ns,
                    "target_offset_ms": 500,
                    "target_ns": target_ns,
                })
                correction_commit_ns = await send_phrase(correction, step="correction_bargein")
            except Exception as exc:
                producer_error = f"{type(exc).__name__}: {exc}"
                log("input_producer_error", {"error": producer_error})
            finally:
                producer_done.set()

        async def maybe_send_segment_ack(key: tuple[str, int, str]) -> None:
            nonlocal ack_sequence
            segment = segments.get(key)
            if not segment or segment["acked"] or not segment["done"]:
                return
            expected = int(segment["end_sample"] - segment["start_sample"])
            if expected <= 0 or int(segment["consumed_samples"]) < expected:
                return
            ack_sequence += 1
            ack = {
                "type": "client_consumed_pcm",
                "response_id": key[0],
                "generation_id": key[1],
                "speaker_segment_id": key[2],
                "start_sample": int(segment["start_sample"]),
                "end_sample": int(segment["end_sample"]),
                "sample_rate": int(segment["sample_rate"]),
                "sequence": ack_sequence,
                "client_monotonic_ns": time.monotonic_ns(),
                "consumption_semantics": "logical_client_device_buffer",
            }
            async with send_lock:
                await ws.send(json.dumps(ack, separators=(",", ":")))
                ack["sent_ns"] = time.monotonic_ns()
            segment["acked"] = True
            playback_acknowledgements.append(ack)
            log("client_consumed_pcm_sent", ack)

        async def playback_clock() -> None:
            nonlocal pending_samples, first_pcm_consumed_ns
            while not stop_playback.is_set():
                tick_start = time.monotonic_ns()
                remaining = FRAME_OUT
                consumed_keys: set[tuple[str, int, str]] = set()
                output = np.zeros(FRAME_OUT, dtype=np.float32)
                written = 0
                while remaining and chunks:
                    chunk = chunks[0]
                    pcm = chunk["pcm"]
                    offset = int(chunk["offset"])
                    take = min(remaining, int(pcm.size - offset))
                    output[written : written + take] = pcm[offset : offset + take]
                    chunk["offset"] = offset + take
                    key = chunk["key"]
                    segments[key]["consumed_samples"] += take
                    consumed_keys.add(key)
                    written += take
                    remaining -= take
                    pending_samples -= take
                    if int(chunk["offset"]) == pcm.size:
                        chunks.popleft()
                if written:
                    consumed_pcm.append(output.copy())
                    if first_pcm_consumed_ns is None:
                        first_pcm_consumed_ns = time.monotonic_ns()
                        first_pcm_consumed.set()
                        log("first_client_pcm_consumed", {
                            "timestamp_ns": first_pcm_consumed_ns,
                            "consumption_semantics": "logical_client_device_buffer",
                            "copied_samples": written,
                            "sample_rate": RATE_OUT,
                        })
                    for key in consumed_keys:
                        await maybe_send_segment_ack(key)
                elapsed = time.monotonic_ns() - tick_start
                await asyncio.sleep(max(0.0, (PERIOD_NS - elapsed) / 1e9))

        producer_task = asyncio.create_task(input_producer())
        playback_task = asyncio.create_task(playback_clock())
        started_ns = time.monotonic_ns()
        try:
            while True:
                now_ns = time.monotonic_ns()
                if now_ns - started_ns >= 100_000_000_000:
                    stop_reason = "max_duration"
                    break
                if producer_done.is_set() and producer_error:
                    stop_reason = "input_error"
                    break
                if (
                    producer_done.is_set()
                    and correction_commit_ns is not None
                    and last_event_ns - correction_commit_ns >= 0
                    and not chunks
                    and pending_samples == 0
                    and time.monotonic_ns() - last_event_ns >= 2_000_000_000
                ):
                    stop_reason = "quiet_after_correction"
                    break
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                received_ns = time.monotonic_ns()
                last_event_ns = received_ns
                event = json.loads(raw)
                event_type = str(event.get("type", "unknown"))
                row = {"type": event_type, "received_ns": received_ns, "event": event}
                event_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                event_file.flush()
                events.append(row)
                if event_type == "response.created":
                    response_id = str(event.get("response_id", ""))
                    if response_id:
                        responses.setdefault(response_id, {
                            "response_id": response_id,
                            "created_ns": received_ns,
                            "text": "",
                            "audio_packets": 0,
                            "audio_samples": 0,
                        })
                        completed_responses.discard(response_id)
                elif event_type == "speaker.segment.opened":
                    response_id = str(event.get("response_id", ""))
                    generation = int(event.get("generation_id", 0))
                    remove_stale(lambda chunk: (
                        chunk["key"][0] == response_id
                        and chunk["key"][1] < generation
                        and response_id not in completed_responses
                    ))
                elif event_type == "response.audio.delta":
                    if first_audio_delta_ns is None:
                        first_audio_delta_ns = received_ns
                    packet = decode_response_audio(event)
                    key = (packet["response_id"], packet["generation_id"], packet["speaker_segment_id"])
                    if (packet["response_id"], packet["generation_id"]) in cancelled:
                        continue
                    segment = segments.setdefault(key, {
                        "start_sample": packet["start_sample"],
                        "end_sample": packet["end_sample"],
                        "sample_rate": packet["sample_rate"],
                        "consumed_samples": 0,
                        "done": False,
                        "acked": False,
                    })
                    segment["start_sample"] = min(int(segment["start_sample"]), packet["start_sample"])
                    segment["end_sample"] = max(int(segment["end_sample"]), packet["end_sample"])
                    if pending_samples + packet["pcm"].size > PLAYBACK_CAPACITY:
                        raise BufferError(
                            f"logical playback exceeded {PLAYBACK_CAPACITY} samples; stopping without dropping PCM"
                        )
                    chunks.append({"key": key, "pcm": packet["pcm"], "offset": 0})
                    pending_samples += int(packet["pcm"].size)
                    response = responses.setdefault(packet["response_id"], {
                        "response_id": packet["response_id"],
                        "created_ns": None,
                        "text": "",
                        "audio_packets": 0,
                        "audio_samples": 0,
                    })
                    response["audio_packets"] += 1
                    response["audio_samples"] += int(packet["pcm"].size)
                    response.setdefault("first_audio_delta_ns", received_ns)
                elif event_type == "response.audio.segment_done":
                    key = (
                        str(event.get("response_id", "")),
                        int(event.get("generation_id", 0)),
                        str(event.get("speaker_segment_id", "")),
                    )
                    segment = segments.setdefault(key, {
                        "start_sample": int(event["start_sample"]),
                        "end_sample": int(event["end_sample"]),
                        "sample_rate": int(event["sample_rate"]),
                        "consumed_samples": 0,
                        "done": False,
                        "acked": False,
                    })
                    segment.update({
                        "start_sample": int(event["start_sample"]),
                        "end_sample": int(event["end_sample"]),
                        "sample_rate": int(event["sample_rate"]),
                        "done": True,
                    })
                    await maybe_send_segment_ack(key)
                elif event_type in {"response.cancelled", "audio.cancelled"}:
                    response_id = str(event.get("response_id", ""))
                    generation_value = event.get("generation_id")
                    generation = int(generation_value) if generation_value is not None else None
                    cancelled.add((response_id, generation))
                    remove_stale(lambda chunk: (
                        chunk["key"][0] == response_id
                        and (generation is None or chunk["key"][1] == generation)
                    ))
                elif event_type == "response.done":
                    completed_responses.add(str(event.get("response_id", "")))
                elif event_type in {
                    "response.output_text.delta",
                    "response.audio_transcript.delta",
                    "response.output_audio_transcript.delta",
                }:
                    response_id = str(event.get("response_id", ""))
                    if response_id:
                        responses.setdefault(response_id, {
                            "response_id": response_id,
                            "created_ns": None,
                            "text": "",
                            "audio_packets": 0,
                            "audio_samples": 0,
                        })["text"] += str(event.get("delta", ""))
                elif event_type == "input_audio_buffer.consumed":
                    input_frames_consumed += 1
                    input_frame_consumed_event.set()
                elif event_type == "error":
                    log("server_error", {"received_ns": received_ns, "error": event.get("error", event)})
                for key in (
                    "duplex_control6.decision",
                    "response.created",
                    "response.cancelled",
                    "audio.cancelled",
                    "response.done",
                ):
                    if event_type == key:
                        break
        finally:
            stop_playback.set()
            for task in (producer_task, playback_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(producer_task, playback_task, return_exceptions=True)

    event_file.close()
    if input_pcm.size:
        with wave.open(str(OUTPUT / "input_sent.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE_IN)
            wav.writeframes(np.clip(input_pcm, -1.0, 1.0).astype("<i2").tobytes())
    if consumed_pcm:
        with wave.open(str(OUTPUT / "assistant_logical_playback.wav"), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE_OUT)
            wav.writeframes(np.clip(np.concatenate(consumed_pcm), -1.0, 1.0).astype("<i2").tobytes())

    decision_rows = [
        row["event"] for row in events
        if row.get("type") == "duplex_control6.decision" and isinstance(row.get("event"), dict)
    ]
    yield_rows = [
        event for event in decision_rows
        if str(event.get("gate_applied_action", event.get("applied_action", ""))).upper() == "YIELD"
    ]
    speak_rows = [
        event for event in decision_rows
        if str(event.get("gate_applied_action", event.get("applied_action", ""))).upper() == "SPEAK"
    ]
    report = {
        "status": "completed" if stop_reason == "quiet_after_correction" else "partial",
        "stop_reason": stop_reason,
        "scenario_id": "control6_single_target_b_bargein_20261020",
        "prompt": TARGET_B_PROMPT,
        "protocol": "heard_action_say_v2",
        "turn_policy": "gemma_native_control6",
        "inputs": {
            "seed_path": str(INPUTS / "pair_seed_it.wav"),
            "seed_samples_16k": int(seed.size),
            "correction_path": str(INPUTS / "correction_bargein.wav"),
            "correction_asr_reference": correction_text,
            "correction_samples_16k": int(correction.size),
            "correction_frame_count": correction_frames,
            "frame_pacing_ms": FRAME_MS,
        },
        "timing": {
            "started_ns": started_ns,
            "first_audio_delta_ns": first_audio_delta_ns,
            "first_logical_pcm_consumed_ns": first_pcm_consumed_ns,
            "seed_commit_ns": seed_commit_ns,
            "correction_first_sample_sent_ns": correction_first_sample_ns,
            "correction_commit_ns": correction_commit_ns,
            "first_pcm_to_correction_injection_ms": (
                (correction_first_sample_ns - first_pcm_consumed_ns) / 1e6
                if correction_first_sample_ns is not None and first_pcm_consumed_ns is not None else None
            ),
            "first_audio_delta_to_first_logical_pcm_ms": (
                (first_pcm_consumed_ns - first_audio_delta_ns) / 1e6
                if first_pcm_consumed_ns is not None and first_audio_delta_ns is not None else None
            ),
        },
        "ingress_frames_sent": input_frames_sent,
        "ingress_frames_consumed_ack": input_frames_consumed,
        "responses": list(responses.values()),
        "control6_decision_count": len(decision_rows),
        "yield_decisions": yield_rows,
        "speak_decisions": speak_rows,
        "playback_acknowledgements": playback_acknowledgements,
        "unacked_segments": [
            {"key": key, **segment}
            for key, segment in segments.items()
            if segment["done"] and not segment["acked"]
        ],
        "producer_error": producer_error,
        "events_jsonl": str(events_path),
        "input_wav": str(OUTPUT / "input_sent.wav"),
        "assistant_logical_playback_wav": str(OUTPUT / "assistant_logical_playback.wav"),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

