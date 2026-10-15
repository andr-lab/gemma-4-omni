#!/usr/bin/env python3
"""Exercise a Gemma QLoRA adapter through real WAV ingress and kernel TTS."""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
import wave

import numpy as np


def _wav_metrics(path: Path, expected_rate: int) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != expected_rate:
            raise RuntimeError(
                f"{path}: WAV inatteso: channels={wav.getnchannels()}, width={wav.getsampwidth()}, "
                f"rate={wav.getframerate()}"
            )
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise RuntimeError(f"{path}: audio vuoto o non finito")
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(pcm)))
    clipping = float(np.mean(np.abs(pcm) >= 0.999))
    duration = pcm.size / float(expected_rate)
    if duration <= 0.1 or rms <= 1e-4 or clipping > 0.05:
        raise RuntimeError(f"{path}: audio non utilizzabile: duration={duration:.3f}s rms={rms:.6f} clip={clipping:.2%}")
    return {"sample_rate_hz": expected_rate, "samples": int(pcm.size), "duration_s": duration,
            "rms": rms, "peak": peak, "clipping_fraction": clipping}


def _smoke_scenario(*, require_cache_hit: bool):
    from dialogue_eval.scenarios import SCENARIO_NAME_NUMBER

    turns = 2 if require_cache_hit else 1
    step_count = 4 if require_cache_hit else 2
    return replace(SCENARIO_NAME_NUMBER, steps=SCENARIO_NAME_NUMBER.steps[:step_count], max_turns=turns), turns


def _summarize_response_events(
    events: list[dict[str, object]],
    *,
    expected_turns: int,
    require_cache_hit: bool,
) -> list[dict[str, object]]:
    response_text: dict[str, str] = {}
    response_start_ns: dict[str, int] = {}
    response_audio_ns: dict[str, int] = {}
    user_end_ns: dict[int, int] = {}
    completed: list[dict[str, object]] = []
    for event in events:
        event_type = event.get("event_type")
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        response_id = str(details.get("response_id", ""))
        try:
            timestamp_ns = int(event.get("timestamp_ns"))
        except (TypeError, ValueError):
            timestamp_ns = 0
        if event_type == "user_speech_end":
            try:
                user_end_ns[int(event.get("turn_index", 0) or 0)] = timestamp_ns
            except (TypeError, ValueError):
                pass
        elif event_type == "response.created" and response_id and timestamp_ns:
            response_start_ns[response_id] = timestamp_ns
        elif event_type == "response.audio.delta" and response_id and timestamp_ns:
            response_audio_ns.setdefault(response_id, timestamp_ns)
        elif event_type == "response.output_text.delta" and response_id:
            response_text[response_id] = response_text.get(response_id, "") + str(details.get("delta", ""))
        elif event_type == "response.done":
            completed.append(event)

    if len(completed) != expected_turns:
        raise RuntimeError(
            f"audio smoke expected {expected_turns} completed responses, observed {len(completed)}"
        )

    summary: list[dict[str, object]] = []
    for event in completed:
        details = event.get("details")
        if not isinstance(details, dict):
            raise RuntimeError("response.done is missing its event details")
        metrics = details.get("runtime_metrics")
        metrics = metrics if isinstance(metrics, dict) else {}
        direct = metrics.get("direct_audio")
        direct = direct if isinstance(direct, dict) else {}
        thinker = metrics.get("thinker")
        thinker = thinker if isinstance(thinker, dict) else {}
        response_id = str(details.get("response_id", ""))
        turn_index = int(event.get("turn_index", 0) or 0)
        started_ns = response_start_ns.get(response_id)
        first_audio_ns = response_audio_ns.get(response_id)
        user_done_ns = user_end_ns.get(turn_index)
        try:
            done_ns = int(event.get("timestamp_ns"))
        except (TypeError, ValueError):
            done_ns = None

        def elapsed_ms(end_ns: int | None, start_ns: int | None) -> float | None:
            if end_ns is None or start_ns is None or end_ns < start_ns:
                return None
            return round((end_ns - start_ns) / 1_000_000.0, 3)

        pcm_samples = int(direct.get("pcm_samples", 0) or 0)
        if require_cache_hit and pcm_samples <= 0:
            raise RuntimeError("direct-audio smoke response is missing its captured user PCM metrics")
        summary.append({
            "turn_index": turn_index,
            "response_id": response_id,
            "response_text": response_text.get(response_id, "").strip(),
            "direct_audio_pcm_samples": pcm_samples,
            "thinker_cache_hit": bool(thinker.get("thinker_cache_hit", False)),
            "thinker_cache_reused_tokens": int(thinker.get("thinker_cache_reused_tokens", 0) or 0),
            "thinker_cache_reason": thinker.get("thinker_cache_reason"),
            "thinker_prefill_ms": thinker.get("thinker_prefill_ms"),
            "thinker_audio_encoder_gpu_ms": thinker.get("thinker_audio_encoder_gpu_ms"),
            "thinker_audio_encoder_wall_ms": thinker.get("thinker_audio_encoder_wall_ms"),
            "first_output_audio_ms": elapsed_ms(first_audio_ns, started_ns),
            "response_done_ms": elapsed_ms(done_ns, started_ns),
            "user_end_to_first_audio_ms": elapsed_ms(first_audio_ns, user_done_ns),
        })

    if require_cache_hit:
        second = summary[1]
        if not second["thinker_cache_hit"] or int(second["thinker_cache_reused_tokens"]) <= 0:
            raise RuntimeError(
                "second turn did not reuse an exact hard-text prefix before the current audio; "
                f"metrics={second}"
            )
    return summary


async def _run(args: argparse.Namespace) -> dict[str, object]:
    from client.duplex_dialogue_harness import DialogueConfig, DialogueOrchestrator

    scenario, turn_count = _smoke_scenario(require_cache_hit=args.require_cache_hit)
    orchestrator = DialogueOrchestrator(
        config=DialogueConfig(
            max_turns=turn_count,
            max_duration_s=60 if turn_count > 1 else 30,
            ws_url=args.server,
        ),
        output_dir=args.output_dir,
        scenario_audio_dir=args.scenario_audio_dir,
    )
    result = await orchestrator.run_mode_a(scenario)
    if result.get("status") != "COMPLETED" or result.get("server_errors"):
        raise RuntimeError(f"audio ingress/response failed: {result.get('server_errors')}")
    event_path = args.output_dir / "dialogue_events.jsonl"
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    response_metrics = _summarize_response_events(
        events,
        expected_turns=turn_count,
        require_cache_hit=args.require_cache_hit,
    )
    paths = result.get("wav_files", {})
    input_path = Path(paths["interlocutor_a_input"])
    output_path = Path(paths["interlocutor_a_output"])
    report = {
        "status": "cache_prefix_reused" if args.require_cache_hit else "audio_io_observed",
        "trained_control6": False,
        "policy": "none",
        "auto_response": False,
        "manual_commit": True,
        "turns": response_metrics,
        "input_wav": str(input_path),
        "output_wav": str(output_path),
        "audio": {"input": _wav_metrics(input_path, 16000), "output": _wav_metrics(output_path, 24000)},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenario-audio-dir", type=Path, default=Path("artifacts/scenario_audio_qwen3tts"))
    parser.add_argument(
        "--require-cache-hit",
        action="store_true",
        help="run both memory turns and fail unless the second direct-audio response reuses a text KV prefix",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
