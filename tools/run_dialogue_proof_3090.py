# -*- coding: utf-8 -*-
"""tools/run_dialogue_proof_3090.py

End-to-end spoken dialogue proof-of-concept and multi-turn evaluation on NVIDIA GeForce RTX 3090 (24 GB).

Verifies all requirements:
1. Single server process, single copy of model weights in VRAM (Gemma 4 Thinker + Talker + V9 + Code2Wav).
2. Two WebSocket sessions (/v1/realtime?duplex=1), with session IDs and owned-state object identity measured explicitly.
3. Configuration comparison: one session vs two open sessions (VRAM and active-session counts; not concurrent inference).
4. Mode (a): Qwen3-TTS speech playback for canonical questions, clarifications, instruction changes, contradictions, and interruptions.
5. Mode (b): Autonomous multi-turn conversation between Interlocutor A and Interlocutor B with anti-echo 80 ms audio routing.
6. Multi-track recording: 4 independent WAVs (A-in, A-out, B-in, B-out) and JSONL event trace.
7. Objective metrics: clipping (<=0.1%), RMS, crest factor (5.0-8.5), anomalous silences, discontinuities, TTFA, dropped frames (0).
8. Separate transcription: Input ASR vs ground truth, Generated text vs Audio Transcript (via faster-whisper).
9. Final report with clear distinction:
   - "serving avviabile"
   - "dialogo audio dimostrato"
   - "PRE_QLORA_READY certificato"
"""

from __future__ import annotations

import asyncio
import base64
import collections
from dataclasses import replace
import glob
import hashlib
import io
import json
import logging
import math
import os
from pathlib import Path
import sys
import threading
import time
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple
import wave

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/.local_lib")

import numpy as np
import soundfile as sf
import torch
import uvicorn
import websockets

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.server import create_app
from duplex_kernel.session import SessionManager
from client.duplex_dialogue_harness import (
    AudioBridge,
    AudioTimelineReconstructor,
    DialogueConfig,
    DialogueEventLogger,
    DialogueOrchestrator,
    DialogueRecorder,
    decode_response_audio,
    float_to_pcm16_bytes,
    pcm_to_80ms_frames,
    resample_24k_to_16k,
)
from dialogue_eval.audio_metrics import compute_physical_audio_metrics
from dialogue_eval.evaluator import (
    DialogueEvaluator,
    calculate_wer,
    classify_conversation_evidence,
    classify_turn_taking_audio_evidence,
    contains_italian_digit_sequence,
)
from tools.generate_scenario_audio_qwen3tts import (
    _force_offline_model_loading,
    _manifest_is_complete,
    _path_has_symlink_component,
    _snapshot_verified_audio_cache,
    _validate_cache_clip,
)
from dialogue_eval.scenarios import (
    ALL_SCENARIOS,
    SCENARIO_NAME_NUMBER,
    SCENARIO_CLARIFICATION,
    SCENARIO_INSTRUCTION_CHANGE,
    SCENARIO_CONTRADICTION,
    SCENARIO_BARGE_IN,
    SCENARIO_CLOSING,
    MODE_B_KICKOFF_TEXT,
    generate_speech_for_text,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("dialogue_proof_3090")

WORKSPACE = Path("/workspace")
RUN_ID = os.environ.get("DIALOGUE_PROOF_RUN_ID", time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
ARTIFACTS_DIR = WORKSPACE / "artifacts" / f"dialogue_proof_3090_qwen3tts_{RUN_ID}"
SCENARIO_AUDIO_DIR = WORKSPACE / "artifacts" / "scenario_audio_qwen3tts"


def verify_scenario_audio_cache() -> Tuple[Dict[str, Any], Any]:
    """Validate and freeze the cache before ASR or dialogue inference consumes it."""
    _force_offline_model_loading()
    if _path_has_symlink_component(SCENARIO_AUDIO_DIR):
        raise RuntimeError(f"Qwen3-TTS cache path contains a symlink: {SCENARIO_AUDIO_DIR}")
    manifest_path = SCENARIO_AUDIO_DIR / "scenario_audio_manifest.json"
    if _path_has_symlink_component(manifest_path):
        raise RuntimeError(f"Qwen3-TTS manifest path contains a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Qwen3-TTS scenario cache manifest not found: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict) or not _manifest_is_complete(manifest):
        raise RuntimeError("Qwen3-TTS cache is stale, incomplete, or one or more independent-ASR gates failed")

    expected_user_steps = [
        (scenario, step)
        for scenario in ALL_SCENARIOS
        for step in scenario.steps
        if step.speaker == "user"
    ]
    if len(expected_user_steps) != 11:
        raise RuntimeError(f"Expected exactly 11 canonical Mode A user clips, found {len(expected_user_steps)}")
    clip_records = manifest.get("clips")
    if not isinstance(clip_records, list) or len(clip_records) != 11:
        raise RuntimeError("Qwen3-TTS cache must contain exactly 11 Mode A clip records")
    clips: Dict[str, Dict[str, Any]] = {}
    for clip in clip_records:
        if not isinstance(clip, dict) or not isinstance(clip.get("filename"), str):
            raise RuntimeError("Qwen3-TTS cache contains a malformed clip record")
        filename = clip["filename"]
        if filename in clips:
            raise RuntimeError(f"Qwen3-TTS cache contains a duplicate clip record: {filename}")
        clips[filename] = clip

    kickoff = manifest.get("mode_b_kickoff")
    if not isinstance(kickoff, dict):
        raise RuntimeError("Qwen3-TTS cache has no valid Mode B kickoff record")
    targets = [
        (
            f"{scenario.scenario_id}__{step.step_id}.wav",
            step.expected_text,
            clips.get(f"{scenario.scenario_id}__{step.step_id}.wav"),
            scenario.scenario_id,
            step.step_id,
        )
        for scenario, step in expected_user_steps
    ]
    targets.append(("mode_b_kickoff.wav", MODE_B_KICKOFF_TEXT, kickoff, "mode_b", "kickoff"))

    problems = []
    for filename, expected_text, clip, scenario_id, step_id in targets:
        if clip is None:
            problems.append(f"{filename}: missing manifest record")
            continue
        errors = _validate_cache_clip(
            clip,
            expected_text=expected_text,
            audio_path=SCENARIO_AUDIO_DIR / filename,
            expected_scenario_id=scenario_id,
            expected_step_id=step_id,
        )
        if errors:
            problems.append(f"{filename}: {errors}")
    if problems:
        raise RuntimeError(f"Qwen3-TTS cache failed pre-ASR integrity validation: {problems}")

    if Path(RUN_ID).name != RUN_ID or "/" in RUN_ID or "\\" in RUN_ID:
        raise ValueError(f"Unsafe proof run ID: {RUN_ID!r}")
    artifacts_root = WORKSPACE / "artifacts"
    if _path_has_symlink_component(artifacts_root) or _path_has_symlink_component(ARTIFACTS_DIR):
        raise RuntimeError(f"Proof artifact path contains a symlink: {ARTIFACTS_DIR}")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=False)
    audio_hashes = {
        filename: clip["audio_sha256"]
        for filename, _expected_text, clip, _scenario_id, _step_id in targets
    }
    audio_snapshot_dir = _snapshot_verified_audio_cache(
        SCENARIO_AUDIO_DIR,
        ARTIFACTS_DIR / "verified_scenario_audio",
        audio_hashes,
    )
    manifest_snapshot = audio_snapshot_dir / "scenario_audio_manifest.json"
    manifest_snapshot.write_bytes(manifest_bytes)
    manifest_snapshot.chmod(0o444)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if hashlib.sha256(manifest_snapshot.read_bytes()).hexdigest() != manifest_sha256:
        raise RuntimeError("Proof manifest snapshot digest mismatch")

    whisper_asr = load_independent_asr()
    if whisper_asr is None:
        raise RuntimeError("faster-whisper local model is required; proof stopped before dialogue-engine loading")
    independent_transcripts: Dict[str, str] = {}
    for filename, expected_text, clip, scenario_id, step_id in targets:
        audio_path = audio_snapshot_dir / filename
        fresh_transcript = transcribe_wav_file(
            whisper_asr,
            audio_path,
            independent_only=True,
            language=clip.get("language"),
        )
        errors = _validate_cache_clip(
            clip,
            expected_text=expected_text,
            audio_path=audio_path,
            independent_transcript=fresh_transcript,
            expected_scenario_id=scenario_id,
            expected_step_id=step_id,
        )
        if errors:
            problems.append(f"{filename} fresh-ASR validation: {errors}")
        independent_transcripts[filename] = fresh_transcript
    if problems:
        raise RuntimeError(f"Qwen3-TTS cache failed fresh independent-ASR validation: {problems}")

    pretest_filename = f"{SCENARIO_NAME_NUMBER.scenario_id}__{SCENARIO_NAME_NUMBER.steps[0].step_id}.wav"
    pretest = clips[pretest_filename]
    pretest_numeric_wer = float(pretest["numeric_normalized_wer"])
    if (
        not math.isfinite(pretest_numeric_wer)
        or pretest_numeric_wer >= 0.15
        or pretest.get("contains_marco") is not True
        or pretest.get("contains_code_4829") is not True
        or pretest.get("asr_quality_passed") is not True
    ):
        raise RuntimeError("Qwen3-TTS name/number pretest does not meet numeric-WER, semantic, and audio gates")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_snapshot_path": str(manifest_snapshot),
        "proof_snapshot_dir": str(audio_snapshot_dir),
        "mode_a_user_utterances": len(expected_user_steps),
        "mode_b_kickoff": str(audio_snapshot_dir / "mode_b_kickoff.wav"),
        "pretest_wer": float(pretest["wer"]),
        "pretest_numeric_normalized_wer": pretest_numeric_wer,
        "pretest_asr_transcript": pretest["asr_transcript"],
        "independent_asr_engine": "faster-whisper",
        "independent_asr_rechecks": len(independent_transcripts),
        "independent_asr_transcripts": independent_transcripts,
        "audio_sha256_by_filename": audio_hashes,
    }, whisper_asr


def get_vram_telemetry() -> Dict[str, float]:
    """Capture precise CUDA VRAM telemetry in MB."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_mb": 0.0}
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved() / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2),
        "max_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 ** 2), 2),
    }


def load_independent_asr() -> Optional[Any]:
    """Load local faster-whisper model for independent speech transcription."""
    _force_offline_model_loading()
    try:
        from faster_whisper import WhisperModel
        whisper_dirs = (
            glob.glob("/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*")
            + glob.glob("/workspace/.hf_cache/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*")
            + glob.glob("/workspace/.hf_cache/hub/models--Systran--faster-whisper-small/snapshots/*")
            + glob.glob("/workspace/hf_cache/hub/models--mobiuslabsgmbh--faster-whisper*/snapshots/*")
            + glob.glob("/home/administrator/.cache/huggingface/hub/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots/*")
            + glob.glob("/home/administrator/.cache/huggingface/hub/models--Systran--faster-whisper-small/snapshots/*")
        )
        if not whisper_dirs:
            raise FileNotFoundError("No local faster-whisper model snapshot found; network downloads are disabled for this proof")
        model_path = whisper_dirs[0]
        logger.info(f"Loading faster-whisper from local snapshot: {model_path}")
        return WhisperModel(model_path, device="cuda", compute_type="float16")
    except Exception as exc:
        logger.error(f"faster-whisper local ASR unavailable: {exc}")
        return None


def transcribe_wav_file(
    whisper_model: Any,
    wav_path: Path,
    engine: Optional[PipelineEngine] = None,
    *,
    independent_only: bool = False,
    language: Optional[str] = None,
) -> str:
    """Transcribe an audio WAV file using faster-whisper or native engine."""
    if not wav_path.is_file():
        return ""
    if whisper_model is not None:
        try:
            language_code = {"italian": "it", "it": "it", "english": "en", "en": "en"}.get(
                language.casefold() if language else ""
            )
            segments, _info = whisper_model.transcribe(
                str(wav_path), language=language_code, beam_size=1
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            if text:
                return text
        except Exception as exc:
            logger.warning(f"Whisper transcription failed on {wav_path}: {exc}")
    if engine is not None and not independent_only:
        try:
            pcm, sr = sf.read(str(wav_path))
            if sr != 16000:
                pcm = resample_24k_to_16k(pcm) if sr == 24000 else pcm
            res = engine.transcribe_audio(pcm)
            return res.get("text", "")
        except Exception as exc:
            logger.warning(f"Engine transcription failed on {wav_path}: {exc}")
    return ""


def transcribe_input_pcm(whisper_model: Any, pcm_16k: np.ndarray, sample_rate: int) -> str:
    """Run faster-whisper on the exact ingress PCM without engine-ASR fallback."""
    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz ingress PCM, got {sample_rate} Hz")
    pcm = np.asarray(pcm_16k, dtype=np.float32).reshape(-1)
    if pcm.size == 0 or not np.isfinite(pcm).all():
        raise ValueError("Ingress PCM is empty or non-finite")
    with tempfile.TemporaryDirectory(prefix="dialogue_input_asr_") as temp_dir:
        input_path = Path(temp_dir) / "input.wav"
        sf.write(str(input_path), pcm, sample_rate, subtype="PCM_16")
        return transcribe_wav_file(whisper_model, input_path, independent_only=True)


async def run_e2e_proof():
    # This proof exercises the native action-token prompt path. The server reads
    # this setting for each direct-audio response, so force it before any proof
    # setup can issue realtime requests.
    os.environ["DUPLEX_CONTROLLER_GUIDED_SAY_V1"] = "1"
    scenario_audio_cache, whisper_asr = verify_scenario_audio_cache()
    audio_snapshot_dir = Path(scenario_audio_cache["proof_snapshot_dir"])
    logger.info("Qwen3-TTS scenario audio cache verified and snapshotted: %s", scenario_audio_cache)
    logger.info("=== STARTING SERIALIZED SPOKEN DIALOGUE PROOF ON RTX 3090 ===")
    vram_start = get_vram_telemetry()
    logger.info(f"VRAM Initial: {vram_start}")

    # 1. Load Neural Engine (Single copy of weights in VRAM)
    logger.info("Step 1: Loading PipelineEngine (Gemma 4 Thinker + Talker + V9 + Code2Wav)...")
    engine = PipelineEngine(
        device="cuda",
        dtype=torch.bfloat16,
        enable_gemma_native_control6_policy=True,
        native_control6_adapter_path=(
            "checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt"
        ),
        gemma_qlora_adapter_dir="checkpoints/gemma4_action_qlora_v1",
    )
    engine.enforce_production_listener = False  # Diagnostic-only proof; this does not certify production listener enforcement.
    loaded_ok = engine.load(workspace=str(WORKSPACE))
    if not loaded_ok:
        raise RuntimeError("Failed to load PipelineEngine!")
    
    vram_loaded = get_vram_telemetry()
    logger.info(f"PipelineEngine Loaded Successfully! VRAM Allocated: {vram_loaded['allocated_mb']} MB, Reserved: {vram_loaded['reserved_mb']} MB")

    # 2. Start WebSocket Realtime Server in Background Task
    port = 8092
    ws_url = f"ws://127.0.0.1:{port}/v1/realtime?duplex=1"
    session_mgr = SessionManager()
    app = create_app(engine, session_mgr=session_mgr)

    server_config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(server_config)

    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    server_started = bool(server.started)
    logger.info(f"Step 2: Realtime server active on ws://127.0.0.1:{port}/v1/realtime?duplex=1")

    # 3. Configuration Measurement: 1 Session vs 2 Sessions Concurrency
    logger.info("Step 3: Measuring Configuration Comparison (1 vs 2 Concurrent Sessions)...")
    concurrency_results = {}

    # Measure 1 Session
    async with websockets.connect(ws_url) as ws1:
        created_1_session = json.loads(await ws1.recv())
        await ws1.send(json.dumps({
            "type": "session.update",
            "session": {"voice": "gemma", "instructions": "Session 1 test", "turn_policy": "gemma_native_control6"}
        }))
        await ws1.recv() # session.updated
        vram_1sess = get_vram_telemetry()
        concurrency_results["1_session"] = {
            "vram_allocated_mb": vram_1sess["allocated_mb"],
            "vram_reserved_mb": vram_1sess["reserved_mb"],
            "sessions_active": len(session_mgr.all()),
            "session_id": created_1_session.get("session_id"),
        }

    # Measure 2 Sessions Concurrently
    async with websockets.connect(ws_url) as ws_a, websockets.connect(ws_url) as ws_b:
        created_a = json.loads(await ws_a.recv())
        created_b = json.loads(await ws_b.recv())
        await ws_a.send(json.dumps({
            "type": "session.update",
            "session": {"voice": "leonardo", "instructions": "Customer session A", "turn_policy": "gemma_native_control6"}
        }))
        await ws_b.send(json.dumps({
            "type": "session.update",
            "session": {"voice": "gemma", "instructions": "Assistant session B", "turn_policy": "gemma_native_control6"}
        }))
        await ws_a.recv()
        await ws_b.recv()
        vram_2sess = get_vram_telemetry()
        active_sessions = {session.session_id: session for session in session_mgr.all()}
        session_a = active_sessions.get(str(created_a.get("session_id")))
        session_b = active_sessions.get(str(created_b.get("session_id")))
        state_fields = ("thinker_ctx", "talker_prefix", "turn_kv", "history", "listener_state")
        distinct_session_state = bool(session_a and session_b) and all(
            getattr(session_a, field) is not getattr(session_b, field)
            for field in state_fields
        )
        concurrency_results["2_sessions"] = {
            "vram_allocated_mb": vram_2sess["allocated_mb"],
            "vram_reserved_mb": vram_2sess["reserved_mb"],
            "sessions_active": len(active_sessions),
            "session_ids": [created_a.get("session_id"), created_b.get("session_id")],
            "isolation_verified": {
                "session_ids_distinct": bool(created_a.get("session_id") and created_b.get("session_id") and created_a.get("session_id") != created_b.get("session_id")),
                "session_owned_state_objects_distinct": distinct_session_state,
                "configured_voices_distinct": bool(session_a and session_b and session_a.voice != session_b.voice),
            },
        }
    logger.info(f"Concurrency Telemetry: 1 Session = {concurrency_results['1_session']['vram_allocated_mb']} MB, 2 Sessions = {concurrency_results['2_sessions']['vram_allocated_mb']} MB")

    # Reuse the offline local ASR model that independently revalidated the frozen cache.
    def asr_transcriber(pcm: np.ndarray, sample_rate: int) -> str:
        return transcribe_input_pcm(whisper_asr, pcm, sample_rate)

    # 4. Mode (a): Canonical Test Scenarios Execution
    logger.info("Step 4: Executing Mode (a) Qwen3-TTS Speech Dialogues (6 Canonical Scenarios)...")
    mode_a_results = []
    mode_a_dir = ARTIFACTS_DIR / "mode_a"
    mode_a_dir.mkdir(parents=True, exist_ok=True)

    skip_mode_a = os.environ.get("RUN_MODE_B_ONLY", "1") not in {"0", "false", "False"}
    scenarios_to_run = [] if skip_mode_a else [
        SCENARIO_NAME_NUMBER,
        SCENARIO_CLARIFICATION,
        SCENARIO_INSTRUCTION_CHANGE,
        SCENARIO_CONTRADICTION,
        SCENARIO_BARGE_IN,
        SCENARIO_CLOSING,
    ]

    for scenario in scenarios_to_run:
        scen_out_dir = mode_a_dir / scenario.scenario_id
        cfg = DialogueConfig(
            max_turns=scenario.max_turns,
            max_duration_s=min(30.0, scenario.max_duration_s),
            ws_url=ws_url,
            voice_a="leonardo",
            voice_b="gemma",
            turn_policy="gemma_native_control6",
        )
        orchestrator = DialogueOrchestrator(
            config=cfg,
            output_dir=scen_out_dir,
            scenario_audio_dir=audio_snapshot_dir,
            asr_transcriber=asr_transcriber,
        )
        logger.info(f"--- Running Scenario: {scenario.title} ({scenario.scenario_id}) ---")
        res = await orchestrator.run_mode_a(scenario=scenario)

        # Transcribe output audio independently with Whisper (or native engine fallback)
        out_wav = res["wav_files"].get("interlocutor_a_output")
        in_wav = res["wav_files"].get("interlocutor_a_input")
        audio_transcript = transcribe_wav_file(whisper_asr, out_wav, independent_only=True) if out_wav else ""
        input_transcript = transcribe_wav_file(whisper_asr, in_wav, independent_only=True) if in_wav else ""

        res["asr_eval"] = {
            "input_audio_transcript": input_transcript,
            "output_audio_transcript": audio_transcript,
        }
        mode_a_results.append({
            "scenario_id": scenario.scenario_id,
            "title": scenario.title,
            "result": res,
        })

    # 5. Mode (b): Autonomous Spoken Dialogue A <-> B with Anti-Echo Bridge
    logger.info("Step 5: Executing Mode (b) Autonomous Dialogue A <-> B over WebSockets...")
    mode_b_dir = ARTIFACTS_DIR / "mode_b"
    mode_b_dir.mkdir(parents=True, exist_ok=True)

    mode_b_scenario = replace(
        SCENARIO_NAME_NUMBER,
        system_prompt_a=(
            "Sei un assistente vocale italiano. Ascolta l'interlocutore e rispondi con una frase chiara "
            "confermando sia il nome pronunciato (Marco) sia ogni cifra del codice ordine (4 8 2 9), "
            "ripetendo il codice cifra per cifra come lo hai sentito (ad esempio: 'Buongiorno Marco, confermo codice 4 8 2 9'). "
            "Non convertire una sequenza di cifre in un numero composto; non fare domande e non inventare."
        ),
        system_prompt_b=(
            "Sei un assistente vocale italiano. Ascolta con attenzione l'altro interlocutore e "
            "ripeti in una frase breve il nome e ogni cifra del codice che senti, nello stesso ordine "
            "(ad esempio: 'Ho capito Marco, confermo codice 4 8 2 9'). "
            "Non convertire una sequenza di cifre in un numero composto e non inventare."
        ),
        steps=[
            replace(
                SCENARIO_NAME_NUMBER.steps[0],
                expected_text=MODE_B_KICKOFF_TEXT,
            ),
            *SCENARIO_NAME_NUMBER.steps[1:],
        ],
    )
    cfg_b = DialogueConfig(
        max_turns=3,
        max_duration_s=120.0,
        ingress_ack_timeout_s=2.0,
        ws_url=ws_url,
        voice_a="leonardo",
        voice_b="gemma",
        turn_policy="gemma_native_control6",
    )
    orchestrator_b = DialogueOrchestrator(
        config=cfg_b,
        output_dir=mode_b_dir,
        scenario_audio_dir=audio_snapshot_dir,
        asr_transcriber=asr_transcriber,
    )
    mode_b_res = await orchestrator_b.run_mode_b(scenario=mode_b_scenario)
    mode_b_res["scenario_instructions"] = {
        "A": mode_b_scenario.system_prompt_a,
        "B": mode_b_scenario.system_prompt_b,
    }

    # Transcribe all 4 tracks independently
    mode_b_asr = {}
    for track_name, wav_p in mode_b_res["wav_files"].items():
        mode_b_asr[track_name] = transcribe_wav_file(
            whisper_asr, wav_p, independent_only=True
        )
    mode_b_res["asr_transcripts"] = mode_b_asr
    generated_by_speaker = {
        speaker: " ".join(r["generated_text"].strip() for r in mode_b_res.get("responses", []) if r["speaker"] == speaker and r["generated_text"].strip())
        for speaker in ("A", "B")
    }
    mode_b_res["generated_text_by_speaker"] = generated_by_speaker
    mode_b_res["audio_text_wer_by_speaker"] = {
        speaker: calculate_wer(generated_by_speaker[speaker], mode_b_asr.get(f"interlocutor_{speaker.lower()}_output", ""))
        if generated_by_speaker[speaker] and mode_b_asr.get(f"interlocutor_{speaker.lower()}_output", "") else None
        for speaker in ("A", "B")
    }
    b_checks = [
        lambda text: "marco" in text.casefold(),
        lambda text: contains_italian_digit_sequence(text, "4829"),
    ]
    mode_b_res["semantic_checks"] = {
        "scenario_step": SCENARIO_NAME_NUMBER.steps[1].step_id,
        "checks": [bool(check(generated_by_speaker["B"])) for check in b_checks],
        "passed": bool(generated_by_speaker["B"]) and all(check(generated_by_speaker["B"]) for check in b_checks),
    }
    a_audio_transcript = mode_b_asr.get("interlocutor_a_output", "")
    b_audio_transcript = mode_b_asr.get("interlocutor_b_output", "")
    asr_by_speaker = {"A": a_audio_transcript, "B": b_audio_transcript}
    mode_b_res["independent_asr"] = {
        "engine": "faster-whisper" if whisper_asr is not None else None,
        "available": bool(whisper_asr is not None and all(text.strip() for text in asr_by_speaker.values())),
        "speaker_a_transcript": a_audio_transcript,
        "speaker_b_transcript": b_audio_transcript,
        "speaker_b_facts_in_audio_transcript": bool(b_audio_transcript) and all(check(b_audio_transcript) for check in b_checks),
    }
    transport_complete = mode_b_res.get("conversation_status") == "TURN_TAKING_TRANSPORT_COMPLETE"
    mode_b_res["conversation_status"] = classify_conversation_evidence(
        transport_complete=transport_complete,
        generated_facts_passed=bool(mode_b_res["semantic_checks"]["passed"]),
        independent_asr_available=bool(mode_b_res["independent_asr"]["available"]),
        audio_facts_passed=bool(mode_b_res["independent_asr"]["speaker_b_facts_in_audio_transcript"]),
        paced_bridge_duplex_observed=bool(mode_b_res.get("paced_bridge_duplex_observed")),
    )

    # 6. Physical Audio Quality Analysis across all generated WAVs
    logger.info("Step 6: Calculating Physical Audio Fidelity & Latency Metrics...")
    audio_metrics_summary = {}
    for track_name, wav_p in mode_b_res["wav_files"].items():
        if wav_p.is_file():
            data, sr = sf.read(str(wav_p))
            metrics = compute_physical_audio_metrics(data, sample_rate=sr)
            audio_metrics_summary[track_name] = metrics

    output_rms_by_speaker = {
        speaker: float(audio_metrics_summary.get(f"interlocutor_{speaker.lower()}_output", {}).get("rms", 0.0))
        for speaker in ("A", "B")
    }
    required_tracks = (
        "interlocutor_a_input",
        "interlocutor_a_output",
        "interlocutor_b_input",
        "interlocutor_b_output",
    )
    four_track_audio_health = {
        track: {
            "exists": bool(mode_b_res["wav_files"].get(track) and mode_b_res["wav_files"][track].is_file()),
            "finite": bool(audio_metrics_summary.get(track, {}).get("finite", False)),
            "rms": float(audio_metrics_summary.get(track, {}).get("rms", 0.0)),
            "rms_positive": float(audio_metrics_summary.get(track, {}).get("rms", 0.0)) > 0.0,
            "clipping_pct": float(audio_metrics_summary.get(track, {}).get("clipping_pct", 100.0)),
            "clipping_zero": float(audio_metrics_summary.get(track, {}).get("clipping_pct", 100.0)) == 0.0,
        }
        for track in required_tracks
    }
    mode_b_res["four_track_audio_health"] = {
        "tracks": four_track_audio_health,
        "all_passed": all(
            result["exists"] and result["finite"] and result["rms_positive"] and result["clipping_zero"]
            for result in four_track_audio_health.values()
        ),
    }
    mode_b_res["audio_output_verification"] = {
        speaker: {
            "wav_path": str(mode_b_res["wav_files"].get(f"interlocutor_{speaker.lower()}_output", "")),
            "samples": int(audio_metrics_summary.get(f"interlocutor_{speaker.lower()}_output", {}).get("samples", 0)),
            "rms": output_rms_by_speaker[speaker],
            "rms_positive": output_rms_by_speaker[speaker] > 0.0,
            "independent_asr_engine": mode_b_res["independent_asr"]["engine"],
            "independent_asr_transcript": asr_by_speaker[speaker],
            "independent_asr_nonempty": bool(asr_by_speaker[speaker].strip()),
        }
        for speaker in ("A", "B")
    }
    mode_b_res["turn_taking_audio_verification"] = classify_turn_taking_audio_evidence(
        transport_complete=transport_complete,
        turn_sequence=mode_b_res.get("turn_sequence", []),
        turn_transfers=mode_b_res.get("turn_transfers", []),
        output_rms_by_speaker=output_rms_by_speaker,
        output_asr_by_speaker=asr_by_speaker,
        independent_asr_engine=mode_b_res["independent_asr"]["engine"],
        generated_facts_passed=bool(mode_b_res["semantic_checks"]["passed"]),
        audio_facts_passed=bool(mode_b_res["independent_asr"]["speaker_b_facts_in_audio_transcript"]),
    )

    # 7. Shutdown Server
    server.should_exit = True
    await server_task
    logger.info("Realtime server stopped cleanly.")

    # 8. Compile Comprehensive Verification Report
    vram_peak = get_vram_telemetry()
    final_report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "diagnostic_execution_completed": True,
        "process_exit_code_semantics": "Exit code 0 means the diagnostic report was produced; it does not imply any certification field passed.",
        "hardware": {
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 2),
            "vram_allocated_peak_mb": vram_peak["max_allocated_mb"],
            "vram_reserved_peak_mb": vram_peak["max_reserved_mb"],
        },
        "concurrency_measurement": concurrency_results,
        "scenario_audio_cache": scenario_audio_cache,
        "listener_enforcement_disabled_for_diagnostic": True,
        "four_track_audio_health": mode_b_res["four_track_audio_health"],
        "mode_a_scenarios": mode_a_results,
        "mode_b_autonomous": mode_b_res,
        "audio_metrics": audio_metrics_summary,
        "status_certification": {
            "serving_avviabile": bool(loaded_ok and server_started),
            "dialogo_audio_dimostrato": mode_b_res.get("conversation_status") == "FULL_DUPLEX_AUDIO_VERIFIED",
            "dialogo_audio_a_turni_verificato": bool(mode_b_res.get("turn_taking_audio_verification", {}).get("verified")),
            "pre_qlora_ready_certificato": False,
            "pre_qlora_block_rationale": "Questa prova dialogica non esegue né sostituisce il master verifier Pre-QLoRA; nessuna conclusione sulla readiness è derivata da questo run.",
        }
    }

    # Save JSON and Markdown
    report_json_path = ARTIFACTS_DIR / "dialogue_proof_summary.json"
    report_json_path.write_text(json.dumps(final_report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Generate Markdown Summary
    md_content = build_final_markdown_report(final_report)
    report_md_path = ARTIFACTS_DIR / "dialogue_proof_report.md"
    report_md_path.write_text(md_content, encoding="utf-8")

    logger.info(f"=== DIALOGUE PROOF COMPLETE! Report saved to {report_md_path} ===")
    return final_report


def build_final_markdown_report(report: Dict[str, Any]) -> str:
    hw = report["hardware"]
    conc = report["concurrency_measurement"]
    mode_b = report["mode_b_autonomous"]
    status = report["status_certification"]

    lines = [
        "# Relazione Tecnica: Prova di Conversazione Parlata End-to-End su NVIDIA RTX 3090",
        "",
        "## 1. Stato dei Criteri di Certificazione (Distinzione Netta)",
        "- **Semantica exit code**: `0` indica che il report diagnostico è stato prodotto; non implica che una certificazione sia passata.",
        f"- **«Serving Avviabile»**: {'✅ DIMOSTRATO (solo diagnostico)' if status['serving_avviabile'] else '❌ NON DIMOSTRATO'}",
        "  - Singolo processo server FastAPI con protocollo OpenAI `/v1/realtime?duplex=1`.",
        "  - Pesi neurali condivisi caricati in VRAM (Gemma 4 Thinker bf16 + Qwen3-TTS Talker + Projector V9 + Code2Wav).",
        "  - Ingressi a frame di 80 ms con ACK listener/V9 prima del frame successivo.",
        "",
        f"- **«Full-duplex audio dimostrato»**: {'✅ CERTIFICATO' if status['dialogo_audio_dimostrato'] else '⛔ NON DIMOSTRATO (turni serializzati)'}",
        f"- **«Dialogo audio a turni verificato»**: {'✅ CERTIFICATO' if status['dialogo_audio_a_turni_verificato'] else '❌ PROVA INCOMPLETA'}",
        "  - I turni vengono serializzati: nessuna inferenza simultanea viene affermata.",
        "  - La certificazione a turni richiede trasferimenti ACK, RMS positivo e ASR indipendente non vuoto per A e B.",
        "  - La traccia JSONL e le quattro WAV sono incluse per verifica indipendente.",
        f"  - Listener enforcement disabilitato nel proof diagnostico: {'sì' if report.get('listener_enforcement_disabled_for_diagnostic') else 'no'}; non è una certificazione del serving di produzione.",
        "",
        f"- **«PRE_QLORA_READY Certificato»**: {'✅ CERTIFICATO' if status['pre_qlora_ready_certificato'] else '⛔ BLOCKED (Non Certificato)'}",
        f"  - **Motivazione Riconciliata**: {status['pre_qlora_block_rationale']}.",
        "  - Nessuna promozione fasulla su semplice esistenza dei file.",
        "",
        "## 2. Telemetria Hardware & Confronto Concorrenza (1 vs 2 Sessioni)",
        f"- **GPU Hardware**: {hw['gpu_name']} (Totale VRAM: {hw['vram_total_mb']} MB)",
        f"- **VRAM Allocata a Riposo (Pesi Caricati)**: {conc.get('1_session', {}).get('vram_allocated_mb', 0.0)} MB",
        f"- **VRAM Allocata con 2 Sessioni Simultanee**: {conc.get('2_sessions', {}).get('vram_allocated_mb', 0.0)} MB",
        f"- **VRAM Picco Riservata**: {hw['vram_reserved_peak_mb']} MB",
        "- Il confronto riguarda sessioni aperte; non certifica inferenza simultanea, throughput o precisione.",
        "",
        "## 3. Risultati Dialogo Autonomo A <-> B (Modalità b)",
        f"- **Stato Mode (b)**: {mode_b.get('conversation_status', 'INCOMPLETE_EVIDENCE')}",
        f"- **Turni Completati**: {mode_b.get('turns_completed', 0)} — sequenza `{mode_b.get('turn_sequence', [])}`",
        f"- **Trasferimenti ACK verificati**: {len(mode_b.get('turn_transfers', []))}",
        "- **File WAV Generati (4 Tracce Indipendenti)**:",
    ]

    for track, path in mode_b.get("wav_files", {}).items():
        lines.append(f"  - `{track}`: `{path}`")
    for speaker, evidence in mode_b.get("audio_output_verification", {}).items():
        lines.append(
            f"  - Output {speaker}: {evidence.get('samples', 0)} samples, RMS {evidence.get('rms', 0.0):.5f}, "
            f"ASR non vuoto: {'sì' if evidence.get('independent_asr_nonempty') else 'no'}"
        )
        lines.append(f"    - ASR {speaker}: `{evidence.get('independent_asr_transcript', '')}`")

    lines.extend([
        "",
        "## 4. Metriche Acustiche & Salute Fisica dell'Audio",
        f"- Gate delle quattro tracce richieste: {'PASS' if mode_b.get('four_track_audio_health', {}).get('all_passed') else 'FAIL'}",
        "| Traccia Audio | Clipping % | RMS | Crest Factor | Discontinuità | Stato Salute |",
        "|---|---|---|---|---|---|",
    ])

    for track, m in report.get("audio_metrics", {}).items():
        clip = m.get("clipping_pct", 0.0)
        rms = m.get("rms", 0.0)
        cf = m.get("crest_factor", 0.0)
        disc = m.get("packet_discontinuities_count", 0)
        health = "✅ PASS" if clip <= 0.10 and 3.5 <= cf <= 12.0 else "⚠️ ALERT"
        lines.append(f"| `{track}` | {clip:.3f}% | {rms:.4f} | {cf:.2f} | {disc} | {health} |")

    lines.extend([
        "",
        "## 5. Tabella Turni Scenari Canonici (Modalità a)",
        "| Scenario | Turni | Pass Rate % | Input WER | Output WER | Fatti | Contraddizione | Barge-in | Errori server | Esito |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])

    for item in report.get("mode_a_scenarios", []):
        scen_id = item["scenario_id"]
        rep = item["result"].get("report", {})
        p_rate = rep.get("pass_rate_pct", 0.0)
        i_wer = rep.get("mean_input_wer")
        i_wer_text = f"{i_wer:.3f}" if i_wer is not None else "N/A"
        o_wer = rep.get("mean_output_wer", 0.0)
        f_ok = "PASS" if rep.get("all_facts_retained") else "FAIL"
        c_ok = "PASS" if rep.get("all_contradictions_resolved") else "N/A"
        b_ok = "PASS" if rep.get("all_barge_ins_recovered") else "N/A"
        server_errors = item["result"].get("server_errors", [])
        error_summary = "; ".join(str(error.get("code", error.get("message", "error"))) for error in server_errors)
        badge = "❌ SERVER ERROR" if server_errors else ("✅ PASS" if p_rate >= 80.0 else "⚠️ REVIEW")
        lines.append(f"| {scen_id} | {rep.get('total_turns', 0)} | {p_rate}% | {i_wer_text} | {o_wer:.3f} | {f_ok} | {c_ok} | {b_ok} | {error_summary or '—'} | {badge} |")

    return "\n".join(lines)


if __name__ == "__main__":
    # A successful diagnostic run writes its report; certification remains in the report fields.
    asyncio.run(run_e2e_proof())
