#!/usr/bin/env python3
"""Reproduce the prefix/Q0-vs-Q2 probe with the production GemmaListener worker.

Default mode is a CPU-only tokenizer/import preflight.  The model/runtime path
is entered only with --run, after the caller has an explicit GPU handoff.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any


TARGET_B_PROMPT = (
    "Sei un assistente conversazionale italiano. Ascolta con attenzione e rispondi in modo naturale e conciso, "
    "al massimo due frasi. Mantieni i dati effettivamente comunicati e applica le correzioni più recenti. "
    "Se non hai capito, chiedi chiarimenti. Non inventare informazioni."
)
RATE_IN = 16_000
FRAME_MS = 80
FRAME_SAMPLES = RATE_IN * FRAME_MS // 1000
DEFAULT_WORKSPACE = Path("/workspace")
DEFAULT_INPUT = Path("/workspace/artifacts/conversation_quality_20261014/inputs/pair_seed_it.wav")
DEFAULT_STAGE2_RESULTS = Path(
    "/workspace/artifacts/conversation_quality_20261021/"
    "control6_prefix_sensitivity_build6_20261021/prefix_probe_results.json"
)
DEFAULT_DEBUG_INPUT = Path(
    "/workspace/artifacts/conversation_quality_20261021/"
    "control6_fullab3000_optimized_build6_20261021/native_debug_inputs/"
    "578bcbd450814802883c72e0595c0b1d.npz"
)
DEFAULT_EVAL_CONTEXT = Path(
    "/workspace/artifacts/conversation_quality_20261021/"
    "control6_prefix_sensitivity_build6_20261021/eval_context.json"
)
DEFAULT_FULL_HEAD = Path(
    "/workspace/checkpoints/native_control6_gemma_native_rows_rank16_"
    "runtime_effective_balanced_fullab_pcm_temporal_native1540_3000_20261021.pt"
)
DEFAULT_WARM_HEAD = Path(
    "/workspace/checkpoints/native_control6_gemma_native_rows_rank16_real_mixed_600.pt"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="run the CUDA probe; omitted means CPU preflight only")
    parser.add_argument("--preflight", action="store_true", help="explicitly select CPU-only preflight (default)")
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--source-root", type=Path, required=True, help="prepared bundle12 serving source root")
    parser.add_argument("--training-source", type=Path, default=Path(
        "/workspace/artifacts/runtime/control6_training_source_balanced_20261020"
    ))
    parser.add_argument("--dso", type=Path, required=True, help="the unchanged bundle12 DSO11 file")
    parser.add_argument("--base-model", type=Path, default=Path(
        "/workspace/.hf_cache/models--google--gemma-4-E2B-it/full"
    ))
    parser.add_argument("--input-wav", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stage2-prefix-results", type=Path, default=DEFAULT_STAGE2_RESULTS)
    parser.add_argument("--debug-input", type=Path, default=DEFAULT_DEBUG_INPUT)
    parser.add_argument("--eval-context", type=Path, default=DEFAULT_EVAL_CONTEXT)
    parser.add_argument("--output-dir", type=Path, default=Path(
        "/workspace/artifacts/conversation_quality_20261021/"
        "control6_prefix_sensitivity_build6_20261021"
    ))
    parser.add_argument("--frames", type=int, default=15, help="capture f0 through this exclusive frame limit")
    parser.add_argument(
        "--head", action="append", default=None, metavar="NAME=CHECKPOINT",
        help="optional native Control6 head to score; may be repeated",
    )
    return parser


def _resolve_head_args(values: list[str] | None) -> list[tuple[str, Path]]:
    if values is None:
        values = [f"full3000={DEFAULT_FULL_HEAD}", f"warm={DEFAULT_WARM_HEAD}"]
    result: list[tuple[str, Path]] = []
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--head must be NAME=CHECKPOINT, got {value!r}")
        result.append((name.strip(), Path(path.strip())))
    return result


def _configure_import_environment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    workspace = args.workspace.resolve()
    source_root = args.source_root.resolve()
    training_source = args.training_source.resolve()
    dso = args.dso.resolve()
    if not (source_root / "duplex_kernel" / "server.py").is_file():
        raise FileNotFoundError(f"serving bundle source is incomplete: {source_root}")
    if not dso.is_file():
        raise FileNotFoundError(f"native DSO is missing: {dso}")
    if not training_source.is_dir():
        raise FileNotFoundError(f"training source mirror is missing: {training_source}")

    # The selected source/DSO/protocol are fixed before any duplex_kernel import.
    os.environ["DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT"] = str(source_root)
    os.environ["DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH"] = str(dso)
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = str(dso)
    os.environ["DUPLEX_AUDIO_RESPONSE_PROTOCOL"] = "heard_action_say_v2"
    os.environ["DUPLEX_CONTROLLER_GUIDED_SAY_V1"] = "0"
    for entry in (workspace, training_source, source_root):
        text = str(entry)
        while text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)
    return workspace, source_root, dso


def _activate_and_check_runtime_source(source_root: Path) -> Any:
    import importlib

    package = importlib.import_module("duplex_kernel")
    package_paths = [Path(value).resolve() for value in getattr(package, "__path__", ())]
    expected = (source_root / "duplex_kernel").resolve()
    if expected not in package_paths:
        raise RuntimeError(f"duplex_kernel package path does not include selected source: {package_paths}")
    for name, module in tuple(sys.modules.items()):
        if name == "duplex_kernel" or name.startswith("duplex_kernel."):
            filename = getattr(module, "__file__", None)
            if filename is not None:
                try:
                    Path(filename).resolve().relative_to(expected)
                except ValueError as exc:
                    raise RuntimeError(
                        f"{name} was imported outside selected bundle: {filename}"
                    ) from exc
    return importlib


def _stage2_prefix_ids(path: Path) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        values = payload["prefixes"]["prefix190"]["prefix"]["token_ids"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"cannot find saved Stage2 prefix190 token IDs in {path}") from exc
    if not isinstance(values, list) or not values or any(type(value) is not int for value in values):
        raise ValueError("saved Stage2 prefix190 token IDs are not a non-empty integer list")
    if len(values) != 190:
        raise ValueError(f"saved Stage2 prefix length changed: expected 190, got {len(values)}")
    return tuple(values)


def _check_resolved_base_path(thinker_module: Any, workspace: Path, requested: Path) -> Path:
    resolved = Path(thinker_module.resolve_thinker_path(str(workspace.resolve()))).resolve()
    expected = requested.resolve()
    if resolved != expected:
        raise RuntimeError(
            "serving resolve_thinker_path differs from --base-model: "
            f"resolved={resolved} requested={expected}"
        )
    return resolved


def _read_runner_pcm(path: Path) -> list[Any]:
    import numpy as np
    from scipy.signal import resample_poly

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
        pcm = resample_poly(pcm, RATE_IN // divisor, rate // divisor).astype(np.float32)
    pcm = np.clip(pcm, -1.0, 1.0)
    frames: list[Any] = []
    for start in range(0, pcm.size, FRAME_SAMPLES):
        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        chunk = pcm[start:start + FRAME_SAMPLES]
        frame[:chunk.size] = chunk
        # Match run_replay.py encode_pcm16(), then the server's PCM16 /32768 decode.
        encoded = np.rint(np.clip(frame, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        decoded = np.frombuffer(encoded, dtype="<i2").astype(np.float32) / 32768.0
        frames.append(decoded)
    return frames


def _builder_prefixes(args: argparse.Namespace, *, cpu_only: bool) -> dict[str, Any]:
    import importlib
    from types import SimpleNamespace

    if cpu_only:
        import torch
        from transformers import AutoConfig, AutoProcessor

        if torch.cuda.is_available():
            raise RuntimeError("CPU preflight requires CUDA_VISIBLE_DEVICES empty")
        if not args.base_model.is_dir():
            raise FileNotFoundError(f"base model directory is not local: {args.base_model}")
        thinker_module = importlib.import_module("duplex_kernel.models.thinker")
        resolved_thinker_path = _check_resolved_base_path(thinker_module, args.workspace, args.base_model)
        processor = AutoProcessor.from_pretrained(str(args.base_model), local_files_only=True)
        config = AutoConfig.from_pretrained(str(args.base_model), local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        engine = SimpleNamespace(
            tokenizer=tokenizer,
            _thinker_wrapper=SimpleNamespace(processor=processor, config=config),
        )
        fake_listener = SimpleNamespace(backend=SimpleNamespace(boa_token_id=getattr(config, "boa_token_id", None)))
    else:
        thinker_module = importlib.import_module("duplex_kernel.models.thinker")
        resolved_thinker_path = _check_resolved_base_path(thinker_module, args.workspace, args.base_model)
        thinker = thinker_module.load_thinker(str(args.workspace.resolve()), device="cuda", dtype=__import__("torch").bfloat16)
        model = getattr(thinker, "model", None)
        if model is None or getattr(thinker, "processor", None) is None:
            raise RuntimeError("serving load_thinker did not return the BF16 Gemma model/processor")
        import torch
        if Path(str(getattr(model, "_gemma4_checkpoint_path", ""))).resolve() != resolved_thinker_path:
            raise RuntimeError("loaded Thinker model provenance differs from serving resolve_thinker_path")
        floating = [p for p in model.parameters() if p.is_floating_point()]
        if not floating or any(p.dtype != torch.bfloat16 for p in floating):
            raise RuntimeError("prefix probe requires the production BF16 serving base")
        if any("lora_" in name.lower() for name, _ in model.named_parameters()):
            raise RuntimeError("prefix probe must use base-only Thinker without response LoRA")
        thinker_for_run = thinker
        engine = SimpleNamespace(tokenizer=thinker.tokenizer, _thinker_wrapper=thinker)
        listener_module = importlib.import_module("duplex_kernel.listener")
        listener = listener_module.GemmaListener.from_thinker(thinker, runtime_profile="production")
        fake_listener = listener

    server = importlib.import_module("duplex_kernel.server")
    builder = server._build_unified_listener_prefix_token_ids
    stage2_session = SimpleNamespace(history=[], instructions="", listener=fake_listener)
    target_session = SimpleNamespace(history=[], instructions=TARGET_B_PROMPT, listener=fake_listener)
    stage2_ids = tuple(int(value) for value in builder(engine, stage2_session))
    target_ids = tuple(int(value) for value in builder(engine, target_session))
    saved_stage2_ids = _stage2_prefix_ids(args.stage2_prefix_results)
    if stage2_ids != saved_stage2_ids:
        raise RuntimeError(
            "bundle prefix builder no longer reproduces recorded Stage2 prefix190 IDs; "
            f"built={len(stage2_ids)} recorded={len(saved_stage2_ids)}"
        )
    if len(target_ids) != 265:
        raise RuntimeError(f"TARGET_B_PROMPT builder prefix length changed: expected 265, got {len(target_ids)}")
    if cpu_only:
        return {
            "classification": "builder_reconstructed_not_production_session_capture",
            "protocol": os.environ["DUPLEX_AUDIO_RESPONSE_PROTOCOL"],
            "builder_source": str(Path(server.__file__).resolve()),
            "target_b_prompt": TARGET_B_PROMPT,
            "history_messages": 0,
            "resolved_thinker_path": str(resolved_thinker_path),
            "target_prefix_len": len(target_ids),
            "target_prefix_token_ids": list(target_ids),
            "stage2_prefix_len": len(stage2_ids),
            "stage2_ids_match_saved_extraction": True,
            "stage2_prefix_token_ids": list(stage2_ids),
        }
    return {
        "engine": engine,
        "thinker": thinker_for_run,
        "listener": listener,
        "builder": builder,
        "target_ids": target_ids,
        "stage2_ids": stage2_ids,
        "builder_source": str(Path(server.__file__).resolve()),
        "resolved_thinker_path": str(resolved_thinker_path),
    }


def _cpu_preflight(args: argparse.Namespace, workspace: Path, source_root: Path) -> None:
    import json

    _activate_and_check_runtime_source(source_root)
    prefixes = _builder_prefixes(args, cpu_only=True)
    frames = _read_runner_pcm(args.input_wav)
    import numpy as np

    phase = np.load(args.debug_input)
    if "phase_t" not in phase.files or "frame_ids" not in phase.files:
        raise ValueError("debug input must contain actual phase_t and frame_ids arrays")
    if phase["phase_t"].shape[1:] != (4,):
        raise ValueError(f"debug phase schema is not native1540: {phase['phase_t'].shape}")
    if not args.eval_context.is_file():
        raise FileNotFoundError(f"missing saved runtime policy context: {args.eval_context}")
    if not frames or len(frames) < args.frames or args.frames < 15:
        raise ValueError(f"input must provide at least {args.frames} frames; got {len(frames)}")
    summary = {
        "mode": "cpu_preflight_no_model_weights_no_dso_load",
        "workspace": str(workspace),
        "source_root": str(source_root),
        "dso_present_not_loaded": str(args.dso.resolve()),
        "requested_base_model": str(args.base_model.resolve()),
        "input_wav": str(args.input_wav),
        "decoded_requantized_frames": len(frames),
        "first_frame_samples": int(frames[0].size),
        "debug_phase_frames": int(phase["phase_t"].shape[0]),
        "eval_context": str(args.eval_context),
        "prefixes": prefixes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _load_eval_context(path: Path, frames: int) -> list[dict[str, Any]]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list):
        raise ValueError("eval_context.json must be a list of per-frame runtime contexts")
    by_id = {int(row["frame_id"]): row for row in values if isinstance(row, dict) and "frame_id" in row}
    if any(index not in by_id for index in range(frames)):
        raise ValueError("saved policy context does not cover every requested input frame")
    return [by_id[index] for index in range(frames)]


def _capture_condition(
    *,
    name: str,
    instructions: str,
    prefix_ids: tuple[int, ...],
    q0_first_two: bool,
    args: argparse.Namespace,
    runtime: dict[str, Any],
    pcm_frames: list[Any],
) -> tuple[dict[str, Any], Any]:
    import inspect
    import numpy as np

    importlib = __import__("importlib")
    torch = importlib.import_module("torch")
    listener_module = importlib.import_module("duplex_kernel.listener")
    serving = importlib.import_module("duplex_kernel.listener_microbatch_serving")
    contract = importlib.import_module("duplex_kernel.microbatch_contract")
    listener = runtime["listener"]
    worker = serving.make_listener_microbatch_worker(
        listener,
        library_path=args.dso.resolve(),
        semantic_prefix_enabled=False,
        unified_kv_enabled=True,
    )
    state = None
    registered = False
    hidden_rows: list[Any] = []
    frame_rows: list[dict[str, Any]] = []
    registration_capture: dict[str, Any] = {}
    effective_info: dict[str, Any] = {}
    try:
        worker.start().result()
        backend = getattr(listener, "backend", None)
        effective_info = getattr(backend, "_listener_microbatch_runtime_info", {})
        if not isinstance(effective_info, dict):
            raise RuntimeError("production worker did not publish native runtime information")
        effective_library = Path(str(effective_info.get("library_path", ""))).resolve()
        if effective_library != args.dso.resolve():
            raise RuntimeError(f"worker loaded unexpected DSO: {effective_library}")
        if int(effective_info.get("stream_q_len", -1)) != 2:
            raise RuntimeError("production worker did not attest effective Q2")
        if getattr(listener, "runtime_profile", None) != "production":
            raise RuntimeError("prefix probe requires GemmaListener production profile")
        if int(getattr(backend, "mel_window_rows", -1)) != 240:
            raise RuntimeError("prefix probe requires effective W240 frontend")

        session_id = f"prefix-probe-{name}"
        state = listener_module.GemmaListenerState(
            sample_rate=RATE_IN,
            frame_duration_ms=FRAME_MS,
            context_frames=listener_module.DEFAULT_CONTEXT_FRAMES,
        )
        session = SimpleNamespace(
            session_id=session_id,
            listener=listener,
            listener_state=state,
            history=[],
            instructions=instructions,
        )
        # Rebuild through the exact selected bundle builder and refuse ID drift.
        built = tuple(int(value) for value in runtime["builder"](runtime["engine"], session))
        if built != prefix_ids:
            raise RuntimeError(f"condition {name} prefix IDs changed between preflight and registration")

        real_factory = serving.make_unified_prefix_import

        def capture_factory(actual_backend: Any, ids: Any) -> Any:
            captured = tuple(int(value) for value in ids)
            registration_capture.update({
                "factory": "duplex_kernel.listener_microbatch_serving.make_unified_prefix_import",
                "factory_source": str(Path(inspect.getsourcefile(real_factory) or "").resolve()),
                "factory_argument_token_ids": list(captured),
                "factory_argument_len": len(captured),
                "matches_session_builder": captured == built,
                "is_production_websocket_session_capture": False,
            })
            if captured != built:
                raise RuntimeError("prefix import factory received token IDs different from session builder")
            return real_factory(actual_backend, captured)

        importer = capture_factory(backend, prefix_ids)
        row_state = serving.initial_listener_session_state(
            session,
            worker.runtime_signature,
            native_context_id=(1 if name == "A_runtime265_q0" else 2 if name == "B_runtime265_q2" else 3),
            generation_id=1,
            cache_position=len(prefix_ids),
        )
        worker.register_session(row_state, prefix_import=importer).result()
        registered = True
        with torch.inference_mode():
            for frame_id, pcm in enumerate(pcm_frames[:args.frames]):
                logical_q_len = 0 if q0_first_two and frame_id < 2 else 2
                frame = contract.ListenerFrame(
                    session_id=session_id,
                    generation_id=1,
                    state_epoch_id=str(state.state_epoch_id),
                    frame_id=frame_id,
                    ready_at_ns=time.monotonic_ns(),
                    pcm=np.asarray(pcm, dtype=np.float32),
                    logical_q_len=logical_q_len,
                    ephemeral_reason="prespeech_silence" if logical_q_len == 0 else None,
                )
                output = worker.submit_tick((frame,)).result()[session_id]
                hidden = output.gemma_hidden_t.detach().float().cpu().numpy().reshape(-1)
                if hidden.shape != (1536,) or not np.isfinite(hidden).all():
                    raise RuntimeError(f"condition {name} returned invalid final-norm Q0 hidden")
                health = output.native_output_health
                if not bool(getattr(health, "checked", False)) or bool(getattr(health, "nonfinite", True)):
                    raise RuntimeError(f"condition {name} failed native output health: {health}")
                hidden_rows.append(hidden)
                frame_rows.append({
                    "frame_id": frame_id,
                    "logical_q_len": int(output.logical_q_len),
                    "cache_position_after_commit": int(output.cache_position),
                    "valid_history_tokens_after_commit": int(output.valid_history_tokens),
                    "sequence_position_after_commit": int(output.sequence_position),
                    "hidden_norm": float(np.linalg.norm(hidden)),
                    "native_health_status": str(getattr(health, "status", "unknown")),
                })
    finally:
        try:
            if registered:
                worker.close_session(session_id).result()
        finally:
            if state is not None:
                state.close()
            worker.stop(drain=True).result()
    return {
        "condition": name,
        "session_instructions": instructions,
        "history_messages": 0,
        "prefix_len": len(prefix_ids),
        "registration_capture": registration_capture,
        "runtime_info": effective_info,
        "prefix_builder_source": runtime["builder_source"],
        "frames": frame_rows,
        "hidden": np.stack(hidden_rows).astype(np.float32),
    }, torch


def _score_heads(
    *,
    hidden: Any,
    phase_rows: Any,
    context_rows: list[dict[str, Any]],
    pcm_frames: list[Any],
    heads: list[tuple[str, Path]],
) -> dict[str, Any]:
    import importlib
    import torch

    native_module = importlib.import_module("training.duplex_qlora.native_control6")
    policy_module = importlib.import_module("duplex_kernel.gemma_native_control6_policy")
    turn_module = importlib.import_module("duplex_kernel.turn_policy")
    scored: dict[str, Any] = {}
    for head_name, checkpoint in heads:
        adapter, load_report = native_module.load_native_control6_adapter(checkpoint, device="cpu")
        policy = policy_module.GemmaNativeControl6Policy(
            adapter,
            checkpoint_path=str(checkpoint.resolve()),
            trace_scores=True,
            phase_profile="native1540",
        )
        score_rows: list[dict[str, Any]] = []
        for frame_id, (hidden_row, phase_t, context) in enumerate(
            zip(hidden, phase_rows, context_rows, strict=True)
        ):
            legal = tuple(str(value) for value in context["legal_actions"])
            peer_turn_active = bool(context.get("peer_turn_active", False))
            speaker_active = bool(context.get("speaker_active", False))
            state = turn_module.PolicyInput(
                session_id="prefix-probe-score",
                frame_id=frame_id,
                user_pcm=pcm_frames[frame_id],
                speaker_active=speaker_active,
                peer_turn_active=peer_turn_active,
                generation_id=1,
                legal_actions=legal,
                native_control_hidden=torch.as_tensor(hidden_row, dtype=torch.float32),
                native_control_hidden_frame_id=frame_id,
                native_control_hidden_at_ns=time.monotonic_ns(),
                native_control_phase_t=torch.as_tensor(phase_t, dtype=torch.float32),
            )
            decision = policy.step(state)
            diagnostics = decision.diagnostics or {}
            effective = diagnostics.get("native_action_effective_probabilities", {})
            legal_effective = diagnostics.get("native_action_legal_effective_probabilities", {})
            score_rows.append({
                "frame_id": frame_id,
                "legal_actions": list(legal),
                "peer_turn_active": peer_turn_active,
                "speaker_active": speaker_active,
                "phase_source": "actual_debug_npz_phase_t",
                "phase_t": [float(value) for value in phase_t],
                "p_speak_effective": float(effective.get("SPEAK", 0.0)),
                "p_speak_legal": float(legal_effective.get("SPEAK", 0.0)),
                "effective_probabilities": effective,
                "legal_effective_probabilities": legal_effective,
                "candidate": diagnostics.get("native_control6_effective_candidate"),
                "applied_action": decision.action.value,
                "hysteresis_count": diagnostics.get("native_control6_hysteresis_count"),
                "hysteresis_pending": diagnostics.get("native_control6_hysteresis_pending"),
                "confidence": float(decision.confidence),
            })
        active = [row for row in score_rows if row["peer_turn_active"] and not row["speaker_active"]]
        scored[head_name] = {
            "checkpoint": str(checkpoint.resolve()),
            "load_report_phase_profile": load_report.get("native_control6_phase_profile"),
            "rows": score_rows,
            "max_p_speak_legal_peer_active": max((row["p_speak_legal"] for row in active), default=None),
            "first_p_speak_legal_ge_0_70": next((row["frame_id"] for row in active if row["p_speak_legal"] >= 0.70), None),
            "first_p_speak_legal_ge_0_85": next((row["frame_id"] for row in active if row["p_speak_legal"] >= 0.85), None),
        }
        policy.reset()
    return scored


def _run(args: argparse.Namespace, workspace: Path, source_root: Path, dso: Path) -> None:
    import importlib
    import json
    import numpy as np

    importlib = _activate_and_check_runtime_source(source_root)
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("--run requires the explicitly handed-off CUDA device")
    if not args.base_model.is_dir():
        raise FileNotFoundError(f"base model directory is not local: {args.base_model}")
    if args.frames < 15:
        raise ValueError("the agreed probe requires f0..f14 (at least 15 frames)")
    heads = _resolve_head_args(args.head)
    for _, path in heads:
        if not path.is_file():
            raise FileNotFoundError(f"head checkpoint missing: {path}")
    for path in (args.debug_input, args.eval_context, args.stage2_prefix_results):
        if not path.is_file():
            raise FileNotFoundError(f"probe input missing: {path}")

    runtime = _builder_prefixes(args, cpu_only=False)
    saved_stage2_ids = _stage2_prefix_ids(args.stage2_prefix_results)
    if runtime["stage2_ids"] != saved_stage2_ids:
        raise RuntimeError("Stage2 prefix IDs differ from saved extraction after serving-source activation")
    pcm_frames = _read_runner_pcm(args.input_wav)
    if len(pcm_frames) < args.frames:
        raise ValueError(f"input has {len(pcm_frames)} frames, requested {args.frames}")
    debug = np.load(args.debug_input)
    debug_frame_ids = np.asarray(debug["frame_ids"], dtype=np.int64)
    if not np.array_equal(debug_frame_ids[:args.frames], np.arange(args.frames, dtype=np.int64)):
        raise ValueError("actual debug input frame IDs do not begin at f0 in order")
    phase_rows = np.asarray(debug["phase_t"][:args.frames], dtype=np.float32)
    context_rows = _load_eval_context(args.eval_context, args.frames)

    target = runtime["target_ids"]
    conditions = (
        ("A_runtime265_q0", TARGET_B_PROMPT, target, True),
        ("B_runtime265_q2", TARGET_B_PROMPT, target, False),
        ("C_stage2_190_q2", "", saved_stage2_ids, False),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
    prefix_capture_rows: list[dict[str, Any]] = []
    condition_results: dict[str, Any] = {}
    try:
        for name, instructions, ids, q0_first_two in conditions:
            captured, _ = _capture_condition(
                name=name,
                instructions=instructions,
                prefix_ids=ids,
                q0_first_two=q0_first_two,
                args=args,
                runtime=runtime,
                pcm_frames=pcm_frames,
            )
            hidden = captured.pop("hidden")
            capture = captured["registration_capture"]
            prefix_capture_rows.append({
                "condition": name,
                "classification": "IDs captured at actual probe worker registration factory call; not a production WebSocket capture",
                "session_instructions": instructions,
                "history_messages": 0,
                **capture,
            })
            scored = _score_heads(
                hidden=hidden,
                phase_rows=phase_rows,
                context_rows=context_rows,
                pcm_frames=pcm_frames,
                heads=heads,
            )
            captured["scores"] = scored
            captured["comparison_to_live_debug_hidden"] = {
                "actual_debug_path": str(args.debug_input.resolve()),
                "comparison_frames": min(args.frames, int(debug["hidden"].shape[0])),
                "max_abs": float(np.max(np.abs(hidden[:args.frames] - debug["hidden"][:args.frames]))),
                "rms_abs": float(np.sqrt(np.mean(np.square(hidden[:args.frames] - debug["hidden"][:args.frames])))),
                "scope": "hidden comparison only; debug phase_t is reused for offline scoring",
            }
            hidden_path = args.output_dir / f"prefix_parity_{run_id}_{name}.npz"
            np.savez_compressed(
                hidden_path,
                frame_ids=np.arange(args.frames, dtype=np.int64),
                hidden=hidden,
                phase_t=phase_rows,
                prefix_len=np.asarray([len(ids)], dtype=np.int32),
            )
            captured["hidden_phase_npz"] = str(hidden_path.resolve())
            condition_results[name] = captured
            output_path = args.output_dir / f"prefix_parity_{run_id}.json"
            output_path.write_text(json.dumps({
                "source_root": str(source_root),
                "dso": str(dso),
                "base_model_path": str(args.base_model.resolve()),
                "resolved_thinker_path": runtime["resolved_thinker_path"],
                "prompt": TARGET_B_PROMPT,
                "input_pcm_contract": "run_replay.py read_audio_16k + encode_pcm16 roundtrip + server PCM16 decode",
                "frame_limit": args.frames,
                "prefix_registration_captures": prefix_capture_rows,
                "conditions": condition_results,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        del runtime["thinker"]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    print(json.dumps({
        "result_json": str((args.output_dir / f"prefix_parity_{run_id}.json").resolve()),
        "conditions": list(condition_results),
        "prefix_registration_capture_count": len(prefix_capture_rows),
    }, indent=2))


def main() -> None:
    args = _parser().parse_args()
    if args.run and args.preflight:
        raise SystemExit("choose either --preflight or --run")
    workspace, source_root, dso = _configure_import_environment(args)
    if args.run:
        _run(args, workspace, source_root, dso)
    else:
        _cpu_preflight(args, workspace, source_root)


if __name__ == "__main__":
    main()
