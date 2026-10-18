#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duplex_kernel/launch.py — Avvio del custom duplex kernel (ADR-029).

Uso (su gpu3090, dentro il container o venv con GPU):
    python duplex_kernel/launch.py --port 8092
    python duplex_kernel/launch.py --smoke   # solo test caricamento + 1 sintesi

Il kernel è narrow (ispirato a DwarfStar/antirez-ds4) adattato alla nostra
architettura: Thinker Gemma4 E2B → V9 → Talker Qwen3-TTS → Code2Wav, con
TurnDetectorHead nativa e KV persistente per sessione.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path


def _prioritize_explicit_pythonpath(*fallback_paths: str | Path) -> None:
    """Keep caller-selected source paths ahead of repository fallbacks."""
    explicit_paths = list(dict.fromkeys(
        item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item
    ))
    rest = [item for item in sys.path if item not in explicit_paths]
    sys.path[:] = explicit_paths + rest
    for fallback in fallback_paths:
        value = str(fallback)
        if value and value not in sys.path:
            sys.path.append(value)


def _log_runtime_source_files(
    server_module,
    listener_module,
    frontend_module,
    worker_module,
    session_module,
    native_listener_module,
) -> None:
    logging.getLogger(__name__).info(
        "[launch] runtime source files server=%s listener=%s frontend=%s worker=%s session=%s native=%s",
        server_module.__file__,
        listener_module.__file__,
        frontend_module.__file__,
        worker_module.__file__,
        session_module.__file__,
        native_listener_module.__file__,
    )


_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_prioritize_explicit_pythonpath(_REPOSITORY_ROOT)
if not os.path.isdir(os.environ.get("GEMMA4_OMNI_WORKSPACE", "/workspace")) and _REPOSITORY_ROOT.is_dir():
    os.environ["GEMMA4_OMNI_WORKSPACE"] = str(_REPOSITORY_ROOT)
from duplex_kernel import config


_DIRECT_AUDIO_CANDIDATE_ENV = "DUPLEX_DIRECT_AUDIO_CANDIDATE"
_AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV = "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE"
_NATIVE_PARTIAL_ASR_ENV = "DUPLEX_NATIVE_PARTIAL_ASR"
_TALKER_COMPILED_DECODE_ENV = "DUPLEX_TALKER_COMPILED_DECODE"
_TALKER_NATIVE_AUDIO_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1"
_QWEN_NATIVE_LIBRARY_ENV = "QWEN_NATIVE_LIBRARY_PATH"
_DEFAULT_QWEN_NATIVE_AUDIO_Q1_LIBRARY = (
    "/workspace/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so"
)


def configure_bundle_runtime_environment(
    bundle_dir: str | Path | None,
    *,
    workspace: str | Path,
    environ: dict[str, str] | None = None,
) -> str | None:
    """Set protocol-specific runtime defaults before any model is loaded.

    A v2 response bundle is an audio-native generation contract, so the
    direct-audio path is mandatory and cross-response prefix reuse stays off
    until cross-response cache reuse is enabled intentionally. Native recurrent
    audio-Q1 and compiled Qwen Q1 are explicit, v2-only opts; native takes
    precedence when both are selected. Partial ASR remains opt-in per session.
    """
    env = os.environ if environ is None else environ
    from duplex_kernel.audio_response_protocol import (
        AUDIO_RESPONSE_PROTOCOL_ENV,
        resolve_audio_response_protocol,
    )

    # Normalize the old compatibility toggle once; the server then sees the
    # same explicit enum selected by this launcher.
    env[AUDIO_RESPONSE_PROTOCOL_ENV] = resolve_audio_response_protocol(env)
    compiled_decode = env.get(_TALKER_COMPILED_DECODE_ENV)
    native_decode = env.get(_TALKER_NATIVE_AUDIO_Q1_ENV)
    if compiled_decode is not None and compiled_decode.strip() not in {"0", "1"}:
        raise ValueError(f"{_TALKER_COMPILED_DECODE_ENV} must be exactly '0' or '1'")
    if compiled_decode is not None:
        compiled_decode = compiled_decode.strip()
    if native_decode is not None and native_decode.strip() not in {"0", "1"}:
        raise ValueError(f"{_TALKER_NATIVE_AUDIO_Q1_ENV} must be exactly '0' or '1'")
    if native_decode is not None:
        native_decode = native_decode.strip()
    if bundle_dir is None:
        if compiled_decode == "1" or native_decode == "1":
            raise ValueError(
                "Qwen optimized audio-Q1 decode requires response protocol "
                "heard_action_say_v2"
            )
        return None
    bundle_path = Path(bundle_dir).expanduser()
    if not bundle_path.is_absolute():
        bundle_path = Path(workspace).expanduser() / bundle_path
    manifest_path = bundle_path / "bundle_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read response bundle manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"response bundle manifest must be a JSON object: {manifest_path}")

    from duplex_kernel.audio_response_protocol import (
        AUDIO_RESPONSE_PROTOCOL_V2,
        normalize_audio_response_protocol,
    )

    protocol = normalize_audio_response_protocol(manifest.get("response_protocol"))
    if protocol != AUDIO_RESPONSE_PROTOCOL_V2:
        if compiled_decode == "1" or native_decode == "1":
            raise ValueError(
                "Qwen optimized audio-Q1 decode requires response protocol "
                f"{AUDIO_RESPONSE_PROTOCOL_V2}"
            )
        return protocol

    direct_audio = env.get(_DIRECT_AUDIO_CANDIDATE_ENV)
    if direct_audio is not None and direct_audio.strip() != "1":
        raise ValueError(
            f"response protocol {protocol} requires {_DIRECT_AUDIO_CANDIDATE_ENV}=1; "
            "remove the incompatible override or set it to 1"
        )
    prefix_reuse = env.get(_AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV)
    if prefix_reuse is not None and prefix_reuse.strip() != "0":
        raise ValueError(
            f"response protocol {protocol} requires {_AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV}=0 "
            "until native cache parity is established"
        )
    partial_asr = env.get(_NATIVE_PARTIAL_ASR_ENV)
    if partial_asr is not None and partial_asr.strip() not in {"0", "1"}:
        raise ValueError(f"{_NATIVE_PARTIAL_ASR_ENV} must be exactly '0' or '1'")

    env[_DIRECT_AUDIO_CANDIDATE_ENV] = "1"
    env[_AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV] = "0"
    env.setdefault(_NATIVE_PARTIAL_ASR_ENV, "0")
    env.setdefault(_TALKER_COMPILED_DECODE_ENV, "0")
    env.setdefault(_TALKER_NATIVE_AUDIO_Q1_ENV, "0")
    if env[_TALKER_NATIVE_AUDIO_Q1_ENV] == "1":
        library_path = env.get(_QWEN_NATIVE_LIBRARY_ENV)
        if library_path is not None and not library_path.strip():
            raise ValueError(f"{_QWEN_NATIVE_LIBRARY_ENV} must be a non-empty path")
        env.setdefault(_QWEN_NATIVE_LIBRARY_ENV, _DEFAULT_QWEN_NATIVE_AUDIO_Q1_LIBRARY)
    return protocol


def validate_smoke_audio(wav, sample_rate: int = 24_000) -> dict[str, float]:
    """Validate the smoke's generated PCM and return measured audio metrics."""
    import numpy as np

    pcm = np.asarray(wav, dtype=np.float32)
    if pcm.ndim != 1:
        raise RuntimeError(f"smoke audio deve essere mono, shape ricevuta: {pcm.shape}")
    if not pcm.size or not np.isfinite(pcm).all():
        raise RuntimeError("smoke audio vuoto o non finito")
    duration_s = len(pcm) / float(sample_rate)
    if not 0.1 <= duration_s <= 20.0:
        raise RuntimeError(f"smoke audio fuori durata attesa: {duration_s:.3f}s")
    peak = float(np.max(np.abs(pcm)))
    rms = float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))
    if peak <= 1e-5 or rms <= 1e-6:
        raise RuntimeError("smoke audio silenzioso")
    if peak > 1.0001:
        raise RuntimeError(f"smoke audio fuori range PCM: peak={peak:.6f}")
    clipping_fraction = float(np.mean(np.abs(pcm) >= 0.999))
    if clipping_fraction > 0.05:
        raise RuntimeError(f"smoke audio presenta clipping: {clipping_fraction:.2%}")
    return {
        "duration_s": duration_s,
        "peak": peak,
        "rms": rms,
        "crest_factor": peak / rms,
        "clipping_fraction": clipping_fraction,
    }


def _load_query_film_v2_adapter_cpu(checkpoint_path: str | Path):
    """Validate the production checkpoint type before allocating model/GPU state."""
    import torch
    from duplex_kernel.frame_residual_adapter import (
        FrameResidualQueryFiLMAdapter,
        load_frame_residual_adapter,
    )

    adapter, payload = load_frame_residual_adapter(
        checkpoint_path,
        device="cpu",
        dtype=torch.float32,
    )
    if (
        not isinstance(adapter, FrameResidualQueryFiLMAdapter)
        or adapter.architecture_id != "response_hidden_q1_query_film_v2"
    ):
        raise ValueError("production serving requires the response_hidden_q1_query_film_v2 Query FiLM v2 checkpoint")
    return adapter, payload


def _require_query_film_v2_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    workspace: str | Path,
) -> Path:
    """Resolve the production checkpoint and fail closed if it is unavailable."""
    if checkpoint_path is None or not str(checkpoint_path).strip():
        raise FileNotFoundError("approved Query FiLM v2 checkpoint is required for serving")
    resolved = Path(checkpoint_path).expanduser()
    if not resolved.is_absolute():
        resolved = Path(workspace).expanduser() / resolved
    if not resolved.is_file():
        raise FileNotFoundError(f"approved Query FiLM v2 checkpoint missing: {resolved}")
    return resolved.resolve()


def main() -> int:
    default_adapter = (
        config.FRAME_RESIDUAL_ADAPTER_CKPT
        if os.path.isfile(config.FRAME_RESIDUAL_ADAPTER_CKPT)
        else None
    )
    default_bundle = os.environ.get(
        "GEMMA4_OMNI_DUPLEX_RESPONSE_BUNDLE_DIR",
        os.path.join(config.WORKSPACE, "checkpoints", "experimental-v5-repair200-fphase-20261003"),
    )
    if not os.path.isfile(os.path.join(default_bundle, "bundle_manifest.json")):
        default_bundle = None
    adapter_defaults_active = default_adapter is not None
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--workspace", default=config.WORKSPACE)
    ap.add_argument(
        "--runtime-backend",
        choices=("pytorch", "native"),
        default="native" if adapter_defaults_active else "pytorch",
        help="backend esplicito FuturePrefixQwen; pytorch resta il default compatibile",
    )
    ap.add_argument(
        "--virtual-prefix-mode", choices=("legacy", "g0", "gated"), default="legacy",
        help="posizioni logiche plain-Qwen e visibilità del prefix V9; g0 lo maschera dagli audio query",
    )
    ap.add_argument("--smoke", action="store_true", help="solo test: load + 1 sintesi + exit")
    ap.add_argument("--enable-gemma-native-control6", action="store_true",
        help="use Gemma native selected output rows with rank-16 adapter; no external Control6 MLP")
    ap.add_argument("--native-control6-adapter", default=None,
        help="native six-row adapter checkpoint, checked against live lm_head rows before readiness")
    ap.add_argument(
        "--enable-duplex-control6",
        action="store_true",
        help="carica adapter/head six-class; il client deve opt-in con turn_policy=duplex_control6",
    )
    ap.add_argument("--duplex-control6-adapter-dir", default=None)
    ap.add_argument("--duplex-control6-head", default=None)
    ap.add_argument(
        "--duplex-response-bundle-dir",
        default=default_bundle,
        help="carica adapter Gemma, head a sei classi e lessico BACKCHANNEL dallo stesso bundle self-contained",
    )
    ap.add_argument(
        "--ablate-response-lora",
        action="store_true",
        default=adapter_defaults_active,
        help="carica control head e lessico del bundle ma lascia Gemma sul modello base, senza applicare la response LoRA",
    )
    ap.add_argument(
        "--v9-conditioning-source",
        choices=("live_generated", "v9_text_replay", "audio_prefill_frame_raw"),
        default="live_generated",
        help="input hidden per V9 nel controller SAY; v9_text_replay è un candidato diagnostico opt-in",
    )
    ap.add_argument(
        "--response-intent-local-cursor-checkpoint",
        default=None,
        help="checkpoint opt-in del projector Gemma→Qwen con cursore testuale nativo; mantiene h_txt Qwen invariato",
    )
    ap.add_argument(
        "--frame-residual-adapter-checkpoint", default=default_adapter,
        help="checkpoint opt-in per residuo h risposta sulle query audio native Qwen Q1",
    )
    ap.add_argument(
        "--ablate-projector",
        choices=("true", "false"),
        default="false",
        help="controllo esplicito Qwen text-only: nessun projector e zero slot KV di prefix",
    )
    ap.add_argument(
        "--ablate-session-voice-conditioning",
        action="store_true",
        help="ignora profilo/ref voce delle sessioni per riprodurre il Talker senza conditioning del probe fisso",
    )
    ap.add_argument(
        "--enforce-v9-training-voice",
        action="store_true",
        default=adapter_defaults_active,
        help="usa solo l'x-vector x_vector_only identificato dal WAV di training del checkpoint V9",
    )
    ap.add_argument(
        "--diagnostic-talker-seed",
        type=int,
        default=None,
        help="seed diagnostico globale applicato una volta all'avvio; le risposte concorrenti non sono riproducibili individualmente",
    )
    ap.add_argument(
        "--controller-guided-talk-context-pool",
        action="store_true",
        default=adapter_defaults_active,
        help="abilita due contesti Talker isolati per risposte controller-guided; pesi frozen condivisi",
    )
    ap.add_argument(
        "--gemma-qlora-adapter-dir",
        default=None,
        help="merge the Gemma adapter for audio/Qwen smoke only; does not enable a turn policy",
    )
    ap.add_argument(
        "--duplex-control6-manifest",
        default="checkpoints/gemma4_duplex_qlora/training_manifest.json",
    )
    ap.add_argument(
        "--thinker-prefill-chunk-tokens",
        type=int,
        default=16,
        help="limite token per forward del prefill Thinker (0 disabilita il chunking)",
    )
    ap.add_argument(
        "--thinker-audio-prefill-chunk-tokens",
        type=int,
        default=0,
        help=(
            "dimensione del chunk per il solo prefill LM dopo l'assemblaggio direct-audio; "
            "0 mantiene il forward multimodale unico"
        ),
    )
    ap.add_argument(
        "--thinker-compile-mode",
        choices=("off", "default", "reduce-overhead"),
        default="off",
        help="compila i forward Thinker Q4/Q1 e li warmup prima della readiness; errori fail-fast",
    )
    ap.add_argument(
        "--thinker-static-cache-max-len",
        type=int,
        default=4096,
        help="capacità esplicita della StaticCache per ogni generazione Thinker compilata",
    )
    args = ap.parse_args()
    try:
        args.frame_residual_adapter_checkpoint = str(
            _require_query_film_v2_checkpoint(
                args.frame_residual_adapter_checkpoint,
                workspace=args.workspace,
            )
        )
    except FileNotFoundError as exc:
        ap.error(str(exc))
    os.environ.setdefault(
        "GEMMA4_OMNI_CUSTOM_VOICE_DIR",
        os.path.join(args.workspace, "custom_voices"),
    )
    from duplex_kernel.audio_response_protocol import (
        AUDIO_RESPONSE_PROTOCOL_ENV,
        resolve_audio_response_protocol,
    )
    try:
        os.environ[AUDIO_RESPONSE_PROTOCOL_ENV] = resolve_audio_response_protocol()
    except ValueError as exc:
        ap.error(str(exc))
    if args.frame_residual_adapter_checkpoint:
        os.environ.setdefault(_TALKER_NATIVE_AUDIO_Q1_ENV, "1")
    if args.frame_residual_adapter_checkpoint:
        if args.response_intent_local_cursor_checkpoint:
            ap.error("frame residual Q1 adapter cannot load a prefix projector checkpoint")
        if args.runtime_backend != "native":
            ap.error("frame residual adapter requires --runtime-backend native")
        if not args.duplex_response_bundle_dir or not args.ablate_response_lora:
            ap.error("frame residual adapter requires controller bundle with --ablate-response-lora")
        if not args.controller_guided_talk_context_pool:
            ap.error("frame residual adapter requires --controller-guided-talk-context-pool")
        if not args.enforce_v9_training_voice or args.ablate_session_voice_conditioning:
            ap.error("frame residual adapter requires exact training-default voice conditioning")
        if args.v9_conditioning_source != "live_generated":
            ap.error("frame residual adapter requires live response semantics")
        if os.environ.get("DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL", "0").strip() != "0":
            ap.error("frame residual adapter requires dual V9 audio context disabled")
        if os.environ.get("DUPLEX_TALKER_NATIVE_AUDIO_Q1", "0").strip() != "1":
            ap.error("frame residual adapter requires DUPLEX_TALKER_NATIVE_AUDIO_Q1=1")
    if args.enforce_v9_training_voice and args.ablate_session_voice_conditioning:
        ap.error("--enforce-v9-training-voice e --ablate-session-voice-conditioning sono incompatibili")

    if args.response_intent_local_cursor_checkpoint:
        if args.runtime_backend != "native":
            ap.error("local response-intent cursor richiede --runtime-backend native")
        if not args.duplex_response_bundle_dir or not args.ablate_response_lora:
            ap.error("local response-intent cursor richiede bundle controller e --ablate-response-lora")
        if not args.controller_guided_talk_context_pool:
            ap.error("local response-intent cursor richiede --controller-guided-talk-context-pool")
        if not args.enforce_v9_training_voice:
            ap.error("local response-intent cursor richiede --enforce-v9-training-voice")
        if args.v9_conditioning_source != "live_generated":
            ap.error("local response-intent cursor richiede hiddens SAY live; audio-prefill/replay non sono compatibili")
        if os.environ.get("DUPLEX_V9_DUAL_AUDIO_CONTEXT", "0").strip() == "1":
            ap.error("local response-intent cursor non è compatibile con DUPLEX_V9_DUAL_AUDIO_CONTEXT=1")

    if args.ablate_projector == "true":
        if args.runtime_backend != "native":
            ap.error("--ablate-projector=true richiede --runtime-backend native")
        if not args.duplex_response_bundle_dir or not args.ablate_response_lora:
            ap.error("--ablate-projector=true richiede bundle controller e --ablate-response-lora")
        if not args.controller_guided_talk_context_pool:
            ap.error("--ablate-projector=true richiede --controller-guided-talk-context-pool")
        if not args.enforce_v9_training_voice:
            ap.error("--ablate-projector=true richiede --enforce-v9-training-voice")
        if args.response_intent_local_cursor_checkpoint:
            ap.error("--ablate-projector=true non carica checkpoint del projector")
        if args.v9_conditioning_source != "live_generated":
            ap.error("--ablate-projector=true richiede il normale percorso h_txt live")
        if os.environ.get("DUPLEX_V9_DUAL_AUDIO_CONTEXT", "0").strip() == "1":
            ap.error("--ablate-projector=true richiede DUPLEX_V9_DUAL_AUDIO_CONTEXT=0")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    _prioritize_explicit_pythonpath(
        args.workspace,
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.workspace + "/qwen3tts_src",
    )

    local_cursor_checkpoint_path = None
    validated_prefix_log_gate = None
    if args.virtual_prefix_mode == "gated" and not args.response_intent_local_cursor_checkpoint:
        ap.error("gated virtual-prefix mode requires --response-intent-local-cursor-checkpoint")
    if args.response_intent_local_cursor_checkpoint:
        from duplex_kernel.response_intent_local import validate_local_cursor_projector_checkpoint

        local_cursor_checkpoint_path = Path(args.response_intent_local_cursor_checkpoint).expanduser()
        if not local_cursor_checkpoint_path.is_absolute():
            local_cursor_checkpoint_path = Path(args.workspace) / local_cursor_checkpoint_path
        try:
            validated_prefix_log_gate = validate_local_cursor_projector_checkpoint(
                local_cursor_checkpoint_path,
                virtual_prefix_mode=args.virtual_prefix_mode,
            )
        except (FileNotFoundError, ValueError) as exc:
            ap.error(str(exc))
        print(
            "[launch] response-intent checkpoint CPU-validated before engine load "
            f"mode={args.virtual_prefix_mode} gate={validated_prefix_log_gate:.9f} "
            f"checkpoint={local_cursor_checkpoint_path.resolve()}",
            flush=True,
        )

    if args.frame_residual_adapter_checkpoint:
        checkpoint_path = Path(args.frame_residual_adapter_checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(args.workspace).expanduser() / checkpoint_path
        try:
            cpu_adapter, cpu_payload = _load_query_film_v2_adapter_cpu(checkpoint_path)
            if cpu_adapter.training or any(parameter.requires_grad for parameter in cpu_adapter.parameters()):
                ap.error("frame residual checkpoint did not load frozen")
            print("[launch] frame_residual_checkpoint_cpu_validated "
                  f"path={checkpoint_path.resolve()} scale={float(cpu_adapter.scale):.6f}", flush=True)
            del cpu_adapter, cpu_payload
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            ap.error(f"frame residual checkpoint invalid: {exc}")
    try:
        configure_bundle_runtime_environment(
            args.duplex_response_bundle_dir,
            workspace=args.workspace,
        )
    except ValueError as exc:
        ap.error(str(exc))
    from duplex_kernel.audio_response_protocol import resolve_audio_response_protocol

    response_protocol = resolve_audio_response_protocol()
    if response_protocol:
        print(
            "[launch] response protocol="
            f"{response_protocol} direct_audio={os.environ.get(_DIRECT_AUDIO_CANDIDATE_ENV)} "
            f"prefix_reuse={os.environ.get(_AUDIO_TEXT_PREFIX_CACHE_REUSE_ENV)} "
            f"native_partial_asr={os.environ.get(_NATIVE_PARTIAL_ASR_ENV)} "
            f"talker_compiled_decode={os.environ.get(_TALKER_COMPILED_DECODE_ENV, '0')} "
            f"talker_native_audio_q1={os.environ.get(_TALKER_NATIVE_AUDIO_Q1_ENV, '0')} "
            f"response_lora_applied={not args.ablate_response_lora} "
            f"v9_conditioning_source={args.v9_conditioning_source} "
            f"session_voice_conditioning_ablation={args.ablate_session_voice_conditioning} "
            f"enforce_v9_training_voice={args.enforce_v9_training_voice} "
            f"diagnostic_talker_seed={args.diagnostic_talker_seed}",
            f"controller_guided_talk_context_pool={args.controller_guided_talk_context_pool}",
            flush=True,
        )
    else:
        print(
            f"[launch] v9_conditioning_source={args.v9_conditioning_source} "
            f"response_lora_applied={not args.ablate_response_lora} "
            f"session_voice_conditioning_ablation={args.ablate_session_voice_conditioning} "
            f"enforce_v9_training_voice={args.enforce_v9_training_voice} "
            f"diagnostic_talker_seed={args.diagnostic_talker_seed}",
            f"controller_guided_talk_context_pool={args.controller_guided_talk_context_pool}",
            flush=True,
        )

    if args.frame_residual_adapter_checkpoint:
        print(f"[launch] frame_residual_adapter={args.frame_residual_adapter_checkpoint} response_lora_applied=False", flush=True)
    from duplex_kernel.engine import PipelineEngine

    print("=" * 68)
    print("  GEMMA-4-OMNI DUPLEX KERNEL (ADR-029, narrow engine)")
    print("=" * 68)

    engine = PipelineEngine(
        runtime_backend=args.runtime_backend,
        enable_gemma_native_control6_policy=args.enable_gemma_native_control6,
        native_control6_adapter_path=args.native_control6_adapter,
        enable_duplex_control6=args.enable_duplex_control6,
        duplex_control6_adapter_dir=args.duplex_control6_adapter_dir,
        duplex_control6_head_path=args.duplex_control6_head,
        duplex_control6_manifest_path=(
            args.duplex_control6_manifest if args.enable_duplex_control6 else None
        ),
        duplex_response_bundle_dir=args.duplex_response_bundle_dir,
        ablate_response_lora=args.ablate_response_lora,
        v9_conditioning_source=args.v9_conditioning_source,
        ablate_session_voice_conditioning=args.ablate_session_voice_conditioning,
        diagnostic_talker_seed=args.diagnostic_talker_seed,
        controller_guided_talk_contexts=(
            2 if args.controller_guided_talk_context_pool else 0
        ),
        gemma_qlora_adapter_dir=args.gemma_qlora_adapter_dir,
        thinker_prefill_chunk_tokens=args.thinker_prefill_chunk_tokens,
        thinker_audio_prefill_chunk_tokens=args.thinker_audio_prefill_chunk_tokens,
        thinker_compile_mode=args.thinker_compile_mode,
        thinker_static_cache_max_len=args.thinker_static_cache_max_len,
        frame_residual_adapter_checkpoint_path=args.frame_residual_adapter_checkpoint,
    )
    engine.enforce_v9_training_voice = bool(args.enforce_v9_training_voice)
    engine.ablate_response_projector = args.ablate_projector == "true"
    engine.virtual_prefix_mode = args.virtual_prefix_mode
    print(f"[launch] virtual_prefix_mode={engine.virtual_prefix_mode}", flush=True)
    print(
        "[launch] response_projector_mode="
        + ("plain_qwen_text_only_no_prefix_kv" if engine.ablate_response_projector else "live_v9"),
        flush=True,
    )
    t0 = time.time()
    ok = engine.load(workspace=args.workspace)
    print(f"[launch] load engine: {'OK' if ok else 'FAIL'} in {time.time()-t0:.1f}s")
    if not ok:
        return 1

    if local_cursor_checkpoint_path is not None:
        from duplex_kernel.response_intent_local import load_local_cursor_projector

        talker_parameters = getattr(getattr(engine.talker, "talker", None), "parameters", None)
        if not callable(talker_parameters):
            raise RuntimeError("local response-intent candidate cannot resolve frozen Qwen Talker dtype")
        projector_dtype = next(talker_parameters()).dtype
        projector = load_local_cursor_projector(
            local_cursor_checkpoint_path,
            device=engine.device,
            dtype=projector_dtype,
            virtual_prefix_mode=args.virtual_prefix_mode,
        )
        actual_prefix_log_gate = float(projector.prefix_key_log_gate)
        if validated_prefix_log_gate is None or abs(actual_prefix_log_gate - validated_prefix_log_gate) > 1e-9:
            raise RuntimeError("response-intent checkpoint gate changed between CPU validation and model load")
        engine.response_intent_local_projector = projector
        engine.response_intent_prefix_key_log_gate = actual_prefix_log_gate
        engine.response_intent_local_checkpoint_path = str(local_cursor_checkpoint_path.resolve())
        print(
            "[launch] response_intent_local_cursor=enabled "
            f"checkpoint={engine.response_intent_local_checkpoint_path} "
            f"virtual_prefix_mode={args.virtual_prefix_mode} gate={actual_prefix_log_gate:.9f} "
            "qwen_text_prefill=unchanged frozen=true",
            flush=True,
        )

    if args.smoke:
        # Smoke: 1 domanda → testo + audio, metriche fisiche
        import numpy as np

        t1 = time.time()
        prompt = "Quali sono i colori primari? Rispondi in una sola breve frase."
        chat = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        think = engine.think(ids, max_new_tokens=48)
        if not isinstance(think.get("text"), str) or not think["text"].strip():
            raise RuntimeError("smoke Gemma non ha prodotto testo di risposta")
        n_words = len(think["text"].split())
        h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])
        wav = engine.talk(think["text"], h_pfx, n_words=n_words)
        dt = time.time() - t1
        audio_metrics = validate_smoke_audio(wav)
        print(f"[smoke] testo: {think['text']}")
        print(f"[smoke] audio: {audio_metrics['duration_s']:.2f}s ({n_words} parole, {audio_metrics['duration_s']*12.5:.0f} frame)")
        print(f"[smoke] tempo: {dt:.1f}s | RTF={dt/audio_metrics['duration_s']:.3f}")
        print(
            f"[smoke] crest={audio_metrics['crest_factor']:.2f} "
            f"peak={audio_metrics['peak']:.3f} rms={audio_metrics['rms']:.4f} "
            f"clipping={audio_metrics['clipping_fraction']:.2%}"
        )

        os.makedirs(os.path.join(args.workspace, "samples_kernel"), exist_ok=True)
        import soundfile as sf

        out = os.path.join(args.workspace, "samples_kernel", "kernel_smoke.wav")
        sf.write(out, wav, 24000)
        print(f"[smoke] WAV: {out}")
        return 0

    import duplex_kernel.server as server_module
    import duplex_kernel.listener as listener_module
    import duplex_kernel.listener_microbatch_frontend as frontend_module
    import duplex_kernel.listener_microbatch_worker as worker_module
    import duplex_kernel.session as session_module
    import duplex_kernel.native_listener_microbatch as native_listener_module

    _log_runtime_source_files(
        server_module,
        listener_module,
        frontend_module,
        worker_module,
        session_module,
        native_listener_module,
    )
    server_module.run(engine, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
