# -*- coding: utf-8 -*-
"""
duplex_kernel/config.py — Configurazione del custom duplex kernel.

Paths e parametri del kernel ispirato a DwarfStar (antirez/ds4): narrow,
specializzato per l'architettura Gemma-4-Omni full-duplex. Tutti i path
sono risolti rispetto alla radice del workspace (/workspace su gpu3090).
"""

from __future__ import annotations

import os
from pathlib import Path

from duplex_kernel.qwen_conditioning import (
    QWEN_AUDIO_CONDITIONING_LEGACY_RECIPE,
    QWEN_AUDIO_CONDITIONING_RECIPE,
)

_DEFAULT_WORKSPACE = (
    "/workspace"
    if os.path.isdir("/workspace")
    else str(Path(__file__).resolve().parent.parent)
)
WORKSPACE = os.environ.get("GEMMA4_OMNI_WORKSPACE", _DEFAULT_WORKSPACE)

# ---------------------------------------------------------------------------
# Checkpoint addestrati (pesi custom, non dai modelli base)
# ---------------------------------------------------------------------------
EXPECTED_CAUSAL_V9_SHA256: str = (
    "ef75c5bcdf8fff470e3527e513dede01"
    "5899a8bb497b4e914ee638d61456fe39"
)
EXPECTED_QUALITY_V9_SHA256: str = (
    "2b8139a66c2dcdeba510f3416ec7cdc58"
    "cfcee8c9ea762072cdd3f27e01cd096"
)

V9_PROFILE_QUALITY_FINAL_ORACLE = "quality_final_oracle"
V9_PROFILE_CAUSAL_L0 = "causal_l0"

V9_QUALITY_RECIPE = "mean_interp16_causalconv_batch_v1"
V9_CAUSAL_L0_RECIPE = "stateful_first_anchor_incremental_v1"

# Qwen audio-input contract is versioned independently from the V9 weights.
# Existing checkpoints retain their historical identity; callers must record
# which Qwen recipe was used rather than silently relabeling old weights.
QWEN_AUDIO_RECIPE = QWEN_AUDIO_CONDITIONING_RECIPE
QWEN_AUDIO_LEGACY_RECIPE = QWEN_AUDIO_CONDITIONING_LEGACY_RECIPE

V9_QUALITY_CKPT = os.environ.get(
    "GEMMA4_OMNI_V9_QUALITY_CKPT",
    os.path.join(WORKSPACE, "checkpoints", "v9_prefix_final.pt"),
)
V9_CAUSAL_L0_CKPT = os.environ.get(
    "GEMMA4_OMNI_V9_CAUSAL_L0_CKPT",
    os.path.join(WORKSPACE, "checkpoints", "v9_causal_l0_final.pt"),
)
FRAME_RESIDUAL_ADAPTER_CKPT = os.environ.get(
    "GEMMA4_OMNI_FRAME_RESIDUAL_ADAPTER_CKPT",
    os.path.join(WORKSPACE, "checkpoints", "frame_residual_adapter_emovo_query_film_v2_candidate.pt"),
)
DEFAULT_VOICE = os.environ.get("GEMMA4_OMNI_DEFAULT_VOICE", "gemma")
FIRST_INPUT_SCHEDULER_ANCHOR_ENV = "GEMMA4_OMNI_FIRST_INPUT_SCHEDULER_ANCHOR"
READY_FRAME_SCHEDULER_DISPATCH_ENV = "GEMMA4_OMNI_READY_FRAME_SCHEDULER_DISPATCH"
NATIVE_LISTENER_CUDA_PROFILE_ENV = "GEMMA4_NATIVE_LISTENER_CUDA_PROFILE"
NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY_ENV = "GEMMA4_NATIVE_LISTENER_CUDA_PROFILE_DIRECTORY"
LISTENER_PAIR_LOCK_HANDOFF_ENV = "GEMMA4_OMNI_LISTENER_PAIR_LOCK_HANDOFF"
NATIVE_CONTROL6_FUSED_READBACK_ENV = "GEMMA4_OMNI_NATIVE_CONTROL6_FUSED_READBACK"


def _strict_env_flag(name: str, *, default: str = "0") -> bool:
    value = os.environ.get(name, default).strip()
    if value not in {"0", "1"}:
        raise RuntimeError(f"{name} must be exactly '0' or '1'")
    return value == "1"


def native_listener_cuda_profile_flag(value: str | None = None) -> bool:
    """Resolve the opt-in native Q4 CUDA-event sampler flag, defaulting off."""
    configured = os.environ.get(NATIVE_LISTENER_CUDA_PROFILE_ENV, "0") if value is None else value
    normalized = str(configured).strip()
    if normalized not in {"0", "1"}:
        raise RuntimeError(f"{NATIVE_LISTENER_CUDA_PROFILE_ENV} must be exactly '0' or '1'")
    return normalized == "1"


def listener_pair_lock_handoff_flag(value: str | None = None) -> bool:
    """Resolve the proof-only two-listener-grant handoff flag, defaulting off."""
    configured = (
        os.environ.get(LISTENER_PAIR_LOCK_HANDOFF_ENV, "0")
        if value is None else value
    )
    normalized = str(configured).strip()
    if normalized not in {"0", "1"}:
        raise RuntimeError(f"{LISTENER_PAIR_LOCK_HANDOFF_ENV} must be exactly '0' or '1'")
    return normalized == "1"


def native_control6_fused_readback_flag(value: str | None = None) -> bool:
    """Resolve the proof-only single-transfer Control6 score readback flag."""
    configured = (
        os.environ.get(NATIVE_CONTROL6_FUSED_READBACK_ENV, "0")
        if value is None else value
    )
    normalized = str(configured).strip()
    if normalized not in {"0", "1"}:
        raise RuntimeError(f"{NATIVE_CONTROL6_FUSED_READBACK_ENV} must be exactly '0' or '1'")
    return normalized == "1"


FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED = _strict_env_flag(
    FIRST_INPUT_SCHEDULER_ANCHOR_ENV
)
READY_FRAME_SCHEDULER_DISPATCH_ENABLED = _strict_env_flag(
    READY_FRAME_SCHEDULER_DISPATCH_ENV
)
NATIVE_LISTENER_CUDA_PROFILE_ENABLED = native_listener_cuda_profile_flag()
LISTENER_PAIR_LOCK_HANDOFF_ENABLED = listener_pair_lock_handoff_flag()
NATIVE_CONTROL6_FUSED_READBACK_ENABLED = native_control6_fused_readback_flag()
if FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED and READY_FRAME_SCHEDULER_DISPATCH_ENABLED:
    raise RuntimeError(
        f"{FIRST_INPUT_SCHEDULER_ANCHOR_ENV} and "
        f"{READY_FRAME_SCHEDULER_DISPATCH_ENV} are mutually exclusive"
    )


def _valid_sha256(value: str) -> bool:
    value = str(value).strip().lower()
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def v9_profile_spec(profile: str) -> dict[str, str]:
    if profile == V9_PROFILE_QUALITY_FINAL_ORACLE:
        expected = EXPECTED_QUALITY_V9_SHA256.strip().lower()
        if not _valid_sha256(expected):
            raise RuntimeError("V9 quality-final SHA non è stato pinned in source")
        return {
            "profile": V9_PROFILE_QUALITY_FINAL_ORACLE,
            "recipe": V9_QUALITY_RECIPE,
            "path": V9_QUALITY_CKPT,
            "sha256": expected,
        }

    if profile == V9_PROFILE_CAUSAL_L0:
        expected = EXPECTED_CAUSAL_V9_SHA256.strip().lower()
        if not _valid_sha256(expected):
            raise RuntimeError("V9 causal-L0 SHA invalido")
        return {
            "profile": V9_PROFILE_CAUSAL_L0,
            "recipe": V9_CAUSAL_L0_RECIPE,
            "path": V9_CAUSAL_L0_CKPT,
            "sha256": expected,
        }

    raise ValueError(f"V9 profile non valido: {profile!r}")


def resolve_v9_checkpoint(profile: str) -> dict[str, str]:
    import hashlib

    spec = dict(v9_profile_spec(profile))
    path = os.path.abspath(os.path.expanduser(spec["path"]))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"V9 checkpoint mancante: {path}")

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != spec["sha256"]:
        raise RuntimeError(
            "V9 checkpoint SHA mismatch: "
            f"profile={profile} expected={spec['sha256']} actual={actual}"
        )
    spec["path"] = path
    spec["actual_sha256"] = actual
    return spec


def _resolve_production_v9() -> str:
    """Resolve V9 checkpoint path without bureaucratic SHA gating."""
    candidate = os.path.abspath(os.path.expanduser(
        os.environ.get("GEMMA4_OMNI_V9_CKPT", V9_CAUSAL_L0_CKPT)
    ))
    return candidate



# Deprecated: P7 code must resolve an explicit profile instead.
V9_CKPT = V9_CAUSAL_L0_CKPT
HEADS_CKPT = os.environ.get("GEMMA4_OMNI_HEADS_CKPT", "")  # custom_duplex_heads.pt (causal_conv)
TURNHEAD_CKPT = os.environ.get(
    "GEMMA4_OMNI_TURNHEAD_CKPT",
    os.path.join(WORKSPACE, "checkpoints", "turn_dynamics_v1_final.pt"),
)
HT_ADAPTER_CKPT = os.environ.get(
    "GEMMA4_OMNI_HT_ADAPTER_CKPT",
    os.path.join(WORKSPACE, "_archive_ckpts", "temporal2", "ht_adapter_temporal2_1000.pt"),
)

# ---------------------------------------------------------------------------
# Modelli base (snapshot HuggingFace locali)
# ---------------------------------------------------------------------------
THINKER_GLOB = os.path.join(
    WORKSPACE,
    ".hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/*",
)
TALKER_GLOB = os.path.join(WORKSPACE, ".hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*")
CODEC_GLOB = os.path.join(WORKSPACE, ".hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*")

# ---------------------------------------------------------------------------
# Parametri architetturali
# ---------------------------------------------------------------------------
# Talker pacing: retain full Italian SAY text while keeping acoustic output bounded.
ARIA_FRAMES_PER_WORD = float(os.environ.get("GEMMA4_ARIA_FRAMES_PER_WORD", 7.0))
ARIA_OVERHEAD_FRAMES = int(os.environ.get("GEMMA4_ARIA_OVERHEAD_FRAMES", 50))
ARIA_MIN_FRAMES = 48
ARIA_MAX_FRAMES = 512
ARIA_BACKCHANNEL_FRAMES_PER_WORD = 4.8
ARIA_BACKCHANNEL_OVERHEAD_FRAMES = 16
ARIA_BACKCHANNEL_MAX_FRAMES = 80

# Prefisso V9: 16 frame di stile prosodico (ADR-022/024).
V9_PREFIX_FRAMES = 16

# Talker: rate frame 12.5 Hz, 15 codebook residui dopo CB0.
TALKER_FPS = 12.5
TALKER_NUM_CODECS = 16  # CB0 semantico + 15 residui
TALKER_EOS_TOKEN_ID = 2150

# P7 audio timeline: codec/network packetization is independent from the
# 80 ms duplex steering clock.  One atomic codec/playout frame is 1920 samples.
CODEC_SAMPLE_RATE = 24000
AUDIO_FRAME_MS = 80
AUDIO_FRAME_SAMPLES = CODEC_SAMPLE_RATE * AUDIO_FRAME_MS // 1000

CODE2WAV_DECODE_FRAMES = int(
    os.environ.get("GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES", "2")
)
if not 2 <= CODE2WAV_DECODE_FRAMES <= 4:
    raise ValueError(
        "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES deve essere tra 2 e 4"
    )

_CODE2WAV_GRAPH_FAMILY_SETTING = os.environ.get(
    "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY", "0"
).strip()
if _CODE2WAV_GRAPH_FAMILY_SETTING not in {"0", "1"}:
    raise ValueError("GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY must be exactly '0' or '1'")
CODE2WAV_GRAPH_FAMILY_ENABLED = _CODE2WAV_GRAPH_FAMILY_SETTING == "1"
if CODE2WAV_GRAPH_FAMILY_ENABLED and CODE2WAV_DECODE_FRAMES != 2:
    raise ValueError(
        "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1 requires "
        "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2"
    )
_CODE2WAV_GRAPH_ODD_TAIL_SETTING = os.environ.get(
    "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL", "0"
).strip()
if _CODE2WAV_GRAPH_ODD_TAIL_SETTING not in {"0", "1"}:
    raise ValueError(
        "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL must be exactly '0' or '1'"
    )
CODE2WAV_GRAPH_ODD_TAIL_ENABLED = _CODE2WAV_GRAPH_ODD_TAIL_SETTING == "1"
if CODE2WAV_GRAPH_ODD_TAIL_ENABLED and not CODE2WAV_GRAPH_FAMILY_ENABLED:
    raise ValueError(
        "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1 requires "
        "GEMMA4_OMNI_CODE2WAV_GRAPH_FAMILY=1"
    )
if CODE2WAV_GRAPH_ODD_TAIL_ENABLED and CODE2WAV_DECODE_FRAMES != 2:
    raise ValueError(
        "GEMMA4_OMNI_CODE2WAV_GRAPH_ODD_TAIL=1 requires "
        "GEMMA4_OMNI_CODE2WAV_DECODE_FRAMES=2"
    )

# Qwen3-TTS Tokenizer V2:
# Qwen3TTSTokenizerV2DecoderConfig.sliding_window == 72.
#
# The upstream chunked_decode(left_context_size=25) value belongs to its
# 300-frame outer chunking strategy.  P7 decodes every 2 frames and must
# preserve the transformer's complete causal receptive attention window in
# order to reproduce the same trajectory as a monolithic <=140-frame decode.
CODE2WAV_ATTENTION_WINDOW_FRAMES = 72

CODE2WAV_LEFT_CONTEXT_FRAMES = int(
    os.environ.get(
        "GEMMA4_OMNI_CODE2WAV_LEFT_CONTEXT_FRAMES",
        str(CODE2WAV_ATTENTION_WINDOW_FRAMES),
    )
)

if (
    CODE2WAV_LEFT_CONTEXT_FRAMES
    != CODE2WAV_ATTENTION_WINDOW_FRAMES
):
    raise ValueError(
        "P7 production richiede "
        "CODE2WAV_LEFT_CONTEXT_FRAMES=72, "
        "uguale alla sliding attention window "
        "del decoder Qwen3-TTS Tokenizer V2"
    )

V9_TRAINING_REFERENCE_SHA256 = (
    "480f55f41c71c3d79c2a9acc48f0bfb3"
    "c5a46222e6e9ebf3e2888e93501a6b5c"
)

SPEAKER_REFERENCE_WAV = os.environ.get(
    "GEMMA4_OMNI_SPEAKER_REFERENCE_WAV",
    os.path.join(WORKSPACE, "artifacts", "voices", "reference.wav"),
).strip()

SPEAKER_REFERENCE_SHA256 = os.environ.get(
    "GEMMA4_OMNI_SPEAKER_REFERENCE_SHA256",
    V9_TRAINING_REFERENCE_SHA256,
).strip()

if bool(SPEAKER_REFERENCE_WAV) != bool(SPEAKER_REFERENCE_SHA256):
    raise ValueError(
        "GEMMA4_OMNI_SPEAKER_REFERENCE_WAV e "
        "GEMMA4_OMNI_SPEAKER_REFERENCE_SHA256 "
        "devono essere definiti insieme"
    )

NETWORK_PACKET_FRAMES = int(
    os.environ.get("GEMMA4_OMNI_NETWORK_PACKET_FRAMES", "2")
)
if not 2 <= NETWORK_PACKET_FRAMES <= 4:
    raise ValueError(
        "GEMMA4_OMNI_NETWORK_PACKET_FRAMES deve essere tra 2 e 4"
    )

AUDIO_BOUNDARY_CROSSFADE_MS = 0
AUDIO_BOUNDARY_CROSSFADE_SAMPLES = 0

# P7 compatibility alias: legacy callers may still import this name.
CODEC_CHUNK_FRAMES = CODE2WAV_DECODE_FRAMES
# P4D: candidate runtime causale V9 a L=0 (zero lookahead acustico, zero delay).
# Il valore storico L=40 e' preservato unicamente come modalita' diagnostica legacy.
FROZEN_QWEN_LOOKAHEAD_FRAMES_LEGACY = 40  # diagnostica storica reference
FROZEN_QWEN_LOOKAHEAD_FRAMES = int(os.environ.get("GEMMA4_OMNI_FROZEN_QWEN_LOOKAHEAD_FRAMES", "0"))
if FROZEN_QWEN_LOOKAHEAD_FRAMES < 0 or FROZEN_QWEN_LOOKAHEAD_FRAMES >= 64:
    raise ValueError("GEMMA4_OMNI_FROZEN_QWEN_LOOKAHEAD_FRAMES deve essere tra 0 e 63")
# Ritardo playout opzionale aggiuntivo; zero conserva il comportamento storico.
# L'abilitazione ha effetto solo sulla catena di playout, mai sul listener.
FROZEN_QWEN_PLAYOUT_DELAY_FRAMES = int(os.environ.get("GEMMA4_OMNI_FROZEN_QWEN_PLAYOUT_DELAY_FRAMES", "0"))
if FROZEN_QWEN_PLAYOUT_DELAY_FRAMES < 0:
    raise ValueError("GEMMA4_OMNI_FROZEN_QWEN_PLAYOUT_DELAY_FRAMES non può essere negativo")
# A missing microphone stream must fail the speaker wait promptly instead of
# occupying the 180-s realtime watchdog.  This is a failure bound, not a
# substitute for the required speaker-only future-conditioning delay.
FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S = float(
    os.environ.get("GEMMA4_OMNI_FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S", "8.0")
)
if FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S <= 0:
    raise ValueError("GEMMA4_OMNI_FROZEN_QWEN_CONDITIONING_WAIT_TIMEOUT_S deve essere positivo")

# ---------------------------------------------------------------------------
# Gestione KV cache (ADR-029 §KV)
# ---------------------------------------------------------------------------
# Thinker: rolling context window — numero massimo di turni mantenuti.
THINKER_MAX_TURNS = 8
# Thinker: massimo token di contesto (sliding window nativa 4:1 in E2B).
THINKER_MAX_CONTEXT_TOKENS = 8192
# Talker: il prefisso [ICL|h_pfx|BOS] è stabile per sessione → KV riusata.
TALKER_PREFIX_KV_CACHE = True
# Talker: tronca la KV ai token del prefisso a fine turno (zero-reset su barge-in).
TALKER_TRUNCATE_TO_PREFIX = True
# Qwen native: capacità persistente dell'arena tra utterance. La slot cache
# Python resta exact-size per turno; l'executor nativo può quindi riusare la
# stessa arena quando il nuovo layout rientra in questa capacità.
QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN = 256

# ---------------------------------------------------------------------------
# GPU / esecuzione
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("GEMMA4_OMNI_DEVICE", "cuda:0")
DTYPE = "bfloat16"

# Clock duplex P2: un tick ogni 80 ms, 1280 sample PCM a 16 kHz.
DUPLEX_CHUNK_PERIOD_MS = 80
DUPLEX_FRAME_SAMPLES = 1280
USER_SAMPLE_RATE = 16000
MEL_BINS = 128
MEL_HOP = 160  # 10ms @ 16kHz
