# -*- coding: utf-8 -*-
"""
duplex_kernel/config.py — Configurazione del custom duplex kernel.

Paths e parametri del kernel ispirato a DwarfStar (antirez/ds4): narrow,
specializzato per l'architettura Gemma-4-Omni full-duplex. Tutti i path
sono risolti rispetto alla radice del workspace (/workspace su gpu3090).
"""

from __future__ import annotations

import os

WORKSPACE = os.environ.get("GEMMA4_OMNI_WORKSPACE", "/workspace")

# ---------------------------------------------------------------------------
# Checkpoint addestrati (pesi custom, non dai modelli base)
# ---------------------------------------------------------------------------
V9_CKPT = os.environ.get("GEMMA4_OMNI_V9_CKPT", os.path.join(WORKSPACE, "checkpoints", "v9_prefix_final.pt"))
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
# ARIA pacing (ADR-023): 4.8 frame/parola + 12 di overhead.
ARIA_FRAMES_PER_WORD = 4.8
ARIA_OVERHEAD_FRAMES = 12
ARIA_MIN_FRAMES = 48
ARIA_MAX_FRAMES = 140

# Prefisso V9: 16 frame di stile prosodico (ADR-022/024).
V9_PREFIX_FRAMES = 16

# Talker: rate frame 12.5 Hz, 15 codebook residui dopo CB0.
TALKER_FPS = 12.5
TALKER_NUM_CODECS = 16  # CB0 semantico + 15 residui
TALKER_EOS_TOKEN_ID = 2150

# Code2Wav: 24 kHz, chunk di 8 frame = 640ms (ADR-023).
CODEC_SAMPLE_RATE = 24000
CODEC_CHUNK_FRAMES = 8

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

# ---------------------------------------------------------------------------
# GPU / esecuzione
# ---------------------------------------------------------------------------
DEVICE = os.environ.get("GEMMA4_OMNI_DEVICE", "cuda:0")
DTYPE = "bfloat16"

# Sessione duplex: chunk audio utente (ms) → 100ms per la turnhead 12.5Hz.
DUPLEX_CHUNK_PERIOD_MS = 100
USER_SAMPLE_RATE = 16000
MEL_BINS = 128
MEL_HOP = 160  # 10ms @ 16kHz
