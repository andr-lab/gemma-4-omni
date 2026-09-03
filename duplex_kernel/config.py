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
EXPECTED_CAUSAL_V9_SHA256: str = "f434aad703e1ea883a82cc948e5a66f42ff4794b185c19236a89c5e75f874fa7"


def _resolve_production_v9() -> str:
    """Risolve il checkpoint causale certificato a L=0 in modalita' fail-closed.

    In produzione:
    - File missing -> RuntimeError (HARD FAIL)
    - Hash mismatch -> RuntimeError (HARD FAIL)
    - Legacy V9 (v9_prefix_final.pt) -> consentito SOLO con GEMMA4_OMNI_ALLOW_LEGACY_V9=1
    """
    env_v9 = os.environ.get("GEMMA4_OMNI_V9_CKPT")
    allow_legacy = os.environ.get("GEMMA4_OMNI_ALLOW_LEGACY_V9", "0").strip().lower() in ("1", "true", "yes")

    if env_v9:
        candidate = env_v9
    else:
        cand_final = os.path.join(WORKSPACE, "checkpoints", "v9_causal_l0_final.pt")
        cand_unk = os.path.join(WORKSPACE, "checkpoints", "v9_causal_l0_unknown.pt")
        if os.path.exists(cand_final):
            candidate = cand_final
        elif not os.path.isdir(WORKSPACE) or not os.path.isdir(os.path.join(WORKSPACE, "checkpoints")):
            # Client locale / packaging senza cartella checkpoints (pesi su 3090/Kaggle)
            candidate = cand_final
        else:
            raise FileNotFoundError(
                f"Checkpoint causale V9 non trovato in {cand_final} ne' in {cand_unk}."
                " In produzione L=0 il fallback automatico a legacy e' vietato."
                " (Impostare GEMMA4_OMNI_ALLOW_LEGACY_V9=1 solo per diagnostica storica)."
            )

    if os.path.isfile(candidate):
        # Se il file esiste, verifica l'hash crittografico
        skip_hash = os.environ.get("GEMMA4_OMNI_SKIP_V9_HASH_CHECK", "0").lower() in ("1", "true")
        if not allow_legacy and not skip_hash:
            import hashlib
            h = hashlib.sha256()
            with open(candidate, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            actual_sha = h.hexdigest()
            if actual_sha != EXPECTED_CAUSAL_V9_SHA256:
                raise RuntimeError(
                    f"Checkpoint V9 hash mismatch per {candidate}!\nAtteso:   {EXPECTED_CAUSAL_V9_SHA256}\nEffettivo: {actual_sha}\nHard fail di sicurezza: impossibile caricare un modello non certificato."
                )

    return candidate


V9_CKPT = _resolve_production_v9()
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
