# -*- coding: utf-8 -*-
"""
duplex_kernel/models/__init__.py — Loader dei modelli addestrati.

- thinker: Gemma 4 E2B (bf16 nativo, transformers, niente unsloth/LoRA/bnb)
- talker: Qwen3-TTS + Code2Wav (pesi HF locali)
- v9: DeepCausalTransformerProjectorV9 + CausalConv1D (checkpoint addestrati)
- turnhead: TurnDetectorHead + StreamingDuplexController (checkpoint v1)

Tutti autocontenuti: nessuna dipendenza da kaggle_pipeline (training) o dal
plugin vLLM-Omni.
"""

from duplex_kernel.models.talker import TalkerModel, load_talker
from duplex_kernel.models.thinker import ThinkerModel, load_thinker
from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    compute_v9_prefix,
    load_v9_conditioning,
)

__all__ = [
    "CausalConv1D",
    "DeepCausalTransformerProjectorV9",
    "TalkerModel",
    "ThinkerModel",
    "compute_v9_prefix",
    "load_talker",
    "load_thinker",
    "load_v9_conditioning",
]
