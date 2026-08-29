# -*- coding: utf-8 -*-
"""
vllm_omni_plugin — Plugin out-of-tree per il serving Full-Duplex Gemma-4-Omni su vLLM-Omni.

Registrazione ufficiale (SDD: vllm_omni/plugins/__init__.py, load_omni_general_plugins):
- entry point group ``vllm_omni.general_plugins`` -> ``register``
- caricato in TUTTI i processi (API server, engine core, worker) via
  ``load_omni_general_plugins()`` chiamato in vllm_omni.engine.arg_utils /
  stage_init_utils / worker.mixins.
"""

from __future__ import annotations

from vllm_omni_plugin.models import (
    GEMMA4_V9_TALKER_ARCH,
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    Gemma4OmniV9TalkerForConditionalGeneration,
    register_v9_models,
)
from vllm_omni_plugin.pipeline import GEMMA4_OMNI_DUPLEX_PIPELINE
from vllm_omni_plugin.processors import (
    thinker2talker_v9_async_chunk,
    thinker2talker_v9_full_payload,
    thinker2talker_v9_token_only,
)

# Registrazione anche sotto "gemma4": il model_type del config.json di
# unsloth/gemma-4-E2B-it è "gemma4" e StageConfigFactory.try_infer_model_type
# usa proprio quel campo per la pipeline auto-detection.
REGISTERED_MODEL_TYPES = ("gemma4_omni_duplex", "gemma4")


def register() -> None:
    """Entry point plugin: registra pipeline + modelli in TUTTI i processi."""
    import os as _os

    from vllm_omni.config import register_pipeline

    from vllm_omni_plugin.bnb_compat import apply_bnb_compat

    # Shim vllm-bnb-plugin 0.0.3 ⇄ WeightsMapper vLLM 0.28 (vedi bnb_compat.py).
    apply_bnb_compat()
    from vllm_omni_plugin.input import install_audio_media_shim

    # Porting L2: shim media connector raw-PCM → WAV (STT upstream Thinker).
    install_audio_media_shim()
    from vllm_omni_plugin.processors import install_async_prewarm_patch

    install_async_prewarm_patch()
    register_v9_models()
    for model_type in REGISTERED_MODEL_TYPES:
        register_pipeline(GEMMA4_OMNI_DUPLEX_PIPELINE, model_type=model_type)
    print(
        "[vllm_omni_plugin] register() OK: "
        f"pid={_os.getpid()} arch={GEMMA4_V9_TALKER_ARCH}",
        flush=True,
    )


__all__ = [
    "GEMMA4_OMNI_DUPLEX_PIPELINE",
    "GEMMA4_V9_TALKER_ARCH",
    "CausalConv1D",
    "DeepCausalTransformerProjectorV9",
    "Gemma4OmniV9TalkerForConditionalGeneration",
    "register",
    "register_v9_models",
    "thinker2talker_v9_async_chunk",
    "thinker2talker_v9_full_payload",
    "thinker2talker_v9_token_only",
]
