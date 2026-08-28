# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/models/__init__.py — Registrazione modelli custom del plugin.

Espone:
- DeepCausalTransformerProjectorV9 / CausalConv1D (architettura + loader V9)
- Gemma4OmniV9TalkerForConditionalGeneration (talker con Prefix KV-Cache V9)
- register_v9_models(): registrazione nel ModelRegistry globale di vLLM
"""

from vllm_omni_plugin.models.talker_v9 import (
    V9_EXTRA_FRAMES,
    V9_PREFIX_FRAMES,
    Gemma4OmniV9TalkerForConditionalGeneration,
)
from vllm_omni_plugin.models.v9_projector import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    compute_v9_prefix,
    load_v9_conditioning,
)

# Nome architettura registrata nel ModelRegistry vLLM.
GEMMA4_V9_TALKER_ARCH = "Gemma4OmniV9TalkerForConditionalGeneration"


def register_v9_models() -> None:
    """Registra le classi modello custom nei registry di vLLM.

    CRITICO: il ModelConfig di vLLM-Omni (vllm_omni/config/model.py:161)
    risolve gli arch tramite ``OmniModelRegistry`` — uno snapshot separato
    costruito a import-time (base + modelli omni). Registrare solo nel
    ModelRegistry upstream NON basta: ogni stage engine core risolve contro
    OmniModelRegistry. Registriamo in ENTRAMBI.
    """
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        GEMMA4_V9_TALKER_ARCH,
        "vllm_omni_plugin.models.talker_v9:Gemma4OmniV9TalkerForConditionalGeneration",
    )
    try:
        from vllm_omni.model_executor.models.registry import OmniModelRegistry

        OmniModelRegistry.register_model(
            GEMMA4_V9_TALKER_ARCH,
            "vllm_omni_plugin.models.talker_v9:Gemma4OmniV9TalkerForConditionalGeneration",
        )
    except Exception as exc:  # pragma: no cover - fail-soft
        import logging

        logging.getLogger(__name__).warning(
            "[V9] Registrazione su OmniModelRegistry fallita (%s).", exc
        )


__all__ = [
    "GEMMA4_V9_TALKER_ARCH",
    "V9_EXTRA_FRAMES",
    "V9_PREFIX_FRAMES",
    "CausalConv1D",
    "DeepCausalTransformerProjectorV9",
    "Gemma4OmniV9TalkerForConditionalGeneration",
    "compute_v9_prefix",
    "load_v9_conditioning",
    "register_v9_models",
]
