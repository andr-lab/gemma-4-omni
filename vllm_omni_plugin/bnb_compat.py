# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/bnb_compat.py — Shim di compatibilità vllm-bnb-plugin 0.0.3 ⇄ vLLM 0.28.

Problema (verificato): vllm_bnb_plugin/bitsandbytes_loader.py:625 chiama
``hf_to_vllm_mapper.get_rename_mapper()`` e poi ``_map_name(name)``, due API
rimosse dal WeightsMapper di vLLM 0.28 (che espone apply/apply_dict/apply_list/
get_unstacked_mapper e usa internamente _map_name_with_shard).

Fix: aggiungiamo al WeightsMapper i due metodi mancanti come alias delle API
0.28. La patch è idempotente e va applicata prima del caricamento dei pesi
(il register() del plugin gira in tutti i processi prima del model load).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_bnb_compat() -> None:
    """Applica lo shim API mancante al WeightsMapper di vLLM 0.28."""
    global _PATCHED
    if _PATCHED:
        return
    try:
        from vllm.model_executor.models.utils import WeightsMapper

        if not hasattr(WeightsMapper, "get_rename_mapper"):

            def _get_rename_mapper(self):
                return self

            WeightsMapper.get_rename_mapper = _get_rename_mapper

        if not hasattr(WeightsMapper, "_map_name"):

            def _map_name(self, name: str) -> str:
                # _map_name_with_shard ritorna (out_name, shard_id): il plugin
                # bnb si aspetta la stringa rinominata.
                result = self._map_name_with_shard(name)
                return result[0] if result is not None else name

            WeightsMapper._map_name = _map_name

        _PATCHED = True
        logger.info("[bnb-compat] Shim WeightsMapper per vllm-bnb-plugin applicato.")
    except Exception as exc:  # pragma: no cover - fail-soft
        logger.warning("[bnb-compat] Shim non applicato (%s): verrà usato il fallback bf16.", exc)


# --------------------------------------------------------------------------
# Strumentazione diagnostica temporanea: traccia la risoluzione dell'arch V9.
# Rimuovere dopo il fix (DEBUG_REGISTRY=1 per attivare).
# --------------------------------------------------------------------------
_PATCHED_RESOLVE = False


def _install_registry_debug() -> None:
    global _PATCHED_RESOLVE
    if _PATCHED_RESOLVE or not os.environ.get("DEBUG_REGISTRY"):
        return
    try:
        from vllm.model_executor.models.registry import _ModelRegistry

        orig = _ModelRegistry.resolve_model_cls

        def _patched(self, architectures, model_config):
            has = [a for a in (architectures if isinstance(architectures, list) else [architectures]) if a in self.models]
            print(
                f"[registry-debug] resolve_model_cls pid={os.getpid()} id(self)={id(self)} "
                f"archs={architectures} found={has} n_models={len(self.models)}",
                flush=True,
            )
            return orig(self, architectures, model_config)

        _ModelRegistry.resolve_model_cls = _patched
        _PATCHED_RESOLVE = True
        print(f"[registry-debug] patch installato pid={os.getpid()}", flush=True)
    except Exception as exc:
        print(f"[registry-debug] patch fallito: {exc}", flush=True)
