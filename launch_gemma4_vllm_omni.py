#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
launch_gemma4_vllm_omni.py — Entrypoint serving Gemma-4-Omni Full-Duplex via CLI ufficiale vLLM-Omni.

La pipeline custom è registrata out-of-tree tramite entry point
``vllm_omni.general_plugins`` (vllm_omni_plugin:register), caricato in tutti i
processi da vLLM-Omni (arg_utils / stage_init_utils / worker.mixins).

Il comando equivalente è:
    vllm serve <GLOBAL_MODEL> --omni --host 0.0.0.0 --port 8091 ...

Nota: l'attributo 'get_explicit_kwargs_dict' (AttributeError del vecchio
launcher) esiste solo sul TrackingNamespace costruito dal CLI TrackingArgumentParser:
per questo si passa DALLA CLI ufficiale e non da parse_args() manuale.
"""

import glob
import os
import sys


def _resolve_global_model() -> str:
    """Thinker Gemma 4 E2B (snapshot locale .hf_cache)."""
    env = os.environ.get("GEMMA4_OMNI_THINKER_MODEL")
    if env and os.path.isdir(env):
        return env
    snaps = sorted(
        glob.glob("/workspace/.hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/*")
    )
    if snaps:
        return snaps[-1]
    snaps = sorted(glob.glob("/workspace/.hf_cache/models--google--gemma-4-E2B-it/snapshots/*"))
    if snaps:
        return snaps[-1]
    raise FileNotFoundError("Thinker Gemma 4 non trovato in /workspace/.hf_cache")


def main() -> None:
    from vllm.entrypoints.cli.main import main as vllm_cli_main

    # Registra la pipeline PRIMA della risoluzione del model_type: nel processo
    # API-server load_omni_general_plugins viene richiamato solo più tardi
    # (engine startup), mentre try_infer_model_type gira già in
    # load_and_resolve_stage_configs. Senza questa registrazione esplicita
    # "gemma4" non risolve e si cade nel fallback diffusion.
    import vllm_omni_plugin

    vllm_omni_plugin.register()

    model = _resolve_global_model()
    deploy_config = os.environ.get(
        "GEMMA4_OMNI_DEPLOY_CONFIG",
        "/workspace/vllm_omni_plugin/deploy/gemma4_omni_duplex.yaml",
    )
    argv = [
        "serve",
        model,
        "--omni",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("GEMMA4_OMNI_PORT", "8091"),
        "--trust-remote-code",
        "--dtype",
        "bfloat16",
        "--deploy-config",
        deploy_config,
        # NESSUN --quantization globale: vLLM inferisce il metodo dal config.json
        # di ciascun modello (docs.vllm.ai/en/stable/features/quantization/bnb):
        # - stage 0 (Gemma unsloth): quantization_config bitsandbytes in config.json
        #   → in-flight nf4 via vllm-bnb-plugin (~4-5GB VRAM).
        # - stage 1/2 (Qwen3-TTS): nessun quantization_config → bf16 nativo
        #   (il talker NON supporta bnb: 'packed_modules_mapping' assente).
    ]
    print(f"[vLLM-Omni] Launch: vllm {' '.join(argv)}", flush=True)
    sys.argv = ["vllm"] + argv
    vllm_cli_main()
    sys.exit(0)


if __name__ == "__main__":
    main()
