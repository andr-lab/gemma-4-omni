# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/pipeline.py — GEMMA4_OMNI_DUPLEX_PIPELINE (API reale vLLM-Omni 0.28).

Topologia a 3 stadi disaggregati (ADR-028):
- Stage 0 (thinker, LLM_AR):  `Gemma4ForConditionalGeneration` (vLLM nativo)
  → testo + hidden states dei token generati (pooler payload, default).
- Stage 1 (talker, LLM_AR):   `Gemma4OmniV9TalkerForConditionalGeneration`
  → Prefix KV-Cache V9 [ICL_text | h_pfx | BOS] + RVQ codes (streaming async_chunk).
- Stage 2 (code2wav, LLM_GEN): `Qwen3TTSCode2Wav` (vLLM-Omni nativo)
  → PCM 24kHz (streaming chunk).

Processori stage 0→1: vllm_omni_plugin.processors (testo + hidden V9).
Processori stage 1→2: built-in qwen3_tts (talker2code2wav_{async_chunk,full_payload,token_only}).

Riferimenti SDD:
- vllm_omni/config/stage_config.py (PipelineConfig/StagePipelineConfig reali)
- vllm_omni/model_executor/models/qwen3_omni/pipeline.py (pattern 3 stadi)
- vllm_omni/deploy/qwen3_tts.yaml (deploy streaming async_chunk)
"""

from __future__ import annotations

from vllm_omni.config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni_plugin.processors"
_PROC_TTS = "vllm_omni.model_executor.stage_input_processors.qwen3_tts"
_CKPT = "vllm_omni_plugin.checkpoint"

GEMMA4_OMNI_DUPLEX_PIPELINE = PipelineConfig(
    model_type="gemma4_omni_duplex",
    default_deploy_config_name="gemma4_omni_duplex.yaml",
    # Default di pipeline = architettura dello stage 0 (thinker nativo vLLM).
    model_arch="Gemma4ForConditionalGeneration",
    hf_architectures=("Gemma4ForConditionalGeneration",),
    # Full-duplex /v1/realtime: extension + adapter (livello 1: chat fallback).
    duplex_runtime_extension="vllm_omni_plugin.duplex_extension.Gemma4OmniDuplexRuntimeExtension",
    duplex_serving_adapter="vllm_omni_plugin.duplex_adapter.Gemma4OmniServingRuntimeAdapter",
    duplex_control_enabled=True,
    stages=(
        # ----------------------------------------------------------------
        # Stage 0: Thinker — Gemma 4 E2B (vLLM nativo, testo + hidden latenti)
        # ----------------------------------------------------------------
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_arch="Gemma4ForConditionalGeneration",
            # Handoff hidden states V9 verso il talker (connector).
            custom_process_next_stage_input_func=f"{_PROC}.thinker2talker_v9_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.thinker2talker_v9_async_chunk",
            sampling_constraints={"detokenize": True},
        ),
        # ----------------------------------------------------------------
        # Stage 1: Talker — Qwen3-TTS + Projector V9 (Prefix KV-Cache)
        # ----------------------------------------------------------------
        StagePipelineConfig(
            stage_id=1,
            model_stage="talker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            engine_output_type="latent",
            model_arch="Gemma4OmniV9TalkerForConditionalGeneration",
            model_path_resolver=f"{_CKPT}.resolve_qwen3_tts_root",
            # Input del talker (orchestrator): placeholder + testo risposta.
            # custom_process_input_func è il canale in modalità async_chunk
            # (_select_processor_funcs: sync_process_input_func vale solo con
            # async_chunk=false); impostiamo entrambi per coprire i due modi.
            custom_process_input_func=f"{_PROC}.thinker2talker_v9_token_only",
            sync_process_input_func=f"{_PROC}.thinker2talker_v9_token_only",
            # Handoff RVQ codes verso code2wav (built-in qwen3_tts).
            custom_process_next_stage_input_func=f"{_PROC_TTS}.talker2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC_TTS}.talker2code2wav_async_chunk",
            requires_full_payload_input=True,
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [2150],  # EOS token ID del Talker Qwen3-TTS
            },
        ),
        # ----------------------------------------------------------------
        # Stage 2: Code2Wav — Vocoder Qwen3TTS 24kHz
        # ----------------------------------------------------------------
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="Qwen3TTSCode2Wav",
            model_path_resolver=f"{_CKPT}.resolve_qwen3_tts_root",
            sync_process_input_func=f"{_PROC_TTS}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": False},
            requires_full_payload_input=True,
        ),
    ),
)
