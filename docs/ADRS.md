# Architecture Decision Records (ADRs) Index

This directory contains the Architecture Decision Records (ADRs) for the **Gemma 4 Full-Duplex Omni Pipeline** project.

## Summary of Decisions

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [ADR-001](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-001-mtp-qwen3-tts.md) | Multi-Token Prediction (MTP) with Qwen3-TTS Code2Wav | Superseded by ADR-012 | 2026-08-05 |
| [ADR-002](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-002-causal-conv1d-circular-buffer.md) | 1D Causal Convolution with Identity-Init and Circular Buffer for Streaming | Accepted | 2026-08-05 (Updated 2026-08-19) |
| [ADR-003](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-003-discoupled-2phase-kaggle-pipeline.md) | Discoupled 2-Phase Kaggle Pipeline Architecture (transformers 4.49 vs 5.x) | Accepted (Superseded for Inference by ADR-011) | 2026-08-05 |
| [ADR-004](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-004-option-a-gemma4-e2b-native-input.md) | Option A: Native Gemma E2B Conformer Input + Qwen3-TTS Code2Wav Output | Accepted | 2026-08-05 |
| [ADR-005](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md) | Qwen3-TTS Teacher Selection (Base Variant) for Sub-Talker Coupling | Superseded by ADR-012 | 2026-08-05 |
| [ADR-006](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-006-eval-teacher-cpu-vram-isolation.md) | CPU Execution of Qwen3-TTS Teacher during Evaluation for VRAM Isolation | Accepted | 2026-08-18 |
| [ADR-007](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-007-reuse-gemma-instance-unsloth-eval.md) | Reuse of Gemma 4 E2B Base Instance in Evaluation to Bypass Unsloth VRAM Retention | Accepted | 2026-08-19 |
| [ADR-008](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-008-fresh-adapter-checkpoint-precedence.md) | Strict Precedence of Fresh Local Checkpoints over Static Input Dataset Weights | Accepted | 2026-08-19 |
| [ADR-009](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-009-runtime-monkey-patch-subtalker-topk.md) | Conditional Runtime Monkey-Patching of Sub-Talker Acoustic Sampling | Accepted | 2026-08-19 |
| [ADR-010](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-010-informational-asymptote-qlora-coadaptation.md) | Recognition of Informational Asymptote and Transition to QLoRA Co-Adaptation | Superseded by ADR-012 | 2026-08-19 |
| [ADR-011](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-011-unified-env-gemma-qwen3tts.md) | Unified Environment Gemma + Qwen3-TTS in Single Process (transformers 5.x shim) | Accepted | 2026-08-25 |
| [ADR-012](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-012-text-anchored-projection-v6.md) | TextAnchoredProjection (Strada B / Thinker-Talker Joint Conditioning) | Accepted | 2026-08-26 |
| [ADR-013](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-013-post-train-manifold-oracle-diagnostics.md) | Post-Training Diagnostic Protocol — Manifold Alignment & SLERP Oracle Probes | Accepted | 2026-08-26 |
| [ADR-014](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-014-decoupled-full-duplex-dynamics.md) | Decoupled Full-Duplex Conversational Dynamics Module with Frozen Projector V7 | Accepted | 2026-08-26 |
| [ADR-015](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-015-affective-prosody-steering-vectors.md) | Affective Prosody & Emotional Steering Vectors for Talker Conditioning | Accepted (Post-V7) | 2026-08-26 |
| [ADR-016](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-016-dense-causal-3d-mrope-parallel-engine.md) | Dense Causal 3D MRoPE Parallel Engine for Talker Conditioning (V8) | Accepted | 2026-08-26 |
| [ADR-021](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-021-acoustic-distortion-metrics-and-mitigation.md) | Acoustic Distortion Metrics Suite & Mitigation for Speech-LLMs | Accepted | 2026-08-27 |
| [ADR-022](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-022-prefix-kv-cache-conditioning.md) | Prefix KV-Cache Conditioning vs Additive Layer 0 Injection | Accepted | 2026-08-27 |
| [ADR-023](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-023-aria-streaming-alignment-and-canonical-inference.md) | ARIA Streaming Alignment, Dynamic Pacing & Canonical Inference | Accepted | 2026-08-27 |
| [ADR-024](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-024-deep-causal-transformer-projector-and-parallel-training-engine.md) | Deep Causal Transformer Projector V9 (211.83M) & Parallel 3D MRoPE Training Engine | Accepted | 2026-08-27 |
| [ADR-027](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-027-turn-detector-head-v9-integration.md) | TurnDetectorHead (Frame-Synchronous 12.5Hz) & StreamingDuplexController Integration with Projector V9 | Accepted | 2026-08-28 |
| [ADR-028](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-028-vllm-omni-native-docker-serving.md) | Official vLLM-Omni Native Docker Serving Runtime for Gemma-4-Omni Full-Duplex | Accepted | 2026-08-28 |
| [ADR-029](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-029-custom-duplex-kernel.md) | Custom Optimized Duplex Kernel (DwarfStar-Inspired Narrow Inference Engine) | **Accepted** | 2026-08-29 |
| [ADR-030](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-030-cuda-graph-subtalker-kernel.md) | CUDA Graph Sub-Talker nel Duplex Kernel — Ricostruzione e Gotcha | **Accepted** | 2026-08-29 |

## Historical & Legacy ADRs
Superseded or exploratory ADR iterations are preserved in _archive/legacy_adrs/.


