# Architecture Decision Records (ADRs) Index

This directory contains the Architecture Decision Records (ADRs) for the **Gemma 4 Full-Duplex Omni Pipeline** project.

## Summary of Decisions

| ADR | Title | Status | Date |
| --- | --- | --- | --- |
| [ADR-001](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-001-mtp-qwen3-tts.md) | Multi-Token Prediction (MTP) with Qwen3-TTS Code2Wav | Accepted | 2026-08-05 |
| [ADR-002](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-002-causal-conv1d-circular-buffer.md) | 1D Causal Convolution with Identity-Init and Circular Buffer for Streaming | Accepted | 2026-08-05 (Updated 2026-08-19) |
| [ADR-003](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-003-discoupled-2phase-kaggle-pipeline.md) | Discoupled 2-Phase Kaggle Pipeline Architecture (transformers 4.49 vs 5.x) | Accepted | 2026-08-05 |
| [ADR-004](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-004-option-a-gemma4-e2b-native-input.md) | Option A: Native Gemma E2B Conformer Input + Qwen3-TTS Code2Wav Output | Accepted | 2026-08-05 |
| [ADR-005](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md) | Qwen3-TTS Teacher Selection (Base Variant) for Sub-Talker Coupling | Accepted | 2026-08-05 |
| [ADR-006](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-006-eval-teacher-cpu-vram-isolation.md) | CPU Execution of Qwen3-TTS Teacher during Evaluation for VRAM Isolation | Accepted | 2026-08-18 |
| [ADR-007](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-007-reuse-gemma-instance-unsloth-eval.md) | Reuse of Gemma 4 E2B Base Instance in Evaluation to Bypass Unsloth VRAM Retention | Accepted | 2026-08-19 |
| [ADR-008](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-008-fresh-adapter-checkpoint-precedence.md) | Strict Precedence of Fresh Local Checkpoints over Static Input Dataset Weights | Accepted | 2026-08-19 |
| [ADR-009](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-009-runtime-monkey-patch-subtalker-topk.md) | Conditional Runtime Monkey-Patching of Sub-Talker Acoustic Sampling | Accepted | 2026-08-19 |
| [ADR-010](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-010-informational-asymptote-qlora-coadaptation.md) | Recognition of Informational Asymptote and Transition to QLoRA Co-Adaptation | Accepted | 2026-08-19 |

## Historical & Legacy ADRs
Superseded or exploratory ADR iterations are preserved in `_archive/legacy_adrs/`.
