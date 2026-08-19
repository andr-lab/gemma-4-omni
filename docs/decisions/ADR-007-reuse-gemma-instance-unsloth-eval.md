# ADR-007: Reuse of Gemma 4 E2B Base Instance in Evaluation to Bypass Unsloth VRAM Retention

## Status
Accepted

## Date
2026-08-19

## Context
During sequential execution of STAGE2 training followed by EVAL2 within the same notebook process, attempting to reload the Gemma 4 E2B model for evaluation caused repeated CUDA Out-Of-Memory (OOM) errors (run v4/v5 requested 4.38 GiB when 13.29 GiB was already retained).
Calling `del gemma_base`, `gc.collect()`, and `torch.cuda.empty_cache()` failed to free the GPU memory because the Unsloth framework retains internal references and compiled graph hooks to the model in an internal registry, holding ~9.96 GiB allocated.

## Decision
1. Eliminate the redundant secondary model load in the evaluation section (`EVAL2`).
2. Pass the existing `gemma_base` instance directly from STAGE2 to the evaluation logic, setting `gemma_base.eval()` and running under `torch.no_grad()`.
3. If execution starts directly at EVAL without STAGE2 (evaluation-only mode), load Gemma once.

## Alternatives Considered
- **Forced Python Process Respawn**: Not natively feasible within a single Kaggle notebook cell without splitting into multi-phase kernels.
- **Aggressive Garbage Collection Hack**: Unreliable and prone to CUDA runtime segmentation faults due to residual C++ bindings in PyTorch/Unsloth.

## Consequences
- Completely resolves OOM crashes in multi-stage notebook runs.
- Reduces notebook runtime by eliminating redundant weight loading and initialization delays.
- Clean execution across training, validation, and audio synthesis stages.
