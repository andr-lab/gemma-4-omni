# ADR-006: CPU Execution of Qwen3-TTS Teacher during Evaluation for VRAM Isolation

## Status
Accepted

## Date
2026-08-18

## Context
On Kaggle execution environments with a single 16GB GPU (NVIDIA T4), running evaluation requires both the base LLM (Gemma 4 E2B in 4-bit) and the teacher oracle (Qwen3-TTS-12Hz-1.7B-Base, ~3.8GB in bf16/fp32) to compute validation metrics (cosine similarity, coupling Cross-Entropy, and target hidden state extraction).
Attempting to host both models concurrently on GPU VRAM causes immediate CUDA Out-Of-Memory (OOM) errors during inference or feature extraction (exceeding the 16GiB limit).

## Decision
1. Keep the Qwen3-TTS teacher model strictly on host CPU memory (`device="cpu"`) throughout the entire evaluation phase (`EVAL_R2`).
2. Implement device-aware tensor routing in `extract_ht` by querying `_tdev = next(talker.parameters()).device` and moving inputs dynamically to match the teacher device.
3. Reserve GPU VRAM exclusively for Gemma 4 E2B and the trainable adapter (`ht_adapter`).

## Alternatives Considered
- **Concurrent GPU Allocation**: Rejected due to deterministic CUDA OOM crashes (>13.5 GiB baseline allocation).
- **Sequential In-Place GPU Swapping (`.to('cuda')` / `.to('cpu')`)**: High transfer overhead and unsloth memory retention issues causing allocator fragmentation.

## Consequences
- Completely eliminates CUDA OOM risks during evaluation runs on 16GB GPUs.
- Negligible latency overhead for validation feature extraction since evaluation is performed across 100 validation windows.
- Stable, reproducible evaluation metrics without VRAM spikes.
