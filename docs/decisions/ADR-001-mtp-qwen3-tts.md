# ADR-001: Multi-Token Prediction (MTP) with Qwen3-TTS Code2Wav

## Status
Accepted

## Date
2026-08-05

## Context
Full-Duplex Speech-to-Speech models require generating high-fidelity audio streams in real-time. Traditional single-codebook tokenizers sacrifice audio quality, while multi-codebook architectures (e.g. Residual Vector Quantization with 15 codebooks) require predicting multiple tokens per time step. Naive multi-head prediction introduces acoustic drift across codebooks.

## Decision
Implement a **Multi-Token Prediction (MTP) Module** paired with the **Qwen3-TTS Tokenizer (15 RVQ Codebooks at 12.5 Hz)**:
1. `semantic_head` predicts Codebook 0 (Semantic token).
2. `MTPModule` embeds Codebook 0, concatenates it with the LLM adapted hidden states $H_t$, and passes through an MLP to predict Codebooks 1..14 (`acoustic_heads`).
3. Discoupled loss: Standard Cross-Entropy on Codebook 0 (weight 1.0) and Weighted Cross-Entropy on Codebooks 1..14 with silence weight 0.2 (`SILENCE_TOKEN_ID = 0`).

## Alternatives Considered
- **Single-Codebook Tokenization**: Lower audio quality, poor voice cloning fidelity.
- **Independent Parallel Heads**: High acoustic drift between semantic and fine-grained acoustic codebooks.

## Consequences
- Eliminates acoustic drift between codebooks.
- Reduces silence token domination during quiet dialogue pauses.
- Allows zero-shot 3-second voice cloning using `speaker_ref_codes`.
