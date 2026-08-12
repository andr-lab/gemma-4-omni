# ADR-003: Discoupled 2-Phase Kaggle Pipeline Architecture

## Status
Accepted

## Date
2026-08-05

## Context
Legacy audio tokenization packages (such as `qwen-tts`) require `transformers==4.49.0`. However, training Gemma 4 with Unsloth requires `transformers>=5.0.0`. Installing both in the same Python environment causes dependency conflicts and runtime import failures.

## Decision
Structure the Kaggle execution notebook into two completely discoupled execution phases:
- **Phase 1 (Data Preprocessing & Tokenization)**: Install `qwen-tts` & `transformers==4.49.0`. Load `DuplexConv` + `Mozilla Data Collective Italian` datasets, tokenize audio to 15 RVQ 12.5Hz codebooks, and export to compressed Parquet (`.parquet`).
- **Phase 2 (Training & Synthesis)**: Restart Kaggle Kernel. Upgrade environment to `transformers>=5.x` and `unsloth`. Load Parquet cache in <0.05s, instantiate a standalone PyTorch `Code2WavDecoder`, and execute QLoRA training on Gemma 4.

## Alternatives Considered
- **Single-Phase Execution**: Breaks due to incompatible `transformers` version requirements between `qwen-tts` and Unsloth/Gemma 4.
- **On-the-fly Tokenization during Training**: Slows down training iterations significantly due to real-time audio encoding overhead.

## Consequences
- Zero dependency conflicts.
- Fast training iteration loops (<0.05s data loading from Parquet).
- Pure cell-based execution in Kaggle compliant with `AGENTS.md`.
