# Gemma 4 Full-Duplex Omni Pipeline

Real-time, dual-channel, synchronous Full-Duplex Speech-to-Speech conversational AI pipeline combining Google's **Gemma 4 E2B** multimodal backbone with Alibaba's **Qwen3-TTS 12Hz** neural audio codec and neural rendering sub-talker.

---

## Architecture Overview

The system enables native simultaneous listening, speaking, and turn-taking (barge-in) via a three-tier decoupled pipeline:

```
[User Audio Mel (Ch 0)] ──┐
                          ├──> Gemma 4 E2B Conformer ──> LLM Hidden States ──> CausalConv1d ──> ResidualAdapter ──> h_t ──┐
[Agent Mel (Self-Listen)] ┘                                                                                               ├──> [Qwen3 Sub-Talker (Frozen)] ──> [Code2Wav] ──> [24kHz PCM]
                                                                                cb0 Semantic Prediction ──────────────────┘
```

- **Perception (Listening & Self-Listening)**: Ingests 100 fps log-mel spectrograms natively through Gemma 4 E2B's 305M USM-Conformer audio encoder, interleaving user audio and agent self-speech with a 1-frame delay for native barge-in awareness.
- **Cognition & Latent Projection**: Gemma 4 E2B generates text tokens and adapted hidden states via a 1D Causal Convolution ($k=3$, left-pad $(2, 0)$) and `ResidualAdapter` (14.17M params) targeting the continuous latent space $h_t$ of the Qwen3-TTS teacher.
- **Synthesis (Speaking)**: A frozen sub-talker (`Qwen3-TTS-12Hz-1.7B-Base` code predictor, ~363MB) generates the 15 residual RVQ acoustic codebooks conditioned on $h_t$ and $CB_0$, which are decoded into 24kHz waveform audio by the standalone bit-exact `Code2WavDecoder`.

---

## Repository Structure

```
gemma-4-omni/
|-- README.md                         # Main repository documentation and quickstart
|-- AGENTS.md                         # Engineering guidelines, DDD/SDD/TDD protocols, and Kaggle rules
|-- NEXT_STEP.md                      # Live project status, experimental verdicts, and technical roadmap
|-- gemma4_training_report.md         # Comprehensive diagnostic training report and run comparison table
|-- kaggle_pipeline.py                # Standalone 2-Phase Kaggle execution pipeline (tokenization & training)
|-- qwen_subtalker.py                 # Pure PyTorch implementation of the Qwen3-TTS sub-talker code predictor
|-- code2wav_decoder.py               # Standalone bit-exact PyTorch decoder for Qwen3-TTS 24kHz audio synthesis
|-- extract_qwen_subtalker.py         # Selective range-extractor for Qwen3-TTS sub-talker weights (~363MB)
|-- phase2_support.py                 # Dataset structures, batch collation, and evaluation helper routines
|-- serve_duplex.py                   # Local serving interface (CLI single-file demo, REST API, Gradio UI)
|-- docs/                             # Architecture documentation, ADRs, and academic sources
|   |-- ADRS.md                       # Master index of Architecture Decision Records
|   |-- decisions/                    # Formal ADR markdown records (ADR-001 through ADR-010)
|   |-- sources/                      # Reference papers, extracted texts, and official model source snapshots
|-- kernel_fitnets_syntest_r3/        # Active Kaggle kernel configuration and Jupyter notebook (R3)
|-- kernel_src/                       # Modularity cell scripts and source templates for Kaggle notebooks
|-- tools/                            # Operational tools, metric evaluation, and diagnostic tests
|   |-- monitor_kernel.py             # Live Kaggle kernel execution monitor and progress logger
|   |-- parse_klog.py                 # Kaggle execution log parser and metric extractor
|   |-- mcd_v3.py                     # Gold-standard Mel-Cepstral Distortion (MCD) computation utility
|   |-- corruption_metrics.py         # Acoustic corruption and spectral flatness evaluation tool
|   |-- syntest_metrics.py            # Latent space metrics, cosine similarity, and SVD rank analysis
|   `-- tests/                        # Unit tests (causal convolution, sampling routines, wrapper fidelity)
`-- _archive/                         # Historical logs, exploratory scripts, and legacy artifacts
    |-- agy_logs/                     # Historical agent run transcripts and execution outputs
    |-- agy_prompts/                  # Historical prompt templates and research inquiries
    |-- audit_reports/                # Detailed perceptual and acoustic audit logs
    |-- legacy_adrs/                  # Superseded ADR versions
    |-- patch_tests/                  # Experimental patch verification scripts
    |-- run_data/                     # Historical training series metrics (JSON)
    `-- scratch/                      # Temporary working scripts
```

---

## How to Launch and Monitor a Kaggle Run

All heavy GPU workloads run remotely on Kaggle (2x T4 or single T4 16GB) in compliance with [AGENTS.md](file:///c:/Users/andre/Documents/gemma-4-omni/AGENTS.md):

### 1. Push Kernel
Deploy the active notebook to Kaggle:
```bash
kaggle kernels push -p kernel_fitnets_syntest_r3/
```

### 2. Monitor Live Execution
Track execution status, GPU memory usage, and streaming output logs:
```bash
python tools/monitor_kernel.py andrlab/gemma4-fitnets-syntest-r3
```

### 3. Download Run Outputs
Retrieve trained checkpoints (`.pt`), synthesis audio (`.wav`), and JSON evaluation reports:
```bash
kaggle kernels output andrlab/gemma4-fitnets-syntest-r3 -p eval_r3_out/
```

---

## Experimental Status & Benchmarks

Full experimental progression across FitNets distillation iterations:

| Run ID | Adapter Type | Trainable Params | Steps | Cosine Sim (Val) | Coupling CE (Val) | Effective SVD Rank | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Linear (1536->2048) | 3.15M | 1500 | 0.6211 | 5.4226 | 1136 / 2043 | Baseline (Capacity saturation) |
| **R2** | Residual MLP (1536->3072->2048) | 14.17M | 1500 | 0.6275 | 5.4121 | 1145 / 2043 | Minor gain (+0.006 cos); clean audio |
| **R3 Smoke** | CausalConv1d (k=3) + Residual MLP | 21.25M | 4 | 0.6835 -> 0.6953 | — | — | Warm-start verified; conv active |
| **R3 Full v7** | CausalConv1d (k=3) + Residual MLP | 21.25M | 200 + Eval | 0.6063 | 5.4817 | 937 / 2044 | Asymptote confirmed; 72 wavs synthesized |

For detailed telemetry, spectrograms, and loss curves, see [gemma4_training_report.md](file:///c:/Users/andre/Documents/gemma-4-omni/gemma4_training_report.md) and [NEXT_STEP.md](file:///c:/Users/andre/Documents/gemma-4-omni/NEXT_STEP.md).

---

## Architecture Decision Records (ADRs)

Key architectural decisions are indexed in [docs/ADRS.md](file:///c:/Users/andre/Documents/gemma-4-omni/docs/ADRS.md):

- [ADR-001: Multi-Token Prediction (MTP) with Qwen3-TTS Code2Wav](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-001-mtp-qwen3-tts.md)
- [ADR-002: 1D Causal Convolution with Identity-Init and Circular Buffer for Streaming](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-002-causal-conv1d-circular-buffer.md)
- [ADR-003: Discoupled 2-Phase Kaggle Pipeline Architecture (transformers 4.49 vs 5.x)](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-003-discoupled-2phase-kaggle-pipeline.md)
- [ADR-004: Option A: Native Gemma E2B Conformer Input + Qwen3-TTS Code2Wav Output](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-004-option-a-gemma4-e2b-native-input.md)
- [ADR-005: Qwen3-TTS Teacher Selection (Base Variant) for Sub-Talker Coupling](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md)
- [ADR-006: CPU Execution of Qwen3-TTS Teacher during Evaluation for VRAM Isolation](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-006-eval-teacher-cpu-vram-isolation.md)
- [ADR-007: Reuse of Gemma 4 E2B Base Instance in Evaluation to Bypass Unsloth VRAM Retention](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-007-reuse-gemma-instance-unsloth-eval.md)
- [ADR-008: Strict Precedence of Fresh Local Checkpoints over Static Input Dataset Weights](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-008-fresh-adapter-checkpoint-precedence.md)
- [ADR-009: Conditional Runtime Monkey-Patching of Sub-Talker Acoustic Sampling](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-009-runtime-monkey-patch-subtalker-topk.md)
- [ADR-010: Recognition of Informational Asymptote and Transition to QLoRA Co-Adaptation](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-010-informational-asymptote-qlora-coadaptation.md)
