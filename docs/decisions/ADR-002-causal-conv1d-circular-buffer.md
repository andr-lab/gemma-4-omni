# ADR-002: 1D Causal Convolution with Identity-Init and Circular Buffer for Streaming

## Status
Accepted (Updated with R3 Identity-Init)

## Date
2026-08-05 (Updated 2026-08-19)

## Context
Standard 1D convolutions with symmetric padding (e.g. `padding=1` on `kernel_size=3`) cause temporal leakage (non-causal future token leakage) during real-time frame-by-frame streaming generation.
Furthermore, during Stage 2 FitNets distillation, a mismatch was identified: production inference applied a 1D causal convolution prior to `ht_adapter`, whereas early FitNets training mapped hidden states frame-wise without convolution. A pure frame-wise adapter cannot compute acoustic temporal derivatives (delta and delta-delta formant transitions) present in the Qwen3-TTS teacher hidden states.

## Decision
1. Implement a 1D Causal Convolution layer (`kernel_size=3, padding=0, bias=False`) with explicit left-padding `F.pad(x, (2, 0))`, providing a 240ms receptive field with zero look-ahead.
2. Initialize convolution weights with **Identity Initialization**: center kernel index $j = k - 1 = 2$ set to Identity ($1.0 / C$ or Identity matrix), and lateral weights set to 0.0. At step 0, $\text{Conv1d}(x) \equiv x$, ensuring zero initial regression and enabling smooth warm-start from previous adapter checkpoints (`strict=False`).
3. Embed `CausalConv1d` directly as the first stage of `ResidualAdapter` so that training and inference execute the exact same computation graph.
4. For streaming inference (`stream_speech`), maintain a 2-frame `conv_state` circular buffer on-device for zero-latency causal convolution per step without re-computing past states.

## Alternatives Considered
- **Symmetric Conv1d (`padding=1`)**: Rejected due to non-causal future frame leakage during training that cannot be replicated during real-time streaming inference.
- **Pure Linear Layer without Conv**: Lacks temporal context (240ms) needed to align with the teacher's causal self-attention trajectory.
- **Random / Kaiming Initialization for Conv**: Causes immediate performance collapse when warm-starting from pre-trained adapter weights.

## Consequences
- Guaranteed zero temporal leakage during streaming inference.
- Exact training/inference consistency between FitNets distillation and runtime pipeline.
- Preserves pre-trained adapter output at step 0 while enabling the convolution weights to learn temporal dynamics.
