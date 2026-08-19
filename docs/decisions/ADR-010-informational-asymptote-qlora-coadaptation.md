# ADR-010: Recognition of Informational Asymptote and Transition to QLoRA Co-Adaptation

## Status
Accepted

## Date
2026-08-19

## Context
Extensive empirical experiments across three adapter generations evaluated the mapping from frozen Gemma 4 E2B hidden states to frozen Qwen3-TTS teacher hidden states $h_t$:
- **R1 (Linear Adapter, 3.15M params)**: Validation cosine similarity = 0.6211, Coupling CE = 5.4226, SVD effective rank = 1136.
- **R2 (Residual MLP Adapter, 14.17M params)**: Validation cosine similarity = 0.6275 (+0.006 gain for 4.5x parameters), Coupling CE = 5.4121, SVD effective rank = 1145.
- **R3 (CausalConv1d k=3 + Residual MLP, 21.25M params)**: Validation cosine similarity = 0.6063, Coupling CE = 5.4817, SVD effective rank = 937 vs target rank 2044.

Increasing adapter capacity and adding causal convolutions confirmed that the bottleneck is not adapter capacity. Rather, a frozen Gemma 4 E2B model operating on short window contexts reaches an **informational ceiling (asymptote)** at $\text{cos\_sim} \approx 0.60 - 0.63$. The predicted representation is constrained to a subspace of approximately half the target rank (937 vs 2044).

## Decision
1. Conclude standalone adapter-only architectural explorations with frozen backbones.
2. Adopt **QLoRA Co-Adaptation on Gemma 4 E2B** as the primary lever to break the informational asymptote:
   - Configuration: $r=16$, $\alpha=32$, dropout=0.05 on attention and MLP projections (`q, k, v, o_proj`, `gate, up, down_proj`).
   - Differential learning rates: Gemma LoRA at low learning rate ($\text{LR} = 1.5 \times 10^{-5}$), Adapter at standard rate ($\text{LR} = 1.5 \times 10^{-4}$), Sub-talker strictly frozen.
3. For streaming synthesis downstream:
   - Use tiered geometric decay sampling (`tdecay`: $0.8 \cdot 0.75^{g-1}$) across the 15 acoustic codebook levels.
   - Apply $+4\text{ dB}$ post-gain on predicted waveform outputs.

## Alternatives Considered
- **Deeper / Wider Feed-Forward Adapters (R4)**: Rejected due to empirical proof of capacity saturation between R1 and R2.
- **Unfreezing the Qwen3-TTS Sub-Talker**: Strictly rejected due to catastrophic forgetting and collapse on 5M+ pre-trained hours.

## Consequences
- Prevents wasteful GPU allocation on saturated adapter variations.
- Provides a clear mathematical roadmap to reach $\text{cos\_sim} \ge 0.80$ via joint backbone representation learning.
- Establishes gold-standard downstream synthesis configuration for full-duplex conversational audio.
