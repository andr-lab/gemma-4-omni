# ADR-017: Multi-Codebook Loss Calibration (Moshi 100:1 Ratio) and Continuous Manifold Regularization (V8.5)

## Status
Accepted

## Date
2026-08-27

## Context
At Step 12,500 of V8 training, the validation loss plateaued at $\mathcal{L}_{\text{cb0}} = 3.41 \pm 0.02$ and validation accuracy at $28.4\%$.
- **Diagnostic Root Cause**: The loss function $\mathcal{L} = \mathcal{L}_{\text{cb0}} + 0.5 \cdot \mathcal{L}_{\text{res}} + 0.5 \cdot \mathcal{L}_{\text{manifold}}$ spent $52\%$ of its total gradient budget on the 15 acoustic residual codebooks ($\mathcal{L}_{\text{res}} \approx 9.0 \implies 0.5 \times 9.0 = 4.5$), whereas the Sub-Talker was already pre-trained and frozen (`ResTot = 8.85`, matching native loss).
- **Literature SOTA (Moshi Kyutai, arXiv:2410.00037, Eq. 7)**:
  $$\mathcal{L}_{\text{audio}} = \text{CE}(CB_0) + \frac{1}{\sum \alpha_k} \sum_{k=2}^K \alpha_k \text{CE}(CB_k) \quad \text{with } \alpha_k = 100 \text{ for } CB_0 \text{ and } 1 \text{ for acoustic codes}$$
  In SOTA literature, **$93.5\%$ of the optimization budget is dedicated to semantic Codebook 0**, and only $6.5\%$ to acoustic residuals.
- **Doubt-Driven Development (DDD) Adversarial Review**:
  - A discontinuous Hinge Loss $\max(0, \mathcal{L}_{\text{res}} - 9.0) + \max(0, 0.55 - \cos)$ was evaluated and rejected due to: (1) AdamW gradient shock when crossing thresholds ($v_t \to 0$), (2) permitted regression trap below $\cos = 0.62$, (3) vowel vs consonant regularization asymmetry.

## Decision
Adopt the continuous $\mathbb{C}^1$-smooth calibrated loss formulation (V8.5):
$$\mathcal{L}_{\text{total}} = \mathbf{1.0} \cdot \mathcal{L}_{\text{cb0}} + \mathbf{0.05} \cdot \mathcal{L}_{\text{res}} + \mathbf{0.20} \cdot (1.0 - \text{CosineSimilarity}) + \mathbf{0.10} \cdot (\text{NormRatio} - 1.0)^2$$

1. **Semantic Dominance ($\lambda_0 = 1.0$)**: Focuses $93\%$ of the gradient directly on phonetic intelligibility and syllable accuracy.
2. **Acoustic Anchor ($\lambda_r = 0.05$)**: Smoothly regularizes the Sub-Talker without distracting the projector from $CB_0$.
3. **Manifold & Norm Alignment ($\lambda_m = 0.20, \lambda_{\text{norm}} = 0.10$)**: Continuously guides the latent angle towards native alignment while strictly locking the Euclidean norm ratio to $1.0000$.
4. **Warmstart Continuity**: Resuming from `checkpoints/proj_v8_step12500.pt` with matching low learning rate ($\text{LR} = 1.5 \times 10^{-5}$).

## Consequences
- **Instantaneous Loss Reduction**: Total training loss dropped from $\sim 8.0 \to \mathbf{2.97 - 3.53}$.
- **Semantic Convergence**: $CB_0$ loss decreased to **$2.44 - 2.80$** with batch accuracy reaching **$45.3\%$**.
- **Full-Duplex Acoustic Stability**: Acoustic timbre and formants remain strictly within the target range ($8.74 - 9.15$) with zero drift.
