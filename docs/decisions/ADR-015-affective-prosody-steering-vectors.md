# ADR-015: Affective Prosody & Emotional Steering Vectors for Talker Conditioning

## Status
Accepted (Planned for Post-V7 Execution)

## Date
2026-08-26

## Context
With the progression of Projector V7 (`TextAnchoredProjectionV7` with calibrated gain $\gamma=0.25$) and the decoupling of the conversational turn dynamics module (`TurnDynamicsModule`, ADR-014), the Gemma4-Omni pipeline achieves fluent Italian speech synthesis aligned with Gemma 4 multimodal cognitive states.

However, the baseline model naturally converges to a neutral, average prosodic profile ($\mathbb{E}[h_t \mid h_{\text{Gemma}}]$) due to the predominantly neutral/read nature of Common Voice 26 and MDC datasets. To enable fine-grained, dynamic control over affective prosody (e.g. empathy, enthusiasm, calmness, assertiveness, curiosity) without fine-tuning or degrading the frozen Gemma 4 backbone, Projector V7, or neural vocoder (`Code2Wav`), we explore representation engineering via **Activation Steering Vectors**.

Recent literature (2025–2026):
1. **CoCoEmo** ([arXiv:2602.03420](https://arxiv.org/abs/2602.03420)): Proves that emotional prosody is generated primarily by the Speech Language Model (Talker) rather than acoustic flow-matching/vocoder, and identifies intermediate-to-high attention layers (layers 12–16) as optimal injection sites.
2. **Geometric Perspective on Emotion Steering** ([arXiv:2607.00946](https://arxiv.org/abs/2607.00946)): Proves that the Talker latent manifold contains a low-dimensional emotion subspace cleanly disentangled from speaker identity, allowing linear composition $\mathbf{v}_{\text{mix}} = \sum w_k \mathbf{v}_k$.
3. **EmoShift** ([arXiv:2601.22873](https://arxiv.org/abs/2601.22873)) & **EmoSteer-TTS** ([arXiv:2508.03543](https://arxiv.org/abs/2508.03543)): Demonstrate training-free continuous intensity control $\alpha$ and Euclidean norm preservation.

## Decision
We adopt a **Dual-Level Affective Steering Framework**:

1. **Ingress Steering at $h_{\text{cond}}$ (Macro-Prosody & Tone)**:
   Modulate the conditioning vector entering the Talker recurrent loop:
   $$h_{\text{cond}}^{\text{steered}}[t] = \frac{\|h_{\text{cond}}[t]\|_2}{\|h_{\text{cond}}[t] + \alpha \cdot \mathbf{v}_{\text{aff}}\|_2 + \epsilon} \cdot \Big(h_{\text{cond}}[t] + \alpha \cdot \mathbf{v}_{\text{aff}}\Big)$$
2. **Deep Attention Hooking (Micro-Prosodia & Expressive Variability)**:
   Attach forward hooks to Talker attention layers (layers 12–16):
   $$H_t^{(l)} \leftarrow \frac{\|H_t^{(l)}\|_2}{\|H_t^{(l)} + \alpha \cdot \mathbf{v}_{\text{aff}}^{(l)}\|_2 + \epsilon} \cdot \Big(H_t^{(l)} + \alpha \cdot \mathbf{v}_{\text{aff}}^{(l)}\Big)$$
3. **Contrastive Extraction (Mean-Difference)**:
   Extract directional vectors using speaker-matched, text-matched emotional pairs from Italian EMOVO, ESD, and synthetic Gemma 4 prompt pairs:
   $$\mathbf{v}_e^{(l)} = \frac{1}{|\mathcal{D}^{(e)}|} \sum_{i \in \mathcal{D}^{(e)}} \mathbf{h}_i^{(l)} - \frac{1}{|\mathcal{D}^{(0)}|} \sum_{j \in \mathcal{D}^{(0)}} \mathbf{h}_j^{(l)}$$

## Consequences

### Positive
- **Zero Retraining of Core Components**: Backbone Gemma 4, Projector V7, and Code2Wav remain completely frozen.
- **Continuous Intensity Scaling**: Continuous control $\alpha \in [0.0, 3.5]$ allows smooth interpolation from neutral to intense emotion.
- **Composable Mixed Emotions**: Linear superposition $\mathbf{v}_{\text{mix}} = \sum w_k \hat{\mathbf{v}}_k$ allows nuanced blends (e.g., calm curiosity or empathetic reassurance).
- **Intelligibility Preservation**: Strict Euclidean norm preservation prevents activation drift and avoids degradation of Italian pronunciation.

### Negative / Risks & Mitigations
- **Risk**: Setting $\alpha > 3.5$ could lead to slight phonetic distortion.
  - *Mitigation*: Hard clamp $\alpha \le 3.0$ and enforce automated ASR sanity checks (`check_voice_quality2.py`).
- **Risk**: Extracted vectors from cross-lingual datasets (e.g. English ESD) might carry subtle accent bias.
  - *Mitigation*: Prioritize native Italian EMOVO corpus and synthetic Gemma 4 Italian pairs for vector construction.

## Execution Sequence
This ADR is scheduled for post-V7 deployment:
1. Complete Projector V7 training run.
2. Validate and activate `TurnDynamicsModule`.
3. Extract steering vector banks with `tools/extract_affective_steering_vectors.py`.
4. Deploy `AffectiveSteeringModule` and verify with multi-tone audio evaluation.
