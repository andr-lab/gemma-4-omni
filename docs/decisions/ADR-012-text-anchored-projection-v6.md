# ADR-012: TextAnchoredProjection (Strada B / Thinker-Talker Joint Conditioning)

## Status
Accepted (2026-08-25 / 2026-08-26)

Supersedes:
- [ADR-001](ADR-001-mtp-qwen3-tts.md) (MTP isolated generation outside the loop)
- [ADR-005](ADR-005-qwen3-tts-teacher-base-variant.md) (Sub-Talker as standalone decoupled generator)
- [ADR-010](ADR-010-informational-asymptote-qlora-coadaptation.md) (Backbone QLoRA co-adaptation)

Related:
- [ADR-011](ADR-011-unified-env-gemma-qwen3tts.md) (Unified monoprocess environment)
- docs/VERDETTO_TASK12_ARCHITETTURA_V5.md (Decision rationale and experimental benchmarks)

---

## Context

Previous iterations relied on:
1. **Isolated Multi-Token Prediction (MTP)**: Generating acoustic codebooks CB1..CB15 via Qwen3TTSSubTalker conditioned on a static latent representation h_t extracted in a single forward pass.
2. **AR-SemanticDecoder (VALL-E pattern)**: Training an explicit 4-layer causal transformer to predict CB0 autoregressively from h_sub.
3. **Pure Acoustic Hidden Projection (V5)**: Projecting only audio hidden states h_raw into the Qwen3-TTS loop.

### Experimental Findings & Failure Analysis
- **Static Latent Failure**: Experiments (LOOP A vs LOOP B vs oracle, 2026-08-25) proved that the Qwen3-TTS MTP code predictor requires the **dynamic past_hidden state** evolved frame-by-frame inside the Qwen3-TTS autoregressive loop. Static pass embeddings are out-of-distribution, causing phonetic degradation or noise.
- **Informational Bottleneck in V5**: Projecting only audio representations h_raw caused semantic collapse: CB0 loss hit a plateau at ~8.32 (the uniform entropy threshold ln(4096)), unable to constrain phonetic content.
- **Corrupted Cache Distillation**: In cache_ars_full_v2.pt, the teacher states 	r_sub had been extracted with uninitialized custom duplex heads (cos(tr_sub, h_sub) ≈ -0.003). Using lambda_d > 0 injected pure gradient noise.

---

## Decision

We adopt **Strada B (V6): TextAnchoredProjection** integrated into the frozen Qwen3-TTS 1.7B autoregressive loop (Thinker-Talker paradigm inspired by Qwen3-Omni 2509.17765, Qwen2.5-Omni 2503.20215, and Moshi 2410.00037):

`
Gemma 4 E2B Thinker (Frozen 4-bit)
   ├─ Audio Hidden: h_raw (T, 1536)  [Prosody, style, acoustic dynamics]
   └─ Text Hidden:  h_tx  (1, 1536)  [Semantic & phonetic content from inner monologue]
            │
            ▼
   TextAnchoredProjection (25.19M params, ONLY trainable module)
      h_cond[t] = LayerNorm( AudioNorm(AudioMLP(h_raw[t]) + Skip(h_raw[t])) + TextNorm(TextMLP(h_tx)) )
            │
            ▼ h_cond (T, 2048)
   Qwen3-TTS 1.7B Talker Loop (Frozen):
      Frame t: inputs_t = codec_agg_{t-1} + h_cond[t]
         ├─ codec_head(last_hidden) ──> CB0 (Semantic Codebook)
         └─ sub_talker([past_hidden, CB0]) ──> CB1..15 (Acoustic Codebooks)
            │
            ▼
   Code2WavDecoder (Official Qwen3TTSTokenizerV2Model, 24kHz PCM)
`

### Module Architecture: TextAnchoredProjection
1. **Audio Branch**:
   - udio_skip: Linear(1536 -> 2048)
   - udio_mlp: Linear(1536 -> 3072) -> SiLU -> Linear(3072 -> 2048)
   - udio_norm: LayerNorm(2048)
   - Initialized via warm-start from ht_adapter_temporal2_1000.pt (cos ≈ 0.63).
2. **Text Branch (ReZero Anchor)**:
   - 	ext_mlp: Linear(1536 -> 3072) -> SiLU -> Linear(3072 -> 2048)
   - 	ext_norm: LayerNorm(2048)
   - Final linear layer zero-initialized (weight=0, ias=0) for a smooth warm-start.
3. **Output Normalization**:
   - out_norm: LayerNorm(2048).

### Joint Training Loss
Loss = Loss_cb0 + lambda_r * Loss_res + lambda_d * Loss_dis
- Loss_cb0: Cross-Entropy on CB0 from codec_head (vocabulary 4096).
- Loss_res: Cross-Entropy on residual codebooks CB1..CB15 via orward_sub_talker_finetune (vocabulary 2048).
- lambda_r = 0.5: Balances acoustic codebook learning.
- lambda_d = 0.0: Distillation loss deactivated to decouple from legacy uninitialized cache.

---

## Consequences

1. **Semantic Convergence**: With the text anchor h_tx, CB0 loss drops well below the uniform random threshold ln(4096) ≈ 8.32 (reaching < 6.8 by step 100 and < 5.3 by step 320).
2. **Architectural Simplicity**: No external AR semantic decoder or complex scheduled sampling required; the native Qwen3-TTS codec_head and MTP handle autoregression.
3. **Monoprocess Deployment**: Operates seamlessly within the unified monoprocess container (gemma4-omni-v5:latest, ADR-011).
4. **Training Efficiency**: Only 25.19M parameters are trained, with Gemma 4 E2B, Qwen3-TTS 1.7B backbone, Sub-Talker, and Code2Wav fully frozen.
