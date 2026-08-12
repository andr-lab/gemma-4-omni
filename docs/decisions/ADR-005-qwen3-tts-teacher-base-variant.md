# ADR-005: Qwen3-TTS Teacher Selection (Base Variant) for Sub-Talker Coupling

## Status
Accepted

## Date
2026-08-05

## Context
The coupling pipeline fine-tunes Gemma 4 E2B to drive the **frozen Qwen3-TTS sub-talker** (code predictor): Gemma emits cb0 + `h_t`, the sub-talker renders the residual acoustic codebooks cb1..cb15, and the standalone Code2Wav decoder produces the waveform. The user's stated priority is *Italian voice quality at the level of Qwen3-TTS* in a full-duplex model.

Three 1.7B variants exist with identical architecture but different `tts_model_type`. The choice of variant is **binding for the extracted sub-talker weights**: verified by hashing `talker.code_predictor.lm_head.0.weight`, the sub-talker weights **differ between variants** (Base ≠ CustomVoice). All facts below were verified on the HF model cards, the `config.json` files, and the qwen-tts 0.1.1 source (`inference/qwen3_tts_model.py`).

Verified variant facts:

| Variant | `tts_model_type` | Voice conditioning | speaker_encoder in checkpoint |
|---|---|---|---|
| **Base** | `base` | **Reference audio required** for cloning (`ref_audio` is Required in `generate_voice_clone`; `extract_speaker_embedding` → `ref_spk_embedding`); no default path without a sample | ✅ 74 tensors (480 total) |
| **CustomVoice** | `custom_voice` | **Preset speaker ID** (`generate_custom_voice(text, speaker=...)`): 9 timbres (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) — **none Italian-native** (card: "recommend using each speaker's native language"); each voice still speaks all 10 languages | ❌ (405 total) |
| **VoiceDesign** | `voice_design` | **Text prompt only** (`generate_voice_design(text, instruct=...)` natural-language voice description) | ❌ (405 total) |

The requirement "Base needs a voice sample for cloning" applies **only at the full-model level** (speaker_encoder + main talker conditioning). The coupling path does **not** extract the speaker path: the sub-talker input is `[h_t, codes]` and the voice comes from **Gemma's `h_t`** (trainable adapter), not from any sample.

**Core quality risk (the central design risk)**: the sub-talker is a conditional renderer trained on `h_t` produced by Qwen's main talker (which encodes content + prosody + speaker identity). Gemma's adapter output starts **out of distribution**; the coupling loss pulls it toward the teacher's `h_t` space. If the adapter cannot converge into distribution, the sub-talker renders degraded audio (artifacts) despite carrying Qwen's own weights.

## Decision
1. **Teacher = `Qwen/Qwen3-TTS-12Hz-1.7B-Base`** for the sub-talker extraction.
2. **Extraction scope**: `talker.code_predictor.*` + `talker.model.codec_embedding` (cb0 row) only. **No** speaker_encoder, **no** main talker. No voice sample is needed at coupling inference.
3. **Quality expectation (explicit, honest)**: the rendering ceiling is literally Qwen's (frozen weights); reaching it depends on `h_t` convergence and is **validated empirically by listening**, not assumed.
4. **Voice identity expectation**: initially neutral, then the average voice of the training corpus (MDC Italian is multi-speaker). A specific Qwen preset voice is **not** guaranteed without explicit speaker conditioning (future feature).
5. **Contingency knobs (deferred, not initial design)**: larger adapter capacity; light LoRA on the sub-talker to adapt input distribution; explicit voice-embedding conditioning in the adapter input.

## Alternatives Considered

### CustomVoice as teacher
- Pros: no reference audio at the full-model level (preset IDs); 9 premium timbres.
- Cons: none of the 9 presets is Italian-native (Italian would carry foreign accent — violates the user's Italian-quality priority); the sub-talker was post-trained for the preset-speaker conditioning system, which the coupling bypasses; model card does not describe it as FT-friendly.
- **Rejected.**

### VoiceDesign as teacher
- Pros: text-only conditioning at the full-model level.
- Cons: the text-to-voice mapping lives in the main talker (not extracted); its sub-talker is post-trained for that conditioning system, which the coupling bypasses; no FT story.
- **Rejected.**

### Full Qwen pipeline at inference (cascade)
- Pros: guarantees Qwen-quality `h_t` (main talker + speaker conditioning).
- Cons: not full-duplex (Qwen3-TTS is text-in → audio-out, no streaming duplex turn-taking); defeats the project's purpose.
- **Rejected.** Kept only as a **PoC ceiling test** (listening benchmark), not as the architecture.

### Light LoRA on the sub-talker from day one
- Pros: directly adapts the sub-talker to Gemma's `h_t` distribution.
- Cons: risks degrading the 5M-hour renderer toward Gemma's distribution; contradicts the "quality preserved by frozen teacher" premise.
- **Deferred as contingency knob #2** — applied only if early listening shows the `h_t` gap is limiting.

## Consequences
- The rendering layer (sub-talker + Code2Wav) carries Qwen's native quality ceiling; the timbre/prosody naturalness can reach Qwen level **if and only if** `h_t` converges into the teacher's distribution.
- The coupling loss (CE through the frozen teacher on GT codes) is the primary mechanism pushing `h_t` into distribution; warm-up via `h_t.detach()` (adapter-first) reduces early-distribution shocks.
- **Validation plan (empirical, mandatory)**:
  1. Cascade PoC: full Qwen pipeline ceiling on 10 Italian holdout sentences (`generate_voice_clone` with an Italian reference sample, else `generate_custom_voice` preset as a sample-free listen).
  2. At the first coupling checkpoint: decode a handful of held-out sentences and listen — clean rendering = `h_t` in distribution; artifacts = apply contingency knobs (adapter capacity first, then sub-talker LoRA).
- No voice sample required at coupling inference; the voice identity is whatever Gemma learns to encode in `h_t`.
- Future work if a specific voice is desired: inject a learned voice embedding into the adapter input (e.g., per-speaker MDC embeddings, or a distilled Qwen preset embedding).
