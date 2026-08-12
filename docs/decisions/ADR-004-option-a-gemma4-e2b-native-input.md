# ADR-004: Option A - Native Gemma E2B Conformer Input + Qwen3-TTS Code2Wav Output

## Status
Accepted

## Date
2026-08-05

## Context
Community empirical evaluation on `google/gemma-4-12B-it` (Hugging Face Discussion #45) demonstrated that Gemma 4 12B's *encoder-free* linear projection audio input suffers from a severe **positional cliff at ~450–500 tokens**, causing multi-turn audio conversation comprehension to collapse to zero. In contrast, Gemma 4 E2B and E4B use a dedicated 305M USM-Conformer audio encoder that is completely immune to this positional cliff.

Furthermore, for inference, the target model must accept user audio via Gemma's native audio encoder while generating agent audio responses via fine-tuned LoRA adapters and custom acoustic heads.

## Decision
Adopt **Option A Hybrid Architecture**:
1. **Model Backbone Selection**: Set `MODEL_NAME = "unsloth/gemma-4-E2B-it"` to use Gemma 4 E2B's 305M USM-Conformer audio encoder for user speech perception.
2. **User Speech Input (Perception)**: User audio is ingested natively by Gemma's Conformer encoder via `Gemma4Processor`, producing continuous LLM embeddings.
3. **Agent Speech Output (Synthesis)**: Gemma 4 outputs predicted RVQ codebooks (15 codebooks @ 12.5 Hz) which are decoded into 24kHz PCM audio waveforms by the standalone **Qwen3-TTS Code2Wav ConvNet Decoder**.

## Alternatives Considered
- **Gemma 4 12B Encoder-Free Input**: Rejected due to the 450-token positional cliff in multi-turn audio conversations.
- **Qwen-only Dual Tokenizer**: Rejected for production inference because it requires running an external Qwen tokenizer service on input user audio instead of using Gemma's native audio input.

## Consequences
- Guaranteed stability in multi-turn audio conversations beyond 800+ tokens.
- Native speech listening paired with state-of-the-art voice synthesis and zero-shot voice cloning.
- Alignment with SOTA academic models (e.g. NVIDIA NemotronLabs VoiceChat 11B, LLaMA-Omni).
