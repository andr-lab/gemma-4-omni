# ADR-028: Official vLLM-Omni Native Docker Serving Runtime for Gemma-4-Omni Full-Duplex

## Status
Accepted (implementation status updated 2026-08-28 evening)

## Implementation Status Update (2026-08-28 evening)

**VALIDATED END-TO-END** (full-payload mode, `async_chunk: false`):
- 3-stage pipeline serving on RTX 3090: thinker `Gemma4ForConditionalGeneration` (unsloth bnb-4bit, nf4 via `vllm-bnb-plugin` 0.0.3, ~7.3GB) → talker `Gemma4OmniV9TalkerForConditionalGeneration` (Qwen3-TTS 1.7B + Prefix KV-Cache V9, ~3.9GB) → `Qwen3TTSCode2Wav` (~0.5GB).
- `POST /v1/chat/completions` con `modalities: ["text", "audio"]` → choice[0]=texto, choice[1]=WAV 24kHz base64.
- **V9 conditioning attivo**: hidden states del thinker → h_tx/h_raw_style → projector 211.83M + causal conv (pesi dai checkpoint addestrati) → prefisso `[ICL | h_pfx | BOS]` (ricetta `interactive_gemma4_omni_live.py`).
- **Qualità**: ASR Whisper 100% fedeltà, Crest 4.63, HNR +3.06 dB, THD+N 33.1% (rif. live: 4.72 / +3.97 / 28.6%). Artefatti in `samples/vllm_omni_serving/`.

**NOT YET WORKING**: streaming audio incrementale (`async_chunk: true` — il sender di stage 0 non emette chunk al connector), duplex runtime extension (TurnDetectorHead su interfaccia reale), WebSocket `/v1/realtime` (da testare). Dettagli in NEXT_STEP.md.

### Lessons learned (implementation)
1. **OmniModelRegistry**: il ModelConfig di vLLM-Omni risolve le architetture via `OmniModelRegistry` (snapshot separato da `ModelRegistry`): registrare in ENTRAMBI.
2. **Entry point**: `vllm.general_plugins` (prefisso `zzz_`) + `vllm_omni.general_plugins` — caricati in tutti i processi.
3. **Pesi extra (V9/causal_conv)**: caricarli in `load_weights()` con `assign=True` e dichiarare i nomi nel set di ritorno (tracker vLLM); i buffer non-persistenti RoPE restano su CPU → `.to(device)` esplicito.
4. **Placeholder talker**: `ph_len = stima_ufficiale + 18` (V9_EXTRA_FRAMES = 16+2); corto di 1 frame = BOS troncato = silenzio.
5. **task_type VoiceDesign** replica il prefill addestrato (il modello 12Hz Base ha `spk_id: {}` — CustomVoice non disponibile); il code predictor nativo NON supporta `top_p<1.0`.
6. **bitsandbytes**: plugin out-of-tree `vllm-bnb-plugin` (vLLM 0.28 non lo supporta in-tree) + shim `WeightsMapper`; nessun flag `--quantization` globale (inferenza per-stage dal config.json).

## Date
2026-08-28

## Context
Following the completion and verification of the Projector V9 (211.83M parameters), the 12.5Hz Frame-Synchronous TurnDetectorHead ($<0.50\text{ms}$ inference latency), and the multi-codebook RVQ alignment with Qwen3-TTS, the project required a production-grade, low-latency, and standard-compliant serving architecture.

Ad-hoc FastAPI runners or monolithic generation loops suffer from several fundamental limitations:
1. **Inefficient Memory Management**: Lack of PagedAttention and continuous batching causes severe KV-cache fragmentation during simultaneous full-duplex text and audio generation.
2. **Standard Non-Compliance**: Clients (web frontends, OpenAI SDKs, real-time audio players) require standard OpenAI `/v1/chat/completions`, Streaming SSE (`delta` chunks), Structured Tool/Function Calling, and Realtime WebSocket (`/v1/realtime` / `/v1/duplex`) interfaces.
3. **Execution Disaggregation**: Gemma-4-Omni is inherently a multi-stage disaggregated pipeline:
   - **Stage 0 (Thinker)**: `unsloth/gemma-4-E2B-it` autoregressive LLM generating text tokens and hidden states.
   - **Stage 1 (Talker)**: `DeepCausalTransformerProjectorV9` prefix KV-cache + `Qwen3TTSTalker` generating multi-codebook RVQ acoustic tokens.
   - **Stage 2 (Code2Wav Vocoder)**: `Qwen3TTSCode2Wav` causal waveform synthesis at 24kHz.

The open-source `vLLM-Omni` framework (based on vLLM `0.28.0`) natively introduces multi-stage disaggregated execution pipelines, bidirectional audio streaming, and OpenAI-compatible endpoints with custom pipeline registration.

## Decision
We deploy the official **vLLM-Omni Docker Runtime** on the local NVIDIA GeForce RTX 3090 (24GB VRAM):

1. **Docker Container Architecture**:
   - Built custom production image `vllm-omni-cuda:latest` based on `vllm/vllm-openai:latest`.
   - Explicitly aligned and matched `vllm==0.28.0` with `vllm-omni==0.28.0rc2` using `uv` to ensure binary compatibility of entrypoint APIs (e.g. `create_error_response`, `omni_run_server`).
   - Execution constraints: Non-root container permissions, `--ipc=host` for zero-copy DataLoader shared memory, and direct port mapping `-p 8091:8091` (preserving host port `8000` for management services).

2. **vLLM-Omni Plugin Architecture (`vllm_omni_plugin/`)**:
   - **Topological Pipeline (`pipeline.py`)**: Defines `GEMMA4_OMNI_DUPLEX_PIPELINE` registering:
     - Stage 0: `StageExecutionType.LLM_AR` (Gemma 4 E2B Thinker, owns tokenizer, multimodal input enabled).
     - Stage 1: `StageExecutionType.LLM_AR` (Projector V9 + Talker, inter-stage prefix function `thinker2talker_v9_prefix`, chunked streaming output `talker2code2wav_async_chunk`).
     - Stage 2: `StageExecutionType.LLM_GENERATION` (Vocoder Code2Wav, full waveform decoding).
   - **Pipeline Registration**: Registered cleanly into vLLM-Omni's registry via `vllm_omni.config.register_pipeline(GEMMA4_OMNI_DUPLEX_PIPELINE, model_type="gemma4_omni_duplex")`.
   - **Quantization & VRAM Footprint**: Native `bfloat16` precision across all stages. Total memory footprint on RTX 3090 is ~13.4GB VRAM, well within the 24GB limit, entirely avoiding bitsandbytes quantization mismatches.

3. **Duplex Conversational Dynamics**:
   - Integrated `TurnDetectorHead` with 5-state Markov conversational machine (`USER_SPEAKING`, `AI_SPEAKING`, `MUTUAL_SILENCE`, `BARGE_IN`, `INTERRUPTED`).
   - Zero-Reset KV-cache manager ensuring seamless context continuity across real-time WebSocket audio turns.

## Alternatives Considered

### Alternative A: Standalone Ad-Hoc FastAPI Server
- *Pros*: Simple single-file implementation.
- *Cons*: No PagedAttention; manual thread/process coordination between Gemma and Qwen3-TTS; unable to benefit from vLLM's continuous batching and CUDA kernel optimizations.
- *Rejected*: Inadequate for low-latency full-duplex production deployment.

### Alternative B: Host-level `uv` Python Environment without Docker
- *Pros*: Direct host execution.
- *Cons*: Risk of host library conflicts with system CUDA/CUDNN; containerization provides isolated, reproducible environments across different GPU nodes.
- *Decision*: Containerized via `vllm-omni-cuda:latest` with volume mounts to host model cache (`~/.cache/huggingface`).

## Consequences

### Positive
- Production-grade serving engine with full OpenAI SDK and Realtime WebSocket compatibility.
- Seamless execution of simultaneous streaming text transcripts and 24kHz audio synthesis.
- Ultra-low latency turn-detection (<0.50ms) allowing instantaneous barge-in.
- Modularity: Model weights, checkpoints, and projectors can be swapped declaratively via YAML deploy configs.

### Negative / Operational Constraints
- Host machine must allocate >= 14GB free VRAM for concurrent multi-stage serving.
- Startup warm-up involves PagedAttention KV-cache pre-allocation (~30s).
