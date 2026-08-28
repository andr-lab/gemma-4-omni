# NEXT_STEP — Gemma 4 Full-Duplex Omni Pipeline (28 Agosto 2026, sera)

## 1. ✅ LINEA PROJECTOR CONSOLIDATA & VALIDATA
- **Projector V9** (211.83M): `v9_prefix_step10000.pt` (Val CB0 2.294, Acc 40.8%).
- **Inference Real-Time**: CUDA Graph Sub-Talker, RTF 0.7541.
- Causal Conv 1D addestrata (7.08M, Task 11i SOTA, `custom_duplex_heads.pt`).

## 2. ✅ TURNDETECTORHEAD (ADR-027)
- `turn_detector_head.py`: 5 classi, <0.45ms/frame, StreamingDuplexController.

## 3. ✅ SERVING vLLM-Omni CON PROJECTOR V9 (ADR-028) — VALIDATO full-payload

### Pipeline (RTX 3090, porta 8091)
- Stage 0: Gemma4ForConditionalGeneration (unsloth bnb-4bit, nf4 via vllm-bnb-plugin, ~7.3GB)
- Stage 1: Gemma4OmniV9TalkerForConditionalGeneration (Qwen3-TTS 1.7B + Prefix KV-Cache V9, ~3.9GB)
- Stage 2: Qwen3TTSCode2Wav 24kHz (~0.5GB)

### 4 CAPABILITY VOCE (tutte VALIDATE in full-payload, async_chunk=false)
1. ✅ **System prompt LLM**: standard nei messaggi chat (testato: risposta conforme).
2. ✅ **Voice style (VoiceDesign + instruct)**: `voice_style` nel request → il talker usa il task VoiceDesign con le istruzioni. Audio pulito (HNR 3.06 dB, THD 33%).
3. ✅ **Ref audio cloning (Base)**: `ref_audio` (base64 WAV) + `ref_text` → task Base con cloning a 3s dal report Qwen3-TTS (arXiv:2601.15621). ASR 100% intelligibile.
4. ✅ **Voci di default**: `voice="gemma"|"leonardo"` → profili x-vector registrati (custom_voice_dir). Generati con `tools/build_voice_profiles.py` (mel slaney esatta + speaker encoder) dai sample validati. ASR 100% intelligibile; qualità acustica leggermente inferiore al default (voce diversa).
- Default (nessun parametro voce): VoiceDesign puro = voce V9 addestrata (la migliore qualità).

### Debug veloce (SDD: docs.vllm.cc configuration/optimization)
- **`GEMMA4_OMNI_MODE=debug`** (default): deploy `gemma4_omni_duplex_debug.yaml` con `enforce_eager` su tutti gli stage + `kv_cache_memory` fisso → **startup 4.5 min** (vs 8 min).
- **Volume cache vLLM persistente**: `/home/administrator/.cache/vllm:/root/.cache/vllm` → il 2° boot in modalità prod è molto più veloce (compile cache).
- `GEMMA4_OMNI_MODE=prod`: deploy completo (torch.compile + CUDA graphs).

### NON funzionante (da investigare con il ciclo rapido)
- ⚠️ **Streaming audio incrementale (async_chunk=true)**: il sender di stage 0 ora emette chunk (`thinker2talker_v9_async_chunk` chiamato, hidden "hidden" key fix, `new_token_ids` accumulati, testo recuperato worker-side), MA il talker produce rumore/512 frame in modalità async (probabile: codec_streaming/placeholder in async + EOS non campionato). Il deploy resta su async_chunk=false (stato validato).
- ⚠️ **Duplex runtime extension** (TurnDetectorHead su interfaccia reale) + WebSocket `/v1/realtime`: da implementare/testare.
- Nota: il trim energetico e la peak-normalizzazione (post-processing del ground truth) NON sono applicati dal serving — i client devono tagliare la coda di silenzio.

### Note tecniche chiave (lezione appresa)
1. Registrare in `ModelRegistry` E `OmniModelRegistry`; entry point in `vllm.general_plugins` (zzz_) e `vllm_omni.general_plugins`.
2. Pesi extra in `load_weights()` (assign=True) + dichiarati al tracker; buffer non-persistenti RoPE → .to(device).
3. Placeholder talker: `ph_len = estimate + V9_EXTRA_FRAMES(18)`; task VoiceDesign default; subtalker top_p=1.0.
4. **async_chunk=true cambia TUTTO il flusso** (payload struct via connector, prewarm, replace_streaming_prompt): il percorso async va validato SEPARATAMENTE — non mischiare con le fix full-payload.
5. Profili voce: formato ufficiale (manifest + safetensors speaker_embedding 2048D); estrazione x-vector con la mel slaney ESATTA del builder (torchaudio melscale_fbanks) — una mel diversa degrada la voce.
6. Debug rapido: enforce_eager + kv_cache_memory + cache compile persistente.

---

## 4. PROSSIMI PASSI (in ordine)
1. **Debug async_chunk** (streaming audio) con il ciclo rapido: il talker in async produce rumore (codec_streaming / placeholder / EOS). Dopo il fix: `/v1/realtime` per il duplex vero.
2. **Duplex runtime extension** (TurnDetectorHead) contro `vllm_omni.experimental.fullduplex.engine.contracts` + WebSocket E2E.
3. **Benchmark finale**: 50 domande conversazionali (WER/Crest/HNR/THD/TTFA), confronto con/senza profili voce.
