# NEXT_STEP — Gemma 4 Full-Duplex Omni Pipeline (28 Agosto 2026)

## 1. ✅ LINEA PROJECTOR CONSOLIDATA & VALIDATA (Invariato)
- **Projector V9 Deep Causal Transformer (211.83M parametri)**: `v9_prefix_step10000.pt` (Val CB0 Loss `2.294`, Top-1 Acc `40.8%`).
- **Inference Engine Real-Time**: CUDA Graph Sub-Talker, RTF 0.7541, qualità 100% bit-pari.
- Causal Conv 1D addestrata (7.08M, Task 11i SOTA, `custom_duplex_heads.pt`).

## 2. ✅ TURNDETECTORHEAD IMPLEMENTATA E VALIDATA E2E (ADR-027, Invariato)
- `turn_detector_head.py`: 5 classi, <0.45ms/frame, StreamingDuplexController con Zero-Reset KV-Cache.

## 3. ✅ SERVING vLLM-OMNI: PIPELINE 3 STADI FUNZIONANTE CON PROJECTOR V9 (ADR-028)

**STATO: FUNZIONANTE E VALIDATO in full-payload mode (async_chunk=false).**

### Cosa funziona (validato su RTX 3090)
- **Pipeline disaggregata 3 stadi** registrata out-of-tree:
  - Stage 0 (thinker): `Gemma4ForConditionalGeneration` (unsloth bnb-4bit, inferenza nf4 via `vllm-bnb-plugin` 0.0.3, ~7.3GB VRAM).
  - Stage 1 (talker): `Gemma4OmniV9TalkerForConditionalGeneration` = Qwen3-TTS 1.7B + **Prefix KV-Cache V9** (projector 211.83M + causal conv, pesi dai checkpoint addestrati, ~3.9GB).
  - Stage 2 (code2wav): `Qwen3TTSCode2Wav` 24kHz (~0.5GB).
- **Endpoint OpenAI**: `POST /v1/chat/completions` con `modalities: ["text", "audio"]` → choice[0]=testo, choice[1]=audio WAV base64. `GET /v1/models`, `/health`.
- **Condizionamento V9 ATTIVO**: la risposta del thinker → hidden states last-layer → h_tx (mean) + h_raw_style (interp 16 + causal conv) → proj → prefisso `[ICL | h_pfx | BOS]` nel talker (ricetta esatta di `interactive_gemma4_omni_live.py`).
- **Qualità validata** (frase lunga, 15.6s, trimmed): ASR Whisper **100% fedeltà lessicale**, Crest Factor **4.63** (rif. live 4.72), HNR **+3.06 dB** (target >3), THD+N **33.1%** (rif. 28.6%).
  - A/B: V9 OFF → audio sano ma senza condizionamento; V9 ON con placeholder corto → silenzio (fix: off-by-one BOS, `V9_EXTRA_FRAMES = 16+2`).
- **Artefatti**: `samples/vllm_omni_serving/vllm_omni_serving_v9_duplex_trimmed.wav` (frase lunga V9), `vllm_omni_serving_v9_ciao.wav`, `vllm_omni_serving_textonly_ciao.wav`.
- **Avvio**: `start_vllm_omni_server.sh` (container `gemma4_omni_server`, porta 8091, ~8 min warmup).

### Cosa NON funziona ancora
- **Streaming audio incrementale (async_chunk=true)**: il percorso è implementato (`thinker2talker_v9_async_chunk` → OmniPayloadStruct con `ids.output` + `hidden_states.output` + `meta.replace_streaming_prompt`) e il talker recupera il testo dai token ids Gemma worker-side, MA **il sender di stage 0 non emette chunk al connector** → il talker riceve solo il prewarm → errore. Da debuggare: catena `stage_sends_async_output` / save-loop del transfer adapter (il role "sender" di stage 0 sembra configurato da `get_connectors_config_for_stage`, ma il save-loop non parte).
- **Duplex runtime extension / TurnDetectorHead nel serving**: NON implementata contro l'interfaccia reale (`DuplexRuntimeExtension` con `configure_sampling_params`/`plan_append`/`decide_output`). `duplex_control_enabled=false` di default.
- **WebSocket `/v1/realtime`**: non testato.
- **`--served-model-name`**: usare un nome pulito (`gemma-4-omni-duplex`) invece del path del checkpoint.
- **`--generation-config vllm`**: il `generation_config.json` di Gemma sovrascrive i sampling params del thinker (temperature 1.0/0.95/64) — aggiungere il flag per usare i parametri del deploy.

### Note tecniche chiave (lezione appresa)
1. **Registrazione modello**: il ModelConfig di vLLM-Omni risolve via `OmniModelRegistry` (snapshot separato!) — registrare in ENTRAMBI (`ModelRegistry` + `OmniModelRegistry`).
2. **Entry point**: registrare in `vllm.general_plugins` (prefisso `zzz_` per ordine) E `vllm_omni.general_plugins` — caricati in tutti i processi.
3. **Pesi extra (V9/causal_conv)**: caricarli DENTRO `load_weights()` con `assign=True` e dichiarare i nomi nel set di ritorno (tracker vLLM).
4. **Buffer non-persistenti RoPE**: `assign=True` NON li sposta → `.to(device)` esplicito nel forward.
5. **Placeholder talker**: `ph_len = estimate + V9_EXTRA_FRAMES (16+2)` — corto di 1 frame = BOS troncato = silenzio.
6. **task_type VoiceDesign**: replica il prefill addestrato (tag think + lingua, nessun speaker — il modello 12Hz Base ha `spk_id: {}`).
7. **SubTalker**: il code predictor nativo vLLM NON supporta top_p<1.0 → `top_p: 1.0` (con top_k=20 il campionamento resta troncato).
8. **bnb**: plugin ufficiale `vllm-bnb-plugin` (non supportato in-tree da vLLM 0.28); serve shim `WeightsMapper.get_rename_mapper/_map_name`; NESSUN flag `--quantization` globale (si inferisce dal config.json per stage).

---

## 4. PROSSIMI PASSI (in ordine)
1. **Debug async_chunk** (streaming audio): capire perché il sender di stage 0 non emette chunk (save-loop / role). Ipotesi: il save-loop richiede `_should_use_async_omni_output` (async scheduling + `use_async_omni_output` sul model_config) oppure la partition dei pooler output; verificare con log del transfer adapter.
2. **Duplex runtime extension** (TurnDetectorHead) contro l'interfaccia reale `vllm_omni.experimental.fullduplex.engine.contracts` e WebSocket `/v1/realtime` E2E.
3. **`--served-model-name gemma-4-omni-duplex`** + `--generation-config vllm`.
4. **Benchmark finale**: 50 domande conversazionali, WER/Crest/HNR/THD/TTFA.
