# NEXT_STEP — Gemma 4 Full-Duplex Omni Pipeline (28 Agosto 2026, notte)

## ✅ COMPLETATO E VALIDATO

### 1. Linea Projector + TurnDetectorHead (ADR-024/027) — invariato
### 2. Serving vLLM-Omni con V9 (ADR-028) — full-payload validato
### 3. Debug rapido vLLM: modalità DEBUG (enforce_eager + kv_cache_memory) → startup 4.5 min; volume cache vLLM persistente.
### 4. 4 capability voce (report Qwen3-TTS arXiv:2601.15621): system prompt ✓, voice_style ✓, ref_audio cloning ✓, voci default (profili x-vector) ✓.
### 5. 🆕 STREAMING ASYNC SBLOCCATO (async_chunk=true)
- **Root cause risolta**: il prewarm del talker usava `compute_talker_prompt_ids_length` (Qwen-specific) → prompt_len=1 → prefill troncato → rumore. Fix: monkey-patch della funzione nel modulo `vllm_omni.distributed.omni_connectors.adapter` (stima generosa del placeholder) — ATTENZIONE: l'import nel prewarm è LOCALE, la patch va sul modulo adapter non sull'orchestrator.
- Fix aggiuntivi: chiave payload "hidden" (non "hidden_states"), accumulo `new_token_ids` (canale affidabile), testo ricostruito worker-side dai token ids, task_type VoiceDesign default.
- **Risultato**: audio streaming incrementale (chunk code2wav), ASR quasi perfetto ("dati zorio" = unico errore), EOS campionato.
### 6. 🆕 FULL DUPLEX WebSocket (livello 1 — chat fallback) FUNZIONANTE
- **`Gemma4OmniServingRuntimeAdapter`** (duplex_adapter.py): protocollo ServingRuntimeAdapter minimale (session state, PcmAppendBuffer, data plane no-op, capabilities).
- **`Gemma4OmniDuplexRuntimeExtension`** (duplex_extension.py): extension minima (configure_sampling_params pass-through, plan_append, decide_output=None).
- Pipeline: duplex_control_enabled=True + extension + adapter. Deploy: `session_mode: duplex` **TOP-LEVEL** (non per-stage!).
- **Endpoint verificati**: `/v1/realtime?duplex=1` → session.created, conversation.item.create, response.create → eventi Realtime (output_item.added, content_part.added audio+text, output_text.delta, **response.audio.delta × 18 chunk**, response.done in 34s). `/v1/duplex` route attivo.
- Nota: il `session.update` del fallback chat si aspetta `model` dentro `session` (diverso dall'STT upstream che lo vuole top-level).

## ⚠️ Limiti noti / da ottimizzare
- **Durata audio**: per risposte brevi il talker può generare code lunghe (max_tokens 512, EOS ritardato) — da applicare ARIA pacing per-request (max_tokens = n_words×4.8+12) e trim client-side (il serving non taglia la coda).
- **TurnDetectorHead nativo (livello 2)**: il barge-in con data plane audio (come nemotron) NON è integrato — il fallback chat gestisce il barge-in client-side (epoch). Il TurnDetectorHead va agganciato come policy in decide_output quando il data plane nativo sarà attivo.
- **Qualità async**: ASR quasi perfetto ma con un micro-errore ("dati zorio") — da investigare (left context chunking).
- **--served-model-name** pulito + `--generation-config vllm` (i sampling del thinker sono sovrascritti dal generation_config di Gemma).

## 📋 Prossimi passi
1. ARIA pacing per-request (max_tokens talker = n_words×4.8+12) + trim client-side.
2. TurnDetectorHead nel data plane nativo (barge-in server-side) — livello 2.
3. Benchmark finale: 50 domande conversazionali (WER/Crest/HNR/THD/TTFA) + confronto voci.
