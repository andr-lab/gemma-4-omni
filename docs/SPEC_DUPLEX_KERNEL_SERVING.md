# SPEC — Duplex Kernel Serving: Funzionalità Richieste (Roadmap Feature-Complete)

**Stato**: Roadmap — documenta le funzionalità che il kernel custom (ADR-029) deve
avere per essere feature-complete rispetto al serving di riferimento vLLM-Omni
(ADR-028) e al report Qwen3-TTS (arXiv:2601.15621, 4 capability voce).

**Data**: 2026-08-29 · **File**: `duplex_kernel/` (server.py, session.py, engine.py,
turnhead.py, kv_cache.py, models/)

Questa spec è il **contratto funzionale** per i prossimi passi del kernel. Per ogni
funzionalità: requisito, stato attuale, contratto API, implementazione target.
La verifica di completamento è: client `client/gemma4_omni_test_client.py` (tutte le
modalità) + benchmark 5 domande (metriche fisiche + ASR).

---

## 1. Input / Output Testo

### Requisiti
- `/v1/chat/completions` con `messages` OpenAI (roles user/assistant/tool/system).
- Streaming SSE token-by-token del testo del thinker (`stream=true`).
- Storia conversazionale per sessione (multi-turno, rolling su THINKER_MAX_TURNS=8).

### Stato attuale
- ✅ `POST /v1/chat/completions` (server.py) — text base, non streaming reale.
- ✅ `engine.think()` con chat template Gemma4 applicato.
- ⚠️ Streaming: `_gen()` manda **un unico blob** di testo/audio, non delta progressivi.
- ⚠️ Sessione: `sessions.create_session()` nuovo per ogni richiesta HTTP (niente
  `session_id` riusabile via header) — la storia è una tantum.

### Implementazione target
```python
POST /v1/chat/completions
{ "model": "gemma-4-omni-duplex", "messages": [...],
  "stream": true, "modalities": ["text"],
  "session_id": "..." }            # opzionale: riusa storia+KV

SSE:
data: {"choices":[{"delta":{"role":"assistant","content":"Ciao"}}]}
data: {"choices":[{"delta":{"content":" mondo"}}]}
data: {"choices":[{"delta":{},"finish_reason":"stop"}}]}
data: [DONE]
```
- Supporto `session_id` nel body → `SessionManager.get()` (riusa KV thinker/talker).
- Thinker: generazione token-by-token con `StreamingDuplexController` (yield testo
  incrementale) invece di `max_new_tokens` monolitico.

---

## 2. Output Audio (text → speech)

### Requisiti
- `modalities: ["text","audio"]` → risposta con `audio.data` (WAV base64).
- ARIA pacing (ADR-023): budget frame da `n_words` (T_gen = max(48,min(140,n*4.8+12))).
- Trim + fade + peak-norm della coda (postprocess già in engine).

### Stato attuale
- ✅ `engine.talk(text, h_pfx, n_words)` → wav 24kHz, postprocess incluso.
- ✅ `audio.data` WAV base64 nel non-streaming.
- ⚠️ Streaming SSE audio: manda **tutto il WAV in un chunk**, non chunk code2wav
  incrementali (il WS usa chunk 640ms, ma l'SSE no).

### Implementazione target
- SSE stream con chunk audio 640ms (`CODEC_CHUNK_FRAMES=8` → 8 frame @12.5Hz):
  ```
  data: {"choices":[{"delta":{"audio":{"chunk":"<wav b64 640ms>"}}}]}
  ```
- Evento `finish_reason: "stop"` a fine audio.
- Riuso KV talker nel turno: `TalkerPrefixKV` + `TurnKVState` (oggi `engine.talk`
  rifà il prefill `[ICL|h_pfx|BOS]` a ogni chiamata).

---

## 3. Conditioning Voce (4 capability — report Qwen3-TTS)

Il kernel deve replicare `vllm_omni_plugin/voices.py::resolve_voice_conditioning`
(PRIORITÀ: voice profilo → voice_style → ref_audio → default VoiceDesign).
Oggi **NON collegato**: `_run_response` ignora `voice`; session.py ha i campi ma
il talker non li riceve.

### 3.1 Voci di default (scelta della voce)
- **Requisito**: `voice: "gemma" | "leonardo"` (e profili custom) → profilo x-vector
  registrato in `custom_voice_dir`.
- **Formato** (ufficiale vLLM-Omni, già implementato in `vllm_omni_plugin/voices.py`):
  `custom_voice_dir/custom_voice_manifest.json`:
  ```json
  {"gemma": {"file": "gemma.safetensors", "mode": "xvec", "ref_text": "..."},
   "leonardo": {"file": "leonardo.safetensors", "mode": "icl", "ref_text": "..."}}
  ```
  Tensore `speaker_embedding` (enc_dim 2048); `ref_code` obbligatorio se mode=icl.
- **Mapping talker**: `task_type=Base` + `voice_clone_prompt={ref_spk_embedding, icl_mode}`,
  `x_vector_only_mode` (True per xvec, False per icl).
- **Target**: portare `load_voice_profiles()` + `_load_profile_tensors()` nel kernel
  (modulo `duplex_kernel/voices.py`), con `CONFIG.CUSTOM_VOICE_DIR`.

### 3.2 Prompting per lo stile della voce (voice_style / instructions)
- **Requisito**: `voice_style: "parla in modo caldo e rassicurante"` → VoiceDesign.
- **Mapping talker**: `task_type=VoiceDesign` + `instruct=[voice_style]`.
- **Stato**: il campo esiste in `DuplexSession.voice_style` ma non è usato.
- **Target**: risolvere `voice_style`/`instructions` → `instruct` nel talker
  (il prompt del talker native supporta instruct).
- **Nota qualità (ADR-028 §2.4)**: VoiceDesign puro dà HNR migliore del path
  x-vector; testare entrambi nel benchmark comparativo.

### 3.3 ref_audio (voice cloning 3s)
- **Requisito**: `ref_audio` (WAV base64) + opzionale `ref_text` → task Base con
  `ref_audio: [[samples, 24000]]` (cloning a 3 secondi, x_vector_only_mode).
- **Stato**: campo `ref_audio` in `DuplexSession` (np.ndarray) ma mai popolato.
- **Target**: decodifica base64→PCM→resample 24kHz (pattern `_decode_wav_b64`),
  passare a `task_type=Base`; `ref_text` presente → `x_vector_only_mode=False`
  (cloning completo), assente → True (solo timbro).

### 3.4 speaker_prompt (alias)
- **Requisito**: `speaker`/`speaker_prompt` accettati come alias di `voice`
  (pattern `_extract_voice_and_language`: `voice or speaker`).
- **Target**: normalizzare in ingresso, passare al resolver.

### Contratto unificato (tutte le modalità)
```python
POST /v1/chat/completions
{ "voice": "gemma",                          # o voice_style / ref_audio
  "voice_style": "tono caldo e gentile",      # VoiceDesign instruct
  "ref_audio": "<wav b64>", "ref_text": "...",# cloning 3s
  "instructions": "..." }                     # alias di voice_style
```

---

## 4. Full-Duplex WebSocket Realtime (con Projector V9 + TurnDetectorHead)

### Requisiti (livello 2 — audio-native, non chat fallback)
- `/v1/realtime?duplex=1`: audio utente in streaming (PCM 16k) → audio risposta
  in streaming, **sovrapposti** (parla mentre ascolta, barge-in).
- **TurnDetectorHead nativa nel loop**: `z_t = [h_t(2048) ∥ u_mel(128) ∥ Δu_mel(128)]`
  (ADR-027), FSM StreamingDuplexController: BARGE_IN → flush/stop; EMIT_BC →
  backchannel; YIELD → auto-commit risposta.
- **h_t dal V9** (media degli h_pfx frame, 1,16,2048) — **MAI ht_adapter** (LEGACY).
- **KV persistente**: prefix talker `[ICL|h_pfx|BOS]` calcolato 1 volta per
  sessione (TalkerPrefixKV); troncamento a fine turno (TurnKVState → zero-reset
  su barge-in); thinker rolling (ThinkerContext).

### Stato attuale
- ✅ `WS /v1/realtime` scaffold: session.update, audio append/commit, response.create.
- ✅ Il Thinker E2B può ricevere PCM 16 kHz direttamente: `AutoProcessor` Gemma4 genera mel 128-bin, il Conformer nativo produce le feature continue e il prefill usa placeholder `audio_token_id` + `input_features_mask`; il decode testuale prosegue con la KV cache.
- ⚠️ Per il percorso audio è obbligatorio il checkpoint ufficiale BF16 `google/gemma-4-E2B-it`: il mirror Unsloth BNB 4-bit non contiene i `quant_state` dell'audio tower e fallisce in `Linear4bit`.
- ⚠️ **TurnHeadRuntime NON alimentato nel loop**: `set_thinker_hidden(h_pfx)` mai
  chiamato; `decide()` su commit usa mel utente ma non h_t.
- ⚠️ `_feed_user_audio_to_turnhead` dipende dal processor Gemma4 (try/except, no
  fallback mel da numpy).
- ⚠️ Il talker non usa KV persistente: `engine.talk` rifà il prefill a ogni risposta.
- ⚠️ Niente barge-in attivo: mentre `active_response`, l'audio utente in arrivo
  non interrompe (serve check turnhead → stop del generate corrente).
- ⚠️ `engine.think` non è streaming: la risposta è generata in blocco, non
  token-by-token con commit progressivo.

### Implementazione target (loop duplex corretto)
```python
while True:
    ev = ws.receive()
    if ev.type == "input_audio_buffer.append":
        pcm = decode(ev.audio)                    # 16k PCM
        mel = gemma4_feature_extractor(pcm)       # (T,128) @12.5Hz
        sess.turnhead.feed_user_mel(mel)          # ascolto continuo
        sess.turn_kv.append_audio(pcm)            # audio del turno
        if sess.active_response:                  # BARGE-IN check
            action = sess.turnhead.decide()
            if action.pred == BARGE_IN:
                engine.stop_generation()          # stop generate talker
                sess.turn_kv.truncate_to_prefix() # zero-reset
                yield response.cancel
    elif ev.type == "input_audio_buffer.commit":
        # fine micro-turn utente: prefill thinker con audio+testo
        h_pfx = engine.v9_prefix(think(...))
        sess.turnhead.set_thinker_hidden(h_pfx)   # h_t al turnhead
        if sess.turnhead.decide().pred == YIELD:
            await _stream_response(sess, ws)      # audio chunk 640ms
    elif ev.type == "response.create":
        ...
```

### Eventi Realtime da emettere (compat client esistente)
`session.updated` · `input_audio_buffer.committed` · `response.created` ·
`response.output_text.delta` · `response.audio.delta` (640ms WAV b64) ·
`response.output_text.done` · `response.done` · `response.cancelled` (barge-in) ·
`error`.

---

## 5. Tool Calling / Function Calling

### Requisiti
- Chat template Gemma4 con `<tool_call>`/`<tool_response>` (già nel tokenizer).
- Parser locale delle chiamate (niente vLLM): estrarre `{"name": ..., "arguments": ...}`
  dal testo generato; eseguire; reiniettare nel contesto; secondo giro di think.

### Stato attuale
- ❌ Non implementato nel kernel (il plugin vLLM lo aveva via
  `--enable-auto-tool-choice --tool-call-parser gemma4`).
- ✅ Il tokenizer Gemma4 ha il template nativo (verificato in ADR-028).

### Implementazione target
```python
POST /v1/chat/completions
{ "messages": [{"role":"user","content":"che ore sono?"}],
  "tools": [{"type":"function","function":{"name":"get_time","parameters":{...}}}],
  "tool_choice": "auto" }
→ prima risposta: {"tool_calls":[{"id":"...","function":{"name":"get_time","arguments":"{}"}}]}
→ client esegue, invia {"role":"tool","tool_call_id":"...","content":"14:30"}
→ seconda risposta: testo finale + audio
```
- Parser: regex/JSON sul testo del thinker prima di EOS 2150; se tool_calls →
  ferma l'audio, emette solo tool_calls.

---

## 6. Altre capability (parità ADR-028)

| Funzionalità | Stato kernel | Target |
|---|---|---|
| `GET /health` | ✅ | ok |
| `GET /v1/models` | ✅ (`gemma-4-omni-duplex`) | ok |
| `max_tokens` thinker | ✅ (default 64) | per-request |
| `language` (it/en/...) | ❌ | `codec_language_id` dal config talker |
| `temperature`/`top_p` thinker | ⚠️ hardcoded | per-request |
| Trim/fade coda audio | ✅ `_postprocess` | ok |
| Voci custom (nuovi profili) | ⚠️ via manifest | `tools/build_voice_profiles.py` |
| Micro-batching multi-sessione | ⚠️ scaffold (SessionManager.decode_ready) | decode batching |
| KV quantizzata FP8 | ❌ | dopo CUDA graph backbone |
| CUDA graph talker backbone | ❌ | 33→~12ms (ADR-030 pattern) |

---

## 7. Priorità e ordine di implementazione

1. **Condizionamento voce** (sez. 3): collegare voice/voice_style/ref_audio al
   talker — massimo impatto funzionale, basso rischio (port di `voices.py`).
2. **Streaming SSE reale** (sez. 1-2): delta testo + chunk audio 640ms.
3. **KV persistente nel talker** (TalkerPrefixKV/TurnKVState nel loop) → TTFA ~0
   al 2° turno + abilita barge-in.
4. **TurnHead nativa nel loop duplex** (sez. 4): set_thinker_hidden, barge-in,
   auto-commit YIELD — il cuore del full-duplex.
5. **Tool calling** (sez. 5).
6. **CUDA graph talker backbone** → target RTF ~0.5.

**Gate**: ogni feature validata con smoke test prima del run completo (TDD/DDD,
AGENTS.md); benchmark finale 5 domande con ASR + metriche fisiche per feature.

## File coinvolti
- `duplex_kernel/server.py` (endpoint, streaming, duplex loop)
- `duplex_kernel/session.py` (popolare voice/voice_style/ref_audio nel talker)
- `duplex_kernel/engine.py` (think streaming, talk con KV, stop_generation)
- `duplex_kernel/voices.py` (NUOVO: port di vllm_omni_plugin/voices.py)
- `duplex_kernel/config.py` (CUSTOM_VOICE_DIR, prompt defaults)
- `duplex_kernel/turnhead.py` (set_thinker_hidden già presente, collegare)
- `duplex_kernel/kv_cache.py` (TalkerPrefixKV/TurnKVState già pronti, usare)
