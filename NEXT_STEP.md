# Gemma 4 Full-Duplex Omni — STATO CONSOLIDATO (29 Agosto 2026, kernel RTF ~1.0)

## Kernel serving slice — verificata su RTX 3090
Il benchmark release su 5 domande (`samples_kernel_release_report.json`) misura
RTF medio **0.995**, crest medio 4.19, clipping 0.208%; il Sub-Talker usa 15
CUDA Graph catturati su StaticCache. Il server espone chat/audio, SSE a chunk
640 ms, session_id, parser tool e WebSocket non bloccante per barge-in.
Il backbone talker CUDA Graph e il vero STT da PCM nel WS restano attività
separate: il WS audio accumula PCM e usa il testo della sessione finché non
viene collegato il percorso Gemma audio encoder.


Documento di riferimento dello stato del progetto. Copre: architettura, milestone completate,
stato del custom duplex kernel (ADR-029/030), stato del serving vLLM-Omni (full duplex),
limiti noti e prossimi passi.

---

## 0. CUSTOM DUPLEX KERNEL (ADR-029) — FUNZIONANTE, RTF ~1.0

Kernel narrow autocontenuto in `duplex_kernel/` (ispirato a DwarfStar, antirez/ds4):
processo singolo, niente unsloth/kaggle_pipeline/vLLM. **CUDA graph sub-talker
ricostruito e validato (ADR-030)**: 99 → 15.5ms/frame (6.4×), sampling bit-exact.

| Metrica | Eager | **CUDA Graph** |
|---|---|---|
| Sub-talker 15 residui | 99 ms/frame | **15.5 ms/frame** |
| RTF end-to-end | 2.28 | **~1.0** (0.97–1.09) |
| ASR Whisper | 5/5 (generate HF) | **3/3 perfetto** |
| Load (tutti i modelli) | — | **~36s** (vLLM: ~4.5min) |

- Architettura: Thinker Gemma4 E2B bf16 (8.82GB) → V9 → Talker Qwen3-TTS →
  Code2Wav; TurnDetectorHead (h_t = media h_pfx dal V9, ht_adapter è LEGACY).
- Gotcha documentati in ADR-030: prefill con talker_emb 2048 + small_to_mtp_projection;
  cache 17 posizioni; reset cumulative_length per StaticLayer; buffer statici per
  cache_position; sampling senza seed fisso + warper T→top_k→top_p.
- **Prossimo**: **serving feature-complete — vedi [SPEC_DUPLEX_KERNEL_SERVING.md](docs/SPEC_DUPLEX_KERNEL_SERVING.md)**
  (conditioning voce: voice/voice_style/ref_audio/speaker_prompt; streaming SSE testo+audio;
  full-duplex con TurnHead nativa e KV persistente; tool calling; CUDA graph backbone → RTF ~0.5).

---

## 1. ARCHITETTURA (consolidata)

```
Utente (testo o voce) → Gemma 4 Thinker (E2B) ──testo+hidden──▶ Projector V9 (211.83M)
      ──h_pfx 16 frame──▶ Talker Qwen3-TTS 1.7B ──RVQ 16 codebook──▶ Code2Wav 24kHz
                                            ▲
                                    Prefix KV-Cache [ICL | h_pfx | BOS]
```

- **Thinker**: `unsloth/gemma-4-E2B-it` (frozen, bf16 / nf4 nel serving).
- **Talker**: `Qwen3-TTS-12Hz-1.7B-Base` (frozen) + **Projector V9** (Deep Causal Transformer, 4 layer, RoPE/SwiGLU/RMSNorm, 211.83M) + **causal conv** addestrata (7.08M, Task 11i SOTA).
- **Checkpoint chiave**: `v9_prefix_final.pt` (step 10000, Val CB0 2.294, Acc 40.8%), `custom_duplex_heads.pt` (causal_conv), `turn_dynamics_v1_final.pt` (TurnDetectorHead), `cache_ars_full_v2.pt` (features).
- **Hardware**: RTX 3090 24GB (`gpu3090`, alias SSH).

---

## 2. FATTO E VALIDATO (tutte le milestone)

### 2.1 Linea di modellazione (completa)
- Data prep 1.83M frame CV26 IT; engine Dense 3D MRoPE (V8/V9); Prefix KV-Cache (ADR-022); Projector V9 (ADR-024); ARIA streaming (ADR-023); TurnDetectorHead (ADR-027); **CUDA Graph Sub-Talker ricostruito nel kernel (ADR-030: 15.5ms/frame, RTF ~1.0)**; distorsioni risolte (SnakeBeta, e_tts_pad, top_k=20, ADR-023/026).

### 2.2 Serving vLLM-Omni (ADR-028) — VALIDATO
- **Pipeline 3 stadi disaggregata** su vLLM-Omni 0.28 + vllm-bnb-plugin 0.0.3, porta 8091:
  - Stage 0: `Gemma4ForConditionalGeneration` (nf4, ~7.3GB)
  - Stage 1: `Gemma4OmniV9TalkerForConditionalGeneration` (talker + Prefix V9, ~3.9GB)
  - Stage 2: `Qwen3TTSCode2Wav` (~0.5GB)
- **Endpoint OpenAI**: `/v1/chat/completions` (text+audio, SSE streaming), `/v1/models`, `/health`.
- **Qualità**: ASR Whisper 100% fedeltà, Crest 4.63, HNR +3.06 dB, THD+N 33.1% (full-payload, VoiceDesign).
- **Plugin out-of-tree**: pipeline registrata (model_type `gemma4_omni_duplex` + alias `gemma4`), modelli custom, bnb-compat shim, entry point in `vllm.general_plugins` (zzz_) e `vllm_omni.general_plugins`.

### 2.3 Debug rapido vLLM
- Modalità DEBUG (`enforce_eager` + `kv_cache_memory` fisso per-stage) → **startup 4.5 min** (da 8).
- Volume cache vLLM persistente (`~/.cache/vllm`) → 2° boot prod più veloce.
- `GEMMA4_OMNI_MODE=debug|prod` nello start script.

### 2.4 4 capability voce (report Qwen3-TTS arXiv:2601.15621)
1. ✅ **System prompt** LLM (chat template standard).
2. ✅ **voice_style** → VoiceDesign + instruct (HNR 3.06 dB, THD 33%).
3. ✅ **ref_audio + ref_text** → Base + cloning 3s (ASR 100%).
4. ✅ **voice="gemma"|"leonardo"** → profili x-vector di default (custom_voice_dir, formato ufficiale vLLM-Omni; ASR 100%).

### 2.5 🆕 Streaming async (async_chunk=true) — SBLOCCATO
- Root cause risolta: il prewarm del talker usava `compute_talker_prompt_ids_length` (Qwen-specific) → prompt_len=1 → prefill troncato → rumore. **Monkey-patch sul modulo `vllm_omni.distributed.omni_connectors.adapter`** (l'import nel prewarm è locale).
- Fix aggiuntivi: chiave payload `hidden`, accumulo `new_token_ids`, testo ricostruito worker-side, task_type VoiceDesign default.
- Risultato: audio streaming incrementale (chunk code2wav), ASR quasi perfetto, EOS campionato.

### 2.6 🆕 Full Duplex WebSocket (livello 1 — chat fallback) — FUNZIONANTE
- `Gemma4OmniServingRuntimeAdapter` (protocollo ServingRuntimeAdapter) + `Gemma4OmniDuplexRuntimeExtension` (protocollo DuplexRuntimeExtension) + `duplex_control_enabled` + `session_mode: duplex` **top-level** nel deploy.
- `/v1/realtime?duplex=1` **verificato**: eventi Realtime completi (output_item.added, content_part audio+text, output_text.delta, response.audio.delta × 18 chunk, response.done in ~34s). Route `/v1/duplex` attivo.

---

## 3. NON FATTO / LIMITI NOTI

| # | Cosa | Stato | Note |
|---|------|-------|------|
| 1 | **TurnDetectorHead nativo nel data plane** (barge-in server-side) | Da fare (livello 2) | Il fallback chat gestisce il barge-in client-side (epoch). Il TurnDetectorHead va agganciato in `decide_output` quando il data plane audio sarà attivo (pattern nemotron_voicechat). |
| 2 | **ARIA pacing per-request** (max_tokens talker) | Da fare | Risposte brevi possono generare code lunghe (max_tokens 512, EOS ritardato). Applicare max_tokens = n_words×4.8+12 per-request. |
| 3 | **Trim/peak-normalizzazione** della coda audio | Client-side | Il serving non taglia la coda di silenzio; i client devono farlo (come il ground truth). |
| 4 | `--served-model-name gemma-4-omni-duplex` | Da fare | Il model id attuale = path del checkpoint. |
| 5 | `--generation-config vllm` | Da fare | Il generation_config di Gemma sovrascrive i sampling del thinker. |
| 6 | Qualità async: micro-errore "dati zorio" | Da investigare | Probabile left-context del chunking code2wav. |
| 7 | Benchmark finale | Da fare | 50 domande conversazionali: WER/Crest/HNR/THD/TTFA, confronto voci. |
| 8 | Voci di default: qualità acustica inferiore al default | Nota | Il path Base+x-vector dà HNR più basso del VoiceDesign puro (voce diversa, intelligibile). |

---

## 4. OPERATIVO (comandi)

```bash
# Avvio server (modalità DEBUG di default, 4.5 min warmup)
bash start_vllm_omni_server.sh            # GEMMA4_OMNI_MODE=debug (default)
GEMMA4_OMNI_MODE=prod bash start_vllm_omni_server.sh

# Test chat (text+audio)
curl -s -X POST http://localhost:8091/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"<MODEL_PATH>","messages":[{"role":"user","content":"Ciao!"}],"max_tokens":40,"modalities":["text","audio"]}'

# Test full duplex
python3 test_realtime_ws3.py   # (client WebSocket in /workspace)
```

- Model path: `/workspace/.hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/0ae102ac289cc3f17500a607ad817849778b8ba7`
- Voci default: `custom_voices/` (gemma, leonardo) — rigenerabili con `tools/build_voice_profiles.py` (mel slaney esatta).
- Artefatti audio: `samples/vllm_omni_serving/`.

---

## 5. PROSSIMI PASSI (in ordine)

1. **ARIA pacing per-request** (max_tokens talker) + `--served-model-name` + `--generation-config vllm` (quick wins).
2. **TurnDetectorHead nel data plane** (livello 2 duplex: barge-in server-side, /v1/realtime audio-native).
3. **Qualità async**: investigare "dati zorio" (left context chunking).
4. **Benchmark finale**: 50 domande, confronto con/senza voci, report `gemma4_training_report.md`.
