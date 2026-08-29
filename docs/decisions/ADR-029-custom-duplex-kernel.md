# ADR-029: Custom Optimized Duplex Kernel — DwarfStar-Inspired Narrow Inference Engine

## Status
**Accepted** (2026-08-29) — kernel funzionante e benchmarkato: RTF ~1.0, ASR perfetto.

## Stato di implementazione (aggiornato 2026-08-29)

Il kernel `duplex_kernel/` è implementato e validato end-to-end su RTX 3090:

| Voce | Stato | Dettaglio |
|---|---|---|
| Load self-contained (niente unsloth/vLLM) | ✅ | Thinker bf16 8.82GB + Talker + V9 + TurnHead, ~36s |
| Smoke test | ✅ | 1 domanda → testo+audio WAV, metriche fisiche |
| **CUDA graph sub-talker** (ADR-030) | ✅ | 99 → 15.5ms/frame (6.4×), sampling bit-exact |
| **RTF end-to-end** | ✅ | **~1.0** (0.97–1.09), da 2.28 eager |
| ASR fedeltà | ✅ | 3/3 perfetto nel benchmark (Crest 4.2–4.5) |
| KV cache per-sessione | ⚠️ | session_id HTTP, rolling history e zero-reset TurnKV; riuso del prefix talker non ancora nel loop di prefill |
| Server HTTP/WS core | ✅ | chat/audio, SSE chunk 640ms, tool parser, voice resolver, WS duplex e barge-in stop event |

**Correzione importante**: il "RTF 0.7541" citato in README/NEXT_STEP (ADR-026)
proveniva da un esperimento CUDA graph il cui codice **non è mai stato
persistito** — `interactive_gemma4_omni_live.py` è eager. Il kernel attuale
raggiunge RTF ~1.0 con graph ricostruiti (ADR-030); il prossimo passo (CUDA
graph sul talker backbone, 33→~12ms) porta il target a ~0.5, sotto la soglia
real-time con margine.

## Gestione KV Cache (focus critico per il duplex continuo)

In full-duplex continuo il contesto cresce senza limite: storia conversazionale,
audio utente in streaming, codici RVQ generati. Sintesi dei meccanismi dalle
fonti (Gemma 4 TR §2, Qwen3.5-Omni §2.5, DwarfStar, VoiceChat-TTS):

### Meccanismi nativi già disponibili (da sfruttare)
| Fonte | Meccanismo | Impatto KV |
|---|---|---|
| Gemma 4 TR §2 | Sliding window 4:1 (locale/globale) in E2B | I layer local evictano da soli |
| Gemma 4 TR §2 | KV shared ratio 20/35 (key riusata come value) | −37.5% KV globale |
| Gemma 4 TR Tab.3 | Quantized KV cache (bf16 → basso bit) | ~2-4× capacità a parità VRAM |
| Qwen3.5-Omni §2.5 | Chunked prefill (audio/vision encoder a chunk) | TTFT ridotto; KV I/O limitato |
| Qwen3.5-Omni §2.5 | GDN (Gated Delta Net) long-context | I/O KV ridotto in sequenze lunghe |
| VoiceChat-TTS | Zero-Reset KV su barge-in | KV del prefisso preservata |

### Meccanismi specifici da implementare nel kernel

**1. Thinker — rolling context window (conversazionale)**
- Contest = `[system | ultimi K turni | audio_chunk corrente]` con K fisso (es. 8).
- A ogni fine turno: evict del turno più vecchio, append del nuovo. Il prefill
  del thinker è quindi limitato (non cresce col tempo di sessione).
- La storia oltre K turni è opzionalmente riassunta (testo compresso iniettato
  come system-followup) — nessuna crescita illimitata.
- KV del thinker: quantizzata (bf16→FP8) per raddoppiare la finestra utile.

**2. Talker — prefix KV persistente + troncamento a fine turno**
- Il prefisso `[ICL_text | h_pfx | BOS]` è STABILE per sessione (stessa voce,
  stesso stile): la sua KV si calcola UNA volta al primo turno e si riusa
  (prefix KV caching).
- Durante il turno: append dei codici RVQ correnti. Su YIELD/stop: **tronca la
  KV ai token del prefisso** (i codici del turno finito sono già stati emessi
  come audio) → il turno successivo riparte dal prefisso con costo ~0.
- Zero-Reset KV su barge-in: il prefisso non viene toccato; si tronca solo la
  coda del turno interrotto (VoiceChat-TTS).

**3. Audio utente — chunked streaming**
- PCM 16k → mel → encoder audio Gemma4 a chunk (chunked prefill): il
  passato audio lontano viene evictato, si mantiene l'ultimo chunk + il
  riassunto semantico implicito nella storia testuale del thinker.

**4. Sub-talker / code2wav — stateless per chunk**
- Sub-talker CUDA graph: KV solo per i 15 codebook del frame corrente
  (nessuna crescita). Code2Wav: convnet causale, stateless.

### Contabilità memoria (RTX 3090 24GB, sessione singola)
| Componente | KV | Note |
|---|---|---|
| Thinker E2B (sliding 4:1) | ~2-4GB @ 8k token | quantizzata FP8 |
| Talker prefix (ICL+h_pfx+BOS) | ~0.5GB | stabile, calcolata 1 volta |
| Talker coda turno | ~0.2GB | troncata a fine turno |
| Sub-talker + code2wav | ~0.1GB | stateless |
| Modelli (Thinker bf16/nf4 + V9 + Talker) | ~13GB | |

## Riferimento architetturale: DwarfStar (antirez/ds4)
Progetto di riferimento: **antirez/ds4 (DwarfStar)** — motore di inferenza nativo
(MIT) specializzato per DeepSeek V4 Flash/GLM 5.2, scritto da Salvatore Sanfilippo
(creatore di Redis). Principi fondanti riutilizzabili:

1. **Deliberatamente narrow**: non un GGUF runner generale — modello, prompt
   rendering, tool calls, KV state, HTTP server e agent costruiti e testati
   insieme. Ogni pezzo conosce l'architettura target.
2. **Session worker residenti + model coordinator**: ogni sessione mantiene la
   propria KV; un coordinator batcha le sessioni decode-ready e serializza
   prefill quantizzati → micro-batching multi-utente (120 t/s aggregati,
   2000 t/s prefill su 8×L40S).
3. **KV cache compressa + ownership per-sessione**: contesti lunghi pratici.
4. **Quantizzazione asimmetrica**: solo i componenti che dominano la memoria e
   tollerano il downgrade (routed MoE experts → IQ2_XXS/Q2_K) vengono quantizzati;
   il resto resta ad alta precisione per qualità.
5. **Speculative decoding / MTP**: DSpark propone token, il main model verifica
   in batch — stesso pattern del nostro sub-talker speculativo (ADR-025/026).
6. **Test vectors ufficiali + QA prima dei release**: riproducibilità garantita
   (`--quality`, `--dspark-strict`).

## Contesto
Il porting full-duplex su vLLM-Omni (ADR-028) ha costi alti: 3 stage disaggregati
+ IPC + plugin out-of-tree, RTF ~2.0 (26s per ~13s audio). Il kernel dedicato
(ADR-030: CUDA graph sub-talker ricostruito e validato) misura RTF ~1.0 end-to-end
con ASR perfetto e carica in ~36s (vs ~4.5min di warmup vLLM). La complessità del
data plane nativo di vLLM-Omni (stage0 duplex runtime, plan_append/decide_output,
TurnDetectorHead nel framework) è sproporzionata rispetto al nostro bisogno: una
pipeline fissa a 3 modelli con checkpoint custom.

## Decisione proposta
**Custom duplex kernel** (narrow, ispirato a DwarfStar): un runtime di inferenza
dedicato in PyTorch/CUDA che carica i checkpoint addestrati (V9, causal_conv,
ht_adapter, TurnDetectorHead) e orchestra i 3 modelli in un processo singolo,
mantenendo l'interfaccia OpenAI-compatibile. vLLM-Omni resta come runtime di
riferimento standard-compliant.

### Architettura (DwarfStar-inspired)
```
┌────────────────────────────────────────────────────────────────────┐
│  duplex_kernel/  (processo singolo, GPU dedicata)                  │
│                                                                    │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │ HTTP+WS layer│ → │ SessionManager   │ → │ ModelCoordinator │    │
│  │ /v1/chat,    │   │ (sessioni duplex │   │ (micro-batching, │    │
│  │ /v1/realtime │   │  KV per-session)│   │  decode batching)│    │
│  └──────────────┘   └────────┬─────────┘   └────────┬─────────┘    │
│                              │                      │              │
│  ┌───────────────────────────▼──────────────────────▼────────────┐  │
│  │  PipelineEngine (CUDA graphs, KV persistente)                  │  │
│  │  Thinker Gemma4 E2B (nf4/bf16)                                │  │
│  │    → h_tx + h_raw_style → causal_conv → V9 (211.83M)          │  │
│  │    → h_t (ht_adapter 2048) [per TurnHead]                     │  │
│  │  Talker Qwen3-TTS (CUDA graph sub-talker 12.25ms)             │  │
│  │    → [ICL | h_pfx | BOS] → RVQ 16 codebook autoregressive     │  │
│  │  Code2Wav 24kHz (CUDA graph)                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────┐                   │
│  │  TurnDetectorHead + StreamingDuplexController│ ← nativo nel loop│
│  │  (z_t = [h_t thinker ∥ u_mel ∥ Δu_mel])      │    (no data plane)│
│  └─────────────────────────────────────────────┘                   │
└────────────────────────────────────────────────────────────────────┘
```

### Ottimizzazioni (già dimostrate o portabili 1:1)
| Ottimizzazione | Riferimento | Risultato |
|---|---|---|
| CUDA graph sub-talker RVQ | ADR-030 | 99 → 15.5ms/frame (6.4×), sampling bit-exact |
| RTF kernel end-to-end | ADR-030 | ~1.0 (0.97–1.09), ASR 3/3 perfetto |
| MTP 15 codebook residui | ADR-025 | 15.5ms/frame (graph, misurato nel kernel) |
| KV-cache persistente per sessione | VoiceChat-TTS Zero-Reset | TTFA → quasi 0 al 2° turno |
| Micro-batching decode multi-sessione | DwarfStar ds4-server | scalare a 2-4 sessioni su 24GB |
| ARIA pacing per-request | ADR-023 | budget frame esatto |
| STT upstream encoder audio Gemma4 | ADR-004 | niente Whisper |

### Checkpoint e moduli (già pronti)
- `v9_prefix_final.pt` (V9 projector 211.83M) + `custom_duplex_heads.pt` (causal_conv)
- `turn_dynamics_v1_final.pt` (TurnDetectorHead in_dim 2304)
- Moduli kernel: `duplex_kernel/` (engine, models/{thinker,talker,v9}, kv_cache,
  turnhead, subtalker_graph ADR-030, session, server, launch, benchmark)
- Loop esatto di riferimento: `tools/interactive_gemma4_omni_live.py` (eager),
  `tools/stream_full_conversation_e2e.py`

### Feature preservate (interfaccia OpenAI-compat)
- Chat text / text+audio (modalities) / streaming SSE
- Tool/function calling (chat template Gemma4 + parser)
- Voci default (gemma/leonardo), voice_style, ref_audio cloning
- Full-duplex WebSocket Realtime (audio in → audio out continuo, auto-commit
  su YIELD, barge-in via turnhead)
- ARIA pacing, speaker anchor (stabilità timbrica)

### Rischio / mitigazione
| Rischio | Mitigazione |
|---|---|
| Multi-tenant limitato (2-4 sessioni su 24GB) | Micro-batching + KV compressa (DwarfStar pattern); scalare a 2 GPU se serve |
| Manutenzione manuale del loop | Riuso integrale del loop validato; test vectors + QA (DwarfStar pattern) |
| Tool calling senza vLLM | Chat template Gemma4 (già nel tokenizer) + parser locale |
| Riuso del lavoro vLLM perso | ADR-028 resta valido; i processors/plugin restano come riferimento |

## Prossimi passi (stato 2026-08-29)

**Serving feature-complete**: il contratto funzionale completo è in
[`docs/SPEC_DUPLEX_KERNEL_SERVING.md`](../../docs/SPEC_DUPLEX_KERNEL_SERVING.md)
(7 sezioni: I/O testo, output audio, conditioning voce a 4 capability, full-duplex
WS con TurnHead+KV persistente, tool calling, parità ADR-028, priorità).

1. ✅ Scaffolding `duplex_kernel/` + smoke test + benchmark (RTF ~1.0, ASR 3/3).
2. ⚠️ Conditioning voce risolto in `duplex_kernel/voices.py`; consumo nel prompt nativo Qwen ancora da completare senza bypassare V9.
3. ✅ Streaming SSE: delta testo + chunk audio 640ms.
4. ✅ Sessioni HTTP riusabili con rolling history; reset sicuro su idle/barge-in. ⏳ Prefix KV nel prefill.
5. ✅ TurnHead alimentata con h_t V9 e mel fallback; stop event su barge-in.
6. ✅ Tool calling locale con parser JSON bilanciato.
7. ⏳ CUDA graph talker backbone (33→~12ms) → target RTF ~0.5.

## Conseguenze
- Positivo: RTF ~1.0 dimostrato (vs ~2.0 del serving vLLM-Omni), load ~36s (vs
  ~4.5min), controllo totale del duplex, zero dipendenza da API interne vLLM-Omni
  in evoluzione.
- Negativo: niente scaling arbitrario multi-tenant; il team mantiene il loop.
- vLLM-Omni resta come runtime standard-compliant di riferimento.
