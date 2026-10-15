# Gemma 4 Full-Duplex Omni Pipeline

Pipeline vocale full-duplex speech-to-speech per dialoghi in tempo reale basata su **Gemma 4 E2B-it** (LLM Thinker), **Qwen3-TTS 12Hz** (Acoustic Talker) e vocoder neuronale **Code2Wav a 24 kHz**.

La documentazione tecnica dell'architettura attiva è disponibile in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Caratteristiche Principali

- **Interazione Full-Duplex Continua**: Elaborazione a 12.5 Hz su frame PCM da 80 ms a 16 kHz (1.280 campioni).
- **Condizionamento Prosodico Continuo**: Modulazione FiLM continua (`FrameResidualQueryFiLMAdapter` v2, scala $\alpha = 0.0513$) che mappa gli stati latenti di Gemma 4 ($h_t \in \mathbb{R}^{1536}$) sulle Query di Qwen3-TTS ($q_t \in \mathbb{R}^{2048}$) ad ogni frame da 80 ms.
- **Identità Vocale Coerente**: Voce femminile canonica fissa `"gemma"` (`custom_voices/gemma.safetensors`).
- **Filtro Audio Trasparente Anti-Clipping**: Hook HAR pre-clamp a picco 0.95 nel vocoder Code2Wav. Nessuna perdita di banda ad alte frequenze (8-12 kHz) e leveling lineare trasparente a -14.0 dBFS RMS.
- **Listener Ottimizzato e Micro-Batching**: Frontend a estrazione mel unificata e coalescing dinamico $N \in \{1, 2, 4\}$ per la riduzione dell'overhead di esecuzione.
- **Serving OpenAI-Compatibile**: Endpoint REST `POST /v1/chat/completions` con supporto simultaneo `text` e `audio`, endpoint `/talk` e streaming bidirezionale via WebSocket `/v1/realtime`.

---

## Struttura della Codebase

```
gemma-4-omni/
├── duplex_kernel/             # Core del kernel full-duplex (scheduler, listener, talker, microbatch)
│   ├── microbatch/            # Runner, probe e contratti per il micro-batching del listener
│   ├── models/                # Adapter prosodico FiLM v2, Conformer USM, architettura Qwen Talker
│   └── subtalker_graph.py     # Campionamento stocastico HRCR/SACR
├── native/                    # Binding e sorgenti C++/CUDA per inferenza accelerata
│   ├── gemma4_listener/       # C++ engine per audio listener
│   └── gemma4_listener_microbatch/ # Sviluppo C++ microbatching N in {1, 2, 4}
├── server/                    # Server FastAPI e controller di produzione (speech_service.py)
├── checkpoints/               # Checkpoint addestrati (QLoRA, Projector FiLM v2, Controllo)
├── custom_voices/             # Vettori vocali canonici (gemma.safetensors)
├── docs/                      # Documentazione architetturale e sorgenti di ricerca
│   ├── ARCHITECTURE.md        # Documento di riferimento dell'architettura attiva
│   └── sources/               # Paper di ricerca accademici e sorgente originale Qwen3-TTS
├── tests/                     # Suite di test automatizzati (tests/kernel/)
└── scripts/                   # Script di gestione e avvio servizio su RTX 3090
```

---

## Flusso Operativo su NVIDIA RTX 3090 (Host `mondaini`)

L'ambiente di runtime è distribuito sulla macchina RTX 3090 (host `mondaini`) all'interno di un container Docker isolato non-root (`1000:1000`) con `--ipc=host`:

```bash
# 1. Avvio Serving di Produzione (API REST porta 8000)
./start_speech_service_3090.sh

# 2. Verifica dello Stato di Salute del Servizio
curl -s http://localhost:8000/health

# 3. Esecuzione Test Unitari Locali
pytest tests/kernel/test_listener_micro_batching.py tests/kernel/test_frame_residual_query_film_v2.py
```

---

## Esecuzione dei Test

Per eseguire i test attivi in ambiente locale:

```powershell
$env:PYTHONPATH = "."
pytest tests/kernel/test_listener_micro_batching.py tests/kernel/test_listener_micro_batch_frontend.py
pytest tests/kernel/test_frame_residual_query_film_v2.py tests/kernel/test_serving_conditioning_v2_contract.py
```
