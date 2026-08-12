# Gemma 4 Full-Duplex Omni Pipeline

High-performance, dual-channel, real-time Full-Duplex Speech-to-Speech conversational AI pipeline based on **Gemma 4 E2B** and **Qwen3-TTS Code2Wav**.

## 🌟 Architecture Overview

This project implements an end-to-end Full-Duplex architecture combining:
- **Speech Perception (Listening)**: Native 305M USM-Conformer Audio Encoder from **Gemma 4 E2B** (`unsloth/gemma-4-E2B-it`), taking log-mel (100 fps) input — **no VQ tokens in input**, coherent with the official E2B design (Report §2.2).
- **Self-Listening (auto-ascolto)**: the input mel stream **interleaves the agent's own channel** (`[user 8 mel, agent 8 mel, ...]` per 80 ms frame, agent mel delayed by 1 frame). The tower produces 4 features per agent frame (2 user + 2 agent); the LLM's causal attention gives each frame the context `[user up to t, own speech up to t-1]` — the model "hears itself talking", which makes **turn-taking and barge-in native** (validated by the community release `dockhardman/gemma-4-E2B-duplex`: "barge-in handled natively by the model").
- **LLM Backbone & Adapters**: Unsloth 4-bit QLoRA fine-tuning (audio encoder frozen, per design) with 1D Causal Convolution (`kernel_size=3, padding=0` with circular buffer state) for streaming non-causal-leakage frame adaptation.
- **Speech Synthesis (Speaking)**: **Coupling Qwen3-TTS (ADR-005) — unico percorso, MTP legacy rimosso**: il sub-talker `Qwen3-TTS-12Hz-1.7B-Base` CONGELATO (replica torch pura di `talker.code_predictor`, ~363MB) genera i 16 codebook: Gemma produce `h_t` (adapter trainabile 3072→2048) + cb0 (semantic_head vocab 4096, clampato a 2048 in decodifica) e il sub-talker genera i codebook acustici 1..15 in catena RVQ a **qualità nativa Qwen** (5M ore di training), decodificati in 24kHz PCM dal **Qwen3-TTS Code2Wav ConvNet Decoder**. La loss di coupling è CE diretta (allineamento RVQ, niente shift) scalata `COUPLING_LOSS_WEIGHT=0.6`, con warm-up `COUPLING_WARMUP_STEPS=200` (h_t detached a monte dell'adapter: solo l'adapter impara, poi il gradiente fluisce anche alla LoRA di Gemma).
- **Frame-level importance weighting** (recipe from the same community release): the "act now" signal (turn opening, barge-in, backchannel) is a tiny fraction of the supervised frames and gets drowned by `<wait>` frames — the loss is amplified on decisive frames (`FRAME_WEIGHT_OPEN=5`, `FRAME_WEIGHT_YIELD=3`, `FRAME_WEIGHT_CO_TALK=3`), with empirical per-codebook silence-code deweighting (`SILENCE_LOSS_WEIGHT=0.2`).

```
[User Mel (Ch 0)] ──┐
                    ├──> Native Conformer (frozen) ──> [user feat, agent feat] ──> LLM ──> Causal Conv1d ──> h_t (adapter) ──┐
[Agent Mel (Ch 1, self-listening, delayed)] ──┘                                                                          ├──> [Sub-Talker Qwen congelato] ──> Code2Wav ──> [Agent Audio]
                                                              Causal Conv1d ──> cb0 (semantic_head) ─────────────────────────┘   (coupling, ADR-005 — unico percorso)
```

## 📚 Architecture Decision Records (ADRs)

Key architectural decisions are formally recorded under `docs/decisions/`:
- [ADR-001: Multi-Token Prediction (MTP) with Qwen3-TTS Code2Wav](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-001-mtp-qwen3-tts.md)
- [ADR-002: 1D Causal Convolution with Circular Buffer for Streaming](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-002-causal-conv1d-circular-buffer.md)
- [ADR-003: Discoupled 2-Phase Kaggle Pipeline Architecture](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-003-discoupled-2phase-kaggle-pipeline.md)
- [ADR-004: Option A - Native Gemma E2B Conformer Input + Qwen3-TTS Output](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-004-option-a-gemma4-e2b-native-input.md)
- [ADR-005: Qwen3-TTS Teacher Selection (Base Variant) for Sub-Talker Coupling](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md)

See [docs/ADRS.md](file:///c:/Users/andre/Documents/gemma-4-omni/docs/ADRS.md) for the complete index.

## 🚀 Execution on Kaggle (2-Phase Architecture)

To run the complete pipeline on Kaggle (2x T4 GPU):

### Phase 1: Preprocessing & Parquet Tokenization
Run cells 1 to 3 in `kaggle_pipeline.py`. Tokenizes `DuplexConv` + `Mozilla Data Collective Italian` into:
- **mel utente + mel agente** (log-mel 100 fps, `*_mel.parquet` + `*_ameld.parquet`) — inputs del percorso nativo (con auto-ascolto). L'estrattore mel è una **replica numpy pura del `Gemma4AudioFeatureExtractor`** (bit-pari, verificata: max diff ~2e-7): funziona anche con `transformers==4.49.0` (necessario per qwen-tts) e non dipende dalla versione. NOTA: DuplexConv è un webdataset con campo `wav` stereo `(T,2)` 48kHz (canale 0 = utente / canale 1 = agente); a seconda della versione di `datasets` il campo arriva come **`torchcodec.decoders.AudioDecoder`** (oggetto con `get_all_samples()`), feature decodificata `{array, sampling_rate}`, o byte WAV grezzi `{path, bytes}` → decode multi-forma `_decode_duplex_audio` (mai il `__getitem__` del wrapper datasets, che fa np.mean sui canali e li miscela); verificato su dati reali in streaming. `encode(waveform, sr=48000)` del Qwen3-TTS tokenizer è supportato (resample interno 48k→24k, verificato sulla sorgente).
- **MDC Italiano** = Common Voice Scripted Speech 26.0 `it` (dataset MDC `cmqini14100vmnq07309ocknr`, ~10GB tar.gz): il loader rinomina le colonne (`audio_path`/`transcription`/`speaker_id`) e gli split `invalidated`/`other` vengono esclusi. I **clip mp3 reali (mono 32kHz) vengono letti e normalizzati** per costruire la traccia stereo con turni alternati + barge-in simulato — niente più rumore bianco sintetico. Dopo l'uso, tar (~10GB) e wav temporaneo (~1.3GB) vengono eliminati per rispettare il disco Kaggle. Serve il secret Kaggle `MDC_API_KEY` (MDC) oltre a `HF_TOKEN`.
- **16 RVQ codebooks agente** (`*_codes.parquet`, Qwen3-TTS 12Hz reale — `encode(waveform, sr=...)`, fix "must provide sr")
- **Testo risposta agente** (`*_text.parquet`): la trascrizione del turno agente MDC (colonna `sentence` dei clip reali) tokenizzata in BPE Gemma 4 E2B (libreria `tokenizers` su `tokenizer.json`, nessuna dipendenza da transformers≥5). **DuplexConv HA le trascrizioni** (archivi `{Edu,edu,none_Edu}/jsons.tar.gz` separati dagli audio, ~230MB): `asr[1][i].labels.txt` per canale con timestamp millisecondo (mappatura traccia↔canale verificata sul segnale: track 0 → canale 0 utente, track 1 → canale 1 agente) → testo allineato per finestra di 64 frame.
- **Disco Kaggle**: `/kaggle/working` ha quota ~20GB; i download pesanti (tar MDC ~10GB + clip ~10GB, jsons DuplexConv ~1GB, wav temporaneo) vanno sul **disco grande** (`KAGGLE_BIG_DISK` env → `/root/kaggle_big` su Kaggle → `data/` in locale); i parquet finali restano in working per la Fase 2.
- **Confini di segmento** (`SEGMENT_ENDS` nel sidecar): fine di ogni conversazione DuplexConv e di ogni turno MDC → base dello split di validazione.
- sidecar `*_hyperparams.json` (codebook 16/4096/2048, codici di silenzio **empirici** estratti da una waveform di silenzio — il tokenizer 12Hz non ha un token di silenzio esplicito; `TEXT_BLOCK_LEN`, `VAL_FRACTION`)

```python
execute_phase_1()
```

### Phase 2: Unsloth Training & Synthesis
Restart Kaggle Kernel. Run cells 4 to 7 in `kaggle_pipeline.py`. Upgrades environment to `transformers>=5.x` and Unsloth, loads the Parquet caches, trains LoRA adapters (finestra 64 frame agente = 5.12s; input mel interleavato 1024×128 → 256 feature tower → 256 placeholder audio; loss discouplata con pesatura per frame e silenzio de-pesato), and saves the final checkpoint.

**Supervisione testuale** (sequenza `[bos, TESTO risposta agente, boa, feat×256, eoa]`): il testo BPE precede le feature audio, quindi causalmente i codici audio attendono alla risposta (direzione nativa E2B testo→audio). La loss testo è una cross-entropy next-token calcolata con la **lm_head condivisa del base** (tied embeddings) sulle posizioni del blocco testo; scalata `TEXT_LOSS_WEIGHT=0.5`. Finestre senza testo (DuplexConv) → blocco pad mascherato, nessuna loss.

**Validation split**: l'ultimo `VAL_FRACTION=10%` dei frame viene tenuto fuori dal train a un **confine di segmento** (`SEGMENT_ENDS`): val_loss riportata per epoca senza gradiente.

```python
duplex_model, _ = load_gemma4_duplex_system()
run_training_pipeline(duplex_model, num_epochs=1)
```

**Diagnostica di qualità**: la loss iniziale attesa è ~13–17 (log 4096 + log 2048), **non** ~0.002. Se compare la loss collassata, i dati sono degenerati (fallback FFT/sintetici) → rigenerare la Fase 1.

### Training (unico percorso: coupling Qwen3-TTS obbligatorio)

Il sub-talker è **obbligatorio** (MTP legacy rimosso): senza la dir `qwen_subtalker/` il caricamento fallisce con un errore chiaro che indica di eseguire l'estrattore.

**In Fase 1** (dopo `execute_phase_1()`, stesso kernel):
```python
# 1) Aggiungi il secret Kaggle HF_TOKEN (repo Qwen3-TTS-12Hz-1.7B-Base è GATED)
# 2) Esegui l'estrattore: scarica SOLO i ~363MB del sub-talker su /kaggle/working
!python extract_qwen_subtalker.py --out-dir qwen_subtalker
# (Range HTTP parziale: 89 tensori su 479, senza scaricare i 3.8GB del checkpoint)
```
Il decoder Code2Wav viene invece estratto **automaticamente** da `execute_phase_1()` in `code2wav_weights/` (271 tensori): in Fase 2 (transformers 5.x) il decoder ufficiale non è caricabile, quindi viene usata la **replica torch pura** `code2wav_decoder.py` (bit-exact verificata vs il decoder ufficiale: MAXDIFF=0.0 su T=5..80).

**In Fase 2** (kernel riavviato):
```python
duplex_model, _ = load_gemma4_duplex_system()   # coupling OBBLIGATORIO (fail-fast se manca la dir)
run_training_pipeline(duplex_model, num_epochs=1)
```
- Loss = cb0 (semantica) + `COUPLING_LOSS_WEIGHT ×` coupling (CE diretta sui codebook 1..15 del sub-talker) + testo.
- Warm-up: nei primi 200 step `h_t` è detached a monte dell'adapter → solo l'adapter impara a mappare Gemma congelata; poi il gradiente fluisce anche alla LoRA.
- A inferenza `generate` usa `sample_acoustic_codes` (catena RVQ a 15 passi del sub-talker); il checkpoint salva l'adapter (`ht_adapter`), NON il sub-talker (congelato, si ricarica dalla dir).

**Performance di inferenza** (misurata su CPU locale fp32, stima T4 bf16 ≈ 30× CPU-fp32 su matmul):

| Percorso | Sub-talker | Decoder Code2Wav | RTF / margine (T4 stimato) |
|---|---|---|---|
| `generate` batch (T=64, 5.12s audio) | ~9.4s CPU / **~0.35s T4** | ~2.0s CPU / ~0.1s T4 | RTF ~0.07 → real-time con margine ~15× |
| `stream_speech` frame-by-frame | ~0.69s CPU per frame / **~23ms T4 per frame** | (su chunk) | 80ms/frame disponibili → margine ~3× su T4 |

Il sub-talker è il costo dominante dopo Gemma (il vecchio MTP costava ~65ms CPU, ~145× meno), ma resta ampiamente sotto il real-time nel percorso batch; nello streaming frame-by-frame il margine scende a ~3× su T4 → ottimizzazione naturale: **KV-cache nel sub-talker** (1 posizione per passo invece di 16 con junk) o `torch.compile`.

### Sintesi & demo (ascolto della qualità)

A fine training vengono generati wav ascoltabili in `samples/` (finestre di validazione): `generated_*.wav` (modello) vs `oracle_*.wav` (codici GT decodificati) per isolare degradazione del modello vs decoder, più un `README.txt` con i testi attesi. Demo interattiva (il kernel Kaggle non ha microfono → wav caricato):
```python
demo_talk(duplex_model)                          # wav utente -> risposta agente
# oppure:  load_gemma4_duplex_system() -> load_gemma4_duplex_checkpoint(m) -> demo_talk(m)
```

## 🖥️ Inferenza fuori Kaggle (serving)

Il kernel Kaggle è effimero: per servire il modello addestrato serve una macchina con GPU e `serve_duplex.py` (riusa gli stessi loader verificati):

```bash
pip install unsloth transformers accelerate bitsandbytes fastapi uvicorn python-multipart soundfile numpy pandas torch safetensors

export HF_TOKEN=<token>   # repo gated (google/gemma-4-E2B-it, Qwen3-TTS)

# Risposta singola a un wav utente
python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota --demo parlato.wav -o risposta.wav

# Server REST (OpenAI-ish minimal): POST /talk (multipart wav) -> wav 24kHz risposta agente
python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota --serve --port 8000

# App Gradio opzionale (microfono nel browser)
python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota --gradio
```

### Perché non posso caricarlo "così com'è" in vLLM-Omni

Il checkpoint NON è un modello HF standard: è multi-componente (LoRA su Gemma 4 E2B + `causal_conv`/`semantic_head`/`ht_adapter` custom + sub-talker Qwen congelato + decoder Code2Wav). vLLM carica solo architetture registrate → `--model <checkpoint>` fallirebbe. La buona notizia (verificato su docs live): il nostro pattern è ESATTAMENTE quello di vLLM-Omni (Thinker → Talker → Code2Wav, come Qwen2.5-Omni/Qwen3-TTS):

| Componente nostro | Stato in vLLM/vLLM-Omni | Lavoro per il porting |
|---|---|---|
| Gemma 4 E2B (LoRA) | ✅ vLLM core: `Gemma4ForConditionalGeneration` supporta `google/gemma-4-E2B-it` (testo+immagini+video+audio) | LoRA: `--lora-modules` nativo (export dell'adapter Unsloth in formato HF) |
| `causal_conv` + `semantic_head` + `ht_adapter` | ❌ custom | wrapper `Gemma4DuplexForConditionalGeneration` in vLLM-Omni: aggiunge gli head al thinker e inietta il mel; emette `h_t`+`cb0` |
| Sub-talker (code predictor AR) | ✅ vLLM-Omni ha il code predictor di Qwen3-TTS (stessa architettura RVQ a catena) | stage AR custom: il nostro forward (`forward_sub_talker_finetune`) va registrato come talker stage con input `h_t` invece del testo |
| Decoder Code2Wav | ✅ vLLM-Omni ha già lo stage Code2Wav di Qwen3-TTS (e la nostra replica è bit-exact) | **nessuno**: si riusa lo stage esistente |

Schema del modello multi-stadio da registrare (`docs/contributing/model/adding_omni_model.md`):
```
Stage 0 (Thinker):  mel utente+agente -> Gemma4ForCausalLM (LoRA) -> hidden -> causal_conv -> h_t (adapter) + cb0
Stage 1 (Talker):   h_t + cb0 -> sub-talker Qwen (congelato) -> codebook RVQ 1..15      [async_chunk, chunk=25]
Stage 2 (Code2Wav): codebook -> waveform 24kHz                                          [già in vLLM-Omni]
```
Stima onesta: porting ~1-2 settimane di lavoro dedicato (implementazione vLLM + test di parità con i nostri selftest). **Finché non serve la concorrenza/throughput di vLLM, `serve_duplex.py` copre il serving.**

## 🗣️ Varianti Qwen3-TTS e scelta del teacher (verificato)

Le tre varianti 1.7B (stessa architettura, `tts_model_type` diverso) differiscono per il **condizionamento della voce** (verificato su config.json, model card e sorgente qwen-tts 0.1.1 `inference/qwen3_tts_model.py`):

| Variante | `tts_model_type` | Condizionamento voce | speaker_encoder nel checkpoint |
|---|---|---|---|
| **Base** | `base` | **Sample audio richiesto** per la clonazione (`ref_audio` è Required in `generate_voice_clone`; `extract_speaker_embedding` → `ref_spk_embedding`); nessun path di default senza sample | ✅ 74 tensori (480 tot) |
| **CustomVoice** | `custom_voice` | **Speaker ID predefinito** (`generate_custom_voice(text, speaker=...)`): 9 timbri (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) — nessuno con italiano nativo (card: "recommend using each speaker's native language"); ogni voce parla comunque le 10 lingue | ❌ (405 tot) |
| **VoiceDesign** | `voice_design` | **Solo prompt testuale** (`generate_voice_design(text, instruct=...)` descrizione naturale della voce/stile) | ❌ (405 tot) |

**Implicazione per il coupling (teacher = `Qwen3-TTS-12Hz-1.7B-Base`)**: estraiamo **solo** il sub-talker (`talker.code_predictor.*` + `talker.model.codec_embedding`) — NON estraiamo lo `speaker_encoder` né il main talker. L'input del sub-talker è `[h_t, codici]`: la voce non arriva da un sample, ma da **h_t di Gemma** (adapter trainabile). Il requisito di "sample vocale per la clonazione" vale per il modello completo, **non** per il nostro percorso di coupling: a inferenza nessun sample è necessario (la voce sarà quella che Gemma impara a codificare in h_t, inizialmente neutra).

Perché **Base** e non le altre (verificato): (1) la model card dice che Base "can be used for fine-tuning (FT) other models" — è esattamente il nostro uso (Gemma fine-tunata che guida il sub-talker); (2) i pesi del sub-talker **differiscono tra varianti** (hash di `code_predictor.lm_head.0.weight` diverso Base vs CustomVoice): le varianti di controllo post-addestrano il talker per i loro sistemi speaker specifici, che bypassiamo; (3) CustomVoice non ha voci native italiane (le preset sono cinese/inglese/giapponese/coreano).

**Test di tetto Qwen (PoC cascata)**: con un sample vocale italiano a disposizione → `generate_voice_clone` su Base (qualità italiana migliore); senza sample → `generate_custom_voice(text, speaker="Serena")` su CustomVoice (ascolto senza campione, timbro non italiano nativo).

**Decisione formale registrata in [ADR-005](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md)**: scelta della variante, scopo dell'estrazione, aspettativa di qualità onesta (tetto = rendering Qwen congelato, raggiungibilità dipende dalla convergenza di `h_t`, validata all'ascolto), piano di validazione e knob di contingenza (adapter più capiente → LoRA leggera sul sub-talker → condizionamento voce esplicito).

## 🧪 Verification & Testing

Verify local Python syntax:
```bash
python -m py_compile kaggle_pipeline.py
```
