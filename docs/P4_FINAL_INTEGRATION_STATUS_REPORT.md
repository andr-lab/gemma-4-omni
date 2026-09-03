# P4 Final Integration Status Report: True 80 ms Full-Duplex Kernel

**Data di attestazione:** 03 Settembre 2026  
**Piattaforma:** NVIDIA GeForce RTX 3090 (24 GB VRAM), Ubuntu 24.04 via Docker (`gemma4-omni-v5:latest`)  
**Repository Head Base:** `2368151398f9684fc4739ebe2a69cf6cc4efc276`  
**Stato Attuale:** **P4 FORMALLY AUDITED — PROOFS 1 & 2 CLOSED, PROOF 3 PHYSICAL BOTTLENECK CHARACTERIZED**

---

## 1. Sommario Esecutivo

In conformità ai requisiti vincolanti del task P4 Final Closure, la pipeline è stata sottoposta a verifica rigorosa su hardware reale senza modificare i pesi dei modelli (Gemma 4 E2B, V9 Causal, Qwen3-TTS 1.7B, SubTalker, Code2Wav) né indebolire alcuna soglia temporale o causale.

### Stato Sintetico dei Gate

| Gate | Descrizione | Requisito | Risultato Misurato | Esito |
|---|---|---|---|:---:|
| **§8 / Proof 1** | Frontend Oracle Parity (5000 frame / 400s) | Mel MAE $= 0.0$, CosMin $\ge 0.9999$ | MAE = **0.00000000**, CosMin = **0.99999994** | **PASS** |
| **§15** | Cryptographic Source Manifest Binding | SHA-256 su tutti i file sorgente | 88 file censiti, digest crittografico salvato | **PASS** |
| **§2-§6 / Proof 2** | Thinker + Continuous Listener Concurrency | $p95 \le 80.0$ ms, miss $\le 1.0\%$, backlog $= 0$ | $p50 = \mathbf{68.45\text{ ms}}, p95 = \mathbf{72.17\text{ ms}}$, miss = $\mathbf{0.59\%}$, backlog = $\mathbf{0}$ | **PASS** |
| **§9 / Proof 3** | Simultaneous Listen + Speak (60s / 5min) | Listener $p95 \le 80$ ms, Speaker $\ge 12.5$ fps, miss $\le 1\%$ | Listener $p95 = 142.32$ ms, Speaker fps = $10.34$, miss = $97.39\%$ | **BLOCKED (Hardware Contention)** |

---

## 2. Dettaglio Risultati e Lavoro Svolto

### A. Frontend Oracle Parity Gate (§8) — PASS
- **Test eseguito**: `tools/kernel_gates/p4_frontend_parity.py` su 5.000 frame reali a 16 kHz (400.0 secondi di audio continuo).
- **Verifica**: Confronto bit-a-bit tra `GemmaAudioFeatureExtractor` nativo Google/Transformers e il kernel acustico locale.
- **Metriche**:
  - `frames_evaluated`: 5.000
  - `mel_mae_max`: **0.00000000**
  - `cosine_min`: **0.99999994**
  - `mismatch_count`: **0**
- **Artefatto**: `validation/kernel_gates/P4_FINAL/P4_FRONTEND_PARITY.json`.

---

### B. Proof 2: Thinker Prefill + Decode Concurrency (§2–§6) — FULL PASS

#### Problemi Individuati e Risolti:
1. **Pausa di ricompilazione Dynamo (100s freeze)**:  
   `transformers.StaticCache` veniva istanziato da zero a ogni chiamata di `engine.think()`, invalidando i guard di `torch.compile`. Risolto con `self._persistent_static_cache` e ripristino istantaneo `cache.reset()`, garantendo passi decode a **12.85 ms** stabili e zero ricompilazioni.
2. **Padding statico a 291 token in `ShapeStableGemmaListenerLM`**:  
   Tutti i 28 layer del listener eseguivano forward su 291 token anziché 4 token di streaming audio. Implementato dual workspace (`stream_length=4` per frame $\le 4$ e `target_length=291` per prefill). Latenza del forward LM scesa da **63.91 ms a 51.63 ms** (-12.28 ms/frame).
3. **Spike di avvio su Frame 0**:  
   Implementato `prime_steady_state()` su `GemmaAudioTowerBackend` e `GemmaListener` per pre-popolare il contesto testuale e allineare i contatori prima dell'inizio del benchmark.
4. **Drenaggio pulito della coda di ingresso**:  
   Arresto ordinato del thread produttore prima del scheduler per garantire backlog finale pari a 0.

#### Risultati Sperimentali su RTX 3090 (128 token generati, 511 frame audio, 40.8s):
- **Listener Complete Step Latency**:
  - $p50$: **68.45 ms**
  - $p95$: **72.17 ms** (Soglia: $\le 80.00\text{ ms}$ — **PASS**)
  - $p99$: **74.01 ms**
  - max: **76.82 ms**
- **Deadline Miss Rate**: **0.59%** (3 / 511 frame; Soglia: $\le 1.0\%$ — **PASS**).
- **Final Backlog**: **0** frame.
- **Salute Stream**: 0 dropped, 0 duplicate, 0 out-of-order.
- **Artefatto**: `validation/kernel_gates/P4_FINAL/P4_THINKER_LISTENER_CONCURRENCY.json`.

---

### C. Proof 3: Simultaneous Listener + Speaker Concurrency Gate (§9) — Analisi di Root-Cause

#### 1. Profiling Fisico dei Singoli Componenti su RTX 3090:
- **Listener Step** (Audio Tower CUDA Graph + Gemma LM + V9):
  - Audio Tower: **7.9 ms**
  - Gemma LM (`ShapeStableGemmaListenerLM`, $Q=4$): **51.8 ms**
  - Incremental V9: **2.7 ms**
  - **Totale Listener Step**: **62.4 ms**
- **Speaker Step** (FuturePrefixQwen L=0 + SubTalker 15 codebooks):
  - In eager mode:
    - Qwen 1.7B text/audio backbone: **35.3 ms**
    - SubTalker (15 CUDA graphs): **15.4 ms**
    - **Totale Speaker Step**: **50.7 ms**
  - Con `torch.compile` su Qwen backbone:
    - Qwen 1.7B compiled: **13.2 ms**
    - SubTalker: **15.4 ms** (12.9 ms greedy)
    - **Totale Speaker Step compilato**: **28.6 ms**

#### 2. Il Limite Fisico del Singolo GPU (RTX 3090):
Nel caso di esecuzione contemporanea al 100% per 60 secondi:
- Per ogni finestra di 80 ms, il budget fisico totale è di **80.0 millisecondi**.
- Somma del calcolo seriale minimo necessario:
  $$\text{Listener (62.4 ms)} + \text{Speaker (50.7 ms)} = \mathbf{113.1\text{ ms}} \quad (\mathbf{1.41\times \text{ del tempo fisico}})$$
  Con Qwen compilato:
  $$\text{Listener (62.4 ms)} + \text{Speaker (28.6 ms)} = \mathbf{91.0\text{ ms}} \quad (\mathbf{1.14\times \text{ del tempo fisico}})$$

#### 3. Comportamento sotto Esecuzione Parallela (CUDA Streams concorrenti):
Sia Gemma 4 E2B (2.6B parametri, 5.2 GB in BF16) che Qwen 1.7B (1.7B parametri, 3.4 GB in BF16) operano a batch size 1 e sequenza corta. Sono operazioni interamente limitate dalla banda di memoria (GEMV memory-bandwidth bound).
- Quando eseguiti simultaneamente su due stream CUDA separati (`listener_stream` e `speaker_stream`):
  - Entrambi saturano il bus GDDR6X da 936 GB/s.
  - La banda effettiva per ciascun modello si dimezza.
  - La latenza di Listener sale a **135.92 ms** ($p95 = 142.32\text{ ms}$).
  - La latenza di Speaker sale a **91.79 ms**.
  - Entrambi mancano sistematicamente la deadline degli 80 ms (miss rate $\approx 97 - 100\%$).

#### 4. Comportamento sotto Mutua Esclusione (Lock / Schedule seriale):
Se i due modelli vengono serializzati per evitare contesa di memoria:
- Finché Speaker non parla, Listener gira a 62.4 ms (0% miss rate).
- Quando Speaker deve generare 1 frame (28.6 ms o 50.7 ms) per ogni tick di 80 ms:
  - $62.4\text{ ms} + 28.6\text{ ms} = 91.0\text{ ms} > 80.0\text{ ms}$.
  - A ogni tick si accumulano 11 ms di ritardo, generando crescita del backlog dell'ingress queue.

---

## 3. Soluzioni Architetturali Possibili per la Chiusura

Per chiudere formalmente Proof 3 mantenendo la soglia rigorosa di $p95 \le 80\text{ ms}$ e $\ge 12.5\text{ fps}$, si delineano tre opzioni:

### Opzione A (Raccomandata per Produzione): Playout Buffering con Chunking
Nel parlato umano e nei sistemi duplex reali (es. WebRTC, VoIP, DwarfStar):
- Il Talker non sintetizza 1 frame alla volta esattamente a ogni 80 ms.
- Sintetizza in **chunk audio** (es. chunk da 8 frame = 640 ms, come già previsto da `engine.py:468`).
- Con Qwen compilato, 8 frame vengono generati in circa $8 \times 28.6 = 229\text{ ms}$.
- Durante i restanti $640 - 229 = 411\text{ ms}$ di riproduzione audio, il GPU è al **100% dedicato al Listener**.
- I frame audio riprodotti vengono immessi in `assistant_reference_buffer` esattamente a 12.5 Hz (ogni 80 ms).
- Risultato atteso: Throughput $= 12.5\text{ fps}$, deadline miss rate $= 0\%$, listener $p95 \le 80\text{ ms}$ per il 100% del tempo di riproduzione.

### Opzione B: Kernel Fusion su SubTalker + Listener
- SubTalker esegue 15 lanci sequenziali di CUDA graph per frame (15.4 ms). Un kernel Triton o C++ fuso potrebbe abbattere SubTalker a ~4-5 ms.
- Ottimizzazione di `ShapeStableGemmaListenerLM` per scendere da 51.8 ms a ~44 ms.
- In questo modo: $T_L (54\text{ ms}) + T_S (20\text{ ms}) = \mathbf{74\text{ ms}} \le 80.0\text{ ms}$, permettendo la serializzazione frame-per-frame su singolo GPU.

### Opzione C: Assegnazione Dual-GPU o Offload Playout
- Allocazione di Listener su GPU 0 e Talker su GPU 1, eliminando la contesa sul bus di memoria GDDR6X.

---

## 4. Conclusione

I vincoli di Proof 1 (Frontend Parity) e Proof 2 (Thinker + Listener Concurrency) sono **completamente e formalmente chiusi**.
Il blocco riscontrato in Proof 3 è di natura puramente fisica (saturazione della banda di memoria GDDR6X su RTX 3090 quando due modelli da 2.6B e 1.7B operano contemporaneamente a saturazione).
Si attende decisione dell'utente sulla strategia di scheduling (Opzione A, B o C) prima di procedere.
