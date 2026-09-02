# P4 Shared-Weight Listener Report — Risoluzione Blocker Runtime & Gate Parity

**Data:** 2026-09-02  
**Target:** Host RTX 3090 Linux (`gpu3090`, 24GB VRAM, CUDA 12.4, Docker `gemma4-omni-v5:latest`)  
**Commit/Head:** `7989c59` + modifiche kernel condivise  
**Verdetto Finale:** **A) SHARED-WEIGHT LISTENER PASS**  
*(Parity 100×64 PASS, Latenza Steady-State p95 <= 80 ms PASS, Concurrency Isolation PASS, Weight Sharing Zero-Copy PASS)*

---

## 1. Verdetto Esecutivo: OPZIONE A (PASS)

Il candidato diagnostico `shape_padded_all_ops_kv` è stato convertito con successo in un runtime di produzione permanente, modulare e ad isolamento totale: `ShapeStableGemmaListenerLM` (in [duplex_kernel/shape_stable_listener.py](file:///c:/Users/andre/Documents/gemma-4-omni/duplex_kernel/shape_stable_listener.py)), integrato in [duplex_kernel/listener.py](file:///c:/Users/andre/Documents/gemma-4-omni/duplex_kernel/listener.py).

Tutte le 5 prove obbligatorie definite nello steering sono state eseguite su GPU NVIDIA GeForce RTX 3090 e risultano **pienamente superate**:

| Prova | Requisito / Soglia | Risultato Misurato | Esito |
|---|---|---|:---:|
| **1. Weight Sharing** | Zero duplicazione pesi, delta alloc/res = 0 | 0 parametri creati, $\Delta$ alloc = 0 B, $\Delta$ res = 0 B | **PASS** |
| **2. Parity Numerica** | 100 seq × 64 frame (6400 frame), MAE $\le 10^{-3}$, Cos $\ge 0.9999$ | MAE = **0.0**, Max Abs = **0.0**, Cos min = **0.99999994** | **PASS** |
| **3. Latenza Steady-State** | p95 $\le 80.0$ ms (senza patch/toggle a ogni frame) | p50 = **74.46 ms**, p95 = **79.79 ms**, p99 = **81.03 ms** | **PASS** |
| **4. Concurrency Isolation** | Listener 12.5 Hz + Thinker Generation + ASR concorrenti | Text logit hash invariato, ASR output identico, MAE = **0.0** | **PASS** |
| **5. Source Isolation** | Zero mutazioni su `module.forward` o registry Transformers | 0 `module.forward =`, 0 modifiche registry globali | **PASS** |

La soluzione multiprocess con replica del modello Gemma è pertanto **scartata**: l'architettura single-model a pesi condivisi e KV-cache disaccoppiata rispetta pienamente tutti i vincoli funzionali, temporali e di memoria.

---

## 2. Architettura del Runtime: Shared-Weight & Session-Owned State

Il runtime implementa la separazione strutturale tra pesi del Thinker e stato di ascolto:

```text
                     Gemma 4 Weights (Frozen / Shared Read-Only)
                     [565 named parameters, ~9.54 GiB residenti]
                                      │
               ┌──────────────────────┴──────────────────────┐
               │                                             │
      Native Thinker Path                          ShapeStableGemmaListenerLM
   (Generation / Transcription)                      (Continuous 12.5 Hz Listener)
   Gemma4Model.forward standard                    Functional step-by-step view
   Inference / SDPA standard                      Padded GEMMs & RMSNorms statici
               │                                             │
      Thinker DynamicCache                         Listener-Owned DynamicCache
      (Session Generation)                         (Rolling 64-frame Context)
```

### Proprietà di Implementazione:
1. **Nessun parametro `nn.Parameter` di proprietà**: `ShapeStableGemmaListenerLM` è un controller funzionale. Tutti i riferimenti puntano direttamente ai tensori dei pesi originali (`parameter.data_ptr()` identici).
2. **Static Padding Workspace (`_StaticPaddingWorkspace`)**: Alloca all'avvio buffer statici dedicati (dimensione target 291 token) per le proiezioni lineari, RMSNorm, MLP e SDPA K/V. Evita riallocazioni continue e mantiene la forma dei GEMM stabile a runtime. Memoria statica occupata: soli **22.5 MB** (~21.46 MiB).
3. **Session-Owned State**: La `DynamicCache`, le maschere 4D e gli shared-KV layers Gemma appartengono alla `GemmaListenerState` della singola sessione; nessun contesto travasa o inquina il Thinker principale.

---

## 3. Risultati Dettagliati delle 5 Prove

### Prova 1: Dimostrazione di Weight Sharing (Zero Duplicazione)

Eseguita tramite [tools/kernel_gates/shared_weight_listener_runtime.py](file:///c:/Users/andre/Documents/gemma-4-omni/tools/kernel_gates/shared_weight_listener_runtime.py):
- **Parametri creati dal listener**: `0`
- **Parametri sorgente verificati**: `565` (SHA-256 dei data pointer: `6b12d6ae881eaf9ca02bb6f9181379d586081205c357bd549879a3fe71a302ae`)
- **Storage condiviso al 100%**: `shared_weight_storage: true`
- **Memoria GPU prima dell'inizializzazione listener view**:
  - Allocated: `10,246,877,696 B` (~9.54 GiB)
  - Reserved: `10,380,902,400 B` (~9.67 GiB)
- **Memoria GPU dopo l'inizializzazione listener view**:
  - Allocated: `10,246,877,696 B`
  - Reserved: `10,380,902,400 B`
- **Delta Allocated**: **0 Bytes**
- **Delta Reserved**: **0 Bytes**
- **Buffer di workspace statici**: `22,508,850 B` (~21.46 MiB)

*Conclusione:* Nessuna copia del modello Gemma viene istanziata.

---

### Prova 2: Parity Numerica con Oracle Nativo BF16 (100 seq × 64 frame)

Esecuzione completa sul dataset reale:
`/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet`  
- Offset di validazione: frame `1,650,688` (finestra testo `25,792`, 100 sequenze consecutive da 64 frame = 6.400 frame totali).
- Oracle di riferimento: forward completo nativo e intatto `Gemma4Model.forward` in BF16.
- Candidate: `GemmaAudioTowerBackend` con `ShapeStableGemmaListenerLM` permanente.

Evidence registrata in:
`validation/kernel_gates/P4_SHARED_WEIGHT_LISTENER_RUNTIME_FINAL.json`

| Metrica | Risultato | Soglia di Gate | Esito |
|---|---:|---:|:---:|
| **Sequenze Valutate** | 100 | $\ge 100$ | PASS |
| **Frame Totali** | 6.400 | $\ge 5.000$ | PASS |
| **Mean Absolute Error (MAE)** | **0.0** | $\le 10^{-3}$ | **PASS** |
| **Max Absolute Error** | **0.0** | diagnostica | **PASS** |
| **Cosine Similarity Minima** | **0.99999994** | $\ge 0.9999$ | **PASS** |
| **Valori Non-Finiti (NaN/Inf)** | **0** | $0$ | PASS |
| **Shape Mismatch** | **0** | $0$ | PASS |

*Conclusione:* Parity perfetta ad errore zero riprodotta sul runtime permanente.

---

### Prova 3: Latenza & Telemetria Steady-State (senza toggle/patch)

Misurata su 6.300 sample steady-state (dopo warmup di 2 sequenze complete e prefill prefisso):
- **Steady-State Audio p50**: **74.46 ms** (centra l'engineering target di 75 ms)
- **Steady-State Audio p95**: **79.79 ms** (rispetta il limite rigido di 80.0 ms)
- **Steady-State Audio p99**: **81.03 ms**
- **Wall-clock time per 6.400 frame**: 503.17 s (~8.38 minuti)
- **Peak VRAM Allocated**: 10.37 GB
- **Peak VRAM Reserved**: 10.45 GB
- **Margine VRAM disponibile su RTX 3090 (24 GB)**: **> 13.5 GB liberi** (più che sufficienti per alloggiare stabilmente Qwen Talker frozen, SubTalker e Code2Wav vocoder).

---

### Prova 4: Concurrency Isolation (Listener 12.5 Hz + Thinker + ASR + Context Update)

Eseguita tramite [tools/kernel_gates/shared_weight_listener_concurrency.py](file:///c:/Users/andre/Documents/gemma-4-omni/tools/kernel_gates/shared_weight_listener_concurrency.py).  
Evidence registrata in:
`validation/kernel_gates/P4_SHARED_WEIGHT_LISTENER_CONCURRENCY.json`

Setup:
- Thread background esegue lo stream listener continuo a 12.5 Hz (64 frame) consumando la view shape-stable e aggiornando la propria `DynamicCache`.
- Contemporaneamente, il thread principale invoca:
  1. Forward testuale del Thinker su input testuale (`thinker.model(...)`).
  2. Trascrizione nativa debug ASR (`thinker.transcribe_audio(...)`).
  3. Cambio di contesto conversazionale nel listener (`text_ids` aggiornati).

Risultati:
- **Thinker Text Forward**: Logit SHA-256 identico bit-a-bit prima e durante l'esecuzione del listener (`text_forward_unchanged: true`).
- **Thinker Debug ASR**: Testo trascritto `"Mi"` e token generati `[31213]` identici al 100% (`debug_asr_unchanged: true`).
- **Listener Hidden durante concorrenza**: MAE = **0.0**, Max Abs = **0.0**, Cosine = **0.999992**, Shape `[1, 64, 1536]` identica.
- **Aggiornamento di Contesto**: La cache della sessione viene invalidata e ricostruita correttamente all'arrivo di nuovi turni testuali senza causare residui stale (`context_update_rebuilt_listener_cache: true`).
- **Interferenze / Deadlock**: Zero eccezioni, zero deadlock, zero corruzione dello stato.

---

### Prova 5: Source Isolation & Contratti Software

Eseguita tramite la suite di test TDD [tests/kernel/test_shape_stable_listener.py](file:///c:/Users/andre/Documents/gemma-4-omni/tests/kernel/test_shape_stable_listener.py):
1. Verifica statica AST/sorgenti:
   - Assenza totale di assegnazioni a `module.forward =` nei file di produzione (`listener.py`, `shape_stable_listener.py`).
   - Assenza totale di mutazioni a `ALL_ATTENTION_FUNCTIONS[...]` o a registry globali di Transformers.
2. Verifica isolamento dei buffer simultanei K e V nel workspace:
   - `padded_key.data_ptr() != padded_value.data_ptr()`, prevenendo race condition o sovrascritture in-place durante il calcolo SDPA.
3. Verifica resilienza su errori e frame non-finiti in [duplex_kernel/listener.py](file:///c:/Users/andre/Documents/gemma-4-omni/duplex_kernel/listener.py):
   - In caso di eccezione o hidden non-finito, la cache del listener LM viene invalidata immediatamente (`state.invalidate_gemma_lm_state()`) prevenendo il riutilizzo di stati parziali.

La suite locale `pytest tests/kernel/` conta **54 test superati al 100% VERDE** (in 8.48s).

---

## 4. Stato dell'Integrazione nel Runtime Principale

L'integrazione del nuovo runtime è stata completata in [duplex_kernel/listener.py](file:///c:/Users/andre/Documents/gemma-4-omni/duplex_kernel/listener.py):
1. `GemmaAudioTowerBackend.__init__`: Inizializza `self._shape_stable_lm = ShapeStableGemmaListenerLM(language_model, target_length=291)` senza duplicare pesi.
2. `GemmaAudioTowerBackend.encode()`:
   - Esegue la torre audio nativa per estrarre i 4 soft token del frame corrente.
   - Avanza il language model multimodale tramite `self._run_incremental_lm()`.
   - Restituisce `gemma_hidden_t` come **ultimo hidden layer del language transformer multimodale** (risolvendo il blocker storico in cui veniva restituito il soft-token audio pre-LM).
   - Preserva `audio_tower_hidden` come metadato sidecar.

---

## 5. Prossimi Passi

Con il superamento di tutte le 5 prove della Variante A:
1. **P4 Runtime Listener**: Il blocco dell'equivalenza BF16 e dell'isolamento a pesi condivisi è **formalmente superato**.
2. **Vincoli invariati**: Non procedere a P5/QLoRA o modifiche Qwen finché non verranno eseguiti i rimanenti test di wiring e riceveremo autorizzazione esplicita.
