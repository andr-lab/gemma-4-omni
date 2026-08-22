# Report Task 10 — Ricerca Common Voice 26 Italiano & Stato Datasets

**Data:** 22 Agosto 2026  
**Hardware:** NVIDIA GeForce RTX 3090 (24 GB VRAM)  
**Ambiente:** Local Docker (`gemma4-omni:latest`)  
**Stato Checkpoint SOTA:** `checkpoints/gemma4_duplex_sota` (Cos Sim 0.9279)  
**Dataset Italiano Corrente:** `data/train_tokens_mdc_16.0h_9185s` (16.0h, 720.014 frame, 9.185 clip)

---

## 1. Obiettivo dell'Indagine

Verificare la disponibilità di nuovi dataset pubblici in lingua italiana, con specifico focus su **Common Voice 26.0** (o Mozilla Data Collective v26), per valutare un'eventuale estensione del corpus di addestramento dell'Adapter e di TurnHead.

---

## 2. Esito Ricerca e Verifica Repository Hugging Face Hub

È stata condotta una scansione sistematica tramite l'API Hugging Face Hub sui repository candidati:

| Repository / Sorgente | Tipo / Licenza | Stato Accesso / Lingua IT | Utilizzabilità Automatica |
| :--- | :---: | :---: | :---: |
| `Peacockery/common-voice-scripted-speech-26` | Public Community Hub | 68 lingue supportate, **ma la lingua italiana (`it`) non è inclusa** | ❌ Assente lingua `it` |
| `deepdml/common_voice_26_0` | Gated Dataset | **Gated Dataset (Richiede accettazione termini web)** -> `HTTP 401 / Unauthorized` | ❌ Bloccato da Gated Auth |
| `mozilla-foundation/common_voice_17_0` (e varianti) | Mozilla Public License | Richiede token con permessi espliciti o download interattivo | ❌ Richiede Auth Manuale |
| `data/train_tokens_mdc_16.0h_9185s_*` | Locale Cache NVMe | 16.0h di parlato MDC italiano pre-processato e pronto (9.185 segmenti) | **✅ DISPONIBILE & SOTA** |

---

## 3. Decisione Architetturale & Mantenimento SOTA

Poiché non sono disponibili ulteriori dump unauthenticated di Common Voice 26 in lingua italiana:
1. **Conservazione SOTA Esistente:** Il checkpoint `checkpoints/gemma4_duplex_sota` (addestrato su 16.0h di MDC italiano con cosine similarity $0.9279$ e SVD rank $1537$) è confermato come modello di riferimento per la lingua italiana.
2. **Nessun Downgrade o Rielaborazione Inutile:** Evitata la rielaborazione ridondante del dataset MDC già ottimizzato su disco in formato float16 memory-mapped (`0.00s` load time, zero copie in RAM).
3. **Integrazione con TurnHead v4:** L'Adapter Italiano SOTA è perfettamente compatibile con le feature $h_t \in \mathbb{R}^{2048}$ estratte da `TurnStateHeadV4`.
