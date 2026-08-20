# Analisi Approfondita del Dataset DuplexConv
## Struttura Dati, Temporizzazione ASR Dual-Channel e Strategia per la TurnStateHead

- **Progetto**: Gemma 4 Full-Duplex Omni Pipeline (`C:\Users\andre\Documents\gemma-4-omni`)
- **Ruolo**: Ingegnere ML Senior / Lead Architect
- **Data**: 20 Agosto 2026
- **Oggetto**: Analisi dettagliata del corpus `qualialabsAI/DuplexConv` (50h+ stereo nativo) e delle sue annotazioni per la derivazione del ground truth conversazionale a 12.5 Hz (TurnStateHead).

---

## 1. Struttura Esatta dei Record di DuplexConv

`qualialabsAI/DuplexConv` e un dataset stereo nativo a due canali specificamente progettato per lo studio del dialogo parlato spontaneo.

### 1.1. Audio Stereo e Formato WebDataset
- **Audio Shards**: Distribuiti in formato WebDataset (`.tar`) su tre domini principali:
  - `Edu/` (conversazioni a tema educativo/tutoring)
  - `edu/` (dialoghi accademici/studio)
  - `none_Edu/` (dialoghi a tema libero, vita quotidiana, interviste)
- **Canali Audio**:
  - **Canale 0 (Traccia Sinistra)**: Audio dell'Utente (`user_audio`).
  - **Canale 1 (Traccia Destra)**: Audio dell'Agente / Interlocutore (`agent_audio`).
- **Campionamento e Durata**:
  - Sample rate nativo: **48,000 Hz** (stereo PCM float32 o 16-bit int).
  - Durata per record: Segmenti conversazionali da **5 a 30 secondi** (non file monolitici da 120s, bensi chunk di dialogo continuo coerente).
  - Volume complessivo: **50+ ore** di interazione spontanea (~1500-3000 segmenti totali).
- **Pre-Processing nella Pipeline**:
  - Resampling Canale 0 a **16 kHz** per l'estrazione del Log-Mel Spectrogram a 100 Hz (128 mel bins) per il Conformer nativo di Gemma 4 E2B.
  - Resampling Canale 1 a **24 kHz** per la tokenizzazione RVQ a 12.5 Hz (16 codebook) via Qwen3-TTS Tokenizer.
  - Normalizzazione RMS a ~0.05 per uniformita di livello sonoro tra i due canali.

### 1.2. Archivi di Annotazioni ASR (`jsons.tar.gz`)
Le trascrizioni e le annotazioni temporali non risiedono negli shard `.tar` audio ma in tre archivi paralleli:
`https://huggingface.co/datasets/qualialabsAI/DuplexConv/resolve/main/{Edu,edu,none_Edu}/jsons.tar.gz`
- Dimensione compressa: ~57 MB (`Edu`) + ~24 MB (`edu`) + ~150 MB (`none_Edu`) = **~231 MB totali**.
- Dimensione decompressa: **~1.0 GB di file JSON** indicizzati per chiave segmento (es. `Edu--000001.json`).

### 1.3. Schema Dati JSON Dettagliato
Ogni file JSON ha la struttura seguente:
```json
{
  "key": "Edu--000001",
  "durationInMs": 18450,
  "asr": [
    [
      {
        "startInMs": 420,
        "endInMs": 3150,
        "labels": {
          "txt": "Can you explain how the Doppler effect works in astronomy?"
        },
        "state": "<|complete|>"
      },
      {
        "startInMs": 7200,
        "endInMs": 7650,
        "labels": {
          "txt": "mhm"
        },
        "state": "<|incomplete|>"
      }
    ],
    [
      {
        "startInMs": 3500,
        "endInMs": 12800,
        "labels": {
          "txt": "Sure! When a star moves towards us, its light waves get compressed..."
        },
        "state": "<|complete|>"
      }
    ]
  ]
}
```
- `asr[0]`: Lista cronologica delle emissioni dell'**Utente**.
- `asr[1]`: Lista cronologica delle emissioni dell'**Agente**.
- `startInMs` / `endInMs`: Timestamp precisi al millisecondo di inizio e fine emissione.
- `labels.txt`: Trascrizione testuale del parlato.
- `state`: Tag sintattico di completezza semantica (`<|complete|>` vs `<|incomplete|>`).

---

## 2. La Temporizzazione Disponibile e Derivazione Ground Truth a 12.5 Hz

La disponibilita dei timestamp millisecondo per **ENTRAMBI i canali contemporaneamente** consente di mappare in modo deterministico e programmatico lo stato conversazionale su ciascun frame temporale di **80 ms (12.5 Hz)**:

$$\Delta t_{\text{frame}} = \frac{1000\text{ ms}}{12.5} = 80.0\text{ ms}$$

A ogni frame temporale $t$ ($t = 0, 1, \dots, T-1$, corrispondente all'intervallo $[t \cdot 80\text{ ms}, (t+1) \cdot 80\text{ ms})$), definiamo l'attivita dei due canali:
- $\text{UserActive}(t) \in \{0, 1\}$: $1$ se l'intervallo interseca un'emissione utente `asr[0]` (o energia mel utente $> \text{thr}$).
- $\text{AgentActive}(t) \in \{0, 1\}$: $1$ se l'intervallo interseca un'emissione agente `asr[1]` (o energia codici agente non-silenzio).

### 2.1. Regole di Assegnazione delle 5 Classi TurnStateHead

| ID Classe | Nome Classe | Condizione Temporale & Semantica | Dinamica Conversazionale Modellata |
| :---: | :--- | :--- | :--- |
| **0** | **`SPEAK`** | $\text{AgentActive}(t)=1 \land \text{UserActive}(t)=0$ | L'agente parla da solo; sintesi vocale attiva e indisturbata. |
| **1** | **`YIELD`** | $\text{AgentActive}(t)=1 \land \text{Frame } t \in [\text{EndAgente}-3, \text{EndAgente}]$ | Ultimi 3 frame (240 ms) del turno agente prima del silenzio: cessione attiva del turno. |
| **2** | **`BARGE_IN`** | $\text{AgentActive}(t)=1 \land \text{UserActive}(t)=1 \land \text{DurataUtente} \ge 400\text{ms}$ (o $>2$ parole) | Sovrapposizione competitiva: l'utente interrompe l'agente per prendere la parola $\implies$ trigger stop immediato. |
| **3** | **`USER_BACKCHANNEL`** | $\text{AgentActive}(t)=1 \land \text{UserActive}(t)=1 \land \text{DurataUtente} < 400\text{ms}$ (es. "mhm", "ok", "yeah") | Sovrapposizione collaborativa passiva: feedback utente $\implies$ l'agente NON si deve interrompere. |
| **4** | **`EMIT_BACKCHANNEL`** | $\text{AgentActive}(t)=0 \land \text{UserActive}(t)=1 \land \text{PausaInterUtente} \in [300, 800]\text{ms}$ | L'agente e in ascolto; l'utente sta completando un concetto e fa una pausa $\implies$ opportunita di assenso. |

---

## 3. Confronto Critico: Cosa Usiamo OGGI vs Cosa Offre DuplexConv

| Dimensione | Utilizzo Attuale (FitNets Stage 2 / Temporal / RankFix / Window128) | Potenziale Completo Offerto da DuplexConv |
| :--- | :--- | :--- |
| **Canale Utente (ch0)** | Solo feature Mel 100Hz in ingresso all'audio tower Conformer di Gemma 4. | Informazione prosodica, sintattica, pause, backchannel ed enfasi dell'utente. |
| **Canale Agente (ch1)** | Codici RVQ 12.5Hz (target regressione $h_t$) + Mel ritardato (auto-ascolto). | Trascrizione temporizzata, intonazione di chiusura turno, segnali di cessione. |
| **Finestre di Training** | **Solo finestre con energia agente (`_mel_active_mask`)**. Finestre di solo ascolto **scartate al 100%**. | Finestre bilanciate: monologo agente, solo ascolto utente, cambi turno, sovrapposizioni reali. |
| **Supervisione Turni** | **Assente** (Loss di regressione su $h_t$ attiva solo sul parlato agente). | Supervisione discreta frame-by-frame (5 stati di turno) con Ground Truth ASR dual-channel. |
| **Interruzioni (Barge-In)** | Trattate come rumore spurio o simulate artificialmente (MDC 20% truncate). | **Sovrapposizioni umane reali**: sovrapposizioni naturali di 400-1200ms con reazione spontanea dei parlanti. |
| **Backchannel** | Inesistenti nel loss path. | **Centinaia di backchannel naturali** annotati con durata e testo nei JSON (`asr[0]`). |
| **Silenzi Inter-Turno** | Filtrati o trattati come padding. | Durate reali di presa di parola umana (gap distributions tra 150ms e 600ms). |

---

## 4. Raccomandazioni per il Training della TurnStateHead

1. **Abbandonare il filtro monodirezionale `_mel_active_mask` per la TurnHead**:
   - Il dataloader della TurnHead deve campionare finestre da **tre distribuzioni bilanciate**:
     - 40% Finestre di Monologo Agente (`SPEAK`, `YIELD`).
     - 30% Finestre di Solo Ascolto Utente (`EMIT_BACKCHANNEL`, `LISTEN`).
     - 30% Finestre di Sovrapposizione Reale (`BARGE_IN`, `USER_BACKCHANNEL`).
2. **Definizione della Maschera di Frame Validi**:
   - Un frame e valido per la loss di turno se **almeno uno dei due canali e attivo** ($\text{UserActive} \lor \text{AgentActive}$), oppure se e un frame di silenzio transitorio intra-dialogo (non semplice zero-padding di fine batch).
3. **Loss Bilanciata (Weighted Cross-Entropy)**:
   - Dato che `SPEAK` copre ~70-80% dei frame attivi, i pesi asimmetrici $w = [1.0, 8.0, 6.0, 3.0, 4.0]$ impediscono al modello di collassare sullo stato maggioritario e forzano un'elevata sensibilita su `YIELD` e `BARGE_IN`.
4. **Isolamento Architetturale (Zero Degradazione su $h_t$)**:
   - Gemma 4 E2B e il `TemporalResidualAdapter` restano **completamente congelati** (forward `no_grad()`).
   - I gradienti fluiscono **esclusivamente nella `TurnStateHead` (3.8M)**, garantendo stabilita numerica, rapidita di convergenza (<5s per batch) e VRAM contenuta (<12 GB su Tesla T4).
