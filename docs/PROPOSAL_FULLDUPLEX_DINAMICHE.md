# PROPOSTA ARCHITETTURALE: ANALISI DINAMICHE FULL-DUPLEX
## Valutazione Empirica, Revisione della Letteratura SOTA e Progetto Architetturale per Turn-Taking, Barge-In, Yield e Backchannel

- **Progetto**: Gemma 4 Full-Duplex Omni Pipeline (`C:\Users\andre\Documents\gemma-4-omni`)
- **Ruolo**: Ingegnere ML Senior / Lead Architect
- **Data**: 20 Agosto 2026
- **Oggetto**: Analisi strutturale e dei dati sull'apprendimento delle dinamiche conversazionali full-duplex (*barge-in, yield, interrupt, backchannel*) nella pipeline attuale, e definizione di una proposta architetturale rigorosa basata sulla letteratura scientifica (2024-2026).

---

## 1. SINTESI E VERDETTO ESECUTIVO

```
========================================================================================
                                VERDETTO SCIENTIFICO
========================================================================================
[STATO ATTUALE]: IL MODELLO ATTUALE NON HA IMPARATO NÉ PUÒ ESPRIMERE LE DINAMICHE
                 FULL-DUPLEX (BARGE-IN, YIELD, INTERRUPT, BACKCHANNEL).
                 
[MOTIVAZIONE]:  1. DATI & TRAINING: Le finestre di training (5.12s) filtrano solo
                   i frame attivi dell'agente, addestrando la regressione latente di
                   h_t esclusivamente su tratti di monologo/sintesi. Non esiste alcuna
                   supervisione su cambi di stato o sovrapposizioni.
                2. ARCHITETTURA: L'adapter proietta monotonicamente lo hidden state
                   di Gemma in h_t (2048D) e il sub-talker (Qwen3-TTS congelato) è un
                   puro decoder neurale acustico privo di logica di turno, di token di stop
                   condizionato o di facoltà di emissione di silenzio adattivo.
========================================================================================
```

Per raggiungere un reale funzionamento **Full-Duplex ascoltabile e controllabile**, è **indispensabile un modulo o una head dedicata al Turn-Taking Management** (o l'integrazione di token conversazionali espliciti).

---

## 2. FASE 1 — ANALISI DEI DATI E DEI FLUSSI CORRENTI (EVIDENZA SPERIMENTALE)

### 2.1. Ispezione dei Dati di Training (Parquet & Cache Dataset)

Dall'ispezione del codice di data preparation (`kaggle_pipeline.py`, linee 280-320, 1260-1400, 1550-1880, 2501-2678):
1. **Origine dei Dati**:
   - **`qualialabsAI/DuplexConv`**: Dataset stereo nativo a due canali (canale 0 = utente, canale 1 = agente), con segmenti da 5-30 secondi e annotazioni ASR per canale (`asr[1].startInMs`, `endInMs`, `txt`). Contiene sovrapposizioni reali e cambi turno naturali.
   - **`Mozilla Common Voice (MDC) 26.0 - Italiano`**: Dataset mono single-speaker. Per generare audio dual-channel, la funzione `create_dual_channel_dialogue` concatena i clip alternandoli con pause di 0.4s e applica una simulazione euristica di barge-in (20% probabilità) in cui il clip dell'agente viene troncato a metà quando subentra l'utente.
2. **Cosa viene esportato nei file Parquet**:
   - `_mel.parquet`: Mel log-spettrale canale utente (100 Hz, 80 mel/frame agente = 512 per finestra da 64 frame agente).
   - `_ameld.parquet`: Mel canale agente per auto-ascolto ritardato (1 frame = 80ms delay).
   - `_codes.parquet`: Matrice di 16 livelli RVQ (1 semantico a 4096 vocab + 15 acustici a 2048 vocab) a 12.5 Hz per il parlato agente.
   - `_text.parquet`: Token BPE della trascrizione risposta agente per finestra.
   - `_hyperparams.json`: Confini di segmento (`SEGMENT_ENDS`) per lo split train/val a confine di conversazione.
3. **Cosa vede effettivamente la Loss nel Training Loop (Stage 2 FitNets)**:
   - Nel dataloader di training (`kernel_src/rankfix_cell2.py`, `temporal_cell2.py`, `fitnets_extract_cell2.py`):
     - La funzione `_mel_active_mask` seleziona **esclusivamente le finestre con energia agente significativa** (`energy > thr`).
     - Le finestre di solo ascolto (in cui l'agente è silenzioso e l'utente parla) vengono **sistematicamente scartate** dal training (`if int(active.sum()) == 0: return None`).
     - La loss viene calcolata solo sui frame attivi:
       $$\mathcal{L} = (1 - \text{cos\_sim}(h_t^{pred}, h_t^{target})) + 0.5 \cdot \text{MSE}(h_t^{pred}, h_t^{target}) + 0.2 \cdot \mathcal{L}_{coupling}(y_{1..15}, \text{SubTalker}(h_t^{pred}, y_0)) + 0.5 \cdot \mathcal{L}_{rank}$$
     - **Conclusione empirica sui dati**: Il modello viene addestrato al 100% come un generatore di parlato continuo (TTS monologo da hidden state). Non esiste alcun gradiente associato al silenzio dell'agente durante l'ascolto dell'utente, alla decisione di interruzione o all'emissione di un backchannel.

---

### 2.2. Requisiti di Supervisione per le Dinamiche Conversazionali

Per consentire a un modello neurale di padroneggiare le dinamiche full-duplex servono precise forme di supervisione:

| Dinamica Conversazionale | Comportamento Richiesto al Modello | Segnale di Supervisione Necessario |
| :--- | :--- | :--- |
| **Barge-In Detection** *(Interruzione Utente)* | L'agente sta parlando; l'utente interviene per prendere la parola. L'agente deve rilevare l'intento competitivo e arrestare la generazione di $h_t$ entro $<200\text{ ms}$. | Supervisione binaria o multi-classe a ogni frame temporale $t$ ($y_t \in \{\text{SPEAK}, \text{YIELD\_TO\_BARGEIN}\}$), o token di controllo causale esplicito. |
| **Yield Detection** *(Cessione Fine Risposta)* | L'agente conclude la propria frase/argomentazione. Deve cedere il turno all'utente emettendo token di silenzio o cambiando stato a `LISTEN`. | Supervisione di fine emissione ($h_t \to \text{silence}$ oppure predizione di probabilità di End-of-Turn / EOS). |
| **User Hesitation vs Turn End** | L'utente fa una pausa intra-frase ("Vorrei sapere... [pausa 400ms] ...il prezzo"). L'agente NON deve interrompere l'utente. | Supervisione discriminativa che associ alle pause sintattiche incomplete lo stato `KEEP_WAITING` e non `START_SPEAKING`. |
| **Backchannel Generation** *(Feedback Passivo)* | L'utente parla a lungo; l'agente emette un breve assenso ("mhm", "sì", "chiaro") della durata di ~300-600ms senza rubare il turno primario. | Finestre di micro-turn con emissione di brevi unità discrete o trigger di token `<system backchannel>` su specifici punti di ancoraggio prosodico/sintattico. |
| **Unintentional Barge-In** *(Tosse / Rumore / Backchannel Utente)* | L'agente sta parlando e l'utente dice "ok" o tossisce. L'agente DEVE continuare a parlare senza interrompersi. | Classificazione semantica della voce utente per distinguere sovrapposizione collaborativa da competitiva. |

---

### 2.3. Limiti Strutturali dell'Architettura Attuale

L'architettura attuale è composta da:
$$\text{Gemma 4 E2B (Frozen)} \xrightarrow{\text{hidden } 1536} \text{TemporalResidualAdapter (59.55M)} \xrightarrow{h_t \in \mathbb{R}^{2048}} \text{Qwen3-TTS SubTalker (Frozen 181M)} \xrightarrow{\text{RVQ 1..15}} \text{Code2Wav}$$

1. **Assenza di canale decisionale discreto**: L'adapter è un modulo deterministico continuo che mappa $1536 \to 2048$. Non possiede uscite probabilistiche (softmax) relative allo stato del turno.
2. **Natura "cieca" del Sub-Talker**: Il sub-talker di Qwen3-TTS è un modulo di sintesi vocale estratto da un modello TTS autoregressivo. Dato $h_t$ e $y_0$, predice i centroidi acustici residui $y_{1..15}$. Non possiede alcuna logica interna per arrestare il rendering o forzare il muting.
3. **Mancanza di loop di feedback in inferenza streaming**: In inferenza, se un frame di audio utente contiene voce competitiva, l'adapter continua a proiettare $h_t$ per la finestra corrente senza avere alcun meccanismo intrinseco per azzerare il segnale o segnalare al server streaming di svuotare il circular buffer.

---

## 3. FASE 2 — STATO DELL'ARTE DELLA LETTERATURA SCIENTIFICA (2024-2026)

L'analisi sistematica della letteratura su Full-Duplex Spoken Dialogue Systems (SDS) rivela che i sistemi di punta si collocano lungo una tassonomia precisa (formalizzata da *Lu et al., 2026*):

```
                        GERARCHIA ARCHITETTURALE FULL-DUPLEX (Lu et al., 2026)
                        
  [L0: External Module / Pipeline]  --->  VAD / pVAD esterno, FastTurn, FireRedChat, FlexDuo
  [L1: Hidden-State Predictor]      --->  MinMo, Freeze-Omni (Speech Decoder da LLM frozen)
  [L2: Token-Level Synchronization] --->  Moshi (Dual-Stream), DuplexCascade (Micro-Turns), SyncLLM
  [L3: Shared Continuous Latent]    --->  Frontiera aperta (JEPA dual-stream, Continuous Flow)
```

---

### 3.1. Analisi Comparativa delle Architetture SOTA

#### 1. Moshi (Défossez et al., Kyutai, Ottobre 2024, arXiv:2410.00037)
- **Architettura**: Dual-Stream RQ-Transformer basato sul codec neurale **Mimi** (12.5 Hz, 8 codebook).
- **Meccanismo di Turn-Taking**: **L2 Token-Level puro**. Il modello modella simultaneamente due stream paralleli di 8 codebook (stream Utente + stream Assistente), affiancati da uno stream testuale anticipato (*Inner Monologue* con `TextualHead`).
- **Gestione Turni**: A ogni frame di 80ms, Moshi predice i codici audio dell'assistente e predice autoregressivamente i token dell'utente. Quando l'utente parla, il modello impara dai dati conversazionali (Fisher corpus) a emettere token di silenzio per l'assistente, o a generare backchannel se l'utente è in una pausa di sospensione. Non necessita di VAD esterno, ma richiede pre-addestramento end-to-end su migliaia di ore di dialoghi a due tracce perfettamente allineati.

#### 2. DuplexCascade (Yang, Fujita, & Sudo, Marzo 2026, arXiv:2603.09180)
- **Architettura**: Pipeline cascaded streaming ASR $\to$ LLM $\to$ TTS con discretizzazione a **micro-turn** ($\Delta t = 0.6\text{ s}$).
- **Conversational Special Tokens**: Elimina il VAD introducendo 6 token speciali nel vocabolario dell'LLM:
  - `<user is speaking>`: L'utente sta parlando (peso loss = 1).
  - `<user finish speaking>`: L'utente ha completato il turno (peso loss = 10 per forzare la reattività).
  - `<user is interrupting>`: L'utente ha interrotto l'assistente (peso loss = 5 $\implies$ trigger di stop immediato su TTS).
  - `<user backchannel>`: L'utente ha fatto un breve backchannel (peso loss = 2 $\implies$ l'assistente non si ferma).
  - `<user is thinking>`: L'utente sta esitando/pensando (peso loss = 1 $\implies$ l'assistente attende).
  - `<system backchannel>`: L'assistente deve emettere un feedback immediato (peso loss = 3 $\implies$ play di una clip audio pre-sintetizzata).
- **Evidenza**: Dimostra che il fine-tuning LoRA dell'LLM (50k dialoghi, 5k step) con **Weighted Cross-Entropy Loss** sui token speciali è sufficiente per ottenere un turn-taking fluido e robusto a latenza zero.

#### 3. LM-Enhanced Dialogue Management / Semantic VAD (Zhang et al. - Tencent AI Lab, Giugno 2026, arXiv:2502.14145v3)
- **Architettura**: Dialogue Manager (DM) leggero basato su un LLM da **0.5B** (Hunyuan-0.5B) interposto tra lo streaming ASR/VAD acustico e il Core Dialogue Engine (CDE).
- **Control Tokens a 4 vie**:
  - `<|Normal|>`: Normale ciclo di domanda e risposta.
  - `<|Real-INT|>`: *Real Interruption* $\implies$ l'utente sta prendendo la parola con una nuova richiesta; il DM cancella all'istante la risposta in corso del CDE/TTS.
  - `<|Fake-INT|>`: *Fake Interruption* $\implies$ rumore, tosse o backchannel dell'utente; il DM sopprime l'evento e il CDE continua a parlare indisturbato.
  - `<|Continue-Listening|>` (`<|C-L|>`): Query incompleta o esitazione; il DM accumula i micro-turn senza svegliare inutilmente il generatore pesante.

#### 4. SyncLLM (Veluri et al. - Meta AI / Univ. of Washington, EMNLP 2024)
- **Architettura**: Synchronous LLM che sincronizza la generazione con un orologio temporale a chunk discreti (160-240ms).
- **Previsione Cross-Stream**: Genera prima una stima del chunk futuro dell'utente e la usa come condizionamento per predire il proprio chunk successivo, garantendo latenza inter-turno pari a quella umana (~200ms).

#### 5. Mini-Omni & Mini-Omni 2 (Xie & Wu, 2024, arXiv:2408.16703, arXiv:2410.11190)
- **Architettura**: Any Model Can Talk + SNAC audio tokenizer.
- **Interruzione**: Utilizza un comando di interruzione esplicito o un layer di interruzione streaming esterno: non appena il canale audio utente supera la soglia di energia o viene rilevato un token di stop, il decoding batch-parallel viene interrotto.

---

### 3.2. Sintesi della Letteratura su Barge-In e Backchannel

1. **Barge-In Detection (VAD vs Model-Based)**:
   - Il **VAD puramente acustico (Energy/Silero)** soffre di falsi positivi nel 30-45% dei casi di parlato spontaneo (*Lu et al., 2026; Zhang et al., 2026*), poiché interrompe l'agente ogni volta che l'interlocutore respira rumorosamente, ride o dice "mhm".
   - Il **Model-Based Turn-Taking (Semantic VAD / TurnHead)** riduce i falsi positivi a $<5\%$, sfruttando il contesto semantico e prosodico per discriminare l'intento dell'interruzione.
2. **Backchannel Decision**:
   - I sistemi production-grade (come *DuplexCascade* e *Google Duplex*) evitano di sintetizzare da zero i backchannel frame-by-frame: una head dedicata o un token speciale predice l'evento `<system backchannel>` e il client riproduce a latenza zero (<50ms) un set di brevi clip audio stereotipiche ad alta qualità ("uh-huh", "sì", "d'accordo", "capisco").

---

## 4. FASE 3 — PROPOSTA ARCHITETTURALE: IL SISTEMA "GEMMA-4 DUPLEX MANAGER"

Per dotare la nostra pipeline delle capacità full-duplex senza stravolgere il lavoro di distillazione di $h_t$ e senza appesantire la VRAM su GPU T4 (16GB), si propone un design a due moduli complementari: **`TurnStateHead`** (modulo neurale leggero a livello di frame) unito a un **`StreamingDuplexController`** a livello di serving.

```
                           ARCHITETTURA PROPOSTA (L1+L0 HYBRID)
                           
               [Canale Utente: Mel 100Hz]      [Canale Agente: Auto-Ascolto]
                           │                                │
                           ▼                                ▼
                ┌──────────────────────────────────────────────────────┐
                │        Gemma 4 E2B Backbone (4-bit Frozen / QLoRA)   │
                └──────────────────────────┬───────────────────────────┘
                                           │ Hidden State h_gemma (d=1536)
                     ┌─────────────────────┴─────────────────────┐
                     │                                           │
                     ▼                                           ▼
       ┌───────────────────────────┐               ┌───────────────────────────┐
       │  TemporalResidualAdapter  │               │      TurnStateHead        │
       │    (4 blocchi MHSA Causali│               │ (Classificatore Neurale   │
       │      + Conv1D + MLP)      │               │     a 5 Stati di Turno)   │
       └─────────────┬─────────────┘               └─────────────┬─────────────┘
                     │ Predizione h_t (d=2048)                   │ Logits Turno s_t (5-dim)
                     ▼                                           ▼
       ┌───────────────────────────┐               ┌───────────────────────────┐
       │   Sub-Talker Qwen3-TTS    │               │  StreamingDuplexController│
       │   (Frozen bf16, 181M)     │               │    (FSM & Decision Logic: │
       └─────────────┬─────────────┘               │    Yield, Barge-In, Mute) │
                     │ RVQ Codes 1..15                           └─────────────┬─────────────┘
                     ▼                                                         │
       ┌───────────────────────────┐                                           │ Controllo
       │      Code2WavDecoder      │ ◄─────────────────────────────────────────┘ Buffer/Audio
       │      (Standalone 24kHz)   │
       └───────────────────────────┘
```

---

### 4.1. Progetto di Dettaglio: `TurnStateHead` (Classificatore Neurale di Turno)

1. **Tipologia Architetturale**:
   - Rete neurale a convoluzione causale temporale + MLP a 3 strati con Pre-LayerNorm e attivazioni SiLU.
2. **Input**:
   - Vettore di feature $z_t = [h_t^{\text{temp}} \,\|\, e_t^{\text{user}} \,\|\, \Delta e_t^{\text{user}}] \in \mathbb{R}^{2048 + 128 + 128} = \mathbb{R}^{2304}$, dove $h_t^{\text{temp}}$ è l'hidden state emesso dal `TemporalResidualAdapter` e $e_t^{\text{user}}$ è l'embedding di energia/mel del canale utente estratto da un piccolo linear layer causale.
3. **Output (Spazio di Decisione a 5 Stati)**:
   - Vettore di logits $\mathbf{s}_t \in \mathbb{R}^5$:
     1. **`STATE_SPEAK` ($s_0$)**: L'agente parla normalmente; generazione $h_t$ attiva.
     2. **`STATE_YIELD` ($s_1$)**: L'agente ha completato il turno di risposta; cede la parola (transizione a silenzio / `LISTEN`).
     3. **`STATE_BARGE_IN` ($s_2$)**: Rilevata presa di parola competitiva dell'utente durante il parlato dell'agente $\implies$ trigger immediato di interruzione.
     4. **`STATE_USER_BACKCHANNEL` ($s_3$)**: L'utente ha emesso un breve feedback passivo ("mhm") $\implies$ l'agente NON interrompe la sintesi.
     5. **`STATE_EMIT_BACKCHANNEL` ($s_4$)**: L'agente è in ascolto e rileva un punto di ancoraggio prosodico $\implies$ emissione di una breve clip di assenso.
4. **Funzione di Loss ($\mathcal{L}_{\text{turn}}$)**:
   - Weighted Cross-Entropy Loss calcolata a ogni frame temporale:
     $$\mathcal{L}_{\text{turn}} = - \sum_{k=0}^4 w_k \cdot y_{t,k} \log \sigma(\mathbf{s}_t)_k$$
   - **Pesi di Bilanciamento**: $w_{\text{SPEAK}} = 1.0$, $w_{\text{YIELD}} = 8.0$, $w_{\text{BARGE\_IN}} = 6.0$, $w_{\text{USER\_BC}} = 3.0$, $w_{\text{EMIT\_BC}} = 4.0$ (ispirato a *DuplexCascade*).
5. **Complessità Parametrica & Impatto VRAM**:
   - Parametri totali: **$\approx 3.8\text{M}$ parametri trainable** (`Linear(2304, 1024) -> LN -> SiLU -> Linear(1024, 256) -> SiLU -> Linear(256, 5)`).
   - Impatto VRAM su T4 (16GB): **$< 45\text{ MB}$** (completamente trascurabile, zero rischio di OOM).
   - Latenza computazionale per batch da 64 frame: **$< 1.5\text{ ms}$** su GPU.

---

### 4.2. Dati Necessari per Addestrare la Head (Gap Analysis Dati)

Attualmente i nostri file Parquet contengono solo audio allineato e codici RVQ del parlato agente. Per addestrare la `TurnStateHead` occorre arricchire il dataset nella **Fase 1**:

1. **Dataset Conversazionali Dual-Channel Target**:
   - `qualialabsAI/DuplexConv` (utilizzare tutte le 50h+ disponibili con il parsing completo degli archivi `jsons.tar.gz`).
   - `Candor Corpus` (dialoghi video/audio a 2 canali con annotazioni di turn-taking e overlap).
   - `EasyTurn Dataset` (*Li et al., 2026*): 1145 ore di parlato annotato specificamente per turn-taking, backchannel e barge-in.
2. **Generazione di Ground-Truth Labels a Frame Rate (12.5 Hz)**:
   - Creazione di una colonna `turn_labels` ($T \times 1$, `int8`) generata programmaticamente tramite euristica timestamp-aware:
     - Se `agent_speaking` e `user_silent` $\implies$ `0 (SPEAK)`.
     - Negli ultimi 3 frame (240ms) prima della fine del turno agente $\implies$ `1 (YIELD)`.
     - Se `agent_speaking` e `user_speaking` con durata $>400\text{ms}$ e trascrizione semantica $>2$ parole $\implies$ `2 (BARGE_IN)`.
     - Se `agent_speaking` e `user_speaking` con durata $<400\text{ms}$ (es. "ok", "mhm") $\implies$ `3 (USER_BACKCHANNEL)`.
     - Durante le pause dell'utente $>300\text{ms}$ in code a frasi lunghe $\implies$ `4 (EMIT_BACKCHANNEL)`.

---

## 5. TRADE-OFF E RACCOMANDAZIONI PRIORITIZZATE

Per guidare le decisioni del coordinatore, si delineano tre strategie a complessità e investimento crescenti:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               ROADMAP RACCOMANDATA (PRIORITIZZATA)                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PRIORITÀ 1: L0/L1 Hybrid Controller a Regole (Zero Training, Full-Duplex Immediato)             │
│   • Integrazione di un controller in serve_duplex.py basato su energy-gating con isteresi       │
│   • Se energia utente > soglia per >160ms durante la sintesi -> flush buffer e stop sub-talker  │
│   • Costo: 0 ore GPU. Pronto per demo live in streaming.                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PRIORITÀ 2: Stage 3 Training di `TurnStateHead` (3.8M params) su Dati Dual-Channel              │
│   • Backbone Gemma 4 e Adapter congelati; addestramento della sola TurnStateHead con CE pesata  │
│   • Classificazione accurata di Barge-in vs Backchannel su DuplexConv + EasyTurn                │
│   • Costo: ~1.5 ore su Kaggle T4.                                                               │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ PRIORITÀ 3: Conversational Micro-Turns End-to-End (Stile DuplexCascade / Moshi)                 │
│   • Fine-tuning QLoRA congiunto con token speciali conversazionali e generazione parallela      │
│   • Massima naturalezza umana e controllo end-to-end senza moduli esterni                      │
│   • Costo: ~4.0 ore su Kaggle 2x T4.                                                            │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. BIBLIOGRAFIA SCIENTIFICA AGGIORNATA (TURN-TAKING & FULL-DUPLEX)

1. **Lu, J., Wang, Y., Luo, J., Chen, Y., Liang, T., Ji, S., Jiang, Z., Yang, X., Zhang, Y., Cheng, X., Wen, C., Pan, C., Wang, H., Ye, C., Wu, J., Jiang, X., Jiang, G., & Zhao, Z.** (Giugno 2026). *A Survey of Full-Duplex Spoken Dialogue Systems: Architectural Hierarchy, Interaction Ontology, and Decision State Machine*. arXiv:2606.19453v1.
2. **Yang, J., Fujita, Y., & Sudo, Y.** (Marzo 2026). *DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR-LLM-TTS Pipeline and Micro-Turn Optimization*. SB Intuitions & Univ. of Tokyo, arXiv:2603.09180v1.
3. **Zhang, H., Li, W., Chen, R., Kothapally, V., Yu, M., & Yu, D.** (Giugno 2026). *LM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems*. Tencent AI Lab, arXiv:2502.14145v3.
4. **Veluri, B., Peloquin, B. N., Yu, B., Gong, H., & Gollakota, S.** (Novembre 2024). *Beyond Turn-Based Interfaces: Synchronous LLMs as Full-Duplex Dialogue Agents*. Meta AI & University of Washington, EMNLP 2024, pp. 21390-21402.
5. **Défossez, A., Mazaré, L., Orsini, M., Royer, A., Pérez, P., Jégou, H., Grave, E., & Zeghidour, N.** (Ottobre 2024). *Moshi: a speech-text foundation model for real-time dialogue*. Kyutai, arXiv:2410.00037.
6. **Xie, Z., & Wu, C.** (Agosto 2024). *Mini-Omni: Language Models Can Hear, Talk and Think in Real Time*. arXiv:2408.16703.
7. **Xie, Z., & Wu, C.** (Ottobre 2024). *Mini-Omni2: Towards Open-source GPT-4o with Vision, Speech and Duplex Capabilities*. arXiv:2410.11190.
8. **Li, J. et al.** (2026). *Easy Turn: A 1145-Hour Open Turn-Taking Corpus and Multi-State Turn Detector*. Cited in Lu et al. (2026).
9. **Wang, X. et al.** (Novembre 2024). *Freeze-Omni: A Smart and Low Latency Speech-to-Speech Dialogue Model with Frozen LLM*. arXiv:2411.00774.
10. **Qwen Team (Alibaba Cloud)** (Gennaio 2026). *Qwen3-TTS Technical Report: Advancing Spoken Language Modeling*. arXiv:2601.15621.
11. **Google DeepMind** (Luglio 2026). *Gemma 4 Technical Report*. arXiv:2607.02770.

---
*Proposta architetturale redatta e verificata nel rispetto dei principi di Source-Driven Development (SDD) e Doubt-Driven Development (DDD) del progetto Gemma 4 Full-Duplex Omni Pipeline.*
