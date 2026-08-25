# Design Evidence-Based — Full-Duplex Gemma-4-E2B × Qwen3-TTS

**Data**: 2026-08-24 (aggiornato 2026-08-25 con verdetto sperimentale v5) · **Scopo**: fondare le scelte architetturali del progetto su letteratura 2024-2026, rispondere alla domanda "errori implementativi, di dati o architetturali?" e proporre il design del modello full-duplex (Gemma-4-E2B = intelligenza/ragionamento/tool calling, Qwen3-TTS = voce, + turn-taking: backchanneling, barge-in, yield, interrupt).

---

## 0. REVISIONE 2026-08-25 — Verdetto sperimentale (SUPERSEDE la sez. 1 per le conclusioni operative)

Gli esperimenti del 25 agosto (blind ASR) hanno rivisto il verdetto: il pattern cb0+residui è corretto, MA il collo NON era l'informazione in h_sub per il predittore cb0 — era **il formato/lo spazio di h dato al MTP**:

- **LOOP A**: past_hidden del backbone in loop (codici GT, cache, testo per frame) + MTP isolato da [h, cb0] → **parlato fluente** → il MTP PUÒ generare dal solo cb0 quando h è ricorrente (riga 1740 modeling_qwen3_tts.py: `past_hidden=hidden_states[:,-1:,:]`).
- **LOOP B**: h_sub dell'adapter (cos 0.63) nel loop → elica → il ponte Gemma→talker non allinea abbastanza.
- **oracle 23 ago** (h_t statico, passata unica): garbled → il formato h della passata unica è fuori distribuzione per il MTP.
- **NATIVE** (talker completo dal testo): ASR esatto → il percorso testo→talker funziona nativamente.

**Conseguenza (architettura v5, FISSATA — vedi `docs/VERDETTO_TASK12_ARCHITETTURA_V5.md`)**:
- Niente AR decoder per cb0, niente MTP isolato, niente coupling cos-MSE statico (tetto ~0.63).
- Runtime = Gemma (congelata) → h_raw → **[PROIEZIONE addestrata, unico componente trainabile]** → backbone Qwen3-TTS in loop (congelato: codec_head cb0 + MTP residui, protocollo nativo) → Code2Wav.
- Lezione Qwen3-Omni (2509.17765): token discreti per il contenuto, hidden per l'espressività → il testo di Gemma resta nel prefill (canale contenuto), h_proj è il canale espressivo.
- Ambiente: Gemma (unsloth, transformers 5.5.0) + teacher (qwen3tts_src patchato, 3 shim) coesistono in UN processo → runtime monoprocesso possibile (container `gemma4-omni-v5:latest`).
- La sezione 1 sotto resta valida come ANALISI DELLA LETTERATURA (perché il pattern era giusto), ma le sue conclusioni operative (P1-P6 su AR decoder) sono superate dal verdetto v5.

---

## 1. Il verdetto (2026-08-24, storico): implementativo, dati o architetturale?

**Sintesi: il pattern architetturale è corretto e confermato dalla letteratura; il collo di bottiglia è (a) informazione: h_sub non contiene l'informazione cb0, (b) scala: 40h vs 5M ore, (c) una mancanza implementativa: l'hybrid training (TF+free-running) non è mai stato applicato.**

### 1.1 Cosa dice la letteratura sul nostro pattern (cb0 semantico + residui acustici)

Il nostro design — predire il **codebook semantico cb0** con un predittore autoregressivo e delegare i **residui acustici cb1..15** a un modulo NAR (il sub-talker MTP di Qwen3-TTS) — è **esattamente il pattern raccomandato** dalla ricerca 2025-2026:

| Sistema | Tokenizer | Chi predice cb0/semantico | Chi fa l'acustica |
|---|---|---|---|
| **Qwen3-TTS 12Hz** (nostro) | RVQ 12.5Hz, cb0 WavLM-semantico | backbone (1.7B, 5M h) | MTP residui |
| **GLM-4-Voice** (2412.02612) | semantico supervisionato 12.5Hz, 175bps, 1 codebook | LLM 9B | flow matching |
| **CosyVoice 2** (2412.10117) | semantico S³ 25Hz | LLM | flow matching |
| **Fun-Audio-Chat** (2512.20156) | semantico S³ (25Hz→5Hz grouping) | LLM 30B-MoE | flow matching + SRH |
| **CaT-TTS** (2509.22062) | S3Codec split-RVQ + distillazione semantica | **dual LM: semantic transformer (coarse) + acoustic transformer (fine)** | AR/NAR a due stadi |
| **Moshi** (2410.00037) | Mimi: RVQ con semantic knowledge distillation (cb0 semantico) | Helium 7B (multi-stream) | RVQ residui (nested transformer) |
| **Mini-Omni** (2408.16725) | SNAC 7 codebook | LLM con **text-instructed delayed parallel** | sub-LM-heads parallele |
| **Mini-Omni Reasoner** (2508.15827) | SNAC | Talker condizionato dai token del Thinker (testo) | — |
| **Qwen2.5-Omni** (2503.20215) | tokens audio | **Talker separato (4B) che riceve gli hidden del Thinker** | — |

**Conclusioni dal confronto:**
1. **Il nostro split cb0+residui è il pattern dominante** (CaT-TTS "dual LM", Moshi "semantic distillation", Qwen3-TTS stesso). NON è un errore architetturale.
2. **Il WER Trap** (2605.29209) lo conferma teoricamente: token semantici puri ultra-compressi causano *articulatory collapse* nella generazione acustica → **"rappresentazioni esplicitamente disaccoppiate: token semantici per la comprensione, token acustici per la generazione"**. È esattamente il nostro caso: cb0 (WavLM-semantico) + residui acustici.
3. **Nessun sistema di successo predice cb0 con un decoder piccolo addestrato su decine di ore.** Chi predice il token semantico lo fa con l'LLM a scala (9B, 7B, 30B) o con un backbone addestrato su milioni di ore (Qwen3-TTS: 5M h). **Il nostro predittore cb0 (ARSemanticDecoder, top-1 12%) è l'anomalia, non il pattern.**

### 1.2 Evidenze quantitative sulla difficoltà di predire token semantici

- **"Bridging the gap between training and inference in LM-based TTS"** (2509.17021): *"speech token accuracy is inherently low, which is determined by the nature of speech token... similar sounds can be represented by different speech tokens, making sound-token mapping not strictly one-to-one"* → **la bassa accuracy TF dei token di parlato è intrinseca e documentata**. Il dato che conta è il GAP TF vs free-running (exposure bias), non il top-1 assoluto.
- Stesso paper: manifestazioni dell'exposure bias nei TTS LM = **accelerazione, EOS misprediction (early stop / ripetizioni), prosodia piatta** → sono **esattamente i sintomi del nostro free-gen** (balbettio, loop, drift).
- **Mini-Omni2** (2410.11190): *"the audio loss of Mini-Omni2 remains high during training, and the tokens for a segment of audio can vary significantly based on the content at both ends"* → stessa osservazione della nostra tolleranza cb0 (target non deterministici a livello di token, ma tollerabili a livello spettrale).
- **Mini-Omni** (2408.16725): *"flattening for audio token modeling... resulted in excessively long tokens, detrimental to streaming and leading to unstable learning"* → il nostro 12.5Hz a 16 codebook (vs SNAC 7 codebook) è già nella direzione giusta.
- **Qwen3-TTS report** (2601.15621): *"purely acoustic tokenizers inject excessive low-level detail that complicates LLM-based modeling and leads to long-horizon error accumulation"* → la scelta di cb0 semantico è motivata esattamente per evitare l'error accumulation.

### 1.3 Le tre cause del nostro blocco cb0

**(a) DATI/INFORMAZIONE — causa primaria (confermata dai nostri probe):**
- Probe per-frame su h_sub: lineare 3.7%, MLP 3.2%; transformer bidirezionale con finestra 64: loss ~6.6 (vs random 7.62 = ln 2048) → **h_sub contiene pochissima informazione su cb0, anche con contesto pieno**.
- Transfer diretto del codec_head preaddestrato su h_sub: top-1 ~0.08% (random) → **h_sub vive fuori dallo spazio hidden del backbone Qwen3-TTS**.
- Interpretazione: **l'adapter è stato addestrato per allineare h_sub ai RESIDUI acustici (sub-talker), non per preservare l'informazione semantica cb0**. L'informazione cb0 è stata sacrificata nell'allenamento dell'adapter → il predittore cb0 non ha materiale su cui lavorare. Questo spiega perché 100k step non hanno superato il 12%.

**(b) SCALA:**
- Qwen3-TTS backbone per lo stesso task (predire cb0): 5M ore, 1.7B. Noi: ~40h, decoder ~10-30M. La letteratura (SemBridge, 2608.07462) mostra che la supervisione semantica esplicita sugli stati dell'LM serve proprio a compensare (vedi §2.3).

**(c) IMPLEMENTATIVO — una mancanza, non un bug:**
- Il nostro scheduled sampling era **gated su val-acc ≥ 25%** e non si è MAI attivato. La letteratura (2509.17021) mostra che l'hybrid training (TF + free-running con prompt protection) **deve partire subito, senza gate**, e riduce WER 6.23→4.21 su CosyVoice2 (~30% relativo). 
- **Nessuna corruzione dei token GT** nel training: Locodec/StableAR (2607.29363) corrompono i prefix GT con rumore (ELD input corruption) proprio per simulare il drift dell'inference.
- Il nostro repetition-aware/nucleus fallback è un palliativo all'inferenza; la letteratura interviene a monte (training).

---

## 2. Design raccomandato (evidence-based)

### 2.0 Ruoli dei componenti (chiarimento architetturale, 2026-08-24)

Domanda: "il cosa dire a livello semantico non dovrebbe venire da Gemma? Allora l'AR Head è un altro adapter tra gli hidden di Gemma e il codebook semantico del subtalker?"

**Risposta: SÌ — l'AR Head è concettualmente l'ADAPTER SEMANTICO.** Il contenuto viene da Gemma; l'AR head lo realizza nella sequenza discreta cb0 che il sub-talker congelato esige in ingresso. Non è un cervello, è un traduttore. La catena completa:

```
Gemma (cosa dire, a livello di significato)
  → hidden 1536
  → TemporalResidualAdapter = ADAPTER ACUSTICO (proietta in spazio talker 2048,
      addestrato per i RESIDUI cb1..15 con cb0 GT) → h_t
  → AR-SemanticDecoder = ADAPTER SEMANTICO (mappa contenuto di Gemma via h_t
      → sequenza discreta cb0, 2048 classi) → cb0
  → Sub-Talker MTP (congelato) → residui → Code2Wav → voce
```

Perché l'AR head NON è solo un'altra proiezione per-frame:
1. **È autoregressivo**: cb0 è una sequenza con struttura fonotattica (12.5Hz); i probe mostrano per-frame 3.7% vs con storia codici 12% → il contesto temporale è necessario.
2. **È generativo**: all'inferenza cb0 va campionato dalla distribuzione, non mappato.
3. **Esce in spazio discreto** (2048 classi).

Perché è scarso (top-1 12%) pur avendo il contenuto in Gemma: il canale h_t è strozzato — l'adapter acustico è stato addestrato per i residui (cb0 GT), non per conservare l'info semantica (probe: 3.7% per-frame). In più il mapping testo→cb0 è intrinsecamente non uno-a-uno (2509.17021) → il top-1 esatto ha un tetto; conta la distribuzione campionata.

**Conseguenza di design**: il fix non è "renderlo più intelligente" ma **riaprire un canale diretto Gemma→cb0**:
1. **Ancora testuale (P3)**: condizionare l'AR head sugli embedding del testo generato da Gemma (inner monologue, Moshi; nel backbone originale Qwen3-TTS cb0 è predetto da text embeddings). È l'unico predittore semantico in letteratura senza condizionamento testuale → il nostro gap principale.
2. **cb0-pressure sull'adapter (P4)**: ri-addestrare l'adapter con loss semantica ausiliaria (SemBridge) perché h_t conservi il contenuto.
3. **Metrica**: non top-1 (tetto intrinseco) ma qualità del parlato campionato (blind ASR + mel cos-sim).

Tabella ruoli: Adapter = proiezione di spazio (✅); AR head = predittore semantico cb0, adapter semantico (⚠️ collo); Sub-Talker = voce (✅ congelato); TurnHead = turn-taking, va ripensata come token di controllo nel prompt di Gemma (DuplexCascade) o stream separato (Moshi), non classificatore esterno (❌ v5 collassato).

### 2.1 Architettura target (full-duplex, requisiti utente)

```
┌─────────────────────────────────────────────────────────────┐
│  Gemma-4-E2B (congelata o QLoRA leggera) — ragionamento,     │
│  tool calling, testo (inner monologue)                       │
│  INPUT: audio utente (via encoder Gemma) + testo/storia      │
│  OUTPUT: hidden states h_t + testo di risposta T             │
└──────────────────────────┬──────────────────────────────────┘
                           │ h_t
              ┌────────────▼────────────┐
              │  Adapter temporale v3   │  ← RITRAIN con
              │  h_sub = Ad(h_t)        │    cb0-pressure
              └────────────┬────────────┘     (loss ausiliaria
                           │ h_sub             semantica, stile
              ┌────────────▼────────────┐     SemBridge)
              │  Predittore cb0 (AR)    │  ← hybrid training
              │  condizionato su:       │    TF+free-running
              │   • h_sub               │    + input corruption
              │   • testo T (ancora     │    + EOS-adaptive
              │     semantica, Moshi)   │
              │   • storia cb0          │
              └────────────┬────────────┘
                           │ cb0
              ┌────────────▼────────────┐
              │  Sub-Talker Qwen3-TTS   │  ← congelato
              │  (MTP residui cb1..15)  │    (voce Qwen)
              └────────────┬────────────┘
                           ▼
                     Code2Wav decoder → audio streaming
```

**Elementi nuovi rispetto all'architettura attuale (tutti evidence-based):**

1. **Ancora testuale (inner monologue)** — Moshi (2410.00037): *"Inner Monologue... jointly model text and audio tokens... significantly improves factuality and linguistic quality of generated speech"*; Mini-Omni Reasoner: thinking-in-speaking con token di ragionamento interleavati (rapporto 2:8). **Gemma genera (o ha già generato) il testo della risposta**: condizionare il predittore cb0 anche sugli embedding/estratto di T è il modo più economico di dare informazione semantica a cb0 — il testo è quasi deterministico per il contenuto, cb0 specifica la realizzazione. In pratica: `cb0_pred = AR(h_sub, text_ctx, cb0_history)`.
   - Questo risolve il paradosso della tolleranza: gli errori sistematici del generatore libero distruggono il contenuto; con l'ancora testuale il contenuto è già fissato dal testo → errori cb0 diventano errori "acustici" (tollerati, mel cos 0.92-0.99), non semantici.

2. **Hybrid training del predittore cb0** — 2509.17021: alternare TF e free-running con **prompt protection** (primi N token sempre GT) e **EOS-adaptive** (se EOS predetto male → meno iterazioni libere). Senza gate sull'accuracy. Costo: 1.5× del TF-only. Risultato atteso: il gap TF/free si chiude (WER −30% su CosyVoice2).

3. **Input corruption** — 2607.29363 (Locodec/MP-ELD): *"to mitigate exposure bias, during training the ground-truth prefix tokens are perturbed with small noise before being fed to the encoder"*. Costo: zero. Applicabile al nostro AR decoder: corrompere con probabilità ε i token cb0 GT nel prefix (sostituzione con campione dalla distribuzione predetta o token casuali).

4. **cb0-pressure sull'adapter (ritrain)** — SemBridge (2608.07462): *"SemBridge uses discrete semantic tokens to directly supervise autoregressive LM states... semantic supervision is used only during training"*. → Ritrainare l'adapter con loss ausiliaria di predizione cb0 (o allineamento WavLM) per re-iniettare l'informazione semantica in h_sub. **Questa è la risposta diretta al finding dei probe (h_sub ≈ senza info cb0).**

5. **Opzione B (backbone Qwen3-TTS come predittore cb0)** — confermata dalla letteratura: Qwen2.5-Omni usa esattamente questo pattern (Talker separato che riceve hidden del Thinker); DS2ST-LM (2601.16023) valida le proiezioni (Linear/Conv1D-Linear/Q-Former) per mappare spazi. MA il nostro test di transfer diretto (0.08%) dimostra che serve una **proiezione addestrata**, non il codec_head preaddestrato. → Se dopo (1)-(4) il top-1 resta ≤15%, questa diventa la strada: proiezione h_sub→spazio backbone addestrata + backbone 1.7B congelato + codec_head.

6. **Full-duplex / turn-taking** — dal survey full-duplex (2606.19453) e DuplexCascade (2603.09180): special tokens di controllo (`<user_is_speaking>`, `<barge_in>`, `<yield>`, `<backchannel>`) addestrati con loss pesata (peso 10 sugli eventi rari); Moshi: multi-stream (user audio + agent audio + text) in un depth transformer. La nostra pipeline (Gemma + adapter + cb0 + sub-talker) è **cascaded-duplex**: il turn-taking va gestito a livello di sistema (VAD semantico/DM) e di condizionamento (storia interleavata). Per il primo rilascio: special tokens nel prompt di Gemma + policy di stato (IDLE/LISTEN/SPEAK/DUAL) — vedi survey L0-L3.

### 2.2 Cosa NON rifare (evidenze contrarie)

- **Non** predire tutti i 16 codebook con il decoder piccolo (flattening → instabile, Mini-Omni; error accumulation, Qwen3-TTS report). Il sub-talker MTP congelato è la scelta giusta.
- **Non** sostituire cb0 con token semantici puri ultra-compressi (<12.5Hz): WER Trap dimostra il collasso generativo.
- **Non** fare QLoRA su Gemma per predire cb0 direttamente: rompe il vincolo "preservare ragionamento/tool calling" e nessuna evidenza lo supporta per il nostro setup (l'opzione C è la meno fondata).
- **Non** ripetere il transfer diretto del codec_head senza proiezione addestrata (già falsificato: 0.08%).

### 2.3 Priorità di esecuzione (Task 12 → release full-duplex)

| # | Azione | Evidenza | Costo | Impatto atteso |
|---|---|---|---|---|
| 1 | **Input corruption** sul training AR (ε=0.1-0.2 sui prefix) | 2607.29363 | ~0 (2 righe) | chiude parte del gap TF/free |
| 2 | **Hybrid training** (TF+free-running, prompt protection, EOS-adaptive), no gate | 2509.17021 | medio (loop di training) | WER −30% rel (letteratura) |
| 3 | **Ancora testuale**: condizionare AR su testo T di Gemma | 2410.00037, 2508.15827 | medio (cache testo + embedding) | il fix strutturale per "free ≈ ancorato" |
| 4 | **cb0-pressure**: ritrain adapter con loss semantica ausiliaria | 2608.07462 + nostri probe | alto (retrain adapter) | ripristina l'info cb0 in h_sub |
| 5 | Se top-1 < 15% dopo 1-4: **Opzione B** (proiezione addestrata + backbone 1.7B) | 2503.20215, 2601.16023 | alto | ceiling di capacità |
| 6 | **Turn-taking**: special tokens nel flusso Gemma (NON TurnHead esterna) | 2603.09180, 2606.19453 | medio | backchannel/barge-in/yield/interrupt |

### 2.4 DECISIONE: niente TurnHead esterna — token di controllo nel flusso (2026-08-24, rev.2)

**Decisione**: abbandonare TurnHead (classificatore esterno su feature grezze, v3-v5
collassati: bACC 41%, YIELD/BARGE_IN collapse) in favore di token di controllo nel
flusso generativo.

**REV.2 — i token NON vanno nel vocabolario di Gemma** (Gemma è congelata per
preservare ragionamento/tool calling; estendere il vocabolario richiederebbe retraining
di embedding+lm_head, stile Mini-Omni2 vocab 181k). I token di controllo vivono nel
componente ADDESTRABILE:
- (a) **nel vocabolario del predittore cb0**: alterna codebook e token di controllo
  (`<silence>`, `<backchannel>`, `<yield>`) nello stesso flusso AR — ricetta 4 del
  survey (silent-token augmentation, OmniFlatten `silent_speech_token`);
- (b) **marcatori testuali BPE esistenti** generati da Gemma ("[backchannel]",
  "[yield]") senza estensione del vocabolario (in-context), tradotti in pattern
  acustici dal predittore cb0 addestrato su coppie testo+marcatore→cb0.

Evidenze:
1. **DuplexCascade (2603.09180)**: turn-taking = special tokens nel flusso LLM con
   loss pesata (10× eventi rari). Il controllo del turno È il modello, non un modulo.
2. **Moshi (2410.00037)**: multi-stream (testo+audio utente+audio agente); la decisione
   di parlare/cedere è implicita nel flusso generativo.
3. **Survey full-duplex (2606.19453)**: i sistemi evoluti implementano la decision
   state machine (IDLE/LISTEN/SPEAK/WAIT/DUAL) dentro il modello; silent-token
   augmentation esprime T5 a livello di token senza modificare l'audio.
4. **Prova negativa interna**: TurnHead v3-v5 — un classificatore su mel/z 2304-dim
   non vede il contesto linguistico/pragmatico dove vive la decisione di turno.

Perché è il pattern giusto nel nostro design:
- Gemma (congelata) genera testo: i marcatori testuali (b) sono in-context, zero retraining.
- Il predittore cb0 (addestrabile) è il punto di sintesi: token di controllo (a) o
  traduzione dei marcatori (b) → pattern acustici (silenzio, backchannel breve).
- La loss pesata (DuplexCascade 10×) risolve lo sbilanciamento che ha ucciso v4/v5.
- Si integra con l'ancora testuale (P3): testo → token di turno → cb0 nello stesso flusso.

**REV.2 — DuplexConv NON addestra le dinamiche italiane** (è cinese; J-Moshi §5:
le statistiche di turn-taking variano per lingua: overlap GT JP 8.1s/min vs EN 3.3).
Ruolo ridotto a: (1) riferimento di FORMATO/etichette GT (VAD frame-level + turn state)
per costruire/validare il nostro pipeline di etichettatura; (2) eventuale detector
acustico di eventi (riconoscimento di overlap/barge-in è lingua-indipendente).
Nessun training diretto dello stack su di esso. Il training delle dinamiche IT viene
SOLO dallo stereo IT sintetico (ricetta 1+2 survey: dual-role TTS + overlap injection
T∼N(0.6,0.4²), generato con Qwen3-TTS).

### 2.5 REV.3 — chi decide il QUANDO: il predittore cb0 come dialogue manager

**Problema sollevato**: (a) marcatori BPE in Gemma richiederebbero insegnarle QUANDO
produrli = retraining (vietato); (b) token nel vocabolario del solo predittore =
Gemma "cieca" alle dinamiche?

**Soluzione**: il predittore cb0 NON è cieco — riceve gli hidden h_t di Gemma che
contengono la comprensione completa del contesto dialogico frame-per-frame (audio
utente + testo + storia). È il pattern Qwen2.5-Omni (Talker ← hidden del Thinker)
e Moshi (RQ-Transformer ← hidden del backbone).

Divisione dei ruoli (definitiva):
| Funzione | Chi la fa | Come |
|---|---|---|
| Capire il contesto (utente parla? interrompe? pausa?) | Gemma (congelata) | hidden h_t frame-per-frame |
| DECIDERE il timing (backchannel/yield/barge-in/silence) | Predittore cb0 (addestrabile) | token di controllo nel suo vocabolario, condizionati su h_t |
| Dire il contenuto | Gemma | testo → ancora testuale (P3) |
| Suonare (voce IT) | Sub-talker (congelato) | — |

Gemma non genera i token di turno: **li autorizza con i suoi hidden**. Il predittore
cb0 è il dialogue manager implicito del sistema (stato IDLE/LISTEN/SPEAK/WAIT/DUAL
implementato come transizioni tra token di controllo).

### 2.6 Pipeline dati a 3 livelli (definitiva)

| Livello | Dato | Ruolo | Uso |
|---|---|---|---|
| 1. Fonetica IT | **MDC26/CV26** (mono, reale) | lingua/fonetica italiana per adapter + predittore cb0 | pre-training predittore (IN CORSO v3) + adapter (fatto) + ancora testuale P3 |
| 2. Dinamiche IT | **Stereo IT sintetico** (da creare: contenuto SerFabio89 → sintesi dual-role Qwen3-TTS → overlap injection → GT by construction) | dinamiche full-duplex italiane + etichette turno perfette | fine-tuning predittore cb0 (token di controllo) |
| 3. Timing/struttura | **SmoothConv** (HF qualialabsAI, 100h ZH stereo, **annotazione UMANA**: turn complete/incomplete/backchannel, speech event, other_turn, timings float) — campione 5-10h | **partitura temporale reale** (overlap/backchannel/gap) per lo stereo IT sintetico + statistiche timing | time-structure transfer: griglia temporale reale riempita con voce IT (Qwen3-TTS) |

**Selezione minimale (rev. 2026-08-25)**: i soli dati necessari sono **MDC26/CV26
(già in uso, nessun download)** + **SerFabio89 SFT (283MB, contenuto)** +
**SmoothConv campione 5-10h (partitura temporale reale)**. DuplexConv (5TB) NON
si scarica: SmoothConv (100h, annotazione umana migliore) basta per statistiche
e template. Nexdata/MagicHub esclusi (stile opzionale).

**Perché SmoothConv sì (rev. utente 2026-08-25)**: il timing del turn-taking è
STRUTTURA TEMPORALE UNIVERSALE (cross-lingua): la forma di un overlap (quando
inizia, quanto dura, come si risolve) e dei backchannel non-lessicali è un
fenomeno acustico/fisico, non linguistico. J-Moshi mostra che NON trasferiscono
le STATISTICHE quantitative (frequenza per lingua: JP 8.1s/min vs EN 3.3s), ma
per l'italiano non esistono dati reali → i pattern reali cinesi battono la
gaussiana sintetica (MinMo T∼N(0.6,0.4²)). Uso = **time-structure transfer**:
griglia temporale di SmoothConv riempita con voce italiana Qwen3-TTS → overlap
strutturalmente reali nel sintetico. NIENTE training diretto del predittore cb0
su SmoothConv (cb0 cinesi = fuori dominio fonetico).

**MDC26 resta necessario** (livello 1: fonetica reale, varietà prosodica umana che il
sintetico non ha). Il sintetico NON lo sostituisce: lo completa con le dinamiche.

### 2.7 Ricognizione dataset conversazionali italiani (2026-08-25)

Esito: **NON esiste stereo italiano pubblico** (conferma survey 2606.19453: dati
Type C pubblici ≈ 5.000h, dominati da inglese). Censimento:

| Dataset | Lingua/Formato | Dimensione | Accesso | Uso |
|---|---|---|---|---|
| **Kaggle nexdatafrank/italian-spontaneous-dialogue-telephony-speech** | IT, mono 8kHz, trascrizioni con disfluenze ([S] silenzio, [N] rumore, [T]) | sample 118KB (7 clip); completo commerciale ~120GB | gratuito solo sample | riferimento di STILE parlato spontaneo per la riscrittura; NON training |
| **MagicHub ASR-ItaCSC (italian-conversational-speech-corpus)** | IT, mono, 28 conversazioni (3 coppie parlanti), trascrizioni | 10.43h, 785MB | commerciale (login) | riferimento di stile; NON training |
| **HF UniDataPro/ud-nlp "499h telephone dialogues"** | IT (asserito) | CSV placeholder 195B, path inesistenti | pubblico | **FAKE/honeypot: scartare** |
| **HF SerFabio89/italian-open-sft-chat-dataset** | IT testuale, multi-turno | 55.439 dialoghi (train 283MB parquet) | pubblico CC-BY-NC | **contenuto testuale per lo stereo IT sintetico (livello 2)** |
| **DuplexConv** (HF qualialabsAI) | ZH stereo, LLM-annotated | 2.000h, ~5TB | pubblico CC-BY-NC | statistiche timing + schema annotazione + detector acustico (livello 3) |

Conclusione operativa: lo stereo IT sintetico si costruisce da **SerFabio89 (contenuto)
+ stile Nexdata/MagicHub (disfluenze) + statistiche DuplexConv (timing) + Qwen3-TTS
(sintesi dual-role)**. I dataset commerciali italiani non sono né necessari né
scaricabili senza licenza.

### 2.8 Costruzione dello stereo IT sintetico (dettaglio)

**Osservazione critica**: i dialoghi testuali italiani (SerFabio89) sono
HALF-DUPLEX: turni puliti senza interruzioni/overlap/backchannel. Il contenuto
testuale da solo NON insegna le dinamiche → le dinamiche vanno INIETTATE come
strato di sintesi (survey 7.4), non cercate nel testo.

Pipeline (ricetta 1+2+3 del survey, adattata):
1. **Script di contenuto**: dialoghi SFT IT + riscrittura stile parlato (disfluenze
   "e, eh...", marcatori [S] silenzio / [N] rumore — stile dal campione Nexdata).
2. **Iniezione dinamiche a livello di segmento**:
   (a) turni puliti (T1);
   (b) BACKCHANNEL agente: clip brevi "mh-mh / sì / capito / ok" inserite DURANTE
       il turno utente (T3·I2);
   (c) BARGE-IN agente: onset anticipato vs fine turno utente (T3·I4), overlap
       injection con T∼N(0.6, 0.4²) (distribuzione MinMo, survey 7.4);
   (d) YIELD agente: l'utente inizia durante il turno agente → l'agente si ferma
       (taglio + silenzio);
   (e) OVERLAP sostenuto (T4): pochi casi, cella più difficile (nessun dato
       pubblico la copre — survey Tab.6).
   Le STATISTICHE di timing (rate backchannel/min, durate overlap, distribuzione
   inter-turn) vengono da DuplexConv (dinamiche vocali lingua-indipendenti).
3. **Sintesi fisica**: Qwen3-TTS dual-role su 2 canali con 2 voci diverse
   (utente = voce X, agente = voce Qwen; voice cloning 3s) → STEREO con
   sovrapposizioni REALI (i canali si sommano dove c'è overlap).
4. **GT by construction**: turn state frame-level USER/AGENT/BOTH/NONE + etichette
   evento (backchannel, barge-in, yield, overlap) → supervisione perfetta per i
   token di controllo del predittore cb0 (niente euristiche stile TurnHead v4).
5. **Selezione WER** (J-Moshi 3.4: 10 campioni → scegli il migliore; WER 24.6%).

**Cosa impara il modello**: non le dinamiche "dal testo", ma a PRODURRE backchannel/
barge-in/yield nei casi sintetici dove l'AGENTE li compie (scenari b/c/d), con GT
frame-level by construction.

---

## 3. Bibliografia essenziale (tutti letti in questa sessione)

1. **Moshi** — Défossez et al., 2410.00037 (inner monologue, Mimi, multi-stream full-duplex)
2. **Qwen3-TTS Technical Report** — 2601.15621 (cb0 WavLM-semantico + MTP; motivazione anti error-accumulation) — già in docs/
3. **GLM-4-Voice** — Zeng et al., 2412.02612 (tokenizer semantico supervisionato 12.5Hz + LLM + flow matching)
4. **Mini-Omni** — Xie et al., 2408.16725 (text-instructed delayed parallel, SNAC) — già in docs/
5. **Mini-Omni2** — 2410.11190 (instabilità loss audio, token variabili ai bordi) — già in docs/
6. **Mini-Omni Reasoner** — 2508.15827 (thinking-in-speaking 2:8) — già in docs/
7. **Fun-Audio-Chat** — 2512.20156 (semantic tokens + flow matching, grouping 25→5Hz)
8. **CaT-TTS / S3Codec** — 2509.22062 (dual LM: semantic coarse + acoustic fine; split-RVQ con distillazione semantica)
9. **Bridging the gap (exposure bias in LM-TTS)** — 2509.17021 (ibrido TF+free-running, prompt protection, EOS-adaptive; accuracy intrinsecamente bassa; WER 6.23→4.21)
10. **Stable AR / Locodec + MP-ELD** — 2607.29363 (input corruption, error accumulation, CFG drift)
11. **SemBridge** — 2608.07462 (semantic-token anchoring training-only)
12. **The WER Trap** — 2605.29209 (token semantici puri → collasso generativo; decoupling semantico/acustico)
13. **XY-Tokenizer** — 2506.23325 (conflitto semantico-acustico nei codec low-bitrate)
14. **DS2ST-LM** — 2601.16023 (proiezioni Linear/Conv1D-Linear/Q-Former per mappare spazi)
15. **Qwen2.5-Omni** — 2503.20215 (Thinker→Talker: modello separato che riceve hidden del Thinker)
16. **DuplexCascade** — 2603.09180 (special tokens micro-turn, loss pesata) — già in docs/
17. **Survey Full-Duplex (L0-L3, T×I×R)** — 2606.19453 — già in docs/
18. **Luna-TTS** — 2608.11593 (critica AR codec-LM: error accumulation, latency)
19. **SemaVoice** — 2605.16964 (mismatch semantico-prosodico nelle rappresentazioni continue)

---

## 4. Nota metodologica

- Tutti i paper citati sono stati letti (fulltext o context-search) in questa sessione con i tool arxiv (search/get/fulltext/context). I numeri (WER, accuracy, frame rate) sono riportati come nei paper.
- I nostri dati sperimentali (top-1 12%, probe 3.7/3.2%, bix 6.6, transfer 0.08%, mel cos 0.92-0.99, 2024/2048 codici) provengono dai run del progetto (log `logs/`, `cache_ars_full.pt`, checkpoints).
- Documento di lavoro: verrà aggiornato con i risultati dei prossimi passi (ADR quando si sceglie la direzione definitiva).
