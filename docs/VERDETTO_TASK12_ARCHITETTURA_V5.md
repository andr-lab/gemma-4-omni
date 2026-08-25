# VERDETTO TASK 12 — Architettura v5: proiezione addestrata nel loop (punto architetturale FISSATO)

**Data**: 25 agosto 2026 | **Stato**: DECISIONE ARCHITETTURALE (fissata con evidenze sperimentali + letteratura)

## Sintesi della decisione

**Il sub-talker (MTP) non viene usato isolato.** Il runtime full-duplex usa il **talker Qwen3-TTS completo in loop** (backbone + codec_head + MTP nativi, tutti congelati), condizionato dagli hidden di Gemma attraverso una **proiezione addestrata** (l'unico componente addestrabile). L'AR decoder per cb0 e il coupling statico dell'adapter vengono **abbandonati** come via principale.

```
Gemma-4-E2B (congelata) — ragionamento, tool calling, turn-taking (token di controllo)
  │  hidden h_raw (1536)
  ▼
[PROIEZIONE ADDESTRATA h_raw → spazio backbone]  ← UNICO componente addestrato
  │  h_proj (2048, formato past_hidden/condizionamento per frame)
  ▼
Backbone Qwen3-TTS in loop (CONGELATO, pattern nativo)
  ├─ codec_head(hidden) → cb0
  ├─ MTP([past_hidden, emb(cb0)]) → residui cb1..15   (protocollo generate ufficiale)
  └─ backbone ri-elabora i codici di ogni frame → past_hidden evolve (hidden-state synchronization)
  ▼
Code2Wav → audio streaming (97-101 ms primo pacchetto)
```

## Evidenze sperimentali (25 agosto 2026, tutte con blind ASR)

| # | Test | Setup | Esito |
|---|------|-------|-------|
| 1 | NATIVE | `teacher.generate(text)` — talker completo in loop nativo | **ASR esatto** su 3 testi ("L'impianto era infine dotato di un fabbric.", "La verdura è anche estremamente popolare a Taiwán.", "Successivamente cercò.") |
| 2 | LOOP A | Backbone in loop con codici GT (cache) → cattura `past_hidden` → **MTP isolato** (`sample_acoustic_codes` v2) da [h, cb0] | **Parlato fluente** ("Sto in un studio!...", "Anna, te ti senti qua?...") → il MTP PUÒ generare dal solo cb0 quando h è il past_hidden ricorrente del loop |
| 3 | LOOP B | h_sub = adapter_temporal2_1000(h_raw Gemma) iniettato nel loop al posto del testo | **Elica** ("Sottotitoli a cura di QTSS" = allucinazione whisper su rumore) → l'adapter non allinea abbastanza (cos 0.51-0.63) |
| 4 | oracle (23 ago) | MTP isolato + teacher h_t (estrazione statica, passata unica) | Contenuto riconoscibile ma fonetica garbled → il formato h statico è fuori distribuzione per il MTP |
| 5 | pred (23 ago) | MTP isolato + h_sub adapter | Elica |
| 6 | N7 (23 ago) | adapter + cb0..7 GT + residui 8-15 predetti | **ASR quasi esatto** → l'adapter basta per i residui alti, non per i bassi |

## Perché questa architettura (catena logica)

1. **Il MTP sa generare dal solo cb0** (evidenza 2 + paper Qwen3-TTS: "the backbone ingests aggregated codebook features to predict the zeroth codebook, and an MTP module then generates all residual codebooks"). Il fallimento storico (residui random, acc 0.002) NON era il MTP: era **h sbagliato** (formato statico o spazio non allineato).

2. **Il formato di h che il MTP esige è il past_hidden del loop nativo**: hidden del backbone DOPO aver ri-elaborato i codici del frame precedente (riga 1740 modeling_qwen3_tts.py: `past_hidden=hidden_states[:, -1:, :]`), con testo aggiunto per frame (riga 1689-1692: `inputs_embeds = codec_hiddens.sum(1) + trailing_text_hidden[t]`). L'estrazione statica (passata unica) produce h fuori distribuzione → garbled (evidenza 4).

3. **h_sub dell'adapter (cos 0.51-0.63) non basta** per guidare i residui bassi 1-7 (evidenza 3,5,6). Il tetto ~0.63 del coupling cos-MSE è un limite informativo del metodo: la proiezione va addestrata **nel loop** (con loss sul parlato/codici), non con allineamento statico di rappresentazioni.

4. **Letteratura convergente**:
   - PRIME-Speech (2606.30944): backbone congelato + talker che consuma **hidden intermedi che evolvono nel loop** (hidden-state synchronization); senza sincronizzazione WER esplode (39→65, 10→129).
   - Qwen2.5-Omni (2503.20215): Talker AR dedicato addestrato **joint** col Thinker (3 stage); riceve hidden + token testo (hidden = semantica, token = fonetica).
   - Freeze-Omni: decoder su hidden di LLM congelato → **richiede retraining dei componenti speech per ogni LLM** (critica esplicita in LLMVoX 2503.04724).
   - LLMVoX (2503.04724): TTS 30M LLM-agnostico condizionato sul **testo streaming** (non hidden) — alternativa valida ma perde il canale hidden.
   - Moshi (2410.00037): Depth Transformer con condizionamento additivo z_s + embedding; inner monologue = 3× qualità (rimuovere il testo crolla 9.2→26.6/62.3/22.8).

## Componenti del runtime v5

| Componente | Stato | Note |
|---|---|---|
| Gemma-4-E2B | 🔒 congelata | ragionamento, tool calling, turn-taking via token di controllo nel suo flusso |
| Proiezione h_raw→spazio backbone | 🎯 **ADDESTRATA** (unica) | dimensione da decidere (~1-5M); addestrata nel loop con CE sui codici del talker (o loss sul parlato) |
| Backbone Qwen3-TTS (1.7B) | 🔒 congelato | predice cb0 (codec_head) + guida il MTP |
| MTP / code_predictor | 🔒 congelato | genera residui dal solo cb0 (protocollo generate ufficiale) |
| Code2Wav | 🔒 congelato | voce italiana pronta |

## Cosa si ABBANDONA (con motivazione)

- **AR decoder cb0** (trainer v2/v3/v4): il codec_head del backbone predice cb0 nativamente nel loop. I risultati TF 7-12% erano su cache contaminata (adapter mai caricato) e comunque il collo era h_sub.
- **Coupling cos-MSE statico** dell'adapter: tetto informativo ~0.63; non scala.
- **sample_acoustic_codes / MTP isolato**: va usato solo DENTRO il loop del backbone (il loop A lo dimostra: funziona solo con past_hidden ricorrente).
- **TurnHead esterno**: invariato — il turn-taking vive nel flusso del predittore/token (decisione precedente).

## Prossimi passi (Task 13+)

1. **Progettare la proiezione**: input h_raw (1536) per frame → output (2048) nel formato `past_hidden` o additivo per frame; architettura candidata: MLP a 2 layer + LayerNorm (come DS2ST-LM projections: Linear/Conv1D-Linear/Q-Former).
2. **Loss**: CE sui 16 codebook del talker (o solo cb0+residui alti) con teacher forcing nel loop; oppure distillazione verso past_hidden del teacher (ma il loop A mostra che la CE diretta è più promettente).
3. **Dati**: MDC26 (fonetica IT) + 16 dialoghi sintetici (turn-taking GT) + SerFabio89 (contenuto).
4. **Smoke test**: 2-4 batch su 3090, gate ASR su 3 finestre con contenuto.
5. **Verifica full-duplex**: Gemma genera → h_raw → proiezione → talker in loop → voce; turn-taking via token.

## Note metodologiche

- Gate percettivo: blind ASR (faster-whisper small) su finestre selezionate con contenuto (`_mel_active_mask` del trainer faithful) — le finestre brute possono essere silenzio.
- La verifica "N7" (23 ago) resta il riferimento di qualità dell'adapter per i residui alti: utile come sanity check futuro, non come percorso.
- Il test LOOP A è un esperimento nuovo (nessun paper pubblicato lo aveva eseguito: gap dichiarato dalla review arXiv).
