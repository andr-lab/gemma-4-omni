# Diagnosi Audio Metallo / Incomprensibile — Report Finale

**Kernel**: `andrlab/gemma4-diag-v9` (COMPLETE 13:20, ckpt ep2 md5 `86a854e9` verificato)
**Data**: 2026-08-10 | **Metodo**: A/B isolation + statistiche + spettrogrammi + correlazione tempo

---

## Sintesi esecutiva

**Le metriche del report erano ingannevoli. Il problema esiste, è reale e grave, e la causa è stata isolata con certezza:**

1. **Il cb0 (semantico) predetto è quasi perfetto** — sostituirlo all'oracle cambia l'audio di pochissimo (RMS diff 0.004-0.022 vs totale 0.022-0.056).
2. **Tutta la distorsione viene dai codebook residui 1..15 generati dal sub-talker** (condizionato su h_t da Gemma). Sostituire cb0 con quello GT NON risolve: l'audio resta degradato.
3. **La correlazione temporale generated-vs-oracle è ~0.0** (win0 +0.017, win2 -0.004, win3 -0.045, win7 +0.001): i due segnali sono praticamente indipendenti.
4. **Il generated è ~65% più debole dell'oracle** (rms 33-44% dell'ampiezza GT): il sub-talker "smorza" il segnale.
5. **Il modello produce molto silenzio**: win005 (frase lunga) → 0s di audio; win001/006 (GT già silenzio) → corretto.

## Verdetto sulle 4 ipotesi dell'utente

| # | Ipotesi | Verdetto | Evidenza |
|---|---------|----------|----------|
| 1 | Discrepanze statistiche h_t (collasso/jitter/norm) | ✅ **CONFERMATA come causa principale** | I residui 1..15 (che dipendono solo da h_t) causano tutta la distorsione; cb0 ok. Le statistiche dei *codici* sono simili (mean/std ~±5%) ma la qualità percettiva crolla → h_t non riproduce il manifold del teacher per i codebook di dettaglio |
| 2 | Eval vs AR inference misalignment | ⚠️ **PARZIALE — non nel modo pensato** | `generate()` è un SOLO forward (non frame-AR), quindi l'exposure bias temporale non esiste. MA c'è exposure bias *nella catena RVQ*: il sub-talker in training riceve i codici GT (teacher forcing), in inferenza i propri campionati. Test A/B però dimostra che anche iniettando cb0 GT i residui restano sbagliati → la causa NON è il loop, è h_t |
| 3 | Incompletezza sub-talker (codebook mancanti/metadati) | ❌ **SMENTITA** | Tutti e 16 i codebook arrivano al decoder in entrambi i percorsi; GT validato (range 0..2047, no silenzio anomalo). Il decoder riceve informazione completa ma corrotta nei residui |
| 4 | Bug masking/posizionamento | ❌ **SMENTITA (probabile)** | Stesso allineamento in train e inferenza; val_loss/text/coupling coerenti (testo 0.18, coupling 0.49); se ci fosse off-by-one il testo sarebbe sbagliato, invece la semantica cb0 è quasi perfetta. Da escludere come causa del metallico |

## Evidenza dettagliata

### Check 1 — A/B isolation (il test decisivo)

4 sintesi per finestra, mescolando codebook GT/predetti:

| combinazione | cosa isola | risultato |
|---|---|---|
| a) GT completo | oracle | perfetto |
| b) cb0 GEN + residui GT | qualità cb0 | **diff 0.004-0.022** (quasi identico all'oracle!) |
| c) cb0 GT + residui GEN | qualità h_t→residui | **diff 0.023-0.055** (≈ totale, tutta la distorsione) |
| d) GEN completo | modello | diff 0.022-0.056 |

**Conclusione**: il colpevole è il **sub-talker che genera i codebook 1..15 da h_t**. cb0 è innocente.

### Check 2 — Statistiche

| win | mean_abs gen/gt | std gen/gt | sil0 gen/gt |
|---|---|---|---|
| 0 | 1045/1077 | 506/547 | 0/0 |
| 2 | 868/1051 | 558/585 | 0/0 |
| 5 | 1107/1054 | 456/488 | 0/0 |
| 7 | 920/1028 | 543/559 | 0/0 |

Le distribuzioni dei *codici* sono simili (~±5%, nessun collasso sul codice 0). **MA** questo è coerente col fatto che i codici hanno entropia marginale simile pur essendo percettivamente sbagliati: il problema è nella **relazione** h_t→residui, non nella statistica marginale.

### Check 3 — Teacher forcing

- `generate()` è one-shot: **non c'è loop temporale** → l'ipotesi "errore accumulato dopo 200ms" NON si applica al nostro design.
- Test A/B (c): anche con cb0 GT, residui degradati → il sub-talker produce residui scadenti **anche con input corretto** → h_t è il problema.

### Check 4 — Spettrogrammi

- Bande mel attive: generated 2-3% vs oracle 4-5% → **spettro "spento"**
- Temp-diff media: 0.008 vs 0.017 → **dinamica temporale dimezzata** (piatto, non "bande verticali")
- RMS: generated 33-44% dell'oracle

## Root cause (formulata con certezza)

**ADR-005 avvisava esattamente di questo**: il sub-talker Qwen è un renderer condizionale addestrato su h_t prodotto dal main talker di Qwen. L'adapter di Gemma (3072→2048) **non converge nel manifold di h_t del teacher** in modo sufficiente: la coupling loss CE (0.494) è una media su tutti i frame (dominati dal silenzio, facili) e non cattura la qualità dei frame parlati di dettaglio. Risultato: i codebook di dettaglio (RVQ alti) prodotti dal sub-talker sono "abbastanza giusti" per la loss ma percettivamente rovinati → audio metallico/incomprensibile, con ampiezza ridotta.

## Fix raccomandati (in ordine di impatto)

1. **LayerNorm sull'h_t post-adapter** + **temporal-derivative loss** (consigliato dall'utente per l'ipotesi 1): normalizza il manifold e penalizza il jitter frame-to-frame. Richiede nuovo training.
2. **Coupling loss pesata sui frame parlati**: la CE media è dominata dal silenzio; pesare i frame con energia per forzare la convergenza sui frame di dettaglio.
3. **Scongelare parzialmente il sub-talker** (loRA sui layer RVQ di dettaglio) — più invasivo, rompe la purezza ADR-005.
4. **Valutazione onesta**: sostituire mel-L1/MCD con (a) RMS diff tempo, (b) correlazione, (c) WER su ASR, (d) metriche solo sui frame parlati.

## File prodotti

- `diag_v9_out/data/ab_report.txt` — metriche A/B
- `diag_v9_out/data/stat_codes.txt` — statistiche distribuzione
- `diag_v9_out/samples/ab_*_*.wav` — 32 wav (4 combinazioni × 8 finestre) per ascolto diretto
- `diag_v9_out/plots/spec_*.png` — spettrogrammi

---

# AGGIORNAMENTO v13 (2026-08-10, kernel andrlab/gemma4-fixrapid-v13) — DISCRIMINAZIONE FINALE

## Check A — CE teacher-forcing per-frame (il test che mancava)

Il test chiave: la coupling loss del sub-talker in **teacher forcing** (tutti i codici GT in input, nessuna catena AR campionata), **splittata per frame parlato vs silenzio**.

| win | n_act/64 | CE parlato | CE silenzio |
|-----|----------|-----------|-------------|
| win0 | 6 | **7.00** | 2.64 |
| win2 | 33 | **7.00** | 5.06 |
| win3 | 8 | **6.84** | 2.03 |
| win4 | 13 | **7.25** | 2.77 |
| win7 | 26 | **6.06** | 4.03 |
| win1/5/6 | 0-4 | (nan/sil) | 0.45-0.63 |

**CE casuale attesa con vocab 2048 = 7.62.** I frame PARLATI hanno CE ≈ 6-7.25 ≈ **casuale**: il sub-talker, dato l'h_t di Gemma e i codici veri in teacher forcing, NON può predire i residui dei frame parlati meglio del caso. I frame di silenzio invece sono facili (CE 0.4-5, il silenzio è ripetitivo). 

### Interpretazione definitiva
- ❌ **NON è exposure bias** (la CE è pessima ANCHE in teacher forcing con codici veri)
- ❌ NON è sampling (temp=0 in tutti i test)
- ✅ **È l'h_t**: l'adapter `ht_adapter` (3072→2048, addestrato su 50h) proietta gli hidden di Gemma in uno spazio che il sub-talker congelato (addestrato su 5M ore) NON riconosce come il suo manifold h_t. Il sub-talker predice i frame parlati quasi a caso → il decoder Code2Wav riceve residui sbagliati → audio metallico.

## Check B — Match rate per codebook (gen vs GT, temp=0)
- cb1-5: 47.8%, cb6-10: 39.1%, cb11-15: 33.4% (win0) → peggiora coi codebook alti (dettaglio HF), coerente con CE crescente.

## Check C — Fix rapido LayerNorm su h_t a runtime: ❌ NON FUNZIONA
- rmsratio: baseline 0.41 → lrn 0.40 (nessun miglioramento)
- corr: baseline 0.02 → lrn 0.00 (nessun miglioramento)
- Il problema non è la scala/offset di h_t; è il contenuto informativo (il manifold).

## Check D — Metriche oneste (conferma)
- rmsratio baseline 0.22-0.41, corr ~0.00-0.02 (decorrelati) → confermano la diagnosi.

## CONCLUSIONE: il fix rapido NON esiste. Serve RIADDESTRAMENTO.
Il test v13 chiude il capitolo diagnostico. La strada è il training con:
1. **LayerNorm appresa sull'h_t** (non statica a runtime) o proiezione migliore nel manifold
2. **Coupling loss pesata sui frame parlati** (la CE media 0.49 era dominata dal silenzio facile; i frame parlati con CE≈7 vanno amplificati)
3. (opzionale) LoRA sul sub-talker per i codebook di dettaglio
