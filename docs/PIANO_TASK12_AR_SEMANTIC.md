# PIANO COLMATURA GAP — Predizione libera cb0 (Task 12)

**Data**: 23 agosto 2026 | **Base**: architettura canonica consolidata (ARCHITECTURE_STATE.md)

## Il gap (unico frontier rimasto)
Con la ricetta temporale fedele validata:
- ✅ Ancoraggio GT cb0..cb7 + residui predetti → italiano corretto e comprensibile
- ❌ **cb0 predetto libero** (testa semantica one-shot argmax/sampling) → silenzio/loop/allucinazioni

## Diagnosi strutturale (perché fallisce)
La testa semantica predice ogni cb0 **indipendentemente**, un frame alla volta, da stati continui.
Ma p(cb0_t | contesto) è intrinsecamente sequenziale: senza condizionamento sui cb0 precedenti
generati, le decisioni per-frame sono incoerenti tra loro → l'argmax amplifica in loop/silenzio.
Questo è il classico limite dei decoder paralleli (NAR) sulla parte semantica.

## Fondamento nella letteratura locale (docs/sources)
1. **Qwen3-TTS Technical Report** (§architettura): il talker nativo genera cb0 in modo
   AUTOREGRESSIVO con il proprio LM, poi il code predictor produce cb1..15 condizionati ai
   livelli precedenti (chain MTP). Il contenuto linguistico vive interamente nel percorso AR.
2. **VALL-E / VALL-E 2 pattern**: primo codebook AR (coerenza), restanti NAR/paralleli
   (qualità acustica). È esattamente la nostra struttura: cb0 = "primo codebook".
3. **Moshi**: stream semantico generato AR dal proprio RQ-Transformer.
4. **Bengio et al. 2015, Scheduled Sampling**: mitiga l'exposure bias del teacher forcing
   addestrando con mischie di input GT e predetti.
5. **FireRedTTS3**: regularizzazione semantica congelata (già adottata parzialmente col
   teacher h_t) + mitigazione error accumulation a livello rappresentazione.

## DECISIONE (massima probabilità di successo)
**Testa semantica autoregressiva (VALL-E-style) con scheduled sampling**, sopra l'architettura
canonica intoccata:

```
h_t (TemporalResidualAdapter, congelato dopo il Task 11I-bis)
  │ sequenza (T, 2048)
  ▼
AR-SemanticDecoder (NUOVO: 4 layer causali, d=1024, 8 heads)
  input passo t: [emb_cb0_{<t} ; h_t_{≤t}]  (causale)
  output: logits cb0_t (2048)
  ▼
cb0_1..T generati step-by-step (greedy/top-k+temperature)
  ▼
Sub-talker frozen (come oggi) → cb1..15 → Code2Wav ufficiale
```

### Perché questa e non le alternative
- **Talker nativo completo (opzione A pura)**: richiede innestare il contesto Gemma negli
  embedding del talker AR (interfaccia complessa, rischio alto, poco controllo).
- **Solo scheduled sampling sull'one-shot attuale**: non risolve la mancanza di coerenza
  sequenziale (il modello resta parallelamente fattorizzato).
- **AR head**: minimal-invasiva (l'adapter validato resta congelato), è il pattern VALL-E
  dimostrato, si addestra in pochi step su CV26 (40.8h), e il suo fallimento sarebbe
  diagnosticabile isolatamente.

### Protocollo di training (TDD)
1. Dataset finestre CV26 (stesso split del 11I-bis)
2. Teacher-forcing iniziale (CE su cb0 GT) fino a convergenza
3. Scheduled sampling: probabilità di usare cb0 predetto come input cresce 0→0.5
4. Gate intermedi: CE val, Top-5, coerenza (n-run ripetute su 100 finestre)
5. **Gate finale obbligatorio**: blind ASR su 8 generated liberi (nessun ancoraggio GT)
   + ascolto utente. Criterio: contenuto coerente con la finestra (anche parafrasi acustica).

### File
- `tools/train_ar_semantic.py` (nuovo, autonomo)
- Output: `checkpoints/ar_semantic_final.pt`
- Report: `docs/REPORT_TASK12_AR_SEMANTIC.md`

## TurnHead v5 (tracciato separato, post-voce)
Ridisegno feature: sostituire lo spettro medio/frame con embedding dal backbone
(stessi hidden del forward esistente) prima di ripetere il training bilanciato.
EOF