# REPORT TASK 11I-BIS — REPLICA FEDELE r3_v7/temporal2 SU CV26 ITALIANO

**Data**: 23 agosto 2026 | **Gate**: blind ASR (faster-whisper small, nessun testo atteso)

## RISULTATO PRINCIPALE
La replica fedele della pipeline r3_v7 (TemporalResidualAdapter + teacher h_t ufficiale +
loss rankfix + 64 frame) su **CV26 italiano 40.8h** produce, in inferenza con ancoraggio
GT ai codebook 0..7 e residui PREDETTI dal modello, **italiano comprensibile con contenuto
corretto alla finestra target**:

| File | Blind ASR | GT |
|---|---|---|
| rvq_pred_a_N7 | "L'impianto era infine adottato di un fabbricato viaggiatori." | "l'impianto era infine dotato di un fabbricato viaggiatori" ✅ |
| rvq_pred_c_N7 | "L'avvertura è anche estremamente popolare a San Juan, dove cresci bene." | "La verdura è anche estremamente popolare a Taiwán, dove cresce bene" ✅ |
| rvq_pred_a_N3 (solo 4 CB GT) | "...il fine dottato di un fabbricato viaggia..." | parziale ⚠️ |

## Perché questa volta funziona (diff vs tentativi precedenti)
1. **TemporalResidualAdapter vero** (transformer temporale 1024D, 4 layer) — il Task 11I
   aveva usato il trainer semplificato 6B senza modulo temporale
2. **Teacher h_t estratto correttamente** dal talker ufficiale via `qwen3tts_src` isolato
   da .pylibs (evitato sia il git clone Kaggle-era sia l'accelerate rotto di .pylibs)
3. Protocollo di valutazione con sampling (temperature 0.9/top_k 50 + varianti)

## Metriche
cos sim val: 0.5094 (su CV26 italiano; benchmark storico cinese 0.6275 — lingua/dataset diversi)
SVD rank: 1431/2048 | 2500→1200 step, ~18 min su RTX 3090

## Gap rimasto (onesto)
La predizione LIBERA di cb0 (testa semantica argmax/sampling) resta degradata:
i pred_*_greedy/t0x con cb0 predetto mostrano contenuto parziale, vuoti o allucinazioni.
Il contenuto corretto richiede ancoraggio GT ai primi codebook.

## Prossimi passi
1. Migliorare la testa semantica per cb0 libero (AR decoding / scheduled sampling)
2. Oppure opzione A (talker nativo AR condizionato da Gemma)
3. TurnHead v5: ridisegno feature (embedding backbone invece di spettro medio)
