# Implementation Plan: P4 — V9 realmente incrementale

## Overview
Portare il Projector V9 dal solo forward batch a un percorso frame-by-frame
causale, mantenendo stato della CausalConv, KV per ciascuno dei quattro layer,
posizione RoPE e anchor testuale fissa. Il percorso batch resta invariato come
reference fino alla verifica di equivalenza. P4-T4 è un gate separato e
obbligatorio prima di QLoRA: se l'ablation causale Qwen fallisce, si interrompe
senza cambiare soglie.

## Contratto

```python
stream = IncrementalV9(projector=v9, causal_conv=causal_conv)
state = stream.new_state(batch_size=1, device="cuda", dtype=torch.bfloat16)
h_pfx_t = stream.step(h_raw_t, h_tx_anchor, state)  # (B, 2048)
```

- `h_raw_t`: un frame `(B, 1536)`; nessun frame futuro letto.
- `h_tx_anchor`: `(B, 1536)` fisso per la sequenza.
- `state`: possiede ultime 2 raw frame della CausalConv, KV `K/V` per layer,
  `rope_position`; ogni step commit una sola posizione.
- Output deve coincidere con l'ultima posizione del forward batch causal,
  entro le soglie BF16 (`MAE <= .01`, cosine min `>= .999`).

## Task ordinati

### Slice 1 — Contratto stateful e test RED/GREEN
- [x] Definire `IncrementalV9State` e `IncrementalV9Projector` in `duplex_kernel/models/v9.py`.
- [x] Implementare CausalConv window `(t-2,t-1,t)` con zero-left-padding e KV append per layer.
- [x] Testare shape, posizione, reset, input invalidi e nessun futuro letto con tiny model.
- [x] Testare equivalenza batch-vs-step con 100 sequenze reali del cache `cache_ars_full_v2.pt` su RTX 3090.

### Checkpoint 1
- [x] Suite locale verde.
- [x] Equivalence report indipendentemente verificabile.
- [x] Nessuna modifica al percorso batch o ai pesi V9.

### Slice 2 — Performance e integrazione runtime
- [x] Aggiungere `v9_step`/state ownership per sessione senza ricostruire sequenza.
- [x] Sostituire nel listener/TurnHead path il fallback hidden-sequence → interpolate → V9.
- [x] Misurare 10.000 step dopo warmup e gate p95 <= 10 ms.
- [x] Testare reset tra segmenti e anchor testuale fisso.

### Checkpoint 2
- [x] 10.000 step GPU con p95 <= 10 ms.
- [x] Nessun forward V9 batch nel percorso realtime.
- [x] Suite e verifier P4 aggiornati.

### Slice 3 — Causal-prefix ablation Qwen
- [x] Usare validation cache/data già presente sul 3090, almeno 100 sequenze e 5.000 frame.
- [x] Confrontare FULL (prefisso completo) vs CAUSAL (solo prefisso <=t) su CB0.
- [x] Ricomputare CE, top-1, top-5, KL per-frame e quartili temporali dal raw trace.
- [x] Se una soglia fallisce: STOP, documentare risultato e non iniziare QLoRA.

### Checkpoint finale
- [x] T1 equivalence PASS.
- [x] T2 no-future PASS.
- [x] T3 p95 PASS.
- [x] T4 ablation PASS.
- [x] Verifier indipendente PASS e evidence pubblicata.

## Rischi e mitigazioni

| Rischio | Mitigazione |
|---|---|
| Drift numerico BF16 tra SDPA batch e single-query | usare stessa proiezione/RoPE, confronto FP32 diagnostico |
| KV mutata dopo errore | step transazionale: commit stato solo dopo output valido |
| Conv iniziale non allineata | due slot zero e confronto diretto col forward `CausalConv1D` |
| Cache validation contiene raw già conv-processed | formato dichiarato nel todo/report; T1–T3 attestano equivalenza del percorso stateful sui vettori reali della cache |
| costo per-sessione GPU | stato per sessione, lock/ownership chiaro, benchmark 10k |
| ablation Qwen non supportata dai dati disponibili | fermare T4 come BLOCKED, non inventare metriche o soglie |
