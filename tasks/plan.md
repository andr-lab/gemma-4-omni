# Implementation Plan: P2 — 80 ms DuplexScheduler e audio ingress indipendente

## Overview
Introdurre un clock duplex logico a 12,5 Hz (80 ms), frame PCM utente standard da 1280 sample a 16 kHz e una coda thread-safe con numerazione, contabilità e nessun drop silenzioso. Collegare l'ingress alla `DuplexSession`/WebSocket senza legarlo al ciclo request/response; il consumo da listener/speaker resta responsabilità delle fasi successive.

## Decisioni architetturali
- `DuplexScheduler` orchestra il clock e callback per tick, senza importare TurnHead o modelli.
- `VirtualFrameClock` è deterministico e permette i gate 10 s/60 s senza wall-clock.
- `MonotonicFrameClock` usa `time.monotonic_ns()` e deadline assolute, evitando drift cumulativo.
- `AudioIngressQueue.append()` segmenta chunk PCM arbitrari in frame da 80 ms e conserva eventuali sample parziali; `put()` accetta soltanto frame completi.
- La sessione possiede l'ingress queue; il server continua a mantenere il buffer di commit legacy per ASR, ma le decisioni di policy vengono alimentate solo da frame completi numerati.

## Task ordinati

### Task 1 — Contratto e unit test RED
- [ ] Definire API `AudioFrame`, `DuplexTick`, `FrameClock`, `AudioIngressQueue`, `DuplexScheduler` nei test.
- [ ] Testare segmentazione lossless di chunk arbitrari, numerazione e rifiuto esplicito di input non valido.
- [ ] Testare 125/750 tick con clock virtuale, ordine e timestamp.
- [ ] Testare callback durante speaking e statistiche zero-drop.

### Task 2 — Implementazione core
- [ ] Implementare `duplex_kernel/scheduler.py` con clock virtuale/reale, queue e scheduler.
- [ ] Eseguire test P2 unitari e test contrattuali esistenti.

### Task 3 — Integrazione sessione/WebSocket
- [ ] Aggiungere `AudioIngressQueue` alla sessione e cleanup.
- [ ] Alimentare l'ingress su `input_audio_buffer.append`; policy per frame completo, non per chunk/request.
- [ ] Mantenere compatibilità del buffer commit e degli eventi esistenti.
- [ ] Aggiungere test di integrazione dell'ingress durante risposta attiva.

### Task 4 — Gate P2 riproducibile
- [ ] Aggiungere gate P2 al runner con trace elementare e report JSON.
- [ ] Estendere verifier indipendente per ricomputare tick, trasporto, speaking overlap e clock reale.
- [ ] Eseguire smoke locale, poi gate real-clock 60 s e verificare soglie.

## Checkpoint
- Dopo Task 2: unit test + contract test verdi.
- Dopo Task 3: WebSocket regression e test ingresso durante output verdi.
- Dopo Task 4: `verify_report.py` PASS sul report P2 e bundle/artifact tracciati.

## Acceptance criteria
- 10 s virtuali → 125 tick; 60 s → 750 tick; nessun duplicate/missing/out-of-order.
- Frame PCM 16 kHz/80 ms: `frame_id` monotono, contabilità sample lossless, zero drop/duplicate/out-of-order.
- Audio ingress continua ad accettare frame mentre una risposta è attiva.
- P2 report contiene raw trace, hash e metriche ricalcolate da verifier.

## Rischi e mitigazioni
| Rischio | Mitigazione |
|---|---|
| Drift del clock reale | Deadline assolute monotonic + misurazione jitter p99 |
| Chunk WebSocket non allineati a 80 ms | Buffer interno con timestamp e split lossless |
| Backpressure | Coda bounded con `queue.Full` esplicita, mai drop silenzioso; default non bounded nel server finché P3 non consuma |
| Regressione protocollo | Policy alimentata solo dai frame completi; commit buffer legacy invariato |
