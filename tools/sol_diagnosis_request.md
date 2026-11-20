# Richiesta Diagnosi Architetturale per GPT 6.1 Sol (Medium Effort)

## Contesto Operativo e Obiettivo
Siamo nella pipeline full-duplex omni per Gemma 4 su GPU NVIDIA RTX 3090 (24GB VRAM, host `mondaini`).
Stiamo validando `PLAN.md` tramite la suite di accettazione end-to-end (`tools/run_acceptance_suite.py --runs 3`).

Il Turn 2 (Barge-in con YIELD e cancellazione deterministica senza drain failure) è stato COMPLETAMENTE RISOLTO E VERIFICATO nel Run 13 (`cancelled_response_count: 1`, zero errori di playback drain).
Tutti i 177 unit test passano verdi.

Tuttavia, nel Run 13, il Turn 1 (Long response playback >= 40s) è fallito al frame 1214 con un'eccezione sul gap ACK del microfono:
`EvidenceError: long response microphone ACK gap 1489.9ms reached the 160.0ms limit`

## Dati Telemetrici e Osservabili (Run 13)
1. **Analisi temporale del Turn 1**:
   - Frame 1213: ACK arrivato a 20:57:58.304 (latenza 71.6ms).
   - Frame 1214: ACK arrivato a 20:57:59.794 (latenza 1480.3ms).
   - Gap calcolato: 1489.9ms (> soglia 160.0ms).
   
2. **Evento 6945 in `events.jsonl` (Frame 1214)**:
   - `server_listener_queue_ms`: 1476.107 ms
   - `server_listener_step_ms`: 1471.371 ms
   - `listener_batch_compute_ms`: 1471.371 ms
   - `listener_batch_control6_policy_ms`: **1430.056 ms**
   
3. **Lock Timeline dell'Host Tick nel Thread Worker (event 6945)**:
   - Ticket 3932 rilasciato a timestamp 6004995009897678 (20:57:58.349).
   - Ticket 3933 richiesto a timestamp 6004996440246808 (20:57:59.779).
   - Tempo trascorso nel thread tra i due ticket: **1430.3 ms** esatti, interamente spesi dentro la chiamata di policy!

4. **Codice Coinvolto nel Microbatch Worker**:
   - `duplex_kernel/listener_microbatch_worker.py`: riga ~7374:
     ```python
     valid_decisions = tuple(mixed_source_policy(output_views, valid_policy_inputs))
     ```
   - Chiamata a `_step_policy_context_views` in `duplex_kernel/gemma_native_control6_batch_policy.py`: righe 675-725.
   - All'interno: valutazione dell'adapter PyTorch, sincronizzazione su stream CUDA (`output.completion_event.wait(consumer_stream)`), e allocazioni di tensori di contesto.

5. **Confronto con Run 12**:
   - Nel Run 12, lo stesso identico Turn 1 è stato eseguito per **122.88 secondi** (1536 frame) SENZA alcun picco sopra 70ms.
   - Nel Run 13, a circa 97 secondi dall'avvio della risposta (frame 1214), si è verificato questo blocco isolato di 1.43s.

## Domande per GPT 6.1 Sol
1. **Root Cause Analysis**: Qual è la causa scatenante più probabile di questo blocco una tantum di 1430ms dentro `listener_batch_control6_policy_ms`?
   - È il Garbage Collector di Python (`gc.collect`) innescato dall'accumulo di oggetti/frame nel microbatch loop continuo?
   - È una deallocazione/compattazione di memoria PyTorch CUDA (`cudaMalloc` / `empty_cache`)?
   - È un blocco di sincronizzazione su `completion_event.wait` o contesa della GPU/CUDA stream con il thread di generazione della risposta assistente?
   - Oppure allocazioni/concatenazioni di tensori o slice crescenti nella context pool durante la riproduzione lunga?
2. **Analisi Architetturale & Rimedi**:
   - Come isolare e proteggere il loop di microbatch tick del listener per garantire che il tempo di computazione della policy non superi MAI la soglia critica di 160ms?
   - Quali ottimizzazioni o guardrail (es. disabilitazione GC / gestione cicli gc periodici, pre-allocazione tensori, policy timeout/fallback o esecuzione asincrona del tick di policy senza bloccare l'invio dell'ACK, gestione CUDA stream) raccomandi?
3. **Piano Operativo Determinante per GPT-6 Luna Max**:
   - Fornisci una specifica tecnica precisa, con file esatti, funzioni, strutture dati e modifiche da affidare al worker operativo GPT-6 Luna Max per eliminare definitivamente questo jitter e superare i 3 run consecutivi della suite di accettazione.
