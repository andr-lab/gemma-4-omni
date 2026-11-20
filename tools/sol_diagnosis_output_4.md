# Diagnosi architetturale — listener ACK, batch 1175 / frame 664

Data: 2 novembre 2026. Richiesta: `tools/sol_diagnostic_prompt_10.txt`. Ambito: diagnosi e contratto per l'esecutore; nessuna patch applicata al runtime, nessun servizio riavviato, nessuna sessione GPU di prova avviata.

## 1. Verdetto

**La violazione ACK è reale; l'attribuzione degli 80 ms alla contesa GPU non è dimostrata.** Esiste invece un difetto architetturale verificabile: la precedenza nominale del listener non copre il lavoro listener fuori dal model lock e l'ammissione del Talker vede soltanto la coda di ingresso, non il lavoro già prelevato e ancora in esecuzione.

Il prompt contiene tre premesse da correggere prima di implementare:

1. I sei hold pesanti del Talker non occupano consecutivamente il batch 664. Intersecando i timestamp reali, tutti gli hold compute visibili che ricadono nel batch sommano **22,921 ms**, non 162,37 ms. La timeline esportata contiene anche eventi molto precedenti al batch.
2. Le sotto-metriche non formano una partizione del batch: alcune sono annidate, altre misurano CUDA e wall time sullo stesso lavoro. Il commit di stato da 7,003 ms avviene addirittura **dopo** la fine del batch. Il residuo di 80,5 ms ottenuto per sottrazione non è una misura valida.
3. L'assenza del vecchio evidence gap VRAM significa che i dati sono raccolti. Non significa che il consumo sia entro budget: nel report finalizzato il reserved growth è **1492 MiB**, contro **1024 MiB**, e `within_budget` è `false`.

La remediation raccomandata è rendere misurabili i confini host/worker e correggere la visibilità della domanda listener prima di ottimizzare la cooperazione per quanto GPU. Togliere soltanto il `return` da `admit_listener` non risolve il contratto.

## 2. Evidenze effettivamente consultate

Ho letto il prompt completo, il codice locale corrente e gli artefatti originali su Mondaini:

`/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T205147331301Z/run_01/`

In particolare: `events.jsonl`, `health_pre_run.json`, `runner_summary.json`, `report.json`, `report.finalized.json` e struttura di `runtime_telemetry.json`. Gli ACK 663 e 664 sono alle righe **5952** e **5976** di quel file eventi remoto.

Il container attivo è `gemma4-speech-s2s-readyframes-20261102`. Il processo PID 1 avvia `/workspace/.venv_e2e_runtime/bin/python /workspace/duplex_kernel/launch.py`, con backend native e capacità del pool Talker pari a 1. Interrogando lo stesso interprete, come utente 1000:1000, risultano PyTorch **2.12.1+cu130**, CUDA del build PyTorch **13.0**, Transformers **5.5.0**, FastAPI **0.141.1**, Uvicorn **0.52.4**. Un import separato risolve `engine.py` in `/workspace/duplex_kernel/engine.py`. Questi controlli descrivono il filesystem/runtime corrente; non ricostruiscono in modo immutabile gli oggetti Python già caricati nel processo storico.

L'health pre-run dichiara unified KV attivo, ready-frame dispatch attivo, pair-lock handoff disabilitato, semantic shadow disabilitato, fused Control6 readback attivo e host lock timeline attiva. Ho inoltre riletto sulla macchina remota il bypass in `engine.py:6111–6120` e l'helper in `server_pcm_tracker.py:814–842`: corrispondono ai punti discussi qui.

## 3. Ricostruzione corretta del picco

### 3.1 ACK, completamento batch e trasporto

| Confine | Timestamp monotonic ns |
|---|---:|
| Fine batch frame 663 | 6436679386341648 |
| Ricezione client ACK 663 | 6436679391745144 |
| Inizio batch frame 664 | 6436679392861484 |
| Fine batch frame 664 | 6436679541370035 |
| Inizio commit output frame 664 | 6436679541779222 |
| Fine commit output frame 664 | 6436679548894608 |
| Ricezione client ACK 664 | 6436679553646912 |

Il gap ACK è **161,901768 ms**. Si scompone esattamente in:

`1,116340 ms prima del batch + 148,508551 ms batch + 12,276877 ms dopo il batch`.

Il termine finale comprende commit e pubblicazione/trasporto/ricezione; non è tutto tempo WebSocket. Il commit di stato da 7,003 ms è contenuto qui. I completamenti batch 663→664 distano **155,028387 ms**; il ritardo batch-end→ACK passa da **5,403496** a **12,276877 ms**. La crescita di quest'ultimo contribuisce per **6,873381 ms** al gap client.

Quindi il batch lungo è il contributo dominante, ma anche il percorso di pubblicazione consuma il margine residuo. Non basta sottrarre 1,9 ms da un forward per ottenere robustezza.

### 3.2 I 162 ms del Talker appartengono a un'altra finestra

La timeline del frame 664 è `truncated`: conserva 64 eventi, con 2843 eventi storici scartati. Gli eventi esportati non sono filtrati all'intervallo batch.

| Ticket | Posizione rispetto all'inizio batch | Hold intersecato col batch |
|---|---|---:|
| 1652, 1655, 1659, 1662, 1666 | Terminati prima dell'inizio | 0 ms |
| 1668 | +0,801 → +0,994 ms | 0,192914 ms |
| 1669 | +18,658 → +40,715 ms | 22,056931 ms |
| 1670 | +41,998 → +42,270 ms | 0,272545 ms |
| 1671 | +92,760 → +93,159 ms | 0,398717 ms |

Somma delle intersezioni visibili: **22,921107 ms**. Sono tempi di possesso di un lock host: non misurano direttamente occupazione SM, durata di tutti i kernel o sovrapposizione effettiva sulla GPU. Lavoro CUDA asincrono lanciato prima può essere ancora pendente dopo il rilascio del lock; dunque la correzione della finestra non esclude la contesa GPU, ma elimina la presunta prova dei «162 ms consecutivi».

`pending_counts.listener == 0` dimostra assenza di waiter listener registrati su quel lock nei campioni visibili; non dimostra assenza di lavoro listener.

### 3.3 Perché gli 80,5 ms non sono attribuibili

`shared_listener_scheduler.py:482–525` avvia il timer prima di `_batch_listener(rows)`, attende `Future.result()` e valida gli output prima di fermarlo. Include anche l'inizializzazione dei campi delle righe e gli aggiornamenti watchdog. `_shared_listener_batch_rows` in `server.py` prepara i frame e chiama `worker.submit_tick`; `listener_microbatch_worker.py:2543–2560` accoda un comando al proprietario. **Il nome compute_ms copre l'intero percorso di servizio**, inclusi eventuale attesa nella coda comandi, schedulazione del worker e risveglio del dispatcher.

Nel worker, `frontend_encode_ms` avvolge `frontend.encode_batch`: Mel, copie e tower non vanno aggiunti automaticamente come durate disgiunte. `native_submit_ms=33,744` e `native_cuda_ms=33,730` descrivono confini differenti dello stesso tratto, non due lavori da sommare. `native_cuda_ms` riguarda il native listener; non va identificato con il forward Qwen del Talker perché i numeri sono simili.

Restano inoltre operazioni fuori dai timer citati: preparazione dei metadati, costruzione/controllo degli input policy, eventuale vista Q2 privata, commit frontend/coordinatore, costruzione output, consegna Future e validazione. Nel frame osservato le metriche decision-view sono `null`: questo non prova che una vista Q2 costosa sia stata eseguita, né autorizza a imputarle il residuo.

Con questi dati non si distingue fra attesa host, GIL/descheduling, altra attività del worker, sincronizzazioni implicite e contesa device. Servono confini ulteriori, non una spiegazione numerica inventata. I batch 661–666 durano rispettivamente **44,148; 84,064; 66,077; 148,509; 78,156; 51,433 ms**: la variabilità è concreta e non esclusiva del frame fallito.

## 4. Difetto di cooperazione confermato nel codice

In `engine.py:6069–6071`, `prioritized_v2_talker` abilita l'arbitraggio host per backend native/compiled V2. In `admit_listener`, però, il percorso prioritizzato salta il callback per `subtalker` e `next_qwen_model`.

C'è un dettaglio aggiuntivo: il loop corrente chiama `admit_listener` prima di `codec_head` (`6948`) e di `next_qwen_model` (`7076`), ma **non lo chiama prima di `run_talker_gpu_step("subtalker", ...)`**. Rimuovere il bypass senza aggiungere quel punto di cooperazione lascerebbe il SubTalker scoperto.

L'helper server legge `sess.audio_ingress.qsize` (`server.py:4098–4114`). Quando il dispatcher ha già estratto il frame, la coda può essere vuota mentre il batch è selezionato, accodato al worker o in esecuzione. La soglia `<=1` ammette inoltre un frame pendente. Cambiarla in `==0` non risolve il lavoro invisibile dopo il dequeue.

L'helper ignora intenzionalmente `callbacks_active`: il flag comprende attività CPU/event-loop e può restare attivo per tutto lo stream. Usarlo come nuovo mutex globale ricreerebbe lo starvation del Talker/prefill. Al timeout di 160 ms l'helper ammette comunque un quanto, affidandosi alla fairness di un lock al quale il listener ordinario non partecipa: il commento non costituisce una garanzia valida per questo percorso.

Il problema è quindi una **discordanza tra dominio di priorità e lavoro effettivo**, non una prova di starvation sul model lock.

## 5. Contratto di remediation per l'esecutore

### 5.1 Rendere osservabile il percorso reale

Aggiungere timestamp monotonic correlati per batch/frame/generazione a: selezione, submit comandi, dequeue worker, inizio/fine elaborazione worker, completamento Future, ritorno da `result()`, fine validazione, commit e pubblicazione ACK. Separare la durata di servizio end-to-end dal wall time del worker, mantenendo compatibili i campi esistenti.

Nel worker misurare intervalli wall **disgiunti** per preparazione/capacità, frontend complessivo, preparazione input/vista Q2, native submit, health/readback, policy, completamento/commit e produzione output. Le sottofasi CUDA restano una vista parallela, mai un totale da sottrarre al wall. Campionare `thread_time_ns` ai confini utili distingue CPU attiva da tempo non eseguito; la differenza wall−CPU non identifica da sola GIL o CUDA.

Telemetria scalare e bounded, senza liste crescenti sul WebSocket degli ACK. Riutilizzare `engine_timing.py` dove appropriato; leggere eventi completati senza aggiungere sincronizzazioni diagnostiche globali. Per un dubbio residuo usare una breve acquisizione di timeline host/device con intervalli nominati, non un profiler pesante durante tutta l'accettazione.

### 5.2 Pubblicare la domanda listener senza buchi

Usare un coordinatore leggero condiviso dal dispatcher, dal worker e dagli ingressi compute. Lo stato minimo comprende richieste listener selezionate/accodate, batch in esecuzione e completamento GPU non ancora osservato. Associare ticket al batch e alla generazione; per batch con più righe conservare tutte le identità. Non usare un booleano di sessione che un vecchio callback possa azzerare mentre un nuovo batch è attivo.

Il trasferimento ingress→selected→worker→GPU deve mantenere continua la visibilità della domanda: prima registrare la nuova ownership, poi rimuovere la precedente. Errore, cancellazione e teardown rilasciano esattamente il ticket posseduto e notificano i waiter. Il coordinatore non deve richiedere letture concorrenti di KV o degli oggetti mutabili posseduti dal worker.

### 5.3 Cooperazione a ogni quanto significativo

Prima di codec head, SubTalker, Qwen Q1 e degli altri stadi GPU potenzialmente concorrenti, il produttore richiede un permesso. Il controllo avviene **prima di acquisire i lock modello/workspace**, con verifica di cancellazione/generazione. Il ramo prioritized V2 non deve scavalcare il controllo. Centralizzare il controllo in un wrapper comune evita chiamate mancanti, purché si preservino le semantiche dei callback già presenti.

Ammissione compute e registrazione della domanda listener devono essere atomiche rispetto allo stesso coordinatore. Un semplice «leggi flag, poi acquisisci model lock» lascia una race. Se il listener arriva dopo l'ammissione di un quanto, quel quanto può completare; il successivo non deve superare il ticket listener già pubblicato. Il modello non è preemptive e non deve tentare di interrompere un kernel o un graph replay in corso.

Il permesso host deve coprire anche l'ordinamento device necessario: se lo stadio restituisce prima del completamento GPU, registrare un evento sul suo stream effettivo e mantenere il lavoro pendente visibile. Per ordinare il listener dopo il quanto già ammesso, usare dipendenze fra gli stream interessati; il passaggio inverso deve attendere l'evento listener corrispondente. Verificare i contratti degli executor native e dei graph sui loro stream reali: non assumere che l'evento sullo stream corrente copra stream interni non raccordati.

Niente `torch.cuda.synchronize()` per frame, sleep fissi o polling stretto. Attendere tramite condition/eventi fuori dai lock di modello, transizione e stato. Non attendere il completamento dei callback di pubblicazione ACK per liberare la GPU: il confine è il completamento del lavoro GPU e delle sue dipendenze, mentre il commit host mantiene le proprie regole di consistenza.

Il timeout non deve fingere che il listener sia inattivo e autorizzare una sequenza illimitata di altri quanta. Su mancato progresso si mantiene visibile il ticket, si registra la fase responsabile e si applica la normale gestione cancellazione/errore. Per backlog persistente serve una politica esplicita di fairness o capacità, misurata congiuntamente al playout: una priorità assoluta per un'intera risposta o un'intera sequenza infinita di frame non è accettabile.

### 5.4 Fattibilità temporale: non promettere 50–70 ms per definizione

50–70 ms di servizio listener e ACK normalmente sotto 120–140 ms sono obiettivi con margine; la soglia esistente di 160 ms resta invariata. Il gap fra ACK non coincide con il tempo del singolo batch: comprende cadenza, code e variazione della pubblicazione.

Una serializzazione integrale potrebbe fallire per capacità: ipotizzando un batch listener e un frame Talker ogni 80 ms, **50–70 + 33 + 13 = 96–116 ms**, prima di codec/vocoder/host. È un controllo di fattibilità condizionale, non una misura della domanda GPU esclusiva: i tempi del run includono interferenze e non tutti gli stadi hanno necessariamente la medesima frequenza. Occorre misurare costi isolati, frequenze reali e sovrapposizione utile prima di imporre esclusione totale.

La prima applicazione deve limitare i burst e proteggere il batch pronto lasciando procedere il Talker negli spazi utili. Se throughput e latenza non soddisfano insieme i vincoli, ottimizzare lo stadio identificato o rivalutare il grado di sovrapposizione. Non aumentare buffer o soglie per occultare un deficit sostenuto, né proclamare garantito un limite temporale su un runtime general purpose soltanto perché esiste un lock.

## 6. File da modificare nella patch successiva

| File | Responsabilità della modifica |
|---|---|
| `duplex_kernel/shared_listener_scheduler.py` | Ticket della domanda prima che la riga diventi invisibile in ingress; tempi submit/Future/validazione; cleanup su ogni uscita. |
| `duplex_kernel/server.py` | Wiring del coordinatore condiviso e passaggio dei ticket; conservare confine queue-close/drain e percorso ACK. |
| `duplex_kernel/listener_microbatch_worker.py` | Dequeue/process/completion, fasi wall disgiunte, pubblicazione completion e rilascio ticket su successo/errore. |
| `duplex_kernel/server_pcm_tracker.py` | Separare il throttle ingress dall'arbitraggio GPU; eliminare l'assunzione che qsize descriva tutto il lavoro e che il fair lock veda il listener normale. |
| `duplex_kernel/engine.py` | Controllo per ogni quanto, incluso SubTalker; nessun bypass native/compiled; cancellazione fuori lock e raccordo eventi stream. |
| `duplex_kernel/engine_timing.py` | Solo se necessario: riuso degli helper per eventi completati e aggregati bounded. |
| Modulo del coordinatore, se serve | Un'unica implementazione piccola, preferibilmente nel modulo scheduler pertinente; evitare di confonderlo con `gpu_admission.py`, che limita conversazioni/HTTP e non arbitra questi quanta. |
| `docs/ARCHITECTURE.md` e test pertinenti | Documentare ownership, ordine lock/eventi e limiti verificati insieme alla patch. |

Non modificare ora `PLAN.md`, archive terminale, soglie del simulatore o classificazioni del finalizer per far passare il run. La richiesta corrente autorizza questo documento diagnostico; l'elenco descrive la patch futura.

## 7. Invarianti e prove funzionali

Prima della patch isolare test RED nei moduli esistenti `test_v2_duplex_admission.py`, `test_shared_listener_tick_dispatcher.py` e `test_listener_microbatch_worker.py`, estendendoli con un piccolo test del coordinatore se necessario:

1. Ingress vuoto e batch selezionato/accodato/attivo: il prossimo quanto compute vede la domanda; non basta `qsize==0`.
2. Listener arrivato tra controllo e acquisizione: al massimo termina il quanto già ammesso; il successivo cede. Coprire interleaving deterministici, non sleep probabilistici.
3. Tutti i siti significativi cooperano sui backend native/compiled, compreso SubTalker. Coprire il percorso pool realmente usato, non soltanto l'helper isolato.
4. `callbacks_active=True` con nessun lavoro GPU non blocca prefill o Talker.
5. Eccezione worker, annullamento fra claim ed enqueue o prima del dequeue, generazione sostituita, teardown e completion tardiva: nessuna domanda fantasma e nessun ticket nuovo liberato dal batch vecchio. Coprire batch con più sessioni e release host precedente alla completion CUDA, inclusi backend con stream interni.
6. Producer e listener progrediscono sotto carico continuativo; nessuna attesa di condition mentre si possiede un lock necessario all'altro. Eventi pendenti impediscono una falsa liberazione del device.
7. I tempi host sono ordinati e coerenti col batch, distinguono attesa worker/risveglio e non sommano metriche CUDA annidate.

Preservare ordine e completezza dei frame, KV canonico, fork Q2 privati, generation checks, rollback/YIELD, commit e drain transazionali, testo EOS e codec EOS. Preservare FIFO playout **4 s / 96000 campioni a 24 kHz**, watermark PCM, chiusura segmenti, `response.done` e archive terminale limitato a 64 record. Non anticipare l'ACK prima del reale consumo previsto dal protocollo.

Eseguire poi su Mondaini, in Docker non-root e senza altri job GPU concorrenti, uno smoke breve con listener e percorso Talker produttivo. Verificare ordinamento cross-stream, cancellazione, primo audio/TTFT **<=800 ms secondo la metrica esistente**, continuità PCM e memoria. Solo dopo lo smoke eseguire la suite richiesta:

```text
python3 tools/run_acceptance_suite.py --runs 3 --container gemma4-speech-s2s-readyframes-20261102 --execution docker
```

Confrontare gap ACK massimi e distribuzione, età ingresso→ACK, servizio worker, attese compute e profondità playout sull'intera risposta lunga e sui turni successivi. Il frame 664 riporta già `server_listener_queue_ms=311,196`: un gap inter-ACK inferiore a 160 ms non dimostra da solo bassa latenza assoluta. Verificare la definizione del contatore e i timestamp raw prima di attribuirlo alla sola coda d'ingresso.

I test CPU provano ownership e interleaving del protocollo; lo smoke GPU prova il percorso reale breve; i tre run completi verificano la conversazione. Nessuno di questi risultati è stato ottenuto in questa diagnosi.

## 8. Stato reale di VRAM e accettazione

`report.finalized.json` conserva `overall_status: INCOMPLETE` e l'errore ACK. Contiene anche l'evidence gap `input counters in the preliminary report do not match raw input events`. Non riclassificare il run come quasi PASS sulla sola differenza di 1,9 ms.

La raccolta VRAM è funzionante, con 183 campioni e uno snapshot post-drain. Crescita allocated di picco: **270,646 MiB**; post-drain: **93,291 MiB**, sotto i budget rispettivi di 1536 e 256 MiB. Crescita reserved di picco e post-drain: **1492 MiB**, sopra entrambi i budget di 1024 MiB. Un solo turno non permette di inferire un leak; la memoria reserved comprende il caching allocator e non equivale alla memoria tensor attiva. Tuttavia il budget di questa suite è violato e resta invariato.

L'esecutore deve preservare questa telemetria e confrontare baseline/active/peak/post-drain con il runtime corretto, senza usare `empty_cache()` per occultare la crescita o ampliare i budget. La risoluzione della raccolta dati non va confusa con la risoluzione del comportamento misurato.

## 9. Fonti e limiti delle conclusioni

Le evidenze numeriche sono quelle del run originale remoto, non vecchi risultati di smoke. I riferimenti implementativi principali sono `shared_listener_scheduler.py:482–525`, `listener_microbatch_worker.py:2543–2560, 8190–8330, 8420–8490, 8665–8800`, `server.py:1390–1435, 4098–4114` e `engine.py:6069–6150, 6948–7105`. I numeri di riga si riferiscono al checkout letto e possono cambiare con le modifiche.

Per i contratti CUDA ho consultato [PyTorch 2.12, CUDA semantics](https://docs.pytorch.org/docs/2.12/notes/cuda.html) e [NVIDIA, Asynchronous Execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html): il submit host può precedere il completamento device e l'ordine fra stream richiede dipendenze esplicite. La priorità degli stream non costituisce una garanzia di preemption o deadline. Queste proprietà sostengono il contratto degli eventi; non dimostrano quale kernel abbia causato il picco storico.

La conclusione causale più forte sostenibile oggi è: **servizio listener end-to-end troppo variabile, con arbitraggio incapace di osservare tutta la domanda reale**. La causa temporale specifica del frame 664 rimane da separare fra host, worker e device con i confini indicati. Nessun dato disponibile giustifica assegnare precisamente 80 ms a scheduling SM, GIL, vista Q2 o attesa del dispatcher.

Revisione DDD: un revisore indipendente con contesto separato ha esaminato il contratto e i punti sorgente. I rilievi utili sono incorporati: distinzione hold host/completion GPU, copertura degli stream interni, fairness sotto ingresso continuo, somma dei costi soltanto condizionale e ownership dei ticket durante cancellazioni. Non è stata eseguita una revisione con un modello esterno.
