# Diagnosi architetturale: ACK del listener e finalizzazione della risposta

## Esito

La finalizzazione coincide con due massimi ACK, ma **i dati non dimostrano che il transition lock o un model lock da 30,5 ms ne siano la causa**. La spiegazione del prompt richiede tre correzioni sostanziali:

1. Sui frame critici il transition lock attende **0,001 ms**. Anche l'attesa dello state lock è **0,001 ms**: i 24–26 ms sono tempo wall *dentro* il lock, non tempo passato ad aspettarne il proprietario.
2. Il model-lock hold da 30,558 ms citato nella timeline è precedente al frame incriminato. Non prova un blocco diretto durante quel batch.
3. Il massimo finale non è l'unica anomalia: nella run 3 esiste già un gap di **164,687 ms tra i frame 1126 e 1127**, circa quattro secondi prima. La run 2 passa il limite corrente di 160 ms, ma raggiunge **143,994 ms** nella finestra lunga: non soddisfa l'obiettivo più stretto di 140 ms.

La diagnosi sostenuta dall'evidenza è una **coda di latenza host nel percorso batch → commit → pubblicazione → ricezione**, accentuata al termine della risposta. Esistono costi architetturali verificabili da rimuovere: copie e costruzione dell'output sotto lo state lock; ripubblicazione dell'intero dizionario delle metriche nell'evento finale; lavoro owner che precede il controllo dei frame pronti. Non è ancora identificato quale di questi costi produca ciascuna porzione dei picchi osservati. La bonifica deve partire da questi punti, senza riscrivere il lifecycle appena stabilizzato e senza promettere una scadenza temporale che lo scheduler cooperativo non può garantire da solo.

## 1. Fonti, runtime e limiti della verifica

Sono stati letti il prompt 12, i sorgenti attuali, i test pertinenti e gli artifact originali su Mondaini in:

`/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261102T234523994806Z/`

In particolare: `suite_summary.json`, i tre `report.finalized.json`, gli eventi dei frame interessati, `health_pre_run.json` e ricerche mirate nei `server.log`. La ricostruzione numerica usa i timestamp degli eventi originali, non soltanto gli estratti del prompt.

I contenuti testuali di sette file sono stati confrontati direttamente tra checkout Windows e repository montato su Mondaini e risultano uguali: `server.py`, `listener.py`, `server_unified_cache.py`, `shared_listener_scheduler.py`, `listener_microbatch_worker.py`, `server_realtime_scheduler.py`, `engine.py`. Nessun manifest aggiunto. Questo verifica i file presenti al momento dell'ispezione; non prova retroattivamente il contenuto in memoria di ogni modulo durante le run precedenti. Gli health delle run indicano i moduli server/listener/worker sotto `/workspace/duplex_kernel`.

Il container ispezionato è `gemma4-speech-s2s-readyframes-20261102`, utente `1000:1000`, IPC host, rete bridge, senza privileged, repository montato in `/workspace`. Il comando del container usa `/workspace/.venv_e2e_runtime/bin/python`. Interrogando quell'interprete risultano Python 3.11.9, distribuzioni Torch 2.12.1, Transformers 5.5.0, NumPy 1.26.4, FastAPI 0.141.1, Starlette 1.6.0 e websockets 17.1. Gli health storici riportano Torch `2.12.1+cu130`. Il `python3` generico del container non rappresenta automaticamente l'ambiente di serving; il Dockerfile non basta a ricostruirlo.

Nessuna patch runtime, sincronizzazione dei sorgenti o riavvio è stato eseguito. Sono passati **49 test locali** con:

```powershell
python -m pytest tests/kernel/test_shared_listener_tick_dispatcher.py tests/kernel/test_server_consumed_ack_order.py tests/kernel/test_cooperative_drain.py tests/kernel/test_unified_controller_lifecycle.py -q
```

Questa verifica conferma i contratti coperti oggi; non dimostra la latenza della correzione proposta. Nessun nuovo smoke GPU o E2E è stato eseguito per questa diagnosi. I 129 test complessivi sono un risultato riferito dal prompt, non ripetuto qui. Una revisione avversariale separata ha verificato i rischi del commit fuori lock e dell'ordinamento sul worker; le correzioni sono integrate sotto.

### Stato reale delle tre run

| Run | Stato finale artifact | Massimo ACK nella finestra lunga | Osservazioni |
| --- | --- | ---: | --- |
| 1 | INCOMPLETE | 173,527530 ms | Errore ACK; `scenario.evidence_gaps` segnala anche discrepanza tra contatori preliminari ed eventi grezzi |
| 2 | PASS | 143,994297 ms | `errors: []`, `scenario.evidence_gaps: []`; il riepilogo conserva però `simulator_exit_code: 2`, con `finalizer_exit_code: 0` |
| 3 | INCOMPLETE | 178,141369 ms | Errore ACK e stessa discrepanza dei contatori preliminari |

La suite complessiva è INCOMPLETE. La run 2 è un PASS del report finalizzato, non un'esecuzione con tutti gli exit code a zero. I tre log server non contengono la stringa `native generation session failed; close it before reuse`: la regressione non si osserva in questo campione, ma «risolto definitivamente» resta una formulazione più forte dell'evidenza.

## 2. Ricostruzione del percorso e delle misure

Il percorso rilevante è:

```text
ingress PCM / pubblicazione domanda listener
  → snapshot dispatcher: sessione in-flight
  → wait_for_compute_quanta, senza model/state lock
  → batch context e acquisizione breve transition lock
  → classificazione frame e worker.submit_tick
  → attesa Future owner + validazione output
  → rilascio priorità GPU / in-flight
  → callback host e commit_microbatch_output
  → costruzione evento consumed e coda scheduler
  → publisher WebSocket / serializzazione / trasporto
  → parsing e timestamp del client
```

Riferimenti: `shared_listener_scheduler.py:699–869`, `server.py:1408–1432`, `server.py:5096–5208`, `server_realtime_scheduler.py:159–185`, `server.py:4951–5007`, `tools/real_dialogue_simulator.py:1340–1361`.

### Durate realmente disgiunte

Valori in millisecondi. «Fine batch → ACK client» include commit e tutto il percorso successivo; non va sommato nuovamente al commit.

| Run/frame | Enqueue → inizio batch | Batch wall | CUDA nativo | State lock wait | State lock hold | Fine batch → ACK client |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 1135 | 0,142 | 48,425 | 23,666 | 0,001 | 5,671 | 10,381 |
| 1 / 1136 | 0,180 | 95,849 | 27,118 | 0,001 | 24,512 | 55,097 |
| 1 / 1137 | 39,262 | 43,002 | 23,676 | 0,001 | 1,452 | 5,535 |
| 3 / 1173 | 0,187 | 48,318 | 23,170 | 0,001 | 5,867 | 9,802 |
| 3 / 1174 | 9,834 | 77,808 | 25,242 | 0,001 | 26,123 | 54,936 |
| 3 / 1175 | 45,436 | 45,928 | 26,487 | 0,001 | 3,137 | 8,473 |

`server_listener_queue_ms` viene calcolato con `now - frame.enqueue_ns` quando si costruisce l'evento ACK (`server_realtime_scheduler.py:176`). Include quindi attesa, batch e commit: il nome non indica una pura attesa in coda. `server_listener_step_ms` nel percorso condiviso è lo stesso batch wall, non un secondo costo indipendente.

`wait_for_compute_quanta()` precede il timestamp di inizio batch. Il transition lock è invece acquisito dalla funzione batch dopo quel timestamp. Nei due frame incriminati il suo wait è 0,001 ms; la misura enqueue → batch start è inoltre appena 0,180 ms nella run 1. Non si può attribuire il raddoppio del batch a una lunga attesa in `wait_for_compute_quanta()` su quel frame.

Il tempo CUDA riportato copre il tratto nativo misurato, non tutta l'attività GPU del processo. «Batch wall meno CUDA» non equivale a tempo CPU puro: include frontend, policy, attesa owner, scheduling, altre fasi GPU e attese non coperte da quell'evento. Per esempio, nella run 3 il frontend del frame critico sale a 16,495 ms dai 6,307 ms del precedente.

Il commit inizia circa 0,5–0,6 ms dopo la fine del batch e termina dopo 25,054/26,777 ms. Rimangono **30,043/28,159 ms tra ritorno del commit e timestamp client dell'ACK**. Non sono spiegati dal solo hold dello state lock. La coda ACK riportata è circa 2,6 ms e il lock WebSocket circa 0,06 ms: servono misure di serializzazione, consegna e ricezione per attribuire il residuo.

Per lo stesso insieme ordinato di frame vale:

`gap_ACK(i) = intervallo_invio(i) + latenza(i) - latenza(i-1)`.

Il gap breve successivo è coerente con recupero del backlog. Non prova che il frame precedente sia stato bloccato da una specifica funzione di finalizzazione. Non bisogna rendere artificialmente regolari gli ACK ritardando quelli veloci.

### La timeline del model lock non dimostra il blocco proposto

Il release con `hold_ns: 30558030` ha timestamp `6447133799225606`; già l'invio del frame 1136 avviene a `6447134507279858`, circa **708 ms dopo**. È storia precedente esportata nella snapshot, non un hold simultaneo al frame 1136. Nei tail esportati dei sei frame esaminati non risultano eventi lock con timestamp nell'intervallo enqueue → ACK. Lo stato della snapshot è `truncated`: questa assenza non esclude attività non esportata e non identifica i thread per ruolo. Occorrono intervalli sovrapposti e identità dell'operazione per attribuire causalità.

### Il problema esiste anche durante lo streaming

Nella run 3 il gap 1126 → 1127 è 164,687301 ms, all'interno della finestra lunga. Il batch sale da 37,472 a 110,999 ms, mentre lo state lock hold è soltanto 2,494 ms. Nella run 1 il gap 978 → 979 è 153,383373 ms; nella run 2 il massimo della finestra lunga è il gap 836 → 837 da 143,994297 ms. Nell'intera sessione della run 2 compare anche 177,370866 ms tra 2060 e 2061, fuori dal massimo della finestra lunga pubblicato dal report.

Pertanto vanno distinti due obiettivi: tre PASS secondo lo scenario corrente, e tutti i gap sotto 140 ms nel perimetro concordato. Il secondo è più severo e non è implicato dal primo. Il picco finale non deve diventare l'unico frame osservato.

## 3. Finalizzazione: cosa contende realmente e cosa disaccoppiare

### Contratto già corretto da preservare

`server_unified_cache.py:324–352` acquisisce il transition lock, pubblica l'eventuale confine di chiusura e accoda le operazioni. Rilascia il lock prima di attendere le Future. `_queue_worker_close_epoch()` chiama un metodo che restituisce una Future: non esegue direttamente tutto il drain sul thread chiamante (`listener_microbatch_worker.py:3132–3145`).

Il lock serve a decidere su quale lato del confine queue-close cade ciascun frame, in coerenza con classificazione e submission del listener. Rimuoverlo o spostare fuori lock il marker senza un protocollo equivalente introduce frame classificati con un'epoch e accodati dall'altro lato del drain. Non è una correzione della latenza.

Il replay Q2 del drain è già cooperativo: `_run_one_queued_audio_drain_quantum()` rinuncia se ci sono comandi o frame pronti e processa un frame Q2 alla volta. La sua Future resta pendente fino al completamento ordinato. Non introdurre un secondo drain in background che perda quell'ordine.

### Percorso effettivo da distinguere

In `server.py:8567–8584`, se `revisioned_live_say` è vero, **non viene chiamato** `_finalize_unified_native_cache_branch()`: viene atteso `worker.end_response_decode(handle)`. Sul worker questo termina in `_end_ephemeral_decode()` (`:5264–5316`), che pubblica un event wait sullo stream e chiama `release_session_async`.

Nei frame interessati il cache cursor rimane 791, `logical_q_len` è 0 e `ephemeral_reason` è `prespeech_silence`; l'evento finale riporta soltanto lo status della finalizzazione senza metriche di drain. Sono indizi compatibili con il percorso privato/revisioned, non una prova completa dell'identità del ramo. Aggiungere una misura compatta della route effettiva prima di scegliere il punto da correggere. Modificare soltanto il drain canonico può non toccare il percorso che ha prodotto il picco.

### Due accoppiamenti concreti

**Lavoro owner prima dei frame.** Il loop del worker (`listener_microbatch_worker.py:3514–3535`) esegue un comando prima di `take_ready()`. `_drain_commands()` ne esegue uno, non l'intera coda. Una chiusura/release lunga può quindi precedere un batch pronto; anche tanti piccoli comandi possono aggiungere un costo a ogni batch. Non basta rendere asincrona la chiamata dal server. L'esecuzione effettiva deve essere breve oppure differita a quanti, con verifica dei frame tra i quanti. Non inserire sul worker un'attesa di permit che richieda al medesimo worker di completare il listener: sarebbe un deadlock.

**Metriche voluminose sul canale realtime.** `_unified_native_finalizer_event()` costruisce già un evento compatto, ma `_send_unified_native_finalizer_event()` sostituisce le sue metriche con tutto `runtime_metrics` (`server_metrics.py:622–642`). Il server passa `runtime_metrics_event(response_id)` (`server.py:8642`). I payload finali misurano circa **152.828 e 153.278 byte**, con le tracce già presenti nella risposta. I payload consumed adiacenti sono circa 20 KB ciascuno. Sono dimensioni della rappresentazione `json.dumps` predefinita, non conteggi esatti dei byte sul wire compatto.

Il sorgente di `Starlette.WebSocket.send_json` installato è stato letto: esegue `json.dumps` sincrono prima dell'await del trasporto. Il server lo chiama sotto `ws_send_lock`. Il client, a sua volta, esegue parsing, ricorsione `_json_safe`, scrittura e flush sincroni nel receiver loop (`real_dialogue_simulator.py:804–826,1340–1361`). Questo lavoro può competere per CPU/GIL e ritardare gestione del messaggio successivo. Un basso `server_ack_lock_wait_ms` non lo esclude: il loop può rimanere occupato prima che la coroutine ACK chieda il lock.

Questo è un accoppiamento dimostrato dal codice e dal volume dei payload; **non è ancora misurato che costi 20 o 30 ms**. Non attribuire automaticamente al JSON tutto il residuo. Il timestamp client attuale è acquisito dopo `json.loads`, quindi la metrica ACK include anche il ritardo del receiver; va conservato per confrontare le run, aggiungendo separatamente i timestamp grezzi se necessari.

### Contratto di fine risposta proposto

- `segment_done`, `response.done`, consumo client, rilascio del decoder e commit della storia restano eventi distinti. `response.done` non autorizza la riutilizzazione anticipata dello stato nativo.
- L'attesa playback/ACK avviene senza model lock, state lock e transition lock. L'owner mantiene vivi handle e buffer fino al fence necessario.
- La pubblicazione del confine queue-close e l'accodamento dei comandi associati restano un'operazione breve ordinata rispetto ai tick. Le mutazioni fisiche vengono eseguite dall'owner, con CAS di generazione/cursore.
- Tra due quanti differibili si ricontrolla la domanda listener. Un frame arrivato dopo l'ammissione può dover attendere il solo quanto già in volo: il sistema cooperativo non offre preemption istantanea dei kernel.
- La finalizzazione invia un delta compatto con route, response ID, status, errore e metriche nuove di rollback/drain/release. Le tracce complete restano disponibili nel meccanismo diagnostico esistente; non vengono rigenerate sul percorso ACK.
- Cancellazione e cleanup restano ordinati e idempotenti. Non liberare una sessione adottata, non riprendere un decoder FAILED, non convertire un errore reale in esito di successo.

## 4. Commit listener: sezione critica minima

### Perché il hold può aumentare

`listener.py:3336–3556` non contiene un ramo speciale «fine risposta» che giustifichi da solo il salto a 24–26 ms. Sotto lo stesso lock esegue validazioni, copie PCM, ricostruzione di entrambe le history mel, copie dei tail, hash PCM/reference, concatenazione della lineage, aggiornamento cursori/revisioni, detach e costruzione dell'output.

Le history PCM sono bounded; il codice non ricopia tutta la storia PCM a ogni tick. Ricopia invece gli elementi presenti nello snapshot mel. Il costo dipende da quantità e dimensioni effettive, che i dati forniti non misurano. Il hold è wall time: può comprendere descheduling, perdita del GIL, allocazioni e rilascio dei riferimenti sostituiti, oltre al lavoro utile. Non dimostra che NumPy abbia eseguito 25 ms di copie né che un altro thread abbia trattenuto `state._lock`.

### Preparazione e pubblicazione atomica

La modifica minima è separare preparazione fallibile e commit, conservando il contratto multi-campo dello stato.

1. **Preparare su input posseduti:** validazione di shape/dtype, copie PCM/reference/mel/tails e digest dei byte PCM/reference; preparazione dei riferimenti hidden/query e dei campi dell'output indipendenti dallo stato live. Verificare prima l'ownership dello snapshot owner; non introdurre alias verso buffer riutilizzati dal frontend.
2. **Acquisire `state._lock` e ricontrollare:** stato aperto, configurazione, identità/epoch, frame atteso e versione successiva. Conservare anche il fence di generazione previsto dal chiamante; non trattare una preparazione diventata stale come un nuovo frame valido.
3. **Pubblicare insieme:** history bounded coerenti, tail, cursori/counter/timestamp, lineage, hidden/query, source frame IDs, revisioni semantiche e cache epoch con le regole monotone attuali. La concatenazione della lineage può rimanere qui, usando il digest precedente protetto e i digest PCM già preparati. Non calcolare la lineage leggendo lo stato live senza lock.
4. **Rilasciare e restituire l'output:** completare soltanto il lavoro che non può fallire dopo aver modificato lo stato o rendere incoerente l'evento. Trattenere, quando necessario, riferimenti ai contenitori sostituiti per rilasciarli fuori lock; non lasciare che la distruzione di una history grande allunghi la sezione critica.
5. **Accodare consumed**, poi notifiche shadow/condizionamento e diagnostica differibile.

Lo stato frontend owner (`ListenerSessionState`) e il mirror applicativo (`GemmaListenerState`) hanno ownership differenti. Non rialiasarli per eliminare copie. `_snapshot_speaking_branch_context()` in `server_audio_snapshot_core.py:140–184` legge cursori, history, tail e source IDs insieme sotto lo state lock: deve continuare a osservare uno snapshot interamente vecchio o interamente nuovo. Verificare i reader prima di sostituire un deque: chi ne conserva un riferimento non deve perdere gli aggiornamenti.

Un errore di validazione o allocazione durante la preparazione deve lasciare lo stato invariato. Il codice attuale, per esempio, appende PCM prima di verificare alcuni campi dello snapshot: anticipare queste verifiche migliora anche l'atomicità del fallimento. Conservare la semantica delle revisioni applicate: una row precedente non può retrocedere una revisione più recente pubblicata da un altro attore.

**Non anticipare l'ACK al solo incremento del cursore.** Oggi il server lo accoda già prima della notifica shadow differita (`server.py:5204`, `test_server_consumed_ack_order.py`). Anticiparlo ulteriormente può certificare un commit ancora fallibile e permettere ai reader di osservare cursori nuovi con history vecchia.

Spostare copie fuori dal lock riduce l'esclusione reciproca, ma se restano sullo stesso thread prima dell'ACK non elimina automaticamente il loro costo end-to-end. Misurare separatamente preparazione, hold e totale commit. Trasferire preparazione al producer è una possibilità successiva, solo se ownership e misure la giustificano; non aggiungere un executor per ogni frame come prima soluzione.

## 5. Specifica operativa per Luna

Sono interventi proposti, non patch applicate. Procedere per unità verificabili, aggiornando i test pertinenti e `docs/ARCHITECTURE.md` quando il comportamento cambia. Non serve un nuovo piano rigido né un sistema di certificazione.

| File | Intervento preciso | Vincolo |
| --- | --- | --- |
| `duplex_kernel/server.py` | Misurare route finale effettiva; ridurre il passaggio di metriche complete all'evento finale; mantenere queue-close e submission brevi; ACK subito dopo commit coerente | Nessuna attesa playback/Future sotto lock di transizione/modello/stato; nessuna riapertura prematura del response slot |
| `duplex_kernel/server_metrics.py` | Conservare l'evento finale come delta delle fasi finalizer/rollback e degli errori; non sostituirlo con tutte le metriche storiche | Non perdere evidenza di drain, errori o provenienza; aggiornare i consumer interessati |
| `duplex_kernel/server_realtime_scheduler.py` | Esporre tempi distinti di prepare/commit/build-event e mantenere il significato dei campi esistenti | Non rinominare una misura cambiandone silenziosamente il denominatore |
| `duplex_kernel/listener.py` | Preparazione fallibile fuori lock; commit atomico completo; riduzione allocazioni/distruzioni nel tratto protetto | Frame/epoch/versione/lineage e snapshot multi-campo coerenti; niente ACK su errore |
| `duplex_kernel/shared_listener_scheduler.py` | Misurare wait dei quanti prima del batch e durata host della fanout; preservare l'attuale handoff GPU | Non estendere `_inflight_sessions` fino al trasporto ACK: toglierebbe opportunità al Talker senza provare la causa |
| `duplex_kernel/server_unified_cache.py` | Preservare submit/await separati, close epoch e CAS; aggiungere misure compatte delle singole operazioni quando questo ramo è attivo | Non introdurre drain monolitico o replay duplicato; mantenere rollback e cancellazione ordinata |
| `duplex_kernel/listener_microbatch_worker.py` | Se i tempi owner mostrano un comando/release lungo, renderne differibile la parte costosa e controllare i frame tra quanti | Non bloccare l'owner su un permit il cui rilascio dipende dal listener; non riordinare comandi con dipendenze causali |
| `tools/real_dialogue_simulator.py` e relativi test | Analizzare tutti i gap nella finestra, non solo il massimo; misurare parsing/logging separatamente se resta residuo client | Non cambiare cutoff, timestamp storico ACK o soglie per ottenere un PASS |

I quattro file citati nel prompt non sono sufficienti da soli: la ripubblicazione delle metriche è in `server_metrics.py`, e l'ordine effettivo del lavoro GPU/native è nel worker. Toccarli soltanto dove il contratto e il test richiedono una modifica.

### Ordine pratico

**A. Catturare la proprietà prima della modifica.** Estendere i test esistenti con un evento finale che contiene tracce estese: l'evento realtime deve mantenere le informazioni terminali senza replicare quelle tracce. Aggiungere il test di stato invariato quando la preparazione della row fallisce. Isolare inoltre i test di stale epoch/versione e di lettura multi-campo concorrente. Questi test devono fallire per la proprietà mancante, non per un numero artificiale di millisecondi.

**B. Ridurre il lavoro di pubblicazione già identificato.** Ripristinare il delta finale compatto mantenendo le tracce nel percorso diagnostico esistente. Verificare i consumer del simulatore, il report finalizzato e i test di cancellazione: l'evento finale resta senza fence del token risposta, perché YIELD può averlo invalidato. Misurare una volta il costo/volume del delta e del payload precedente, distinguendo microbenchmark da latenza reale.

**C. Ridurre la sezione critica del commit.** Applicare la separazione prepare/commit descritta sopra. Prima validare ownership, reader e lineage; poi ottimizzare le copie effettivamente dominanti. Nessuna condivisione mutabile di array per risparmiare memoria.

**D. Intervenire sul worker solo sul costo osservato.** Rilevare enqueue comando, dispatch owner, inizio/fine FFI release, ritorno Future ed eventuale fence. Il release attuale usa un event wait CUDA senza attesa host esplicita: non attribuirgli un blocco prima di misurarlo. Se è costoso, separare la chiusura logica dal rilascio fisico sicuro, con ownership esplicita fino al fence. Per il ramo canonico conservare il replay Q2 già a quanti.

**E. Ripetere scenario e analizzare anche i picchi non finali.** Se il gap 1126 → 1127 ha un'origine diversa, correggere quel percorso dopo averlo localizzato. Non proseguire aggiungendo nuovi lock sulla premessa che tutti i picchi abbiano la stessa causa.

### Strumentazione minima utile

Riutilizzare le strutture di timing esistenti, con pochi campi numerici bounded:

- attesa permit e operazione ammessa, separata dal batch;
- owner queue wait e durata dell'operazione finale, distinguendo canonical drain e private release;
- prepare/commit wall e, per localizzare descheduling, CPU del thread; numero/byte delle copie mel;
- evento ACK pronto, inizio/fine serializzazione e handoff al trasporto;
- costo parsing/logging del client e loop lag se rimane una latenza non attribuita.

Non esportare stack o intere timeline a ogni tick. Non aggiungere `torch.cuda.synchronize()` globale al percorso realtime. Correlare ogni misura con frame, batch, generazione e response ID; confrontare timestamp assoluti tra processi soltanto dove il clock comune è stato verificato. Qui client e server giravano sullo stesso host Linux: non generalizzare questa proprietà a Windows e Mondaini.

## 6. Invarianti e verifiche funzionali

### Test mirati prima della GPU

Estendere preferibilmente le suite già presenti:

- `test_server_consumed_ack_order.py`: consumed segue il commit completo e precede lavoro shadow/diagnostico differibile; mai duplicato né emesso per row stale/fallita.
- `test_shared_listener_tick_dispatcher.py`: domanda pubblicata atomicamente, frame arrivato durante un quanto servito prima del successivo; dipendenze CUDA conservate; callback host non tiene la priorità GPU.
- `test_cooperative_drain.py`: tick pronto servito tra replay; epoch di chiusura rispettata per frame arrivati prima/durante/dopo il confine; Future completata solo dopo il drain; cancellazione attende cleanup ordinato.
- `test_listener_microbatch_worker.py` e `test_unified_kv_server_gate.py`: release privata e chiusura canonica percorse separatamente; nessun comando differito supera un prerequisito; nessuna perdita del cursore o del watermark.
- Test del bridge listener: errore di prepare senza mutazioni parziali, reset/stale row, snapshot reader coerente, digest/lineage equivalenti al comportamento valido attuale, history bounded e revisioni monotone.
- `test_unified_controller_lifecycle.py`: terminal replay idempotente, causa FAILED conservata, adozione e cleanup senza doppio rilascio. Mantenere i test attuali anche se il difetto non riappare.
- Test dei consumer della telemetria: delta finale sufficiente a conservare status, errore, ACK/drain/rollback; report non dipendente dalla ripetizione della trace completa nell'ultimo evento.

Usare barriere/eventi per controllare l'interleaving nei test di concorrenza. Non usare sleep e deadline reali per dimostrare un limite GPU di 140 ms su CPU. Non aggiungere mock che facciano passare capacità, EOS o completamento playback non osservati.

### Smoke ed esecuzioni reali

Su Mondaini, serializzare l'uso della RTX 3090. Prima di una suite lunga eseguire uno smoke breve che includa almeno un confine reale di risposta e alcuni frame microphone dopo la chiusura, non soltanto pochi frame a inizio sessione. Confermare il percorso importato e il ramo di finalizzazione attivo usando i diagnostici esistenti. Preservare checkpoint e vincoli Docker del repository.

Poi ripetere:

```text
python3 tools/run_acceptance_suite.py --runs 3 --container gemma4-speech-s2s-readyframes-20261102 --execution docker
```

Eseguire con l'ambiente previsto dal runner e conservare i parametri effettivi. Il comando non viene eseguito in questa diagnosi.

Esaminare report preliminare, report finalizzato ed eventi grezzi insieme. Per ogni run verificare tutti i turni, continuità ingress, consumo degli ultimi frame entro il cutoff esistente, ACK univoci e ordinati, playback/segment completion, rollback causale, follow-up di memoria, assenza di riuso FAILED e crescita VRAM post-drain. Le discrepanze dei contatori nelle run abortite devono essere spiegate dalla finestra osservata o corrette nel report; non vanno ignorate perché l'errore principale è un gap.

Per il nuovo obiettivo, calcolare il massimo di **tutte** le coppie ACK ammesse nella finestra lunga inclusi i frame di coda, e mostrare separatamente gli altri turni. Se si richiede <140 ms per l'intera sessione, includere anche quei turni: il PASS corrente della finestra lunga non copre quel requisito. Mantenere il limite attuale di 160 ms per confrontabilità e riportare esplicitamente il requisito aggiuntivo di 140 ms; non restringere la finestra o scartare gli ultimi frame.

Per passare da 178,141 a meno di 140 ms serve ridurre di oltre 38,141 ms il gap peggiore, a parità delle altre condizioni. Non è un budget automaticamente recuperato eliminando un hold da 26 ms: gap, latenza assoluta e variazione di latenza sono grandezze diverse. Analogamente, togliere l'evento finale non corregge automaticamente il picco precedente da 164,687 ms.

Il contratto deterministico ottenibile con questi interventi riguarda l'ordine delle operazioni, il ricontrollo della domanda listener tra quanti differibili e l'atomicità del commit. Un limite numerico ai quanti davanti a un frame richiede anche una policy esplicita per comandi, prerequisiti e permit: il FIFO owner attuale non basta a dimostrarlo. Il limite wall dipende inoltre da durata dei quanti, scheduling host, frontend, ricezione e trasporto. Tre run consecutive sotto 140 ms sono il risultato sperimentale richiesto; non una garanzia matematica ricavabile dalle modifiche suggerite.

## Indicazione finale all'esecutore

Non rifare il lifecycle e non rimuovere i lock sulla base dei nomi delle metriche. Iniziare dal delta finale compatto e dal commit preparato fuori lock, misurando il costo totale fino al client. Conservare l'owner e i fence, distinguere il percorso revisioned dal drain canonico e correggere gli ulteriori picchi dello streaming con la stessa disciplina causale. La diagnosi non autorizza a dichiarare tre PASS, EOS naturale o latenza <140 ms finché tali risultati non sono osservati nelle nuove run.
