# sol_diag_3 — Gap ACK durante la risposta lunga

Diagnosi del 30 ottobre 2026. Ambito: ispezione del codice e lettura delle evidenze originali su Mondaini; nessuna patch applicata, nessun riavvio e nessun job GPU avviato.

## 1. Conclusione operativa

Il difetto architetturale individuato è **la pubblicazione del profilo completo per token del Thinker sullo stesso WebSocket realtime degli ACK**, con serializzazione sincrona sul server e parsing, ricopia ricorsiva e logging sincroni nel receiver del simulatore.

Nei due episodi segnalati passa, immediatamente prima dell'ACK ritardato, un `response.thinker_metrics` rispettivamente di **520.316 e 540.813 byte di payload JSON UTF-8 compatto**. Contiene le liste cumulative `unified_native_profile.append_steps` e `sampling_steps`. Questi payload crescono con i token generati. La pubblicazione finale del profilo avviene al termine del decode del Thinker, mentre il percorso audio/microfono è ancora attivo: non occorre che finisca il playout per incontrare il problema.

La conclusione è sostenuta sia dal percorso implementativo sia dalla ricorrenza temporale nei due run. Il batch listener da solo **non spiega** il gap: tra due completamenti batch consecutivi trascorrono 96,335 e 113,277 ms; il gap osservato dal client è 172,207 e 184,925 ms. Il ritardo dopo il batch aumenta di circa 72–76 ms proprio intorno al grosso evento di profiling.

**Limite della diagnosi:** gli eventi storici non separano il tempo trascorso nel lock WebSocket, nella codifica/compressione/trasmissione e nel logging del receiver. Non è possibile attribuire rigorosamente una quota di quel ritardo a ciascuno, né dichiarare dimostrato un preciso intervallo di possesso del GIL. È invece verificato il difetto che mette lavoro proporzionale alla lunghezza della risposta nel percorso degli ACK. La prima correzione raccomandata è eliminare questo lavoro dal canale realtime, conservando i tempi scalari necessari; misurare i confini mancanti permette di verificare la causalità con il successivo smoke e di individuare eventuali contributi residui.

## 2. Evidenze e runtime effettivamente verificati

Suite originale su Mondaini:

`/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261030T135715955563Z/`

Ho letto gli `events.jsonl`, gli `health_pre_run.json`, i report preliminari e quelli finalizzati. I riferimenti di riga agli eventi riportati sotto si riferiscono a questi file remoti, non agli artefatti storici locali.

Il runtime interrogato usa `/workspace/.venv_e2e_runtime/bin/python`; il modulo server si risolve in `/workspace/duplex_kernel/server.py`. Versioni lette dai metadata installati: FastAPI 0.141.1, Starlette 1.6.0, Uvicorn 0.52.4, websockets 17.1. Il Dockerfile non basta a ricostruire le versioni: `Dockerfile:58–60` lascia FastAPI/Uvicorn senza pin e specifica websockets 17.1. Ho quindi letto anche il sorgente Starlette installato. I punti chiave di server, helper ACK, publisher Thinker e receiver hanno gli stessi numeri di riga nel checkout locale e nella copia remota corrente; non è una ricostruzione immutabile del sorgente storico del processo.

Gli health dei tre run dichiarano `unified_kv_enabled: true`, microbatch condiviso attivo, `listener_pair_lock_handoff_enabled: false`, semantic shadow disabilitato. Questo esclude il *legacy pair handoff* come spiegazione specifica dei run osservati; non esclude altri lock condivisi.

Precisazione sui risultati: `run_01/report.finalized.json` è PASS; il report preliminare è INCOMPLETE. In questa suite il post-drain allocated growth finalizzato è **82,087 MiB**, sempre sotto 256 MiB. Il valore 59,4 MiB fornito nel contesto appartiene quindi a un'altra misurazione e non va attribuito a questo file. I report finalizzati dei run 2 e 3 sono INCOMPLETE e contengono la violazione ACK riportata; non li riclassifico come PASS sulla base dell'ACK ratio.

### Run 2: frame 781 → 782 → 783

| Evidenza | Riga in `run_02/events.jsonl` | Valore |
|---|---:|---:|
| ACK 781 osservato | 4603 | 6153007207634529 ns |
| `response.audio.delta` | 4608 | 6153007278905771 ns |
| `response.output_text.delta` | 4609 | 6153007292475818 ns |
| `response.thinker_metrics` | 4610 | 6153007343491136 ns; 520.316 byte |
| ACK 782 osservato | 4611 | 6153007379841168 ns |
| ACK 783 osservato | 4614 | 6153007382543542 ns |

Il profilo contiene **427 append e 855 sampling step**. Il gap 781→782 è 172,206639 ms; 782→783 è 2,702374 ms. Rispetto alla descrizione iniziale, c'è anche un pacchetto audio tra i due ACK.

### Run 3: frame 807 → 808 → 809

| Evidenza | Riga in `run_03/events.jsonl` | Valore |
|---|---:|---:|
| ACK 807 osservato | 4765 | 6153178064856156 ns |
| Ultimo text delta nell'intervallo | 4771 | 6153178154542731 ns |
| `response.audio.delta` | 4772 | 6153178166812479 ns |
| `response.thinker_metrics` | 4773 | 6153178213018263 ns; 540.813 byte |
| ACK 808 osservato | 4774 | 6153178249781525 ns |
| ACK 809 osservato | 4779 | 6153178265066912 ns |

Il profilo contiene **444 append e 889 sampling step**. Il gap 807→808 è 184,925369 ms; 808→809 è 15,285387 ms.

### Riscontro nel run che passa

Nel Run 1 l'evento di profiling della prima risposta lunga è alla riga 3870: **402.686 byte**, 330 append, 661 sampling step. Il gap ACK 668→669 che lo attraversa è **136,197102 ms**, il massimo della finestra lunga. Anche qui lo stesso meccanismo è presente, ma resta sotto soglia. Un altro evento da 402.694 byte alla riga 18676 è seguito da un gap di 139,816241 ms. Questa relazione tra dimensione, posizione dell'evento e gap è un forte riscontro; tre run non bastano a stimare una legge quantitativa del jitter.

## 3. Ricostruzione temporale e significato delle metriche

Definisco `E_i` = fine batch listener del frame i, `C_i` = fine `commit_microbatch_output`, `R_i` = timestamp client registrato per il consumed ACK. La scomposizione è:

`R_i - R_(i-1) = (E_i - E_(i-1)) + [(R_i - E_i) - (R_(i-1) - E_(i-1))]`.

| Intervallo, ms | Run 2: 781→782 | Run 3: 807→808 |
|---|---:|---:|
| Gap ACK osservato | 172,207 | 184,925 |
| Distanza tra fine batch `E_i - E_prev` | 96,335 | 113,277 |
| Fine batch → ricezione ACK precedente | 7,253 | 9,442 |
| Fine batch → ricezione ACK critico | 83,125 | 81,091 |
| Aumento del tratto successivo al batch | 75,872 | 71,649 |
| Distanza tra fine commit `C_i - C_prev` | 109,099 | 125,233 |
| Fine commit → ricezione ACK precedente | 3,065 | 6,021 |
| Fine commit → ricezione ACK critico | 66,173 | 65,714 |
| `server_ack_queue_ms` del frame critico | 16,610 | 15,746 |

I timestamp server sono nel dominio monotonic del runtime Linux; la suite esegue il simulatore nello stesso ambiente host/container, come previsto da `tools/run_acceptance_suite.py:149–159`. Non usare queste sottrazioni server/client senza verificare il dominio di clock quando il client gira su un'altra macchina. Le differenze tra due timestamp entrambi server e tra due timestamp entrambi client restano comunque le misure più solide.

Tre precisazioni impediscono conclusioni sbagliate:

1. **`server_listener_queue_ms` non è una pura coda.** `server_realtime_scheduler.py:176–184` lo calcola come tempo trascorso dall'enqueue alla costruzione dell'ACK, dopo batch e commit. Non sommare 854,55 ms a 87,457 ms: si conterebbe due volte parte del lavoro. È una latenza accumulata del frame, non il gap tra ACK.
2. **`server_ack_queue_ms` non include tutto il trasporto.** Viene calcolato in `server.py:3892–3897`, prima di entrare in `send_ws`; `_ack_enqueued_ns` è creato in `server_realtime_scheduler.py:184`. Esclude l'attesa successiva del `ws_send_lock`, il `send_json`, il trasporto e il receiver. Include invece il ritardo nel bridge thread→async e nell'event publisher fino a quel punto. I 16,61 ms non scagionano il percorso WebSocket.
3. **Il timestamp client non è l'istante di arrivo sul socket.** `real_dialogue_simulator.py:1333–1342` assegna `received_ns` dopo `json.loads`; prima di leggere il messaggio successivo esegue `EventLog.write` e il trattamento dell'evento. Fra il timestamp del grosso evento metrics e quello dell'ACK seguente trascorrono 36,350 ms nel Run 2 e 36,763 ms nel Run 3. Questi intervalli includono logging ed eventuale ulteriore attesa: non sono misure isolate della sola scrittura su disco.

Il completamento batch del frame 783 è a 6153007363288693 ns, quindi precede già l'osservazione client dell'ACK 782. L'arrivo quasi simultaneo di 782 e 783 è coerente con recupero di ACK già pronti dopo uno stallo a valle del batch; non con una perdita di frame. Non prova, da solo, che lo stallo sia interamente nel client.

## 4. Percorso implementativo che produce il problema

### 4.1 Produzione del profilo cumulativo

In `duplex_kernel/server_unified_thinker.py`:

- `144–158`: il profilo della risposta contiene due liste inizialmente vuote.
- `317`, `448`: ogni append aggiunge il relativo record.
- `568–591`: ogni sampling aggiunge un record; gli errori possono forzare una pubblicazione.
- `207–218`: `publish_profile()` ricopia **tutte** le liste e passa l'intero `token_timings` a `_on_thinker_metrics`.
- `500`: pubblicazione dei tempi del primo token.
- `799`: pubblicazione finale dopo il decode/cleanup del Thinker.
- `841–847`: il risultato interno contiene ancora le liste complete.

Il costo della pubblicazione finale è quindi proporzionale ai token generati. Il fenomeno non richiede un accumulo di memoria fra sessioni: una singola risposta più lunga basta ad aumentare il payload. Non risulta motivato annullare la correzione delle risorse residenti o intervenire sul VRAM leak.

### 4.2 Pubblicazione senza proiezione nel canale realtime

`duplex_kernel/server_speech_ack.py:88–114` costruisce `metrics = dict(values)`, aggiorna le metriche locali e invia `{"type": "response.thinker_metrics", ..., **metrics}`. La copia superficiale conserva le liste annidate complete.

Il callback in `server.py:8101–8120` programma `send_ws(event, token)` sull'event loop e ne aspetta l'esito con `run_coroutine_threadsafe(...).result(timeout=5.0)`. Si usa quindi il medesimo writer degli ACK.

Esiste già una proiezione di metriche selezionate in `server_metrics.py:142` e `server.py:3477–3482`, ma il payload di questo evento diretto viene costruito dai `values` originali. Applicare filtri solo alle metriche memorizzate o a `response.done` non corregge il bypass del `response.thinker_metrics`.

### 4.3 Trasporto serializzato e lavoro sincrono

`server.py:3756–3766` acquisisce `ws_send_lock` per ogni scrittura. `3816–3821` controlla il generation fence per gli eventi della risposta. `3836` esegue `await asyncio.wait_for(ws.send_json(payload), timeout=0.08)` mantenendo il lock WebSocket.

Il sorgente **installato**, `/opt/conda/lib/python3.11/site-packages/starlette/websockets.py:171–178`, contiene `text = json.dumps(...)` alla riga 174, prima del primo `await` alla riga 176. La serializzazione del mezzo megabyte avviene quindi sincronicamente nell'event loop. Il timeout asincrono non preempie quella sezione CPU e non comprende l'attesa precedente del lock. Il fatto che non sia apparso un timeout di invio a 80 ms non dimostra un gap ACK inferiore a 160 ms.

Anche prioritizzando l'ACK non si può interrompere un messaggio metrics già entrato in serializzazione/invio. Serve prima ridurre il lavoro e la dimensione del messaggio.

### 4.4 Receiver e logging

`tools/real_dialogue_simulator.py:1333–1354` legge, decodifica e registra ogni evento prima di proseguire. `EventLog.write`, righe `803–820`, applica `_json_safe`, serializza tutto il record, scrive e chiama `flush()` a ogni riga. `_json_safe`, righe `313–324`, attraversa ricorsivamente mapping e liste. Il receiver ricopia quindi i profili per token e li serializza una seconda volta per il JSONL, prima di poter osservare l'ACK seguente.

Il timestamp dell'ACK in `1380–1390` è quello già preso a `1342`: un ritardo introdotto dall'elaborazione dell'evento precedente entra nella misura del gap. Il requisito osservato dal client resta valido; non va sottratto questo ritardo per trasformare il risultato in PASS.

## 5. Risposta alle ipotesi concorrenti

| Ipotesi | Verdetto supportato dalle evidenze |
|---|---|
| `_generation_lock` blocca direttamente il consumed ACK | Il consumed non porta fence di risposta; `server.py:3812–3817` evita quel lock per gli eventi senza token/generation. Possibili attese indirette restano nei callback e nei writer della risposta; non sono cronometrate come causa di questo episodio. |
| Transition lock | Il percorso effettivo è `unified_turn_transition_lock`, acquisito in `server.py:1068–1092`. Per i frame critici l'attesa registrata è 0,001 ms. Non spiega il gap osservato. Non confonderlo con un generico `_transition_lock`. |
| Pair handoff | Disabilitato negli health; anche `server.py:829–837` lo disabilita nel percorso unified. Non è la correzione da tentare. |
| State lock del commit | Attesa registrata 0,001 ms per entrambi i frame critici. Il tratto batch→fine commit cresce a circa 15–17 ms, ma il tempo totale successivo al commit arriva a circa 66 ms: non basta da solo. |
| GIL / scheduler CPU | Plausibile contributo durante copie e JSON sincroni, anche alla durata wall del commit. Nessuna misura storica separa CPU/GIL/descheduling: non assegnare una colpa esclusiva al GIL. |
| Listener o talker GPU | I batch critici durano 87,457 e 102,394 ms. Il loro wall time comprende il percorso batch/future e non equivale al solo kernel GPU (`shared_listener_scheduler.py:486–526`). Contribuiscono al budget ma non spiegano il residuo successivo. Nessuna prova per cambiare stream o kernel come prima mossa. |
| Coda ACK / lock WebSocket | Esiste un contributo osservato di circa 16 ms prima del writer; manca il timer del lock e dell'invio. Il grande evento sul canale condiviso introduce un meccanismo concreto di blocco a valle. |
| Receiver del client | Ha lavoro sincrono proporzionale al payload ed è parte della misura. Non è dimostrato che tutti i 65–66 ms dopo il commit siano client-side. |

Gli ACK sono emessi dopo consumo/commit e prima della policy corrente (`server.py:4070–4087`). La policy può tuttavia ritardare **il frame successivo**: il dispatcher attende il fanout (`shared_listener_scheduler.py:542–563`, `656–666`), e `scheduler.py:933–947` esegue listener, policy e speaker in sequenza. Questo accoppiamento esiste ma non richiede un refactoring preventivo per correggere il difetto di telemetria confermato.

## 6. Specifica per i worker GPT-6 Luna xhigh

Implementare una modifica circoscritta in TDD, tramite Codex CLI come richiesto da AGENTS.md. In questa consulenza non è stata avviata l'implementazione. I worker devono conservare i contratti di consumo, generazione, rollback, EOS e cancellazione.

### A. Riprodurre il difetto a livello del publisher

Estendere i test pertinenti in `tests/kernel/test_unified_kv_server_gate.py`, dove sono già verificati `_publish_thinker_metrics_event`, cancellazione e clock del token0 (`191–224`, `1212–1355`, `1366` e seguenti).

Costruire un profilo con almeno 500 append e 1.000 sampling record, usando le forme reali osservate. Il test deve verificare la proprietà: **il payload realtime mantiene dimensione limitata indipendentemente dal numero di token** e conserva identità, tempi token0/TTFT e provenienza. Nello stato attuale deve fallire perché `send_event` riceve le liste complete.

Separare il contratto del callback interno da quello del messaggio WebSocket: i test che ispezionano il profilo interno non devono essere indeboliti per far passare il test del trasporto. Conservare i test di cancellazione: aggiornamento locale consentito, nessun evento tardivo di risposta dopo cancellazione.

### B. Correzione primaria: proiezione piccola prima della pubblicazione

1. Introdurre una proiezione esplicita e tipizzata per `response.thinker_metrics`, preferibilmente negli helper di metriche esistenti. Conservare `response_id`, tempi scalari e clock già usati dai consumer, i metadati token0 di dimensione fissa e aggregati numerici del profilo. Non inoltrare `append_steps`, `sampling_steps`, o strutture annidate arbitrarie fornite nei valori.
2. Usare questa proiezione in `_publish_thinker_metrics_event` **prima** di `send_event`, senza alterare il generation fence e la gestione degli errori. Una allowlist protegge anche da nuovi campi voluminosi futuri; un controllo solo sui due nomi delle liste non basta come contratto durevole.
3. Obiettivo implementativo proposto: evento JSON UTF-8 sotto **8 KiB**, con il profilo del test e con profili molto più grandi. È un budget di progetto per il nuovo payload, non un limite già esistente né un motivo per troncare silenziosamente campi obbligatori. Limitare strutturalmente i campi, senza serializzare l'intero profilo solo per controllarne la dimensione.
4. In `server_unified_thinker.py:207–218`, evitare la ricopia completa delle liste per la sola pubblicazione realtime. Mantenere separati snapshot diagnostico completo e snapshot realtime. Per gli aggregati, aggiornare contatori/somme/massimi al momento dell'append, evitando nuove scansioni complete per ogni pubblicazione.
5. Conservare il profilo completo dove è realmente richiesto dai consumer diagnostici interni. Prima di cambiare il risultato `runtime_metrics` alle righe `841–847`, cercare e aggiornare i consumer: non cancellare indiscriminatamente i profili dai risultati interni. Se si rende opzionale la raccolta dettagliata o si introduce un buffer limitato, documentare quantità totale, quantità conservata e troncamento; non presentare un campione come trace completo.
6. L'eventuale esportazione dettagliata deve avvenire fuori dal canale realtime e fuori dall'event loop, usando il percorso diagnostico esistente e senza scrittura sincrona durante il tratto sensibile. La prima patch può semplicemente conservare il profilo interno e pubblicare il riepilogo: non serve introdurre un nuovo sottosistema di logging.

Non basta mettere `json.dumps` in `asyncio.to_thread`: il grande messaggio resterebbe sullo stesso trasporto e nel receiver, e il lavoro Python potrebbe ancora contendere CPU/GIL. Non basta eliminare l'evento solo dal logger client: il payload continuerebbe a bloccare il server e il WebSocket.

### C. Misure piccole per verificare il tratto mancante

Aggiungere timestamp server scalari per enqueue effettivo, dequeue, richiesta/acquisizione del `ws_send_lock` e inizio del send dell'ACK. Riutilizzare i timestamp di fine batch e commit già presenti. Registrare completamento del send in una metrica locale limitata: **non può essere inserito retroattivamente nello stesso ACK già inviato**. Specificare clock e confine di ogni campo.

Conservare il significato di `server_ack_queue_ms` per compatibilità e documentarlo. Aggiungere campi distinti per attesa writer e invio, senza rinominare la metrica in modo da cambiare l'interpretazione dei run pregressi. Evitare timeline per-token complete dentro gli ACK: basta una serie limitata di scalari.

Sul client mantenere `received_ns` attuale per la continuità dei report; aggiungere, se necessario, un timestamp immediatamente dopo l'ottenimento di `raw`, prima di `json.loads`, e durata dell'handler precedente/logging. Questo misura la consegna al codice applicativo, non l'arrivo nel kernel. Qualsiasi writer asincrono del JSONL deve essere limitato, preservare ordine e timestamp originari, completare il drain e propagare errori: nessun drop silenzioso di evidenze. Non rifattorizzare il receiver nella prima patch se il payload piccolo risolve già il problema.

### D. Verifica funzionale e criteri di esito

1. RED→GREEN sul payload limitato e sui clock/cancellation già coperti; nessun test wall-clock stretto da usare come prova di realtime su CPU condivisa.
2. Smoke breve su Mondaini, nel container non-root esistente, senza avviare job concorrenti sulla RTX 3090. Verificare che l'evento lungo sia diventato piccolo, che gli ACK restino ordinati/unici e che non siano comparsi errori di cancellazione o EOS. Conservare le risorse worker/dispatcher residenti.
3. Eseguire poi la suite reale `tools/run_acceptance_suite.py --runs 3`, con output distinto, senza cambiare fixture, soglie o budget. Confrontare dimensione del metrics finale, gap ACK intorno a quel messaggio, fine batch→commit→writer e tempi di osservazione client. Servono quattro turni e gli altri controlli esistenti per ogni run, non solo l'assenza dell'eccezione segnalata.
4. Preservare FIFO 96.000 campioni/4 s, in-flight client 15, timeout input 3,5 s, **gap ACK strettamente inferiore a 160 ms**, pacing tolerance corrente 60 ms e budget VRAM. ACK ratio 100% e FIFO entro limite non sostituiscono il requisito sul gap.
5. Se rimane una violazione, usare i nuovi intervalli per scegliere il passo successivo. Se cresce attesa writer con loop libero, valutare priorità ACK limitata e corretta per ordine/fence; se cresce batch/commit, intervenire sul segmento dimostrato; se cresce l'handler client, spostare il logging fuori dal receiver. Non introdurre preventivamente un writer globale, un nuovo processo o un refactoring del dispatcher.

Aggiornare direttamente `docs/ARCHITECTURE.md` con la separazione fra telemetria realtime limitata e profiling dettagliato, e con il significato dei timer. Nessun nuovo manifest SHA, piano rigido, verificatore a cascata o report di certificazione. Nessuna modifica ai checkpoint.

## 7. Esito della consulenza

È identificato un difetto concreto e correggibile nel percorso di pubblicazione della telemetria, con evidenze originali concordanti nei tre run. La quota esatta di ritardo di server, trasporto e client resta da separare con misure mirate; la rimozione del payload cumulativo e la nuova suite devono verificarne l'effetto. Non è ancora dimostrato che questa singola correzione renda tutti i run conformi a 160 ms.
