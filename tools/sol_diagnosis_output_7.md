# Diagnosi 7 — spike ACK alla fine della risposta

Diagnosi del 4 novembre 2026. Ambito: lettura del codice e degli artefatti, senza patch runtime, riavvii o nuove sessioni GPU.

## Esito

Lo spike è reale, ma **non è ancora dimostrata una singola causa radice**. L'evidenza sostiene un'interferenza nel percorso di servizio del listener durante la coda della generazione. Non sostiene invece «il finalizzatore asyncio blocca l'adapter Control6 per 48 ms».

Tre correzioni sono decisive:

1. `listener_batch_compute_ms` misura il **tempo di servizio del batch**, non il solo calcolo del thread owner né il tempo GPU. Il codice lo documenta espressamente in `duplex_kernel/shared_listener_scheduler.py:92–101`: comprende coda worker e validazione. I 0,225 ms precedono l'ingresso nella callback; non escludono una successiva attesa nella coda dell'owner.
2. In Run 1 il log delle `00:23:25,118` descrive il frame 2701 **già completato**. `Thinker/Talker done` viene registrato alle `00:23:25,168`, dopo quel frame. Il successivo flush asyncio e il finalizzatore non possono iniziare retroattivamente quello spike. Il lavoro terminale **dentro** l'inferenza resta candidato.
3. Il punto dove si accumula il ritardo cambia fra le run: Control6 è lento in Run 1; il frontend domina Run 2; Run 3 ha circa 139 ms fuori dalle principali fasi temporizzate. Correggere soltanto `adapter.forward` non copre il problema osservato.

Esistono costi e accoppiamenti concreti da correggere: postprocess completo della risposta nel ramo streaming, diagnostica terminale CUDA fuori dall'admission ordinaria, proiezione completa del ledger a ogni ACK. Sono difetti di isolamento/costo verificabili nel sorgente; il loro contributo ai singoli spike deve essere separato con misure mirate. Non attribuisco percentuali causali che i log non consentono di ricavare.

## Fonti e ambiente verificati

Artefatti originali letti direttamente su Mondaini:

`/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261104T001943152237Z/`

Sono stati consultati `suite_summary.json`, i tre `report.finalized.json`, gli eventi dei frame critici, il `server.log` e l'health della prima run. Tutte le run e la suite risultano `INCOMPLETE`, con errore sul gap ACK. Tutti e tre i report riportano anche `scenario.evidence_gaps = ["input counters in the preliminary report do not match raw input events"]`: il 100% di ACK dichiarato non elimina questa discrepanza del reporting. Non la uso per negare gli ACK effettivamente presenti.

Container: `gemma4-speech-s2s-readyframes-20261102`, cwd `/workspace`, utente `1000:1000`, IPC host. Interrogato l'interprete indicato nel comando di avvio: `/workspace/.venv_e2e_runtime/bin/python`, Python **3.11.9**, Torch **2.12.1+cu130**, CUDA build **13.0**. Un nuovo processo con quell'interprete risolve worker e batch policy sotto `/workspace/duplex_kernel/`. Questo non è un dump retroattivo dei moduli in memoria del server durante l'acceptance.

Confronto diretto del testo locale/remoto, senza manifest: uguali le righe di `engine.py`, `server.py`, `listener_microbatch_worker.py`, entrambe le policy Control6, `response_event_ledger.py`, `server_pcm_tracker.py`, `native_listener_microbatch.py`, `training/duplex_qlora/native_control6.py` e dei due sorgenti nativi citati sotto. Le differenze iniziali nei tre file principali erano terminatori di riga. I riferimenti seguenti riguardano questi sorgenti correnti; non provano da soli la revisione storica della libreria `.so` caricata nelle run.

I 141 test sono un risultato fornito in input: non ho rieseguito la suite. È stato eseguito soltanto il microbenchmark CPU descritto sotto. Nessun modello scaricato, checkpoint modificato o job GPU avviato.

## 1. Ricostruzione numerica e temporale

| Run / frame | Gap ACK ms | Servizio batch ms | Frontend ms | Native submit ms | Native CUDA ms | Health ms | Policy ms | Completion ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 / 2701 | 202,108 | 176,241 | 2,317 | 46,719 | 44,940 | 17,090 | 65,447 | 6,826 |
| 2 / 2255 | 262,837 | 210,075 | 125,743 | 44,633 | 44,633 | 0,348 | 5,508 | 0,040 |
| 3 / 2973 | 203,357 | 177,076 | 9,140 | 24,904 | 25,002 | 0,096 | 4,071 | 0,072 |

`native_submit` e `native_cuda` si sovrappongono: **non vanno sommati**. Neppure mel/copy/tower vanno aggiunti al frontend che già li contiene. `cuda_completion_wait_ms` include anche il lavoro successivo alla barriera fino alla fine del blocco: `listener_microbatch_worker.py:8734–8766`.

Sottraendo solo frontend, native submit, health, policy e completion dal servizio totale rimangono rispettivamente **37,842 / 33,803 / 138,793 ms**. È un residuo diagnostico, non una fase misurata: comprende porzioni non temporizzate, coda owner, costruzione/commit output e ritorno della callback. In particolare Run 3 impedisce di dichiarare risolta la causa ottimizzando solo la policy.

In Run 2 il frontend contiene mel **19,396 ms**, copy host **76,277 ms**, intervallo CUDA copy **71,256 ms**, tower CUDA **29,892 ms**. È quindi necessario osservare anche gli intervalli CUDA e la submission host, non soltanto gli stack Python della policy.

La cronologia corretta di Run 1, usando gli eventi originali:

| Evento | Timestamp |
| --- | --- |
| ACK frame 2700 ricevuto | monotonic 6535721581245144; UTC 00:23:24.917973 |
| Inizio servizio frame 2701 | monotonic 6535721598053301 |
| Inizio chiamata policy | perf counter 6535721680734727 |
| Fine chiamata policy | perf counter 6535721745794902 |
| Fine servizio frame 2701 | monotonic 6535721774294392 |
| Trace del risultato frame 2701 | log UTC 00:23:25.118 |
| ACK frame 2701 ricevuto | monotonic 6535721783352872; UTC 00:23:25.120092 |
| Thinker/Talker done | log UTC 00:23:25.168 |
| `response.audio.segment_done` ricevuto | UTC 00:23:25.180892 |
| `response.done` ricevuto | UTC 00:23:25.202233 |
| `response.unified_native_finalized` ricevuto | UTC 00:23:25.228428 |

I timestamp perf-counter sono riportati come tali; l'ordine causale rispetto a `done` si verifica anche senza convertirli, tramite trace del risultato e ricezione ACK precedenti al log `done`. Le `00:23:25,381` del prompt non corrispondono a questi eventi nel file ispezionato.

Il batch termina **9,058 ms** prima della ricezione ACK: in questo frame il danno principale precede la pubblicazione. In Run 2 questo tratto vale **28,599 ms**, in Run 3 **6,080 ms**. Anche il callback/trasporto finale va mantenuto sotto osservazione.

Il frame 2699 ha già `turn_generation_id=1` e `native_control6_peer_turn_active=False`, con servizio **45,037 ms**. Quel cambio di contesto non avviene per la prima volta sul 2701 e non dimostra né una race né un reset lento. Il 2702 impiega ancora **90,782 ms** e il 2974 di Run 3 **73,802 ms**: un unico massimo oltre soglia ACK non significa un solo batch perturbato.

## 2. Perché il listener rallenta: ciò che sappiamo

### A. Il confine temporizzato è più largo del worker

`SharedListenerTickResult.batch_compute_ms` è un alias di `batch_service_ms` (`shared_listener_scheduler.py:92–101`). L'owner misura già separatamente:

- `worker_queue_wait_ms`, all'ingresso `_process_batch`, `listener_microbatch_worker.py:8073–8080`;
- `worker_wall_ms`, dopo costruzione output e callback, `:9135–9145`.

Il server copia questi valori nel tick (`server.py:1580–1583`). Occorre esporli **per il frame critico** negli eventi esistenti: i campi esaminati non consentono di assegnare il residuo di Run 3 a queue, compute o validazione. Non serve creare un altro sistema di metriche.

Il blocco post-completion include commit frontend/coordinator (`listener_microbatch_worker.py:8821–8844`), retention delle righe Q2, eventuali callback (`:9030–9039`), snapshot e costruzione output (`:9050–9134`). Anche queste porzioni possono subire descheduling/GIL; una durata lunga qui non indica necessariamente molto lavoro algoritmico nel listener.

### B. GIL plausibile, ma non provato come causa esclusiva

Il ramo terminale Talker esegue, **prima** del suo ritorno:

1. Chiusura della lease Q1: `engine.py:7243–7251`.
2. Flush dell'accumulatore: `:7273–7278`.
3. Concatenazione di tutto il PCM: `:7294`.
4. Controllo finite e `_postprocess(wav)`: `:7297–7299`.
5. Eventuale cattura diagnostica: `:7300–7388`.
6. Finalizzazione residual metrics e timing: `:7415–7416`.

`_postprocess` copia l'intero segnale e attraversa in Python finestre da 20 ms, invocando NumPy per ciascuna (`engine.py:7428–7443`), poi esegue riduzioni sull'intero array (`:7463–7464`). Una risposta acustica di 114,56 s comporta **5.728 iterazioni**. Molte operazioni NumPy possono rilasciare il GIL; questo non elimina il costo del loop Python, delle allocazioni e della riacquisizione fra chiamate. Non è corretto affermare che l'intera funzione trattenga continuativamente il GIL.

**Microbenchmark isolato verificato:** nel container e nell'interprete di serving, tre chiamate dirette a `PipelineEngine._postprocess` sullo stesso array sintetico zero di **2.749.440 campioni float32**, dimensione del WAV ricevuto della Run 1. Risultati wall **60,564 / 55,811 / 51,399 ms**, thread CPU **60,518 / 55,779 / 51,393 ms**. Non esegue inferenza o CUDA. Dimostra un costo CPU terminale dell'ordine di decine di ms, non riproduce la contesa storica e non misura il GIL del listener.

Il WAV ricevuto ha durata acustica **114,56 s**, distinta dai **192,47 s** di durata del playout nel report: non dimensionare il postprocess usando automaticamente la durata wall del playout.

### C. CUDA: sincronizzazioni reali e submission terminale non isolata

La verifica health passa dalla binding (`native_listener_microbatch.py:1950–1957`) a `gemma4_listener_batch_output_health` (`native/gemma4_listener_microbatch/gemma4_listener_microbatch_impl.inc:3695–3774`) e quindi a `gemma4_listener_output_health`: **`cudaStreamSynchronize(ctx->last_stream)` seguito da `cudaMemcpy` D2H sincrono** (`native/gemma4_listener_microbatch/gemma4_listener.cu:3402–3412`). Non è solo lettura di un intero da quattro byte.

Il tempo Python health può includere validazione, attesa CUDA, runtime/driver e riacquisizione del GIL al ritorno da ctypes. I **17,09 ms** non attribuiscono da soli il ritardo alla copia. La misura native CUDA delimita un intervallo sul device (`listener_microbatch_worker.py:8506–8523`, `:10091–10095`): può includere interferenze fra stream e intervalli vuoti causati dalla submission host. Non dimostra che i kernel abbiano raddoppiato il proprio lavoro.

`finalize_frame_residual_metrics` costruisce tensori scalari e una raccolta CUDA→CPU (`engine.py:6469–6495`) **fuori** da `run_talker_gpu_step`. La trace è limitata a **256 entry**, non a 256 scalari: ciascuna può avere norm e cosine (`:6410–6411`, `:6486–6494`). La costruzione CPU della trace prosegue a `:6504–6515`. È un bypass dell'isolamento GPU ordinario da sistemare, ma non è stato profilato durante il frame storico.

La `native_executor.close()` qui può essere la **lease persistente**, il cui `close` chiama `fence_q1_graph()` (`engine.py:302–321`), non la distruzione del modello. `native_qwen_talker.py:1153–1168` può fare snapshot dello stato device o ordinare gli stream. Non attribuire allo stop normale un `qwen_talker_destroy` senza verificare il tipo concreto. Conservare il fence necessario prima del riuso dello slot.

### D. Lock e ACK

Per Run 1: transition wait **0,001 ms**, state-lock wait **0,001 ms**, state-lock hold **0,096 ms**, ACK-lock wait **0,007 ms**. Nessuna evidenza di un lock di commit che spieghi 176 ms. I campi model-lock `null` significano assenza di misura applicabile, non prova che qualunque lock nell'intero processo sia sempre libero.

La policy esaminata non acquisisce direttamente `sess._generation_lock` o `model_lock`. Gli input del batch derivano dal contesto immutabile già associato al frame (`listener_microbatch_serving.py:123–170`, `:187–195`). Il lock ordine quantum→Thinker in `engine.py:6164–6185` è coerente con la correzione descritta e non va invertito di nuovo.

Gli ACK PCM arrivano circa ogni 10 ms anche durante la generazione; non sono soltanto una raffica eccezionale alla fine. Run 1 mostra sequenze 11747–11759 fra UTC 24.927 e 25.046. Il server li elabora sincronicamente sotto `spoken_prefix_lock` (`server.py:11091–11092`). `_record_client_consumed_pcm` copia solo l'intervallo nuovo (`server_pcm_tracker.py:261`), ma `ResponseEventLedger.record_pcm_consumed` esegue **`on_commit(self.project())` a ogni commit** (`response_event_ledger.py:847–848`). La proiezione ricopia/ordina i source token e ricostruisce provenienza e testo (`:874–959`), mentre il callback locale ignora `_projection` (`server_pcm_tracker.py:365`). È costo dipendente dalla storia ripetuto ad alta frequenza, distinto dalla serializzazione delle metriche già corretta. Può contendere il GIL; senza timing non è identificato come causa dominante dello spike.

## 3. Risposta precisa sui 65,4 ms di Control6

Il timer esterno `listener_batch_control6_policy_ms` copre più della sola chiamata alla policy (`listener_microbatch_worker.py:8582–8711`). La chiamata effettiva temporizzata internamente dura **65,060 ms**, con thread CPU **16,520 ms**, differenza **48,540 ms**. Il timer più largo vale 65,447 ms. La sovrapposizione GC registrata per la chiamata è **0 ms**; non esclude deallocazioni o altre attese.

Dentro quella chiamata avvengono:

- controlli di provenienza e `completion_event.wait(consumer_stream)`, che ordina lavoro CUDA (`gemma_native_control6_batch_policy.py:693–698`);
- costruzione hidden e upload di `phase_t` (`:730–742`);
- forward dell'adapter (`:749–754`);
- eventuali letture diagnostiche CPU, se abilitate (`:760–768`);
- `step_with_logits` di ogni sessione (`:794–798`), con operazioni eager e readback `packed_scores.detach().cpu().tolist()` nel ramo fused (`gemma_native_control6_policy.py:447–458`).

L'adapter stesso contiene controlli finite potenzialmente sincronizzanti quando il defer non è attivo, poi `F.linear`/concatenazioni (`training/duplex_qlora/native_control6.py:173–201`). Il batch abilita il defer CUDA solo se tutte le policy coinvolte usano `fused_readback` (`gemma_native_control6_batch_policy.py:743–748`). Il ramo non fused ha ulteriori letture scalari.

**Quindi non sappiamo quale istruzione abbia atteso per quei 48,5 ms.** Possono essere distribuiti fra descheduling, GIL, dispatch/runtime CUDA, upload e readback. `thread_time_ns` non conta il tempo eseguito da altri thread o dal GPU. Né spostare il medesimo forward in un altro thread Python né aumentare la priorità dello stream garantisce di eliminare queste attese.

## 4. Interventi concreti e contratti per gli esecutori

Questi sono interventi proposti, non patch applicate. La sequenza evita di ottimizzare un costo diverso da quello osservato.

### A. Chiudere il buco di misura con i campi già esistenti

Propagare queue/wall dell'owner negli eventi per-frame insieme al servizio esistente. Aggiungere pochi timestamp e delta thread-CPU intorno a: submission owner, ingresso owner, frontend, policy-context, forward, readback, commit/snapshot, consegna future e ritorno callback. Sul Talker: fence Q1, flush, concatenate/postprocess, diagnostica residual, ritorno inferenza. Correlare con session/response/frame e native thread ID.

**Contratto:** misure passive, raccolta bounded, nessun log JSON voluminoso per frame, nessuna nuova `synchronize` per misurare. Usare eventi CUDA esistenti o query dopo la barriera già necessaria. Se resta ambiguous GIL versus GPU, fare una breve acquisizione con stack di tutti i thread e timeline CUDA/scheduling, non un profiler pesante per tutta la risposta.

Separare esplicitamente `service_wall`, `owner_queue`, `owner_wall`, `gpu_interval`. Non sostituire il valore storico con un numero più piccolo per far passare l'acceptance. Il residuo di Run 3 è il primo caso da spiegare.

### B. Togliere il lavoro sull'intero WAV dal percorso live

Il punto è `engine.py:7294–7299`. Prima verificare tutti i consumer del ritorno: risultato WAV, diagnostica `final_pcm`, sample count, percorso legacy e archivi. Il PCM già pubblicato arriva da `emit_decoded_chunk` (`:6926`); normalizzare il WAV a fine turno non modifica quei campioni già sentiti.

**Contratto:** nel serving streaming la chiusura dipende dalla coda finale bounded, non dalla durata dell'intera risposta. Conservare il comportamento offline/non-streaming. Scegliere esplicitamente il contratto del WAV restituito; non restituire un array vuoto come scorciatoia e non cambiare silenziosamente normalizzazione, durata, diagnostica o contatori. Se un consumer richiede l'export completo normalizzato, produrlo fuori dal percorso temporale del listener con proprietà degli array e coda bounded definite. Un altro thread nello stesso processo non elimina automaticamente la contesa GIL.

Non introdurre clipping/fade nuovi o una seconda emissione dell'intero WAV; preservare tutti i campioni della coda EOS e il comportamento cancel/YIELD.

### C. Rendere cooperativo il lavoro CUDA terminale

Separare in `finalize_frame_residual_metrics` raccolta minima di scalari GPU e costruzione Python della trace. La parte CUDA deve usare il compute quantum esistente, senza Thinker lock quando non accede al modello condiviso, registrando il completamento sullo stream produttore corretto. La formattazione CPU deve avvenire dopo il rilascio del permit.

**Contratto:** non racchiudere l'intera finalizzazione in un singolo quantum lungo: trasformerebbe diagnostica in un nuovo blocco d'admission del listener. Preferire un riassunto GPU compatto e raccolta trace opt-in/bounded; non perdere le metriche funzionali necessarie. Considerare anche i calcoli norm/cosine prodotti durante la generazione (`engine.py:6385–6411`), non soltanto lo stack finale. Conservare lifetime dei tensori, fence e proprietà della lease finché l'evento di completamento consente il riuso dello slot. Non cancellare il fence Q1.

### D. Ridurre la proiezione ACK ripetuta, con semantica transazionale

Misurare `record_pcm_consumed`/`project` e il numero di source token. Se il costo è rilevante, evitare la proiezione completa quando il callback non la usa: prevedere un percorso di commit senza proiezione oppure mantenere una proiezione incrementale/versionata. Migrare esplicitamente i chiamanti che dipendono da `on_commit(projection)`.

**Contratto:** validazione range e sequenza, duplicate idempotenti, commit ledger/branch e rollback su eccezione restano atomici. Preservare watermark sample/KV, ACK finali e late-cancel. Non scartare ACK, non anticipare `consumed`, non coalescere ignorando validation/final e non eseguire `project()` sotto un lock nuovo condiviso col listener. Usare `tests/kernel/test_server_consumed_ack_order.py` e i test del ledger come base; aggiungere una proprietà funzionale che il commit senza consumer di proiezione conservi la stessa proiezione finale.

### E. Ottimizzare Control6 solo dopo aver distinto i sottostadi

Se il forward/readback eager domina ancora, preallocare hidden/phase/packet e valutare un piccolo grafo per le forme effettivamente usate. È un'ottimizzazione successiva, non una spiegazione già provata. Non spostare la policy su CPU o usare la decisione del frame precedente per rispettare il budget.

**Contratto:** stessi logits entro tolleranza motivata, stesso mask legale, phase_t, hysteresis, turn/listener generation e identità della sorgente Q2. Il controllo finite deve restare prima del commit/ACK. Un readback/evento deve riguardare proprio quella generazione d'output. Eventuale buffering pinned va preallocato e riusato soltanto dopo completion; non rimuovere la health check.

## 5. Limite dei 60 ms e verifica funzionale

Un worker Python dedicato esiste già. **CPython 3.11 su Linux general purpose e una singola GPU condivisa non offre un limite wall deterministico di 60 ms** attraverso una modifica minima a lock o thread. Restano scheduling OS, GIL, runtime/driver, allocator e lavoro GPU non preemptibile. La priorità stream è un aiuto di scheduling, non un termine massimo di esecuzione. Affinity o scheduling realtime non vanno introdotti come cura della submission non isolata.

Il contratto utile è: nessun lavoro terminale proporzionale alla durata della risposta nel percorso live; nessuna submission CUDA terminale che bypassi l'admission; nessun falso ACK; tempo di servizio **misurato** entro 60 ms nel carico target, lasciando 20 ms nel periodo d'ingresso da 80 ms. Le tre run future possono verificare quel risultato per il campione, non dimostrare un worst case assoluto. Se il limite assoluto è indispensabile, serve uno studio separato di isolamento processo/GIL, scheduling e budget GPU; un processo listener separato introduce IPC, ownership CUDA/KV e memoria aggiuntiva su 24 GB e non è la patch minima.

Gli esecutori devono prima isolare test funzionali RED sui contratti modificati, poi implementare il minimo e aggiornare i test pertinenti o `docs/ARCHITECTURE.md`. Per B: stessi PCM emessi una volta, coda EOS integra, contratto ritorno WAV esplicito, offline preservato. Per C: completamento prima del riuso slot e listener ammesso fra i quanta. Per D: stessa proiezione finale anche con duplicati/cancel/errore del callback. Evitare test che confrontano soltanto stringhe del sorgente o soglie temporali flaky su CPU.

Dopo i test mirati: smoke GPU breve nel container autorizzato, poi acceptance a tre run con input continuo, includendo esplicitamente EOS e post-drain. Riportare separatamente tempo queue/owner/service e ACK gap, frame inviati/consumati, response.done/finalized e VRAM post-drain. Non cambiare la finestra di misura per escludere lo spike, non rilassare la soglia di 160 ms e correggere la discrepanza dei contatori prima di dichiarare complete le evidenze. Nessun nuovo manifest o meccanismo di certificazione è necessario.

## Revisione avversariale e fonti di piattaforma

Una revisione indipendente in sola lettura ha richiesto: contabilizzare il residuo non misurato, separare GIL/GPU, non trattenere il quantum per tutta la diagnostica, contare le 256 entry correttamente e preservare il contratto del WAV restituito. I punti sono integrati nelle proposte sopra. Cross-model omesso nel contesto di esecuzione da stdin.

La documentazione ufficiale conferma che il [GIL CPython limita l'esecuzione Python concorrente](https://docs.python.org/3.11/library/threading.html) e che [le copie CUDA D2H sincrone terminano dopo il completamento della copia; le API possono inoltre bloccare per risorse interne](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html). Queste sono proprietà generali, non misure del tempo perso in questa sessione. Per la sincronizzazione specifica del runtime fanno fede le chiamate effettive nei sorgenti citati, con il limite già dichiarato sulla `.so` storica.
