**Il gap da 1567,4 ms è localizzato nella valutazione Control6 eseguita dall’owner del listener. Inoltre, nella stessa risposta esiste un secondo gap da 164,179 ms causato dallo scheduler a griglia fissa: correggere soltanto Control6 non basta.**

Ho letto il prompt, esaminato i sorgenti e recuperato su Mondaini la traccia originale di `artifacts/acceptance_suite_verify_8/run_01/`. Ho aggiunto diagnosi, contratti e specifiche per gli esecutori in [docs/ARCHITECTURE.md:497](C:/Users/andre/Documents/gemma-4-omni/docs/ARCHITECTURE.md:497). Non ho applicato patch funzionali né modificato il servizio.

**1. Dove si interrompe il consumo**

La traccia completa del frame 1100 contiene queste misure:

| Misura | Durata |
|---|---:|
| Gap ACK 1099 → 1100 | **1567,416 ms** |
| Batch listener complessivo | 1553,604 ms |
| Valutazione Control6 | **1504,248 ms** |
| Vista decisionale Q2 | 17,895 ms |
| Frontend audio | 4,137 ms |
| Attesa lock di transizione | 0,001 ms |
| Attesa completamento CUDA finale | 0,456 ms |
| Coda ACK prima dell’invio WebSocket | 0,471 ms |

Control6 occupa circa il **96,8% del batch**. Nei frame immediatamente vicini richiede normalmente circa 1,5–2,3 ms.

La vista Q2 presenta `cache_hit=true`, un solo frame riprodotto e nessun nuovo fork. Non emerge un replay crescente capace di spiegare lo stallo. Il finalizzatore, successivo, completa con zero frame trattenuti e drain da 0,368806 ms.

Il percorso responsabile è:

```text
SharedListenerTickDispatcher
  → submit_tick()
  → attesa della Future del batch
  → ListenerMicrobatchWorker._process_batch()
      → step_native_output_views()
      → _step_policy_context_views()
      → step_with_logits() / _step_impl()
  → fanout, commit del frame e pubblicazione ACK
```

Il dispatcher non passa al frame successivo finché quel batch non termina. Il blocco Control6 trattiene quindi l’owner, ritarda il consumo e fa crescere la coda client fino a 15 frame in-flight. Il successivo invio attende circa 917 ms per backpressure.

Un dettaglio del prompt va corretto: **il server aveva già ricevuto il frame 1100 prima dell’ACK del 1099**. Gli append mostrati durante lo stallo appartengono a frame successivi.

**2. Cosa è dimostrato e cosa resta da misurare**

È dimostrato **dove** vengono assorbiti i 1,5 secondi. Non è ancora dimostrato quale singola operazione interna li provochi.

Il timer `listener_batch_control6_policy_ms` comprende preparazione host, submit CUDA, sincronizzazioni e readback. Nei sorgenti ho verificato:

- due controlli sincroni di finitezza in `NativeControl6LoRA.forward`;
- trasferimenti di `phase_t` verso GPU;
- operazioni CUDA su sei logits;
- diversi `.item()`, conversioni booleane e readback nella policy ordinaria.

Il file `health_pre_run.json` conferma che **`native_control6_fused_readback_enabled=false`**.

Questi sono candidati concreti, ma la traccia non separa attesa GPU, allocatore/driver e sospensione del thread host. Le copie CPU/GPU possono sincronizzare lavoro asincrono precedente: attribuire tutto al calcolo della piccola head sarebbe prematuro. [Documentazione ufficiale PyTorch 2.12](https://docs.pytorch.org/docs/2.12/notes/cuda.html#asynchronous-execution).

Non ci sono quindi evidenze sufficienti per indicare il Thinker lock, un prefill o il Talker come causa specifica del picco.

**3. Il secondo difetto: frame pronto dopo un tick vuoto**

Nella stessa finestra di playout, il gap ACK 1723 → 1724 raggiunge **164,179 ms**.

Qui Control6 richiede appena 2,051 ms. Il problema è temporale:

- il frame arriva circa **1,511 ms dopo** la griglia nominale;
- lo snapshot di quel tick non lo include;
- il dispatcher attende la griglia successiva;
- il frame accumula **78,489 ms prima del dispatch**;
- seguono 44,115 ms di batch.

Il salto di un intervallo e la variazione del tempo di servizio portano il gap oltre 160 ms. `_wait_until()` non viene risvegliato dall’arrivo dell’audio.

**4. Soluzione architetturale da implementare**

La specifica completa, funzione per funzione, parte da [ARCHITECTURE.md:528](C:/Users/andre/Documents/gemma-4-omni/docs/ARCHITECTURE.md:528). Gli interventi sono questi.

**A. Rendere misurabile e ridurre il tratto Control6**

In `listener_microbatch_worker.py::_process_batch` e `gemma_native_control6_batch_policy.py::_step_policy_context_views` separare:

```text
attesa owner
preparazione input
adapter submit/return
readback submit/completion
applicazione policy host
commit listener
```

Usare timestamp host ed eventi CUDA sullo stream effettivo, senza aggiungere una sincronizzazione globale per tick.

Il primo esperimento deve usare il percorso già disponibile:

```text
GEMMA4_OMNI_NATIVE_CONTROL6_FUSED_READBACK=1
```

È una riduzione mirata delle sincronizzazioni, **non una soluzione già provata**: i controlli finite dell’adapter restano.

Se le nuove misure confermano i readback come collo di bottiglia, introdurre un unico pacchetto compatto per batch contenente finitezza, indici raw/effettivi, massa legale e probabilità necessarie. L’aggiornamento host di endpoint e hysteresis deve avvenire dopo la validazione del pacchetto, una sola volta per frame.

Preservare:

- raggruppamento delle sei azioni nelle azioni fisiche;
- precisione BF16 del ramo base e FP32 del delta;
- controlli nonfinite;
- generazioni, provenienza del frame e hysteresis.

Il vecchio helper `step_batch()` non è un sostituto equivalente della policy completa.

Se invece il picco risulta attesa di lavoro GPU precedente, occorre identificare e suddividere **quel lavoro**. Ridurre i readback o aumentare la priorità dello stream non garantisce di eliminarlo.

**B. Risvegliare il dispatcher quando arriva un frame completo**

In `scheduler.py::AudioIngressQueue` introdurre una notifica di disponibilità collegata al dispatcher condiviso. La notifica deve soltanto risvegliarlo, senza consumare il frame e senza invertire l’ordine dei lock.

In `shared_listener_scheduler.py` modificare `_wait_until`, `_dispatch_next_tick` e `_snapshot_ready_rows` affinché:

- l’attesa sia interrompibile da ingresso, resume e stop;
- un frame pronto possa essere selezionato senza attendere la prossima griglia;
- un backlog possa essere recuperato senza pause artificiali da 80 ms;
- resti una sola Future batch in-flight;
- ogni batch contenga al massimo una riga per sessione pronta.

Servono un predicato/contatore di notifica che eviti risvegli persi e identificatori distinti per dispatch e griglia temporale. Il deadline nominale deve rimanere disponibile per misurare il ritardo: non va riscritto per far apparire puntuale il backlog.

Il wiring va collegato nel lifecycle shared di `server.py`. Attivare una modalità del solo scheduler per sessione non modifica automaticamente questo percorso.

**C. Limitare anche l’elaborazione dei comandi dell’owner**

Il drain è già cooperativo: `_run_one_queued_audio_drain_quantum()` esegue un Q2 per giro.

Rimane però un rischio indipendente: `_drain_commands()` continua fino a svuotare la coda e torna dopo `submit_tick` soltanto quando esiste un drain attivo.

La correzione proposta è:

1. eseguire al massimo un comando per giro;
2. dopo `submit_tick`, tornare sempre al batch pronto;
3. limitare anche la scansione di comandi cancellati o differiti;
4. preservare FIFO, close epoch, CAS e ordine delle mutazioni.

Questo elimina una possibile starvation, ma non interrompe una singola valutazione Control6 lunga e non spiega da solo il picco osservato.

**5. Il simulatore non deve compensare il gap ACK**

Le formule attuali distinguono correttamente:

```text
send_gap_net = send_gap_raw − backpressure_wait
ack_gap      = distanza fra ricezioni ACK consecutive
```

`window_backpressure_waits` serve ad attribuire il ritardo del sender. Sottrarlo dal gap ACK nasconderebbe il blocco server che ha provocato la backpressure.

Restano invariati FIFO da 96.000 campioni, massimo operativo di 15 frame in-flight, timeout da 3,5 s e soglia stretta di 160 ms.

Va invece migliorata la diagnosi: conservare metriche e coppia di frame responsabile anche quando il controllo fallisce. Attualmente `long_response_input_window` resta `null`.

**6. Un ulteriore ostacolo all’accettazione**

Il `server.log` raccolto dal runner è vuoto. `server-log-window.json` dichiara `source=unavailable`: il runner non dispone di Docker né di una sorgente log esplicita.

I log esistono sul Docker host. `run_acceptance_suite.py` deve raccoglierli da lì oppure ricevere `--server-log-source`, mantenendo la finestra temporale corretta. Superare il gap potrebbe altrimenti lasciare un’altra evidenza mancante.

La traccia originale corregge anche un dato del prompt: questa risposta riporta **450 token generati**, non 284.

**Verifica svolta e lavoro restante**

I test esistenti mirati su readback Control6 e continuità sono passati: **15 passed, 115 deselected**. Non sono prove GPU del rimedio.

La verifica degli esecutori deve riprodurre prima il frame arrivato 1,5 ms dopo uno snapshot vuoto, verificare fairness e parità della policy, quindi svolgere uno smoke breve su RTX 3090 e infine tre run seriali complete. Il risultato deve rispettare tutta la finestra di playout: eliminare soltanto il massimo da 1567 ms lascerebbe ancora il gap da 164,179 ms.

