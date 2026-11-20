# Diagnosi e specifica per Luna: sessioni native e latenza ACK alle transizioni

## Ambito ed esito

Questa diagnosi usa esclusivamente `tools/sol_diagnostic_prompt_11.txt`. Non sono stati ispezionati altri sorgenti, log o processi, né eseguiti test. I riferimenti di linea sono quelli dell'estratto e devono essere riagganciati alle funzioni nell'implementazione corrente. Gli esempi di codice sotto specificano il comportamento richiesto: non sono patch immediatamente applicabili senza leggere i tipi e i contratti effettivi.

Run 3 dimostra che lo scenario completo può riuscire. Non dimostra ancora ripetibilità su tre esecuzioni consecutive. I 122 test riportati non coprono sufficientemente almeno una sequenza di errore/ripresa osservata; il picco ACK richiede invece di chiarire prima la semantica delle misure. Non serve ridisegnare lo scheduler sulla base di un solo campione.

Due conclusioni hanno gradi di certezza diversi:

1. È documentato un tentativo di avanzamento di una sessione nativa già marcata fallita. È un difetto nel contratto tra sessione, controller e handoff. L'eccezione che l'ha originato non è inclusa: la sua natura resta sconosciuta.
2. È documentato un intervallo ACK oltre soglia. CPU/GIL, contesa GPU, drain Talker ed endpointing sono ipotesi concorrenti, non cause accertate. Le durate riportate non possono essere sommate come fasi disgiunte dello stesso frame.

## 1. Sessione nativa: diagnosi

### Evidenza e catena causale

`advance()` marca `_failed = True` quando intercetta un `BaseException`. L'estratto non mostra una registrazione terminale equivalente per quel ramo. Una chiamata successiva supera il controllo su `_final_step`, entra in `_live()` e riceve `native generation session failed; close it before reuse`.

Il traceback prova che il percorso `server_response_runtime.run_response -> resume -> run_controller_to_completion -> advance_controller -> session_decoder.advance` ha raggiunto quello stato. Gli estratti mostrano inoltre che `handoff_adopted_session(exc)` può invocare `run_to_completion` durante la gestione di un'eccezione. Non provano che la sessione adottata e quella fallita siano sempre lo stesso oggetto: Luna deve verificare identità e proprietà prima di modificare quel percorso.

Il controllo in `_live()` è utile e va mantenuto. Il difetto è permettere a un controller di riprendere una sessione non riprendibile, perdendo il significato e potenzialmente la causa del primo errore. Rimuovere `_failed` o ignorare l'eccezione consentirebbe di usare stato nativo/KV parzialmente modificato.

`BaseException` non significa cancellazione: include errori reali e segnali di arresto. Né un timeout né un generico abort provano che esista una revisione valida. Restituire sempre `revision_handoff: True` trasformerebbe guasti reali in risposte apparentemente riuscite.

### Contratto richiesto

Separare esito della generazione e proprietà delle risorse. Non introdurre stati ridondanti se il codice dispone già di equivalenti.

| Stato logico | `advance` | `run_to_completion` / `resume` | Risorse |
| --- | --- | --- | --- |
| OPEN | Avanzamento consentito a un solo proprietario | Prosegue fino a esito o trasferimento esplicito | Possedute dal controller corrente |
| DONE | Ripete lo stesso esito terminale | Ripete lo stesso risultato | Rilasciate o trasferite secondo il contratto esistente |
| CANCELLED | Ripete lo stesso esito di cancellazione | Restituisce la cancellazione; handoff solo se validato | Nessun ulteriore decode |
| FAILED | Rilancia la causa originaria registrata | Rilancia la stessa causa, senza avanzare | Sessione inutilizzabile, cleanup dovuto |

La chiusura delle risorse è una proprietà distinta, idempotente. `close()` non deve cancellare l'esito registrato né riabilitare una sessione fallita. Una sessione chiusa senza esito non è riprendibile. Una nuova generazione richiede una nuova sessione.

Invarianti:

- Ogni uscita definitiva da `advance`, compresa un'eccezione, registra un esito terminale prima di propagarsi al chiamante.
- Dopo un esito terminale non si eseguono più kernel, campionamenti, mutazioni KV o callback di pubblicazione per quella sessione.
- Una sola esecuzione avanza la sessione. La sincronizzazione deve funzionare tra thread e tra i distinti event loop creati da `asyncio.run`; non aggiungere un `asyncio.Lock` condiviso tra loop.
- Il risultato finale è pubblicato una sola volta; ripetere `resume()` non ripete testo, audio, rollback o side effect di finalizzazione.
- Le risorse hanno un solo proprietario. L'adozione trasferisce la proprietà una volta; il cleanup del cedente non chiude il decoder adottato.
- Una richiesta di cancellazione concorrente invalida logicamente la generazione; la liberazione delle risorse attende il completamento dello step già in volo. Mai chiudere memoria usata da un kernel.
- L'errore originale resta la causa principale. Un errore di cleanup è secondario e non lo sostituisce.

### Specifica di implementazione

In `unified_native_decoder.py`, riusare o estendere il contenitore terminale esistente affinché rappresenti anche FAILED e conservi la prima causa. Non fabbricare un `NativeDecodeStep` CANCELLED per un errore GPU. Il ramo di ingresso di `advance()` deve consultare il terminale prima di `_live()`, anche per FAILED.

Ordine richiesto nel ramo eccezionale: classificare l'interruzione, registrare il terminale sotto la sincronizzazione del ciclo di vita, avviare il cleanup sicuro, propagare l'esito appropriato. La registrazione dell'esito non può dipendere dal successo del cleanup. Se `close()` modifica lo stato utilizzato dal replay, conservare il record terminale separatamente.

Schema semantico da adattare ai tipi esistenti:

```python
async def advance(self, ...):
    # Accesso esclusivo già acquisito dal protocollo di ownership.
    if self._terminal is not None:
        return self._replay_terminal()  # FAILED rilancia la causa salvata.
    decoder, state = self._live()
    try:
        return await self._advance_live(decoder, state, ...)
    except BaseException as exc:
        # Questo predicato usa le cause esplicite del protocollo.
        # Non equivale a isinstance(exc, Exception) o a un match del testo.
        if self._is_expected_cancel(exc):
            self._record_cancelled_once(exc)
        else:
            self._record_failed_once(exc)
        self._request_safe_cleanup_once()
        raise
```

Lo schema lascia intenzionalmente al contratto corrente la modalità di consegna della cancellazione. Se `asyncio.CancelledError` è usato per cancellare un task, preservarne la propagazione dopo la registrazione e il cleanup; il controller leggerà l'esito registrato alla successiva consultazione. Se il decoder usa già un risultato CANCELLED per la cancellazione cooperativa, preservare quel canale. Non convertire arbitrariamente i due meccanismi l'uno nell'altro.

In `server_unified_thinker.py`, rendere `run_controller_to_completion` un accesso all'esito del controller, non un ciclo che riparte sempre da `advance_controller`. Prima di avanzare deve verificare terminale e proprietà. Se il controller è terminale, restituisce/rilancia il suo esito. Se un altro chiamante lo sta completando, si unisce al completamento esistente attraverso un meccanismo compatibile con thread e loop, senza avviare un secondo ciclo. Evitare attese sotto lock che possano bloccare callback o rientranze dello stesso proprietario.

Separare due operazioni oggi potenzialmente confuse:

1. consultare l'esito del controller cedente;
2. completare il controller adottato, se esiste ed è autorizzato a proseguire.

`handoff_adopted_session(exc)` deve verificare che l'adozione sia già stata registrata per la stessa revisione/generazione, che l'oggetto adottato sia quello atteso e che non sia FAILED. Non deve riprendere l'oggetto cedente terminale. Un controller adottato DONE può restituire il risultato memorizzato; uno CANCELLED non genera una nuova revisione per il solo fatto di essere cancellato.

`revision_handoff: True` è ammesso solo per un trasferimento di revisione effettivamente validato, con esito e ownership coerenti. In assenza di adozione valida, restituire l'esito CANCELLED previsto dal protocollo oppure propagare il guasto originale. Non mascherare un errore CUDA come handoff riuscito.

In `server_response_runtime.py`, il ramo intorno a `think = resume()` deve consumare lo stesso contratto: risultato memorizzato, cancellazione esplicita o errore originale. Finalizzazione e pubblicazione devono essere eseguite una volta per generazione. Il vecchio produttore non può pubblicare dopo il trasferimento alla revisione. Mantenere i confini esistenti di rollback causale e PCM già confermato.

Un errore reale può richiedere l'abbandono della risposta o, per un contesto CUDA compromesso, del worker. Questo lavoro elimina il riuso illecito e la perdita della causa; non promette che qualsiasi errore renda sicuro continuare la stessa sessione WebSocket/GPU.

## 2. ACK finale: diagnosi

### Cosa si può dedurre dai timestamp

Assumendo che gli eventi usino lo stesso orologio o timestamp confrontabili:

| Intervallo | Durata |
| --- | ---: |
| Invio 1127 -> invio 1128 | 81,688 ms |
| Invio 1127 -> ACK 1127 | 55,703 ms |
| Invio 1128 -> ACK 1128 | 155,867 ms |
| ACK 1127 -> ACK 1128 | 181,852 ms |
| Invio 1128 -> consumed 1128 | 153,604 ms |
| consumed 1128 -> ACK 1128 | 2,263 ms |

L'identità `gap ACK = intervallo invii + latenza ACK corrente - latenza ACK precedente` produce 181,852 ms. La differenza rispetto ai 182,27 ms del report va ricondotta agli esatti eventi, agli arrotondamenti o alla definizione della metrica; non cambia il superamento della soglia.

Il ritardo osservabile è quasi interamente precedente a `consumed`. Non c'è evidenza, in questo campione, di una coda di invio ACK da decine di millisecondi dopo quel timestamp. Va comunque verificato dove il server assegna il timestamp di `consumed`: creazione del payload, commit o invio effettivo non sono sinonimi.

`117,568 ms` di queue più `111,071 ms` di compute danno `228,639 ms`, superiori ai `153,604 ms` tra invio e consumed. Non sono quindi fasi seriali disgiunte attribuibili al frame 1128 secondo questa lettura. Possibili spiegazioni da verificare: queue misurata sul frame più vecchio del batch, aggregato condiviso tra frame, finestre sovrapposte, timestamp di origine diversi o clock non confrontabili. Anche i sottotimer possono essere annidati: non sommarli senza verificarne i confini.

### Risposte alle ipotesi causali

**Mel e CUDA più lenti.** L'aumento contemporaneo di fasi CPU, submit GPU e lock è compatibile con una pausa di scheduling o contesa comune. Non identifica il GIL. Un timer host attorno a un'operazione CUDA può includere attese, sincronizzazioni o lavoro precedente; un timer con eventi CUDA misura un intervallo diverso. Non conosciamo qui quale dei due produca `native_cuda_ms`.

**Queue a 117 ms.** Può indicare dispatch tardivo, attesa del quanto non preemptibile precedente, accumulo di batch o attribuzione del timer al batch. Non prova da sola che `compute_quantum` non funzioni. L'ammissione cooperativa non interrompe kernel già lanciati e non limita automaticamente lavoro CPU, lock o code di invio.

**Frame finale.** È l'ultimo frame della richiesta lunga riportata; non è dimostrato che il server sapesse già che sarebbe stato l'ultimo quando lo ha accodato. La transizione successiva può essere conseguenza del consumo del frame, non la causa del suo ritardo. Non introdurre una priorità speciale per il numero 1128 né presumere un messaggio di fine stream che l'estratto non mostra.

### Contratto ACK e confine di transizione

Conservare la semantica di `input_audio_buffer.consumed`: l'ACK conferma consumo e commit coerente dello stato listener, non la semplice ricezione in una coda. Emetterlo prima del compute per far scendere la metrica falsificherebbe il contratto. Se esiste un evento di ricezione distinto può restare distinto, senza sostituire consumed nel test.

Ordine logico richiesto per ogni frame/batch:

```text
ricezione -> coda listener -> ammissione -> compute listener
         -> commit minimo dello stato -> ACK consumed pronto
         -> accodamento ACK nel percorso di uscita ordinato
         -> ammissione del lavoro della transizione
```

Il lavoro necessario a decidere lo stato listener rimane prima del commit. Preparazione Thinker/Talker, prefill e attività di finalizzazione differibili non devono essere eseguiti nel tratto critico tra commit e accodamento ACK. Non inserire un'attesa di round trip del client prima di avviare la risposta: basta rendere l'ACK pronto e cedibile al sender.

Usare il writer WebSocket ordinato esistente. Se oggi più produttori chiamano direttamente `send`, Luna deve rendere esplicita la serializzazione e impedire che blocchi di audio/testo già accodati ritardino indefinitamente gli ACK. Preservare l'ordine degli ACK e le dipendenze tra eventi; una priorità non autorizza a scavalcare eventi causalmente necessari. Non introdurre una seconda connessione o un secondo writer concorrente come scorciatoia.

Il dispatch deve dare precedenza al listener pendente al successivo confine cooperativo, includendo il lavoro di transizione nelle stesse regole di ammissione. Separare dalla sezione critica di commit callback, serializzazione pesante e lavoro di risposta. Non aggiungere lock globali o una condition che resti occupata durante operazioni di rete.

Se l'evidenza mostra un singolo quanto di transizione troppo lungo, suddividerlo dove il modello lo consente mantenendo lo stato corretto; non basta rinominarlo o trasferirlo in un thread. Se emerge contesa CPU, ridurre il lavoro eccessivo o il parallelismo responsabile sulla base della misura. Non fissare thread count, affinity o priorità del sistema operativo senza prova.

### Misure minime necessarie prima della patch prestazionale

Estendere la telemetria esistente senza creare manifest o infrastrutture di certificazione. Per frame e batch, correlare identificativi di sessione, generazione, frame, batch e tipo di quanto. Usare un clock monotono locale per le durate server:

- ricezione/enqueue del singolo frame, ingresso dispatch, attesa ammissione, inizio/fine compute;
- inizio/fine commit, ACK pronto, enqueue al sender, inizio/fine send;
- inizio/fine dei quanti Thinker/Talker/transizione concorrenti;
- frame iniziale/finale del batch e distinzione tra durata per batch e attesa per frame.

Riutilizzare misure già presenti quando corrette. Esplicitare se il timer GPU è host o CUDA e quali sottofasi sono annidate. Non aggiungere `torch.cuda.synchronize()` per frame al percorso di produzione: altererebbe il fenomeno. La timeline deve spiegare il ritardo del frame senza sommare intervalli sovrapposti né sottrarre clock di macchine diverse.

L'obiettivo operativo è osservare ACK gap <140 ms anche alle transizioni, conservando la soglia esistente di fallimento a 160 ms. Un limite assoluto deterministico su Docker/Linux, rete e client ordinari non è dimostrabile con queste evidenze. Le tre run provano la ripetibilità osservata nello scenario, non un vincolo temporale universale.

## 3. File e ordine di lavoro per Luna

L'elenco seguente è la superficie di modifica identificabile dall'input. Non inventare percorsi di test o il nome del modulo che emette ACK: non sono forniti. Leggere i proprietari effettivi prima di modificarli; nessun nuovo documento di piano è richiesto.

| File | Intervento |
| --- | --- |
| `duplex_kernel/unified_native_decoder.py` | Esiti terminali completi, conservazione prima causa, replay senza compute, cleanup idempotente e ownership |
| `duplex_kernel/server_unified_thinker.py` | Completamento idempotente, esclusione del doppio avanzamento, handoff validato, distinzione cedente/adottato |
| `duplex_kernel/server_response_runtime.py` | Adeguamento del consumo di `resume()` e della finalizzazione/pubblicazione al contratto terminale |
| `duplex_kernel/server.py` | Verifica e modifica, se posseduti qui, del commit listener, accodamento consumed e gestione dell'errore originario nel produttore |
| `engine.py` — percorso completo non fornito | Modifica solo se la timeline prova che dispatch o quanti di transizione violano la precedenza listener; l'input non documenta già quel difetto |
| `tools/run_acceptance_suite.py` | Riutilizzare il runner; modificare solo se serve a distinguere correttamente le misure o identificare il campione, senza rilassare soglie |
| Test pertinenti esistenti — percorsi non forniti | Aggiungere regressioni del ciclo di vita e dell'ordine commit/ACK/transizione vicino ai test attuali |
| `docs/ARCHITECTURE.md` | Aggiornare direttamente contratti terminali, proprietà del controller e semantica ACK, in forma breve |

Se commit/ACK o scheduler appartengono a un modulo diverso da quelli elencati, intervenire sul proprietario effettivo e riportare il percorso nel risultato finale. Dagli estratti non è possibile dare onestamente un elenco esaustivo di percorsi verificati.

Sequenza di implementazione:

1. Isolare una regressione che riproduca errore iniziale seguito da `resume()`; deve fallire prima della correzione mostrando il riuso o la perdita della causa.
2. Correggere sessione e controller, poi adattare i chiamanti. Rivedere in modo avversariale i casi «errore reale scambiato per revisione», «adottato uguale al cedente», «cleanup chiude risorse trasferite» e «due resume pubblicano due volte».
3. Chiarire i timer e raccogliere una timeline breve della transizione. Correggere il proprietario del ritardo dimostrato; mantenere commit e consumed coerenti. Non applicare simultaneamente più ipotesi prestazionali senza poter attribuire l'effetto.
4. Aggiornare test e architettura, eseguire smoke breve sulla 3090, quindi la suite di tre run consecutive.

## 4. Verifiche funzionali richieste

### Sessione e controller

- DONE seguito da due `resume()` restituisce lo stesso risultato; contatore di avanzamenti e pubblicazioni invariato dopo il primo completamento.
- Cancellazione cooperativa prima dello step, durante una pausa prevista e subito prima del completamento: terminale coerente, nessun riavvio.
- Cancellazione del task async: propagazione preservata, stato terminale registrato e risorse chiuse in sicurezza.
- Eccezione reale nel decoder seguita da `resume()`: si osserva la causa originale, nessun nuovo step nativo; mai il solo errore secondario di riuso.
- Cleanup che fallisce: causa primaria conservata, errore di cleanup osservabile, nessun riuso.
- Handoff valido: trasferimento una volta, revisione corretta, cedente fermo e adottato completato una volta.
- Adozione assente, obsoleta o riferita al cedente fallito: nessuna falsa risposta `revision_handoff: True`.
- Due chiamanti concorrenti di `resume()` da thread/loop distinti: un solo avanzamento e una sola finalizzazione. La prova deve attraversare il confine reale `asyncio.run` usato dal controller.
- Rollback e cancellazione non modificano il PCM già confermato né consentono pubblicazioni tardive dalla generazione cedente.

### ACK e scheduler

- Test d'ordine: un frame risulta consumed solo dopo commit; il lavoro differibile della transizione parte dopo l'accodamento dell'ACK.
- Con writer rallentato e output Talker concorrente, gli ACK mantengono ordine e dipendenze; nessun send concorrente non controllato.
- Con listener pendente e un quanto di transizione già in volo, il listener è ammesso al confine previsto. Verificare anche rilascio della condition e delle prenotazioni in caso di errore/cancellazione.
- I timer per frame e batch sono distinguibili. Il test non deve imporre la falsa uguaglianza queue + compute quando le finestre si sovrappongono.

I test locali verificano stato e ordine, non dimostrano un limite di latenza GPU/rete. Le misure prestazionali vanno eseguite nel runtime Docker effettivo su Mondaini.

### Smoke GPU e tre run consecutive

Usare Mondaini, una sola attività GPU alla volta, senza download pesanti locali né modifica dei checkpoint. Se occorre creare un container, mantenere uid:gid 1000:1000, `--ipc=host`, volume repository in `/workspace`, nessun privilegio o rete host. Non assumere che un processo già avviato abbia caricato il codice aggiornato: verificarlo nel runtime con un controllo mirato, senza manifest SHA.

Prima uno smoke breve con generazione, cancellazione/revisione e completamento: nessun riuso di sessione fallita, nessuna doppia pubblicazione o cleanup concorrente. Poi eseguire il comando già usato:

```bash
python3 tools/run_acceptance_suite.py --runs 3 --container gemma4-speech-s2s-readyframes-20261102 --execution docker
```

Criteri di riuscita:

- Tre run consecutive PASS nella stessa verifica finale sul codice corretto, senza selezionare retroattivamente le run riuscite.
- In ogni run: `overall_status: PASS`, `evidence_gaps: []`, `errors: []`, tutti i quattro turni completati, inclusi risposta lunga, revisione/barge-in, memoria e chiusura.
- Nessun riuso di sessione fallita, nessuna perdita della causa originaria, nessun handoff falso, rollback causale e pubblicazione una sola volta.
- TTFT, continuità audio, ACK e VRAM post-drain rispettano i budget esistenti. Non rilassare soglie, allungare timeout o cambiare consumed per ottenere PASS.
- Riportare separatamente il massimo ACK gap e il margine rispetto a 140 ms per ciascuna run, includendo il confine finale. Una run tra 140 e 160 ms può soddisfare la soglia attuale ma non l'obiettivo aggiuntivo <140 ms.

Luna deve consegnare i file effettivamente modificati, l'esito dei test eseguiti e la causa del ritardo supportata dalla timeline. Se il picco non ricompare e manca una prova della causa, dichiarare che la causa prestazionale rimane non isolata, anche in presenza di tre PASS. Non attribuire a CPU, GIL o Talker una responsabilità che i dati non distinguono.
