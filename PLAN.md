# Implementazione finale del contesto multimodale continuo

## 1. Contratti e problemi da correggere

### Architettura scelta

- Listener con Q2 persistente ogni 80 ms.
- Control6 decide l’azione; V1 genera soltanto il parlato.
- Thinker su snapshot privato della cache canonica.
- Un solo stream Qwen/Talker per risposta.
- History con audio utente e testo assistant confermato.
- Compattazione tramite sintesi semantica dei contenuti vecchi, preservando gli ultimi turni originali.
- Nessun ASR utente richiesto per commit, memoria o ricostruzione KV.

### Buchi confermati nei sorgenti

| Problema | Correzione richiesta |
|---|---|
| Il drain della history può aspettare `transcript` e `assistant_finalized`. | Il commit del turno utente dipende dall’archiviazione dell’audio, non da una trascrizione. |
| La compattazione richiede copertura ASR dei frame espulsi. | Sostituire la copertura ASR con riferimenti audio e copertura della memoria semantica. |
| Il ring recente contiene soltanto 200 frame. | Archiviare anche i soft-token necessari prima che escano dal ring. |
| Un intervento lungo ancora aperto non è compattabile. | Permettere chunk audio immutabili appartenenti a un turno ancora aperto. |
| Ogni frame voiced può richiedere una revisione. | Solo REPLAN autorizza un candidato di revisione. |
| La pulizia del parent chiude indiscriminatamente i branch figli. | Separare sostituzione della cache canonica e chiusura della sessione. |
| Alcuni percorsi arrivano a `await future` dopo aver lasciato il ciclo con timeout. | Applicare la deadline all’intera lifetime della risposta. |
| Il riferimento assistant segue l’invio, non il rendering. | Introdurre un watermark distinto del playback del dispositivo. |

Le quattro correzioni già implementate rimangono e devono continuare a passare i propri test.

## 2. Modello dei dati, proprietà e budget

### Archivio autorevole fuori GPU

Estendere la history della sessione con record tipizzati:

```text
UserAudioSpan
  turn_id, chunk_id, revision
  first_frame_id, last_frame_id
  pcm_reference, soft_tokens_reference
  encoder_profile
  turn_open

AssistantTextSpan
  turn_id, response_id, revision
  confirmed_text
  source_character_range
  pcm_reference, rendered_sample_frontier

MemoryRevision
  memory_id, parent_memory_id
  covered_source_ranges
  source_revisions
  semantic_notes
  open_turn_state
```

Regole:

- Un chunk audio diventa immutabile quando i suoi campioni e soft-token sono stati archiviati.
- `turn_open` distingue un chunk completo dalla fine dell’intervento.
- Il testo assistant entra nella history soltanto fino alla frontiera confermata.
- Gli intervalli indicano esattamente quali contenuti sono stati letti dalla sintesi. Non certificano che ogni dettaglio sia sopravvissuto al riepilogo.
- Le trascrizioni diagnostiche restano in un canale separato e non modificano questi record.

Archiviare il PCM16 ricevuto dal server in segmenti, con scrittura fuori dal thread realtime. Conservare i soft-token pre-decoder con provenienza dell’encoder: non usare gli hidden finali del Listener come sostituti dell’audio.

Il ring RAM rimane bounded; i contenuti necessari alla manutenzione vengono letti dall’archivio tramite buffer di staging. Il tick non attende scritture su disco o copie GPU→CPU.

### Proprietà della KV

Estendere il worker esistente in [listener_microbatch_worker.py](C:/Users/andre/Documents/gemma-4-omni/duplex_kernel/listener_microbatch_worker.py), evitando un secondo manager.

Tre ruoli:

| Ruolo | Uso |
|---|---|
| CANONICAL | Listener e audio continuo. |
| ACTIVE | Risposta corrente. |
| WORK | REPLAN, SUMMARY oppure REBUILD, mutuamente esclusivi. |

Separare negli handle:

- identità della sessione;
- generazione della risposta;
- revisione della risposta;
- epoch del proprietario KV;
- revisione della memoria e riferimenti sorgente.

L’epoch canonica registrata all’origine di ACTIVE descrive il suo snapshot. Non deve essere confrontata indiscriminatamente con l’epoch canonica corrente a ogni append: il branch possiede una copia isolata e può continuare dopo una compattazione.

La chiusura della sessione invalida tutti i proprietari. La sola sostituzione di CANONICAL no.

### Budget iniziali

| Parametro | Valore |
|---|---:|
| Budget operativo per cache | 4.096 token |
| Avvio preparazione memoria | 2.048 token canonici |
| Soglia desiderata di sostituzione | 3.072 token |
| Obiettivo della nuova cache | ≤2.048 token |
| Prefisso iniziale importato | ≤512 token |
| Memoria semantica generata | ≤256 token |
| Audio recente originale di riferimento | 200 frame, 16 secondi |
| Testo assistant per risposta, incluse revisioni | ≤768 token |

Le soglie sono parametri del servizio. La capacità ABI microbatch resta 16.384.

Prima di aprire ACTIVE o CANDIDATE verificare la somma effettiva di prefisso, transizione, prefisso assistant congelato e generazione restante. REPLAN non rinnova il budget totale della risposta.

## 3. Ciclo completo della compattazione

### Preparazione della memoria

Implementare una macchina a stati:

```text
IDLE → SUMMARIZING → BUILDING → CATCH_UP → READY → SWAP → IDLE
```

`CANCELLED` e `FAILED` rilasciano WORK senza modificare memoria o cache canonica correnti.

All’avvio:

1. Scegliere un cutoff esplicito dei contenuti vecchi.
2. Congelare gli intervalli sorgente e le loro revisioni.
3. Conservare fuori dalla sintesi l’audio recente e il testo assistant non ancora confermato.
4. Preparare un input privato con memoria precedente, audio da compattare e testo assistant confermato.

**Non fare fork della cache Listener più recente per poi dichiarare un cutoff precedente:** quel contesto potrebbe già contenere audio successivo. SUMMARY deve vedere soltanto le sorgenti dichiarate.

Usare i soft-token archiviati per il prefill privato; non rieseguire il Conformer live e non avanzarne lo stato.

### Percorso SUMMARY isolato

Creare un handle di manutenzione distinto da `ResponseDecodeHandle`.

Il job SUMMARY:

- usa gli stessi pesi Gemma congelati;
- non acquisisce un talk-context;
- non passa attraverso `run_response`;
- non crea `response.created`;
- non alimenta LiveQwenSayStream o TTS;
- non modifica `session.response_id` o la generazione della risposta;
- non pubblica hidden verso FiLM;
- procede in quantum brevi nel worker GPU.

Il risultato contiene soltanto note semantiche:

- fatti e preferenze rilevanti;
- decisioni;
- richieste aperte;
- correzioni o contraddizioni;
- stato dell’eventuale intervento ancora in corso.

I riferimenti sorgente vengono validati contro l’input del job. Non promuovere un risultato troncato, malformato o riferito a sorgenti inesistenti.

La memoria viene serializzata come dati di contesto con delimitazione fissa. Non concatenare il testo generato dal summarizer alle istruzioni privilegiate del sistema.

### Interventi utente lunghi

Non aspettare necessariamente la fine del turno.

- Suddividere l’audio ricevuto in chunk immutabili.
- Compattare il prefisso vecchio del turno mantenendo `turn_open=true`.
- Conservare l’audio recente originale, inclusa la continuazione dell’enunciato.
- Il riepilogo deve rappresentare come provvisorie intenzioni o richieste non ancora completate.

Questo evita che una singola richiesta lunga renda impossibile qualsiasi compattazione.

### Costruzione e sostituzione della nuova KV

Dopo SUMMARY, riutilizzare WORK per REBUILD:

1. Prefill delle istruzioni stabili e della nuova memoria.
2. Replay cronologico dei segmenti multimodali mantenuti.
3. Aggiunta dei frame arrivati durante la preparazione.
4. Verifica di copertura e revisione delle sorgenti.
5. Scambio del proprietario canonico al confine di un tick.
6. Rilascio del vecchio proprietario dopo gli eventi CUDA pendenti.

La ricostruzione usa il template effettivo del modello: delimitatori user/assistant, BOA, EOA e chiusure di turno. Frame sorgente, indice di replay e posizione KV sono contatori distinti.

Nuovi contenuti successivi al cutoff non invalidano automaticamente la sintesi: vengono aggiunti durante il catch-up. Una modifica a contenuti già coperti, oppure alla memoria genitore, la invalida.

Mantenere una sola richiesta di manutenzione pending, aggiornata allo stato più recente. Non ricominciare SUMMARY per ogni nuovo frame.

### Contesa e failure

- Un REPLAN può interrompere la manutenzione; l’ultimo checkpoint valido della preparazione resta riutilizzabile solo se le sorgenti sono ancora compatibili.
- Sotto pressione di contesto, completare una ricostruzione già valida ha precedenza su un nuovo candidato non ancora iniziato.
- Il Listener non attende il job di manutenzione.
- Nessuna operazione alloca un quarto slot perché un vecchio proprietario è ancora in retirement.
- Se la nuova cache non è pronta alla soglia desiderata, usare il margine residuo.
- Al limite operativo, emettere `context_limit` e terminare ordinatamente: mai scartare audio o promuovere una sintesi incompleta.

“Immediata” significa scambio di una cache già preparata. Non significa generazione istantanea della memoria al limite.

## 4. Patch runtime, native e training

### P0 — History senza ASR operativo

In [session.py](C:/Users/andre/Documents/gemma-4-omni/duplex_kernel/session.py):

- Sostituire l’attesa del transcript nel commit con lo stato di archiviazione dello span audio.
- Rendere indipendenti il completamento del turno utente e quello assistant.
- Aggiornare la history tramite record audio/testo, non tramite messaggi user testuali sintetizzati dall’ASR.
- Conservare l’ASR diagnostico senza effetti sullo stato conversazionale.

Nel preparatore di compattazione eliminare `_VerifiedSpan` ASR come requisito per rimuovere audio dalla KV. Il nuovo criterio è la copertura esplicita delle sorgenti incluse in una `MemoryRevision` valida.

### P1 — Canonico Q2 e controllo delle revisioni

- Q2 persistente su ogni frame ordinario.
- Eliminare il bypass della head per PCM esattamente nullo.
- Rimuovere `request_revision` dal ramo VAD voiced.
- SPEAK/BACKCHANNEL/REPLAN aprono esclusivamente i branch previsti.
- Le istruzioni relative all’azione restano private; il prefisso canonico è stabile.

Il tracker vocale può annotare confini e attività, ma non sostituisce l’azione scelta da Control6.

### P2 — Proprietari nativi e lifecycle

Riutilizzare le API response già presenti. Aggiungere comandi di manutenzione equivalenti a:

```text
begin_memory_job(source_snapshot)
append_memory_input(handle, prepared_chunk)
advance_memory_decode(handle)
begin_canonical_rebuild(memory_revision, retained_sources)
append_rebuild_chunk(handle, prepared_chunk)
promote_canonical_rebuild(handle, expected_source_revisions)
cancel_maintenance(handle)
```

La promozione aggiorna descrittori e identità del proprietario. Non deve chiamare la pulizia generale dei branch figli della sessione.

Dopo promozione:

- il nuovo WORK diventa CANONICAL o ACTIVE;
- il vecchio proprietario passa a RETIRING;
- il suo slot torna WORK soltanto dopo le fence;
- gli append usano l’identità del proprietario, non un riferimento mutabile al parent corrente.

### P3 — Miglioramenti CUDA necessari

Nel backend cache nativo:

- Separare capacità ABI e capacità allocata per slot.
- Preallocare K/V e scratch per il budget operativo.
- Conservare la copia del solo prefisso valido, già implementata.
- Preservare gli alias dei layer che condividono KV.
- Promuovere tramite descrittori senza ricopiare l’intera cache ricostruita.
- Mantenere i cursori aggiornati sul worker e gli ordinamenti fra stream tramite eventi.
- Evitare allocazioni e sincronizzazioni globali nel tick.

Non usare la semplice compattazione fisica delle ultime righe KV per rappresentare una nuova memoria semantica: le righe dipendono dal vecchio contesto. REBUILD esegue un nuovo prefill.

Mantenere i kernel di attenzione correnti. Paged attention e quantizzazione KV non sono necessarie per questa implementazione.

### P4 — Deadline e cancellazioni

Applicare la deadline all’intera operazione, anche quando il ciclo di emissione termina anticipatamente.

- Il percorso di cancellazione chiude tutte le code.
- Nessun `await future` resta senza deadline.
- Cancellare una Future dell’executor non equivale a interrompere il lavoro nativo.
- Il lease resta occupato fino al ritorno effettivo del producer e al completamento CUDA.
- Un worker bloccato rende il runtime non pronto; non si liberano buffer ancora utilizzati e non si accettano nuove inferenze sullo stato compromesso.

Separare il fallimento della manutenzione da quello della risposta: un SUMMARY fallito non cancella una risposta valida.

### P5 — Audio e playback

Restano indispensabili:

- Graph Code2Wav per forme corte e coda dispari.
- Graph family compatibile con un solo talk-context.
- Eliminazione delle attese di ammissione ridondanti fra sottopassi Talker.
- Prebuffer UI iniziale di 160 ms, adattabile fino a 320 ms.
- Cancellazione client→server quando la UI abbandona una risposta.
- Watermark distinto del rendering, separato dal consumo logico dei buffer.
- Riferimento assistant associato al frame di cattura microfono, anziché al momento dell’invio WebSocket.

Il supporto agli altoparlanti richiede questa corrispondenza temporale e prove di eco reale. Nessuna soglia RMS deve sopprimere arbitrariamente voce debole.

Il rollback parola per parola rimane un componente distinto: deve allineare **testo assistant già noto e PCM assistant**, non trascrivere l’utente. Finché quel confine non è verificato, mantenere il prefisso prudente precedente e non attribuire agli ACK PCM una precisione linguistica che non possiedono.

### P6 — Training coerente

In [stage2.py](C:/Users/andre/Documents/gemma-4-omni/training/duplex_qlora/stage2.py):

- Eliminare la dipendenza dal vecchio mirror.
- Riutilizzare builder e percorso Q2 del serving.
- Rigenerare le feature nel contesto nuovo.
- Includere contesti con memoria compatta, turni audio recenti e interventi aperti.
- Usare gli stessi quattro valori di fase e lo stesso hidden corrente.
- Mantenere split disgiunti per conversazione/parlante; test held-out non usato per scegliere soglie.
- Addestrare inizialmente solo la head Control6.
- Sei azioni distinte e legal mask diretta, senza accorpamenti runtime che trasformino HOLD in SPEAK/YIELD.
- Valutare falsi eventi, ritardo delle transizioni, eco e doppio parlato oltre alle metriche per frame.

Il checkpoint precedente resta conservato. Il nuovo profilo di contesto non viene dichiarato compatibile con quei pesi solo perché le dimensioni dei tensori coincidono.

## 5. Ordine di lavoro e verifiche

Ordine:

1. Record audio/testo e rimozione della dipendenza ASR.
2. Proprietà, budget e lifecycle dei branch.
3. Q2 canonico continuo.
4. SUMMARY privato e REBUILD.
5. Promozione, catch-up e finalizer.
6. Deadline, playback, riferimento eco e vocoder.
7. Feature coerenti e nuovo training Control6.
8. Integrazione nella UI.

Test funzionali da aggiungere alle suite esistenti:

- Nessun transcript: il turno audio entra comunque nella history.
- Turno utente oltre il ring recente: compattazione per chunk senza chiuderlo.
- SUMMARY non emette eventi di risposta, testo o audio.
- Il job non vede sorgenti successive al proprio cutoff.
- Arrivo di nuovi frame durante SUMMARY e REBUILD senza perdita o duplicazione.
- Modifica di una sorgente coperta: risultato scartato.
- Promozione canonica mentre ACTIVE continua sul proprio snapshot.
- Tre slot compresi i proprietari in retirement.
- YIELD durante copia, decode e catch-up.
- REPLAN ripetuti senza riavviare Qwen.
- Fine deadline dopo uscita anticipata dal ciclo di emissione.
- Mancanza di spazio o fallimento della sintesi: nessun trimming silenzioso.
- Memoria ricorsiva con correzioni, numeri, preferenze cambiate e richieste aperte.
- Conversazione lunga con più compattazioni, prima in cuffia e poi con altoparlanti.

Sul runtime GPU: smoke breve prima delle sessioni lunghe. Verificare latenze, underrun, occupazione dei lease e crescita della memoria; la correttezza dei test CPU non prova la stabilità della conversazione.

Aggiornare direttamente `docs/ARCHITECTURE.md` e i test pertinenti. Nessun nuovo manifesto SHA, sistema di certificazione o documento orfano. Modelli e build pesanti soltanto su Mondaini; checkpoint esistenti preservati.