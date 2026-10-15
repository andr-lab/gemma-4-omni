# Transcript del run pair `owner_pin_fix`

Run iniziato `2026-10-15T15:38:42.294242+00:00 UTC`; durata `300.54 s`; esito `partial`: 6 risposte create, 6 concluse e 6 con PCM udibile su 32 richieste. Gli artefatti d’ingresso sono `A_input.wav` (B→A) e `B_input.wav` (A→B). Il prompt A contiene una persona che deve comunicare gradualmente i vincoli e infine chiedere un riepilogo; il prompt B dice di usare soltanto ciò che viene comunicato e di non conoscere in anticipo alcun fatto.

**Come leggere i tempi.** Le risposte sono ordinate per primo `probe.input_frame_sent` con `source_response_id` corrispondente alla risposta. L’intervallo di relay va dal primo frame al timestamp dell’ultimo frame più 80 ms. `response.created` e `response.done` sono riportati separatamente: il primo audio può essere inoltrato mentre la generazione continua, e i tempi di relay non provano l’overlap acustico campione per campione.

## Risposte ordinate per primo PCM inoltrato

### 1. A → B · SPEAK · 00:09.644–00:29.554
- Creata 00:03.267; `response.done` 00:22.982; PCM 18.72 s, RMS 0.0610.
- **ASR sul WAV della risposta e sullo stream B_input:** corrispondono al contenuto generato; l’ASR scrive “on weekend” e “17.0”.
- **Testo generato:** “Ciao, sono Marco. Stavo pensando a un weekend a Torino dal 13 al 15 novembre 2026 con Elena. Il budget iniziale totale è di 350 euro per entrambi e arriveremo in treno venerdì alle 18:30, ripartendo domenica alle 17:00.”

### 2. B → A · SPEAK · 00:27.229–00:33.206
- Creata 00:12.241; `response.done` 00:31.608; PCM 5.92 s, RMS 0.0787.
- **ASR sul WAV della risposta e sullo stream A_input:** “Ciao Marco, sono qui per aiutarti a pianificare il tuo viaggio. Cosa hai in mente?”
- **Testo generato:** “Ciao Marco, sono qui per aiutarti a pianificare il tuo viaggio. Cosa hai in mente?”
- La risposta viene avviata su un prefisso parziale: la trascrizione ASR di `B_input_prefix_first_B_SPEAK.wav` è solo “Ciao, sono Marco”. Il JSONL runtime non registra onset/endpoint; un replay CPU del tracker server sul WAV completo ricava onset f115=9.20 s, ultimo frame vocale f133=10.64 s, endpoint f143=11.44 s dopo 10 frame quieti, nuovo onset f147=11.76 s. La decisione B è al frame f145, nella pausa interna di 800 ms e prima che riprenda il parlato A.

### 3. A → B · BACKCHANNEL · 00:34.407–00:35.927
- Creata 00:29.067; `response.done` 00:35.615; PCM 1.44 s, RMS 0.0437.
- **ASR sul WAV della risposta:** “Ascolto.”
- **Testo generato:** “Ascolto.”

### 4. A → B · SPEAK · 00:40.406–00:52.881
- Creata 00:36.058; `response.done` 00:50.406; PCM 9.12 s, RMS 0.0685.
- **ASR sul WAV della risposta:** “Elena è vegetariana, non abbiamo un auto e ci interessano cinema e cibo con un ritmo tranquillo. Avresti qualche idea per un possibile piano?”
- **Testo generato:** “Elena è vegetariana, non abbiamo un'auto e ci interessano cinema e cibo con un ritmo tranquillo. Avresti qualche idea per un possibile piano?”

### 5. B → A · SPEAK · 00:41.366–00:56.146
- Creata 00:33.517; `response.done` 00:53.100; PCM 11.36 s, RMS 0.0689.
- **ASR sul WAV della risposta e sullo stream A_input:** “Perfetto! Abbiamo un budget totale di 350 euro e le date sono da venerdì alle 18.30 a domenica alle 17.00. Avete già una destinazione in mente?”
- **Testo generato:** “Perfetto, abbiamo un budget totale di 350 euro e le date sono da venerdì alle 18:30 a domenica alle 17:00. Avete già una destinazione in mente?”
- L’output B si sovrappone nel relay alla risposta A sulle preferenze (00:41.366–00:52.881). B aveva creato questa risposta prima che la risposta A iniziasse a inoltrare le preferenze; non è una risposta a quelle preferenze. L’ASR dell’intero `B_input.wav` contiene Torino, ma l’audio attribuito al prefill B2 è il crop `B_input_span_225-349_for_response_987920c0af48.wav`: 160.000 campioni (125 frame, f225–349), con budget e orari ma senza Torino. Sul crop l’ASR legge 00:00–00:05.70 «Il budget iniziale totale è di 350 euro per entrambi e arriveremo in treno venerdì alle» e 00:05.70–00:09.04 «18.30, ripartendo domenica alle 17.0». Torino compare prima, nei frame f147–217, mentre B1 era pending. Il report associa a B2 `completed_history_roles=["user", "assistant"]`, ma non registra span o contenuto della history; il carry semantico è quindi non dimostrato. La domanda sulla destinazione non prova una perdita di contesto del modello.

### 6. B → A · SPEAK · 00:59.985–01:13.137
- Creata 00:55.668; `response.done` 01:08.642; PCM 12.56 s, RMS 0.0671.
- **ASR sul WAV della risposta e sullo stream A_input:** “Considerando il budget e gli interessi, potremmo pensare a una città con opzioni di ristorazione vegetariana e attività culturali rilassanti. Avete qualche preferenza per il tipo di ambiente che cercate?”
- **Testo generato:** “Considerando il budget e gli interessi, potremmo pensare a una città con opzioni di ristorazione vegetariana e attività culturali rilassanti. Avete qualche preferenza per il tipo di ambiente che cercate?”
- Questa risposta è stata creata dopo che l’audio delle preferenze A era stato inoltrato (fine circa 00:52.881); il testo riprende vegetarianismo/interessi/ritmo, ma propone una città generica e domanda di nuovo una preferenza. Non nomina Torino né offre un piano specifico.

## Causa: decisione del listener, harness o training

| Punto | Evidenza del run | Lettura e limite |
|---|---|---|
| Primo SPEAK B anticipato | Nel prefisso A→B, faster-whisper riconosce solo “Ciao, sono Marco”. Il replay CPU del tracker sui frame del WAV trova ultimo frame vocale f133=10.64 s, endpoint f143=11.44 s dopo 10 frame quieti e nuovo onset f147=11.76 s; B decide SPEAK al frame f145, nella pausa interna da 800 ms. Il relay B inizia a 27.229 s mentre A termina a 29.554 s. | La risposta è stata decisa su un segmento/utterance parziale dopo l’endpoint, non sull’intera intro. Il relay poi si sovrappone al tratto finale. Il JSONL di serving non serializza VAD/onset/endpoint; i marker sono ricostruiti sul WAV con il tracker CPU. Il server configura 800 ms e conta frame quieti in `duplex_kernel/server.py:800,1683-1743`. |
| Correzione del pin audio | Nel run non compare `direct_audio_snapshot_invalid` tra le rejection; `response.created` e `response.done` sono registrati per le sei risposte. | Il precedente failure di pin non si riproduce in questa sessione; questo run non prova tutti i casi stale/reset. |
| Bootstrap manuale A | `report.json`: `peer_a.phase=complete`, un commit inviato, risposta `resp_ac2a7f5a7aee`, auto-response riabilitata e ACKata; B resta in auto-response. Il report elenca sei response create, nessun budget di creazione esaurito. | Il difetto di stato del bootstrap non si riproduce qui; non ci sono trigger `response.create` aggiuntivi manuali dopo il seed. |
| A dopo il parlato B | Da 73.744 a 300.480 s si registrano 2.820 decisioni A. Raw: 2.541 `LISTEN`, 279 `CONTINUE`; selected/applied: 2.820 `LISTEN`. In tutte, `legal_actions` include `SPEAK`, hidden fresco (age 0 frame; 3.7–5.0 ms), testa Control6 valutata, nessun rejection. Al primo frame f855 la probabilità di LISTEN è 0.999988. `auto_response_disable.acknowledged=false`, `created_budget_exhausted=false`; l’evento riporta `native_control6_semantic_prefix_mixed_candidate600.pt`. | Il controller sceglie LISTEN: non c’è evidenza che harness/admission sopprima SPEAK. È un fallimento conversazionale osservato nell’inferenza di questo run, non una dimostrazione della causa di training. Il nome checkpoint è metadato auto-riportato dall’evento e non prova da solo quali pesi fossero effettivamente online. |
| Timeout/cleanup | Le sei risposte sono terminate entro 73.137 s; il resto della run è silenzio. A deadline non ci sono response attive, PCM in coda o frame non inviati; restano 4 ACK pendenti per peer e il report segnala sender bloccati al cleanup. | Il cleanup spiega il blocco allo shutdown, non i ~227 s senza SPEAK. Il limite di 300 s e il target di 32 risposte non hanno troncato una risposta attiva. |
| Domanda di destinazione B | B2 (`resp_987920c0af48`) è creato a 33.517 s. L’ASR dell’intero `B_input.wav` contiene Torino, ma l’audio del suo prefill è esattamente 160.000 campioni/125 frame, f225–349: ASR 00:00–00:05.70 budget 350 e arrivo venerdì; 00:05.70–00:09.04 orari, senza Torino. Torino è nei frame f147–217, mentre B1 era pending. Il report associa a B2 `completed_history_roles=["user", "assistant"]`, senza span/contenuto della history. | Il crop audio B2 non contiene Torino, ma il report non consente di stabilire se la storia testuale/semantica precedente la includesse. Lo stream aggregato non prova che B2 l’abbia ricevuta; il carry resta non verificato e non si può attribuire una perdita al modello. |

Nel sorgente verificato, `duplex_kernel/duplex_control6.py:32-49` rende SPEAK legale solo con user turn pending e senza peer/response attivi; `duplex_kernel/gemma_native_control6_policy.py:51-62,136-155,203-233` controlla hidden fresco, valuta i logits sei-vie e applica la maschera legale. `duplex_kernel/server.py:7461-7495` registra se il head è stato valutato e la freschezza; `duplex_kernel/server.py:7947-7954` descrive i rifiuti di admission delle azioni vocali. Il decision event f855 riporta `native_control6_step_called=true`, `native_control6_head_scored=true`, `listener_hidden_age_frames=0`, legal SPEAK e nessun rejection; `gate_applied_action=LISTEN` coincide con `selected_action`, ma al frame f855 il raw è `CONTINUE`.

I metadati dichiarano schema `semantic_prefix_mixed_training_v1`, `prefix_v2` e `context_conditioned=true`; il report riporta macro-F1 real all-classes 0.2365298 (present classes 0.35479) e supporto SPEAK pari a 9. Sono un indizio di copertura/qualità fragile della classe SPEAK, non una prova causale. Il checkpoint è self-report: la corrispondenza dei pesi online con la candidate non è dimostrata; inoltre l’evento `session.updated` di B riporta `applied_revision=null` e `context_current=false`, e il report non contiene span/content della history B2. La diagnosi sostenibile è quindi: nel run il controller ha realmente scelto LISTEN per 2.820 tick pur avendo SPEAK legale e hidden fresco; non emerge una soppressione del harness, mentre la causa training e il carry del contesto restano non provati.

## Esito della conversazione

L’ordine e il contenuto generato mostrano l’apertura A con i vincoli iniziali, una breve risposta di B e la successiva comunicazione delle preferenze A. Il B2 ripete budget e date ma non ha Torino nel crop audio verificato; quindi la domanda sulla destinazione non dimostra che il modello abbia ignorato Torino. Il B3, creato dopo le preferenze A, restituisce una proposta generica e chiede un’altra preferenza. Non vengono comunicati l’aggiornamento del budget a 280 euro né dell’arrivo alle 20:30, non si arriva al confronto delle opzioni e non c’è il riepilogo richiesto dal prompt A.

Tutte le sei risposte udibili terminano entro 01:13.137; per il resto della deadline di 300.54 s non ci sono altri response. Dopo 01:13 circa, A registra decisioni LISTEN durante il silenzio del peer. Il bundle riporta bootstrap manuale A `complete` (un commit e ACK di riabilitazione auto-response), 6 risposte create e nessun errore `direct_audio_snapshot_invalid` tra le rejection delle decisioni. A deadline non ci sono response attive o frame audio in coda; restano 4 ACK pendenti per ciascun endpoint, e l’harness segnala che i sender tasks sono rimasti bloccati al cleanup. Nessuna causa VAD/onset per il silenzio è dimostrabile da questo JSONL.

**ASR esterno:** faster-whisper small, CPU int8, 4 thread, cache-only; lingua italiana con probabilità 1.0 per tutti i file. È un singolo modello fuori dal serving; non equivale a una seconda valutazione umana/modello indipendente. Le frasi ASR per risposta sono abbinate ai WAV di risposta, mentre gli stream aggregati possono fondere turni e silenzi.

**ASR aggregato dei flussi d’ingresso (tempi locali dei WAV, non wall-clock del run):**

- `A_input.wav` (audio di B ricevuto da A): 0.00–0.92 “Inizia pure”; 25.50–31.02 primo SPEAK B; 39.60–51.14 secondo SPEAK B; 55.82–67.58 ultimo SPEAK B. Il dettaglio per risposta è riportato sopra.
- `B_input.wav` (audio di A ricevuto da B): 8.94–16.40 apertura “...a Torino dal 13 al 15 novembre 2026 con Elena”; 17.92–23.70 budget 350 € e treno/orari; 23.70–41.86 completa gli orari, “Ascolto” e avvia le preferenze; 41.86–47.64 preferenze e richiesta di piano. La segmentazione fonde frasi/turni e non va letta come confine esatto della risposta.

Il prefisso `B_input_prefix_first_B_SPEAK.wav` viene riconosciuto solo come “Ciao, sono Marco”, coerente con la decisione B presa dopo 33 frame/2.64 s del primo audio A. L’ASR aggregato `B_input.wav` contiene “...a Torino dal 13 al 15 novembre 2026...”; tuttavia il crop audio del prefill B2 è `B_input_span_225-349_for_response_987920c0af48.wav` (160.000 campioni, f225–349) e il suo ASR riporta budget/orari, senza Torino. Torino era stata pronunciata ai frame f147–217, durante B1 pending. Il report dichiara ruoli history user/assistant, ma non gli span/contenuti; il flusso aggregato non prova quindi che Torino fosse nel contesto semantico di B2. L’ASR aggregato `A_input.wav` conferma le tre risposte B sopra. L’ASR interno del servizio è disabilitato. **Limite:** tempi di relay e RMS non certificano intelligibilità né ascolto umano; il crop ASR è una singola trascrizione automatica e non dimostra da solo la rappresentazione semantica interna.
