# **Gemma 4 Omni — Piano di validazione e modifiche prima della prossima run** 

_Documento tecnico finale • 12 agosto 2026_ 

## **Executive summary** 

La prossima run non dovrebbe partire da ulteriori modifiche radicali all’architettura. Le diagnostiche disponibili indicano che il ramo semantic/cb0 è promettente, mentre il problema principale è nella generazione cb1–cb15 e nel collegamento tra lo spazio latente di Gemma e quello atteso dal sub-talker Qwen. Il test più informativo è quello in cui GT full e cb0 generated + acoustic GT producono voce comprensibile, mentre GT cb0 + acoustic generated e full generated producono voce metallica/incomprensibile. 

Nota correttiva: il warning REAL_ENCODING visto in una delle ultime run è stato interpretato erroneamente come prova di fallback FFT/sintetico. In quella run il warning era un falso allarme; non va usato per invalidare il training. Le conclusioni qui sotto non si basano su quella interpretazione errata. 

## **1. Cosa hanno dimostrato le ultime diagnostiche** 

- GT full → voce comprensibile: la catena Qwen/sub-talker + decoder è in grado di produrre speech valido. 

- cb0 generated + cb1–15 GT → voce comprensibile: la predizione semantic/cb0 di Gemma è sufficientemente buona; un errore in cb0 non spiega da solo la degradazione acustica. 

- cb0 GT + cb1–15 generated → voce metallica/incomprensibile: il failure è localizzato principalmente nel ramo che genera cb1–15 o nel suo condizionamento. 

- cb0 generated + cb1–15 generated → voce metallica/incomprensibile: il problema del ramo acustico rimane anche quando cb0 è corretto. 

- Le statistiche dei codebook non mostrano un collasso banale verso un singolo codice: il ramo generativo non sembra semplicemente emettere sempre silenzio. 

## **2. Ipotesi principale: co-adattamento Gemma ↔ adapter ↔ sub-talker** 

L’ipotesi è tecnicamente plausibile. Se il sub-talker Qwen è congelato, il suo manifold di ingresso rimane fisso, mentre i pesi LoRA di Gemma modificano continuamente la distribuzione degli hidden state. L’adapter deve quindi apprendere una trasformazione verso un target che, dal suo punto di vista, cambia continuamente perché cambia la sorgente Gemma. Il problema è meno 'il sub-talker si muove' e più 'il sistema di coordinate della sorgente si muove mentre l’adapter sta imparando la mappa'. 

Non è ancora una prova causale. Va verificata con un ablation controllato: Gemma frozen + Qwen frozen + solo adapter trainable, seguito da un secondo stage in cui si sbloccano le LoRA di Gemma con learning rate più basso dell’adapter. 

## **3. Modifiche obbligatorie prima del training** 

**A. Inserire un protocollo di latent alignment —** Aggiungere una latent alignment loss tra adapter(Gemma hidden) e una rappresentazione target estratta dal sub-talker Qwen sullo stesso audio. Usarla come regolarizzatore, non come unico obiettivo. 

**B. Separare le fasi di ottimizzazione —** Stage 0: Gemma frozen, Qwen frozen, adapter trainable. Stage 1: aggiungere acoustic CE. Stage 2: sbloccare LoRA Gemma, mantenendo Qwen frozen e adapter a LR inferiore. 

**C. Aggiungere la turn head, ma mantenerla leggerissima —** Head Linear/LayerNorm → pochi logits sopra gli hidden già calcolati da Gemma. Nessun nuovo encoder, LSTM/GRU/Transformer o forward Gemma separato. Stati consigliati: WAIT, BACKCHANNEL, START/ SPEAK, CONTINUE, YIELD, INTERRUPT. 

**D. Rendere il controllo turn-taking un segnale ausiliario —** La turn head deve inizialmente essere addestrata come task ausiliario. Solo dopo aver verificato che impari segnali temporali utili va usata nel runtime per controllare la generazione. 

**E. Rafforzare il data pipeline DuplexConv —** Campionamento per durata cumulativa e stratificazione dei domini; verificare esplicitamente il mapping dei track a partecipanti/ruoli. Conservare timestamp/VAD/overlap invece di ridurli a semplice frame weighting. 

**F. Correggere il pipeline MDC —** Separare train/dev/test; non trasformare tutto il dataset read-speech in pseudoconversazione. Randomizzare gap e punto di barge-in; evitare una pausa fissa e un cutoff fisso del 50%. 

**G. Non cambiare ancora il sub-talker —** Il test GT full dimostra che il decoder può produrre voce comprensibile. Prima va risolto il mapping Gemma→Qwen; cambiare contemporaneamente il decoder renderebbe l’esperimento non diagnostico. 

## **4. Protocollo sperimentale consigliato** 

|Run|Gemma|Adapter|Qwen|Scopo|
|---|---|---|---|---|
|R0 — baseline<br>diagnostica|frozen|frozen|frozen|Misurare A/B/C/D<br>sullo stesso<br>checkpoint e<br>dataset.<br>|
|R1 — alignment|frozen|train|frozen|Verifcare se esiste<br>una mappa stabile<br>verso il manifold<br>Qwen.|
|R2 — acoustic|frozen|train|frozen|Aggiungere acoustic<br>CE e verifcare cb1–<br>15.|
|R3 — Gemma<br>adaptation|LoRA train|train, LR basso|frozen|Testare l’ipotesi del<br>moving<br>representation<br>space.|
|R4 — turn auxiliary|LoRA train|train|frozen|Aggiungere turn loss<br>senza cambiare il<br>runtime.|



## **5. Loss consigliata** 

Durante R3, usare concettualmente: 

#### **L_total = L_acoustic + λ · L_latent + μ · L_text + ν · L_turn** 

λ deve essere più alto nella fase di warm-up e poi ridotto. L’obiettivo non è rendere gli hidden di Gemma identici a quelli Qwen, ma renderli funzionalmente compatibili con il sub-talker. 

## **6. Test diagnostici da automatizzare** 

- A/B/C/D waveform test: GT full, cb0GEN/restGT, cb0GT/restGEN, fullGEN. 

- cb0 top-1 e top-5 accuracy, perplexity ed entropy. 

- Per-codebook CE/accuracy per cb1…cb15. 

- Latent cosine similarity prima e dopo adapter. 

- Norma e covariance spectrum di Gemma hidden, adapter output e Qwen hidden. 

- Temporal smoothness: ||z_t-z_{t-1}|| confrontato con il target Qwen. 

- Nearest-neighbor retrieval nello spazio Qwen: verificare se l’embedding mappato recupera il frame corretto. 

- Audio quality: non usare RMS waveform error come unica metrica; aggiungere ASR/WER o CER e ascolto umano su un set fisso. 

- Silence/code collapse: distribuzione per codebook, entropy, percentuale dei codici dominanti. 

- Turn head: macro-F1, confusion matrix e detection latency su START/INTERRUPT/BACKCHANNEL/YIELD. 

## **7. Criteri di go/no-go** 

|Segnale|GO|NO-GO|
|---|---|---|
|R1 latent alignment|alignment loss converge + cosine<br>aumenta|nessun miglioramento nonostante<br>adapter train|
|cb1–15|CE e/o accuracy migliorano e audio<br>diventa intelligibile|audio resta metallico con GT cb0|
|Gemma LoRA|migliora acoustic metrics senza<br>destabilizzare alignment|alignment degrada rapidamente|
|Turn head|impara oltre baseline e non richiede<br>forward extra|richiede modello/forward aggiuntivo<br>opeggiora training|
|Full A/B|fullGEN si avvicina progressivamente<br>a cb0GEN/restGT|fullGEN resta incomprensibile<br>mentre oracle acoustic resta buono|



## **8. Turn head e latenza** 

La turn head non dovrebbe costituire un problema di latenza se è una piccola projection head applicata agli hidden state già calcolati da Gemma. Va evitato qualsiasi forward separato o modulo temporale pesante. In runtime il ramo turn e il ramo Qwen possono essere eseguiti in parallelo. La turn head va quindi considerata un rischio di latenza trascurabile rispetto a Gemma + autoregressione Qwen, a condizione di mantenere questa struttura. 

## **9. Priorità operative** 

1. Automatizzare A/B/C/D e salvarne audio + metriche su un set fisso. 

2. Implementare adapter-only alignment con Gemma e Qwen congelati. 

3. Aggiungere latent alignment loss al training acustico. 

4. Ripetere una run con Gemma frozen per verificare il ramo cb1–15. 

5. Sbloccare LoRA Gemma con LR più basso dell’adapter e confrontare direttamente con la run frozen. 

6. Aggiungere la turn head come loss ausiliaria, senza inserirla inizialmente nel percorso critico di inferenza. 

7. Solo dopo i test sopra, rifinire sampling/augmentation DuplexConv e MDC. 

## **10. Conclusione** 

La prossima run deve essere soprattutto un esperimento diagnostico controllato. I risultati A/B già ottenuti localizzano il problema nel ramo acoustic-code generation/latent conditioning, non nel decoder e non principalmente nel cb0. L’ipotesi di co-adattamento Gemma/adapter è sufficientemente forte da meritare un’ablation dedicata. La modifica architetturale più promettente è quindi un adapter stabilizzato da latent alignment, seguito da un graduale sblocco delle LoRA di Gemma. La turn head è compatibile con il requisito di bassa latenza se implementata come head leggera sugli hidden già disponibili. 

## **Appendice operativa — clipping, dataset e uso delle informazioni** 

## **DuplexConv/MDC** 

Questa appendice rende espliciti i punti che devono essere gate obbligatori prima della prossima run: clipping a 2048, preservazione delle informazioni temporali e conversazionali di DuplexConv, separazione dei ruoli tra DuplexConv e MDC Italian Speech Dataset 2026, e controlli della pipeline. 

### **A. Clipping a 2048: non deve essere un semplice truncation** 

Il limite di contesto a 2048 non deve comportare la perdita arbitraria della parte audio o delle annotazioni che determinano il comportamento conversazionale. Il problema è la perdita di causalità tra contesto, turn-taking e target speech. 

- Misurare prima della run la distribuzione delle sequenze: percentuale >2048, mediana/p95 e quota realmente clippata. 

- Preferire chunking su boundary semantici/temporali quando possibile, non taglio cieco nel mezzo di un evento. 

- Conservare overlap sufficiente tra finestre consecutive per non perdere il contesto prima di START, BACKCHANNEL, YIELD o INTERRUPT. 

- Non troncare separatamente input, cb0 e cb1–15: tutti i target devono restare temporalmente allineati. 

- Registrare original_length, clipped_length, clip_fraction, timestamp iniziale/finale e turn boundary attraversati. 

- Misurare quante finestre clippate attraversano un evento di turn-taking. 

### **B. DuplexConv: usare l'informazione conversazionale** 

DuplexConv non va trattato come semplice speech italiano. Il suo valore è nella struttura dialogica e temporale: partecipanti, turni, overlap, pause, silenzi e interruzioni devono diventare supervisione. 

- Conservare speaker/participant identity e role information quando disponibile. 

- Preservare timestamp e ricostruire WAIT → START → CONTINUE → YIELD → INTERRUPT/BACKCHANNEL. 

- Derivare target temporali per la turn head dagli eventi reali, non solo dal frame weighting. 

- Marcare esplicitamente overlap/co-talk. 

- Separare silence conversazionale, pausa prosodica e attesa quando i dati lo permettono. 

- Usare speaker-disjoint split per evitare leakage. 

- Bilanciare anche per evento: START/INTERRUPT/BACKCHANNEL/YIELD sono rari rispetto a SPEAK/SILENCE. 

### **C. MDC Italian Speech Dataset 2026: usarlo per speech, non per inventare conversazione** 

MDC è principalmente utile per copertura linguistica/acustica italiana e varietà del parlato. Non dovrebbe essere trasformato in conversazione artificiale tramite una singola pausa o un cutoff deterministico. 

- Usare MDC per fonetica, lessico, speaker/accent variation e robustezza acustica. 

- Mantenere testo/trascrizione rigorosamente allineati all'audio e ai codebook Qwen. 

- Se si sintetizzano interazioni, randomizzare gap, durata pause, posizione del barge-in e combinazioni speaker. 

- Non usare una pausa fissa o un cutoff fisso: diventerebbe un artefatto apprendibile. 

- Marcare chiaramente esempi reali e sintetici. 

- Usare MDC per aumentare il volume speech, senza permettergli di dominare la distribuzione degli eventi conversazionali. 

### **D. Ruolo corretto dei due dataset** 

|Funzione|DuplexConv|MDC 2026|Regola|
|---|---|---|---|
|Speech / codebook|Sì|Sì|Entrambi, con mix|
|learning|||controllato.|
|Italian linguistic coverage|Medio|Alto|MDC può fornire maggiore<br>copertura.|



|Turn-taking|Alto|Basso / solo se annotato|Non inventarlo da read<br>speech.|
|---|---|---|---|
|Overlap / interruption|Alto|Solo se realmente<br>presente|Preservare temporalità.|
|Text alignment|Sì|Sì|Obbligatorio.|
|Speaker split|Obbligatorio|Obbligatorio|Niente leakage.|
|Synthetic augmentation|Limitata|Possibile, marcata|Non confondere<br>reale/sintetico.|



### **E. Sampling e weighting** 

- Loggare per run la quota di batch proveniente da DuplexConv e MDC. 

- Loggare la quota di esempi con turn event, overlap, backchannel, interruption e silence. 

- Usare sampling/probability per task oltre al frame weighting. 

- Separare nei report loss speech, text, turn e latent per dataset. 

- Avere validation DuplexConv e validation MDC separate: una media unica può nascondere un failure domainspecifico. 

### **F. Caching e REAL_ENCODING: controllo robusto** 

Dato il falso allarme osservato sul controllo REAL_ENCODING, il preflight deve verificare il contenuto effettivo della cache, non solo un flag. 

- Salvare encoder/version/hash e modalità di encoding nei metadata di ogni shard. 

- Verificare un piccolo campione audio → embedding/codebook durante il preflight. 

- Confrontare statistiche cache vs fresh encode su un campione. 

- Fallire il training solo quando il contenuto verificato è incompatibile. 

### **G. Checklist preflight obbligatoria** 

- ☐ Nessun sample supera 2048 senza policy di chunking/overlap. 

- ☐ Input, cb0 e cb1–15 hanno identica segmentazione temporale. 

- ☐ Sono registrati original_length e clipped_length. 

- ☐ Le finestre che attraversano turn boundary sono identificate. 

- ☐ Speaker/participant split verificato. 

- ☐ DuplexConv conserva turn/overlap/timestamp. 

- ☐ MDC non introduce pause/cutoff deterministici. 

- ☐ Train/val/test separati per speaker quando possibile. 

- ☐ Mix DuplexConv/MDC misurato per batch e ore. 

- ☐ REAL_ENCODING verificato sul contenuto cache. 

- ☐ A/B/C/D eseguito su un set fisso. 

- ☐ Salvate metriche cb0, cb1–15, latent alignment e ASR/CER/WER. 

### **H. Ordine definitivo delle modifiche** 

1. Audit token budget e clipping 2048. 

2. Audit builder DuplexConv: ruoli, timestamp, overlap e turn events. 

3. Audit builder MDC: split, allineamento, augmentation e assenza di pattern conversazionali artificiali. 

4. Rendere il preflight REAL_ENCODING basato sul contenuto. 

5. Congelare Gemma/Qwen e implementare adapter-only latent alignment. 

6. Automatizzare A/B/C/D + metriche audio/ASR. 

7. Run acoustic con Gemma frozen. 

8. Sbloccare LoRA Gemma gradualmente con LR più basso dell'adapter. 

9. Aggiungere turn head come task ausiliario leggero. 

10. Solo dopo ottimizzare sampling, weighting e policy runtime. 

## **Decisione per la prossima run** 

La prossima run deve essere un esperimento controllato, non semplicemente una ripetizione con più step. Clipping 2048, mix DuplexConv/MDC, preservazione degli eventi conversazionali e allineamento Gemma→Qwen devono essere osservabili e misurabili. Il criterio principale è che full generated inizi a convergere verso la qualità di cb0 generated + acoustic GT, accompagnato da miglioramento di latent alignment e cb1–15. 

