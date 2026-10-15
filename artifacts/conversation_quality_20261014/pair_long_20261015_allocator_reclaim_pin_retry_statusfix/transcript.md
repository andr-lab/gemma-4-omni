# Transcript del run pair statusfix

Run iniziato 2026-10-15T15:05:06.376583+00:00 UTC; durata 300.54 s. Esito `partial`: 17 risposte create, 16 concluse con `response.done`, 15 con PCM udibile; target 32 risposte udibili. Nessun recap finale.

**Ordine e tempi.** Le risposte udibili sono ordinate per primo frame audio inoltrato al peer, non per `response.created`; ogni riga riporta anche creazione e terminale. `t_end` è l’invio dell’ultimo frame più 80 ms. Gli intervalli sono trasporto di frame verso l’altro endpoint: non misurano l’ascolto umano né l’overlap acustico campione per campione. La durata PCM generata è indicata a parte.

**ASR.** faster-whisper small, CPU int8, modello già in cache (`local_files_only`), italiano (probabilità 1.0). L’ASR è stato eseguito sui due WAV aggregati in ingresso: `A_input.wav` (audio ricevuto da A, quindi output di B) e `B_input.wav` (audio ricevuto da B, quindi output di A). Le frasi ASR qui sotto sono abbinate per contenuto alla risposta corrispondente; i confini segmentati dal modello possono accorpare turni e silenzi, quindi non li uso come tempi wall-clock.

## Turni ordinati per audio inoltrato

### 01. A → B · SPEAK · 00:09.71–00:27.35 verso B (199 frame; 15.92 s PCM)
- `response.created` 00:03.26; terminale response.done 00:23.45; 382080 campioni generati, RMS 0.0650.
- ASR sullo stream ricevuto (B_input.wav, cioè l’audio di A ricevuto da B): “Ciao, sono Marco. Stavo pensando da un weekend a Torino dal 13 al 15 novembre 2026 per me ed Elena. Il budget iniziale è di 350 euro per entrambi e arriveremo in treno venerdì alle 18.30, ripartendo domenica alle 17.0.”
- Testo generato dal server: “Ciao, sono Marco. Stavo pensando a un weekend a Torino dal 13 al 15 novembre 2026 per me ed Elena. Il budget iniziale è di 350 euro per entrambi e arriveremo in treno venerdì alle 18:30, ripartendo domenica alle 17:00.”

### 02. B → A · BACKCHANNEL · 00:11.31–00:13.31 verso A (20 frame; 1.60 s PCM)
- `response.created` 00:10.16; terminale response.done 00:13.02; 38400 campioni generati, RMS 0.0237.
- ASR sullo stream ricevuto (A_input.wav, cioè l’audio di B ricevuto da A): “Certo.”
- Testo generato dal server: “Certo.”

### 03. B → A · SPEAK · 01:12.02–01:29.32 verso A (188 frame; 15.04 s PCM)
- `response.created` 00:53.19; terminale response.done 01:22.99; 360960 campioni generati, RMS 0.0710.
- ASR sullo stream ricevuto (A_input.wav): “Capisco. Quindi, un weekend a Torino per te ed Elena, dal 13 al 15 novembre 2026 con un bagge di 350 euro e arrivo in treno venerdì alle 18.30 e partenza domenica alle 17.00. Avete già pensato a dove alloggiare?”
- Testo generato dal server: “Capisco. Quindi, un weekend a Torino per te ed Elena dal 13 al 15 novembre 2026 con un budget di 350 euro e arrivo in treno venerdì alle 18:30 e partenza domenica alle 17:00. Avete già pensato a dove alloggiare?”

### 04. A → B · BACKCHANNEL · 01:26.30–01:27.34 verso B (13 frame; 1.04 s PCM)
- `response.created` 01:13.76; terminale response.done 01:26.80; 24960 campioni generati, RMS 0.0593.
- ASR sullo stream ricevuto (B_input.wav): “Ascolto.”
- Testo generato dal server: “Ascolto.”

### 05. B → A · BACKCHANNEL · 01:29.32–01:31.33 verso A (25 frame; 2.00 s PCM)
- `response.created` 01:27.41; terminale response.done 01:30.22; 48000 campioni generati, RMS 0.0212.
- ASR sullo stream ricevuto (A_input.wav): “Perfetto.”
- Testo generato dal server: “Perfetto.”

### 06. B → A · SPEAK · 01:34.28–01:38.21 verso A (48 frame; 3.84 s PCM)
- `response.created` 01:31.50; terminale response.done 01:37.60; 92160 campioni generati, RMS 0.0596.
- ASR sullo stream ricevuto (A_input.wav): “Cosa ti piacerebbe fare a Torino durante il weekend?”
- Testo generato dal server: “Cosa ti piacerebbe fare a Torino durante il weekend?”

### 07. A → B · SPEAK · 01:39.98–01:54.71 verso B (168 frame; 13.44 s PCM)
- `response.created` 01:30.46; terminale response.done 01:49.81; 322560 campioni generati, RMS 0.0682.
- ASR sullo stream ricevuto (B_input.wav): “Ho pensato a diverse cose. Elena è vegetariana, non abbiamo un auto. Ci interessano cinema e cibo e preferiamo un ritmo tranquillo. Avete qualche idea di cosa vorreste fare?”
- Testo generato dal server: “Sì, ho pensato a diverse cose. Elena è vegetariana, non abbiamo un'auto, ci interessano cinema e cibo e preferiamo un ritmo tranquillo. Avete qualche idea di cosa vorreste fare?”

### 08. B → A · BACKCHANNEL · 01:53.03–01:55.51 verso A (31 frame; 2.48 s PCM)
- `response.created` 01:40.64; terminale response.done 01:54.67; 59520 campioni generati, RMS 0.0487.
- ASR sullo stream ricevuto (A_input.wav): “Potremmo esplorare alcuni musei.”
- Testo generato dal server: “Potremmo esplorare alcuni musei”

### 09. A → B · BACKCHANNEL · 01:57.59–01:58.63 verso B (13 frame; 1.04 s PCM)
- `response.created` 01:54.07; terminale response.done 01:58.04; 24960 campioni generati, RMS 0.0310.
- ASR sullo stream ricevuto (B_input.wav): “Perfetto.”
- Testo generato dal server: “Perfetto.”

### 10. A → B · SPEAK · 02:02.15–02:08.77 verso B (73 frame; 5.84 s PCM)
- `response.created` 01:58.91; terminale response.done 02:08.27; 140160 campioni generati, RMS 0.0617.
- ASR sullo stream ricevuto (B_input.wav): “Che ne dici di un piano che combini qualche visita culturale con ottimi posti dove mangiare?”
- Testo generato dal server: “Che ne dici di un piano che combini qualche visita culturale con ottimi posti dove mangiare?”

### 11. B → A · SPEAK · 02:03.35–02:21.81 verso A (197 frame; 15.76 s PCM)
- `response.created` 01:56.71; terminale response.done 02:16.94; 378240 campioni generati, RMS 0.0418.
- ASR sullo stream ricevuto (A_input.wav): “Potremmo considerare visite ai siti storici o magari passeggiate nei quartieri più caratteristici. Avete preferenze per attività culturali, gastronomiche o relax?”
- Testo generato dal server: “Potremmo considerare visite ai siti storici o magari passeggiate nei quartieri più caratteristici. Avete preferenze per attività culturali, gastronomiche o relax?”

### 12. A → B · SPEAK · 02:48.80–02:58.36 verso B (113 frame; 9.04 s PCM)
- `response.created` 02:44.62; terminale response.done 02:55.36; 216960 campioni generati, RMS 0.0733.
- ASR sullo stream ricevuto (B_input.wav): “Ottimo, allora aggiorniamo un po’ i dettagli. Il budget totale è ora di 20.80 euro e l’arrivo sarà venerdì alle 20.30.”
- Testo generato dal server: “Ottimo, allora aggiorniamo un po' i dettagli. Il budget totale è ora di 280 euro e l'arrivo sarà venerdì alle 20:30.”

### 13. B → A · BACKCHANNEL · 02:58.44–02:59.56 verso A (14 frame; 1.12 s PCM)
- `response.created` 02:51.85; terminale response.done 02:58.88; 26880 campioni generati, RMS 0.0426.
- ASR sullo stream ricevuto (A_input.wav): “Perfetto.”
- Testo generato dal server: “Perfetto.”

### 14. B → A · BACKCHANNEL · nessun PCM inoltrato
- `response.created` 02:59.02; terminale response.done 03:00.29; 0 campioni generati, RMS 0.0000.
- ASR sullo stream ricevuto (A_input.wav): “Nessun parlato riconosciuto.”
- Testo generato dal server: “[nessun testo generato]”

### 15. A → B · BACKCHANNEL · 03:04.04–03:07.00 verso B (35 frame; 2.80 s PCM)
- `response.created` 02:59.17; terminale response.done 03:06.70; 67200 campioni generati, RMS 0.0847.
- ASR sullo stream ricevuto (B_input.wav): “Dobbiamo confrontare due opzioni.”
- Testo generato dal server: “Dobbiamo confrontare due opzioni”

### 16. B → A · SPEAK · 03:11.32–03:27.12 verso A (178 frame; 14.24 s PCM)
- `response.created` 03:01.92; terminale response.done 03:21.59; 341760 campioni generati, RMS 0.0688.
- ASR sullo stream ricevuto (A_input.wav): “Capisco. Quindi, il bagge totale è di 280 euro e l’arrivo sarà venerdì alle 20.30. Questo ci aiuta a definire meglio le opzioni per alloggio e trasporti locali. Avete già pensato a come gestirete i pasti?”
- Testo generato dal server: “Capisco. Quindi, il budget totale è di 280 euro e l'arrivo sarà venerdì alle 20:30. Questo ci aiuta a definire meglio le opzioni per alloggio e trasporti locali. Avete già pensato a come gestirete i pasti?”

### 17. A → B · BACKCHANNEL · nessun PCM inoltrato
- `response.created` 03:11.83; terminale response.cancelled 03:12.59 (replan); 0 campioni generati, RMS 0.0000.
- ASR sullo stream ricevuto (B_input.wav): “Nessun parlato riconosciuto; response.cancelled prima di qualsiasi PCM.”
- Testo generato dal server: “[nessun testo generato]”

## Terminali senza audio

- B BACKCHANNEL: creato 02:59.02, terminale response.done 03:00.29 (nessun motivo esplicito); zero campioni, nessun audio inoltrato.
- A BACKCHANNEL: creato 03:11.83, terminale response.cancelled 03:12.59 (replan); zero campioni, nessun audio inoltrato.

## Lettura dei contenuti

- Il seed apre con “Inizia pure”. A poi introduce Torino, Elena, 13–15 novembre, 350 euro, treno, arrivo venerdì 18:30 e partenza domenica 17:00. B ripete quasi tutti i fatti nella risposta SPEAK iniziale e domanda dell’alloggio.
- La domanda di B “Cosa ti piacerebbe fare a Torino durante il weekend?” è inoltrata a 01:34.28–01:38.21; A comunica le preferenze (Elena vegetariana, niente auto, cinema/cibo, ritmo tranquillo) da 01:39.98 a 01:54.71. Quella domanda precede l’audio delle preferenze, quindi non è evidenza di perdita di quel contesto.
- Diverso il turno B successivo: le ultime preferenze di A risultano inviate e ACKate al peer B entro 01:54.89; B crea la risposta a 01:56.71 (prefill riportato 457 ms). Eppure il testo poi chiede di nuovo “Avete preferenze per attività culturali, gastronomiche o relax?” dopo aver già ricevuto cinema/cibo/ritmo tranquillo. È una ridondanza supportata temporalmente.
- Le risposte A “Che ne dici di un piano...” (creata 01:58.91; audio 02:02.15) e B “Potremmo considerare...” (creata 01:56.71; audio 02:03.35) si sovrappongono davvero nel trasporto fino a 02:08.78. Il testo B era già in generazione prima di ricevere l’audio della proposta A; non va letto come risposta a quella proposta.
- Il testo generato di A in SPEAK riporta l’aggiornamento a 280 euro e arrivo venerdì 20:30; l’ASR aggregato dello stream B e lo stesso modello ASR applicato al WAV della risposta leggono “20,80 euro”; B in seguito riepiloga “280 euro”. È una discrepanza tra testo generato A, riconoscimento automatico e risposta successiva di B, non una conferma indipendente della pronuncia effettiva. L’ASR aggregato sul solo aggiornamento A trascrive `20.80 euro`, mentre l’ASR della successiva risposta B legge `280 euro`: il testo server e la successiva risposta B riportano 280, mentre sia l’ASR aggregato sia il WAV A trascritto dallo stesso modello riportano 20,80. Senza ascolto umano o un secondo ASR indipendente, la realizzazione parlata resta ambigua.
- Due BACKCHANNEL contengono proposte sostantive: “Potremmo esplorare alcuni musei” (2.48 s) e “Dobbiamo confrontare due opzioni” (2.80 s). Un ulteriore BACKCHANNEL di B è vuoto e senza PCM. I filler “Certo”, “Ascolto” e i ripetuti “Perfetto” sono brevi.
- L’ultimo SPEAK di B ripete il budget aggiornato e l’orario e chiede dei pasti; l’audio finisce a 03:27.12. A non risponde: c’è un BACKCHANNEL cancellato senza PCM, poi il run resta senza nuovo parlato fino alla deadline 05:00.54. Risultato parziale, senza recap finale.

### Riepilogo ASR aggregato

**A_input.wav (A riceve B):**

- 0.00–1.00 s WAV: “Inizia pure.”
- 1.00–12.11 s WAV: “Certo.”
- 12.11–70.04 s WAV: “Capisco.”
- 70.04–78.00 s WAV: “Quindi, un weekend a Torino per te ed Elena, dal 13 al 15 novembre 2026 con un bagge di 350”
- 78.00–82.52 s WAV: “euro e arrivo in treno venerdì alle 18.30 e partenza domenica alle 17.00.”
- 82.52–84.72 s WAV: “Avete già pensato a dove alloggiare?”
- 84.72–85.72 s WAV: “Perfetto.”
- 85.72–93.20 s WAV: “Cosa ti piacerebbe fare a Torino durante il weekend?”
- 93.20–109.92 s WAV: “Potremmo esplorare alcuni musei.”
- 118.11–124.47 s WAV: “Potremmo considerare visite ai siti storici o magari passeggiate nei quartieri più caratteristici.”
- 124.47–170.69 s WAV: “Perfetto.”
- 170.69–183.14 s WAV: “Capisco.”
- 183.14–190.14 s WAV: “Quindi, il bagge totale è di 280 euro e l'arrivo sarà venerdì alle 20.30.”
- 190.14–194.14 s WAV: “Questo ci aiuta a definire meglio le opzioni per alloggio e trasporti locali.”
- 194.14–197.14 s WAV: “Avete già pensato a come gestirete i pasti?”

**B_input.wav (B riceve A):**

- 8.98–16.58 s WAV: “Ciao, sono Marco. Stavo pensando da un weekend a Torino dal 13 al 15 novembre 2026 per me”
- 16.58–22.46 s WAV: “ed Elena. Il budget iniziale è di 350 euro per entrambi e arriveremo in treno venerdì”
- 22.46–82.99 s WAV: “alle 18.30, ripartendo domenica alle 17.0. Ascolto.”
- 82.99–101.74 s WAV: “Ho pensato a diverse cose. Elena è vegetariana, non abbiamo un auto.”
- 101.74–107.78 s WAV: “Ci interessano cinema e cibo e preferiamo un ritmo tranquillo. Avete qualche idea di”
- 107.78–119.15 s WAV: “cosa vorreste fare? Perfetto. Che ne dici di un piano che combini qualche”
- 119.15–162.83 s WAV: “visita culturale con ottimi posti dove mangiare? Ottimo, allora aggiorniamo un”
- 162.83–169.55 s WAV: “po' i dettagli. Il budget totale è ora di 20.80 euro e l'arrivo sarà venerdì alle 20.30.”
- 169.55–178.47 s WAV: “Dobbiamo confrontare due opzioni.”
