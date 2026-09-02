# Report P4 — blocker che impedisce la chiusura P4/P4D

**Data:** 2026-09-02
**Verdetto corrente:** `P4 FAIL — BLOCKERS REMAIN`
**P4D:** non chiuso; le evidenze offline non certificano il runtime realtime.

## 1. Contratto che deve essere soddisfatto

`gemma_hidden_t` deve essere il hidden dell'ultimo layer del language model
multimodale Gemma, alla prima posizione audio causale del frame `t`, prima di
una sola `causal_conv` V9.

Il percorso incrementale deve essere equivalente al forward nativo Gemma,
con:

- lo stesso prefisso `[BOS | testo | BOA]`;
- quattro soft audio token per frame;
- la stessa PLE (Per-Layer Embedding);
- le maschere native ibride `sliding_attention` / `full_attention`;
- la stessa semantica di posizione e cache condivisa;
- nessuna dipendenza da frame futuri.

Soglie di parity:

- almeno 100 sequenze e 5000 frame;
- MAE `<= 1e-3`;
- cosine minima `>= 0.9999`;
- zero valori non finiti e zero shape mismatch.

Il listener deve inoltre rispettare il tick realtime di 80 ms. Il ritardo
consentito per P4D appartiene soltanto allo speaker/playout, non a Gemma.

## 2. Problema osservato

Il forward full nativo `Gemma4Model.forward` e il percorso listener stateful
che usa `DynamicCache` e shared-KV non producono lo stesso hidden in BF16.

La costruzione degli input e dell'oracle è stata isolata e verificata: il
nuovo helper full restituisce differenza `0.0` rispetto al forward nativo. La
divergenza nasce quindi nel passaggio chunked/cache, non nella selezione del
hidden o nell'allineamento dei placeholder audio.

La causa tecnica più probabile è una combinazione di:

1. algoritmi di attenzione diversi per query full e query da quattro token;
2. arrotondamento BF16 dipendente dalla forma dei tensori;
3. amplificazione dell'errore nei router/MoE;
4. propagazione nei shared-KV layer Gemma.

La causa è ancora da chiudere con un'implementazione runtime definitiva: non è
lecito dichiarare parity sulla base di due percorsi incrementali che condividono
lo stesso codice.

## 3. Evidenza numerica

### Oracle full

`tools/kernel_gates/representation_parity.py` invoca direttamente il modello
nativo e confronta la cache runtime nativa:

- sequenze: `100`;
- frame: `6400`;
- MAE: `0`;
- cosine minima: `0.99999994`;
- non-finite: `0`;
- shape mismatch: `0`;
- gate: **PASS**.

Evidence locale:

`validation/kernel_gates/P4/representation_runtime_v1/representation_runtime_v1/`

### Listener stateful BF16 contro oracle full

Probe realtime su quattro frame PCM:

| frame | max diff | MAE | cosine |
|---:|---:|---:|---:|
| 0 | 5.75 | 0.7196545 | 0.984672 |
| 1 | 6.50 | 1.1404827 | 0.938604 |
| 2 | 6.75 | 0.4248498 | 0.986723 |
| 3 | 5.50 | 0.4398242 | 0.974057 |

Un probe indipendente su una fixture casuale ha prodotto, per il percorso
incrementale corrente, max diff `12.0`, MAE `0.4281`, cosine minima `0.9600`.
Una variante con prefill stabile separato ha migliorato i valori a max `6.5`,
MAE `0.2891`, cosine `0.9891`, ma resta molto oltre le soglie.

Il fatto che `NATIVE_VS_HELPER` sia `max=0.0, mae=0.0` nello stesso probe
esclude l'oracle come causa della discrepanza.

## 4. Performance osservata

Percorso BF16 listener:

- primo frame: circa `837 ms`;
- steady state: circa `98.6 ms`;
- limite richiesto: `80 ms`.

Esperimento diagnostico con language model FP32 e audio tower BF16:

- parity su quattro frame: max diff circa `0.0007`, MAE circa `0.00011`,
  cosine circa `1.0`;
- memoria GPU: circa `24.2 GB`;
- primo frame: circa `796 ms`;
- steady state: p50 circa `83.5 ms`, p95 circa `85.4 ms`.

L'esperimento FP32 è un'indicazione tecnica, non una chiusura del gate: non è
stato validato su 100 sequenze/5000 frame e supera ancora il budget realtime,
oltre a lasciare margine VRAM insufficiente per il runtime completo con Qwen.

## 5. Impatto su P4 e P4D

### P4

- V9 T1 equivalence: **PASS** — MAE `0.0008888`, cosine minima `0.9999936`.
- V9 T2 causalità: **PASS** — differenza futura massima `0`.
- V9 T3 performance: **PASS** — p95 circa `2.886 ms`.
- T4 causal Qwen baseline: **FAIL** — CE ratio `1.4389`, top-1 drop `5.44 pp`,
  top-5 drop `9.52 pp`.
- Representation parity offline: **PASS**.
- Representation parity del listener runtime contro oracle full: **FAIL**.

Il passaggio dei soli test V9 non basta a chiudere P4, perché il listener non
ha ancora dimostrato di fornire la rappresentazione canonica in realtime.

### P4D

La pipeline `FuturePrefixQwen`, il fencing generation/segment e il
`SpeakerPlayoutBuffer` sono implementati e i test fake mirati passano. Tuttavia:

- la parity numerica dello slot-KV Qwen resta irrisolta;
- il runtime probe predefinito è appena fuori soglia: CE ratio `1.1000517`;
- il risultato `first_raw_frame + L=40` con variante sperimentale `x1.5` è
  solo diagnostico e non sostituisce il gate predefinito;
- mancano gate realtime end-to-end, cancellation race, LISTEN e source binding
  finale.

## 6. Soluzioni candidate da valutare

1. **Percorso LM FP32 o misto:** è l'unica variante già osservata con parity
   numerica vicina alla soglia, ma richiede riduzione di memoria e latenza.
2. **Runtime stateful con shape/kernel deterministici:** mantenere cache e
   causalità, ma impedire che full e chunk usino percorsi numerici incompatibili.
3. **Ricalcolo full del prefisso:** probabilmente corretto numericamente, ma
   incompatibile con il budget realtime salvo ottimizzazione sostanziale.
4. **Modifica del contratto/soglie:** non accettabile senza una decisione
   esplicita; non è stata applicata alcuna deroga.

Non sono accettabili: oracle definito riusando il percorso incrementale,
maschere generiche, ritardo del listener, doppia causal convolution,
modifiche ai pesi/codice funzionale Qwen, SubTalker o Code2Wav, oppure soglie
indebolite.

## 7. Decisioni richieste

Per procedere serve scegliere, in ordine di priorità:

1. se il limite di 80 ms è rigido anche durante il primo warm-up;
2. il budget VRAM disponibile quando Gemma e Qwen sono residenti insieme;
3. se è accettabile un LM Gemma FP32/misto in produzione;
4. quale compromesso è preferibile tra parity numerica, memoria e latenza;
5. se il gate deve certificare anche la parity del listener su 100/5000 prima di
   riattivare lo sweep P4D.

## 8. Stato Git

Prima della pubblicazione di questo report il repository era stato riallineato
su `main` e sincronizzato con `origin/main` al commit:

`86d4a9f feat(p4): add canonical Gemma and stateful Qwen runtime`

Sono state escluse da Git le evidenze raw e le probe temporanee locali tramite
`.gitignore`; restano disponibili sul checkout per la diagnosi. Il report viene
committato separatamente e il controllo finale dovrà verificare:

```text
git status --short --branch       # nessuna modifica non tracciata
 git rev-parse HEAD origin/main   # stesso SHA
```
