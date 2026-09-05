# P4-NATIVE — Stato sessione

**Data:** 2026-09-05 17:35 UTC
**Stato:** sperimentale / **NO-GO production** fino alla parity strict

## Sintesi

È disponibile un executor C/CUDA nativo per il listener Gemma 4, integrato
esclusivamente come opt-in tramite `GEMMA4_NATIVE_LISTENER=1` o
`use_native_lm=True`. Il default PyTorch `ShapeStableGemmaListenerLM` non è
stato modificato.

Il backend è stato compilato su RTX 3090 (`sm_86`, senza `--use_fast_math`) e
misura circa **10.47 ms medi / 11.69 ms p95** per frame `batch=1, Q=4`.
Il percorso steady-state non esegue `cudaMalloc`, readback o
`torch.cuda.synchronize()` e il gate end-to-end di `GemmaListener.encode()`
riporta delta memoria zero anche con un input Q=4 non contiguo. Il gate ha
eseguito 65 frame di warm-up più 10 frame steady-state, con posizione LM
`294 -> 334`.

La parity numerica completa **non è certificata**. Risultati correnti:

- prefix + step canonico: MAE circa `0.480`, cosine `0.9737`, max diff `17.82`;
- step unprimed diagnostico: MAE circa `0.171`, cosine `0.9988`, max diff `4.0`;
- prefill component-wise: cache K/V con cosine elevata, ma hidden finale
  post-RMSNorm circa MAE `0.543`, cosine `0.968`.
- con il bridge diagnostico ATen SDPA + `pre_ff_norm`, il multi-frame resta
  **NO-GO**: MAE `0.3115/0.6501/0.6796`, cosine
  `0.9890/0.9768/0.9736`.

Di conseguenza il native backend resta opt-in e non deve essere usato in
produzione.

## Artefatti

| Componente | File | Stato |
|---|---|---|
| ABI Python/C | `duplex_kernel/native_listener.py`, `native/gemma4_listener/gemma4_listener.h` | implementato; ABI v3 + size/offset handshake + tensor descriptors |
| Executor | `native/gemma4_listener/gemma4_listener.cu` | implementato; CUBLAS + kernel CUDA |
| Kernel | `native/gemma4_listener/gemma4_listener_kernels.cuh` | RMSNorm, GeGLU, RoPE, SDPA Q4/prefill |
| Cache | `native/gemma4_listener/gemma4_listener_cache.cu` | KV device-resident, layer sharing attestato |
| Topologia | `native/gemma4_listener/generated_topology.h` | 35 layer, manifest SHA attestato |
| Integrazione | `duplex_kernel/listener.py` | opt-in, prefix separato, frame Q=4 persistenti |
| Test/gate | `tools/kernel_gates/` e `tools/tests/` | smoke, parity, diagnostica, steady-state; bisect layer-local strict |
| Bisect diagnostico | `tools/kernel_gates/test_native_layer_local_parity.py` | cache prefix exact, tap per-op CUDA, first-divergence + future-leak gate |
| ATen RMSNorm oracle | `native/gemma4_listener/gemma4_listener_aten_norm.*`, `GEMMA4_ATEN_RMSNORM_BRIDGE=1` | bridge literal opt-in, solo pre-FF norm |
| FFN isolation | `tools/kernel_gates/test_native_ffn_primitives.py` | cuBLAS/GELU/mul/down exact con input oracle |

Contratto numerico: prefix esatto `BOS + 32 slot testo + BOA = 34`, prefill
shape-stable `M=291` con sole 34 righe valide in cache, mask CUDA `int64`
binaria obbligatoria, frame `batch=1, Q=4`, pesi borrowed CUDA BF16 senza
copia dei pesi.

## Verifiche superate

- Build CUDA RTX 3090 e primitive native: **PASS**.
- Manifest SHA interno e SHA esportato dalla libreria: **PASS**.
- ABI version e dimensioni delle struct Python/C: **PASS**.
- Validazione topology, shared-KV, shapes, epsilon, scalari e RoPE: **PASS**.
- Pesi su un solo device CUDA, BF16 e contigui: **PASS**.
- Final RMSNorm isolato: **bit-exact** nel gate dedicato.
- Integrazione: prefix position `34`, step position `38`, reset position `0`: **PASS**.
- Continuità frame 63/64/65: posizioni `290 -> 294 -> 298`, nessun rollover LM
  artificiale: **PASS**.
- Steady-state executor 100 frame: allocazioni `0`, memoria riservata `0`,
  readback `0`, sync Python `0`: **PASS**.
- Steady-state end-to-end `GemmaListener.encode()`: 10 frame dopo 65 frame di
  warm-up, allocazioni `0`, memoria riservata `0`, sync Python `0`: **PASS**.
- Benchmark corrente: `10.47 ms` medi, `11.69 ms` p95 su RTX 3090.
- Import cache ShapeStable esatto: K/V importati bit-identici e posizione nativa `34 -> 38` dopo il probe Q=4.
- Bisect layer-local default: primo failure `layer 0 / attn_output`, MAE `2.0633e-4`, max `0.0625`, cosine `0.99999899`.
- RMSNorm isolata layer 2: il custom kernel diverge prima a `row_sum_fp32`
  (1–4 ULP), mentre il literal ATen è exact BF16; layer 4 `post_attn_norm`
  riproduce lo stesso pattern.
- FFN layer 2 isolata: gate/up, GELU-tanh, gate×up e down sono tutti exact
  BF16 (`MAE=0`), quindi nessun bridge GEMM/attivazione è stato promosso.
- Scan local ATen RMSNorm su tutti i 35 layer: 945 boundary, primo nuovo
  nonzero layer 4/post-attn norm, primo failure formale layer 9/post-FF norm;
  future-leak `0.0`. Report: `P4_NATIVE_LAYER_LOCAL_SCAN_ALL_ATEN_RMSNORM.json`.
- Future-leak numerico sul prefisso mascherato: max abs `0.0` (gate `<=1e-5`).
- Test locali mirati: `16 passed`; `py_compile` e `git diff --check`: **PASS**.
- `pytest -q` completo locale: non gate valido; collection interrotta da 73 moduli
  legacy che richiedono checkpoint/GPU/dipendenze non disponibili.

Il frame non contiguo viene copiato in un buffer CUDA persistente preallocato,
poi consumato dal C executor; non viene creato un temporary Python per frame.
Il C executor non usa più uno staging buffer intermedio per il frame.

## Hardening applicato dopo la review adversarial

- handshake ABI con `sizeof` delle tre struct C/Python;
- controllo esplicito `rope_parameters` del checkpoint e uso dei base RoPE dalla
topologia generata nel codice C;
- validazione `frame_pad_ple` di device, dtype, contiguità e shape;
- validazione `stream_q_len=4` e capacità massima coerente con il kernel;
- `producer_stream` opzionale con `wait_stream` CUDA non bloccante per la
  dipendenza producer/consumer;
- controllo dei return code dei readback diagnostici KV;
- reset dello stato Python in `finally`; dopo un reset CUDA fallito il contesto
  nativo viene scartato e non riutilizzato;
- `cudaStreamSynchronize` nel reset C marca il contesto poisoned se fallisce;
- output borrowed persistente documentato valido fino alla chiamata successiva;
- ABI v3 verifica `sizeof`/offset e riceve descriptor host per pointer, dtype,
  rank, shape, stride e layout di pesi/IO; il descriptor è una attestazione del
  chiamante, non una prova della capacità reale dell’allocazione CUDA;
- ogni output è passato da un guard CUDA device-side: eventuali BF16 non finiti
  vengono neutralizzati a zero e registrati in un flag persistente; `verify_output_health()`
  è il controllo diagnostico sincronizzante esplicito, fuori dal fast path;
- prime/step osservano anche il precedente stream con `cudaStreamQuery`: `NotReady`
  è mantenuto come lavoro ordinato, ogni altro errore osservato marca il contesto
  poisoned senza sincronizzare l’host;
- `wait_for_output()` inserisce il fencing device-side richiesto quando un consumer
  usa uno stream diverso dal producer, mentre l’output borrowed resta valido solo
  fino alla chiamata successiva;
- la capacità KV esaurita porta ora lo stato nativo in errore terminale: non viene
  tentato un reprime infinito. Sono attestati 503 frame dopo il prefisso; è richiesto
  `GemmaListenerState.reset()`.

## Finding ancora aperti — blocco production

1. La visibilità degli errori CUDA asincroni è ora fail-closed al confine della
   transazione successiva: `cudaStreamQuery` non blocca e marca poisoned ogni
   errore diverso da `cudaSuccess`/`cudaErrorNotReady`; `verify_output_health()`
   resta la barriera esplicita per una certificazione immediata. Non è possibile
   restituire dal metodo asincrono un errore di esecuzione che non è ancora avvenuto
   senza attendere lo stream.
2. Il percorso default resta divergente a `layer 0 / attn_output` perché
   ShapeStable/PyTorch seleziona CUTLASS mentre il native usa SDPA scalar
   custom; l’ATen SDPA diagnostic-only elimina quel drift in isolamento. Con
   il solo bridge `pre_ff_norm`, il nuovo scan local mostra il primo nonzero a
   layer 4/post-attn norm e il primo failure formale a layer 9/post-FF norm.
   La parity strict multi-frame non è raggiunta e il native resta NO-GO.
3. Il prefill primed continua a essere un confine separato: il test cache
   import è exact per K/V, ma il passo da cache identica resta NO-GO. Il bridge
   RMSNorm non viene considerato certificato sul full path.
4. Il debug layer-local cattura copie D2D solo quando l’entrypoint diagnostico è
   attivo; le allocazioni dei tap sono create una volta nel contesto e non
   entrano nello steady-state production path.
5. La cache append-only ha capacità configurabile ma limitata a `4096` token
   dal kernel score; il percorso listener istanzia `max_seq_len=2048` (503 frame
   dopo il prefix). A capacità esaurita l’integrazione fallisce chiusa e richiede
   `GemmaListenerState.reset()`; non esiste rollover/reprime implicito.
6. Gli output e gli input sono borrowed; i consumatori su stream diversi devono
   usare `producer_stream`/`wait_for_output()` e consumare l’output prima della
   chiamata successiva.
7. L’attestazione copre manifest, topologia, parametri strutturali e gli hash dei
   file del checkpoint dichiarati nel manifest; non può provare capacità/ownership
   del raw CUDA pointer oltre la validazione del device e del descriptor.

## Prossimo gate obbligatorio

Prima di qualsiasi commit/release del percorso nativo:

1. localizzare il drift con i gate component-wise e verificare GEMM, shared-KV,
   PLE, attenzione e MLP contro il percorso ShapeStable;
2. definire un protocollo fail-closed per gli errori CUDA asincroni senza
   violare il contratto di latenza, oppure dichiarare esplicitamente il limite;
3. rieseguire parity strict su prefix, step, frame 63/64/65 e final norm;
4. eseguire code review finale e aggiornare questo snapshot con risultati
   riproducibili;
5. solo dopo, sincronizzare il repository remoto, creare il commit e notificare
   l’utente.
