# ADR-030: CUDA Graph Sub-Talker nel Duplex Kernel — Ricostruzione e Gotcha

## Status
Accepted

## Date
2026-08-29

## Context

Il sub-talker RVQ (15 codebook residui, `Qwen3TTSTalkerCodePredictorModel`) è il
collo di bottiglia del talker: il `code_predictor.generate` HF fa 15 forward
autoregressivi sequenziali + sampling → **~99ms/frame** misurati su RTX 3090.
A 12.5 frame/s (ARIA 12Hz), questo da solo supera il budget real-time
(99ms > 80ms/frame).

ADR-029 citava un "RTF 0.7541" storico (README, NEXT_STEP) attribuito a un
"CUDA Graph Sub-Talker 12.25ms/frame (8.6×)". **Verifica critica fatta oggi**:
il codice di quell'esperimento **non è mai stato persistito** nei `tools/`
(`interactive_gemma4_omni_live.py` è eager, zero import CUDAGraph). Il valore
0.754 era reale ma non riproducibile da nessun file del repo. Il primo tentativo
di ricostruzione (`subtalker_graph.py` v1) falliva e — peggio — **degradava
l'ASR** (casi "Sottotitoli a cura di QTSS" nei benchmark).

Questo ADR documenta la ricostruzione funzionante e i **quattro gotcha** che
hanno richiesto iterazioni per essere trovati. Sono esattamente il tipo di
conoscenza che, se non scritta, verrà ripagata in cicli di debug da ogni futuro
agente.

## Decision

Ricostruire l'accelerazione del sub-talker come **15 CUDA graph statici**
(1 prefill + 14 step autoregressivi) su **StaticCache** preallocata, con
sampling top_k/top_p **esterno ai graph** (bit-exact con il generate HF).

### Architettura del graph (per frame, 12.5Hz)

```
PRELLOOP (per frame):
  c0_embed = talker.get_input_embeddings()(c0)          # embedding del TALKER, 2048 dim
  prefill_inputs = cat([last_h, c0_embed], dim=1)        # (1, 2, 2048)
  reset cumulative_length di OGNI StaticLayer → 0
  static_prefill_inputs.copy_(prefill_inputs)
  prefill_graph.replay()                                  # scrive KV pos 0,1; logits → residuo 1
  r1 = sample(static_logits)                              # top_k=20, top_p=0.90, T=0.80, RNG globale
  static_tok.copy_([[r1]])
  for g in 1..13:
      static_pos.fill_(g+1)                               # RoPE position
      step_graphs[g-1].replay()                           # scrive KV pos g+1
      r = sample(static_logits)
      static_tok.copy_([[r]])
  tokens = [c0] + [r1, ..., r15]
```

Il modello sub-talker è un decoder MTP con **una testa dedicata per step**
(`lm_head[generation_steps]`): lo step g usa `generation_steps=g` e scrive il
KV alla posizione `cumulative_length` (avanzante). I graph congelano la forma:
input (1,1) + StaticCache (1, n_heads, 17, head_dim) → replay con `copy_` dei
buffer statici.

### Gotcha 1 — Il prefill del sub-talker usa l'embedding del TALKER, non il proprio

`torch.cat((last_h, last_id_h))` nel generate HF ufficiale concatena la hidden
del **talker** (2048 dim) con l'embedding di c0 del **talker** (2048 dim);
il code predictor proietta tutto con `small_to_mtp_projection` (2048→1024).
Usare `code_predictor.get_input_embeddings()[0]` (1024 dim) produce un mismatch
di shape silenzioso → `index_copy_` out of bounds device-side.

**Fix**: passare `talker_embed = talker.get_input_embeddings()` al graph;
mai `cp.get_input_embeddings()` per il primo token.

### Gotcha 2 — La cache deve avere 17 posizioni, non 16

`max_cache_len = num_code_groups + 2` (15 + 2) = **17**: prefill scrive le
posizioni 0,1, poi 14 step scrivono 2..15... in realtà il 15° residuo scrive
alla posizione 16 (cumulative_length parte da 2 dopo il prefill). Con
`max_cache_len=16` il penultimo/ultimo `index_copy_` va fuori bounds.
Il bug si manifesta solo al replay (device-side assert async) — difficile da
localizzare senza `CUDA_LAUNCH_BLOCKING=1`.

### Gotcha 3 — `cumulative_length` vive su OGNI StaticLayer ed è mutato in-place

`StaticCache.update` fa `cache_position = arange(kv_length) + cumulative_length`
e poi `cumulative_length.add_(kv_length)`. Nei CUDA graph il tensore è un
indirizzo statico: al replay parte dal valore lasciato dal replay precedente
(16), quindi `index_copy_` va fuori bounds. **Il reset va fatto su ogni layer**
(`for l in cache.layers: l.cumulative_length.zero_()`), non sulla cache —
`StaticCache` non espone `cumulative_length` come attributo proprio.

### Gotcha 4 — Mai allocare tensori dentro il graph

`cache_position=torch.arange(0, 2, device=...)` creato **dentro** il blocco
`torch.cuda.graph()` viene catturato con un puntatore a memoria che il replay
riusa in modo incoerente → errori asincroni. Tutti gli input (cache_position,
input_ids, prefill_inputs) devono essere **buffer statici preallocati prima
della cattura** e aggiornati con `copy_`/`fill_` prima di ogni replay.

### Gotcha 5 — Sampling bit-exact: mai seed fisso per residuo, mai dimenticare top_p

La v1 del sampler usava `torch.manual_seed(42+g)` per ogni residuo g → **la
stessa sequenza random a ogni frame** → artefatti audio sistematici (casi ASR
degradati). Il generate HF usa il **RNG globale che avanza naturalmente**:
`torch.multinomial(probs, 1)` senza seed. Inoltre il warper order di
transformers è `temperature → top_k → top_p` — la v1 ometteva `top_p=0.90`.
`_warp_logits()` in `subtalker_graph.py` replica l'ordine esatto.

## Risultati misurati (RTX 3090, container gemma4-omni-v5, transformers 5.5.0)

| Metrica | Eager (generate HF) | **CUDA Graph** | Speedup |
|---|---|---|---|
| Sub-talker 15 residui | 99 ms/frame | **15.5 ms/frame** | 6.4× |
| RTF kernel end-to-end | 2.28 | **~1.0** (0.97–1.09) | 2.3× |
| ASR Whisper | 5/5 (generate) | **3/3 perfetto** | — |
| Crest Factor | 5.16 | 4.19–4.52 | — |
| HNR | 4.15 dB | 0.27–2.88 dB | — |

Il residuo vs l'ADR-026 storico (12.25ms vs 15.5ms) è l'overhead del sampling
Python + 15 `copy_` + sincronizzazioni per frame — candidato a un graph unico
del loop completo o sampling CUDA-side in futuro.

## Alternative Considered

### torch.compile(mode="reduce-overhead") sul code predictor
- **Pros**: applica CUDA graphs automaticamente, zero gestione manuale.
- **Cons**: **fallisce** con StaticCache in-place (`index_copy_` non compilabile
  senza `cudagraph_mark_step_begin()`); errore noto di transformers/torch.
- **Rejected**: non applicabile al sub-talker con cache statica.

### Graph a bucket (un graph per lunghezza)
- **Pros**: niente StaticCache, forme fisse per bucket.
- **Cons**: 15 allocazioni KV distinte, più memoria, stessa complessità.
- **Rejected**: StaticCache a 17 posizioni è più semplice e già validata.

### Loop eager a passo singolo senza graph
- **Pros**: corretto e semplice (85ms con StaticCache vs 99ms generate).
- **Cons**: ancora sopra il budget real-time (85ms > 80ms).
- **Rejected**: serve il graph per scendere sotto il frame budget.

## Consequences

- **Positivo**: sub-talker da 99→15.5ms/frame (6.4×), RTF kernel end-to-end
  ~1.0, ASR bit-exact con l'eager, zero dipendenza da generate HF nel loop caldo.
- **Positivo**: il pattern (StaticCache + graph per step + sampling esterno +
  reset cumulative) è riusabile per il **talker backbone** (33ms/frame → ~12ms
  attesi) e per il futuro kernel C (stesse invarianti).
- **Negativo**: 15 graph consumano ~150MB di pool CUDA; il reset della cache
  per frame è obbligatorio (se dimenticato: crash async difficile da debuggare).
- **Negativo**: il sampling resta Python-side (15 `multinomial` + sync per
  frame) — prossimo collo di bottiglia dopo il graph del backbone.

## File coinvolti

- `duplex_kernel/subtalker_graph.py` — `SubtalkerGraph` (15 graph), `_warp_logits`
- `duplex_kernel/engine.py` — integrazione: `sg.generate(last_h, c0, c0_embed=c0_embed)`
- Benchmark: `duplex_kernel/benchmark.py` → `samples_kernel/kernel_benchmark_report.json`
