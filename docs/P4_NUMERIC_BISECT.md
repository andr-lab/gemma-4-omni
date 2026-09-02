# P4 Numeric Bisect — Gemma incremental BF16

**Status:** diagnostico; non è un gate e non modifica l'oracle canonico.
**Fixture:** 4 frame, seed `7`, audio soft hash `e04ea782b7f93bc030f988d443057f6b15cac0ec1d50c52e7a63f376c6991a36`.

## Primo punto divergente

- Full BF16 vs incremental BF16: layer `0`, componente `k_projection`, MAE max `0.031108856201171875`, cosine min `0.9999939203262329`.
- Full FP32 diagnostico vs incremental FP32: nessuna divergenza significativa rilevata.
- Significatività: MAE > `0.001` oppure cosine < `0.9999`; per top-k basta un expert mismatch.

## Representation finale

| confronto | MAE | max abs | cosine min | shape mismatch | non-finite |
|---|---:|---:|---:|---:|---:|
| full BF16 → incremental BF16 | 0.22266688104718924 | 6.5 | 0.9891278743743896 | 0 | 0 |
| full FP32 → incremental FP32 (diagnostico) | 2.1439267896994352e-05 | 0.000484466552734375 | 0.9999998211860657 | 0 | 0 |
| full BF16 → full FP32 (precision drift) | 0.1881718127988279 | 4.7226715087890625 | 0.9838064908981323 | 0 | 0 |

## MoE routing

| confronto | primo layer mismatch | mismatch rate per layer | margin al mismatch |
|---|---:|---|---:|
| BF16 incremental | None | n/a | None |
| FP32 incremental | None | n/a | None |

## Precision ablation presente

- `full_bf16`: forward nativo Gemma con pesi/attivazioni BF16.
- `incremental_bf16`: prefill stabile `[BOS|text|BOA]` separato, poi 4 soft token/frame con cache native.
- `full_fp32` + `incremental_fp32`: percorso nativo e transizione incrementale con il language model interamente FP32 **solo come riferimento diagnostico**.
- Non sono ancora state applicate shadow projection, attention FP32 selettiva o router FP32: saranno valutate dopo il bisect.

## Timing/memoria del probe strumentato

| variante | total ms | stable p50 ms | audio p50/p95/p99 ms | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|
| full_bf16 | 300.83933705464005 | None | None/None/None | 10263095808 | 10385096704 |
| incremental_bf16 | 662.0301879011095 | 71.55350293032825 | 128.59252153430134/130.14138573780656/130.2964379824698 | 10260432896 | 10385096704 |
| full_fp32 | 230.29586113989353 | None | None/None/None | 19566100480 | 24738004992 |
| incremental_fp32 | 712.9307540599257 | 71.76928711123765 | 143.88475904706866/147.58186532417312/147.84784973831847 | 19561785856 | 24738004992 |

## Interpretazione

Il percorso full BF16 sopra è l'oracle nativo e non viene ridefinito in base al risultato incremental; il full FP32 è solo il controllo diagnostico della precisione C. Questo report localizza il primo scarto numerico e separa la possibile amplificazione MoE dalla causa iniziale; non autorizza ancora la scelta di una variante di produzione.

## Candidate shape-stable su validation reale (2026-09-02)

Il probe `p4_candidate_validation.py` ha valutato `shape_padded_all_ops_kv` contro
l'oracle BF16 nativo su 100 sequenze reali da 64 frame (6.400 frame), con ogni
candidate eseguito in cache nuova e senza patchare l'oracle.

| metrica | risultato | soglia |
|---|---:|---:|
| MAE | 0.0 | <= 1e-3 |
| max abs | 0.0 | diagnostica |
| cosine minima | 0.99999994 | >= 0.9999 |
| non-finite / shape mismatch | 0 / 0 | 0 / 0 |
| audio p50 / p95 | 76.36 / 77.63 ms | p95 < 80 ms |
| peak allocated / reserved | 10,352,332,800 / 10,454,302,720 B | budget RTX 3090 24 GiB |

La split è stata corretta: `SEGMENT_ENDS` portava a frame `1,650,667`, non
multiplo di 64. La validation ora parte al primo bordo completo successivo,
`1,650,688` (window offset `25,792`), e la cache attesta entrambi i valori.
Usare il precedente cache runtime senza questi campi è rifiutato dai gate di
parity/source alignment.

**Stato:** il risultato rimuove il blocco numerico del candidato diagnostico,
ma non chiude P4: il patch altera dinamicamente i `forward` dei moduli Gemma;
non è stato ancora integrato in modo concurrency-safe nel listener che condivide
il Thinker con la generazione testuale. L'oracle e il percorso di produzione
restano invariati finché quel confine non è progettato e validato.
