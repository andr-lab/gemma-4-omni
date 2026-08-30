# P4 checklist

- [x] Definire contratto `IncrementalV9State` / `IncrementalV9Projector`
- [x] Implementare CausalConv stateful + KV state per 4 Transformer layer + RoPE position
- [x] Aggiungere test unitari tiny model: shape, reset, causalità, commit/error semantics
- [x] Equivalence BF16: 100 sequenze reali validation vs forward batch
- [x] Integrare V9 step nel listener/TurnHead path con anchor fisso per sessione
- [x] Performance: >=10.000 step post-warmup, p95 <=10 ms
- [x] Causal-prefix Qwen ablation: >=100 sequenze, >=5.000 frame
- [x] Gate/verifier/evidence P4 e workflow CI
- [x] Code review finale con correctness/simplification/DDD/TDD prima di dichiarare P4 PASS

## Stato

P4 PASS indipendente pubblicato per il commit `3929a3d581359958de115e6f6aa93819c8a90907`.
La validation cache remota `cache_ars_full_v2.pt` contiene 100 sequenze (`va_raw`
64x1536, `va_sub` 64x2048, `va_htx` 1536, `va_c` 64); `va_raw` è il tensore
post-CausalConv prodotto da `extract_ars_cache_v2.py`, quindi il gate attesta
esplicitamente equivalenza dello stesso percorso stateful su vettori reali già
normalizzati dalla conv. L'allineamento codebook è verificato contro la split
validation parquet a `codes_validation_start=1650667`.

P4 evidence: `validation/kernel_gates/3929a3d581359958de115e6f6aa93819c8a90907/P4/run7/`.
QLoRA può essere sbloccato solo dopo conferma utente e aggiornamento del piano
post-P4; P5+ non sono stati avviati.
