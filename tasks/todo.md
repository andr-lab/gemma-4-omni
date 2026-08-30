# P4 checklist

- [ ] Definire contratto `IncrementalV9State` / `IncrementalV9Projector`
- [ ] Implementare CausalConv stateful + KV state per 4 Transformer layer + RoPE position
- [ ] Aggiungere test unitari tiny model: shape, reset, causalità, commit/error semantics
- [ ] Equivalence BF16: 100 sequenze reali validation vs forward batch
- [ ] Integrare V9 step nel listener/TurnHead path con anchor fisso per sessione
- [ ] Performance: >=10.000 step post-warmup, p95 <=10 ms
- [ ] Causal-prefix Qwen ablation: >=100 sequenze, >=5.000 frame
- [ ] Gate/verifier/evidence P4 e workflow CI
- [ ] Code review finale con correctness/simplification/DDD/TDD prima di dichiarare P4 PASS

## Stato

P3 PASS e QLoRA ancora bloccato. P4 appena autorizzata esplicitamente dall'utente.
La validation cache remota `cache_ars_full_v2.pt` contiene 100 sequenze (`va_raw`
64x1536, `va_sub` 64x2048, `va_htx` 1536, `va_c` 64), da verificare rispetto al
formato raw pre-CausalConv richiesto dal nuovo step.
