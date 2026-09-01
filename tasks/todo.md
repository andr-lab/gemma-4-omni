# P4 checklist

- [x] Definire contratto `IncrementalV9State` / `IncrementalV9Projector`
- [x] Implementare CausalConv stateful + KV state per 4 Transformer layer + RoPE position
- [x] Aggiungere test unitari tiny model: shape, reset, causalità, commit/error semantics
- [x] Equivalence BF16: 100 sequenze reali validation vs forward batch
- [x] Integrare V9 step nel listener/TurnHead path con anchor fisso per sessione
- [x] Performance: >=10.000 step post-warmup, p95 <=10 ms
- [x] Causal-prefix Qwen ablation: >=100 sequenze, >=5.000 frame
- [x] Gate/verifier/evidence P4 e workflow CI
- [x] Correggere P4-T4 alle soglie originali e pubblicare il baseline FAIL
- [x] P4D: misurare lookahead speaker-only con Qwen frozen e anchor causale
- [x] Implementare `SpeakerPlayoutBuffer` senza ritardare listener/Gemma
- [ ] Integrare il conditioning V9 frame-stateful + lookahead nel percorso speaker
- [ ] Code review finale P4D prima della chiusura

## Stato

P4-T1/T2/T3 restano PASS, ma P4-T4 baseline è correttamente FAIL alle soglie
originali: commit `269f4cf`, CE ratio `1.3047` (>1.10), top-5 drop `6.953 pp`
(>3 pp). Il precedente PASS del commit `3929a3d` era dovuto a un gate T4
incompleto che controllava solo che i logit cambiassero.
La validation cache remota `cache_ars_full_v2.pt` contiene 100 sequenze (`va_raw`
64x1536, `va_sub` 64x2048, `va_htx` 1536, `va_c` 64); `va_raw` è il tensore
post-CausalConv prodotto da `extract_ars_cache_v2.py`, quindi il gate attesta
esplicitamente equivalenza dello stesso percorso stateful su vettori reali già
normalizzati dalla conv. L'allineamento codebook è verificato contro la split
validation parquet a `codes_validation_start=1650667`.

P4 baseline evidence: `validation/kernel_gates/269f4cf60119da3bfe6ce565896ec6d77cf070fa/P4/run8/`.
P4D evidence frozen-Qwen: `validation/kernel_gates/5a795d8dfcf866a83cd635b8e214e49b445299ce/P4D/run1/`;
verifier indipendente PASS. `first_raw_frame` causale + 7 frame lookahead
(560 ms) passa; 0–6 frame fallano. Qwen checkpoint hashes attestati,
parametri trainable vuoti, pesi non modificati. `SpeakerPlayoutBuffer` è
abilitabile con delay 0 di default e resta speaker-only. QLoRA/P5+ non sono
stati avviati.
