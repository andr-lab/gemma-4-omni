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
- [x] P4D preliminare: misurare lookahead speaker-only con Qwen frozen e anchor causale
- [x] Invalidare P4D preliminare per doppia causal_conv e introdurre cache pre-causal_conv
- [x] Implementare `SpeakerPlayoutBuffer` senza ritardare listener/Gemma
- [ ] Rigenerare cache Gemma pre-causal_conv e rieseguire P4 T1/T2/T3 + sweep P4D
- [ ] Integrare il conditioning V9 frame-stateful + lookahead nel percorso speaker
- [ ] Code review finale P4D prima della chiusura

## Stato

P4-T1/T2/T3 restano PASS, ma P4-T4 baseline è correttamente FAIL alle soglie
originali: commit `269f4cf`, CE ratio `1.3047` (>1.10), top-5 drop `6.953 pp`
(>3 pp). Il precedente PASS del commit `3929a3d` era dovuto a un gate T4
incompleto che controllava solo che i logit cambiassero.
La cache precedente `cache_ars_full_v2.pt` contiene `va_raw` post-CausalConv e
non è più valida per P4/P4D: il vecchio runner applicava una seconda conv. Il
nuovo extractor supporta `--raw-stage pre_causal_conv`, il metadata obbligatorio
`v9_raw_stage=pre_causal_conv` e il runner applica causal_conv una sola volta.
L'allineamento codebook resta verificato contro la split validation parquet a
`codes_validation_start=1650667`.

P4 baseline evidence: `validation/kernel_gates/269f4cf60119da3bfe6ce565896ec6d77cf070fa/P4/run8/`.
La precedente evidence P4D `validation/kernel_gates/5a795d8dfcf866a83cd635b8e214e49b445299ce/P4D/run1/` è **INVALIDATA** per doppia causal_conv; non usarla per decisioni. Va rigenerata su cache pre-causal_conv. `SpeakerPlayoutBuffer` resta abilitabile con delay 0 di default e speaker-only. QLoRA/P5+ non sono stati avviati.
