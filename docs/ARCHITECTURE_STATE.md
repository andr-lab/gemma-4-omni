# ARCHITECTURE STATE — Stato Consolidato (23 agosto 2026)

**Checkpoint canonico**: `checkpoints/ht_adapter_temporal2_final.pt` (3090) + copia standard `ht_adapter_stage2r2_final.pt`
**Pipeline canonica**: `train_temporal2_cv26_faithful.py` (replica fedele r3_v7/temporal2, adattata al 3090)
**Dataset canonico**: CV26 italiano 40.8h — `data/train_tokens_cv26_it_40.8h_31916s_*.parquet`
**Decoder**: ufficiale `Qwen3TTSTokenizerV2Model` via subprocess venv phase1 (`decode_official_worker.py`, ADR-003)

---

## 1. ARCHITETTURA ATTUALE (VALIDATA)

```
Gemma 4 E2B (congelata, unsloth 4-bit)
   │ hidden 1536, finestre 64 frame @12.5Hz
   ▼
TemporalResidualAdapter  ← UNICO componente addestrabile
   (residual MLP + transformer temporale 1024D, 4 layer, 8 heads)
   │ h_t 2048 (bf16)
   ▼
Qwen3-TTS Sub-Talker (congelato, MTP: cb1..cb15 condizionati a h_t+cb0)
   ▼
Code2Wav ufficiale (subprocess venv phase1) → wav 24kHz
```

**Loss** (Task 11I-bis, validata): `L = cos_loss + 0.5·MSE + 0.2·L_coupling` su frame attivi.
**Teacher h_t**: estratto dal talker ufficiale Qwen3-TTS (sorgente isolata in `qwen3tts_src/`,
mai da `.pylibs` che contiene accelerate rotto).
**Risultato chiave** (blind ASR, nessun testo atteso): con ancoraggio GT ai codebook 0..7,
i residui predetti dal modello producono **italiano comprensibile e contenuto corretto**
("L'impianto era infine adottato di un fabbricato viaggiatori." = GT).

## 2. REGISTRO FALLIMENTI — perché le architetture precedenti non funzionano

| Architettura | Perché è stata provata | Perché ha fallito | Evidenza |
|---|---|---|---|
| **v3-CE + QLoRA** (Task 11G/H): DeepSemanticHead(2048) + ResidualAdapter + CE multi-codebook + QLoRA co-adattamento | Predire direttamente i codici con CE; QLoRA per co-adattare il backbone | Inferenza libera: silenzio/allucinazioni; interfaccia h_t→sub-talker rotta anche con cb0 GT (diag ibrida) | Blind ASR 23/8 mattina; hybrid diag |
| **Trainer semplificato 6B** (Task 11I prima versione): causal_conv + Linear head + Sequential adapter, SENZA modulo temporale | Replica "leggera" della ricetta | Non fedele: mancava il transformer temporale e l'estrazione teacher corretta → stesso fallimento in free-gen | Blind ASR 23/8: silenzio/allucinato |
| **TurnHead v3/v4** (etichette euristiche energetiche) | Dare supervisione di turno senza annotazioni | Le label simulate non corrispondono al comportamento umano; accuratezza insufficiente al controllo (bACC 41%) | Task 10 report |
| **TurnHead v5 run1/run2** (GT nativo, pesi classe) | Usare VAD frame-level + state nativi | Collasso verso la classe favorita dai pesi (YIELD 98% / BARGE_IN 85%): le feature di input (spettro mel medio/frame) non discriminano le classi reali | Log run 1/2 |
| **MDC v2 adapter** (Task 11B, cos sim 0.9366) | Più dati italiano mono | Metrica latente alta ma voce robotica: la cos sim non predice la qualità percettiva | Feedback utente + blind ASR |
| **QLoRA su Gemma** (qlora_r1) | Co-adattare il backbone | Nessun beneficio in accuratezza, costo computazionale alto | Valutazione utente |

## 3. LEZIONI METODOLOGICHE (vincolanti)

1. **Gate percettivo cieco OBBLIGATORIO**: blind ASR (faster-whisper, nessun testo atteso)
   + ascolto utente. Mai dare il testo atteso al giudice prima della trascrizione.
2. **Le metriche latenti (cos sim, mel-L1) NON predicono l'intelligibilità** in free-generation.
3. **Replica fedele > re-invenzione**: quando una pipeline storica funziona, replicarla
   1:1 prima di modificare qualunque cosa (il fallimento del 11I prima versione è la prova).
4. I wav storici "buoni" (N7/oracle/gt_ref) erano ancorati al GT: mai usarli come prova
   di free-generation senza verificare la composizione dei codebook.

## 4. GAP APERTO (unico frontier rimasto)

**Predizione libera di cb0**: la testa semantica produce codici degradati senza ancoraggio GT
(silenzio/loop). L'ancoraggio ai primi codebook + residui predetti funziona.
Strade candidate (da letteratura): decoding autoregressivo/scheduled sampling della testa
semantica; talker nativo AR condizionato da Gemma (opzione consigliata, coerente con
Moshi/FireRedTTS3); verifica con blind ASR a ogni passo.

## 4b. TASK 12 IN CORSO — AR-SemanticDecoder (24 agosto 2026)

**Pattern VALL-E** per colmare il gap cb0 libero, sopra l'adapter validato CONGELATO:
```
Gemma frozen → TemporalResidualAdapter (congelato) → h_t
                                                     ▼
        [NUOVO] AR-SemanticDecoder: 4 layer causali, d=1024, 8 heads
        passo t: emb(cb0_{t-1}) + proj(h_t) → logits cb0_t (rolling)
                                                     ▼
        sequenza cb0 completa → Sub-Talker frozen → residui → Code2Wav
```
- Training: teacher forcing CE (BOS=0) → scheduled sampling 35% (Bengio 2015)
- Condition dropout 10% (abilita future classifier-free guidance, VALL-E 2)
- Inferenza: repetition-aware sampling (penalità su token ripetuti nelle ultime 12 posizioni,
  VALL-E 2) + top-k 50 + temp 0.9
- File: `tools/train_ar_semantic.py`, `tools/gen_ar_semantic.py`
- Infrastruttura: estrazione h_t/cb0 precomputata e cachata su disco
  (`cache_ars_full.pt`, 25.791 finestre train + 2.865 val) — il training con cache costa
  ~40s per 2.000 step
- Evidenza primo run (2.000 step): loss ~7.5 (ln 2048=7.62, appena sotto il caso casuale),
  TF-acc val 3,4% → modello sotto-addestrato; rilanciato a 100.000 step (chain detached
  `ars_chain`, log `/workspace/logs/chain_ars.log`)
- Gate pendente: generazione libera + blind ASR (mai testo atteso prima della trascrizione)
- Revisione avversariale a contesto fresco delegata ad agy (letteratura via API arxiv ufficiale:
  VALL-E 2 repetition-aware sampling/CFG, ELLA-V constrained decoding)

## 5. MANIFEST PULIZIA (23 agosto 2026)

### Rimossi dal 3090 (~7GB → `_archive_ckpts/`)
`gemma4_duplex_sota_v2|_v3`, `task11i_cv26_v2recipe`, `adapter_it`, `turnhead_v2`,
`gemma4_duplex_sota(|_ckpt)`, `temporal2` (vecchio), `turnhead_v3_*.pt`,
`turn_state_head_v3|v4.pt`, step intermedi temporal2.

### Archiviati localmente
- `_archive/agy_prompts/`: 22 prompt task conclusi
- `_archive/agy_logs/`: 25 output agy + ascolti
- `_archive/runners/`: runner e harvester task conclusi

### Radice: solo infrastruttura attiva
`kaggle_pipeline.py` (pipeline principale), `train_temporal2_cv26_faithful.py` (trainer
canonico), `code2wav_decoder.py` + `decode_official_worker.py` (decoder ufficiale),
`qwen_subtalker.py`, `phase2_support.py`, `paths.py`, `serve_duplex.py`, `run_3090.py`,
`setup_3090.sh` / `docker_run_3090.sh` / `_dl_models_3090.sh`,
`train_turnhead_v5_final.py` + `train_task11i_cv26_v2recipe.py` (storico trainer,
in revisione), `tools/` (blind ASR, builder dataset, merge), `_test_worker_daemon4.py`
e `_test_wrapper_e2e.py` (verifica decoder).
