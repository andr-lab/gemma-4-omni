# NEXT STEP — dopo il VERDETTO V5 (2026-08-25)

> ⚠️ QUESTO DOCUMENTO SOSTITUISCE la versione del 2026-08-24 (piano P1-P6 su AR decoder + MTP isolato).
> Il punto architetturale è stato RIVISTO con esperimenti: vedi `docs/VERDETTO_TASK12_ARCHITETTURA_V5.md`.

## Verdetto sperimentale (25 agosto 2026, blind ASR)

| Test | Percorso | Esito |
|---|---|---|
| NATIVE | talker completo in loop (dal testo) | **ASR esatto** |
| LOOP A | backbone in loop (codici GT) → past_hidden → **MTP isolato** da [h, cb0] | **Parlato fluente** (il MTP PUÒ dal solo cb0 con h ricorrente del loop) |
| LOOP B | h_sub adapter (cos 0.63) iniettato nel loop | Elica (adapter non allinea abbastanza) |
| oracle 23 ago | MTP isolato + h_t statico (passata unica) | garbled (formato h fuori distribuzione) |

**Conclusione**: il collo NON è il MTP (funziona dal solo cb0 nel loop nativo) — è il **ponte Gemma→talker**.
Niente AR decoder per cb0, niente MTP isolato, niente coupling cos-MSE statico. La proiezione va addestrata **nel loop**.

## Architettura v5 (FISSATA)

```
Gemma (congelata) → h_raw (1536) → [PROIEZIONE ADDESTRATA, unico componente trainabile] → h_proj (2048)
  → backbone Qwen3-TTS in loop (congelato: codec_head cb0 + MTP residui, protocollo generate nativo)
  → Code2Wav → voce
```
Lezione Qwen3-Omni (2509.17765): token discreti = contenuto, hidden = espressività. Il testo di Gemma
(inner monologue) resta nel prefill del talker (canale contenuto, percorso NATIVE già ASR esatto);
h_proj aggiunge il canale espressivo frame-per-frame.

## Stato corrente

- [x] Ambiente unificato verificato: python di sistema (unsloth, transformers 5.5.0) + qwen3tts_src
      patchato (3 shim: `@check_model_inputs` senza parentesi, `ROPE_INIT_FUNCTIONS["default"]`,
      `talker_config.pad_token_id=None`) → **Gemma + teacher coesistono in un processo**
      (container `gemma4-omni-v5:latest`). Patch nel repo: `qwen3tts_src/.../modeling_qwen3_tts_tokenizer_v2.py` riga 499.
- [x] Trainer v5 scritto: `tools/train_loop_coupling_v5.py` (proiezione ResidualAdapter-style 1536→2048,
      warm-start da `_archive_ckpts/temporal2/ht_adapter_temporal2_1000.pt`, teacher forcing nel loop,
      loss CE cb0 + λ·CE residui, eval ASR nel loop).
- [ ] Smoke test v5 (gate: loss scende, eval ASR produce contenuto).
- [ ] Training completo v5 (100k step, checkpoint `checkpoints/proj_v5_final.pt`).

## Prossimi passi

1. **Training v5 completo** (se smoke OK) — unico componente addestrato per la voce.
2. **DM head (seconda run, ORTOGONALE)**: predizione eventi full-duplex (backchannel/yield/barge_in/
   silence) da h_raw di Gemma, addestrato sui 16 dialoghi sintetici (GT eventi già pronti in
   `data/synth_it/out/`). NON tocca il talker né la proiezione. I backchannel si INIETTANO (clip
   esistenti: 20 sintetiche + 168 reali SmoothConv), non si generano col talker.
3. **Runtime monoprocesso**: Gemma (unsloth) + proiezione + talker in loop, con DM head che decide
   SPEAK/BACKCHANNEL/IDLE; barge-in = stop del talker.
4. **Vector steering affettivo** (post-full-duplex): vettore arousal/valence da h_raw → condizionamento
   additivo nel loop (pattern Moshi embedding sum).
5. **Context window** (post-full-duplex): finestra scorrevole Gemma 30-60s + compaction; cache talker
   reset per turno (PRIME-Speech: senza reset WER esplode).

## Riferimenti chiave

- Verdetto v5: `docs/VERDETTO_TASK12_ARCHITETTURA_V5.md`
- Letteratura: `docs/DESIGN_FULLDUPLEX_LETTERATURA.md` (sez. 2 rivista), `docs/sources/SUMMARY_ALL_PAPERS.md`
- Report ricerca arXiv thinker-talker: vedi conversazione (2503.20215, 2509.17765, 2606.30944, 2503.04724, 2410.00037)
