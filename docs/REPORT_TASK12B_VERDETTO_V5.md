# REPORT TASK 12B — Verdetto sperimentale e architettura v5 (coupling in loop)

**Data**: 25 agosto 2026 | **Stato**: decisione architetturale FISSATA + trainer v5 in addestramento

## 1. Contesto (per chi arriva dopo)
Dalla sessione precedente (Task 12): l'AR decoder (cb0 da h_sub) stagnava (TF 7-12%) e l'ancorato (cb0 GT + residui predetti dal MTP) produceva "elica". Cache v1/v2 estratte con adapter RANDOM (bug radice: `load_gemma4_duplex_checkpoint` non caricava mai la TemporalResidualAdapter). La domanda aperta: il sub-talker può generare i residui dal solo cb0?

## 2. Esperimenti del 25 agosto (tutti con blind ASR faster-whisper small)

| Test | Descrizione | Risultato ASR |
|---|---|---|
| **NATIVE** | `teacher.generate(text)` talker completo in loop nativo (3 testi reali) | **Esatto** ("L'impianto era infine dotato di un fabbric.", "La verdura è anche estremamente popolare a Taiwán.", "Successivamente cercò.") |
| **LOOP A** | Backbone in loop (codici GT, cache, formato generate_icl_prompt) → cattura `past_hidden` → **MTP isolato** `sample_acoustic_codes` da [h, cb0] | **Parlato fluente** ("Sto in un studio!...", "Anna, te ti senti qua?...", "E' tra l'esistica...") — finestre con contenuto |
| **LOOP B** | `h_sub = adapter_temporal2_1000(h_raw Gemma)` iniettato nel loop al posto del testo | Elica ("Sottotitoli a cura di QTSS") |
| oracle 23 ago | MTP isolato + teacher h_t (passata unica statica) | Contenuto ma garbled |
| pred 23 ago | MTP isolato + h_sub adapter | Elica |
| N7 23 ago | adapter + cb0..7 GT + residui 8-15 predetti | ASR quasi esatto |

### Interpretazione
1. **Il MTP PUÒ generare dal solo cb0** — quando h è il `past_hidden` ricorrente del backbone in loop (formato esatto di training). Il fallimento storico era il FORMATO di h (passata statica) e lo SPAZIO (adapter non allineato, cos 0.51-0.63).
2. **Il collo è il ponte Gemma→talker**: anche l'adapter "buono" (cos 0.6304) produce elica nel loop.
3. La letteratura converge (Qwen2.5-Omni joint training; Qwen3-Omni: token per testo + hidden per espressività; PRIME-Speech: hidden-state synchronization nel loop; Freeze-Omni: retraining per LLM).

## 3. Architettura v5 (FISSATA)
```
Gemma-4-E2B (congelata) → h_raw (1536) → [PROIEZIONE, unico trainabile] → h_proj (2048)
  → backbone Qwen3-TTS in loop (congelato): codec_head → cb0, MTP → residui, protocollo nativo
  → Code2Wav
```
- Il testo di Gemma (inner monologue) resta nel prefill del talker (canale contenuto, percorso NATIVE esatto).
- h_proj sommato al codec_agg per frame (canale espressivo/condizionamento).
- Dettagli: `docs/VERDETTO_TASK12_ARCHITETTURA_V5.md`, `NEXT_STEP.md` (rivisto).

## 4. Ambiente unificato (supera ADR-003 per l'inferenza)
Gemma (unsloth, transformers 5.5.0) + teacher (qwen3tts_src patchato, 3 shim) coesistono in UN processo:
1. `@check_model_inputs()` → `@check_model_inputs` (decorator deprecato in 5.x)
2. `ROPE_INIT_FUNCTIONS["default"]` registrata manualmente
3. `talker_config.pad_token_id = None`
Container: `gemma4-omni-v5:latest`. Verifica: teacher load missing=0 unexpected=0 su cuda con unsloth importato.
→ **ADR-011** creato. Il runtime full-duplex è monoprocesso.

## 5. Trainer v5 (`tools/train_loop_coupling_v5.py`)
- Proiezione ResidualAdapter-style (skip_proj + MLP 3072 + norm), 14.2M params, warm-start 8/8 chiavi da `_archive_ckpts/temporal2/ht_adapter_temporal2_1000.pt`
- Teacher forcing nel loop: per t, `inputs = codec_agg_{t-1} + h_proj[t]`; loss = CE(cb0 da codec_head) + 0.5·CE(residui da `forward_sub_talker_finetune`)
- Eval ASR periodico: loop + MTP sample v2 dal solo cb0 GT
- Dati: cache v2 (h_raw, mmap) + parquet CV26 (codici GT + testo), finestra 64 frame, allineamento `finestra i ↔ frame [i*64, (i+1)*64) ↔ tbw[i]`

## 6. Prossimi passi
1. ✅ Documentazione (questo report, NEXT_STEP, ADR-011, Verdetto v5)
2. ⏳ Smoke test v5 (gate: loss scende, eval ASR con contenuto)
3. Training completo v5 (100k step) → `checkpoints/proj_v5_final.pt`
4. **Run 2 (ortogonale)**: DM head per turn-taking (eventi da h_raw, GT dai 16 dialoghi sintetici) — i backchannel si iniettano, non si generano
5. Runtime monoprocesso + barge-in + context window
6. Vector steering affettivo (post-full-duplex)
