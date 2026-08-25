# ADR-011: Ambiente Unificato Gemma + Qwen3-TTS (supera ADR-003 per l'inferenza)

## Status
Accepted (2026-08-25)

## Context
ADR-003 (2-phase decoupled) risolveva il conflitto transformers (`qwen-tts` richiede 4.x, Unsloth/Gemma richiede ≥5.x) separando le fasi in processi/ambienti diversi. Ma il **runtime full-duplex** (architettura v5: Gemma produce h_raw live → proiezione → talker in loop) richiede che Gemma e teacher Qwen3-TTS coesistano **nello stesso processo** (niente IPC tra venv).

## Decision
**Ambiente unificato nel python di sistema (transformers 5.5.0 + unsloth)** con qwen3tts_src patchato per compatibilità transformers 5.x. Tre shim:

1. **Decorator `check_model_inputs`** (riga 499 di `qwen3tts_src/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`):
   `@check_model_inputs()` → `@check_model_inputs` (in transformers 5.x il decorator ha firma `(func)`, non più chiamabile con parentesi).
2. **`ROPE_INIT_FUNCTIONS["default"]`** registrata manualmente prima dell'import del modulo (in 5.x le rope si chiamano `linear/dynamic/yarn/...`; 'default' non esiste più). Pattern già usato dal trainer faithful del 23 ago (riga 293): `_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init` (funzione custom che calcola inv_freq da rope_theta).
3. **`talker_config.pad_token_id = None`** (il default che transformers 4.57 iniettava nel config).

Verifica: `teacher load missing=0 unexpected=0`, teacher su cuda OK sotto transformers 5.5.0 con unsloth importato.

## Container
`gemma4-omni-v5:latest` = `gemma4-omni:latest` + librosa/sox/onnxruntime/einops/faster-whisper + patch. Il training v5 può girare anche in venv_phase1 (solo teacher, h_raw da cache) — l'ambiente unificato serve all'INFERENZA/runtime e al debug end-to-end.

## Consequences
- Runtime full-duplex monoprocesso: `mel → Gemma (unsloth) → h_raw → proiezione → talker (stesso processo) → Code2Wav`.
- ADR-003 resta valido per il TRAINING Kaggle (fasi separate), ma NON è più un vincolo per l'inferenza.
- La patch al repo qwen3tts_src va preservata nei rebuild (sed nel Dockerfile o commit dedicato).

## Files
- `tools/_env_check3_tmp.py` — verifica ambiente unificato (teacher load sotto 5.5.0)
- `qwen3tts_src/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` (riga 499, patch decorator)
- `Dockerfile.v5` (base per l'ambiente unificato; i pacchetti si installano al primo run, poi `docker commit`)
