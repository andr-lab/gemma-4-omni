# CODEBASE MAP — gemma-4-omni

Aggiornata: 22 agosto 2026 (Task 11H, pulizia post-porting 3090)

## Struttura viva (root)

| File | Ruolo | Note |
|---|---|---|
| `kaggle_pipeline.py` | Pipeline principale (prep + training + sintesi) | Nato come notebook Kaggle, ora eseguito su 3090 via Docker (`PIPELINE_ENV=local3090`). I blocchi funzionali Kaggle (kaggle_secrets, path moduli, disco Kaggle) sono stati rimossi il 22/8/2026; restano SOLO riferimenti testuali storici nei commenti/print (workflow notebook originale) — non influenzano l'esecuzione |
| `code2wav_decoder.py` | Wrapper Code2Wav con 5 livelli di fallback | Il livello `official_subprocess` (venv phase1, ADR-003) è quello verificato |
| `decode_official_worker.py` | Daemon JSON-lines per il decoder ufficiale Qwen3TTSTokenizerV2Model | Gira nel container standard via `/opt/venv_phase1/bin/python` |
| `qwen_subtalker.py` | Sub-talker estratto da Qwen3-TTS | Frozen; riceve h_t dall'adapter |
| `phase2_support.py` | Utility Fase 2 | |
| `paths.py` | Risoluzione percorsi dischi | |
| `serve_duplex.py` | Serving REST FastAPI (porta 8000) | Attualmente half-duplex request/response — integrazione TurnHead/policy = prossima fase (vedi REVIEW_PIPELINE_FULLDUPLEX.md) |
| `run_3090.py` | Wrapper esecuzione unificata SSH+Docker | |
| `setup_3090.sh` / `docker_run_3090.sh` | Provisioning host ed esecuzione container | Non-root uid/gid 1000, --ipc=host |
| `train_adapter_it_3090.py` / `prep_mdc_v2_3090.py` | Trainer/prep standalone era-3090 (Task 6B/9B) | Precedenti al porting completo di kaggle_pipeline; mantenuti come riferimento |
| `check_voice_quality*.py` | Gate numerici qualità voce | Firma decoder ufficiale: corr@12.5Hz ∈ [-0.1,+0.1], sub<100Hz <2%, peak/rms 3-10 |
| `tools/build_*.py` | Generatori celle/notebook e builder dataset | `build_turnhead_v5_native.py` = GT nativo DuplexConv (Task 11H) |

## File attivi di sessione (radice, `_`-prefixed)

- `_dl_models_3090.sh` — download modelli HF sulla 3090
- `_run_agy_t11f.sh`, `_run_agy_t11g.sh` — runner agy task correnti
- `_test_worker_daemon4.py`, `_test_wrapper_e2e.py` — verifica decoder/wrapper end-to-end

## Archivio

- `_archive/kaggle_era/` — script one-off dell'era Kaggle e runner agy dei task conclusi:
  - `_run_agy/`: runner Task 2→11E (sostituiti dai prompt archiviati in `_archive/agy_prompts/`)
  - `_untracked/`: sync celle, monitor kernel, test daemon obsoleti, piani download completati
- `_archive/agy_prompts/` — TUTTI i prompt agy (storico decisionale)
- `_archive/agy_logs/` — output agy dei task chiusi

## Modifiche della pulizia (22/8/2026)

1. Rimossi da `kaggle_pipeline.py`: blocco `kaggle_secrets` (i secret ora passano SOLO via env), sys.path `/kaggle/input/...`, branch disco `/root/kaggle_big`. Comportamento locale/remoto identico (py_compile OK).
2. Archiviati ~40 script one-off (nessuno importato da moduli vivi, verificato).
3. `.pyc` de-tracciati da git (già in .gitignore).

## Regola per il futuro

Gli script one-off (_-prefixed) vanno creati direttamente sotto `_archive/<task>/`
oppure spostati lì a task concluso. I prompt agy SEMPRE in `_archive/agy_prompts/`
prima del lancio.
