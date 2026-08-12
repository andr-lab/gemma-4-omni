# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 2 — TRAINING FIX v14 COMPLETO (run da 9h, commit / batch session)
# Riparte dal ckpt FINALE ep2 (86a854e9, step 24236) con:
#   - FRAME_WEIGHT_SPEECH=5.0: frame parlati pesati >=5x (fix segnale v14)
#   - lr 5e-5, max_grad_norm 1.0 (stabilità epoca 2)
#   - checkpoint periodici su disco (CKPT_EVERY=1000) per resume intra-kernel
#   - UPLOAD periodico del ckpt su dataset 'andrlab/gemma4-fix14-ckpt'
#     (ogni PHASE2_UPLOAD_EVERY step): il working del commit non è scaricabile
#     (mmap ~7GB), quindi il ckpt nuovo esce SOLO via dataset. Upload
#     Kaggle->Kaggle veloce; ogni upload = nuova versione dataset (zip, il
#     Check A monterà l'ultima). L'upload è in try/except: non deve MAI
#     far crashare il training.
# ============================================================================
import unsloth  # noqa: F401  (patch in processo fresco)
import torch
import subprocess
import kaggle_pipeline as KP

# ---- FIX v14: amplifica il peso dei frame PARLATI nella loss ----
KP.FRAME_WEIGHT_SPEECH = float(os.environ.get("PHASE2_FRAME_WEIGHT_SPEECH", "5.0"))
_print_speech = KP.FRAME_WEIGHT_SPEECH
_fix_seen = {"speech_pesati": 0, "check": 0}
_orig_getitem = KP.FullDuplexDataset.__getitem__
def _getitem_speech_weighted(self, idx):
    global _fix_seen
    batch = _orig_getitem(self, idx)
    fw = batch["frame_weights"].float().clone()
    tgt = batch["audio_targets"]  # (T, 16); PAD_TARGET_ID=-100 sui frame pad
    real = (tgt != KP.PAD_TARGET_ID).all(dim=-1)   # frame reali (non pad)
    sil = (tgt == 0).all(dim=-1)                   # silenzio = tutti i cb a 0
    speech = real & (~sil)                         # frame PARLATI
    fw = torch.where(speech, torch.maximum(fw, torch.full_like(fw, KP.FRAME_WEIGHT_SPEECH)), fw)
    batch["frame_weights"] = fw
    if _fix_seen["check"] < 3:  # verifica TDD: frame parlati effettivamente pesati
        _fix_seen["check"] += 1
        _fix_seen["speech_pesati"] += int(speech.sum().item())
        _fix_seen["min_fw_speech"] = min(_fix_seen.get("min_fw_speech", 9e9),
                                         float(fw[speech].min().item()) if speech.any() else 9e9)
    return batch
KP.FullDuplexDataset.__getitem__ = _getitem_speech_weighted
print(f"[FIX v14] frame parlati pesati >= {_print_speech}x (monkeypatch __getitem__)")
sys.stdout.flush()

# ---- env training ----
os.environ["PHASE2_RESUME_CKPT"] = "checkpoints/gemma4_duplex_sota_ckpt"
os.environ["PHASE2_START_STEP"] = os.environ.get("PHASE2_START_STEP", "24236")
os.environ["PHASE2_LR"] = os.environ.get("PHASE2_LR", "5e-5")
os.environ["PHASE2_MAX_GRAD_NORM"] = os.environ.get("PHASE2_MAX_GRAD_NORM", "1.0")
os.environ["PHASE2_RUN_LOG"] = os.environ.get("PHASE2_RUN_LOG", "data/phase2_run_log_fix14.tsv")
os.environ["PHASE2_CKPT_LOG"] = os.environ.get("PHASE2_CKPT_LOG", "data/phase2_ckpt_log_fix14.txt")
os.environ.setdefault("PHASE2_CKPT_EVERY", "1000")
os.environ.setdefault("PHASE2_UPLOAD_EVERY", "2000")  # step assoluti? relativo: ogni N step
_UPLOAD_EVERY = int(os.environ["PHASE2_UPLOAD_EVERY"])
print(f"=== FIX TRAIN v14 FULL: LR={os.environ['PHASE2_LR']} GN={os.environ['PHASE2_MAX_GRAD_NORM']} "
      f"SPEECH={_print_speech} CKPT_EVERY={os.environ['PHASE2_CKPT_EVERY']} UPLOAD_EVERY={_UPLOAD_EVERY} ===")
sys.stdout.flush()

# ---- Upload periodico del ckpt su dataset Kaggle (solo ckpt, zip) ----
_UPLOAD_DIR = "fix14_ckpt_upload"
_UPLOAD_DS = "andrlab/gemma4-fix14-ckpt"
_upload_n = {"done": 0}
def _upload_ckpt(step):
    """Copia il ckpt corrente in fix14_ckpt_upload/ e fa kaggle datasets create/version.
    In try/except: un errore di upload NON deve mai fermare il training."""
    global _upload_n
    import shutil as _sh
    os.makedirs(_UPLOAD_DIR, exist_ok=True)
    _src = "checkpoints/gemma4_duplex_sota_ckpt"
    # dataset-metadata.json per il dataset
    with open(os.path.join(_UPLOAD_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as _mf:
        _mf.write('{"id": "' + _UPLOAD_DS + '", "title": "gemma4-fix14-ckpt", '
                  '"subtitle": "ckpt training fix v14 (step ' + str(step) + ')", '
                  '"licenses": [{"name": "other"}]}')
    # pulizia vecchio ckpt nella dir di upload (zip ricreato ogni volta)
    for _item in os.listdir(_UPLOAD_DIR):
        if _item != "dataset-metadata.json":
            _p = os.path.join(_UPLOAD_DIR, _item)
            if os.path.isdir(_p):
                _sh.rmtree(_p)
            else:
                os.remove(_p)
    _sh.copytree(_src, os.path.join(_UPLOAD_DIR, "gemma4_duplex_sota_ckpt"))
    # prover version; se il dataset non esiste -> create. Errore robusto.
    _r = subprocess.run(
        ["kaggle", "datasets", "version", "-p", _UPLOAD_DIR, "-m", f"fix14 step {step}", "--dir-mode", "zip"],
        capture_output=True, text=True, timeout=3600)
    _out = _r.stdout + _r.stderr
    if _r.returncode != 0 and ("not exist" in _out or "No such" in _out or "404" in _out):
        _r = subprocess.run(["kaggle", "datasets", "create", "-p", _UPLOAD_DIR],
                            capture_output=True, text=True, timeout=3600)
        _out = _r.stdout + _r.stderr
    _ok = _r.returncode == 0 and ("Dataset version" in _out or "Success" in _out or "created" in _out.lower())
    _upload_n["done"] += 1
    print(f"[FIX v14] UPLOAD ckpt @ step {step}: rc={_r.returncode} ok={_ok} "
          f"({_out[-160:]})")
    sys.stdout.flush()
    return _ok

# Wrapper del loop di training per l'upload periodico: monitoriamo il run_log.
# Il log è in formato tab: epoch, abs_step, loss_tot, coupling, text (abs_step
# INCLUDE il resume: parte da 24236+). Il primo upload avviene a start+EVERY.
_run_log = os.environ["PHASE2_RUN_LOG"]
_start_abs = int(os.environ.get("PHASE2_START_STEP", "24236"))
_last_uploaded_step = 0
_upload_target = _start_abs + _UPLOAD_EVERY  # primo upload dopo N step DAL RESUME
def _maybe_upload():
    global _last_uploaded_step, _upload_target
    if not os.path.exists(_run_log):
        return
    try:
        with open(_run_log, encoding="utf-8") as _rf:
            _lines = [l for l in _rf.read().strip().splitlines() if l.strip()]
        if not _lines:
            return
        _last = _lines[-1].split("\t")
        _step = int(_last[1]) if len(_last) > 1 else 0
        if _step >= _upload_target and _step > _last_uploaded_step:
            _last_uploaded_step = _step
            _upload_target = _step + _UPLOAD_EVERY
            _upload_ckpt(_step)
    except Exception as _e:
        print(f"[FIX v14] monitor upload: {str(_e)[:150]}")
        sys.stdout.flush()

# Upload di VERIFICA immediato (prima del training): valida che il dataset
# fix14 esista e che le credenziali kaggle nel container funzionino. Se fallisce
# il training continua (non deve bloccarsi), ma lo vediamo nel log.
try:
    _upload_ckpt(int(os.environ["PHASE2_START_STEP"]))
except Exception as _e:
    print(f"[FIX v14] upload verifica iniziale fallito: {str(_e)[:150]}")
    sys.stdout.flush()

_upload_target = _start_abs + _UPLOAD_EVERY  # riarmato dopo l'upload di verifica

# Hook leggero: il training loop scrive il run_log a ogni step; facciamo un
# thread daemon che controlla ogni 30s e lancia l'upload senza bloccare il loop.
import threading, time as _time
def _upload_monitor_loop():
    while True:
        _time.sleep(30)
        try:
            _maybe_upload()
        except Exception as _e:
            print(f"[FIX v14] upload monitor ERR: {str(_e)[:150]}")
            sys.stdout.flush()
_threading = threading.Thread(target=_upload_monitor_loop, daemon=True)
_threading.start()
print("[FIX v14] thread upload periodico avviato (ogni 30s check, upload ogni "
      f"{_UPLOAD_EVERY} step)")
sys.stdout.flush()

duplex_model, _ = KP.load_gemma4_duplex_system()
KP.run_training_pipeline(duplex_model, num_epochs=1)

# Upload finale (a fine epoca / run)
try:
    _step_final = 0
    if os.path.exists(_run_log):
        with open(_run_log, encoding="utf-8") as _rf:
            _lines = [l for l in _rf.read().strip().splitlines() if l.strip()]
        if _lines:
            _step_final = int(_lines[-1].split("\t")[1])
    _upload_ckpt(_step_final)
except Exception as _e:
    print(f"[FIX v14] upload finale fallito: {str(_e)[:150]}")
    sys.stdout.flush()

print(f"=== FIX TRAIN v14 COMPLETATO === (TDD check: {_fix_seen}, upload: {_upload_n})")
sys.stdout.flush()
