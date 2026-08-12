# -*- coding: utf-8 -*-
# ============================================================================
# CELLA 1 — TRAINING FIX (v14): coupling loss pesata sui frame PARLATI
# Riparte dal ckpt FINALE ep2 (86a854e9, step 24236) con:
#   - FRAME_WEIGHT_SPEECH=5.0: tutti i frame parlati pesano >=5x (la ricetta
#     dockhardman pesava SOLO le transizioni; il parlato normale aveva peso 1.0
#     e il suo gradiente CE~7 era annegato nel silenzio CE~0.5, ~70% dei frame)
#   - lr 5e-5, max_grad_norm 1.0 (stabilità, come epoca 2)
#   - PHASE2_SMOKE_BATCHES=2-4 (se impostato) => smoke test
#   - checkpoint periodici ogni PHASE2_CKPT_EVERY (default 1000) => resume
# La LayerNorm appresa sull'h_t è GIÀ dentro ht_adapter (Linear+LayerNorm).
# ============================================================================
import unsloth  # noqa: F401  (patch in processo fresco)
import torch
import kaggle_pipeline as KP

# ---- FIX v14: amplifica il peso dei frame PARLATI nella loss ----
# La ricetta dockhardman (USE_FRAME_WEIGHTING) pesa apertura/barge-in/cessione,
# ma i frame parlati NORMALI restano a peso 1.0: la CE media (0.49) era dominata
# dal silenzio (CE~0.5 su ~70% dei frame) e il gradiente del parlato (CE~7) era
# annegato. Con FRAME_WEIGHT_SPEECH ogni frame parlato pesa >= 5x.
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
# Resume dal ckpt FINALE ep2 (cell0 lo copia in checkpoints/gemma4_duplex_sota_ckpt)
os.environ["PHASE2_RESUME_CKPT"] = "checkpoints/gemma4_duplex_sota_ckpt"
os.environ["PHASE2_START_STEP"] = os.environ.get("PHASE2_START_STEP", "24236")
os.environ["PHASE2_LR"] = os.environ.get("PHASE2_LR", "5e-5")
os.environ["PHASE2_MAX_GRAD_NORM"] = os.environ.get("PHASE2_MAX_GRAD_NORM", "1.0")
os.environ["PHASE2_RUN_LOG"] = os.environ.get("PHASE2_RUN_LOG", "data/phase2_run_log_fix14.tsv")
os.environ["PHASE2_CKPT_LOG"] = os.environ.get("PHASE2_CKPT_LOG", "data/phase2_ckpt_log_fix14.txt")
_print = {
    "RESUME": os.environ["PHASE2_RESUME_CKPT"],
    "START_STEP": os.environ["PHASE2_START_STEP"],
    "LR": os.environ["PHASE2_LR"],
    "MAX_GN": os.environ["PHASE2_MAX_GRAD_NORM"],
    "SPEECH_WEIGHT": _print_speech,
}
print(f"=== FIX TRAIN v14: {_print} ===")
sys.stdout.flush()

duplex_model, _ = KP.load_gemma4_duplex_system()
KP.run_training_pipeline(duplex_model, num_epochs=1)

print(f"=== FIX TRAIN v14 COMPLETATO === (TDD check: {_fix_seen})")
sys.stdout.flush()
