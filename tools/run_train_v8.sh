#!/bin/bash
set -e
mkdir -p /workspace/logs /workspace/checkpoints /workspace/samples_v8
echo "[V8-LAUNCH] Avvio Training V8.5 SOTA Calibrato (Step 12501 -> 25000, batch 4)..."
python tools/train_loop_coupling_v8.py \
    --smoke 0 \
    --start-step 12500 \
    --steps 12500 \
    --batch 4 \
    --lr 1.5e-5 \
    --lambda-r 0.05 \
    --lambda-m 0.20 \
    --lambda-d 0.0 \
    --eval-every 100 \
    --save-every 2500 \
    --warmstart /workspace/checkpoints/proj_v8_step12500.pt \
    --out /workspace/checkpoints \
    2>&1 | tee -a /workspace/logs/train_v8_parallel.log
echo "[V8-LAUNCH] Training V8.5 SOTA Calibrato (25000 Step) completato con successo."
echo "[V8-EVAL] Esecuzione Comparazione Audio Side-by-Side (Native Base vs Omni Projector) con Faster-Whisper..."
python tools/test_v8_teacher_forcing_audio.py 2>&1 | tee -a /workspace/logs/train_v8_parallel.log
echo "[V8-COMPLETED] Valutazione completata con successo."
