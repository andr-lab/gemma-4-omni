#!/bin/bash
# chain_ars.sh — training AR 100k step → generazione libera → blind ASR. Tutto lato remoto.
LOG=/workspace/logs/chain_ars.log
echo "[chain] start $(date +%H:%M:%S)" >> $LOG
docker run --rm --user 1000:1000 --ipc=host --gpus all \
  -v /home/administrator/gemma-4-omni:/workspace -w /workspace \
  -e PIPELINE_ENV=local3090 -e PIPELINE_HEAD_ARCH=v2 \
  -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache \
  gemma4-omni:latest bash -c '
    pip install --quiet librosa faster-whisper >/dev/null 2>&1
    echo "[1] TRAIN 100k" && python3 /workspace/tools/train_ar_semantic.py >> /workspace/logs/job_ars_full.log 2>&1
    echo "[2] GEN LIBERA" && python3 /workspace/tools/gen_ar_semantic.py --n 4 > /tmp/gen.log 2>&1
    grep -E "GEN\]" /tmp/gen.log | tee -a /workspace/logs/chain_ars.log
  ' >> $LOG 2>&1
echo "[chain] done $(date +%H:%M:%S)" >> $LOG
