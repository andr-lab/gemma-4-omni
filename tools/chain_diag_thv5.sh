#!/bin/bash
# chain_diag_thv5.sh — attende la fine del training TurnHead v5 (run 2), poi lancia
# la diagnostica ibrida. Tutto lato remoto sotto nohup: immune a disconnessioni ssh.
LOG=/tmp/chain_diag.log
echo "[chain] start $(date +%H:%M:%S)" >> $LOG
for i in $(seq 1 520); do
  N=$(docker ps --format '{{.Names}}' | grep -vE 'portainer' | head -1)
  if [ -z "$N" ]; then echo "[chain] GPU libera $(date +%H:%M:%S)" >> $LOG; break; fi
  sleep 10
done
docker run --rm --user 1000:1000 --ipc=host --gpus all \
  -v /home/administrator/gemma-4-omni:/workspace -w /workspace \
  -e PIPELINE_ENV=local3090 -e PIPELINE_HEAD_ARCH=v2 \
  -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache \
  gemma4-omni:latest bash -c 'pip install --quiet faster-whisper >/dev/null 2>&1; python3 /workspace/tools/diag_hybrid.py > /tmp/diag_hybrid.log 2>&1; echo EXIT=$?'
echo "[chain] diag exit=$(tail -1 /tmp/diag_hybrid.log) $(date +%H:%M:%S)" >> $LOG
