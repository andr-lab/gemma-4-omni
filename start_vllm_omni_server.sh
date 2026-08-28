#!/bin/bash
# Avvio ufficiale vLLM-Omni Server con plugin Gemma-4-Omni Duplex (V9).
set -e
PORT="${GEMMA4_OMNI_PORT:-8091}"

echo "============================================================"
echo "  Avvio vLLM-Omni Server Gemma-4-Omni Duplex (V9) :${PORT}"
echo "============================================================"

docker stop gemma4_omni_server 2>/dev/null || true
docker rm gemma4_omni_server 2>/dev/null || true

docker run -d \
    --name gemma4_omni_server \
    --gpus all \
    --ipc=host \
    -v /home/administrator/gemma-4-omni:/workspace \
    -v /home/administrator/.cache/huggingface:/root/.cache/huggingface \
    -p "${PORT}:8091" \
    -w /workspace \
    -e GEMMA4_OMNI_PORT=8091 \
    vllm-omni-cuda:latest \
    bash -c "pip install -e /workspace/vllm_omni_plugin --no-build-isolation -q 2>&1 | tail -2 && exec python3 /workspace/launch_gemma4_vllm_omni.py"

echo "Server avviato nel container 'gemma4_omni_server' su porta ${PORT}."
