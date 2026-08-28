#!/bin/bash
# Avvio ufficiale vLLM-Omni Server con plugin Gemma-4-Omni Duplex (V9).
#
# Modalità (env GEMMA4_OMNI_MODE):
#   debug (default) → gemma4_omni_duplex_debug.yaml (enforce_eager: startup ~1-2 min)
#   prod            → gemma4_omni_duplex.yaml        (torch.compile + CUDA graphs ~8 min)
#
# Volume cache vLLM persistente: la compile cache (~/.cache/vllm) sopravvive ai
# riavvii → il 2° boot in prod è molto più veloce (SDD: docs.vllm.cc
# configuration/optimization → "Reuse the compile cache").
set -e
PORT="${GEMMA4_OMNI_PORT:-8091}"
MODE="${GEMMA4_OMNI_MODE:-debug}"

if [ "$MODE" = "prod" ]; then
    DEPLOY="gemma4_omni_duplex.yaml"
else
    DEPLOY="gemma4_omni_duplex_debug.yaml"
fi

echo "============================================================"
echo "  vLLM-Omni Gemma-4-Omni Duplex (V9) :${PORT} [MODE=${MODE}]"
echo "  deploy: ${DEPLOY}"
echo "============================================================"

docker stop gemma4_omni_server 2>/dev/null || true
docker rm gemma4_omni_server 2>/dev/null || true

docker run -d \
    --name gemma4_omni_server \
    --gpus all \
    --ipc=host \
    -v /home/administrator/gemma-4-omni:/workspace \
    -v /home/administrator/.cache/huggingface:/root/.cache/huggingface \
    -v /home/administrator/.cache/vllm:/root/.cache/vllm \
    -p "${PORT}:8091" \
    -w /workspace \
    -e GEMMA4_OMNI_PORT=8091 \
    -e GEMMA4_OMNI_DEPLOY_CONFIG="/workspace/vllm_omni_plugin/deploy/${DEPLOY}" \
    -e GEMMA4_OMNI_CUSTOM_VOICE_DIR=/workspace/custom_voices \
    -e DEBUG_REGISTRY=1 \
    vllm-omni-cuda:latest \
    bash -c "pip install -e /workspace/vllm_omni_plugin --no-build-isolation -q 2>&1 | tail -2 && exec python3 /workspace/launch_gemma4_vllm_omni.py"

echo "Server avviato nel container 'gemma4_omni_server' su porta ${PORT} (mode=${MODE})."
