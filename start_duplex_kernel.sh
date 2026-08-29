#!/bin/bash
# start_duplex_kernel.sh — Avvia il custom duplex kernel (ADR-029) nel container.
#
# Il kernel è un runtime narrow (ispirato a DwarfStar/antirez-ds4) che NON usa
# vLLM: carica i checkpoint addestrati e orchestra Thinker→V9→Talker→Code2Wav
# in un processo singolo con CUDA graphs e KV persistente per sessione.
#
# Uso:
#   bash start_duplex_kernel.sh            # serve su :8092
#   bash start_duplex_kernel.sh --smoke    # solo test load + 1 sintesi + exit
set -e

PORT="${GEMMA4_OMNI_KERNEL_PORT:-8092}"
ARGS="$*"

echo "============================================================"
echo "  Gemma-4-Omni DUPLEX KERNEL (ADR-029) :${PORT}"
echo "  args: ${ARGS:-serve}"
echo "============================================================"

docker stop gemma4_omni_kernel 2>/dev/null || true
docker rm gemma4_omni_kernel 2>/dev/null || true

docker run -d \
    --name gemma4_omni_kernel \
    --gpus all \
    --ipc=host \
    -v /home/administrator/gemma-4-omni:/workspace \
    -v /home/administrator/.cache/huggingface:/root/.cache/huggingface \
    -p "${PORT}:8092" \
    -w /workspace \
    -e GEMMA4_OMNI_WORKSPACE=/workspace \
    -e PIPELINE_ENV=local3090 \
    gemma4-omni-kernel:latest \
    bash -c "python3 /workspace/duplex_kernel/launch.py --workspace /workspace --port 8092 ${ARGS}"

echo "Kernel avviato nel container 'gemma4_omni_kernel' su porta ${PORT}."
echo "Log: docker logs -f gemma4_omni_kernel"
