#!/bin/bash
set -e
echo "============================================================"
echo "🚀 [vLLM-Omni Setup] Installazione vLLM-Omni su RTX 3090"
echo "============================================================"

cd ~/gemma-4-omni

# 1. Clone vllm-omni repository se non esiste
if [ ! -d "vllm-omni" ]; then
    echo "Clonazione repository vllm-project/vllm-omni..."
    git clone https://github.com/vllm-project/vllm-omni.git
else
    echo "Repository vllm-omni già presente, aggiornamento..."
    cd vllm-omni && git pull || true
    cd ~/gemma-4-omni
fi

# 2. Installazione nel container gemma4-omni-v5
echo "Installazione di vllm-omni nel container di runtime..."
docker run --rm --gpus all --ipc=host \
    -v ~/gemma-4-omni:/workspace \
    -w /workspace \
    gemma4-omni-v5:latest \
    bash -c "pip install -e /workspace/vllm-omni && python3 -c 'import vllm_omni; print(\"✓ vLLM-Omni installato con successo! Versione:\", getattr(vllm_omni, \"__version__\", \"dev\"))'"

echo "============================================================"
echo "✅ Installazione vLLM-Omni completata!"
echo "============================================================"
