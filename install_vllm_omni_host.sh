#!/bin/bash
set -e
echo "============================================================"
echo "🚀 [vLLM-Omni Installation] Setup Environment via uv on RTX 3090"
echo "============================================================"

export PATH="/home/administrator/.local/bin:$PATH"
cd /home/administrator/gemma-4-omni

# 1. Creazione ambiente virtuale Python 3.12 con uv
if [ ! -d ".venv_vllm_omni" ]; then
    echo "Creazione virtual environment .venv_vllm_omni..."
    uv venv --python 3.12 .venv_vllm_omni
fi

source .venv_vllm_omni/bin/activate

# 2. Installazione vLLM 0.28.0 con backend CUDA automatico
echo "Installazione vllm==0.28.0..."
uv pip install vllm==0.28.0 --torch-backend=auto

# 3. Installazione vLLM-Omni
echo "Installazione vllm-omni..."
uv pip install vllm-omni

# 4. Installazione dipendenze aggiuntive per audio/qwen3-tts
echo "Installazione dipendenze audio..."
uv pip install soundfile einops transformers==5.14.1 safetensors

# 5. Verifica dell'installazione
echo "Verifica del runtime vLLM-Omni..."
python -c "import vllm, vllm_omni; print('✓ SUCCESS! vLLM:', vllm.__version__, '| vLLM-Omni:', getattr(vllm_omni, '__version__', 'ready'))"

echo "============================================================"
echo "✅ Installazione vLLM-Omni completata con successo!"
echo "============================================================"
