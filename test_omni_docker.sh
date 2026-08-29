#!/bin/bash
set -e
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "import vllm, vllm_omni; print('? OFFICIAL vLLM-Omni Docker Runtime Ready! vLLM:', vllm.__version__, '| vLLM-Omni:', vllm_omni.__file__)"