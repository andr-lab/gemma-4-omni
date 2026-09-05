#!/usr/bin/env bash
# docker_run_3090.sh - Execution helper for Docker container on RTX 3090 host
# Pure ASCII encoding - no non-ASCII characters, no emojis.

set -euo pipefail

IMAGE_NAME="gemma4-omni:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

echo "[docker_run_3090] Workspace: ${PROJECT_DIR}"

# 1. Build image if it does not exist
if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
    echo "[docker_run_3090] Image '${IMAGE_NAME}' not found. Building now..."
    docker build -t "${IMAGE_NAME}" -f "${PROJECT_DIR}/Dockerfile" "${PROJECT_DIR}"
fi

# 2. Check HF_TOKEN
if [ -z "${HF_TOKEN:-}" ]; then
    echo "[docker_run_3090] Warning: HF_TOKEN is not set in environment."
    echo "[docker_run_3090] Gated models (Gemma 4, Qwen3-TTS) may require 'export HF_TOKEN=your_token'."
fi

# Check if stdin is a terminal for -it vs -i
INTERACTIVE_FLAG="-i"
if [ -t 0 ]; then
    INTERACTIVE_FLAG="-it"
fi

CUDA_HOST_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || echo /usr/local/cuda)"

# 3. Base Docker arguments
# Enforcing safety: ipc=host for PyTorch dataloader, non-root user matching host, unprivileged mode, isolated network
DOCKER_BASE_ARGS=(
    --rm
    ${INTERACTIVE_FLAG}
    --gpus all
    --ipc=host
    --user "$(id -u):$(id -g)"
    -v "${PROJECT_DIR}:/workspace"
    -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro"
    -e "PIPELINE_ENV=local3090"
    -e "CUDA_HOME=/usr/local/cuda"
    -e "PATH=/usr/local/cuda/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    -e "HF_TOKEN=${HF_TOKEN:-}"
    -e "HF_HOME=/workspace/hf_cache"
    -e "HF_HUB_CACHE=/workspace/hf_cache/hub"
    -e "CUDA_VISIBLE_DEVICES=0"
    -e "PYTHONUNBUFFERED=1"
    -w /workspace
)

# 4. Command dispatch
if [ $# -eq 0 ]; then
    echo "[docker_run_3090] No arguments provided. Running default smoke training..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase train --smoke
elif [ "$1" == "smoke" ]; then
    echo "[docker_run_3090] Running Phase 2 Smoke Test..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase train --smoke
elif [ "$1" == "train" ]; then
    shift
    echo "[docker_run_3090] Running Phase 2 Full Training..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase train "$@"
elif [ "$1" == "prep" ]; then
    shift
    echo "[docker_run_3090] Running Phase 1 Data Preparation..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase prep "$@"
elif [ "$1" == "prep-smoke" ]; then
    shift
    echo "[docker_run_3090] Running Phase 1 Smoke Preparation..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase prep --smoke "$@"
elif [ "$1" == "eval" ]; then
    shift
    echo "[docker_run_3090] Running Phase 2 Evaluation..."
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py --phase eval "$@"
elif [ "$1" == "serve" ]; then
    shift
    echo "[docker_run_3090] Starting Serving API on port 8000..."
    docker run "${DOCKER_BASE_ARGS[@]}" -p 8000:8000 "${IMAGE_NAME}" python run_3090.py --phase serve --port 8000 "$@"
elif [ "$1" == "bash" ] || [ "$1" == "sh" ]; then
    if [ $# -eq 1 ]; then
        echo "[docker_run_3090] Opening interactive bash inside container..."
        docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" /bin/bash
    else
        docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" "$@"
    fi
elif [ "$1" == "python" ] || [ "$1" == "python3" ] || [ "$1" == "exec" ]; then
    if [ "$1" == "exec" ]; then
        shift
    fi
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" "$@"
else
    # Forward all custom args directly to run_3090.py
    docker run "${DOCKER_BASE_ARGS[@]}" "${IMAGE_NAME}" python run_3090.py "$@"
fi
