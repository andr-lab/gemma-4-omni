#!/usr/bin/env bash
# setup_3090.sh - Host provisioning and environment setup for RTX 3090 (Linux)
# Pure ASCII encoding - no non-ASCII characters, no emojis.

set -euo pipefail

echo "===================================================================="
echo "=== PROVISIONING HOST RTX 3090 FOR GEMMA 4 FULL-DUPLEX ==="
echo "===================================================================="

# 1. Verify NVIDIA Driver and nvidia-smi
echo "[1/6] Verifying NVIDIA GPU driver..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Please install NVIDIA Linux drivers (>= 535.x for CUDA 12.4)."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "NVIDIA driver verified successfully."

# 2. Verify Docker Engine
echo "[2/6] Verifying Docker installation..."
if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found. Please install Docker Engine: https://docs.docker.com/engine/install/"
    exit 1
fi
docker --version
echo "Docker verified successfully."

# 3. Verify NVIDIA Container Toolkit
echo "[3/6] Verifying NVIDIA Container Toolkit..."
if ! docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: NVIDIA Container Toolkit is not functional in Docker."
    echo "Please configure nvidia-container-toolkit:"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo nvidia-ctk runtime configure --runtime=docker"
    echo "  sudo systemctl restart docker"
    exit 1
fi
echo "NVIDIA Container Toolkit test passed."

# 4. Create Project Directory Tree
echo "[4/6] Creating workspace directory structure..."
TARGET_DIR="${HOME}/gemma-4-omni"
mkdir -p "${TARGET_DIR}"/{data,checkpoints,samples,logs,qwen_subtalker,code2wav_weights}
chmod -R 755 "${TARGET_DIR}"
echo "Directory structure created at: ${TARGET_DIR}"

# 5. Check Hugging Face Authentication
echo "[5/6] Checking Hugging Face credentials..."
if [ -n "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN environment variable detected."
elif command -v huggingface-cli &>/dev/null && huggingface-cli whoami &>/dev/null; then
    echo "Hugging Face CLI authenticated user: $(huggingface-cli whoami)"
else
    echo "Notice: HF_TOKEN is not set. For gated models, run: 'export HF_TOKEN=your_token_here'"
fi

# 6. Build Unified Docker Image
echo "[6/6] Building Docker image 'gemma4-omni:latest'..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker build -t gemma4-omni:latest -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Docker image built successfully."

echo "===================================================================="
echo "=== SETUP COMPLETE ==="
echo "===================================================================="
echo "To run a fast smoke test:"
echo "  cd ${TARGET_DIR} && ./docker_run_3090.sh smoke"
echo "Or using docker-compose:"
echo "  docker compose run --rm train-smoke"
echo "===================================================================="
