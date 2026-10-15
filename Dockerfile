# Dockerfile - Unified execution container for Gemma 4 Full-Duplex on RTX 3090
# Pure ASCII encoding - Base: PyTorch 2.4.0 with CUDA 12.4 and cuDNN 9

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIPELINE_ENV=local3090
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    build-essential \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user matching host UID/GID 1000
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -m -u ${USER_ID} -g appuser -s /bin/bash appuser

# ----------------------------------------------------------------------------
# Phase 1 Isolated Virtualenv (ADR-003: transformers 4.49 + qwen-tts 0.1.1)
# ----------------------------------------------------------------------------
RUN python3 -m venv /opt/venv_phase1 && \
    /opt/venv_phase1/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv_phase1/bin/pip install --no-cache-dir \
        qwen-tts==0.1.1 \
        soundfile \
        librosa \
        datasets \
        datacollective \
        pandas \
        pyarrow \
        tqdm \
        huggingface_hub

# ----------------------------------------------------------------------------
# Phase 2 Main Environment (ADR-003: Unsloth, Gemma 4, Code2Wav, Serving)
# ----------------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 && \
    pip install --no-cache-dir \
        transformers==5.5.0 \
        git+https://github.com/unslothai/unsloth.git \
        git+https://github.com/unslothai/unsloth-zoo.git \
        accelerate \
        bitsandbytes \
        peft \
        trl \
        fastapi \
        uvicorn \
        websockets==17.1 \
        python-multipart \
        scipy==1.17.1 \
        soundfile \
        pandas \
        pyarrow \
        datasets \
        tqdm \
        huggingface_hub \
        jsonschema

# Grant read/execute permissions to appuser
RUN chown -R appuser:appuser /opt/venv_phase1

WORKDIR /workspace

# Switch to non-root user
USER appuser

# Default entrypoint
CMD ["python", "run_3090.py", "--phase", "train", "--smoke"]
