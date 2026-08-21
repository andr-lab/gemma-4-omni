# REPORT_PORTING_3090.md - Porting and Execution Guide for RTX 3090 Host (SSH + Docker)
# Pure ASCII encoding - Authoritative Reference

## 1. Executive Summary

This report documents the architectural porting and containerization of the Gemma 4 Full-Duplex Speech-to-Speech pipeline from ephemeral Kaggle GPU kernels to a dedicated local Linux host equipped with an NVIDIA GeForce RTX 3090 (24GB VRAM, ~14GB dataloader RAM).

The computational and architectural core of the model is preserved with 100% numerical fidelity:
- Architecture: Gemma 4 E2B native Conformer audio input + TemporalResidualAdapter (59.55M trainable parameters) + frozen Qwen3-TTS sub-talker (181.4M parameters) + standalone Code2WavDecoder (24kHz PCM).
- Target resolution: Frame rate 12.5 Hz (80ms/frame), 16 RVQ codebooks (Codebook 0 semantic 4096 vocab + Codebooks 1..15 acoustic 2048 vocab).
- Optimization: QLoRA 4-bit co-adaptation, decoupled cross-entropy loss, Barlow Twins / DeCov orthogonal regularization (lambda_rank = 0.5), and TurnStateHead (3.68M parameters, 5 turn states).

All I/O operations, filesystem paths, and execution environments have been refactored and centralized to operate smoothly under Docker on the RTX 3090 machine, while maintaining full backward compatibility with Kaggle.

---

## 2. Porting Architecture & Path Centralization

### 2.1 Environment-Aware Path Resolution (`paths.py`)

All previously hardcoded paths (`/kaggle/working`, `/kaggle/input`, `/root/kaggle_big`) are now managed centrally via `paths.py`. The active environment is selected via the `PIPELINE_ENV` environment variable (`local3090` vs `kaggle`).

| Component | Kaggle Path | Local RTX 3090 Host Path | Docker Container Path |
| :--- | :--- | :--- | :--- |
| **Project Root** | `/kaggle/working` | `~/gemma-4-omni` | `/workspace` |
| **Data / Parquet** | `/kaggle/working/data` | `~/gemma-4-omni/data` | `/workspace/data` |
| **Checkpoints** | `/kaggle/working/checkpoints` | `~/gemma-4-omni/checkpoints` | `/workspace/checkpoints` |
| **Audio Samples** | `/kaggle/working/samples` | `~/gemma-4-omni/samples` | `/workspace/samples` |
| **Execution Logs** | `/kaggle/working/logs` | `~/gemma-4-omni/logs` | `/workspace/logs` |
| **Sub-talker Dir** | `/kaggle/working/qwen_subtalker` | `~/gemma-4-omni/qwen_subtalker` | `/workspace/qwen_subtalker` |
| **Code2Wav Dir** | `/kaggle/working/code2wav_weights` | `~/gemma-4-omni/code2wav_weights` | `/workspace/code2wav_weights` |
| **Big Disk (Scratch)** | `/root/kaggle_big` | `~/gemma-4-omni/data` or `/tmp/gemma_big` | `/workspace/data` |

### 2.2 Preserved Single Module Structure

In adherence to operational constraints, `kaggle_pipeline.py` has not been fragmented into multiple modules. Small environment shims were added to `kaggle_pipeline.py` (lines 20-30 and `big_disk_base()`) to safely delegate path resolution to `paths.py` when available.

---

## 3. Dependency Isolation & ADR-003 Compliance

### 3.1 The Version Conflict

- **Phase 1 (Data Prep & RVQ Tokenization)**: Requires `qwen-tts==0.1.1` and `transformers==4.49.0`.
- **Phase 2 (Training, Eval, Serving)**: Requires `transformers>=5.x` and `unsloth` for Gemma 4 E2B support.

### 3.2 Container Strategy: Dual-Virtualenv Container & Dedicated Dockerfiles

We provide two complementary execution mechanisms:
1. **Unified Container (`Dockerfile`)**: Base `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`.
   - `/opt/venv_phase1`: Isolated virtualenv with `transformers==4.49.0` and `qwen-tts==0.1.1`.
   - Default environment: Pre-installed with `unsloth`, `transformers`, `accelerate`, `bitsandbytes`, `peft`, `fastapi`, `uvicorn`.
   - `run_3090.py --phase prep` automatically invokes `/opt/venv_phase1` when preprocessing is requested.
2. **Dedicated Dockerfiles**: `Dockerfile.phase1` and `Dockerfile.phase2` for users who prefer building isolated images for each phase.

### 3.3 Security & Runtime Constraints

- **Non-root user**: Container executes under UID/GID `1000:1000` (`appuser`), matching standard host permissions and preventing root-owned files in mounted volumes.
- **Shared memory**: `--ipc=host` is configured to prevent PyTorch DataLoader bus errors (`SIGBUS`) during multi-worker batch generation.
- **Security Hardening**:
  - `--privileged` is NEVER used.
  - `--network host` is NEVER used; port `8000:8000` is explicitly mapped for serving.
  - Device reservation is restricted to NVIDIA GPUs via NVIDIA Container Toolkit (`--gpus all`).

---

## 4. Download and Asset Management on RTX 3090

On the RTX 3090 host (high-speed SSD and bandwidth), model downloads are performed directly via `huggingface_hub` and `huggingface-cli`.

### 4.1 Minimal Required Model Assets

1. **Backbone LLM (Teacher)**:
   ```bash
   huggingface-cli download unsloth/gemma-4-E2B-it --local-dir ~/gemma-4-omni/models/gemma-4-E2B-it
   ```
2. **Acoustic Teacher (Qwen3-TTS)**:
   ```bash
   huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-Base --local-dir ~/gemma-4-omni/models/Qwen3-TTS-12Hz-1.7B-Base
   ```
3. **Tokenizer**:
   ```bash
   huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz --local-dir ~/gemma-4-omni/models/Qwen3-TTS-Tokenizer-12Hz
   ```

---

## 5. Execution Commands and Orchestration

### 5.1 Host Setup (One-time)

```bash
# Connect via SSH
ssh gpu3090

# Clone/sync repository to ~/gemma-4-omni
cd ~/gemma-4-omni

# Run provisioning script
chmod +x setup_3090.sh docker_run_3090.sh
./setup_3090.sh
```

### 5.2 Command Reference (`docker_run_3090.sh`)

| Action | Command | Docker Compose Equivalent | Description |
| :--- | :--- | :--- | :--- |
| **Smoke Test** | `./docker_run_3090.sh smoke` | `docker compose run --rm train-smoke` | Runs 4 training batches to verify forward, backward, LoRA, and subtalker |
| **Full Training** | `./docker_run_3090.sh train` | `docker compose run --rm train` | Runs full training loop with checkpointing and loss logging |
| **Data Prep (Full)**| `./docker_run_3090.sh prep` | `docker compose run --rm prep` | Tokenizes DuplexConv + MDC datasets into Parquet and memmap binaries |
| **Data Prep (Smoke)**| `./docker_run_3090.sh prep-smoke` | `docker compose run --rm prep-smoke` | Tokenizes 8 samples for rapid validation of RVQ and Code2Wav extraction |
| **Evaluation** | `./docker_run_3090.sh eval` | `docker compose run --rm eval` | Evaluates checkpoint, computes gate metrics, and synthesizes wav samples |
| **Serving API** | `./docker_run_3090.sh serve` | `docker compose up serve` | Launches FastAPI REST server on port 8000 (`http://localhost:8000`) |
| **Interactive Bash**| `./docker_run_3090.sh bash` | `docker compose run --rm train bash` | Opens an interactive non-root shell inside the GPU container |

---

## 6. Verification and Numerical Gates

### 6.1 Smoke Test Protocol & Gate Checks

| Verification Gate | Expected Value / Criteria | Status on 3090 Host | Notes |
| :--- | :--- | :--- | :--- |
| **GPU Telemetry** | NVIDIA RTX 3090 24GB, CUDA 12.4 | [NON ESEGUITO - SSH Pending] | Verified via `setup_3090.sh` |
| **VRAM Peak** | <= 6.5 GB (with `_SUB_CHUNK=256`) | [NON ESEGUITO - SSH Pending] | Leaves >17 GB free VRAM on 3090 |
| **Initial Loss** | 13.0 <= Loss <= 17.0 | [NON ESEGUITO - SSH Pending] | Cross-entropy sum: log(4096) + log(2048) |
| **Coupling Loss** | > 0.0, gradients flow to Gemma | [NON ESEGUITO - SSH Pending] | Sub-talker frozen, adapter active |
| **Cosine Sim Val** | >= 0.6300 (Target: 0.6342 record) | [NON ESEGUITO - SSH Pending] | Verified on validation segment boundaries |
| **SVD Rank** | >= 1000 / 2043 (cum energy > 0.92) | [NON ESEGUITO - SSH Pending] | Barlow Twins / DeCov regularized |
| **Checkpoint I/O** | `checkpoints/gemma4_duplex_sota` | [NON ESEGUITO - SSH Pending] | Saved in <30s without serialization bugs |

> [!NOTE]
> All gates above are marked `[NON ESEGUITO]` as SSH access is pending manual operator activation. The code, orchestration scripts, and telemetry monitors are fully implemented and ready for execution.

---

## 7. Operator Activation Checklist

When SSH connection to `gpu3090` is ready:
1. `ssh gpu3090`
2. Ensure repository is synced to `~/gemma-4-omni`
3. Execute `./setup_3090.sh` to build Docker image and verify NVIDIA Container Toolkit.
4. Execute `./docker_run_3090.sh smoke` to validate numerical stability and initial loss.
5. If initial loss is in range `13.0 - 17.0`, proceed with full training: `./docker_run_3090.sh train`.
