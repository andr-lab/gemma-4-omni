#!/usr/bin/env bash
# Staged data-preparation/training runner; the active exported bundle uses
# tools/run_duplex_control6_active_3090.sh for serving and end-to-end capture.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

IMAGE="${DUPLEX_QLORA_IMAGE:-gemma4-omni:latest}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-data/synthetic_it_t6_v4}"
SYNTHETIC_V3_CLEAN_ASR_ROOT="${SYNTHETIC_V3_CLEAN_ASR_ROOT:-data/synthetic_it_t6_v3}"
SMOOTHCONV_ROOT="${SMOOTHCONV_ROOT:-}"
DUPLEXCONV_ROOT="${DUPLEXCONV_ROOT:-}"
PREPARED_DIR="${PREPARED_DIR:-data/prepared_duplex_qlora_v4}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/gemma4_duplex_qlora}"
BASE_MODEL_PATH="${GEMMA_BASE_MODEL_PATH:-/workspace/.hf_cache/models--google--gemma-4-E2B-it/full}"
TARGET_TRAIN_HOURS="${TARGET_TRAIN_HOURS:-1}"
MAX_STEPS="${QLORA_MAX_STEPS:-1000}"
CONTROL_STEPS="${CONTROL_STEPS:-500}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="${PROJECT_DIR}/.tmp/duplex_qlora_runs/${RUN_ID}"
LOG_ROOT="${PROJECT_DIR}/.tmp/duplex_qlora_runs"
LOCK_PATH="${LOG_ROOT}/run.lock"

mkdir -p "${RUN_DIR}" "${LOG_ROOT}"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "A duplex QLoRA pipeline is already active (${LOCK_PATH})." >&2
    exit 2
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "Docker image not found: ${IMAGE}. Build it from the project Dockerfile first." >&2
    exit 2
fi
if [[ ! -d /usr/local/cuda ]]; then
    echo "CUDA toolkit path /usr/local/cuda is missing on the host." >&2
    exit 2
fi
CUDA_HOST_DIR="$(readlink -f /usr/local/cuda)"

COMMON_ARGS=(
    --rm
    --ipc=host
    --network bridge
    --user 1000:1000
    -v "${PROJECT_DIR}:/workspace"
    -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro"
    -e "CUDA_HOME=/usr/local/cuda"
    -e "HF_TOKEN=${HF_TOKEN:-}"
    -e "HF_HOME=/workspace/hf_cache"
    -e "HF_HUB_CACHE=/workspace/hf_cache/hub"
    -e "CUDA_VISIBLE_DEVICES=0"
    -e "PYTHONUNBUFFERED=1"
    -w /workspace
)

run_logged() {
    local name="$1"
    shift
    echo "[$name] log: ${RUN_DIR}/${name}.log"
    "$@" 2>&1 | tee "${RUN_DIR}/${name}.log"
}

run_cpu() {
    local name="$1"
    shift
    run_logged "${name}" docker run "${COMMON_ARGS[@]}" "${IMAGE}" "$@"
}

run_gpu() {
    local name="$1"
    shift
    run_logged "${name}" docker run --gpus all "${COMMON_ARGS[@]}" "${IMAGE}" "$@"
}

prepare_data() {
    local args=(
        tools/train_duplex_qlora.py prepare
        --synthetic-root "${SYNTHETIC_ROOT}"
        --synthetic-v3-clean-asr-root "${SYNTHETIC_V3_CLEAN_ASR_ROOT}"
        --output-dir "${PREPARED_DIR}"
    )
    if [[ -n "${SMOOTHCONV_ROOT}" ]]; then
        args+=(--smoothconv-root "${SMOOTHCONV_ROOT}")
    fi
    if [[ -n "${DUPLEXCONV_ROOT}" ]]; then
        args+=(--duplexconv-root "${DUPLEXCONV_ROOT}")
    fi
    run_cpu prepare "${args[@]}"
}

baseline_smoke() {
    run_gpu kernel-baseline python duplex_kernel/launch.py --workspace /workspace --smoke
    cp -f samples_kernel/kernel_smoke.wav "${RUN_DIR}/kernel_baseline.wav"
}

stage1_smoke() {
    run_gpu qlora-smoke \
        python tools/train_duplex_qlora.py train \
        --train-jsonl "${PREPARED_DIR}/train.jsonl" \
        --preparation-manifest "${PREPARED_DIR}/preparation_manifest.json" \
        --model-path "${BASE_MODEL_PATH}" \
        --adapter-dir "${CHECKPOINT_ROOT}/adapter-smoke" \
        --smoke
}

adapter_kernel_smoke() {
    local server_name="duplex-qlora-adapter-${RUN_ID//[^A-Za-z0-9_.-]/-}"
    local server_log="${RUN_DIR}/kernel-adapter-server.log"
    local server_id
    server_id="$(docker run -d \
        --name "${server_name}" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -v "${PROJECT_DIR}:/workspace" \
        -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e "CUDA_HOME=/usr/local/cuda" \
        -e "HF_HOME=/workspace/hf_cache" \
        -e "HF_HUB_CACHE=/workspace/hf_cache/hub" \
        -e "CUDA_VISIBLE_DEVICES=0" -e "PYTHONUNBUFFERED=1" \
        -w /workspace -p 8000:8000 "${IMAGE}" \
        python duplex_kernel/launch.py --workspace /workspace --host 0.0.0.0 --port 8000 \
        --gemma-qlora-adapter-dir "${CHECKPOINT_ROOT}/adapter-smoke")"
    trap 'docker rm -f "${server_name}" >/dev/null 2>&1 || true' EXIT
    echo "[kernel-adapter-smoke] server ${server_name} (${server_id}); log: ${server_log}"
    while true; do
        docker logs "${server_name}" >"${server_log}" 2>&1 || true
        if grep -q '\[launch\] load engine: OK' "${server_log}" \
            && grep -q '\[engine\] Gemma QLoRA adapter-only merged before listener pointer binding' "${server_log}" \
            && grep -Fq "${CHECKPOINT_ROOT}/adapter-smoke" "${server_log}"; then
            break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${server_name}" 2>/dev/null || echo false)" != true ]]; then
            cat "${server_log}" >&2
            return 1
        fi
        sleep 3
    done
    run_logged kernel-adapter-audio-smoke docker run --rm --ipc=host --network bridge --user 1000:1000 \
        --add-host host.docker.internal:host-gateway \
        -v "${PROJECT_DIR}:/workspace" -w /workspace "${IMAGE}" \
        python tools/smoke_duplex_qlora_audio.py \
        --server ws://host.docker.internal:8000/v1/realtime?duplex=1 \
        --output-dir "samples_kernel/adapter_smoke_${RUN_ID}"
    trap - EXIT
    docker rm -f "${server_name}" >/dev/null
}

train_stage1() {
    run_gpu qlora-train \
        python tools/train_duplex_qlora.py train \
        --train-jsonl "${PREPARED_DIR}/train.jsonl" \
        --preparation-manifest "${PREPARED_DIR}/preparation_manifest.json" \
        --model-path "${BASE_MODEL_PATH}" \
        --adapter-dir "${CHECKPOINT_ROOT}/adapter" \
        --max-steps "${MAX_STEPS}"
}

train_stage2() {
    run_gpu extract-control \
        python tools/train_duplex_qlora.py extract-control \
        --prepared-dir "${PREPARED_DIR}" \
        --base-model-path "${BASE_MODEL_PATH}" \
        --adapter-dir "${CHECKPOINT_ROOT}/adapter" \
        --output-dir "${CHECKPOINT_ROOT}/features"
    run_cpu train-control \
        python tools/train_duplex_qlora.py train-control \
        --feature-dir "${CHECKPOINT_ROOT}/features" \
        --output "${CHECKPOINT_ROOT}/control_head.pt" \
        --training-manifest "${CHECKPOINT_ROOT}/training_manifest.json" \
        --max-steps "${CONTROL_STEPS}"
}

run_e2e() {
    local server_name="duplex-qlora-${RUN_ID//[^A-Za-z0-9_.-]/-}"
    local server_log="${RUN_DIR}/kernel-server.log"
    local server_id
    local expected_run_id
    server_id="$(docker run -d \
        --name "${server_name}" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -v "${PROJECT_DIR}:/workspace" \
        -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e "CUDA_HOME=/usr/local/cuda" \
        -e "HF_TOKEN=${HF_TOKEN:-}" \
        -e "HF_HOME=/workspace/hf_cache" \
        -e "HF_HUB_CACHE=/workspace/hf_cache/hub" \
        -e "CUDA_VISIBLE_DEVICES=0" \
        -e "PYTHONUNBUFFERED=1" \
        -w /workspace -p 8000:8000 "${IMAGE}" \
        python duplex_kernel/launch.py --workspace /workspace --host 0.0.0.0 --port 8000 \
        --enable-duplex-control6 \
        --duplex-control6-adapter-dir "${CHECKPOINT_ROOT}/adapter" \
        --duplex-control6-head "${CHECKPOINT_ROOT}/control_head.pt" \
        --duplex-control6-manifest "${CHECKPOINT_ROOT}/training_manifest.json")"
    trap 'docker rm -f "${server_name}" >/dev/null 2>&1 || true' EXIT
    echo "[e2e] server container ${server_name} (${server_id}); log: ${server_log}"
    while true; do
        docker logs "${server_name}" >"${server_log}" 2>&1 || true
        if grep -q '\[launch\] load engine: OK' "${server_log}"; then
            break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${server_name}" 2>/dev/null || echo false)" != true ]]; then
            cat "${server_log}" >&2
            return 1
        fi
        sleep 3
    done
    expected_run_id="$(python3 -c 'import json; print(json.load(open("checkpoints/gemma4_duplex_qlora/training_manifest.json", encoding="utf-8"))["run_id"])')"
    run_logged e2e docker run --rm --ipc=host --network bridge --user 1000:1000 \
        --add-host host.docker.internal:host-gateway \
        -v "${PROJECT_DIR}:/workspace" -w /workspace "${IMAGE}" \
        python tools/eval_duplex_qlora_e2e.py \
        --checkpoint-root "${CHECKPOINT_ROOT}" \
        --server ws://host.docker.internal:8000/v1/realtime?duplex=1 \
        --expected-run-id "${expected_run_id}" \
        --output-dir ".tmp/duplex_qlora_runs/${RUN_ID}/e2e"
    trap - EXIT
    docker rm -f "${server_name}" >/dev/null
}

wait_for_synthetic_and_gpu() {
    local manifest="${PROJECT_DIR}/${SYNTHETIC_ROOT%/}/manifest.jsonl"
    local train_hours="0"
    local producer_pids=""
    local stable_idle=0
    local gpu_util="100"
    local gpu_memory="999999"
    echo "[wait] dataset=${SYNTHETIC_ROOT}; target_train_hours=${TARGET_TRAIN_HOURS}"
    while true; do
        if [[ -f "${manifest}" ]]; then
            train_hours="$(python3 - "${manifest}" <<'PY'
import json
import sys
total = 0.0
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if row.get("split") == "train":
                    total += float(row.get("duration_hours", 0.0))
except (OSError, ValueError, json.JSONDecodeError):
    total = 0.0
print(f"{total:.6f}")
PY
)"
        fi
        producer_pids="$(pgrep -af '[g]enerate_synthetic_it_t6.py.*synthetic_it_t6_v4' || true)"
        IFS=, read -r gpu_util gpu_memory < <(
            nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '
        )
        if [[ -z "${producer_pids}" && "${train_hours}" != "0.000000" \
            && "$(awk -v got="${train_hours}" -v need="${TARGET_TRAIN_HOURS}" 'BEGIN {print (got >= need) ? 1 : 0}')" == 1 \
            && "${gpu_util:-100}" -le 5 && "${gpu_memory:-999999}" -le 512 ]]; then
            stable_idle=$((stable_idle + 1))
        else
            stable_idle=0
        fi
        echo "[wait] train_hours=${train_hours}; producer=${producer_pids:-idle}; gpu=${gpu_util:-?}%/${gpu_memory:-?}MiB; stable_idle=${stable_idle}/2"
        if [[ ${stable_idle} -ge 2 ]]; then
            break
        fi
        sleep 60
    done
}

full_pipeline() {
    wait_for_synthetic_and_gpu
    prepare_data
    baseline_smoke
    stage1_smoke
    adapter_kernel_smoke
    train_stage1
    train_stage2
    run_e2e
    echo "[done] artifacts and logs: ${RUN_DIR}"
}

case "${1:-}" in
    wait-and-run) full_pipeline ;;
    prepare) prepare_data ;;
    baseline-smoke) baseline_smoke ;;
    qlora-smoke) stage1_smoke ;;
    adapter-smoke) adapter_kernel_smoke ;;
    train) train_stage1 ;;
    stage2) train_stage2 ;;
    e2e) run_e2e ;;
    *)
        echo "Usage: $0 {wait-and-run|prepare|baseline-smoke|qlora-smoke|adapter-smoke|train|stage2|e2e}" >&2
        echo "Set SYNTHETIC_ROOT, TARGET_TRAIN_HOURS, QLORA_MAX_STEPS, or GEMMA_BASE_MODEL_PATH to override defaults." >&2
        exit 2
        ;;
esac
