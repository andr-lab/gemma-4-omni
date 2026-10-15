#!/usr/bin/env bash
# Active Gemma QLoRA + native listener bundle only. No legacy checkpoint fallback.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

IMAGE="${DUPLEX_QLORA_IMAGE:-gemma4-omni:latest}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints/gemma4_duplex_control6_event_onset_mlp_20261001}"
SCENARIO_AUDIO_DIR="${SCENARIO_AUDIO_DIR:-artifacts/scenario_audio_qwen3tts}"
SERVER_URL="${SERVER_URL:-ws://host.docker.internal:8000/v1/realtime?duplex=1}"
PREFILL_CHUNK_TOKENS="${THINKER_PREFILL_CHUNK_TOKENS:-16}"
ASR_MODEL_PATH="${ASR_MODEL_PATH:-/workspace/hf_cache/hub/models--openai--whisper-large-v3-turbo/snapshots/41f01f3fe87f28c78e2fbf8b568835947dd65ed9}"
CONTAINER_NAME="${DUPLEX_CONTROL6_CONTAINER_NAME:-gemma-duplex-control6-active}"
RESPONSE_BUNDLE_DIR="${DUPLEX_RESPONSE_BUNDLE_DIR:-checkpoints/gemma4_duplex_response_bundle}"

usage() {
    cat <<'EOF'
Usage: tools/run_duplex_control6_active_3090.sh {serve|serve-response-bundle|candidate-audio-smoke|candidate-response-e2e|status|stop|capture|asr}

serve    Start the active duplex_control6 service in the foreground (uses GPU).
serve-response-bundle Start a self-contained adapter+head bundle with direct audio on the selected port (uses GPU).
  DUPLEX_TALKER_COMPILED_DECODE=1 opts into compiled Qwen one-token decode for this service; default 0.
  DUPLEX_TALKER_NATIVE_AUDIO_Q1=1 opts into native recurrent Qwen audio-Q1 for v2; default 0.
  DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL=1 opts into experimental dual audio/SAY V9 conditioning; default 0.
  DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL=1 opts into experimental dual audio/SAY V9 conditioning; default 0.
  DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE=1 logs one private pre-DFA action-token probe; default 0.
candidate-audio-smoke Start a short-lived direct-audio LM + compiled KV service, run the explicit prefix-cache diagnostic, then stop it (uses GPU; no duplex_control6 head).
candidate-response-e2e Start a self-contained new LM+head bundle with direct audio, capture the six real-audio scenarios, stop it, then run offline Whisper scoring (uses GPU). Set DUPLEX_NATIVE_PARTIAL_ASR=1 to also opt in to native live transcript events; default is off.
Cross-response audio hard-text KV prefix reuse is off by default. Set DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=1 only for the explicit diagnostic.
status   Show whether this runner's service container is running and its mapped port.
stop     Stop this runner's service and release the GPU.
capture  Run the six sequential scenarios and three qualitative probes (CPU client).
asr      Add offline Whisper large-v3-turbo measurements to E2E_REPORT after the server stops (uses GPU).

For the compiled Thinker, set THINKER_COMPILE_MODE=default for Inductor without CUDA graphs.
The reduce-overhead mode is retained for diagnosis but the latest probe segfaulted in decode.
TORCHINDUCTOR_COMPILE_THREADS accepts 1 or 2 (default 2) when compilation is enabled.

The checkpoint root defaults to checkpoints/gemma4_duplex_control6_event_onset_mlp_20261001.
Override with CHECKPOINT_ROOT only when selecting another complete bundle.

The direct-audio Thinker candidate is opt-in: set DUPLEX_DIRECT_AUDIO_CANDIDATE=1
for candidate-audio-smoke; the baseline serve command always keeps it off.
candidate-audio-smoke uses GEMMA_QLORA_ADAPTER_DIR (default checkpoints/gemma4_duplex_qlora/adapter)
and requires THINKER_COMPILE_MODE=default for compiled KV prefix reuse.
For serve-response-bundle, DUPLEX_TALKER_COMPILED_DECODE=1 enables the compiled Qwen Talker Q1 path;
the service warms it before readiness. Confirm /health reports talker_q1_compile_status=ready.
For v2, DUPLEX_TALKER_NATIVE_AUDIO_Q1=1 selects the native recurrent audio-Q1 path and takes
precedence over compiled decode. The service warms it before readiness and reports its status.
Synthetic TTS tensor capture is disabled by default; set DUPLEX_TTS_QUALITY_DIAGNOSTICS=1 only
for a bounded harness run that marks synthetic_dinner_dialogue_v1 in session.update.
EOF
}

die() {
    echo "[active-duplex] $*" >&2
    exit 2
}

require_image() {
    docker image inspect "${IMAGE}" >/dev/null 2>&1 \
        || die "Docker image not found: ${IMAGE}; use the existing gemma4-omni:latest image."
}

bundle_values() {
    local root_rel
    root_rel="$(realpath --relative-to="${PROJECT_DIR}" "${CHECKPOINT_ROOT}")"
    [[ "${root_rel}" != ".." && "${root_rel}" != ../* ]] \
        || die "CHECKPOINT_ROOT must stay inside the mounted repository: ${CHECKPOINT_ROOT}"
    [[ -s "${CHECKPOINT_ROOT}/training_manifest.json" ]] \
        || die "active bundle manifest missing: ${CHECKPOINT_ROOT}/training_manifest.json"
    [[ -s "${CHECKPOINT_ROOT}/control_head.pt" ]] \
        || die "active listener head missing: ${CHECKPOINT_ROOT}/control_head.pt"
    [[ -s "${CHECKPOINT_ROOT}/adapter/adapter_model.safetensors" ]] \
        || die "complete QLoRA adapter missing: ${CHECKPOINT_ROOT}/adapter/adapter_model.safetensors"

    RUN_ID="$(python3 - "${CHECKPOINT_ROOT}/training_manifest.json" "${root_rel}" <<'PY'
import json
import sys

manifest_path, root_rel = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
expected_root = "/workspace/" + root_rel
expected = {
    "schema_version": 1,
    "checkpoint_root": expected_root,
    "adapter_dir": expected_root + "/adapter",
    "control_head_path": expected_root + "/control_head.pt",
}
for key, value in expected.items():
    if manifest.get(key) != value:
        raise SystemExit(f"manifest {key}={manifest.get(key)!r}; expected {value!r}")
run_id = manifest.get("run_id")
if not isinstance(run_id, str) or not run_id:
    raise SystemExit("manifest run_id missing")
print(run_id)
PY
)"
    WORKSPACE_CHECKPOINT_ROOT="/workspace/${root_rel}"
    ADAPTER_DIR="${WORKSPACE_CHECKPOINT_ROOT}/adapter"
    CONTROL_HEAD="${WORKSPACE_CHECKPOINT_ROOT}/control_head.pt"
    MANIFEST="${WORKSPACE_CHECKPOINT_ROOT}/training_manifest.json"
}

response_bundle_values() {
    local root_rel
    root_rel="$(realpath --relative-to="${PROJECT_DIR}" "${RESPONSE_BUNDLE_DIR}")"
    [[ "${root_rel}" != ".." && "${root_rel}" != ../* ]] \
        || die "DUPLEX_RESPONSE_BUNDLE_DIR must stay inside the mounted repository: ${RESPONSE_BUNDLE_DIR}"
    [[ -s "${RESPONSE_BUNDLE_DIR}/bundle_manifest.json" ]] \
        || die "response bundle manifest missing: ${RESPONSE_BUNDLE_DIR}/bundle_manifest.json"
    RESPONSE_BUNDLE_ID="$(python3 - "${RESPONSE_BUNDLE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
manifest = json.loads((root / "bundle_manifest.json").read_text(encoding="utf-8"))
actions = ["LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN"]
feature = {
    "queries_per_tick": 4, "pooling": "last_audio_query",
    "tick_ms": 80, "source": "native_gemma_listener",
}
acoustic_phase = {
    "schema_version": 1,
    "names": [
        "user_log10_rms_current_80ms",
        "user_log10_rms_past_400ms_inclusive",
        "assistant_log10_rms_prior_80ms",
        "user_log_rms_delta_clip6_div6",
    ],
    "sample_rate_hz": 16000,
    "tick_ms": 80,
    "history_ms": 400,
    "epsilon": 1e-6,
    "log10_rms_clip": [-6.0, 0.0],
    "log_rms_delta_clip": [-6.0, 6.0],
    "log_rms_delta_divisor": 6.0,
    "history_padding": "left_zero_pad_at_stream_start_to_400ms",
    "assistant_reference": "previous_input_tick_t_minus_1_only",
    "pcm_domain": "mono_float32_normalized_pcm",
}
runtime_abi = {
    "hidden_dim": 1536, "query_index": 3, "classes": actions,
    "sample_rate_hz": 16000, "channels": 1, "tick_ms": 80,
}
if manifest.get("bundle_format") != "gemma4_duplex_response" or manifest.get("schema_version") != 1:
    raise SystemExit("unsupported response bundle schema")
bundle_id = manifest.get("bundle_id")
if not isinstance(bundle_id, str) or not bundle_id.strip():
    raise SystemExit("response bundle bundle_id missing")
if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in bundle_id):
    raise SystemExit("response bundle bundle_id may contain only letters, digits, dot, underscore and hyphen")
model, policy, backchannel = manifest.get("model"), manifest.get("policy"), manifest.get("backchannel")
if not all(isinstance(item, dict) for item in (model, policy, backchannel)):
    raise SystemExit("response bundle requires model, policy and backchannel objects")
if not isinstance(model.get("base_model_id"), str) or not model["base_model_id"].strip():
    raise SystemExit("response bundle model.base_model_id missing")
if model.get("adapter_format") != "peft" or policy.get("id") != "duplex_control6":
    raise SystemExit("response bundle adapter/policy ABI is unsupported")
feature_contract = policy.get("feature_contract")
if not isinstance(feature_contract, dict):
    raise SystemExit("response bundle feature contract is missing")
input_dim = feature_contract.get("input_dim")
if input_dim == 1536:
    expected_feature_contract = {**feature, "input_dim": 1536}
elif input_dim == 1540:
    expected_feature_contract = {
        **feature,
        "input_dim": 1540,
        "acoustic_phase": acoustic_phase,
    }
else:
    raise SystemExit("response bundle feature input_dim must be 1536 or acoustic-phase 1540")
if (policy.get("actions") != actions or feature_contract != expected_feature_contract
        or policy.get("abi") != runtime_abi):
    raise SystemExit("response bundle action/feature ABI mismatch")
feature_source = policy.get("feature_source")
if (not isinstance(feature_source, dict)
        or feature_source.get("base_model_id") != model.get("base_model_id")
        or feature_source.get("adapter_dir") != model.get("adapter_dir")):
    raise SystemExit("response bundle head feature provenance does not match its model adapter")
if backchannel.get("language") != "it":
    raise SystemExit("response bundle backchannel language must be it")

def contained(raw, label, directory=False):
    if not isinstance(raw, str) or not raw.strip() or Path(raw).is_absolute():
        raise SystemExit(f"response bundle {label} must be a relative path")
    path = (root / raw).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f"response bundle {label} escapes the bundle root")
    if directory and not path.is_dir() or not directory and not path.is_file():
        raise SystemExit(f"response bundle {label} is missing: {path}")
    return path

adapter = contained(model.get("adapter_dir"), "model.adapter_dir", True)
if not (adapter / "adapter_config.json").is_file() or not (adapter / "adapter_model.safetensors").is_file():
    raise SystemExit("response bundle PEFT adapter files are incomplete")
contained(policy.get("head_path"), "policy.head_path")
lexicon_path = contained(backchannel.get("lexicon_path"), "backchannel.lexicon_path")
lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
if (not isinstance(lexicon, dict) or lexicon.get("schema_version") != 1
        or lexicon.get("language") != "it" or lexicon.get("source_split", "train") != "train"
        or not isinstance(lexicon.get("phrases"), list) or not lexicon["phrases"]):
    raise SystemExit("response bundle backchannel lexicon is invalid")
print(bundle_id.strip())
PY
)" || die "response bundle manifest or runtime assets are invalid"
    WORKSPACE_RESPONSE_BUNDLE_DIR="/workspace/${root_rel}"
}

cuda_mount() {
    [[ -d /usr/local/cuda ]] || die "host CUDA toolkit path /usr/local/cuda is missing"
    CUDA_HOST_DIR="$(readlink -f /usr/local/cuda)"
}

serve() {
    require_image
    bundle_values
    cuda_mount
    local thinker_compile_args=()
    local torch_logs_args=()
    local compile_threads="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
    local direct_audio_candidate="${DUPLEX_DIRECT_AUDIO_CANDIDATE:-0}"
    [[ "${direct_audio_candidate}" == 0 ]] \
        || die "serve loads the legacy duplex_control6 bundle; use candidate-audio-smoke for direct audio without that head"
    case "${THINKER_COMPILE_MODE:-default}" in
        off) ;;
        default)
            [[ "${compile_threads}" == 1 || "${compile_threads}" == 2 ]] \
                || die "TORCHINDUCTOR_COMPILE_THREADS must be 1 or 2"
            thinker_compile_args+=(--thinker-compile-mode default)
            ;;
        reduce-overhead)
            [[ "${compile_threads}" == 1 || "${compile_threads}" == 2 ]] \
                || die "TORCHINDUCTOR_COMPILE_THREADS must be 1 or 2"
            echo "[active-duplex] WARNING: reduce-overhead mode segfaulted in compiled greedy decode; use default for the next comparison." >&2
            thinker_compile_args+=(--thinker-compile-mode reduce-overhead)
            ;;
        *) die "THINKER_COMPILE_MODE must be off, default, or reduce-overhead" ;;
    esac
    if [[ -n "${TORCH_LOGS:-}" ]]; then
        torch_logs_args+=( -e "TORCH_LOGS=${TORCH_LOGS}" )
    fi
    if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        die "container ${CONTAINER_NAME} already exists; run status or stop first"
    fi
    echo "[active-duplex] serving run_id=${RUN_ID}, bundle=${WORKSPACE_CHECKPOINT_ROOT}, direct_audio_candidate=0"
    docker run --rm --name "${CONTAINER_NAME}" \
        --label "org.gemma.duplex.run_id=${RUN_ID}" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -p 8000:8000 \
        -v "${PROJECT_DIR}:/workspace" -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e PIPELINE_ENV=local3090 -e CUDA_HOME=/usr/local/cuda \
        -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
        -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache/hub \
        -e TORCHINDUCTOR_CACHE_DIR=/workspace/.tmp/torchinductor_cache \
        -e TRITON_CACHE_DIR=/workspace/.tmp/triton_cache \
        -e "TORCHINDUCTOR_COMPILE_THREADS=${compile_threads}" \
        -e "DUPLEX_TRACE_ASR_TEXT=${DUPLEX_TRACE_ASR_TEXT:-0}" \
        -e DUPLEX_DIRECT_AUDIO_CANDIDATE=0 \
        "${torch_logs_args[@]}" \
        -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
        -w /workspace "${IMAGE}" /workspace/.venv_e2e_runtime/bin/python duplex_kernel/launch.py \
        --workspace /workspace --host 0.0.0.0 --port 8000 --enable-duplex-control6 \
        --thinker-prefill-chunk-tokens "${PREFILL_CHUNK_TOKENS}" \
        "${thinker_compile_args[@]}" \
        --duplex-control6-adapter-dir "${ADAPTER_DIR}" \
        --duplex-control6-manifest "${MANIFEST}" \
        --duplex-control6-head "${CONTROL_HEAD}"
}

serve_response_bundle() {
    require_image
    response_bundle_values
    cuda_mount
    local compile_threads="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
    [[ "${THINKER_COMPILE_MODE:-default}" == default ]] \
        || die "serve-response-bundle requires THINKER_COMPILE_MODE=default for compiled KV reuse"
    [[ "${compile_threads}" == 1 || "${compile_threads}" == 2 ]] \
        || die "TORCHINDUCTOR_COMPILE_THREADS must be 1 or 2"
    local response_name="${DUPLEX_RESPONSE_CONTAINER_NAME:-gemma-duplex-response-bundle}"
    local response_port="${DUPLEX_RESPONSE_PORT:-8000}"
    local prefix_reuse_enabled="${DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE:-0}"
    local talker_compiled_decode="${DUPLEX_TALKER_COMPILED_DECODE:-0}"
    local talker_native_audio_q1="${DUPLEX_TALKER_NATIVE_AUDIO_Q1:-0}"
    local v9_dual_audio_context_experimental="${DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL:-0}"
    local v9_dual_audio_context_experimental="${DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL:-0}"
    local tts_quality_diagnostics="${DUPLEX_TTS_QUALITY_DIAGNOSTICS:-0}"
    local action_logit_probe_once="${DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE:-0}"
    local qwen_native_library="${QWEN_NATIVE_LIBRARY_PATH:-/workspace/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so}"
    [[ "${prefix_reuse_enabled}" == 0 || "${prefix_reuse_enabled}" == 1 ]] \
        || die "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE must be 0 or 1"
    [[ "${talker_compiled_decode}" == 0 || "${talker_compiled_decode}" == 1 ]] \
        || die "DUPLEX_TALKER_COMPILED_DECODE must be 0 or 1"
    [[ "${talker_native_audio_q1}" == 0 || "${talker_native_audio_q1}" == 1 ]] \
        || die "DUPLEX_TALKER_NATIVE_AUDIO_Q1 must be 0 or 1"
    [[ "${v9_dual_audio_context_experimental}" == 0 || "${v9_dual_audio_context_experimental}" == 1 ]] \
        || die "DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL must be 0 or 1"
    [[ "${v9_dual_audio_context_experimental}" == 0 || "${v9_dual_audio_context_experimental}" == 1 ]] \
        || die "DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL must be 0 or 1"
    [[ "${action_logit_probe_once}" == 0 || "${action_logit_probe_once}" == 1 ]] \
        || die "DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE must be 0 or 1"
    [[ "${tts_quality_diagnostics}" == 0 || "${tts_quality_diagnostics}" == 1 ]] \
        || die "DUPLEX_TTS_QUALITY_DIAGNOSTICS must be 0 or 1"
    [[ -n "${qwen_native_library}" ]] \
        || die "QWEN_NATIVE_LIBRARY_PATH must be non-empty"
    if docker container inspect "${response_name}" >/dev/null 2>&1; then
        die "container ${response_name} already exists; inspect it before retrying"
    fi
    echo "[active-duplex] serving response bundle=${WORKSPACE_RESPONSE_BUNDLE_DIR} bundle_id=${RESPONSE_BUNDLE_ID} direct_audio=1 port=${response_port}"
    docker run --rm --name "${response_name}" \
        --label "org.gemma.duplex.bundle_id=${RESPONSE_BUNDLE_ID}" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -p "${response_port}:8000" \
        -v "${PROJECT_DIR}:/workspace" -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e PIPELINE_ENV=local3090 -e CUDA_HOME=/usr/local/cuda \
        -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
        -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache/hub \
        -e TORCHINDUCTOR_CACHE_DIR=/workspace/.tmp/torchinductor_cache \
        -e TRITON_CACHE_DIR=/workspace/.tmp/triton_cache \
        -e "TORCHINDUCTOR_COMPILE_THREADS=${compile_threads}" \
        -e DUPLEX_DIRECT_AUDIO_CANDIDATE=1 \
        -e "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=${prefix_reuse_enabled}" \
        -e "DUPLEX_TALKER_COMPILED_DECODE=${talker_compiled_decode}" \
        -e "DUPLEX_TALKER_NATIVE_AUDIO_Q1=${talker_native_audio_q1}" \
        -e "DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL=${v9_dual_audio_context_experimental}" \
        -e "DUPLEX_V9_DUAL_AUDIO_CONTEXT_EXPERIMENTAL=${v9_dual_audio_context_experimental}" \
        -e "DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE=${action_logit_probe_once}" \
        -e "DUPLEX_TTS_QUALITY_DIAGNOSTICS=${tts_quality_diagnostics}" \
        -e "QWEN_NATIVE_LIBRARY_PATH=${qwen_native_library}" \
        -e "DUPLEX_NATIVE_PARTIAL_ASR=${DUPLEX_NATIVE_PARTIAL_ASR:-0}" \
        -e "DUPLEX_TALKER_CUDA_TIMING=${DUPLEX_TALKER_CUDA_TIMING:-0}" \
        -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
        -w /workspace "${IMAGE}" /workspace/.venv_e2e_runtime/bin/python duplex_kernel/launch.py \
        --workspace /workspace --host 0.0.0.0 --port 8000 \
        --thinker-prefill-chunk-tokens "${PREFILL_CHUNK_TOKENS}" \
        --thinker-compile-mode default \
        --duplex-response-bundle-dir "${WORKSPACE_RESPONSE_BUNDLE_DIR}"
}

status() {
    if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "[active-duplex] stopped (container ${CONTAINER_NAME} not present)"
        return 0
    fi
    docker container inspect --format \
        '[active-duplex] state={{.State.Status}} running={{.State.Running}} run_id={{index .Config.Labels "org.gemma.duplex.run_id"}} ports={{json .NetworkSettings.Ports}}' \
        "${CONTAINER_NAME}"
}

candidate_audio_smoke() {
    require_image
    cuda_mount
    [[ "${THINKER_COMPILE_MODE:-default}" == default ]] \
        || die "candidate-audio-smoke requires THINKER_COMPILE_MODE=default for compiled KV prefix reuse"
    local compile_threads="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
    [[ "${compile_threads}" == 1 || "${compile_threads}" == 2 ]] \
        || die "TORCHINDUCTOR_COMPILE_THREADS must be 1 or 2"

    local adapter_input="${GEMMA_QLORA_ADAPTER_DIR:-checkpoints/gemma4_duplex_qlora/adapter}"
    local adapter_abs adapter_rel output_dir output_rel run_tag
    adapter_abs="$(realpath "${adapter_input}")"
    adapter_rel="$(realpath --relative-to="${PROJECT_DIR}" "${adapter_abs}")"
    [[ "${adapter_rel}" != ".." && "${adapter_rel}" != ../* ]] \
        || die "GEMMA_QLORA_ADAPTER_DIR must stay inside the mounted repository: ${adapter_input}"
    [[ -s "${adapter_abs}/adapter_config.json" && -s "${adapter_abs}/adapter_model.safetensors" ]] \
        || die "complete Gemma QLoRA adapter is missing at ${adapter_abs}"
    [[ -d "${SCENARIO_AUDIO_DIR}" ]] || die "scenario audio directory missing: ${SCENARIO_AUDIO_DIR}"

    local candidate_name="${DUPLEX_AUDIO_CANDIDATE_CONTAINER_NAME:-gemma-duplex-audio-candidate}"
    local server_port="${DUPLEX_AUDIO_CANDIDATE_PORT:-8000}"
    local startup_timeout="${SERVE_START_TIMEOUT_SECONDS:-900}"
    candidate_container_id=""
    if docker container inspect "${candidate_name}" >/dev/null 2>&1; then
        die "candidate container ${candidate_name} already exists; inspect it before retrying"
    fi

    run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="${OUTPUT_DIR:-.tmp/duplex_audio_candidate/${run_tag}}"
    output_rel="$(realpath -m --relative-to="${PROJECT_DIR}" "${output_dir}")"
    [[ "${output_rel}" != ".." && "${output_rel}" != ../* ]] \
        || die "OUTPUT_DIR must stay inside the mounted repository: ${output_dir}"
    mkdir -p "${output_dir}"

    cleanup_candidate() {
        if [[ -n "${candidate_container_id:-}" ]] \
            && docker container inspect "${candidate_container_id}" >/dev/null 2>&1; then
            if [[ -n "${output_dir:-}" ]]; then
                docker logs "${candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
            fi
            docker stop --time "${STOP_TIMEOUT_SECONDS:-30}" "${candidate_container_id}" >/dev/null || true
        fi
    }
    trap cleanup_candidate EXIT

    echo "[active-duplex] starting direct-audio candidate (no duplex_control6 head), adapter=/workspace/${adapter_rel}"
    candidate_container_id="$(docker run -d --rm --name "${candidate_name}" \
        --label "org.gemma.duplex.candidate=direct-audio-lm-cache" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -p "${server_port}:8000" \
        -v "${PROJECT_DIR}:/workspace" -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e PIPELINE_ENV=local3090 -e CUDA_HOME=/usr/local/cuda \
        -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
        -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache/hub \
        -e TORCHINDUCTOR_CACHE_DIR=/workspace/.tmp/torchinductor_cache \
        -e TRITON_CACHE_DIR=/workspace/.tmp/triton_cache \
        -e "TORCHINDUCTOR_COMPILE_THREADS=${compile_threads}" \
        -e DUPLEX_DIRECT_AUDIO_CANDIDATE=1 \
        -e DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=1 \
        -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
        -w /workspace "${IMAGE}" /workspace/.venv_e2e_runtime/bin/python duplex_kernel/launch.py \
        --workspace /workspace --host 0.0.0.0 --port 8000 \
        --thinker-prefill-chunk-tokens "${PREFILL_CHUNK_TOKENS}" \
        --thinker-compile-mode default \
        --gemma-qlora-adapter-dir "/workspace/${adapter_rel}")"

    local deadline=$((SECONDS + startup_timeout))
    local logs=""
    while (( SECONDS < deadline )); do
        if ! docker container inspect --format '{{.State.Running}}' "${candidate_container_id}" 2>/dev/null | grep -qx true; then
            docker logs "${candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
            die "direct-audio candidate exited before becoming ready; see ${output_rel}/candidate_server.log"
        fi
        logs="$(docker logs "${candidate_container_id}" 2>&1 || true)"
        if [[ "${logs}" == *"[launch] load engine: OK"* ]]; then
            break
        fi
        sleep 2
    done
    [[ "${logs}" == *"[launch] load engine: OK"* ]] || {
        docker logs "${candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
        die "candidate startup timed out after ${startup_timeout}s; see ${output_rel}/candidate_server.log"
    }

    local server_url="ws://host.docker.internal:${server_port}/v1/realtime?duplex=1"
    echo "[active-duplex] running two-turn direct-audio and text-prefix-cache smoke"
    docker run --rm --user 1000:1000 --ipc=host --network bridge \
        --add-host=host.docker.internal:host-gateway \
        -v "${PROJECT_DIR}:/workspace" -e PYTHONPATH=/workspace -w /workspace "${IMAGE}" \
        /workspace/.venv_e2e_runtime/bin/python tools/smoke_duplex_qlora_audio.py \
        --server "${server_url}" --scenario-audio-dir "/workspace/${SCENARIO_AUDIO_DIR}" \
        --output-dir "/workspace/${output_rel}/client" --require-cache-hit

    docker logs "${candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
    echo "[active-duplex] candidate report: ${output_rel}/client/report.json"
    echo "[active-duplex] server log: ${output_rel}/candidate_server.log"
    cleanup_candidate
    candidate_container_id=""
    trap - EXIT
}

candidate_response_e2e() {
    require_image
    response_bundle_values
    cuda_mount
    [[ "${THINKER_COMPILE_MODE:-default}" == default ]] \
        || die "candidate-response-e2e requires THINKER_COMPILE_MODE=default for compiled KV reuse"
    local compile_threads="${TORCHINDUCTOR_COMPILE_THREADS:-2}"
    [[ "${compile_threads}" == 1 || "${compile_threads}" == 2 ]] \
        || die "TORCHINDUCTOR_COMPILE_THREADS must be 1 or 2"
    [[ -d "${SCENARIO_AUDIO_DIR}" ]] || die "scenario audio directory missing: ${SCENARIO_AUDIO_DIR}"

    local candidate_name="${DUPLEX_RESPONSE_CANDIDATE_CONTAINER_NAME:-gemma-duplex-response-e2e-candidate}"
    local server_port="${DUPLEX_RESPONSE_CANDIDATE_PORT:-8124}"
    local partial_asr_enabled="${DUPLEX_NATIVE_PARTIAL_ASR:-0}"
    [[ "${partial_asr_enabled}" == 0 || "${partial_asr_enabled}" == 1 ]] \
        || die "DUPLEX_NATIVE_PARTIAL_ASR must be 0 or 1"
    local prefix_reuse_enabled="${DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE:-0}"
    [[ "${prefix_reuse_enabled}" == 0 || "${prefix_reuse_enabled}" == 1 ]] \
        || die "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE must be 0 or 1"
    local transcription_args=()
    if [[ "${partial_asr_enabled}" == 1 ]]; then
        transcription_args+=(--input-audio-transcription)
    fi
    local startup_timeout="${SERVE_START_TIMEOUT_SECONDS:-900}"
    local run_tag output_dir output_rel report_rel
    response_candidate_container_id=""
    if docker container inspect "${candidate_name}" >/dev/null 2>&1; then
        die "candidate container ${candidate_name} already exists; inspect it before retrying"
    fi
    run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="${OUTPUT_DIR:-.tmp/duplex_response_bundle_runs/${RESPONSE_BUNDLE_ID}/e2e_${run_tag}}"
    output_rel="$(realpath -m --relative-to="${PROJECT_DIR}" "${output_dir}")"
    [[ "${output_rel}" != ".." && "${output_rel}" != ../* ]] \
        || die "OUTPUT_DIR must stay inside the mounted repository: ${output_dir}"
    mkdir -p "${output_dir}"

    cleanup_response_candidate() {
        if [[ -n "${response_candidate_container_id:-}" ]] \
            && docker container inspect "${response_candidate_container_id}" >/dev/null 2>&1; then
            docker logs "${response_candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
            docker stop --time "${STOP_TIMEOUT_SECONDS:-30}" "${response_candidate_container_id}" >/dev/null || true
        fi
    }
    trap cleanup_response_candidate EXIT

    echo "[active-duplex] starting complete response bundle candidate id=${RESPONSE_BUNDLE_ID} direct_audio=1 port=${server_port}"
    response_candidate_container_id="$(docker run -d --rm --name "${candidate_name}" \
        --label "org.gemma.duplex.candidate=response-bundle-e2e" \
        --label "org.gemma.duplex.bundle_id=${RESPONSE_BUNDLE_ID}" \
        --gpus all --ipc=host --network bridge --user 1000:1000 \
        -p "${server_port}:8000" \
        -v "${PROJECT_DIR}:/workspace" -v "${CUDA_HOST_DIR}:/usr/local/cuda:ro" \
        -e PIPELINE_ENV=local3090 -e CUDA_HOME=/usr/local/cuda \
        -e "LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}" \
        -e HF_HOME=/workspace/.hf_cache -e HF_HUB_CACHE=/workspace/.hf_cache/hub \
        -e TORCHINDUCTOR_CACHE_DIR=/workspace/.tmp/torchinductor_cache \
        -e TRITON_CACHE_DIR=/workspace/.tmp/triton_cache \
        -e "TORCHINDUCTOR_COMPILE_THREADS=${compile_threads}" \
        -e DUPLEX_DIRECT_AUDIO_CANDIDATE=1 \
        -e "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=${prefix_reuse_enabled}" \
        -e "DUPLEX_NATIVE_PARTIAL_ASR=${partial_asr_enabled}" \
        -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -e PYTHONUNBUFFERED=1 \
        -w /workspace "${IMAGE}" /workspace/.venv_e2e_runtime/bin/python duplex_kernel/launch.py \
        --workspace /workspace --host 0.0.0.0 --port 8000 \
        --thinker-prefill-chunk-tokens "${PREFILL_CHUNK_TOKENS}" \
        --thinker-compile-mode default \
        --duplex-response-bundle-dir "${WORKSPACE_RESPONSE_BUNDLE_DIR}")"

    local deadline=$((SECONDS + startup_timeout))
    local logs=""
    while (( SECONDS < deadline )); do
        if ! docker container inspect --format '{{.State.Running}}' "${response_candidate_container_id}" 2>/dev/null | grep -qx true; then
            docker logs "${response_candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
            die "response bundle candidate exited before readiness; see ${output_rel}/candidate_server.log"
        fi
        logs="$(docker logs "${response_candidate_container_id}" 2>&1 || true)"
        if [[ "${logs}" == *"[launch] load engine: OK"* ]]; then
            break
        fi
        sleep 2
    done
    [[ "${logs}" == *"[launch] load engine: OK"* ]] || {
        docker logs "${response_candidate_container_id}" >"${output_dir}/candidate_server.log" 2>&1 || true
        die "response bundle candidate startup timed out after ${startup_timeout}s; see ${output_rel}/candidate_server.log"
    }

    local server_url="ws://host.docker.internal:${server_port}/v1/realtime?duplex=1"
    local capture_status=0
    echo "[active-duplex] capturing six canonical real-audio scenarios plus qualitative correction probe"
    if docker run --rm --user 1000:1000 --ipc=host --network bridge \
        --add-host=host.docker.internal:host-gateway \
        -v "${PROJECT_DIR}:/workspace" -e PYTHONPATH=/workspace -w /workspace "${IMAGE}" \
        /workspace/.venv_e2e_runtime/bin/python tools/capture_duplex_qlora_sequential.py \
        --server "${server_url}" --checkpoint-root "${WORKSPACE_RESPONSE_BUNDLE_DIR}" \
        --turn-policy duplex_control6 --expected-bundle-id "${RESPONSE_BUNDLE_ID}" \
        --scenario-audio-dir "/workspace/${SCENARIO_AUDIO_DIR}" \
        "${transcription_args[@]}" \
        --include-qualitative-prompts --output-dir "/workspace/${output_rel}"; then
        capture_status=0
    else
        capture_status=$?
        echo "[active-duplex] capture returned ${capture_status}; keeping the partial report for diagnosis"
    fi
    report_rel="${output_rel}/report.json"
    cleanup_response_candidate
    response_candidate_container_id=""
    trap - EXIT

    if [[ -s "${output_dir}/report.json" ]]; then
        echo "[active-duplex] candidate stopped; running offline Whisper scoring on captured input/output WAVs"
        docker run --rm --gpus all --user 1000:1000 --ipc=host --network bridge \
            -v "${PROJECT_DIR}:/workspace" \
            -e HF_HOME=/workspace/hf_cache -e HF_HUB_CACHE=/workspace/hf_cache/hub \
            -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -w /workspace "${IMAGE}" \
            /workspace/.venv_e2e_runtime/bin/python tools/eval_duplex_qlora_e2e.py \
            --offline-asr-report "/workspace/${report_rel}" \
            --asr-model-path "${ASR_MODEL_PATH}" --asr-backend transformers --asr-device cuda
    else
        echo "[active-duplex] capture report was not produced; inspect ${output_rel}/candidate_server.log" >&2
        return 2
    fi
    echo "[active-duplex] response bundle report: ${report_rel}"
    return "${capture_status}"
}

stop() {
    if ! docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "[active-duplex] already stopped (container ${CONTAINER_NAME} not present)"
        return 0
    fi
    if [[ "$(docker container inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" == true ]]; then
        docker stop --time "${STOP_TIMEOUT_SECONDS:-30}" "${CONTAINER_NAME}"
        echo "[active-duplex] stopped; GPU resources released"
    else
        echo "[active-duplex] container exists but is already stopped; state: $(docker container inspect --format '{{.State.Status}}' "${CONTAINER_NAME}")"
    fi
}

capture() {
    require_image
    bundle_values
    [[ -d "${SCENARIO_AUDIO_DIR}" ]] || die "scenario audio directory missing: ${SCENARIO_AUDIO_DIR}"
    local run_tag output_dir output_rel
    run_tag="$(date -u +%Y%m%dT%H%M%SZ)"
    output_dir="${OUTPUT_DIR:-.tmp/duplex_qlora_runs/${RUN_ID}/e2e_${run_tag}}"
    output_rel="$(realpath --relative-to="${PROJECT_DIR}" "${output_dir}")"
    [[ "${output_rel}" != ".." && "${output_rel}" != ../* ]] \
        || die "OUTPUT_DIR must stay inside the mounted repository: ${output_dir}"
    mkdir -p "${output_dir}"
    echo "[active-duplex] capture run_id=${RUN_ID}, output=${output_dir}"
    docker run --rm --user 1000:1000 --ipc=host --network bridge \
        --add-host=host.docker.internal:host-gateway \
        -v "${PROJECT_DIR}:/workspace" -w /workspace "${IMAGE}" \
        /workspace/.venv_e2e_runtime/bin/python tools/capture_duplex_qlora_sequential.py \
        --server "${SERVER_URL}" --checkpoint-root "${WORKSPACE_CHECKPOINT_ROOT}" \
        --turn-policy duplex_control6 --expected-run-id "${RUN_ID}" \
        --scenario-audio-dir "/workspace/${SCENARIO_AUDIO_DIR}" \
        --include-qualitative-prompts --output-dir "/workspace/${output_rel}"
    echo "[active-duplex] report: ${output_rel}/report.json"
}

offline_asr() {
    require_image
    bundle_values
    [[ -n "${E2E_REPORT:-}" ]] || die "set E2E_REPORT to the capture report.json path"
    local report_rel
    report_rel="$(realpath --relative-to="${PROJECT_DIR}" "${E2E_REPORT}")"
    [[ "${report_rel}" != ".." && "${report_rel}" != ../* ]] \
        || die "E2E_REPORT must stay inside the mounted repository: ${E2E_REPORT}"
    [[ -s "${E2E_REPORT}" ]] || die "E2E report missing: ${E2E_REPORT}"
    docker run --rm --gpus all --user 1000:1000 --ipc=host --network bridge \
        -v "${PROJECT_DIR}:/workspace" \
        -e HF_HOME=/workspace/hf_cache -e HF_HUB_CACHE=/workspace/hf_cache/hub \
        -e HF_HUB_OFFLINE=1 -e CUDA_VISIBLE_DEVICES=0 -w /workspace "${IMAGE}" \
        /workspace/.venv_e2e_runtime/bin/python tools/eval_duplex_qlora_e2e.py \
        --offline-asr-report "/workspace/${report_rel}" \
        --asr-model-path "${ASR_MODEL_PATH}" --asr-backend transformers --asr-device cuda
}

case "${1:-}" in
    serve) serve ;;
    serve-response-bundle) serve_response_bundle ;;
    candidate-audio-smoke) candidate_audio_smoke ;;
    candidate-response-e2e) candidate_response_e2e ;;
    status) status ;;
    stop) stop ;;
    capture) capture ;;
    asr) offline_asr ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
esac
