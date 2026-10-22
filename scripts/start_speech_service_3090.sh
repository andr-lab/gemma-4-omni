#!/usr/bin/env bash
set -euo pipefail

# Canonical experimental speech service on gpu3090. This script starts or
# restarts only this named container; it never removes a container.
readonly CONTAINER="${DUPLEX_SERVICE_CONTAINER:-gemma4-listener-control6-h3000-eos-build14-20261021}"
readonly IMAGE="${DUPLEX_SERVICE_IMAGE:-gemma4-listener-control6-baseline-snapshot-20261014:latest}"
readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HEALTH_TIMEOUT_S="${DUPLEX_HEALTH_TIMEOUT_S:-600}"
readonly PYTHON="/workspace/.venv_e2e_runtime/bin/python"
readonly UNIFIED_KV="${DUPLEX_UNIFIED_KV:-1}"
case "$UNIFIED_KV" in
  0|1) ;;
  *) echo "speech service: DUPLEX_UNIFIED_KV must be 0 or 1" >&2; exit 1 ;;
esac
if [[ "$UNIFIED_KV" == "1" ]]; then
  if [[ ${DUPLEX_AUDIO_RESPONSE_PROTOCOL+x} ]]; then
    RESPONSE_PROTOCOL="$DUPLEX_AUDIO_RESPONSE_PROTOCOL"
  else
    case "${DUPLEX_CONTROLLER_GUIDED_SAY_V1:-1}" in
      1) RESPONSE_PROTOCOL="controller_guided_say_v1" ;;
      0) RESPONSE_PROTOCOL="heard_action_say_v2" ;;
      *) echo "speech service: DUPLEX_CONTROLLER_GUIDED_SAY_V1 must be 0 or 1" >&2; exit 1 ;;
    esac
  fi
elif [[ ${DUPLEX_AUDIO_RESPONSE_PROTOCOL+x} ]]; then
  RESPONSE_PROTOCOL="$DUPLEX_AUDIO_RESPONSE_PROTOCOL"
else
  case "${DUPLEX_CONTROLLER_GUIDED_SAY_V1:-1}" in
    1) RESPONSE_PROTOCOL="controller_guided_say_v1" ;;
    0) RESPONSE_PROTOCOL="heard_action_say_v2" ;;
    *) echo "speech service: DUPLEX_CONTROLLER_GUIDED_SAY_V1 must be 0 or 1" >&2; exit 1 ;;
  esac
fi
case "$RESPONSE_PROTOCOL" in
  controller_guided_say_v1|heard_action_say_v2) ;;
  *) echo "speech service: DUPLEX_AUDIO_RESPONSE_PROTOCOL must be controller_guided_say_v1 or heard_action_say_v2" >&2; exit 1 ;;
esac
readonly RESPONSE_PROTOCOL
readonly LISTENER_MICROBATCH="${DUPLEX_LISTENER_MICROBATCH:-1}"
LISTENER_MICROBATCH_SEMANTIC_SHADOW="${DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW:-1}"
if [[ "$UNIFIED_KV" == "1" ]]; then
  [[ "$LISTENER_MICROBATCH" == "1" ]] || {
    echo "speech service: DUPLEX_UNIFIED_KV=1 requires DUPLEX_LISTENER_MICROBATCH=1" >&2
    exit 1
  }
  # Unified mode has one native cache owner; the legacy semantic shadow is off.
  LISTENER_MICROBATCH_SEMANTIC_SHADOW=0
fi
readonly LISTENER_MICROBATCH_SEMANTIC_SHADOW
readonly LISTENER_MICROBATCH_GEMM_GRAPHS="${DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS:-0}"
readonly LISTENER_SOURCE_ROOT_RELATIVE="${DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT:-artifacts/runtime/control6_runtime_pendingpcm_phaseonly_q2cachemeta_build14_stableturn_q2_context_h3000_eos_20261021}"
readonly NATIVE_CONTROL6_TRACE="${DUPLEX_NATIVE_CONTROL6_TRACE:-0}"
readonly NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY="${DUPLEX_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY:-}"
readonly STABLE_TURN_POLICY_Q2_CONTEXT="${DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT:-1}"
readonly NATIVE_LISTENER_CUDA_GRAPH="${GEMMA4_NATIVE_LISTENER_CUDA_GRAPH:-0}"
readonly ALLOW_CUSTOM_EAGER_DIAGNOSTIC="${GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC:-0}"
readonly ATTENTION_BACKEND="${GEMMA4_ATTENTION_BACKEND:-NATIVE_SPLITK_GQA}"
readonly SPLITK_LAYER_MASK="${GEMMA4_SPLITK_LAYER_MASK:-STATE_SAFE_HYBRID}"
readonly INPUT_STREAM_PRIORITY="${GEMMA4_DUPLEX_INPUT_STREAM_PRIORITY:-default}"
readonly CUDA_TIMING="${GEMMA4_DUPLEX_CUDA_TIMING:-0}"
readonly NATIVE_Q4_GEMV4="${GEMMA4_NATIVE_Q4_GEMV4:-0}"
case "$LISTENER_MICROBATCH" in
  0|1) ;;
  *) echo "speech service: DUPLEX_LISTENER_MICROBATCH must be 0 or 1" >&2; exit 1 ;;
esac
case "$LISTENER_MICROBATCH_SEMANTIC_SHADOW" in
  0|1) ;;
  *) echo "speech service: DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW must be 0 or 1" >&2; exit 1 ;;
esac
case "$LISTENER_MICROBATCH_GEMM_GRAPHS" in
  0|1) ;;
  *) echo "speech service: DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS must be 0 or 1" >&2; exit 1 ;;
esac
case "$NATIVE_CONTROL6_TRACE" in
  0|1) ;;
  *) echo "speech service: DUPLEX_NATIVE_CONTROL6_TRACE must be 0 or 1" >&2; exit 1 ;;
esac
case "$STABLE_TURN_POLICY_Q2_CONTEXT" in
  0|1) ;;
  *) echo "speech service: DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT must be 0 or 1" >&2; exit 1 ;;
esac
case "$NATIVE_LISTENER_CUDA_GRAPH" in
  0|1) ;;
  *) echo "speech service: GEMMA4_NATIVE_LISTENER_CUDA_GRAPH must be 0 or 1" >&2; exit 1 ;;
esac
case "$ALLOW_CUSTOM_EAGER_DIAGNOSTIC" in
  0|1) ;;
  *) echo "speech service: GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC must be 0 or 1" >&2; exit 1 ;;
esac
readonly LISTENER_SILENCE_SKIP="${GEMMA4_OMNI_LISTENER_SILENCE_SKIP:-1}"
if [[ "$INPUT_STREAM_PRIORITY" != "default" && "$INPUT_STREAM_PRIORITY" != "high" ]]; then
  echo "speech service: GEMMA4_DUPLEX_INPUT_STREAM_PRIORITY must be default or high" >&2; exit 1
fi
case "$CUDA_TIMING" in
  0|1) ;;
  *) echo "speech service: GEMMA4_DUPLEX_CUDA_TIMING must be 0 or 1" >&2; exit 1 ;;
esac
case "$NATIVE_Q4_GEMV4" in
  0|1) ;;
  *) echo "speech service: GEMMA4_NATIVE_Q4_GEMV4 must be 0 or 1" >&2; exit 1 ;;
esac
case "$LISTENER_SILENCE_SKIP" in
  0|1) ;;
  *) echo "speech service: GEMMA4_OMNI_LISTENER_SILENCE_SKIP must be 0 or 1" >&2; exit 1 ;;
esac
readonly NATIVE_CONTROL6_SEMANTIC_PREFIX="${DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX:-1}"
case "$NATIVE_CONTROL6_SEMANTIC_PREFIX" in
  1)
    DEFAULT_LISTENER_LIB_RELATIVE_PATH="artifacts/runtime/libgemma4_listener_microbatch_prefix512_q2cachemeta_20261021_build11_text_health.so"
    DEFAULT_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH="checkpoints/native_control6_gemma_native_rows_rank16_control6_temporal100_candidate_h_3000_20261021.pt"
    DEFAULT_PREFIX_SCHEMA_VERSION="2"
    DEFAULT_PREFIX_CONTRACT="gemma_native_control6_semantic_prefix_boa_v2"
    ;;
  0)
    # Explicit rollback only: pair the legacy 34-token listener with its
    # pre-semantic native Control6 checkpoint.
    DEFAULT_LISTENER_LIB_RELATIVE_PATH="artifacts/runtime/libgemma4_listener.so"
    DEFAULT_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH="checkpoints/native_control6_gemma_native_rows_rank16_600.pt"
    DEFAULT_PREFIX_SCHEMA_VERSION="0"
    DEFAULT_PREFIX_CONTRACT="disabled"
    ;;
  *)
    echo "speech service: DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX must be 0 or 1" >&2
    exit 1
    ;;
esac

readonly QWEN_LIB_HOST="$REPO_ROOT/artifacts/runtime/libqwen_talker_abi4.so"
readonly LISTENER_LIB_RELATIVE_PATH="${GEMMA4_NATIVE_LISTENER_LIBRARY_RELATIVE_PATH:-$DEFAULT_LISTENER_LIB_RELATIVE_PATH}"
readonly LISTENER_LIB_HOST="$REPO_ROOT/$LISTENER_LIB_RELATIVE_PATH"
readonly LISTENER_LIB_CONTAINER="/workspace/$LISTENER_LIB_RELATIVE_PATH"
readonly MICROBATCH_LIB_CONTAINER="$LISTENER_LIB_CONTAINER"
readonly PREFIX_SCHEMA_VERSION="${DUPLEX_NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION:-$DEFAULT_PREFIX_SCHEMA_VERSION}"
readonly PREFIX_CONTRACT="${DUPLEX_NATIVE_CONTROL6_PREFIX_CONTRACT:-$DEFAULT_PREFIX_CONTRACT}"
readonly ADAPTER_HOST="$REPO_ROOT/checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt"
readonly ADAPTER_RUNTIME_HOST="$REPO_ROOT/artifacts/runtime/frame_residual_adapter_emovo_query_film_v2_candidate.pt"
readonly ENABLE_NATIVE_CONTROL6="${DUPLEX_ENABLE_GEMMA_NATIVE_CONTROL6:-1}"
readonly NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH="${DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH:-$DEFAULT_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH}"
readonly NATIVE_CONTROL6_ADAPTER_HOST="$REPO_ROOT/$NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH"
readonly QWEN_LIB_CONTAINER="/workspace/artifacts/runtime/libqwen_talker_abi4.so"
readonly LISTENER_SOURCE_ROOT_HOST="$REPO_ROOT/$LISTENER_SOURCE_ROOT_RELATIVE"
readonly LISTENER_SOURCE_ROOT_CONTAINER="/workspace/$LISTENER_SOURCE_ROOT_RELATIVE"
readonly LISTENER_PYTHONPATH="$LISTENER_SOURCE_ROOT_CONTAINER:$LISTENER_SOURCE_ROOT_CONTAINER/runtimepackages_original:/workspace"
readonly ADAPTER_CONTAINER="/workspace/checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt"
readonly NATIVE_CONTROL6_ADAPTER_CONTAINER="/workspace/$NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH"

NATIVE_CONTROL6_ARGS=()
if [[ "$ENABLE_NATIVE_CONTROL6" == "1" ]]; then
  NATIVE_CONTROL6_ARGS=(
    --enable-gemma-native-control6
    --native-control6-adapter "$NATIVE_CONTROL6_ADAPTER_CONTAINER"
  )
elif [[ "$ENABLE_NATIVE_CONTROL6" != "0" ]]; then
  echo "speech service: DUPLEX_ENABLE_GEMMA_NATIVE_CONTROL6 must be 0 or 1" >&2
  exit 1
fi

fail() { echo "speech service: $*" >&2; exit 1; }

if [[ -n "${DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH:-}" && \
      "$DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH" != "$MICROBATCH_LIB_CONTAINER" ]]; then
  fail "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH must match GEMMA4_NATIVE_LISTENER_LIBRARY_PATH ($MICROBATCH_LIB_CONTAINER)"
fi

if [[ "$NATIVE_LISTENER_CUDA_GRAPH" == "1" ]]; then
  [[ "$ATTENTION_BACKEND" == "CUSTOM" ]] || fail "CUDA graph listener requires CUSTOM attention"
  [[ "$ALLOW_CUSTOM_EAGER_DIAGNOSTIC" == "0" ]] || fail "CUSTOM eager diagnostic cannot be combined with CUDA graphs"
elif [[ "$ALLOW_CUSTOM_EAGER_DIAGNOSTIC" == "1" ]]; then
  [[ "$ATTENTION_BACKEND" == "CUSTOM" ]] || fail "CUSTOM eager diagnostic requires CUSTOM attention"
else
  [[ "$ATTENTION_BACKEND" == "NATIVE_SPLITK_GQA" ]] || fail "graph-off production listener requires NATIVE_SPLITK_GQA"
fi

[[ "$NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH" != /* ]] || fail "DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH must stay inside the repository"
[[ "/$NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH/" != *"/../"* ]] || fail "DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH must not traverse outside checkpoints"
[[ "$NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH" == checkpoints/* ]] || fail "native Control6 checkpoint must be under checkpoints/"
[[ "$LISTENER_LIB_RELATIVE_PATH" != /* ]] || fail "GEMMA4_NATIVE_LISTENER_LIBRARY_RELATIVE_PATH must be repository-relative"
[[ "$LISTENER_LIB_RELATIVE_PATH" == artifacts/runtime/* ]] || fail "listener library must be under artifacts/runtime/"
[[ "/$LISTENER_LIB_RELATIVE_PATH/" != *"/../"* ]] || fail "listener library path must not traverse outside artifacts/runtime/"
[[ "$LISTENER_SOURCE_ROOT_RELATIVE" != /* ]] || fail "DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT must be repository-relative"
[[ "$LISTENER_SOURCE_ROOT_RELATIVE" == artifacts/runtime/* ]] || fail "listener source root must be under artifacts/runtime/"
[[ "/$LISTENER_SOURCE_ROOT_RELATIVE/" != *"/../"* ]] || fail "listener source root must not traverse outside artifacts/runtime/"
[[ "$LISTENER_SOURCE_ROOT_RELATIVE" != *\\* ]] || fail "listener source root must use repository-relative POSIX separators"
[[ "$LISTENER_SOURCE_ROOT_RELATIVE" != *:* ]] || fail "listener source root must not contain PYTHONPATH separators"

[[ "$UNIFIED_KV" == "1" || "$NATIVE_CONTROL6_SEMANTIC_PREFIX" != "1" || "$LISTENER_MICROBATCH" != "1" || "$LISTENER_MICROBATCH_SEMANTIC_SHADOW" == "1" ]] || \
  fail "semantic-prefix Control6 microbatch requires DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW=1"
[[ "$LISTENER_MICROBATCH" != "1" || "$LISTENER_MICROBATCH_GEMM_GRAPHS" == "0" ]] || \
  fail "production listener microbatch requires DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS=0"
[[ "$NATIVE_CONTROL6_SEMANTIC_PREFIX" != "0" || "$LISTENER_MICROBATCH" != "1" ]] || \
  fail "the legacy listener DSO requires DUPLEX_LISTENER_MICROBATCH=0"

for file in "$QWEN_LIB_HOST" "$LISTENER_LIB_HOST" "$ADAPTER_HOST" "$ADAPTER_RUNTIME_HOST"; do
  [[ -f "$file" ]] || fail "required runtime artifact is missing: $file"
done
RUNTIME_HOST_REAL="$(realpath -e "$REPO_ROOT/artifacts/runtime")" || fail "artifacts/runtime is missing"
LISTENER_LIB_REAL="$(realpath -e "$LISTENER_LIB_HOST")" || fail "listener library does not resolve inside artifacts/runtime"
[[ "$LISTENER_LIB_REAL" == "$RUNTIME_HOST_REAL/"* ]] || fail "listener library resolves outside artifacts/runtime"
[[ -d "$LISTENER_SOURCE_ROOT_HOST" ]] || fail "persistent listener source root is missing: $LISTENER_SOURCE_ROOT_HOST"
LISTENER_SOURCE_ROOT_REAL="$(realpath -e "$LISTENER_SOURCE_ROOT_HOST")" || fail "persistent listener source root does not resolve"
[[ "$LISTENER_SOURCE_ROOT_REAL" == "$RUNTIME_HOST_REAL/"* ]] || fail "listener source root resolves outside artifacts/runtime"
for relative_module in \
  duplex_kernel/__init__.py \
  duplex_kernel/launch.py \
  duplex_kernel/server.py \
  duplex_kernel/gemma_representation.py \
  duplex_kernel/gemma_native_control6_batch_policy.py \
  duplex_kernel/gemma_native_control6_policy.py \
  duplex_kernel/listener.py \
  duplex_kernel/listener_microbatch.py \
  duplex_kernel/listener_microbatch_serving.py \
  duplex_kernel/listener_microbatch_worker.py \
  duplex_kernel/listener_microbatch_frontend.py \
  duplex_kernel/native_listener.py \
  duplex_kernel/native_listener_microbatch.py \
  duplex_kernel/microbatch_contract.py \
  duplex_kernel/shared_listener_scheduler.py \
  duplex_kernel/native_control6_runtime.py \
  duplex_kernel/turn_policy.py \
  runtimepackages_original/qwen_tts/__init__.py; do
  [[ -f "$LISTENER_SOURCE_ROOT_HOST/$relative_module" ]] || fail "required persistent listener source module is missing: $LISTENER_SOURCE_ROOT_HOST/$relative_module"
done
cmp -s "$ADAPTER_HOST" "$ADAPTER_RUNTIME_HOST" || fail "runtime adapter copy differs from the approved Query FiLM v2 checkpoint"
if [[ "$ENABLE_NATIVE_CONTROL6" == "1" ]]; then
  [[ -f "$NATIVE_CONTROL6_ADAPTER_HOST" ]] || fail "native Control6 adapter is missing: $NATIVE_CONTROL6_ADAPTER_HOST"
  echo "native Control6 checkpoint: $NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH"
fi
if [[ "$NATIVE_CONTROL6_SEMANTIC_PREFIX" == "1" ]]; then
  [[ "$ENABLE_NATIVE_CONTROL6" == "1" ]] || fail "semantic-prefix mode requires native Control6"
  [[ "$PREFIX_SCHEMA_VERSION" == "2" ]] || fail "semantic-prefix schema version must be 2"
  [[ "$PREFIX_CONTRACT" == "gemma_native_control6_semantic_prefix_boa_v2" ]] || fail "semantic-prefix contract mismatch"
fi
[[ -d /usr/local/cuda-13.1 ]] || fail "expected CUDA mount source /usr/local/cuda-13.1 is missing"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "serving image is not present locally: $IMAGE"

check_env() {
  local key="$1" expected="$2" actual
  actual="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" | awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}')"
  # Existing legacy containers predate the explicit unified toggle; absence is
  # equivalent to the documented opt-out, but never to an opt-in.
  if [[ "$key" == "DUPLEX_UNIFIED_KV" && "$expected" == "0" && -z "$actual" ]]; then
    return
  fi
  if [[ "$key" == "DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT" && "$expected" == "0" && -z "$actual" ]]; then
    return
  fi
  [[ "$actual" == "$expected" ]] || fail "existing $CONTAINER has unexpected $key; refusing to mutate it"
}

check_response_protocol_env() {
  local protocol_env legacy_toggle
  protocol_env="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" | awk -F= '$1 == "DUPLEX_AUDIO_RESPONSE_PROTOCOL" {sub(/^[^=]*=/, ""); print; exit}')"
  if [[ -n "$protocol_env" ]]; then
    [[ "$protocol_env" == "$RESPONSE_PROTOCOL" ]] || fail "existing $CONTAINER has unexpected DUPLEX_AUDIO_RESPONSE_PROTOCOL; refusing to mutate it"
    return
  fi
  if [[ "$UNIFIED_KV" == "0" && ( "$RESPONSE_PROTOCOL" == "controller_guided_say_v1" || "$RESPONSE_PROTOCOL" == "heard_action_say_v2" ) ]]; then
    legacy_toggle="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$CONTAINER" | awk -F= '$1 == "DUPLEX_CONTROLLER_GUIDED_SAY_V1" {sub(/^[^=]*=/, ""); print; exit}')"
    if [[ "$RESPONSE_PROTOCOL" == "controller_guided_say_v1" ]]; then
      [[ "$legacy_toggle" == "1" ]] || fail "existing $CONTAINER has no matching audio-response protocol setting; refusing to mutate it"
    else
      [[ "$legacy_toggle" == "0" ]] || fail "existing $CONTAINER has no matching audio-response protocol setting; refusing to mutate it"
    fi
    return
  fi
  fail "existing $CONTAINER has no explicit DUPLEX_AUDIO_RESPONSE_PROTOCOL; refusing to mutate it"
}

check_existing_container() {
  local image user ipc port mounts cmd workdir
  image="$(docker inspect --format '{{.Config.Image}}' "$CONTAINER")"
  user="$(docker inspect --format '{{.Config.User}}' "$CONTAINER")"
  ipc="$(docker inspect --format '{{.HostConfig.IpcMode}}' "$CONTAINER")"
  port="$(docker inspect --format '{{with index .HostConfig.PortBindings "8000/tcp"}}{{(index . 0).HostPort}}{{end}}' "$CONTAINER" 2>/dev/null || true)"
  mounts="$(docker inspect --format '{{range .Mounts}}{{println .Source "=>" .Destination}}{{end}}' "$CONTAINER")"
  cmd="$(docker inspect --format '{{json .Config.Cmd}}' "$CONTAINER")"
  workdir="$(docker inspect --format '{{.Config.WorkingDir}}' "$CONTAINER")"
  [[ "$image" == "$IMAGE" ]] || fail "existing $CONTAINER uses a different image; refusing to mutate it"
  [[ "$user" == "1000:1000" && "$ipc" == "host" && "$port" == "8000" ]] || fail "existing $CONTAINER does not match uid/ipc/port contract"
  [[ "$mounts" == *"$REPO_ROOT => /workspace"* && "$mounts" == *"/usr/local/cuda-13.1 => /usr/local/cuda"* ]] || fail "existing $CONTAINER mounts do not match this checkout/CUDA runtime"
  [[ "$workdir" == "$LISTENER_SOURCE_ROOT_CONTAINER" ]] || fail "existing $CONTAINER does not use the persistent listener source as its working directory"
  [[ "$cmd" == *"$LISTENER_SOURCE_ROOT_CONTAINER/duplex_kernel/launch.py"* ]] || fail "existing $CONTAINER does not use the persistent listener source entrypoint"
  for arg in "$ADAPTER_CONTAINER" --frame-residual-adapter-checkpoint --ablate-response-lora --controller-guided-talk-context-pool --enforce-v9-training-voice; do
    [[ "$cmd" == *"$arg"* ]] || fail "existing $CONTAINER is missing launch argument: $arg"
  done
  if [[ "$ENABLE_NATIVE_CONTROL6" == "1" ]]; then
    for arg in --enable-gemma-native-control6 --native-control6-adapter "$NATIVE_CONTROL6_ADAPTER_CONTAINER"; do
      [[ "$cmd" == *"$arg"* ]] || fail "existing $CONTAINER is missing launch argument: $arg"
    done
  elif [[ "$cmd" == *"--enable-gemma-native-control6"* ]]; then
    fail "existing $CONTAINER enables native Control6 but this launch requested it disabled"
  fi
  check_env DUPLEX_ENABLE_GEMMA_NATIVE_CONTROL6 "$ENABLE_NATIVE_CONTROL6"
  if [[ "$NATIVE_CONTROL6_TRACE" == "1" ]]; then
    check_env DUPLEX_NATIVE_CONTROL6_TRACE 1
  fi
  check_env DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT "$STABLE_TURN_POLICY_Q2_CONTEXT"
  check_env GEMMA4_OMNI_LISTENER_SILENCE_SKIP "$LISTENER_SILENCE_SKIP"
  check_env GEMMA4_OMNI_FRAME_RESIDUAL_ADAPTER_CKPT "$ADAPTER_CONTAINER"
  check_env GEMMA4_OMNI_DEFAULT_VOICE gemma
  check_env GEMMA4_OMNI_CUSTOM_VOICE_DIR /workspace/custom_voices
  check_env GEMMA4_OMNI_WORKSPACE /workspace
  check_env DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX "$NATIVE_CONTROL6_SEMANTIC_PREFIX"
  check_env DUPLEX_NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION "$PREFIX_SCHEMA_VERSION"
  check_env DUPLEX_NATIVE_CONTROL6_PREFIX_CONTRACT "$PREFIX_CONTRACT"
  check_env DUPLEX_DIRECT_AUDIO_CANDIDATE 1
  check_env DUPLEX_UNIFIED_KV "$UNIFIED_KV"
  check_response_protocol_env
  check_env GEMMA4_OMNI_SPEAKER_REFERENCE_WAV /workspace/artifacts/voices/reference.wav
  check_env GEMMA4_OMNI_SPEAKER_REFERENCE_SHA256 480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c
  check_env DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU 1
  check_env DUPLEX_TALKER_NATIVE_AUDIO_Q1 1
  check_env DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH 1
  check_env QWEN_NATIVE_LIBRARY_PATH "$QWEN_LIB_CONTAINER"
  check_env GEMMA4_NATIVE_LISTENER_CUDA_GRAPH "$NATIVE_LISTENER_CUDA_GRAPH"
  check_env GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC "$ALLOW_CUSTOM_EAGER_DIAGNOSTIC"
  check_env GEMMA4_NATIVE_LISTENER_LIBRARY_PATH "$LISTENER_LIB_CONTAINER"
  check_env DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH "$MICROBATCH_LIB_CONTAINER"
  check_env DUPLEX_LISTENER_MICROBATCH "$LISTENER_MICROBATCH"
  check_env DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW "$LISTENER_MICROBATCH_SEMANTIC_SHADOW"
  check_env DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS "$LISTENER_MICROBATCH_GEMM_GRAPHS"
  check_env DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT "$LISTENER_SOURCE_ROOT_RELATIVE"
  check_env PYTHONPATH "$LISTENER_PYTHONPATH"
  check_env GEMMA4_ATTENTION_BACKEND "$ATTENTION_BACKEND"
  check_env GEMMA4_SPLITK_LAYER_MASK "$SPLITK_LAYER_MASK"
  check_env GEMMA4_RMSNORM_BACKEND NATIVE_EXACT
  check_env GEMMA4_DUPLEX_INPUT_STREAM_PRIORITY "$INPUT_STREAM_PRIORITY"
  check_env GEMMA4_DUPLEX_CUDA_TIMING "$CUDA_TIMING"
  check_env GEMMA4_NATIVE_Q4_GEMV4 "$NATIVE_Q4_GEMV4"
}

if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  check_existing_container
  state="$(docker inspect --format '{{.State.Status}}' "$CONTAINER")"
  if [[ "$state" == running ]]; then
    docker restart --time 20 "$CONTAINER" >/dev/null
  elif [[ "$state" == exited || "$state" == created ]]; then
    docker start "$CONTAINER" >/dev/null
  else
    fail "container is in state '$state'; inspect it manually"
  fi
else
  port_users="$(docker ps --filter publish=8000 --format '{{.Names}}')"
  [[ -z "$port_users" ]] || fail "host port 8000 is already published by: $port_users; refusing to stop or remove it"

  docker run --detach \
    --name "$CONTAINER" \
    --label io.gemma4-omni.role=speech-runtime \
    --label io.gemma4-omni.managed-by=start_speech_service_3090.sh \
    --restart unless-stopped \
    --gpus all \
    --user 1000:1000 \
    --ipc=host \
    --publish 8000:8000 \
    --volume "$REPO_ROOT:/workspace" \
    --volume /usr/local/cuda-13.1:/usr/local/cuda:ro \
    --workdir "$LISTENER_SOURCE_ROOT_CONTAINER" \
    --env DUPLEX_DIRECT_AUDIO_CANDIDATE=1 \
    --env DUPLEX_UNIFIED_KV="$UNIFIED_KV" \
    --env DUPLEX_AUDIO_RESPONSE_PROTOCOL="$RESPONSE_PROTOCOL" \
    --env GEMMA4_OMNI_FRAME_RESIDUAL_ADAPTER_CKPT="$ADAPTER_CONTAINER" \
    --env GEMMA4_OMNI_DEFAULT_VOICE=gemma \
    --env GEMMA4_OMNI_CUSTOM_VOICE_DIR=/workspace/custom_voices \
    --env GEMMA4_OMNI_WORKSPACE=/workspace \
    --env GEMMA4_OMNI_SPEAKER_REFERENCE_WAV=/workspace/artifacts/voices/reference.wav \
    --env GEMMA4_OMNI_SPEAKER_REFERENCE_SHA256=480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c \
    --env DUPLEX_ENABLE_GEMMA_NATIVE_CONTROL6="$ENABLE_NATIVE_CONTROL6" \
    --env DUPLEX_NATIVE_CONTROL6_TRACE="$NATIVE_CONTROL6_TRACE" \
    --env DUPLEX_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY="$NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY" \
    --env DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT="$STABLE_TURN_POLICY_Q2_CONTEXT" \
    --env GEMMA4_OMNI_LISTENER_SILENCE_SKIP="$LISTENER_SILENCE_SKIP" \
    --env DUPLEX_NATIVE_CONTROL6_SEMANTIC_PREFIX="$NATIVE_CONTROL6_SEMANTIC_PREFIX" \
    --env DUPLEX_NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION="$PREFIX_SCHEMA_VERSION" \
    --env DUPLEX_NATIVE_CONTROL6_PREFIX_CONTRACT="$PREFIX_CONTRACT" \
    --env DUPLEX_AUDIO_HISTORY_ARCHIVE_CPU=1 \
    --env DUPLEX_TALKER_NATIVE_AUDIO_Q1=1 \
    --env DUPLEX_TALKER_NATIVE_AUDIO_Q1_GRAPH=1 \
    --env QWEN_NATIVE_LIBRARY_PATH="$QWEN_LIB_CONTAINER" \
    --env GEMMA4_NATIVE_LISTENER_CUDA_GRAPH="$NATIVE_LISTENER_CUDA_GRAPH" \
    --env GEMMA4_NATIVE_LISTENER_ALLOW_CUSTOM_EAGER_DIAGNOSTIC="$ALLOW_CUSTOM_EAGER_DIAGNOSTIC" \
    --env GEMMA4_NATIVE_LISTENER_LIBRARY_PATH="$LISTENER_LIB_CONTAINER" \
    --env DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH="$MICROBATCH_LIB_CONTAINER" \
    --env DUPLEX_LISTENER_MICROBATCH="$LISTENER_MICROBATCH" \
    --env DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW="$LISTENER_MICROBATCH_SEMANTIC_SHADOW" \
    --env DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS="$LISTENER_MICROBATCH_GEMM_GRAPHS" \
    --env DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT="$LISTENER_SOURCE_ROOT_RELATIVE" \
    --env PYTHONPATH="$LISTENER_PYTHONPATH" \
    --env GEMMA4_ATTENTION_BACKEND="$ATTENTION_BACKEND" \
    --env GEMMA4_SPLITK_LAYER_MASK="$SPLITK_LAYER_MASK" \
    --env GEMMA4_RMSNORM_BACKEND=NATIVE_EXACT \
    --env GEMMA4_DUPLEX_INPUT_STREAM_PRIORITY="$INPUT_STREAM_PRIORITY" \
    --env GEMMA4_DUPLEX_CUDA_TIMING="$CUDA_TIMING" \
    --env GEMMA4_NATIVE_Q4_GEMV4="$NATIVE_Q4_GEMV4" \
    "$IMAGE" \
    "$PYTHON" "$LISTENER_SOURCE_ROOT_CONTAINER/duplex_kernel/launch.py" \
    --workspace /workspace --host 0.0.0.0 --port 8000 \
    --thinker-prefill-chunk-tokens 16 --thinker-compile-mode default \
    --duplex-response-bundle-dir /workspace/checkpoints/experimental-v5-repair200-fphase-20261003 \
    --runtime-backend native \
    --frame-residual-adapter-checkpoint "$ADAPTER_CONTAINER" \
    --ablate-response-lora \
    --controller-guided-talk-context-pool \
    --enforce-v9-training-voice \
    "${NATIVE_CONTROL6_ARGS[@]}"
fi

for ((elapsed=0; elapsed<HEALTH_TIMEOUT_S; elapsed++)); do
  if body="$(curl --connect-timeout 1 --max-time 2 -fsS http://127.0.0.1:8000/health 2>/dev/null)"; then
    if printf '%s' "$body" | python3 -c '
import json, sys
d = json.load(sys.stdin)
requested = sys.argv[1] == "1"
response_protocol = sys.argv[2]
unified_kv = sys.argv[3] == "1"
microbatch = d.get("listener_microbatch")
ok = (
    d.get("status") == "ok"
    and d.get("engine_loaded") is True
    and d.get("talker_q1_backend") == "native_audio_q1"
    and d.get("talker_q1_native_status") == "ready"
    and d.get("response_protocol") == response_protocol
    and d.get("unified_kv_enabled") is unified_kv
    and isinstance(microbatch, dict)
    and microbatch.get("enabled") is requested
)
if requested:
    ok = ok and microbatch.get("semantic_shadow_setting") == ("0" if unified_kv else "1")
    ok = ok and microbatch.get("gemm_graphs_setting") == "0"
sys.exit(0 if ok else 1)
' "$LISTENER_MICROBATCH" "$RESPONSE_PROTOCOL" "$UNIFIED_KV" 2>/dev/null; then
      printf '%s\n' "$body"
      exit 0
    fi
  fi
  sleep 1
done

echo "speech service did not reach engine/talker readiness within ${HEALTH_TIMEOUT_S}s" >&2
docker logs --tail 100 "$CONTAINER" >&2 || true
exit 5
