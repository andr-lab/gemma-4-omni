import inspect
import json
import os
from pathlib import Path

expected_root = Path(
    "/workspace/artifacts/runtime/control6_runtime_pendingpcm_phaseonly_q2cachemeta_build14_stableturn_q2_context_h3000_eos_20261021"
).resolve()
expected_flag = "DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT"
expected_library = (
    "/workspace/artifacts/runtime/"
    "libgemma4_listener_microbatch_prefix512_q2cachemeta_20261021_build11_text_health.so"
)
expected_checkpoint = Path(
    "/workspace/checkpoints/"
    "native_control6_gemma_native_rows_rank16_control6_temporal100_candidate_h_3000_20261021.pt"
)

import duplex_kernel.engine as engine
import duplex_kernel.listener_microbatch_serving as serving
import duplex_kernel.listener_microbatch_worker as worker
import torch

engine_path = Path(engine.__file__).resolve()
worker_path = Path(worker.__file__).resolve()
serving_path = Path(serving.__file__).resolve()
for path in (engine_path, worker_path, serving_path):
    assert path.is_relative_to(expected_root), (str(path), str(expected_root))

assert os.environ.get("DUPLEX_LISTENER_MICROBATCH_SOURCE_ROOT") == str(expected_root).removeprefix("/workspace/")
assert os.environ.get(expected_flag) == "1", os.environ.get(expected_flag)
assert os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH") == expected_library
assert os.environ.get("DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH") == expected_library
assert os.environ.get("GEMMA4_ATTENTION_BACKEND") == "NATIVE_SPLITK_GQA"
assert os.environ.get("GEMMA4_RMSNORM_BACKEND") == "NATIVE_EXACT"
assert expected_checkpoint.is_file(), str(expected_checkpoint)

assert "live_text_eos_drained_frame_cap" in inspect.getsource(engine.PipelineEngine.talk)
factory_source = inspect.getsource(serving.make_listener_microbatch_worker)
assert "stable_turn_policy_q2_context=stable_turn_policy_q2_context" in factory_source
assert "STABLE_TURN_POLICY_Q2_CONTEXT_ENV" in factory_source
assert serving._strict_env_flag(serving.STABLE_TURN_POLICY_Q2_CONTEXT_ENV) is True
worker_default = inspect.signature(worker.ListenerMicrobatchWorker).parameters[
    "stable_turn_policy_q2_context"
].default
assert worker_default is False, worker_default

checkpoint_payload = torch.load(expected_checkpoint, map_location="cpu", weights_only=False)
assert isinstance(checkpoint_payload, dict)
phase_schema = checkpoint_payload.get("phase_schema")
assert checkpoint_payload.get("native_control6_phase_profile") == "native1540"
assert isinstance(phase_schema, dict) and phase_schema.get("profile") == "native1540"
assert phase_schema.get("width") == 4
assert tuple(checkpoint_payload["base_rows"].shape) == (6, 1536)

result = {
    "engine_source": str(engine_path),
    "engine_has_live_eos_continuation": True,
    "worker_source": str(worker_path),
    "serving_source": str(serving_path),
    "stable_flag": os.environ[expected_flag],
    "worker_default_off": worker_default is False,
    "checkpoint": str(expected_checkpoint),
    "checkpoint_keys": sorted(checkpoint_payload.keys()),
    "checkpoint_shapes": {
        str(key): list(value.shape)
        for key, value in checkpoint_payload.items()
        if isinstance(value, torch.Tensor)
    },
    "phase_schema": checkpoint_payload.get("phase_schema"),
    "native_control6_phase_profile": checkpoint_payload.get("native_control6_phase_profile"),
    "native_library": os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"),
    "wrapper_library": os.environ.get("DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH"),
    "attention_backend": os.environ.get("GEMMA4_ATTENTION_BACKEND"),
    "rmsnorm_backend": os.environ.get("GEMMA4_RMSNORM_BACKEND"),
}
print(json.dumps(result, indent=2, sort_keys=True))
