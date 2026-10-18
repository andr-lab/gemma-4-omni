import importlib.util
import runpy
import sys

import transformers

overlay = "/workspace/.tmp/listener_batch_source_overlay"
print(f"bootstrap_transformers={transformers.__file__}", flush=True)
sys.path.insert(0, "/workspace")
package_root = "/workspace/duplex_kernel"
overlay_package = f"{overlay}/duplex_kernel"
spec = importlib.util.spec_from_file_location(
    "duplex_kernel",
    f"{package_root}/__init__.py",
    submodule_search_locations=[overlay_package, package_root],
)
if spec is None or spec.loader is None:
    raise RuntimeError("could not create overlay-first duplex_kernel package")
duplex_kernel = importlib.util.module_from_spec(spec)
sys.modules["duplex_kernel"] = duplex_kernel
spec.loader.exec_module(duplex_kernel)
sys.path.append(f"{overlay}/runtimepackages_original")

import duplex_kernel.listener
import duplex_kernel.listener_microbatch_worker
import duplex_kernel.native_listener_microbatch
import duplex_kernel.scheduler
import duplex_kernel.server

for module in (
    duplex_kernel.listener,
    duplex_kernel.listener_microbatch_worker,
    duplex_kernel.native_listener_microbatch,
    duplex_kernel.scheduler,
    duplex_kernel.server,
):
    module_path = str(module.__file__)
    print(f"overlay_module={module.__name__}:{module_path}", flush=True)
    if not module_path.startswith(overlay_package + "/"):
        raise RuntimeError(f"{module.__name__} did not load from the source overlay")

sys.argv = [
    "launch.py",
    "--workspace", "/workspace",
    "--host", "127.0.0.1",
    "--port", "8092",
    "--frame-residual-adapter-checkpoint",
    "/workspace/checkpoints/frame_residual_adapter_emovo_query_film_v2_candidate.pt",
    "--ablate-response-lora",
    "--controller-guided-talk-context-pool",
    "--enforce-v9-training-voice",
]
runpy.run_path(
    f"{overlay}/duplex_kernel/launch.py",
    run_name="__main__",
)
