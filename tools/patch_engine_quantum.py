import re
from pathlib import Path

path = Path("duplex_kernel/engine.py")
content = path.read_text(encoding="utf-8")

# 1. Update admit_listener to check not callable(listener_compute_quantum)
target_admit = '''        def admit_listener(phase: str, site: str) -> None:
            if (
                prioritized_v2_talker
                and phase == "talker"
                and site in {"codec_head", "subtalker", "next_qwen_model"}
            ):
                return'''

replacement_admit = '''        def admit_listener(phase: str, site: str) -> None:
            if (
                prioritized_v2_talker
                and not callable(listener_compute_quantum)
                and phase == "talker"
                and site in {"codec_head", "subtalker", "next_qwen_model"}
            ):
                return'''

assert target_admit in content, "target_admit not found"
content = content.replace(target_admit, replacement_admit, 1)

# 2. Update run_talker_gpu_step to acquire quantum permit and record completion
target_step = '''        def run_talker_gpu_step(site: str, forward: Callable[[], Any]) -> Any:
            if not priority_lock_talker:
                return forward()
            wait_started = time.perf_counter()
            self._listener_priority_lock()
            acquired = time.perf_counter()
            wait_ms = (acquired - wait_started) * 1000.0
            if metrics is not None:
                wait_key = f"listener_priority_{site}_lock_wait_ms"
                wait_max_key = f"listener_priority_{site}_lock_wait_max_ms"
                metrics[wait_key] = metrics.get(wait_key, 0.0) + wait_ms
                metrics[wait_max_key] = max(
                    metrics.get(wait_max_key, 0.0), wait_ms
                )
            try:
                return forward()
            finally:'''

replacement_step = '''        def run_talker_gpu_step(site: str, forward: Callable[[], Any]) -> Any:
            if not priority_lock_talker:
                return forward()
            wait_started = time.perf_counter()
            self._listener_priority_lock()
            acquired = time.perf_counter()
            wait_ms = (acquired - wait_started) * 1000.0
            if metrics is not None:
                wait_key = f"listener_priority_{site}_lock_wait_ms"
                wait_max_key = f"listener_priority_{site}_lock_wait_max_ms"
                metrics[wait_key] = metrics.get(wait_key, 0.0) + wait_ms
                metrics[wait_max_key] = max(
                    metrics.get(wait_max_key, 0.0), wait_ms
                )
            quantum_ctx = (
                listener_compute_quantum()
                if callable(listener_compute_quantum)
                else nullcontext(None)
            )
            try:
                with quantum_ctx as permit:
                    result = forward()
                    record_completion = getattr(self, "_record_talker_cuda_completion", None)
                    if callable(record_completion) and permit is not None and permit is not False:
                        record_completion(permit)
                    return result
            finally:'''

assert target_step in content, "target_step not found"
content = content.replace(target_step, replacement_step, 1)

# 3. Add admit_listener for code2wav_decode
target_decode = '''                    if profiler is None:
                        return run_talker_gpu_step("code2wav_decode", invoke)'''

replacement_decode = '''                    admit_listener("talker", "code2wav_decode")
                    if profiler is None:
                        return run_talker_gpu_step("code2wav_decode", invoke)'''

assert target_decode in content, "target_decode not found"
content = content.replace(target_decode, replacement_decode, 1)

# 4. Add admit_listener for subtalker
target_subtalker = '''            # Sub-talker: CUDA graph (ADR-026, 15 residui in ~12ms) oppure
            # fallback eager bit-exact (stesso warper order di transformers).
            timing_pair = begin_cuda_timing("subtalker")'''

replacement_subtalker = '''            # Sub-talker: CUDA graph (ADR-026, 15 residui in ~12ms) oppure
            # fallback eager bit-exact (stesso warper order di transformers).
            admit_listener("talker", "subtalker")
            timing_pair = begin_cuda_timing("subtalker")'''

assert target_subtalker in content, "target_subtalker not found"
content = content.replace(target_subtalker, replacement_subtalker, 1)

# 5. Add admit_listener for terminal_qwen_model and terminal_codec_head
target_terminal_qwen = '''            terminal_input = apply_frame_residual(
                terminal_input, generation_step=len(codes) - 1
            )
            if native_executor is not None:'''

replacement_terminal_qwen = '''            terminal_input = apply_frame_residual(
                terminal_input, generation_step=len(codes) - 1
            )
            admit_listener("talker", "terminal_qwen_model")
            if native_executor is not None:'''

assert target_terminal_qwen in content, "target_terminal_qwen not found"
content = content.replace(target_terminal_qwen, replacement_terminal_qwen, 1)

target_terminal_head = '''            terminal_queries = 1
            terminal_logits = run_talker_gpu_step(
                "terminal_codec_head",'''

replacement_terminal_head = '''            terminal_queries = 1
            admit_listener("talker", "terminal_codec_head")
            terminal_logits = run_talker_gpu_step(
                "terminal_codec_head",'''

assert target_terminal_head in content, "target_terminal_head not found"
content = content.replace(target_terminal_head, replacement_terminal_head, 1)

path.write_text(content, encoding="utf-8")
print("Successfully patched duplex_kernel/engine.py")
