import os
import sys

def patch_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    old_block = """            quantum_ctx = (
                listener_compute_quantum()
                if callable(listener_compute_quantum)
                else nullcontext(None)
            )
            lock_acquired = False
            acquired_at = 0.0
            wait_started = 0.0
            try:
                with quantum_ctx as permit:
                    if priority_lock_talker and thinker_lock:
                        wait_started = time.perf_counter()
                        if not self._listener_priority_lock():
                            raise GenerationAdmissionAborted(
                                "generation stopped before Thinker lock acquisition"
                            )
                        acquired_at = time.perf_counter()
                        lock_acquired = True
                        if metrics is not None:
                            wait_ms = (acquired_at - wait_started) * 1000.0
                            wait_key = f"listener_priority_{site}_lock_wait_ms"
                            wait_max_key = f"listener_priority_{site}_lock_wait_max_ms"
                            metrics[wait_key] = metrics.get(wait_key, 0.0) + wait_ms
                            metrics[wait_max_key] = max(
                                metrics.get(wait_max_key, 0.0), wait_ms
                            )
                    result = forward()
                    record_completion = getattr(self, "_record_talker_cuda_completion", None)
                    if callable(record_completion) and permit is not None and permit is not False:
                        record_completion(permit)
                    return result"""

    new_block = """            lock_acquired = False
            acquired_at = 0.0
            wait_started = 0.0
            try:
                if priority_lock_talker and thinker_lock:
                    wait_started = time.perf_counter()
                    if not self._listener_priority_lock():
                        raise GenerationAdmissionAborted(
                            "generation stopped before Thinker lock acquisition"
                        )
                    acquired_at = time.perf_counter()
                    lock_acquired = True
                    if metrics is not None:
                        wait_ms = (acquired_at - wait_started) * 1000.0
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
                with quantum_ctx as permit:
                    result = forward()
                    record_completion = getattr(self, "_record_talker_cuda_completion", None)
                    if callable(record_completion) and permit is not None and permit is not False:
                        record_completion(permit)
                    return result"""

    # Normalize newlines for search
    is_crlf = "\r\n" in content
    normalized_content = content.replace("\r\n", "\n")
    if old_block not in normalized_content:
        print(f"ERROR: old_block not found in {path}")
        return False

    patched_content = normalized_content.replace(old_block, new_block, 1)
    if is_crlf:
        patched_content = patched_content.replace("\n", "\r\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(patched_content)
    print(f"SUCCESS: patched {path}")
    return True

if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else "duplex_kernel/engine.py"
    if not patch_file(target_path):
        sys.exit(1)
