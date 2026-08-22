# -*- coding: utf-8 -*-
# Monitor kernel Kaggle: attende COMPLETE/ERROR poi scarica il log UTF-8 e termina.
# Uso: python _monitor_kernel.py <kernel_id> [timeout_min]
import subprocess, sys, time, os

KERNEL = sys.argv[1] if len(sys.argv) > 1 else "andrlab/gemma4-qlora-r1"
TIMEOUT = int(sys.argv[2]) * 60 if len(sys.argv) > 2 else 90 * 60
OUT = os.path.join(os.environ.get("TEMP", "/tmp"), KERNEL.split("/")[-1] + "_kernel.log")
POLL = 30

def sh(cmd, **kw):
    env = dict(os.environ); env["PYTHONIOENCODING"] = "utf-8"
    kw.setdefault("env", env)
    kw.setdefault("capture_output", True)
    kw.setdefault("text", True)
    kw.setdefault("timeout", 180)
    return subprocess.run(cmd, **kw)

def fetch_log():
    try:
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run(["kaggle", "kernels", "logs", KERNEL],
                           env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=180)
        with open(OUT, "w", encoding="utf-8", errors="replace") as f:
            f.write(r.stdout or "")
            if r.stderr:
                f.write("\n--- STDERR ---\n" + r.stderr)
        return os.path.getsize(OUT) if os.path.exists(OUT) else 0
    except Exception as e:
        print(f"fetch_log warning: {e}", flush=True)
        return 0

t0 = time.time()
last = None
print(f"Monitor avviato per {KERNEL} (timeout {TIMEOUT//60} min, poll {POLL}s)...", flush=True)
while time.time() - t0 < TIMEOUT:
    r = sh(["kaggle", "kernels", "status", KERNEL])
    status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR {r.stderr[-200:]}"
    if status != last:
        print(f"[{int(time.time()-t0)}s] {status}", flush=True)
        last = status
    s_low = status.lower()
    is_kernel_complete = "kernelworkerstatus.complete" in s_low or ("has status" in s_low and "complete" in s_low)
    is_kernel_error = "kernelworkerstatus.error" in s_low or ("has status" in s_low and "error" in s_low)
    is_kernel_cancel = "kernelworkerstatus.cancel" in s_low or ("has status" in s_low and "cancel" in s_low)

    if is_kernel_complete or is_kernel_error or is_kernel_cancel:
        time.sleep(15)
        n = fetch_log()
        if n == 0:
            print("retry log fetch...", flush=True)
            time.sleep(15)
            n = fetch_log()
        print(f"LOG salvato ({n} bytes): {OUT}", flush=True)
        if n > 0:
            with open(OUT, encoding="utf-8", errors="replace") as f:
                txt = f.read()
            print("--- TAIL (ultimi 4000 caratteri) ---", flush=True)
            print(txt[-4000:], flush=True)
        else:
            print("LOG NON SALVATO", flush=True)
        sys.exit(0 if is_kernel_complete else 1)
    time.sleep(POLL)
print("TIMEOUT monitor", flush=True)
sys.exit(2)
