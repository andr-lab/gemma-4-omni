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
    with open(OUT, "w", encoding="utf-8", errors="replace") as f:
        r = subprocess.run(["kaggle", "kernels", "logs", KERNEL],
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                           capture_output=True, text=True, timeout=180)
        f.write(r.stdout)
    return os.path.getsize(OUT) if os.path.exists(OUT) else 0

t0 = time.time()
last = None
print(f"Monitor avviato per {KERNEL} (timeout {TIMEOUT//60} min, poll {POLL}s)...", flush=True)
while time.time() - t0 < TIMEOUT:
    r = sh(["kaggle", "kernels", "status", KERNEL])
    status = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"ERR {r.stderr[-200:]}"
    if status != last:
        print(f"[{int(time.time()-t0)}s] {status}", flush=True)
        last = status
    if "complete" in status.lower() or "error" in status.lower() or "cancel" in status.lower():
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
        sys.exit(0 if "complete" in status.lower() else 1)
    time.sleep(POLL)
print("TIMEOUT monitor", flush=True)
sys.exit(2)
