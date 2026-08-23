import os, json, re

log_path = os.path.join(os.environ.get('TEMP', '/tmp'), 'gemma4-window128-r1_kernel.log')
with open(log_path, encoding='utf-8', errors='replace') as f:
    raw_json = f.read()

try:
    entries = json.loads(raw_json)
    clean_text = "".join(e.get("data", "") for e in entries)
except Exception:
    clean_text = raw_json

with open("window128_clean_log.txt", "w", encoding="utf-8") as f:
    f.write(clean_text)

print("Parsed clean log bytes:", len(clean_text))
for line in clean_text.splitlines():
    if any(k in line for k in ("[WINDOW128", "[EVAL_WINDOW128", "stage2window128_report", "eval_window128_report", "STAGE 2", "EVAL", "step ", "cos_sim", "rank SVD", "PARAMETRI", "COMPLETAT", "warm-start", "PATCH", "ROPE_FIX", "THETA_FIX", "GPU rilevata")):
        print(line)
