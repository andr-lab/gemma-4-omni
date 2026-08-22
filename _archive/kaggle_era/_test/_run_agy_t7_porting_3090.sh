#!/bin/bash
# Task 7: porting pipeline su RTX 3090 locale (SSH + Docker) — retry x4 su errori 500
set -e
cd "$(dirname "$0")"
OUT=_agy_out_t7_porting_3090.txt
: > "$OUT"
for attempt in 1 2 3 4; do
  echo "=== TENTATIVO $attempt ===" >> "$OUT"
  if agy -p "$(cat _agy_prompt_t7_porting_3090.txt)" \
       --model gemini-3.7-flash-high --effort high \
       --print-timeout 90m --dangerously-skip-permissions \
       >> "$OUT" 2>&1; then
    echo "=== T7 OK (tentativo $attempt) ===" >> "$OUT"
    exit 0
  fi
  echo "=== T7 fallito (tentativo $attempt), retry in 30s ===" >> "$OUT"
  sleep 30
done
echo "=== T7 FALLITO DOPO 4 TENTATIVI ===" >> "$OUT"
exit 1
