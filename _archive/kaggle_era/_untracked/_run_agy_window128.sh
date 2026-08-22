#!/bin/bash
# agy Task 3: finestra 128 frame (10.24s) — implementazione + smoke (POI si ferma)
# Prompt letto da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- WINDOW128 (tentativo $tries/$max) ---"
  agy -p "$(cat _archive/agy_prompts/_agy_prompt_t3_window128.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 55m --dangerously-skip-permissions > _agy_out_t3_window128.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_t3_window128.txt && ! grep -q "timeout waiting for response" _agy_out_t3_window128.txt; then
    log "--- WINDOW128 OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- WINDOW128 fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_t3_window128.txt
  sleep 30
done
log "--- WINDOW128: FALLITO dopo $max tentativi ---"
exit 1
