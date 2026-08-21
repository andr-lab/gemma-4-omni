#!/bin/bash
# agy Task 6A: Prep DuplexConv esteso + TurnStateHead v3 (bilanciamento calibrato)
# Prompt letto da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- TASK 6A PREP+V3 (tentativo $tries/$max) ---"
  agy -p "$(cat _archive/agy_prompts/_agy_prompt_t6a_prep_v3.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 140m --dangerously-skip-permissions > _agy_out_t6a_prep_v3.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_t6a_prep_v3.txt && ! grep -q "timeout waiting for response" _agy_out_t6a_prep_v3.txt; then
    log "--- TASK 6A OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- TASK 6A fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_t6a_prep_v3.txt
  sleep 30
done
log "--- TASK 6A: FALLITO dopo $max tentativi ---"
exit 1
