#!/bin/bash
# agy Task 4: TurnStateHead stage 3 — analisi DuplexConv + implementazione + smoke + RUN COMPLETO + report
# Prompt letto da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- TURNHEAD (tentativo $tries/$max) ---"
  agy -p "$(cat _archive/agy_prompts/_agy_prompt_t4_turnhead.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 120m --dangerously-skip-permissions > _agy_out_t4_turnhead.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_t4_turnhead.txt && ! grep -q "timeout waiting for response" _agy_out_t4_turnhead.txt; then
    log "--- TURNHEAD OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- TURNHEAD fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_t4_turnhead.txt
  sleep 30
done
log "--- TURNHEAD: FALLITO dopo $max tentativi ---"
exit 1
