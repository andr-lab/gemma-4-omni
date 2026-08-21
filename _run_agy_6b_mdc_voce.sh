#!/bin/bash
# agy Task 6B: Revisione architetturale MDC-only + Prep MDC italiano + Retrain adapter voce italiana
# Prompt letto da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- TASK 6B MDC+VOCE (tentativo $tries/$max) ---"
  agy -p "$(cat _archive/agy_prompts/_agy_prompt_t6b_mdc_voce.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 140m --dangerously-skip-permissions > _agy_out_t6b_mdc_voce.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_t6b_mdc_voce.txt && ! grep -q "timeout waiting for response" _agy_out_t6b_mdc_voce.txt; then
    log "--- TASK 6B OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- TASK 6B fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_t6b_mdc_voce.txt
  sleep 30
done
log "--- TASK 6B: FALLITO dopo $max tentativi ---"
exit 1
