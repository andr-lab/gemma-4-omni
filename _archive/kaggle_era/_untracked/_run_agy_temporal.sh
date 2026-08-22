#!/bin/bash
# agy: modulo temporale causale — letteratura arXiv + implementazione + smoke (POI si ferma)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- TEMPORAL (tentativo $tries/$max) ---"
  agy -p "$(cat _agy_prompt_temporal.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 55m --dangerously-skip-permissions > _agy_out_temporal.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_temporal.txt && ! grep -q "timeout waiting for response" _agy_out_temporal.txt; then
    log "--- TEMPORAL OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- TEMPORAL fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_temporal.txt
  sleep 30
done
log "--- TEMPORAL: FALLITO dopo $max tentativi ---"
exit 1
