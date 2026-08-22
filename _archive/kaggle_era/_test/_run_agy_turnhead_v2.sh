#!/bin/bash
# agy Task 5: TurnStateHead v2 — scalata dati DuplexConv + bilanciamento frame 5 classi + analisi lingua/timbrica
# Prompt letto da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

tries=0; max=4
while [ $tries -lt $max ]; do
  tries=$((tries+1))
  log "--- TURNHEAD V2 (tentativo $tries/$max) ---"
  agy -p "$(cat _archive/agy_prompts/_agy_prompt_t5_turnhead_v2.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 130m --dangerously-skip-permissions > _agy_out_t5_turnhead_v2.txt 2>&1
  code=$?
  if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" _agy_out_t5_turnhead_v2.txt && ! grep -q "timeout waiting for response" _agy_out_t5_turnhead_v2.txt; then
    log "--- TURNHEAD V2 OK (tentativo $tries) ---"
    exit 0
  fi
  log "--- TURNHEAD V2 fallito (exit=$code), retry tra 30s ---"
  tail -3 _agy_out_t5_turnhead_v2.txt
  sleep 30
done
log "--- TURNHEAD V2: FALLITO dopo $max tentativi ---"
exit 1
