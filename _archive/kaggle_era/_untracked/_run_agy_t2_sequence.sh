#!/bin/bash
# Sequenza agy Task 2: A=potenzia modulo temporale, B=rank SVD, C=dinamiche full-duplex
# I prompt si leggono da _archive/agy_prompts/ (lezione: agy puo' archiviare i prompt e i task successivi falliscono)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_agy() {
  local label="$1" prompt="$2" out="$3"
  local tries=0 max=4
  while [ $tries -lt $max ]; do
    tries=$((tries+1))
    log "--- $label (tentativo $tries/$max) ---"
    agy -p "$(cat "$prompt")" --model gemini-3.7-flash-high --effort high --print-timeout 55m --dangerously-skip-permissions > "$out" 2>&1
    local code=$?
    if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" "$out" && ! grep -q "timeout waiting for response" "$out"; then
      log "--- $label OK (tentativo $tries) ---"
      return 0
    fi
    log "--- $label fallito (exit=$code), retry tra 30s ---"
    tail -3 "$out"
    sleep 30
  done
  log "--- $label: FALLITO dopo $max tentativi ---"
  return 1
}

run_agy "TASK A temporal2 (potenzia modulo temporale)" "_archive/agy_prompts/_agy_prompt_t2a_temporal2.txt" "_agy_out_t2a_temporal2.txt" || exit 1
run_agy "TASK B rankfix (aumenta rank SVD)" "_archive/agy_prompts/_agy_prompt_t2b_rank.txt" "_agy_out_t2b_rank.txt" || exit 1
run_agy "TASK C fullduplex (dinamiche turn-taking)" "_archive/agy_prompts/_agy_prompt_t2c_fullduplex.txt" "_agy_out_t2c_fullduplex.txt" || exit 1

log "=== SEQUENZA TASK 2 COMPLETATA (A+B+C) ==="
