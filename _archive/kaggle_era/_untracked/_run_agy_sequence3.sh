#!/bin/bash
# Sequenza agy 2-4 con retry (il backend Gemini ha dato errori 500 transitori)
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

run_agy() {
  local label="$1" prompt_file="$2" out_file="$3" timeout_min="$4"
  local tries=0 max=4
  while [ $tries -lt $max ]; do
    tries=$((tries+1))
    log "--- $label (tentativo $tries/$max) ---"
    agy -p "$(cat "$prompt_file")" --model gemini-3.7-flash-high --effort high --print-timeout "${timeout_min}m" --dangerously-skip-permissions > "$out_file" 2>&1
    local code=$?
    if [ $code -eq 0 ] && ! grep -q "Error: Agent execution terminated due to error" "$out_file"; then
      log "--- $label OK (tentativo $tries) ---"
      return 0
    fi
    log "--- $label fallito (exit=$code), retry tra 30s ---"
    tail -3 "$out_file"
    sleep 30
  done
  log "--- $label: FALLITO dopo $max tentativi ---"
  return 1
}

run_agy "TASK2 PULIZIA CODICE" "_archive/agy_prompts/_agy_prompt_code_clean.txt" "_agy_out_code_clean.txt" 25
echo "TASK2_EXIT=$?"

run_agy "TASK3 REVISIONE LETTERATURA" "_archive/agy_prompts/_agy_prompt_lit_review.txt" "_agy_out_lit_review.txt" 30
echo "TASK3_EXIT=$?"

run_agy "TASK4 QLORA" "_archive/agy_prompts/_agy_prompt_qlora.txt" "_agy_out_qlora.txt" 40
echo "TASK4_EXIT=$?"

log "=== SEQUENZA AGY COMPLETATA ==="
