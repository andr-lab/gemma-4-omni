#!/bin/bash
# Sequenza agy: 1) repo/docs  2) pulizia codice  3) revisione letteratura  4) QLoRA
cd /c/Users/andre/Documents/gemma-4-omni || exit 1

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "=== TASK 1/4: ORDINE REPO E DOCUMENTAZIONE ==="
agy -p "$(cat _agy_prompt_repo_order.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 25m --dangerously-skip-permissions > _agy_out_repo_order.txt 2>&1
echo "TASK1_EXIT=$?"
tail -5 _agy_out_repo_order.txt

log "=== TASK 2/4: PULIZIA CODICE ==="
agy -p "$(cat _agy_prompt_code_clean.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 25m --dangerously-skip-permissions > _agy_out_code_clean.txt 2>&1
echo "TASK2_EXIT=$?"
tail -5 _agy_out_code_clean.txt

log "=== TASK 3/4: REVISIONE LETTERATURA ==="
agy -p "$(cat _agy_prompt_lit_review.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 30m --dangerously-skip-permissions > _agy_out_lit_review.txt 2>&1
echo "TASK3_EXIT=$?"
tail -5 _agy_out_lit_review.txt

log "=== TASK 4/4: QLoRA su Gemma E2B (smoke + run completo) ==="
agy -p "$(cat _agy_prompt_qlora.txt)" --model gemini-3.7-flash-high --effort high --print-timeout 40m --dangerously-skip-permissions > _agy_out_qlora.txt 2>&1
echo "TASK4_EXIT=$?"
tail -5 _agy_out_qlora.txt

log "=== SEQUENZA AGY COMPLETATA ==="
