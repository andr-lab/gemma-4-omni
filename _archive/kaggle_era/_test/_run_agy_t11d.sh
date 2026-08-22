#!/bin/bash
# _run_agy_t11d.sh - Task 11D: ascolto multimodale + diagnosi decoder/dataset + MDC 26
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11d_voce_reale.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11d_voce_reale.txt 2>&1

echo "AGY_T11D_EXIT=$?"
tail -5 _agy_out_t11d_voce_reale.txt
