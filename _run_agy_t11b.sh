#!/bin/bash
# _run_agy_t11b.sh - Lancia agy Task 11B (training completo adapter v2 voci reali)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11b_adapter_v2.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11b_adapter_v2.txt 2>&1

echo "AGY_T11B_EXIT=$?"
tail -5 _agy_out_t11b_adapter_v2.txt
