#!/bin/bash
# _run_agy_t9b_adapter_it.sh - Lancia agy Task 9B (training adapter italiano su RTX 3090)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t9b_adapter_it_3090.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t9b_adapter_it.txt 2>&1

echo "AGY_T9B_EXIT=$?"
tail -5 _agy_out_t9b_adapter_it.txt
