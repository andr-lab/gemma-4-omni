#!/bin/bash
# _run_agy_t8_3090.sh - Lancia agy Task 8 (proseguimento lavoro su RTX 3090)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t8_3090.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t8_3090.txt 2>&1

echo "AGY_T8_EXIT=$?"
tail -5 _agy_out_t8_3090.txt
