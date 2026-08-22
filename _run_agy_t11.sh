#!/bin/bash
# _run_agy_t11.sh - Lancia agy Task 11 (fix voci reali + CV26 + nuovo training adapter)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11_voci_reali.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11_voci_reali.txt 2>&1

echo "AGY_T11_EXIT=$?"
tail -5 _agy_out_t11_voci_reali.txt
