#!/bin/bash
# _run_agy_t11c.sh - Lancia agy Task 11C (fix clipping decoder + rigenerazione wav)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11c_fix_clipping.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11c_fix_clipping.txt 2>&1

echo "AGY_T11C_EXIT=$?"
tail -5 _agy_out_t11c_fix_clipping.txt
