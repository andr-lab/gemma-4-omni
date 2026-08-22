#!/bin/bash
# _run_agy_t11f.sh - Task 11F: tokenizzazione CV26 it (MDC 26) + verifica percettiva + prep v3
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11f_prep_cv26.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11f_prep_cv26.txt 2>&1

echo "AGY_T11F_EXIT=$?"
tail -5 _agy_out_t11f_prep_cv26.txt
