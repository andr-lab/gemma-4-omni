#!/bin/bash
# _run_agy_t10.sh - Lancia agy Task 10 (TurnHead v4 + valutazione italiano + retraining MDC)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t10_turnhead_italiano.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t10_turnhead_italiano.txt 2>&1

echo "AGY_T10_EXIT=$?"
tail -5 _agy_out_t10_turnhead_italiano.txt
