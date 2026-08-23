#!/bin/bash
set -e
cd /c/Users/andre/Documents/gemma-4-omni
agy -p "$(cat _agy_prompt_t11h_ascolto.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11h_ascolto.txt 2>&1
echo "AGY_LISTEN_EXIT=$?"
tail -30 _agy_out_t11h_ascolto.txt
