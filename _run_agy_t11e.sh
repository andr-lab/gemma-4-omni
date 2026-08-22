#!/bin/bash
# _run_agy_t11e.sh - Task 11E: integra decoder ufficiale nella pipeline (subprocess venv phase1)
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11e_decoder_pipeline.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11e_decoder_pipeline.txt 2>&1

echo "AGY_T11E_EXIT=$?"
tail -5 _agy_out_t11e_decoder_pipeline.txt
