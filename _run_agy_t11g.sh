#!/bin/bash
# _run_agy_t11g.sh - Task 11G: diagnosi adapter voce robotica + fix + retrain
set -e
cd /c/Users/andre/Documents/gemma-4-omni

agy -p "$(cat _agy_prompt_t11g_adapter_robot.txt)" \
    --model gemini-3.7-flash-high \
    --effort high \
    --print-timeout 55m \
    --dangerously-skip-permissions \
    > _agy_out_t11g_adapter_robot.txt 2>&1

echo "AGY_T11G_EXIT=$?"
tail -5 _agy_out_t11g_adapter_robot.txt
