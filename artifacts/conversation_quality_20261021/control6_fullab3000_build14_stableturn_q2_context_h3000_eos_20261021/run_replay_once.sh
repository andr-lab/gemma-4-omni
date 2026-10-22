#!/usr/bin/env bash
set -uo pipefail
cd /home/administrator/gemma-4-omni
container=gemma4-listener-control6-h3000-eos-build14-20261021
out=/home/administrator/gemma-4-omni/artifacts/conversation_quality_20261021/control6_fullab3000_build14_stableturn_q2_context_h3000_eos_20261021
docker exec "$container" /workspace/.venv_e2e_runtime/bin/python /workspace/artifacts/conversation_quality_20261020/control6_q2cache_build6_continuous_mic_bargein_20261020/run_replay.py --output-dir "/workspace/artifacts/conversation_quality_20261021/control6_fullab3000_build14_stableturn_q2_context_h3000_eos_20261021" > "$out/replay_stdout.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$out/replay.exit"
exit "$rc"
