import asyncio
import json
from dataclasses import replace
from pathlib import Path

from client.duplex_dialogue_harness import (
    DialogueConfig,
    DialogueOrchestrator,
    streaming_scenario_from_fixtures,
)

scenario = streaming_scenario_from_fixtures(
    "A", Path("/workspace/artifacts/system_convergence_final_20261023/attempt7_A4s/inputs_4s")
)
scenario = replace(scenario, steps=scenario.steps[:1])
config = DialogueConfig(
    max_duration_s=30.0,
    silence_timeout_s=3.0,
    frame_pacing_s=0.080,
    ws_url="ws://127.0.0.1:8000/v1/realtime?duplex=1",
    streaming_voice_profile="gemma",
    streaming_turn_policy="gemma_native_control6",
)
runner = DialogueOrchestrator(
    config=config,
    output_dir=Path("/workspace/artifacts/system_convergence_final_20261023/attempt8_final/smoke/one_turn_output"),
)
result = asyncio.run(runner.run_streaming_scenario(scenario))
print("SMOKE_RESULT " + json.dumps({"status": result.get("status"), "keys": sorted(result)}, ensure_ascii=False), flush=True)
if result.get("status") != "COMPLETED":
    raise SystemExit(1)
