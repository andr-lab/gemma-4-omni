import json

r2 = "/home/administrator/gemma-4-omni/artifacts/acceptance_suite_20261104T012215626940Z/run_02"
log_path = f"{r2}/server.log"

with open(log_path) as f:
    for line in f:
        if 'frame_id": 456' in line or 'frame_id": 457' in line:
            print(line.strip())
