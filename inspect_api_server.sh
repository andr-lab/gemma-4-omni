#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import inspect
import vllm_omni.entrypoints.openai.api_server as srv
print('vLLM-Omni API Server file:', srv.__file__)
print('Members in api_server:', [m for m in dir(srv) if not m.startswith('_')])
"