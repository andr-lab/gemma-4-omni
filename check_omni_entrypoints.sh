#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import vllm_omni
import pkg_resources
import importlib

print('vllm-omni entrypoints:')
for ep in pkg_resources.iter_entry_points('console_scripts'):
    if 'omni' in ep.name or 'vllm' in ep.name:
        print(' -', ep.name, '->', ep.module_name)

import vllm_omni.entrypoints.openai.api_server as oas
print('Found OpenAI Omni API Server entrypoint:', oas.__file__)
"