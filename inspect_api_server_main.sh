#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import vllm_omni.entrypoints.openai.api_server as srv
import inspect
print('omni_run_server signature:', inspect.signature(srv.omni_run_server))
"