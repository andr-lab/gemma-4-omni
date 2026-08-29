#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import inspect
from vllm_omni.entrypoints.openai.api_server import make_arg_parser
print('make_arg_parser signature:', inspect.signature(make_arg_parser))
"