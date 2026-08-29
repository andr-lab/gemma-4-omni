#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import inspect
from vllm_omni.config import register_pipeline
print('register_pipeline signature:', inspect.signature(register_pipeline))
print('register_pipeline doc:', register_pipeline.__doc__)
"