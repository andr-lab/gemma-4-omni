#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import vllm_omni.config as cfg
print('Config exports:', [m for m in dir(cfg) if not m.startswith('_')])

from vllm_omni.config.config_factory import (
    register_pipeline_config,
    get_pipeline_config,
)
print('register_pipeline_config available:', register_pipeline_config)
"