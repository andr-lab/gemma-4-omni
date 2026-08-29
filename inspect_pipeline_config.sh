#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -c "
import dataclasses
from vllm_omni.config import PipelineConfig, StagePipelineConfig, StageExecutionType
print('PipelineConfig fields:', [f.name for f in dataclasses.fields(PipelineConfig)])
print('StagePipelineConfig fields:', [f.name for f in dataclasses.fields(StagePipelineConfig)])
print('StageExecutionType values:', [e.value for e in StageExecutionType])
"