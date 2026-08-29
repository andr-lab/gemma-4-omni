#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest python3 -m vllm_omni.entrypoints.openai.api_server --help