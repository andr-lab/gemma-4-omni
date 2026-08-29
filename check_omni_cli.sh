#!/bin/bash
docker run --rm --gpus all --ipc=host vllm-omni-cuda:latest vllm serve --help