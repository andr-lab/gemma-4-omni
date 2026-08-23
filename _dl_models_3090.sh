#!/bin/bash
# Download modelli HF sulla 3090 via API Python snapshot_download (robusto, indipendente dal CLI)
set -e
HOST=administrator@gpu3090
CACHE=/home/administrator/gemma-4-omni/hf_cache

ssh -o BatchMode=yes $HOST "mkdir -p $CACHE && echo CACHE_OK"

# Script python che gira nel container one-shot
cat > /tmp/hf_dl.py << 'PYEOF'
import os, sys
from huggingface_hub import snapshot_download
repos = [
    "unsloth/gemma-4-E2B-it",
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
]
for r in repos:
    print(f"=== DOWNLOAD {r} ===", flush=True)
    path = snapshot_download(repo_id=r, local_dir_use_symlinks=False)
    print(f"DONE {r} -> {path}", flush=True)
print("ALL_DOWNLOADS_DONE", flush=True)
PYEOF

scp -o BatchMode=yes /tmp/hf_dl.py $HOST:/tmp/hf_dl.py

ssh -o BatchMode=yes $HOST "docker run --rm \
  -e HF_HOME=$CACHE \
  -v $CACHE:$CACHE \
  -v /tmp/hf_dl.py:/tmp/hf_dl.py \
  python:3.11-slim bash -lc 'pip install -q huggingface_hub 2>/dev/null; python /tmp/hf_dl.py'"

echo "=== VERIFICA CACHE ==="
ssh -o BatchMode=yes $HOST "du -sh $CACHE; find $CACHE/hub -maxdepth 1 -type d | head"
echo "SCRIPT_ALL_DONE"
