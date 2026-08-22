#!/bin/bash
ls /opt/venv_phase1/bin/python 2>/dev/null && echo VENV_ESISTE || echo VENV_MANCANTE
/opt/venv_phase1/bin/python -c 'import qwen_tts, transformers; print("qwen_tts OK, transformers", transformers.__version__)' 2>&1 | tail -1
/opt/venv_phase1/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>&1 | tail -1
