# CELLA INSTALL - dipendenze Fase 2 su GPU P100 (sm_60)
# P100 non supporta torch cu128 (solo sm_70+). Serve torch cu118.
# Pattern VERIFICATO in test kernel: torch cu118 + --no-deps per non far toccare torch.
import subprocess, sys, os, re as _re

def gpu_info():
    try:
        r = subprocess.run([sys.executable, '-c',
            "import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"],
            capture_output=True, text=True, timeout=120)
        lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
        cap = None
        if len(lines) > 1:
            m = _re.search(r'(\d+)', lines[1])
            cap = int(m.group(1)) if m else None
        return (lines[0] if lines else "?"), cap
    except Exception as e:
        return f"ERR {str(e)[:50]}", None

name, cap = gpu_info()
print("GPU rilevata:", name, "| compute capability:", cap)
sys.stdout.flush()

if cap is not None and cap < 7:
    # P100 (sm_60): stack cu118 (supporta Pascal), torch 2.6.0
    print(">>> GPU Pascal rilevata: installo torch 2.6.0+cu118 (supporta sm_60) <<<")
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet',
         '--index-url', 'https://download.pytorch.org/whl/cu118',
         'torch==2.6.0', 'torchvision==0.21.0', 'torchaudio==2.6.0'],
        capture_output=True, text=True, timeout=1800)
    print("torch cu118 rc:", r.returncode, (r.stdout + r.stderr)[-150:])
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-deps',
         'transformers==5.5.0', 'unsloth', 'unsloth_zoo', 'peft', 'safetensors', 'soundfile', 'pyarrow',
         'trl', 'xformers==0.0.29.post3', 'bitsandbytes', 'triton', 'tyro', 'structlog',
         'nest-asyncio', 'hf_transfer', 'diffusers', 'typer', 'sentencepiece', 'datasets'],
        capture_output=True, text=True, timeout=1800)
    print("stack cu118 rc:", r.returncode, (r.stdout + r.stderr)[-200:])
else:
    # T4 o superiore (cap>=7): stack normale cu128 (il kernel batch preinstalla torch cu128)
    # NOTA: qui NON usiamo --no-deps perche' le deps di transformers/unsloth (accelerate,
    # tokenizers, huggingface-hub, etc.) devono risolversi. torch resta quello preinstallato
    # (2.10+cu128) perche' transformers 5.5.0 richiede torch>=2.4 e unsloth torch<2.12.
    print(">>> GPU Volta+ rilevata: stack normale cu128 <<<")
    sys.stdout.flush()
    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--quiet', '--upgrade',
         'transformers==5.5.0', 'unsloth', 'safetensors', 'soundfile', 'pyarrow'],
        capture_output=True, text=True, timeout=1800)
    print("stack cu128 rc:", r.returncode, (r.stdout + r.stderr)[-250:])
    sys.stdout.flush()

import importlib.metadata as md
for pkg in ['transformers', 'unsloth', 'peft', 'torch']:
    try: print(pkg + ": " + md.version(pkg))
    except Exception: print(pkg + ": MANCANTE")
# verifica GPU compatibile con torch
r2 = subprocess.run([sys.executable, '-c',
    "import torch; print('cuda ok:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"],
    capture_output=True, text=True, timeout=120)
print("check cuda:", (r2.stdout + r2.stderr).strip()[-200:])
