# paths.py - Centralized path resolution for Gemma 4 Full-Duplex pipeline
# Pure ASCII encoding - compatible with both local Linux (RTX 3090 / Docker) and Kaggle kernels.

import os
import sys
from typing import Dict, Optional

# Cache for resolved paths
_ROOT_CACHE: Optional[str] = None
_BIG_DISK_CACHE: Optional[str] = None


def get_pipeline_env() -> str:
    """Return active environment: 'kaggle' or 'local3090' (default)."""
    env = os.environ.get("PIPELINE_ENV", "").strip().lower()
    if env in ("kaggle", "local3090"):
        return env
    if os.path.isdir("/kaggle/working"):
        return "kaggle"
    return "local3090"


def get_project_root() -> str:
    """Return the absolute root path of the project workspace."""
    global _ROOT_CACHE
    if _ROOT_CACHE is not None:
        return _ROOT_CACHE

    env = get_pipeline_env()
    custom_root = os.environ.get("PROJECT_ROOT", "").strip()

    if custom_root:
        root = os.path.abspath(os.path.expanduser(custom_root))
    elif env == "kaggle":
        root = "/kaggle/working"
    else:
        # local3090: check ~/gemma-4-omni or current working directory
        home_gemma = os.path.abspath(os.path.expanduser("~/gemma-4-omni"))
        if os.path.isdir(home_gemma):
            root = home_gemma
        elif os.path.isdir("/workspace"):
            root = "/workspace"
        else:
            root = os.path.abspath(os.getcwd())

    _ROOT_CACHE = root
    return _ROOT_CACHE


def get_data_dir() -> str:
    """Return path to data directory for tokenized datasets and cache."""
    return os.path.join(get_project_root(), "data")


def get_checkpoints_dir() -> str:
    """Return path to checkpoints directory."""
    return os.path.join(get_project_root(), "checkpoints")


def get_samples_dir() -> str:
    """Return path to audio synthesis samples directory."""
    return os.path.join(get_project_root(), "samples")


def get_logs_dir() -> str:
    """Return path to training and execution logs directory."""
    return os.path.join(get_project_root(), "logs")


def get_subtalker_dir() -> str:
    """Return path to extracted Qwen3-TTS sub-talker weights."""
    custom = os.environ.get("QWEN_SUBTALKER_DIR", "").strip()
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.join(get_project_root(), "qwen_subtalker")


def get_code2wav_dir() -> str:
    """Return path to extracted Code2Wav decoder weights."""
    custom = os.environ.get("CODE2WAV_WEIGHTS_DIR", "").strip()
    if custom:
        return os.path.abspath(os.path.expanduser(custom))
    return os.path.join(get_project_root(), "code2wav_weights")


def get_big_disk_dir() -> str:
    """Return path for heavy scratch files (tar downloads, temp wavs, raw extractions)."""
    global _BIG_DISK_CACHE
    if _BIG_DISK_CACHE is not None:
        return _BIG_DISK_CACHE

    env = get_pipeline_env()
    candidates = []

    # 1. Explicit environment overrides
    env_big = (os.environ.get("BIG_DISK_DIR") or os.environ.get("KAGGLE_BIG_DISK", "")).strip()
    if env_big:
        candidates.append(os.path.abspath(os.path.expanduser(env_big)))

    # 2. Kaggle default
    if env == "kaggle" or os.path.isdir("/kaggle/working"):
        candidates.append("/root/kaggle_big")

    # 3. local3090 defaults: data/scratch under project root or /tmp/gemma_big
    root = get_project_root()
    candidates.append(os.path.join(root, "data"))
    candidates.append(os.path.join(root, "data", "scratch"))
    candidates.append("/tmp/gemma_big")

    for c in candidates:
        try:
            os.makedirs(c, exist_ok=True)
            probe = os.path.join(c, ".write_probe")
            with open(probe, "w", encoding="ascii") as f:
                f.write("1")
            os.remove(probe)
            _BIG_DISK_CACHE = c
            return _BIG_DISK_CACHE
        except Exception:
            continue

    fallback = os.path.join(get_project_root(), "data")
    os.makedirs(fallback, exist_ok=True)
    _BIG_DISK_CACHE = fallback
    return _BIG_DISK_CACHE


def resolve_path(rel_or_abs: str, base_dir: Optional[str] = None) -> str:
    """Resolve a relative or absolute path against base_dir or project root."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    if base_dir is None:
        base_dir = get_project_root()
    return os.path.abspath(os.path.join(base_dir, rel_or_abs))


def ensure_all_dirs() -> Dict[str, str]:
    """Ensure all required workspace directories exist and return their paths."""
    dirs = {
        "root": get_project_root(),
        "data": get_data_dir(),
        "checkpoints": get_checkpoints_dir(),
        "samples": get_samples_dir(),
        "logs": get_logs_dir(),
        "subtalker": get_subtalker_dir(),
        "code2wav": get_code2wav_dir(),
        "big_disk": get_big_disk_dir(),
    }
    for name, path in dirs.items():
        try:
            os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"[paths] Warning: could not create {name} dir at {path}: {e}")
            sys.stdout.flush()
    return dirs
