"""Atomic Stage1 adapter/optimizer checkpoints with exact RNG restoration."""
from __future__ import annotations

import json
import os
import random
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture all RNG streams used by the Stage1 sampler and CUDA dropout."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contiene stato RNG CUDA ma CUDA non è disponibile")
        torch.cuda.set_rng_state_all(cuda_state)


def save_training_checkpoint(
    *,
    checkpoint_root: str | Path,
    completed_steps: int,
    optimizer: torch.optim.Optimizer,
    training_config: Mapping[str, Any],
    losses: list[float],
    step_runtime: list[dict[str, Any]],
    run_metadata: Mapping[str, Any] | None = None,
    save_adapter: Callable[[Path], None],
) -> Path:
    """Write a complete step directory, then atomically publish it by rename."""
    if completed_steps <= 0:
        raise ValueError("completed_steps checkpoint deve essere positivo")
    root = Path(checkpoint_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"step_{completed_steps:06d}"
    if final_path.exists():
        raise FileExistsError(f"checkpoint già esistente, non sovrascritto: {final_path}")
    temporary = root / f".{final_path.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    (temporary / "adapter").mkdir()
    save_adapter(temporary / "adapter")
    state = {
        "schema_version": 1,
        "completed_steps": int(completed_steps),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
    }
    torch.save(state, temporary / "training_state.pt")
    metadata = {
        "schema_version": 1,
        "completed_steps": int(completed_steps),
        "training_config": dict(training_config),
        "losses": [float(value) for value in losses],
        "step_runtime": step_runtime,
        "run_metadata": dict(run_metadata or {}),
    }
    (temporary / "training_state.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.rename(temporary, final_path)
    return final_path


def read_training_checkpoint_metadata(
    *, checkpoint_dir: str | Path, expected_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate the small checkpoint contract before loading a GPU model."""
    directory = Path(checkpoint_dir).expanduser().resolve()
    metadata = json.loads((directory / "training_state.json").read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("training_config") != dict(expected_config):
        raise ValueError(
            "configurazione Stage1 incompatibile col checkpoint: "
            f"saved={metadata.get('training_config')}, requested={dict(expected_config)}"
        )
    completed = metadata.get("completed_steps")
    if isinstance(completed, bool) or not isinstance(completed, int) or completed <= 0:
        raise ValueError("completed_steps checkpoint Stage1 non valido")
    if len(metadata.get("losses", [])) != completed or len(metadata.get("step_runtime", [])) != completed:
        raise ValueError("traccia loss/step incompleta nel checkpoint Stage1")
    adapter_dir = directory / "adapter"
    if not (adapter_dir / "adapter_config.json").is_file():
        raise ValueError(f"config PEFT mancante nel checkpoint: {adapter_dir}")
    if not any((adapter_dir / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"pesi PEFT mancanti nel checkpoint: {adapter_dir}")
    if not (directory / "training_state.pt").is_file():
        raise ValueError(f"stato optimizer mancante nel checkpoint: {directory}")
    return metadata


def restore_training_checkpoint(
    *,
    checkpoint_dir: str | Path,
    optimizer: torch.optim.Optimizer,
    expected_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore optimizer and RNG after the PEFT adapter has been loaded."""
    directory = Path(checkpoint_dir).expanduser().resolve()
    metadata = read_training_checkpoint_metadata(
        checkpoint_dir=directory, expected_config=expected_config
    )
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    if state.get("schema_version") != 1 or state.get("completed_steps") != metadata["completed_steps"]:
        raise ValueError("stato optimizer Stage1 incompleto o incoerente col manifest checkpoint")
    optimizer.load_state_dict(state["optimizer_state_dict"])
    restore_rng_state(state["rng_state"])
    return metadata
