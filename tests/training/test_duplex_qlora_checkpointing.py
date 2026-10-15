from __future__ import annotations

import copy
import random
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.duplex_qlora.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
)


def _update(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    scale = random.random() + float(np.random.random())
    features = torch.randn((4, 3)) * scale
    target = torch.randn((4, 2))
    loss = (model(features) - target).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _save_fake_adapter(directory, model_state):
    (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    torch.save(model_state, directory / "adapter_model.bin")


def test_checkpoint_restores_adapter_optimizer_and_rng_for_identical_next_update(tmp_path):
    config = {"max_steps": 2, "learning_rate": 0.01, "seed": 260930, "dataset": "tiny-test"}
    random.seed(10)
    np.random.seed(11)
    torch.manual_seed(12)
    model = nn.Linear(3, 2, bias=False)
    initial_state = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    first_loss = _update(model, optimizer)

    checkpoint_dir = save_training_checkpoint(
        checkpoint_root=tmp_path / "stage1_checkpoints",
        completed_steps=1,
        optimizer=optimizer,
        training_config=config,
        losses=[first_loss],
        step_runtime=[{"step": 1, "lora_gradients_finite": True}],
        run_metadata={"run_id": "test-run", "conditioning_probe": {"text_only_loss": 0.1}},
        save_adapter=lambda path: _save_fake_adapter(path, model.state_dict()),
    )
    expected_loss = _update(model, optimizer)
    expected_state = copy.deepcopy(model.state_dict())

    resumed_model = nn.Linear(3, 2, bias=False)
    resumed_model.load_state_dict(initial_state)
    resumed_model.load_state_dict(
        torch.load(checkpoint_dir / "adapter" / "adapter_model.bin", map_location="cpu", weights_only=True)
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    metadata = restore_training_checkpoint(
        checkpoint_dir=checkpoint_dir,
        optimizer=resumed_optimizer,
        expected_config=config,
    )
    resumed_loss = _update(resumed_model, resumed_optimizer)

    assert metadata["completed_steps"] == 1
    assert metadata["losses"] == [first_loss]
    assert metadata["run_metadata"]["run_id"] == "test-run"
    assert resumed_loss == expected_loss
    assert all(torch.equal(resumed_model.state_dict()[name], value) for name, value in expected_state.items())


def test_checkpoint_does_not_overwrite_existing_step(tmp_path):
    optimizer = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=0.01)
    save_kwargs = {
        "checkpoint_root": tmp_path,
        "completed_steps": 1,
        "optimizer": optimizer,
        "training_config": {"max_steps": 2},
        "losses": [1.0],
        "step_runtime": [],
        "save_adapter": lambda path: _save_fake_adapter(path, {}),
    }
    saved = save_training_checkpoint(**save_kwargs)
    try:
        save_training_checkpoint(**save_kwargs)
    except FileExistsError as exc:
        assert "non sovrascritto" in str(exc)
    else:
        raise AssertionError("il checkpoint esistente è stato sovrascritto")
    assert saved.joinpath("training_state.pt").is_file()


def test_checkpoint_rejects_training_config_change(tmp_path):
    optimizer = torch.optim.AdamW(nn.Linear(1, 1).parameters(), lr=0.01)
    saved = save_training_checkpoint(
        checkpoint_root=tmp_path,
        completed_steps=1,
        optimizer=optimizer,
        training_config={"max_steps": 10, "learning_rate": 0.01},
        losses=[1.0],
        step_runtime=[],
        save_adapter=lambda path: _save_fake_adapter(path, {}),
    )
    try:
        restore_training_checkpoint(
            checkpoint_dir=saved,
            optimizer=optimizer,
            expected_config={"max_steps": 20, "learning_rate": 0.01},
        )
    except ValueError as exc:
        assert "configurazione Stage1 incompatibile" in str(exc)
    else:
        raise AssertionError("il resume ha accettato una configurazione Stage1 differente")


if __name__ == "__main__":
    with TemporaryDirectory(prefix="duplex-checkpoint-test-") as temp_dir:
        temp_root = Path(temp_dir)
        test_checkpoint_restores_adapter_optimizer_and_rng_for_identical_next_update(temp_root / "resume")
        test_checkpoint_does_not_overwrite_existing_step(temp_root / "collision")
        test_checkpoint_rejects_training_config_change(temp_root / "config")
    print("checkpoint CPU smoke: 3 tests PASS")
