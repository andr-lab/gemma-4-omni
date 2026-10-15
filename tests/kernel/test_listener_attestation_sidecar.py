from __future__ import annotations

import json
from pathlib import Path

import pytest

from duplex_kernel import listener_attestation


def test_sidecar_preserves_only_actual_listener_identity_fields(tmp_path: Path) -> None:
    attestation_dir = tmp_path / "run" / "listener_attestations"
    runtime_info = {
        "library_path": "/workspace/.tmp/listener_candidate.so",
        "library_sha256": "a" * 64,
        "abi_version": 3,
        "attention_backend": "NATIVE_SPLITK_GQA",
    }

    path = listener_attestation.write_listener_attestation_sidecar(
        attestation_dir,
        workspace_root=tmp_path,
        session_id="session/A",
        runtime_info=runtime_info,
    )

    assert path == attestation_dir / listener_attestation.listener_attestation_filename("session/A")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": 1,
        "session_id": "session/A",
        "library_path": runtime_info["library_path"],
        "library_sha256": runtime_info["library_sha256"],
        "abi_version": 3,
    }
    assert sorted(item.name for item in attestation_dir.iterdir()) == [path.name]
    with pytest.raises(FileExistsError, match="already exists"):
        listener_attestation.write_listener_attestation_sidecar(
            attestation_dir,
            workspace_root=tmp_path,
            session_id="session/A",
            runtime_info={"library_path": "/wrong.so", "library_sha256": "f" * 64},
        )
    assert json.loads(path.read_text(encoding="utf-8")) == payload


def test_sidecar_environment_absent_performs_no_file_io(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(listener_attestation.LISTENER_ATTESTATION_DIRECTORY_ENV, raising=False)
    assert listener_attestation.write_listener_attestation_from_environment(
        "session-a", {"library_path": "/candidate.so", "library_sha256": "b" * 64},
        workspace_root=tmp_path,
    ) is None
    assert list(tmp_path.iterdir()) == []


def test_sidecar_rejects_directory_outside_workspace(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    outside = tmp_path / "outside"
    with pytest.raises(ValueError, match="inside the workspace"):
        listener_attestation.write_listener_attestation_sidecar(
            outside,
            workspace_root=workspace_root,
            session_id="session-a",
            runtime_info={"library_path": "/candidate.so", "library_sha256": "c" * 64},
        )
    assert not outside.exists()


def test_sidecar_atomic_publish_failure_leaves_no_attestation_or_temp_file(
    monkeypatch, tmp_path: Path,
) -> None:
    attestation_dir = tmp_path / "run" / "listener_attestations"

    def fail_link(*_args, **_kwargs):
        raise OSError("simulated atomic publish failure")

    monkeypatch.setattr(listener_attestation.os, "replace", fail_link)
    with pytest.raises(OSError, match="simulated atomic publish failure"):
        listener_attestation.write_listener_attestation_sidecar(
            attestation_dir,
            workspace_root=tmp_path,
            session_id="session-a",
            runtime_info={"library_path": "/candidate.so", "library_sha256": "d" * 64},
        )
    assert list(attestation_dir.iterdir()) == []
