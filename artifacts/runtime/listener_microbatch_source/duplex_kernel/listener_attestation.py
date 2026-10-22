"""Proof-only atomic sidecars for per-session native listener identity."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid


LISTENER_ATTESTATION_DIRECTORY_ENV = "PAIR_DUPLEX_LISTENER_ATTESTATION_DIR"


def listener_attestation_filename(session_id: str) -> str:
    if not isinstance(session_id, str) or not session_id or len(session_id) > 512:
        raise ValueError("listener attestation requires a nonempty bounded session ID")
    if any(ord(character) < 32 or ord(character) == 127 for character in session_id):
        raise ValueError("listener attestation session ID contains control characters")
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest() + ".json"


def write_listener_attestation_sidecar(
    directory: str | Path,
    *,
    workspace_root: str | Path,
    session_id: str,
    runtime_info: Mapping[str, Any],
) -> Path:
    """Atomically publish only the real listener library identity for one session."""
    if not isinstance(runtime_info, Mapping):
        raise TypeError("native listener runtime info must be a mapping")
    filename = listener_attestation_filename(session_id)
    root = Path(workspace_root).resolve()
    if not root.is_dir():
        raise ValueError("listener attestation workspace root must be an existing directory")
    requested_directory = Path(directory)
    if not requested_directory.is_absolute():
        raise ValueError("listener attestation directory must be absolute")
    resolved_directory = requested_directory.resolve()
    if resolved_directory == root or root not in resolved_directory.parents:
        raise ValueError("listener attestation directory must be inside the workspace")

    resolved_directory.mkdir(parents=True, exist_ok=True)
    resolved_directory = resolved_directory.resolve()
    if resolved_directory == root or root not in resolved_directory.parents:
        raise ValueError("listener attestation directory escaped the workspace")

    record: dict[str, Any] = {"schema_version": 1, "session_id": session_id}
    available = runtime_info.get("available")
    if type(available) is bool:
        record["available"] = available
    for key in ("library_path", "library_sha256"):
        if key in runtime_info:
            value = runtime_info[key]
            if value is not None and not isinstance(value, str):
                raise TypeError(f"native listener {key} must be a string or null")
            record[key] = value
    if "abi_version" in runtime_info:
        abi_version = runtime_info["abi_version"]
        if type(abi_version) is not int or abi_version < 0:
            raise TypeError("native listener abi_version must be a nonnegative integer")
        record["abi_version"] = abi_version

    destination = resolved_directory / filename
    if destination.exists():
        raise FileExistsError(f"listener attestation already exists for session {session_id!r}")
    temporary: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".tmp", dir=resolved_directory,
        )
        temporary = Path(temporary_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(record, handle, ensure_ascii=False, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # Same-directory replace publishes the complete fsynced JSON atomically.
        os.replace(temporary, destination)
        temporary = None
        return destination
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_listener_attestation_from_environment(
    session_id: str,
    runtime_info: Mapping[str, Any],
    *,
    workspace_root: str | Path = "/workspace",
) -> Path | None:
    """Write a sidecar only when the proof-only environment setting is present."""
    directory = os.environ.get(LISTENER_ATTESTATION_DIRECTORY_ENV)
    if directory is None:
        return None
    if not directory.strip():
        raise ValueError(f"{LISTENER_ATTESTATION_DIRECTORY_ENV} cannot be empty")
    return write_listener_attestation_sidecar(
        directory,
        workspace_root=workspace_root,
        session_id=session_id,
        runtime_info=runtime_info,
    )


__all__ = [
    "LISTENER_ATTESTATION_DIRECTORY_ENV",
    "listener_attestation_filename",
    "write_listener_attestation_from_environment",
    "write_listener_attestation_sidecar",
]
