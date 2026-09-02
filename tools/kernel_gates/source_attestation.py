"""Content-addressed source attestation for kernel gate artifacts.

The attestation is valid only for a clean Git checkout.  The verifier checks
Git object bytes and blob IDs, so a JSON summary cannot silently claim a
different source revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_FILES = (
    "duplex_kernel/__init__.py",
    "duplex_kernel/config.py",
    "duplex_kernel/engine.py",
    "duplex_kernel/kv_cache.py",
    "duplex_kernel/listener.py",
    "duplex_kernel/server.py",
    "duplex_kernel/session.py",
    "duplex_kernel/scheduler.py",
    "duplex_kernel/turn_policy.py",
    "duplex_kernel/turnhead.py",
    "duplex_kernel/turnhead_adapter.py",
    "duplex_kernel/models/thinker.py",
    "duplex_kernel/models/__init__.py",
    "duplex_kernel/models/v9.py",
    "duplex_kernel/models/talker.py",
    "duplex_kernel/playout.py",
    "duplex_kernel/future_prefix.py",
    "duplex_kernel/gemma_representation.py",
    "tools/extract_runtime_cache.py",
    "tools/kernel_gates/representation_parity.py",
    "tools/kernel_gates/runtime_dataset.py",
    "tests/kernel/test_runtime_dataset.py",
    "tests/kernel/test_p4d_runtime.py",
    "tools/kernel_gates/build_manifest.py",
    "tools/kernel_gates/p1_checks.py",
    "tools/kernel_gates/p2_checks.py",
    "tools/kernel_gates/p3_checks.py",
    "tools/extract_ars_cache_v2.py",
    "tools/kernel_gates/p4_checks.py",
    "tools/kernel_gates/run_phase.py",
    "tools/kernel_gates/source_attestation.py",
    "tools/kernel_gates/verify_report.py",
    "tools/kernel_gates/p4d_checks.py",
    "tools/kernel_gates/verify_p4d_report.py",
    "tests/test_duplex_kernel_contract.py",
    "tests/kernel/test_p1_control_plane.py",
    "tests/kernel/test_p2_scheduler.py",
    "tests/kernel/test_p3_listener.py",
    "tests/kernel/test_p4_v9_incremental.py",
    "tests/kernel/test_p4d_playout.py",
    "tests/kernel/test_gemma_pcm_mel.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def verify_workspace_attestation(attestation_path: Path, workspace: Path) -> dict[str, Any]:
    """Bind an attestation to the exact bytes in the runtime workspace."""
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    if attestation.get("schema_version") != 1 or attestation.get("git_dirty") is not False:
        raise ValueError("source attestation non valida o dirty")
    workspace = workspace.resolve()
    verified_files = []
    for relative, expected in attestation.get("files", {}).items():
        path = (workspace / relative).resolve()
        if workspace not in path.parents or not path.is_file():
            raise FileNotFoundError(f"workspace file non attestato: {relative}")
        data = path.read_bytes()
        actual_sha256 = sha256_bytes(data)
        actual_blob = git_blob_sha1(data)
        if len(data) != expected.get("bytes") or actual_sha256 != expected.get("sha256"):
            raise ValueError(f"byte workspace diversi da attestation: {relative}")
        if actual_blob != expected.get("git_blob_sha1"):
            raise ValueError(f"Git blob workspace diversa da attestation: {relative}")
        verified_files.append({"path": relative, "bytes": len(data), "sha256": actual_sha256,
                               "git_blob_sha1": actual_blob})
    if not verified_files:
        raise ValueError("source attestation senza file")
    return {
        "workspace": str(workspace),
        "attestation_git_sha": attestation["git_sha"],
        "files_verified": len(verified_files),
        "passed": True,
        "files": verified_files,
    }


def build_attestation(repo: Path, files: tuple[str, ...] = DEFAULT_FILES) -> dict[str, Any]:
    repo = repo.resolve()
    git_sha = _git(repo, "rev-parse", "HEAD")
    dirty = bool(_git(repo, "status", "--porcelain"))
    entries: dict[str, Any] = {}
    for relative in files:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        data = path.read_bytes()
        git_data = subprocess.check_output(("git", "show", f"{git_sha}:{relative}"), cwd=repo)
        if data != git_data:
            raise RuntimeError(f"worktree non coincide con HEAD per {relative}")
        entries[relative] = {
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "git_blob_sha1": git_blob_sha1(data),
            "git_show_sha256": sha256_bytes(git_data),
        }
    return {
        "schema_version": 1,
        "attestation_type": "git-clean-worktree-content-addressed",
        "git_sha": git_sha,
        "git_dirty": dirty,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attestation = build_attestation(args.repo)
    if attestation["git_dirty"]:
        raise SystemExit("source attestation rifiutata: checkout Git dirty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # write_bytes avoids Windows newline translation: this digest is part of
    # the report contract and must be identical on Windows, Linux and CI.
    payload = (json.dumps(attestation, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
