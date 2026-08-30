"""Content-addressed source attestation for kernel gate artifacts.

The attestation is valid only for a clean Git checkout.  The verifier compares
both the worktree bytes and Git blob IDs, so a JSON summary cannot silently
claim a different source revision.
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
    "duplex_kernel/server.py",
    "duplex_kernel/session.py",
    "duplex_kernel/turn_policy.py",
    "duplex_kernel/turnhead.py",
    "duplex_kernel/turnhead_adapter.py",
    "duplex_kernel/models/thinker.py",
    "duplex_kernel/models/v9.py",
    "duplex_kernel/models/talker.py",
    "tools/kernel_gates/build_manifest.py",
    "tools/kernel_gates/p1_checks.py",
    "tools/kernel_gates/run_phase.py",
    "tools/kernel_gates/verify_report.py",
    "tests/test_duplex_kernel_contract.py",
    "tests/kernel/test_p1_control_plane.py",
    "tests/kernel/test_gemma_pcm_mel.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=repo, text=True, stderr=subprocess.STDOUT).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


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
    args.output.write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
