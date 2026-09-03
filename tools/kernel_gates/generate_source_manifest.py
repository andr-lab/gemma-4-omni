#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/kernel_gates/generate_source_manifest.py — Generate cryptographic source manifest for P4 Final.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_manifest() -> dict:
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        git_sha = "unknown"

    try:
        git_dirty_raw = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(ROOT), text=True).strip()
        git_dirty = bool(git_dirty_raw)
    except Exception:
        git_dirty = True

    source_files = {}
    for p in sorted(ROOT.glob("duplex_kernel/**/*.py")):
        rel = p.relative_to(ROOT).as_posix()
        source_files[rel] = sha256_file(p)

    for p in sorted(ROOT.glob("tools/kernel_gates/**/*.py")):
        rel = p.relative_to(ROOT).as_posix()
        source_files[rel] = sha256_file(p)

    manifest = {
        "schema_version": 1,
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": source_files,
        "source_file_count": len(source_files),
    }

    out_dir = ROOT / "validation" / "kernel_gates" / "P4_FINAL"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "source_manifest.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    manifest_sha = sha256_file(out_path)
    print(f"[OK] Source manifest generated: {out_path}")
    print(f"   Git SHA: {git_sha} (dirty: {git_dirty})")
    print(f"   Files hashed: {len(source_files)}")
    print(f"   Manifest SHA-256: {manifest_sha}")

    return manifest


if __name__ == "__main__":
    generate_manifest()
