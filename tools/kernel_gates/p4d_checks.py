#!/usr/bin/env python3
"""P4D frozen-Qwen future-conditioning experiments.

This module deliberately does not modify Qwen.  It measures whether a causal
V9 prefix can recover the original P4-T4 quality gate by delaying only speaker
playout: frame t is evaluated after ``lookahead_frames`` additional V9 frames
are available.  Anchor variants are also evaluated independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import IncrementalV9Projector, load_v9_conditioning
from tools.kernel_gates.p4_checks import (
    ABLATION_CE_RATIO_MAX,
    ABLATION_TOP1_DROP_MAX,
    ABLATION_TOP5_DROP_MAX,
    _load_codes,
    _stack_cache,
    _run_qwen_ablation,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_files(workspace: Path) -> list[dict[str, Any]]:
    """Return immutable Qwen checkpoint files visible in the runtime cache."""
    # The container runs as an unprivileged user; inspect only the mounted
    # workspace, where the configured HF cache is available.
    roots = [workspace, workspace / ".hf_cache", workspace / ".cache"]
    candidates = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.glob("**/*Qwen3-TTS-12Hz-1.7B-Base*/snapshots/**/*.safetensors")
        if path.is_file()
    )
    # A mounted workspace can expose the same snapshot through several paths.
    unique: dict[str, Path] = {}
    for path in candidates:
        unique.setdefault(_sha256(path), path)
    return [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
        for digest, path in sorted(unique.items(), key=lambda item: str(item[1]))
    ]


def _anchor_variants(raw: torch.Tensor, anchors: torch.Tensor) -> dict[str, torch.Tensor]:
    # ``oracle_text`` is the P4 diagnostic baseline.  The other anchors are
    # available from the current/past audio only and therefore causal.
    first = raw[:, 0, :].to(dtype=anchors.dtype).clone()
    variants = {
        "oracle_text": anchors,
        "zero": torch.zeros_like(anchors),
        "first_raw_frame": first,
    }
    # No learned parameter is introduced: these are deterministic causal
    # anchor calibrations, useful to find whether a short playout is enough.
    for scale in (0.5, 0.75, 1.25, 1.5, 2.0):
        variants[f"first_raw_frame_x{scale:g}"] = first * scale
    return variants


def _quality(metrics: dict[str, Any]) -> dict[str, Any]:
    ratio = float(metrics["causal_ce"]) / max(float(metrics["full_ce"]), 1e-12)
    top1_drop = float(metrics["full_top1"]) - float(metrics["causal_top1"])
    top5_drop = float(metrics["full_top5"]) - float(metrics["causal_top5"])
    passed = bool(
        metrics["sequences"] >= 100
        and metrics["frames"] >= 5_000
        and metrics["finite"]
        and metrics["mask_future_prefix"]
        and float(metrics["max_logit_delta"]) > 1e-6
        and ratio <= ABLATION_CE_RATIO_MAX
        and top1_drop <= ABLATION_TOP1_DROP_MAX
        and top5_drop <= ABLATION_TOP5_DROP_MAX
    )
    return {
        **metrics,
        "ce_ratio": ratio,
        "top1_drop": top1_drop,
        "top5_drop": top5_drop,
        "thresholds": {
            "ce_ratio_max": ABLATION_CE_RATIO_MAX,
            "top1_drop_max": ABLATION_TOP1_DROP_MAX,
            "top5_drop_max": ABLATION_TOP5_DROP_MAX,
        },
        "passed": passed,
    }


def run_experiment(
    *,
    workspace: Path,
    cache_path: Path,
    codes_path: Path,
    codes_metadata_path: Path | None,
    lookaheads: list[int],
    anchor_names: list[str],
    output_dir: Path,
    git_sha: str = "attested-by-caller",
    source_attestation: Path | None = None,
) -> dict[str, Any]:
    raw, anchors, cache_codes, raw_stage = _stack_cache(cache_path)
    if cache_codes is None:
        raise ValueError("cache validation priva di va_c")
    codes, codes_start = _load_codes(codes_path, codes_metadata_path, cache_codes)

    projector, causal_conv = load_v9_conditioning(device="cuda", dtype=torch.bfloat16)
    stream = IncrementalV9Projector(projector, causal_conv, max_seq_len=16_384)
    talker = load_talker(str(workspace), device="cuda")
    talker.talker.eval()
    talker.talker.requires_grad_(False)
    qwen_trainable = [name for name, parameter in talker.talker.named_parameters() if parameter.requires_grad]
    checkpoint_files = _checkpoint_files(workspace)
    if not checkpoint_files:
        raise RuntimeError("Qwen checkpoint safetensors non trovati: impossibile attestare byte-identità")

    variants: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    try:
        anchor_map = _anchor_variants(raw, anchors)
        for anchor_name in anchor_names:
            if anchor_name not in anchor_map:
                raise ValueError(f"anchor non supportato: {anchor_name}")
            for lookahead in lookaheads:
                if lookahead < 0:
                    raise ValueError("lookahead non può essere negativo")
                experiment_id = f"{anchor_name}_L{lookahead}"
                local_trace: list[dict[str, Any]] = []
                metrics = _run_qwen_ablation(
                    talker,
                    stream,
                    raw,
                    anchor_map[anchor_name],
                    codes,
                    trace=local_trace,
                    prefix_lookahead=lookahead,
                    event_context={
                        "experiment_id": experiment_id,
                        "anchor_mode": anchor_name,
                        "anchor_causal": anchor_name != "oracle_text",
                    },
                )
                metrics = _quality(metrics)
                metrics.update({
                    "experiment_id": experiment_id,
                    "anchor_mode": anchor_name,
                    "anchor_causal": anchor_name != "oracle_text",
                    "lookahead_frames": lookahead,
                    "playout_delay_ms": lookahead * 80,
                    "codes_validation_start": codes_start,
                })
                variants[experiment_id] = metrics
                trace.extend(local_trace)
                print(
                    f"{experiment_id}: CE ratio={metrics['ce_ratio']:.4f} "
                    f"top1_drop={metrics['top1_drop']:.4f} "
                    f"top5_drop={metrics['top5_drop']:.4f} passed={metrics['passed']}",
                    flush=True,
                )
    finally:
        del talker, projector, causal_conv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_trace.jsonl"
    raw_path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in trace), encoding="utf-8")
    report = {
        "schema_version": 1,
        "phase": "P4D",
        "git_sha": git_sha,
        "git_dirty": False,
        "platform": platform.platform(),
        "real_model_required": True,
        "qwen": {
            "checkpoint_files": checkpoint_files,
            "trainable_parameter_names": qwen_trainable,
            "frozen": not qwen_trainable,
            "weights_modified": False,
            "architecture": "Qwen3TTSTalkerModel",
        },
        "inputs": {
            "cache_path": str(cache_path),
            "raw_stage": raw_stage,
            "causal_conv_applied_by_runner": True,
            "codes_path": str(codes_path),
            "sequences": int(raw.shape[0]),
            "frames_per_sequence": int(raw.shape[1]),
            "lookahead_frames": lookaheads,
            "anchor_modes": anchor_names,
            "codes_validation_start": codes_start,
        },
        "variants": variants,
        "raw_trace": raw_path.name,
        "raw_trace_sha256": _sha256(raw_path),
        "passed": any(metrics["passed"] for metrics in variants.values()),
    }
    if source_attestation is not None:
        destination = output_dir / "source_attestation.json"
        destination.write_bytes(source_attestation.read_bytes())
        report["source_attestation"] = destination.name
        report["source_attestation_sha256"] = _sha256(destination)
    (output_dir / "P4D_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--codes", type=Path, required=True)
    parser.add_argument("--codes-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--git-sha", default="attested-by-caller")
    parser.add_argument("--source-attestation", type=Path, default=None)
    parser.add_argument("--lookaheads", default="0,1,2,4")
    parser.add_argument("--anchors", default="oracle_text,zero,first_raw_frame")
    args = parser.parse_args()
    report = run_experiment(
        workspace=args.workspace,
        cache_path=args.cache,
        codes_path=args.codes,
        codes_metadata_path=args.codes_metadata,
        lookaheads=[int(item) for item in args.lookaheads.split(",") if item.strip()],
        anchor_names=[item.strip() for item in args.anchors.split(",") if item.strip()],
        output_dir=args.output_dir,
        git_sha=args.git_sha,
        source_attestation=args.source_attestation,
    )
    print(f"report={args.output_dir / 'P4D_report.json'} passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
