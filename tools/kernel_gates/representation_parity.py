#!/usr/bin/env python3
"""Independent Gemma representation-parity gate.

The cache under test is produced by the serving-model extraction path.  The
oracle in this gate does not call the incremental listener or its helper: it
invokes the native ``Gemma4Model.forward`` directly and selects audio slots
from the returned final LM hidden state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_split(dataset: Path) -> tuple[Any, int]:
    import kaggle_pipeline as kp

    prefix = str(dataset).removesuffix(".parquet")
    user_mel, agent_mel, codes, text_by_window = kp.load_tokens_from_parquet(str(dataset))
    sidecar = Path(prefix + "_hyperparams.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    val_start = int(0.90 * codes.shape[1])
    for end in (int(value) for value in metadata.get("SEGMENT_ENDS", [])):
        if end >= val_start:
            val_start = end
            break
    mel_per_frame = int(kp.MEL_PER_AGENT_FRAME)
    ds = kp.FullDuplexDataset(
        user_mel[val_start * mel_per_frame :],
        agent_mel[val_start * mel_per_frame :],
        codes[:, val_start:],
        seq_len=64,
        text_by_window=text_by_window,
    )
    return ds, int(kp.TEXT_BLOCK_LEN)


def run(
    *,
    dataset: Path,
    cache_path: Path,
    output_dir: Path,
    batch_size: int,
    limit: int,
    git_sha: str,
) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("v9_raw_stage") != "pre_causal_conv":
        raise ValueError("representation cache non attestata come pre_causal_conv")
    cached = cache.get("va_raw")
    if not isinstance(cached, list) or not cached:
        raise ValueError("representation cache priva di va_raw")

    ds, text_block_len = _dataset_split(dataset)
    count = min(int(limit), len(ds), len(cached))
    if count <= 0:
        raise ValueError("nessuna sequenza validation disponibile")

    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    device = torch.device("cuda")
    audio_token_id = int(getattr(model.config, "audio_token_id", 258881))
    audio_features_per_frame = 4
    trace: list[dict[str, Any]] = []
    abs_errors: list[torch.Tensor] = []
    cosines: list[float] = []
    shape_mismatches = 0
    nonfinite_count = 0
    compared_frames = 0

    for offset in range(0, count, batch_size):
        items = [ds[index] for index in range(offset, min(offset + batch_size, count))]
        batch = {
            key: torch.stack([item[key] for item in items]).to(device)
            for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
        }
        with torch.inference_mode():
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_features=batch["input_features"],
                input_features_mask=batch["input_features_mask"],
                use_cache=True,
                return_dict=True,
            )
        hidden = output.last_hidden_state
        for row, item in enumerate(items):
            positions = (batch["input_ids"][row] == audio_token_id).nonzero(as_tuple=False).flatten()
            expected = cached[offset + row]
            expected_tensor = torch.as_tensor(expected, dtype=torch.float32)
            if positions.numel() % audio_features_per_frame:
                shape_mismatches += 1
                trace.append({
                    "event": "representation_parity_sequence",
                    "sequence_id": offset + row,
                    "shape_mismatch": True,
                    "audio_placeholder_count": int(positions.numel()),
                })
                continue
            selected = hidden[row, positions[::audio_features_per_frame], :].float().cpu()
            if selected.shape != expected_tensor.shape:
                shape_mismatches += 1
                trace.append({
                    "event": "representation_parity_sequence",
                    "sequence_id": offset + row,
                    "shape_mismatch": True,
                    "oracle_shape": list(selected.shape),
                    "cache_shape": list(expected_tensor.shape),
                })
                continue
            if not torch.isfinite(selected).all() or not torch.isfinite(expected_tensor).all():
                nonfinite_count += int((~torch.isfinite(selected)).sum() + (~torch.isfinite(expected_tensor)).sum())
            error = (selected - expected_tensor).abs()
            frame_cos = torch.nn.functional.cosine_similarity(
                selected, expected_tensor, dim=-1, eps=1e-12
            )
            abs_errors.append(error.reshape(-1))
            cosines.extend(frame_cos.tolist())
            compared_frames += int(selected.shape[0])
            trace.append({
                "event": "representation_parity_sequence",
                "sequence_id": offset + row,
                "frames": int(selected.shape[0]),
                "mae": float(error.mean()),
                "max_abs": float(error.max()),
                "cosine_min": float(frame_cos.min()),
                "finite": bool(torch.isfinite(selected).all() and torch.isfinite(expected_tensor).all()),
                "shape_mismatch": False,
            })

    all_errors = torch.cat(abs_errors) if abs_errors else torch.empty(0)
    mae = float(all_errors.mean()) if all_errors.numel() else float("inf")
    max_abs = float(all_errors.max()) if all_errors.numel() else float("inf")
    cosine_min = min(cosines) if cosines else -1.0
    thresholds = {
        "sequences_min": 100,
        "frames_min": 5000,
        "mae_max": 1e-3,
        "cosine_min": 0.9999,
        "nonfinite_max": 0,
        "shape_mismatches_max": 0,
    }
    passed = bool(
        count >= thresholds["sequences_min"]
        and compared_frames >= thresholds["frames_min"]
        and mae <= thresholds["mae_max"]
        and cosine_min >= thresholds["cosine_min"]
        and nonfinite_count <= thresholds["nonfinite_max"]
        and shape_mismatches <= thresholds["shape_mismatches_max"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_trace.jsonl"
    raw_path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in trace), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "phase": "P4",
        "gate": "representation_parity",
        "git_sha": git_sha,
        "git_dirty": False,
        "platform": platform.platform(),
        "real_model_required": True,
        "oracle": {
            "implementation": "native Gemma4Model.forward",
            "selection": "final_layer_hidden_at_audio_position_before_v9_causal_conv",
            "masks": "Gemma4Model.forward create_masks_for_generate sliding_attention/full_attention",
            "audio_token_id": audio_token_id,
            "text_block_len": text_block_len,
            "audio_features_per_frame": audio_features_per_frame,
        },
        "model": {
            "path": resolve_thinker_path("/workspace"),
            "backend": "Gemma4Model",
            "dtype": "bfloat16",
            "frozen": True,
        },
        "cache": {
            "path": str(cache_path),
            "sha256": sha256(cache_path),
            "raw_stage": cache.get("v9_raw_stage"),
        },
        "metrics": {
            "sequences": count,
            "frames": compared_frames,
            "mae": mae,
            "max_abs": max_abs,
            "cosine_min": cosine_min,
            "nonfinite_count": nonfinite_count,
            "shape_mismatches": shape_mismatches,
        },
        "thresholds": thresholds,
        "raw_trace": raw_path.name,
        "raw_trace_sha256": sha256(raw_path),
        "passed": passed,
    }
    (output_dir / "P4_REPRESENTATION_PARITY.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--git-sha", default="attested-by-caller")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit <= 0:
        parser.error("batch-size e limit devono essere positivi")
    report = run(
        dataset=args.dataset,
        cache_path=args.cache,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        git_sha=args.git_sha,
    )
    print(
        f"sequences={report['metrics']['sequences']} frames={report['metrics']['frames']} "
        f"mae={report['metrics']['mae']:.8f} cosine_min={report['metrics']['cosine_min']:.8f} "
        f"passed={report['passed']}"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
