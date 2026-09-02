#!/usr/bin/env python3
"""Run a precision candidate on the real 64-frame validation windows.

This is a diagnostic gate: the native full Gemma forward remains the oracle,
while the stateful 4-token/frame path is executed under one isolated candidate
patch.  No V9/Qwen code or oracle code is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

from duplex_kernel.gemma_representation import _audio_soft_tokens  # noqa: E402
from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path  # noqa: E402
from tools.kernel_gates.p4_numeric_bisect import _compare, _incremental  # noqa: E402
from tools.kernel_gates.p4_precision_ablations import NullCapture, _candidate  # noqa: E402
from tools.kernel_gates.runtime_dataset import aligned_validation_start  # noqa: E402


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] * (upper - rank) + ordered[upper] * (rank - lower)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validation_dataset(dataset: Path) -> tuple[Any, int]:
    import kaggle_pipeline as kp

    prefix = str(dataset).removesuffix(".parquet")
    user_mel, agent_mel, codes, text_by_window = kp.load_tokens_from_parquet(str(dataset))
    metadata_path = Path(prefix + "_hyperparams.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    val_start = aligned_validation_start(
        total_frames=int(codes.shape[1]),
        validation_fraction=float(metadata.get("VAL_FRACTION", 0.10)),
        segment_ends=(int(value) for value in metadata.get("SEGMENT_ENDS", [])),
        window_frames=64,
    )
    dataset = kp.FullDuplexDataset(
        user_mel[val_start * int(kp.MEL_PER_AGENT_FRAME):],
        agent_mel[val_start * int(kp.MEL_PER_AGENT_FRAME):],
        codes[:, val_start:],
        seq_len=64,
        text_by_window=text_by_window[val_start // 64 :],
    )
    return dataset, int(kp.TEXT_BLOCK_LEN), val_start


def run(
    *,
    dataset_path: Path,
    candidate_name: str,
    limit: int,
    batch_size: int,
    cache_path: Path | None,
    enforce_minimum: bool = True,
    start_index: int = 0,
) -> dict[str, Any]:
    ds, text_block_len, validation_start_frame = _validation_dataset(dataset_path)
    start_index = int(start_index)
    if start_index < 0 or start_index >= len(ds):
        raise ValueError(f"start-index fuori range: {start_index} / {len(ds)}")
    count = min(int(limit), len(ds) - start_index)
    if count <= 0 or (enforce_minimum and count < 100):
        raise ValueError(f"validation insufficiente: {count} sequenze, minimo 100")

    cache = torch.load(cache_path, map_location="cpu", weights_only=False) if cache_path else None
    cached = cache.get("va_raw") if cache else None
    if cached is not None and len(cached) < count:
        raise ValueError(f"cache validation insufficiente: {len(cached)} < {count}")

    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    audio_token_id = int(getattr(model.config, "audio_token_id", 258881))
    device = torch.device("cuda")
    sequence_lengths: set[int] = set()
    frame_maes: list[float] = []
    frame_maxima: list[float] = []
    frame_cosines: list[float] = []
    final_comparisons: list[dict[str, Any]] = []
    sequence_metrics: list[dict[str, Any]] = []
    audio_timings: list[float] = []
    peak_allocated = 0
    peak_reserved = 0
    cache_errors: list[float] = []
    cache_cosines: list[float] = []
    compared_sequences = 0
    compared_frames = 0
    started = time.perf_counter()

    try:
        for offset in range(0, count, batch_size):
            items = [ds[index] for index in range(start_index + offset, start_index + min(offset + batch_size, count))]
            batch = {
                key: torch.stack([item[key] for item in items]).to(device)
                for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
            }
            ids = batch["input_ids"]
            attention = batch["attention_mask"]
            features = batch["input_features"]
            feature_mask = batch["input_features_mask"]
            sequence_lengths.add(int(ids.shape[1]))
            positions = (ids[0] == audio_token_id).nonzero(as_tuple=False).flatten().tolist()
            if len(positions) != 256 or len(positions) % 4:
                raise ValueError(f"sequenza {offset}: placeholder audio inattesi: {len(positions)}")
            audio_start = int(positions[0])
            audio_soft = _audio_soft_tokens(
                model,
                input_features=features,
                input_features_mask=feature_mask,
            )
            with torch.inference_mode():
                full_output = model(
                    input_ids=ids,
                    attention_mask=attention,
                    input_features=features,
                    input_features_mask=feature_mask,
                    use_cache=True,
                    return_dict=True,
                )
            full = full_output.last_hidden_state[:, positions, :].float().cpu().contiguous()
            del full_output

            model.language_model._p4_padding_length = int(ids.shape[1])
            capture = NullCapture(set(positions))
            with _candidate(model.language_model, candidate_name) as applicable:
                if not applicable:
                    raise RuntimeError(f"candidate non applicabile: {candidate_name}")
                result = _incremental(
                    model,
                    mode_name=f"validation_{candidate_name}",
                    input_ids=ids,
                    attention_mask=attention,
                    audio_soft=audio_soft,
                    audio_start=audio_start,
                    capture=capture,
                )
            if capture.final is None:
                raise RuntimeError(f"candidate {candidate_name} non ha prodotto hidden finale")
            candidate = capture.final.float().cpu().contiguous()
            comparison = _compare(full, candidate)
            final_comparisons.append(comparison)
            for row in range(len(items)):
                row_full = full[row]
                row_candidate = candidate[row]
                row_error = (row_full - row_candidate).abs()
                row_cosine = torch.nn.functional.cosine_similarity(row_full, row_candidate, dim=-1, eps=1e-12)
                sequence_metrics.append({
                    "sequence_id": start_index + offset + row,
                    "mae": float(row_error.mean()),
                    "max_abs": float(row_error.max()),
                    "cosine_min": float(row_cosine.min()),
                })
            if comparison.get("shape_mismatch") or not comparison.get("available"):
                raise RuntimeError(f"shape/finite mismatch nella sequenza {offset}: {comparison}")

            full_frames = full[:, ::4, :]
            candidate_frames = candidate[:, ::4, :]
            error = (full_frames - candidate_frames).abs()
            cosine = torch.nn.functional.cosine_similarity(full_frames, candidate_frames, dim=-1, eps=1e-12)
            frame_maes.extend(error.mean(dim=-1).reshape(-1).tolist())
            frame_maxima.extend(error.amax(dim=-1).reshape(-1).tolist())
            frame_cosines.extend(cosine.reshape(-1).tolist())
            compared_sequences += len(items)
            compared_frames += int(full_frames.shape[0] * full_frames.shape[1])
            audio_timings.extend(float(value) for value in result.audio_ms)
            peak_allocated = max(peak_allocated, int(result.peak_allocated))
            peak_reserved = max(peak_reserved, int(result.peak_reserved))

            if cached is not None:
                for row in range(len(items)):
                    expected = torch.as_tensor(cached[start_index + offset + row], dtype=torch.float32)
                    cache_value = full[row, ::4, :]
                    cache_error = (cache_value - expected).abs()
                    cache_errors.append(float(cache_error.mean()))
                    cache_cosines.append(float(torch.nn.functional.cosine_similarity(cache_value, expected, dim=-1, eps=1e-12).min()))
            del batch, ids, attention, features, feature_mask, audio_soft, full, candidate
            torch.cuda.empty_cache()
            print(f"[P4-CANDIDATE] {compared_sequences}/{count}", flush=True)

        thresholds = {
            "sequences_min": 100,
            "frames_min": 5000,
            "mae_max": 1e-3,
            "cosine_min": 0.9999,
            "nonfinite_max": 0,
            "shape_mismatches_max": 0,
            "audio_p95_ms_max": 80.0,
        }
        mae = sum(frame_maes) / len(frame_maes) if frame_maes else float("inf")
        max_abs = max(frame_maxima) if frame_maxima else float("inf")
        cosine_min = min(frame_cosines) if frame_cosines else -1.0
        audio_p50 = _percentile(audio_timings, 50)
        audio_p95 = _percentile(audio_timings, 95)
        passed = bool(
            compared_sequences >= thresholds["sequences_min"]
            and compared_frames >= thresholds["frames_min"]
            and mae <= thresholds["mae_max"]
            and cosine_min >= thresholds["cosine_min"]
            and audio_p95 is not None
            and audio_p95 < thresholds["audio_p95_ms_max"]
            and all(torch.isfinite(torch.tensor(value)) for value in (*frame_maes, *frame_maxima, *frame_cosines))
        )
        report = {
            "schema_version": 1,
            "phase": "P4",
            "gate": "candidate_real_validation",
            "status": "diagnostic",
            "candidate": candidate_name,
            "oracle": "untouched native Gemma4Model.forward BF16",
            "contract": "real FullDuplexDataset validation windows; stable prefill + 4 audio soft tokens/frame + native masks/cache",
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path) if dataset_path.is_file() else "derived-sidecars-attested",
            "text_block_len": text_block_len,
            "validation_start_frame": validation_start_frame,
            "validation_text_window_offset": validation_start_frame // 64,
            "start_index": start_index,
            "sequence_lengths": sorted(sequence_lengths),
            "sequence_metrics": sequence_metrics,
            "model": {
                "path": resolve_thinker_path("/workspace"),
                "dtype": "bfloat16",
                "frozen": True,
            },
            "metrics": {
                "sequences": compared_sequences,
                "frames": compared_frames,
                "mae": mae,
                "max_abs": max_abs,
                "cosine_min": cosine_min,
                "nonfinite_count": 0,
                "shape_mismatches": 0,
                "audio_p50_ms": audio_p50,
                "audio_p95_ms": audio_p95,
                "audio_samples": len(audio_timings),
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "full_validation_wall_ms": (time.perf_counter() - started) * 1000.0,
            },
            "cache_crosscheck": {
                "provided": cache is not None,
                "path": str(cache_path) if cache_path else None,
                "sha256": _sha256(cache_path) if cache_path else None,
                "mae": sum(cache_errors) / len(cache_errors) if cache_errors else None,
                "cosine_min": min(cache_cosines) if cache_cosines else None,
            },
            "thresholds": thresholds,
            "passed": passed,
        }
        return report
    finally:
        del thinker, model
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    parser.add_argument("--candidate", default="shape_padded_rmsnorm_kv")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--allow-small", action="store_true", help="solo diagnostica; non chiude il gate da 100 sequenze")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.limit <= 0 or args.batch_size <= 0:
        parser.error("limit e batch-size devono essere positivi")
    report = run(
        dataset_path=args.dataset,
        candidate_name=args.candidate,
        limit=args.limit,
        batch_size=args.batch_size,
        cache_path=args.cache,
        enforce_minimum=not args.allow_small,
        start_index=args.start_index,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
