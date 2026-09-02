#!/usr/bin/env python3
"""Extract the canonical Gemma representation with the serving model.

This is intentionally separate from ``extract_ars_cache_v2.py``.  The latter
loads the training/QLoRA pipeline (and therefore may select a PEFT/Unsloth
base), while P4/P4D runtime evidence must consume the exact native thinker
loaded by ``duplex_kernel.models.thinker.load_thinker``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

from duplex_kernel.gemma_representation import canonical_full_lm_hidden  # noqa: E402
from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path  # noqa: E402
from tools.kernel_gates.runtime_dataset import aligned_validation_start  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet",
    )
    parser.add_argument("--output", default="/workspace/cache_ars_runtime_v1.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.limit <= 0:
        parser.error("batch-size e limit devono essere positivi")

    # Importing the dataset definitions is safe; unlike
    # load_gemma4_duplex_system(), this does not instantiate the training model.
    import kaggle_pipeline as kp

    dataset = Path(args.dataset)
    prefix = str(dataset).removesuffix(".parquet")
    print("[RUNTIME-CACHE] loading dataset", flush=True)
    user_mel, agent_mel, agent_codes, text_by_window = kp.load_tokens_from_parquet(str(dataset))
    sidecar = Path(prefix + "_hyperparams.json")
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    segment_ends = [int(value) for value in metadata.get("SEGMENT_ENDS", [])]
    validation_fraction = float(metadata.get("VAL_FRACTION", 0.10))
    # The text sidecar is indexed in fixed 64-frame windows.  Never start a
    # validation dataset mid-window: that pairs audio after the split with
    # text from an unrelated earlier window.
    val_start = aligned_validation_start(
        total_frames=int(agent_codes.shape[1]),
        validation_fraction=validation_fraction,
        segment_ends=segment_ends,
        window_frames=64,
    )
    val_text_by_window = text_by_window[val_start // 64 :]

    mel_per_frame = int(kp.MEL_PER_AGENT_FRAME)
    train_ds = kp.FullDuplexDataset(
        user_mel[: val_start * mel_per_frame],
        agent_mel[: val_start * mel_per_frame],
        agent_codes[:, :val_start],
        seq_len=64,
        text_by_window=text_by_window,
    )
    val_ds = kp.FullDuplexDataset(
        user_mel[val_start * mel_per_frame :],
        agent_mel[val_start * mel_per_frame :],
        agent_codes[:, val_start:],
        seq_len=64,
        text_by_window=val_text_by_window,
    )

    print(
        f"[RUNTIME-CACHE] validation_start={val_start} "
        f"text_window_offset={val_start // 64}",
        flush=True,
    )
    print("[RUNTIME-CACHE] loading native thinker", flush=True)
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    device = torch.device("cuda")
    fpf = int(kp.FEATURES_PER_AGENT_FRAME if kp.SELF_LISTENING else kp.AGENT_FEATURES_PER_FRAME)
    audio_start = 2 + int(kp.TEXT_BLOCK_LEN)
    print(
        f"[RUNTIME-CACHE] model={resolve_thinker_path('/workspace')} "
        f"text_block={kp.TEXT_BLOCK_LEN} audio_start={audio_start} fpf={fpf}",
        flush=True,
    )

    def extract(ds: object, limit: int) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        raw: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        text_hidden: list[torch.Tensor] = []
        count = min(limit, len(ds))
        for offset in range(0, count, args.batch_size):
            items = [ds[index] for index in range(offset, min(offset + args.batch_size, count))]
            batch = {
                key: torch.stack([item[key] for item in items]).to(device)
                for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
            }
            hidden, _ = canonical_full_lm_hidden(
                model,
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                input_features=batch["input_features"],
                input_features_mask=batch["input_features_mask"],
            )
            frames = int(items[0]["audio_targets"].shape[0])
            selected = hidden[:, audio_start : audio_start + fpf * frames : fpf, :].float()
            if selected.shape[1] != frames or selected.shape[2] != 1536:
                raise RuntimeError(f"runtime cache shape inattesa: {tuple(selected.shape)}")
            text = hidden[:, 1 : 1 + int(kp.TEXT_BLOCK_LEN), :]
            token_ids = batch["input_ids"][:, 1 : 1 + int(kp.TEXT_BLOCK_LEN)]
            valid = token_ids.ne(0).unsqueeze(-1).to(text.dtype)
            pooled = (text * valid).sum(1) / valid.sum(1).clamp_min(1.0)
            for index, item in enumerate(items):
                raw.append(selected[index].cpu())
                targets.append(item["audio_targets"][:, 0].clone().cpu())
                text_hidden.append(pooled[index].cpu())
            print(f"[RUNTIME-CACHE] {len(raw)}/{count}", flush=True)
        return raw, targets, text_hidden

    train_raw: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    train_text: list[torch.Tensor] = []
    if not args.validation_only:
        train_raw, train_targets, train_text = extract(train_ds, len(train_ds))
    val_raw, val_targets, val_text = extract(val_ds, args.limit)

    output = Path(args.output)
    torch.save(
        {
            "tr_raw": train_raw,
            "tr_sub": [],
            "tr_htx": train_text,
            "tr_c": train_targets,
            "va_raw": val_raw,
            "va_sub": [],
            "va_htx": val_text,
            "va_c": val_targets,
            "v9_raw_stage": "pre_causal_conv",
            "v9_raw_description": "native load_thinker Gemma4Model final LM hidden before V9 causal_conv",
            "thinker_model_path": resolve_thinker_path("/workspace"),
            "text_block_len": int(kp.TEXT_BLOCK_LEN),
            "audio_start": audio_start,
            "features_per_agent_frame": fpf,
            "validation_start_frame": val_start,
            "validation_text_window_offset": val_start // 64,
            "validation_fraction": validation_fraction,
        },
        output,
    )
    print(f"[RUNTIME-CACHE] saved {output}: {len(train_raw)} train / {len(val_raw)} val", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
