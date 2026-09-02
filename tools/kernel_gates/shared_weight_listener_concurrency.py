#!/usr/bin/env python3
"""Concurrency isolation gate for the installed shared-weight listener.

It overlaps a live Q=4 listener stream with the normal Thinker text forward
and native debug-ASR transcription, then verifies that their byte outputs and
listener representation remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch

from duplex_kernel.gemma_representation import _audio_soft_tokens
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from tools.kernel_gates.p4_candidate_validation import _percentile, _validation_dataset
from tools.kernel_gates.p4_numeric_bisect import _compare
from tools.kernel_gates.shape_stable_listener_check import _module_forward_ids
from tools.kernel_gates.shared_weight_listener_runtime import _runtime_sequence


def _tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _text_forward(model: Any, token_id: int, device: torch.device) -> torch.Tensor:
    ids = torch.tensor([[token_id, token_id]], device=device, dtype=torch.long)
    with torch.inference_mode():
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, return_dict=True).logits


def _assert_context_update_rebuilds_cache(
    *,
    backend: GemmaAudioTowerBackend,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    audio_soft: torch.Tensor,
) -> bool:
    state = GemmaListenerState(context_frames=64)
    zero = np.zeros(state.frame_samples, dtype=np.float32)
    state.user_pcm_history.append(zero)
    state.assistant_pcm_history.append(zero)
    first = backend._run_incremental_lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_soft=audio_soft,
        state=state,
        text_ids=(),
    )
    first_cache = state.gemma_lm_cache
    state.user_pcm_history.append(zero)
    state.assistant_pcm_history.append(zero)
    state.next_frame_id = 1
    updated = backend._run_incremental_lm(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_soft=audio_soft,
        state=state,
        text_ids=(1,),
    )
    return bool(
        first_cache is not state.gemma_lm_cache
        and torch.isfinite(first).all().item()
        and torch.isfinite(updated).all().item()
    )


def run(dataset: Path) -> dict[str, Any]:
    ds, _, validation_start_frame = _validation_dataset(dataset)
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    lm = model.language_model
    device = torch.device("cuda")
    backend = GemmaAudioTowerBackend(thinker)
    item = ds[0]
    batch = {key: item[key].unsqueeze(0).to(device) for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")}
    ids = batch["input_ids"]
    positions = (ids[0] == int(model.config.audio_token_id)).nonzero(as_tuple=False).flatten().tolist()
    with torch.inference_mode():
        audio_soft = _audio_soft_tokens(
            model,
            input_features=batch["input_features"],
            input_features_mask=batch["input_features_mask"],
        )
        native_listener = model(
            input_ids=ids,
            attention_mask=batch["attention_mask"],
            input_features=batch["input_features"],
            input_features_mask=batch["input_features_mask"],
            use_cache=True,
            return_dict=True,
        ).last_hidden_state[:, positions[::4], :].float().cpu()

    # Warm listener-only buffers outside the measured concurrent execution.
    _runtime_sequence(backend=backend, input_ids=ids, attention_mask=batch["attention_mask"], audio_soft=audio_soft)
    token_id = int(thinker.tokenizer.bos_token_id or thinker.tokenizer.eos_token_id)
    pcm = np.zeros(16_000, dtype=np.float32)
    baseline_text = _text_forward(thinker.model, token_id, device)
    baseline_asr = thinker.transcribe_audio(pcm, max_new_tokens=1)
    before_forwards = _module_forward_ids(lm)

    started = threading.Event()
    listener_result: dict[str, Any] = {}
    listener_error: list[BaseException] = []

    def listener_worker() -> None:
        try:
            started.set()
            hidden, timings = _runtime_sequence(
                backend=backend,
                input_ids=ids,
                attention_mask=batch["attention_mask"],
                audio_soft=audio_soft,
            )
            listener_result["hidden"] = hidden.float().cpu()
            listener_result["timings"] = timings[1:]
        except BaseException as exc:  # surface the original exception after join
            listener_error.append(exc)

    worker = threading.Thread(target=listener_worker, name="shape-stable-listener", daemon=False)
    worker.start()
    if not started.wait(timeout=5):
        raise RuntimeError("thread listener non avviato")
    concurrent_text = _text_forward(thinker.model, token_id, device)
    concurrent_asr = thinker.transcribe_audio(pcm, max_new_tokens=1)
    worker.join(timeout=60)
    if worker.is_alive():
        raise RuntimeError("thread listener non terminato")
    if listener_error:
        raise RuntimeError("listener concorrente fallito") from listener_error[0]
    context_update_rebuilt = _assert_context_update_rebuilds_cache(
        backend=backend,
        input_ids=ids,
        attention_mask=batch["attention_mask"],
        audio_soft=audio_soft,
    )
    after_forwards = _module_forward_ids(lm)
    listener_comparison = _compare(native_listener, listener_result["hidden"])
    text_unchanged = _tensor_hash(baseline_text) == _tensor_hash(concurrent_text)
    asr_unchanged = baseline_asr == concurrent_asr
    report = {
        "schema_version": 1,
        "phase": "P4",
        "gate": "shared_weight_listener_concurrency",
        "validation_start_frame": validation_start_frame,
        "listener": {
            "comparison": listener_comparison,
            "p50_ms": _percentile(listener_result["timings"], 50),
            "p95_ms": _percentile(listener_result["timings"], 95),
            "p99_ms": _percentile(listener_result["timings"], 99),
            "frames": 64,
        },
        "thinker": {
            "text_forward_unchanged": text_unchanged,
            "debug_asr_unchanged": asr_unchanged,
            "baseline_asr": baseline_asr,
            "concurrent_asr": concurrent_asr,
        },
        "isolation": {
            "module_forward_identity_unchanged": before_forwards == after_forwards,
            "context_update_rebuilt_listener_cache": context_update_rebuilt,
            "shared_weights_only": bool(backend._shape_stable_lm.weight_sharing_report()["shared_weight_storage"]),
        },
    }
    report["passed"] = bool(
        listener_comparison.get("mae") == 0.0
        and text_unchanged
        and asr_unchanged
        and context_update_rebuilt
        and before_forwards == after_forwards
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
