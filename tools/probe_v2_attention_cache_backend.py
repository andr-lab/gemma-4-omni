#!/usr/bin/env python3
"""No-update Gemma4 comparison of cache-on/off under SDPA and eager attention."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.prepare import read_jsonl
from training.duplex_qlora.stage1 import _load_qlora
from tools.eval_audio_response_protocol_v2 import _encode_prompt


def _summary_tensor(value: torch.Tensor) -> dict[str, Any]:
    flat = value.detach().to(dtype=torch.float32).contiguous().cpu()
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256_f32": hashlib.sha256(flat.numpy().tobytes()).hexdigest(),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
        "finite_min": float(finite_values.min()) if finite_values.numel() else None,
        "finite_max": float(finite_values.max()) if finite_values.numel() else None,
    }


def _summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _summary_tensor(value)
    if isinstance(value, Mapping):
        return {str(k): _summarize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summarize(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _delta(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a, b = left.float(), right.float()
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0}
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rms_abs": float(torch.sqrt(torch.mean(diff.square()))),
        "cosine": float(torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()),
    }


def _key(row: Mapping[str, Any]) -> str:
    return f"{row.get('conversation_id', '')}::{row.get('target_utterance_id', '')}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("probe requires CUDA")
    torch.cuda.set_device(0)

    rows = read_jsonl(args.test_jsonl)
    direct = [row for row in rows if row.get("task_kind") == "conversation_response"]
    selected = {action: next(row for row in direct if row.get("action") == action)
                for action in ("SPEAK", "BACKCHANNEL", "REPLAN")}
    model, processor, _trainable, _modules, quantization = _load_qlora(
        args.model_path, dtype=torch.bfloat16, resume_adapter_dir=args.adapter_dir,
    )
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_config = base.config.get_text_config()
    original_impls = {
        "base": getattr(base.config, "_attn_implementation", None),
        "text": getattr(text_config, "_attn_implementation", None),
    }
    import transformers.models.gemma4.modeling_gemma4 as gemma4
    original_factory = getattr(gemma4, "create_masks_for_generate", None)
    result: dict[str, Any] = {
        "status": "in_progress",
        "transformers_version": __import__("transformers").__version__,
        "model_path": args.model_path,
        "adapter_path": str(args.adapter_dir),
        "quantization": str(quantization),
        "initial_attention_implementation": original_impls,
        "probe_contract": "four forward-only paths per heldout class; no updates; eager and SDPA use same loaded weights/audio",
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, args.output)

    persist()
    try:
        for action, row in selected.items():
            encoded = _encode_prompt(processor, row, device=torch.device("cuda:0"))
            item: dict[str, Any] = {
                "record_id": _key(row),
                "prompt_tokens": int(encoded["input_ids"].shape[1]),
                "audio_feature_shape": list(encoded["input_features"].shape),
                "paths": {},
            }
            hidden_states: dict[str, tuple[torch.Tensor, ...]] = {}
            last_logits: dict[str, torch.Tensor] = {}
            for backend in ("sdpa", "eager"):
                # Gemma4TextAttention reads this config at each invocation, so the loaded modules/weights stay fixed.
                base.config._attn_implementation = backend
                text_config._attn_implementation = backend
                for cache_mode in ("off", "on"):
                    label = f"{backend}_cache_{cache_mode}"
                    factory_calls: list[Any] = []
                    if original_factory is not None:
                        def wrapped_factory(*f_args: Any, **f_kwargs: Any) -> Any:
                            masks = original_factory(*f_args, **f_kwargs)
                            factory_calls.append(_summarize(masks))
                            return masks
                        gemma4.create_masks_for_generate = wrapped_factory
                    call = dict(encoded)
                    call.update({"use_cache": cache_mode == "on", "output_hidden_states": True, "return_dict": True})
                    with torch.inference_mode():
                        output = model(**call)
                    states = getattr(output, "hidden_states", None)
                    if not states:
                        raise RuntimeError(f"hidden_states unavailable for {label}")
                    hidden_states[label] = tuple(x.detach().to(dtype=torch.float16).cpu() for x in states)
                    last_logits[label] = output.logits[0, -1].detach().float().cpu()
                    item["paths"][label] = {
                        "attention_backend_config": {
                            "base": getattr(base.config, "_attn_implementation", None),
                            "text": getattr(text_config, "_attn_implementation", None),
                        },
                        "logits_shape": list(output.logits.shape),
                        "last_logit_argmax": int(last_logits[label].argmax()),
                        "output_cache_length": int(output.past_key_values.get_seq_length()) if getattr(output, "past_key_values", None) is not None else None,
                        "mask_factory_calls": factory_calls,
                    }
                    del output
                    torch.cuda.empty_cache()
            if original_factory is not None:
                gemma4.create_masks_for_generate = original_factory

            item["cache_off_vs_on_by_backend"] = {}
            for backend in ("sdpa", "eager"):
                off, on = f"{backend}_cache_off", f"{backend}_cache_on"
                layers = []
                for layer_index, (a, b) in enumerate(zip(hidden_states[off], hidden_states[on])):
                    layers.append({"hidden_state_index": layer_index, **_delta(a, b)})
                item["cache_off_vs_on_by_backend"][backend] = {
                    "logits": _delta(last_logits[off], last_logits[on]),
                    "first_layer_with_max_abs_over_1e-3": next((x["hidden_state_index"] for x in layers if x["max_abs"] > 1e-3), None),
                    "layer_deltas": layers,
                }
            item["backend_deltas_by_cache_mode"] = {}
            for cache_mode in ("off", "on"):
                sdpa, eager = f"sdpa_cache_{cache_mode}", f"eager_cache_{cache_mode}"
                item["backend_deltas_by_cache_mode"][cache_mode] = {
                    "logits": _delta(last_logits[sdpa], last_logits[eager]),
                    "first_layer_with_max_abs_over_1e-3": next((i for i, (a, b) in enumerate(zip(hidden_states[sdpa], hidden_states[eager])) if _delta(a, b)["max_abs"] > 1e-3), None),
                }
            result["probes"][action] = item
            persist()
            print(json.dumps({
                "action": action,
                "cache_delta_sdpa": item["cache_off_vs_on_by_backend"]["sdpa"]["logits"],
                "cache_delta_eager": item["cache_off_vs_on_by_backend"]["eager"]["logits"],
                "first_layer_sdpa": item["cache_off_vs_on_by_backend"]["sdpa"]["first_layer_with_max_abs_over_1e-3"],
                "first_layer_eager": item["cache_off_vs_on_by_backend"]["eager"]["first_layer_with_max_abs_over_1e-3"],
            }, ensure_ascii=True), flush=True)
    finally:
        if original_factory is not None:
            gemma4.create_masks_for_generate = original_factory
        base.config._attn_implementation = original_impls["base"]
        text_config._attn_implementation = original_impls["text"]
    result["status"] = "complete"
    persist()
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
