#!/usr/bin/env python3
"""Compare Gemma 4 prefill masks/logits for implicit and explicit empty KV caches."""
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


def _cache_length(value: Any) -> int | None:
    try:
        return int(value.get_seq_length())
    except Exception:
        return None


def _tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().cpu().contiguous()
    numeric = cpu.float()
    finite = torch.isfinite(numeric)
    finite_values = numeric[finite]
    raw = numeric.numpy().tobytes()
    summary: dict[str, Any] = {
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "sha256_f32": hashlib.sha256(raw).hexdigest(),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int((~finite).sum()),
    }
    if finite_values.numel():
        summary["finite_min"] = float(finite_values.min())
        summary["finite_max"] = float(finite_values.max())
        summary["first_values"] = numeric.reshape(-1)[:12].tolist()
        summary["last_values"] = numeric.reshape(-1)[-12:].tolist()
    return summary


def _summarize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _tensor_summary(value)
    if isinstance(value, Mapping):
        return {str(k): _summarize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summarize(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "get_seq_length"):
        return {"type": type(value).__name__, "sequence_length": _cache_length(value)}
    return {"type": type(value).__name__}


def _tensor_mapping(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, Mapping):
        return {str(k): v for k, v in value.items() if isinstance(v, torch.Tensor)}
    return {}


def _max_delta(left: torch.Tensor, right: torch.Tensor) -> float | None:
    if left.shape != right.shape:
        return None
    a, b = left.float(), right.float()
    both_finite = torch.isfinite(a) & torch.isfinite(b)
    if not both_finite.any():
        return 0.0 if torch.equal(a, b) else None
    delta = (a[both_finite] - b[both_finite]).abs().max()
    same_nonfinite = torch.equal(torch.isfinite(a), torch.isfinite(b))
    return float(delta) if same_nonfinite else float("inf")


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
    direct = [r for r in rows if r.get("task_kind") == "conversation_response"]
    selected = {
        action: next(r for r in direct if r.get("action") == action)
        for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
    }
    model, processor, _trainable, _modules, quantization = _load_qlora(
        args.model_path, dtype=torch.bfloat16, resume_adapter_dir=args.adapter_dir,
    )
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    import transformers.models.gemma4.modeling_gemma4 as gemma4
    from transformers.cache_utils import DynamicCache

    text_config = getattr(base.config, "text_config", None)
    if text_config is None:
        raise RuntimeError("Gemma 4 config has no text_config")
    result: dict[str, Any] = {
        "status": "in_progress",
        "transformers_version": __import__("transformers").__version__,
        "model_path": args.model_path,
        "adapter_path": str(args.adapter_dir),
        "quantization": str(quantization),
        "config": {
            "attn_implementation": getattr(base.config, "_attn_implementation", None),
            "text_attn_implementation": getattr(text_config, "_attn_implementation", None),
            "use_bidirectional_attention": getattr(base.config, "use_bidirectional_attention", None),
            "sliding_window": getattr(text_config, "sliding_window", None),
        },
        "probes": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        temp = args.output.with_name(args.output.name + ".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, args.output)

    persist()

    # Record the mask factory outputs from the real outer Gemma4 forward path.
    original_factory = getattr(gemma4, "create_masks_for_generate", None)
    for action, row in selected.items():
        encoded = _encode_prompt(processor, row, device=torch.device("cuda:0"))
        prompt_len = int(encoded["input_ids"].shape[1])
        action_report: dict[str, Any] = {
            "record_id": _key(row),
            "prompt_length": prompt_len,
            "audio_feature_shape": list(encoded["input_features"].shape),
            "modes": {},
        }
        saved_logits: dict[str, torch.Tensor] = {}

        for mode in ("no_cache", "implicit_empty_cache", "explicit_empty_cache"):
            factory_calls: list[dict[str, Any]] = []
            if original_factory is not None:
                def wrapped_factory(*f_args: Any, **f_kwargs: Any) -> Any:
                    output = original_factory(*f_args, **f_kwargs)
                    call = {
                        "arguments": _summarize({
                            **{f"arg_{i}": v for i, v in enumerate(f_args)},
                            **f_kwargs,
                        }),
                        "output": _summarize(output),
                    }
                    factory_calls.append(call)
                    return output
                gemma4.create_masks_for_generate = wrapped_factory

            call_args = dict(encoded)
            call_args.update({"return_dict": True})
            cache_obj = None
            if mode == "no_cache":
                call_args["use_cache"] = False
            elif mode == "implicit_empty_cache":
                call_args["use_cache"] = True
            else:
                cache_obj = DynamicCache(config=text_config)
                call_args["past_key_values"] = cache_obj
                call_args["use_cache"] = True

            with torch.inference_mode():
                output = model(**call_args)
            logits = output.logits[0, -1].detach().float().cpu()
            saved_logits[mode] = logits
            result_row = {
                "input_attention_mask": _tensor_summary(encoded.get("attention_mask", torch.empty(0))),
                "input_position_ids": _tensor_summary(encoded["position_ids"]) if "position_ids" in encoded else None,
                "cache_argument": {"type": type(cache_obj).__name__, "sequence_length": _cache_length(cache_obj)} if cache_obj is not None else None,
                "output_cache": {"type": type(output.past_key_values).__name__, "sequence_length": _cache_length(output.past_key_values)} if getattr(output, "past_key_values", None) is not None else None,
                "logits_shape": list(output.logits.shape),
                "last_logit_argmax": int(logits.argmax()),
                "mask_factory_calls": factory_calls,
            }
            action_report["modes"][mode] = result_row
            del output, logits
            torch.cuda.empty_cache()

        if original_factory is not None:
            gemma4.create_masks_for_generate = original_factory
        pairwise: dict[str, Any] = {}
        names = list(saved_logits)
        for i, left_name in enumerate(names):
            for right_name in names[i + 1:]:
                left, right = saved_logits[left_name], saved_logits[right_name]
                pairwise[f"{left_name}__vs__{right_name}"] = {
                    "max_abs_logit_delta": _max_delta(left, right),
                    "argmax_equal": int(left.argmax()) == int(right.argmax()),
                    "top1_left": int(left.argmax()),
                    "top1_right": int(right.argmax()),
                }
        action_report["pairwise_logits"] = pairwise

        # Compare actual constructed full/sliding masks across modes by digest.
        factory_names = [k for k in action_report["modes"]]
        first = action_report["modes"][factory_names[0]]["mask_factory_calls"]
        for mode in factory_names[1:]:
            calls = action_report["modes"][mode]["mask_factory_calls"]
            if first and calls:
                action_report.setdefault("mask_factory_digest_equal_to_no_cache", {})[mode] = (
                    first[0]["output"] == calls[0]["output"]
                )
        result["probes"][action] = action_report
        persist()
        print(json.dumps({
            "action": action,
            "pairwise_logits": pairwise,
            "mask_factory_digest_equal": action_report.get("mask_factory_digest_equal_to_no_cache"),
            "factory_call_counts": {m: len(v["mask_factory_calls"]) for m, v in action_report["modes"].items()},
        }, ensure_ascii=True), flush=True)

    result["status"] = "complete"
    persist()
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
