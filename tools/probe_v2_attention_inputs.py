#!/usr/bin/env python3
"""Trace first language attention q/k/v, mask, and output with cache on/off."""
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


def _clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).cpu().clone()
    if isinstance(value, Mapping):
        return {str(k): _clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clone(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        data = value.detach().float().contiguous().cpu()
        finite = torch.isfinite(data)
        vals = data[finite]
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256_f32": hashlib.sha256(data.numpy().tobytes()).hexdigest(),
            "finite_count": int(finite.sum()),
            "nonfinite_count": int((~finite).sum()),
            "finite_min": float(vals.min()) if vals.numel() else None,
            "finite_max": float(vals.max()) if vals.numel() else None,
        }
    if isinstance(value, Mapping):
        return {str(k): _summary(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_summary(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"type": type(value).__name__}


def _delta(left: Any, right: Any) -> Any:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape:
            return {"shape_left": list(left.shape), "shape_right": list(right.shape)}
        a, b = left.float(), right.float()
        both_finite = torch.isfinite(a) & torch.isfinite(b)
        finite_delta = (a[both_finite] - b[both_finite]).abs()
        return {
            "shape": list(a.shape),
            "max_abs": float(finite_delta.max()) if finite_delta.numel() else 0.0,
            "mean_abs": float(finite_delta.mean()) if finite_delta.numel() else 0.0,
            "nonfinite_mask_equal": torch.equal(torch.isfinite(a), torch.isfinite(b)),
            "exact_equal": torch.equal(a, b),
        }
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        return {key: _delta(left.get(key), right.get(key)) for key in keys}
    if left == right:
        return {"equal": True, "value": left}
    return {"equal": False, "left": _summary(left), "right": _summary(right)}


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
    selected = {action: next(r for r in direct if r.get("action") == action)
                for action in ("SPEAK", "BACKCHANNEL", "REPLAN")}
    model, processor, _trainable, _modules, _quantization = _load_qlora(
        args.model_path, dtype=torch.bfloat16, resume_adapter_dir=args.adapter_dir,
    )
    model.eval()
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    text_config = base.config.get_text_config()
    import transformers.models.gemma4.modeling_gemma4 as gemma4
    from transformers.cache_utils import DynamicCache

    result: dict[str, Any] = {"status": "in_progress", "transformers_version": __import__("transformers").__version__, "probes": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist() -> None:
        tmp = args.output.with_name(args.output.name + ".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, args.output)

    persist()
    for action, row in selected.items():
        encoded = _encode_prompt(processor, row, device=torch.device("cuda:0"))
        item: dict[str, Any] = {"action": action, "input_tokens": int(encoded["input_ids"].shape[1]), "backends": {}}
        for backend in ("sdpa", "eager"):
            base.config._attn_implementation = backend
            text_config._attn_implementation = backend
            original_attention_function = gemma4.ALL_ATTENTION_FUNCTIONS[backend]
            vectors: dict[str, Any] = {}
            modes: dict[str, Any] = {}
            for cache_on in (False, True):
                mode = "cache_on" if cache_on else "cache_off"
                captured = {"first_call": False}

                def wrapped_attention(module: Any, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                      attention_mask: torch.Tensor | None, **kwargs: Any) -> Any:
                    output = original_attention_function(module, query, key, value, attention_mask, **kwargs)
                    if not captured["first_call"]:
                        captured["first_call"] = True
                        output_tensor = output[0] if isinstance(output, (tuple, list)) else output
                        vectors[mode] = {
                            "query_states": _clone(query),
                            "key_states_after_cache_update": _clone(key),
                            "value_states_after_cache_update": _clone(value),
                            "attention_mask": _clone(attention_mask),
                            "attention_output": _clone(output_tensor),
                        }
                        modes[mode] = {
                            "layer_index": getattr(module, "layer_idx", None),
                            "query": _summary(query),
                            "key_after_cache_update": _summary(key),
                            "value_after_cache_update": _summary(value),
                            "mask": _summary(attention_mask),
                            "attention_output": _summary(output_tensor),
                            "attention_kwargs": _summary(kwargs),
                        }
                    return output

                gemma4.ALL_ATTENTION_FUNCTIONS[backend] = wrapped_attention
                call = dict(encoded)
                call.update({"use_cache": cache_on, "return_dict": True})
                if cache_on:
                    call["past_key_values"] = DynamicCache(config=text_config)
                with torch.inference_mode():
                    out = model(**call)
                modes[mode]["final_logits_shape"] = list(out.logits.shape)
                del out
                gemma4.ALL_ATTENTION_FUNCTIONS[backend] = original_attention_function
                torch.cuda.empty_cache()
            item["backends"][backend] = {
                "first_language_attention": modes,
                "cache_on_vs_off": _delta(vectors["cache_off"], vectors["cache_on"]),
            }
        result["probes"][action] = item
        persist()
        print(json.dumps({
            "action": action,
            "sdpa": item["backends"]["sdpa"]["cache_on_vs_off"],
            "eager": item["backends"]["eager"]["cache_on_vs_off"],
        }, ensure_ascii=True), flush=True)
    base.config._attn_implementation = "sdpa"
    text_config._attn_implementation = "sdpa"
    result["status"] = "complete"
    persist()
    print(json.dumps({"status": result["status"], "output": str(args.output)}, ensure_ascii=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
