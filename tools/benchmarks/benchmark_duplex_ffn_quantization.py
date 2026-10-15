#!/usr/bin/env python3
"""Isolated Gemma 4 FFN quantization microbenchmark.

Loads one or two real Gemma 4 text MLP blocks from local safetensors, without
constructing the full language model or touching the live duplex engine.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_TORCH = "2.12.1"
EXPECTED_TRANSFORMERS = "5.5.0"
EXPECTED_TORCHAO = "0.18.0"
DEFAULT_MODEL_RELATIVE_PATH = Path(".hf_cache/models--google--gemma-4-E2B-it/full")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    token_count: int
    role: str


def benchmark_cases() -> tuple[BenchmarkCase, ...]:
    """Return the fixed shapes used to approximate the requested FFN workloads."""
    return (
        BenchmarkCase("listener_q4", 4, "native listener query shape proxy"),
        BenchmarkCase("thinker_decode_q1", 1, "autoregressive decode shape proxy"),
        BenchmarkCase("thinker_prefill_q4", 4, "compiled prefill shape proxy"),
    )


def parse_layer_indices(value: str, *, num_layers: int, maximum: int = 2) -> tuple[int, ...]:
    """Parse one or two unique layer indices and fail closed on bad input."""
    if num_layers <= 0 or maximum <= 0:
        raise ValueError("num_layers e maximum devono essere positivi")
    parts = value.split(",") if value else []
    if not parts or len(parts) > maximum or any(not part.strip() for part in parts):
        raise ValueError(f"specificare da 1 a {maximum} layer, separati da virgola")
    try:
        indices = tuple(int(part.strip(), 10) for part in parts)
    except ValueError as exc:
        raise ValueError("gli indici layer devono essere interi") from exc
    if len(set(indices)) != len(indices):
        raise ValueError("gli indici layer devono essere distinti")
    if any(index < 0 or index >= num_layers for index in indices):
        raise ValueError(f"ogni indice deve essere in [0, {num_layers - 1}]")
    return indices


def find_ffn_tensor_names(keys: Iterable[str], *, layer_index: int) -> dict[str, str]:
    """Resolve exactly one real checkpoint tensor for each text MLP projection."""
    available = tuple(str(key) for key in keys)
    result: dict[str, str] = {}
    for projection in PROJECTIONS:
        suffix = re.compile(
            rf"(?:^|\.)layers\.{int(layer_index)}\.mlp\.{re.escape(projection)}\.weight$"
        )
        matches = sorted(key for key in available if suffix.search(key))
        if len(matches) != 1:
            raise ValueError(
                f"layer {layer_index} {projection}: atteso un solo peso, trovati {matches}"
            )
        result[projection] = matches[0]
    return result


def native_listener_contract() -> dict[str, Any]:
    """Describe the current native listener boundary recorded by the source."""
    return {
        "query_shape": [1, 4, 1536],
        "query_dtype": "bfloat16",
        "tick_ms": 80,
        "weight_dtype_required": "bfloat16",
        "weights_are_borrowed_cuda_pointers": True,
        "quantized_shared_ffn_supported": False,
        "native_listener_integration_tested": False,
        "reason": (
            "native_listener._ptr accepts contiguous CUDA BF16 tensors only; "
            "the native cuBLAS linear path reads CUDA_R_16BF weights"
        ),
    }


def report_header(*, model_path: str, layers: Sequence[int]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "single_ffn_block_quantization_microbenchmark",
        "model_path": str(model_path),
        "layers": [int(index) for index in layers],
        "quantized_modules": list(PROJECTIONS),
        "input_source": "synthetic normalized BF16 hidden states; not audio-derived activations",
        "memory_measurement": {
            "baseline": "one fixed listener_q4 input is allocated before every module baseline",
            "steady_model_cuda_bytes": "sampled after one forward and after releasing its output; lazy weights are included",
            "peak_first_forward_cuda_bytes": "first-forward peak above the same input baseline; includes transient activations",
            "backend_allocator_note": "a zero delta for a successful backend is reported as null when its persistent CUDA storage is not visible to the PyTorch allocator",
        },
        "native_listener": native_listener_contract(),
        "scope": {
            "full_language_model_loaded": False,
            "live_engine_modified": False,
            "native_cuda_modified": False,
            "query_abi_preserved_by_experiment": True,
            "end_to_end_latency_claim": False,
        },
    }


def is_exact_stable_version(value: str, expected: str) -> bool:
    """Accept only the exact stable x.y.z release, not rc/dev/post/local builds."""
    return bool(re.fullmatch(r"\d+\.\d+\.\d+", str(value))) and str(value) == expected


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _local_checkpoint_path(value: str | None, workspace: Path) -> Path:
    path = Path(value).expanduser() if value else workspace / DEFAULT_MODEL_RELATIVE_PATH
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint locale assente: {path}; nessun download verrà avviato")
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"config.json non trovato in {path}")
    if not list(path.glob("*.safetensors")):
        raise FileNotFoundError(f"nessun safetensors locale trovato in {path}")
    return path


def _checkpoint_weight_map(model_path: Path) -> dict[str, Path]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"weight_map non valido in {index_path}")
        result: dict[str, Path] = {}
        for tensor_name, filename in weight_map.items():
            shard = (model_path / str(filename)).resolve()
            if model_path.resolve() not in shard.parents or not shard.is_file():
                raise ValueError(f"shard fuori checkpoint o assente: {filename}")
            result[str(tensor_name)] = shard
        return result

    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors è richiesto; non verrà installato automaticamente") from exc
    result = {}
    for shard in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for tensor_name in handle.keys():
                if tensor_name in result:
                    raise ValueError(f"peso duplicato in più shard: {tensor_name}")
                result[tensor_name] = shard
    if not result:
        raise ValueError(f"nessun tensore trovato nei safetensors di {model_path}")
    return result


def _read_tensor(model_path: Path, locations: Mapping[str, Path], name: str):
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise RuntimeError("safetensors è richiesto; non verrà installato automaticamente") from exc
    shard = locations[name]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def _get_text_config(model_path: Path):
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("Transformers non è installato nel runtime richiesto") from exc
    config = AutoConfig.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    getter = getattr(config, "get_text_config", None)
    text_config = getter(decoder=True) if callable(getter) else getattr(config, "text_config", config)
    required = ("hidden_size", "intermediate_size", "num_hidden_layers", "hidden_activation")
    absent = [name for name in required if not hasattr(text_config, name)]
    if absent:
        raise ValueError(f"text_config incompleta; campi assenti: {absent}")
    return text_config


def _load_real_mlp(model_path: Path, layer_index: int, locations: Mapping[str, Path], text_config: Any, torch):
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP
    except ImportError as exc:
        raise RuntimeError(
            "Transformers 5.5.0 deve esporre Gemma4TextMLP; benchmark non compatibile"
        ) from exc
    module = Gemma4TextMLP(text_config, layer_index)
    resolved = find_ffn_tensor_names(locations.keys(), layer_index=layer_index)
    with torch.no_grad():
        for projection in PROJECTIONS:
            source = _read_tensor(model_path, locations, resolved[projection])
            target = getattr(module, projection).weight
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError(
                    f"layer {layer_index} {projection}: checkpoint {tuple(source.shape)} "
                    f"!= Transformers {tuple(target.shape)}"
                )
            if source.dtype != torch.bfloat16:
                raise ValueError(
                    f"layer {layer_index} {projection}: atteso BF16, trovato {source.dtype}"
                )
            target.copy_(source)
            target.requires_grad_(False)
    module.eval()
    return module, resolved


def _torchao_configs() -> tuple[Any, Any, Any] | None:
    version = _distribution_version("torchao")
    if version is None:
        return None
    if not is_exact_stable_version(version, EXPECTED_TORCHAO):
        raise RuntimeError(
            f"torchao {version} != pin sperimentale {EXPECTED_TORCHAO}; "
            "nessuna API non verificata verrà usata"
        )
    try:
        from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise RuntimeError(
            f"torchao {version} non espone l'API Int4/Int8WeightOnlyConfig attesa; "
            "non verrà sostituito né installato automaticamente"
        ) from exc
    return (
        quantize_,
        Int8WeightOnlyConfig(),
        Int4WeightOnlyConfig(group_size=128),
    )


def _bnb_version() -> str | None:
    return _distribution_version("bitsandbytes")


def _replace_with_bnb4(module: Any, *, torch, bnb, dequantize_each_forward: bool) -> Any:
    class DequantizedLinear(torch.nn.Module):
        def __init__(self, source):
            super().__init__()
            packed, state = bnb.functional.quantize_4bit(
                source.weight.detach().contiguous(),
                blocksize=128,
                compress_statistics=True,
                quant_type="nf4",
            )
            self.register_buffer("packed_weight", packed)
            self.quant_state = state
            self.in_features = source.in_features
            self.out_features = source.out_features
            self.register_buffer("bias", None)

        def dequantize_weight(self):
            value = bnb.functional.dequantize_4bit(self.packed_weight, self.quant_state)
            return value.reshape(self.out_features, self.in_features)

        def forward(self, inputs):
            weight = self.dequantize_weight().to(dtype=inputs.dtype)
            return torch.nn.functional.linear(inputs, weight, self.bias)

    for projection in PROJECTIONS:
        source = getattr(module, projection)
        if dequantize_each_forward:
            replacement = DequantizedLinear(source)
        else:
            replacement = bnb.nn.Linear4bit(
                source.in_features,
                source.out_features,
                bias=False,
                compute_dtype=torch.bfloat16,
                compress_statistics=True,
                quant_type="nf4",
            )
            replacement = replacement.to(dtype=torch.bfloat16)
            replacement.load_state_dict({"weight": source.weight.detach().cpu()})
            replacement = replacement.to(device=source.weight.device)
        setattr(module, projection, replacement)
    module.eval()
    return module


def _output_metrics(reference, candidate, torch) -> dict[str, float]:
    ref = reference.detach().float().cpu().reshape(-1)
    got = candidate.detach().float().cpu().reshape(-1)
    delta = got - ref
    denominator = ref.norm().clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(ref, got, dim=0).item()
    return {
        "max_abs_error": float(delta.abs().max().item()),
        "relative_l2_error": float((delta.norm() / denominator).item()),
        "cosine_similarity": float(cosine),
    }


def _validate_candidate_output(reference, candidate, *, torch) -> None:
    if tuple(candidate.shape) != tuple(reference.shape):
        raise ValueError(
            f"quantizzazione ha cambiato la forma: {tuple(candidate.shape)} "
            f"!= {tuple(reference.shape)}"
        )
    if candidate.dtype != torch.bfloat16:
        raise ValueError(
            f"quantizzazione ha cambiato il dtype di output: {candidate.dtype} != bfloat16"
        )
    if not bool(torch.isfinite(candidate).all().item()):
        raise ValueError("quantizzazione ha prodotto output non finiti")


def _time_forward(module, inputs, *, warmup: int, iterations: int, compile_mode: str, torch) -> dict[str, Any]:
    runner = module
    compile_error = None
    compile_ms = 0.0
    if compile_mode != "off":
        compile_start = time.perf_counter()
        try:
            runner = torch.compile(module, mode=compile_mode, fullgraph=False, dynamic=False)
            # Compilation is lazy; warmup below pays for it before timings start.
        except Exception as exc:
            compile_error = f"{type(exc).__name__}: {exc}"
            runner = module
            compile_mode = "off_after_compile_failure"
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

    with torch.inference_mode():
        for _ in range(warmup):
            runner(inputs)
        torch.cuda.synchronize(inputs.device)
        wall_start = time.perf_counter()
        events = []
        for _ in range(iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            runner(inputs)
            end.record()
            events.append((begin, end))
        torch.cuda.synchronize(inputs.device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
    latencies_ms = sorted(float(begin.elapsed_time(end)) for begin, end in events)
    return {
        "compile_mode": compile_mode,
        "compile_setup_ms": round(compile_ms, 3),
        "compile_error": compile_error,
        "iterations": iterations,
        "cuda_event_ms": {
            "median": round(statistics.median(latencies_ms), 5),
            "p95": round(_percentile(latencies_ms, 0.95), 5),
            "mean": round(statistics.fmean(latencies_ms), 5),
        },
        "wall_ms_per_call": round(wall_ms / iterations, 5),
    }


def _case_input(case: BenchmarkCase, *, seed: int, layer_index: int, hidden_size: int, device, torch):
    torch.manual_seed(seed + layer_index + case.token_count)
    return torch.randn(
        (1, case.token_count, hidden_size),
        dtype=torch.bfloat16,
        device=device,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile richiede almeno un valore")
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return float(values[index])


def _measure_dequant_only(module, *, warmup: int, iterations: int, torch) -> dict[str, float]:
    linears = [getattr(module, projection) for projection in PROJECTIONS]
    method = getattr(linears[0], "dequantize_weight", None)
    if not callable(method):
        return {"median_cuda_event_ms": 0.0, "p95_cuda_event_ms": 0.0}
    with torch.inference_mode():
        for _ in range(warmup):
            temporary = [linear.dequantize_weight() for linear in linears]
            del temporary
        torch.cuda.synchronize()
        events = []
        for _ in range(iterations):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            temporary = [linear.dequantize_weight() for linear in linears]
            end.record()
            events.append((begin, end))
            del temporary
        torch.cuda.synchronize()
    values = sorted(float(begin.elapsed_time(end)) for begin, end in events)
    return {
        "median_cuda_event_ms": round(statistics.median(values), 5),
        "p95_cuda_event_ms": round(_percentile(values, 0.95), 5),
    }


def _build_torchao_variant(base_cpu, bits: int, device, torch):
    backend = _torchao_configs()
    if backend is None:
        raise RuntimeError("torchao non installato; nessun pacchetto verrà scaricato")
    quantize_, int8_config, int4_config = backend
    candidate = copy.deepcopy(base_cpu).to(device)
    quantize_(candidate, int8_config if bits == 8 else int4_config)
    return candidate


def _build_bnb_variant(base_cpu, *, dequantize_each_forward: bool, device, torch):
    try:
        import bitsandbytes as bnb
    except ImportError as exc:
        raise RuntimeError("bitsandbytes non installato nel runtime; nessuna installazione automatica") from exc
    candidate = copy.deepcopy(base_cpu).to(device)
    return _replace_with_bnb4(candidate, torch=torch, bnb=bnb, dequantize_each_forward=dequantize_each_forward)


def _variant_builders(base_cpu, *, device, torch) -> tuple[list[tuple[str, Any, str]], list[str]]:
    builders: list[tuple[str, Any, str]] = []
    notes: list[str] = []
    try:
        torchao = _torchao_configs()
    except RuntimeError as exc:
        torchao = None
        notes.append(str(exc))
    else:
        if torchao is None:
            notes.append("torchao non installato; varianti TorchAO saltate")
    if torchao is not None:
        builders.extend(
            [
                ("torchao_int8_weight_only", lambda: _build_torchao_variant(base_cpu, 8, device, torch), "Int8WeightOnlyConfig"),
                ("torchao_int4_group128", lambda: _build_torchao_variant(base_cpu, 4, device, torch), "Int4WeightOnlyConfig(group_size=128)"),
            ]
        )
    if _bnb_version() is not None:
        builders.extend(
            [
                ("bnb_nf4_fused", lambda: _build_bnb_variant(base_cpu, dequantize_each_forward=False, device=device, torch=torch), "bitsandbytes Linear4bit NF4"),
                ("bnb_nf4_dequant_bf16_each_forward", lambda: _build_bnb_variant(base_cpu, dequantize_each_forward=True, device=device, torch=torch), "NF4 weights dequantized to BF16 on every forward"),
            ]
        )
    else:
        notes.append("bitsandbytes non installato; varianti BNB saltate")
    return builders, notes


def _runtime_metadata(torch, model_path: Path) -> dict[str, Any]:
    import transformers

    return {
        "torch": str(torch.__version__),
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "transformers": str(transformers.__version__),
        "torchao": _distribution_version("torchao"),
        "bitsandbytes": _distribution_version("bitsandbytes"),
        "safetensors": _distribution_version("safetensors"),
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "compute_capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        "model_path": str(model_path),
    }


def _validate_runtime(torch, args) -> None:
    torch_version = str(torch.__version__).split("+", 1)[0]
    transformers_version = _distribution_version("transformers")
    if args.expected_torch_version and torch_version != args.expected_torch_version:
        raise RuntimeError(
            f"PyTorch {torch_version} != atteso {args.expected_torch_version}; benchmark interrotto"
        )
    if args.expected_transformers_version and transformers_version != args.expected_transformers_version:
        raise RuntimeError(
            f"Transformers {transformers_version} != atteso {args.expected_transformers_version}; benchmark interrotto"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA non disponibile; il benchmark GPU non verrà simulato su CPU")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("il benchmark è fissato al runtime RTX 3090/Ampere o successivo")


def run_benchmark(args) -> dict[str, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:
        raise RuntimeError("il benchmark richiede Torch e Transformers già installati") from exc
    _validate_runtime(torch, args)
    workspace = Path(__file__).resolve().parents[2]
    model_path = _local_checkpoint_path(args.model_path, workspace)
    text_config = _get_text_config(model_path)
    if int(text_config.hidden_size) != 1536:
        raise RuntimeError(
            f"hidden_size {text_config.hidden_size} != 1536 richiesto dal contratto listener attivo"
        )
    layer_indices = parse_layer_indices(
        args.layer_indices,
        num_layers=int(text_config.num_hidden_layers),
    )
    locations = _checkpoint_weight_map(model_path)
    runtime = _runtime_metadata(torch, model_path)
    torch.cuda.init()
    device = torch.device("cuda", torch.cuda.current_device())
    output = report_header(model_path=str(model_path), layers=layer_indices)
    output["runtime"] = runtime
    output["settings"] = {
        "compile_mode": args.compile_mode,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "input_distribution": "standard normal with RMS approximately 1, cast to BF16",
        "random_seed": args.seed,
        "bnb_dequantization_included_in_forward": True,
    }
    output["torchao_runtime_gate"] = {
        "exact_experiment_pin": EXPECTED_TORCHAO,
        "available_version": runtime["torchao"],
        "reason": (
            "TorchAO 0.18.0 release drops PyTorch <2.11 and moves CI to PyTorch 2.12; "
            "the exact API still must pass import/quantize smoke in the serving image"
        ),
    }
    output["bnb_dequantization_method"] = (
        "NF4 packed weights are dequantized to BF16 on each FFN forward before F.linear; "
        "this isolates the conversion cost required by a BF16-pointer consumer and is not a fused low-bit kernel"
    )
    output["layers_result"] = []

    for layer_index in layer_indices:
        base_cpu, selected_keys = _load_real_mlp(model_path, layer_index, locations, text_config, torch)
        base_cpu = base_cpu.to(dtype=torch.bfloat16, device="cpu")
        memory_probe_inputs = _case_input(
            benchmark_cases()[0],
            seed=args.seed,
            layer_index=layer_index,
            hidden_size=int(text_config.hidden_size),
            device=device,
            torch=torch,
        )
        empty_baseline_bytes = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        base = base_cpu.to(device).eval()
        torch.cuda.synchronize(device)
        bf16_construction_peak_bytes = max(
            0, int(torch.cuda.max_memory_allocated(device)) - empty_baseline_bytes
        )
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            memory_probe_output = base(memory_probe_inputs)
        torch.cuda.synchronize(device)
        bf16_first_forward_peak_bytes = max(
            0, int(torch.cuda.max_memory_allocated(device)) - empty_baseline_bytes
        )
        del memory_probe_output
        torch.cuda.synchronize(device)
        bf16_resident_bytes = max(
            0, int(torch.cuda.memory_allocated(device)) - empty_baseline_bytes
        )
        layer_result: dict[str, Any] = {
            "layer_index": layer_index,
            "weight_keys": selected_keys,
            "hidden_size": int(text_config.hidden_size),
            "intermediate_size": int(base_cpu.intermediate_size),
            "quantization": [],
        }

        reference_outputs: dict[str, Any] = {}
        bf16_case_results: dict[str, Any] = {}
        for case in benchmark_cases():
            inputs = _case_input(
                case,
                seed=args.seed,
                layer_index=layer_index,
                hidden_size=int(text_config.hidden_size),
                device=device,
                torch=torch,
            )
            with torch.inference_mode():
                reference_outputs[case.name] = base(inputs).detach().cpu()
            timing = _time_forward(
                base,
                inputs,
                warmup=args.warmup,
                iterations=args.iterations,
                compile_mode=args.compile_mode,
                torch=torch,
            )
            bf16_case_results[case.name] = {
                "timing": timing,
                "output_error_vs_bf16": {
                    "max_abs_error": 0.0,
                    "relative_l2_error": 0.0,
                    "cosine_similarity": 1.0,
                },
            }
            del inputs
        layer_result["quantization"].append(
            {
                "variant": "bf16_reference",
                "quantization_config": "none",
                "steady_model_cuda_bytes": bf16_resident_bytes,
                "peak_construction_cuda_bytes": bf16_construction_peak_bytes,
                "peak_first_forward_cuda_bytes": bf16_first_forward_peak_bytes,
                "dequant_only_timing": None,
                "cases": bf16_case_results,
            }
        )
        del base
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

        builders, backend_notes = _variant_builders(base_cpu, device=device, torch=torch)
        layer_result["backend_notes"] = backend_notes
        for name, builder, quantization_config in builders:
            empty_baseline_bytes = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            candidate = None
            candidate_output = None
            case_inputs = None
            try:
                candidate = builder().eval()
                torch.cuda.synchronize(device)
                construction_peak_bytes = max(
                    0,
                    int(torch.cuda.max_memory_allocated(device)) - empty_baseline_bytes,
                )
                torch.cuda.reset_peak_memory_stats(device)
                with torch.inference_mode():
                    candidate_output = candidate(memory_probe_inputs)
                torch.cuda.synchronize(device)
                first_forward_peak_bytes = max(
                    0, int(torch.cuda.max_memory_allocated(device)) - empty_baseline_bytes
                )
                candidate_output = None
                torch.cuda.synchronize(device)
                resident_bytes = max(
                    0, int(torch.cuda.memory_allocated(device)) - empty_baseline_bytes
                )
                case_results: dict[str, Any] = {}
                dequant_timing = None
                if name == "bnb_nf4_dequant_bf16_each_forward":
                    dequant_timing = _measure_dequant_only(
                        candidate, warmup=args.warmup, iterations=args.iterations, torch=torch
                    )
                for case in benchmark_cases():
                    case_inputs = _case_input(
                        case,
                        seed=args.seed,
                        layer_index=layer_index,
                        hidden_size=int(text_config.hidden_size),
                        device=device,
                        torch=torch,
                    )
                    with torch.inference_mode():
                        candidate_output = candidate(case_inputs)
                    torch.cuda.synchronize(device)
                    _validate_candidate_output(
                        reference_outputs[case.name], candidate_output, torch=torch
                    )
                    error = _output_metrics(reference_outputs[case.name], candidate_output, torch)
                    timing = _time_forward(
                        candidate,
                        case_inputs,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        compile_mode=args.compile_mode,
                        torch=torch,
                    )
                    case_results[case.name] = {
                        "timing": timing,
                        "output_error_vs_bf16": error,
                    }
                    candidate_output = None
                    case_inputs = None
                resident_report = resident_bytes
                first_forward_peak_report = first_forward_peak_bytes
                steady_memory_status = "measured_by_torch_cuda_allocator"
                first_forward_peak_status = "measured_by_torch_cuda_allocator"
                if name.startswith("bnb_") and resident_bytes == 0:
                    resident_report = None
                    steady_memory_status = "unavailable_backend_storage_not_visible_to_torch_allocator"
                if name.startswith("bnb_") and first_forward_peak_bytes == 0:
                    first_forward_peak_report = None
                    first_forward_peak_status = "unavailable_backend_storage_not_visible_to_torch_allocator"
                layer_result["quantization"].append(
                    {
                        "variant": name,
                        "quantization_config": quantization_config,
                        "steady_model_cuda_bytes": resident_report,
                        "steady_model_cuda_bytes_status": steady_memory_status,
                        "peak_construction_cuda_bytes": construction_peak_bytes,
                        "peak_first_forward_cuda_bytes": first_forward_peak_report,
                        "peak_first_forward_cuda_bytes_status": first_forward_peak_status,
                        "dequant_only_timing": dequant_timing,
                        "cases": case_results,
                    }
                )
            except Exception as exc:
                layer_result["quantization"].append(
                    {
                        "variant": name,
                        "quantization_config": quantization_config,
                        "status": "unsupported_or_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                if candidate_output is not None:
                    del candidate_output
                if case_inputs is not None:
                    del case_inputs
                if candidate is not None:
                    del candidate
                torch.cuda.empty_cache()

        output["layers_result"].append(layer_result)
        del base_cpu, reference_outputs, bf16_case_results, memory_probe_inputs
        torch.cuda.empty_cache()

    return output


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="directory of the already-cached official Gemma 4 safetensors")
    parser.add_argument("--layer-indices", default="15", help="one or two layer indices, e.g. 14,15")
    parser.add_argument("--compile-mode", choices=("off", "default", "reduce-overhead"), default="off")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=260902)
    parser.add_argument("--expected-torch-version", default=EXPECTED_TORCH)
    parser.add_argument("--expected-transformers-version", default=EXPECTED_TRANSFORMERS)
    parser.add_argument("--output-json", default="reports/ffn_quantization_microbenchmark.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    for name, value in (("warmup", args.warmup), ("iterations", args.iterations)):
        if value <= 0:
            print(f"{name} deve essere positivo", file=sys.stderr)
            return 2
    try:
        result = run_benchmark(args)
        workspace = Path(__file__).resolve().parents[2]
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = workspace / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"benchmark interrotto: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"report scritto: {output_path}")
    print("Questo microbenchmark non modifica né certifica l'integrazione col listener nativo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
