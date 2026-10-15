"""BF16-base QLoRA audio-conditioned conversational text fine-tuning."""
from __future__ import annotations

import json
import hashlib
import inspect
import math
import os
import random
import shutil
import time
import uuid
from collections import OrderedDict
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from .checkpointing import read_training_checkpoint_metadata, restore_training_checkpoint, save_training_checkpoint
from .prepare import read_jsonl
from .schema import ACTIONS
from duplex_kernel.audio_response_conditioning import (
    DEFAULT_MAX_AUDIO_MS,
    causal_audio_window,
    build_audio_response_messages,
    build_audio_response_protocol_v2_messages,
)


_AUDIO_RESPONSE_PROTOCOL_V1 = "audio_response_conditioning_v1"
_AUDIO_RESPONSE_PROTOCOL_V2 = "heard_action_say_v2"


_AUDIO_PCM_CACHE_MAX_BYTES = 512 * 1024 * 1024
_TRAINING_INPUT_CACHE_MAX_BYTES = 1024 * 1024 * 1024
_AUDIO_PCM_CACHE: OrderedDict[tuple[str, int, int], tuple[np.ndarray, int]] = OrderedDict()
_AUDIO_PCM_CACHE_BYTES = 0
_AUDIO_PCM_CACHE_STATS: Counter[str] = Counter()
_VERIFIED_AUDIO_FILE_HASHES: set[tuple[str, int, int, str]] = set()
_TRAINING_INPUT_CACHE: OrderedDict[tuple[int, str], tuple[dict[str, torch.Tensor], int]] = OrderedDict()
_TRAINING_INPUT_CACHE_BYTES = 0
_TRAINING_INPUT_CACHE_STATS: Counter[str] = Counter()


_LORA_LANGUAGE_PROJECTIONS = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_FLOAT_TOWER_NAMES = ("audio_tower", "vision_tower")
_FLOAT_BRIDGE_NAMES = (
    "model.embed_audio.embedding_projection",
    "model.embed_vision.embedding_projection",
)
_QUANTIZATION_SKIP_MODULES = (
    "model.audio_tower",
    "model.vision_tower",
    *_FLOAT_BRIDGE_NAMES,
    "lm_head",
)


def _is_linear4bit(module: Any) -> bool:
    return type(module).__name__ == "Linear4bit"


def _is_normalization_module(module: Any) -> bool:
    class_name = type(module).__name__.lower()
    return isinstance(module, torch.nn.LayerNorm) or class_name.endswith(("rmsnorm", "layernorm"))


def _prepare_gemma_for_kbit_training(
    model: Any, *, use_cache_for_training: bool = True,
) -> dict[str, Any]:
    """Freeze Gemma's base weights without PEFT's blanket FP32 cast of multimodal embeddings."""
    has_shared_kv = any(bool(getattr(module, "is_kv_shared_layer", False)) for _, module in model.named_modules())
    if has_shared_kv and not use_cache_for_training:
        raise ValueError(
            "Gemma4 shared-KV training requires use_cache=True; disabling it changes the model's K/V-sharing semantics"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    language_model_roots = [
        name for name, _ in model.named_modules()
        if name and name.split(".")[-1] == "language_model"
    ]
    if len(language_model_roots) != 1:
        raise RuntimeError(
            "Gemma4 richiede un unico language_model per la preparazione selettiva; "
            f"trovati: {language_model_roots}"
        )
    language_model_root = language_model_roots[0]
    norms_cast_to_fp32: list[str] = []
    for name, module in model.named_modules():
        if not name.startswith(language_model_root + ".") or not _is_normalization_module(module):
            continue
        parameters = list(module.parameters(recurse=False))
        if not parameters or not any(parameter.dtype in (torch.float16, torch.bfloat16) for parameter in parameters):
            continue
        module.to(dtype=torch.float32)
        for parameter in module.parameters(recurse=False):
            if parameter.dtype != torch.float32 or parameter.requires_grad:
                raise RuntimeError(f"normalizzazione Gemma non congelata in FP32: {name}")
        norms_cast_to_fp32.append(name)
    if not norms_cast_to_fp32:
        raise RuntimeError(f"nessuna normalizzazione floating trovata sotto {language_model_root}")

    cache_configs: list[Any] = []
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "model", None), "config", None),
        getattr(getattr(getattr(model, "model", None), "language_model", None), "config", None),
    ):
        if config is not None and hasattr(config, "use_cache") and all(config is not item for item in cache_configs):
            config.use_cache = bool(use_cache_for_training)
            cache_configs.append(config)
    checkpointing_kwargs = None
    if not use_cache_for_training:
        enable_gradient_checkpointing = getattr(model, "gradient_checkpointing_enable", None)
        if not callable(enable_gradient_checkpointing):
            raise RuntimeError("Gemma4 runtime non espone gradient_checkpointing_enable")
        if "gradient_checkpointing_kwargs" not in inspect.signature(enable_gradient_checkpointing).parameters:
            raise RuntimeError("Transformers runtime non espone gradient_checkpointing_kwargs")
        checkpointing_kwargs = {"use_reentrant": False}
        enable_gradient_checkpointing(gradient_checkpointing_kwargs=checkpointing_kwargs)
    else:
        # Gemma4 5.5 uses the active cache to carry full-length K/V tensors
        # between KV-shared decoder layers. Reentrant decoder checkpointing
        # would replay mutations to that cache, so this mode keeps each
        # training forward uncached between batches but enables its fresh
        # per-forward DynamicCache and disables gradient checkpointing.
        disable_gradient_checkpointing = getattr(model, "gradient_checkpointing_disable", None)
        if callable(disable_gradient_checkpointing):
            disable_gradient_checkpointing()
        if bool(getattr(model, "is_gradient_checkpointing", False)):
            raise RuntimeError("Gemma4 shared-KV training requires gradient checkpointing disabled")
    return {
        "base_parameters_frozen": True,
        "norms_cast_to_fp32": norms_cast_to_fp32,
        "cache_disabled": bool(cache_configs) and not use_cache_for_training,
        "cache_enabled_for_shared_kv": bool(cache_configs) and use_cache_for_training,
        "gradient_checkpointing": not use_cache_for_training,
        "gradient_checkpointing_kwargs": checkpointing_kwargs,
        "input_require_grads_hook": False,
    }


def _validate_native_shared_kv_cache(model: Any, output: Any) -> dict[str, Any]:
    """Confirm one forward used Gemma's native cache-backed K/V sharing."""
    shared_by_layer: dict[int, int] = {}
    for _, module in model.named_modules():
        attention = getattr(module, "self_attn", module)
        if not bool(getattr(attention, "is_kv_shared_layer", False)):
            continue
        layer_index = getattr(attention, "layer_idx", None)
        source_index = getattr(attention, "kv_shared_layer_index", None)
        if not isinstance(layer_index, int) or not isinstance(source_index, int):
            raise RuntimeError(
                "Gemma4 espone un layer K/V condiviso senza layer_idx/kv_shared_layer_index interi; "
                "impossibile confermare la semantica del cache nativo"
            )
        previous_source = shared_by_layer.get(layer_index)
        if previous_source is not None and previous_source != source_index:
            raise RuntimeError(
                f"Il layer Gemma4 {layer_index} dichiara sorgenti K/V condivise incompatibili"
            )
        shared_by_layer[layer_index] = source_index
    shared_layers = sorted(shared_by_layer.items())
    if not shared_layers:
        raise RuntimeError(
            "Il modello Gemma4 caricato non espone layer K/V condivisi; "
            "la configurazione cache nativa non può essere verificata"
        )

    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError(
            "use_cache=True non ha restituito past_key_values nel forward audio Gemma4"
        )
    cache_sources = getattr(cache, "shared_layers", None)
    if not isinstance(cache_sources, Mapping):
        raise RuntimeError(
            "Il past_key_values Gemma4 non espone la mappa shared_layers attesa per la condivisione K/V nativa"
        )

    expected_sources = sorted({source_index for _, source_index in shared_layers})
    try:
        observed_sources = sorted({int(source_index) for source_index in cache_sources.keys()})
    except (TypeError, ValueError) as exc:
        raise RuntimeError("La mappa shared_layers contiene indici sorgente non interi") from exc
    if observed_sources != expected_sources:
        raise RuntimeError(
            "Le sorgenti K/V nel cache non corrispondono alla topologia Gemma4: "
            f"attese={expected_sources}, osservate={observed_sources}"
        )
    return {
        "cache_type": type(cache).__name__,
        "first_shared_layer_index": shared_layers[0][0],
        "shared_attention_layer_count": len(shared_layers),
        "expected_cache_source_indices": expected_sources,
        "observed_cache_source_indices": observed_sources,
    }


def _cast_lm_head_input_to_weight_dtype(module: Any, inputs: tuple[Any, ...]):
    if not inputs or not isinstance(inputs[0], torch.Tensor):
        return None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.Tensor) or not weight.is_floating_point():
        raise RuntimeError("lm_head Gemma non espone un peso floating per il bridge di dtype")
    if inputs[0].dtype == weight.dtype:
        return None
    # Tensor.to keeps autograd connected; this only changes the activation dtype
    # at the frozen output projection boundary.
    return (inputs[0].to(dtype=weight.dtype), *inputs[1:])


def _freeze_float_towers(model: Any, *, dtype: torch.dtype) -> dict[str, dict[str, Any]]:
    """Keep Gemma audio/vision towers frozen and in a floating compute dtype."""
    named_modules = list(model.named_modules())
    report: dict[str, dict[str, Any]] = {}
    for tower_name in _FLOAT_TOWER_NAMES:
        roots = [
            (name, module)
            for name, module in named_modules
            if name and name.split(".")[-1] == tower_name
        ]
        if len(roots) != 1:
            raise RuntimeError(
                f"Gemma richiede un unico sottoalbero {tower_name}; trovati: {[name for name, _ in roots]}"
            )
        root_name, root = roots[0]
        quantized_modules = [
            name for name, module in named_modules
            if (name == root_name or name.startswith(root_name + ".")) and _is_linear4bit(module)
        ]
        quantized_parameters = [
            f"{name}.{parameter_name}" if parameter_name else name
            for name, module in named_modules
            if name == root_name or name.startswith(root_name + ".")
            for parameter_name, parameter in module.named_parameters(recurse=False)
            if type(parameter).__name__ == "Params4bit"
        ]
        if quantized_modules or quantized_parameters:
            raise RuntimeError(
                f"tower Gemma {tower_name} quantizzato; Linear4bit={quantized_modules}, "
                f"Params4bit={quantized_parameters}"
            )
        root.to(dtype=dtype)
        for parameter in root.parameters():
            parameter.requires_grad_(False)
        parameters = [
            (f"{name}.{parameter_name}" if parameter_name else name, parameter)
            for name, module in named_modules
            if name == root_name or name.startswith(root_name + ".")
            for parameter_name, parameter in module.named_parameters(recurse=False)
        ]
        if not parameters:
            raise RuntimeError(f"tower Gemma {tower_name} senza parametri verificabili")
        bad_dtype = [name for name, parameter in parameters if not parameter.is_floating_point()]
        trainable = [name for name, parameter in parameters if parameter.requires_grad]
        if bad_dtype or trainable:
            raise RuntimeError(
                f"tower Gemma {tower_name} non è floating/frozen: "
                f"dtype_nonfloating={bad_dtype}, parametri_trainabili={trainable}"
            )
        report[tower_name] = {
            "dtypes": sorted({str(parameter.dtype) for _, parameter in parameters}),
            "requires_grad": False,
            "parameter_count": sum(parameter.numel() for _, parameter in parameters),
            "linear4bit_count": 0,
        }
    return report


def _quantization_runtime_report(
    model: Any,
    towers: Mapping[str, Mapping[str, Any]],
    bridges: Mapping[str, Mapping[str, Any]],
    input_embeddings: Mapping[str, Any],
) -> dict[str, Any]:
    """Record that only language-model projections were quantized."""
    quantized = [
        name for name, module in model.named_modules()
        if name and _is_linear4bit(module)
    ]
    language_quantized = [
        name for name in quantized if "language_model" in name.split(".")
    ]
    outside_language = [name for name in quantized if "language_model" not in name.split(".")]
    if outside_language:
        raise RuntimeError(f"Linear4bit fuori dal language_model Gemma: {outside_language}")
    if not language_quantized:
        raise RuntimeError("QLoRA Gemma non ha quantizzato alcun modulo del language_model")
    return {
        "llm_int8_skip_modules": list(_QUANTIZATION_SKIP_MODULES),
        **{name: dict(status) for name, status in towers.items()},
        "embedding_bridges": {name: dict(status) for name, status in bridges.items()},
        "input_embeddings": dict(input_embeddings),
        "language_model_quantized_module_count": len(language_quantized),
        "language_model_quantized_module_names": sorted(language_quantized),
    }


def _freeze_float_bridges(model: Any, *, dtype: torch.dtype) -> dict[str, dict[str, Any]]:
    """Keep audio/vision embedding projections frozen in the Gemma input dtype."""
    named_modules = list(model.named_modules())
    report: dict[str, dict[str, Any]] = {}
    for bridge_name in _FLOAT_BRIDGE_NAMES:
        matches = [
            (name, module)
            for name, module in named_modules
            if name == bridge_name or name.endswith("." + bridge_name)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Gemma richiede un unico embedding bridge {bridge_name}; "
                f"trovati: {[name for name, _ in matches]}"
            )
        root_name, root = matches[0]
        quantized_modules = [
            name for name, module in named_modules
            if (name == root_name or name.startswith(root_name + ".")) and _is_linear4bit(module)
        ]
        parameters = list(root.named_parameters())
        if not parameters or quantized_modules or any(type(p).__name__ == "Params4bit" for _, p in parameters):
            raise RuntimeError(
                f"embedding bridge Gemma non può restare quantizzato: {bridge_name}; "
                f"Linear4bit={quantized_modules}"
            )
        root.to(dtype=dtype)
        for parameter in root.parameters():
            parameter.requires_grad_(False)
        parameters = list(root.named_parameters())
        if any(not parameter.is_floating_point() or parameter.requires_grad for _, parameter in parameters):
            raise RuntimeError(f"embedding bridge Gemma non è floating/frozen: {bridge_name}")
        report[bridge_name] = {
            "dtypes": sorted({str(parameter.dtype) for _, parameter in parameters}),
            "requires_grad": False,
            "parameter_count": sum(parameter.numel() for _, parameter in parameters),
            "linear4bit_count": 0,
        }
    return report


def _freeze_input_embeddings(model: Any, *, dtype: torch.dtype) -> dict[str, Any]:
    """Match text token embeddings to the audio soft-token dtype after PEFT upcasting."""
    get_base = getattr(model, "get_base_model", None)
    owner = get_base() if callable(get_base) else model
    get_input = getattr(owner, "get_input_embeddings", None)
    get_output = getattr(owner, "get_output_embeddings", None)
    if not callable(get_input) or not callable(get_output):
        raise RuntimeError("Gemma deve esporre input/output embeddings per il merge audio-testuale")
    input_module = get_input()
    output_module = get_output()
    input_weight = getattr(input_module, "weight", None)
    output_weight = getattr(output_module, "weight", None)
    if not isinstance(input_weight, torch.Tensor) or not isinstance(output_weight, torch.Tensor):
        raise RuntimeError("Gemma non espone pesi verificabili per input embedding e output head")
    tied_before = input_weight is output_weight or input_weight.data_ptr() == output_weight.data_ptr()
    if not input_weight.is_floating_point():
        raise RuntimeError(f"input embedding Gemma non floating: {input_weight.dtype}")
    if not output_weight.is_floating_point():
        raise RuntimeError(f"output head Gemma non floating: {output_weight.dtype}")
    # Gemma's source embedding/head weights are tied. Keep the input side in
    # BF16 for the audio bridge, then give the frozen CE head its own FP32 copy.

    input_module.to(dtype=dtype)
    for parameter in input_module.parameters():
        parameter.requires_grad_(False)

    input_weight = getattr(get_input(), "weight", None)
    output_module = get_output()
    output_weight = getattr(output_module, "weight", None)
    if tied_before:
        if not isinstance(input_weight, torch.Tensor) or not isinstance(output_weight, torch.Tensor):
            raise RuntimeError("pesi input/output embedding Gemma persi durante il cast")
        if output_weight is input_weight or output_weight.data_ptr() == input_weight.data_ptr():
            output_head_copy = output_weight.detach().to(dtype=torch.float32)
            output_module.weight = torch.nn.Parameter(output_head_copy, requires_grad=False)
            output_weight = getattr(get_output(), "weight", None)
    else:
        output_module.to(dtype=torch.float32)
        output_weight = getattr(get_output(), "weight", None)
    if not isinstance(input_weight, torch.Tensor) or input_weight.dtype != dtype:
        raise RuntimeError(f"input embedding Gemma non è nel compute dtype {dtype}")
    if not isinstance(output_weight, torch.Tensor) or not output_weight.is_floating_point():
        raise RuntimeError("output head Gemma non è rimasto floating dopo la separazione dtype")
    if tied_before and (output_weight is input_weight or output_weight.data_ptr() == input_weight.data_ptr()):
        raise RuntimeError("input embedding BF16 e output head devono essere separati per il dtype CE")
    if not getattr(output_module, "_gemma4_lm_head_dtype_bridge_registered", False):
        register_pre_hook = getattr(output_module, "register_forward_pre_hook", None)
        if not callable(register_pre_hook):
            raise RuntimeError("lm_head Gemma non supporta il cast differenziabile dell’attivazione")
        register_pre_hook(_cast_lm_head_input_to_weight_dtype)
        output_module._gemma4_lm_head_dtype_bridge_registered = True
    if any(parameter.requires_grad for parameter in get_input().parameters()):
        raise RuntimeError("input embedding Gemma deve restare congelato in Stage1")
    for parameter in get_output().parameters():
        parameter.requires_grad_(False)
    return {
        "input_embedding_dtype": str(input_weight.dtype),
        "output_head_dtype": str(output_weight.dtype),
        "output_head_was_tied": tied_before,
        "output_head_tied": output_weight is input_weight,
        "output_head_input_cast": str(output_weight.dtype),
        "requires_grad": False,
        "parameter_count": sum(parameter.numel() for parameter in get_input().parameters()),
    }


def _move_batch_to_device(
    inputs: Mapping[str, torch.Tensor], *, device: torch.device, compute_dtype: torch.dtype
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Move processor tensors without deriving feature dtype from a quantized weight."""
    if not compute_dtype.is_floating_point:
        raise ValueError(f"dtype di compute audio non floating: {compute_dtype}")
    features = inputs.get("input_features")
    labels = inputs.get("labels")
    if features is not None and (not isinstance(features, torch.Tensor) or not features.is_floating_point()):
        raise RuntimeError("input_features audio devono essere un tensor floating")
    if not isinstance(labels, torch.Tensor):
        raise RuntimeError("labels causali mancanti nel batch QLoRA")
    non_ignored = int(labels.ne(-100).sum().item())
    if non_ignored <= 0:
        raise RuntimeError("batch QLoRA senza token target assistant non mascherati")
    moved = {
        key: value.to(device=device, dtype=compute_dtype if key == "input_features" else value.dtype)
        for key, value in inputs.items()
    }
    if "input_features" in moved and (
        moved["input_features"].dtype != compute_dtype or not moved["input_features"].is_floating_point()
    ):
        raise RuntimeError("input_features non convertite al compute dtype floating configurato")
    return moved, {
        "input_features_dtype": str(moved["input_features"].dtype) if "input_features" in moved else None,
        "input_features_shape": list(moved["input_features"].shape) if "input_features" in moved else None,
        "non_ignored_label_tokens": non_ignored,
    }


def _normalize_decoded_text(text: str) -> str:
    return " ".join(text.split())


def _validate_probe_target(
    processor: Any, record: Mapping[str, Any], inputs: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    """Prove that the first numerical probe supervises only the assistant answer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt_text = _format_chat(processor, record, add_target=False)
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("chat template non ha restituito un prompt generation per la probe Stage1")
    input_ids, labels = inputs.get("input_ids"), inputs.get("labels")
    if (
        not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
        or not isinstance(labels, torch.Tensor)
        or labels.shape != input_ids.shape
    ):
        raise ValueError("probe Stage1 richiede input_ids/labels [1, sequence] allineati")
    target_positions = labels[0].ne(-100).nonzero(as_tuple=False).flatten()
    if target_positions.numel() == 0:
        raise ValueError("probe Stage1 senza label assistant supervisionate")
    first = int(target_positions[0].item())
    if first <= 0 or not bool(labels[0, :first].eq(-100).all()):
        raise ValueError("probe Stage1 richiede il prefisso prompt mascherato")
    expected_suffix = torch.arange(first, labels.shape[1], device=target_positions.device)
    if not torch.equal(target_positions, expected_suffix):
        raise ValueError("label assistant probe non contigue fino alla fine del full-chat; padding non supportato")
    target_ids = labels[0, first:].detach().cpu().tolist()
    actual_ids = input_ids[0, first:].detach().cpu().tolist()
    if target_ids != actual_ids:
        raise ValueError("label probe divergono dai token assistant nel full-chat input")
    target_encoding = tokenizer(str(record["target_text"]), add_special_tokens=False)
    expected_target_ids = target_encoding["input_ids"]
    if isinstance(expected_target_ids, torch.Tensor):
        expected_target_ids = expected_target_ids.detach().cpu().reshape(-1).tolist()
    elif expected_target_ids and isinstance(expected_target_ids[0], (list, tuple)):
        expected_target_ids = expected_target_ids[0]
    expected_target_ids = [int(value) for value in expected_target_ids]
    if not expected_target_ids or [int(value) for value in target_ids[:len(expected_target_ids)]] != expected_target_ids:
        raise ValueError(
            "token assistant probe non iniziano con la codifica esatta del target; "
            f"labels={target_ids}, target_token_ids={expected_target_ids}"
        )
    trailing_ids = [int(value) for value in target_ids[len(expected_target_ids):]]
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", ())}
    eot_ids = {
        int(value) for value in (
            getattr(tokenizer, "eot_id", None),
            getattr(tokenizer, "eos_token_id", None),
        ) if value is not None
    }
    decoded = tokenizer.decode(target_ids, skip_special_tokens=True)
    decoded_matches = _normalize_decoded_text(decoded) == _normalize_decoded_text(str(record["target_text"]))
    language = str(record.get("language") or "").strip().lower()
    italian = language in {"it", "ita", "italian"} or language.startswith("it-") or "italian" in language
    if record.get("synthetic") is not True or record.get("task_kind") != "conversation_response" or not italian:
        raise ValueError("probe richiede il primo record italiano gold di risposta conversazionale sintetica")
    return {
        "prompt_generation_header": bool(prompt_text.strip()),
        "prompt_token_count": first,
        "decoded_target_matches": decoded_matches,
        "decoded_label_matches_target": decoded_matches,
        "decoded_label_char_count": len(decoded),
        "target_text_char_count": len(str(record["target_text"])),
        "assistant_target_token_count": len(expected_target_ids),
        "supervised_label_token_count": len(target_ids),
        "trailing_label_token_ids_after_target": trailing_ids,
        "eot_token_ids": sorted(eot_ids),
        "eot_present_after_target": any(value in (special_ids | eot_ids) for value in trailing_ids),
        "record_source": str(record.get("source", "unknown")),
        "record_conversation_id": str(record.get("conversation_id", "")),
    }


def run_text_conditioning_probe(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    inputs: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    use_cache_for_training: bool = True,
) -> dict[str, Any]:
    """Run the exact text-only conversation-response path used by runtime mode."""
    if record.get("conditioning_mode") not in {"runtime_asr_text", "action_conditioned_text"} or record.get("audio_mask") is not False:
        raise ValueError("probe testuale richiede record conversation_response/runtime_asr_text action-conditioned senza audio")
    prompt_text = _format_chat(processor, record, add_target=False)
    full_text = _format_chat(processor, record, add_target=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    audio_markers = {
        str(marker) for marker in (
            getattr(processor, "audio_token", None),
            getattr(tokenizer, "audio_token", None),
            "<|audio|>",
            "<|soft_audio|>",
            "<|audio_perception|>",
            "<|soft_audio_text|>",
        ) if marker
    }
    audio_placeholder_count = sum(
        prompt_text.count(marker) + full_text.count(marker) for marker in audio_markers
    )
    if audio_placeholder_count:
        raise ValueError("runtime_asr_text probe contiene placeholder audio nel chat input")
    if "input_features" in inputs or "input_features_mask" in inputs:
        raise ValueError("runtime_asr_text probe deve usare esattamente input testuale senza feature audio")
    target_report = _validate_probe_target(processor, record, inputs)
    moved, batch_status = _move_batch_to_device(inputs, device=device, compute_dtype=compute_dtype)
    if "input_features" in moved or "input_features_mask" in moved:
        raise RuntimeError("runtime_asr_text probe ha ricevuto feature audio dopo il trasferimento")
    was_training = bool(getattr(model, "training", True))
    model.eval()
    try:
        with torch.no_grad():
            output = model(**moved, use_cache=use_cache_for_training, return_dict=True)
            loss = getattr(output, "loss", None)
            logits = getattr(output, "logits", None)
            if (
                not isinstance(loss, torch.Tensor)
                or loss.ndim != 0
                or not bool(torch.isfinite(loss))
                or not isinstance(logits, torch.Tensor)
                or logits.ndim != 3
                or logits.shape[:2] != moved["input_ids"].shape
                or not bool(torch.isfinite(logits).all())
            ):
                raise RuntimeError("probe runtime_asr_text ha logits/loss non finiti o shape inattesa")
            causal_targets = moved["labels"][:, 1:].ne(-100)
            if not bool(causal_targets.any()):
                raise RuntimeError("probe runtime_asr_text senza predizioni target next-token")
            text_loss = float(loss.detach().float().cpu())
    finally:
        model.train(was_training)
    return {
        **target_report,
        "conditioning_mode": str(record.get("conditioning_mode")),
        "text_only_loss": text_loss,
        "audio_placeholder_count": audio_placeholder_count,
        "input_features_present": False,
        "input_features_dtype": batch_status["input_features_dtype"],
        "input_features_shape": batch_status["input_features_shape"],
        "non_ignored_label_tokens": batch_status["non_ignored_label_tokens"],
        "causal_target_prediction_count": int(causal_targets.sum().item()),
    }


def run_audio_text_probe(
    model: Any,
    processor: Any,
    record: Mapping[str, Any],
    inputs: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    use_cache_for_training: bool = True,
) -> dict[str, Any]:
    """Compare one gold example with/without only its real audio tensors."""
    if "input_features" not in inputs:
        raise ValueError("probe audio/text senza input_features audio")
    target_report = _validate_probe_target(processor, record, inputs)
    audio_inputs, batch_status = _move_batch_to_device(
        inputs, device=device, compute_dtype=compute_dtype
    )
    audio_only_keys = {key for key in ("input_features", "input_features_mask") if key in audio_inputs}
    if not audio_only_keys:
        raise ValueError("probe non ha individuato le feature/mask audio da rimuovere")
    text_inputs = {key: value for key, value in audio_inputs.items() if key not in audio_only_keys}
    if set(audio_inputs) - set(text_inputs) != audio_only_keys:
        raise RuntimeError("probe text-only ha rimosso input oltre alle feature audio e alla loro mask")
    was_training = bool(getattr(model, "training", True))
    model.eval()
    try:
        with torch.no_grad():
            audio_output = model(**audio_inputs, use_cache=use_cache_for_training, return_dict=True)
            native_shared_kv_cache = (
                _validate_native_shared_kv_cache(model, audio_output)
                if use_cache_for_training else None
            )
            audio_loss = getattr(audio_output, "loss", None)
            audio_logits = getattr(audio_output, "logits", None)
            if (
                not isinstance(audio_loss, torch.Tensor)
                or audio_loss.ndim != 0
                or not bool(torch.isfinite(audio_loss))
                or not isinstance(audio_logits, torch.Tensor)
                or not bool(torch.isfinite(audio_logits).all())
            ):
                raise RuntimeError("probe audio-conditioned ha logits/loss non finiti")
            next_token_labels = audio_inputs["labels"][:, 1:]
            supervised = next_token_labels.ne(-100)
            if not bool(supervised.any()):
                raise RuntimeError("probe senza predizioni next-token assistant supervisionate")
            audio_target_logits = audio_logits[:, :-1][supervised].detach().float().cpu()
            audio_loss_value = float(audio_loss.detach().float().cpu())
            del audio_output, audio_logits, audio_loss

            text_output = model(**text_inputs, use_cache=use_cache_for_training, return_dict=True)
            text_loss = getattr(text_output, "loss", None)
            text_logits = getattr(text_output, "logits", None)
            if (
                not isinstance(text_loss, torch.Tensor)
                or text_loss.ndim != 0
                or not bool(torch.isfinite(text_loss))
                or not isinstance(text_logits, torch.Tensor)
                or not bool(torch.isfinite(text_logits).all())
            ):
                raise RuntimeError("probe text-only ha logits/loss non finiti")
            text_target_logits = text_logits[:, :-1][supervised].detach().float().cpu()
            text_loss_value = float(text_loss.detach().float().cpu())
    finally:
        model.train(was_training)
    max_logit_diff = float((audio_target_logits - text_target_logits).abs().max().item())
    if not math.isfinite(max_logit_diff) or max_logit_diff <= 0:
        raise RuntimeError("probe audio/text non mostra alcuna differenza nei logits target")
    return {
        **target_report,
        "audio_loss": audio_loss_value,
        "text_only_loss": text_loss_value,
        "audio_minus_text_only_loss": audio_loss_value - text_loss_value,
        "max_target_logit_abs_diff": max_logit_diff,
        "causal_target_prediction_count": int(supervised.sum().item()),
        "audio_input_keys_removed_for_text_only": sorted(audio_only_keys),
        "input_features_dtype": batch_status["input_features_dtype"],
        "input_features_shape": batch_status["input_features_shape"],
        "non_ignored_label_tokens": batch_status["non_ignored_label_tokens"],
        "native_shared_kv_cache": native_shared_kv_cache,
    }


def _language_model_lora_target_modules(model: Any) -> list[str]:
    """Return exact projection module paths from the model's language_model branch."""
    module_names = [name for name, _ in model.named_modules() if name]
    roots = {
        ".".join(parts[: index + 1])
        for name in module_names
        for parts in (name.split("."),)
        for index, part in enumerate(parts)
        if part == "language_model"
    }
    if len(roots) != 1:
        raise RuntimeError(
            "Gemma QLoRA richiede un unico sottoalbero language_model verificabile; "
            f"trovati: {sorted(roots)}"
        )
    root = next(iter(roots))
    targets = sorted(
        name
        for name in module_names
        if name.startswith(root + ".") and name.rsplit(".", 1)[-1] in _LORA_LANGUAGE_PROJECTIONS
    )
    found_projections = {name.rsplit(".", 1)[-1] for name in targets}
    missing = sorted(_LORA_LANGUAGE_PROJECTIONS - found_projections)
    if missing:
        raise RuntimeError(
            "Gemma language_model non espone tutte le proiezioni LoRA approvate; "
            f"mancano: {missing}"
        )
    return targets


def _verify_lora_scope(model: Any, target_modules: list[str]) -> list[str]:
    """Fail closed unless PEFT injected LoRA on exactly the selected module paths."""
    targets = set(target_modules)
    if not targets or len(targets) != len(target_modules):
        raise RuntimeError("target LoRA Gemma vuoti o duplicati")
    matched: set[str] = set()
    unexpected: list[str] = []
    for name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)
        if lora_a is None or lora_b is None or not len(lora_a) or not len(lora_b):
            continue
        path_matches = [target for target in targets if name == target or name.endswith("." + target)]
        if len(path_matches) != 1:
            unexpected.append(name)
            continue
        matched.add(path_matches[0])
    missing = sorted(targets - matched)
    if unexpected or missing:
        raise RuntimeError(
            "PEFT LoRA scope differs from the selected Gemma language_model targets; "
            f"outside the selected Gemma language_model targets={sorted(unexpected)}, missing={missing}"
        )
    return sorted(matched)


def _training_manifest_path(adapter_dir: str | Path) -> Path:
    """Keep the Stage1 report beside its adapter, scoped to that checkpoint root."""
    return Path(adapter_dir).expanduser().resolve().parent / "training_manifest.json"


def _summarize_batch_runtime(step_runtime: list[Mapping[str, Any]]) -> dict[str, Any]:
    elapsed = [float(item["step_elapsed_seconds"]) for item in step_runtime]
    allocated = [
        int(item["cuda_peak_memory_allocated_bytes"])
        for item in step_runtime if item.get("cuda_peak_memory_allocated_bytes") is not None
    ]
    reserved = [
        int(item["cuda_peak_memory_reserved_bytes"])
        for item in step_runtime if item.get("cuda_peak_memory_reserved_bytes") is not None
    ]
    return {
        "examples_by_task": dict(sorted(Counter(str(item["task_kind"]) for item in step_runtime).items())),
        "examples_by_source": dict(sorted(Counter(str(item["source"]) for item in step_runtime).items())),
        "examples_by_action": dict(sorted(Counter(
            str(item["action"]) for item in step_runtime if item.get("action") is not None
        ).items())),
        "mean_step_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else None,
        "max_step_elapsed_seconds": max(elapsed) if elapsed else None,
        "cuda_peak_memory_allocated_bytes": max(allocated) if allocated else None,
        "cuda_peak_memory_reserved_bytes": max(reserved) if reserved else None,
    }


def _write_training_manifest(path: str | Path, manifest: dict[str, Any]) -> Path | None:
    """Atomically publish final metadata while retaining any prior smoke report."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup_path = None
    if destination.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = destination.with_name(
            f"{destination.stem}.backup-{stamp}-{uuid.uuid4().hex[:8]}{destination.suffix}"
        )
        shutil.copy2(destination, backup_path)
        manifest["previous_manifest_backup"] = str(backup_path)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    return backup_path


def _save_final_adapter(model: Any, processor: Any, adapter_dir: str | Path) -> Path:
    """Publish the final adapter atomically without replacing an earlier run."""
    destination = Path(adapter_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"adapter finale già esistente, non sovrascritto: {destination}")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir()
    model.save_pretrained(temporary, save_embedding_layers=False)
    processor.save_pretrained(temporary)
    os.rename(temporary, destination)
    return destination


def _role(segment: Mapping[str, Any], channel_map: Mapping[str, Any]) -> str | None:
    explicit = segment.get("role", segment.get("speaker_id"))
    if isinstance(explicit, str) and explicit in {"user", "assistant"}:
        return str(explicit)
    # Synthetic v3 explicitly defines utterance.speaker as the fictional
    # conversation role. Imported corpora use speaker for track IDs, so only
    # interpret it as a role when the canonicalizer marked the row synthetic.
    if segment.get("synthetic_role") is True and segment.get("speaker") in {"user", "assistant"}:
        return str(segment["speaker"])
    channel = segment.get("channel", segment.get("channel_id"))
    mapped = channel_map.get(str(channel)) if channel is not None else None
    return str(mapped) if isinstance(mapped, str) and mapped in {"user", "assistant"} else None


def _assistant_segments(conversation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if conversation.get("language_training_eligible") is False:
        return []
    channel_map = conversation.get("channel_map") or {}
    return [
        segment for segment in conversation.get("segments", ())
        if _role(segment, channel_map) == "assistant"
        and segment.get("target_mask", True) is True
        and segment.get("text_mask", True) is True
        and segment.get("time_mask", True) is True
        and isinstance(segment.get("text", segment.get("script_text", segment.get("transcript"))), str)
        and str(segment.get("text", segment.get("script_text", segment.get("transcript")))).strip()
    ]


def count_response_examples(conversations: Iterable[Mapping[str, Any]]) -> int:
    return sum(len(_assistant_segments(row)) for row in conversations)


def _audio_cache_stats() -> dict[str, Any]:
    return {
        "source_pcm": {
            **dict(_AUDIO_PCM_CACHE_STATS),
            "entries": len(_AUDIO_PCM_CACHE),
            "bytes": _AUDIO_PCM_CACHE_BYTES,
            "max_bytes": _AUDIO_PCM_CACHE_MAX_BYTES,
        },
        "source_integrity": {
            "hash_checks": _AUDIO_PCM_CACHE_STATS["hash_checks"],
            "verified_files": len(_VERIFIED_AUDIO_FILE_HASHES),
            "policy": "verify each unchanged source file once by SHA256; stat changes trigger re-verification",
        },
        "encoded_record_inputs": {
            **dict(_TRAINING_INPUT_CACHE_STATS),
            "entries": len(_TRAINING_INPUT_CACHE),
            "bytes": _TRAINING_INPUT_CACHE_BYTES,
            "max_bytes": _TRAINING_INPUT_CACHE_MAX_BYTES,
        },
    }


def _read_cached_audio_file(path: Path) -> tuple[np.ndarray, int]:
    """Decode each unchanged source WAV once, with an explicit bounded LRU."""
    global _AUDIO_PCM_CACHE_BYTES
    resolved = path.resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _AUDIO_PCM_CACHE.get(key)
    if cached is not None:
        _AUDIO_PCM_CACHE.move_to_end(key)
        _AUDIO_PCM_CACHE_STATS["hits"] += 1
        return cached
    # A changed file invalidates older entries at this path before a fresh decode.
    for old_key in [item for item in _AUDIO_PCM_CACHE if item[0] == str(resolved)]:
        old_audio, _ = _AUDIO_PCM_CACHE.pop(old_key)
        _AUDIO_PCM_CACHE_BYTES -= int(old_audio.nbytes)
    import soundfile as sf

    audio, sample_rate = sf.read(str(resolved), dtype="float32", always_2d=True)
    audio = np.asarray(audio, dtype=np.float32)
    _AUDIO_PCM_CACHE_STATS["decodes"] += 1
    if audio.nbytes <= _AUDIO_PCM_CACHE_MAX_BYTES:
        while _AUDIO_PCM_CACHE and _AUDIO_PCM_CACHE_BYTES + audio.nbytes > _AUDIO_PCM_CACHE_MAX_BYTES:
            _, (old_audio, _) = _AUDIO_PCM_CACHE.popitem(last=False)
            _AUDIO_PCM_CACHE_BYTES -= int(old_audio.nbytes)
            _AUDIO_PCM_CACHE_STATS["evictions"] += 1
        _AUDIO_PCM_CACHE[key] = (audio, int(sample_rate))
        _AUDIO_PCM_CACHE_BYTES += int(audio.nbytes)
    else:
        _AUDIO_PCM_CACHE_STATS["oversized_not_cached"] += 1
    return audio, int(sample_rate)


def _training_input_cache_key(processor: Any, record: Mapping[str, Any]) -> tuple[int, str] | None:
    if not _is_audio_response_record(record):
        return None
    path = record.get("audio_path")
    if not isinstance(path, str) or not Path(path).is_file():
        return None
    stat = Path(path).stat()
    payload = json.dumps(
        {
            "record": record,
            "audio_stat": [int(stat.st_size), int(stat.st_mtime_ns)],
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    ).encode("utf-8")
    return id(processor), hashlib.sha256(payload).hexdigest()


def _cache_training_inputs(key: tuple[int, str], inputs: dict[str, torch.Tensor]) -> None:
    global _TRAINING_INPUT_CACHE_BYTES
    size = sum(int(tensor.numel() * tensor.element_size()) for tensor in inputs.values())
    if size > _TRAINING_INPUT_CACHE_MAX_BYTES:
        _TRAINING_INPUT_CACHE_STATS["oversized_not_cached"] += 1
        return
    existing = _TRAINING_INPUT_CACHE.pop(key, None)
    if existing is not None:
        _TRAINING_INPUT_CACHE_BYTES -= existing[1]
    while _TRAINING_INPUT_CACHE and _TRAINING_INPUT_CACHE_BYTES + size > _TRAINING_INPUT_CACHE_MAX_BYTES:
        _, (_, old_size) = _TRAINING_INPUT_CACHE.popitem(last=False)
        _TRAINING_INPUT_CACHE_BYTES -= old_size
        _TRAINING_INPUT_CACHE_STATS["evictions"] += 1
    _TRAINING_INPUT_CACHE[key] = ({name: tensor.detach().cpu() for name, tensor in inputs.items()}, size)
    _TRAINING_INPUT_CACHE_BYTES += size


def _read_audio_for_record(record: Mapping[str, Any], *, target_sample_rate_hz: int = 16000) -> np.ndarray:
    """Read an actual mono channel/span and resample it to Gemma's 16 kHz input."""
    path = record.get("audio_path")
    if not path or not Path(str(path)).is_file():
        raise FileNotFoundError(f"audio QLoRA mancante: {path}")
    source_path = Path(str(path)).resolve()
    source_stat = source_path.stat()
    expected_sha = record.get("source_audio_sha256")
    if expected_sha is not None:
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ValueError(f"{source_path}: SHA256 sorgente non valido")
        hash_key = (str(source_path), int(source_stat.st_size), int(source_stat.st_mtime_ns), expected_sha.lower())
        stale_hash_keys = {
            key for key in _VERIFIED_AUDIO_FILE_HASHES
            if key[0] == str(source_path) and key != hash_key
        }
        _VERIFIED_AUDIO_FILE_HASHES.difference_update(stale_hash_keys)
        if hash_key not in _VERIFIED_AUDIO_FILE_HASHES:
            digest = hashlib.sha256()
            with source_path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != expected_sha.lower():
                raise ValueError(f"{source_path}: audio source hash non coincide col record preparato")
            _VERIFIED_AUDIO_FILE_HASHES.add(hash_key)
            _AUDIO_PCM_CACHE_STATS["hash_checks"] += 1
    audio, sample_rate = _read_cached_audio_file(source_path)
    declared_rate = record.get("sample_rate_hz")
    if declared_rate is not None and sample_rate != int(declared_rate):
        raise ValueError(f"{path}: sample rate dichiarato diverso dal file")
    channel = record.get("audio_channel_index")
    if isinstance(channel, bool) or not isinstance(channel, int):
        raise ValueError(f"{path}: audio_channel_index esplicito richiesto")
    if channel >= audio.shape[1]:
        raise ValueError(f"{path}: canale {channel} fuori range ({audio.shape[1]})")
    if channel < 0:
        raise ValueError(f"{path}: canale negativo: {channel}")
    start_ms, end_ms = record.get("audio_start_ms"), record.get("audio_end_ms")
    if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
        raise ValueError(f"{path}: intervallo audio non numerico")
    if not (math.isfinite(start_ms) and math.isfinite(end_ms) and 0 <= start_ms < end_ms):
        raise ValueError(f"{path}: intervallo audio non valido: [{start_ms}, {end_ms})")
    start = round(float(start_ms) * sample_rate / 1000.0)
    end = round(float(end_ms) * sample_rate / 1000.0)
    if end > audio.shape[0] or start < 0 or end <= start:
        raise ValueError(f"{path}: span audio oltre i campioni presenti: [{start}, {end}) / {len(audio)}")
    mono = np.asarray(audio[start:end, channel], dtype=np.float32)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"{path}: PCM vuoto/non finito per l'obiettivo audio")
    if not float(np.max(np.abs(mono))) > 1e-7:
        raise ValueError(f"{path}: PCM silenzioso; non usare un audio marker come se fosse input supervisionato")
    if sample_rate != target_sample_rate_hz:
        from scipy.signal import resample_poly
        from math import gcd

        divisor = gcd(sample_rate, target_sample_rate_hz)
        mono = resample_poly(
            mono,
            up=target_sample_rate_hz // divisor,
            down=sample_rate // divisor,
        ).astype(np.float32, copy=False)
    if mono.size == 0 or not np.isfinite(mono).all():
        raise ValueError(f"{path}: PCM resampled vuoto/non finito")
    return mono


def _segment_text(segment: Mapping[str, Any]) -> str:
    value = segment.get("text", segment.get("script_text", segment.get("transcript", "")))
    return value.strip() if isinstance(value, str) else ""


def _training_records(
    conversations: Iterable[Mapping[str, Any]],
    *,
    conditioning_mode: str = "baselineaudio_response",
) -> list[dict[str, Any]]:
    """Build response and role-free ASR tasks from supported, timed audio spans."""
    if conditioning_mode not in {"baselineaudio_response", "runtime_asr_text"}:
        raise ValueError(f"conditioning_mode Stage1 non supportata: {conditioning_mode}")
    # Match GemmaListener's rolling PCM history rather than training on whole
    # dialogue files that runtime can never present to the listener.
    from duplex_kernel.listener import DEFAULT_CONTEXT_FRAMES

    listener_history_ms = int(DEFAULT_CONTEXT_FRAMES) * 80
    records: list[dict[str, Any]] = []
    for conversation in conversations:
        # A source may retain valid audio/event frames while its transcript is
        # explicitly rejected for language learning. Keep that row available
        # to Stage2 feature extraction, but do not turn its text into either a
        # target or conversational context for Stage1.
        if conversation.get("language_training_eligible") is False:
            continue
        channel_map = conversation.get("channel_map") or {}
        source = str(conversation.get("source", "unknown"))
        synthetic = bool(conversation.get("synthetic"))
        duration_ms = conversation.get("duration_ms")
        if duration_ms is None and isinstance(conversation.get("duration_s"), (int, float)):
            duration_ms = float(conversation["duration_s"]) * 1000.0
        segments = sorted(
            conversation.get("segments", ()),
            key=lambda item: float(item.get("start_ms", item.get("start_s", 0) * 1000))
            if isinstance(item.get("start_ms", item.get("start_s", 0) * 1000), (int, float)) else 0,
        )
        assistant_channel = next(
            (int(channel) for channel, role in channel_map.items() if role == "assistant" and str(channel).isdigit()),
            None,
        )
        user_channel = next(
            (int(channel) for channel, role in channel_map.items() if role == "user" and str(channel).isdigit()),
            None,
        )
        for target in segments:
            text = _segment_text(target)
            if (
                not text
                or target.get("target_mask", True) is not True
                or target.get("text_mask", True) is not True
                or target.get("time_mask", True) is not True
            ):
                continue
            start_ms = target.get("start_ms")
            end_ms = target.get("end_ms")
            if start_ms is None and isinstance(target.get("start_s"), (int, float)):
                start_ms = float(target["start_s"]) * 1000.0
            if end_ms is None and isinstance(target.get("end_s"), (int, float)):
                end_ms = float(target["end_s"]) * 1000.0
            if not isinstance(start_ms, (int, float)) or not isinstance(end_ms, (int, float)):
                continue
            if not (math.isfinite(start_ms) and math.isfinite(end_ms) and 0 <= start_ms < end_ms):
                continue
            if isinstance(duration_ms, (int, float)) and end_ms > float(duration_ms) + 1e-4:
                continue

            role = _role(target, channel_map)
            is_asr = target.get("task_kind") == "asr"
            if is_asr:
                # Aux transcripts remain role-free and use only the physical
                # track/channel pairing verified by the source converter.
                channel_index = target.get("channel_index")
                if role is not None or target.get("audio_channel_verified") is not True:
                    continue
                if isinstance(channel_index, bool) or not isinstance(channel_index, int):
                    continue
                task_kind = "asr"
                audio_start_ms, audio_end_ms = float(start_ms), float(end_ms)
                context_text = ""
                language = target.get("language") or conversation.get("language")
            elif role == "assistant":
                if user_channel is None:
                    continue
                channel_index = user_channel
                task_kind = "conversation_response"
                probe_audio_start_ms = max(0.0, float(start_ms) - listener_history_ms)
                probe_audio_end_ms = float(start_ms)
                if conditioning_mode == "baselineaudio_response":
                    audio_start_ms = probe_audio_start_ms
                    audio_end_ms = probe_audio_end_ms
                    context_floor_ms = audio_start_ms
                else:
                    # Match serving: the user turn is first transcribed by
                    # Gemma ASR, then response generation receives chat text.
                    audio_start_ms = None
                    audio_end_ms = None
                    context_floor_ms = None
                context = []
                history_entries: list[tuple[float, float, str, dict[str, str]]] = []
                for segment in segments:
                    segment_start = segment.get("start_ms")
                    segment_end = segment.get("end_ms")
                    if segment_start is None and isinstance(segment.get("start_s"), (int, float)):
                        segment_start = float(segment["start_s"]) * 1000.0
                    if segment_end is None and isinstance(segment.get("end_s"), (int, float)):
                        segment_end = float(segment["end_s"]) * 1000.0
                    segment_role = _role(segment, channel_map)
                    segment_text = _segment_text(segment)
                    if (
                        segment is not target
                        and isinstance(segment_start, (int, float))
                        and isinstance(segment_end, (int, float))
                        and float(segment_start) < float(start_ms)
                        and float(segment_end) <= float(start_ms)
                        and (context_floor_ms is None or float(segment_end) >= context_floor_ms)
                        and segment.get("text_mask", True) is True
                        and segment.get("time_mask", True) is True
                        and segment_role in {"user", "assistant"}
                        and segment_text
                    ):
                        context.append(f"{segment_role}: {segment_text}")
                        history_entries.append((
                            float(segment_end),
                            float(segment_start),
                            segment_role,
                            {"role": segment_role, "content": segment_text},
                        ))
                # The live session commits assistant BCs when they finish and
                # appends the user transcript only after that utterance ends.
                # Order training history by completion to match this runtime
                # ordering when a BC overlaps an unfinished user turn.
                history_messages = [entry[3] for entry in sorted(
                    history_entries,
                    key=lambda entry: (entry[0], entry[1], entry[2]),
                )]
                if conditioning_mode == "runtime_asr_text" and (
                    not history_messages or history_messages[-1]["role"] != "user"
                ):
                    # Serving starts response generation only after a user
                    # transcript has been committed to the chat history.
                    continue
                context_text = "\n".join(context)
                language = target.get("language") or conversation.get("language") or "Italian"
                audio_history_cap_ms = listener_history_ms if conditioning_mode == "baselineaudio_response" else None
            else:
                continue
            if audio_start_ms is not None and audio_end_ms is not None and audio_end_ms <= audio_start_ms:
                continue
            if is_asr:
                probe_audio_start_ms = probe_audio_end_ms = None
                audio_history_cap_ms = None
            records.append(
                {
                    "conversation_id": str(conversation.get("conversation_id", "")),
                    "source_group_id": str(conversation.get("source_group_id", conversation.get("conversation_id", ""))),
                    "source": source,
                    "synthetic": synthetic,
                    "task_kind": task_kind,
                    "conditioning_mode": "native_audio_asr" if is_asr else conditioning_mode,
                    "language": language,
                    "audio_path": conversation.get("audio_path"),
                    "sample_rate_hz": conversation.get("sample_rate_hz", 16000),
                    "audio_channel_index": channel_index,
                    "audio_start_ms": float(audio_start_ms) if audio_start_ms is not None else None,
                    "audio_end_ms": float(audio_end_ms) if audio_end_ms is not None else None,
                    "audio_history_cap_ms": audio_history_cap_ms,
                    "audio_probe_start_ms": probe_audio_start_ms,
                    "audio_probe_end_ms": probe_audio_end_ms,
                    "audio_mask": is_asr or conditioning_mode == "baselineaudio_response",
                    "context_text": context_text,
                    "history_messages": history_messages if task_kind == "conversation_response" else [],
                    "target_text": text,
                }
            )
    return records


def _make_training_schedule(
    records: list[Mapping[str, Any]],
    *,
    max_steps: int,
    target_token_counts: list[int],
    synthetic_fraction: float = 0.8,
    seed: int = 20260930,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Oversample Italian synth and enforce its minimum target-token share."""
    if len(records) != len(target_token_counts):
        raise ValueError("training records e conteggi token non allineati")
    synthetic_indices = [i for i, row in enumerate(records) if bool(row.get("synthetic"))]
    other_indices = [i for i, row in enumerate(records) if not bool(row.get("synthetic"))]
    if not synthetic_indices:
        raise ValueError("stage1 richiede sintetico italiano come sorgente primaria")
    if max_steps <= 0 or not 0.7 <= synthetic_fraction <= 1.0:
        raise ValueError("max_steps positivo e synthetic_fraction tra 0.7 e 1.0 richiesti")
    if any(isinstance(value, bool) or int(value) <= 0 for value in target_token_counts):
        raise ValueError("ogni esempio deve avere almeno un token target")
    rng = random.Random(int(seed))
    n_primary = max(1, math.ceil(int(max_steps) * float(synthetic_fraction)))
    schedule = [rng.choice(synthetic_indices) for _ in range(n_primary)]
    schedule.extend(rng.choice(other_indices) for _ in range(max_steps - n_primary) if other_indices)
    while len(schedule) < max_steps:
        schedule.append(rng.choice(synthetic_indices))
    rng.shuffle(schedule)

    def token_fraction(indices: list[int]) -> float:
        primary_tokens = sum(int(target_token_counts[index]) for index in indices if index in synthetic_indices)
        all_tokens = sum(int(target_token_counts[index]) for index in indices)
        return primary_tokens / all_tokens if all_tokens else 0.0

    while token_fraction(schedule) < 0.7:
        aux_positions = [position for position, index in enumerate(schedule) if index in other_indices]
        if not aux_positions:
            break
        # Replace the largest auxiliary target first; this converges quickly
        # without silently allowing long aux transcripts to dominate loss.
        position = max(aux_positions, key=lambda item: int(target_token_counts[schedule[item]]))
        schedule[position] = rng.choice(synthetic_indices)
    selected = [records[index] for index in schedule]
    source_counts = Counter(str(row.get("source", "unknown")) for row in selected)
    source_tokens: Counter[str] = Counter()
    for index in schedule:
        source_tokens[str(records[index].get("source", "unknown"))] += int(target_token_counts[index])
    report = {
        "examples": len(selected),
        "synthetic_examples": sum(bool(row.get("synthetic")) for row in selected),
        "synthetic_example_fraction": sum(bool(row.get("synthetic")) for row in selected) / len(selected),
        "target_tokens_by_source": dict(source_tokens),
        "examples_by_source": dict(source_counts),
        "synthetic_target_token_fraction": sum(
            int(target_token_counts[index]) for index in schedule if index in synthetic_indices
        ) / sum(int(target_token_counts[index]) for index in schedule),
    }
    if report["synthetic_example_fraction"] < 0.7 or report["synthetic_target_token_fraction"] < 0.7:
        raise RuntimeError("training sampler non rispetta la quota sintetica primaria del 70%")
    return selected, report


def _validate_direct_audio_records(
    records: Iterable[Mapping[str, Any]], *, conditioning_mode: str,
) -> list[dict[str, Any]]:
    """Validate the flat per-response format emitted by the causal audio builder."""
    validated: list[dict[str, Any]] = []
    action_strata = {
        "BACKCHANNEL": "labelled_backchannel",
        "SPEAK": "labelled_speak",
        "REPLAN": "labelled_replan",
        "UNKNOWN": "conversational_regular",
    }
    for index, source in enumerate(records):
        record = dict(source)
        if record.get("task_kind") != "conversation_response":
            raise ValueError(f"direct audio record {index}: task_kind non è conversation_response")
        if record.get("conditioning_mode") != conditioning_mode or conditioning_mode != "baselineaudio_response":
            raise ValueError(f"direct audio record {index}: conditioning_mode incompatibile")
        if record.get("audio_response_contract") != "audio_response_conditioning_v1":
            raise ValueError(f"direct audio record {index}: marker del contratto audio mancante")
        if record.get("audio_mask") is not True or record.get("action_is_prompt_conditioning") is not False:
            raise ValueError(f"direct audio record {index}: audio reale richiesto e azione solo metadata")
        action = str(record.get("action", "")).upper()
        if action not in action_strata:
            raise ValueError(f"direct audio record {index}: action_label non supportata")
        target_start = record.get("target_start_ms")
        if isinstance(target_start, bool) or not isinstance(target_start, (int, float)) or not math.isfinite(float(target_start)):
            raise ValueError(f"direct audio record {index}: target_start_ms non valido")
        if record.get("action_label") != action.lower() or record.get("sampling_stratum") != action_strata[action]:
            raise ValueError(f"direct audio record {index}: action label/sampling stratum incoerenti")
        if action == "UNKNOWN":
            if record.get("action_label_provenance") != "unknown_no_aligned_event":
                raise ValueError(f"direct audio record {index}: target unknown deve restare senza action ground truth")
            if record.get("current_user_active_at_target") is not False:
                raise ValueError(f"direct audio record {index}: conversational_regular richiede user turn completo")
            if record.get("action_event_start_ms") is not None or record.get("action_event_end_ms") is not None:
                raise ValueError(f"direct audio record {index}: action UNKNOWN non può esporre un evento GT")
        elif record.get("action_label_provenance") not in {
            "onset_aligned_source_event", "measured_v5_event_schedule",
        }:
            raise ValueError(f"direct audio record {index}: action GT priva di provenienza da evento onset-aligned")
        elif (
            not isinstance(record.get("action_event_start_ms"), (int, float))
            or not isinstance(record.get("action_event_end_ms"), (int, float))
            or not float(record["action_event_start_ms"]) < float(record["action_event_end_ms"])
            or not (float(record["action_event_start_ms"]) <= float(record["target_start_ms"]) < float(record["action_event_end_ms"])
                    or abs(float(record["action_event_start_ms"]) - float(record["target_start_ms"])) <= 80.0)
        ):
            raise ValueError(f"direct audio record {index}: action event non temporalmente allineato")
        if not isinstance(record.get("target_text"), str) or not record["target_text"].strip():
            raise ValueError(f"direct audio record {index}: target_text vuoto")
        if not isinstance(record.get("audio_path"), str) or not record["audio_path"].strip():
            raise ValueError(f"direct audio record {index}: audio_path mancante")
        if isinstance(record.get("audio_channel_index"), bool) or not isinstance(record.get("audio_channel_index"), int):
            raise ValueError(f"direct audio record {index}: canale fisico user esplicito richiesto")
        if int(record.get("sample_rate_hz", 0)) <= 0:
            raise ValueError(f"direct audio record {index}: sample rate non valido")
        try:
            expected_window = causal_audio_window(
                action="SPEAK" if action == "UNKNOWN" else action,
                target_start_ms=record["target_start_ms"],
                current_user_start_ms=record["current_user_start_ms"],
                current_user_end_ms=record["current_user_end_ms"],
                current_user_active=record.get("current_user_active_at_target"),
                max_audio_ms=int(record.get("max_audio_ms", DEFAULT_MAX_AUDIO_MS)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"direct audio record {index}: finestra audio causale non valida: {exc}") from exc
        for key, expected in expected_window.items():
            actual = record.get(key)
            if isinstance(expected, float):
                if not isinstance(actual, (int, float)) or not math.isclose(float(actual), expected, abs_tol=1e-6):
                    raise ValueError(f"direct audio record {index}: {key} non coincide con la finestra causale")
            elif actual != expected:
                raise ValueError(f"direct audio record {index}: {key} non coincide con la finestra causale")
        if record.get("audio_start_ms") != record.get("current_user_start_ms"):
            raise ValueError(f"direct audio record {index}: l'audio non parte dall'onset utente")
        if not isinstance(record.get("source_audio_sha256"), str) or len(record["source_audio_sha256"]) != 64:
            raise ValueError(f"direct audio record {index}: SHA256 sorgente non valido")
        history = record.get("history_messages") or []
        if not isinstance(history, list):
            raise ValueError(f"direct audio record {index}: history_messages non è una lista")
        for message_index, message in enumerate(history):
            if not isinstance(message, Mapping) or message.get("role") not in {"user", "assistant"}:
                raise ValueError(f"direct audio record {index}: ruolo storico {message_index} non supportato")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError(f"direct audio record {index}: messaggio storico {message_index} vuoto")
        if record.get("context_text"):
            raise ValueError(f"direct audio record {index}: context_text potrebbe duplicare la trascrizione corrente")
        record["action"] = action
        validated.append(record)
    if not validated:
        raise ValueError("dataset direct audio vuoto")
    return validated


def _validate_direct_audio_protocol_v2_records(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the exact v2 JSON target and retain all v1 causal audio checks."""
    from duplex_kernel.audio_response_protocol import format_audio_response_v2_target

    validated: list[dict[str, Any]] = []
    for index, source in enumerate(records):
        record = dict(source)
        if record.get("audio_response_protocol") != _AUDIO_RESPONSE_PROTOCOL_V2:
            raise ValueError(f"direct audio v2 record {index}: marker heard/action/say mancante")
        try:
            payload = json.loads(str(record.get("target_text", "")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"direct audio v2 record {index}: target non è JSON valido") from exc
        if not isinstance(payload, dict) or list(payload) != ["heard", "action", "say"]:
            raise ValueError(f"direct audio v2 record {index}: target deve avere solo heard/action/say in quest'ordine")
        action = str(record.get("action", "")).upper()
        if action not in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
            raise ValueError(f"direct audio v2 record {index}: action non supportata")
        heard, say = payload.get("heard"), payload.get("say")
        if not isinstance(heard, str) or not isinstance(say, str) or not say.strip():
            raise ValueError(f"direct audio v2 record {index}: heard/say devono essere stringhe e say non vuoto")
        if payload.get("action") != action or record.get("target_say_text") != say:
            raise ValueError(f"direct audio v2 record {index}: target action/say divergono dai metadata")
        if record.get("target_heard_text") != heard:
            raise ValueError(f"direct audio v2 record {index}: heard diverge dal target-only metadata")
        if action == "BACKCHANNEL":
            if heard:
                raise ValueError(f"direct audio v2 record {index}: BACKCHANNEL deve supervisionare heard vuoto")
        else:
            cutoff, user_end = record.get("input_cutoff_ms"), record.get("current_user_end_ms")
            if (not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool)
                    or not isinstance(user_end, (int, float)) or isinstance(user_end, bool)
                    or float(user_end) > float(cutoff) or not heard.strip()):
                raise ValueError(f"direct audio v2 record {index}: heard S/R deve essere completo entro il cutoff")
        if record.get("target_text") != format_audio_response_v2_target(heard=heard, action=action, say=say):
            raise ValueError(f"direct audio v2 record {index}: target JSON non è compatto/canonico")
        legacy_audio_view = {
            **record,
            "audio_response_contract": _AUDIO_RESPONSE_PROTOCOL_V1,
            "target_text": say,
        }
        _validate_direct_audio_records([legacy_audio_view], conditioning_mode="baselineaudio_response")
        validated.append(record)
    if not validated:
        raise ValueError("dataset direct audio v2 vuoto")
    return validated


def _validate_audio_asr_mix_records(
    records: Iterable[Mapping[str, Any]], *, split: str, response_protocol: str = "v1",
    require_asr: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Validate mixed rows per task while keeping each task's native prompt path."""
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError(f"split {split} audio_asr_mix_{response_protocol} vuoto")
    if response_protocol not in {"v1", "v2"}:
        raise ValueError(f"split {split}: response_protocol non supportato: {response_protocol!r}")
    mix_contract = f"audio_asr_mix_{response_protocol}"
    if any(row.get("task_mix_contract") != mix_contract for row in rows):
        raise ValueError(f"split {split}: marker {mix_contract} mancante")
    direct_rows = [
        row for row in rows
        if (_is_audio_response_record(row) if response_protocol == "v1"
            else row.get("audio_response_protocol") == _AUDIO_RESPONSE_PROTOCOL_V2)
    ]
    asr_rows = [row for row in rows if row.get("task_kind") == "asr"]
    if len(direct_rows) + len(asr_rows) != len(rows):
        raise ValueError(f"split {split}: task_kind misto non supportato")
    direct = (
        _validate_direct_audio_records(direct_rows, conditioning_mode="baselineaudio_response")
        if response_protocol == "v1"
        else _validate_direct_audio_protocol_v2_records(direct_rows)
    )
    if any(row["action"] == "UNKNOWN" for row in direct):
        raise ValueError(f"split {split}: action UNKNOWN non appartiene al mix 20/40/20/20")
    if any(
        row["action"] == "BACKCHANNEL"
        and row.get("action_label_provenance") != "measured_v5_event_schedule"
        for row in direct
    ):
        raise ValueError(f"split {split}: BACKCHANNEL onset-only legacy escluso dal mix")

    validated_asr: list[dict[str, Any]] = []
    for index, row in enumerate(asr_rows):
        if row.get("source") != "synthetic_v3_clean_asr" or row.get("conditioning_mode") != "native_audio_asr":
            raise ValueError(f"ASR record {index}: provenance o conditioning_mode incompatibile")
        if row.get("audio_mask") is not True or not row.get("audio_path"):
            raise ValueError(f"ASR record {index}: audio reale richiesto")
        if not isinstance(row.get("audio_channel_index"), int) or isinstance(row.get("audio_channel_index"), bool):
            raise ValueError(f"ASR record {index}: canale fisico non dichiarato")
        if int(row.get("sample_rate_hz", 0)) <= 0:
            raise ValueError(f"ASR record {index}: sample rate non valido")
        start, end = row.get("audio_start_ms"), row.get("audio_end_ms")
        if (
            not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isinstance(end, (int, float)) or isinstance(end, bool)
            or not math.isfinite(float(start)) or not math.isfinite(float(end))
            or float(start) < 0 or float(end) <= float(start)
        ):
            raise ValueError(f"ASR record {index}: span audio non valido")
        if not isinstance(row.get("target_text"), str) or not row["target_text"].strip():
            raise ValueError(f"ASR record {index}: target transcript vuoto")
        if row.get("history_messages") or row.get("context_text"):
            raise ValueError(f"ASR record {index}: il prompt ASR non usa cronologia conversazionale")
        if not (row.get("source_group_id") or row.get("conversation_id")):
            raise ValueError(f"ASR record {index}: source group mancante")
        validated_asr.append(row)
    if not direct or (require_asr and not validated_asr):
        raise ValueError(f"split {split}: record direct audio e clean ASR richiesti={require_asr}")
    for row in direct:
        if row.get("current_user_transcript") or row.get("current_user_text"):
            raise ValueError(f"split {split}: il transcript corrente non può essere un campo di input")
    return rows, {"conversation_response": direct, "asr": validated_asr}


def _make_audio_asr_mix_schedule(
    records: list[Mapping[str, Any]], *, max_steps: int, seed: int,
    target_token_counts: list[int] | None = None,
    response_protocol: str = "v1",
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Sample exact 20/40/20/20 ASR/SPEAK/BC/REPLAN strata in each five-step cycle."""
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    if response_protocol not in {"v1", "v2"}:
        raise ValueError(f"response_protocol non supportato: {response_protocol!r}")
    mix_contract = f"audio_asr_mix_{response_protocol}"
    strata: dict[str, list[int]] = {"asr": [], "SPEAK": [], "BACKCHANNEL": [], "REPLAN": []}
    source_strata: dict[str, dict[str, list[int]]] = {
        action: {"authored_synthetic_audio_candidate": [], "synthetic_it_t6": []}
        for action in ("SPEAK", "BACKCHANNEL", "REPLAN")
    }
    for index, row in enumerate(records):
        if row.get("task_mix_contract") != mix_contract:
            raise ValueError(f"record privo del marker {mix_contract}")
        if row.get("task_kind") == "asr":
            strata["asr"].append(index)
        elif row.get("task_kind") == "conversation_response" and row.get("action") in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
            action = str(row["action"])
            source = str(row.get("source") or "")
            if source not in source_strata[action]:
                raise ValueError(f"{mix_contract} source non supportata per {action}: {source!r}")
            strata[action].append(index)
            source_strata[action][source].append(index)
        else:
            raise ValueError(f"record non campionabile nel mix: {row.get('task_kind')}/{row.get('action')}")
    missing = [name for name, values in strata.items() if not values]
    if missing:
        raise ValueError(f"{mix_contract} privo di strati richiesti: {missing}")
    rng = random.Random(int(seed))
    pattern = ("asr", "SPEAK", "BACKCHANNEL", "REPLAN", "SPEAK")
    schedule: list[int] = []
    action_occurrences: Counter[str] = Counter()

    def pick_for_stratum(stratum: str) -> int:
        if stratum == "asr":
            return rng.choice(strata["asr"])
        pools = source_strata[stratum]
        v5_pool = pools["authored_synthetic_audio_candidate"]
        legacy_pool = pools["synthetic_it_t6"]
        if stratum in {"SPEAK", "REPLAN"} and v5_pool and legacy_pool:
            # Prefer three v5 samples for each legacy support sample per action.
            source = "authored_synthetic_audio_candidate" if action_occurrences[stratum] % 4 < 3 else "synthetic_it_t6"
            pool = pools[source]
        else:
            pool = v5_pool or legacy_pool
        action_occurrences[stratum] += 1
        return rng.choice(pool)

    for start in range(0, max_steps, len(pattern)):
        chunk = [pick_for_stratum(pattern[(start + offset) % len(pattern)]) for offset in range(min(len(pattern), max_steps - start))]
        rng.shuffle(chunk)
        schedule.extend(chunk)
    selected = [records[index] for index in schedule]
    action_counts = Counter(
        str(row["action"]) for row in selected if row.get("task_kind") == "conversation_response"
    )
    task_counts = Counter(str(row["task_kind"]) for row in selected)
    action_source_counts: dict[str, dict[str, int]] = {}
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        sources = Counter(
            str(row.get("source") or "unknown")
            for row in selected
            if row.get("task_kind") == "conversation_response" and row.get("action") == action
        )
        action_source_counts[action] = dict(sorted(sources.items()))
    synthetic_examples = sum(bool(row.get("synthetic")) for row in selected)
    if target_token_counts is not None and len(target_token_counts) != len(records):
        raise ValueError(f"target_token_counts deve allinearsi ai record {mix_contract}")
    selected_tokens = [int(target_token_counts[index]) for index in schedule] if target_token_counts is not None else None
    synthetic_tokens = (
        sum(int(target_token_counts[index]) for index in schedule if bool(records[index].get("synthetic")))
        if target_token_counts is not None else None
    )
    report = {
        "examples": len(selected),
        "task_mix_contract": mix_contract,
        "sampling": "20% ASR, 40% SPEAK, 20% BACKCHANNEL, 20% REPLAN per 5 updates",
        "examples_by_task": {name: task_counts[name] for name in ("asr", "conversation_response")},
        "audio_response_by_action": {name: action_counts[name] for name in ("SPEAK", "BACKCHANNEL", "REPLAN")},
        "audio_response_by_action_source": action_source_counts,
        "examples_by_source": dict(Counter(str(row.get("source", "unknown")) for row in selected)),
        "synthetic_example_fraction": synthetic_examples / len(selected),
        "synthetic_target_token_fraction": (
            synthetic_tokens / sum(selected_tokens) if selected_tokens and sum(selected_tokens) else None
        ),
        "steps_include_all_strata_when_at_least_four": max_steps >= 4,
    }
    return selected, report


def _make_ordered_audio_asr_schedule(
    records: list[Mapping[str, Any]], *, schedule_rows: list[Mapping[str, Any]],
    max_steps: int, response_protocol: str = "v2",
    target_token_counts: list[int] | None = None,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Resolve a complete ordered schedule against unique train-record IDs.

    Schedule rows contain identifiers and provenance only. The selected model
    inputs always come from the validated training JSONL, preventing the
    schedule file from supplying or rewriting targets/audio.
    """
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    if response_protocol not in {"v1", "v2"}:
        raise ValueError(f"response_protocol non supportato: {response_protocol!r}")
    mix_contract = f"audio_asr_mix_{response_protocol}"
    if len(schedule_rows) < max_steps:
        raise ValueError(
            f"ordered schedule contiene {len(schedule_rows)} righe, meno dei {max_steps} update richiesti"
        )
    if target_token_counts is not None and len(target_token_counts) != len(records):
        raise ValueError(f"target_token_counts deve allinearsi ai record {mix_contract}")

    records_by_id: dict[str, int] = {}
    for index, record in enumerate(records):
        example_id = str(record.get("training_example_id") or "").strip()
        if not example_id:
            raise ValueError(f"training record {index}: training_example_id mancante per ordered schedule")
        if example_id in records_by_id:
            raise ValueError(f"duplicate training_example_id nel train dataset: {example_id}")
        if record.get("task_mix_contract") != mix_contract:
            raise ValueError(f"training record {example_id}: marker {mix_contract} mancante")
        records_by_id[example_id] = index

    required_metadata = (
        "task_mix_contract", "split", "task_kind", "action", "source",
        "source_group_id", "conversation_id", "target_variant_id",
    )
    seen_schedule_ids: set[str] = set()
    resolved_indices: list[int] = []
    for schedule_index, entry in enumerate(schedule_rows):
        if not isinstance(entry, Mapping):
            raise ValueError(f"ordered schedule row {schedule_index}: record non è un oggetto")
        example_id = str(entry.get("training_example_id") or "").strip()
        if not example_id:
            raise ValueError(f"ordered schedule row {schedule_index}: training_example_id mancante")
        if example_id in seen_schedule_ids:
            raise ValueError(f"ordered schedule duplicate training_example_id: {example_id}")
        seen_schedule_ids.add(example_id)
        record_index = records_by_id.get(example_id)
        if record_index is None:
            raise ValueError(f"ordered schedule unknown training_example_id: {example_id}")
        record = records[record_index]
        if record.get("split") != "train":
            raise ValueError(f"ordered schedule ID {example_id} is not a train record")
        for key in required_metadata:
            if entry.get(key) != record.get(key):
                raise ValueError(
                    f"ordered schedule metadata mismatch for {example_id}: {key}"
                )
        task_kind = str(record.get("task_kind") or "")
        if task_kind == "asr":
            if record.get("source") != "synthetic_v3_clean_asr":
                raise ValueError(f"ordered schedule ASR source non supportata per {example_id}")
        elif task_kind == "conversation_response":
            if record.get("action") not in {"SPEAK", "BACKCHANNEL", "REPLAN"}:
                raise ValueError(f"ordered schedule action non supportata per {example_id}")
            if record.get("source") not in {"authored_synthetic_audio_candidate", "synthetic_it_t6"}:
                raise ValueError(f"ordered schedule response source non supportata per {example_id}")
        else:
            raise ValueError(f"ordered schedule task non supportato per {example_id}: {task_kind}")
        if schedule_index < max_steps:
            resolved_indices.append(record_index)

    selected = [records[index] for index in resolved_indices]
    task_counts = Counter(str(row["task_kind"]) for row in selected)
    action_counts = Counter(
        str(row["action"]) for row in selected if row.get("task_kind") == "conversation_response"
    )
    action_source_counts: dict[str, dict[str, int]] = {}
    for action in ("SPEAK", "BACKCHANNEL", "REPLAN"):
        action_source_counts[action] = dict(sorted(Counter(
            str(row.get("source") or "unknown")
            for row in selected
            if row.get("task_kind") == "conversation_response" and row.get("action") == action
        ).items()))
    synthetic_examples = sum(bool(row.get("synthetic")) for row in selected)
    selected_tokens = [target_token_counts[index] for index in resolved_indices] if target_token_counts is not None else None
    synthetic_tokens = (
        sum(target_token_counts[index] for index in resolved_indices if bool(records[index].get("synthetic")))
        if target_token_counts is not None else None
    )
    report = {
        "examples": len(selected),
        "task_mix_contract": mix_contract,
        "schedule_mode": "ordered_jsonl",
        "ordered_schedule_entries_validated": len(schedule_rows),
        "ordered_schedule_entries_used": len(selected),
        "ordered_schedule_ids_unique": True,
        "sampling": "explicit ordered schedule JSONL; no internal fallback or resampling",
        "examples_by_task": {name: task_counts[name] for name in ("asr", "conversation_response")},
        "audio_response_by_action": {name: action_counts[name] for name in ("SPEAK", "BACKCHANNEL", "REPLAN")},
        "audio_response_by_action_source": action_source_counts,
        "examples_by_source": dict(sorted(Counter(str(row.get("source", "unknown")) for row in selected).items())),
        "synthetic_example_fraction": synthetic_examples / len(selected),
        "synthetic_target_token_fraction": (
            synthetic_tokens / sum(selected_tokens) if selected_tokens and sum(selected_tokens) else None
        ),
    }
    return selected, report


def _make_action_weighted_schedule(
    records: list[Mapping[str, Any]], *, max_steps: int, seed: int = 20261002,
) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    """Mix regular replies with event-grounded actions using available strata."""
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    by_stratum: dict[str, list[int]] = {
        stratum: [] for stratum in (
            "conversational_regular", "labelled_speak", "labelled_backchannel", "labelled_replan",
        )
    }
    for index, record in enumerate(records):
        stratum = str(record.get("sampling_stratum", ""))
        if stratum not in by_stratum:
            raise ValueError(f"sampling_stratum non supportato: {stratum}")
        by_stratum[stratum].append(index)
    action_strata = ("labelled_speak", "labelled_backchannel", "labelled_replan")
    missing_actions = [stratum for stratum in action_strata if not by_stratum[stratum]]
    if missing_actions:
        raise ValueError(f"sampler audio-response privo di action GT necessarie: {missing_actions}")
    rng = random.Random(int(seed))
    if by_stratum["conversational_regular"]:
        # Guarantee a four-update smoke sees all strata; then repeat exact 40/30/15/15 proportions.
        pattern = ["conversational_regular", "labelled_speak", "labelled_backchannel", "labelled_replan"] + [
            "conversational_regular"
        ] * 7 + ["labelled_speak"] * 5 + ["labelled_backchannel"] * 2 + ["labelled_replan"] * 2
        sampling = "40/30/15/15 conversational_regular/labelled_speak/labelled_backchannel/labelled_replan; unknown action labels receive response loss only"
    else:
        # The current v4 corpus has no unannotated regular replies. Use an
        # explicit known-action fallback: 50% SPEAK, 25% BACKCHANNEL, 25% REPLAN.
        pattern = ["labelled_speak", "labelled_backchannel", "labelled_replan", "labelled_speak"] + [
            "labelled_speak"
        ] * 8 + ["labelled_backchannel"] * 4 + ["labelled_replan"] * 4
        sampling = "50/25/25 labelled_speak/labelled_backchannel/labelled_replan; regular unknown-action targets unavailable"
    schedule = [rng.choice(by_stratum[pattern[position % len(pattern)]]) for position in range(max_steps)]
    chunks = [schedule[start:start + len(pattern)] for start in range(0, len(schedule), len(pattern))]
    for chunk in chunks:
        rng.shuffle(chunk)
    schedule = [index for chunk in chunks for index in chunk]
    selected = [records[index] for index in schedule]
    counts = Counter(str(record["action"]) for record in selected)
    stratum_counts = Counter(str(record["sampling_stratum"]) for record in selected)
    report = {
        "examples": len(selected),
        "examples_by_action_label": {
            action: counts[action] for action in ("BACKCHANNEL", "SPEAK", "REPLAN", "UNKNOWN")
        },
        "examples_by_sampling_stratum": {stratum: stratum_counts[stratum] for stratum in by_stratum},
        "sampling_fraction_by_stratum": {stratum: stratum_counts[stratum] / len(selected) for stratum in by_stratum},
        "source_groups_seen": len({str(record.get("source_group_id", "")) for record in selected}),
        "sampling": sampling,
    }
    return selected, report


def _response_language_instruction(language: Any) -> str:
    value = str(language or "").strip().lower().replace("_", "-")
    if value in {"it", "ita", "italian"} or value.startswith("it-") or "italian" in value:
        return "Rispondi in italiano in modo pertinente all'ultimo messaggio dell'utente."
    if value in {"cn", "zh", "chinese", "mandarin"} or value.startswith(("zh-", "cn-")) or "chinese" in value:
        return "请用中文针对用户的最后一条消息作出相关回复。"
    if value in {"en", "eng", "english"} or value.startswith("en-") or "english" in value:
        return "Reply in English to the user's latest message."
    return "Reply to the user's latest message in the same language."


def _language_name(language: Any) -> str:
    from duplex_kernel.asr_prompt import asr_language_name

    return asr_language_name(language)


def _format_chat(processor: Any, record: Mapping[str, Any], *, add_target: bool) -> str:
    tokenizer = getattr(processor, "tokenizer", processor)
    if record.get("task_kind") == "asr":
        from duplex_kernel.asr_prompt import asr_prompt_content

        audio_token = (
            getattr(processor, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": asr_prompt_content(record.get("language"), audio_token),
        }]
    elif record.get("audio_response_protocol") == _AUDIO_RESPONSE_PROTOCOL_V2:
        audio_token = (
            getattr(processor, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        messages = build_audio_response_protocol_v2_messages(
            history_messages=record.get("history_messages") or [],
            audio_token=audio_token,
        )
    elif record.get("audio_response_contract") == _AUDIO_RESPONSE_PROTOCOL_V1:
        audio_token = (
            getattr(processor, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        messages = build_audio_response_messages(
            language=record.get("language"),
            history_messages=record.get("history_messages") or [],
            audio_token=audio_token,
        )
    elif record.get("conditioning_mode") in {"runtime_asr_text", "action_conditioned_text"}:
        system_instruction = _response_language_instruction(record.get("language"))
        history_messages = record.get("history_messages") or []
        if record.get("conditioning_mode") == "action_conditioned_text":
            from duplex_kernel.action_conditioning import action_conditioned_messages

            messages = action_conditioned_messages(
                system_instruction,
                str(record.get("action", "")),
                history_messages,
            )
        else:
            messages = [{"role": "system", "content": system_instruction}]
            if not history_messages or history_messages[-1].get("role") != "user":
                raise ValueError("runtime_asr_text richiede una history con ultimo messaggio user trascritto")
            messages.extend(
                {"role": str(item["role"]), "content": str(item["content"])}
                for item in history_messages
            )
    else:
        prompt_text = _response_language_instruction(record.get("language"))
        if record.get("context_text"):
            prompt_text += "\nContesto testuale precedente:\n" + str(record["context_text"])
        content = prompt_text
        if record.get("audio_mask"):
            audio_token = (
                getattr(processor, "audio_token", None)
                or getattr(tokenizer, "audio_token", None)
                or "<|audio|>"
            )
            content = f"{content} {audio_token}"
        messages = [{"role": "user", "content": content}]
    if add_target:
        messages.append({"role": "assistant", "content": str(record["target_text"])})
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not add_target,
        )
    except Exception as exc:
        raise RuntimeError(f"Gemma chat template non supporta audio/conversazione: {exc}") from exc
    return text


def build_training_inputs(processor: Any, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Build targets using the configured runtime-matched text or audio path."""
    cache_key = _training_input_cache_key(processor, record)
    if cache_key is not None:
        cached = _TRAINING_INPUT_CACHE.get(cache_key)
        if cached is not None:
            _TRAINING_INPUT_CACHE.move_to_end(cache_key)
            _TRAINING_INPUT_CACHE_STATS["hits"] += 1
            return cached[0]
        _TRAINING_INPUT_CACHE_STATS["misses"] += 1
    prompt_text = _format_chat(processor, record, add_target=False)
    full_text = _format_chat(processor, record, add_target=True)
    audio_conditioned = record.get("audio_mask") is True
    if record.get("task_kind") == "asr" and not audio_conditioned:
        raise ValueError("record ASR deve passare audio reale al processor Gemma")
    if audio_conditioned:
        audio = _read_audio_for_record(record)
        processor_kwargs: dict[str, Any] = {
            "audio": [audio],
            "sampling_rate": 16000,
            "return_tensors": "pt",
            "padding": True,
        }
    else:
        if record.get("conditioning_mode") not in {"runtime_asr_text", "action_conditioned_text"} or record.get("task_kind") != "conversation_response":
            raise ValueError("solo conversation_response/runtime_asr_text action-conditioned supporta input testuale senza feature audio")
        processor_kwargs = {"return_tensors": "pt", "padding": True}
    try:
        # Encode both sides with the same processor/config so prompt length and
        # boundary come from actual model input IDs (audio-expanded for ASR).
        prompt_encoded = processor(text=[prompt_text], **processor_kwargs)
        encoded = processor(text=[full_text], **processor_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Gemma processor non ha potuto codificare l'esempio col percorso conditioning richiesto; "
            f"controllare chat template, audio marker/sample rate e API installata: {exc}"
        ) from exc
    prompt_ids = prompt_encoded.get("input_ids")
    input_ids = encoded.get("input_ids")
    if (
        not isinstance(prompt_ids, torch.Tensor)
        or prompt_ids.ndim != 2
        or prompt_ids.shape[0] != 1
        or not isinstance(input_ids, torch.Tensor)
        or input_ids.ndim != 2
        or input_ids.shape[0] != 1
    ):
        raise RuntimeError("processor Gemma non ha restituito input_ids [1,sequence] per prompt e full-chat")
    prompt_length = int(prompt_ids.shape[1])
    if input_ids.shape[1] <= prompt_length:
        raise ValueError(
            "target assistant vuoto dopo il processor multimodale: "
            f"prompt-only={prompt_length} token, full-chat={input_ids.shape[1]} token"
        )
    actual_prefix = input_ids[:, :prompt_length]
    if not torch.equal(actual_prefix.cpu(), prompt_ids.cpu()):
        compared = min(prompt_length, int(input_ids.shape[1]))
        differences = torch.nonzero(prompt_ids[0, :compared].cpu() != input_ids[0, :compared].cpu()).flatten()
        first_difference = int(differences[0]) if differences.numel() else compared
        window_start = max(0, first_difference - 3)
        window_end = min(prompt_length, first_difference + 4)
        prompt_window = prompt_ids[0, window_start:window_end].cpu().tolist()
        full_window = input_ids[0, window_start:window_end].cpu().tolist()
        raise ValueError(
            "prompt-only e full-chat divergono nel prefisso multimodale; "
            f"primo indice={first_difference}, finestra=[{window_start}:{window_end}], "
            f"prompt_ids={prompt_window}, full_ids={full_window}; "
            "il boundary assistant non è sicuro per questa chat template"
        )
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    encoded["labels"] = labels
    input_features = encoded.get("input_features")
    if audio_conditioned:
        if not isinstance(input_features, torch.Tensor) or input_features.numel() == 0:
            raise RuntimeError("processor Gemma ha ignorato l'audio; ASR audio-conditioned rifiutato")
        if not bool(torch.isfinite(input_features).all()):
            raise RuntimeError("feature audio Gemma non finite")
        prompt_features = prompt_encoded.get("input_features")
        if (
            not isinstance(prompt_features, torch.Tensor)
            or prompt_features.shape != input_features.shape
            or not torch.equal(prompt_features.cpu(), input_features.cpu())
        ):
            raise RuntimeError("prompt-only e full-chat non hanno generato le stesse feature dall'audio reale")
    elif input_features is not None or encoded.get("input_features_mask") is not None:
        raise RuntimeError("response runtime_asr_text ha ricevuto feature audio inattese dal processor")
    result = {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}
    if cache_key is not None:
        _cache_training_inputs(cache_key, result)
    return result


def _load_qlora(
    model_path: str, *, dtype: torch.dtype, resume_adapter_dir: str | Path | None = None,
    use_cache_for_training: bool = True,
):
    try:
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "QLoRA richiede transformers>=5, peft e bitsandbytes nel runtime GPU Linux"
        ) from exc
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
        # In Transformers 5.5 this option is also consumed by Bnb4BitQuantizer.
        # Keep Gemma's audio/vision computation in floating point; its audio
        # encoder calls torch.finfo on linear weight dtype during forward.
        llm_int8_skip_modules=list(_QUANTIZATION_SKIP_MODULES),
    )
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            device_map="auto",
            quantization_config=quantization,
        )
        processor = AutoProcessor.from_pretrained(model_path)
        preparation_report = _prepare_gemma_for_kbit_training(
            model, use_cache_for_training=use_cache_for_training,
        )
        target_modules = _language_model_lora_target_modules(model)
        if resume_adapter_dir is None:
            lora = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora)
        else:
            model = PeftModel.from_pretrained(
                model, str(resume_adapter_dir), is_trainable=True
            )
        matched_lora_modules = _verify_lora_scope(model, target_modules)
        tower_report = _freeze_float_towers(model, dtype=dtype)
        bridge_report = _freeze_float_bridges(model, dtype=dtype)
        input_embedding_report = _freeze_input_embeddings(model, dtype=dtype)
        quantization_report = _quantization_runtime_report(
            model, tower_report, bridge_report, input_embedding_report
        )
        quantization_report["training_preparation"] = preparation_report
    except Exception as exc:
        raise RuntimeError(
            "Gemma4 E2B QLoRA topology/API non compatibile con il runtime installato: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    trainable_lora = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" in name
    ]
    target_set = set(target_modules)
    trainable_target_paths: set[str] = set()
    unexpected_trainable: list[str] = []
    for name, _ in trainable_lora:
        module_path = name.split(".lora_", 1)[0]
        path_matches = [target for target in target_set if module_path == target or module_path.endswith("." + target)]
        if len(path_matches) != 1:
            unexpected_trainable.append(name)
        else:
            trainable_target_paths.add(path_matches[0])
    if not trainable_lora or unexpected_trainable or trainable_target_paths != target_set:
        raise RuntimeError("PEFT non ha reso trainabili LoRA sul backbone Gemma language_model")
    model.train()
    return model, processor, trainable_lora, matched_lora_modules, quantization_report


def _is_italian_record(record: Mapping[str, Any]) -> bool:
    language = str(record.get("language") or "").strip().lower()
    return language in {"it", "ita", "italian"} or language.startswith("it-") or "italian" in language


def _training_config(
    *, train_jsonl: str | Path, model_path: str, max_steps: int,
    learning_rate: float, seed: int, conditioning_mode: str,
    validation_jsonl: str | Path | None = None,
    initial_adapter_dir: str | Path | None = None,
    checkpoint_interval: int = 100,
    use_cache_for_training: bool = True,
) -> dict[str, Any]:
    if conditioning_mode not in {"baselineaudio_response", "runtime_asr_text", "audio_asr_mix_v1", "audio_asr_mix_v2"}:
        raise ValueError(f"conditioning_mode Stage1 non supportata: {conditioning_mode}")
    if checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval Stage1 deve essere positivo")
    config = {
        "train_jsonl": str(Path(train_jsonl).expanduser().resolve()),
        "model_path": model_path,
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "checkpoint_interval": int(checkpoint_interval),
        "conditioning_mode": conditioning_mode,
        "objective": f"{conditioning_mode}_assistant_text_causal_lm",
    }
    if validation_jsonl is not None:
        config["validation_jsonl"] = str(Path(validation_jsonl).expanduser().resolve())
    if initial_adapter_dir is not None:
        initial_adapter = Path(initial_adapter_dir).expanduser().resolve()
        config["initial_adapter_dir"] = str(initial_adapter)
        manifest_path = initial_adapter.parent / "training_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, Mapping) and isinstance(manifest.get("run_id"), str):
                config["initial_adapter_run_id"] = manifest["run_id"]
    if use_cache_for_training:
        config["use_cache_for_training"] = True
        config["kv_sharing_semantics"] = "transformers_gemma4_native_dynamic_cache"
    return config


def _resolve_adapter_dir_for_load(
    *, resume_from: str | Path | None, initial_adapter_dir: str | Path | None,
) -> Path | None:
    """Load a checkpoint adapter on resume, otherwise warm-start a fresh optimizer from the given adapter."""
    if resume_from is not None:
        return Path(resume_from).expanduser().resolve() / "adapter"
    if initial_adapter_dir is None:
        return None
    adapter = Path(initial_adapter_dir).expanduser().resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter iniziale non trovato: {adapter}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json mancante: {adapter}")
    if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise FileNotFoundError(f"pesi PEFT mancanti: {adapter}")
    return adapter


def _is_audio_response_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("audio_response_contract") == _AUDIO_RESPONSE_PROTOCOL_V1
        or record.get("audio_response_protocol") == _AUDIO_RESPONSE_PROTOCOL_V2
    )


def _audio_window_identity(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(record.get("audio_path", "")), int(record.get("audio_channel_index", -1)),
        float(record.get("audio_start_ms", -1)), float(record.get("audio_end_ms", -1)),
    )


def _evaluate_validation_loss_by_task(
    model: Any,
    processor: Any,
    records: list[Mapping[str, Any]],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    use_cache_for_training: bool = True,
) -> dict[str, Any]:
    """Report held-out response and ASR losses separately, token-weighted."""
    if not records:
        raise ValueError("validation task metrics richiede record held-out")
    was_training = bool(model.training)
    totals: dict[str, dict[str, float]] = {}
    model.eval()
    try:
        with torch.no_grad():
            for record in records:
                task = str(record.get("task_kind", "unknown"))
                inputs = build_training_inputs(processor, record)
                moved, status = _move_batch_to_device(inputs, device=device, compute_dtype=compute_dtype)
                output = model(**moved, use_cache=use_cache_for_training, return_dict=True)
                loss = getattr(output, "loss", None)
                if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise RuntimeError(f"validation loss non finita per task {task}")
                bucket = totals.setdefault(task, {
                    "examples": 0.0, "tokens": 0.0,
                    "token_weighted_loss_sum": 0.0, "mean_example_loss_sum": 0.0,
                })
                token_count = int(status["non_ignored_label_tokens"])
                loss_value = float(loss.detach().cpu())
                bucket["examples"] += 1
                bucket["tokens"] += token_count
                bucket["token_weighted_loss_sum"] += loss_value * token_count
                bucket["mean_example_loss_sum"] += loss_value
    finally:
        model.train(was_training)
    return {
        task: {
            "examples": int(values["examples"]),
            "target_tokens": int(values["tokens"]),
            "token_weighted_loss": values["token_weighted_loss_sum"] / max(1, values["tokens"]),
            "mean_example_loss": values["mean_example_loss_sum"] / max(1, values["examples"]),
        }
        for task, values in totals.items()
    }


def _evaluate_shuffled_audio_dependency(
    model: Any,
    processor: Any,
    records: list[Mapping[str, Any]],
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    use_cache_for_training: bool = True,
    max_pairs: int = 6,
    seed: int = 20261002,
) -> dict[str, Any]:
    """Measure held-out LM loss under exact-duration, wrong-user-audio swaps."""
    if not records or max_pairs <= 0:
        raise ValueError("valutazione audio shuffle richiede record held-out e max_pairs positivo")
    by_duration: dict[float, list[int]] = {}
    for index, record in enumerate(records):
        duration = record.get("input_audio_duration_ms")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration <= 0:
            continue
        by_duration.setdefault(round(float(duration), 6), []).append(index)
    candidate_pairs: list[tuple[int, int]] = []
    for indices in by_duration.values():
        for target_index in indices:
            target_identity = _audio_window_identity(records[target_index])
            donors = [index for index in indices if _audio_window_identity(records[index]) != target_identity]
            if donors:
                candidate_pairs.append((target_index, donors[0]))
    if not candidate_pairs:
        return {
            "status": "unavailable",
            "reason": "nessun audio held-out di durata identica ma contenuto differente per lo shuffle",
            "heldout_records": len(records),
            "shuffled_pair_count": 0,
        }
    rng = random.Random(int(seed))
    rng.shuffle(candidate_pairs)
    chosen = candidate_pairs[:max_pairs]
    was_training = bool(getattr(model, "training", True))
    model.eval()
    pair_rows: list[dict[str, Any]] = []
    try:
        with torch.no_grad():
            for target_index, donor_index in chosen:
                target = dict(records[target_index])
                donor = records[donor_index]
                shuffled = dict(target)
                for key in (
                    "audio_path", "source_audio_sha256", "sample_rate_hz", "audio_channel_index",
                    "audio_start_ms", "audio_end_ms", "input_audio_duration_ms",
                ):
                    shuffled[key] = donor[key]
                correct_inputs = build_training_inputs(processor, target)
                shuffled_inputs = build_training_inputs(processor, shuffled)
                for key in ("input_ids", "labels", "attention_mask"):
                    if key in correct_inputs and key in shuffled_inputs and not torch.equal(correct_inputs[key], shuffled_inputs[key]):
                        raise RuntimeError(f"audio shuffle ha alterato input testuale/{key}; il confronto non è isolato")
                for key in ("input_features", "input_features_mask"):
                    if key not in correct_inputs or key not in shuffled_inputs or correct_inputs[key].shape != shuffled_inputs[key].shape:
                        raise RuntimeError("audio shuffle a durata identica ha prodotto feature incompatibili")
                correct, correct_status = _move_batch_to_device(correct_inputs, device=device, compute_dtype=compute_dtype)
                wrong, wrong_status = _move_batch_to_device(shuffled_inputs, device=device, compute_dtype=compute_dtype)
                if correct_status["non_ignored_label_tokens"] != wrong_status["non_ignored_label_tokens"]:
                    raise RuntimeError("audio shuffle ha cambiato il numero di token supervisionati")
                correct_output = model(**correct, use_cache=use_cache_for_training, return_dict=True)
                wrong_output = model(**wrong, use_cache=use_cache_for_training, return_dict=True)
                correct_loss = getattr(correct_output, "loss", None)
                wrong_loss = getattr(wrong_output, "loss", None)
                if (
                    not isinstance(correct_loss, torch.Tensor) or correct_loss.ndim != 0
                    or not bool(torch.isfinite(correct_loss))
                    or not isinstance(wrong_loss, torch.Tensor) or wrong_loss.ndim != 0
                    or not bool(torch.isfinite(wrong_loss))
                ):
                    raise RuntimeError("heldout audio shuffle ha prodotto loss non valida")
                pair_rows.append({
                    "action": str(target.get("action", "unknown")),
                    "sampling_stratum": str(target.get("sampling_stratum", "unknown")),
                    "correct_audio_loss": float(correct_loss.float().cpu()),
                    "shuffled_audio_loss": float(wrong_loss.float().cpu()),
                    "shuffled_minus_correct_loss": float((wrong_loss - correct_loss).float().cpu()),
                    "target_conversation_id": str(target.get("conversation_id", "")),
                    "donor_conversation_id": str(donor.get("conversation_id", "")),
                    "equal_duration_ms": float(target["input_audio_duration_ms"]),
                })
                del correct_output, wrong_output, correct_loss, wrong_loss
    finally:
        model.train(was_training)
    deltas = [row["shuffled_minus_correct_loss"] for row in pair_rows]
    action_deltas = {
        action: [row["shuffled_minus_correct_loss"] for row in pair_rows if row["action"] == action]
        for action in ("BACKCHANNEL", "SPEAK", "REPLAN", "UNKNOWN")
    }
    stratum_deltas = {
        stratum: [row["shuffled_minus_correct_loss"] for row in pair_rows if row["sampling_stratum"] == stratum]
        for stratum in (
            "conversational_regular", "labelled_speak", "labelled_backchannel", "labelled_replan",
        )
    }
    mean_delta = sum(deltas) / len(deltas)
    return {
        "status": "measured",
        "heldout_records": len(records),
        "candidate_pair_count": len(candidate_pairs),
        "shuffled_pair_count": len(pair_rows),
        "mean_correct_audio_loss": sum(row["correct_audio_loss"] for row in pair_rows) / len(pair_rows),
        "mean_shuffled_audio_loss": sum(row["shuffled_audio_loss"] for row in pair_rows) / len(pair_rows),
        "mean_shuffled_minus_correct_loss": mean_delta,
        "audio_dependency_pass": mean_delta > 0.0,
        "pairs_by_target_action": {
            action: {"count": len(values), "mean_delta": sum(values) / len(values) if values else None}
            for action, values in action_deltas.items()
        },
        "pairs_by_sampling_stratum": {
            stratum: {"count": len(values), "mean_delta": sum(values) / len(values) if values else None}
            for stratum, values in stratum_deltas.items()
        },
        "pair_details": pair_rows,
    }


def _preflight_checkpoint_run(
    *, adapter_path: Path, resume_from: str | Path | None,
    expected_config: Mapping[str, Any], max_steps: int,
) -> tuple[Path, dict[str, Any] | None]:
    run_root = adapter_path.parent.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if adapter_path.exists():
        raise FileExistsError(f"adapter finale già esistente, non sovrascritto: {adapter_path}")
    if resume_from is None:
        old_steps = sorted(path.name for path in run_root.glob("step_[0-9][0-9][0-9][0-9][0-9][0-9]"))
        if old_steps:
            raise FileExistsError(
                f"checkpoint Stage1 già presenti in {run_root}; specificare --resume-from {run_root / old_steps[-1]}"
            )
        return run_root, None
    checkpoint_dir = Path(resume_from).expanduser().resolve()
    if checkpoint_dir.parent != run_root:
        raise ValueError(
            "--resume-from deve appartenere allo stesso checkpoint root dell'adapter finale: "
            f"resume={checkpoint_dir}, adapter_root={run_root}"
        )
    metadata = read_training_checkpoint_metadata(
        checkpoint_dir=checkpoint_dir, expected_config=expected_config
    )
    completed = int(metadata["completed_steps"])
    if completed >= max_steps:
        raise ValueError(f"checkpoint già a {completed} step, max_steps richiesto={max_steps}")
    if checkpoint_dir.name != f"step_{completed:06d}":
        raise ValueError("directory --resume-from non corrisponde al completed_steps del checkpoint")
    run_metadata = metadata.get("run_metadata")
    if not isinstance(run_metadata, dict) or not isinstance(run_metadata.get("conditioning_probe"), dict):
        raise ValueError("checkpoint privo della probe numerica di conditioning originaria")
    next_checkpoint = ((completed // 100) + 1) * 100
    future_steps = list(range(next_checkpoint, max_steps + 1, 100))
    if max_steps % 100 and (not future_steps or future_steps[-1] != max_steps):
        future_steps.append(max_steps)
    collisions = [run_root / f"step_{step:06d}" for step in future_steps if (run_root / f"step_{step:06d}").exists()]
    if collisions:
        raise FileExistsError(f"checkpoint di destinazione già esistenti, non sovrascritti: {collisions}")
    return run_root, metadata


def _save_training_step_checkpoint(
    *, model: Any, optimizer: torch.optim.Optimizer, run_root: Path,
    completed_steps: int, expected_config: Mapping[str, Any],
    losses: list[float], step_runtime: list[dict[str, Any]],
    run_metadata: Mapping[str, Any],
) -> Path:
    def save_adapter(path: Path) -> None:
        model.save_pretrained(path, save_embedding_layers=False)

    return save_training_checkpoint(
        checkpoint_root=run_root,
        completed_steps=completed_steps,
        optimizer=optimizer,
        training_config=expected_config,
        losses=losses,
        step_runtime=step_runtime,
        run_metadata=run_metadata,
        save_adapter=save_adapter,
    )


def train_qlora(
    *,
    train_jsonl: str | Path,
    model_path: str,
    adapter_dir: str | Path,
    max_steps: int,
    learning_rate: float = 2e-4,
    audio_code_loss_weight: float = 0.0,
    preparation_manifest: str | Path | None = None,
    validation_jsonl: str | Path | None = None,
    seed: int = 20260930,
    resume_from: str | Path | None = None,
    initial_adapter_dir: str | Path | None = None,
    schedule_jsonl: str | Path | None = None,
    checkpoint_interval: int = 100,
    use_cache_for_training: bool = True,
    conditioning_mode: str,
) -> dict[str, Any]:
    """Fine-tune Gemma LoRA with an explicitly selected serving-aligned objective."""
    if not use_cache_for_training:
        raise ValueError(
            "Stage1 Gemma4 non supporta training senza DynamicCache: use_cache=False disattiva la condivisione K/V nativa"
        )
    if audio_code_loss_weight != 0:
        raise NotImplementedError(
            "Gemma4ForConditionalGeneration in Transformers 5.5 exposes audio input features "
            "and causal text logits, but no audio codec/subtalker output head; audio-code loss is unsupported"
        )
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    adapter_path = Path(adapter_dir).expanduser().resolve()
    expected_config = _training_config(
        train_jsonl=train_jsonl, model_path=model_path, max_steps=max_steps,
        learning_rate=learning_rate, seed=seed, conditioning_mode=conditioning_mode,
        validation_jsonl=validation_jsonl, initial_adapter_dir=initial_adapter_dir,
        checkpoint_interval=checkpoint_interval,
        use_cache_for_training=use_cache_for_training,
    )
    ordered_schedule_rows: list[dict[str, Any]] | None = None
    if schedule_jsonl is not None:
        if conditioning_mode != "audio_asr_mix_v2":
            raise ValueError("--schedule-jsonl è supportato soltanto con audio_asr_mix_v2")
        schedule_path = Path(schedule_jsonl).expanduser().resolve()
        if not schedule_path.is_file():
            raise FileNotFoundError(f"ordered schedule JSONL non trovato: {schedule_path}")
        expected_config["ordered_schedule_jsonl"] = str(schedule_path)
        expected_config["ordered_schedule_sha256"] = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
        ordered_schedule_rows = read_jsonl(schedule_path)
    source_rows = read_jsonl(train_jsonl)
    direct_rows = [row for row in source_rows if _is_audio_response_record(row)]
    mixed_task_records = None
    if conditioning_mode in {"audio_asr_mix_v1", "audio_asr_mix_v2"}:
        response_protocol = "v2" if conditioning_mode == "audio_asr_mix_v2" else "v1"
        mix_contract = f"audio_asr_mix_{response_protocol}"
        if validation_jsonl is None:
            raise ValueError(f"{mix_contract} richiede un validation JSONL held-out")
        records, task_records = _validate_audio_asr_mix_records(
            source_rows, split="train", response_protocol=response_protocol,
        )
        validation_rows = read_jsonl(validation_jsonl)
        validation_records, _ = _validate_audio_asr_mix_records(
            validation_rows, split="validation", response_protocol=response_protocol,
        )
        train_groups = {
            (str(row.get("source", "")), str(row.get("source_group_id") or row.get("conversation_id") or ""))
            for row in records
        }
        validation_groups = {
            (str(row.get("source", "")), str(row.get("source_group_id") or row.get("conversation_id") or ""))
            for row in validation_records
        }
        if train_groups & validation_groups:
            raise ValueError("source group audio/ASR sovrapposti fra train e validation")
        def mix_source_is_valid(row: Mapping[str, Any]) -> bool:
            if row.get("task_kind") == "asr":
                return row.get("source") == "synthetic_v3_clean_asr"
            return row.get("task_kind") == "conversation_response" and row.get("synthetic") is True

        if not all(mix_source_is_valid(row) for row in records):
            raise ValueError(f"{mix_contract} train contiene source non eleggibili")
        if not all(mix_source_is_valid(row) for row in validation_records):
            raise ValueError(f"{mix_contract} validation contiene source non eleggibili")
        direct_rows = task_records["conversation_response"]
        mixed_task_records = task_records
        expected_config["task_mix_contract"] = mix_contract
        if response_protocol == "v2":
            expected_config["audio_response_protocol"] = _AUDIO_RESPONSE_PROTOCOL_V2
        conversations = source_rows
    elif direct_rows:
        if len(direct_rows) != len(source_rows) or conditioning_mode != "baselineaudio_response":
            raise ValueError("il contratto audio_response_conditioning_v1 richiede un train split omogeneo baselineaudio_response")
        if validation_jsonl is None:
            raise ValueError("il training audio_response_conditioning_v1 richiede un validation JSONL held-out esplicito")
        records = _validate_direct_audio_records(source_rows, conditioning_mode=conditioning_mode)
        train_action_counts = Counter(str(row["action"]) for row in records)
        missing_actions = sorted({"BACKCHANNEL", "SPEAK", "REPLAN"} - set(train_action_counts))
        missing_strata = sorted(
            {"labelled_speak", "labelled_backchannel", "labelled_replan"}
            - {str(row.get("sampling_stratum")) for row in records}
        )
        if missing_actions or missing_strata:
            raise ValueError(
                "training audio_response_conditioning_v1 privo di coverage supervisionabile: "
                f"action={missing_actions}, strata={missing_strata}"
            )
        validation_rows = read_jsonl(validation_jsonl)
        if not validation_rows or any(not _is_audio_response_record(row) for row in validation_rows):
            raise ValueError("validation split deve contenere soltanto record audio_response_conditioning_v1")
        validation_records = _validate_direct_audio_records(validation_rows, conditioning_mode=conditioning_mode)
        train_groups = {str(row.get("source_group_id", "")) for row in records}
        validation_groups = {str(row.get("source_group_id", "")) for row in validation_records}
        if "" in train_groups or "" in validation_groups or train_groups & validation_groups:
            raise ValueError("source_group_id deve essere presente e disgiunto fra train e validation")
        if not all(row.get("synthetic") is True for row in records):
            raise ValueError("dataset audio_response_conditioning_v1 richiede target sintetici verificati")
        if not all(row.get("synthetic") is True for row in validation_records):
            raise ValueError("validation audio_response_conditioning_v1 deve mantenere la provenance sintetica")
        expected_config["audio_response_contract"] = "audio_response_conditioning_v1"
        conversations = source_rows
    else:
        if validation_jsonl is not None:
            raise ValueError("--validation-jsonl è supportato per il dataset audio_response_conditioning_v1")
        validation_records = []
        conversations = source_rows
        records = _training_records(conversations, conditioning_mode=conditioning_mode)
    if not records:
        raise ValueError("nessun target audio/testo con label esplicita nel train split")
    if not any(record["synthetic"] for record in records):
        raise ValueError("train split privo di target supervisionati italiani sintetici")
    if ordered_schedule_rows is not None:
        _make_ordered_audio_asr_schedule(
            records,
            schedule_rows=ordered_schedule_rows,
            max_steps=int(max_steps),
            response_protocol="v2",
        )
    model_adapter_dir = _resolve_adapter_dir_for_load(
        resume_from=resume_from, initial_adapter_dir=initial_adapter_dir,
    )
    run_root, resume_metadata = _preflight_checkpoint_run(
        adapter_path=adapter_path,
        resume_from=resume_from,
        expected_config=expected_config,
        max_steps=max_steps,
    )
    if not torch.cuda.is_available():
        raise RuntimeError("training QLoRA richiede CUDA; questa CLI non esegue fallback CPU")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    compute_dtype = torch.bfloat16
    model, processor, trainable, matched_lora_modules, quantization_report = _load_qlora(
        model_path, dtype=compute_dtype, resume_adapter_dir=model_adapter_dir,
        use_cache_for_training=use_cache_for_training,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    target_token_counts: list[int] = []
    for record in records:
        tokenized = tokenizer(str(record["target_text"]), add_special_tokens=False)
        token_ids = tokenized["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            count = int(token_ids.shape[-1])
        elif token_ids and isinstance(token_ids[0], (list, tuple)):
            count = len(token_ids[0])
        else:
            count = len(token_ids)
        target_token_counts.append(count)
    if conditioning_mode in {"audio_asr_mix_v1", "audio_asr_mix_v2"}:
        response_protocol = "v2" if conditioning_mode == "audio_asr_mix_v2" else "v1"
        if ordered_schedule_rows is not None:
            schedule, source_composition = _make_ordered_audio_asr_schedule(
                records,
                schedule_rows=ordered_schedule_rows,
                max_steps=int(max_steps),
                response_protocol=response_protocol,
                target_token_counts=target_token_counts,
            )
        else:
            schedule, source_composition = _make_audio_asr_mix_schedule(
                records, max_steps=int(max_steps), seed=seed,
                target_token_counts=target_token_counts,
                response_protocol=response_protocol,
            )
    elif direct_rows:
        schedule, source_composition = _make_action_weighted_schedule(
            records, max_steps=int(max_steps), seed=seed,
        )
    else:
        schedule, source_composition = _make_training_schedule(
            records,
            max_steps=int(max_steps),
            target_token_counts=target_token_counts,
            synthetic_fraction=0.8,
            seed=seed,
        )
    start_step = int(resume_metadata["completed_steps"]) if resume_metadata else 0
    if resume_metadata:
        run_metadata = dict(resume_metadata.get("run_metadata", {}))
        run_id = str(run_metadata["run_id"])
        losses = [float(value) for value in resume_metadata["losses"]]
        step_runtime = list(resume_metadata["step_runtime"])
    else:
        selected_probe = next(
            (
                record for record in records
                if record.get("synthetic") is True
                and record.get("task_kind") == "conversation_response"
                and _is_italian_record(record)
                and (
                    conditioning_mode == "runtime_asr_text"
                    or _is_audio_response_record(record)
                    or float(record.get("audio_probe_end_ms") or 0)
                    > float(record.get("audio_probe_start_ms") or 0)
                )
            ),
            None,
        )
        if selected_probe is None:
            raise ValueError("probe numerica Stage1 richiede un record italiano gold conversation_response")
        probe_record = dict(selected_probe)
        lora_b = [
            (name, parameter) for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_B" in name
        ]
        if not lora_b:
            raise RuntimeError("probe di conditioning non ha trovato i parametri LoRA B trainabili")
        lora_b_zero_initialized = not any(bool(parameter.detach().ne(0).any()) for _, parameter in lora_b)
        if model_adapter_dir is None and not lora_b_zero_initialized:
            raise RuntimeError("nuova LoRA senza adapter sorgente deve inizializzare LoRA B a zero")
        if conditioning_mode == "runtime_asr_text":
            probe_inputs = build_training_inputs(processor, probe_record)
            conditioning_probe = run_text_conditioning_probe(
                model,
                processor,
                probe_record,
                probe_inputs,
                device=next(model.parameters()).device,
                compute_dtype=compute_dtype,
                use_cache_for_training=use_cache_for_training,
            )
        else:
            probe_inputs = build_training_inputs(processor, probe_record)
            conditioning_probe = run_audio_text_probe(
                model,
                processor,
                probe_record,
                probe_inputs,
                device=next(model.parameters()).device,
                compute_dtype=compute_dtype,
                use_cache_for_training=use_cache_for_training,
            )
        conditioning_probe["lora_b_zero_initialized"] = lora_b_zero_initialized
        conditioning_probe["adapter_initialization"] = (
            "resume_checkpoint" if resume_from is not None
            else "warm_start_adapter" if initial_adapter_dir is not None
            else "fresh_lora"
        )
        conditioning_probe["training_conditioning_mode"] = conditioning_mode
        print("Stage1 conditioning probe: " + json.dumps(conditioning_probe, ensure_ascii=False))
        run_id = uuid.uuid4().hex
        run_metadata = {
            "run_id": run_id,
            "conditioning_mode": conditioning_mode,
            "adapter_initialization": {
                "mode": (
                    "warm_start_adapter" if initial_adapter_dir is not None
                    else "fresh_lora"
                ),
                "initial_adapter_dir": str(Path(initial_adapter_dir).expanduser().resolve())
                if initial_adapter_dir is not None else None,
                "initial_adapter_run_id": expected_config.get("initial_adapter_run_id"),
                "optimizer_state": "fresh_adamw",
            },
            "conditioning_probe": conditioning_probe,
        }
        losses = []
        step_runtime = []
    optimizer_parameters = [parameter for _, parameter in trainable]
    trainable_model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if {id(parameter) for parameter in optimizer_parameters} != {id(parameter) for parameter in trainable_model_parameters}:
        raise RuntimeError("optimizer non contiene esattamente tutti e soli i parametri LoRA trainabili")
    adapter_parameter_snapshot = [parameter.detach().float().clone() for parameter in optimizer_parameters]
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=float(learning_rate))
    if resume_metadata:
        resume_metadata = restore_training_checkpoint(
            checkpoint_dir=resume_from,
            optimizer=optimizer,
            expected_config=expected_config,
        )
    device = next(model.parameters()).device
    model.train()
    for schedule_index in range(start_step, max_steps):
        record = schedule[schedule_index]
        inputs = build_training_inputs(processor, record)
        inputs, batch_status = _move_batch_to_device(
            inputs, device=device, compute_dtype=compute_dtype
        )
        optimizer.zero_grad(set_to_none=True)
        cuda_step = device.type == "cuda" and torch.cuda.is_available()
        if cuda_step:
            torch.cuda.reset_peak_memory_stats(device)
        step_started = time.perf_counter()
        output = model(**inputs, use_cache=use_cache_for_training, return_dict=True)
        loss = getattr(output, "loss", None)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("Gemma audio/text forward non ha restituito una loss scalare finita")
        loss.backward()
        gradients = [parameter.grad for parameter in optimizer_parameters if parameter.grad is not None]
        if not gradients:
            raise RuntimeError("nessun gradiente raggiunge i parametri LoRA Gemma")
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("gradiente LoRA Gemma non finito")
        if not any(bool(gradient.ne(0).any()) for gradient in gradients):
            raise RuntimeError("tutti i gradienti LoRA Gemma sono zero")
        optimizer.step()
        global_step = schedule_index + 1
        loss_value = float(loss.detach().cpu())
        step_elapsed_seconds = time.perf_counter() - step_started
        cuda_peak_memory_allocated = (
            int(torch.cuda.max_memory_allocated(device)) if cuda_step else None
        )
        cuda_peak_memory_reserved = (
            int(torch.cuda.max_memory_reserved(device)) if cuda_step else None
        )
        losses.append(loss_value)
        step_runtime.append({
            "step": global_step,
            "loss": loss_value,
            "task_kind": str(record.get("task_kind", "unknown")),
            "source": str(record.get("source", "unknown")),
            "action": (
                str(record.get("action"))
                if record.get("task_kind") == "conversation_response" and record.get("action") is not None
                else None
            ),
            "sampling_stratum": record.get("sampling_stratum"),
            "step_elapsed_seconds": step_elapsed_seconds,
            "cuda_peak_memory_allocated_bytes": cuda_peak_memory_allocated,
            "cuda_peak_memory_reserved_bytes": cuda_peak_memory_reserved,
            "input_features_dtype": batch_status["input_features_dtype"],
            "input_features_shape": batch_status["input_features_shape"],
            "non_ignored_label_tokens": batch_status["non_ignored_label_tokens"],
            "lora_gradient_count": len(gradients),
            "lora_gradients_finite": True,
            "lora_has_nonzero_gradient": True,
            "use_cache_for_training": use_cache_for_training,
        })
        del output, loss
        if global_step % int(expected_config["checkpoint_interval"]) == 0 or global_step == max_steps:
            checkpoint_dir = _save_training_step_checkpoint(
                model=model,
                optimizer=optimizer,
                run_root=run_root,
                completed_steps=global_step,
                expected_config=expected_config,
                losses=losses,
                step_runtime=step_runtime,
                run_metadata=run_metadata,
            )
            print(f"Stage1 checkpoint step={global_step} path={checkpoint_dir}")
    max_delta = max(
        float((parameter.detach().float() - before).abs().max().cpu())
        for parameter, before in zip(optimizer_parameters, adapter_parameter_snapshot)
    )
    if not max_delta > 0:
        raise RuntimeError("nessun parametro LoRA Gemma è cambiato durante gli optimizer step del run/resume")
    validation_loss_by_task = None
    if conditioning_mode in {"audio_asr_mix_v1", "audio_asr_mix_v2"}:
        validation_loss_by_task = _evaluate_validation_loss_by_task(
            model,
            processor,
            validation_records,
            device=device,
            compute_dtype=compute_dtype,
            use_cache_for_training=use_cache_for_training,
        )
    heldout_audio_dependency = None
    if direct_rows:
        heldout_audio_dependency = _evaluate_shuffled_audio_dependency(
            model,
            processor,
            [
                row for row in validation_records
                if _is_audio_response_record(row)
            ],
            device=device,
            compute_dtype=compute_dtype,
            use_cache_for_training=use_cache_for_training,
        )
    _save_final_adapter(model, processor, adapter_path)
    prep_counts = {}
    if preparation_manifest and Path(preparation_manifest).is_file():
        prep_counts = json.loads(Path(preparation_manifest).read_text(encoding="utf-8"))
    manifest_path = _training_manifest_path(adapter_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    try:
        import peft
        peft_version = peft.__version__
    except Exception:
        peft_version = "unknown"
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "base_model_id": model_path,
        "base_model_path": str(Path(model_path).resolve()) if Path(model_path).exists() else model_path,
        "checkpoint_root": str(manifest_path.parent.resolve()),
        "adapter_dir": str(adapter_path.resolve()),
        "resumed_from": str(Path(resume_from).resolve()) if resume_from else None,
        "initial_adapter_dir": expected_config.get("initial_adapter_dir"),
        "initial_adapter_run_id": expected_config.get("initial_adapter_run_id"),
        "control_head_path": str((manifest_path.parent / "control_head.pt").resolve()),
        "control_head": {
            "input_dim": 1536,
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(ACTIONS),
        },
        "versions": {"torch": torch.__version__, "transformers": transformers_version, "peft": peft_version},
        "training": {
            "stage1_examples": len(records),
            "stage1_usable_records_by_source": dict(Counter(record["source"] for record in records)),
            "stage1_steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "lora_backbone_parameter_count": sum(parameter.numel() for _, parameter in trainable),
            "lora_target_modules": matched_lora_modules,
            "lora_max_abs_parameter_delta": max_delta,
            "lora_max_abs_parameter_delta_reference": (
                "resume_checkpoint" if resume_metadata
                else "initial_adapter" if initial_adapter_dir is not None
                else "run_start"
            ),
            "conditioning_probe": run_metadata["conditioning_probe"],
            "quantization_runtime": quantization_report,
            "batch_runtime": {
                "batch_count": len(step_runtime),
                **_summarize_batch_runtime(step_runtime),
                "input_features_dtype": sorted({
                    item["input_features_dtype"] for item in step_runtime
                    if item["input_features_dtype"] is not None
                }),
            "input_features_shapes": sorted(
                    {tuple(item["input_features_shape"]) for item in step_runtime
                     if item["input_features_shape"] is not None}
                ),
                "non_ignored_label_tokens_total": sum(
                    item["non_ignored_label_tokens"] for item in step_runtime
                ),
                "non_ignored_label_tokens_per_batch": [
                    item["non_ignored_label_tokens"] for item in step_runtime
                ],
                "steps": step_runtime,
            },
            "conditioning_mode": conditioning_mode,
            "objective": expected_config["objective"],
            "audio_code_objective": "unsupported_by_gemma4_transformers_api",
            "source_composition": source_composition,
            "synthetic_primary_sample_fraction": source_composition.get(
                "synthetic_example_fraction",
                sum(bool(row.get("synthetic")) for row in schedule) / len(schedule),
            ),
            "synthetic_primary_target_token_fraction": source_composition.get("synthetic_target_token_fraction", 1.0),
            "text_target_counts": (
                {
                    "records_by_action": dict(Counter(row["action"] for row in records if _is_audio_response_record(row))),
                    "records_by_task": dict(Counter(row["task_kind"] for row in records)),
                }
                if conditioning_mode in {"audio_asr_mix_v1", "audio_asr_mix_v2"}
                else {"records_by_action": dict(Counter(row["action"] for row in records))}
                if direct_rows else _text_target_counts(conversations)
            ),
            "heldout_audio_dependency": heldout_audio_dependency,
            "validation_loss_by_task": validation_loss_by_task,
            "audio_input_cache": _audio_cache_stats(),
        },
        "data_sources": {
            "split_conversation_counts": prep_counts.get("split_conversation_counts", {
                "train": len({str(row.get("conversation_id", "")) for row in records}),
                **({"validation": len({str(row.get("conversation_id", "")) for row in validation_records})} if direct_rows else {}),
            }),
            "source_split_conversation_counts": prep_counts.get("source_split_conversation_counts", {}),
            "preparation_manifest": str(Path(preparation_manifest).resolve()) if preparation_manifest else None,
        },
    }
    _write_training_manifest(manifest_path, manifest)
    return manifest


def _text_target_counts(conversations: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Report complete/cropped synthetic assistant text without restoring masked labels."""
    counts = Counter()
    for conversation in conversations:
        if not conversation.get("synthetic"):
            continue
        for segment in conversation.get("segments", ()):
            if _role(segment, conversation.get("channel_map") or {}) != "assistant":
                continue
            text = _segment_text(segment)
            if not text:
                continue
            measured = segment.get("measured_duration_s")
            start = segment.get("start_ms", float(segment.get("start_s", 0) or 0) * 1000)
            end = segment.get("end_ms", float(segment.get("end_s", 0) or 0) * 1000)
            if isinstance(measured, (int, float)) and isinstance(start, (int, float)) and isinstance(end, (int, float)):
                cropped = (float(end) - float(start)) / 1000.0 < float(measured) - 0.08
            else:
                cropped = segment.get("text_mask", True) is False
            counts["cropped_assistant_text"] += int(cropped)
            counts["complete_assistant_text"] += int(not cropped)
            counts["text_ce_usable"] += int(not cropped and segment.get("text_mask", True) is True)
    return dict(counts)


__all__ = ["build_training_inputs", "count_response_examples", "train_qlora"]
