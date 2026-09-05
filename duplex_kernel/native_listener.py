# -*- coding: utf-8 -*-
"""duplex_kernel/native_listener.py — Python adapter for native Gemma 4 Listener LM.

Loads libgemma4_listener.so via ctypes, binds PyTorch parameter data_ptr()
with zero weight duplication, validates cryptographic topology hashes,
and exposes the single-call gemma4_listener_step_async interface.

Conforms strictly to P4-NATIVE Implementation Plan §6, §7, §22.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT

logger = logging.getLogger("duplex_kernel.native_listener")

GEMMA4_NUM_LAYERS = 35
GEMMA4_HIDDEN_SIZE = 1536
GEMMA4_PLE_DIM = 256
_MANIFEST_RELATIVE_PATH = Path("validation/kernel_gates/P4_NATIVE/GEMMA4_NATIVE_TOPOLOGY.json")


def _manifest_path() -> Path:
    candidates = (
        Path("/workspace") / _MANIFEST_RELATIVE_PATH,
        Path(__file__).resolve().parents[1] / _MANIFEST_RELATIVE_PATH,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"manifest topology non trovato: {_MANIFEST_RELATIVE_PATH}")


def _load_attested_manifest() -> dict[str, Any]:
    path = _manifest_path()
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    claimed_sha = manifest.get("manifest_sha256")
    if not isinstance(claimed_sha, str):
        raise RuntimeError("manifest topology privo di manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    computed_sha = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if computed_sha != claimed_sha:
        raise RuntimeError("manifest topology con hash interno non valido")
    return manifest


def _verify_attested_checkpoint(manifest: dict[str, Any]) -> Path:
    checkpoint_path = manifest.get("checkpoint_path")
    checkpoint_files = manifest.get("checkpoint_files")
    if not isinstance(checkpoint_path, str) or not isinstance(checkpoint_files, dict):
        raise RuntimeError("manifest privo di attestazione checkpoint completa")
    root = Path(checkpoint_path)
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint attestato non trovato: {root}")
    for relative_name, expected_sha in checkpoint_files.items():
        if not isinstance(relative_name, str) or not isinstance(expected_sha, str):
            raise RuntimeError("entry checkpoint_files non valida")
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError("percorso checkpoint non relativo al root attestato")
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"file checkpoint attestato non trovato: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha = digest.hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"hash checkpoint non valido per {relative_name}: {actual_sha} != {expected_sha}"
            )
    return root


class Gemma4TensorMeta(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("dtype", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("contiguous", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("dims", ctypes.c_uint64 * 4),
        ("strides", ctypes.c_uint64 * 4),
    ]


class Gemma4ListenerConfig(ctypes.Structure):
    _fields_ = [
        ("max_seq_len", ctypes.c_uint32),
        ("stream_q_len", ctypes.c_uint32),
        ("enable_cuda_graph", ctypes.c_bool),
        ("output_tensor_ptr", ctypes.c_void_p),
        ("frame_pad_ple_ptr", ctypes.c_void_p),
        ("weight_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("weight_meta_count", ctypes.c_uint32),
        ("output_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("frame_pad_ple_meta", ctypes.POINTER(Gemma4TensorMeta)),
    ]


class Gemma4LayerWeights(ctypes.Structure):
    _fields_ = [
        ("input_norm_w", ctypes.c_void_p),
        ("post_attn_norm_w", ctypes.c_void_p),
        ("pre_ff_norm_w", ctypes.c_void_p),
        ("post_ff_norm_w", ctypes.c_void_p),
        ("q_proj_w", ctypes.c_void_p),
        ("k_proj_w", ctypes.c_void_p),
        ("v_proj_w", ctypes.c_void_p),
        ("o_proj_w", ctypes.c_void_p),
        ("q_norm_w", ctypes.c_void_p),
        ("k_norm_w", ctypes.c_void_p),
        ("v_norm_w", ctypes.c_void_p),
        ("gate_proj_w", ctypes.c_void_p),
        ("up_proj_w", ctypes.c_void_p),
        ("down_proj_w", ctypes.c_void_p),
        ("ple_gate_w", ctypes.c_void_p),
        ("ple_proj_w", ctypes.c_void_p),
        ("ple_norm_w", ctypes.c_void_p),
    ]


class Gemma4ListenerWeights(ctypes.Structure):
    _fields_ = [
        ("final_norm_w", ctypes.c_void_p),
        ("per_layer_model_proj_w", ctypes.c_void_p),
        ("per_layer_proj_norm_w", ctypes.c_void_p),
        ("per_layer_model_proj_scale", ctypes.c_float),
        ("per_layer_input_scale", ctypes.c_float),
        ("layers", Gemma4LayerWeights * GEMMA4_NUM_LAYERS),
    ]


class Gemma4ImportedKV(ctypes.Structure):
    _fields_ = [
        ("k_ptr", ctypes.c_void_p),
        ("k_meta", ctypes.POINTER(Gemma4TensorMeta)),
        ("v_ptr", ctypes.c_void_p),
        ("v_meta", ctypes.POINTER(Gemma4TensorMeta)),
    ]


class Gemma4ImportedCache(ctypes.Structure):
    _fields_ = [("layers", Gemma4ImportedKV * GEMMA4_NUM_LAYERS)]


def _find_library() -> str:
    candidates = [
        Path("/workspace/native/gemma4_listener/libgemma4_listener.so"),
        Path(__file__).resolve().parents[1] / "native" / "gemma4_listener" / "libgemma4_listener.so",
        Path("native/gemma4_listener/libgemma4_listener.so").resolve(),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError("libgemma4_listener.so non trovata nei percorsi candidati")


class NativeGemmaListenerLM:
    """DS4-style native model-specific executor for Gemma 4 Listener LM.

    Single C ABI boundary per frame, device-resident KV and activation arena,
    zero weight duplication (borrowed data_ptr()); numerical parity is guarded
    by external gates before production use.
    """

    def __init__(
        self,
        language_model: Any,
        *,
        max_seq_len: int = 4096,
        stream_q_len: int = 4,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.language_model = language_model
        self.max_seq_len = int(max_seq_len)
        self.stream_q_len = int(stream_q_len)
        if self.stream_q_len != 4:
            raise ValueError("il backend Gemma nativo richiede stream_q_len=4")
        if self.max_seq_len < AUDIO_START_AFTER_TEXT + self.stream_q_len:
            raise ValueError("max_seq_len troppo piccolo per prefix canonico + un frame")
        if self.max_seq_len > 4096:
            raise ValueError("max_seq_len oltre la capacità del kernel SDPA nativo")
        self._pinned_tensors: list[torch.Tensor] = []
        self._cache_import_tensors: list[torch.Tensor] = []
        self._last_execution_stream: torch.cuda.Stream | None = None

        lib_path = _find_library()
        logger.info("[native_listener] Caricamento %s...", lib_path)
        self._lib = ctypes.CDLL(lib_path)
        self._setup_c_signatures()

        # 1. Attest ABI and Topology Hash
        abi_ver = self._lib.gemma4_listener_abi_version()
        if abi_ver != 3:
            raise RuntimeError(f"Mismatch ABI version nativa: atteso 3, ottenuto {abi_ver}")
        abi_sizes = {
            "config": int(self._lib.gemma4_listener_abi_config_size()),
            "tensor_meta": int(self._lib.gemma4_listener_abi_tensor_meta_size()),
            "layer_weights": int(self._lib.gemma4_listener_abi_layer_weights_size()),
            "weights": int(self._lib.gemma4_listener_abi_weights_size()),
            "imported_kv": int(self._lib.gemma4_listener_abi_imported_kv_size()),
            "imported_cache": int(self._lib.gemma4_listener_abi_imported_cache_size()),
        }
        expected_abi_sizes = {
            "config": ctypes.sizeof(Gemma4ListenerConfig),
            "tensor_meta": ctypes.sizeof(Gemma4TensorMeta),
            "layer_weights": ctypes.sizeof(Gemma4LayerWeights),
            "weights": ctypes.sizeof(Gemma4ListenerWeights),
            "imported_kv": ctypes.sizeof(Gemma4ImportedKV),
            "imported_cache": ctypes.sizeof(Gemma4ImportedCache),
        }
        if abi_sizes != expected_abi_sizes:
            raise RuntimeError(f"layout ABI nativa incompatibile: {abi_sizes} != {expected_abi_sizes}")
        abi_fields = {
            "tensor_meta": (Gemma4TensorMeta, (
                "ptr", "dtype", "ndim", "contiguous", "reserved", "dims", "strides",
            )),
            "config": (Gemma4ListenerConfig, (
                "max_seq_len", "stream_q_len", "enable_cuda_graph", "output_tensor_ptr",
                "frame_pad_ple_ptr", "weight_meta", "weight_meta_count", "output_meta",
                "frame_pad_ple_meta",
            )),
            "layer_weights": (Gemma4LayerWeights, (
                "input_norm_w", "post_attn_norm_w", "pre_ff_norm_w", "post_ff_norm_w",
                "q_proj_w", "k_proj_w", "v_proj_w", "o_proj_w", "q_norm_w", "k_norm_w",
                "v_norm_w", "gate_proj_w", "up_proj_w", "down_proj_w", "ple_gate_w",
                "ple_proj_w", "ple_norm_w",
            )),
            "weights": (Gemma4ListenerWeights, (
                "final_norm_w", "per_layer_model_proj_w", "per_layer_proj_norm_w",
                "per_layer_model_proj_scale", "per_layer_input_scale", "layers",
            )),
            "imported_kv": (Gemma4ImportedKV, ("k_ptr", "k_meta", "v_ptr", "v_meta")),
            "imported_cache": (Gemma4ImportedCache, ("layers",)),
        }
        for kind, (struct_type, fields) in abi_fields.items():
            offsets = (ctypes.c_size_t * len(fields))()
            count = int(self._lib[f"gemma4_listener_abi_{kind}_offsets"](offsets, len(fields)))
            expected_offsets = [int(getattr(struct_type, field).offset) for field in fields]
            if count != len(fields) or list(offsets) != expected_offsets:
                raise RuntimeError(
                    f"offset ABI nativa incompatibile per {kind}: "
                    f"{list(offsets)} != {expected_offsets} (count={count})"
                )

        manifest = _load_attested_manifest()
        checkpoint_root = _verify_attested_checkpoint(manifest)
        loaded_root = getattr(language_model, "_gemma4_checkpoint_path", None)
        if not isinstance(loaded_root, str) or Path(loaded_root).resolve() != checkpoint_root.resolve():
            raise RuntimeError(
                "provenienza checkpoint del language model non attestata o inattesa: "
                f"{loaded_root!r} != {str(checkpoint_root)!r}"
            )
        manifest_sha = self._lib.gemma4_listener_manifest_sha256().decode("utf-8")
        if manifest_sha != manifest["manifest_sha256"]:
            raise RuntimeError(
                "Manifest SHA nativa non corrisponde alla topologia attestata "
                f"({manifest_sha} != {manifest['manifest_sha256']})"
            )
        self._validate_model_topology(language_model, manifest)
        logger.info("[native_listener] Attestata lib nativa (ABI v%d, Manifest SHA: %s)", abi_ver, manifest_sha[:16])

        parameters = iter(language_model.parameters())
        first_parameter = next(parameters, None)
        if first_parameter is None or first_parameter.device.type != "cuda":
            raise ValueError("il checkpoint nativo deve risiedere interamente su CUDA")
        device = first_parameter.device
        if device.index != torch.cuda.current_device():
            raise ValueError("il checkpoint nativo deve usare il device CUDA corrente")
        for parameter in parameters:
            if parameter.device != device:
                raise ValueError("i pesi nativi devono risiedere su un solo device CUDA")
        self._model_device = device

        # 2. Register borrowed parameter pointers
        self._weights_struct = self._build_weights_struct(language_model)

        # 3. Preallocate persistent device output tensor and PAD PLE identities
        self.output_tensor = torch.empty(
            (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE),
            dtype=torch.bfloat16,
            device=self._model_device,
        )
        # Accept the non-contiguous Q=4 view produced by the rolling audio
        # tensor without allocating a temporary on every frame.
        self._frame_input = torch.empty_like(self.output_tensor)

        pad_token_id = int(getattr(getattr(language_model, "config", None), "pad_token_id", 0))
        pad_ids = torch.full((1, self.stream_q_len), pad_token_id, device=device, dtype=torch.long)
        with torch.inference_mode():
            pad_embeds = language_model.embed_tokens(pad_ids)
            self.frame_pad_ple = language_model.get_per_layer_inputs(pad_ids, pad_embeds)
        if (
            self.frame_pad_ple.device != device
            or self.frame_pad_ple.dtype != torch.bfloat16
            or not self.frame_pad_ple.is_contiguous()
            or tuple(self.frame_pad_ple.shape) != (1, self.stream_q_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM)
        ):
            raise ValueError("frame_pad_ple deve essere CUDA BF16 contiguo con shape [1,Q,35,256]")
        self._weight_meta = self._build_weight_meta(language_model)
        self._output_meta = self._tensor_meta(self.output_tensor, dtype_code=1)
        self._frame_meta = self._tensor_meta(self._frame_input, dtype_code=1)
        self._frame_pad_ple_meta = self._tensor_meta(self.frame_pad_ple, dtype_code=1)
        self._frame_meta_ptr = ctypes.pointer(self._frame_meta)
        self._prefix_embeds_meta: Gemma4TensorMeta | None = None
        self._prefix_ple_meta: Gemma4TensorMeta | None = None
        self._prefix_attention_meta: Gemma4TensorMeta | None = None

        # 4. Create native context
        self._config_struct = Gemma4ListenerConfig(
            max_seq_len=self.max_seq_len,
            stream_q_len=self.stream_q_len,
            enable_cuda_graph=enable_cuda_graph,
            output_tensor_ptr=int(self.output_tensor.data_ptr()),
            frame_pad_ple_ptr=int(self.frame_pad_ple.data_ptr()),
            weight_meta=ctypes.cast(self._weight_meta, ctypes.POINTER(Gemma4TensorMeta)),
            weight_meta_count=len(self._weight_meta),
            output_meta=ctypes.pointer(self._output_meta),
            frame_pad_ple_meta=ctypes.pointer(self._frame_pad_ple_meta),
        )

        self._ctx = ctypes.c_void_p()
        ret = self._lib.gemma4_listener_create(
            ctypes.byref(self._config_struct),
            ctypes.byref(self._weights_struct),
            ctypes.byref(self._ctx),
        )
        if ret != 0 or not self._ctx:
            raise RuntimeError(f"gemma4_listener_create failed with code {ret}")

        logger.info("[native_listener] Context creato con successo (Arena e Output residenti su GPU).")

    @staticmethod
    def _validate_model_topology(lm: Any, manifest: dict[str, Any]) -> None:
        def check_shape(tensor: Any, expected_shape: Any, label: str) -> None:
            if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected_shape):
                actual = None if tensor is None else tuple(tensor.shape)
                raise ValueError(f"{label}: shape inattesa {actual} != {tuple(expected_shape)}")

        def check_float(actual: Any, expected: Any, label: str) -> None:
            actual_float = float(actual)
            expected_float = float(expected)
            if not math.isfinite(actual_float) or not math.isfinite(expected_float):
                raise ValueError(f"{label}: valore non finito")
            if abs(actual_float - expected_float) > 1e-7:
                raise ValueError(f"{label}: valore inatteso {actual} != {expected}")

        config = getattr(lm, "config", None)
        layers = list(getattr(lm, "layers", ()))
        if config is None or len(layers) != GEMMA4_NUM_LAYERS:
            raise ValueError("checkpoint Gemma incompatibile: numero layer inatteso")
        if int(getattr(config, "hidden_size", -1)) != GEMMA4_HIDDEN_SIZE:
            raise ValueError("checkpoint Gemma incompatibile: hidden_size inatteso")
        if int(getattr(config, "num_hidden_layers", -1)) != GEMMA4_NUM_LAYERS:
            raise ValueError("checkpoint Gemma incompatibile: num_hidden_layers inatteso")
        if int(getattr(lm, "hidden_size_per_layer_input", -1)) != GEMMA4_PLE_DIM:
            raise ValueError("checkpoint Gemma incompatibile: PLE dim inattesa")
        if int(manifest.get("canonical_prefix_len", -1)) != AUDIO_START_AFTER_TEXT:
            raise ValueError("manifest topology incompatibile: prefix canonico inatteso")
        if int(manifest.get("canonical_lm_prefill_target_len", -1)) != 291:
            raise ValueError("manifest topology incompatibile: target prefill inatteso")
        ple_config = manifest.get("ple_global_config", {})
        check_shape(
            getattr(getattr(lm, "per_layer_model_projection", None), "weight", None),
            ple_config.get("per_layer_model_projection_shape", ()),
            "per_layer_model_projection",
        )
        check_shape(
            getattr(getattr(lm, "per_layer_projection_norm", None), "weight", None),
            (GEMMA4_PLE_DIM,),
            "per_layer_projection_norm",
        )
        check_float(
            getattr(lm, "per_layer_model_projection_scale", None),
            ple_config.get("per_layer_model_projection_scale"),
            "per_layer_model_projection_scale",
        )
        check_float(
            getattr(lm, "per_layer_input_scale", None),
            ple_config.get("per_layer_input_scale"),
            "per_layer_input_scale",
        )
        check_float(getattr(getattr(lm, "norm", None), "eps", None), manifest["final_norm_eps"], "final_norm_eps")
        rope_parameters = getattr(config, "rope_parameters", None)
        if not isinstance(rope_parameters, dict):
            raise ValueError("checkpoint Gemma privo di rope_parameters attestabili")

        expected_layers = manifest.get("layers", ())
        if len(expected_layers) != GEMMA4_NUM_LAYERS:
            raise ValueError("manifest topology incompatibile: numero layer inatteso")
        for index, (layer, expected) in enumerate(zip(layers, expected_layers)):
            attention = layer.self_attn
            if int(expected.get("index", -1)) != index:
                raise ValueError(f"manifest layer index inatteso: {index}")
            actual_type = str(config.layer_types[index])
            if actual_type != expected["layer_type"]:
                raise ValueError(f"layer {index}: attention type inatteso")
            actual_num_heads = getattr(
                attention,
                "num_attention_heads",
                getattr(attention, "num_heads", getattr(config, "num_attention_heads", -1)),
            )
            if int(actual_num_heads) != int(expected["num_attention_heads"]):
                raise ValueError(f"layer {index}: num_heads inatteso")
            actual_num_kv_heads = getattr(
                attention,
                "num_key_value_heads",
                getattr(config, "num_key_value_heads", -1),
            )
            if int(actual_num_kv_heads) != int(expected["num_key_value_heads"]):
                raise ValueError(f"layer {index}: num_key_value_heads inatteso")
            if int(attention.head_dim) != int(expected["head_dim"]):
                raise ValueError(f"layer {index}: head_dim inatteso")
            rope_type = actual_type
            expected_rope = expected["rope_scaling"][rope_type]
            actual_rope = rope_parameters.get(rope_type)
            if not isinstance(actual_rope, dict):
                raise ValueError(f"layer {index}: rope_parameters inattesi")
            if str(actual_rope.get("rope_type")) != str(expected_rope["rope_type"]):
                raise ValueError(f"layer {index}: rope_type inatteso")
            check_float(actual_rope.get("rope_theta"), expected_rope["rope_theta"], f"layer {index} rope_theta")
            if "partial_rotary_factor" in expected_rope:
                check_float(
                    actual_rope.get("partial_rotary_factor"),
                    expected_rope["partial_rotary_factor"],
                    f"layer {index} partial_rotary_factor",
                )
            actual_window = int(getattr(attention, "sliding_window", 0) or 0)
            expected_window = int(expected.get("sliding_window", 0) or 0)
            if actual_window != expected_window:
                raise ValueError(f"layer {index}: sliding_window inattesa")
            check_float(getattr(attention, "scaling", None), expected["attention_scaling"], f"layer {index} scaling")
            check_float(getattr(layer, "layer_scalar", None), expected["layer_scalar"], f"layer {index} layer_scalar")
            if int(getattr(layer, "hidden_size_per_layer_input", -1)) != GEMMA4_PLE_DIM:
                raise ValueError(f"layer {index}: PLE dim inattesa")
            actual_shared = bool(attention.is_kv_shared_layer)
            expected_shared = bool(expected["is_kv_shared_layer"])
            if actual_shared != expected_shared:
                raise ValueError(f"layer {index}: shared-KV topology inattesa")
            if actual_shared and int(attention.kv_shared_layer_index) != int(expected["kv_shared_layer_index"]):
                raise ValueError(f"layer {index}: shared-KV source inattesa")
            if bool(getattr(attention, "store_full_length_kv", False)) != bool(expected["store_full_length_kv"]):
                raise ValueError(f"layer {index}: store_full_length_kv inatteso")

            for name, module in (
                ("input_layernorm", layer.input_layernorm),
                ("post_attention_layernorm", layer.post_attention_layernorm),
                ("pre_feedforward_layernorm", layer.pre_feedforward_layernorm),
                ("post_feedforward_layernorm", layer.post_feedforward_layernorm),
                ("post_per_layer_input_norm", layer.post_per_layer_input_norm),
            ):
                check_shape(module.weight, (GEMMA4_HIDDEN_SIZE,), f"layer {index} {name}")
            for name, module in (("q_proj", attention.q_proj), ("k_proj", attention.k_proj), ("v_proj", attention.v_proj), ("o_proj", attention.o_proj)):
                check_shape(module.weight, expected[f"{name}_shape"], f"layer {index} {name}")
            for name, module in (("q_norm", attention.q_norm), ("k_norm", attention.k_norm)):
                check_shape(module.weight, (int(expected["head_dim"]),), f"layer {index} {name}")
            if getattr(attention.v_norm, "weight", None) is not None:
                raise ValueError(f"layer {index} v_norm: il checkpoint nativo atteso non ha peso")
            for name, module in (
                ("input_layernorm_eps", layer.input_layernorm),
                ("post_attention_layernorm_eps", layer.post_attention_layernorm),
                ("pre_feedforward_layernorm_eps", layer.pre_feedforward_layernorm),
                ("post_feedforward_layernorm_eps", layer.post_feedforward_layernorm),
                ("q_norm_eps", attention.q_norm),
                ("k_norm_eps", attention.k_norm),
                ("v_norm_eps", attention.v_norm),
                ("post_per_layer_input_norm_eps", layer.post_per_layer_input_norm),
            ):
                check_float(getattr(module, "eps", None), expected[name], f"layer {index} {name}")
            for name in ("gate_proj", "up_proj", "down_proj"):
                check_shape(getattr(layer.mlp, name).weight, expected[f"{name}_shape"], f"layer {index} {name}")
            check_shape(layer.per_layer_input_gate.weight, expected["per_layer_input_gate_shape"], f"layer {index} per_layer_input_gate")
            check_shape(layer.per_layer_projection.weight, expected["per_layer_projection_shape"], f"layer {index} per_layer_projection")

    def _setup_c_signatures(self) -> None:
        self._lib.gemma4_listener_manifest_sha256.restype = ctypes.c_char_p
        self._lib.gemma4_listener_manifest_sha256.argtypes = []

        self._lib.gemma4_listener_abi_version.restype = ctypes.c_int
        self._lib.gemma4_listener_abi_version.argtypes = []
        for name in (
            "gemma4_listener_abi_config_size",
            "gemma4_listener_abi_tensor_meta_size",
            "gemma4_listener_abi_layer_weights_size",
            "gemma4_listener_abi_weights_size",
            "gemma4_listener_abi_imported_kv_size",
            "gemma4_listener_abi_imported_cache_size",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_size_t
            function.argtypes = []
        for name in (
            "gemma4_listener_abi_config_offsets",
            "gemma4_listener_abi_tensor_meta_offsets",
            "gemma4_listener_abi_layer_weights_offsets",
            "gemma4_listener_abi_weights_offsets",
            "gemma4_listener_abi_imported_kv_offsets",
            "gemma4_listener_abi_imported_cache_offsets",
        ):
            function = getattr(self._lib, name)
            function.restype = ctypes.c_size_t
            function.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]

        self._lib.gemma4_listener_create.restype = ctypes.c_int
        self._lib.gemma4_listener_create.argtypes = [
            ctypes.POINTER(Gemma4ListenerConfig),
            ctypes.POINTER(Gemma4ListenerWeights),
            ctypes.POINTER(ctypes.c_void_p),
        ]

        self._lib.gemma4_listener_import_cache_async.restype = ctypes.c_int
        self._lib.gemma4_listener_import_cache_async.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4ImportedCache),
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_execute_layer.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_execute_layer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.POINTER(Gemma4ImportedCache),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_debug_attention.restype = ctypes.c_int
        self._lib.gemma4_listener_debug_attention.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_double,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_prime_async.restype = ctypes.c_int
        self._lib.gemma4_listener_prime_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_prime_debug.restype = ctypes.c_int
        self._lib.gemma4_listener_prime_debug.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_step_async.restype = ctypes.c_int
        self._lib.gemma4_listener_step_async.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_void_p,
        ]

        self._lib.gemma4_listener_step_debug.restype = ctypes.c_int
        self._lib.gemma4_listener_step_debug.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(Gemma4TensorMeta),
            ctypes.c_uint32,
            ctypes.c_bool,
            ctypes.c_void_p,
        ]


        self._lib.gemma4_listener_output_ptr.restype = ctypes.c_void_p
        self._lib.gemma4_listener_output_ptr.argtypes = [ctypes.c_void_p]
        self._lib.gemma4_listener_output_health.restype = ctypes.c_int
        self._lib.gemma4_listener_output_health.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)
        ]

        self._lib.gemma4_listener_debug_ptr.restype = ctypes.c_void_p
        self._lib.gemma4_listener_debug_ptr.argtypes = [ctypes.c_void_p, ctypes.c_char_p]


        self._lib.gemma4_listener_reset.restype = ctypes.c_int
        self._lib.gemma4_listener_reset.argtypes = [ctypes.c_void_p]

        self._lib.gemma4_listener_destroy.restype = None
        self._lib.gemma4_listener_destroy.argtypes = [ctypes.c_void_p]

        self._lib.gemma4_listener_current_position.restype = ctypes.c_uint32
        self._lib.gemma4_listener_current_position.argtypes = [ctypes.c_void_p]

        self._lib.gemma4_listener_get_cached_k.restype = ctypes.c_void_p
        self._lib.gemma4_listener_get_cached_k.argtypes = [ctypes.c_void_p, ctypes.c_int]

        self._lib.gemma4_listener_get_cached_v.restype = ctypes.c_void_p
        self._lib.gemma4_listener_get_cached_v.argtypes = [ctypes.c_void_p, ctypes.c_int]

    @staticmethod
    def _tensor_meta(tensor: torch.Tensor | None, *, dtype_code: int) -> Gemma4TensorMeta:
        meta = Gemma4TensorMeta()
        if tensor is None:
            return meta
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
            raise ValueError("metadata nativo richiede tensor CUDA")
        expected_dtype = {
            1: torch.bfloat16,
            2: torch.int64,
            3: torch.float32,
            4: torch.bool,
        }.get(int(dtype_code))
        if expected_dtype is None or tensor.dtype != expected_dtype:
            raise ValueError(f"dtype metadata nativo inatteso: {tensor.dtype} != {expected_dtype}")
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) > 4 or any(value <= 0 for value in shape):
            raise ValueError(f"rank/shape tensor nativo inattesi: {shape}")
        expected_strides = []
        running = 1
        for size in reversed(shape):
            expected_strides.append(running)
            running *= size
        expected_strides = tuple(reversed(expected_strides))
        if tuple(int(value) for value in tensor.stride()) != expected_strides:
            raise ValueError("metadata nativo richiede layout row-major contiguous")
        strides = tuple(int(value) for value in tensor.stride())
        meta.ptr = int(tensor.data_ptr())
        meta.dtype = int(dtype_code)
        meta.ndim = len(shape)
        meta.contiguous = 1
        meta.reserved = 0
        meta.dims = (ctypes.c_uint64 * 4)(*(shape + (0,) * (4 - len(shape))))
        meta.strides = (ctypes.c_uint64 * 4)(*(strides + (0,) * (4 - len(strides))))
        return meta

    def _ptr(self, tensor: torch.Tensor | None) -> int:
        if tensor is None:
            return 0
        if tensor.device.type != "cuda" or tensor.dtype != torch.bfloat16 or not tensor.is_contiguous():
            raise ValueError("tutti i pesi nativi devono essere tensor CUDA BF16 contigui")
        self._pinned_tensors.append(tensor)
        return int(tensor.data_ptr())

    def _build_weight_meta(self, lm: Any) -> Any:
        tensors: list[torch.Tensor | None] = [
            getattr(getattr(lm, "norm", None), "weight", None),
            getattr(getattr(lm, "per_layer_model_projection", None), "weight", None),
            getattr(getattr(lm, "per_layer_projection_norm", None), "weight", None),
        ]
        for layer in lm.layers[:GEMMA4_NUM_LAYERS]:
            attn = layer.self_attn
            tensors.extend([
                getattr(layer.input_layernorm, "weight", None),
                getattr(layer.post_attention_layernorm, "weight", None),
                getattr(layer.pre_feedforward_layernorm, "weight", None),
                getattr(layer.post_feedforward_layernorm, "weight", None),
                getattr(attn.q_proj, "weight", None),
                getattr(attn.k_proj, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.v_proj, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.o_proj, "weight", None),
                getattr(attn.q_norm, "weight", None),
                getattr(attn.k_norm, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(attn.v_norm, "weight", None) if not attn.is_kv_shared_layer else None,
                getattr(layer.mlp.gate_proj, "weight", None),
                getattr(layer.mlp.up_proj, "weight", None),
                getattr(layer.mlp.down_proj, "weight", None),
                getattr(layer.per_layer_input_gate, "weight", None),
                getattr(layer.per_layer_projection, "weight", None),
                getattr(layer.post_per_layer_input_norm, "weight", None),
            ])
        expected_count = 3 + GEMMA4_NUM_LAYERS * 17
        if len(tensors) != expected_count:
            raise RuntimeError(f"metadata pesi nativi incompleti: {len(tensors)} != {expected_count}")
        metadata = (Gemma4TensorMeta * expected_count)()
        for index, tensor in enumerate(tensors):
            metadata[index] = self._tensor_meta(tensor, dtype_code=1)
        return metadata

    def _build_weights_struct(self, lm: Any) -> Gemma4ListenerWeights:
        w = Gemma4ListenerWeights()
        w.final_norm_w = self._ptr(getattr(getattr(lm, "norm", None), "weight", None))

        if hasattr(lm, "per_layer_model_projection") and hasattr(lm.per_layer_model_projection, "weight"):
            w.per_layer_model_proj_w = self._ptr(lm.per_layer_model_projection.weight)
        if hasattr(lm, "per_layer_projection_norm") and hasattr(lm.per_layer_projection_norm, "weight"):
            w.per_layer_proj_norm_w = self._ptr(lm.per_layer_projection_norm.weight)

        w.per_layer_model_proj_scale = float(getattr(lm, "per_layer_model_projection_scale", 1.0))
        w.per_layer_input_scale = float(getattr(lm, "per_layer_input_scale", 1.0))

        for i, layer in enumerate(lm.layers[:GEMMA4_NUM_LAYERS]):
            lw = w.layers[i]
            lw.input_norm_w = self._ptr(layer.input_layernorm.weight)
            lw.post_attn_norm_w = self._ptr(layer.post_attention_layernorm.weight)
            lw.pre_ff_norm_w = self._ptr(layer.pre_feedforward_layernorm.weight)
            lw.post_ff_norm_w = self._ptr(layer.post_feedforward_layernorm.weight)

            attn = layer.self_attn
            lw.q_proj_w = self._ptr(attn.q_proj.weight)
            lw.q_norm_w = self._ptr(getattr(attn.q_norm, "weight", None))

            if not attn.is_kv_shared_layer:
                lw.k_proj_w = self._ptr(attn.k_proj.weight if attn.k_proj is not None else None)
                lw.v_proj_w = self._ptr(attn.v_proj.weight if attn.v_proj is not None else None)
                lw.k_norm_w = self._ptr(getattr(attn.k_norm, "weight", None))
                lw.v_norm_w = self._ptr(getattr(attn.v_norm, "weight", None))
            else:
                lw.k_proj_w = 0
                lw.v_proj_w = 0
                lw.k_norm_w = 0
                lw.v_norm_w = 0

            lw.o_proj_w = self._ptr(attn.o_proj.weight)

            mlp = layer.mlp
            lw.gate_proj_w = self._ptr(mlp.gate_proj.weight)
            lw.up_proj_w = self._ptr(mlp.up_proj.weight)
            lw.down_proj_w = self._ptr(mlp.down_proj.weight)

            if getattr(layer, "hidden_size_per_layer_input", 0):
                lw.ple_gate_w = self._ptr(layer.per_layer_input_gate.weight)
                lw.ple_proj_w = self._ptr(layer.per_layer_projection.weight)
                lw.ple_norm_w = self._ptr(getattr(layer.post_per_layer_input_norm, "weight", None))
            else:
                lw.ple_gate_w = 0
                lw.ple_proj_w = 0
                lw.ple_norm_w = 0

        return w

    def _cuda_stream(self, stream: torch.cuda.Stream | None) -> torch.cuda.Stream:
        if stream is not None:
            if torch.device(stream.device) != self.output_tensor.device:
                raise ValueError("CUDA stream e output nativo devono usare lo stesso device")
            return stream
        return torch.cuda.current_stream(device=self.output_tensor.device)

    def _resolve_stream(
        self,
        stream: torch.cuda.Stream | None,
        producer_stream: torch.cuda.Stream | None,
    ) -> torch.cuda.Stream:
        active = self._cuda_stream(stream)
        producer = self._cuda_stream(producer_stream)
        if active.cuda_stream != producer.cuda_stream:
            # Non-blocking CUDA dependency: the caller declares the stream on
            # which the borrowed inputs were produced; no host synchronization.
            active.wait_stream(producer)
        return active

    @classmethod
    def _canonicalize_cuda_tensor(cls, tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> torch.Tensor:
        cls._require_cuda_tensor(tensor, name=name, dtype=dtype)
        shape = tuple(int(value) for value in tensor.shape)
        if len(shape) > 4:
            raise ValueError(f"{name} rank inatteso: {len(shape)}")
        running = 1
        expected = []
        for size in reversed(shape):
            expected.append(running)
            running *= size
        expected_strides = tuple(reversed(expected))
        if tuple(int(value) for value in tensor.stride()) == expected_strides:
            return tensor
        canonical = torch.empty(shape, dtype=dtype, device=tensor.device)
        canonical.copy_(tensor)
        return canonical

    @staticmethod
    def _require_cuda_contiguous(tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} deve essere un torch.Tensor")
        if tensor.device.type != "cuda" or tensor.dtype != dtype:
            raise ValueError(f"{name} deve essere CUDA {dtype} contiguo")
        shape = tuple(int(value) for value in tensor.shape)
        running = 1
        expected = []
        for size in reversed(shape):
            expected.append(running)
            running *= size
        if tuple(int(value) for value in tensor.stride()) != tuple(reversed(expected)):
            raise ValueError(f"{name} deve avere stride row-major contiguo")

    @staticmethod
    def _require_cuda_tensor(tensor: torch.Tensor, *, name: str, dtype: torch.dtype) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} deve essere un torch.Tensor")
        if tensor.device.type != "cuda" or tensor.dtype != dtype:
            raise ValueError(f"{name} deve essere CUDA {dtype}")

    def _validate_prime_inputs(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None,
        prefix_attention: torch.Tensor | None,
    ) -> tuple[int, torch.cuda.Stream]:
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        self._require_cuda_contiguous(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        self._require_cuda_contiguous(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        self._require_cuda_contiguous(prefix_attention, name="prefix_attention", dtype=torch.int64)
        if prefix_embeds.ndim != 3 or tuple(prefix_embeds.shape[:1]) != (1,) or prefix_embeds.shape[2] != GEMMA4_HIDDEN_SIZE:
            raise ValueError(f"prefix_embeds inatteso: {tuple(prefix_embeds.shape)}")
        prefix_len = int(prefix_embeds.shape[1])
        expected_ple = (1, prefix_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM)
        if tuple(prefix_ple.shape) != expected_ple:
            raise ValueError(f"prefix_ple inatteso: atteso {expected_ple}, ottenuto {tuple(prefix_ple.shape)}")
        if tuple(prefix_attention.shape) != (1, prefix_len):
            raise ValueError(f"prefix_attention inatteso: atteso {(1, prefix_len)}, ottenuto {tuple(prefix_attention.shape)}")
        if prefix_embeds.device != self.output_tensor.device or prefix_ple.device != prefix_embeds.device or prefix_attention.device != prefix_embeds.device:
            raise ValueError("tutti gli input prime devono risiedere sul device CUDA del listener")
        if prefix_len != AUDIO_START_AFTER_TEXT:
            raise ValueError(
                f"prefix_len canonico inatteso: {prefix_len}, atteso {AUDIO_START_AFTER_TEXT}"
            )
        if not bool(torch.all((prefix_attention == 0) | (prefix_attention == 1)).item()):
            raise ValueError("prefix_attention deve contenere esclusivamente 0/1")
        self._prefix_embeds_meta = self._tensor_meta(prefix_embeds, dtype_code=1)
        self._prefix_ple_meta = self._tensor_meta(prefix_ple, dtype_code=1)
        self._prefix_attention_meta = self._tensor_meta(prefix_attention, dtype_code=2)
        return prefix_len, self._cuda_stream(None)

    def _prepare_imported_cache(
        self,
        past_key_values: Any,
        current_position: int,
    ) -> tuple[Gemma4ImportedCache, list[Gemma4TensorMeta], list[torch.Tensor]]:
        layers = getattr(past_key_values, "layers", None)
        source_layer_count = sum(
            not bool(getattr(layer.self_attn, "is_kv_shared_layer", False))
            for layer in self.language_model.layers[:GEMMA4_NUM_LAYERS]
        )
        if layers is None or len(layers) < source_layer_count:
            raise ValueError(
                "DynamicCache canonica priva dei layer KV sorgente richiesti: "
                f"{len(layers) if layers is not None else 0} < {source_layer_count}"
            )
        get_seq_length = getattr(past_key_values, "get_seq_length", None)
        if callable(get_seq_length) and int(get_seq_length(0)) != current_position:
            raise ValueError("DynamicCache canonica con lunghezza inattesa")

        imported = Gemma4ImportedCache()
        descriptors: list[Gemma4TensorMeta] = []
        imported_tensors: list[torch.Tensor] = []
        shared_layers = getattr(past_key_values, "shared_layers", None)
        for layer_idx in range(GEMMA4_NUM_LAYERS):
            attention = self.language_model.layers[layer_idx].self_attn
            if bool(getattr(attention, "is_kv_shared_layer", False)):
                continue
            source = None
            if shared_layers is not None and hasattr(shared_layers, "get"):
                source = shared_layers.get(layer_idx)
            if source is None:
                source_layer = layers[layer_idx]
                keys = getattr(source_layer, "keys", None)
                values = getattr(source_layer, "values", None)
            else:
                keys, values = source
            head_dim = int(attention.head_dim)
            expected_shape = (1, 1, current_position, head_dim)
            keys = self._canonicalize_cuda_tensor(
                keys, name=f"cache K layer {layer_idx}", dtype=torch.bfloat16
            )
            values = self._canonicalize_cuda_tensor(
                values, name=f"cache V layer {layer_idx}", dtype=torch.bfloat16
            )
            if tuple(keys.shape) != expected_shape or tuple(values.shape) != expected_shape:
                raise ValueError(
                    f"cache KV layer {layer_idx} inattesa: "
                    f"{tuple(keys.shape)}/{tuple(values.shape)} != {expected_shape}"
                )
            k_meta = self._tensor_meta(keys, dtype_code=1)
            v_meta = self._tensor_meta(values, dtype_code=1)
            descriptors.extend((k_meta, v_meta))
            entry = imported.layers[layer_idx]
            entry.k_ptr = int(keys.data_ptr())
            entry.k_meta = ctypes.pointer(descriptors[-2])
            entry.v_ptr = int(values.data_ptr())
            entry.v_meta = ctypes.pointer(descriptors[-1])
            imported_tensors.extend((keys, values))
        return imported, descriptors, imported_tensors

    def import_cache_async(
        self,
        past_key_values: Any,
        current_position: int = AUDIO_START_AFTER_TEXT,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Import an exact ShapeStable DynamicCache with D2D K/V copies.

        Only the non-shared source layers are described. Native shared layers
        already alias sources 13 and 14, matching Gemma's shared-KV topology.
        Descriptor validation is synchronous, while every K/V payload transfer
        is device-to-device and ordered on ``stream``.
        """
        current_position = int(current_position)
        if current_position != AUDIO_START_AFTER_TEXT:
            raise ValueError(
                f"cache import richiede posizione canonica {AUDIO_START_AFTER_TEXT}, "
                f"ottenuta {current_position}"
            )
        imported, descriptors, imported_tensors = self._prepare_imported_cache(
            past_key_values, current_position
        )
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in imported_tensors:
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        ret = self._lib.gemma4_listener_import_cache_async(
            self._ctx,
            ctypes.byref(imported),
            ctypes.c_uint32(current_position),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_import_cache_async failed with code {ret}")
        self._cache_import_tensors = imported_tensors
        self._last_execution_stream = active_stream

    def debug_execute_layer(
        self,
        layer_idx: int,
        hidden_input: torch.Tensor,
        projected_ple: torch.Tensor,
        past_key_values: Any,
        current_position: int = AUDIO_START_AFTER_TEXT,
        query_position: int = AUDIO_START_AFTER_TEXT,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Execute one layer from exact oracle hidden/PLE/KV tensors.

        This is diagnostic only: it does not advance ``current_position`` or
        mark the context primed. Reset the context between independent probes.
        """
        layer_idx = int(layer_idx)
        if not 0 <= layer_idx < GEMMA4_NUM_LAYERS:
            raise ValueError(f"layer_idx inatteso: {layer_idx}")
        hidden_input = self._canonicalize_cuda_tensor(
            hidden_input, name="debug hidden_input", dtype=torch.bfloat16
        )
        projected_ple = self._canonicalize_cuda_tensor(
            projected_ple, name="debug projected_ple", dtype=torch.bfloat16
        )
        if tuple(hidden_input.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"debug hidden_input inatteso: {tuple(hidden_input.shape)}")
        expected_ple = (
            1, self.stream_q_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM
        )
        if tuple(projected_ple.shape) != expected_ple:
            raise ValueError(f"debug projected_ple inatteso: {tuple(projected_ple.shape)}")
        imported, descriptors, imported_tensors = self._prepare_imported_cache(
            past_key_values, int(current_position)
        )
        # Keep descriptor storage alive until the C ABI call returns.
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (*imported_tensors, hidden_input, projected_ple):
            tensor.record_stream(active_stream)
        hidden_meta = self._tensor_meta(hidden_input, dtype_code=1)
        ple_meta = self._tensor_meta(projected_ple, dtype_code=1)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        ret = self._lib.gemma4_listener_debug_execute_layer(
            self._ctx,
            ctypes.c_uint32(layer_idx),
            ctypes.c_void_p(hidden_input.data_ptr()),
            ctypes.byref(hidden_meta),
            ctypes.c_void_p(projected_ple.data_ptr()),
            ctypes.byref(ple_meta),
            ctypes.byref(imported),
            ctypes.c_uint32(int(current_position)),
            ctypes.c_uint32(int(query_position)),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_execute_layer failed with code {ret}")
        self._cache_import_tensors = imported_tensors
        self._last_execution_stream = active_stream
        self.output_tensor.record_stream(active_stream)
        return self.output_tensor

    def debug_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor,
        scale: float = 1.0,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Run the public ATen SDPA parity bridge on borrowed CUDA tensors.

        This is diagnostic-only.  It deliberately mirrors Transformers'
        explicit-mask path: K/V are repeated before calling SDPA with
        ``enable_gqa=False``.  The bridge writes into a fresh BF16 output and
        does not alter the native listener position or cache.
        """
        tensors = (("query", query), ("key", key), ("value", value))
        for name, tensor in tensors:
            self._require_cuda_tensor(tensor, name=name, dtype=torch.bfloat16)
            if not tensor.is_contiguous() or tensor.ndim != 4:
                raise ValueError(f"{name} deve essere CUDA BF16 contiguo rank-4")
        if not isinstance(attn_mask, torch.Tensor):
            raise TypeError("attn_mask deve essere un torch.Tensor")
        self._require_cuda_tensor(attn_mask, name="attn_mask", dtype=attn_mask.dtype)
        if attn_mask.dtype not in (torch.bfloat16, torch.float32, torch.bool):
            raise ValueError("attn_mask deve essere BF16, FP32 o BOOL")
        if not attn_mask.is_contiguous() or attn_mask.ndim != 4:
            raise ValueError("attn_mask deve essere contiguo rank-4")
        batch, query_heads, query_len, head_dim = (int(v) for v in query.shape)
        if tuple(int(v) for v in key.shape[:1]) != (batch,) or tuple(int(v) for v in value.shape[:1]) != (batch,):
            raise ValueError("batch Q/K/V incompatibile")
        if tuple(key.shape) != tuple(value.shape) or int(key.shape[3]) != head_dim:
            raise ValueError("shape K/V incompatibile con Q")
        if query_heads % int(key.shape[1]) != 0:
            raise ValueError("Q heads devono essere multipli dei KV heads")
        if tuple(int(v) for v in attn_mask.shape[2:]) != (query_len, int(key.shape[2])):
            raise ValueError("shape attn_mask incompatibile con Q/K")
        if int(attn_mask.shape[0]) not in (1, batch) or int(attn_mask.shape[1]) not in (1, query_heads):
            raise ValueError("attn_mask non broadcastable sulle teste")

        active_stream = self._resolve_stream(stream, producer_stream)
        output = torch.empty_like(query)
        for tensor in (query, key, value, attn_mask, output):
            tensor.record_stream(active_stream)
        query_meta = self._tensor_meta(query, dtype_code=1)
        key_meta = self._tensor_meta(key, dtype_code=1)
        value_meta = self._tensor_meta(value, dtype_code=1)
        mask_dtype_code = {
            torch.bfloat16: 1,
            torch.float32: 3,
            torch.bool: 4,
        }[attn_mask.dtype]
        mask_meta = self._tensor_meta(attn_mask, dtype_code=mask_dtype_code)
        output_meta = self._tensor_meta(output, dtype_code=1)
        ret = self._lib.gemma4_listener_debug_attention(
            ctypes.c_void_p(query.data_ptr()),
            ctypes.byref(query_meta),
            ctypes.c_void_p(key.data_ptr()),
            ctypes.byref(key_meta),
            ctypes.c_void_p(value.data_ptr()),
            ctypes.byref(value_meta),
            ctypes.c_void_p(attn_mask.data_ptr()),
            ctypes.byref(mask_meta),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.byref(output_meta),
            ctypes.c_double(float(scale)),
            ctypes.c_void_p(active_stream.cuda_stream),
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_debug_attention failed with code {ret}")
        output.record_stream(active_stream)
        return output

    def prime_async(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None = None,
        prefix_attention: torch.Tensor | None = None,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Prime the canonical [BOS|text|BOA] prefix and its exact mask."""
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        prefix_embeds = self._canonicalize_cuda_tensor(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        prefix_ple = self._canonicalize_cuda_tensor(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        prefix_attention = self._canonicalize_cuda_tensor(prefix_attention, name="prefix_attention", dtype=torch.int64)
        prefix_len, _ = self._validate_prime_inputs(prefix_embeds, prefix_ple, prefix_attention)
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (prefix_embeds, prefix_ple, prefix_attention):
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_embeds = ctypes.c_void_p(prefix_embeds.data_ptr())
        p_ple = ctypes.c_void_p(prefix_ple.data_ptr())
        p_attention = ctypes.c_void_p(prefix_attention.data_ptr())

        ret = self._lib.gemma4_listener_prime_async(
            self._ctx,
            p_embeds,
            ctypes.byref(self._prefix_embeds_meta),
            p_ple,
            ctypes.byref(self._prefix_ple_meta),
            p_attention,
            ctypes.byref(self._prefix_attention_meta),
            ctypes.c_uint32(prefix_len),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_prime_async failed with code {ret}")

    def prime_debug(
        self,
        prefix_embeds: torch.Tensor,
        prefix_ple: torch.Tensor | None = None,
        prefix_attention: torch.Tensor | None = None,
        num_layers: int = 1,
        apply_final_norm: bool = False,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Prime the prefix for diagnostics using the exact canonical mask."""
        if prefix_ple is None or prefix_attention is None:
            raise ValueError("prime nativo richiede PLE e prefix_attention espliciti")
        prefix_embeds = self._canonicalize_cuda_tensor(prefix_embeds, name="prefix_embeds", dtype=torch.bfloat16)
        prefix_ple = self._canonicalize_cuda_tensor(prefix_ple, name="prefix_ple", dtype=torch.bfloat16)
        prefix_attention = self._canonicalize_cuda_tensor(prefix_attention, name="prefix_attention", dtype=torch.int64)
        prefix_len, _ = self._validate_prime_inputs(prefix_embeds, prefix_ple, prefix_attention)
        active_stream = self._resolve_stream(stream, producer_stream)
        for tensor in (prefix_embeds, prefix_ple, prefix_attention):
            tensor.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_embeds = ctypes.c_void_p(prefix_embeds.data_ptr())
        p_ple = ctypes.c_void_p(prefix_ple.data_ptr())
        p_attention = ctypes.c_void_p(prefix_attention.data_ptr())

        ret = self._lib.gemma4_listener_prime_debug(
            self._ctx,
            p_embeds,
            ctypes.byref(self._prefix_embeds_meta),
            p_ple,
            ctypes.byref(self._prefix_ple_meta),
            p_attention,
            ctypes.byref(self._prefix_attention_meta),
            ctypes.c_uint32(prefix_len),
            ctypes.c_uint32(num_layers),
            ctypes.c_bool(apply_final_norm),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_prime_debug failed with code {ret}")

    def step_async(
        self,
        frame_audio_soft: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Executes one continuous 80 ms audio frame (Q=4 soft tokens).

        Returns a borrowed persistent output view [1, 4, 1536]. It remains
        valid until the next native invocation or ``close()``; consumers must
        finish using it before advancing the executor again.
        """
        active_stream = self._resolve_stream(stream, producer_stream)
        self._require_cuda_tensor(frame_audio_soft, name="frame_audio_soft", dtype=torch.bfloat16)
        if frame_audio_soft.device != self.output_tensor.device:
            raise ValueError("frame_audio_soft deve risiedere sul device del listener nativo")
        if tuple(frame_audio_soft.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"frame_audio_soft inatteso: {tuple(frame_audio_soft.shape)}")
        frame_audio_soft.record_stream(active_stream)
        with torch.cuda.stream(active_stream):
            self._frame_input.copy_(frame_audio_soft, non_blocking=True)
        self._frame_input.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_audio = ctypes.c_void_p(self._frame_input.data_ptr())

        ret = self._lib.gemma4_listener_step_async(
            self._ctx,
            p_audio,
            self._frame_meta_ptr,
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_step_async failed with code {ret}")
        self._last_execution_stream = active_stream
        # The output storage is persistent and reused by the next invocation.
        # This records its stream for allocator lifetime tracking; callers
        # consuming it on another stream must call wait_for_output().
        self.output_tensor.record_stream(active_stream)

        return self.output_tensor

    def wait_for_output(self, stream: torch.cuda.Stream | None = None) -> None:
        """Fence the last borrowed output before consuming it on ``stream``.

        Execution is stream-affine and same-stream consumers are already
        ordered. A different CUDA stream gets a device-side wait; no host
        synchronization is performed.
        """
        producer = self._last_execution_stream
        if producer is None:
            return
        consumer = self._cuda_stream(stream)
        if consumer.cuda_stream != producer.cuda_stream:
            consumer.wait_stream(producer)
        self.output_tensor.record_stream(consumer)

    def step_debug(
        self,
        frame_audio_soft: torch.Tensor,
        num_layers: int = 1,
        apply_final_norm: bool = False,
        stream: torch.cuda.Stream | None = None,
        producer_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        """Executes the first num_layers decoder layers for fine-grained debugging/parity checks.

        Returns output_tensor containing the intermediate hidden states after layer num_layers-1.
        """
        active_stream = self._resolve_stream(stream, producer_stream)
        self._require_cuda_tensor(frame_audio_soft, name="frame_audio_soft", dtype=torch.bfloat16)
        if frame_audio_soft.device != self.output_tensor.device:
            raise ValueError("frame_audio_soft deve risiedere sul device del listener nativo")
        if tuple(frame_audio_soft.shape) != (1, self.stream_q_len, GEMMA4_HIDDEN_SIZE):
            raise ValueError(f"frame_audio_soft inatteso: {tuple(frame_audio_soft.shape)}")
        frame_audio_soft.record_stream(active_stream)
        with torch.cuda.stream(active_stream):
            self._frame_input.copy_(frame_audio_soft, non_blocking=True)
        self._frame_input.record_stream(active_stream)
        c_stream = ctypes.c_void_p(active_stream.cuda_stream)
        p_audio = ctypes.c_void_p(self._frame_input.data_ptr())

        ret = self._lib.gemma4_listener_step_debug(
            self._ctx,
            p_audio,
            self._frame_meta_ptr,
            ctypes.c_uint32(num_layers),
            ctypes.c_bool(apply_final_norm),
            c_stream,
        )
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_step_debug failed with code {ret}")
        self._last_execution_stream = active_stream
        self.output_tensor.record_stream(active_stream)

        return self.output_tensor

    def copy_debug_tensor(self, name: str, shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Copies an internal device scratch buffer into a new PyTorch tensor for inspection."""
        ptr = self._lib.gemma4_listener_debug_ptr(self._ctx, name.encode("utf-8"))
        if not ptr:
            raise ValueError(f"Unknown or null debug buffer: {name}")
        out = torch.empty(shape, dtype=dtype, device=self.output_tensor.device)
        libcudart = ctypes.CDLL("libcudart.so")
        ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(out.numel() * out.element_size()),
            ctypes.c_int(3),  # cudaMemcpyDeviceToDevice
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy failed with code {ret}")
        return out

    def get_cached_kv(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Copies the cached K and V tensors for layer_idx up to current_position."""
        if not 0 <= int(layer_idx) < GEMMA4_NUM_LAYERS:
            raise ValueError(f"layer_idx inatteso: {layer_idx}")
        k_ptr = self._lib.gemma4_listener_get_cached_k(self._ctx, ctypes.c_int(layer_idx))
        v_ptr = self._lib.gemma4_listener_get_cached_v(self._ctx, ctypes.c_int(layer_idx))
        head_dim = 512 if self.language_model.config.layer_types[layer_idx] == "full_attention" else 256
        seq_len = self.current_position
        if seq_len > 0 and (not k_ptr or not v_ptr):
            raise RuntimeError("cache K/V nativa non disponibile")
        k_out = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=self.output_tensor.device)
        v_out = torch.empty((1, 1, seq_len, head_dim), dtype=torch.bfloat16, device=self.output_tensor.device)
        libcudart = ctypes.CDLL("libcudart.so")
        k_ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(k_out.data_ptr()),
            ctypes.c_void_p(k_ptr),
            ctypes.c_size_t(seq_len * head_dim * 2),
            ctypes.c_int(3),
        )
        if k_ret != 0:
            raise RuntimeError(f"cudaMemcpy K failed with code {k_ret}")
        v_ret = libcudart.cudaMemcpy(
            ctypes.c_void_p(v_out.data_ptr()),
            ctypes.c_void_p(v_ptr),
            ctypes.c_size_t(seq_len * head_dim * 2),
            ctypes.c_int(3),
        )
        if v_ret != 0:
            raise RuntimeError(f"cudaMemcpy V failed with code {v_ret}")
        return k_out, v_out

    def reset(self) -> None:
        """Reset the KV position and invalidate the primed prefix."""
        ret = self._lib.gemma4_listener_reset(self._ctx)
        if ret != 0:
            raise RuntimeError(f"gemma4_listener_reset failed with code {ret}")
        self._cache_import_tensors = []
        self._last_execution_stream = None

    def verify_output_health(self) -> bool:
        """Synchronously validate the last output; never call in steady-state."""
        status = ctypes.c_int(0)
        rc = int(self._lib.gemma4_listener_output_health(self._ctx, ctypes.byref(status)))
        if rc < 0:
            raise RuntimeError(f"health check output nativo fallito: rc={rc}")
        if rc != 0 or int(status.value) != 0:
            raise RuntimeError("output nativo non finito; contesto marcato poisoned")
        return True

    @property
    def current_position(self) -> int:
        return int(self._lib.gemma4_listener_current_position(self._ctx))

    def close(self) -> None:
        """Release native storage after draining the executor stream."""
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self._lib.gemma4_listener_destroy(ctx)
            self._ctx = None
        self._pinned_tensors.clear()
        self._cache_import_tensors.clear()
        self._last_execution_stream = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors must not mask interpreter shutdown or a prior CUDA
            # failure. The C destructor still performs best-effort draining.
            self._ctx = None


__all__ = [
    "NativeGemmaListenerLM",
    "Gemma4TensorMeta",
    "Gemma4ListenerConfig",
    "Gemma4ListenerWeights",
    "Gemma4ImportedKV",
    "Gemma4ImportedCache",
]
