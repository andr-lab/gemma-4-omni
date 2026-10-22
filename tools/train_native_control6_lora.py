#!/usr/bin/env python3
"""Prepare and train a six-row native Gemma conversational-control LoRA.

Subcommands:
  prepare-targets  Project the existing authored Italian action schedule to
                   explicit 80 ms targets, retaining its conversation split.
  extract         Replay each selected stereo dialogue through the production
                   Gemma listener and cache final-normalized newest-frame hiddens.
  train           Fit only a rank-r delta on six tied native LM output rows.

The extractor loads only the local Gemma base model (no PEFT response adapter,
Qwen model, or network download). Do not start `extract` while another job owns
the GPU. The dataset/cache formats fail closed on split, token, and hidden-state
contract mismatches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.native_control6 import (  # noqa: E402
    ACTION_TO_TOKEN_ID,
    ACTIONS,
    BASE_WEIGHT_SOURCE,
    DELTA_INPUT_WIDTH,
    HIDDEN_CONTRACT,
    HIDDEN_NORM,
    HIDDEN_SOURCE,
    HIDDEN_WIDTH,
    MODEL_ID,
    NATIVE1540_PHASE_SCHEMA,
    PHASE_T_WIDTH,
    NativeControl6LoRA,
    TICK_MS,
    TOKEN_ID_TO_STRING,
    load_base_rows,
    load_native_control6_adapter,
    save_base_rows,
)
from training.duplex_qlora.features import (  # noqa: E402
    native_control6_feature_contract,
)
from duplex_kernel.native_control6_prompt import (  # noqa: E402
    NATIVE_CONTROL6_PREFIX_CONTRACT,
    NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
    NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION,
)
from duplex_kernel.gemma_native_control6_policy import (  # noqa: E402
    native_control6_effective_members,
)

TARGET_SCHEMA_VERSION = 1
TARGET_SCHEMA = "native_control6_action_targets_v1"
FAMILY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
PHYSICAL_ACTIONS = tuple(action for action in ACTIONS if action != "REPLAN")
RUNTIME_BUCKETS = ("HOLD", "SPEAK", "YIELD", "BACKCHANNEL")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def _require_synthetic_manifest_row(row: Mapping[str, Any]) -> None:
    conversation_id = row.get("conversation_id")
    if not isinstance(conversation_id, str) or not conversation_id.startswith("synthetic-it-t6-"):
        raise ValueError("only the explicit synthetic-it-t6 manifest is eligible here")
    if row.get("human_annotated") is not False:
        raise ValueError(f"{conversation_id}: expected authored synthetic, non-human annotation flag")
    if not isinstance(row.get("template_family_id"), str) or not row["template_family_id"].strip():
        raise ValueError(f"{conversation_id}: missing template_family_id for grouped split audit")
    if row.get("split") not in {"train", "validation", "test"}:
        raise ValueError(f"{conversation_id}: missing explicit train/validation/test split")
    if not isinstance(row.get("action_segments"), list):
        raise ValueError(f"{conversation_id}: missing explicit authored action_segments")
    if not isinstance(row.get("audio_path"), str) or not row.get("audio_path"):
        raise ValueError(f"{conversation_id}: missing source PCM path")


def _validate_source_group_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    ids: set[str] = set()
    family_to_split: dict[str, str] = {}
    family_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    for row in rows:
        _require_synthetic_manifest_row(row)
        conversation_id = str(row["conversation_id"])
        if conversation_id in ids:
            raise ValueError(f"duplicate conversation_id: {conversation_id}")
        ids.add(conversation_id)
        family = str(row["template_family_id"])
        split = str(row["split"])
        previous = family_to_split.setdefault(family, split)
        if previous != split:
            raise ValueError(f"source template family {family!r} leaks across {previous}/{split}")
        family_counts[split] += 1
        group_counts[f"{split}:{family}"] += 1
    if not family_counts["train"] or not family_counts["validation"]:
        raise ValueError("synthetic manifest needs nonempty explicit train and validation groups")
    return {
        "conversations_train": family_counts["train"],
        "conversations_validation": family_counts["validation"],
        "template_families_train": sum(key.startswith("train:") for key in group_counts),
        "template_families_validation": sum(key.startswith("validation:") for key in group_counts),
        "template_families_test": sum(key.startswith("test:") for key in group_counts),
    }


def _target_action_for_tick(
    segments: Sequence[Mapping[str, Any]], start_ms: float, end_ms: float
) -> str | None:
    # A target is accepted only when one explicit segment fully covers the
    # complete tick. Partial coverage, gaps, overlap, and unknown names mask it.
    covering: list[str] = []
    for segment in segments:
        action = segment.get("action")
        start = segment.get("start_ms")
        end = segment.get("end_ms")
        if (
            isinstance(action, str)
            and action in ACTIONS
            and isinstance(start, (int, float))
            and isinstance(end, (int, float))
            and float(start) <= start_ms + 1e-6
            and float(end) >= end_ms - 1e-6
        ):
            covering.append(action)
    return covering[0] if len(covering) == 1 else None


def project_manifest_targets(manifest_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    source_rows = _read_jsonl(manifest_path)
    split_audit = _validate_source_group_split(source_rows)
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "test")}
    per_split_counts: dict[str, Counter[str]] = {name: Counter() for name in by_split}
    per_split_masked: Counter[str] = Counter()
    per_split_manifest_counts: dict[str, Counter[str]] = {
        name: Counter() for name in by_split
    }

    for source in source_rows:
        conversation_id = str(source["conversation_id"])
        split = str(source["split"])
        family = str(source["template_family_id"])
        source_counts = source.get("action_frame_counts") or {}
        if isinstance(source_counts, dict):
            for action, count in source_counts.items():
                if action in ACTIONS and isinstance(count, int) and count >= 0:
                    per_split_manifest_counts[split][action] += count
        duration_s = source.get("duration_s")
        if not isinstance(duration_s, (int, float)) or not math.isfinite(float(duration_s)) or duration_s <= 0:
            raise ValueError(f"{conversation_id}: invalid duration_s")
        duration_ms = float(duration_s) * 1000.0
        frame_count = math.ceil(duration_ms / TICK_MS)
        segments = source["action_segments"]
        for frame_id in range(frame_count):
            start_ms = frame_id * TICK_MS
            end_ms = min(duration_ms, start_ms + TICK_MS)
            full_tick = end_ms - start_ms >= TICK_MS - 1e-6
            action = _target_action_for_tick(segments, start_ms, end_ms) if full_tick else None
            action_mask = action is not None
            target = {
                "schema_version": TARGET_SCHEMA_VERSION,
                "schema": TARGET_SCHEMA,
                "conversation_id": conversation_id,
                "template_family_id": family,
                "split": split,
                "frame_id": frame_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "valid_ms": max(0.0, end_ms - start_ms),
                "target_action": action,
                "action_mask": action_mask,
                "label_origin": str(source.get("label_origin", "authored_action_segments")),
                "human_annotated": False,
            }
            by_split[split].append(target)
            if action_mask:
                per_split_counts[split][action] += 1
            else:
                per_split_masked[split] += 1

    for split_rows in by_split.values():
        split_rows.sort(key=lambda row: (row["conversation_id"], row["frame_id"]))
    summary = {
        "schema_version": TARGET_SCHEMA_VERSION,
        "schema": TARGET_SCHEMA,
        "manifest_path": str(manifest_path),
        "tick_ms": TICK_MS,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "source_group_split": split_audit,
        "label_origin": "authored synthetic action schedule; not human annotation",
        "counts": {
            split: {
                "conversations": len({r["conversation_id"] for r in rows}),
                "frames": len(rows),
                "action_masked_frames": int(per_split_masked[split]),
                "actions": {action: int(per_split_counts[split][action]) for action in ACTIONS},
                "manifest_action_frame_counts": {
                    action: int(per_split_manifest_counts[split][action]) for action in ACTIONS
                },
                "strict_full_tick_delta_vs_manifest": {
                    action: int(per_split_counts[split][action] - per_split_manifest_counts[split][action])
                    for action in ACTIONS
                },
                "template_families": sorted({r["template_family_id"] for r in rows}),
            }
            for split, rows in by_split.items()
        },
    }
    return by_split, summary


def prepare_targets(manifest: Path, output_dir: Path) -> dict[str, Any]:
    split_rows, summary = project_manifest_targets(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        if rows:
            _write_jsonl(output_dir / f"{split}_targets.jsonl", rows)
    temp = output_dir / "summary.json.tmp"
    temp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(output_dir / "summary.json")
    return summary


def _normalise_channel_map(row: Mapping[str, Any]) -> dict[str, str]:
    raw = row.get("channel_map")
    if not isinstance(raw, dict):
        raise ValueError(f"{row.get('conversation_id')}: source channel_map missing")
    role_aliases = {
        "user": {"user", "fictional_user"},
        "assistant": {"assistant", "fictional_assistant"},
    }
    resolved: dict[str, str] = {}
    for channel, value in raw.items():
        if not str(channel).isdigit():
            continue
        role = next((canonical for canonical, aliases in role_aliases.items() if value in aliases), None)
        if role is not None:
            if role in resolved.values():
                raise ValueError(f"{row.get('conversation_id')}: ambiguous {role} audio channel")
            resolved[str(channel)] = role
    if set(resolved.values()) != {"user", "assistant"}:
        raise ValueError(f"{row.get('conversation_id')}: expected explicit two-track user/assistant channels")
    return resolved


def _load_base_thinker(base_model_path: Path, device: str) -> tuple[Any, Any, Any, torch.Tensor, Path]:
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"local-only Gemma base directory is missing: {base_model_path}")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("Gemma feature extraction requested CUDA but CUDA is unavailable")
    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("native hidden extraction requires the configured transformers runtime") from exc
    # Transformers model docs: from_pretrained supports a local directory,
    # `local_files_only=True` disables downloads, and `dtype`/`device_map`
    # control model placement: https://huggingface.co/docs/transformers/main_classes/model
    # Transformers model docs: from_pretrained supports a local directory,
    # `local_files_only=True` disables downloads, and `dtype`/`device_map`
    # control model placement: https://huggingface.co/docs/transformers/main_classes/model
    model = AutoModelForImageTextToText.from_pretrained(
        str(base_model_path),
        dtype=torch.bfloat16,
        device_map={"": device},
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(str(base_model_path), local_files_only=True)
    text_config = getattr(model.config, "text_config", None)
    tied_embeddings = getattr(text_config, "tie_word_embeddings", getattr(model.config, "tie_word_embeddings", False))
    if tied_embeddings is not True:
        raise RuntimeError("Gemma config no longer declares tied input/output embeddings")
    if (
        getattr(text_config, "hidden_size", None) != HIDDEN_WIDTH
        or getattr(text_config, "vocab_size", None) != 262144
    ):
        raise RuntimeError("loaded Gemma text config no longer matches the verified native output-row shape")
    nested = getattr(model, "model", None)
    language_model = getattr(nested, "language_model", None)
    if language_model is None:
        language_model = getattr(model, "language_model", None)
    if language_model is None or not hasattr(language_model, "embed_tokens"):
        raise RuntimeError("loaded Gemma model lacks the expected language_model.embed_tokens weight")
    resolved_base = str(base_model_path.resolve())
    for module in (model, nested, language_model):
        setattr(module, "_gemma4_checkpoint_path", resolved_base)
    tokenizer = getattr(processor, "tokenizer", processor)
    for token_id, token_text in TOKEN_ID_TO_STRING.items():
        if not callable(getattr(tokenizer, "convert_tokens_to_ids", None)):
            raise RuntimeError("Gemma processor tokenizer cannot verify native action token ids")
        if int(tokenizer.convert_tokens_to_ids(token_text)) != token_id:
            raise RuntimeError(f"tokenizer mapping drift for {token_text} (expected id {token_id})")
        if callable(getattr(tokenizer, "convert_ids_to_tokens", None)) and tokenizer.convert_ids_to_tokens(token_id) != token_text:
            raise RuntimeError(f"tokenizer reverse mapping drift at native action id {token_id}")
    token_embedding = language_model.embed_tokens
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or not isinstance(getattr(lm_head, "weight", None), torch.Tensor):
        raise RuntimeError("loaded Gemma model lacks the actual lm_head.weight output projection")
    if tuple(lm_head.weight.shape) != (262144, HIDDEN_WIDTH):
        raise RuntimeError(f"native Gemma lm_head shape changed: {tuple(lm_head.weight.shape)}")
    if lm_head.weight.data_ptr() != token_embedding.weight.data_ptr():
        raise RuntimeError("Gemma config is tied but lm_head.weight does not share the embedding storage")
    from duplex_kernel.models.thinker import ThinkerModel

    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("extractor requires the base Gemma model with response LoRA OFF")
    model.eval()
    thinker = ThinkerModel(model, tokenizer, processor, device=device)
    native_library_path = Path(os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH", ""))
    if not native_library_path.is_file():
        raise RuntimeError(
            "extraction requires the explicitly configured serving CUSTOM graph library at "
            "GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"
        )
    if os.environ.get("GEMMA4_NATIVE_LISTENER_CUDA_GRAPH", "0").strip() != "1":
        raise RuntimeError("extractor requires the serving CUDA-graph CUSTOM profile")
    from duplex_kernel.listener import GemmaListener
    from duplex_kernel.native_listener import (
        GEMMA4_ATTN_CUSTOM,
        GEMMA4_RMS_NATIVE_EXACT,
    )

    listener = GemmaListener.from_thinker(
        thinker,
        runtime_profile="production",
        use_native_lm=True,
        attention_backend="CUSTOM",
        rmsnorm_backend="NATIVE_EXACT",
    )
    backend = listener.backend
    if (
        getattr(listener, "runtime_profile", None) != "production"
        or not bool(getattr(backend, "_use_native_lm", False))
        or getattr(backend, "_native_attention_backend", None) != GEMMA4_ATTN_CUSTOM
        or getattr(backend, "_native_rmsnorm_backend", None) != GEMMA4_RMS_NATIVE_EXACT
        or int(backend.native_max_seq_len) != 4096
    ):
        raise RuntimeError("extractor native listener profile differs from serving CUSTOM graph/4096/NATIVE_EXACT")
    return model, listener, processor, lm_head.weight, native_library_path.resolve()


def _safe_conversation_filename(conversation_id: str) -> str:
    if not FAMILY_ID_RE.fullmatch(conversation_id):
        raise ValueError(f"unsafe conversation id for feature file: {conversation_id!r}")
    return f"{conversation_id}.pt"


def _write_feature_cache_metadata(
    output_dir: Path, *, base_model_path: Path, native_library_path: Path
) -> None:
    metadata = {
        "schema_version": 1,
        "contract": HIDDEN_CONTRACT,
        "model_id": MODEL_ID,
        "base_model_path": str(base_model_path.resolve()),
        "response_adapter": "OFF (base Gemma only)",
        "listener_profile": "production graph-enabled native LM; serving audio frontend/features",
        "native_cuda_graph_enabled": True,
        "native_library_path": str(native_library_path.resolve()),
        "native_attention_backend": "CUSTOM",
        "native_rmsnorm_backend": "NATIVE_EXACT",
        "native_max_seq_len": 4096,
        "native_rebase_retain_frames": 64,
        "hidden_source": HIDDEN_SOURCE,
        "normalization": HIDDEN_NORM,
        "hidden_width": HIDDEN_WIDTH,
        "tick_ms": TICK_MS,
        "audio_queries_per_tick": 4,
        "selected_query_position": 0,
        "per_frame_semantics": "newest production listener Gemma audio-query output after language_model.norm",
        "freshness_policy": "stale reused hidden retained with fresh_mask=false and excluded by trainer",
        "context_policy": "one GemmaListenerState per conversation; prime once; chronological user PCM and previous assistant track frame references; production bounded context/rebase",
        "audio_resampling": "existing stage2._audio_frames 16 kHz polyphase path",
        "qwen_loaded": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "hidden_cache_metadata.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "contract", "model_id", "hidden_width", "tick_ms", "selected_query_position",
            "audio_queries_per_tick", "per_frame_semantics", "freshness_policy", "context_policy",
            "qwen_loaded", "response_adapter", "base_model_path", "listener_profile",
            "native_library_path", "native_attention_backend", "native_rmsnorm_backend",
            "native_cuda_graph_enabled", "native_max_seq_len", "native_rebase_retain_frames",
        ):
            if prior.get(key) != metadata[key]:
                raise ValueError(f"existing hidden cache metadata differs at {key}")
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)


def extract_hidden_cache(
    manifest_path: Path,
    base_model_path: Path,
    output_dir: Path,
    *,
    splits: Sequence[str] = ("train", "validation"),
    device: str = "cuda",
    resume: bool = True,
) -> dict[str, Any]:
    from training.duplex_qlora.stage2 import _audio_frames
    from duplex_kernel.listener import DEFAULT_CONTEXT_FRAMES, GemmaListenerState
    from duplex_kernel.reference_buffer import AssistantReferenceFrame

    source_rows = _read_jsonl(manifest_path)
    _validate_source_group_split(source_rows)
    selected = [row for row in source_rows if row.get("split") in set(splits)]
    if not selected or set(splits) - {"train", "validation"}:
        raise ValueError("extractor accepts only nonempty train/validation synthetic splits")
    native_library_path = Path(os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH", ""))
    if not native_library_path.is_file():
        raise RuntimeError("missing serving CUSTOM graph library in GEMMA4_NATIVE_LISTENER_LIBRARY_PATH")
    _write_feature_cache_metadata(
        output_dir, base_model_path=base_model_path, native_library_path=native_library_path
    )
    conversation_dir = output_dir / "conversations"
    conversation_dir.mkdir(parents=True, exist_ok=True)

    model, listener, _processor, lm_head_weight, loaded_native_library = _load_base_thinker(base_model_path, device)
    backend = listener.backend
    if (
        str(loaded_native_library) != str(native_library_path.resolve())
        or int(backend.native_max_seq_len) != 4096
        or int(DEFAULT_CONTEXT_FRAMES) != 64
    ):
        raise RuntimeError("listener capacity/rebase contract differs from serving CUSTOM 4096/64")
    base_rows_path = output_dir / "native_control6_base_rows.pt"
    save_base_rows(base_rows_path, lm_head_weight)
    del lm_head_weight
    counts: dict[str, Any] = {}
    extraction_started = time.monotonic()
    extracted_frames = 0
    runtime_contract_checked = False
    total_expected_frames = 49402
    print(json.dumps({
        "event": "native_control6_extract_start",
        "listener_profile": "production_graph_native",
        "attention_backend": "CUSTOM",
        "rmsnorm_backend": "NATIVE_EXACT",
        "native_library_path": str(loaded_native_library),
        "native_max_seq_len": int(backend.native_max_seq_len),
        "rebase_retain_frames": int(DEFAULT_CONTEXT_FRAMES),
        "response_adapter": "OFF",
        "qwen_loaded": False,
    }), flush=True)
    try:
        for row in selected:
            conversation_id = str(row["conversation_id"])
            destination = conversation_dir / _safe_conversation_filename(conversation_id)
            if resume and destination.is_file():
                prior = torch.load(destination, map_location="cpu", weights_only=True)
                prior_hidden = prior.get("hidden_states") if isinstance(prior, dict) else None
                prior_frame_ids = prior.get("frame_ids") if isinstance(prior, dict) else None
                if (
                    isinstance(prior, dict)
                    and prior.get("conversation_id") == conversation_id
                    and prior.get("split") == row.get("split")
                    and prior.get("template_family_id") == row.get("template_family_id")
                    and prior.get("audio_sha256", "") == str(row.get("audio_sha256", ""))
                    and isinstance(prior_hidden, torch.Tensor)
                    and isinstance(prior_frame_ids, torch.Tensor)
                    and prior_hidden.ndim == 2
                    and tuple(prior_hidden.shape) == (int(prior_frame_ids.numel()), HIDDEN_WIDTH)
                    and int(prior_frame_ids.numel()) > 0
                ):
                    counts[conversation_id] = {"status": "reused", "frames": len(prior["frame_ids"])}
                    continue
                raise ValueError(f"resume cache is incompatible for {conversation_id}")

            replay_row = dict(row)
            replay_row["channel_map"] = _normalise_channel_map(row)
            user_frames, assistant_frames, valid_samples = _audio_frames(
                replay_row,
                frame_samples=16000 * TICK_MS // 1000,
            )
            state = GemmaListenerState(
                sample_rate=16000,
                frame_duration_ms=TICK_MS,
                context_frames=DEFAULT_CONTEXT_FRAMES,
            )
            if state.context_frames != 64:
                raise RuntimeError(f"{conversation_id}: listener context/rebase retention must be 64 frames")
            hidden_rows: list[torch.Tensor] = []
            frame_ids: list[int] = []
            source_frame_ids: list[int] = []
            assistant_reference_frame_ids: list[int] = []
            fresh_mask: list[bool] = []
            listener_tensor_contract_checked = False
            try:
                listener.prime_steady_state(state)
                with torch.inference_mode():
                    for frame_id in range(len(user_frames)):
                        reference = None
                        reference_id = -1
                        if assistant_frames is not None and frame_id > 0:
                            reference_id = frame_id - 1
                            reference = AssistantReferenceFrame(
                                frame_id=reference_id,
                                pcm=assistant_frames[frame_id - 1],
                                emitted_at_ns=reference_id * TICK_MS * 1_000_000,
                                sample_rate=16000,
                            )
                        output = listener.step(user_frames[frame_id], reference, state)
                        if not runtime_contract_checked:
                            native_runtime = state.gemma_lm_native
                            if native_runtime is None:
                                raise RuntimeError("native listener did not construct a CUSTOM LM at the first frame")
                            runtime_info = native_runtime.native_runtime_info()
                            runtime_expected = {
                                "library_path": str(loaded_native_library),
                                "attention_backend": "CUSTOM",
                                "attention_backend_id": 0,
                                "rmsnorm_backend": "NATIVE_EXACT",
                                "rmsnorm_backend_id": 2,
                                "cuda_graph_enabled": True,
                                "cuda_graph_stream_non_default": True,
                                "max_seq_len": 4096,
                            }
                            runtime_mismatches = {
                                key: (runtime_info.get(key), value)
                                for key, value in runtime_expected.items()
                                if runtime_info.get(key) != value
                            }
                            if runtime_mismatches:
                                raise RuntimeError(
                                    f"live native Gemma runtime differs from serving contract: {runtime_mismatches}"
                                )
                            runtime_contract_checked = True
                        if int(output.frame_id) != frame_id:
                            raise RuntimeError(
                                f"{conversation_id}: listener frame {output.frame_id} != input {frame_id}"
                            )
                        hidden = output.gemma_hidden_t
                        if not isinstance(hidden, torch.Tensor):
                            raise RuntimeError(f"{conversation_id}: listener did not return a tensor hidden")
                        queries = getattr(output, "gemma_audio_queries_t", None)
                        if not listener_tensor_contract_checked:
                            if not isinstance(queries, torch.Tensor) or tuple(queries.shape) != (1, 4, HIDDEN_WIDTH):
                                raise RuntimeError(
                                    f"{conversation_id}: expected newest tick's four Gemma LM outputs [1,4,1536]"
                                )
                            first_query = queries[0, 0, :]
                            candidate = hidden[0] if hidden.ndim == 2 and hidden.shape[0] == 1 else hidden
                            if tuple(candidate.shape) != (HIDDEN_WIDTH,) or not torch.equal(candidate, first_query):
                                raise RuntimeError(
                                    "ListenerOutput.gemma_hidden_t must equal newest four-query LM output at position 0"
                                )
                            listener_tensor_contract_checked = True
                        if hidden.ndim == 2 and tuple(hidden.shape) == (1, HIDDEN_WIDTH):
                            hidden = hidden[0]
                        if tuple(hidden.shape) != (HIDDEN_WIDTH,) or not torch.isfinite(hidden).all():
                            raise RuntimeError(
                                f"{conversation_id}: expected finite final-normalized [1536], got {tuple(hidden.shape)}"
                            )
                        is_fresh = bool(getattr(output, "fresh", True))
                        source_frame = getattr(output, "source_frame_id", frame_id)
                        if source_frame is None:
                            source_frame = frame_id
                        source_frame = int(source_frame)
                        if is_fresh and source_frame != frame_id:
                            raise RuntimeError(
                                f"{conversation_id}: fresh hidden source frame {source_frame} != {frame_id}"
                            )
                        # Copy the borrowed listener output before the next tick
                        # reuses it; transfer one stacked sequence per dialogue.
                        hidden_rows.append(hidden.detach().to(dtype=torch.bfloat16).clone())
                        frame_ids.append(frame_id)
                        source_frame_ids.append(source_frame)
                        assistant_reference_frame_ids.append(reference_id)
                        fresh_mask.append(is_fresh)
                        extracted_frames += 1
                        if extracted_frames == 100 or extracted_frames % 1000 == 0:
                            elapsed = max(time.monotonic() - extraction_started, 1e-6)
                            rate = extracted_frames / elapsed
                            print(json.dumps({
                                "event": "native_control6_extract_progress",
                                "frames": extracted_frames,
                                "expected_frames": total_expected_frames,
                                "elapsed_seconds": round(elapsed, 2),
                                "frames_per_second": round(rate, 3),
                                "estimated_total_seconds": round(total_expected_frames / max(rate, 1e-6), 1),
                                "runtime_contract_checked": runtime_contract_checked,
                            }), flush=True)
            finally:
                state.close()

            payload = {
                "conversation_id": conversation_id,
                "template_family_id": str(row["template_family_id"]),
                "split": str(row["split"]),
                "audio_sha256": str(row.get("audio_sha256", "")),
                "frame_ids": torch.tensor(frame_ids, dtype=torch.int64),
                "source_frame_ids": torch.tensor(source_frame_ids, dtype=torch.int64),
                "assistant_reference_frame_ids": torch.tensor(assistant_reference_frame_ids, dtype=torch.int64),
                "fresh_mask": torch.tensor(fresh_mask, dtype=torch.bool),
                "valid_samples": torch.as_tensor(valid_samples, dtype=torch.int32),
                "hidden_states": torch.stack(hidden_rows, dim=0).cpu(),
            }
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(destination)
            counts[conversation_id] = {
                "status": "extracted",
                "frames": len(frame_ids),
                "fresh_frames": sum(fresh_mask),
                "masked_stale_frames": len(fresh_mask) - sum(fresh_mask),
                "valid_samples": int(torch.as_tensor(valid_samples).sum().item()),
            }
    finally:
        close = getattr(listener, "close", None)
        if callable(close):
            close()
        del listener, model
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    report = {
        "manifest_path": str(manifest_path),
        "cache_dir": str(output_dir),
        "base_rows_path": str(base_rows_path),
        "response_adapter": "OFF",
        "model_id": MODEL_ID,
        "contract": HIDDEN_CONTRACT,
        "conversations": counts,
        "frames_total": sum(int(row["frames"]) for row in counts.values()),
        "status": "complete",
    }
    report_path = output_dir / "extract_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _read_target_rows(path: Path, expected_split: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in _read_jsonl(path):
        if row.get("schema") != TARGET_SCHEMA or row.get("schema_version") != TARGET_SCHEMA_VERSION:
            raise ValueError(f"{path}: wrong action-target schema")
        if row.get("split") != expected_split:
            raise ValueError(f"{path}: target split {row.get('split')!r} != {expected_split!r}")
        conversation_id = row.get("conversation_id")
        frame_id = row.get("frame_id")
        if not isinstance(conversation_id, str) or not isinstance(frame_id, int) or frame_id < 0:
            raise ValueError(f"{path}: invalid conversation/frame id")
        key = (conversation_id, frame_id)
        if key in seen:
            raise ValueError(f"{path}: duplicate target frame {key}")
        seen.add(key)
        if row.get("action_mask"):
            if row.get("target_action") not in ACTIONS:
                raise ValueError(f"{path}: masked target has invalid action {row.get('target_action')!r}")
        elif row.get("target_action") is not None:
            raise ValueError(f"{path}: unknown/masked target must not carry an action")
        grouped[conversation_id].append(row)
    for conversation_id, rows in grouped.items():
        rows.sort(key=lambda row: row["frame_id"])
        family_values = {row.get("template_family_id") for row in rows}
        if len(family_values) != 1:
            raise ValueError(f"{conversation_id}: multiple template family ids")
    return dict(grouped)


def _hidden_cache_metadata(cache_dir: Path) -> dict[str, Any]:
    metadata_path = cache_dir / "hidden_cache_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"native hidden cache metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "contract": HIDDEN_CONTRACT,
        "model_id": MODEL_ID,
        "hidden_source": HIDDEN_SOURCE,
        "normalization": HIDDEN_NORM,
        "hidden_width": HIDDEN_WIDTH,
        "tick_ms": TICK_MS,
        "selected_query_position": 0,
        "audio_queries_per_tick": 4,
        "per_frame_semantics": "newest production listener Gemma audio-query output after language_model.norm",
        "freshness_policy": "stale reused hidden retained with fresh_mask=false and excluded by trainer",
        "context_policy": "one GemmaListenerState per conversation; prime once; chronological user PCM and previous assistant track frame references; production bounded context/rebase",
        "qwen_loaded": False,
        "response_adapter": "OFF (base Gemma only)",
        "listener_profile": "production graph-enabled native LM; serving audio frontend/features",
        "native_cuda_graph_enabled": True,
        "native_attention_backend": "CUSTOM",
        "native_rmsnorm_backend": "NATIVE_EXACT",
        "native_max_seq_len": 4096,
        "native_rebase_retain_frames": 64,
    }
    mismatches = {key: (metadata.get(key), value) for key, value in expected.items() if metadata.get(key) != value}
    if not isinstance(metadata.get("native_library_path"), str) or not metadata["native_library_path"]:
        mismatches["native_library_path"] = (metadata.get("native_library_path"), "nonempty path")
    if mismatches:
        raise ValueError(f"hidden cache does not match native final-norm listener contract: {mismatches}")
    return metadata


def _load_conversation_feature(cache_dir: Path, conversation_id: str) -> dict[str, Any]:
    path = cache_dir / "conversations" / _safe_conversation_filename(conversation_id)
    if not path.is_file():
        raise FileNotFoundError(f"no native final-hidden cache for conversation {conversation_id}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("conversation_id") != conversation_id:
        raise ValueError(f"invalid hidden cache record for {conversation_id}")
    frame_ids = payload.get("frame_ids")
    source_frame_ids = payload.get("source_frame_ids")
    assistant_reference_frame_ids = payload.get("assistant_reference_frame_ids")
    fresh_mask = payload.get("fresh_mask")
    hidden = payload.get("hidden_states")
    if not all(isinstance(v, torch.Tensor) for v in (frame_ids, source_frame_ids, assistant_reference_frame_ids, fresh_mask, hidden)):
        raise ValueError(f"{conversation_id}: feature cache fields must be tensors")
    n = int(frame_ids.numel())
    if (
        frame_ids.ndim != 1
        or source_frame_ids.shape != frame_ids.shape
        or assistant_reference_frame_ids.shape != frame_ids.shape
        or fresh_mask.shape != frame_ids.shape
    ):
        raise ValueError(f"{conversation_id}: cache frame metadata shape mismatch")
    if hidden.ndim != 2 or tuple(hidden.shape) != (n, HIDDEN_WIDTH):
        raise ValueError(f"{conversation_id}: hidden states must have shape [{n}, {HIDDEN_WIDTH}]")
    if n == 0 or not hidden.is_floating_point() or not torch.isfinite(hidden).all():
        raise ValueError(f"{conversation_id}: hidden states are empty, non-floating, or non-finite")
    frame_list = frame_ids.to(torch.int64).tolist()
    if frame_list != list(range(n)):
        raise ValueError(f"{conversation_id}: hidden frames must cover the chronological input from frame zero")
    fresh_bad = fresh_mask.bool() & (source_frame_ids.to(torch.int64) != frame_ids.to(torch.int64))
    if bool(fresh_bad.any()):
        raise ValueError(f"{conversation_id}: fresh features have stale source-frame ids")
    expected_references = torch.arange(n, dtype=torch.int64) - 1
    if not torch.equal(assistant_reference_frame_ids.to(torch.int64), expected_references):
        raise ValueError(f"{conversation_id}: audio reference stream must be absent at frame 0 and previous-frame-only afterwards")
    return payload


def load_sequences(
    target_path: Path,
    cache_dir: Path,
    expected_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = _read_target_rows(target_path, expected_split)
    cache_metadata = _hidden_cache_metadata(cache_dir)
    sequences: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    masked_stale = 0
    feature_rows = 0
    for conversation_id, target_rows in labels.items():
        feature = _load_conversation_feature(cache_dir, conversation_id)
        if feature.get("split") != expected_split:
            raise ValueError(f"{conversation_id}: hidden cache split mismatch")
        if feature.get("template_family_id") != target_rows[0].get("template_family_id"):
            raise ValueError(f"{conversation_id}: feature/label template group mismatch")
        frame_to_index = {
            int(frame): idx for idx, frame in enumerate(feature["frame_ids"].tolist())
        }
        hidden_parts: list[torch.Tensor] = []
        target_ids: list[int] = []
        loss_masks: list[bool] = []
        ordered_ids: list[int] = []
        for row in target_rows:
            frame_id = int(row["frame_id"])
            idx = frame_to_index.get(frame_id)
            if idx is None:
                if bool(row.get("action_mask")):
                    raise ValueError(f"{conversation_id}: no Gemma hidden for labeled target frame {frame_id}")
                hidden_parts.append(torch.zeros(HIDDEN_WIDTH, dtype=torch.float32))
                is_fresh = False
            else:
                hidden_parts.append(feature["hidden_states"][idx].to(dtype=torch.float32))
                is_fresh = bool(feature["fresh_mask"][idx].item())
            ordered_ids.append(frame_id)
            action = row.get("target_action")
            mask = bool(row.get("action_mask")) and is_fresh
            if bool(row.get("action_mask")) and not is_fresh:
                masked_stale += 1
            target_ids.append(ACTIONS.index(action) if mask else -100)
            loss_masks.append(mask)
            if mask:
                class_counts[str(action)] += 1
        if ordered_ids != sorted(ordered_ids):
            raise ValueError(f"{conversation_id}: labels are not chronological")
        sequences.append({
            "conversation_id": conversation_id,
            "template_family_id": target_rows[0]["template_family_id"],
            "frame_ids": torch.tensor(ordered_ids, dtype=torch.int64),
            "hidden_states": torch.stack(hidden_parts),
            "target_ids": torch.tensor(target_ids, dtype=torch.int64),
            "loss_mask": torch.tensor(loss_masks, dtype=torch.bool),
        })
        feature_rows += len(hidden_parts)
    if not sequences or not any(bool(sequence["loss_mask"].any()) for sequence in sequences):
        raise ValueError(f"{expected_split}: no fresh, explicitly labeled examples remain")
    return sequences, {
        "split": expected_split,
        "conversations": len(sequences),
        "template_families": len({seq["template_family_id"] for seq in sequences}),
        "feature_rows_aligned": feature_rows,
        "masked_stale_labeled_frames": masked_stale,
        "action_counts": {action: int(class_counts[action]) for action in ACTIONS},
        "cache_metadata": cache_metadata,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_stage2_feature_sequences(
    feature_dir: Path,
    expected_split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a Stage2 1540 feature split while retaining source provenance."""
    if expected_split not in {"train", "validation"}:
        raise ValueError("native1540 training accepts train and validation only")
    manifest_path = feature_dir / "feature_manifest.json"
    npz_path = feature_dir / f"{expected_split}.npz"
    metadata_path = feature_dir / f"{expected_split}.json"
    for required in (manifest_path, npz_path, metadata_path):
        if not required.is_file():
            raise FileNotFoundError(f"native1540 Stage2 input is missing: {required}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("phase_schema") != NATIVE1540_PHASE_SCHEMA:
        raise ValueError("Stage2 feature manifest does not declare the exact native1540 phase schema")
    if manifest.get("feature_contract") != native_control6_feature_contract(HIDDEN_WIDTH + PHASE_T_WIDTH):
        raise ValueError("Stage2 feature manifest is not 1540-wide")
    native_extract = manifest.get("native_control6_extraction")
    if not isinstance(native_extract, Mapping) or (
        native_extract.get("hidden_source") != HIDDEN_SOURCE
        or native_extract.get("gemma_audio_queries_per_tick") != 2
        or native_extract.get("audio_tower_queries_per_tick") != 2
        or native_extract.get("mel_window_rows_effective") != 240
    ):
        raise ValueError("native1540 features lack the runtime Q2/W240 hidden extraction contract")
    serving_loader = manifest.get("serving_base_loader")
    serving_worker = native_extract.get("serving_worker")
    runtime_info = serving_worker.get("runtime_info") if isinstance(serving_worker, Mapping) else None
    runtime_signature = serving_worker.get("runtime_signature") if isinstance(serving_worker, Mapping) else None
    source_root = (
        Path(str(serving_worker.get("runtime_source_root", ""))).resolve()
        if isinstance(serving_worker, Mapping) and serving_worker.get("runtime_source_root")
        else None
    )
    prefix_builder_source = (
        Path(str(serving_worker.get("prefix_builder_source", ""))).resolve()
        if isinstance(serving_worker, Mapping) and serving_worker.get("prefix_builder_source")
        else None
    )
    worker_factory_source = (
        Path(str(serving_worker.get("worker_factory_source", ""))).resolve()
        if isinstance(serving_worker, Mapping) and serving_worker.get("worker_factory_source")
        else None
    )
    runtime_workspace = (
        Path(str(serving_loader.get("workspace", ""))).resolve()
        if isinstance(serving_loader, Mapping) and serving_loader.get("workspace")
        else None
    )
    worker_library = serving_worker.get("native_library_path") if isinstance(serving_worker, Mapping) else None
    runtime_library = runtime_info.get("library_path") if isinstance(runtime_info, Mapping) else None
    prefix_source_attested = (
        source_root is not None
        and prefix_builder_source is not None
        and prefix_builder_source.is_relative_to(source_root)
    )
    worker_source_attested = (
        worker_factory_source is not None
        and (
            source_root is not None and worker_factory_source.is_relative_to(source_root)
            or runtime_workspace is not None
            and worker_factory_source.is_relative_to(runtime_workspace / "duplex_kernel")
        )
    )
    serving_base_only = (
        manifest.get("serving_base_only") is True
        and manifest.get("merged_before_native_listener") is False
        and manifest.get("adapter_dir") is None
        and isinstance(serving_loader, Mapping)
        and serving_loader.get("loader") == "duplex_kernel.models.thinker.load_thinker"
        and serving_loader.get("model_dtype") == "bfloat16"
        and serving_loader.get("response_lora_applied") is False
        and serving_loader.get("stage1_adapter_applied") is False
        and isinstance(serving_worker, Mapping)
        and serving_worker.get("worker") == "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker"
        and serving_worker.get("prefix_builder") == "duplex_kernel.server._build_unified_listener_prefix_token_ids"
        and isinstance(runtime_info, Mapping)
        and int(runtime_info.get("stream_q_len", -1)) == 2
        and isinstance(runtime_signature, Mapping)
        and int(runtime_signature.get("stream_q_len", -1)) == 2
        and prefix_source_attested
        and worker_source_attested
        and worker_library is not None
        and runtime_library is not None
        and Path(str(worker_library)).resolve() == Path(str(runtime_library)).resolve()
        and isinstance(serving_worker.get("prefix"), Mapping)
        and serving_worker["prefix"].get("contract") == "serving_unified_chat_prefix_v2"
        and int(serving_worker["prefix"].get("length", 0)) > 0
    )
    if not serving_base_only:
        raise ValueError(
            "native1540 training requires serving base-only BF16 weights, the serving shared Q2 worker, "
            "and the exact unified-prefix builder; merged/adapter features are not accepted"
        )
    prefix = serving_worker.get("prefix")
    prefix_token_ids = prefix.get("token_ids") if isinstance(prefix, Mapping) else None
    if (
        not isinstance(prefix, Mapping)
        or prefix.get("contract") != "serving_unified_chat_prefix_v2"
        or not isinstance(prefix_token_ids, list)
        or any(isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in prefix_token_ids)
        or not prefix_token_ids
        or int(prefix.get("length", -1)) != len(prefix_token_ids)
        or len(prefix_token_ids) > NATIVE_CONTROL6_PREFIX_MAX_TOKENS
        or prefix_token_ids[-1] != 256000
        or prefix.get("history_messages") != 0
        or prefix.get("session_instructions") != "empty_session_default"
    ):
        raise ValueError(
            "native1540 training requires the actual serving unified-prefix token IDs from the "
            "empty-history/default-instructions extraction"
        )
    semantic_prefix_metadata = {
        "semantic_prefix_schema_version": NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION,
        "semantic_prefix_contract": NATIVE_CONTROL6_PREFIX_CONTRACT,
        "semantic_prefix_context_conditioned": True,
        # Capacity/max_tokens record the prefix length seen in this extraction;
        # the canonical prime and runtime import buffer have separate fields.
        "semantic_prefix_capacity": len(prefix_token_ids),
        "semantic_prefix_max_tokens": len(prefix_token_ids),
        "semantic_prefix_boa_token_id": prefix_token_ids[-1],
        "semantic_prefix_actual_prefix_tokens": len(prefix_token_ids),
        "semantic_prefix_history_messages": 0,
        "semantic_prefix_session_instructions": "empty_session_default",
        "semantic_prefix_runtime_import_capacity_tokens": NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
        "semantic_prefix_canonical_prime_target_tokens": 291,
    }
    if manifest.get("test_data_read") is True or manifest.get("test_used_for_training_or_calibration") is True:
        raise ValueError("native1540 training refuses a feature manifest that accessed held-out test data")

    with np.load(npz_path, allow_pickle=False) as values:
        features = np.asarray(values["features"], dtype=np.float32)
        labels = np.asarray(values["labels"], dtype=np.int64)
        frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
        listener_ids = np.asarray(values["listener_frame_ids"], dtype=np.int64)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or any(not isinstance(row, dict) for row in metadata):
        raise ValueError(f"{metadata_path}: expected a JSON array of feature metadata objects")
    manifest_state_context = manifest.get("training_state_context")
    row_state_contexts = sorted({
        value if isinstance(value, str) and value else "unspecified"
        for value in (row.get("training_state_context") for row in metadata)
    })
    training_state_context = (
        manifest_state_context
        if isinstance(manifest_state_context, str) and manifest_state_context
        else row_state_contexts[0] if len(row_state_contexts) == 1 else "unspecified"
    )
    n = len(features)
    if (
        features.ndim != 2
        or features.shape[1] != HIDDEN_WIDTH + PHASE_T_WIDTH
        or labels.shape != (n,)
        or frame_ids.shape != (n,)
        or listener_ids.shape != (n,)
        or len(metadata) != n
    ):
        raise ValueError(f"{expected_split}: native1540 feature arrays and metadata are not aligned")
    if not np.isfinite(features).all():
        raise ValueError(f"{expected_split}: native1540 feature vectors contain NaN/Inf")

    groups: dict[str, dict[str, Any]] = {}
    seen_frames: set[tuple[str, int]] = set()
    action_counts: Counter[str] = Counter()
    runtime_target_action_counts: Counter[str] = Counter()
    runtime_derived_target_counts: Counter[str] = Counter()
    runtime_bucket_counts: Counter[str] = Counter()
    context_unsupported_targets: Counter[str] = Counter()
    source_rows: Counter[str] = Counter()
    source_supervised: Counter[str] = Counter()
    masked_reasons: Counter[str] = Counter()
    source_group_ids: set[str] = set()
    audio_sha256s: set[str] = set()
    for index, row in enumerate(metadata):
        conversation_id = row.get("conversation_id")
        source_group_id = row.get("source_group_id")
        frame_id = row.get("frame_id")
        listener_frame_id = row.get("listener_frame_id")
        if not isinstance(conversation_id, str) or not conversation_id:
            raise ValueError(f"{expected_split}:{index}: missing conversation_id")
        if not isinstance(source_group_id, str) or not source_group_id:
            raise ValueError(f"{conversation_id}:{index}: missing source_group_id provenance")
        if (
            isinstance(frame_id, bool)
            or not isinstance(frame_id, int)
            or frame_id < 0
            or int(frame_ids[index]) != frame_id
            or isinstance(listener_frame_id, bool)
            or not isinstance(listener_frame_id, int)
            or int(listener_ids[index]) != listener_frame_id
        ):
            raise ValueError(f"{conversation_id}:{index}: frame IDs disagree with Stage2 feature arrays")
        frame_key = (conversation_id, frame_id)
        if frame_key in seen_frames:
            raise ValueError(f"{expected_split}: duplicate conversation/frame metadata {frame_key}")
        seen_frames.add(frame_key)
        source = str(row.get("source", "unknown"))
        source_rows[source] += 1
        action_mask = row.get("action_mask")
        action = row.get("action")
        speaker_active = row.get("speaker_active_at_frame_start")
        user_activity_detected = row.get("native_control6_user_activity_detected")
        context_valid = row.get("native_control6_context_valid", type(speaker_active) is bool)
        if (
            type(context_valid) is not bool
            or (context_valid and type(speaker_active) is not bool)
            or type(user_activity_detected) is not bool
        ):
            raise ValueError(
                f"{conversation_id}:{frame_id}: native1540 rows require source-derived runtime context flags"
            )
        target = int(labels[index])
        if type(action_mask) is not bool:
            raise ValueError(f"{conversation_id}:{frame_id}: action_mask must be boolean")
        if action_mask:
            if action not in ACTIONS or target != ACTIONS.index(action):
                raise ValueError(
                    f"{conversation_id}:{frame_id}: Stage2 action and label array disagree"
                )
            action_counts[str(action)] += 1
            source_supervised[source] += 1
            if context_valid:
                runtime_action = row.get("native_control6_runtime_target_action", action)
                if runtime_action not in ACTIONS:
                    raise ValueError(
                        f"{conversation_id}:{frame_id}: invalid runtime target {runtime_action!r}"
                    )
                runtime_target_action_counts[str(runtime_action)] += 1
                if row.get("native_control6_runtime_target_derived") is True:
                    runtime_derived_target_counts[str(row.get("native_control6_runtime_target_reason"))] += 1
                members = native_control6_effective_members(
                    speaker_active=speaker_active,
                    user_activity_detected=user_activity_detected,
                )
                if tuple(members) != RUNTIME_BUCKETS:
                    raise ValueError("runtime effective member mapping changed bucket order or names")
                matched_buckets = [bucket for bucket in RUNTIME_BUCKETS if runtime_action in members[bucket]]
                if len(matched_buckets) > 1:
                    raise ValueError(f"runtime effective mapping duplicates raw action {runtime_action} in {conversation_id}:{frame_id}")
                if matched_buckets:
                    runtime_bucket_counts[matched_buckets[0]] += 1
                else:
                    context_unsupported_targets[f"{source}:{runtime_action}"] += 1
            else:
                context_unsupported_targets[f"{source}:{action}:pre_action_context_unknown"] += 1
        elif target != -100:
            raise ValueError(f"{conversation_id}:{frame_id}: masked Stage2 row must use ignore label -100")
        elif row.get("action_masked_reason"):
            masked_reasons[str(row["action_masked_reason"])] += 1
        declared_split = row.get("split")
        if declared_split is not None and declared_split != expected_split:
            raise ValueError(
                f"{conversation_id}:{frame_id}: metadata split {declared_split!r} != {expected_split!r}"
            )
        group = groups.setdefault(conversation_id, {
            "source_group_id": source_group_id,
            "source": source,
            "indices": [],
        })
        if group["source_group_id"] != source_group_id or group["source"] != source:
            raise ValueError(f"{conversation_id}: mixed source provenance within one sequence")
        group["indices"].append(index)
        source_group_ids.add(source_group_id)
        audio_sha256 = row.get("audio_sha256")
        if isinstance(audio_sha256, str) and audio_sha256:
            audio_sha256s.add(audio_sha256)

    sequences: list[dict[str, Any]] = []
    for conversation_id, group in groups.items():
        indices = sorted(group["indices"], key=lambda item: int(frame_ids[item]))
        target_ids = labels[indices].copy()
        runtime_target_ids = np.asarray([
            ACTIONS.index(
                metadata[item].get("native_control6_runtime_target_action", metadata[item].get("action"))
            )
            if int(target_ids[position]) >= 0
            and metadata[item].get("native_control6_context_valid", True)
            else -100
            for position, item in enumerate(indices)
        ], dtype=np.int64)
        sequences.append({
            "conversation_id": conversation_id,
            "template_family_id": group["source_group_id"],
            "source_group_id": group["source_group_id"],
            "source": group["source"],
            "frame_ids": torch.as_tensor(frame_ids[indices].copy(), dtype=torch.int64),
            "hidden_states": torch.as_tensor(features[indices].copy(), dtype=torch.float32),
            "target_ids": torch.as_tensor(target_ids, dtype=torch.int64),
            "runtime_target_ids": torch.as_tensor(runtime_target_ids, dtype=torch.int64),
            "loss_mask": torch.as_tensor(target_ids >= 0, dtype=torch.bool),
            "speaker_active": torch.as_tensor(
                [bool(metadata[item].get("speaker_active_at_frame_start")) for item in indices],
                dtype=torch.bool,
            ),
            "user_activity_detected": torch.as_tensor(
                [bool(metadata[item]["native_control6_user_activity_detected"]) for item in indices],
                dtype=torch.bool,
            ),
        })
    if not sequences or not any(bool(sequence["loss_mask"].any()) for sequence in sequences):
        raise ValueError(f"{expected_split}: no explicitly labeled native1540 rows remain")
    return sequences, {
        "split": expected_split,
        "training_state_context": training_state_context,
        "training_state_context_row_values": row_state_contexts or ["unspecified"],
        "conversations": len(sequences),
        "source_groups": len(source_group_ids),
        "source_group_ids": sorted(source_group_ids),
        "audio_source_sha256_count": len(audio_sha256s),
        "audio_source_sha256": sorted(audio_sha256s),
        "feature_rows_aligned": n,
        "source_rows": dict(sorted(source_rows.items())),
        "source_supervised_rows": dict(sorted(source_supervised.items())),
        "masked_reasons": dict(sorted(masked_reasons.items())),
        "action_counts": {action: int(action_counts[action]) for action in ACTIONS},
        "runtime_target_action_counts": {action: int(runtime_target_action_counts[action]) for action in ACTIONS},
        "runtime_derived_target_counts": dict(sorted(runtime_derived_target_counts.items())),
        "runtime_bucket_counts": {bucket: int(runtime_bucket_counts[bucket]) for bucket in RUNTIME_BUCKETS},
        "context_unsupported_targets_by_source_action": dict(sorted(context_unsupported_targets.items())),
        "context_supervision_rows": int(sum(action_counts.values())),
        "semantic_prefix_metadata": semantic_prefix_metadata,
        "feature_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "feature_npz_sha256": _file_sha256(npz_path),
        "feature_metadata_sha256": _file_sha256(metadata_path),
    }


def assert_disjoint_source_groups(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]]) -> None:
    train_groups = {str(row["template_family_id"]) for row in train}
    validation_groups = {str(row["template_family_id"]) for row in validation}
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(f"source template families leak across train/validation: {sorted(overlap)}")


def _sequence_windows(
    sequences: Sequence[Mapping[str, Any]],
    *,
    batch_frames: int,
    generator: random.Random | None,
    target_key: str = "target_ids",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    order = list(range(len(sequences)))
    if generator is not None:
        generator.shuffle(order)
    windows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index in order:
        sequence = sequences[index]
        n = int(sequence["frame_ids"].numel())
        for start in range(0, n, batch_frames):
            end = min(n, start + batch_frames)
            mask = sequence["loss_mask"][start:end]
            if bool(mask.any()):
                windows.append((
                    sequence["hidden_states"][start:end],
                    sequence[target_key][start:end],
                    mask,
                    sequence["speaker_active"][start:end],
                    sequence["user_activity_detected"][start:end],
                ))
    return windows


def _forward_features(model: NativeControl6LoRA, features: torch.Tensor) -> torch.Tensor:
    if features.shape[-1] == HIDDEN_WIDTH:
        return model(features)
    if features.shape[-1] == HIDDEN_WIDTH + PHASE_T_WIDTH:
        return model(features[..., :HIDDEN_WIDTH], phase_t=features[..., HIDDEN_WIDTH:])
    raise ValueError(f"native Control6 features must be 1536 or 1540 wide, got {tuple(features.shape)}")


def _effective_action_names(replan_target_alias_to_yield: bool) -> tuple[str, ...]:
    return PHYSICAL_ACTIONS if replan_target_alias_to_yield else ACTIONS


def _effective_logits(logits: torch.Tensor, replan_target_alias_to_yield: bool) -> torch.Tensor:
    if not replan_target_alias_to_yield:
        return logits
    # Runtime pools probability mass from the two rows into one physical YIELD
    # action. logsumexp is the exact logit-space equivalent before softmax.
    yield_index = ACTIONS.index("YIELD")
    replan_index = ACTIONS.index("REPLAN")
    return torch.stack((
        logits[..., ACTIONS.index("LISTEN")],
        logits[..., ACTIONS.index("SPEAK")],
        logits[..., ACTIONS.index("CONTINUE")],
        torch.logsumexp(logits[..., (yield_index, replan_index)], dim=-1),
        logits[..., ACTIONS.index("BACKCHANNEL")],
    ), dim=-1)


def _effective_targets(target_ids: torch.Tensor, replan_target_alias_to_yield: bool) -> torch.Tensor:
    if not replan_target_alias_to_yield:
        return target_ids
    result = target_ids.clone()
    replan_index = ACTIONS.index("REPLAN")
    yield_index = PHYSICAL_ACTIONS.index("YIELD")
    result[result == replan_index] = yield_index
    valid = result >= 0
    if bool(valid.any()) and int(result[valid].max()) >= len(PHYSICAL_ACTIONS):
        raise ValueError("native physical target mapping produced an unknown action index")
    return result


def _runtime_effective_target_ids(
    target_ids: torch.Tensor,
    speaker_active: torch.Tensor,
    user_activity_detected: torch.Tensor,
) -> torch.Tensor:
    """Project each raw action label into the shared runtime bucket, or ignore it.

    A raw action that runtime excludes in the current state is deliberately
    left unlabelled; the data contract does not invent a different action for
    an incompatible target.
    """
    targets = target_ids.reshape(-1)
    speaker = speaker_active.reshape(-1).to(device=targets.device, dtype=torch.bool)
    user = user_activity_detected.reshape(-1).to(device=targets.device, dtype=torch.bool)
    if speaker.shape != targets.shape or user.shape != targets.shape:
        raise ValueError("runtime action context vectors must match the raw target shape")
    result = torch.full_like(targets, -100)
    for speaker_value in (False, True):
        for user_value in (False, True):
            rows = (speaker == speaker_value) & (user == user_value)
            if not bool(rows.any()):
                continue
            members = native_control6_effective_members(
                speaker_active=speaker_value,
                user_activity_detected=user_value,
            )
            if tuple(members) != RUNTIME_BUCKETS:
                raise ValueError("runtime effective member mapping changed bucket order or names")
            raw_to_bucket: dict[str, int] = {}
            for bucket_index, bucket in enumerate(RUNTIME_BUCKETS):
                for action in members[bucket]:
                    if action in raw_to_bucket:
                        raise ValueError(f"raw action {action} appears in more than one runtime bucket")
                    raw_to_bucket[action] = bucket_index
            for raw_index, action in enumerate(ACTIONS):
                if action in raw_to_bucket:
                    result[rows & (targets == raw_index)] = raw_to_bucket[action]
    return result.reshape(target_ids.shape)


def _hold_raw_aux_eligible_mask(
    runtime_target_ids: torch.Tensor,
    supervised_mask: torch.Tensor,
    speaker_active: torch.Tensor,
    user_activity_detected: torch.Tensor,
) -> torch.Tensor:
    """Select supervised raw LISTEN/CONTINUE rows physically mapped to HOLD."""
    targets = runtime_target_ids.reshape(-1)
    supervised = supervised_mask.reshape(-1).to(device=targets.device, dtype=torch.bool)
    speaker = speaker_active.reshape(-1).to(device=targets.device, dtype=torch.bool)
    user = user_activity_detected.reshape(-1).to(device=targets.device, dtype=torch.bool)
    if any(values.shape != targets.shape for values in (supervised, speaker, user)):
        raise ValueError("hold raw auxiliary labels, masks, and context vectors must match")
    effective_targets = _runtime_effective_target_ids(targets, speaker, user)
    hold_index = RUNTIME_BUCKETS.index("HOLD")
    raw_hold_actions = (targets == ACTIONS.index("LISTEN")) | (
        targets == ACTIONS.index("CONTINUE")
    )
    eligible = supervised & raw_hold_actions & (effective_targets == hold_index)
    return eligible.reshape(runtime_target_ids.shape)


def _hold_raw_aux_loss(
    raw_logits: torch.Tensor,
    runtime_target_ids: torch.Tensor,
    supervised_mask: torch.Tensor,
    speaker_active: torch.Tensor,
    user_activity_detected: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean six-way CE on raw LISTEN/CONTINUE rows pooled into HOLD.

    With no eligible rows, the empty reduction is a finite differentiable zero.
    """
    if raw_logits.shape[-1] != len(ACTIONS):
        raise ValueError("hold raw auxiliary loss requires all six raw action logits")
    flat_logits = raw_logits.reshape(-1, len(ACTIONS))
    targets = runtime_target_ids.reshape(-1).to(device=raw_logits.device, dtype=torch.long)
    if targets.shape[0] != flat_logits.shape[0]:
        raise ValueError("hold raw auxiliary targets must match logits rows")
    eligible = _hold_raw_aux_eligible_mask(
        targets,
        supervised_mask,
        speaker_active,
        user_activity_detected,
    ).reshape(-1)
    if not bool(eligible.any()):
        return flat_logits.reshape(-1)[:0].sum(), eligible.reshape(runtime_target_ids.shape)
    return (
        F.cross_entropy(flat_logits[eligible], targets[eligible], reduction="mean"),
        eligible.reshape(runtime_target_ids.shape),
    )


def _runtime_effective_log_probs(
    logits: torch.Tensor,
    speaker_active: torch.Tensor,
    user_activity_detected: torch.Tensor,
) -> torch.Tensor:
    """Return absolute full-sixway log probability mass per runtime bucket.

    Empty buckets remain ``-inf`` and raw rows excluded by the context keep
    their share in the six-way softmax denominator. This matches the policy's
    confidence mass and avoids renormalizing over only currently legal rows.
    """
    if logits.shape[-1] != len(ACTIONS):
        raise ValueError("runtime effective loss requires all six raw action logits")
    flat_logits = logits.reshape(-1, len(ACTIONS))
    speaker = speaker_active.reshape(-1).to(device=logits.device, dtype=torch.bool)
    user = user_activity_detected.reshape(-1).to(device=logits.device, dtype=torch.bool)
    if speaker.shape[0] != flat_logits.shape[0] or user.shape[0] != flat_logits.shape[0]:
        raise ValueError("runtime action context vectors must match the logits batch")
    raw_log_probs = F.log_softmax(flat_logits, dim=-1)
    result = torch.full(
        (flat_logits.shape[0], len(RUNTIME_BUCKETS)),
        float("-inf"),
        dtype=raw_log_probs.dtype,
        device=raw_log_probs.device,
    )
    for speaker_value in (False, True):
        for user_value in (False, True):
            rows = (speaker == speaker_value) & (user == user_value)
            if not bool(rows.any()):
                continue
            members = native_control6_effective_members(
                speaker_active=speaker_value,
                user_activity_detected=user_value,
            )
            if tuple(members) != RUNTIME_BUCKETS:
                raise ValueError("runtime effective member mapping changed bucket order or names")
            context_log_probs = raw_log_probs[rows]
            for bucket_index, bucket in enumerate(RUNTIME_BUCKETS):
                member_indices = [ACTIONS.index(action) for action in members[bucket]]
                if member_indices:
                    result[rows, bucket_index] = torch.logsumexp(
                        context_log_probs[:, member_indices], dim=-1
                    )
    return result.reshape(*logits.shape[:-1], len(RUNTIME_BUCKETS))


def _unweighted_runtime_bucket_nll(
    bucket_log_probabilities: torch.Tensor,
    target_bucket_ids: torch.Tensor,
) -> torch.Tensor:
    if bucket_log_probabilities.ndim != 2 or bucket_log_probabilities.shape[1] != len(RUNTIME_BUCKETS):
        raise ValueError("runtime bucket log probabilities must have shape [batch, 4]")
    targets = target_bucket_ids.reshape(-1).to(
        device=bucket_log_probabilities.device, dtype=torch.long
    )
    if targets.shape[0] != bucket_log_probabilities.shape[0]:
        raise ValueError("runtime bucket targets must match the batch size")
    if targets.numel() == 0 or bool(((targets < 0) | (targets >= len(RUNTIME_BUCKETS))).any()):
        raise ValueError("runtime bucket targets must be non-empty valid bucket IDs")
    return -bucket_log_probabilities.gather(1, targets[:, None]).mean()


def _runtime_bucket_source_pools(
    sequences: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[tuple[torch.Tensor, int, bool, bool]]]]:
    pools = {
        bucket: {"real": [], "synthetic": []}
        for bucket in RUNTIME_BUCKETS
    }
    for sequence in sequences:
        features = sequence["hidden_states"]
        if features.ndim != 2 or features.shape[1] != HIDDEN_WIDTH + PHASE_T_WIDTH:
            raise ValueError("balanced runtime sampling requires 1540-wide native features")
        runtime_targets = sequence.get("runtime_target_ids", sequence["target_ids"])
        speaker = sequence["speaker_active"]
        user = sequence["user_activity_detected"]
        loss_mask = sequence["loss_mask"]
        bucket_targets = _runtime_effective_target_ids(runtime_targets, speaker, user)
        source_name = str(sequence.get("source", "unknown"))
        source_kind = (
            "synthetic"
            if source_name == "authored_synthetic_audio_candidate"
            or source_name.startswith("authored_synthetic_audio_")
            else "real"
        )
        for frame_index in range(features.shape[0]):
            bucket_id = int(bucket_targets[frame_index])
            if not bool(loss_mask[frame_index]) or bucket_id < 0:
                continue
            bucket = RUNTIME_BUCKETS[bucket_id]
            pools[bucket][source_kind].append((
                features[frame_index],
                bucket_id,
                bool(speaker[frame_index]),
                bool(user[frame_index]),
            ))
    return pools


def _sample_runtime_bucket_source_batch(
    pools: Mapping[str, Mapping[str, Sequence[Any]]],
    *,
    batch_size: int,
    rng: random.Random,
    batch_index: int,
    real_fraction: float = 0.8,
) -> list[tuple[str, str, Any]]:
    if batch_size <= 0 or batch_size % len(RUNTIME_BUCKETS):
        raise ValueError("balanced runtime batch size must be a positive multiple of four")
    if isinstance(batch_index, bool) or not isinstance(batch_index, int) or batch_index < 0:
        raise ValueError("balanced runtime batch index must be a nonnegative integer")
    if not math.isfinite(real_fraction) or not 0.0 <= real_fraction <= 1.0:
        raise ValueError("real_fraction must be finite and between zero and one")
    per_bucket = batch_size // len(RUNTIME_BUCKETS)
    before = math.floor(batch_index * per_bucket * real_fraction + 0.5)
    after = math.floor((batch_index + 1) * per_bucket * real_fraction + 0.5)
    requested_real = after - before
    sampled: list[tuple[str, str, Any]] = []
    for bucket in RUNTIME_BUCKETS:
        sources = pools.get(bucket)
        if not isinstance(sources, Mapping):
            raise ValueError(f"balanced runtime sampler is missing bucket {bucket}")
        real = sources.get("real", ())
        synthetic = sources.get("synthetic", ())
        if real and synthetic:
            real_count = requested_real
            synthetic_count = per_bucket - real_count
        elif real:
            real_count, synthetic_count = per_bucket, 0
        elif synthetic:
            real_count, synthetic_count = 0, per_bucket
        else:
            raise ValueError(f"balanced runtime sampler has no examples for bucket {bucket}")
        sampled.extend((bucket, "real", rng.choice(real)) for _ in range(real_count))
        sampled.extend((bucket, "synthetic", rng.choice(synthetic)) for _ in range(synthetic_count))
    rng.shuffle(sampled)
    return sampled


def _runtime_natural_validation_metrics(
    model: NativeControl6LoRA,
    sequences: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    chunk_frames: int = 1024,
) -> dict[str, Any]:
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    target_probabilities: dict[str, list[float]] = {name: [] for name in ("SPEAK", "YIELD")}
    target_correct: dict[str, list[bool]] = {name: [] for name in ("SPEAK", "YIELD")}
    false_probabilities: dict[str, list[float]] = {
        "SPEAK_on_HOLD_idle": [],
        "YIELD_on_HOLD_active": [],
    }
    false_selected: dict[str, list[bool]] = {
        "SPEAK_on_HOLD_idle": [],
        "YIELD_on_HOLD_active": [],
    }
    target_bucket = {name: RUNTIME_BUCKETS.index(name) for name in ("SPEAK", "YIELD")}
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            if sequence.get("source") not in {"DuplexConv", "SmoothConv"}:
                continue
            n = int(sequence["frame_ids"].numel())
            for start in range(0, n, chunk_frames):
                end = min(n, start + chunk_frames)
                features = sequence["hidden_states"][start:end].to(device)
                target_ids = sequence.get("runtime_target_ids", sequence["target_ids"])[start:end].to(device)
                mask = sequence["loss_mask"][start:end].to(device)
                speaker = sequence["speaker_active"][start:end].to(device)
                user = sequence["user_activity_detected"][start:end].to(device)
                mapped = _runtime_effective_target_ids(target_ids, speaker, user)
                valid = mask & (mapped >= 0)
                if not bool(valid.any()):
                    continue
                log_probs = _runtime_effective_log_probs(_forward_features(model, features), speaker, user)
                probabilities = log_probs.exp()
                predictions = log_probs.argmax(dim=-1)
                for action in ("SPEAK", "YIELD"):
                    bucket_id = target_bucket[action]
                    rows = valid & (mapped == bucket_id)
                    if bool(rows.any()):
                        target_probabilities[action].extend(
                            probabilities[rows, bucket_id].cpu().tolist()
                        )
                        target_correct[action].extend(
                            (predictions[rows] == bucket_id).cpu().tolist()
                        )
                hold_rows = valid & (mapped == RUNTIME_BUCKETS.index("HOLD"))
                idle_hold = hold_rows & ~speaker & ~user
                active_hold = hold_rows & speaker
                for key, rows, action in (
                    ("SPEAK_on_HOLD_idle", idle_hold, "SPEAK"),
                    ("YIELD_on_HOLD_active", active_hold, "YIELD"),
                ):
                    if bool(rows.any()):
                        bucket_id = RUNTIME_BUCKETS.index(action)
                        false_probabilities[key].extend(probabilities[rows, bucket_id].cpu().tolist())
                        false_selected[key].extend((predictions[rows] == bucket_id).cpu().tolist())

    def summarize(values: Sequence[float], correct: Sequence[bool] | None = None) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        result: dict[str, Any] = {
            "support": int(array.size),
            "median_target_probability": float(np.median(array)) if array.size else None,
            "mean_target_probability": float(np.mean(array)) if array.size else None,
            "count_ge_0_70": int(np.sum(array >= 0.70)),
            "count_ge_0_85": int(np.sum(array >= 0.85)),
        }
        if correct is not None:
            result["argmax_correct"] = int(sum(bool(value) for value in correct))
            result["argmax_accuracy"] = (
                float(sum(bool(value) for value in correct) / len(correct)) if correct else None
            )
        else:
            result["false_argmax_selected"] = int(sum(bool(value) for value in false_selected_key))
        return result

    natural = {
        action: summarize(target_probabilities[action], target_correct[action])
        for action in ("SPEAK", "YIELD")
    }
    false_actions = {}
    for key, values in false_probabilities.items():
        false_selected_key = false_selected[key]
        false_actions[key] = summarize(values)
    return {"natural_targets": natural, "false_actions_on_hold": false_actions}


def _hold_raw_aux_eligible_counts(
    sequences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_source_action: dict[str, Counter[str]] = defaultdict(Counter)
    for sequence in sequences:
        target_ids = sequence.get("runtime_target_ids", sequence["target_ids"])
        eligible = _hold_raw_aux_eligible_mask(
            target_ids,
            sequence["loss_mask"],
            sequence["speaker_active"],
            sequence["user_activity_detected"],
        )
        source_counts = by_source_action[str(sequence.get("source", "unknown"))]
        for target in target_ids[eligible].tolist():
            source_counts[ACTIONS[int(target)]] += 1
    normalized = {
        source: {action: int(counts[action]) for action in ("LISTEN", "CONTINUE")}
        for source, counts in sorted(by_source_action.items())
    }
    by_source = {
        source: sum(action_counts.values())
        for source, action_counts in normalized.items()
    }
    by_action = {
        action: sum(action_counts.get(action, 0) for action_counts in normalized.values())
        for action in ("LISTEN", "CONTINUE")
    }
    return {
        "eligible_train_rows": sum(by_source.values()),
        "eligible_train_rows_by_source": by_source,
        "eligible_train_rows_by_raw_action": by_action,
        "eligible_train_rows_by_source_and_raw_action": normalized,
    }


def _balanced_class_weights(
    counts: Mapping[str, int],
    *,
    actions: Sequence[str] = ACTIONS,
) -> torch.Tensor:
    absent = [action for action in actions if int(counts.get(action, 0)) <= 0]
    if absent:
        raise ValueError(f"training split lacks explicit examples for actions: {absent}")
    total = sum(int(counts[action]) for action in actions)
    weights = [total / (len(actions) * int(counts[action])) for action in actions]
    return torch.tensor(weights, dtype=torch.float32)


def _balanced_supported_class_weights(
    counts: Mapping[str, int],
    *,
    actions: Sequence[str],
) -> torch.Tensor:
    """Balance only buckets with supported rows; absent runtime buckets get zero."""
    supported = [action for action in actions if int(counts.get(action, 0)) > 0]
    if not supported:
        raise ValueError("training split has no context-compatible runtime action targets")
    total = sum(int(counts[action]) for action in supported)
    weight = {action: total / (len(supported) * int(counts[action])) for action in supported}
    return torch.tensor([weight.get(action, 0.0) for action in actions], dtype=torch.float32)


def _macro_f1(
    model: NativeControl6LoRA,
    sequences: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_frames: int,
    *,
    replan_target_alias_to_yield: bool = False,
    loss_space: str = "six_class",
) -> dict[str, Any]:
    if loss_space not in {"six_class", "runtime_effective"}:
        raise ValueError("loss_space must be six_class or runtime_effective")
    if loss_space == "runtime_effective" and replan_target_alias_to_yield:
        raise ValueError("runtime_effective loss already uses the policy's contextual REPLAN bucket")
    action_names = (
        RUNTIME_BUCKETS
        if loss_space == "runtime_effective"
        else _effective_action_names(replan_target_alias_to_yield)
    )
    confusion = torch.zeros((len(action_names), len(action_names)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for sequence in sequences:
            n = int(sequence["frame_ids"].numel())
            for start in range(0, n, batch_frames):
                end = min(n, start + batch_frames)
                mask = sequence["loss_mask"][start:end]
                if not bool(mask.any()):
                    continue
                hidden = sequence["hidden_states"][start:end].to(device)
                target = (
                    sequence["runtime_target_ids"]
                    if loss_space == "runtime_effective" and "runtime_target_ids" in sequence
                    else sequence["target_ids"]
                )[start:end].to(device)
                if loss_space == "runtime_effective":
                    speaker_active = sequence["speaker_active"][start:end].to(device)
                    user_active = sequence["user_activity_detected"][start:end].to(device)
                    target = _runtime_effective_target_ids(target, speaker_active, user_active)
                    mask = mask.to(device) & (target >= 0)
                    prediction = _runtime_effective_log_probs(
                        _forward_features(model, hidden), speaker_active, user_active
                    ).argmax(dim=-1)
                else:
                    prediction = _effective_logits(
                        _forward_features(model, hidden), replan_target_alias_to_yield
                    ).argmax(dim=-1)
                    target = _effective_targets(target, replan_target_alias_to_yield)
                    mask = mask.to(device)
                truth = target[mask]
                predicted = prediction[mask]
                bins = truth * len(action_names) + predicted
                confusion += torch.bincount(bins.cpu(), minlength=len(action_names) ** 2).reshape(len(action_names), len(action_names))
    support = confusion.sum(dim=1)
    predicted_support = confusion.sum(dim=0)
    diagonal = confusion.diag().float()
    denom = 2 * diagonal + (predicted_support - diagonal) + (support - diagonal)
    f1 = torch.where(denom > 0, 2 * diagonal / denom, torch.zeros_like(denom))
    examples = int(support.sum().item())
    accuracy = float(diagonal.sum().item() / max(1, examples))
    return {
        "examples": examples,
        "accuracy": accuracy,
        "macro_f1": float(f1.mean().item()),
        "class_support": {action: int(support[i].item()) for i, action in enumerate(action_names)},
        "confusion_matrix": confusion.tolist(),
        "action_names": list(action_names),
    }


def train_adapter(
    *,
    train_targets: Path | None = None,
    validation_targets: Path | None = None,
    hidden_cache: Path | None = None,
    base_rows_path: Path | None = None,
    output_path: Path,
    steps: int = 600,
    batch_frames: int = 128,
    rank: int = 16,
    alpha: float = 16.0,
    learning_rate: float = 1e-3,
    seed: int = 260008,
    device_name: str = "cpu",
    feature_dir: Path | None = None,
    phase_profile: str = "legacy_zero",
    warmstart_checkpoint: Path | None = None,
    replan_target_alias_to_yield: bool = False,
    loss_space: str = "six_class",
    hold_raw_aux_weight: float = 0.0,
    sampling_strategy: str = "sequence_windows",
    phase_only: bool = False,
    snapshot_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    if steps <= 0 or batch_frames <= 0:
        raise ValueError("steps and batch_frames must be positive")
    if phase_profile not in {"legacy_zero", "native1540"}:
        raise ValueError("phase_profile must be 'legacy_zero' or 'native1540'")
    if type(replan_target_alias_to_yield) is not bool:
        raise ValueError("replan_target_alias_to_yield must be boolean")
    if type(phase_only) is not bool:
        raise ValueError("phase_only must be boolean")
    if snapshot_steps is not None:
        if (
            not snapshot_steps
            or any(isinstance(step, bool) or not isinstance(step, int) for step in snapshot_steps)
            or len(set(snapshot_steps)) != len(snapshot_steps)
            or any(step < 0 or step > steps for step in snapshot_steps)
            or 0 not in snapshot_steps
            or steps not in snapshot_steps
        ):
            raise ValueError("snapshot_steps must be unique in-range integers including 0 and the final step")
    if loss_space not in {"six_class", "runtime_effective"}:
        raise ValueError("loss_space must be 'six_class' or 'runtime_effective'")
    if sampling_strategy not in {"sequence_windows", "balanced_runtime_bucket_source"}:
        raise ValueError("unknown sampling_strategy")
    if (
        isinstance(hold_raw_aux_weight, bool)
        or not isinstance(hold_raw_aux_weight, (int, float))
        or not math.isfinite(hold_raw_aux_weight)
        or hold_raw_aux_weight < 0
    ):
        raise ValueError("hold_raw_aux_weight must be a finite nonnegative float")
    if hold_raw_aux_weight and (phase_profile != "native1540" or loss_space != "runtime_effective"):
        raise ValueError(
            "nonzero hold_raw_aux_weight requires phase_profile='native1540' and "
            "loss_space='runtime_effective'"
        )
    if loss_space == "runtime_effective" and replan_target_alias_to_yield:
        raise ValueError("runtime_effective loss is exclusive with global REPLAN→YIELD pooling")
    if sampling_strategy == "balanced_runtime_bucket_source":
        if phase_profile != "native1540" or loss_space != "runtime_effective":
            raise ValueError("balanced runtime sampling requires native1540 runtime_effective training")
        if hold_raw_aux_weight != 0:
            raise ValueError("balanced runtime sampling requires hold_raw_aux_weight=0")
        if batch_frames % len(RUNTIME_BUCKETS):
            raise ValueError("balanced runtime batch size must be divisible by four")
    if phase_only and (
        phase_profile != "native1540"
        or loss_space != "runtime_effective"
        or sampling_strategy != "balanced_runtime_bucket_source"
        or warmstart_checkpoint is None
        or hold_raw_aux_weight != 0
        or replan_target_alias_to_yield
    ):
        raise ValueError(
            "phase_only requires native1540 runtime_effective balanced training from a warmstart, "
            "without auxiliary loss or global REPLAN pooling"
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite native Control6 checkpoint: {output_path}")
    if output_path.with_suffix(output_path.suffix + ".tmp").exists():
        raise FileExistsError(f"refusing to overwrite stale checkpoint temporary file: {output_path}.tmp")

    if phase_profile == "native1540":
        if feature_dir is None:
            raise ValueError("native1540 training requires Stage2 --feature-dir input")
        if train_targets is not None or validation_targets is not None or hidden_cache is not None:
            raise ValueError("native1540 Stage2 input cannot be mixed with legacy target/cache inputs")
        train, train_info = load_stage2_feature_sequences(feature_dir, "train")
        validation, validation_info = load_stage2_feature_sequences(feature_dir, "validation")
    else:
        if feature_dir is not None:
            raise ValueError("Stage2 feature input requires phase_profile='native1540'")
        if train_targets is None or validation_targets is None or hidden_cache is None:
            raise ValueError("legacy training requires train targets, validation targets, and hidden cache")
        train, train_info = load_sequences(train_targets, hidden_cache, "train")
        validation, validation_info = load_sequences(validation_targets, hidden_cache, "validation")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training requested CUDA but CUDA is unavailable")
    assert_disjoint_source_groups(train, validation)
    initialization: dict[str, Any]
    if warmstart_checkpoint is not None:
        if phase_profile != "native1540":
            raise ValueError("native Control6 warm-start is only supported for explicit native1540 training")
        warmstart_checkpoint = Path(warmstart_checkpoint)
        if not warmstart_checkpoint.is_file():
            raise FileNotFoundError(f"native Control6 warm-start checkpoint missing: {warmstart_checkpoint}")
        base_rows = None
        base_metadata = None
        if base_rows_path is not None:
            base_rows, base_metadata = load_base_rows(base_rows_path)
            if base_metadata.get("base_weight_source") != BASE_WEIGHT_SOURCE:
                raise ValueError("training base rows are not the approved native Gemma lm_head slice")
        model, warmstart_report = load_native_control6_adapter(
            warmstart_checkpoint,
            base_rows=base_rows,
            device="cpu",
        )
        if base_rows is None:
            base_rows = model.base_rows.detach().cpu().clone()
        if model.rank != rank or model.alpha != float(alpha):
            raise ValueError(
                f"warm-start rank/alpha {(model.rank, model.alpha)} != requested {(rank, float(alpha))}"
            )
        if warmstart_report.get("native_control6_phase_profile") not in {"legacy", "native1540"}:
            raise ValueError("warm-start checkpoint did not report a recognized native Control6 phase profile")
        initialization = {
            "kind": "checkpoint_warmstart",
            "path": str(warmstart_checkpoint.resolve()),
            "sha256": _file_sha256(warmstart_checkpoint),
            "phase_profile": warmstart_report["native_control6_phase_profile"],
            "phase_t_checkpoint_migration": warmstart_report.get("phase_t_checkpoint_migration"),
            "phase_t_checkpoint_migration_details": warmstart_report.get("phase_t_checkpoint_migration_details"),
        }
        if base_metadata is None:
            base_weight_source = BASE_WEIGHT_SOURCE
        else:
            base_weight_source = base_metadata["base_weight_source"]
    else:
        if base_rows_path is None:
            raise ValueError("training from scratch requires --base-rows")
        base_rows, base_metadata = load_base_rows(base_rows_path)
        if base_metadata.get("base_weight_source") != BASE_WEIGHT_SOURCE:
            raise ValueError("training base rows are not the approved native Gemma lm_head slice")
        model = NativeControl6LoRA(base_rows, rank=rank, alpha=alpha)
        base_weight_source = base_metadata["base_weight_source"]
        initialization = {"kind": "fresh", "base_rows_path": str(base_rows_path.resolve())}
    if phase_profile != "native1540" and initialization.get("phase_profile") == "native1540":
        raise ValueError("native1540 checkpoint cannot be warm-started into a legacy zero-phase run")
    model = model.to(device)
    model.train()
    phase_only_initial: dict[str, torch.Tensor] | None = None
    phase_only_hook = None
    if phase_only:
        phase_only_initial = {
            "hidden_A": model.lora_A[:, :HIDDEN_WIDTH].detach().clone(),
            "B": model.lora_B.detach().clone(),
            "base_rows": model.base_rows.detach().clone(),
        }
        phase_mask = torch.zeros_like(model.lora_A)
        phase_mask[:, HIDDEN_WIDTH:] = 1
        phase_only_hook = model.lora_A.register_hook(lambda gradient: gradient * phase_mask)
        model.lora_B.requires_grad_(False)
        optimizer = torch.optim.AdamW((model.lora_A,), lr=learning_rate, weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW((model.lora_A, model.lora_B), lr=learning_rate)
    action_names = (
        RUNTIME_BUCKETS
        if loss_space == "runtime_effective"
        else _effective_action_names(replan_target_alias_to_yield)
    )
    original_train_action_counts = {
        action: int(train_info["action_counts"].get(action, 0)) for action in ACTIONS
    }
    runtime_unsupported_train_targets: Counter[str] = Counter()
    effective_train_action_counts = {action: 0 for action in action_names}
    if loss_space == "runtime_effective":
        for sequence in train:
            mapped = _runtime_effective_target_ids(
                sequence.get("runtime_target_ids", sequence["target_ids"]),
                sequence["speaker_active"],
                sequence["user_activity_detected"],
            )
            for raw_target, runtime_target, bucket_target, supervised in zip(
                sequence["target_ids"].tolist(),
                sequence.get("runtime_target_ids", sequence["target_ids"]).tolist(),
                mapped.tolist(),
                sequence["loss_mask"].tolist(),
            ):
                if not supervised:
                    continue
                if bucket_target < 0:
                    runtime_unsupported_train_targets[
                        f"{sequence['source']}:{ACTIONS[int(raw_target)]}"
                    ] += 1
                else:
                    effective_train_action_counts[RUNTIME_BUCKETS[bucket_target]] += 1
        class_weights = (
            torch.ones(len(action_names), dtype=torch.float32, device=device)
            if sampling_strategy == "balanced_runtime_bucket_source"
            else _balanced_supported_class_weights(
                effective_train_action_counts,
                actions=action_names,
            ).to(device)
        )
    else:
        for action in ACTIONS:
            count = int(original_train_action_counts[action])
            effective_action = "YIELD" if replan_target_alias_to_yield and action == "REPLAN" else action
            if effective_action in effective_train_action_counts:
                effective_train_action_counts[effective_action] += count
        class_weights = _balanced_class_weights(
            effective_train_action_counts,
            actions=action_names,
        ).to(device)
    hold_raw_aux_counts = (
        _hold_raw_aux_eligible_counts(train)
        if phase_profile == "native1540" and loss_space == "runtime_effective"
        else {
            "eligible_train_rows": 0,
            "eligible_train_rows_by_source": {},
            "eligible_train_rows_by_raw_action": {},
            "eligible_train_rows_by_source_and_raw_action": {},
        }
    )
    epoch = 0
    window_target_key = "runtime_target_ids" if loss_space == "runtime_effective" else "target_ids"
    balanced_pools = (
        _runtime_bucket_source_pools(train)
        if sampling_strategy == "balanced_runtime_bucket_source"
        else None
    )
    sampler_rng = random.Random(seed)
    sampled_by_bucket_source: Counter[str] = Counter()
    windows = (
        []
        if balanced_pools is not None
        else _sequence_windows(
            train,
            batch_frames=batch_frames,
            generator=random.Random(seed),
            target_key=window_target_key,
        )
    )
    if balanced_pools is None and not windows:
        raise ValueError("training split has no valid sequence windows")
    step_losses: list[float] = []
    physical_step_loss_sum = 0.0
    auxiliary_step_loss_sum = 0.0
    phase_a_initial = model.lora_A[:, HIDDEN_WIDTH:].detach().clone()
    phase_gradient_max_norm = 0.0
    checkpoint_evaluation_curve: list[dict[str, Any]] = []
    checkpoint_snapshots: dict[str, dict[str, torch.Tensor]] = {}
    evaluation_steps = (
        {0, 150, 600} if phase_only else {0, 50, 150, 300, 600}
    ) if snapshot_steps is None else set(snapshot_steps)

    def evaluate_checkpoint(step: int) -> None:
        metrics = _runtime_natural_validation_metrics(model, validation, device)
        checkpoint_evaluation_curve.append({"step": step, **metrics})
        checkpoint_snapshots[str(step)] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.adapter_state_dict().items()
        }
        model.train()

    if balanced_pools is not None:
        evaluate_checkpoint(0)
    while len(step_losses) < steps:
        if balanced_pools is None and epoch:
            windows = _sequence_windows(
                train,
                batch_frames=batch_frames,
                generator=random.Random(seed + epoch),
                target_key=window_target_key,
            )
        batches = (
            [
                _sample_runtime_bucket_source_batch(
                    balanced_pools,
                    batch_size=batch_frames,
                    rng=sampler_rng,
                    batch_index=len(step_losses),
                )
            ]
            if balanced_pools is not None
            else windows
        )
        for batch in batches:
            if len(step_losses) >= steps:
                break
            if balanced_pools is not None:
                sampled = batch
                records = [entry[2] for entry in sampled]
                features = torch.stack([record[0] for record in records]).to(device)
                effective_targets = torch.tensor(
                    [record[1] for record in records], dtype=torch.long, device=device
                )
                speaker_active = torch.tensor(
                    [record[2] for record in records], dtype=torch.bool, device=device
                )
                user_activity_detected = torch.tensor(
                    [record[3] for record in records], dtype=torch.bool, device=device
                )
                mask = torch.ones(batch_frames, dtype=torch.bool, device=device)
                targets = torch.full((batch_frames,), -100, dtype=torch.long, device=device)
                for bucket_name, source_name, _ in sampled:
                    sampled_by_bucket_source[f"{bucket_name}:{source_name}"] += 1
            else:
                hidden, targets, mask, speaker_active, user_activity_detected = batch
                features = hidden.to(device)
                targets = targets.to(device)
                mask = mask.to(device)
                speaker_active = speaker_active.to(device)
                user_activity_detected = user_activity_detected.to(device)
            optimizer.zero_grad(set_to_none=True)
            raw_logits = _forward_features(model, features)
            if loss_space == "runtime_effective":
                if balanced_pools is None:
                    effective_targets = _runtime_effective_target_ids(
                        targets, speaker_active, user_activity_detected
                    )
                effective_mask = mask & (effective_targets >= 0)
                if not bool(effective_mask.any()):
                    continue
                log_probabilities = _runtime_effective_log_probs(
                    raw_logits, speaker_active, user_activity_detected
                )
                selected_targets = effective_targets[effective_mask]
                per_item = -log_probabilities[effective_mask].gather(
                    1, selected_targets.unsqueeze(1)
                ).squeeze(1)
            else:
                logits = _effective_logits(raw_logits, replan_target_alias_to_yield)
                effective_targets = _effective_targets(targets, replan_target_alias_to_yield)
                effective_mask = mask
                selected_targets = effective_targets[effective_mask]
                per_item = F.cross_entropy(
                    logits[effective_mask], selected_targets, reduction="none"
                )
            if sampling_strategy == "balanced_runtime_bucket_source":
                physical_loss = _unweighted_runtime_bucket_nll(
                    log_probabilities[effective_mask], selected_targets
                )
            else:
                selected_weights = class_weights.index_select(0, selected_targets)
                physical_loss = (per_item * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)
            if hold_raw_aux_weight:
                auxiliary_raw_loss, _ = _hold_raw_aux_loss(
                    raw_logits,
                    targets,
                    mask,
                    speaker_active,
                    user_activity_detected,
                )
                auxiliary_loss = auxiliary_raw_loss * hold_raw_aux_weight
                loss = physical_loss + auxiliary_loss
            else:
                # Keep the default objective and optimizer inputs exactly the
                # existing physical weighted runtime-bucket loss.
                auxiliary_loss = physical_loss.detach().new_zeros(())
                loss = physical_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {len(step_losses) + 1}")
            loss.backward()
            if phase_only:
                if model.lora_A.grad is None or model.lora_B.grad is not None:
                    raise RuntimeError("phase-only training must update A while freezing B")
                if bool(torch.count_nonzero(model.lora_A.grad[:, :HIDDEN_WIDTH]).item()):
                    raise RuntimeError("phase-only training produced a hidden-column A gradient")
                if not bool(torch.count_nonzero(model.lora_A.grad[:, HIDDEN_WIDTH:]).item()):
                    raise RuntimeError("phase-only training produced no phase-column A gradient")
            elif model.lora_A.grad is None or model.lora_B.grad is None:
                raise RuntimeError("both native output-row LoRA factors must participate in backward")
            if phase_profile == "native1540":
                phase_gradient = model.lora_A.grad[:, HIDDEN_WIDTH:]
                phase_gradient_max_norm = max(
                    phase_gradient_max_norm,
                    float(torch.linalg.vector_norm(phase_gradient.detach()).cpu()),
                )
            optimizer.step()
            step_losses.append(float(loss.detach().cpu()))
            physical_step_loss_sum += float(physical_loss.detach().cpu())
            auxiliary_step_loss_sum += float(auxiliary_loss.detach().cpu())
            completed_step = len(step_losses)
            if balanced_pools is not None and completed_step in evaluation_steps:
                evaluate_checkpoint(completed_step)
        epoch += 1
    validation_raw_six = _macro_f1(model, validation, device, batch_frames)
    validation_metrics = (
        _macro_f1(
            model,
            validation,
            device,
            batch_frames,
            loss_space="runtime_effective",
        )
        if loss_space == "runtime_effective"
        else _macro_f1(
            model,
            validation,
            device,
            batch_frames,
            replan_target_alias_to_yield=True,
        )
        if replan_target_alias_to_yield
        else validation_raw_six
    )
    phase_learnability: dict[str, Any] | None = None
    if phase_profile == "native1540":
        phase_a_final = model.lora_A[:, HIDDEN_WIDTH:].detach()
        phase_probe = next(
            sequence["hidden_states"][: min(128, len(sequence["hidden_states"]))]
            for sequence in train
            if sequence["hidden_states"].shape[-1] == HIDDEN_WIDTH + PHASE_T_WIDTH
        ).to(device)
        probe_hidden = phase_probe[..., :HIDDEN_WIDTH]
        probe_phase = phase_probe[..., HIDDEN_WIDTH:]
        model.eval()
        with torch.no_grad():
            phase_delta = (
                model(probe_hidden, phase_t=probe_phase)
                - model(probe_hidden, phase_t=torch.zeros_like(probe_phase))
            )
        phase_learnability = {
            "phase_a_gradient_max_l2": phase_gradient_max_norm,
            "phase_a_update_l2": float(torch.linalg.vector_norm(phase_a_final - phase_a_initial).cpu()),
            "phase_a_weight_l2": float(torch.linalg.vector_norm(phase_a_final).cpu()),
            "phase_conditioned_logit_delta_rms": float(torch.sqrt(torch.mean(phase_delta.float() ** 2)).cpu()),
            "probe_rows": int(len(phase_probe)),
            "last_four_a_columns_are_trainable": bool(model.lora_A.requires_grad),
        }
        numeric = [
            phase_learnability["phase_a_gradient_max_l2"],
            phase_learnability["phase_a_update_l2"],
            phase_learnability["phase_a_weight_l2"],
            phase_learnability["phase_conditioned_logit_delta_rms"],
        ]
        if not all(math.isfinite(value) for value in numeric):
            raise RuntimeError("native1540 phase learnability metrics are non-finite")
        if phase_gradient_max_norm <= 0 or phase_learnability["phase_conditioned_logit_delta_rms"] <= 0:
            raise RuntimeError("native1540 phase columns received no usable gradient or logit contribution")

    phase_only_preservation: dict[str, bool] | None = None
    if phase_only:
        assert phase_only_initial is not None
        phase_only_preservation = {
            "hidden_A_bitwise_unchanged": torch.equal(
                model.lora_A[:, :HIDDEN_WIDTH].detach(), phase_only_initial["hidden_A"]
            ),
            "lora_B_bitwise_unchanged": torch.equal(
                model.lora_B.detach(), phase_only_initial["B"]
            ),
            "base_rows_bitwise_unchanged": torch.equal(
                model.base_rows.detach(), phase_only_initial["base_rows"]
            ),
        }
        if not all(phase_only_preservation.values()):
            raise RuntimeError(f"phase-only training changed frozen weights: {phase_only_preservation}")
        if phase_only_hook is not None:
            phase_only_hook.remove()

    train_target_remapping = {
        "enabled": replan_target_alias_to_yield,
        "mapping": {"REPLAN": "YIELD"} if replan_target_alias_to_yield else {},
        "application": "in-memory targets and loss only; prepared data and metadata unchanged",
        "loss_action_names": list(action_names),
        "loss_pooling": (
            "YIELD logit = logsumexp(raw YIELD, raw REPLAN); five-way softmax exactly matches "
            "the runtime sum of the two raw softmax probabilities"
            if replan_target_alias_to_yield
            else "unpooled six-way native logits"
        ),
        "original_train_action_counts": original_train_action_counts,
        "effective_train_action_counts": effective_train_action_counts,
        "remapped_train_target_rows": (
            original_train_action_counts["REPLAN"] if replan_target_alias_to_yield else 0
        ),
        "dataset_action_provenance_preserved": True,
    }
    runtime_effective_loss = None
    if loss_space == "runtime_effective":
        runtime_effective_loss = {
            "enabled": True,
            "bucket_order": list(RUNTIME_BUCKETS),
            "context_members_by_state": {
                f"speaker_active={speaker};user_activity_detected={user}": {
                    bucket: list(members)
                    for bucket, members in native_control6_effective_members(
                        speaker_active=speaker,
                        user_activity_detected=user,
                    ).items()
                }
                for speaker in (False, True)
                for user in (False, True)
            },
            "log_target_mass": "logsumexp(all six raw logits) - logsumexp(raw logits in the matching runtime bucket)",
            "excluded_raw_rows_remain_in_six_way_denominator": True,
            "legal_action_mask_applied_before_loss": False,
            "context_incompatible_target_policy": "mask only that row's loss; retain raw action and provenance",
            "unsupported_train_targets_by_source_action": dict(sorted(runtime_unsupported_train_targets.items())),
            "train_runtime_bucket_counts": effective_train_action_counts,
            "validation_runtime_bucket_counts": validation_info.get("runtime_bucket_counts", {}),
            "validation_unsupported_targets_by_source_action": validation_info.get(
                "context_unsupported_targets_by_source_action", {}
            ),
            "validation_metrics_action_order": list(RUNTIME_BUCKETS),
        }
    source_provenance = {
        "input_mode": "stage2_native1540_features" if phase_profile == "native1540" else "legacy_hidden_cache",
        "train": {
            key: train_info[key]
            for key in (
                "training_state_context", "source_rows", "source_supervised_rows", "source_groups", "source_group_ids",
                "audio_source_sha256_count", "audio_source_sha256", "feature_manifest_sha256",
                "feature_npz_sha256", "feature_metadata_sha256",
            )
            if key in train_info
        },
        "validation": {
            key: validation_info[key]
            for key in (
                "training_state_context", "source_rows", "source_supervised_rows", "source_groups", "source_group_ids",
                "audio_source_sha256_count", "audio_source_sha256", "feature_manifest_sha256",
                "feature_npz_sha256", "feature_metadata_sha256",
            )
            if key in validation_info
        },
    }
    report = {
        "schema_version": 1,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": MODEL_ID,
        "base_weight_source": base_weight_source,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "hidden_source": HIDDEN_SOURCE,
        "hidden_normalization": HIDDEN_NORM,
        "hidden_width": HIDDEN_WIDTH,
        "input_width": DELTA_INPUT_WIDTH if phase_profile == "native1540" else HIDDEN_WIDTH,
        "native_control6_phase_profile": "native1540" if phase_profile == "native1540" else "legacy",
        "phase_schema": NATIVE1540_PHASE_SCHEMA if phase_profile == "native1540" else None,
        "loss_space": loss_space,
        "sampling_strategy": sampling_strategy,
        "phase_only": phase_only,
        "phase_only_preservation": phase_only_preservation,
        "sampling_strategy_config": (
            {
                "bucket_order": list(RUNTIME_BUCKETS),
                "batch_rows_per_bucket": batch_frames // len(RUNTIME_BUCKETS),
                "real_fraction_per_bucket": 0.8,
                "synthetic_fraction_per_bucket": 0.2,
                "sampling_with_replacement": True,
                "loss_weights": "unweighted mean runtime-effective NLL",
                "sampled_rows_by_bucket_and_source": dict(sorted(sampled_by_bucket_source.items())),
            }
            if sampling_strategy == "balanced_runtime_bucket_source"
            else None
        ),
        "checkpoint_evaluation_curve": checkpoint_evaluation_curve,
        "checkpoint_saved_steps": [int(step) for step in checkpoint_snapshots],
        "checkpoint_selected_step": len(step_losses) if balanced_pools is not None else None,
        "checkpoint_selection": (
            "final training step; all requested evaluation-step adapter snapshots are preserved for review"
            if balanced_pools is not None
            else None
        ),
        "hold_raw_aux_weight": float(hold_raw_aux_weight),
        "hold_raw_aux": {
            "enabled": hold_raw_aux_weight > 0,
            "weight": float(hold_raw_aux_weight),
            **hold_raw_aux_counts,
            "eligible_rows_scope": "supervised train rows with runtime target LISTEN or CONTINUE mapped to HOLD",
            "auxiliary_loss": "weighted mean cross-entropy over all six raw logits",
        },
        "runtime_effective_loss": runtime_effective_loss,
        "training_state_context": (
            train_info.get("training_state_context", "unspecified")
            if phase_profile == "native1540"
            else None
        ),
        **(train_info["semantic_prefix_metadata"] if phase_profile == "native1540" else {}),
        "tick_ms": TICK_MS,
        "rank": rank,
        "alpha": alpha,
        "steps": len(step_losses),
        "batch_frames": batch_frames,
        "learning_rate": learning_rate,
        "seed": seed,
        "response_adapter": "OFF",
        "evaluation_scope": (
            "source-group-disjoint train/validation for mixed real and authored audio; held-out Candidate F not read"
            if phase_profile == "native1540"
            else "directional synthetic held-out template families; not human conversation quality proof"
        ),
        "initialization": initialization,
        "train": {k: v for k, v in train_info.items() if k != "cache_metadata"},
        "validation": {k: v for k, v in validation_info.items() if k != "cache_metadata"},
        "source_provenance": source_provenance,
        "target_remapping": train_target_remapping,
        "class_weights": {action: float(class_weights[i].cpu()) for i, action in enumerate(action_names)},
        "mean_training_loss": sum(step_losses) / len(step_losses),
        "mean_physical_step_loss": physical_step_loss_sum / len(step_losses),
        "mean_auxiliary_step_loss": auxiliary_step_loss_sum / len(step_losses),
        "mean_combined_step_loss": sum(step_losses) / len(step_losses),
        "last_32_step_mean_loss": sum(step_losses[-32:]) / min(32, len(step_losses)),
        "validation_metrics": validation_metrics,
        "validation_metrics_physical": (
            validation_metrics
            if replan_target_alias_to_yield or loss_space == "runtime_effective"
            else None
        ),
        "validation_metrics_raw_six_class": validation_raw_six,
        "native1540_phase_learnability": phase_learnability,
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
    }
    artifact = {
        "schema_version": 1,
        "architecture": report["architecture"],
        "model_id": MODEL_ID,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "rank": rank,
        "alpha": alpha,
        "base_weight_source": base_weight_source,
        "base_rows": base_rows.detach().cpu(),
        "response_adapter": "OFF",
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
        "adapter_state_dict": model.adapter_state_dict(),
        "training_snapshots": checkpoint_snapshots if balanced_pools is not None else {},
        "native_control6_phase_profile": "native1540" if phase_profile == "native1540" else "legacy",
        "report": report,
        **(train_info["semantic_prefix_metadata"] if phase_profile == "native1540" else {}),
    }
    if phase_profile == "native1540":
        artifact["phase_schema"] = dict(NATIVE1540_PHASE_SCHEMA)
    else:
        # Keep the old hidden-only ABI when no PCM phase values were trained.
        artifact["adapter_state_dict"]["lora_A"] = artifact["adapter_state_dict"]["lora_A"][:, :HIDDEN_WIDTH].clone()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    if output_path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite native Control6 checkpoint output: {output_path}")
    torch.save(artifact, temporary)
    if output_path.exists():
        temporary.unlink()
        raise FileExistsError(f"native Control6 checkpoint appeared during training: {output_path}")
    temporary.replace(output_path)
    return report


def cpu_smoke() -> dict[str, Any]:
    torch.manual_seed(260008)
    base = torch.randn(len(ACTIONS), HIDDEN_WIDTH)
    hidden = torch.randn(12, HIDDEN_WIDTH)
    model = NativeControl6LoRA(base, rank=16, alpha=16)
    initial = model(hidden)
    frozen_native_rows = model.base_rows.clone()
    expected = F.linear(hidden, base)
    if not torch.equal(initial, expected):
        raise AssertionError("zero-initialized adapter must exactly preserve native six-row logits")
    targets = torch.arange(len(ACTIONS), dtype=torch.long).repeat(2)
    loss = F.cross_entropy(initial, targets)
    loss.backward()
    if model.lora_A.grad is None or model.lora_B.grad is None:
        raise AssertionError("native output-row LoRA factors did not receive gradients")
    if model.base_rows.requires_grad or not torch.equal(model.base_rows, frozen_native_rows):
        raise AssertionError("base Gemma rows must remain frozen")
    if set(model.adapter_state_dict()) != {"lora_A", "lora_B"}:
        raise AssertionError("adapter artifact may contain only the selected-row LoRA factors")
    return {
        "status": "passed",
        "logits_shape": list(initial.shape),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "base_rows_unchanged": True,
        "ordinary_vocabulary_rows_modified": False,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
    }


def _finite_nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a finite nonnegative float") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative float")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-targets", help="project authored synthetic schedules to 80 ms targets")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)

    extract = subparsers.add_parser("extract", help="replay train/validation WAVs through the production Gemma listener")
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--base-model", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--split", choices=("train", "validation"), action="append")
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--no-resume", action="store_true")

    train = subparsers.add_parser("train", help="fit only native Gemma output-row LoRA deltas")
    train.add_argument("--train-targets", type=Path)
    train.add_argument("--validation-targets", type=Path)
    train.add_argument("--hidden-cache", type=Path)
    train.add_argument("--base-rows", type=Path)
    train.add_argument("--feature-dir", type=Path, help="Stage2 train/validation NPZ output with 1540 native features")
    train.add_argument(
        "--phase-profile",
        choices=("legacy_zero", "native1540"),
        default="legacy_zero",
        help="select the explicit input/checkpoint ABI; legacy_zero preserves the 1536-column checkpoint ABI",
    )
    train.add_argument("--warmstart-checkpoint", type=Path)
    train.add_argument(
        "--replan-target-alias-to-yield",
        action="store_true",
        help="map REPLAN targets to the physical YIELD loss pool for this run only",
    )
    train.add_argument(
        "--loss-space",
        choices=("six_class", "runtime_effective"),
        default="six_class",
        help="use raw six-way loss or contextual runtime bucket probability mass",
    )
    train.add_argument(
        "--hold-raw-aux-weight",
        type=_finite_nonnegative_float,
        default=0.0,
        help="add weighted raw six-way CE for LISTEN/CONTINUE targets pooled into runtime HOLD",
    )
    train.add_argument(
        "--sampling-strategy",
        choices=("sequence_windows", "balanced_runtime_bucket_source"),
        default="sequence_windows",
        help="balanced source-aware 4-bucket sampling for targeted native1540 runtime-effective training",
    )
    train.add_argument(
        "--phase-only",
        action="store_true",
        help="freeze hidden LoRA columns, B, and base rows; optimize only native1540 phase columns",
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=None,
        help="optional validation/checkpoint steps; must include zero and the final training step",
    )
    train.add_argument("--batch-frames", type=int, default=128)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--alpha", type=float, default=16.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=260008)
    train.add_argument("--device", default="cpu")

    subparsers.add_parser("smoke", help="CPU-only native row/LoRA shape and gradient smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "train" and args.hold_raw_aux_weight and (
        args.phase_profile != "native1540" or args.loss_space != "runtime_effective"
    ):
        parser.error(
            "--hold-raw-aux-weight requires --phase-profile native1540 and "
            "--loss-space runtime_effective when nonzero"
        )
    if args.command == "prepare-targets":
        result = prepare_targets(args.manifest, args.output_dir)
    elif args.command == "extract":
        result = extract_hidden_cache(
            args.manifest,
            args.base_model,
            args.output_dir,
            splits=tuple(args.split or ("train", "validation")),
            device=args.device,
            resume=not args.no_resume,
        )
    elif args.command == "train":
        result = train_adapter(
            train_targets=args.train_targets,
            validation_targets=args.validation_targets,
            hidden_cache=args.hidden_cache,
            base_rows_path=args.base_rows,
            output_path=args.output,
            steps=args.steps,
            batch_frames=args.batch_frames,
            rank=args.rank,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device_name=args.device,
            feature_dir=args.feature_dir,
            phase_profile=args.phase_profile,
            warmstart_checkpoint=args.warmstart_checkpoint,
            replan_target_alias_to_yield=args.replan_target_alias_to_yield,
            loss_space=args.loss_space,
            hold_raw_aux_weight=args.hold_raw_aux_weight,
            sampling_strategy=args.sampling_strategy,
            phase_only=args.phase_only,
            snapshot_steps=args.snapshot_steps,
        )
    else:
        result = cpu_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
