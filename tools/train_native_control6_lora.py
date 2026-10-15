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

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.duplex_qlora.native_control6 import (  # noqa: E402
    ACTION_TO_TOKEN_ID,
    ACTIONS,
    BASE_WEIGHT_SOURCE,
    HIDDEN_CONTRACT,
    HIDDEN_NORM,
    HIDDEN_SOURCE,
    HIDDEN_WIDTH,
    MODEL_ID,
    NativeControl6LoRA,
    TICK_MS,
    TOKEN_ID_TO_STRING,
    load_base_rows,
    save_base_rows,
)

TARGET_SCHEMA_VERSION = 1
TARGET_SCHEMA = "native_control6_action_targets_v1"
FAMILY_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


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
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    order = list(range(len(sequences)))
    if generator is not None:
        generator.shuffle(order)
    windows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for index in order:
        sequence = sequences[index]
        n = int(sequence["frame_ids"].numel())
        for start in range(0, n, batch_frames):
            end = min(n, start + batch_frames)
            mask = sequence["loss_mask"][start:end]
            if bool(mask.any()):
                windows.append((
                    sequence["hidden_states"][start:end],
                    sequence["target_ids"][start:end],
                    mask,
                ))
    return windows


def _balanced_class_weights(counts: Mapping[str, int]) -> torch.Tensor:
    absent = [action for action in ACTIONS if int(counts.get(action, 0)) <= 0]
    if absent:
        raise ValueError(f"training split lacks explicit examples for actions: {absent}")
    total = sum(int(counts[action]) for action in ACTIONS)
    weights = [total / (len(ACTIONS) * int(counts[action])) for action in ACTIONS]
    return torch.tensor(weights, dtype=torch.float32)


def _macro_f1(model: NativeControl6LoRA, sequences: Sequence[Mapping[str, Any]], device: torch.device, batch_frames: int) -> dict[str, Any]:
    confusion = torch.zeros((len(ACTIONS), len(ACTIONS)), dtype=torch.int64)
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
                target = sequence["target_ids"][start:end].to(device)
                prediction = model(hidden).argmax(dim=-1)
                truth = target[mask.to(device)]
                predicted = prediction[mask.to(device)]
                bins = truth * len(ACTIONS) + predicted
                confusion += torch.bincount(bins.cpu(), minlength=len(ACTIONS) ** 2).reshape(len(ACTIONS), len(ACTIONS))
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
        "class_support": {action: int(support[i].item()) for i, action in enumerate(ACTIONS)},
        "confusion_matrix": confusion.tolist(),
    }


def train_adapter(
    *,
    train_targets: Path,
    validation_targets: Path,
    hidden_cache: Path,
    base_rows_path: Path,
    output_path: Path,
    steps: int = 600,
    batch_frames: int = 128,
    rank: int = 16,
    alpha: float = 16.0,
    learning_rate: float = 1e-3,
    seed: int = 260008,
    device_name: str = "cpu",
) -> dict[str, Any]:
    if steps <= 0 or batch_frames <= 0:
        raise ValueError("steps and batch_frames must be positive")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training requested CUDA but CUDA is unavailable")
    train, train_info = load_sequences(train_targets, hidden_cache, "train")
    validation, validation_info = load_sequences(validation_targets, hidden_cache, "validation")
    assert_disjoint_source_groups(train, validation)
    base_rows, base_metadata = load_base_rows(base_rows_path)
    if base_metadata.get("base_weight_source") != BASE_WEIGHT_SOURCE:
        raise ValueError("training base rows are not the approved native Gemma lm_head slice")
    model = NativeControl6LoRA(base_rows, rank=rank, alpha=alpha).to(device)
    optimizer = torch.optim.AdamW((model.lora_A, model.lora_B), lr=learning_rate)
    class_weights = _balanced_class_weights(train_info["action_counts"]).to(device)
    epoch = 0
    windows = _sequence_windows(train, batch_frames=batch_frames, generator=random.Random(seed))
    if not windows:
        raise ValueError("training split has no valid sequence windows")
    step_losses: list[float] = []
    while len(step_losses) < steps:
        if epoch:
            windows = _sequence_windows(train, batch_frames=batch_frames, generator=random.Random(seed + epoch))
        for hidden, targets, mask in windows:
            if len(step_losses) >= steps:
                break
            hidden = hidden.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(hidden)
            per_item = F.cross_entropy(logits[mask], targets[mask], reduction="none")
            selected_weights = class_weights.index_select(0, targets[mask])
            loss = (per_item * selected_weights).sum() / selected_weights.sum().clamp_min(1e-12)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at step {len(step_losses) + 1}")
            loss.backward()
            if model.lora_A.grad is None or model.lora_B.grad is None:
                raise RuntimeError("both native output-row LoRA factors must participate in backward")
            optimizer.step()
            step_losses.append(float(loss.detach().cpu()))
        epoch += 1
    validation_metrics = _macro_f1(model, validation, device, batch_frames)
    report = {
        "schema_version": 1,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": MODEL_ID,
        "base_weight_source": base_metadata["base_weight_source"],
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "hidden_source": HIDDEN_SOURCE,
        "hidden_normalization": HIDDEN_NORM,
        "hidden_width": HIDDEN_WIDTH,
        "tick_ms": TICK_MS,
        "rank": rank,
        "alpha": alpha,
        "steps": len(step_losses),
        "batch_frames": batch_frames,
        "learning_rate": learning_rate,
        "seed": seed,
        "response_adapter": "OFF",
        "evaluation_scope": "directional synthetic held-out template families; not human conversation quality proof",
        "train": {k: v for k, v in train_info.items() if k != "cache_metadata"},
        "validation": {k: v for k, v in validation_info.items() if k != "cache_metadata"},
        "class_weights": {action: float(class_weights[i].cpu()) for i, action in enumerate(ACTIONS)},
        "mean_training_loss": sum(step_losses) / len(step_losses),
        "last_32_step_mean_loss": sum(step_losses[-32:]) / min(32, len(step_losses)),
        "validation_metrics": validation_metrics,
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
        "base_weight_source": base_metadata["base_weight_source"],
        "base_rows": base_rows.detach().cpu(),
        "response_adapter": "OFF",
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
        "adapter_state_dict": model.adapter_state_dict(),
        "report": report,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, temporary)
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
    train.add_argument("--train-targets", type=Path, required=True)
    train.add_argument("--validation-targets", type=Path, required=True)
    train.add_argument("--hidden-cache", type=Path, required=True)
    train.add_argument("--base-rows", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--batch-frames", type=int, default=128)
    train.add_argument("--rank", type=int, default=16)
    train.add_argument("--alpha", type=float, default=16.0)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--seed", type=int, default=260008)
    train.add_argument("--device", default="cpu")

    subparsers.add_parser("smoke", help="CPU-only native row/LoRA shape and gradient smoke")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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
        )
    else:
        result = cpu_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
