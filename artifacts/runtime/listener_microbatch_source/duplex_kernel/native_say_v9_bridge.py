"""Matched native SAY-hidden -> V9 -> Qwen Audio-Q1 bridge preparation.

The online path has already generated and validated the full v2 JSON response
before Qwen synthesis.  This module keeps that boundary: Gemma SAY-token
states are the complete Qwen prefill prefix, and only teacher-forced RVQ audio
inputs follow BOS.  It does not share the older V9-per-audio-query/Q2 trainer.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from duplex_kernel.qwen_conditioning import build_qwen_audio_conditioning_sequence


V5_PROFILE_VERSION = "synthetic-it-t6-audio-grounded-v5-plan.1"
V5_SOURCE = "authored_synthetic_audio_candidate"
V2_PROTOCOL = "heard_action_say_v2"
ACTION_ORDER = ("SPEAK", "BACKCHANNEL", "REPLAN")
ACOUSTIC_CODEBOOKS = 16
ACOUSTIC_CB0_VOCAB = 2048
MIN_EOS_FRAMES = 2
DEFAULT_TRAIN_QUOTA = {"SPEAK": 24, "BACKCHANNEL": 16, "REPLAN": 24}
DEFAULT_VALIDATION_QUOTA = {"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3}

_EXPECTED_TARGET_ROLES = {
    "SPEAK": {"assistant_start", "assistant_start_interruptible"},
    "BACKCHANNEL": {"assistant_backchannel"},
    "REPLAN": {"assistant_replan"},
}
_VOICE_REFERENCE_ROLES = {
    "assistant_start",
    "assistant_start_interruptible",
    "assistant_replan",
}


def _int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if converted != value:
        raise ValueError(f"{label} must be an exact integer")
    return converted


def _is_v5_record(row: Mapping[str, Any]) -> bool:
    return (
        row.get("source") == V5_SOURCE
        and row.get("profile") == "audio_grounded_v5"
        and row.get("profile_version") == V5_PROFILE_VERSION
        and row.get("audio_response_protocol") == V2_PROTOCOL
    )


def _workspace_path(path: Any) -> str:
    value = str(path or "").strip()
    if not value:
        raise ValueError("audio_path missing")
    if value.startswith("/workspace/"):
        return value
    if value.startswith("data/"):
        return f"/workspace/{value}"
    raise ValueError(f"audio_path must resolve under /workspace: {value}")


def _stable_rank(seed: int, row: Mapping[str, Any]) -> str:
    payload = "|".join(
        (
            str(seed),
            str(row.get("conversation_id", "")),
            str(row.get("action", "")),
            str(row.get("target_utterance_id", "")),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _select_action_quota(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    quotas: Mapping[str, int],
    seed: int,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    group_counts: Counter[str] = Counter()
    conversation_counts: Counter[str] = Counter()
    selected_ids: set[tuple[str, str]] = set()

    for action in ACTION_ORDER:
        quota = _int(quotas.get(action, 0), f"{split}.{action} quota")
        if quota < 0:
            raise ValueError(f"{split}.{action} quota must be non-negative")
        candidates = [
            row for row in rows
            if row.get("split") == split and row.get("action") == action
        ]
        candidates.sort(key=lambda row: _stable_rank(seed, row))
        for _ in range(quota):
            eligible = [
                row for row in candidates
                if (str(row.get("conversation_id")), str(row.get("target_utterance_id"))) not in selected_ids
                and conversation_counts[str(row.get("conversation_id"))] < 3
            ]
            if not eligible:
                raise ValueError(
                    f"insufficient {split}/{action} unique records for quota {quota}; "
                    f"available={len(candidates)} selected={sum(1 for item in selected if item.get('action') == action)}"
                )
            chosen = min(
                eligible,
                key=lambda row: (
                    group_counts[str(row.get("source_group_id", ""))],
                    conversation_counts[str(row.get("conversation_id", ""))],
                    _stable_rank(seed, row),
                ),
            )
            selected.append(chosen)
            selected_ids.add((str(chosen.get("conversation_id")), str(chosen.get("target_utterance_id"))))
            group_counts[str(chosen.get("source_group_id", ""))] += 1
            conversation_counts[str(chosen.get("conversation_id", ""))] += 1
            candidates.remove(chosen)
    return selected


def _target_and_voice_reference(
    record: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    action = str(record.get("action", ""))
    if action not in _EXPECTED_TARGET_ROLES:
        raise ValueError(f"unexpected target action: {action!r}")
    sample_rate = _int(raw.get("sample_rate"), "raw.sample_rate")
    if sample_rate != 16000 or raw.get("channels") != 2:
        raise ValueError("v5 target waveform must be stereo 16 kHz")
    channel_map = raw.get("channel_map")
    if not isinstance(channel_map, Mapping) or channel_map.get("0") != "user" or channel_map.get("1") != "assistant":
        raise ValueError("v5 channel_map must define user=0 and assistant=1")
    if str(raw.get("source")) != V5_SOURCE or str(raw.get("profile_version")) != V5_PROFILE_VERSION:
        raise ValueError("raw audio manifest is not the requested authored v5 source")
    if str(raw.get("conversation_id")) != str(record.get("conversation_id")):
        raise ValueError("record/raw conversation_id mismatch")
    if str(raw.get("topic_group_id")) != str(record.get("source_group_id")):
        raise ValueError("record/raw source_group_id mismatch")
    if str(raw.get("split")) != str(record.get("split")):
        raise ValueError("record/raw split mismatch")
    record_audio = _workspace_path(record.get("audio_path"))
    raw_audio = _workspace_path(raw.get("audio_path"))
    if record_audio != raw_audio:
        raise ValueError("record/raw audio_path mismatch")

    utterances = raw.get("utterances")
    if not isinstance(utterances, list):
        raise ValueError("raw utterances must be a list")
    target_id = str(record.get("target_utterance_id", ""))
    target_candidates = [
        item for item in utterances
        if isinstance(item, Mapping) and str(item.get("utterance_id")) == target_id
    ]
    if len(target_candidates) != 1:
        raise ValueError(f"target utterance id must identify exactly one raw utterance: {target_id!r}")
    target = target_candidates[0]
    if target.get("speaker") != "assistant" or target.get("role") not in _EXPECTED_TARGET_ROLES[action]:
        raise ValueError(f"action {action} does not match target assistant utterance role")
    say_text = str(record.get("target_say_text", "")).strip()
    if not say_text or str(target.get("text", "")).strip() != say_text:
        raise ValueError("target SAY text differs from authored assistant utterance")
    start = _int(target.get("start_sample"), "target.start_sample")
    end = _int(target.get("end_sample"), "target.end_sample")
    if start < 0 or end <= start:
        raise ValueError("target assistant waveform interval is empty or invalid")
    target_start_ms = float(record.get("target_start_ms"))
    target_end_ms = float(record.get("target_end_ms"))
    cutoff_ms = float(record.get("input_cutoff_ms"))
    input_end_ms = float(record.get("audio_end_ms"))
    if _int(record.get("audio_channel_index"), "input.audio_channel_index") != 0:
        raise ValueError("Gemma input must use the user channel 0")
    if abs(input_end_ms - cutoff_ms) > 1000.0 / sample_rate:
        raise ValueError("Gemma input audio span must end at its causal input cutoff")
    tolerance_ms = 1000.0 / sample_rate
    if target_start_ms + tolerance_ms < cutoff_ms:
        raise ValueError("target begins before input cutoff")
    if abs(target_start_ms - start * 1000.0 / sample_rate) > tolerance_ms:
        raise ValueError("target start_ms and raw assistant sample interval differ")
    if abs(target_end_ms - end * 1000.0 / sample_rate) > tolerance_ms:
        raise ValueError("target end_ms and raw assistant sample interval differ")

    target_duration_s = (end - start) / sample_rate
    target_audio = {
        "audio_path": raw_audio,
        "channel_index": 1,
        "sample_rate_hz": sample_rate,
        "start_sample": start,
        "end_sample": end,
        "duration_s": target_duration_s,
        # The encoder output is authoritative; this value is for preflight only.
        "estimated_rvq_frames": int(round(target_duration_s * 12.5)),
        "utterance_id": target_id,
    }

    references: list[tuple[int, str, Mapping[str, Any]]] = []
    fallback: list[tuple[int, str, Mapping[str, Any]]] = []
    for item in utterances:
        if not isinstance(item, Mapping) or item.get("speaker") != "assistant":
            continue
        if item.get("role") not in _VOICE_REFERENCE_ROLES:
            continue
        ref_start = _int(item.get("start_sample"), "voice_reference.start_sample")
        ref_end = _int(item.get("end_sample"), "voice_reference.end_sample")
        duration_samples = ref_end - ref_start
        if duration_samples <= 0:
            continue
        entry = (duration_samples, str(item.get("utterance_id", "")), item)
        fallback.append(entry)
        if duration_samples >= 2 * sample_rate and str(item.get("utterance_id")) != target_id:
            references.append(entry)
    pool = references or fallback
    if not pool:
        raise ValueError("conversation has no assistant speech suitable for x-vector conditioning")
    duration_samples, reference_id, reference = max(pool, key=lambda item: (item[0], item[1]))
    ref_start = _int(reference.get("start_sample"), "voice_reference.start_sample")
    ref_end = _int(reference.get("end_sample"), "voice_reference.end_sample")
    voice_reference = {
        "audio_path": raw_audio,
        "channel_index": 1,
        "sample_rate_hz": sample_rate,
        "start_sample": ref_start,
        "end_sample": ref_end,
        "duration_s": duration_samples / sample_rate,
        "utterance_id": reference_id,
        "mode": "x_vector_only",
        "text_or_ref_code_used": False,
    }
    return target_audio, voice_reference


def prepare_native_say_bridge_pilot(
    prepared_records_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_manifest_rows: Sequence[Mapping[str, Any]],
    *,
    train_action_quota: Mapping[str, int] = DEFAULT_TRAIN_QUOTA,
    validation_action_quota: Mapping[str, int] = DEFAULT_VALIDATION_QUOTA,
    seed: int = 260928,
) -> dict[str, Any]:
    """Select v5-only train/validation targets and materialize audio spans.

    The official test split is deliberately omitted. Topic-group split checks
    are run across every authored v5 record before any smaller pilot sampling.
    """
    raw_by_conversation: dict[str, Mapping[str, Any]] = {}
    for raw in raw_manifest_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("raw manifest rows must be objects")
        if raw.get("source") != V5_SOURCE or raw.get("profile_version") != V5_PROFILE_VERSION:
            continue
        conversation_id = str(raw.get("conversation_id", ""))
        if not conversation_id or conversation_id in raw_by_conversation:
            raise ValueError(f"raw manifest conversation id missing or duplicated: {conversation_id!r}")
        raw_by_conversation[conversation_id] = raw

    v5_by_split: dict[str, list[Mapping[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split in ("train", "validation", "test"):
        for row in prepared_records_by_split.get(split, ()):
            if not isinstance(row, Mapping) or not _is_v5_record(row):
                continue
            if row.get("split") != split:
                raise ValueError(f"prepared record split mismatch: file={split} row={row.get('split')!r}")
            v5_by_split[split].append(row)

    groups_by_split: dict[str, set[str]] = {split: set() for split in v5_by_split}
    raw_groups_by_split: dict[str, set[str]] = {split: set() for split in v5_by_split}
    for split, rows in v5_by_split.items():
        for row in rows:
            group = str(row.get("source_group_id", "")).strip()
            conversation_id = str(row.get("conversation_id", ""))
            if not group:
                raise ValueError("authored v5 record missing source_group_id")
            if conversation_id not in raw_by_conversation:
                raise ValueError(f"raw v5 manifest row missing for {conversation_id}")
            raw = raw_by_conversation[conversation_id]
            if raw.get("split") != split or str(raw.get("topic_group_id")) != group:
                raise ValueError("prepared/raw split or source_group_id mismatch")
            groups_by_split[split].add(group)
        for raw in raw_by_conversation.values():
            if raw.get("split") == split:
                raw_groups_by_split[split].add(str(raw.get("topic_group_id", "")))

    if raw_groups_by_split["train"] & raw_groups_by_split["validation"]:
        raise ValueError("authored v5 train/validation topic groups are not group-disjoint")
    if raw_groups_by_split["train"] & raw_groups_by_split["test"]:
        raise ValueError("authored v5 train/test topic groups are not group-disjoint")
    if raw_groups_by_split["validation"] & raw_groups_by_split["test"]:
        raise ValueError("authored v5 validation/test topic groups are not group-disjoint")
    for split in v5_by_split:
        if not groups_by_split[split].issubset(raw_groups_by_split[split]):
            raise ValueError(f"prepared {split} groups absent from raw manifest")

    selected = []
    for split, quota in (("train", train_action_quota), ("validation", validation_action_quota)):
        selected.extend(
            _select_action_quota(v5_by_split[split], split=split, quotas=quota, seed=seed)
        )

    materialized: list[dict[str, Any]] = []
    for row in selected:
        conversation_id = str(row["conversation_id"])
        raw = raw_by_conversation[conversation_id]
        target_audio, voice_reference = _target_and_voice_reference(row, raw)
        result = dict(row)
        result["sample_id"] = f"{conversation_id}:{row['target_utterance_id']}:{row['action']}"
        result["source_group_id"] = str(row["source_group_id"])
        result["target_audio"] = target_audio
        result["voice_reference"] = voice_reference
        result["rvq_frames_estimate_is_approximate"] = True
        materialized.append(result)

    counts = {
        split: {
            action: sum(1 for row in materialized if row["split"] == split and row["action"] == action)
            for action in ACTION_ORDER
        }
        for split in ("train", "validation")
    }
    expected_counts = {
        "train": {action: _int(train_action_quota.get(action, 0), f"train.{action}") for action in ACTION_ORDER},
        "validation": {action: _int(validation_action_quota.get(action, 0), f"validation.{action}") for action in ACTION_ORDER},
    }
    if counts != expected_counts:
        raise ValueError(f"pilot action counts differ from requested quotas: {counts!r} != {expected_counts!r}")
    return {
        "records": materialized,
        "report": {
            "schema": "native_say_v9_bridge_pilot_manifest_v1",
            "seed": int(seed),
            "source": V5_SOURCE,
            "profile_version": V5_PROFILE_VERSION,
            "official_test_included": False,
            "counts_by_split_action": counts,
            "groups_by_split": {split: sorted(groups_by_split[split]) for split in ("train", "validation")},
            "selected_group_counts": {
                split: len({row["source_group_id"] for row in materialized if row["split"] == split})
                for split in ("train", "validation")
            },
            "selected_conversation_counts": {
                split: len({row["conversation_id"] for row in materialized if row["split"] == split})
                for split in ("train", "validation")
            },
            "estimated_rvq_frames_by_split_action": {
                split: {
                    action: sum(
                        int(row["target_audio"]["estimated_rvq_frames"])
                        for row in materialized
                        if row["split"] == split and row["action"] == action
                    )
                    for action in ACTION_ORDER
                }
                for split in ("train", "validation")
            },
            "all_authored_v5_groups_by_split": {
                split: sorted(groups_by_split[split]) for split in groups_by_split
            },
            "all_authored_v5_record_counts_by_split": {
                split: len(v5_by_split[split]) for split in v5_by_split
            },
            "rvq_frame_estimate_note": "duration*12.5Hz is a preflight estimate; official tokenizer output is authoritative",
        },
    }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSONL row") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def write_native_say_bridge_pilot(
    result: Mapping[str, Any],
    *,
    manifest_path: str | Path,
    report_path: str | Path,
) -> None:
    manifest = Path(manifest_path)
    report = Path(report_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for row in result["records"]:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    report.write_text(
        json.dumps(result["report"], ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def say_hidden_input_positions(*, prompt_length: int, say_token_indices: Sequence[int]) -> list[int]:
    """Map output callback token indices to forward-input hidden positions.

    The production callback runs after the selected token's forward; therefore
    the generated token at index i is represented at input position prompt+i.
    """
    prompt = _int(prompt_length, "prompt_length")
    if prompt < 0:
        raise ValueError("prompt_length must be non-negative")
    indices = [_int(value, "say_token_index") for value in say_token_indices]
    if not indices or any(value < 0 for value in indices):
        raise ValueError("SAY token indices must be non-empty and non-negative")
    if indices != sorted(set(indices)):
        raise ValueError("SAY token indices must be strictly increasing")
    return [prompt + index for index in indices]


def collect_say_token_indices(
    tokenizer: Any,
    generated_token_ids: Sequence[int],
    *,
    expected_text: str,
    expected_action: str,
    expected_say: str | None = None,
) -> tuple[dict[str, str], list[int]]:
    """Use the runtime v2 parser/decode-prefix rule to locate SAY body tokens."""
    from duplex_kernel.audio_response_protocol import AudioResponseV2Collector

    collector = AudioResponseV2Collector(expected_action=expected_action)
    token_ids = [_int(value, "generated_token_id") for value in generated_token_ids]
    previous = ""
    for index, token_id in enumerate(token_ids):
        decoded = tokenizer.decode(token_ids[: index + 1], skip_special_tokens=True)
        if not isinstance(decoded, str) or not decoded.startswith(previous):
            raise ValueError("Gemma target token decode is not prefix-stable")
        fragment = decoded[len(previous) :]
        previous = decoded
        collector.feed_token(token_index=index, token_id=token_id, text=fragment, hidden=object())
    parsed = collector.finish(expected_text=expected_text)
    if parsed["action"] != expected_action:
        raise ValueError("teacher-forced response action differs from selected native label")
    if expected_say is not None and parsed["say"].strip() != str(expected_say).strip():
        raise ValueError("teacher-forced SAY differs from the authored target")
    if parsed["say"].strip() == "":
        raise ValueError("teacher-forced SAY body is empty")
    indices = [int(item.token_index) for item in collector.say_tokens]
    if not indices:
        raise ValueError("runtime response parser found no SAY body token indices")
    return parsed, indices


def differentiable_stateful_v9_prefix(
    h_raw: torch.Tensor,
    projector: Any,
    causal_conv: Any,
) -> torch.Tensor:
    """Batched differentiable form of production's causal-L0 V9 stream.

    Production consumes every SAY hidden state once, with the first state held
    as the fixed text anchor.  Causal convolution plus the projector's causal
    attention is numerically equivalent to incremental ``step`` while keeping
    gradients available for the projector during training.
    """
    raw_dim = int(getattr(getattr(projector, "audio_in", None), "in_features", -1))
    output_dim = int(getattr(getattr(projector, "audio_in", None), "out_features", -1))
    if h_raw.ndim != 3 or h_raw.shape[0] != 1 or h_raw.shape[1] <= 0 or h_raw.shape[2] != raw_dim:
        raise ValueError(f"h_raw must have shape (1,nonempty,{raw_dim})")
    parameter = next(projector.parameters())
    raw = h_raw.to(device=parameter.device, dtype=parameter.dtype)
    anchor = raw[:, 0, :]
    convolved = causal_conv(raw).to(dtype=parameter.dtype)
    result = projector(convolved, anchor)
    if result.shape != (1, int(raw.shape[1]), output_dim):
        raise RuntimeError(f"causal V9 output shape is unexpected: {tuple(result.shape)}")
    return result


def build_native_q1_training_embeddings(
    talker_wrapper: Any,
    *,
    h_pfx: torch.Tensor,
    codes: torch.Tensor,
    text: str,
    voice_conditioning: Mapping[str, Any],
) -> tuple[torch.Tensor, int, int]:
    """Build exact full SAY prefill followed by all teacher-forced RVQ inputs."""
    if h_pfx.ndim != 3 or h_pfx.shape[0] != 1 or h_pfx.shape[1] <= 0:
        raise ValueError("h_pfx must have shape (1,nonempty,2048)")
    if int(h_pfx.shape[-1]) != 2048:
        raise ValueError("h_pfx hidden size must match Qwen Talker 2048")
    if codes.ndim != 3 or tuple(codes.shape[:1]) != (1,) or int(codes.shape[-1]) != ACOUSTIC_CODEBOOKS:
        raise ValueError("codes must have shape (1,frames,16)")
    frame_count = int(codes.shape[1])
    if frame_count < MIN_EOS_FRAMES:
        raise ValueError(f"natural EOS training requires at least {MIN_EOS_FRAMES} real RVQ frames")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("target SAY text must be non-empty")
    model = talker_wrapper.talker
    dtype = h_pfx.dtype
    device = h_pfx.device
    icl, text_part, bos, tts_pad = talker_wrapper.build_prefill_parts(
        text.strip(), voice_conditioning=dict(voice_conditioning)
    )
    icl = icl.to(device=device, dtype=dtype)
    text_part = text_part.to(device=device, dtype=dtype)
    bos = bos.to(device=device, dtype=dtype)
    tts_pad = tts_pad.to(device=device, dtype=dtype)
    h_pfx = h_pfx.to(device=device, dtype=dtype)
    prefix = torch.cat((icl, text_part, h_pfx, bos), dim=1)
    code_inputs = codes.to(device=device, dtype=torch.long)
    audio_inputs = build_qwen_audio_conditioning_sequence(
        model,
        code_inputs,
        tts_pad_embed=tts_pad,
    ).to(dtype=dtype)
    if int(audio_inputs.shape[1]) != frame_count:
        raise RuntimeError("Qwen audio input count differs from the complete target RVQ frame count")
    embeddings = torch.cat((prefix, audio_inputs), dim=1)
    bos_position = int(prefix.shape[1]) - 1
    return embeddings, bos_position, frame_count


def native_q1_causal_mask(
    sequence_length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Standard causal Qwen mask: full SAY prefix, no future audio-code inputs."""
    length = _int(sequence_length, "sequence_length")
    if length <= 0:
        raise ValueError("sequence_length must be positive")
    mask = torch.zeros((1, 1, length, length), device=device, dtype=dtype)
    future = torch.triu(
        torch.ones((length, length), device=device, dtype=torch.bool), diagonal=1
    )
    return mask.masked_fill(future.view(1, 1, length, length), torch.finfo(dtype).min)


def teacher_forced_residual_logits(
    talker_wrapper: Any,
    last_hidden_per_frame: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    """Run Qwen's real autoregressive code predictor with target RVQ history.

    Frames are batched together, while the fifteen residual-codebook steps are
    advanced sequentially with the same codebook index convention as the
    production SubTalkerGraph. Frozen weights still propagate gradients from
    residual loss back through each frame's Qwen hidden state.
    """
    if last_hidden_per_frame.ndim == 3 and last_hidden_per_frame.shape[1] == 1:
        last_hidden_per_frame = last_hidden_per_frame[:, 0, :]
    if last_hidden_per_frame.ndim != 2 or int(last_hidden_per_frame.shape[-1]) != 2048:
        raise ValueError("last_hidden_per_frame must have shape (frames,2048)")
    if codes.ndim != 2 or tuple(codes.shape) != (int(last_hidden_per_frame.shape[0]), ACOUSTIC_CODEBOOKS):
        raise ValueError("residual target codes must have shape (frames,16)")
    if codes.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("RVQ targets must be integer tensors")
    if bool((codes < 0).any()) or bool((codes >= ACOUSTIC_CB0_VOCAB).any()):
        raise ValueError("RVQ targets must be in [0,2048)")

    model = talker_wrapper.talker
    predictor = getattr(model, "code_predictor", None)
    if predictor is None:
        raise RuntimeError("Qwen Talker code_predictor is missing")
    code0 = codes[:, 0:1].to(device=last_hidden_per_frame.device, dtype=torch.long)
    code0_embed = model.get_input_embeddings()(code0)
    prefill = torch.cat((last_hidden_per_frame.unsqueeze(1), code0_embed), dim=1)
    output = predictor(inputs_embeds=prefill, use_cache=True, return_dict=True)
    if output.past_key_values is None:
        raise RuntimeError("Qwen code_predictor prefill did not return a KV cache")
    logits_by_codebook = [output.logits[:, -1, :]]
    past = output.past_key_values
    for codebook_index in range(1, ACOUSTIC_CODEBOOKS - 1):
        previous_target = codes[:, codebook_index : codebook_index + 1].to(
            device=last_hidden_per_frame.device, dtype=torch.long
        )
        output = predictor(
            input_ids=previous_target,
            past_key_values=past,
            use_cache=True,
            generation_steps=codebook_index,
            return_dict=True,
        )
        if output.past_key_values is None:
            raise RuntimeError(f"Qwen code_predictor lost KV at residual group {codebook_index}")
        past = output.past_key_values
        logits_by_codebook.append(output.logits[:, -1, :])
    logits = torch.stack(logits_by_codebook, dim=1)
    if tuple(logits.shape) != (int(codes.shape[0]), ACOUSTIC_CODEBOOKS - 1, ACOUSTIC_CB0_VOCAB):
        raise RuntimeError(f"Qwen residual logits have unexpected shape: {tuple(logits.shape)}")
    return logits


def forward_native_say_q1(
    talker_wrapper: Any,
    *,
    h_pfx: torch.Tensor,
    codes: torch.Tensor,
    text: str,
    voice_conditioning: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return CB0/EOS and 15-book logits from a single exact native-Q1 sequence."""
    embeddings, bos_position, frame_count = build_native_q1_training_embeddings(
        talker_wrapper,
        h_pfx=h_pfx,
        codes=codes,
        text=text,
        voice_conditioning=voice_conditioning,
    )
    length = int(embeddings.shape[1])
    position = torch.arange(length, device=embeddings.device, dtype=torch.long).view(1, 1, length)
    position_ids = position.expand(3, 1, length)
    output = talker_wrapper.talker.model(
        inputs_embeds=embeddings,
        attention_mask=native_q1_causal_mask(length, device=embeddings.device, dtype=embeddings.dtype),
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    hidden = output.last_hidden_state
    q1_hidden = hidden[:, bos_position : bos_position + frame_count + 1, :]
    if int(q1_hidden.shape[1]) != frame_count + 1:
        raise RuntimeError("Qwen full-sequence output omitted an acoustic/EOS query")
    cb0_logits = talker_wrapper.talker.codec_head(q1_hidden).squeeze(0)
    residual_logits = teacher_forced_residual_logits(
        talker_wrapper,
        q1_hidden[:, :frame_count, :].squeeze(0),
        codes.squeeze(0),
    )
    return cb0_logits, residual_logits


def compute_native_say_bridge_loss(
    cb0_logits: torch.Tensor,
    residual_logits: torch.Tensor,
    codes: torch.Tensor,
    *,
    eos_token_id: int,
    acoustic_vocab_size: int = ACOUSTIC_CB0_VOCAB,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CB0 CE + one natural EOS CE + mean CE over all 15 real residual books."""
    if codes.ndim != 2 or int(codes.shape[1]) != ACOUSTIC_CODEBOOKS or int(codes.shape[0]) < MIN_EOS_FRAMES:
        raise ValueError(f"codes must have shape (frames>= {MIN_EOS_FRAMES},16)")
    frames = int(codes.shape[0])
    if acoustic_vocab_size != ACOUSTIC_CB0_VOCAB:
        raise ValueError("production bridge requires the 2048-code acoustic CB0 vocabulary")
    if cb0_logits.ndim != 2 or int(cb0_logits.shape[0]) != frames + 1:
        raise ValueError("CB0 logits must contain exactly frames+1 queries including EOS")
    if eos_token_id < acoustic_vocab_size or eos_token_id >= int(cb0_logits.shape[-1]):
        raise ValueError("codec EOS id must be in logits and outside acoustic CB0 support")
    if tuple(residual_logits.shape) != (frames, ACOUSTIC_CODEBOOKS - 1, acoustic_vocab_size):
        raise ValueError("residual logits must have shape (frames,15,2048)")
    if bool((codes < 0).any()) or bool((codes >= acoustic_vocab_size).any()):
        raise ValueError("all target RVQ codes must be in [0,2048)")

    allowed_cb0 = torch.cat(
        (cb0_logits[:, :acoustic_vocab_size], cb0_logits[:, eos_token_id : eos_token_id + 1]),
        dim=-1,
    ).float()
    cb0_targets = codes[:, 0].to(device=cb0_logits.device, dtype=torch.long)
    cb0_loss = F.cross_entropy(allowed_cb0[:frames], cb0_targets)
    eos_target = torch.full((1,), acoustic_vocab_size, device=cb0_logits.device, dtype=torch.long)
    eos_loss = F.cross_entropy(allowed_cb0[frames : frames + 1], eos_target)
    residual_targets = codes[:, 1:].to(device=residual_logits.device, dtype=torch.long)
    residual_loss = F.cross_entropy(
        residual_logits.float().reshape(-1, acoustic_vocab_size),
        residual_targets.reshape(-1),
    )
    total = cb0_loss + eos_loss + residual_loss
    return total, {
        "total_loss": float(total.detach().item()),
        "cb0_ce_loss": float(cb0_loss.detach().item()),
        "natural_eos_ce_loss": float(eos_loss.detach().item()),
        "residual_mean_ce_loss": float(residual_loss.detach().item()),
        "cb0_frames": float(frames),
        "eos_targets": 1.0,
        "residual_targets": float(frames * (ACOUSTIC_CODEBOOKS - 1)),
    }


__all__ = [
    "ACOUSTIC_CB0_VOCAB",
    "ACOUSTIC_CODEBOOKS",
    "DEFAULT_TRAIN_QUOTA",
    "DEFAULT_VALIDATION_QUOTA",
    "V5_PROFILE_VERSION",
    "build_native_q1_training_embeddings",
    "collect_say_token_indices",
    "compute_native_say_bridge_loss",
    "differentiable_stateful_v9_prefix",
    "forward_native_say_q1",
    "native_q1_causal_mask",
    "prepare_native_say_bridge_pilot",
    "read_jsonl",
    "say_hidden_input_positions",
    "teacher_forced_residual_logits",
    "write_native_say_bridge_pilot",
]
