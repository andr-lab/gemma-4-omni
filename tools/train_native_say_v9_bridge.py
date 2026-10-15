#!/usr/bin/env python3
"""Extract fresh Gemma/Qwen targets, then train the matched native-SAY Audio-Q1 V9 bridge.

This runner is intentionally separate from listener/head training.  ``extract``
uses the current Gemma audio response adapter to cache post-selected SAY-token
hidden states, then official Qwen Base APIs for real 12-Hz RVQ and speaker
x-vectors. ``train`` updates only the causal-L0 V9 projector; Gemma, Qwen,
the Qwen code predictor and the V9 causal convolution stay frozen.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duplex_kernel.native_say_v9_bridge import (
    ACTION_ORDER,
    V5_PROFILE_VERSION,
    V5_SOURCE,
    collect_say_token_indices,
    compute_native_say_bridge_loss,
    differentiable_stateful_v9_prefix,
    forward_native_say_q1,
    read_jsonl,
    say_hidden_input_positions,
)


DEFAULT_PILOT = Path("/workspace/data/native_say_v9_bridge_pilot_20261005/pilot_manifest.jsonl")
DEFAULT_FEATURE_DIR = Path("/workspace/data/native_say_v9_bridge_features_20261005")
DEFAULT_GEMMA_ADAPTER = Path(
    "/workspace/checkpoints/gemma4_audio_response_v2_v5_target_repair_200_kvshared_20261002/adapter"
)
DEFAULT_QWEN_BASE = Path(
    "/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3"
)
DEFAULT_OUTPUT_CKPT = Path("/workspace/checkpoints/native_say_v9_bridge_v5_pilot_20261005.pt")


def _sample_key(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id", ""))
    if not sample_id:
        raise ValueError("pilot row lacks sample_id")
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _load_pilot(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"pilot manifest is empty: {path}")
    if any(row.get("split") not in {"train", "validation"} for row in rows):
        raise ValueError("bridge pilot must omit the official test split")
    if any(
        row.get("source") != V5_SOURCE
        or row.get("profile_version") != V5_PROFILE_VERSION
        or row.get("audio_response_protocol") != "heard_action_say_v2"
        for row in rows
    ):
        raise ValueError("bridge pilot contains non-v5 or non-v2 records")
    if any(row.get("action") not in ACTION_ORDER for row in rows):
        raise ValueError("bridge pilot has an unsupported action")
    groups: dict[str, set[str]] = {"train": set(), "validation": set()}
    for row in rows:
        group = str(row.get("source_group_id", "")).strip()
        if not group:
            raise ValueError("pilot row lacks source_group_id")
        groups[str(row["split"])].add(group)
    if groups["train"] & groups["validation"]:
        raise ValueError("pilot train/validation groups overlap")
    return rows


def _write_torch_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _validate_cached_row(cache: Mapping[str, Any], row: Mapping[str, Any], *, schema: str) -> None:
    if cache.get("schema") != schema:
        raise ValueError(f"cached feature schema mismatch for {row.get('sample_id')}")
    for key in ("sample_id", "split", "action", "source_group_id", "target_say_text"):
        if cache.get(key) != row.get(key):
            raise ValueError(f"cached feature {key} mismatch for {row.get('sample_id')}; use a fresh feature directory")


def _extract_gemma_hidden(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    from duplex_kernel.duplex_control6 import merge_gemma_adapter_into_thinker
    from duplex_kernel.models.thinker import load_thinker, resolve_thinker_path
    from training.duplex_qlora import stage1

    torch.manual_seed(args.seed)
    thinker = load_thinker(args.workspace, device="cuda", dtype=torch.bfloat16)
    actual_base = merge_gemma_adapter_into_thinker(thinker, args.gemma_adapter)
    resolved_base = str(Path(resolve_thinker_path(args.workspace)).resolve())
    if str(Path(actual_base).resolve()) != resolved_base:
        raise RuntimeError(f"Gemma adapter merged onto unexpected base: {actual_base}")
    if thinker.processor is None:
        raise RuntimeError("Gemma audio processor is unavailable")
    model = thinker.model.eval()
    tokenizer = thinker.tokenizer
    output_dir = args.feature_dir / "gemma_hidden"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        out_path = output_dir / f"{_sample_key(row)}.pt"
        if out_path.is_file():
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            _validate_cached_row(cached, row, schema="native_say_v9_bridge_gemma_hidden_v1")
            if cached.get("gemma_base_path") != resolved_base or cached.get("gemma_adapter_dir") != str(Path(args.gemma_adapter).resolve()):
                raise ValueError(f"stale Gemma cache for {row['sample_id']}; use a fresh feature directory")
            continue
        if row.get("audio_mask") is not True or int(row.get("audio_channel_index", -1)) != 0:
            raise ValueError("Gemma hidden extraction requires causal user-channel audio input")
        inputs = stage1.build_training_inputs(thinker.processor, row)
        labels = inputs.get("labels")
        input_ids = inputs.get("input_ids")
        if not isinstance(labels, torch.Tensor) or not isinstance(input_ids, torch.Tensor):
            raise RuntimeError("stage1 builder did not return input_ids and target labels")
        target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
        if target_positions.numel() <= 0:
            raise ValueError("Gemma target is empty")
        prompt_length = int(target_positions[0])
        if not torch.equal(target_positions, torch.arange(prompt_length, labels.shape[1])):
            raise ValueError("Gemma labels are not one contiguous response suffix")
        target_ids = input_ids[0, prompt_length:].detach().cpu().tolist()
        generated_text = tokenizer.decode(target_ids, skip_special_tokens=True)
        if generated_text.strip() != str(row.get("target_text", "")).strip():
            raise ValueError("Gemma target token decode differs from authored v2 JSON")
        parsed, say_indices = collect_say_token_indices(
            tokenizer,
            target_ids,
            expected_text=generated_text,
            expected_action=str(row["action"]),
            expected_say=str(row["target_say_text"]),
        )
        positions = say_hidden_input_positions(
            prompt_length=prompt_length,
            say_token_indices=say_indices,
        )
        device = next(model.parameters()).device
        model_inputs = {
            key: value.to(device)
            for key, value in inputs.items()
            if key != "labels"
        }
        with torch.inference_mode():
            output = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
            )
            hidden = output.hidden_states[-1]
        if output.past_key_values is None:
            raise RuntimeError("Gemma teacher-forced pass did not create a fresh KV cache")
        if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[-1] != 1536:
            raise RuntimeError(f"Gemma hidden ABI mismatch: {tuple(hidden.shape)}")
        if hidden.shape[1] != input_ids.shape[1]:
            raise RuntimeError(
                "Gemma multimodal hidden positions do not align with input_ids; "
                "refusing to infer an audio-expansion offset"
            )
        if max(positions) >= hidden.shape[1]:
            raise RuntimeError("SAY hidden position is outside the Gemma output sequence")
        say_hidden = hidden[0, positions, :].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        if say_hidden.shape != (len(say_indices), 1536) or not bool(torch.isfinite(say_hidden).all()):
            raise RuntimeError("Gemma SAY hidden states have unexpected shape or values")
        _write_torch_atomic(out_path, {
            "schema": "native_say_v9_bridge_gemma_hidden_v1",
            "sample_id": str(row["sample_id"]),
            "source_group_id": str(row["source_group_id"]),
            "split": str(row["split"]),
            "action": str(row["action"]),
            "target_say_text": str(row["target_say_text"]),
            "parsed_response": parsed,
            "gemma_base_path": resolved_base,
            "gemma_adapter_dir": str(Path(args.gemma_adapter).resolve()),
            "say_token_indices": say_indices,
            "say_hidden_input_positions": positions,
            "prompt_length": prompt_length,
            "hidden_source": "production_callback_postselected_token_state",
            "h_raw": say_hidden,
        })
        if hasattr(stage1, "_TRAINING_INPUT_CACHE"):
            stage1._TRAINING_INPUT_CACHE.clear()
        del inputs, output, hidden, model_inputs
        if index % 8 == 0:
            print(f"gemma_hidden {index}/{total}", flush=True)
    del thinker, model
    gc.collect()
    torch.cuda.empty_cache()


def _read_channel_span(row: Mapping[str, Any], span: Mapping[str, Any]) -> np.ndarray:
    import soundfile as sf

    path = Path(str(span["audio_path"]))
    if not path.is_file():
        raise FileNotFoundError(f"audio file missing: {path}")
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if int(sample_rate) != 16000 or audio.shape[1] != 2:
        raise ValueError("v5 raw audio must remain stereo 16 kHz")
    channel = int(span["channel_index"])
    if channel not in (0, 1):
        raise ValueError("audio span channel must be 0 or 1")
    start, end = int(span["start_sample"]), int(span["end_sample"])
    if start < 0 or end <= start or end > len(audio):
        raise ValueError("audio span is outside the authored WAV")
    values = np.ascontiguousarray(audio[start:end, channel], dtype=np.float32)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("audio span is empty or non-finite")
    return values


def _resample_16k_to_24k(values: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly

    return np.ascontiguousarray(resample_poly(values, 3, 2).astype(np.float32, copy=False))


def _extract_qwen_features(rows: Sequence[Mapping[str, Any]], args: argparse.Namespace) -> None:
    from duplex_kernel.models.talker import _install_shims

    _install_shims(str(args.workspace))
    try:
        from qwen_tts import Qwen3TTSModel
    except Exception as exc:
        raise RuntimeError("installed qwen-tts lacks official Qwen3TTSModel API") from exc

    model = Qwen3TTSModel.from_pretrained(
        str(args.qwen_base_path),
        device_map="cuda:0",
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    qwen = model.model
    speech_tokenizer = getattr(qwen, "speech_tokenizer", None)
    if speech_tokenizer is None or not callable(getattr(speech_tokenizer, "encode", None)):
        raise RuntimeError("loaded Qwen Base model does not expose its official speech_tokenizer.encode")
    if not callable(getattr(qwen, "extract_speaker_embedding", None)):
        raise RuntimeError("loaded Qwen Base model lacks extract_speaker_embedding")

    output_dir = args.feature_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        out_path = output_dir / f"{_sample_key(row)}.pt"
        if out_path.is_file():
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            _validate_cached_row(cached, row, schema="native_say_v9_bridge_feature_v1")
            if cached.get("qwen_base_path") != str(args.qwen_base_path.resolve()) or cached.get("gemma_adapter_dir") != str(Path(args.gemma_adapter).resolve()):
                raise ValueError(f"stale Qwen/Gemma cache for {row['sample_id']}; use a fresh feature directory")
            continue
        hidden_path = args.feature_dir / "gemma_hidden" / f"{_sample_key(row)}.pt"
        if not hidden_path.is_file():
            raise FileNotFoundError(f"run extract --phase gemma first: {hidden_path}")
        hidden_cache = torch.load(hidden_path, map_location="cpu", weights_only=False)
        _validate_cached_row(hidden_cache, row, schema="native_say_v9_bridge_gemma_hidden_v1")
        if hidden_cache.get("gemma_adapter_dir") != str(Path(args.gemma_adapter).resolve()):
            raise ValueError(f"Gemma hidden cache adapter differs for {row['sample_id']}")
        target_audio = _read_channel_span(row, row["target_audio"])
        voice_audio = _read_channel_span(row, row["voice_reference"])
        if int(row["target_audio"]["channel_index"]) != 1 or int(row["voice_reference"]["channel_index"]) != 1:
            raise ValueError("Qwen target and x-vector reference must use assistant channel 1")
        if len(voice_audio) < 2 * 16000:
            raise ValueError("x-vector reference must contain at least two seconds of real assistant speech")
        target_encoded = speech_tokenizer.encode(target_audio, sr=16000)
        code_list = getattr(target_encoded, "audio_codes", None)
        if not isinstance(code_list, (list, tuple)) or len(code_list) != 1:
            raise RuntimeError("Qwen 12-Hz encode did not return one real audio_codes sequence")
        codes = torch.as_tensor(code_list[0]).detach().to(device="cpu", dtype=torch.long).contiguous()
        if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] < 2:
            raise RuntimeError(f"Qwen RVQ output must be (frames>=2,16), got {tuple(codes.shape)}")
        if bool((codes < 0).any()) or bool((codes >= 2048).any()):
            raise RuntimeError("Qwen RVQ codes are outside the 2048-code acoustic vocabulary")
        ref_24k = _resample_16k_to_24k(voice_audio)
        speaker = qwen.extract_speaker_embedding(audio=ref_24k, sr=24000)
        speaker = torch.as_tensor(speaker).detach().to(device="cpu", dtype=torch.float32).reshape(-1).contiguous()
        if speaker.numel() != 2048 or not bool(torch.isfinite(speaker).all()):
            raise RuntimeError(f"Qwen x-vector ABI mismatch: {tuple(speaker.shape)}")
        _write_torch_atomic(out_path, {
            "schema": "native_say_v9_bridge_feature_v1",
            "sample_id": str(row["sample_id"]),
            "source_group_id": str(row["source_group_id"]),
            "split": str(row["split"]),
            "action": str(row["action"]),
            "target_say_text": str(row["target_say_text"]),
            "h_raw": hidden_cache["h_raw"],
            "gemma_base_path": hidden_cache["gemma_base_path"],
            "gemma_adapter_dir": hidden_cache["gemma_adapter_dir"],
            "rvq_codes": codes,
            "speaker_embedding": speaker,
            "voice_conditioning": {
                "mode": "x_vector_only",
                "reference_utterance_id": str(row["voice_reference"]["utterance_id"]),
                "reference_duration_s": float(row["voice_reference"]["duration_s"]),
                "text_or_ref_code_used": False,
                "speaker_encoder_sample_rate_hz": 24000,
            },
            "target_audio": dict(row["target_audio"]),
            "qwen_base_path": str(args.qwen_base_path.resolve()),
            "rvq_encoder": "Qwen3TTSModel.model.speech_tokenizer.encode",
            "rvq_frames": int(codes.shape[0]),
        })
        del target_encoded, target_audio, voice_audio, hidden_cache
        if index % 8 == 0:
            print(f"qwen_features {index}/{total}", flush=True)
    del model, qwen, speech_tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def _feature_rows(
    rows: Sequence[Mapping[str, Any]],
    feature_dir: Path,
    *,
    gemma_adapter: Path,
    qwen_base_path: Path,
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        path = feature_dir / "features" / f"{_sample_key(row)}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"missing freshly extracted feature cache: {path}")
        feature = torch.load(path, map_location="cpu", weights_only=False)
        if feature.get("schema") != "native_say_v9_bridge_feature_v1":
            raise ValueError("feature cache schema mismatch")
        for key in ("sample_id", "split", "action", "source_group_id", "target_say_text"):
            if feature.get(key) != row.get(key):
                raise ValueError(f"feature cache {key} mismatch")
        if feature["h_raw"].shape[-1] != 1536 or feature["rvq_codes"].shape[-1] != 16:
            raise ValueError("feature cache hidden/RVQ ABI mismatch")
        if feature.get("gemma_adapter_dir") != str(gemma_adapter.resolve()):
            raise ValueError(f"feature cache Gemma adapter is stale for {row['sample_id']}")
        if feature.get("qwen_base_path") != str(qwen_base_path.resolve()):
            raise ValueError(f"feature cache Qwen Base is stale for {row['sample_id']}")
        result[str(row["sample_id"])] = feature
    return result


def _evaluate_validation(rows: Sequence[Mapping[str, Any]], feature_map: Mapping[str, Mapping[str, Any]], projector: Any, causal_conv: Any, talker: Any, eos_id: int, device: torch.device) -> dict[str, Any]:
    totals: list[float] = []
    by_action: dict[str, list[float]] = {action: [] for action in ACTION_ORDER}
    for row in rows:
        feature = feature_map[str(row["sample_id"])]
        h_raw = feature["h_raw"].to(device=device, dtype=torch.float32).unsqueeze(0)
        codes = feature["rvq_codes"].to(device=device, dtype=torch.long).unsqueeze(0)
        voice = {"mode": "x_vector_only", "speaker_embedding": feature["speaker_embedding"]}
        with torch.no_grad():
            h_pfx = differentiable_stateful_v9_prefix(h_raw, projector, causal_conv)
        qwen_dtype = next(talker.talker.parameters()).dtype
        cb0, residual = forward_native_say_q1(
            talker, h_pfx=h_pfx.to(dtype=qwen_dtype), codes=codes,
            text=str(row["target_say_text"]), voice_conditioning=voice,
        )
        loss, stats = compute_native_say_bridge_loss(cb0, residual, codes[0], eos_token_id=eos_id)
        value = float(loss.detach().cpu())
        totals.append(value)
        by_action[str(row["action"])].append(value)
    return {
        "mean_total_loss": float(np.mean(totals)),
        "mean_total_loss_by_action": {key: (float(np.mean(values)) if values else None) for key, values in by_action.items()},
        "support_by_action": {key: len(values) for key, values in by_action.items()},
    }


def _train(args: argparse.Namespace, rows: Sequence[Mapping[str, Any]]) -> None:
    from duplex_kernel.models.talker import load_talker
    from duplex_kernel.models.v9 import load_v9_conditioning

    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows or len(train_rows) < 3:
        raise ValueError("training and held-out validation rows are both required")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("official test rows must not be used in this pilot")
    feature_map = _feature_rows(
        rows,
        args.feature_dir,
        gemma_adapter=args.gemma_adapter,
        qwen_base_path=args.qwen_base_path,
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    projector, causal_conv = load_v9_conditioning(profile="causal_l0", device="cuda:0", dtype=torch.bfloat16)
    projector = projector.to(device=device, dtype=torch.float32).train()
    causal_conv = causal_conv.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in projector.parameters():
        parameter.requires_grad_(True)
    for parameter in causal_conv.parameters():
        parameter.requires_grad_(False)
    talker = load_talker(args.workspace, device="cuda:0", load_code2wav=False)
    talker.talker.eval()
    for parameter in talker.talker.parameters():
        parameter.requires_grad_(False)
    eos_id = getattr(talker.tcfg, "codec_eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(talker.tcfg, "codec_eos_id", None)
    if isinstance(eos_id, bool) or not isinstance(eos_id, int) or eos_id < 2048 or eos_id >= int(talker.talker.codec_head.out_features):
        raise RuntimeError(f"Qwen codec EOS ID is invalid: {eos_id!r}")
    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.learning_rate, weight_decay=0.0)
    by_action = {action: [row for row in train_rows if row["action"] == action] for action in ACTION_ORDER}
    if any(not by_action[action] for action in ACTION_ORDER):
        raise ValueError("balanced class sampler needs actual examples for SPEAK/BACKCHANNEL/REPLAN")
    action_cycle = list(ACTION_ORDER)
    random.shuffle(action_cycle)
    selected_counts = {action: 0 for action in ACTION_ORDER}
    best_val = math.inf
    best_metrics: dict[str, Any] | None = None
    output_path = args.checkpoint
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite candidate checkpoint: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_path.with_suffix(output_path.suffix + ".jsonl")
    if trace_path.exists():
        raise FileExistsError(f"refusing to overwrite training trace: {trace_path}")

    def save_checkpoint(step: int, validation: Mapping[str, Any]) -> None:
        payload = {
            "bridge_format": "native_say_v9_qwen_audio_q1_v1",
            "proj": {key: value.detach().cpu() for key, value in projector.state_dict().items()},
            "causal_conv": {key: value.detach().cpu() for key, value in causal_conv.conv.state_dict().items()},
            "step": int(step),
            "validation": dict(validation),
            "metadata": {
                "condition_prefix": "Qwen Base eager [ICL|target text|V9(postselected SAY states)|BOS]",
                "audio_decoder": "Audio-Q1, all real teacher-forced 16-codebook RVQ frames then natural EOS",
                "v9_profile": str(getattr(projector, "_v9_profile", "causal_l0")),
                "v9_recipe": str(getattr(projector, "_v9_recipe", "stateful_first_anchor_incremental_v1")),
                "gemma_adapter": str(args.gemma_adapter),
                "qwen_base_path": str(args.qwen_base_path),
                "pilot_manifest": str(args.manifest),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "selected_train_counts_by_action": dict(selected_counts),
                "trainable_parameters": "V9 projector only",
                "frozen_parameters": "Gemma; Qwen Talker; Qwen residual code predictor; V9 causal convolution",
                "official_test_used": False,
            },
        }
        temporary = output_path.with_name(output_path.name + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, output_path)

    # Record the initial held-out state, then checkpoint the best validation mean.
    initial_val = _evaluate_validation(validation_rows, feature_map, projector, causal_conv, talker, int(eos_id), device)
    best_val = initial_val["mean_total_loss"]
    best_metrics = initial_val
    save_checkpoint(0, initial_val)
    with trace_path.open("a", encoding="utf-8") as trace:
        trace.write(json.dumps({"step": 0, "validation": initial_val}, allow_nan=False) + "\n")
        for step in range(1, args.steps + 1):
            action = action_cycle[(step - 1) % len(action_cycle)]
            if step % len(action_cycle) == 1 and step > 1:
                random.shuffle(action_cycle)
                action = action_cycle[0]
            row = random.choice(by_action[action])
            selected_counts[action] += 1
            feature = feature_map[str(row["sample_id"])]
            raw = feature["h_raw"].to(device=device, dtype=torch.float32).unsqueeze(0)
            codes = feature["rvq_codes"].to(device=device, dtype=torch.long).unsqueeze(0)
            voice = {"mode": "x_vector_only", "speaker_embedding": feature["speaker_embedding"]}
            h_pfx = differentiable_stateful_v9_prefix(raw, projector, causal_conv)
            qwen_dtype = next(talker.talker.parameters()).dtype
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cb0, residual = forward_native_say_q1(
                    talker, h_pfx=h_pfx.to(dtype=qwen_dtype), codes=codes,
                    text=str(row["target_say_text"]), voice_conditioning=voice,
                )
                loss, component = compute_native_say_bridge_loss(
                    cb0, residual, codes[0], eos_token_id=int(eos_id)
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), args.max_grad_norm)
            if not torch.isfinite(torch.as_tensor(grad_norm)):
                raise RuntimeError("V9 projector gradient norm is non-finite")
            optimizer.step()
            trace.write(json.dumps({
                "step": step,
                "action": action,
                "sample_id": str(row["sample_id"]),
                "loss": component,
                "grad_norm": float(grad_norm.detach().cpu()),
            }, allow_nan=False) + "\n")
            if step % args.eval_every == 0 or step == args.steps:
                validation = _evaluate_validation(validation_rows, feature_map, projector, causal_conv, talker, int(eos_id), device)
                trace.write(json.dumps({"step": step, "validation": validation}, allow_nan=False) + "\n")
                trace.flush()
                print(json.dumps({"step": step, "validation": validation, "selected": selected_counts}, allow_nan=False), flush=True)
                if validation["mean_total_loss"] < best_val:
                    best_val = validation["mean_total_loss"]
                    best_metrics = validation
                    save_checkpoint(step, validation)
    print(json.dumps({"checkpoint": str(output_path), "best_validation": best_metrics, "selected_train_counts_by_action": selected_counts}, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract fresh Gemma SAY states, real Qwen RVQ and x-vectors")
    extract.add_argument("--manifest", type=Path, default=DEFAULT_PILOT)
    extract.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    extract.add_argument("--phase", choices=("all", "gemma", "qwen"), default="all")
    extract.add_argument("--gemma-adapter", type=Path, default=DEFAULT_GEMMA_ADAPTER)
    extract.add_argument("--qwen-base-path", type=Path, default=DEFAULT_QWEN_BASE)
    extract.add_argument("--workspace", type=Path, default=Path("/workspace"))
    extract.add_argument("--seed", type=int, default=260928)

    train = subparsers.add_parser("train", help="train only V9 on cached native-SAY/Qwen-Q1 examples")
    train.add_argument("--manifest", type=Path, default=DEFAULT_PILOT)
    train.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    train.add_argument("--gemma-adapter", type=Path, default=DEFAULT_GEMMA_ADAPTER)
    train.add_argument("--qwen-base-path", type=Path, default=DEFAULT_QWEN_BASE)
    train.add_argument("--workspace", type=Path, default=Path("/workspace"))
    train.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT_CKPT)
    train.add_argument("--steps", type=int, default=200)
    train.add_argument("--eval-every", type=int, default=25)
    train.add_argument("--learning-rate", type=float, default=1e-5)
    train.add_argument("--max-grad-norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=260928)

    args = parser.parse_args()
    if args.command == "extract":
        rows = _load_pilot(args.manifest)
        args.feature_dir.mkdir(parents=True, exist_ok=True)
        if args.phase in {"all", "gemma"}:
            _extract_gemma_hidden(rows, args)
        if args.phase in {"all", "qwen"}:
            _extract_qwen_features(rows, args)
        print(f"fresh_features={len(list((args.feature_dir / 'features').glob('*.pt')))} dir={args.feature_dir}")
        return 0
    if args.steps <= 0 or args.eval_every <= 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0:
        parser.error("steps, eval-every, learning-rate and max-grad-norm must be positive")
    rows = _load_pilot(args.manifest)
    _train(args, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
