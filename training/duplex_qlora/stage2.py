"""Post-merge native listener feature extraction and six-action head training."""
from __future__ import annotations

import json
import hashlib
import os
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .control import ControlHead, load_control_head
from .features import FEATURE_CONTRACT, last_audio_query, listener_feature_contract
from .prepare import read_jsonl
from .schema import ACTIONS


def load_merged_base(
    base_model_path: str,
    adapter_dir: str,
    *,
    device: str = "cuda",
) -> tuple[Any, Any, Any]:
    """Load BF16 Gemma, merge the final PEFT adapter, and return its processor.

    This must run before GemmaListener creates NativeGemmaListenerLM, whose
    C ABI borrows the language model's parameter pointers directly.
    """
    if not Path(adapter_dir).is_dir():
        raise FileNotFoundError(f"adapter LoRA finale mancante: {adapter_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("estrazione listener stage2 richiede il runtime CUDA del kernel")
    try:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("stage2 richiede transformers e peft nel runtime CUDA") from exc
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype=torch.bfloat16,
            device_map={"": device},
        )
        processor = AutoProcessor.from_pretrained(base_model_path)
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False)
        model = model.merge_and_unload()
    except Exception as exc:
        raise RuntimeError(
            "BF16 base + PEFT merge fallito prima del binding nativo: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    base_checkpoint_path = _attest_merged_model_source(model, base_model_path)
    floating = [parameter for parameter in model.parameters() if parameter.is_floating_point()]
    if not floating or any(parameter.dtype != torch.bfloat16 for parameter in floating):
        raise RuntimeError("Gemma merged runtime richiede tutti i parametri floating BF16")
    if any("lora_" in name for name, _ in model.named_parameters()):
        raise RuntimeError("merge_and_unload ha lasciato parametri PEFT prima del listener nativo")
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    from duplex_kernel.models.thinker import ThinkerModel

    thinker = ThinkerModel(model, tokenizer, processor, device=device)
    setattr(model, "_duplex_qlora_merged_adapter_dir", str(Path(adapter_dir).resolve()))
    return thinker, processor, model


def _attest_merged_model_source(model: Any, base_model_path: str) -> str:
    """Preserve the original BF16 base path on the nested LM borrowed by native CUDA."""
    nested = getattr(model, "model", None)
    language_model = getattr(nested, "language_model", None)
    if nested is None or language_model is None:
        raise RuntimeError("Gemma merged model non espone model.language_model per il listener nativo")
    resolved = str(Path(base_model_path).resolve())
    for module in (model, nested, language_model):
        setattr(module, "_gemma4_checkpoint_path", resolved)
    return resolved


def _channel_index(conversation: Mapping[str, Any], role: str) -> int | None:
    channel_map = conversation.get("channel_map") or {}
    values = [int(channel) for channel, value in channel_map.items() if value == role and str(channel).isdigit()]
    return values[0] if values else None


def _is_synthetic_source(row: Mapping[str, Any]) -> bool:
    """Treat an explicit authored-V5 source/profile as synthetic even if a legacy flag is absent."""
    return bool(row.get("synthetic")) or (
        row.get("source") == "authored_synthetic_audio_candidate"
        or row.get("profile") == "audio_grounded_v5"
    )


def _causal_action_targets(
    conversation: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], bool]:
    """Normalize v4 frame labels and v5 input-t/target-t+1 frame targets.

    Legacy ``frames`` keep their existing same-index contract. Presence of
    ``frame_targets`` selects the audio-grounded causal contract, independent
    of source/profile names.
    """
    is_v5 = "frame_targets" in conversation
    key = "frame_targets" if is_v5 else "frames"
    raw_targets = conversation.get(key, ()) or ()
    if not isinstance(raw_targets, (list, tuple)):
        raise ValueError(f"{conversation.get('conversation_id', 'unknown')}: {key} must be a sequence")
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    frame_samples = 1280
    v5_onset_action_by_target_frame: dict[int, str] = {}
    if is_v5:
        for event in conversation.get("event_schedule", ()) or ():
            if not isinstance(event, Mapping):
                continue
            action = event.get("action")
            start_sample = event.get("start_sample")
            if action not in ACTIONS or isinstance(start_sample, bool) or not isinstance(start_sample, int):
                continue
            if start_sample < 0 or start_sample % frame_samples:
                raise ValueError("audio-grounded event onset is not aligned to a target frame sample")
            target_start = start_sample // frame_samples
            previous_action = v5_onset_action_by_target_frame.setdefault(target_start, str(action))
            if previous_action != action:
                raise ValueError("audio-grounded events have conflicting event-onset actions")
    for index, raw in enumerate(raw_targets):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{key}[{index}] must be an object")
        frame = dict(raw)
        if is_v5:
            input_id = frame.get("input_frame_id")
            target_id = frame.get("target_frame_id")
            if isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
                raise ValueError("input_frame_id must be a non-negative integer")
            if isinstance(target_id, bool) or not isinstance(target_id, int) or target_id != input_id + 1:
                raise ValueError("target_frame_id must equal input_frame_id + 1")
            declared_frame_id = frame.get("frame_id", input_id)
            if (
                isinstance(declared_frame_id, bool)
                or not isinstance(declared_frame_id, int)
                or declared_frame_id != input_id
            ):
                raise ValueError("frame_id must equal input_frame_id for audio-grounded targets")
            expected_samples = {
                "input_audio_start_sample": input_id * frame_samples,
                "input_audio_end_sample": (input_id + 1) * frame_samples,
                "target_action_start_sample": (input_id + 1) * frame_samples,
                "target_action_end_sample": (input_id + 2) * frame_samples,
            }
            for field, expected in expected_samples.items():
                value = frame.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value != expected:
                    raise ValueError(f"{field} is not aligned to input_frame_id/target_frame_id")
            if frame.get("input_text_visible") is not False:
                raise ValueError("input_text_visible must be false for audio-grounded targets")
            if frame.get("future_audio_excluded") is not True:
                raise ValueError("future_audio_excluded must be true for causal targets")
            if type(frame.get("target_action_mask")) is not bool:
                raise ValueError("target_action_mask must be boolean")
            frame["frame_id"] = input_id
            frame["target_action"] = frame.get("action")
            frame["action_mask"] = frame["target_action_mask"]
            event_action = v5_onset_action_by_target_frame.get(target_id)
            if event_action is not None and frame["target_action"] != event_action:
                raise ValueError("audio-grounded event onset action disagrees with its target frame")
            frame["event_onset"] = event_action is not None
        else:
            input_id = frame.get("frame_id", -1)
            target_id = frame.get("target_frame_id", input_id)
            frame["input_frame_id"] = input_id
            frame.setdefault("target_frame_id", target_id)
            frame["event_onset"] = frame.get("event_onset") is True
        if isinstance(input_id, bool) or not isinstance(input_id, int) or input_id < 0:
            raise ValueError("stage2 target input frame id must be a non-negative integer")
        if input_id in seen:
            raise ValueError(f"duplicate stage2 input frame id: {input_id}")
        seen.add(input_id)
        action = frame.get("target_action")
        mask = frame.get("action_mask")
        if is_v5 and action not in ACTIONS:
            raise ValueError(f"frame_targets[{index}] has an action outside the six-class ABI")
        if mask is not True or action not in ACTIONS:
            continue
        frame["_stage2_is_v5_target"] = is_v5
        result.append(frame)
    result.sort(key=lambda frame: int(frame["frame_id"]))
    return result, is_v5


def _audio_frames(
    conversation: Mapping[str, Any],
    *,
    frame_samples: int,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return 16 kHz channel frames and compact valid-sample prefix masks.

    ``scipy.signal.resample_poly`` upsamples, applies a zero-phase low-pass FIR,
    then downsamples; see its official reference:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html.
    A count per frame marks real samples; any partial-frame suffix is zero padded.
    """
    import soundfile as sf

    audio_path = conversation.get("audio_path")
    if not audio_path or not Path(str(audio_path)).is_file():
        raise FileNotFoundError(f"audio conversation stage2 mancante: {audio_path}")
    resolved_path = str(Path(str(audio_path)).resolve())
    cached = pcm_cache.get(resolved_path) if pcm_cache is not None else None
    source_sample_rate = None
    if cached is None:
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        sample_rate = int(sample_rate)
        source_sample_rate = sample_rate
        if sample_rate <= 0:
            raise ValueError(f"{audio_path}: sample rate audio non valido: {sample_rate}")
        if audio.shape[0] and sample_rate != 16000:
            from math import gcd

            try:
                from scipy.signal import resample_poly
            except ImportError as exc:
                raise RuntimeError("stage2 richiede scipy.signal.resample_poly per audio non a 16 kHz") from exc
            common = gcd(sample_rate, 16000)
            audio = resample_poly(
                audio,
                up=16000 // common,
                down=sample_rate // common,
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            audio = np.asarray(audio, dtype=np.float32)
        sample_rate = 16000
        if pcm_cache is not None:
            # The caller scopes this cache to one physical source group so
            # paired role views reuse one decode/resample without corpus caching.
            pcm_cache[resolved_path] = (audio, 16000)
    else:
        audio, sample_rate = cached
    if sample_rate != 16000:
        raise RuntimeError(f"{audio_path}: PCM cache non normalizzata a 16 kHz")
    if audio.ndim != 2 or audio.shape[1] <= 0:
        raise ValueError(f"{audio_path}: PCM deve essere [samples, channels]")
    user_channel = _channel_index(conversation, "user")
    assistant_channel = _channel_index(conversation, "assistant")
    if user_channel is None or user_channel >= audio.shape[1]:
        raise ValueError(f"{audio_path}: channel_map non assegna un canale user esplicito")
    if assistant_channel is not None and assistant_channel >= audio.shape[1]:
        raise ValueError(f"{audio_path}: channel_map assistant fuori range")
    if "frame_targets" in conversation:
        declared_sample_rate = conversation.get("sample_rate_hz", conversation.get("sample_rate", 16000))
        if (
            (source_sample_rate is not None and source_sample_rate != 16000)
            or int(declared_sample_rate) != 16000
            or audio.shape[1] != 2
        ):
            raise ValueError(f"{audio_path}: audio-grounded frames require stereo 16 kHz PCM")
        if user_channel != 0 or assistant_channel != 1:
            raise ValueError(f"{audio_path}: audio-grounded channels require user=0 and assistant=1")
    total_frames = max(1, (audio.shape[0] + frame_samples - 1) // frame_samples)
    user = np.zeros((total_frames, frame_samples), dtype=np.float32)
    assistant = None if assistant_channel is None else np.zeros_like(user)
    valid_samples = np.zeros(total_frames, dtype=np.int32)
    for frame_index in range(total_frames):
        start = frame_index * frame_samples
        end = min(audio.shape[0], start + frame_samples)
        valid_count = max(0, end - start)
        valid_samples[frame_index] = valid_count
        if valid_count:
            user[frame_index, :valid_count] = audio[start:end, user_channel]
        if assistant is not None:
            if valid_count:
                assistant[frame_index, :valid_count] = audio[start:end, assistant_channel]
    return user, assistant, valid_samples


def extract_conversation_features(
    listener: Any,
    conversation: Mapping[str, Any],
    *,
    session_id: str | None = None,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
    timing: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Replay one causal conversation stream and keep only supervised frames."""
    from duplex_kernel.listener import DEFAULT_CONTEXT_FRAMES, GemmaListenerState
    from duplex_kernel.reference_buffer import AssistantReferenceFrame

    if getattr(listener, "runtime_profile", None) != "production":
        raise RuntimeError("stage2 usa solo GemmaListener runtime_profile=production")
    if not bool(getattr(listener.backend, "_use_native_lm", False)):
        raise RuntimeError("stage2 richiede il listener LM nativo del kernel")
    tick_ms = int(conversation.get("tick_ms", 80))
    if tick_ms != 80:
        raise ValueError(f"stage2 richiede frame Gemma da 80 ms, ricevuto {tick_ms}")
    frame_samples = 16000 * tick_ms // 1000
    target_rows, is_v5 = _causal_action_targets(conversation)
    sparse_targets: dict[int, Mapping[str, Any]] = {
        int(frame["frame_id"]): frame for frame in target_rows
    }
    if not sparse_targets:
        return np.empty((0, 1536), dtype=np.float32), []
    clusters = _conversation_replay_clusters(list(sparse_targets.values()))
    if timing is not None:
        timing["conversation_replays"] = int(timing.get("conversation_replays", 0)) + len(clusters)
        causal_steps = sum(int(cluster_targets[-1]["frame_id"]) - cluster_start + 1 for cluster_start, cluster_targets in clusters)
        timing["causal_input_frame_steps"] = int(timing.get("causal_input_frame_steps", 0)) + causal_steps
        timing.setdefault("conversation_replay_input_frames", []).extend(
            int(cluster_targets[-1]["frame_id"]) - cluster_start + 1
            for cluster_start, cluster_targets in clusters
        )
    decode_start = time.perf_counter()
    user_frames, assistant_frames, valid_samples = _audio_frames(
        conversation, frame_samples=frame_samples, pcm_cache=pcm_cache
    )
    if timing is not None:
        timing["audio_decode_resample_frame_pack_seconds"] = float(
            timing.get("audio_decode_resample_frame_pack_seconds", 0.0)
        ) + (time.perf_counter() - decode_start)
    if any(frame_id >= len(user_frames) for frame_id in sparse_targets):
        raise ValueError(f"{conversation['conversation_id']}: frame_id supervisionato oltre la durata del PCM")
    if is_v5:
        actual_samples = max(0, (len(valid_samples) - 1) * frame_samples + int(valid_samples[-1]))
        if any(
            int(frame["target_action_end_sample"]) > actual_samples
            for frame in (conversation.get("frame_targets", ()) or ())
        ):
            raise ValueError(f"{conversation['conversation_id']}: target action oltre la durata del PCM")
    features: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    for cluster_index, (cluster_start, cluster_targets) in enumerate(clusters):
        target_by_id = {int(frame["frame_id"]): frame for frame in cluster_targets}
        last_target_id = int(cluster_targets[-1]["frame_id"])
        state_create_start = time.perf_counter()
        state = GemmaListenerState(
            sample_rate=16000,
            frame_duration_ms=80,
            context_frames=DEFAULT_CONTEXT_FRAMES,
        )
        if timing is not None:
            timing["state_create_host_seconds"] = float(
                timing.get("state_create_host_seconds", 0.0)
            ) + (time.perf_counter() - state_create_start)
        try:
            prime_start = time.perf_counter()
            listener.prime_steady_state(state)
            if timing is not None:
                timing["prime_host_seconds"] = float(timing.get("prime_host_seconds", 0.0)) + (
                    time.perf_counter() - prime_start
                )
            for source_frame_id in range(cluster_start, last_target_id + 1):
                local_frame_id = source_frame_id - cluster_start
                user_pcm = user_frames[source_frame_id]
                reference = None
                # At a cluster reset, the first warm-up frame has no visible
                # past reference; later frames use only the previous physical
                # assistant frame, preserving causal delay across the cluster.
                if assistant_frames is not None and source_frame_id > cluster_start:
                    reference = AssistantReferenceFrame(
                        frame_id=local_frame_id - 1,
                        pcm=assistant_frames[source_frame_id - 1],
                        emitted_at_ns=(local_frame_id - 1) * tick_ms * 1_000_000,
                        sample_rate=16000,
                    )
                assistant_reference_frame_id = (
                    source_frame_id - 1
                    if assistant_frames is not None and source_frame_id > 0
                    else None
                )
                if assistant_reference_frame_id is not None and assistant_reference_frame_id > source_frame_id - 1:
                    raise RuntimeError("assistant reference frame is newer than input_frame_id - 1")
                step_start = time.perf_counter()
                output = listener.step(user_pcm, reference, state)
                if timing is not None:
                    timing["listener_step_host_seconds"] = float(
                        timing.get("listener_step_host_seconds", 0.0)
                    ) + (time.perf_counter() - step_start)
                if int(output.frame_id) != local_frame_id:
                    raise RuntimeError(
                        f"listener_frame_id disallineato: window={local_frame_id}, output={output.frame_id}"
                    )
                frame = target_by_id.get(source_frame_id)
                if frame is None:
                    continue
                queries = getattr(output, "gemma_audio_queries_t", None)
                if queries is None:
                    raise RuntimeError(
                        "listener installato non espone le quattro query Gemma per tick; "
                        "estrazione control6 rifiutata invece di usare la feature V9"
                    )
                feature_copy_start = time.perf_counter()
                vector = last_audio_query(queries).detach().float().cpu().numpy()
                if vector.shape != (1536,) or not np.isfinite(vector).all():
                    raise RuntimeError("native Gemma listener ABI must yield a finite 1536-vector")
                if timing is not None:
                    timing["target_feature_copy_seconds"] = float(
                        timing.get("target_feature_copy_seconds", 0.0)
                    ) + (time.perf_counter() - feature_copy_start)
                action = frame.get("target_action") if frame.get("action_mask") else None
                context_start = max(cluster_start, source_frame_id - (DEFAULT_CONTEXT_FRAMES - 1))
                features.append(vector)
                metadata.append(
                    {
                        "conversation_id": str(frame.get("origin_conversation_id") or conversation["conversation_id"]),
                        "source_group_id": str(conversation.get("source_group_id", conversation["conversation_id"])),
                        "session_id": f"{session_id or conversation['conversation_id']}:conversation-{cluster_index}",
                        "frame_id": int(source_frame_id),
                        "input_frame_id": int(source_frame_id),
                        "target_frame_id": int(frame.get("target_frame_id", source_frame_id)),
                        "listener_frame_id": int(output.frame_id),
                        "assistant_reference_frame_id": assistant_reference_frame_id,
                        "cluster_start_frame_id": int(cluster_start),
                        "cluster_frame_count": int(source_frame_id - cluster_start + 1),
                        "native_kv_replay_start_frame_id": int(cluster_start),
                        "native_kv_replay_input_frame_count": int(source_frame_id - cluster_start + 1),
                        "context_start_frame_id": int(context_start),
                        "context_frame_count": int(source_frame_id - context_start + 1),
                        "valid_sample_count": int(valid_samples[source_frame_id]),
                        "frame_sample_count": frame_samples,
                        "frame_partial": int(valid_samples[source_frame_id]) < frame_samples,
                        "source": str(conversation.get("source", "unknown")),
                        "synthetic": _is_synthetic_source(conversation),
                        "profile": conversation.get("profile"),
                        "profile_version": conversation.get("profile_version"),
                        "action": action if action in ACTIONS else None,
                        "action_mask": bool(action in ACTIONS),
                        "target_action_mask": bool(frame.get("action_mask")),
                        "event_onset": frame.get("event_onset") is True,
                        "input_audio_start_sample": frame.get("input_audio_start_sample"),
                        "input_audio_end_sample": frame.get("input_audio_end_sample"),
                        "target_action_start_sample": frame.get("target_action_start_sample"),
                        "target_action_end_sample": frame.get("target_action_end_sample"),
                        "input_text_visible": frame.get("input_text_visible"),
                        "future_audio_excluded": frame.get("future_audio_excluded"),
                        "action_masked_reason": frame.get("action_masked_reason"),
                    }
                )
        finally:
            state.close()
    return np.asarray(features, dtype=np.float32), metadata


def _cluster_causal_targets(
    target_frames: Sequence[Mapping[str, Any]],
    *,
    context_frames: int = 64,
    max_target_gap_frames: int = 64,
    max_cluster_frames: int = 128,
) -> list[tuple[int, list[Mapping[str, Any]]]]:
    """Cluster row-local targets without exceeding their causal replay bounds."""
    if context_frames <= 0 or max_target_gap_frames <= 0 or max_cluster_frames < context_frames:
        raise ValueError("limiti cluster stage2 non validi")
    ordered = sorted(target_frames, key=lambda frame: int(frame.get("frame_id", -1)))
    clusters: list[tuple[int, list[Mapping[str, Any]]]] = []
    cluster_start: int | None = None
    cluster_targets: list[Mapping[str, Any]] = []
    previous_target = -1
    for frame in ordered:
        frame_id = int(frame.get("frame_id", -1))
        if frame_id < 0 or frame_id <= previous_target:
            raise ValueError("target frame_id deve essere univoco, crescente e non negativo")
        proposed_start = max(0, frame_id - (context_frames - 1))
        if not cluster_targets:
            cluster_start = proposed_start
            cluster_targets = [frame]
        else:
            assert cluster_start is not None
            joins_cluster = (
                frame_id - previous_target <= max_target_gap_frames
                and frame_id - cluster_start + 1 <= max_cluster_frames
            )
            if joins_cluster:
                cluster_targets.append(frame)
            else:
                clusters.append((cluster_start, cluster_targets))
                cluster_start = proposed_start
                cluster_targets = [frame]
        previous_target = frame_id
    if cluster_targets:
        assert cluster_start is not None
        clusters.append((cluster_start, cluster_targets))
    return clusters


def _conversation_replay_clusters(
    target_frames: Sequence[Mapping[str, Any]],
) -> list[tuple[int, list[Mapping[str, Any]]]]:
    """Advance one native listener state from conversation frame zero to its last target."""
    ordered = sorted(target_frames, key=lambda frame: int(frame.get("frame_id", -1)))
    previous = -1
    for frame in ordered:
        frame_id = int(frame.get("frame_id", -1))
        if frame_id < 0 or frame_id <= previous:
            raise ValueError("conversation replay targets must be unique, increasing, and non-negative")
        previous = frame_id
    return [(0, ordered)] if ordered else []


@contextmanager
def _profile_native_listener_components(listener: Any, timing: dict[str, Any]):
    """Collect optional CPU front-end and CUDA component timings for one benchmark."""
    if not torch.cuda.is_available():
        yield
        return
    backend = getattr(listener, "backend", None)
    if backend is None:
        yield
        return
    events: dict[str, list[tuple[Any, Any]]] = {"audio_tower": [], "lm_forward": []}
    restorers: list[tuple[Any, str, bool, Any]] = []

    def timed_method(owner: Any, attribute: str, phase: str) -> None:
        original = getattr(owner, attribute, None)
        if not callable(original):
            return
        existed = attribute in vars(owner)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            stream = torch.cuda.current_stream()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            try:
                return original(*args, **kwargs)
            finally:
                end_event.record(stream)
                events[phase].append((start_event, end_event))

        setattr(owner, attribute, wrapped)
        restorers.append((owner, attribute, existed, original))

    frontend = getattr(backend, "_extract_single_frame", None)
    if callable(frontend):
        existed = "_extract_single_frame" in vars(backend)

        def timed_frontend(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return frontend(*args, **kwargs)
            finally:
                timing["mel_frontend_cpu_seconds"] = float(
                    timing.get("mel_frontend_cpu_seconds", 0.0)
                ) + (time.perf_counter() - started)

        setattr(backend, "_extract_single_frame", timed_frontend)
        restorers.append((backend, "_extract_single_frame", existed, frontend))

    timed_method(backend, "_forward_incremental_lm", "lm_forward")
    gemma_model = getattr(backend, "gemma_model", None)
    if gemma_model is not None:
        timed_method(gemma_model, "get_audio_features", "audio_tower")

    graph = getattr(backend, "_cuda_graph", None)
    graph_stream = getattr(backend, "_graph_stream", None)
    if graph is not None and graph_stream is not None:
        existed = "_cuda_graph" in vars(backend)

        class _TimedGraph:
            def replay(self, *args: Any, **kwargs: Any) -> Any:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record(graph_stream)
                try:
                    return graph.replay(*args, **kwargs)
                finally:
                    end_event.record(graph_stream)
                    events["audio_tower"].append((start_event, end_event))

        setattr(backend, "_cuda_graph", _TimedGraph())
        restorers.append((backend, "_cuda_graph", existed, graph))

    timing["_cuda_events"] = events
    try:
        yield
    finally:
        for owner, attribute, existed, original in reversed(restorers):
            if existed:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)


def _resolve_profile_events(timing: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    events = timing.pop("_cuda_events", {})
    for phase, pairs in events.items():
        values[f"{phase}_cuda_event_seconds"] = round(
            sum(float(start.elapsed_time(end)) for start, end in pairs) / 1000.0,
            6,
        )
        values[f"{phase}_cuda_event_calls"] = len(pairs)
    return values


def _public_phase_timings(timing: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in timing.items()
        if not key.startswith("_")
    }


def benchmark_native_listener_windows(
    listener: Any,
    conversation: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    windows: int = 8,
    pcm_cache: dict[str, tuple[np.ndarray, int]] | None = None,
    include_feature_vectors: bool = False,
) -> dict[str, Any]:
    """Measure real merged-model cost while replaying selected conversations from frame zero."""
    if windows <= 0:
        raise ValueError("windows benchmark deve essere positivo")
    conversations = [conversation] if isinstance(conversation, Mapping) else list(conversation)
    candidates: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for row in conversations:
        targets, _ = _causal_action_targets(row)
        if targets:
            candidates.append((row, targets))
    available = sum(len(targets) for _, targets in candidates)
    if not available:
        raise ValueError("nessun target audio supervisionato disponibile nel benchmark")
    target_count = min(int(windows), available)

    # Share the sample budget evenly across natural and synthetic rows when
    # both are supplied, reallocating any unused quota to rows with more labels.
    allocations = [0] * len(candidates)
    remaining = target_count
    while remaining:
        progressed = False
        for index, (_, targets) in enumerate(candidates):
            if allocations[index] >= len(targets):
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise RuntimeError("benchmark non è riuscito ad assegnare tutti i target richiesti")

    def select_clustered(targets: list[Mapping[str, Any]], count: int) -> list[Mapping[str, Any]]:
        clusters = _cluster_causal_targets(targets)
        ordered_clusters = sorted(clusters, key=lambda item: (-len(item[1]), item[0]))
        selected: list[Mapping[str, Any]] = []
        for _, cluster_targets in ordered_clusters:
            take = min(count - len(selected), len(cluster_targets))
            if take == len(cluster_targets):
                selected.extend(cluster_targets)
            elif take > 1:
                start_index = max(0, (len(cluster_targets) - take) // 2)
                selected.extend(cluster_targets[start_index:start_index + take])
            elif take == 1:
                selected.append(cluster_targets[len(cluster_targets) // 2])
            if len(selected) >= count:
                break
        return sorted(selected[:count], key=lambda frame: int(frame["frame_id"]))

    selected_rows: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
    for (row, targets), allocation in zip(candidates, allocations):
        if not allocation:
            continue
        selected_rows.append((row, select_clustered(targets, allocation)))

    cuda_active = torch.cuda.is_available()
    if cuda_active:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        allocated_before = int(torch.cuda.memory_allocated())
    else:
        allocated_before = 0
    start = time.perf_counter()
    metadata: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    phase_timings: dict[str, Any] = {}
    cache = pcm_cache if pcm_cache is not None else {}
    with _profile_native_listener_components(listener, phase_timings):
        for row, selected_targets in selected_rows:
            target_key = "frame_targets" if "frame_targets" in row else "frames"
            bounded_row = {**row, target_key: selected_targets}
            row_features, row_metadata = extract_conversation_features(
                listener,
                bounded_row,
                session_id=f"benchmark:{row.get('conversation_id', 'unknown')}",
                pcm_cache=cache,
                timing=phase_timings,
            )
            feature_rows.append(row_features)
            metadata.extend(row_metadata)
    if cuda_active:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    causal_steps = int(phase_timings.get("causal_input_frame_steps", 0))
    count = len(metadata)
    if count != target_count or causal_steps <= 0:
        raise RuntimeError("benchmark non ha estratto il numero richiesto di target/context frame")
    features = np.concatenate(feature_rows, axis=0)
    if not np.isfinite(features).all():
        raise RuntimeError("benchmark listener ha prodotto query non finite")
    per_window = elapsed / count
    per_causal_step = elapsed / causal_steps
    projected_windows = {"train": 768, "validation": 192, "test": 128}
    projected_seconds = {
        split: round(per_window * split_count, 3)
        for split, split_count in projected_windows.items()
    }
    allocated_peak = int(torch.cuda.max_memory_allocated()) if cuda_active else 0
    if cuda_active:
        phase_timings["cuda_event_seconds"] = _resolve_profile_events(phase_timings)
    selected_conversations = []
    for row, selected_targets in selected_rows:
        selected_conversations.append(
            {
                "conversation_id": str(row.get("conversation_id", "")),
                "source": str(row.get("source", "unknown")),
                "synthetic": _is_synthetic_source(row),
                "target_frame_ids": [int(frame["frame_id"]) for frame in selected_targets],
            }
        )
    report = {
        "conversations": selected_conversations,
        "windows": count,
        "selected_target_frame_ids": [int(row["frame_id"]) for row in metadata],
        "selected_target_frames": [
            {
                "conversation_id": row["conversation_id"],
                "source": row["source"],
                "synthetic": row["synthetic"],
                "frame_id": row["frame_id"],
                "context_frame_count": row["context_frame_count"],
            }
            for row in metadata
        ],
        "context_frame_counts": [int(row["context_frame_count"]) for row in metadata],
        "causal_input_frame_steps": causal_steps,
        "conversation_replays": int(phase_timings.get("conversation_replays", 0)),
        "tick_ms": 80,
        "input_channel_index": _channel_index(selected_rows[0][0], "user"),
        "input_channel_indices": [
            _channel_index(row, "user") for row, _ in selected_rows
        ],
        "elapsed_seconds": round(elapsed, 6),
        "seconds_per_target_window": round(per_window, 6),
        "seconds_per_causal_input_frame_step": round(per_causal_step, 6),
        "timings": _public_phase_timings(phase_timings),
        "projected_target_windows": projected_windows,
        "projected_seconds_by_split_at_sampled_window_rate": projected_seconds,
        "projected_total_target_window_seconds": round(sum(projected_seconds.values()), 3),
        "cuda_allocated_before_bytes": allocated_before,
        "cuda_peak_allocated_bytes": allocated_peak,
        "cuda_peak_delta_bytes": max(0, allocated_peak - allocated_before),
        "last_query_shape": [int(value) for value in features[-1].shape] if len(features) else None,
        "queries_finite": bool(np.isfinite(features).all()),
    }
    if include_feature_vectors:
        report["_feature_vectors"] = features
        report["_feature_metadata"] = metadata
    return report


def _write_features(
    path: Path,
    features: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    input_dim: int = 1536,
) -> None:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != input_dim or len(features) != len(metadata):
        raise ValueError(
            f"listener features must align as [N,{input_dim}] with frame metadata"
        )
    if not np.isfinite(features).all():
        raise ValueError("native listener features contain NaN/Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(
        [ACTIONS.index(row["action"]) if row.get("action_mask") else -100 for row in metadata],
        dtype=np.int64,
    )
    frame_ids = np.asarray([int(row["frame_id"]) for row in metadata], dtype=np.int64)
    listener_ids = np.asarray([int(row["listener_frame_id"]) for row in metadata], dtype=np.int64)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=features,
            labels=labels,
            frame_ids=frame_ids,
            listener_frame_ids=listener_ids,
        )
    temporary.replace(path)
    path.with_suffix(".json").write_text(
        json.dumps(list(metadata), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def augment_control_features_with_acoustic_phase(
    *,
    native_feature_dir: str | Path,
    prepared_dir: str | Path,
    output_dir: str | Path,
    additional_prepared_dir: str | Path | None = None,
    splits: Sequence[str] = ("train", "validation"),
    mask_train_natural_backchannels_without_overlap: bool = True,
    frozen_head_path: str | Path | None = None,
) -> dict[str, Any]:
    """Append four causal PCM features to saved native hidden vectors.

    Only the requested train/validation splits are read. The native hidden
    vectors must already have been extracted from the final merged adapter;
    this function never loads a model or recomputes those vectors.
    """
    from duplex_kernel.acoustic_phase_features import (
        ACOUSTIC_PHASE_FEATURE_CONTRACT,
        acoustic_phase_features,
    )

    native_dir = Path(native_feature_dir)
    prepared_root = Path(prepared_dir)
    additional_root = Path(additional_prepared_dir) if additional_prepared_dir else None
    output = Path(output_dir)
    requested_splits = tuple(splits)
    if not requested_splits or any(split not in {"train", "validation", "test"} for split in requested_splits):
        raise ValueError("acoustic phase augmentation accepts train, validation, or held-out test")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("acoustic phase split list contains duplicates")
    if "test" in requested_splits:
        if requested_splits != ("test",) or frozen_head_path is None:
            raise ValueError("test augmentation is held-out evaluation only and requires a frozen Candidate C head")
        frozen_path = Path(frozen_head_path)
        if not frozen_path.is_file():
            raise FileNotFoundError(f"frozen Candidate C head is required before test access: {frozen_path}")
        frozen_head = load_control_head(frozen_path, device="cpu")
        if frozen_head.input_dim != 1540:
            raise ValueError("held-out Candidate C evaluation requires the frozen 1540-feature head")
        del frozen_head
    elif frozen_head_path is not None:
        raise ValueError("frozen_head_path is accepted only for held-out test evaluation")
    if output.exists():
        raise FileExistsError(f"Candidate C output must be a fresh directory: {output}")

    source_manifest_path = native_dir / "feature_manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"native feature manifest missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("merged_before_native_listener") is not True:
        raise ValueError("acoustic augmentation requires post-merge native listener features")
    source_contract = source_manifest.get("feature_contract")
    if source_contract != listener_feature_contract(1536):
        raise ValueError("acoustic augmentation accepts only the saved native 1536 feature ABI")

    def file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def source_rows(root: Path, split: str) -> list[dict[str, Any]]:
        path = root / f"{split}.jsonl"
        return read_jsonl(path) if path.is_file() else []

    report: dict[str, Any] = {
        "status": "completed",
        "feature_contract": listener_feature_contract(1540),
        "source_feature_dir": str(native_dir.resolve()),
        "source_feature_manifest_sha256": file_sha256(source_manifest_path),
        "base_model_path": source_manifest.get("base_model_path"),
        "adapter_dir": source_manifest.get("adapter_dir"),
        "merged_before_native_listener": True,
        "prepared_sources": [str(prepared_root.resolve())]
        + ([str(additional_root.resolve())] if additional_root else []),
        "acoustic_phase_contract": ACOUSTIC_PHASE_FEATURE_CONTRACT,
        "causality": {
            "native_hidden_frame": "input_frame_id t",
            "user_pcm_history": "observed user PCM ending at input frame t; at most 400 ms; left-zero-padded at stream start",
            "assistant_pcm": "only already-emitted assistant reference frame t-1",
            "future_audio_or_supervision_used": False,
            "model_reextraction": False,
        },
        "heldout_evaluation_only": requested_splits == ("test",),
        "frozen_head_path": str(Path(frozen_head_path).resolve()) if frozen_head_path else None,
        "splits": {},
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    staging_output = output.with_name(output.name + ".staging")
    if staging_output.exists():
        raise FileExistsError(f"stale Candidate C staging directory exists: {staging_output}")
    staging_output.mkdir(parents=False, exist_ok=False)
    total_masked_rows = 0
    total_verified_overlap_rows = 0
    masked_groups: set[str] = set()
    verified_overlap_groups: set[str] = set()
    for split in requested_splits:
        npz_path = native_dir / f"{split}.npz"
        metadata_path = native_dir / f"{split}.json"
        if not npz_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"native feature split is incomplete: {split}")
        with np.load(npz_path, allow_pickle=False) as values:
            native_features = np.asarray(values["features"], dtype=np.float32)
            native_labels = np.asarray(values["labels"], dtype=np.int64)
            frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
            listener_ids = np.asarray(values["listener_frame_ids"], dtype=np.int64)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            native_features.ndim != 2
            or native_features.shape[1] != 1536
            or native_labels.shape != (len(native_features),)
            or frame_ids.shape != native_labels.shape
            or listener_ids.shape != native_labels.shape
            or len(metadata) != len(native_features)
        ):
            raise ValueError(f"native {split} feature arrays/metadata are not aligned at width 1536")
        if not np.isfinite(native_features).all():
            raise ValueError(f"native {split} hidden vectors contain NaN/Inf")

        prepared_by_id: dict[str, dict[str, Any]] = {}
        for root in (prepared_root, additional_root):
            if root is None:
                continue
            for conversation in source_rows(root, split):
                conversation_id = str(conversation.get("conversation_id", ""))
                if not conversation_id:
                    raise ValueError(f"{root}/{split}.jsonl contains a row without conversation_id")
                previous = prepared_by_id.get(conversation_id)
                if previous is not None:
                    if (
                        previous.get("audio_sha256") != conversation.get("audio_sha256")
                        or previous.get("audio_path") != conversation.get("audio_path")
                        or _is_synthetic_source(previous) != _is_synthetic_source(conversation)
                    ):
                        raise ValueError(f"conflicting prepared source rows for {conversation_id}")
                    # The additional prepared source has the uncapped V5
                    # timeline. For matching V5 audio identity it supersedes
                    # the capped target list in the base prepared directory.
                    if root == additional_root and _is_synthetic_source(conversation):
                        prepared_by_id[conversation_id] = conversation
                    continue
                prepared_by_id[conversation_id] = conversation

        by_conversation: dict[str, list[int]] = {}
        for index, row in enumerate(metadata):
            conversation_id = str(row.get("conversation_id", ""))
            if conversation_id not in prepared_by_id:
                raise ValueError(f"no {split} prepared source row for feature conversation {conversation_id!r}")
            conversation = prepared_by_id[conversation_id]
            if str(row.get("source_group_id")) != str(
                conversation.get("source_group_id") or conversation_id
            ):
                raise ValueError(f"source_group_id mismatch for {conversation_id}")
            if _is_synthetic_source(row) != _is_synthetic_source(conversation):
                raise ValueError(f"synthetic/natural provenance mismatch for {conversation_id}")
            # Normalize convenience metadata from the authoritative source/profile.
            row["synthetic"] = _is_synthetic_source(conversation)
            input_frame_id = row.get("input_frame_id", row.get("frame_id"))
            if (
                isinstance(input_frame_id, bool)
                or not isinstance(input_frame_id, int)
                or int(row.get("frame_id", -1)) != input_frame_id
                or int(frame_ids[index]) != input_frame_id
                or int(listener_ids[index]) != int(row.get("listener_frame_id", -2))
            ):
                raise ValueError(f"saved native frame IDs disagree with their own metadata at {conversation_id}:{index}")
            assistant_reference_id = row.get("assistant_reference_frame_id")
            if input_frame_id == 0:
                if assistant_reference_id not in (None, -1):
                    raise ValueError(f"stream-start assistant reference must be absent at {conversation_id}:0")
            elif (
                isinstance(assistant_reference_id, bool)
                or not isinstance(assistant_reference_id, int)
                or assistant_reference_id != input_frame_id - 1
            ):
                raise ValueError(f"assistant reference is not the causal t-1 frame at {conversation_id}:{input_frame_id}")
            targets, is_v5 = _causal_action_targets(conversation)
            target_by_input = {int(target["frame_id"]): target for target in targets}
            source_target = target_by_input.get(input_frame_id)
            if source_target is None:
                raise ValueError(f"no supervised source target at {conversation_id}:{input_frame_id}")
            if (
                row.get("action") != source_target.get("target_action")
                or bool(row.get("action_mask")) != bool(source_target.get("action_mask"))
                or int(native_labels[index])
                != (ACTIONS.index(source_target["target_action"]) if source_target.get("action_mask") else -100)
            ):
                raise ValueError(f"feature action disagrees with prepared source at {conversation_id}:{input_frame_id}")
            if is_v5:
                if (
                    int(row.get("target_frame_id", -1)) != input_frame_id + 1
                    or int(source_target.get("target_frame_id", -1)) != input_frame_id + 1
                    or row.get("input_text_visible") is not False
                    or row.get("future_audio_excluded") is not True
                ):
                    raise ValueError(f"V5 causal target contract mismatch at {conversation_id}:{input_frame_id}")
            else:
                if int(row.get("target_frame_id", input_frame_id)) != input_frame_id:
                    raise ValueError(f"natural legacy source target is not same-index at {conversation_id}:{input_frame_id}")
            by_conversation.setdefault(conversation_id, []).append(index)

        augmented = np.empty((len(native_features), 1540), dtype=np.float32)
        augmented[:, :1536] = native_features
        result_metadata = [dict(row) for row in metadata]
        pcm_cache: dict[str, tuple[np.ndarray, int]] = {}
        verified_overlap_rows = 0
        masked_rows = 0
        audio_hashes: dict[str, str] = {}
        cached_source_group: str | None = None
        for conversation_id, indices in by_conversation.items():
            conversation = prepared_by_id[conversation_id]
            current_source_group = str(conversation.get("source_group_id") or conversation_id)
            if current_source_group != cached_source_group:
                # Paired role views reuse one decode within a physical source
                # group; never retain corpus-wide PCM in memory.
                pcm_cache.clear()
                cached_source_group = current_source_group
            source_path_value = conversation.get("audio_path")
            if not source_path_value:
                raise ValueError(f"prepared source lacks audio_path: {conversation_id}")
            source_path = Path(str(source_path_value))
            if not source_path.is_file():
                raise FileNotFoundError(f"prepared source audio is missing: {source_path}")
            resolved_audio_path = str(source_path.resolve())
            if resolved_audio_path not in audio_hashes:
                actual_sha256 = file_sha256(source_path)
                declared_sha256 = conversation.get("audio_sha256")
                if declared_sha256 and actual_sha256 != declared_sha256:
                    raise ValueError(f"prepared source audio SHA-256 mismatch: {source_path}")
                audio_hashes[resolved_audio_path] = actual_sha256
            user_frames, assistant_frames, _ = _audio_frames(
                conversation, frame_samples=1280, pcm_cache=pcm_cache
            )
            user_channel = _channel_index(conversation, "user")
            if user_channel is None:
                raise ValueError(f"prepared source has no explicit user channel: {conversation_id}")
            for index in indices:
                row = result_metadata[index]
                input_frame_id = int(row.get("input_frame_id", row["frame_id"]))
                if input_frame_id >= len(user_frames):
                    raise ValueError(f"input frame extends beyond PCM source: {conversation_id}:{input_frame_id}")
                history_start = max(0, input_frame_id - 4)
                user_history = user_frames[history_start : input_frame_id + 1].reshape(-1)
                if len(user_history) < 5 * 1280:
                    user_history = np.pad(user_history, (5 * 1280 - len(user_history), 0))
                assistant_previous = (
                    assistant_frames[input_frame_id - 1]
                    if assistant_frames is not None and input_frame_id > 0
                    else None
                )
                phase = acoustic_phase_features(
                    user_history,
                    assistant_previous,
                    sample_rate_hz=16000,
                    tick_ms=80,
                )
                augmented[index, 1536:] = phase

                if (
                    split == "train"
                    and mask_train_natural_backchannels_without_overlap
                    and not _is_synthetic_source(row)
                    and row.get("action") == "BACKCHANNEL"
                ):
                    source_targets, is_v5 = _causal_action_targets(conversation)
                    if is_v5:
                        raise ValueError("natural BACKCHANNEL training mask unexpectedly matched synthetic V5 source")
                    source_frame = next(
                        (target for target in source_targets if int(target["frame_id"]) == input_frame_id),
                        None,
                    )
                    if (
                        source_frame is None
                        or source_frame.get("target_action") != "BACKCHANNEL"
                        or source_frame.get("target_action_source") != "verified_native_assistant_action"
                    ):
                        raise ValueError(f"natural BACKCHANNEL provenance is not verified at {conversation_id}:{input_frame_id}")
                    tick_start = float(source_frame.get("start_ms", input_frame_id * 80))
                    tick_end = float(source_frame.get("end_ms", (input_frame_id + 1) * 80))
                    if not tick_end > tick_start:
                        raise ValueError(f"natural BACKCHANNEL source interval invalid at {conversation_id}:{input_frame_id}")
                    onset_segments = []
                    overlapping_user_segments = []
                    ambiguous_user_segments = []
                    for segment in conversation.get("segments", ()) or ():
                        if not isinstance(segment, Mapping):
                            raise ValueError(f"source segment is not an object in {conversation_id}")
                        try:
                            start_ms = float(segment["start_ms"])
                            end_ms = float(segment["end_ms"])
                        except (KeyError, TypeError, ValueError) as exc:
                            if segment.get("role") in {"assistant", "user"}:
                                raise ValueError(f"role segment lacks a valid time interval in {conversation_id}") from exc
                            continue
                        if not end_ms > start_ms:
                            raise ValueError(f"role segment has an invalid interval in {conversation_id}")
                        if (
                            segment.get("role") == "assistant"
                            and segment.get("native_turn_state") == "backchannel"
                            and int(start_ms // 80) == input_frame_id
                        ):
                            if (
                                segment.get("audio_channel_verified") is not True
                                or segment.get("time_mask") is not True
                                or segment.get("channel_index") != _channel_index(conversation, "assistant")
                            ):
                                raise ValueError(f"assistant BACKCHANNEL onset lacks verified channel/time in {conversation_id}")
                            onset_segments.append(segment)
                        if end_ms > tick_start and start_ms < tick_end:
                            if segment.get("role") == "user":
                                if (
                                    segment.get("audio_channel_verified") is not True
                                    or segment.get("time_mask") is not True
                                    or segment.get("channel_index") != user_channel
                                ):
                                    ambiguous_user_segments.append(segment)
                                else:
                                    overlapping_user_segments.append(segment)
                            elif (
                                segment.get("role") not in {"assistant", None}
                                or segment.get("channel_index") == user_channel
                            ):
                                ambiguous_user_segments.append(segment)
                    if len(onset_segments) != 1:
                        raise ValueError(
                            f"expected one verified native BACKCHANNEL onset at {conversation_id}:{input_frame_id}, found {len(onset_segments)}"
                        )
                    if ambiguous_user_segments:
                        raise ValueError(f"cannot safely classify user overlap at {conversation_id}:{input_frame_id}")
                    source_group = str(row.get("source_group_id") or conversation_id)
                    if overlapping_user_segments:
                        verified_overlap_rows += 1
                        verified_overlap_groups.add(source_group)
                    else:
                        row["action_mask"] = False
                        row["action_masked_reason"] = "natural_backchannel_without_verified_remote_overlap_at_source_onset"
                        masked_rows += 1
                        masked_groups.add(source_group)
        if not np.isfinite(augmented).all():
            raise ValueError(f"augmented {split} feature matrix contains NaN/Inf")
        _write_features(staging_output / f"{split}.npz", augmented, result_metadata, input_dim=1540)
        report["splits"][split] = {
            "rows": int(len(augmented)),
            "supervised_rows": int(sum(bool(row.get("action_mask")) for row in result_metadata)),
            "per_action_support": {
                action: int(sum(bool(row.get("action_mask")) and row.get("action") == action for row in result_metadata))
                for action in ACTIONS
            },
            "audio_source_files": len(audio_hashes),
            "audio_source_sha256": audio_hashes,
        }
        if split == "train":
            total_masked_rows += masked_rows
            total_verified_overlap_rows += verified_overlap_rows
            report["natural_train_backchannel_masking"] = {
                "enabled": bool(mask_train_natural_backchannels_without_overlap),
                "verified_overlap_rows": int(verified_overlap_rows),
                "masked_rows": int(masked_rows),
                "verified_overlap_groups": len(verified_overlap_groups),
                "masked_groups": len(masked_groups),
                "remaining_supervised_natural_backchannel_rows": int(
                    sum(
                        bool(row.get("action_mask"))
                        and row.get("action") == "BACKCHANNEL"
                        and not _is_synthetic_source(row)
                        for row in result_metadata
                    )
                ),
                "criterion": (
                    "training-only ignore_index for verified natural assistant BACKCHANNEL onset rows "
                    "without a verified user audio/time overlap in the same 80 ms source interval; "
                    "action provenance string is preserved"
                ),
            }

    report["split_scope"] = list(requested_splits)
    report["test_data_read"] = "test" in requested_splits
    report["test_used_for_training_or_calibration"] = False
    report["source_feature_sha256"] = {
        split: {
            "npz": file_sha256(native_dir / f"{split}.npz"),
            "metadata": file_sha256(native_dir / f"{split}.json"),
        }
        for split in requested_splits
    }
    manifest_path = staging_output / "feature_manifest.json"
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(staging_output, output)
    return report


def _clustered_target_context_workload(conversations: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count full causal conversation replay steps before native extraction."""
    windows = 0
    causal_steps = 0
    conversation_replays = 0
    max_conversation_frames = 0
    for conversation in conversations:
        targets, _ = _causal_action_targets(conversation)
        clusters = _conversation_replay_clusters(targets)
        windows += len(targets)
        conversation_replays += len(clusters)
        for cluster_start, cluster_targets in clusters:
            cluster_steps = int(cluster_targets[-1]["frame_id"]) - cluster_start + 1
            causal_steps += cluster_steps
            max_conversation_frames = max(max_conversation_frames, cluster_steps)
    return {
        "target_windows": windows,
        "causal_input_frame_steps": causal_steps,
        "conversation_replays": conversation_replays,
        "max_conversation_replay_input_frames": max_conversation_frames,
    }


def _v5_event_onset_selection_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score V5 event onsets and false event outputs on true L/C background.

    Event-action interiors remain valid action labels. The legacy rate is
    included for continuity with earlier reports, but is explicitly not a
    false-action metric because it counts correct SPEAK/BACKCHANNEL interiors.
    """
    targets = np.asarray(targets, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if targets.ndim != 1 or predictions.shape != targets.shape:
        raise ValueError("V5 selection targets/predictions must be aligned 1D arrays")
    if len(metadata) != len(targets):
        raise ValueError("V5 selection metadata/targets must be aligned")
    if np.any((targets < 0) | (targets >= len(ACTIONS))):
        raise ValueError("V5 selection targets must contain only supervised action indices")
    if np.any((predictions < 0) | (predictions >= len(ACTIONS))):
        raise ValueError("V5 selection predictions contain an action index outside the ABI")

    v5_positions = [
        index
        for index, row in enumerate(metadata)
        if row.get("profile") == "audio_grounded_v5"
        or row.get("source") == "authored_synthetic_audio_candidate"
    ]
    onset_positions = [
        index for index in v5_positions if metadata[index].get("event_onset") is True
    ]
    non_onset_positions = [
        index for index in v5_positions if metadata[index].get("event_onset") is not True
    ]
    background_indices = {ACTIONS.index("LISTEN"), ACTIONS.index("CONTINUE")}
    background_positions = [
        index
        for index in non_onset_positions
        if int(targets[index]) in background_indices
    ]
    event_output_actions = ("SPEAK", "YIELD", "BACKCHANNEL", "REPLAN")
    event_output_indices = {ACTIONS.index(action) for action in event_output_actions}

    def summarize(positions: Sequence[int]) -> dict[str, Any]:
        matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=np.int64)
        for position in positions:
            matrix[int(targets[position]), int(predictions[position])] += 1
        support = matrix.sum(axis=1)
        predicted_support = matrix.sum(axis=0)
        per_class: dict[str, Any] = {}
        f1_values: list[float] = []
        recalls: list[float] = []
        for class_index, action in enumerate(ACTIONS):
            true_positive = int(matrix[class_index, class_index])
            recall = float(true_positive / support[class_index]) if support[class_index] else None
            precision = (
                float(true_positive / predicted_support[class_index])
                if predicted_support[class_index]
                else 0.0
            )
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if recall is not None and precision + recall
                else 0.0
            )
            per_class[action] = {
                "support": int(support[class_index]),
                "precision": precision,
                "recall": recall,
                "f1": f1 if support[class_index] else None,
            }
            if support[class_index]:
                f1_values.append(f1)
                recalls.append(float(recall))
        examples = len(positions)
        correct = int(np.trace(matrix))
        return {
            "examples": int(examples),
            "accuracy": float(correct / examples) if examples else None,
            "macro_f1": float(np.mean(f1_values)) if f1_values else None,
            "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
            "class_support": {action: int(support[index]) for index, action in enumerate(ACTIONS)},
            "per_class": per_class,
            "confusion_matrix": {
                "true_actions": list(ACTIONS),
                "predicted_actions": list(ACTIONS),
                "values": matrix.tolist(),
            },
        }

    background_predictions = predictions[background_positions]
    background_false_counts = {
        action: int(np.count_nonzero(background_predictions == ACTIONS.index(action)))
        for action in event_output_actions
    }
    legacy_predictions = predictions[non_onset_positions]
    legacy_event_outputs = int(
        np.isin(legacy_predictions, tuple(event_output_indices)).sum()
    )
    return {
        "v5_examples": int(len(v5_positions)),
        "v5_full_six_class": summarize(v5_positions),
        "event_onset_examples": int(len(onset_positions)),
        "event_onset_macro_f1": summarize(onset_positions)["macro_f1"],
        "event_onsets_six_class": summarize(onset_positions),
        "background_definition": (
            "supervised V5 rows outside event_schedule onsets whose target action is LISTEN or CONTINUE; "
            "CONTINUE event-onset rows and SPEAK/BACKCHANNEL interiors are excluded"
        ),
        "background_examples": int(len(background_positions)),
        "background_false_event_predictions": int(sum(background_false_counts.values())),
        "background_false_event_rate": (
            float(sum(background_false_counts.values()) / len(background_positions))
            if background_positions
            else None
        ),
        "background_false_event_predictions_by_class": background_false_counts,
        "event_action_outputs": list(event_output_actions),
        "legacy_off_onset_examples": int(len(non_onset_positions)),
        "legacy_off_onset_event_output_predictions": legacy_event_outputs,
        "legacy_off_onset_event_output_rate_not_false_action_rate": (
            float(legacy_event_outputs / len(non_onset_positions))
            if non_onset_positions
            else None
        ),
        "legacy_metric_note": (
            "counts event-class outputs on every non-onset target, including genuine SPEAK/BACKCHANNEL "
            "interiors; descriptive only and not a false-action rate"
        ),
    }


def _natural_validation_macro_f1(
    head: ControlHead,
    features: np.ndarray,
    labels: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
    *,
    train_class_support: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Score checkpoint selection only on labeled real four-action validation rows."""
    natural_actions = ("LISTEN", "SPEAK", "CONTINUE", "BACKCHANNEL")
    action_indices = [ACTIONS.index(action) for action in natural_actions]
    positions = [
        index
        for index, row in enumerate(metadata)
        if int(labels[index]) in action_indices and not _is_synthetic_source(row)
    ]
    if not positions:
        raise ValueError("checkpoint selection richiede esempi naturali etichettati in validation")
    selected_labels = labels[positions].astype(np.int64, copy=False)
    was_training = bool(head.training)
    head.eval()
    try:
        with torch.inference_mode():
            predictions = head(torch.from_numpy(features[positions])).argmax(dim=-1).cpu().numpy()
    finally:
        if was_training:
            head.train()
    row_index = {action_index: index for index, action_index in enumerate(action_indices)}
    matrix = np.zeros((len(natural_actions), len(ACTIONS)), dtype=np.int64)
    for target, prediction in zip(selected_labels, predictions):
        matrix[row_index[int(target)], int(prediction)] += 1
    support = matrix.sum(axis=1)
    predicted_support = matrix.sum(axis=0)
    correct_predictions = sum(
        int(matrix[index, ACTIONS.index(action)])
        for index, action in enumerate(natural_actions)
    )
    per_class: dict[str, Any] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, action in enumerate(natural_actions):
        global_index = ACTIONS.index(action)
        tp = int(matrix[index, global_index])
        recall = float(tp / support[index]) if support[index] else None
        precision = float(tp / predicted_support[global_index]) if predicted_support[global_index] else 0.0
        f1 = float(
            2 * precision * recall / (precision + recall)
        ) if recall is not None and precision + recall else 0.0
        per_class[action] = {
            "support": int(support[index]),
            "precision": precision,
            "recall": recall,
            "f1": f1 if support[index] else None,
        }
        if support[index]:
            f1_values.append(f1)
            recalls.append(float(recall))
    examples = int(len(selected_labels))
    present_indices = [index for index, value in enumerate(support) if value]
    present_actions = [natural_actions[index] for index in present_indices]
    majority_train_action = max(
        present_actions,
        key=lambda action: int((train_class_support or {}).get(action, 0)),
        default=None,
    )
    majority_accuracy = (
        float(support[natural_actions.index(majority_train_action)] / examples)
        if majority_train_action is not None
        else None
    )
    true_counts = support[present_indices]
    return {
        "examples": examples,
        "accuracy": float(correct_predictions / examples),
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "actions": list(natural_actions),
        "class_support": {action: int(support[index]) for index, action in enumerate(natural_actions)},
        "absent_classes": [
            action for index, action in enumerate(natural_actions) if not support[index]
        ],
        "chance_baselines": {
            "uniform_random_over_present_classes_accuracy": float(1.0 / len(present_indices)),
            "empirical_prior_random_expected_accuracy": float(
                np.square(true_counts).sum() / (examples * examples)
            ),
        },
        "majority_baseline_from_train_prior": {
            "action": majority_train_action,
            "accuracy": majority_accuracy,
            "training_class_support": {
                action: int((train_class_support or {}).get(action, 0))
                for action in natural_actions
            },
        },
        "per_class": per_class,
        "confusion_matrix": {
            "true_actions": list(natural_actions),
            "predicted_actions": list(ACTIONS),
            "values": matrix.tolist(),
        },
    }


def extract_stage2_features(
    *,
    prepared_dir: str | Path,
    base_model_path: str,
    adapter_dir: str,
    output_dir: str | Path,
    device: str = "cuda",
    benchmark_only: bool = False,
    benchmark_then_extract_if_under_hours: float | None = None,
    verify_prefix_cache: bool = False,
) -> dict[str, Any]:
    """Merge final LoRA, benchmark bounded extraction, then extract all splits."""
    if not Path(adapter_dir).is_dir():
        raise FileNotFoundError(f"adapter LoRA finale mancante: {adapter_dir}")
    if benchmark_then_extract_if_under_hours is not None and benchmark_then_extract_if_under_hours <= 0:
        raise ValueError("la soglia auto-extract deve essere positiva")
    thinker, processor, model = load_merged_base(base_model_path, adapter_dir, device=device)
    from duplex_kernel.listener import GemmaListener

    listener = GemmaListener.from_thinker(thinker, runtime_profile="production")
    listener_backend = getattr(listener, "backend", None)
    configure_prefix_cache = getattr(listener_backend, "configure_canonical_prefix_cache", None)
    prefix_cache_stats = getattr(listener_backend, "canonical_prefix_cache_stats", None)
    if verify_prefix_cache and not callable(configure_prefix_cache):
        raise RuntimeError("verifica cache prefisso richiede il backend Gemma nativo cache-aware")
    if callable(configure_prefix_cache):
        configure_prefix_cache(enabled=False, clear=True)
    output = Path(output_dir)
    report: dict[str, Any] = {
        "feature_contract": dict(FEATURE_CONTRACT),
        "base_model_path": str(Path(base_model_path).resolve()),
        "adapter_dir": str(Path(adapter_dir).resolve()),
        "merged_before_native_listener": True,
        "extractor": "GemmaListener.production",
        "state_reset": "one GemmaListenerState per conversation; reset only at conversation boundary, replay from absolute frame 0 through the last supervised input",
        "cluster_contract": {
            "state_resets_per_conversation": 1,
            "replay_start_frame_id": 0,
            "advance_continuously_through_last_target": True,
            "reset_between_target_groups": False,
            "rolling_frontend_pcm_mel_context_frames": 64,
            "native_lm_cache_capacity_and_rollover": "listener production defaults; derived from the loaded native backend",
        },
        "audio_resampling": {
            "method": "scipy.signal.resample_poly",
            "source_rates_supported": "any positive PCM rate",
            "target_rate_hz": 16000,
            "tick_ms": 80,
            "partial_frame": "zero padded; valid_sample_count is a compact prefix mask",
        },
        "sparse_frame_extraction": "store explicitly supervised targets; for each conversation replay input from absolute frame zero through its last selected target",
        "counts": {},
    }
    split_conversations: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "test"):
        input_path = Path(prepared_dir) / f"{split}.jsonl"
        if input_path.is_file():
            split_conversations[split] = read_jsonl(input_path)
    workload = {
        split: _clustered_target_context_workload(rows)
        for split, rows in split_conversations.items()
    }
    report["planned_target_context_workload"] = workload

    training_conversations = split_conversations.get("train", [])
    benchmark_real = next(
        (
            row for row in training_conversations
            if not _is_synthetic_source(row)
            and row.get("audio_path")
            and Path(str(row["audio_path"])).is_file()
            and bool(_causal_action_targets(row)[0])
        ),
        None,
    )
    benchmark_synthetic = next(
        (
            row for row in training_conversations
            if _is_synthetic_source(row)
            and row.get("audio_path")
            and Path(str(row["audio_path"])).is_file()
            and bool(_causal_action_targets(row)[0])
        ),
        None,
    )
    benchmark_conversations = [
        row for row in (benchmark_real, benchmark_synthetic) if row is not None
    ]
    benchmark_groups = {
        str(row.get("source_group_id") or row["conversation_id"])
        for row in benchmark_conversations
    }
    benchmark_pcm_cache: dict[str, tuple[np.ndarray, int]] = {}
    if benchmark_real is not None:
        if verify_prefix_cache:
            cold = benchmark_native_listener_windows(
                listener,
                benchmark_conversations,
                windows=8,
                pcm_cache=benchmark_pcm_cache,
                include_feature_vectors=True,
            )
            configure_prefix_cache(enabled=True, clear=True)
            benchmark = benchmark_native_listener_windows(
                listener,
                benchmark_conversations,
                windows=8,
                pcm_cache=benchmark_pcm_cache,
                include_feature_vectors=True,
            )
            cold_features = cold.pop("_feature_vectors")
            cold_metadata = cold.pop("_feature_metadata")
            cached_features = benchmark.pop("_feature_vectors")
            cached_metadata = benchmark.pop("_feature_metadata")
            if len(cold_metadata) != len(cached_metadata) or any(
                any(
                    cold_row.get(key) != cached_row.get(key)
                    for key in (
                        "conversation_id", "source_group_id", "frame_id", "listener_frame_id",
                        "cluster_start_frame_id", "cluster_frame_count", "context_start_frame_id",
                        "context_frame_count", "valid_sample_count", "action",
                    )
                )
                for cold_row, cached_row in zip(cold_metadata, cached_metadata)
            ):
                raise RuntimeError("cache prefisso ha alterato target, causal context o confini stream")
            if cold_features.shape != cached_features.shape:
                raise RuntimeError("shape query cached/cold prefisso non coincide")
            max_abs_difference = float(np.max(np.abs(cold_features - cached_features)))
            backend_stats = prefix_cache_stats()
            expected_clusters = int(benchmark["conversation_replays"])
            if (
                not np.isfinite(cached_features).all()
                or max_abs_difference > 1e-3
                or int(backend_stats["prefix_compute_count"]) != 1
                or int(backend_stats["prefix_reuse_count"]) != expected_clusters - 1
            ):
                raise RuntimeError(
                    "parità/cache prefisso non valida: "
                    f"max_abs_diff={max_abs_difference}, stats={backend_stats}, clusters={expected_clusters}"
                )
            benchmark["canonical_prefix_cache_verification"] = {
                "status": "passed",
                "cold_target_windows": len(cold_features),
                "cached_target_windows": len(cached_features),
                "cold_and_cached_metadata_identical": True,
                "query_shape": list(cached_features.shape),
                "queries_finite": True,
                "max_abs_query_difference": max_abs_difference,
                "absolute_tolerance": 1e-3,
                "conversation_replays": expected_clusters,
                "canonical_prefix_backend_stats": backend_stats,
                "native_state_reset_between_conversations": True,
                "causal_targets_and_context_bounds_preserved": True,
            }
        else:
            if callable(configure_prefix_cache):
                configure_prefix_cache(enabled=True, clear=True)
            benchmark = benchmark_native_listener_windows(
                listener,
                benchmark_conversations,
                windows=8,
                pcm_cache=benchmark_pcm_cache,
            )
        if callable(prefix_cache_stats):
            benchmark["canonical_prefix_cache_after_benchmark"] = prefix_cache_stats()
        benchmark_steps_per_second = float(benchmark["seconds_per_causal_input_frame_step"])
        benchmark["projected_seconds_by_split_at_measured_context_rate"] = {
            split: round(
                benchmark_steps_per_second * float(split_workload["causal_input_frame_steps"]),
                3,
            )
            for split, split_workload in workload.items()
        }
        benchmark["projected_total_seconds_at_measured_context_rate"] = round(
            sum(benchmark["projected_seconds_by_split_at_measured_context_rate"].values()),
            3,
        )
        benchmark["projection_basis"] = (
            "elapsed benchmark time divided by the exact causal context advances for eight sampled targets; "
            "includes per-target steady-state priming and one PCM decode/resample per sampled conversation"
        )
        report["native_listener_benchmark"] = benchmark
    else:
        report["native_listener_benchmark"] = {"status": "no_real_train_conversation_available"}
    if benchmark_only:
        report["status"] = "native_listener_benchmark_only"
        output.mkdir(parents=True, exist_ok=True)
        report_path = output / "benchmark_manifest.json"
        report["benchmark_manifest_path"] = str(report_path.resolve())
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        del processor, model
        return report
    if benchmark_then_extract_if_under_hours is not None:
        benchmark = report["native_listener_benchmark"]
        projected_seconds = benchmark.get("projected_total_seconds_at_measured_context_rate")
        if projected_seconds is None:
            raise RuntimeError("auto-extract richiede un benchmark cluster nativo reale")
        threshold_seconds = float(benchmark_then_extract_if_under_hours) * 3600.0
        report["benchmark_auto_extract_gate"] = {
            "projected_seconds": float(projected_seconds),
            "threshold_seconds": threshold_seconds,
            "proceed": float(projected_seconds) <= threshold_seconds,
            "same_merged_model_reused_for_extraction": True,
        }
        output.mkdir(parents=True, exist_ok=True)
        benchmark_path = output / "benchmark_manifest.json"
        report["benchmark_manifest_path"] = str(benchmark_path.resolve())
        benchmark_snapshot = dict(report)
        benchmark_snapshot["status"] = "native_listener_benchmark_auto_gate"
        benchmark_path.write_text(
            json.dumps(benchmark_snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if float(projected_seconds) > threshold_seconds:
            report["status"] = "benchmark_projection_over_auto_extract_limit"
            del processor, model
            return report
        report["benchmark_auto_extract_gate"]["proceeded_in_same_process"] = True
    for split in ("train", "validation", "test"):
        conversations = split_conversations.get(split)
        if conversations is None:
            continue
        split_features: list[np.ndarray] = []
        split_metadata: list[dict[str, Any]] = []
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for conversation in conversations:
            group = str(conversation.get("source_group_id") or conversation["conversation_id"])
            grouped.setdefault(group, []).append(conversation)
        for group_conversations in grouped.values():
            # Retain decoded/resampled PCM only while paired role views of one
            # physical recording are advanced through the causal listener.
            group_key = str(
                group_conversations[0].get("source_group_id")
                or group_conversations[0]["conversation_id"]
            )
            pcm_cache = benchmark_pcm_cache if split == "train" and group_key in benchmark_groups else {}
            for conversation in group_conversations:
                if not _causal_action_targets(conversation)[0]:
                    continue
                values, metadata = extract_conversation_features(
                    listener,
                    conversation,
                    session_id=f"{split}:{conversation['conversation_id']}",
                    pcm_cache=pcm_cache,
                )
                split_features.append(values)
                split_metadata.extend(metadata)
        features = np.concatenate(split_features, axis=0) if split_features else np.empty((0, 1536), dtype=np.float32)
        _write_features(output / f"{split}.npz", features, split_metadata)
        report["counts"][split] = {
            "conversations": len(conversations),
            "frames": len(split_metadata),
            "labeled_frames": sum(bool(row["action_mask"]) for row in split_metadata),
            "causal_input_frame_steps": int(
                workload.get(split, {}).get("causal_input_frame_steps", 0)
            ),
            "conversation_replays": int(workload.get(split, {}).get("conversation_replays", 0)),
            "per_target_context_frame_sum": sum(
                int(row["context_frame_count"]) for row in split_metadata
            ),
        }
    if callable(prefix_cache_stats):
        report["canonical_prefix_cache_after_extraction"] = prefix_cache_stats()
    report_path = output / "feature_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    del processor, model
    return report


def train_control_head(
    *,
    feature_dir: str | Path,
    output_path: str | Path,
    training_manifest_path: str | Path,
    max_steps: int = 500,
    learning_rate: float = 1e-3,
    architecture: str = "linear",
    class_weighting: str = "none",
    event_onset_loss_multiplier: float = 1.0,
    save_infeasible_diagnostic_checkpoint: bool = False,
    validation_checkpoint_dir: str | Path | None = None,
    selection_metric: str | None = None,
    selection_interval: int = 25,
    sampling_strategy: str = "class_uniform",
    max_validation_v5_background_false_event_rate: float | None = None,
    early_stopping_patience: int | None = None,
    minimum_steps: int = 0,
    seed: int = 20260930,
) -> dict[str, Any]:
    """Fit the six-class head; unknown/unresolved action labels use ignore_index."""
    feature_dir = Path(feature_dir)
    train_path = feature_dir / "train.npz"
    if not train_path.is_file():
        raise FileNotFoundError(f"feature stage2 train mancanti: {train_path}")
    feature_manifest_path = feature_dir / "feature_manifest.json"
    if not feature_manifest_path.is_file():
        raise FileNotFoundError(f"feature stage2 manifest mancante: {feature_manifest_path}")
    feature_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if feature_manifest.get("merged_before_native_listener") is not True:
        raise ValueError("stage2 rifiuta feature non estratte dopo il merge LoRA")
    feature_contract = feature_manifest.get("feature_contract")
    if not isinstance(feature_contract, dict):
        raise ValueError("feature stage2 privo di un feature_contract valido")
    input_dim = feature_contract.get("input_dim")
    try:
        expected_feature_contract = listener_feature_contract(input_dim)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"feature stage2 input_dim non supportato: {input_dim!r}") from exc
    if feature_contract != expected_feature_contract:
        raise ValueError("feature stage2 non rispettano il contratto native listener dichiarato")
    train_metadata_path = feature_dir / "train.json"
    if not train_metadata_path.is_file():
        raise FileNotFoundError(f"metadata frame/session stage2 mancanti: {train_metadata_path}")
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    with np.load(train_path, allow_pickle=False) as values:
        train_array = values["features"].copy()
        train_label_array = values["labels"].copy()
        train_frame_ids = values["frame_ids"].copy()
        train_listener_ids = values["listener_frame_ids"].copy()
    if train_array.ndim != 2 or train_array.shape[1] != input_dim:
        raise ValueError(
            f"features stage2 devono essere [N,{input_dim}], ricevuto {train_array.shape}"
        )
    if len(train_metadata) != len(train_label_array) or train_frame_ids.shape != train_label_array.shape or train_listener_ids.shape != train_label_array.shape:
        raise ValueError("feature/label/metadata/frame IDs stage2 non sono allineati")
    if not np.isfinite(train_array).all():
        raise ValueError("features stage2 contengono NaN/Inf")
    if np.any((train_label_array < -100) | (train_label_array >= len(ACTIONS))):
        raise ValueError("stage2 label index fuori range")
    for index, row in enumerate(train_metadata):
        if int(row.get("frame_id", -1)) != int(train_frame_ids[index]):
            raise ValueError("stage2 frame_id metadata non combacia con NPZ")
        if int(row.get("listener_frame_id", -2)) != int(train_listener_ids[index]):
            raise ValueError("stage2 listener_frame_id metadata non combacia con NPZ")
        if int(train_label_array[index]) >= 0:
            expected_action = ACTIONS[int(train_label_array[index])]
            if not bool(row.get("action_mask")) or row.get("action") != expected_action:
                raise ValueError("stage2 label non corrisponde all'azione esplicita/mask metadata")
    labels = torch.from_numpy(train_label_array).long()
    features = torch.from_numpy(train_array)
    valid = labels.ne(-100)
    if not bool(valid.any()):
        raise ValueError("nessuna action supervision esplicita nel train split stage2")
    if max_steps <= 0:
        raise ValueError("max_steps deve essere positivo")
    if architecture not in {
        "linear",
        "mlp_256_gelu_dropout_0.1",
        "native_phase_fusion_mlp",
    }:
        raise ValueError(f"architettura control head non supportata: {architecture!r}")
    if class_weighting not in {"none", "sqrt_inverse_frequency"}:
        raise ValueError(f"class weighting non supportato: {class_weighting!r}")
    if not np.isfinite(float(event_onset_loss_multiplier)) or event_onset_loss_multiplier <= 0:
        raise ValueError("event_onset_loss_multiplier deve essere finito e positivo")
    if save_infeasible_diagnostic_checkpoint and selection_metric != "validation_v5_event_onset_macro_f1":
        raise ValueError("il checkpoint diagnostico infeasible si applica solo alla selezione V5 event-onset")
    if validation_checkpoint_dir is not None and selection_metric != "validation_v5_event_onset_macro_f1":
        raise ValueError("i checkpoint di stato validation richiedono la selezione V5 event-onset")
    if selection_metric not in {
        None,
        "validation_natural_macro_f1",
        "validation_v5_event_onset_macro_f1",
    }:
        raise ValueError(f"metrica di selezione checkpoint non supportata: {selection_metric!r}")
    if selection_metric is not None and selection_interval <= 0:
        raise ValueError("selection_interval deve essere positivo")
    if sampling_strategy not in {"class_uniform", "empirical_rows"}:
        raise ValueError(f"sampling strategy non supportata: {sampling_strategy!r}")
    if selection_metric == "validation_v5_event_onset_macro_f1":
        if (
            max_validation_v5_background_false_event_rate is None
            or not 0 <= float(max_validation_v5_background_false_event_rate) <= 1
        ):
            raise ValueError(
                "selezione V5 event-onset richiede un limite background false-event tra 0 e 1"
            )
    elif max_validation_v5_background_false_event_rate is not None:
        raise ValueError("il limite V5 false-event si applica solo alla selezione V5 event-onset")
    if early_stopping_patience is not None:
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience deve essere positivo")
        if minimum_steps < 0 or minimum_steps > max_steps:
            raise ValueError("minimum_steps deve essere tra zero e max_steps")
    elif minimum_steps:
        raise ValueError("minimum_steps richiede early_stopping_patience")

    # Ensure whole conversation/source groups remain in a single split.
    metadata_by_split: dict[str, list[dict[str, Any]]] = {"train": train_metadata}
    feature_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "train": (train_array, train_label_array)
    }
    for split in ("validation", "test"):
        array_path = feature_dir / f"{split}.npz"
        json_path = feature_dir / f"{split}.json"
        if not array_path.is_file() or not json_path.is_file():
            continue
        split_metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(array_path, allow_pickle=False) as values:
            split_features = values["features"].copy()
            split_labels = values["labels"].copy()
            frame_ids = values["frame_ids"].copy()
            listener_ids = values["listener_frame_ids"].copy()
        if split_features.ndim != 2 or split_features.shape[1] != input_dim or not np.isfinite(split_features).all():
            raise ValueError(f"{split} stage2 features hanno shape o valori non validi")
        if len(split_metadata) != len(split_labels) or frame_ids.shape != split_labels.shape or listener_ids.shape != split_labels.shape:
            raise ValueError(f"{split} stage2 feature/label/metadata non allineati")
        for index, row in enumerate(split_metadata):
            if int(row.get("frame_id", -1)) != int(frame_ids[index]) or int(row.get("listener_frame_id", -2)) != int(listener_ids[index]):
                raise ValueError(f"{split} stage2 frame IDs non allineati")
            if int(split_labels[index]) >= 0:
                expected_action = ACTIONS[int(split_labels[index])]
                if not bool(row.get("action_mask")) or row.get("action") != expected_action:
                    raise ValueError(f"{split} label non corrisponde all'azione esplicita/mask metadata")
        metadata_by_split[split] = split_metadata
        feature_arrays[split] = (split_features, split_labels)
    group_split: dict[str, str] = {}
    for split, rows in metadata_by_split.items():
        for row in rows:
            group = str(row.get("source_group_id") or row.get("conversation_id") or "")
            if not group:
                raise ValueError(f"{split} stage2 metadata privo di conversation/source_group id")
            previous = group_split.setdefault(group, split)
            if previous != split:
                raise ValueError(f"stage2 conversation group leakage: {group} in {previous} e {split}")
    if selection_metric == "validation_v5_event_onset_macro_f1":
        validation_rows = metadata_by_split.get("validation", [])
        validation_labels = feature_arrays.get("validation", (None, None))[1]
        if validation_labels is None or not any(
            int(validation_labels[index]) >= 0
            and (
                row.get("profile") == "audio_grounded_v5"
                or row.get("source") == "authored_synthetic_audio_candidate"
            )
            and row.get("event_onset") is True
            for index, row in enumerate(validation_rows)
        ):
            raise ValueError("selezione V5 event-onset richiede validation V5 uncapped con event_onset")

    validation_checkpoint_path: Path | None = None
    if validation_checkpoint_dir is not None:
        validation_checkpoint_path = Path(validation_checkpoint_dir)
        if validation_checkpoint_path.exists():
            raise FileExistsError(
                f"la directory dei checkpoint validation deve essere nuova: {validation_checkpoint_path}"
            )
        validation_checkpoint_path.mkdir(parents=True)

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    head = ControlHead(architecture=architecture, input_dim=input_dim).cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(learning_rate))
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1).tolist()
    class_indices = {
        action_index: [index for index in valid_indices if int(labels[index]) == action_index]
        for action_index in range(len(ACTIONS))
    }
    supported_classes = [index for index, indices in class_indices.items() if indices]
    if not supported_classes:
        raise ValueError("nessuna classe action presente nel train split stage2")
    train_class_support = {
        action: len(class_indices[index]) for index, action in enumerate(ACTIONS)
    }
    class_weight_values = np.zeros(len(ACTIONS), dtype=np.float32)
    if class_weighting == "sqrt_inverse_frequency":
        for class_index in supported_classes:
            class_weight_values[class_index] = 1.0 / np.sqrt(len(class_indices[class_index]))
        class_weight_values /= float(class_weight_values[supported_classes].mean())
        class_weight_tensor: torch.Tensor | None = torch.from_numpy(class_weight_values)
    else:
        class_weight_tensor = None
    training_event_onsets_by_action = Counter(
        ACTIONS[int(labels[index])]
        for index in valid_indices
        if train_metadata[index].get("event_onset") is True
    )
    head.train()
    losses: list[float] = []
    update_exposures = Counter()
    event_onset_update_exposures = Counter()
    selection_history: list[dict[str, Any]] = []
    best_selection_score: float | None = None
    best_selection_step: int | None = None
    best_selection_state: dict[str, torch.Tensor] | None = None
    best_selection_key: tuple[float, float, float] | None = None
    best_diagnostic_score: float | None = None
    best_diagnostic_step: int | None = None
    best_diagnostic_state: dict[str, torch.Tensor] | None = None
    best_diagnostic_metrics: dict[str, Any] | None = None
    last_selection_improvement_step = 0
    stopped_early = False
    for step in range(int(max_steps)):
        if sampling_strategy == "class_uniform":
            selected_class = random.choice(supported_classes)
            index = random.choice(class_indices[selected_class])
        else:
            index = random.choice(valid_indices)
            selected_class = int(labels[index])
        logits = head(features[index:index + 1])
        per_example_loss = F.cross_entropy(
            logits,
            labels[index:index + 1],
            weight=class_weight_tensor,
            reduction="none",
        )
        is_event_onset = train_metadata[index].get("event_onset") is True
        onset_multiplier = float(event_onset_loss_multiplier) if is_event_onset else 1.0
        loss = (per_example_loss * onset_multiplier).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        update_exposures[ACTIONS[selected_class]] += 1
        if is_event_onset:
            event_onset_update_exposures[ACTIONS[selected_class]] += 1
        completed_steps = step + 1
        if selection_metric is not None and (
            completed_steps % int(selection_interval) == 0 or completed_steps == int(max_steps)
        ):
            if "validation" not in feature_arrays:
                raise ValueError("checkpoint selection richiede lo split validation")
            val_features, val_labels = feature_arrays["validation"]
            val_metadata = metadata_by_split["validation"]
            if selection_metric == "validation_natural_macro_f1":
                validation_score = _natural_validation_macro_f1(
                    head, val_features, val_labels, val_metadata
                )
                record = {"step": completed_steps, **validation_score}
                score = validation_score.get("macro_f1")
                selection_key = (
                    float(score),
                    0.0,
                    0.0,
                ) if score is not None else None
                selection_is_feasible = score is not None
                selection_value = float(score) if score is not None else None
            else:
                valid_positions = np.flatnonzero(val_labels >= 0)
                head.eval()
                with torch.inference_mode():
                    val_predictions = head(
                        torch.from_numpy(val_features[valid_positions])
                    ).argmax(dim=-1).cpu().numpy()
                head.train()
                validation_score = _v5_event_onset_selection_metrics(
                    val_labels[valid_positions],
                    val_predictions,
                    [val_metadata[int(index)] for index in valid_positions],
                )
                event_per_class = validation_score["event_onsets_six_class"]["per_class"]
                supported_event_actions = [
                    action
                    for action in validation_score["event_action_outputs"]
                    if event_per_class[action]["support"] > 0
                ]
                meaningful_onset_recall = bool(supported_event_actions) and all(
                    float(event_per_class[action]["recall"] or 0.0) > 0.0
                    for action in supported_event_actions
                )
                background_rate = validation_score["background_false_event_rate"]
                budget_met = (
                    background_rate is not None
                    and float(background_rate) <= float(max_validation_v5_background_false_event_rate)
                )
                score = validation_score.get("event_onset_macro_f1")
                full_macro_f1 = validation_score["v5_full_six_class"].get("macro_f1")
                selection_value = float(score) if score is not None else None
                selection_key = (
                    float(score),
                    float(full_macro_f1 or 0.0),
                    -float(background_rate if background_rate is not None else 1.0),
                ) if score is not None else None
                selection_is_feasible = bool(
                    budget_met and meaningful_onset_recall and score is not None
                )
                record = {
                    "step": completed_steps,
                    **validation_score,
                    "false_event_budget": float(max_validation_v5_background_false_event_rate),
                    "background_false_event_budget_met": bool(budget_met),
                    "meaningful_event_onset_recall": meaningful_onset_recall,
                    "checkpoint_feasible": selection_is_feasible,
                }
            selection_history.append(record)
            if validation_checkpoint_path is not None:
                state_path = validation_checkpoint_path / f"step_{completed_steps:04d}.pt"
                head.save(state_path)
                record["cpu_validation_state_checkpoint"] = str(state_path.resolve())
            current_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            if selection_value is not None and (
                best_diagnostic_score is None or selection_value > best_diagnostic_score
            ):
                best_diagnostic_score = selection_value
                best_diagnostic_step = completed_steps
                best_diagnostic_state = current_state
                if selection_metric == "validation_v5_event_onset_macro_f1":
                    best_diagnostic_metrics = validation_score
            if selection_value is not None and selection_is_feasible and (
                selection_key is not None
                and (best_selection_key is None or selection_key > best_selection_key)
            ):
                best_selection_key = selection_key
                best_selection_score = selection_value
                best_selection_step = completed_steps
                best_selection_state = current_state
                last_selection_improvement_step = completed_steps
            elif selection_metric == "validation_natural_macro_f1" and selection_value is not None and (
                best_selection_score is None or selection_value > best_selection_score
            ):
                best_selection_score = selection_value
                best_selection_step = completed_steps
                best_selection_state = current_state
                last_selection_improvement_step = completed_steps
            if (
                early_stopping_patience is not None
                and completed_steps >= max(int(minimum_steps), int(selection_interval))
                and last_selection_improvement_step > 0
                and completed_steps - last_selection_improvement_step >= int(early_stopping_patience)
            ):
                stopped_early = True
                break
    if selection_metric is not None:
        if selection_metric == "validation_v5_event_onset_macro_f1" and best_selection_state is None:
            if best_diagnostic_state is not None:
                head.load_state_dict(best_diagnostic_state, strict=True)
        elif best_selection_state is None or best_selection_step is None or best_selection_score is None:
            raise RuntimeError("checkpoint selection non ha prodotto alcuno stato validation valido")
        else:
            head.load_state_dict(best_selection_state, strict=True)
    head.eval()

    def _split_metrics(split_name: str) -> dict[str, Any] | None:
        if split_name not in feature_arrays:
            return None
        split_features, split_labels_array = feature_arrays[split_name]
        valid_positions = np.flatnonzero(split_labels_array >= 0)
        if not len(valid_positions):
            return {"examples": 0, "reason": "no_explicit_action_labels"}
        with torch.inference_mode():
            predictions = head(torch.from_numpy(split_features[valid_positions])).argmax(dim=-1).cpu().numpy()
        targets = split_labels_array[valid_positions].astype(np.int64, copy=False)

        train_prior = {
            action: len(class_indices[index]) for index, action in enumerate(ACTIONS)
        }

        def summarize(
            selected_targets: np.ndarray,
            selected_predictions: np.ndarray,
            actions: Sequence[str],
        ) -> dict[str, Any]:
            action_indices = [ACTIONS.index(action) for action in actions]
            keep = np.isin(selected_targets, action_indices)
            selected_targets = selected_targets[keep]
            selected_predictions = selected_predictions[keep]
            if not len(selected_targets):
                return {
                    "examples": 0,
                    "actions": list(actions),
                    "absent_classes": list(actions),
                    "reason": "no_examples_for_selected_actions",
                }
            local_index = {action_index: index for index, action_index in enumerate(action_indices)}
            # Keep every model output column even for the natural four-class
            # subset; YIELD/REPLAN predictions must count as false positives.
            matrix = np.zeros((len(actions), len(ACTIONS)), dtype=np.int64)
            for target, prediction in zip(selected_targets, selected_predictions):
                if int(target) in local_index:
                    matrix[local_index[int(target)], int(prediction)] += 1
            support = matrix.sum(axis=1)
            predicted_support = matrix.sum(axis=0)
            correct_predictions = sum(
                int(matrix[index, ACTIONS.index(action)])
                for index, action in enumerate(actions)
            )
            per_class: dict[str, Any] = {}
            f1_values = []
            recalls = []
            for index, action in enumerate(actions):
                global_index = ACTIONS.index(action)
                tp = int(matrix[index, global_index])
                recall = float(tp / support[index]) if support[index] else None
                precision = float(tp / predicted_support[global_index]) if predicted_support[global_index] else 0.0
                f1 = float(2 * precision * recall / (precision + recall)) if recall is not None and precision + recall else 0.0
                per_class[action] = {
                    "support": int(support[index]),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1 if support[index] else None,
                }
                if support[index]:
                    f1_values.append(f1)
                    recalls.append(recall)
            present_indices = [index for index, count in enumerate(support) if count]
            present_actions = [actions[index] for index in present_indices]
            majority_train_action = max(
                present_actions,
                key=lambda action: train_prior.get(action, 0),
                default=None,
            )
            examples = int(len(selected_targets))
            true_counts = support[present_indices]
            majority_accuracy = (
                float(support[actions.index(majority_train_action)] / examples)
                if majority_train_action is not None
                else None
            )
            prior_random_accuracy = float(np.square(true_counts).sum() / (examples * examples))
            return {
                "examples": examples,
                "actions": list(actions),
                "accuracy": float(correct_predictions / examples),
                "macro_f1": float(np.mean(f1_values)) if f1_values else None,
                "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
                "class_support": {action: int(support[index]) for index, action in enumerate(actions)},
                "absent_classes": [action for index, action in enumerate(actions) if not support[index]],
                "per_class": per_class,
                "confusion_matrix": {
                    "true_actions": list(actions),
                    "predicted_actions": list(ACTIONS),
                    "values": matrix.tolist(),
                },
                "chance_baselines": {
                    "uniform_random_over_present_classes_accuracy": float(1.0 / len(present_indices)),
                    "empirical_prior_random_expected_accuracy": prior_random_accuracy,
                },
                "majority_baseline_from_train_prior": {
                    "action": majority_train_action,
                    "accuracy": majority_accuracy,
                    "training_class_support": {
                        action: int(train_prior.get(action, 0)) for action in actions
                    },
                },
            }

        overall = summarize(targets, predictions, ACTIONS)
        row_metadata = metadata_by_split[split_name]
        if len(row_metadata) != len(split_labels_array):
            raise ValueError(f"{split_name} metadata non allineati durante il report per-source")
        valid_metadata = [row_metadata[int(index)] for index in valid_positions]
        source_indices: dict[str, list[int]] = {}
        natural_actions = ("LISTEN", "SPEAK", "CONTINUE", "BACKCHANNEL")
        natural_action_indices = {ACTIONS.index(action) for action in natural_actions}
        natural_positions: list[int] = []
        synthetic_positions: list[int] = []
        audio_grounded_v5_positions: list[int] = []
        audio_grounded_v5_event_onset_positions: list[int] = []
        audio_grounded_v5_non_onset_positions: list[int] = []
        for position, row in enumerate(valid_metadata):
            source = str(row.get("source", "unknown"))
            source_indices.setdefault(source, []).append(position)
            if _is_synthetic_source(row):
                synthetic_positions.append(position)
            if (
                row.get("profile") == "audio_grounded_v5"
                or source == "authored_synthetic_audio_candidate"
            ):
                audio_grounded_v5_positions.append(position)
                if row.get("event_onset") is True:
                    audio_grounded_v5_event_onset_positions.append(position)
                else:
                    audio_grounded_v5_non_onset_positions.append(position)
            if not _is_synthetic_source(row) and int(targets[position]) in natural_action_indices:
                natural_positions.append(position)
        by_source = {
            source: summarize(targets[indices], predictions[indices], ACTIONS)
            for source, indices in sorted(source_indices.items())
        }
        subgroup_metrics = {
            "natural_four_class": (
                _natural_validation_macro_f1(
                    head,
                    split_features,
                    split_labels_array,
                    row_metadata,
                    train_class_support=train_prior,
                )
                if natural_positions
                else summarize(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), natural_actions)
            ),
            "synthetic_six_class": summarize(
                targets[synthetic_positions], predictions[synthetic_positions], ACTIONS
            ),
            "audio_grounded_v5_six_class": summarize(
                targets[audio_grounded_v5_positions],
                predictions[audio_grounded_v5_positions],
                ACTIONS,
            ),
            "audio_grounded_v5_event_onsets_six_class": summarize(
                targets[audio_grounded_v5_event_onset_positions],
                predictions[audio_grounded_v5_event_onset_positions],
                ACTIONS,
            ),
        }
        v5_selection_metrics = _v5_event_onset_selection_metrics(
            targets, predictions, valid_metadata
        )
        subgroup_metrics["audio_grounded_v5_event_onset_background_false_event_rate"] = {
            "examples": v5_selection_metrics["background_examples"],
            "event_action_outputs": v5_selection_metrics["event_action_outputs"],
            "false_event_action_predictions": v5_selection_metrics[
                "background_false_event_predictions"
            ],
            "false_event_rate": v5_selection_metrics["background_false_event_rate"],
            "false_event_predictions_by_class": v5_selection_metrics[
                "background_false_event_predictions_by_class"
            ],
            "denominator": v5_selection_metrics["background_definition"],
        }
        subgroup_metrics[
            "audio_grounded_v5_event_onset_legacy_off_onset_event_output_rate_not_false_action_rate"
        ] = {
            "examples": v5_selection_metrics["legacy_off_onset_examples"],
            "event_action_outputs": v5_selection_metrics["event_action_outputs"],
            "event_output_predictions": v5_selection_metrics[
                "legacy_off_onset_event_output_predictions"
            ],
            "event_output_rate": v5_selection_metrics[
                "legacy_off_onset_event_output_rate_not_false_action_rate"
            ],
            "note": v5_selection_metrics["legacy_metric_note"],
        }
        source_support = {}
        for split, rows in metadata_by_split.items():
            counts: dict[str, Counter[str]] = {}
            source_groups: dict[str, set[str]] = {}
            for row in rows:
                if not bool(row.get("action_mask")) or row.get("action") not in ACTIONS:
                    continue
                source = str(row.get("source", "unknown"))
                counts.setdefault(source, Counter())[str(row["action"])] += 1
                source_groups.setdefault(source, set()).add(
                    str(row.get("source_group_id") or row.get("conversation_id") or "")
                )
            source_support[split] = {
                source: {
                    "examples": int(sum(action_counts.values())),
                    "source_groups": len(source_groups.get(source, set())),
                    "class_support": {
                        action: int(action_counts.get(action, 0)) for action in ACTIONS
                    },
                    "absent_classes": [
                        action for action in ACTIONS if not action_counts.get(action, 0)
                    ],
                }
                for source, action_counts in sorted(counts.items())
            }
        overall["by_source"] = by_source
        overall["subgroups"] = subgroup_metrics
        overall["source_class_support_by_split"] = source_support
        overall["source_groups_by_split"] = {
            split: {
                "count": len({
                    str(row.get("source_group_id") or row.get("conversation_id") or "")
                    for row in rows
                }),
                "disjoint_with_other_splits": True,
            }
            for split, rows in metadata_by_split.items()
        }
        return overall

    output = Path(output_path)
    candidate_selection_infeasible = (
        selection_metric == "validation_v5_event_onset_macro_f1"
        and best_selection_state is None
    )
    diagnostic_checkpoint_saved = bool(
        candidate_selection_infeasible
        and save_infeasible_diagnostic_checkpoint
        and best_diagnostic_state is not None
    )
    head_saved = not candidate_selection_infeasible or diagnostic_checkpoint_saved
    if head_saved:
        head.save(output)
    if sampling_strategy == "class_uniform":
        sampling_probabilities = {
            action: (1.0 / len(supported_classes) if class_indices[index] else 0.0)
            for index, action in enumerate(ACTIONS)
        }
        sampling_description = "uniform_over_observed_action_classes_then_uniform_within_class"
    else:
        sampling_probabilities = {
            action: (len(class_indices[index]) / len(valid_indices))
            for index, action in enumerate(ACTIONS)
        }
        sampling_description = "uniform_over_all_supervised_training_rows"
    if selection_metric == "validation_v5_event_onset_macro_f1":
        diagnostic_history = next(
            (row for row in selection_history if row["step"] == best_diagnostic_step),
            None,
        )
        selected_history = next(
            (row for row in selection_history if row["step"] == best_selection_step),
            None,
        )
        checkpoint_selection: dict[str, Any] | None = {
            "metric": selection_metric,
            "validation_population": "all_explicitly_supervised_uncapped_V5_validation_event_schedule_onsets",
            "interval_updates": int(selection_interval),
            "evaluated_steps": [int(row["step"]) for row in selection_history],
            "selected_step": best_selection_step,
            "selected_event_onset_macro_f1": best_selection_score,
            "selected_validation_event_onset_examples": (
                int(selected_history["event_onset_examples"]) if selected_history else None
            ),
            "background_false_event_rate_budget": float(
                max_validation_v5_background_false_event_rate
            ),
            "background_definition": (
                "off-onset supervised V5 targets whose true action is LISTEN or CONTINUE; "
                "CONTINUE event onsets and event-action interiors are excluded"
            ),
            "background_false_event_budget_met": bool(selected_history is not None),
            "meaningful_event_onset_recall_required": True,
            "selection_status": "feasible_checkpoint_selected" if selected_history else "infeasible_no_checkpoint_met_budget_and_recall_gate",
            "promotion_eligible": bool(selected_history is not None),
            "diagnostic_checkpoint_saved_for_validation_calibration_only": diagnostic_checkpoint_saved,
            "diagnostic_best_onset_f1_step": best_diagnostic_step,
            "diagnostic_best_onset_f1_metrics": best_diagnostic_metrics,
            "history": selection_history,
            "validation_state_checkpoint_dir": (
                str(validation_checkpoint_path.resolve())
                if validation_checkpoint_path is not None
                else None
            ),
            "test_used_for_selection": False,
            "heldout_test_input_present": "test" in feature_arrays,
        }
    elif selection_metric is None:
        checkpoint_selection = None
    else:
        selected_history = next(
            row for row in selection_history if row["step"] == best_selection_step
        )
        checkpoint_selection = {
            "metric": selection_metric,
            "validation_population": "natural_labeled_LISTEN_SPEAK_CONTINUE_BACKCHANNEL_only",
            "interval_updates": int(selection_interval),
            "evaluated_steps": [int(row["step"]) for row in selection_history],
            "selected_step": best_selection_step,
            "selected_macro_f1": best_selection_score,
            "selected_validation_examples": int(selected_history["examples"]),
            "history": selection_history,
            "test_used_for_selection": False,
        }
    metadata = {
        "control_head_path": str(output.resolve()) if head_saved else None,
        "control_head": {
            "input_dim": input_dim,
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(ACTIONS),
            "architecture": architecture,
            "feature_contract": feature_contract,
        },
        "stage2_training": {
            "examples": int(len(valid_indices)),
            "masked_frames": int(len(labels) - len(valid_indices)),
            "steps": len(losses),
            "mean_loss": sum(losses) / len(losses),
            "architecture": architecture,
            "head_parameters": sum(parameter.numel() for parameter in head.parameters()),
            "trainable_head_parameters": sum(
                parameter.numel() for parameter in head.parameters() if parameter.requires_grad
            ),
            "precomputed_feature_encoder_frozen": True,
            "class_support": {
                action: int(train_class_support[action]) for index, action in enumerate(ACTIONS)
            },
            "sampling_strategy": sampling_strategy,
            "class_sampling_probabilities": sampling_probabilities,
            "loss": (
                {"function": "unweighted_cross_entropy", "class_weights": None}
                if class_weighting == "none"
                else {
                    "function": "weighted_cross_entropy_reduction_none_then_example_scale",
                    "class_weighting": class_weighting,
                    "class_weights": {
                        action: float(class_weight_values[index])
                        for index, action in enumerate(ACTIONS)
                    },
                    "class_weight_normalization": "arithmetic_mean_1_over_observed_training_classes",
                }
            ),
            "event_onset_loss_weighting": {
                "multiplier": float(event_onset_loss_multiplier),
                "applied_when": "training_metadata.event_onset is true for any supervised action, including CONTINUE transitions",
                "training_event_onset_examples_by_action": {
                    action: int(training_event_onsets_by_action.get(action, 0))
                    for action in ACTIONS
                },
                "optimizer_exposures_to_event_onsets_by_action": {
                    action: int(event_onset_update_exposures.get(action, 0))
                    for action in ACTIONS
                },
            },
            "optimizer_update_exposures_by_action": {
                action: int(update_exposures.get(action, 0)) for action in ACTIONS
            },
            "expected_updates_per_observed_class": {
                action: float(len(losses) * sampling_probabilities[action])
                for action in ACTIONS
            },
            "checkpoint_selection": checkpoint_selection,
            "early_stopping": {
                "patience_updates": early_stopping_patience,
                "minimum_steps": int(minimum_steps),
                "stopped_early": stopped_early,
                "completed_steps": len(losses),
            },
            "checkpoint_saved": head_saved,
            "diagnostic_checkpoint_role": (
                "best_infeasible_event_onset_checkpoint_saved_for_validation_calibration_only"
                if diagnostic_checkpoint_saved
                else "best_infeasible_event_onset_checkpoint_not_saved"
                if candidate_selection_infeasible
                else "selected_checkpoint"
            ),
            "candidate_promotion_eligible": not candidate_selection_infeasible,
            "absent_classes": [ACTIONS[index] for index in range(len(ACTIONS)) if not class_indices[index]],
            "sampling": sampling_description,
            "seed": int(seed),
            "train_diagnostics": _split_metrics("train"),
            "validation": _split_metrics("validation"),
            "heldout_test": _split_metrics("test"),
            "feature_manifest": str(feature_manifest_path.resolve()),
        },
    }
    manifest_path = Path(training_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest stage1 mancante: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("training manifest schema_version non supportato")
    manifest.update(metadata)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "extract_conversation_features",
    "extract_stage2_features",
    "load_merged_base",
    "train_control_head",
]
