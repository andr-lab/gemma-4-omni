from __future__ import annotations

import json, logging, os, threading, time
from collections import deque
from contextlib import contextmanager
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from duplex_kernel.audio_response_conditioning import (
    build_audio_response_messages,
    build_audio_response_protocol_v2_messages,
    build_controller_guided_say_v1_messages,
)
from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS,
    CONTROLLER_GUIDED_SAY_V1,
    AudioResponseV2Collector,
    normalize_audio_response_protocol,
)
from duplex_kernel.config import DEFAULT_VOICE, USER_SAMPLE_RATE
from duplex_kernel.response_intent_local import align_generated_response_hidden_to_qwen_tokens
from duplex_kernel.server_diagnostics import _alignment_failure_diagnostics
from duplex_kernel.server_state import _verify_v9_training_reference
from duplex_kernel.turn_policy import ControlAction
from duplex_kernel.voices import resolve_voice_conditioning

logger = logging.getLogger("duplex_kernel.server_response")
_ACTION_TOKEN_MAP = {"SPEAK": "<unused90>", "BACKCHANNEL": "<unused93>", "REPLAN": "<unused94>"}
_AUDIO_RESPONSE_V2_FAILURE_TRACE_LOCK = threading.Lock()
_AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS = deque[dict[str, Any]](maxlen=AUDIO_RESPONSE_V2_FAILURE_TRACE_MAX_RECORDS)
_AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK, _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED = threading.Lock(), False

def _empty_controller_say_think_from_eos(
    exc: BaseException,
    *,
    prompt_len: int,
) -> dict[str, Any] | None:
    """Convert a natural empty EOS into the controller's no-speech result."""
    if getattr(exc, "empty_eos", False) is not True or getattr(exc, "terminal_reason", None) != "eos":
        return None
    runtime_metrics = getattr(exc, "runtime_metrics", {})
    return {
        "text": "", "gen_ids": [], "hidden": None,
        "hidden_prompt_len": int(prompt_len), "prompt_len": int(prompt_len),
        "runtime_metrics": dict(runtime_metrics) if isinstance(runtime_metrics, dict) else {},
        "terminal_reason": "eos", "cancelled": False,
    }

def _claim_audio_response_v2_action_probe() -> bool:
    """Enable one private, action-token-only diagnostic per server process."""
    global _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED
    if os.environ.get("DUPLEX_V2_ACTION_LOGIT_PROBE_ONCE") != "1": return False
    with _AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK:
        if _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED: return False
        _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED = True
        return True

def _is_v2_no_speech_result(
    *,
    response_protocol: str | None,
    modalities: Any,
    wav: Any,
    talker_metrics: Any,
    cancelled: bool = False,
) -> bool:
    """Recognize a successful v2 response whose Talker emitted no CB0/PCM."""
    if (
        response_protocol != AUDIO_RESPONSE_PROTOCOL_V2
        or "audio" not in (modalities or ["text"]) or cancelled
        or wav is None or not isinstance(talker_metrics, dict)
    ):
        return False
    pcm = np.asarray(wav).reshape(-1)
    generated_cb0_frames = talker_metrics.get("generated_cb0_frames")
    return pcm.size == 0 and isinstance(generated_cb0_frames, int) and generated_cb0_frames == 0

def _record_audio_response_v2_private_failure(
    *,
    collector: AudioResponseV2Collector,
    reason: str,
    response_id: str | None,
    action: str | None,
    budget: dict[str, int] | None,
) -> None:
    """Persist only bounded token IDs and parser counts, never raw text/audio."""
    record = {
        "timestamp_unix_ms": int(time.time() * 1000),
        "response_id": str(response_id or ""),
        "selected_action": str(action or ""),
        "budget": dict(budget) if budget is not None else None,
        **collector.private_failure_snapshot(reason=reason),
    }
    with _AUDIO_RESPONSE_V2_FAILURE_TRACE_LOCK:
        _AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS.append(record)
        path = Path(__file__).resolve().parent.parent / ".tmp" / "audio_response_v2_private_failures.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(".tmp")
        payload = "".join(
            json.dumps(item, ensure_ascii=True, separators=(",", ":")) + "\n"
            for item in _AUDIO_RESPONSE_V2_FAILURE_TRACE_RECORDS
        )
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(path)

def _resolve_request_voice_conditioning(
    engine: Any,
    sess: Any,
    request: dict[str, Any],
) -> dict[str, Any] | None:
    """Resolve per-request/session voice, unless the startup ablation is active."""
    enforce_training_voice = bool(getattr(engine, "enforce_v9_training_voice", False))
    ablate_voice_conditioning = bool(getattr(engine, "ablate_session_voice_conditioning", False))
    if enforce_training_voice and ablate_voice_conditioning:
        raise ValueError("ablation voce e enforcement V9 sono incompatibili")
    if ablate_voice_conditioning:
        return None
    if enforce_training_voice:
        conditioning = getattr(sess, "voice_conditioning", None)
        if (
            not isinstance(conditioning, dict)
            or conditioning.get("conditioning_source") != "training_default"
            or conditioning.get("mode") != "x_vector_only"
        ):
            raise ValueError("session voice non è il training default V9 verificato")
        _verify_v9_training_reference(conditioning)
        return conditioning
    explicit_voice = request.get("voice") or request.get("speaker")
    if (
        explicit_voice is None
        and not request.get("voice_style")
        and not request.get("ref_audio")
        and getattr(sess, "voice_conditioning", None) is not None
    ):
        return sess.voice_conditioning
    return resolve_voice_conditioning(
        explicit_voice or getattr(sess, "voice", None) or DEFAULT_VOICE,
        request.get("voice_style") or sess.voice_style,
        request.get("ref_audio"), request.get("ref_text"),
    )

@contextmanager
def _diagnostic_talker_rng_scope(engine: Any):
    """Expose startup seed metadata without mutating process-global RNG state."""
    yield getattr(engine, "diagnostic_talker_seed", None)

def _append_native_action_token(chat: str, action: str | None) -> str:
    """Append the fixed native action token to an assistant generation prompt."""
    if action is None: return chat
    normalized_action = str(action).strip().upper()
    token = _ACTION_TOKEN_MAP.get(normalized_action)
    if token and not chat.rstrip().endswith(token): return chat + token + " "
    return chat

def _prepare_direct_audio_processor_inputs(
    engine: Any,
    *,
    language: Any,
    history_messages: list[dict[str, Any]],
    pcm: np.ndarray,
    response_protocol: str | None = None,
    session_instructions: str | None = None,
    controller_action: str | None = None,
) -> tuple[list[int], dict[str, Any], float]:
    """Build processor-native Gemma multimodal input without a transcript/action label."""
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    if processor is None or not callable(processor):
        raise RuntimeError("processor audio Gemma non disponibile per il Thinker diretto")
    audio_marker = (
        getattr(processor, "audio_token", None)
        or getattr(engine.tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    completed_history = [
        message for message in history_messages
        if message.get("role") in {"user", "assistant"}
    ]
    normalized_protocol = normalize_audio_response_protocol(response_protocol)
    if normalized_protocol == AUDIO_RESPONSE_PROTOCOL_V2:
        prompt_messages = build_audio_response_protocol_v2_messages(
            history_messages=completed_history,
            audio_token=str(audio_marker),
            session_instructions=session_instructions,
        )
    elif normalized_protocol == CONTROLLER_GUIDED_SAY_V1:
        prompt_messages = build_controller_guided_say_v1_messages(
            language=language,
            history_messages=completed_history,
            audio_token=str(audio_marker),
            session_instructions=session_instructions,
            action=controller_action or "",
        )
    else:
        prompt_messages = build_audio_response_messages(
            language=language,
            history_messages=completed_history,
            audio_token=str(audio_marker),
        )
    chat = engine.tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    if normalized_protocol != CONTROLLER_GUIDED_SAY_V1:
        chat = _append_native_action_token(chat, controller_action)
    started = time.perf_counter()
    processor_audio_inputs = processor(
        text=[chat],
        audio=[np.asarray(pcm, dtype=np.float32).reshape(-1)],
        sampling_rate=USER_SAMPLE_RATE,
        return_tensors="pt",
        padding=True,
    )
    if not isinstance(processor_audio_inputs, dict):
        processor_audio_inputs = dict(processor_audio_inputs)
    required_audio_keys = {"input_ids", "input_features", "input_features_mask"}
    if not required_audio_keys.issubset(processor_audio_inputs):
        raise RuntimeError(
            "processor Gemma non ha prodotto le chiavi audio richieste: "
            f"{sorted(required_audio_keys - set(processor_audio_inputs))}"
        )
    input_ids = processor_audio_inputs["input_ids"]
    if getattr(input_ids, "ndim", None) != 2 or input_ids.shape[0] != 1:
        raise RuntimeError("processor Gemma deve produrre input_ids con batch 1")
    ids = [int(value) for value in input_ids[0].tolist()]
    tensors = {
        key: processor_audio_inputs[key]
        for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")
        if key in processor_audio_inputs
    }
    return ids, tensors, (time.perf_counter() - started) * 1000.0

def _direct_audio_prompt_history(
    messages: list[dict[str, Any]],
    *,
    history_is_committed: bool,
) -> list[dict[str, Any]]:
    """Exclude an API-supplied current transcript; preserve canonical session history."""
    history = [message for message in messages if message.get("role") in {"user", "assistant"}]
    if not history_is_committed and history and history[-1].get("role") == "user":
        history = history[:-1]
    return history

def _align_generated_response_hidden_with_diagnostics(
    *,
    response_hidden: Any,
    visible_generated_ids: list[int],
    gemma_tokenizer: Any,
    qwen_tokenizer: Any,
    qwen_processor: Any,
    generated_text: str,
    generated_ids: list[int] | tuple[int, ...],
    hidden_prompt_len: int,
    raw_hidden: Any,
):
    try:
        return align_generated_response_hidden_to_qwen_tokens(
            response_hidden,
            visible_generated_ids,
            gemma_tokenizer=gemma_tokenizer,
            qwen_tokenizer=qwen_tokenizer,
            qwen_processor=qwen_processor,
            text=generated_text,
        )
    except Exception:
        try:
            diagnostics = _alignment_failure_diagnostics(
                generated_ids=generated_ids,
                visible_generated_ids=visible_generated_ids,
                generated_text=generated_text,
                hidden_prompt_len=hidden_prompt_len,
                raw_hidden=raw_hidden,
                say_hidden=response_hidden,
                tokenizer=gemma_tokenizer,
            )
            logger.exception(
                "h_txt generated-response alignment failed; diagnostics=%s",
                json.dumps(diagnostics, ensure_ascii=False, sort_keys=True),
            )
        except Exception as diagnostic_error:
            try:
                logger.exception(
                    "h_txt generated-response alignment failed; diagnostics unavailable: %s",
                    diagnostic_error,
                )
            except Exception:
                pass
        raise

def _qwen_text_token_surfaces(
    text: str,
    offsets: Sequence[tuple[int, int]],
) -> tuple[str, ...]:
    """Partition canonical text using the Qwen tokenizer's exact character offsets."""
    if not isinstance(text, str) or not text:
        raise ValueError("Qwen surface text must be nonempty")
    if not offsets:
        raise ValueError("Qwen surface offsets must be nonempty")
    surfaces: list[str] = []
    cursor = 0
    for index, raw_offset in enumerate(offsets):
        if not isinstance(raw_offset, (tuple, list)) or len(raw_offset) != 2:
            raise ValueError("Qwen surface offset must contain a start and end")
        start, end = int(raw_offset[0]), int(raw_offset[1])
        if start < 0 or end <= start or end > len(text):
            raise ValueError("Qwen surface offset is outside canonical text")
        surface_end = len(text) if index == len(offsets) - 1 else max(cursor, end)
        surfaces.append(text[cursor:surface_end])
        cursor = surface_end
    if cursor != len(text) or "".join(surfaces) != text:
        raise ValueError("Qwen token surfaces do not reconstruct canonical text")
    return tuple(surfaces)

def _backchannel_budget_without_eos(
    action: str | None,
    *,
    private: bool,
    controller_guided: bool,
    terminal_reason: Any,
    forced_structure_overrides: Any = 0,
    cancelled: bool,
    buffered_tokens: Sequence[Any] = (),
) -> str | None:
    """Reject cancelled or structurally forced BACKCHANNEL termination."""
    terminal = str(terminal_reason or "")
    if str(action or "").strip().upper() != ControlAction.BACKCHANNEL.value:
        return None
    if bool(cancelled):
        return "cancelled"
    forced_structure = (
        bool(forced_structure_overrides)
        if isinstance(forced_structure_overrides, bool)
        else isinstance(forced_structure_overrides, Integral)
        and int(forced_structure_overrides) > 0
    )
    if terminal == "structure" or forced_structure or terminal not in {"eos", "stop"}:
        return "backchannel_without_eos"
    return None

def _publish_backchannel_tokens_after_eos(
    buffered_tokens: Sequence[tuple[int, int, str, Any]],
    *,
    terminal_reason: Any,
    forced_structure_overrides: Any = 0,
    publish_batch: Callable[[tuple[tuple[int, int, str, Any], ...]], Any],
    can_publish: Callable[[], bool] | None = None,
) -> bool:
    """Publish original backchannel packets as one validated, cancellable batch."""
    forced_structure = (
        bool(forced_structure_overrides)
        if isinstance(forced_structure_overrides, bool)
        else isinstance(forced_structure_overrides, Integral)
        and int(forced_structure_overrides) > 0
    )
    if (
        str(terminal_reason or "") not in {"eos", "stop"}
        or forced_structure
        or not buffered_tokens
    ):
        return False
    packets = tuple(
        (int(token_index), int(token_id), str(token_text), hidden)
        for token_index, token_id, token_text, hidden in buffered_tokens
    )
    if can_publish is not None and not can_publish():
        return False
    publish_batch(packets)
    return True

def _message_text(content: Any) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list): return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict)).strip()
    return ""
