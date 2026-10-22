"""Shared, bounded semantic prefix for native Gemma Control6 training/runtime."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


# The old semantic checkpoint was trained with a 291-token prefix bound.  The
# runtime now accepts longer actual prefixes while retaining the canonical
# prime shape at 291; metadata validation below keeps older v2 checkpoints
# compatible.
NATIVE_CONTROL6_PREFIX_MAX_TOKENS = 512
NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION = 2
NATIVE_CONTROL6_PREFIX_CONTRACT = "gemma_native_control6_semantic_prefix_boa_v2"
NATIVE_CONTROL6_MESSAGE_MAX_TOKENS = 80
NATIVE_CONTROL6_INSTRUCTIONS_MAX_TOKENS = 80
NATIVE_CONTROL6_SYSTEM_INSTRUCTION_IT = (
    "Controller conversazionale audio in italiano. A ogni frame scegli una sola azione: "
    "LISTEN ascolta; SPEAK risponde al nuovo turno utente; CONTINUE prosegue la risposta attiva; "
    "YIELD interrompe la risposta per ascoltare; BACKCHANNEL dà un breve riscontro mentre l'utente parla; "
    "REPLAN rivede il piano dopo una correzione. Usa l'audio corrente e la cronologia completata."
)


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        if "input_ids" not in value:
            raise TypeError("tokenizer output mapping has no input_ids")
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise TypeError("tokenizer output must be a list of token ids")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("native Control6 prefix builder accepts exactly one conversation")
        value = value[0]
    return [int(item) for item in value]


def validate_native_control6_prefix_ids(
    prefix_ids: Any, *, boa_token_id: int
) -> tuple[int, ...]:
    """Validate the actual token-only prefix against the runtime import bound."""
    if isinstance(boa_token_id, bool) or not isinstance(boa_token_id, int) or boa_token_id < 0:
        raise ValueError("boa_token_id must be a non-negative integer")
    ids = _as_token_ids(prefix_ids)
    if not 1 <= len(ids) <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS:
        raise ValueError(
            "native Control6 semantic prefix length must be in [1,"
            f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}], got {len(ids)}"
        )
    if ids[-1] != boa_token_id:
        raise ValueError("native Control6 semantic prefix must end with the real BOA token")
    return tuple(ids)


def _encode(tokenizer: Any, text: str) -> list[int]:
    return _as_token_ids(tokenizer.encode(str(text), add_special_tokens=False))


def _decode(tokenizer: Any, token_ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))
    except TypeError:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=True))


def _bounded_text(tokenizer: Any, text: str, max_tokens: int) -> str:
    value = str(text or "").strip()
    ids = _encode(tokenizer, value)
    if len(ids) <= int(max_tokens):
        return value
    bounded_ids = ids[:int(max_tokens)]
    bounded_text = _decode(tokenizer, bounded_ids).strip()
    while bounded_text and len(_encode(tokenizer, bounded_text)) > int(max_tokens):
        bounded_ids = bounded_ids[:-1]
        bounded_text = _decode(tokenizer, bounded_ids).strip()
    return bounded_text


def build_native_control6_prefix_messages(
    tokenizer: Any,
    *,
    instructions: str | None,
    completed_messages: Sequence[Mapping[str, Any]],
    audio_marker: str,
) -> list[dict[str, str]]:
    """Build the causal system/history/current-audio messages shared by train and serve."""
    marker = str(audio_marker or "").strip()
    if not marker:
        raise ValueError("native Control6 semantic prefix requires the Gemma audio BOA marker")
    bounded_instructions = _bounded_text(
        tokenizer, str(instructions or ""), NATIVE_CONTROL6_INSTRUCTIONS_MAX_TOKENS
    )
    system_content = NATIVE_CONTROL6_SYSTEM_INSTRUCTION_IT
    if bounded_instructions:
        system_content += "\nIstruzioni della sessione: " + bounded_instructions

    history: list[dict[str, str]] = []
    for index, message in enumerate(completed_messages):
        if not isinstance(message, Mapping):
            raise TypeError(f"completed_messages[{index}] must be a mapping")
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError(f"completed_messages[{index}] must contain a user/assistant text message")
        normalized = content.strip()
        if normalized:
            history.append({
                "role": role,
                "content": _bounded_text(
                    tokenizer, normalized, NATIVE_CONTROL6_MESSAGE_MAX_TOKENS
                ),
            })
    history = history[-16:]
    return [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": marker},
    ]


def build_native_control6_prefix_ids(
    tokenizer: Any,
    *,
    instructions: str | None,
    completed_messages: Sequence[Mapping[str, Any]],
    audio_marker: str,
    boa_token_id: int,
) -> tuple[list[int], int]:
    """Return chat-template IDs ending at the real BOA token, bounded to 512.

    Oldest completed messages are removed first when the full role/template
    overhead would exceed the native listener's fixed semantic-prefix capacity.
    ``audio_marker`` is only a location anchor in the serialized user message;
    its placeholder ID is removed and replaced with the model's real BOA ID.
    Audio payload/soft placeholders are never embedded as semantic text.
    """
    messages = build_native_control6_prefix_messages(
        tokenizer,
        instructions=instructions,
        completed_messages=completed_messages,
        audio_marker=audio_marker,
    )
    marker_ids = _encode(tokenizer, audio_marker)
    if len(marker_ids) != 1:
        raise ValueError("audio_marker must map to exactly one placeholder token")
    marker_id = marker_ids[0]
    boa_id = int(boa_token_id)
    if boa_id < 0:
        raise ValueError("boa_token_id must be non-negative")

    def serialize(current_messages: list[dict[str, str]]) -> list[int]:
        tokenized = tokenizer.apply_chat_template(
            current_messages, tokenize=True, add_generation_prompt=False
        )
        ids = _as_token_ids(tokenized)
        positions = [index for index, token_id in enumerate(ids) if token_id == marker_id]
        if not positions:
            raise ValueError("Gemma chat template omitted the final audio BOA marker")
        # The template marker locates the audio boundary, but is not the
        # semantic-prefix token. Prefix queries start after the real BOA.
        return ids[:positions[-1]] + [boa_id]

    prefix_ids = serialize(messages)
    while len(prefix_ids) > NATIVE_CONTROL6_PREFIX_MAX_TOKENS and len(messages) > 2:
        messages.pop(1)
        prefix_ids = serialize(messages)
    prefix_ids = list(validate_native_control6_prefix_ids(
        prefix_ids, boa_token_id=boa_id
    ))
    return prefix_ids, len(prefix_ids)


def select_native_control6_replay_suffix(
    observed_frames: Sequence[tuple[int, Any]],
    *,
    expected_last_frame_id: int,
    max_frames: int = 64,
) -> tuple[tuple[int, ...], tuple[Any, ...]] | None:
    """Select the contiguous observed Q4 suffix ending at the current watermark.

    ``None`` means the required endpoint is not present, so the caller must
    defer semantic-prefix refresh rather than fabricate a frame or replay stale
    features. The returned tensors are borrowed from the bounded session ring.
    """
    expected = int(expected_last_frame_id)
    limit = int(max_frames)
    if limit <= 0 or limit > 64:
        raise ValueError("max_frames deve essere compreso tra 1 e 64")
    if expected < 0:
        return (), ()
    pairs = [(int(frame_id), tensor) for frame_id, tensor in observed_frames]
    end = next((index for index in range(len(pairs) - 1, -1, -1) if pairs[index][0] == expected), None)
    if end is None:
        return None
    start = end
    while start > 0 and end - start + 1 < limit and pairs[start][0] == pairs[start - 1][0] + 1:
        start -= 1
    selected = pairs[start:end + 1]
    frame_ids = tuple(frame_id for frame_id, _ in selected)
    if any(right != left + 1 for left, right in zip(frame_ids, frame_ids[1:])):
        raise AssertionError("selected replay suffix must be contiguous")
    return frame_ids, tuple(tensor for _, tensor in selected)


def validate_native_control6_semantic_prefix_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the selected-row checkpoint matches the BOA-v2 runtime."""
    report = payload.get("report")
    if not isinstance(report, Mapping):
        raise ValueError("native Control6 checkpoint is missing its training report")
    if payload.get("architecture") != "native_gemma_selected_output_rows_lora":
        raise ValueError("native Control6 semantic-prefix checkpoint architecture mismatch")
    expected = {
        "semantic_prefix_schema_version": NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION,
        "semantic_prefix_contract": NATIVE_CONTROL6_PREFIX_CONTRACT,
        "semantic_prefix_context_conditioned": True,
    }
    result: dict[str, Any] = {}
    for key, wanted in expected.items():
        top_value = payload.get(key)
        report_value = report.get(key)
        if top_value != wanted or report_value != wanted:
            raise ValueError(
                f"native Control6 semantic-prefix metadata mismatch for {key}: "
                f"checkpoint={top_value!r}, report={report_value!r}, expected={wanted!r}"
            )
        result[key] = top_value
    for key in ("semantic_prefix_capacity", "semantic_prefix_max_tokens"):
        top_value = payload.get(key)
        report_value = report.get(key)
        if (
            isinstance(top_value, bool)
            or not isinstance(top_value, int)
            or not 1 <= top_value <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS
            or report_value != top_value
        ):
            raise ValueError(
                f"native Control6 semantic-prefix metadata mismatch for {key}: "
                f"checkpoint={top_value!r}, report={report_value!r}, runtime_max="
                f"{NATIVE_CONTROL6_PREFIX_MAX_TOKENS}"
            )
        result[key] = top_value
    if result["semantic_prefix_capacity"] != result["semantic_prefix_max_tokens"]:
        raise ValueError("native Control6 semantic-prefix capacity/max_tokens differ")
    top_boa = payload.get("semantic_prefix_boa_token_id")
    report_boa = report.get("semantic_prefix_boa_token_id")
    if not isinstance(top_boa, int) or isinstance(top_boa, bool) or top_boa < 0 or report_boa != top_boa:
        raise ValueError(
            "native Control6 semantic-prefix BOA token metadata is missing or inconsistent"
        )
    result["semantic_prefix_boa_token_id"] = top_boa
    if payload.get("response_adapter") != "OFF":
        raise ValueError("native Control6 semantic-prefix checkpoint requires response adapter OFF")
    if payload.get("ordinary_vocabulary_rows_modified") is not False:
        raise ValueError("native Control6 checkpoint must leave ordinary vocabulary rows unchanged")
    if payload.get("base_gemma_weights_modified") is not False:
        raise ValueError("native Control6 checkpoint must keep base Gemma weights frozen")
    return result


def native_control6_semantic_prefix_metadata_is_supported(
    payload: Any,
) -> bool:
    """Whether validated v2 training bounds fit the current runtime capacity."""
    if not isinstance(payload, Mapping):
        return False
    capacity = payload.get("semantic_prefix_capacity")
    max_tokens = payload.get("semantic_prefix_max_tokens")
    return bool(
        payload.get("semantic_prefix_schema_version") == NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION
        and payload.get("semantic_prefix_contract") == NATIVE_CONTROL6_PREFIX_CONTRACT
        and isinstance(capacity, int)
        and not isinstance(capacity, bool)
        and 1 <= capacity <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS
        and isinstance(max_tokens, int)
        and not isinstance(max_tokens, bool)
        and max_tokens == capacity
    )


__all__ = [
    "NATIVE_CONTROL6_INSTRUCTIONS_MAX_TOKENS",
    "NATIVE_CONTROL6_MESSAGE_MAX_TOKENS",
    "NATIVE_CONTROL6_PREFIX_MAX_TOKENS",
    "NATIVE_CONTROL6_PREFIX_SCHEMA_VERSION",
    "NATIVE_CONTROL6_PREFIX_CONTRACT",
    "NATIVE_CONTROL6_SYSTEM_INSTRUCTION_IT",
    "build_native_control6_prefix_ids",
    "build_native_control6_prefix_messages",
    "select_native_control6_replay_suffix",
    "validate_native_control6_prefix_ids",
    "native_control6_semantic_prefix_metadata_is_supported",
    "validate_native_control6_semantic_prefix_metadata",
]
