"""Stable audio prompt, listener queue, and transition template helpers."""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Mapping, Sequence

import numpy as np

from duplex_kernel.audio_response_protocol import (
    AUDIO_RESPONSE_PROTOCOL_V2,
    CONTROLLER_GUIDED_SAY_V1,
    normalize_audio_response_protocol,
    resolve_audio_response_protocol,
)
from duplex_kernel.session import DuplexSession

def _strict_env_switch(name: str, *, default: bool = False) -> bool:
    """Read one explicit 0/1 switch without silently accepting typos."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise RuntimeError(f"{name} must be exactly '0' or '1'")

def _listener_frame_is_voiced(pcm: Any) -> bool:
    """Use only a no-signal fallback; serving VAD state takes precedence."""
    frame = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if not frame.size or not np.isfinite(frame).all():
        return False
    return bool(np.any(frame != 0.0))

def _shared_listener_queue_voice_activity(
    policy_context: Any,
    voice_activity_provider: Any = None,
) -> bool | None:
    """Resolve current speech or its exact endpoint frame for queue retention."""
    endpoint = getattr(policy_context, "speech_endpoint", False)
    if not isinstance(endpoint, bool):
        endpoint = False
    voice_state = getattr(policy_context, "policy_input", None)
    peer_active = getattr(voice_state, "peer_turn_active", None)
    if isinstance(peer_active, bool):
        return bool(peer_active or endpoint)
    if endpoint:
        return True
    if callable(voice_activity_provider):
        return bool(voice_activity_provider())
    return None

def _shared_listener_queue_retention_reason(
    policy_context: Any,
    voice_active: bool | None,
) -> str | None:
    """Name the causal frame kind that authorizes assistant-phase retention."""
    if getattr(policy_context, "speech_endpoint", False) is True:
        return "endpoint_tail"
    return "utterance_body" if voice_active is True else None

def _shared_listener_queue_onset_frame_id(policy_context: Any) -> int | None:
    """Keep the VAD's original capture ID when onset recognition is delayed."""
    value = getattr(policy_context, "voice_run_onset_frame_id", None)
    if value is None:
        if getattr(policy_context, "new_voice_onset", False) is True:
            raise RuntimeError("causal voice onset has no valid source frame ID")
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError("causal voice run has no valid source frame ID")
    return int(value)

def _unified_listener_frame_contract(
    pcm: Any,
    *,
    assistant_audio_queue_open: bool,
    user_input_pending: bool,
    voice_active: bool | None = None,
) -> tuple[int, str | None, bool]:
    """Commit two canonical audio tokens for every ordinary 80 ms frame."""
    del pcm, assistant_audio_queue_open, user_input_pending, voice_active
    return 2, None, False


def _unified_response_owns_shared_audio_queue(
    response_branch: Mapping[str, Any] | None,
) -> bool:
    """Q0 belongs to the current shared response branch once its KV queue is open."""
    return bool(
        isinstance(response_branch, Mapping)
        and response_branch.get("queue_open") is True
        and response_branch.get("ephemeral_private") is not True
    )

def _shared_listener_response_branch(
    binding: Mapping[str, Any] | None,
    response_id: Any,
) -> Mapping[str, Any] | None:
    """Resolve playback state from the session binding shared with the listener callback."""
    if not isinstance(binding, Mapping) or response_id is None:
        return None
    branches = binding.get("playback_ack_branches")
    if not isinstance(branches, Mapping):
        return None
    branch = branches.get(str(response_id))
    return branch if isinstance(branch, Mapping) else None

def _shared_listener_frame_queue_contract(
    binding: Mapping[str, Any] | None,
    pcm: Any,
    *,
    response_id: Any,
    user_input_pending: bool,
    voice_active: bool | None = None,
) -> tuple[int, str | None, bool]:
    """Apply Q/queue ownership from the current response branch, including prompt prefill."""
    response_branch = _shared_listener_response_branch(binding, response_id)
    assistant_audio_queue_open = _unified_response_owns_shared_audio_queue(
        response_branch,
    )
    return _unified_listener_frame_contract(
        pcm,
        assistant_audio_queue_open=assistant_audio_queue_open,
        user_input_pending=user_input_pending,
        voice_active=voice_active,
    )

def _clear_shared_listener_response_branches(
    binding: dict[str, Any] | None,
    branches: dict[str, dict[str, Any]],
) -> None:
    """Detach only the registry owned by this websocket from its session binding."""
    if binding is not None and binding.get("playback_ack_branches") is branches:
        binding.pop("playback_ack_branches", None)
    branches.clear()

def _attach_shared_listener_response_branches(
    binding: dict[str, Any] | None,
    branches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Attach the websocket-owned branch registry after session registration."""
    if binding is None:
        raise RuntimeError("shared listener response registry requires a registered session")
    binding["playback_ack_branches"] = branches
    return binding

def _validate_private_transition_chunk(
    transition_token_ids: Sequence[int],
    current_offset: int,
    token_ids: Sequence[int],
    *,
    source_cache_position: int,
    current_cache_position: int,
) -> int:
    """Validate one logical Q<=8 prefill chunk against the exact prompt and private cursor."""
    prompt_ids = tuple(int(value) for value in transition_token_ids)
    chunk = tuple(int(value) for value in token_ids)
    offset = int(current_offset)
    if not chunk or len(chunk) > 8 or offset < 0 or offset + len(chunk) > len(prompt_ids):
        raise RuntimeError("private transition chunk is outside the original Thinker transition")
    if int(current_cache_position) != int(source_cache_position) + offset:
        raise RuntimeError("private transition cursor differs from its contiguous Q2 prompt offset")
    if chunk != prompt_ids[offset:offset + len(chunk)]:
        raise RuntimeError("private transition chunk differs from the original Thinker transition")
    return offset + len(chunk)

def _configured_audio_response_protocol(engine: Any) -> str | None:
    del engine
    return resolve_audio_response_protocol()

def _require_unified_stable_audio_protocol(protocol: Any) -> str:
    """Unified KV requires a prompt contract that stays stable after import."""
    normalized = normalize_audio_response_protocol(protocol)
    if normalized not in {AUDIO_RESPONSE_PROTOCOL_V2, CONTROLLER_GUIDED_SAY_V1}:
        raise RuntimeError(
            "DUPLEX_UNIFIED_KV requires a supported stable audio response protocol"
        )
    return normalized

def _tokenize_template_text(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> list[int]:
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RuntimeError("unified prompt requires the loaded Gemma chat template")
    if not callable(getattr(tokenizer, "encode", None)):
        raise RuntimeError("unified prompt requires tokenizer.encode")
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=bool(add_generation_prompt),
    )
    token_ids = tokenizer.encode(str(rendered), add_special_tokens=False)
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if not isinstance(token_ids, list) or (token_ids and isinstance(token_ids[0], list)):
        raise TypeError("tokenizer.encode must return one flat token-ID sequence")
    result: list[int] = []
    for token_id in token_ids:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise TypeError("tokenizer.encode returned an invalid token ID")
        result.append(int(token_id))
    return result

def _unified_audio_template_ids(engine: Any, session: DuplexSession, *, add_generation_prompt: bool) -> tuple[list[int], int, int, int]:
    """Return tokenized one-user audio template and Gemma BOA/EOA IDs."""
    tokenizer = getattr(engine, "tokenizer", None)
    processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
    audio_marker = (
        getattr(processor, "audio_token", None)
        or getattr(tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    audio_marker = str(audio_marker).strip()
    if not audio_marker:
        raise RuntimeError("unified audio template requires the processor audio marker")
    token_ids = _tokenize_template_text(
        tokenizer,
        [{"role": "user", "content": audio_marker}],
        add_generation_prompt=add_generation_prompt,
    )
    marker_id_values = tokenizer.encode(audio_marker, add_special_tokens=False)
    if hasattr(marker_id_values, "tolist"):
        marker_id_values = marker_id_values.tolist()
    if isinstance(marker_id_values, tuple):
        marker_id_values = list(marker_id_values)
    if not isinstance(marker_id_values, (tuple, list)) or len(marker_id_values) != 1:
        raise RuntimeError("processor audio marker must map to exactly one tokenizer token")
    marker_id = int(marker_id_values[0])
    marker_positions = [index for index, value in enumerate(token_ids) if value == marker_id]
    if len(marker_positions) != 1:
        raise RuntimeError("one-user chat template must contain exactly one audio marker")
    backend = getattr(getattr(session, "listener", None), "backend", None)
    config = getattr(getattr(engine, "_thinker_wrapper", None), "config", None)
    boa_token_id = getattr(backend, "boa_token_id", None)
    if boa_token_id is None:
        boa_token_id = getattr(config, "boa_token_id", None)
    eoa_token_id = getattr(backend, "eoa_token_id", None)
    if eoa_token_id is None:
        eoa_token_id = getattr(config, "eoa_token_id", None)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (boa_token_id, eoa_token_id)):
        raise RuntimeError("unified audio template requires configured Gemma BOA and EOA token IDs")
    return token_ids, marker_positions[0], int(boa_token_id), int(eoa_token_id)

def _unified_turn_end_token_id(engine: Any) -> int:
    """Read the chat turn-end ID from the loaded tokenizer's own metadata."""
    tokenizer = getattr(engine, "tokenizer", None)
    candidates: list[int] = []
    for name in ("eot_id", "eot_token_id"):
        value = getattr(tokenizer, name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"loaded tokenizer exposes an invalid {name}")
        candidates.append(int(value))
    if not candidates:
        eot_token = getattr(tokenizer, "eot_token", None)
        convert = getattr(tokenizer, "convert_tokens_to_ids", None)
        if isinstance(eot_token, str) and eot_token and callable(convert):
            value = convert(eot_token)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError("loaded tokenizer cannot resolve its configured EOT token")
            candidates.append(int(value))
    if not candidates:
        raise RuntimeError("unified audio template requires the loaded tokenizer's EOT token ID")
    if len(set(candidates)) != 1:
        raise RuntimeError("loaded tokenizer exposes conflicting EOT token IDs")
    return candidates[0]

def _validate_unified_audio_turn_end_suffix(
    engine: Any,
    suffix_ids: Sequence[int],
) -> int:
    """Require one configured EOT while preserving every template tail token."""
    eot_token_id = _unified_turn_end_token_id(engine)
    if sum(int(token_id) == eot_token_id for token_id in suffix_ids) != 1:
        raise RuntimeError(
            "unified audio template suffix must contain exactly one configured EOT token"
        )
    return eot_token_id

def _build_unified_audio_response_suffix(engine: Any, session: DuplexSession) -> tuple[int, ...]:
    """Build EOA + user turn end + assistant generation header without audio preprocessing."""
    token_ids, marker_position, _boa_token_id, eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=True
    )
    template_suffix = token_ids[marker_position + 1 :]
    _validate_unified_audio_turn_end_suffix(engine, template_suffix)
    return (eoa_token_id, *template_suffix)

async def _prewarm_unified_v1_transition(
    worker: Any,
    *,
    session_id: str,
    generation_id: int,
    cache_position: int,
    transition_ids: Sequence[int],
) -> dict[str, Any]:
    """Prepare and execute the six-token V1 transition on a disposable KV fork."""
    token_ids = tuple(int(value) for value in transition_ids)
    if len(token_ids) != 6:
        raise RuntimeError("V1 native session prewarm requires its exact logical Q6 transition")
    prepare_started_ns = time.perf_counter_ns()
    prepared = await asyncio.wrap_future(worker.prepare_text_token_inputs(
        token_ids,
        protocol=CONTROLLER_GUIDED_SAY_V1,
    ))
    prepare_ms = (time.perf_counter_ns() - prepare_started_ns) / 1_000_000.0
    physical_q_len = max(2, len(token_ids) + (len(token_ids) & 1))
    if (
        int(prepared.get("logical_q_len", -1)) != len(token_ids)
        or int(prepared.get("physical_q_len", -1)) != physical_q_len
        or physical_q_len != 6
    ):
        raise RuntimeError("prewarmed V1 transition has incompatible physical-Q metadata")

    handle = await asyncio.wrap_future(worker.begin_ephemeral_decode(
        session_id,
        expected_generation_id=int(generation_id),
        expected_cache_position=int(cache_position),
    ))
    native_roundtrip_ms = 0.0
    private_cache_position = int(handle.cache_position)
    try:
        if private_cache_position != int(cache_position):
            raise RuntimeError("V1 transition prewarm fork changed the canonical cache boundary")
        native_started_ns = time.perf_counter_ns()
        appended = await asyncio.wrap_future(worker.append_private_transition_tokens(
            handle,
            token_ids,
            expected_cache_position=private_cache_position,
        ))
        native_roundtrip_ms = (time.perf_counter_ns() - native_started_ns) / 1_000_000.0
        private_cache_position = int(appended.cache_position)
        if private_cache_position != int(cache_position) + len(token_ids):
            raise RuntimeError("V1 native Q6 prewarm advanced an unexpected private KV cursor")
    finally:
        await asyncio.wrap_future(worker.end_ephemeral_decode(handle))
    return {
        "logical_q_len": len(token_ids),
        "physical_q_len": physical_q_len,
        "cache_size": int(prepared.get("cache_size", 0)),
        "cache_misses": int(prepared.get("cache_misses", 0)),
        "prepare_ms": prepare_ms,
        "native_roundtrip_ms": native_roundtrip_ms,
        "canonical_cache_position": int(cache_position),
        "private_cache_position": private_cache_position,
    }

def _unified_v1_transition_prewarm_key(output: Any) -> tuple[int, str, int, int]:
    """Identify the exact imported prefix generation that owns a V1 prewarm."""
    generation_id = getattr(output, "generation_id", None)
    state_epoch_id = getattr(output, "state_epoch_id", None)
    semantic_context_revision = getattr(output, "semantic_context_revision", None)
    semantic_prefix_len = getattr(output, "semantic_prefix_len", None)
    if (
        generation_id is None
        or not isinstance(state_epoch_id, str)
        or not state_epoch_id
        or semantic_context_revision is None
        or semantic_prefix_len is None
    ):
        raise RuntimeError("V1 transition prewarm requires the imported prefix identity")
    return (
        int(generation_id),
        state_epoch_id,
        int(semantic_context_revision),
        int(semantic_prefix_len),
    )

def _prewarm_unified_v1_transition_after_import(
    worker: Any,
    *,
    binding: dict[str, Any],
    output: Any,
    session_id: str,
    transition_ids: Sequence[int],
) -> dict[str, Any] | None:
    """Run one native Q6 roundtrip after the real current prefix import.

    Shared-listener result callbacks run on the dispatcher thread. Bridge the
    async owner-worker operations to the session event loop, which is free
    after the submitted tick has completed and before policy dispatch starts.
    """
    key = _unified_v1_transition_prewarm_key(output)
    if binding.get("unified_v1_transition_prewarm_key") == key:
        return None
    event_loop = binding.get("event_loop")
    if not isinstance(event_loop, asyncio.AbstractEventLoop) or event_loop.is_closed():
        raise RuntimeError("V1 transition prewarm has no live session event loop")
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None
    if running_loop is event_loop:
        raise RuntimeError("V1 transition prewarm cannot block its session event loop")

    coroutine = _prewarm_unified_v1_transition(
        worker,
        session_id=session_id,
        generation_id=key[0],
        cache_position=int(output.cache_position),
        transition_ids=transition_ids,
    )
    try:
        scheduled = asyncio.run_coroutine_threadsafe(coroutine, event_loop)
    except BaseException:
        coroutine.close()
        raise
    result = scheduled.result()
    # A failed Q6 operation remains visible to the dispatcher and is retried
    # only after the caller resolves that failure; never mark it as warmed.
    binding["unified_v1_transition_prewarm_key"] = key
    return result

def _build_unified_backchannel_response_suffix(
    engine: Any,
    session: DuplexSession,
) -> tuple[int, ...]:
    """Build a private backchannel suffix without changing the imported prompt."""
    protocol = _require_unified_stable_audio_protocol(
        _configured_audio_response_protocol(engine)
    )
    if protocol == CONTROLLER_GUIDED_SAY_V1:
        # V1 is zero-prefill: action choice comes from the controller, while
        # the already-imported system prompt only requests spoken text. Adding
        # an action-specific instruction here would create a hidden late prompt.
        return _build_unified_audio_response_suffix(engine, session)
    token_ids, marker_position, _boa_token_id, eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=True
    )
    template_suffix = tuple(int(value) for value in token_ids[marker_position + 1 :])
    _validate_unified_audio_turn_end_suffix(engine, template_suffix)
    tokenizer = getattr(engine, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise RuntimeError("private backchannel requires the loaded Thinker tokenizer")
    instruction = (
        " Rispondi soltanto con un brevissimo cenno di ascolto in italiano. "
        "Non aggiungere spiegazioni."
    )
    instruction_ids = encode(instruction, add_special_tokens=False)
    if hasattr(instruction_ids, "tolist"):
        instruction_ids = instruction_ids.tolist()
    if isinstance(instruction_ids, tuple):
        instruction_ids = list(instruction_ids)
    if (
        not isinstance(instruction_ids, (list, tuple))
        or not instruction_ids
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in instruction_ids)
    ):
        raise RuntimeError("loaded Thinker tokenizer returned invalid private backchannel prompt IDs")
    return (int(eoa_token_id), *(int(value) for value in instruction_ids), *template_suffix)

def _build_unified_next_user_header(engine: Any, session: DuplexSession) -> tuple[int, ...]:
    """Build the next user header through BOA, excluding the global BOS token."""
    token_ids, marker_position, boa_token_id, _eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=False
    )
    header = token_ids[:marker_position]
    backend = getattr(getattr(session, "listener", None), "backend", None)
    config = getattr(getattr(engine, "_thinker_wrapper", None), "config", None)
    bos_token_id = getattr(backend, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(config, "bos_token_id", None)
    if header and isinstance(bos_token_id, int) and header[0] == int(bos_token_id):
        header = header[1:]
    if not header:
        raise RuntimeError("unified next-user header template is empty")
    return (*header, boa_token_id)

def _unified_end_of_turn_token_id(engine: Any, session: DuplexSession) -> int:
    """Resolve the single EOT in the active template, allowing trailing whitespace IDs."""
    token_ids, marker_position, _boa_token_id, _eoa_token_id = _unified_audio_template_ids(
        engine, session, add_generation_prompt=False
    )
    suffix = token_ids[marker_position + 1 :]
    return _validate_unified_audio_turn_end_suffix(engine, suffix)
