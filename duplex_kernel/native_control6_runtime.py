"""Runtime wiring for the shared native Control6 semantic prefix."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.native_control6_prompt import (
    build_native_control6_prefix_ids,
    select_native_control6_replay_suffix,
)


def _latest_audio_soft_q4(soft: Any, *, hidden_size: int = 1536) -> Any:
    """Return the current tick's real Q4 tower rows from a fresh output.

    The first listener call after reset can expose the rolling-window audio
    sequence, while steady-state outputs expose only the newest Q4. Replay
    stores one observed tick at a time, so retain the final complete Q4 rows.
    """
    shape = tuple(getattr(soft, "shape", ()))
    if getattr(soft, "ndim", None) != 2 or len(shape) != 2:
        raise RuntimeError(f"Q4 audio soft inatteso per replay semantico: {shape}")
    if int(shape[1]) != hidden_size or int(shape[0]) < 4 or int(shape[0]) % 4:
        raise RuntimeError(f"audio soft non allineato a tick Q4: {shape}")
    return soft[-4:, :]


def _prior_audio_replay_batch(
    observed: Any,
    *,
    current_frame_id: int,
    hidden_size: int = 1536,
) -> tuple[Any | None, tuple[int, ...], str | None]:
    """Build the exact observed Q4 suffix ending before the current frame."""
    if current_frame_id < 0:
        raise RuntimeError("frame corrente senza ID assoluto")
    replay = select_native_control6_replay_suffix(
        observed,
        expected_last_frame_id=current_frame_id - 1,
        max_frames=64,
    )
    if replay is None:
        return None, (), "no_contiguous_observed_history_before_current_frame"
    replay_ids, replay_frames = replay
    if not replay_frames:
        return None, (), None
    normalized = tuple(
        _latest_audio_soft_q4(frame, hidden_size=hidden_size)
        for frame in replay_frames
    )
    batch = torch.stack(normalized, dim=0).reshape(
        1, len(normalized) * 4, hidden_size
    )
    return batch, tuple(int(frame_id) for frame_id in replay_ids), None


@dataclass(frozen=True, slots=True)
class NativeControl6SemanticPrefixRequest:
    """Immutable deadline snapshot for one session's prefix refresh."""

    session_id: str
    session_generation_id: int
    state_epoch_id: str
    listener_state_version: int
    current_frame_id: int
    semantic_context_revision: int
    prefix_token_ids: tuple[int, ...]
    prefix_len: int
    replay_q4: tuple[Any, ...]
    replay_frame_ids: tuple[int, ...]
    fallback_reason: str | None

    @property
    def cache_position(self) -> int:
        return self.prefix_len + 4 * len(self.replay_q4)


def build_native_control6_semantic_prefix_request(
    engine: Any,
    session: Any,
    *,
    current_frame_id: int,
    pending_speak: Any,
) -> NativeControl6SemanticPrefixRequest | None:
    """Capture prompt and strictly prior Q4 history without touching CUDA."""
    if not isinstance(getattr(session, "turn_policy", None), GemmaNativeControl6Policy):
        return None
    if not bool(getattr(engine, "native_control6_semantic_prefix_enabled", True)):
        return None
    state = getattr(session, "listener_state", None)
    backend = getattr(getattr(session, "listener", None), "backend", None)
    if state is None or backend is None or not callable(
        getattr(backend, "build_semantic_prefix_cache", None)
    ):
        raise RuntimeError("native Control6 semantic-prefix cache builder is unavailable")
    if isinstance(current_frame_id, bool) or not isinstance(current_frame_id, int) or current_frame_id < 0:
        raise ValueError("current_frame_id must be a non-negative integer")

    with session._generation_lock:
        revision = int(getattr(session, "semantic_context_revision", 0))
        if state.semantic_context_revision == revision:
            return None
        if (
            bool(getattr(session, "active_response", False))
            or bool(getattr(session, "policy_response_pending", False))
            or pending_speak is not None
        ):
            return None
        if int(state.next_input_frame_id) != current_frame_id:
            raise RuntimeError("semantic prefix request does not match the listener frame watermark")
        instructions = str(getattr(session, "instructions", "") or "")
        completed_messages = tuple(
            dict(message) for message in getattr(session, "history", ())
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        )
        tokenizer = getattr(engine, "tokenizer", None)
        processor = getattr(getattr(engine, "_thinker_wrapper", None), "processor", None)
        if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise RuntimeError("tokenizer Gemma chat-template unavailable for native Control6")
        marker = (
            getattr(processor, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        boa_token_id = getattr(backend, "boa_token_id", None)
        if boa_token_id is None:
            raise RuntimeError("listener backend does not expose the true BOA token ID")
        prefix_ids, prefix_len = build_native_control6_prefix_ids(
            tokenizer,
            instructions=instructions,
            completed_messages=completed_messages,
            audio_marker=str(marker),
            boa_token_id=int(boa_token_id),
        )
        if (
            prefix_len != len(prefix_ids)
            or prefix_ids[-1] != int(boa_token_id)
            or not 1 <= prefix_len <= 291
        ):
            raise RuntimeError("semantic prefix must end in the real BOA within capacity")
        with state._lock:
            observed = tuple(state.semantic_audio_soft_history)
            listener_state_version = int(state.listener_state_version)
            state_epoch_id = str(state.state_epoch_id)
        replay = select_native_control6_replay_suffix(
            observed,
            expected_last_frame_id=current_frame_id - 1,
            max_frames=64,
        )
        if replay is None:
            replay_ids: tuple[int, ...] = ()
            replay_q4: tuple[Any, ...] = ()
            fallback_reason = (
                "no_contiguous_observed_history_before_current_frame"
                if current_frame_id > 0 else None
            )
        else:
            selected_ids, selected_frames = replay
            replay_ids = tuple(int(value) for value in selected_ids)
            replay_q4 = tuple(
                _latest_audio_soft_q4(
                    frame,
                    hidden_size=int(getattr(backend, "hidden_size", 1536)),
                )
                for frame in selected_frames
            )
            fallback_reason = None
        return NativeControl6SemanticPrefixRequest(
            session_id=str(session.session_id),
            session_generation_id=int(getattr(session, "session_generation_id", 0)),
            state_epoch_id=state_epoch_id,
            listener_state_version=listener_state_version,
            current_frame_id=current_frame_id,
            semantic_context_revision=revision,
            prefix_token_ids=tuple(int(value) for value in prefix_ids),
            prefix_len=prefix_len,
            replay_q4=replay_q4,
            replay_frame_ids=replay_ids,
            fallback_reason=fallback_reason,
        )


def prepare_native_control6_semantic_prefix(
    engine: Any,
    session: Any,
    *,
    current_frame_id: int,
    pending_speak: Any,
) -> bool:
    """Prime committed semantic context from audio observed strictly before t."""
    if not isinstance(getattr(session, "turn_policy", None), GemmaNativeControl6Policy):
        return False
    if not bool(getattr(engine, "native_control6_semantic_prefix_enabled", True)):
        return False
    state = getattr(session, "listener_state", None)
    listener = getattr(session, "listener", None)
    backend = getattr(listener, "backend", None)
    if state is None or backend is None or not callable(getattr(backend, "prime_semantic_prefix", None)):
        raise RuntimeError("native Control6 semantic-prefix listener API non disponibile")
    request = build_native_control6_semantic_prefix_request(
        engine,
        session,
        current_frame_id=current_frame_id,
        pending_speak=pending_speak,
    )
    if request is None:
        return False
    replay_audio_soft = (
        torch.stack(request.replay_q4, dim=0).reshape(
            1, len(request.replay_q4) * 4, int(getattr(backend, "hidden_size", 1536))
        )
        if request.replay_q4 else None
    )
    backend.prime_semantic_prefix(
        state,
        request.prefix_token_ids,
        prefix_len=request.prefix_len,
        replay_audio_soft=replay_audio_soft,
        replay_frame_ids=request.replay_frame_ids,
    )
    with state._lock:
        state.semantic_context_revision = request.semantic_context_revision
        state.semantic_replay_fallback_reason = request.fallback_reason
    return True


def record_native_control6_semantic_audio(
    engine: Any,
    session: Any,
    output: Any,
    *,
    expected_session_generation: int | None = None,
) -> None:
    """Record the actual current Q4 after its fresh listener query at frame t."""
    if not isinstance(getattr(session, "turn_policy", None), GemmaNativeControl6Policy):
        return
    if not bool(getattr(engine, "native_control6_semantic_prefix_enabled", True)):
        return
    if not bool(getattr(output, "fresh", False)):
        return
    state = getattr(session, "listener_state", None)
    listener = getattr(session, "listener", None)
    backend = getattr(listener, "backend", None)
    soft = getattr(output, "audio_tower_hidden", None)
    if state is None or soft is None or not callable(getattr(soft, "detach", None)):
        raise RuntimeError("native Control6 richiede la soft feature Q4 fresca corrente")
    soft = _latest_audio_soft_q4(soft, hidden_size=int(getattr(backend, "hidden_size", 1536)))
    frame_id = int(getattr(output, "frame_id", -1))
    if frame_id < 0:
        raise RuntimeError("listener output senza frame ID assoluto per replay semantico")
    owned_soft = soft.detach().clone()
    with session._generation_lock:
        if (
            expected_session_generation is not None
            and int(getattr(session, "session_generation_id", -1)) != expected_session_generation
        ):
            return
        with state._lock:
            history = state.semantic_audio_soft_history
            if history and frame_id <= int(history[-1][0]):
                if frame_id == int(history[-1][0]):
                    history.pop()
                else:
                    raise RuntimeError("frame Q4 corrente retrocede rispetto alla history semantica")
            if history and frame_id != int(history[-1][0]) + 1:
                history.clear()
            history.append((frame_id, owned_soft))


__all__ = [
    "NativeControl6SemanticPrefixRequest",
    "build_native_control6_semantic_prefix_request",
    "prepare_native_control6_semantic_prefix",
    "record_native_control6_semantic_audio",
]
