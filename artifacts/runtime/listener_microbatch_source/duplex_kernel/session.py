# -*- coding: utf-8 -*-
"""
duplex_kernel/session.py — SessionManager (sessioni duplex con KV ownership).

Pattern DwarfStar adattato alla nostra architettura: ogni sessione duplex
mantiene la propria KV cache (Thinker rolling + Talker prefix persistente) e
il proprio stato turnhead. Il kernel orchestra il micro-batching delle
sessioni decode-ready.

Sessione duplex (per utente):
  - ThinkerContext: rolling window degli ultimi K turni.
  - TalkerPrefixKV: KV stabile del prefisso [ICL|h_pfx|BOS], calcolata 1 volta.
  - TurnKVState: KV del turno corrente (troncata a fine turno).
  - TurnHeadRuntime: ascolta u_mel + h_t thinker, governa listen/speak/barge-in.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from duplex_kernel.config import (
    DEFAULT_VOICE,
    FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED,
    READY_FRAME_SCHEDULER_DISPATCH_ENABLED,
    FROZEN_QWEN_PLAYOUT_DELAY_FRAMES,
    USER_SAMPLE_RATE,
)
from duplex_kernel.conditioning_provenance import validate_trace_row
from duplex_kernel.audio_response_conditioning import DEFAULT_MAX_AUDIO_MS
from duplex_kernel.future_prefix import ConditioningFrameBuffer
from duplex_kernel.kv_cache import TalkerPrefixKV, ThinkerContext, TurnKVState
from duplex_kernel.listener import AssistantReferenceBuffer, GemmaListenerState
from duplex_kernel.scheduler import (
    AudioIngressQueue,
    REALTIME_INGRESS_QUEUE_FRAMES,
)
from duplex_kernel.turn_policy import ControlState

logger = logging.getLogger("duplex_kernel.session")

MAX_PENDING_AUDIO_HISTORY_ARCHIVES = 2
MAX_AUDIO_HISTORY_ARCHIVE_SAMPLES = USER_SAMPLE_RATE * 30
POLICY_AUDIO_FRAME_MS = 80
POLICY_AUDIO_FRAME_SAMPLES = USER_SAMPLE_RATE * POLICY_AUDIO_FRAME_MS // 1000
MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES = USER_SAMPLE_RATE * DEFAULT_MAX_AUDIO_MS // 1000
TURN_LOCAL_SPEECH_FIFO_CAPACITY = 4
GEMMA_SPEECH_TICK_MS = 80
GEMMA_SPEECH_TOKENS_PER_TICK = 2


@dataclass(frozen=True)
class SpokenTokenPacket:
    """One visible SAY delta with its exact post-token Gemma state and KV cursor."""

    token_id: int
    text_piece: str
    gemma_hidden_1536: Any = field(repr=False, compare=False)
    kv_position: int

    def __post_init__(self) -> None:
        if isinstance(self.token_id, bool) or int(self.token_id) < 0:
            raise ValueError("Gemma token_id deve essere non negativo")
        if not isinstance(self.text_piece, str):
            raise TypeError("text_piece deve essere string")
        if isinstance(self.kv_position, bool) or int(self.kv_position) < 0:
            raise ValueError("kv_position deve essere non negativo")
        shape = tuple(getattr(self.gemma_hidden_1536, "shape", ()))
        if shape != (1, 1536):
            raise ValueError(f"gemma_hidden_1536 deve avere shape (1,1536), ricevuto {shape}")
        hidden = self.gemma_hidden_1536
        detach = getattr(hidden, "detach", None)
        if callable(detach):
            hidden = detach()
            clone = getattr(hidden, "clone", None)
            if callable(clone):
                hidden = clone()
        object.__setattr__(self, "token_id", int(self.token_id))
        object.__setattr__(self, "kv_position", int(self.kv_position))
        object.__setattr__(self, "gemma_hidden_1536", hidden)


@dataclass(frozen=True)
class SpeechQwenToken:
    """A Qwen text ID and the Gemma packet whose SAY span produced it."""

    token_id: int
    packet: SpokenTokenPacket
    packet_token_index: int
    packet_token_count: int


@dataclass
class _QueuedSpokenPacket:
    packet: SpokenTokenPacket
    qwen_token_ids: tuple[int, ...]
    next_token_index: int = 0


class TurnLocalSpeechFIFO:
    """Bounded, response-owned FIFO; packet heads live until all Qwen IDs drain."""

    def __init__(self, capacity: int = TURN_LOCAL_SPEECH_FIFO_CAPACITY) -> None:
        if isinstance(capacity, bool) or not 1 <= int(capacity) <= TURN_LOCAL_SPEECH_FIFO_CAPACITY:
            raise ValueError(f"capacity deve essere compresa tra 1 e {TURN_LOCAL_SPEECH_FIFO_CAPACITY}")
        self.capacity = int(capacity)
        self._condition = threading.Condition(threading.RLock())
        self._entries: deque[_QueuedSpokenPacket] = deque()
        self._inflight: SpeechQwenToken | None = None
        self._generation_id: int | None = None
        self._closed = False
        self._frozen = False
        self._initial_token_taken = False
        self._last_vocalized_kv_position: int | None = None
        self._vocalized_text_parts: list[str] = []
        self._peak_depth = 0

    def __len__(self) -> int:
        with self._condition:
            return len(self._entries)

    @property
    def last_vocalized_kv_position(self) -> int | None:
        """Last Gemma KV position consumed by Qwen text synthesis, not playback ACK."""
        with self._condition:
            return self._last_vocalized_kv_position

    @property
    def vocalized_text(self) -> str:
        """Gemma packet text whose complete Qwen ID spans were accepted by Q1."""
        with self._condition:
            return "".join(self._vocalized_text_parts)

    @property
    def peak_depth(self) -> int:
        """Maximum queued plus in-flight Gemma packet count this turn."""
        with self._condition:
            return self._peak_depth

    @property
    def generation_id(self) -> int | None:
        with self._condition:
            return self._generation_id

    def begin_turn(self, *, generation_id: int) -> None:
        if isinstance(generation_id, bool) or int(generation_id) < 0:
            raise ValueError("generation_id deve essere non negativo")
        with self._condition:
            self._entries.clear()
            self._inflight = None
            self._generation_id = int(generation_id)
            self._closed = False
            self._frozen = False
            self._initial_token_taken = False
            self._last_vocalized_kv_position = None
            self._vocalized_text_parts.clear()
            self._peak_depth = 0
            self._condition.notify_all()

    def put(
        self,
        packet: SpokenTokenPacket,
        *,
        qwen_token_ids: tuple[int, ...] | list[int],
        generation_id: int,
        stop_check: Callable[[], bool] | None = None,
        timeout_s: float | None = None,
    ) -> None:
        if not isinstance(packet, SpokenTokenPacket):
            raise TypeError("packet deve essere SpokenTokenPacket")
        ids = tuple(qwen_token_ids)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in ids):
            raise ValueError("qwen_token_ids deve contenere ID interi non negativi")
        if packet.text_piece and not ids:
            raise ValueError("SAY visibile senza ID Qwen: gestire esplicitamente il carry, non scartarlo")
        started = time.monotonic()
        while True:
            with self._condition:
                self._raise_generation(generation_id)
                if self._frozen:
                    raise RuntimeError("turn-local speech FIFO snapshot is frozen")
                if self._closed:
                    raise RuntimeError("turn-local speech FIFO closed")
                if len(self._entries) < self.capacity:
                    if not ids:
                        # Empty/structural Gemma deltas have explicit zero-vocal-ID
                        # semantics and never advance the synthesis cursor.
                        return
                    self._entries.append(_QueuedSpokenPacket(packet, ids))
                    self._peak_depth = max(self._peak_depth, len(self._entries))
                    self._condition.notify_all()
                    return
                if timeout_s is None:
                    self._condition.wait(0.02)
                else:
                    remaining = float(timeout_s) - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("turn-local speech FIFO full")
                    self._condition.wait(min(0.02, remaining))
            # stop_check can acquire the session generation lock. Never call it
            # while holding the FIFO condition: invalidate_generation takes
            # those locks in the opposite order to clear this queue.
            if stop_check is not None and stop_check():
                raise RuntimeError("turn-local speech FIFO producer cancelled")

    def _raise_generation(self, generation_id: int) -> None:
        if self._generation_id != int(generation_id):
            raise RuntimeError("turn-local speech FIFO generation changed")

    def take_qwen_token(
        self,
        *,
        generation_id: int,
        initial: bool = False,
        stop_check: Callable[[], bool] | None = None,
        timeout_s: float | None = None,
    ) -> SpeechQwenToken | None:
        started = time.monotonic()
        while True:
            with self._condition:
                self._raise_generation(generation_id)
                if initial and self._initial_token_taken:
                    raise RuntimeError("Qwen initial token was already consumed")
                if self._inflight is not None:
                    if timeout_s is not None and time.monotonic() - started >= timeout_s:
                        raise TimeoutError("turn-local speech FIFO token remains in flight")
                    self._condition.wait(0.02)
                elif self._entries:
                    head = self._entries[0]
                    index = head.next_token_index
                    token = SpeechQwenToken(
                        token_id=head.qwen_token_ids[index],
                        packet=head.packet,
                        packet_token_index=index,
                        packet_token_count=len(head.qwen_token_ids),
                    )
                    self._inflight = token
                    if initial:
                        self._initial_token_taken = True
                    return token
                elif self._closed:
                    return None
                elif timeout_s is not None:
                    remaining = float(timeout_s) - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("turn-local speech FIFO empty")
                    self._condition.wait(min(0.02, remaining))
                else:
                    self._condition.wait(0.02)
            # Avoid lock inversion with session.invalidate_generation().
            if stop_check is not None and stop_check():
                raise RuntimeError("turn-local speech FIFO consumer cancelled")

    def ack_qwen_token(self, *, generation_id: int) -> int | None:
        """Commit one reserved ID only after the Qwen forward accepted it."""
        with self._condition:
            self._raise_generation(generation_id)
            if self._frozen:
                raise RuntimeError("turn-local speech FIFO snapshot is frozen")
            token = self._inflight
            if token is None:
                raise RuntimeError("turn-local speech FIFO has no Qwen token in flight")
            head = self._entries[0]
            if (
                head.packet is not token.packet
                or head.next_token_index != token.packet_token_index
            ):
                raise RuntimeError("turn-local speech FIFO in-flight token lost its head owner")
            head.next_token_index += 1
            cursor = None
            if head.next_token_index == len(head.qwen_token_ids):
                self._entries.popleft()
                self._last_vocalized_kv_position = head.packet.kv_position
                self._vocalized_text_parts.append(head.packet.text_piece)
                cursor = head.packet.kv_position
            self._inflight = None
            self._condition.notify_all()
            return cursor

    def close(self) -> None:
        """Stop producers and let the consumer drain already accepted packets."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def freeze_snapshot(self, *, generation_id: int) -> tuple[int | None, str]:
        """Freeze the synthesis ACK boundary and return its atomic text/cursor."""
        with self._condition:
            self._raise_generation(generation_id)
            self._frozen = True
            self._closed = True
            snapshot = (
                self._last_vocalized_kv_position,
                "".join(self._vocalized_text_parts),
            )
            self._condition.notify_all()
            return snapshot

    def clear(self) -> int | None:
        """Fence the turn, discard queued packets, and return its synthesis cursor."""
        with self._condition:
            cursor = self._last_vocalized_kv_position
            self._entries.clear()
            self._inflight = None
            self._closed = True
            self._frozen = True
            self._condition.notify_all()
            return cursor


class GemmaTokenTickPacer:
    """Hard per-80ms-window token budget with no catch-up burst after delays."""

    def __init__(
        self,
        *,
        tick_ms: int = GEMMA_SPEECH_TICK_MS,
        max_tokens_per_tick: int = GEMMA_SPEECH_TOKENS_PER_TICK,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        if isinstance(tick_ms, bool) or int(tick_ms) <= 0:
            raise ValueError("tick_ms deve essere positivo")
        if isinstance(max_tokens_per_tick, bool) or int(max_tokens_per_tick) < 1:
            raise ValueError("max_tokens_per_tick deve essere positivo")
        self.tick_s = int(tick_ms) / 1000.0
        self.max_tokens_per_tick = int(max_tokens_per_tick)
        self._monotonic = monotonic
        self._sleep = sleep
        self._epoch = float(monotonic())
        self._tick_index = 0
        self._tokens_in_tick = 0
        self._total_acquired = 0
        self._peak_tokens_in_tick = 0

    async def acquire(self, *, stop_check: Callable[[], bool] | None = None) -> None:
        while True:
            now = float(self._monotonic())
            tick = max(0, int((now - self._epoch) // self.tick_s))
            if tick > self._tick_index:
                self._tick_index = tick
                self._tokens_in_tick = 0
            if self._tokens_in_tick < self.max_tokens_per_tick:
                if stop_check is not None and stop_check():
                    raise RuntimeError("Gemma speech token pacing cancelled")
                self._tokens_in_tick += 1
                self._total_acquired += 1
                self._peak_tokens_in_tick = max(
                    self._peak_tokens_in_tick, self._tokens_in_tick
                )
                return
            next_boundary = self._epoch + (self._tick_index + 1) * self.tick_s
            if stop_check is not None and stop_check():
                raise RuntimeError("Gemma speech token pacing cancelled")
            result = self._sleep(max(0.0, next_boundary - now))
            if inspect.isawaitable(result):
                await result

    def snapshot(self) -> dict[str, int]:
        return {
            "tick_ms": int(round(self.tick_s * 1000.0)),
            "max_tokens_per_tick": self.max_tokens_per_tick,
            "total_tokens": self._total_acquired,
            "peak_tokens_in_tick": self._peak_tokens_in_tick,
        }


@dataclass(frozen=True)
class GenerationToken:
    session_generation_id: int
    response_id: str
    speaker_segment_id: str


@dataclass
class SpeakerSegmentState:
    segment_id: str
    generation_id: int
    planned_text: str = ""
    synthesized_text: str = ""
    emitted_text: str = ""
    emitted_audio_frames: int = 0
    control_opened: bool = False
    control_action: str | None = None
    control_close_requested: bool = False
    control_frames_advanced: int = 0
    control_frames_remaining: int | None = None


@dataclass(frozen=True)
class CompletedUtteranceAudioSnapshot:
    """Immutable PCM and exact source fence for one completed user utterance."""

    generation_id: int
    first_frame_id: int
    last_frame_id: int
    sample_count: int
    source_cutoff_sample_id: int
    pcm_bytes: bytes = field(repr=False)


@dataclass
class DuplexSession:
    """Stato di una sessione duplex (KV ownership + turnhead + history)."""

    session_id: str
    created_at: float = field(default_factory=time.time)
    # KV ownership
    thinker_ctx: ThinkerContext = field(default_factory=ThinkerContext)
    talker_prefix: TalkerPrefixKV = field(default_factory=TalkerPrefixKV)
    turn_kv: TurnKVState = field(default_factory=TurnKVState)
    talker_prefix_text: str | None = None
    talker_prefix_h: np.ndarray | None = None
    thinker_kv: Any | None = None
    thinker_cache_ids: list[int] = field(default_factory=list)
    thinker_hidden: Any | None = None
    # Compiled Thinker cache owns only a prompt prefix. Every response branches
    # into a private cache, so cancellation can never mutate these committed KV.
    thinker_static_cache: Any | None = field(default=None, repr=False, compare=False)
    thinker_static_cache_ids: list[int] = field(default_factory=list, repr=False, compare=False)
    thinker_static_cache_fingerprint: Any | None = field(default=None, repr=False, compare=False)
    # Control plane (TurnHead is only an adapter implementation).
    turn_policy: Any | None = None
    _policy_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)
    _policy_revision: int = field(default=0, init=False, repr=False, compare=False)
    control_state: ControlState = ControlState.LISTENING
    # Generation fencing
    session_generation_id: int = 0
    response_id: str | None = None
    speaker_segment_id: str | None = None
    frame_id: int = 0
    response_cancel_notified: bool = False
    speaker_segment: SpeakerSegmentState | None = None
    stale_audio_events: int = 0
    stale_text_events: int = 0
    stale_history_writes: int = 0
    stale_kv_writes: int = 0
    stale_session_mutations: int = 0
    stale_write_attempts: dict[str, int] = field(default_factory=dict)
    # Gemma-to-Qwen speech packets are owned by this response generation.
    speech_fifo: TurnLocalSpeechFIFO = field(default_factory=TurnLocalSpeechFIFO, repr=False, compare=False)
    _speech_fifo_snapshots: dict[int, tuple[int | None, str]] = field(
        default_factory=dict, repr=False, compare=False
    )
    _silence_skip_transition_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    _generation_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _closed: bool = field(default=False, repr=False, compare=False)
    # Stato conversazione
    history: list[dict[str, Any]] = field(default_factory=list)
    instructions: str = ""
    # Revision consumed by the native listener's bounded semantic prefix.
    semantic_context_revision: int = 0
    # Audio ingress indipendente dal ciclo request/response (P2). P3 collega
    # questo flusso al listener: pending_audio resta solo per il commit API.
    audio_ingress: AudioIngressQueue = field(
        default_factory=lambda: AudioIngressQueue(max_frames=REALTIME_INGRESS_QUEUE_FRAMES)
    )
    audio_ingress_capacity_frames: int = REALTIME_INGRESS_QUEUE_FRAMES
    listener: Any | None = field(default=None, repr=False, compare=False)
    listener_state: GemmaListenerState = field(default_factory=GemmaListenerState, repr=False)
    assistant_reference_buffer: AssistantReferenceBuffer = field(
        default_factory=AssistantReferenceBuffer, repr=False, compare=False
    )
    listener_hidden: Any | None = field(default=None, repr=False, compare=False)
    listener_hidden_frame_id: int | None = field(default=None, repr=False, compare=False)
    listener_hidden_at_ns: int | None = field(default=None, repr=False, compare=False)
    listener_audio_queries: Any | None = field(default=None, repr=False, compare=False)
    listener_audio_frame_id: int | None = field(default=None, repr=False, compare=False)
    last_listener_output: Any | None = field(default=None, repr=False, compare=False)
    # Opt-in diagnostic trace: disabled on normal serving to avoid a GPU
    # scalar readback on every 80 ms listener tick.
    conditioning_trace_enabled: bool = False
    conditioning_trace_sample_id: str = "live-session"
    conditioning_trace_sample_id_locked: bool = field(default=False, repr=False, compare=False)
    conditioning_trace_stream_id: str | None = field(default=None, repr=False, compare=False)
    conditioning_trace_source_fingerprint: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    conditioning_trace_runtime_provenance: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    conditioning_trace: list[dict[str, Any]] = field(default_factory=list, repr=False, compare=False)
    v9_state: Any | None = field(default=None, repr=False, compare=False)
    v9_text_anchor: Any | None = field(default=None, repr=False, compare=False)
    v9_state_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    conditioning_buffer: ConditioningFrameBuffer = field(default_factory=ConditioningFrameBuffer, repr=False, compare=False)
    assistant_reference_next_frame_id: int = 0
    last_listener_frame_id: int | None = field(default=None, repr=False)
    # Micro-turn in corso (compatibilità commit/ASR)
    pending_audio: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    _pending_audio_lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False, compare=False)
    pending_mel: np.ndarray = field(default_factory=lambda: np.empty((0, 128), dtype=np.float32))
    last_activity: float = field(default_factory=time.time)
    active_response: bool = False
    stop_requested: bool = False
    # Action that requested the current cooperative/destructive stop.  It is
    # control-plane metadata consumed by the realtime response cleanup.
    stop_reason: str | None = None
    voice: str = DEFAULT_VOICE
    voice_style: str | None = None
    voice_conditioning: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    # Explicit synthetic-only opt-in for bounded local TTS tensor/code capture.
    tts_quality_diagnostic_fixture: str | None = field(default=None, repr=False, compare=False)
    tts_quality_diagnostic_seed_sha256: str | None = field(default=None, repr=False, compare=False)
    ref_audio: np.ndarray | None = None
    ref_text: str | None = None
    language: str = "Italian"
    tools: list[dict[str, Any]] = field(default_factory=list)
    auto_response: bool = False
    # Audio captured by the 80 ms policy clock for action-driven turns.  Frame
    # IDs below the watermark were already committed by the WebSocket API.
    policy_audio_frames: deque[np.ndarray] = field(default_factory=deque, repr=False, compare=False)
    policy_audio_frame_ids: deque[int] = field(default_factory=deque, repr=False, compare=False)
    policy_audio_samples: int = 0
    policy_audio_min_sample_id: int = 0
    policy_audio_source_cutoff_sample_id: int | None = field(default=None, repr=False, compare=False)
    policy_audio_source_generation_id: int | None = field(default=None, repr=False, compare=False)
    _completed_utterance_audio_snapshot: CompletedUtteranceAudioSnapshot | None = field(
        default=None, init=False, repr=False, compare=False
    )
    # Highest policy frame whose retained PCM has been destructively committed.
    # Direct-audio snapshots of an ongoing utterance must start after this ID.
    policy_audio_consumed_through_frame_id: int = -1
    # One observed user onset remains pending until a SPEAK response completes.
    # A newer onset supersedes pending work without being consumed by the old response.
    pending_user_onset_frame_id: int | None = None
    reserved_user_onset_frame_id: int | None = None
    last_answered_onset_frame_id: int | None = None
    assistant_history_committed_generation_id: int | None = field(default=None, repr=False, compare=False)
    next_audio_history_archive_id: int = 1
    pending_audio_history_archive: dict[str, Any] | None = field(default=None, repr=False, compare=False)
    queued_audio_history_archives: deque[dict[str, Any]] = field(default_factory=deque, repr=False, compare=False)
    audio_history_archive_overflow_count: int = 0
    policy_response_pending: bool = False
    # Speaker-only P4D delay.  The scheduler/listener clock is independent.
    speaker_playout_delay_frames: int = FROZEN_QWEN_PLAYOUT_DELAY_FRAMES
    scheduler: Any = field(init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "session_id" and "session_id" in self.__dict__:
            raise AttributeError("session_id è immutabile")
        super().__setattr__(name, value)

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError("session_id non può essere vuoto")
        if (
            isinstance(self.audio_ingress_capacity_frames, bool)
            or not isinstance(self.audio_ingress_capacity_frames, int)
            or self.audio_ingress_capacity_frames <= 0
        ):
            raise ValueError("audio_ingress_capacity_frames deve essere un intero positivo")
        if self.audio_ingress.max_frames != self.audio_ingress_capacity_frames:
            self.audio_ingress = AudioIngressQueue(max_frames=self.audio_ingress_capacity_frames)
        # The scheduler owns consumption of audio_ingress.  Server adapters
        # inject callbacks but never call get()/policy.step() themselves.
        from duplex_kernel.scheduler import DuplexScheduler
        self.scheduler = DuplexScheduler(
            self.audio_ingress,
            anchor_first_input=FIRST_INPUT_SCHEDULER_ANCHOR_ENABLED,
            ready_frame_dispatch=READY_FRAME_SCHEDULER_DISPATCH_ENABLED,
        )

    def touch(self) -> None:
        self.last_activity = time.time()

    def set_turn_policy(self, policy: Any | None) -> None:
        with self._policy_lock:
            self.turn_policy = policy
            self._policy_revision += 1

    def run_turn_policy(self, policy: Any, revision: int, policy_input: Any) -> Any | None:
        """Serialize policy execution without holding the session generation lock."""
        with self._policy_lock:
            if self._closed or self._policy_revision != int(revision) or self.turn_policy is not policy:
                return None
            return policy.step(policy_input)

    def reset_turn_policy(self) -> None:
        with self._policy_lock:
            if self.turn_policy is not None and hasattr(self.turn_policy, "reset"):
                self.turn_policy.reset()
            self._policy_revision += 1

    def observe_user_onset(self, frame_id: int) -> bool:
        """Retain the newest real onset independently of endpoint consumption."""
        onset = int(frame_id)
        if onset < 0:
            raise ValueError("user onset frame id deve essere non negativo")
        with self._generation_lock:
            if self.last_answered_onset_frame_id is not None and onset <= self.last_answered_onset_frame_id:
                return False
            snapshot = self._completed_utterance_audio_snapshot
            if snapshot is not None and onset > snapshot.first_frame_id:
                self._completed_utterance_audio_snapshot = None
            if self.pending_user_onset_frame_id is None or onset > self.pending_user_onset_frame_id:
                previous_onset = self.pending_user_onset_frame_id
                if (
                    previous_onset is not None
                    and self.reserved_user_onset_frame_id != previous_onset
                ):
                    self._finalize_audio_history_archive_absent_locked(previous_onset)
                self.pending_user_onset_frame_id = onset
                return True
            return False

    def _finalize_audio_history_archive_absent_locked(self, onset_frame_id: int) -> None:
        """Finish an unassociated archive when a newer user turn supersedes it."""
        pending = self.pending_audio_history_archive
        candidates = ([pending] if isinstance(pending, dict) else []) + list(
            self.queued_audio_history_archives
        )
        for archive in candidates:
            if (
                int(archive.get("first_frame_id", -1)) == int(onset_frame_id)
                and not bool(archive.get("assistant_finalized"))
            ):
                archive["assistant_text"] = ""
                archive["assistant_status"] = "absent"
                archive["assistant_finalized"] = True
        self._drain_audio_history_archives_locked()

    def reserve_user_onset_for_response(self, frame_id: int) -> bool:
        """Reserve the pending onset at SPEAK launch; never reserve an older turn."""
        onset = int(frame_id)
        with self._generation_lock:
            if self.pending_user_onset_frame_id != onset:
                return False
            if self.last_answered_onset_frame_id is not None and onset <= self.last_answered_onset_frame_id:
                return False
            if self.reserved_user_onset_frame_id not in {None, onset}:
                return False
            self.reserved_user_onset_frame_id = onset
            return True

    def complete_user_onset_response(self, frame_id: int) -> bool:
        """Consume the reserved onset only after response.done is sent."""
        onset = int(frame_id)
        with self._generation_lock:
            if self.reserved_user_onset_frame_id != onset:
                return False
            if self.last_answered_onset_frame_id is None or onset > self.last_answered_onset_frame_id:
                self.last_answered_onset_frame_id = onset
            if self.pending_user_onset_frame_id == onset:
                self.pending_user_onset_frame_id = None
            self.reserved_user_onset_frame_id = None
            return True

    def cancel_user_onset_response(self, frame_id: int) -> bool:
        """Release reservation; leave pending untouched unless a newer onset superseded it."""
        onset = int(frame_id)
        with self._generation_lock:
            if self.reserved_user_onset_frame_id != onset:
                return False
            self.reserved_user_onset_frame_id = None
            if (
                self.pending_user_onset_frame_id is None
                and (self.last_answered_onset_frame_id is None or onset > self.last_answered_onset_frame_id)
            ):
                self.pending_user_onset_frame_id = onset
            return True

    def begin_response(
        self,
        response_id: str,
        *,
        preserve_pending_user_turn_audio: bool = False,
        preserve_newer_pending_user_turn_audio_after_onset_frame_id: int | None = None,
    ) -> GenerationToken:
        with self._silence_skip_transition_lock:
            with self._generation_lock:
                if self._closed:
                    raise RuntimeError("sessione già chiusa")
                self.policy_response_pending = False
                newer_pending_owner = (
                    None
                    if preserve_newer_pending_user_turn_audio_after_onset_frame_id is None
                    else int(preserve_newer_pending_user_turn_audio_after_onset_frame_id)
                )
                pending_onset = self.pending_user_onset_frame_id
                carry_newer_pending_audio = bool(
                    newer_pending_owner is not None
                    and pending_onset is not None
                    and pending_onset > newer_pending_owner
                    and self.reserved_user_onset_frame_id == newer_pending_owner
                    and self._completed_utterance_pin_matches_pending_locked(
                        allowed_reserved_onset_frame_id=newer_pending_owner,
                    )
                )
                carry_completed_audio = bool(
                    (
                        preserve_pending_user_turn_audio
                        and self._completed_utterance_pin_matches_pending_locked()
                    )
                    or carry_newer_pending_audio
                )
                completed_audio = self._completed_utterance_audio_snapshot
                if not carry_completed_audio:
                    self._clear_completed_utterance_audio_locked()
                    self._clear_policy_audio_source_watermark_locked()
                self.session_generation_id += 1
                if carry_completed_audio and completed_audio is not None:
                    # BACKCHANNEL, or a SPEAK owned by an older reserved onset,
                    # advances the generation without consuming this exact
                    # pending utterance. Rebind only its generation fences
                    # atomically; PCM/span/cutoff remain unchanged.
                    self._completed_utterance_audio_snapshot = replace(
                        completed_audio,
                        generation_id=self.session_generation_id,
                    )
                    self.policy_audio_source_generation_id = self.session_generation_id
                self.speech_fifo.begin_turn(generation_id=self.session_generation_id)
                self.response_id = response_id
                self.speaker_segment_id = f"seg_{uuid.uuid4().hex[:12]}"
                self.frame_id = 0
                self.stop_requested = False
                self.stop_reason = None
                self.response_cancel_notified = False
                self.control_state = ControlState.SPEAKING
                self.speaker_segment = SpeakerSegmentState(self.speaker_segment_id, self.session_generation_id)
                # A listener step commits its state before its scheduler callback
                # publishes the hidden. If begin_response fences that callback
                # mid-step, last_listener_frame_id trails the processed listener
                # clock by one. Anchor after the committed clock so the first
                # post-boundary hidden is still relative frame zero.
                processed_listener_through = self.listener_state.next_input_frame_id - 1
                base_frame_id = max(
                    0,
                    processed_listener_through,
                    self.last_listener_frame_id if self.last_listener_frame_id is not None else 0,
                )
                self.conditioning_buffer.begin_segment(
                    generation_id=self.session_generation_id,
                    segment_id=self.speaker_segment_id,
                    base_frame_id=base_frame_id,
                )
                return GenerationToken(self.session_generation_id, response_id, self.speaker_segment_id)

    def next_frame(self) -> int:
        with self._generation_lock:
            self.frame_id += 1
            return self.frame_id

    def mark_listener_frame(self, frame_id: int) -> None:
        """Anchor emitted assistant references to the listener timeline."""
        with self._generation_lock:
            if self._closed:
                return
            frame_id = int(frame_id)
            if self.last_listener_frame_id is None or frame_id > self.last_listener_frame_id:
                self.last_listener_frame_id = frame_id

    def record_conditioning_trace(self, row: dict[str, Any], *, expected_generation_id: int | None = None) -> bool:
        """Validate and append a trace row only for the current generation."""
        validate_trace_row(row)
        with self._generation_lock:
            if self._closed:
                self.stale_session_mutations += 1
                return False
            if row["stream_id"] != self.session_id:
                raise ValueError("conditioning trace stream_id non coincide con la sessione")
            if self.conditioning_trace_sample_id_locked and row["sample_id"] != self.conditioning_trace_sample_id:
                raise ValueError("conditioning trace sample_id mutato dopo il primo frame")
            if not self.conditioning_trace_enabled:
                return False
            self.conditioning_trace_sample_id = row["sample_id"]
            self.conditioning_trace_sample_id_locked = True
            if expected_generation_id is not None and expected_generation_id != self.session_generation_id:
                self.stale_session_mutations += 1
                return False
            self.conditioning_trace.append(dict(row))
            return True

    def reserve_assistant_reference_frames(self, count: int) -> int:
        """Reserve ids starting at the latest listener frame already consumed."""
        if count <= 0:
            raise ValueError("count deve essere positivo")
        with self._generation_lock:
            anchor = self.last_listener_frame_id
            first = self.assistant_reference_next_frame_id
            if anchor is not None:
                first = max(first, anchor)
            self.assistant_reference_next_frame_id = first + int(count)
            return first

    def end_response(self, token: GenerationToken) -> None:
        with self._generation_lock:
            # A stopped worker may be stale by generation, but it must not be
            # allowed to clear a newer response's active flag.
            if self.response_id == token.response_id:
                self.active_response = False
                self.policy_response_pending = False
                self.control_state = ControlState.LISTENING

    def append_policy_audio(
        self,
        frame_id: int,
        pcm: np.ndarray,
    ) -> bool:
        """Retain post-commit mic samples for policy SPEAK (max 30 s)."""
        with self._generation_lock:
            frame = np.asarray(pcm, dtype=np.float32).reshape(-1)
            if not frame.size or not np.isfinite(frame).all():
                raise ValueError("policy PCM deve essere non vuoto e finito")
            sample_start = int(frame_id) * int(frame.size)
            sample_end = sample_start + int(frame.size)
            self.policy_audio_source_cutoff_sample_id = sample_end
            self.policy_audio_source_generation_id = self.session_generation_id
            cutoff = self.policy_audio_min_sample_id
            if sample_end <= cutoff:
                return False
            if sample_start < cutoff:
                frame = frame[min(cutoff - sample_start, frame.size):]
                if not frame.size:
                    return False
            samples = USER_SAMPLE_RATE * 30
            value = np.array(frame[-samples:], dtype=np.float32, copy=True)
            self.policy_audio_frames.append(value)
            self.policy_audio_frame_ids.append(int(frame_id))
            self.policy_audio_samples += int(value.size)
            while self.policy_audio_samples > samples and self.policy_audio_frames:
                excess = self.policy_audio_samples - samples
                oldest = self.policy_audio_frames[0]
                if oldest.size <= excess:
                    self.policy_audio_frames.popleft()
                    self.policy_audio_frame_ids.popleft()
                    self.policy_audio_samples -= int(oldest.size)
                else:
                    self.policy_audio_frames[0] = oldest[excess:].copy()
                    self.policy_audio_samples -= excess
            return True

    def take_policy_audio(self) -> np.ndarray:
        """Atomically commit the consumed frames through the current policy tick."""
        with self._generation_lock:
            if not self.policy_audio_frames:
                return np.empty(0, dtype=np.float32)
            audio = np.concatenate(tuple(self.policy_audio_frames)).astype(np.float32, copy=False)
            self.policy_audio_consumed_through_frame_id = max(
                self.policy_audio_consumed_through_frame_id,
                max(self.policy_audio_frame_ids),
            )
            self.policy_audio_frames.clear()
            self.policy_audio_frame_ids.clear()
            self.policy_audio_samples = 0
            return audio

    def take_policy_audio_through_frame(self, frame_id: int) -> np.ndarray:
        """Commit only policy PCM through an observed causal frame boundary."""
        boundary = int(frame_id)
        with self._generation_lock:
            frames: list[np.ndarray] = []
            retained_frames: deque[np.ndarray] = deque()
            retained_ids: deque[int] = deque()
            retained_samples = 0
            consumed_frame_ids: list[int] = []
            while self.policy_audio_frames:
                audio_frame = self.policy_audio_frames.popleft()
                audio_frame_id = self.policy_audio_frame_ids.popleft()
                if audio_frame_id <= boundary:
                    frames.append(audio_frame)
                    consumed_frame_ids.append(audio_frame_id)
                else:
                    retained_frames.append(audio_frame)
                    retained_ids.append(audio_frame_id)
                    retained_samples += int(audio_frame.size)
            self.policy_audio_frames = retained_frames
            self.policy_audio_frame_ids = retained_ids
            self.policy_audio_samples = retained_samples
            if not frames:
                return np.empty(0, dtype=np.float32)
            self.policy_audio_consumed_through_frame_id = max(
                self.policy_audio_consumed_through_frame_id,
                max(consumed_frame_ids),
            )
            return np.concatenate(frames).astype(np.float32, copy=False)

    def snapshot_policy_audio_range(
        self,
        first_frame_id: int,
        last_frame_id: int,
    ) -> np.ndarray:
        """Copy an inclusive, contiguous policy audio span without consuming it."""
        first = int(first_frame_id)
        last = int(last_frame_id)
        if first < 0 or last < first:
            return np.empty(0, dtype=np.float32)
        with self._generation_lock:
            by_frame: dict[int, np.ndarray] = {}
            for frame_id, audio in zip(self.policy_audio_frame_ids, self.policy_audio_frames):
                key = int(frame_id)
                if first <= key <= last:
                    if key in by_frame:
                        return np.empty(0, dtype=np.float32)
                    by_frame[key] = audio
            if any(frame_id not in by_frame for frame_id in range(first, last + 1)):
                return np.empty(0, dtype=np.float32)
            return np.concatenate(
                [by_frame[frame_id] for frame_id in range(first, last + 1)]
            ).astype(np.float32, copy=True)

    def pin_completed_utterance_audio(
        self,
        first_frame_id: int,
        last_frame_id: int,
        *,
        source_cutoff_sample_id: int,
        generation_id: int | None = None,
    ) -> bool:
        """Pin one exact, complete endpoint span independently of ring eviction."""
        first = int(first_frame_id)
        last = int(last_frame_id)
        cutoff = int(source_cutoff_sample_id)
        with self._generation_lock:
            generation = self.session_generation_id if generation_id is None else int(generation_id)
            expected_samples = (last - first + 1) * POLICY_AUDIO_FRAME_SAMPLES
            existing = self._completed_utterance_audio_snapshot
            if (
                existing is not None
                and generation == self.session_generation_id
                and existing.generation_id == generation
                and existing.first_frame_id == first
                and existing.last_frame_id == last
                and existing.source_cutoff_sample_id == cutoff
                and existing.sample_count == expected_samples
                and expected_samples > 0
                and expected_samples <= MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES
            ):
                return True
            # A failed/new endpoint must never leave a different utterance eligible.
            self._completed_utterance_audio_snapshot = None
            if (
                self._closed
                or generation != self.session_generation_id
                or self.policy_audio_source_generation_id != generation
                or first < 0
                or last < first
                or expected_samples <= 0
                or expected_samples > MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES
                or cutoff != (last + 1) * POLICY_AUDIO_FRAME_SAMPLES
                or self.policy_audio_source_cutoff_sample_id != cutoff
            ):
                return False

            by_frame: dict[int, np.ndarray] = {}
            for frame_id, audio in zip(self.policy_audio_frame_ids, self.policy_audio_frames):
                frame_id = int(frame_id)
                if first <= frame_id <= last:
                    if frame_id in by_frame:
                        return False
                    value = np.asarray(audio, dtype=np.float32).reshape(-1)
                    if value.size != POLICY_AUDIO_FRAME_SAMPLES or not np.isfinite(value).all():
                        return False
                    by_frame[frame_id] = value
            if any(frame_id not in by_frame for frame_id in range(first, last + 1)):
                return False
            pcm = np.concatenate([by_frame[frame_id] for frame_id in range(first, last + 1)])
            if pcm.size != expected_samples or pcm.size > MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES:
                return False
            self._completed_utterance_audio_snapshot = CompletedUtteranceAudioSnapshot(
                generation_id=generation,
                first_frame_id=first,
                last_frame_id=last,
                sample_count=int(pcm.size),
                source_cutoff_sample_id=cutoff,
                pcm_bytes=pcm.astype(np.float32, copy=False).tobytes(order="C"),
            )
            return True

    def snapshot_completed_utterance_audio(
        self,
        first_frame_id: int,
        last_frame_id: int,
        *,
        source_cutoff_sample_id: int,
        generation_id: int | None = None,
    ) -> np.ndarray | None:
        """Copy the pinned PCM only for its exact utterance and source fence."""
        first = int(first_frame_id)
        last = int(last_frame_id)
        cutoff = int(source_cutoff_sample_id)
        with self._generation_lock:
            generation = self.session_generation_id if generation_id is None else int(generation_id)
            snapshot = self._completed_utterance_audio_snapshot
            if (
                self._closed
                or snapshot is None
                or generation != self.session_generation_id
                or snapshot.generation_id != generation
                or snapshot.first_frame_id != first
                or snapshot.last_frame_id != last
                or snapshot.source_cutoff_sample_id != cutoff
                or cutoff != (last + 1) * POLICY_AUDIO_FRAME_SAMPLES
                or snapshot.sample_count != (last - first + 1) * POLICY_AUDIO_FRAME_SAMPLES
                or snapshot.sample_count > MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES
            ):
                return None
            return np.frombuffer(snapshot.pcm_bytes, dtype=np.float32).copy()

    def release_completed_utterance_audio(
        self,
        first_frame_id: int | None = None,
        last_frame_id: int | None = None,
        *,
        source_cutoff_sample_id: int | None = None,
        generation_id: int | None = None,
    ) -> bool:
        """Release the pin, optionally requiring an exact endpoint identity."""
        with self._generation_lock:
            snapshot = self._completed_utterance_audio_snapshot
            if snapshot is None:
                return False
            if (
                (first_frame_id is not None and snapshot.first_frame_id != int(first_frame_id))
                or (last_frame_id is not None and snapshot.last_frame_id != int(last_frame_id))
                or (
                    source_cutoff_sample_id is not None
                    and snapshot.source_cutoff_sample_id != int(source_cutoff_sample_id)
                )
                or (generation_id is not None and snapshot.generation_id != int(generation_id))
            ):
                return False
            self._completed_utterance_audio_snapshot = None
            return True

    def _clear_completed_utterance_audio_locked(self) -> None:
        self._completed_utterance_audio_snapshot = None

    def _clear_policy_audio_source_watermark_locked(self) -> None:
        self.policy_audio_source_cutoff_sample_id = None
        self.policy_audio_source_generation_id = None

    def reserve_audio_history_archive(
        self,
        *,
        pcm: np.ndarray | None,
        first_frame_id: int,
        last_frame_id: int,
        base_messages: list[dict[str, Any]],
        source: str = "post_response_asr",
        response_created_at_monotonic_ns: int | None = None,
        coverage: str = "complete",
    ) -> int | None:
        """Queue bounded actual user audio for ordered asynchronous archival."""
        normalized_source = str(source).strip()
        if normalized_source not in {"post_response_asr", "heard_action_say_v2"}:
            raise ValueError("sorgente archivio audio non supportata")
        if pcm is None:
            if normalized_source != "heard_action_say_v2":
                raise ValueError("solo heard_action_say_v2 può archiviare senza PCM")
            audio = None
        else:
            audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
        first = int(first_frame_id)
        last = int(last_frame_id)
        normalized_coverage = str(coverage).strip().lower()
        if normalized_coverage not in {"partial", "complete"}:
            raise ValueError("copertura archivio deve essere partial o complete")
        if (
            (audio is not None and (not audio.size or audio.size > MAX_AUDIO_HISTORY_ARCHIVE_SAMPLES or not np.isfinite(audio).all()))
            or first < 0
            or last < first
        ):
            raise ValueError("audio archive richiede uno span frame completo e PCM finito entro 30 s")
        with self._generation_lock:
            if self._closed:
                raise RuntimeError("sessione chiusa: non si può archiviare audio")
            existing = self.pending_audio_history_archive
            if isinstance(existing, dict) and int(existing.get("first_frame_id", -1)) == first:
                return int(existing["archive_id"])
            for existing in self.queued_audio_history_archives:
                if int(existing.get("first_frame_id", -1)) == first:
                    return int(existing["archive_id"])
            pending_count = int(self.pending_audio_history_archive is not None) + len(self.queued_audio_history_archives)
            if pending_count >= MAX_PENDING_AUDIO_HISTORY_ARCHIVES:
                self.audio_history_archive_overflow_count += 1
                return None
            archive_id = self.next_audio_history_archive_id
            self.next_audio_history_archive_id += 1
            reserved_at_ns = time.monotonic_ns()
            pending = {
                "archive_id": archive_id,
                "first_frame_id": first,
                "last_frame_id": last,
                "coverage": normalized_coverage,
                "pcm": audio.copy() if audio is not None else None,
                "base_messages": [dict(message) for message in base_messages[-16:]],
                "source": normalized_source,
                "created_at_monotonic_ns": reserved_at_ns,
                "reserved_at_monotonic_ns": reserved_at_ns,
                "response_created_at_monotonic_ns": (
                    None if response_created_at_monotonic_ns is None
                    else int(response_created_at_monotonic_ns)
                ),
                "submit_started_at_monotonic_ns": None,
                "submitted_at_monotonic_ns": None,
                "completed_at_monotonic_ns": None,
                "response_done_at_monotonic_ns": None,
                "sample_rate": USER_SAMPLE_RATE,
                "transcript": None,
                "assistant_text": "",
                "assistant_status": "pending",
                "assistant_finalized": False,
                "status": "pending",
            }
            if self.pending_audio_history_archive is None:
                self.pending_audio_history_archive = pending
            else:
                self.queued_audio_history_archives.append(pending)
            return archive_id

    def mark_audio_history_archive_submitted(
        self,
        archive_id: int,
        *,
        submit_started_at_monotonic_ns: int,
        submitted_at_monotonic_ns: int,
    ) -> bool:
        """Record CPU-worker submission without coupling it to response generation."""
        with self._generation_lock:
            pending = self._find_audio_history_archive_locked(archive_id)
            if self._closed or pending is None:
                self.stale_history_writes += 1
                return False
            started = int(submit_started_at_monotonic_ns)
            submitted = int(submitted_at_monotonic_ns)
            if submitted < started:
                raise ValueError("archive submit timestamp precedes submit start")
            pending["submit_started_at_monotonic_ns"] = started
            pending["submitted_at_monotonic_ns"] = submitted
            return True

    def get_audio_history_archive(self, archive_id: int) -> dict[str, Any] | None:
        """Return a detached snapshot for the CPU ASR worker, if still pending."""
        with self._generation_lock:
            pending = self._find_audio_history_archive_locked(archive_id)
            if pending is None:
                return None
            result = dict(pending)
            result["pcm"] = pending["pcm"].copy() if pending.get("pcm") is not None else None
            result["base_messages"] = [dict(message) for message in pending.get("base_messages", [])]
            return result

    def _find_audio_history_archive_locked(self, archive_id: int) -> dict[str, Any] | None:
        target = int(archive_id)
        pending = self.pending_audio_history_archive
        if isinstance(pending, dict) and int(pending.get("archive_id", -1)) == target:
            return pending
        for item in self.queued_audio_history_archives:
            if int(item.get("archive_id", -1)) == target:
                return item
        return None

    def _advance_audio_history_archive_locked(self) -> None:
        self.pending_audio_history_archive = (
            self.queued_audio_history_archives.popleft()
            if self.queued_audio_history_archives else None
        )

    def _drain_audio_history_archives_locked(self) -> None:
        while isinstance(self.pending_audio_history_archive, dict):
            pending = self.pending_audio_history_archive
            status = str(pending.get("status", "pending"))
            transcript = str(pending.get("transcript") or "").strip()
            assistant_finalized = bool(pending.get("assistant_finalized"))
            if status in {"failed", "empty"}:
                # Empty user ASR cannot discard completed assistant text.
                if not assistant_finalized:
                    break
                self._commit_audio_history_archive_locked(pending)
                self._advance_audio_history_archive_locked()
                continue
            if transcript and assistant_finalized:
                self._commit_audio_history_archive_locked(pending)
                self._advance_audio_history_archive_locked()
                continue
            break

    def complete_audio_history_archive(self, archive_id: int, transcript: str) -> bool:
        """Commit only real non-empty transcripts, preserving archive order."""
        value = str(transcript).strip()
        with self._generation_lock:
            pending = self._find_audio_history_archive_locked(archive_id)
            if self._closed or pending is None:
                self.stale_history_writes += 1
                return False
            pending["completed_at_monotonic_ns"] = time.monotonic_ns()
            if not value:
                pending["status"] = "empty"
                self._drain_audio_history_archives_locked()
                return True
            pending["transcript"] = value
            pending["status"] = "transcript_ready"
            self._drain_audio_history_archives_locked()
            return True

    def update_audio_history_archive_assistant(
        self,
        archive_id: int,
        text: str,
        *,
        status: str = "completed",
        response_done_at_monotonic_ns: int | None = None,
    ) -> bool:
        """Attach the actual completed assistant turn to its archive slot."""
        normalized_status = str(status).strip().lower()
        if normalized_status not in {"completed", "partial", "cancelled", "absent"}:
            raise ValueError("stato assistant archive non supportato")
        with self._generation_lock:
            pending = self._find_audio_history_archive_locked(archive_id)
            if self._closed or pending is None:
                self.stale_history_writes += 1
                return False
            assistant_text = (
                str(text).strip()
                if normalized_status in {"completed", "partial"}
                else ""
            )
            pending["assistant_text"] = assistant_text
            pending["assistant_status"] = normalized_status
            pending["assistant_finalized"] = True
            if response_done_at_monotonic_ns is not None:
                pending["response_done_at_monotonic_ns"] = int(response_done_at_monotonic_ns)
            self._drain_audio_history_archives_locked()
            return True

    def fail_audio_history_archive(self, archive_id: int, reason: str) -> bool:
        """Skip failed archive work so a later valid turn can still commit."""
        with self._generation_lock:
            pending = self._find_audio_history_archive_locked(archive_id)
            if self._closed or pending is None:
                self.stale_history_writes += 1
                return False
            pending["status"] = "failed"
            pending["failure_reason"] = str(reason)[:240]
            pending["completed_at_monotonic_ns"] = time.monotonic_ns()
            self._drain_audio_history_archives_locked()
            return True

    def _commit_audio_history_archive_locked(self, pending: dict[str, Any]) -> None:
        snapshot = [dict(message) for message in pending.get("base_messages", [])]
        # A prior queued archive may have committed since this snapshot was
        # reserved. Current history wins so turns cannot overwrite each other.
        base_messages = [dict(message) for message in self.history] if self.history else snapshot
        transcript = str(pending.get("transcript") or "").strip()
        if transcript:
            base_messages.append({"role": "user", "content": transcript})
        assistant_text = (
            str(pending.get("assistant_text", "")).strip()
            if pending.get("assistant_status", "completed") in {"completed", "partial"}
            else ""
        )
        if assistant_text:
            base_messages.append({"role": "assistant", "content": assistant_text})
        updated_history = base_messages[-16:]
        if updated_history != self.history:
            self.history = updated_history
            self.semantic_context_revision += 1

    def update_instructions(self, instructions: str) -> bool:
        """Update session instructions and invalidate semantic listener context."""
        value = str(instructions)
        with self._generation_lock:
            if value == self.instructions:
                return False
            self.instructions = value
            self.semantic_context_revision += 1
            return True

    def note_semantic_context_changed(self) -> None:
        """Mark an already-locked direct history mutation for listener refresh."""
        with self._generation_lock:
            self.semantic_context_revision += 1

    def reset_policy_audio(self, *, committed_sample_count: int) -> None:
        """Discard committed frames and trim an ingress frame straddling commit."""
        with self._generation_lock:
            self._clear_completed_utterance_audio_locked()
            self._clear_policy_audio_source_watermark_locked()
            if self.policy_audio_frame_ids:
                self.policy_audio_consumed_through_frame_id = max(
                    self.policy_audio_consumed_through_frame_id,
                    max(self.policy_audio_frame_ids),
                )
            self.policy_audio_frames.clear()
            self.policy_audio_frame_ids.clear()
            self.policy_audio_samples = 0
            self.policy_audio_min_sample_id = max(
                self.policy_audio_min_sample_id,
                int(committed_sample_count),
            )

    def open_control_segment(self, action: str) -> None:
        """Materialize a SPEAK/BACKCHANNEL control action at the segment boundary."""
        with self._generation_lock:
            if self.speaker_segment is None or not self.active_response:
                segment_id = f"seg_{uuid.uuid4().hex[:12]}"
                self.speaker_segment_id = segment_id
                self.speaker_segment = SpeakerSegmentState(
                    segment_id,
                    self.session_generation_id,
                )
            self.speaker_segment.control_opened = True
            self.speaker_segment.control_action = action
            self.speaker_segment.control_frames_remaining = 4 if action == "BACKCHANNEL" else None

    def advance_control_segment(self) -> bool:
        """Account exactly one CONTINUE frame on the current speaker segment."""
        with self._generation_lock:
            if self.speaker_segment is None or self.control_state not in {
                ControlState.SPEAKING,
                ControlState.BACKCHANNELING,
            }:
                return False
            self.speaker_segment.control_frames_advanced += 1
            if self.speaker_segment.control_frames_remaining is not None:
                self.speaker_segment.control_frames_remaining -= 1
                if self.speaker_segment.control_frames_remaining <= 0:
                    self.speaker_segment.control_close_requested = True
            return True

    def request_control_segment_close(self) -> None:
        """Record a close request without invalidating cooperative YIELD PCM."""
        with self._generation_lock:
            if self.speaker_segment is not None:
                self.speaker_segment.control_close_requested = True

    def _segment_current(self, token: GenerationToken) -> bool:
        return self.is_generation_current(token) and self.speaker_segment is not None

    def record_planned_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.planned_text = text
            return True

    def record_synthesized_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.synthesized_text += text
            return True

    def record_emitted_text(self, token: GenerationToken, text: str) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["text"] = self.stale_write_attempts.get("text", 0) + 1
                return False
            self.speaker_segment.emitted_text += text
            return True

    def record_emitted_audio(self, token: GenerationToken, frames: int) -> bool:
        with self._generation_lock:
            if not self._segment_current(token):
                self.stale_write_attempts["audio"] = self.stale_write_attempts.get("audio", 0) + 1
                return False
            self.speaker_segment.emitted_audio_frames += int(frames)
            return True

    def invalidate_generation(self, reason: str = "") -> int:
        with self._generation_lock:
            if self._closed:
                return self.session_generation_id
            old_generation_id = int(self.session_generation_id)
            if self.speech_fifo.generation_id == old_generation_id:
                self._speech_fifo_snapshots[old_generation_id] = self.speech_fifo.freeze_snapshot(
                    generation_id=old_generation_id
                )
            while len(self._speech_fifo_snapshots) > 4:
                self._speech_fifo_snapshots.pop(next(iter(self._speech_fifo_snapshots)))
            self._clear_completed_utterance_audio_locked()
            self._clear_policy_audio_source_watermark_locked()
            self.session_generation_id += 1
            self.speech_fifo.clear()
            self.stop_requested = True
            self.policy_response_pending = False
            self.conditioning_buffer.invalidate()
            self.listener_hidden = None
            self.listener_hidden_frame_id = None
            self.listener_hidden_at_ns = None
            self.listener_audio_queries = None
            self.listener_audio_frame_id = None
            self.last_listener_output = None
            # V9 state is turn-owned.  A destructive control action must not
            # let the next segment continue A's causal stream or anchor.
            with self.v9_state_lock:
                self.v9_state = None
                self.v9_text_anchor = None
            logger.debug("[session] generation invalidata %s: %s", self.session_id, reason)
            return self.session_generation_id

    def freeze_speech_snapshot(self, *, generation_id: int) -> tuple[int | None, str]:
        """Return the final ACKed speech prefix for one generation exactly once."""
        generation_id = int(generation_id)
        with self._generation_lock:
            snapshot = self._speech_fifo_snapshots.get(generation_id)
            if snapshot is None:
                snapshot = self.speech_fifo.freeze_snapshot(generation_id=generation_id)
                self._speech_fifo_snapshots[generation_id] = snapshot
                while len(self._speech_fifo_snapshots) > 4:
                    oldest = next(iter(self._speech_fifo_snapshots))
                    if oldest == generation_id:
                        break
                    self._speech_fifo_snapshots.pop(oldest)
            return snapshot

    def is_generation_current(self, token: GenerationToken | None = None, *, generation_id: int | None = None,
                              response_id: str | None = None, speaker_segment_id: str | None = None) -> bool:
        with self._generation_lock:
            if token is not None:
                generation_id = token.session_generation_id
                response_id = token.response_id
                speaker_segment_id = token.speaker_segment_id
            return (
                not self._closed
                and generation_id == self.session_generation_id
                and (response_id is None or response_id == self.response_id)
                and (speaker_segment_id is None or speaker_segment_id == self.speaker_segment_id)
            )

    def guarded_write(self, token: GenerationToken, kind: str, writer: Any) -> bool:
        """Esegue una mutazione solo se il generation token è ancora attivo."""
        with self._generation_lock:
            if not self.is_generation_current(token):
                self.stale_write_attempts[kind] = self.stale_write_attempts.get(kind, 0) + 1
                return False
            writer()
            return True

    def commit_history(self, token: GenerationToken, base_messages: list[dict[str, Any]], text: str) -> bool:
        with self._generation_lock:
            if not self.is_generation_current(token):
                self.stale_write_attempts["history"] = self.stale_write_attempts.get("history", 0) + 1
                return False
            committed_text = self.speaker_segment.emitted_text if self.speaker_segment is not None else text
            if not committed_text:
                return False
            updated_history = [dict(m) for m in base_messages[-16:]] + [{"role": "assistant", "content": committed_text}]
            if updated_history != self.history:
                self.history = updated_history
                self.semantic_context_revision += 1
            return True

    def commit_completed_assistant_response(
        self,
        token: GenerationToken,
        response_id: str,
        text: str,
        *,
        archive_id: int | None = None,
        user_onset_frame_id: int | None = None,
        response_done_at_monotonic_ns: int | None = None,
    ) -> bool:
        """Commit completed assistant text once, preserving any user-ASR slot."""
        assistant_text = str(text).strip()
        if not assistant_text:
            return False
        with self._generation_lock:
            if not self.is_generation_current(token) or str(response_id) != token.response_id:
                self.stale_write_attempts["history"] = self.stale_write_attempts.get("history", 0) + 1
                return False
            if self.assistant_history_committed_generation_id == token.session_generation_id:
                return True

            pending = (
                self._find_audio_history_archive_locked(archive_id)
                if archive_id is not None else None
            )
            if pending is None and user_onset_frame_id is not None:
                onset = int(user_onset_frame_id)
                candidates = ([self.pending_audio_history_archive]
                              if isinstance(self.pending_audio_history_archive, dict) else [])
                candidates += list(self.queued_audio_history_archives)
                pending = next((
                    item for item in candidates
                    if int(item.get("first_frame_id", -1)) == onset
                ), None)
            if pending is not None:
                pending["assistant_text"] = assistant_text
                pending["assistant_status"] = "completed"
                pending["assistant_finalized"] = True
                if response_done_at_monotonic_ns is not None:
                    pending["response_done_at_monotonic_ns"] = int(response_done_at_monotonic_ns)
                self.assistant_history_committed_generation_id = token.session_generation_id
                self._drain_audio_history_archives_locked()
                return True

            # ASR may be disabled, empty, or over its bounded queue. Append
            # against current history so pending user archives are preserved.
            messages = [dict(message) for message in self.history]
            messages.append({"role": "assistant", "content": assistant_text})
            updated_history = messages[-16:]
            if updated_history != self.history:
                self.history = updated_history
                self.semantic_context_revision += 1
            self.assistant_history_committed_generation_id = token.session_generation_id
            return True

    def close(self) -> None:
        """Fence all outstanding workers before releasing session-owned state."""
        with self._generation_lock:
            if self._closed:
                return
            self._closed = True
            self._clear_completed_utterance_audio_locked()
            self._clear_policy_audio_source_watermark_locked()
            self.session_generation_id += 1
            self.speech_fifo.clear()
            self.stop_requested = True
            self.policy_response_pending = False
            self.policy_audio_frames.clear()
            self.policy_audio_frame_ids.clear()
            self.policy_audio_samples = 0
            self.pending_audio_history_archive = None
            self.queued_audio_history_archives.clear()
            self.response_id = None
            with self._pending_audio_lock:
                self.pending_audio = np.empty(0, dtype=np.float32)
            self.conditioning_buffer.invalidate()
            self.listener_hidden = None
            self.listener_hidden_frame_id = None
            self.listener_hidden_at_ns = None
            self.listener_audio_queries = None
            self.listener_audio_frame_id = None
            self.last_listener_output = None
            self.thinker_static_cache = None
            self.thinker_static_cache_ids.clear()
            self.thinker_static_cache_fingerprint = None
            with self._policy_lock:
                if self.turn_policy is not None and hasattr(self.turn_policy, "reset"):
                    self.turn_policy.reset()
                self._policy_revision += 1
            with self.v9_state_lock:
                self.v9_state = None
                self.v9_text_anchor = None

    def reset_turn(self) -> None:
        """Clear turn audio evidence and response-local buffers after a consumed turn."""
        with self._generation_lock:
            self._clear_completed_utterance_audio_locked()
            self._clear_policy_audio_source_watermark_locked()
            self._reset_response_local_state_locked()

    def reset_backchannel_response(self) -> None:
        """Clean response-local state while keeping a valid pending user-turn pin."""
        with self._generation_lock:
            if not self._completed_utterance_pin_matches_pending_locked():
                self._clear_completed_utterance_audio_locked()
                self._clear_policy_audio_source_watermark_locked()
            self._reset_response_local_state_locked()

    def reset_speak_response(
        self,
        *,
        consumed_user_onset_frame_id: int | None = None,
    ) -> bool:
        """Clean SPEAK state while retaining only a valid, unconsumed newer pin."""
        with self._generation_lock:
            pending = self.pending_user_onset_frame_id
            valid_pending_pin = self._completed_utterance_pin_matches_pending_locked(
                allowed_reserved_onset_frame_id=consumed_user_onset_frame_id,
            )
            preserve_newer_pending_pin = bool(
                valid_pending_pin
                and (
                    consumed_user_onset_frame_id is None
                    or (
                        pending is not None
                        and pending > int(consumed_user_onset_frame_id)
                    )
                )
            )
            if not preserve_newer_pending_pin:
                self._clear_completed_utterance_audio_locked()
                self._clear_policy_audio_source_watermark_locked()
            self._reset_response_local_state_locked()
            return preserve_newer_pending_pin

    def _completed_utterance_pin_matches_pending_locked(
        self,
        *,
        allowed_reserved_onset_frame_id: int | None = None,
    ) -> bool:
        """Validate that the pin belongs to the exact unconsumed onset in this generation."""
        snapshot = self._completed_utterance_audio_snapshot
        pending = self.pending_user_onset_frame_id
        source_cutoff = self.policy_audio_source_cutoff_sample_id
        source_generation = self.policy_audio_source_generation_id
        return bool(
            not self._closed
            and snapshot is not None
            and pending is not None
            and snapshot.first_frame_id == pending
            and snapshot.generation_id == self.session_generation_id
            and source_generation == self.session_generation_id
            and source_cutoff is not None
            and source_cutoff >= snapshot.source_cutoff_sample_id
            and snapshot.last_frame_id >= snapshot.first_frame_id
            and snapshot.source_cutoff_sample_id
            == (snapshot.last_frame_id + 1) * POLICY_AUDIO_FRAME_SAMPLES
            and snapshot.sample_count
            == (snapshot.last_frame_id - snapshot.first_frame_id + 1)
            * POLICY_AUDIO_FRAME_SAMPLES
            and snapshot.sample_count > 0
            and snapshot.sample_count <= MAX_COMPLETED_UTTERANCE_AUDIO_SAMPLES
            and (
                self.last_answered_onset_frame_id is None
                or pending > self.last_answered_onset_frame_id
            )
            and self.reserved_user_onset_frame_id
            in {None, allowed_reserved_onset_frame_id}
        )

    def _reset_response_local_state_locked(self) -> None:
        """Release renderer/thinker state without deciding audio-turn ownership."""
        with self._pending_audio_lock:
            self.pending_audio = np.empty(0, dtype=np.float32)
        self.pending_mel = np.empty((0, 128), dtype=np.float32)
        self.turn_kv.clear()
        self.conditioning_buffer.invalidate()
        # V9 state is response-owned. Keep the listener clock/history alive,
        # but create a fresh state and first-raw-frame anchor next time.
        with self.v9_state_lock:
            self.v9_state = None
            self.v9_text_anchor = None
        self.control_state = ControlState.LISTENING
        self.stop_reason = None
        with self._policy_lock:
            if self.turn_policy is not None and hasattr(self.turn_policy, "reset"):
                self.turn_policy.reset()
            self._policy_revision += 1


class SessionManager:
    """Gestore delle sessioni duplex (multi-sessione, micro-batching)."""

    def __init__(
        self,
        *,
        max_sessions: int | None = None,
        session_idle_timeout_s: float = 600.0,
        session_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if max_sessions is not None:
            if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
                raise TypeError("max_sessions must be a positive integer or None")
            if max_sessions <= 0:
                raise ValueError("max_sessions must be positive when configured")
        self.sessions: dict[str, DuplexSession] = {}
        self.max_sessions = max_sessions
        self.idle_timeout_s = session_idle_timeout_s
        self._session_id_factory = session_id_factory or (lambda: uuid.uuid4().hex)
        self._lock = threading.Lock()

    def create_session(
        self,
        *,
        instructions: str = "",
        voice: str = DEFAULT_VOICE,
        language: str = "Italian",
        audio_ingress_capacity_frames: int = REALTIME_INGRESS_QUEUE_FRAMES,
    ) -> DuplexSession:
        with self._lock:
            self._reap_idle()
            if self.max_sessions is not None and len(self.sessions) >= self.max_sessions:
                raise RuntimeError(f"Limite sessioni raggiunto ({self.max_sessions})")
            session_id = str(self._session_id_factory())
            if session_id in self.sessions:
                raise RuntimeError(f"session_id duplicato: {session_id}")
            sess = DuplexSession(
                session_id=session_id,
                instructions=instructions,
                voice=voice,
                language=language,
                audio_ingress_capacity_frames=audio_ingress_capacity_frames,
            )
            self.sessions[sess.session_id] = sess
            logger.info("[session] creata %s (totale %d)", sess.session_id, len(self.sessions))
            return sess

    def get(self, session_id: str) -> DuplexSession | None:
        with self._lock:
            sess = self.sessions.get(session_id)
            if sess is not None:
                sess.touch()
            return sess

    @staticmethod
    def _release_session_resources(sess: DuplexSession) -> None:
        sess.turn_kv.clear()
        sess.talker_prefix.clear()
        sess.listener_state.close()
        sess.assistant_reference_buffer.clear()
        sess.conditioning_buffer.close()
        sess.audio_ingress.close()
        with sess.v9_state_lock:
            sess.v9_state = None
            sess.v9_text_anchor = None

    def _dispose_session(self, sess: DuplexSession) -> None:
        sess.close()
        sess.scheduler.stop(timeout=0.1)
        if not sess.scheduler.is_running:
            self._release_session_resources(sess)
            return
        logger.warning(
            "[session] scheduler callback still active for %s; deferring resource cleanup",
            sess.session_id,
        )
        threading.Thread(
            target=self._finish_deferred_cleanup,
            args=(sess,),
            name=f"duplex-cleanup-{sess.session_id[:8]}",
            daemon=True,
        ).start()

    def _finish_deferred_cleanup(self, sess: DuplexSession) -> None:
        while sess.scheduler.is_running:
            sess.scheduler.stop(timeout=1.0)
        self._release_session_resources(sess)

    def close(self, session_id: str) -> None:
        with self._lock:
            sess = self.sessions.pop(session_id, None)
        if sess is not None:
            self._dispose_session(sess)
            logger.info("[session] chiusa %s", session_id)

    def _reap_idle(self) -> None:
        # Chiamato mentre _lock è acquisito: non richiamare close(), che
        # tenterebbe di acquisire lo stesso Lock e bloccherebbe il server.
        now = time.time()
        expired = [sid for sid, sess in self.sessions.items()
                   if now - sess.last_activity > self.idle_timeout_s]
        for sid in expired:
            sess = self.sessions.pop(sid)
            self._dispose_session(sess)
            logger.info("[session] reap idle %s", sid)

    def decode_ready(self) -> list[DuplexSession]:
        """Sessioni con audio pendente o risposta attiva (per il micro-batching)."""
        with self._lock:
            return [s for s in self.sessions.values() if s.pending_audio.size or s.active_response]

    def all(self) -> list[DuplexSession]:
        with self._lock:
            return list(self.sessions.values())


__all__ = ["DuplexSession", "GenerationToken", "SessionManager", "SpeakerSegmentState"]
