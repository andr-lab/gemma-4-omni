"""Bounded per-response mutable state for concurrent Talker synthesis.

Frozen model weights stay shared. CUDA-graph scratch and all response-local
resources are owned by one lease until generation and synchronous callbacks
have returned; cancellation only marks a lease and never frees it early.
"""

from __future__ import annotations

import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from typing import Any

from duplex_kernel.audio_stream import DeviceRVQCodeRing


@dataclass
class _TalkContextSlot:
    slot_id: int
    subtalker_graph: Any
    code2wav_graph: Any = None
    rvq_history: DeviceRVQCodeRing = field(default_factory=DeviceRVQCodeRing)
    active: bool = False
    active_lease_id: int = 0
    session_id: str | None = None
    response_id: str | None = None
    generation_id: int | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    callback_count: int = 0
    qwen_kv: Any = None
    future_prefix_qwen: Any = None
    native_q1_executor: Any = None
    cuda_stream: Any = None
    q1_cuda_stream: Any = None
    quarantined: bool = False
    pcm_chunks: list[Any] = field(default_factory=list)
    hook_state: Any = None

    def prepare(self, lease_id: int, session_id: str, response_id: str, generation_id: int) -> None:
        if self.active and self.active_lease_id != lease_id:
            raise RuntimeError("talk context slot is already leased")
        stream_context = (
            torch.cuda.stream(self.cuda_stream)
            if self.cuda_stream is not None and torch.cuda.is_available()
            else nullcontext()
        )
        with stream_context:
            reset = getattr(self.subtalker_graph, "reset", None)
            if callable(reset):
                reset()
            self.rvq_history.reset()
        self.active = True
        self.active_lease_id = lease_id
        self.session_id = session_id
        self.response_id = response_id
        self.generation_id = generation_id
        self.cancel_event.clear()
        self.callback_count = 0
        self.qwen_kv = None
        self.future_prefix_qwen = None
        self.native_q1_executor = None
        self.pcm_chunks.clear()
        self.hook_state = None

    def fence_cuda_streams(self) -> None:
        """Finish this slot's GPU work before its scratch can be reset or reused."""
        seen: set[tuple[str, int] | tuple[str, int]] = set()
        for stream in (self.cuda_stream, self.q1_cuda_stream):
            if stream is None:
                continue
            handle = getattr(stream, "cuda_stream", None)
            key = (str(getattr(stream, "device", "")), int(handle)) if handle is not None else ("object", id(stream))
            if key in seen:
                continue
            seen.add(key)
            synchronize = getattr(stream, "synchronize", None)
            if not callable(synchronize):
                raise RuntimeError("response slot CUDA stream has no synchronize method")
            synchronize()

    def clear(self) -> None:
        executor = self.native_q1_executor
        self.native_q1_executor = None
        try:
            if executor is not None:
                close = getattr(executor, "close", None)
                if callable(close):
                    close()
        finally:
            self.qwen_kv = None
            self.future_prefix_qwen = None
            self.pcm_chunks.clear()
            self.hook_state = None
            self.session_id = None
            self.response_id = None
            self.generation_id = None
            self.active = False


@dataclass
class ResponseTalkLease:
    """Unforgeable generation fence for one use of a context slot."""

    _pool: "ResponseTalkContextPool"
    _slot: _TalkContextSlot
    lease_id: int

    @property
    def slot_id(self) -> int:
        return self._slot.slot_id

    @property
    def session_id(self) -> str:
        return str(self._slot.session_id)

    @property
    def response_id(self) -> str:
        return str(self._slot.response_id)

    @property
    def generation_id(self) -> int:
        return int(self._slot.generation_id)

    @property
    def cuda_stream(self) -> Any:
        return self._slot.cuda_stream

    @property
    def subtalker_graph(self) -> Any:
        return self._slot.subtalker_graph

    @property
    def code2wav_graph(self) -> Any:
        return self._slot.code2wav_graph

    @property
    def rvq_history(self) -> DeviceRVQCodeRing:
        return self._slot.rvq_history

    @property
    def cancel_event(self) -> threading.Event:
        return self._slot.cancel_event

    @property
    def qwen_kv(self) -> Any:
        return self._slot.qwen_kv

    @qwen_kv.setter
    def qwen_kv(self, value: Any) -> None:
        self._require_current()
        self._slot.qwen_kv = value

    @property
    def future_prefix_qwen(self) -> Any:
        return self._slot.future_prefix_qwen

    @future_prefix_qwen.setter
    def future_prefix_qwen(self, value: Any) -> None:
        self._require_current()
        self._slot.future_prefix_qwen = value

    @property
    def native_q1_executor(self) -> Any:
        return self._slot.native_q1_executor

    @native_q1_executor.setter
    def native_q1_executor(self, value: Any) -> None:
        self._require_current()
        self._slot.native_q1_executor = value

    @property
    def pcm_chunks(self) -> list[Any]:
        return self._slot.pcm_chunks

    @property
    def hook_state(self) -> Any:
        return self._slot.hook_state

    @hook_state.setter
    def hook_state(self, value: Any) -> None:
        self._require_current()
        self._slot.hook_state = value

    def _require_current(self) -> None:
        if not self._pool.is_current(self):
            raise RuntimeError("stale response talk lease")

    def is_current(self) -> bool:
        return self._pool.is_current(self)

    def cancel(self) -> bool:
        return self._pool.cancel(self)

    def is_cancelled(self) -> bool:
        return self._pool.is_cancelled(self)

    def callback_started(self) -> None:
        self._pool._callback_started(self)

    def callback_finished(self) -> None:
        self._pool._callback_finished(self)


class ResponseTalkContextPool:
    """Fixed-size lease pool; it never grows or releases on cancel alone."""

    def __init__(self, subtalker_graphs: list[Any], code2wav_graphs: list[Any] | None = None) -> None:
        if not subtalker_graphs:
            raise ValueError("response talk context pool requires at least one graph")
        identities = [id(graph) for graph in subtalker_graphs]
        if len(set(identities)) != len(identities):
            raise ValueError("each response talk context must own a distinct SubtalkerGraph")
        if code2wav_graphs is not None:
            if len(code2wav_graphs) != len(subtalker_graphs):
                raise ValueError("one Code2Wav graph is required per response slot")
            if len({id(graph) for graph in code2wav_graphs}) != len(code2wav_graphs):
                raise ValueError("each response slot must own distinct Code2Wav graph scratch")
        else:
            code2wav_graphs = [None] * len(subtalker_graphs)
        self._condition = threading.Condition()
        self._slots = []
        for i, graph in enumerate(subtalker_graphs):
            device = getattr(graph, "device", None)
            if device is None and code2wav_graphs[i] is not None:
                device = getattr(code2wav_graphs[i], "device", None)
            slot_stream = None
            if device is not None and torch.cuda.is_available() and torch.device(device).type == "cuda":
                slot_stream = torch.cuda.Stream(device=torch.device(device), priority=0)
            self._slots.append(_TalkContextSlot(
                i, graph, code2wav_graphs[i],
                rvq_history=DeviceRVQCodeRing(device=device) if device is not None else DeviceRVQCodeRing(),
                cuda_stream=slot_stream,
            ))
        self._available = list(self._slots)
        self._next_lease_id = 1

    @property
    def capacity(self) -> int:
        return len(self._slots)

    @property
    def active_count(self) -> int:
        with self._condition:
            return sum(slot.active for slot in self._slots)

    def acquire(
        self,
        session_id: str,
        response_id: str,
        *,
        generation_id: int,
        timeout: float | None = None,
    ) -> ResponseTalkLease:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._available:
                if not any(slot.active for slot in self._slots):
                    raise RuntimeError("all response talk-context slots are quarantined after a CUDA fence failure")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("no response talk context available before admission deadline")
                self._condition.wait(remaining)
            slot = self._available.pop(0)
            lease_id = self._next_lease_id
            self._next_lease_id += 1
            slot.active = True
            slot.active_lease_id = lease_id
        try:
            # Reset the graph only after the previous worker and its callbacks
            # returned and this slot was exclusively leased to the new response.
            slot.prepare(lease_id, str(session_id), str(response_id), int(generation_id))
        except BaseException:
            try:
                slot.fence_cuda_streams()
                slot.clear()
            except BaseException:
                slot.quarantined = True
            with self._condition:
                slot.active = False
                if not slot.quarantined:
                    self._available.append(slot)
                    self._available.sort(key=lambda item: item.slot_id)
                self._condition.notify_all()
            raise
        return ResponseTalkLease(self, slot, lease_id)

    def response_stream_for_slot(self, slot_id: int) -> Any:
        slot_id = int(slot_id)
        with self._condition:
            if slot_id < 0 or slot_id >= len(self._slots):
                raise IndexError(f"response talk-context slot {slot_id} fuori intervallo")
            slot = self._slots[slot_id]
            if slot.active:
                raise RuntimeError("cannot bind a response stream while its slot is leased")
            return slot.cuda_stream

    def bind_q1_stream_for_slot(self, slot_id: int, stream: Any) -> None:
        slot_id = int(slot_id)
        with self._condition:
            if slot_id < 0 or slot_id >= len(self._slots):
                raise IndexError(f"response talk-context slot {slot_id} fuori intervallo")
            slot = self._slots[slot_id]
            if slot.active:
                raise RuntimeError("cannot bind the Q1 stream while its slot is leased")
            slot.q1_cuda_stream = stream

    def release(self, lease: ResponseTalkLease) -> None:
        with self._condition:
            if not self._matches_locked(lease):
                raise RuntimeError("stale or already released response talk lease")
            if lease._slot.callback_count:
                raise RuntimeError("cannot release response talk lease while callbacks still active")
            # Fence late callbacks immediately, then clean response-owned
            # resources without holding the pool condition across GPU cleanup.
            lease._slot.active = False
        try:
            lease._slot.fence_cuda_streams()
            lease._slot.clear()
        except BaseException:
            with self._condition:
                lease._slot.quarantined = True
                lease._slot.active = False
                self._condition.notify_all()
            raise
        with self._condition:
            self._available.append(lease._slot)
            self._available.sort(key=lambda item: item.slot_id)
            self._condition.notify_all()

    def is_current(self, lease: ResponseTalkLease) -> bool:
        with self._condition:
            return self._matches_locked(lease)

    def cancel(self, lease: ResponseTalkLease) -> bool:
        with self._condition:
            if not self._matches_locked(lease):
                return False
            lease._slot.cancel_event.set()
            return True

    def is_cancelled(self, lease: ResponseTalkLease) -> bool:
        with self._condition:
            return (
                not self._matches_locked(lease)
                or lease._slot.cancel_event.is_set()
            )

    def _matches_locked(self, lease: ResponseTalkLease) -> bool:
        slot = lease._slot
        return (
            slot in self._slots
            and slot.active
            and slot.active_lease_id == lease.lease_id
        )

    def _callback_started(self, lease: ResponseTalkLease) -> None:
        with self._condition:
            if not self._matches_locked(lease):
                raise RuntimeError("stale response talk lease cannot start an audio callback")
            lease._slot.callback_count += 1

    def _callback_finished(self, lease: ResponseTalkLease) -> None:
        with self._condition:
            if not self._matches_locked(lease):
                raise RuntimeError("stale response talk lease callback completed after reuse")
            if lease._slot.callback_count <= 0:
                raise RuntimeError("response talk callback counter underflow")
            lease._slot.callback_count -= 1
