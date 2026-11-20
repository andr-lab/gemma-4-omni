"""Process-wide admission for HTTP inference and realtime GPU conversations."""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


GPU_ADMISSION_CAPACITY_ENV = "DUPLEX_GPU_ADMISSION_CAPACITY"
DEFAULT_GPU_ADMISSION_CAPACITY = 1


def parse_gpu_admission_capacity(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("GPU admission capacity must be a positive integer")
    if isinstance(value, int):
        capacity = value
    elif isinstance(value, str) and value.strip().isdigit():
        capacity = int(value.strip())
    else:
        raise ValueError("GPU admission capacity must be a positive integer")
    if capacity < 1:
        raise ValueError("GPU admission capacity must be a positive integer")
    return capacity


@dataclass(slots=True)
class _LeaseRecord:
    lease_id: str
    kind: str
    owner_id: str | None
    references: int = 1
    closed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class GpuAdmissionLease:
    """One idempotent reference to a shared GPU admission lease."""

    __slots__ = ("_limiter", "_record", "_released")

    def __init__(self, limiter: "GpuAdmissionLimiter", record: _LeaseRecord):
        self._limiter = limiter
        self._record = record
        self._released = False

    @property
    def lease_id(self) -> str:
        return self._record.lease_id

    @property
    def kind(self) -> str:
        return self._record.kind

    def retain(self) -> "GpuAdmissionLease":
        record = self._record
        with record.lock:
            if self._released or record.closed:
                raise RuntimeError("cannot retain a released GPU admission lease")
            record.references += 1
        return GpuAdmissionLease(self._limiter, record)

    def release(self) -> bool:
        record = self._record
        should_remove = False
        with record.lock:
            if self._released:
                return False
            self._released = True
            record.references -= 1
            if record.references < 0:
                raise RuntimeError("GPU admission lease reference count underflow")
            if record.references == 0 and not record.closed:
                record.closed = True
                should_remove = True
        if should_remove:
            self._limiter._release(record.lease_id)
        return True


class GpuAdmissionLimiter:
    """Nonblocking N-slot lease pool shared by all server inference entrypoints."""

    def __init__(self, capacity: int, *, source: str):
        self.capacity = parse_gpu_admission_capacity(capacity)
        self.source = str(source).strip() or "unspecified"
        self._lock = threading.Lock()
        self._leases: dict[str, _LeaseRecord] = {}
        self._total_acquired = 0
        self._total_released = 0
        self._rejected = 0

    def try_acquire(self, kind: str, *, owner_id: str | None = None) -> GpuAdmissionLease | None:
        normalized_kind = str(kind).strip()
        if not normalized_kind:
            raise ValueError("GPU admission lease kind must be non-empty")
        with self._lock:
            if len(self._leases) >= self.capacity:
                self._rejected += 1
                return None
            record = _LeaseRecord(
                lease_id=uuid.uuid4().hex,
                kind=normalized_kind,
                owner_id=None if owner_id is None else str(owner_id),
            )
            self._leases[record.lease_id] = record
            self._total_acquired += 1
            return GpuAdmissionLease(self, record)

    def record_rejection(self) -> None:
        with self._lock:
            self._rejected += 1

    def _release(self, lease_id: str) -> None:
        with self._lock:
            if self._leases.pop(str(lease_id), None) is not None:
                self._total_released += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_by_kind: dict[str, int] = {}
            for record in self._leases.values():
                active_by_kind[record.kind] = active_by_kind.get(record.kind, 0) + 1
            return {
                "configured_capacity": self.capacity,
                "active_leases": len(self._leases),
                "active_by_kind": active_by_kind,
                "rejected": self._rejected,
                "total_acquired": self._total_acquired,
                "total_released": self._total_released,
                "source": self.source,
                "concurrency_verified": False,
            }


class GpuAdmissionASGIMiddleware:
    """Hold the frontend reference until the full HTTP response body ends."""

    def __init__(
        self,
        app: Any,
        *,
        acquire: Callable[[str], tuple[GpuAdmissionLease | None, str | None]],
        inference_paths: Iterable[str],
    ):
        self.app = app
        self.acquire = acquire
        self.inference_paths = frozenset(str(path) for path in inference_paths)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method", "").upper() != "POST"
            or scope.get("path") not in self.inference_paths
        ):
            await self.app(scope, receive, send)
            return

        lease, rejection_reason = self.acquire("http")
        if lease is None:
            unavailable = rejection_reason not in {None, "capacity"}
            payload = json.dumps(
                {
                    "error": {
                        "code": (
                            "GPU_ADMISSION_UNAVAILABLE"
                            if unavailable else "GPU_CAPACITY_EXCEEDED"
                        ),
                        "message": (
                            "GPU inference capacity is unavailable"
                            if unavailable else "GPU inference capacity is full"
                        ),
                    }
                },
                separators=(",", ":"),
            ).encode("utf-8")
            status = 503 if unavailable else 429
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"cache-control", b"no-store"),
                ],
            })
            await send({"type": "http.response.body", "body": payload, "more_body": False})
            return

        state = scope.setdefault("state", {})
        if not isinstance(state, dict):
            lease.release()
            raise RuntimeError("ASGI request state must be a dictionary")
        state["gpu_admission_lease"] = lease
        try:
            await self.app(scope, receive, send)
        finally:
            lease.release()
