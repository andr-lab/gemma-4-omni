from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from duplex_kernel.gpu_admission import (
    DEFAULT_GPU_ADMISSION_CAPACITY,
    GpuAdmissionASGIMiddleware,
    GpuAdmissionLimiter,
    parse_gpu_admission_capacity,
)
from duplex_kernel.server import SessionManager, create_app


class _Engine:
    loaded = True
    enforce_production_listener = False

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace()


def test_gpu_capacity_defaults_to_one_and_rejects_invalid_values() -> None:
    assert DEFAULT_GPU_ADMISSION_CAPACITY == 1
    assert parse_gpu_admission_capacity("1") == 1
    assert parse_gpu_admission_capacity(2) == 2
    for invalid in (0, -1, True, "0", "1.5", None):
        with pytest.raises(ValueError, match="positive integer"):
            parse_gpu_admission_capacity(invalid)


def _scope(path: str, body: bytes = b"") -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "_test_body": body,
    }


async def _request(app, scope: dict) -> list[dict]:
    messages: list[dict] = []
    received = False

    async def receive() -> dict:
        nonlocal received
        if received:
            return {"type": "http.disconnect"}
        received = True
        return {"type": "http.request", "body": scope.pop("_test_body", b""), "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    await app(scope, receive, send)
    return messages


def test_gpu_admission_lease_releases_only_after_all_idempotent_references() -> None:
    limiter = GpuAdmissionLimiter(1, source="test")
    frontend = limiter.try_acquire("http")
    assert frontend is not None
    producer = frontend.retain()

    assert frontend.release() is True
    assert frontend.release() is False
    assert limiter.snapshot()["active_leases"] == 1

    assert producer.release() is True
    assert producer.release() is False
    snapshot = limiter.snapshot()
    assert snapshot["active_leases"] == 0
    assert snapshot["total_acquired"] == snapshot["total_released"] == 1


def test_http_middleware_holds_frontend_lease_until_response_body_finishes() -> None:
    limiter = GpuAdmissionLimiter(1, source="test")
    started = asyncio.Event()
    finish_first = asyncio.Event()
    calls: list[str] = []

    async def inference_app(scope, _receive, send):
        calls.append(scope["path"])
        assert scope["state"]["gpu_admission_lease"].kind == "http"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        started.set()
        await finish_first.wait()
        await send({"type": "http.response.body", "body": b"done", "more_body": False})

    middleware = GpuAdmissionASGIMiddleware(
        inference_app,
        acquire=lambda kind: (limiter.try_acquire(kind), None),
        inference_paths=("/infer",),
    )

    async def scenario() -> None:
        first_messages: list[dict] = []

        async def send_first(message: dict) -> None:
            first_messages.append(message)

        first = asyncio.create_task(
            middleware(_scope("/infer"), lambda: None, send_first)
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        assert limiter.snapshot()["active_leases"] == 1

        rejected = await _request(middleware, _scope("/infer"))
        rejection_start = next(message for message in rejected if message["type"] == "http.response.start")
        rejection_body = next(message for message in rejected if message["type"] == "http.response.body")
        assert rejection_start["status"] == 429
        assert json.loads(rejection_body["body"])["error"]["code"] == "GPU_CAPACITY_EXCEEDED"
        assert calls == ["/infer"]
        assert limiter.snapshot()["rejected"] == 1

        finish_first.set()
        await asyncio.wait_for(first, timeout=2)
        assert first_messages[-1]["body"] == b"done"
        assert limiter.snapshot()["active_leases"] == 0

    asyncio.run(scenario())


def test_realtime_capacity_rejection_precedes_session_allocation() -> None:
    sessions = SessionManager()
    app = create_app(_Engine(), sessions)
    occupied = app.state.gpu_admission_limiter.try_acquire("realtime")
    assert occupied is not None

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?duplex=1") as websocket:
            rejection = websocket.receive_json()
            assert rejection["type"] == "error"
            assert rejection["error"]["code"] == "GPU_CAPACITY_EXCEEDED"
            with pytest.raises(WebSocketDisconnect) as disconnect:
                websocket.receive_json()
            assert disconnect.value.code == 1013

    assert sessions.all() == []
    assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 1
    occupied.release()
    assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 0


def test_realtime_conversation_holds_lease_and_rejects_second_socket(monkeypatch) -> None:
    from duplex_kernel.turn_policy import NoOpPolicy
    import duplex_kernel.server as server_module

    sessions = SessionManager()
    engine = _Engine()
    engine.default_turn_policy_factory = NoOpPolicy
    app = create_app(engine, sessions)
    monkeypatch.setattr(server_module, "_resolve_p7_session_voice", lambda *_args, **_kwargs: None)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?duplex=1") as first:
            created = first.receive_json()
            assert created["type"] == "session.created"
            assert len(sessions.all()) == 1
            assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 1

            with client.websocket_connect("/v1/realtime?duplex=1") as second:
                rejection = second.receive_json()
                assert rejection["type"] == "error"
                assert rejection["error"]["code"] == "GPU_CAPACITY_EXCEEDED"
                with pytest.raises(WebSocketDisconnect) as disconnect:
                    second.receive_json()
                assert disconnect.value.code == 1013

            assert len(sessions.all()) == 1
            first.send_json({"type": "session.close"})

    assert sessions.all() == []
    assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 0


def test_realtime_startup_failure_releases_lease_and_session(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    sessions = SessionManager()
    app = create_app(_Engine(), sessions)
    monkeypatch.setattr(
        server_module,
        "_resolve_p7_session_voice",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )

    with TestClient(app) as client:
        with pytest.raises(RuntimeError, match="startup failed"):
            with client.websocket_connect("/v1/realtime?duplex=1"):
                pass

    assert sessions.all() == []
    assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 0


def test_incompatible_talk_context_pool_is_reported_and_rejects_before_route() -> None:
    sessions = SessionManager()
    engine = _Engine()
    engine.gpu_admission_capacity = 2
    engine.gpu_admission_capacity_source = "test-config"
    engine.response_talk_context_pool = SimpleNamespace(admission_capacity=1)
    app = create_app(engine, sessions)

    with TestClient(app) as client:
        health = client.get("/health").json()
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert health["status"] == "degraded"
    assert health["gpu_capacity"]["configured_capacity"] == 2
    assert health["gpu_capacity"]["source"] == "test-config"
    assert health["gpu_capacity"]["talk_context_compatible"] is False
    assert health["gpu_capacity"]["configuration_error"] == "talk_context_pool_below_gpu_admission_capacity"
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "GPU_ADMISSION_UNAVAILABLE"
    assert sessions.all() == []


def test_stream_disconnect_releases_frontend_but_keeps_worker_lease_until_exit(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    worker_started = threading.Event()
    release_worker = threading.Event()
    worker_exited = threading.Event()
    sessions = SessionManager()

    def run_response(_engine, _session, _messages, request, **_kwargs):
        request["_on_text_chunk"]("first")
        worker_started.set()
        assert release_worker.wait(5), "test did not release blocked inference worker"
        worker_exited.set()
        return {"text": "first", "wav": None, "tool_calls": [], "finish_reason": "stop"}

    monkeypatch.setattr(server_module, "_server_response_run_response", run_response)
    app = create_app(_Engine(), sessions)

    async def scenario() -> None:
        incoming: asyncio.Queue[dict] = asyncio.Queue()
        outgoing: asyncio.Queue[dict] = asyncio.Queue(maxsize=4)
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "modalities": ["text"],
            "stream": True,
        }
        await incoming.put({
            "type": "http.request",
            "body": json.dumps(payload).encode("utf-8"),
            "more_body": False,
        })

        async def receive() -> dict:
            return await incoming.get()

        async def send(message: dict) -> None:
            await outgoing.put(message)

        task = asyncio.create_task(app(_scope("/v1/chat/completions"), receive, send))
        response_start = await asyncio.wait_for(outgoing.get(), timeout=3)
        assert response_start["status"] == 200
        assert await asyncio.to_thread(worker_started.wait, 3)
        first_body = await asyncio.wait_for(outgoing.get(), timeout=3)
        assert b'"content": "first"' in first_body.get("body", b"")
        assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 1

        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=3)
        assert not worker_exited.is_set()
        assert len(sessions.all()) == 1, "the session must outlive its still-running producer"
        assert app.state.gpu_admission_limiter.snapshot()["active_leases"] == 1

        second = await _request(
            app,
            _scope(
                "/v1/chat/completions",
                json.dumps(payload).encode("utf-8"),
            ),
        )
        second_start = next(message for message in second if message["type"] == "http.response.start")
        second_body = next(message for message in second if message["type"] == "http.response.body")
        assert second_start["status"] == 429
        assert json.loads(second_body["body"])["error"]["code"] == "GPU_CAPACITY_EXCEEDED"
        assert len(sessions.all()) == 1

        release_worker.set()
        assert await asyncio.to_thread(worker_exited.wait, 3)
        deadline = asyncio.get_running_loop().time() + 3
        while app.state.gpu_admission_limiter.snapshot()["active_leases"] != 0:
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("producer completion did not release GPU admission lease")
            await asyncio.sleep(0.01)
        assert sessions.all() == []

    asyncio.run(scenario())

