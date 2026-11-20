from __future__ import annotations

import asyncio
import base64
import json
import threading
import wave
from io import BytesIO
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from duplex_kernel.audio_stream import AUDIO_FRAME_SAMPLES
from duplex_kernel.server import SessionManager, create_app


class _Engine:
    loaded = True
    enforce_production_listener = False

    def __init__(self) -> None:
        self.tokenizer = SimpleNamespace()


def _result(*, text: str = "", finish_reason: str = "stop", wav=None, tool_calls=None):
    return {
        "text": text,
        "wav": wav,
        "tool_calls": [] if tool_calls is None else tool_calls,
        "finish_reason": finish_reason,
    }


def _events(body: bytes) -> list[dict]:
    parsed = []
    for frame in body.decode("utf-8").split("\n\n"):
        if not frame.startswith("data: ") or frame == "data: [DONE]":
            continue
        parsed.append(json.loads(frame[6:]))
    return parsed


async def _open_asgi_stream(app, payload: dict):
    incoming: asyncio.Queue[dict] = asyncio.Queue()
    outgoing: asyncio.Queue[dict] = asyncio.Queue(maxsize=1)
    await incoming.put({
        "type": "http.request",
        "body": json.dumps(payload).encode("utf-8"),
        "more_body": False,
    })

    async def receive():
        return await incoming.get()

    async def send(message):
        await outgoing.put(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    task = asyncio.create_task(app(scope, receive, send))
    start = await asyncio.wait_for(outgoing.get(), timeout=5)
    assert start["type"] == "http.response.start"
    return task, incoming, outgoing, start


async def _collect_asgi_stream(app, payload: dict, *, after_body=None) -> tuple[int, bytes]:
    task, _incoming, outgoing, start = await _open_asgi_stream(app, payload)
    chunks = []
    while True:
        message = await asyncio.wait_for(outgoing.get(), timeout=5)
        if message["type"] != "http.response.body":
            continue
        chunks.append(message.get("body", b""))
        if after_body is not None:
            await after_body(message.get("body", b""))
        if not message.get("more_body", False):
            break
    await asyncio.wait_for(task, timeout=5)
    return int(start["status"]), b"".join(chunks)


def test_nonstream_completion_uses_runtime_finish_reason(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    monkeypatch.setattr(server_module, "_server_response_run_response", lambda *_args, **_kwargs: _result(
        text="limit reached", finish_reason="length"
    ))
    with TestClient(create_app(_Engine())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "modalities": ["text"]},
        )
    assert response.status_code == 200
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_health_reports_loaded_runtime_sources_and_unverified_gpu_capacity() -> None:
    with TestClient(create_app(_Engine())) as client:
        response = client.get("/health")

    assert response.status_code == 200
    health = response.json()
    sources = health["runtime_sources"]
    assert sources["server"].replace("\\", "/").endswith("duplex_kernel/server.py")
    assert sources["listener"].replace("\\", "/").endswith("duplex_kernel/listener.py")
    assert set(sources["versions"]) == {"torch", "transformers", "tokenizers"}
    capacity = health["gpu_capacity"]
    assert capacity["configured_response_contexts"] == 1
    assert capacity["pool_capacity"] is None
    assert capacity["configured_capacity"] == 1
    assert capacity["active_leases"] == 0
    assert capacity["rejected"] == 0
    assert capacity["source"] == "default"
    assert capacity["concurrency_verified"] is False


def test_stream_sends_callback_text_while_producer_is_still_running(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    producer_done = threading.Event()
    release_producer = threading.Event()

    def run_response(_engine, _session, _messages, request, **_kwargs):
        request["_on_text_chunk"]("first")
        assert release_producer.wait(5), "test did not release the producer after the first event"
        request["_on_text_chunk"]("second")
        producer_done.set()
        return _result(text="firstsecond", finish_reason="length")

    monkeypatch.setattr(server_module, "_server_response_run_response", run_response)
    app = create_app(_Engine())
    first_content_seen = False

    async def after_body(body: bytes) -> None:
        nonlocal first_content_seen
        if not first_content_seen:
            first_content_seen = True
            event = _events(body)[0]
            assert event["choices"][0]["delta"]["content"] == "first"
            assert not producer_done.is_set()
            release_producer.set()

    status, body = asyncio.run(_collect_asgi_stream(
        app,
        {"messages": [{"role": "user", "content": "hello"}], "modalities": ["text"], "stream": True},
        after_body=after_body,
    ))

    events = _events(body)
    deltas = [event["choices"][0]["delta"]["content"] for event in events if event["choices"][0]["delta"].get("content")]
    endings = [event["choices"][0]["finish_reason"] for event in events if event["choices"][0].get("finish_reason")]
    assert status == 200
    assert first_content_seen and producer_done.is_set()
    assert deltas == ["first", "second"]
    assert endings == ["length"]
    assert body.count(b"data: [DONE]") == 1


def test_stream_audio_packet_is_wav_with_private_generation_and_24khz_metadata(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    producer_done = threading.Event()
    release_producer = threading.Event()
    ownership = {}
    pcm = np.full(2 * AUDIO_FRAME_SAMPLES, 0.25, dtype=np.float32)

    def run_response(_engine, _session, _messages, request, **_kwargs):
        token = request["_generation_token"]
        ownership["generation_id"] = token.session_generation_id
        ownership["speaker_segment_id"] = token.speaker_segment_id
        request["_on_audio_chunk_with_horizon"](pcm, 1)
        assert release_producer.wait(5), "test did not release the producer after the first audio packet"
        producer_done.set()
        return _result(finish_reason="stop", wav=pcm.copy())

    monkeypatch.setattr(server_module, "_server_response_run_response", run_response)
    app = create_app(_Engine())
    first_audio_seen = False

    async def after_body(body: bytes) -> None:
        nonlocal first_audio_seen
        if first_audio_seen:
            return
        events = _events(body)
        audio_events = [event for event in events if event["choices"][0]["delta"].get("audio")]
        if audio_events:
            first_audio_seen = True
            audio = audio_events[0]["choices"][0]["delta"]["audio"]
            assert not producer_done.is_set()
            assert audio["generation_id"] == ownership["generation_id"]
            assert audio["speaker_segment_id"] == ownership["speaker_segment_id"]
            assert (audio["first_frame_id"], audio["last_frame_id"]) == (0, 1)
            assert (audio["start_sample"], audio["end_sample"], audio["sample_rate"]) == (0, 3840, 24000)
            assert audio["mime_type"] == "audio/wav"
            wav_bytes = base64.b64decode(audio["chunk"])
            with wave.open(BytesIO(wav_bytes), "rb") as wav_file:
                assert wav_file.getframerate() == 24000
                assert wav_file.getnframes() == 3840
            release_producer.set()

    status, body = asyncio.run(_collect_asgi_stream(
        app,
        {
            "messages": [{"role": "user", "content": "speak"}],
            "modalities": ["text", "audio"],
            "stream": True,
        },
        after_body=after_body,
    ))
    events = _events(body)
    audio_events = [event for event in events if event["choices"][0]["delta"].get("audio")]
    assert status == 200
    assert first_audio_seen and producer_done.is_set()
    assert len(audio_events) == 1
    assert [event["choices"][0].get("finish_reason") for event in events if event["choices"][0].get("finish_reason")] == ["stop"]
    assert body.count(b"data: [DONE]") == 1


def test_stream_tool_calls_are_published_once_and_errors_end_with_one_done(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
    }

    def run_response(_engine, _session, _messages, request, **_kwargs):
        if request.get("tools"):
            request["_on_text_chunk"]("private pre-tool text")
            return _result(text="private pre-tool text", tool_calls=[tool_call])
        request["_on_text_chunk"]("partial")
        raise RuntimeError("producer failed after headers")

    monkeypatch.setattr(server_module, "_server_response_run_response", run_response)
    with TestClient(create_app(_Engine())) as client:
        tools_response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "lookup"}],
                "modalities": ["text"],
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            },
        )
        error_response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "fail"}], "modalities": ["text"], "stream": True},
        )

    tool_events = _events(tools_response.content)
    calls = [event for event in tool_events if event["choices"][0]["delta"].get("tool_calls")]
    assert tools_response.status_code == 200
    assert len(calls) == 1
    assert calls[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_1"
    assert "private pre-tool text" not in tools_response.text
    assert [event["choices"][0].get("finish_reason") for event in tool_events if event["choices"][0].get("finish_reason")] == ["tool_calls"]

    error_events = _events(error_response.content)
    assert error_response.status_code == 200
    assert any(event.get("error", {}).get("code") == "INFERENCE_ERROR" for event in error_events)
    assert not any(event.get("choices", [{}])[0].get("finish_reason") == "stop" for event in error_events)
    assert error_response.text.count("data: [DONE]") == 1


def test_disconnect_cancels_blocked_callback_and_next_request_uses_execution_gate(monkeypatch) -> None:
    import duplex_kernel.server as server_module

    calls = 0
    calls_lock = threading.Lock()
    first_producer_done = threading.Event()

    def run_response(_engine, _session, _messages, request, **_kwargs):
        nonlocal calls
        with calls_lock:
            calls += 1
            call_number = calls
        if call_number == 1:
            for index in range(1000):
                if request["_stop_check"]():
                    break
                request["_on_text_chunk"](str(index))
            first_producer_done.set()
            return _result(finish_reason="cancelled")
        request["_on_text_chunk"]("recovered")
        return _result(text="recovered", finish_reason="stop")

    monkeypatch.setattr(server_module, "_server_response_run_response", run_response)
    app = create_app(_Engine(), SessionManager())

    async def disconnect_and_retry() -> None:
        task, incoming, outgoing, start = await _open_asgi_stream(
            app,
            {"messages": [{"role": "user", "content": "slow"}], "modalities": ["text"], "stream": True},
        )
        assert start["status"] == 200
        first_body = await asyncio.wait_for(outgoing.get(), timeout=5)
        assert first_body["type"] == "http.response.body"
        assert _events(first_body.get("body", b""))[0]["choices"][0]["delta"]["content"] == "0"
        await asyncio.sleep(0.05)
        assert not first_producer_done.is_set(), "producer should be backpressured by the bounded event queue"
        await incoming.put({"type": "http.disconnect"})
        await asyncio.wait_for(task, timeout=5)
        assert await asyncio.to_thread(first_producer_done.wait, 5)

        status, body = await _collect_asgi_stream(
            app,
            {"messages": [{"role": "user", "content": "retry"}], "modalities": ["text"], "stream": True},
        )
        assert status == 200
        assert any(
            event["choices"][0]["delta"].get("content") == "recovered"
            for event in _events(body)
        )

    asyncio.run(disconnect_and_retry())
