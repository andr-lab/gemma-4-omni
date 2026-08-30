"""OpenAI-compatible HTTP/SSE and realtime duplex WebSocket server."""
from __future__ import annotations
import asyncio, base64, io, json, logging, queue, threading, time, uuid
from typing import Any
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from duplex_kernel.config import CODEC_SAMPLE_RATE, USER_SAMPLE_RATE, TURNHEAD_CKPT
from duplex_kernel.turn_policy import ControlAction, NoOpPolicy, PolicyInput, ScriptedPolicy, SpeakerGate
from duplex_kernel.turnhead_adapter import TurnHeadPolicy
from duplex_kernel.session import SessionManager, DuplexSession
from duplex_kernel.tool_parser import extract_tool_calls, tool_system_prompt
from duplex_kernel.voices import resolve_voice_conditioning

logger = logging.getLogger("duplex_kernel.server")
MODEL_ID = "gemma-4-omni-duplex"


def _wav_b64(wav: np.ndarray, sr: int = CODEC_SAMPLE_RATE) -> str:
    import soundfile as sf
    buf = io.BytesIO(); sf.write(buf, np.asarray(wav, dtype=np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _message_text(content: Any) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return " ".join(str(x.get("text", "")) for x in content if isinstance(x, dict)).strip()
    return ""


def create_app(engine: Any, session_mgr: SessionManager | None = None) -> FastAPI:
    app = FastAPI(title="Gemma-4-Omni Duplex Kernel", version="0.3.0")
    sessions = session_mgr or SessionManager()
    inference_lock = threading.Lock()  # GPU model is single-stream by design.

    speaker_gate = SpeakerGate()

    def _make_policy(name: str, cfg: dict[str, Any] | None = None) -> Any:
        cfg = cfg or {}
        if name == "none":
            return NoOpPolicy()
        if name == "scripted":
            return ScriptedPolicy(cfg.get("scripted_actions", [ControlAction.LISTEN.value]))
        if name != "turnhead":
            raise ValueError(f"turn_policy non valida: {name}")
        return TurnHeadPolicy(device=getattr(engine, "device", "cuda"), checkpoint=TURNHEAD_CKPT)

    def _ensure_policy(sess: DuplexSession) -> None:
        if sess.turn_policy is None:
            sess.turn_policy = _make_policy("turnhead")

    def get_session(request: dict[str, Any]) -> tuple[DuplexSession, bool]:
        sid = request.get("session_id")
        if sid:
            sess = sessions.get(str(sid))
            if sess is None: raise ValueError("session_id non valido o scaduto")
            _ensure_policy(sess)
            return sess, False
        sess = sessions.create_session(
            instructions=str(request.get("instructions", "")),
            voice=str(request.get("voice", "gemma")),
            language=str(request.get("language", "Italian")),
        )
        _ensure_policy(sess)
        return sess, True

    def run_response(sess: DuplexSession, messages: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
        clean = [m for m in messages if isinstance(m, dict) and m.get("role") in {"system", "user", "assistant", "tool"}]
        if sess.instructions and not any(m.get("role") == "system" for m in clean):
            clean.insert(0, {"role": "system", "content": sess.instructions})
        tools = request.get("tools") or sess.tools
        if tools:
            clean.insert(0, {"role": "system", "content": tool_system_prompt(tools)})
        generation_id = request.get("_generation_id")
        if generation_id is not None and not sess.is_generation_current(generation_id=generation_id):
            sess.stale_session_mutations += 1
            return {"text": "", "tool_calls": [], "wav": None, "elapsed": 0.0, "cancelled": True}
        # La sessione è la proprietaria del rolling context: il prompt non
        # cresce senza limite e non dipende da una lista globale del server.
        sess.thinker_ctx.system_prompt = sess.instructions
        sess.thinker_ctx.set_messages(clean)
        clean = sess.thinker_ctx.messages()
        prompt = _message_text(clean[-1].get("content", "")) if clean else ""
        chat = engine.tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=True)
        ids = engine.tokenizer(chat, return_tensors="pt")["input_ids"][0].tolist()
        cond = resolve_voice_conditioning(
            request.get("voice") or request.get("speaker") or sess.voice,
            request.get("voice_style") or request.get("instructions") or sess.voice_style,
            request.get("ref_audio"), request.get("ref_text"),
        )
        streamed_audio: list[np.ndarray] = []
        stream_requested = callable(request.get("_on_audio_chunk")) and not tools
        def render_text_chunk(chunk_text: str, chunk_hidden: Any, chunk_prompt_len: int) -> None:
            if not chunk_text or not stream_requested:
                return
            chunk_h = engine.v9_prefix(chunk_hidden, chunk_prompt_len)
            if sess.turn_policy is not None and (generation_id is None or sess.is_generation_current(generation_id=generation_id)):
                sess.turn_policy.update_hidden(chunk_h)
            elif generation_id is not None:
                sess.stale_session_mutations += 1
            if request.get("_generation_token") is not None:
                sess.record_synthesized_text(request["_generation_token"], chunk_text)
            if callable(request.get("_on_text_chunk")):
                request["_on_text_chunk"](chunk_text)
            chunk_wav = engine.talk(chunk_text, chunk_h, n_words=max(1, len(chunk_text.split())),
                                    voice_conditioning=cond, stop_check=request.get("_stop_check"),
                                    state=sess, on_audio_chunk=request.get("_on_audio_chunk"), min_frames=8,
                                    generation_id=generation_id)
            streamed_audio.append(chunk_wav)
        think = engine.think(ids, max_new_tokens=int(request.get("max_tokens", 64)),
                             temperature=float(request.get("temperature", 0.7)), top_p=float(request.get("top_p", 0.9)),
                             stop_check=request.get("_stop_check"), state=sess,
                             generation_id=generation_id,
                             on_text_chunk=render_text_chunk if stream_requested else None)
        if request.get("_stop_check") and request["_stop_check"]():
            return {"text": "", "tool_calls": [], "wav": None, "elapsed": 0.0, "cancelled": True}
        calls = extract_tool_calls(think["text"], tools) if tools else []
        if calls:
            return {"text": "", "tool_calls": calls, "wav": None, "elapsed": 0.0}
        text = think["text"]
        if request.get("_generation_token") is not None:
            sess.record_planned_text(request["_generation_token"], text)
        h_pfx = engine.v9_prefix(think["hidden"], think["prompt_len"])
        if sess.turn_policy is not None and (generation_id is None or sess.is_generation_current(generation_id=generation_id)):
            sess.turn_policy.update_hidden(h_pfx)
        elif generation_id is not None:
            sess.stale_session_mutations += 1
        wav = None
        if "audio" in (request.get("modalities") or ["text"]):
            if stream_requested and streamed_audio:
                wav = np.concatenate(streamed_audio)
            else:
                wav = engine.talk(text, h_pfx, n_words=max(1, len(text.split())), voice_conditioning=cond,
                                   stop_check=lambda: sess.stop_requested, state=sess,
                                   on_audio_chunk=request.get("_on_audio_chunk"), generation_id=generation_id)
        if request.get("_defer_history"):
            history_messages = clean[-16:]
        elif generation_id is None or sess.is_generation_current(generation_id=generation_id):
            sess.history = clean[-16:] + [{"role": "assistant", "content": text}]
        else:
            sess.stale_history_writes += 1
        return {"text": text, "wav": wav, "tool_calls": [], "elapsed": 0.0,
                "history_messages": clean[-16:]}

    def sse_event(payload: dict[str, Any]) -> str:
        return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "engine_loaded": bool(getattr(engine, "loaded", False)), "model": MODEL_ID}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "duplex_kernel"}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: dict[str, Any]):
        if not getattr(engine, "loaded", False):
            return JSONResponse({"error": {"code": "ENGINE_NOT_READY", "message": "engine non caricato"}}, status_code=503)
        messages = request.get("messages")
        try:
            max_tokens = int(request.get("max_tokens", 64)); temperature = float(request.get("temperature", 0.7)); top_p = float(request.get("top_p", 0.9))
            if max_tokens < 1 or not 0.0 < temperature or not 0.0 < top_p <= 1.0:
                raise ValueError
        except (TypeError, ValueError):
            return JSONResponse({"error": {"code": "VALIDATION_ERROR", "message": "max_tokens/temperature/top_p non validi"}}, status_code=422)
        if not isinstance(messages, list) or not messages:
            return JSONResponse({"error": {"code": "VALIDATION_ERROR", "message": "messages deve essere una lista non vuota"}}, status_code=422)
        try: sess, ephemeral = get_session(request)
        except (ValueError, RuntimeError) as exc:
            return JSONResponse({"error": {"code": "SESSION_ERROR", "message": str(exc)}}, status_code=422)
        sess.tools = request.get("tools") or sess.tools
        sess.voice = str(request.get("voice") or request.get("speaker") or sess.voice)
        sess.voice_style = request.get("voice_style") or sess.voice_style
        sess.ref_text = request.get("ref_text") or sess.ref_text
        loop = asyncio.get_running_loop()
        context_messages = (sess.history + messages) if request.get("session_id") else messages
        def infer():
            with inference_lock: return run_response(sess, context_messages, request)
        try: result = await loop.run_in_executor(None, infer)
        except Exception:
            if ephemeral: sessions.close(sess.session_id)
            logger.exception("richiesta fallita")
            return JSONResponse({"error": {"code": "INFERENCE_ERROR", "message": "inferenza fallita"}}, status_code=500)
        if ephemeral: result["session_id"] = sess.session_id
        if result["tool_calls"]:
            msg = {"role": "assistant", "content": None, "tool_calls": result["tool_calls"]}
            body = {"id": "chatcmpl_" + uuid.uuid4().hex, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}], "session_id": sess.session_id}
            return JSONResponse(body)
        if not request.get("stream"):
            msg = {"role": "assistant", "content": result["text"]}
            if result["wav"] is not None: msg["audio"] = {"data": _wav_b64(result["wav"]), "expires_at": int(time.time()) + 86400}
            body = {"id": "chatcmpl_" + uuid.uuid4().hex, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
            if ephemeral: body["session_id"] = sess.session_id
            return JSONResponse(body)

        def generate():
            if result["tool_calls"]:
                for index, call in enumerate(result["tool_calls"]):
                    yield sse_event({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": index, **call}]}}]})
                yield sse_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
                yield "data: [DONE]\n\n"; return
            words = result["text"].split(" ")
            for i, word in enumerate(words):
                yield sse_event({"choices": [{"index": 0, "delta": {"content": word + (" " if i + 1 < len(words) else "")}}]})
            wav = result["wav"]
            if wav is not None:
                chunk = int(CODEC_SAMPLE_RATE * 0.640)
                for start in range(0, len(wav), chunk):
                    yield sse_event({"choices": [{"index": 0, "delta": {"audio": {"chunk": _wav_b64(wav[start:start + chunk])}}}]})
            yield sse_event({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket, duplex: int = 0):
        if not duplex:
            await ws.close(code=1008, reason="usa /v1/realtime?duplex=1"); return
        await ws.accept(); sess = sessions.create_session(); _ensure_policy(sess); response_task: asyncio.Task | None = None
        auto_task: asyncio.Task | None = None
        loop = asyncio.get_running_loop()
        scheduler_events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def publish_scheduler_events() -> None:
            while True:
                event = await scheduler_events.get()
                if event is None:
                    return
                try:
                    await ws.send_json(event)
                except (WebSocketDisconnect, RuntimeError):
                    return

        def scheduler_policy(frame: Any, tick: Any) -> None:
            """Policy callback invoked only by DuplexScheduler's tick thread."""
            if frame is None:
                return
            policy = sess.turn_policy
            if policy is None:
                return
            decision = policy.step(PolicyInput(
                session_id=sess.session_id,
                frame_id=frame.frame_id,
                user_pcm=frame.pcm,
                speaker_active=sess.active_response,
                generation_id=sess.session_generation_id,
                speaker_segment_id=sess.speaker_segment_id,
            ))
            gate = speaker_gate.apply(decision, sess)
            logger.info("[realtime %s] scheduler action=%s frame=%d active=%s", sess.session_id[:8], decision.action.value, frame.frame_id, sess.active_response)
            if not gate.stop_synthesis:
                return
            if hasattr(engine, "stop_generation"):
                engine.stop_generation()
            with sess._generation_lock:
                if sess.response_cancel_notified:
                    return
                sess.response_cancel_notified = True
                event = {"type": "response.cancelled", "reason": gate.action.value.lower(), "response_id": sess.response_id}
            loop.call_soon_threadsafe(scheduler_events.put_nowait, event)

        def scheduler_error(exc: BaseException) -> None:
            loop.call_soon_threadsafe(
                scheduler_events.put_nowait,
                {"type": "error", "error": {"code": "SCHEDULER_ERROR", "message": str(exc)[:240]}},
            )

        # The scheduler is the sole consumer of audio_ingress and owns the
        # 80 ms cadence; the WebSocket path only validates and appends PCM.
        sess.scheduler.on_policy = scheduler_policy
        sess.scheduler.on_error = scheduler_error
        sess.scheduler.start(name=f"duplex-scheduler-{sess.session_id[:8]}")
        scheduler_event_task = asyncio.create_task(publish_scheduler_events())

        async def produce_response(committed_audio: np.ndarray) -> None:
            nonlocal response_task
            response_id = "resp_" + uuid.uuid4().hex[:12]
            started = time.perf_counter()
            token = sess.begin_response(response_id)
            sess.active_response = True
            logger.info("[realtime %s] start %s audio_samples=%d auto=%s", sess.session_id[:8], response_id, committed_audio.size, sess.auto_response)
            try:
                await ws.send_json({"type": "response.created", "response_id": response_id})
                audio_queue: queue.Queue[np.ndarray] = queue.Queue()
                text_queue: queue.Queue[str] = queue.Queue()
                streamed_chunks = 0
                def on_audio_chunk(chunk: np.ndarray) -> None:
                    array = np.asarray(chunk, dtype=np.float32)
                    sess.guarded_write(token, "audio", lambda: audio_queue.put(array))
                def on_text_chunk(chunk: str) -> None:
                    sess.guarded_write(token, "text", lambda: text_queue.put(str(chunk)))

                def infer_ws():
                    with inference_lock:
                        history = list(sess.history)
                        if committed_audio.size and hasattr(engine, "transcribe_audio"):
                            logger.info("[realtime %s %s] ASR start samples=%d", sess.session_id[:8], response_id, committed_audio.size)
                            audio_text = engine.transcribe_audio(committed_audio)["text"]
                            logger.info("[realtime %s %s] ASR done text_chars=%d elapsed=%.2fs", sess.session_id[:8], response_id, len(audio_text), time.perf_counter() - started)
                            if audio_text:
                                history.append({"role": "user", "content": audio_text})
                        logger.info("[realtime %s %s] Thinker/Talker start", sess.session_id[:8], response_id)
                        result = run_response(sess, history or [{"role": "user", "content": "Ciao!"}], {"modalities": ["text", "audio"], "tools": sess.tools, "voice": sess.voice, "voice_style": sess.voice_style, "ref_text": sess.ref_text, "_generation_id": token.session_generation_id, "_generation_token": token, "_defer_history": True, "_stop_check": lambda: sess.stop_requested, "_on_audio_chunk": on_audio_chunk, "_on_text_chunk": on_text_chunk})
                        logger.info("[realtime %s %s] Thinker/Talker done elapsed=%.2fs", sess.session_id[:8], response_id, time.perf_counter() - started)
                        return result
                loop = asyncio.get_running_loop()
                future = loop.run_in_executor(None, infer_ws)
                deadline = loop.time() + 180.0
                while not future.done():
                    if loop.time() >= deadline:
                        sess.stop_requested = True
                        sess.invalidate_generation("realtime_timeout")
                        if hasattr(engine, "stop_generation"):
                            engine.stop_generation()
                        logger.error("[realtime %s %s] timeout dopo %.1fs", sess.session_id[:8], response_id, time.perf_counter() - started)
                        await ws.send_json({"type": "error", "error": {"code": "INFERENCE_TIMEOUT", "message": "risposta oltre 180 secondi"}, "response_id": response_id})
                        return
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        await asyncio.sleep(0.005)
                    else:
                        streamed_chunks += 1
                        await ws.send_json({"type": "response.audio.delta", "delta": _wav_b64(chunk_wav), "response_id": response_id})
                        sess.record_emitted_audio(token, 8)
                    try:
                        chunk_text = text_queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        await ws.send_json({"type": "response.output_text.delta", "delta": chunk_text, "response_id": response_id})
                        sess.record_emitted_text(token, chunk_text)
                result = await future
                while True:
                    try:
                        chunk_wav = audio_queue.get_nowait()
                    except queue.Empty:
                        break
                    streamed_chunks += 1
                    await ws.send_json({"type": "response.audio.delta", "delta": _wav_b64(chunk_wav), "response_id": response_id})
                    sess.record_emitted_audio(token, 8)
                while True:
                    try:
                        chunk_text = text_queue.get_nowait()
                    except queue.Empty:
                        break
                    await ws.send_json({"type": "response.output_text.delta", "delta": chunk_text, "response_id": response_id})
                    sess.record_emitted_text(token, chunk_text)
                if not sess.is_generation_current(token):
                    return
                if result.get("cancelled"):
                    if not sess.response_cancel_notified:
                        await ws.send_json({"type": "response.cancelled", "reason": "cooperative_stop", "response_id": response_id})
                    return
                if result["tool_calls"]:
                    for call in result["tool_calls"]:
                        fn = call["function"]
                        await ws.send_json({"type": "response.function_call_arguments.done", "call_id": call["id"], "name": fn["name"], "arguments": fn["arguments"]})
                    return
                if not sess.speaker_segment or not sess.speaker_segment.emitted_text:
                    await ws.send_json({"type": "response.output_text.delta", "delta": result["text"], "response_id": response_id})
                    sess.record_emitted_text(token, result["text"])
                sess.commit_history(token, result.get("history_messages", []), result["text"])
                if result["wav"] is not None and streamed_chunks == 0:
                    chunk = int(CODEC_SAMPLE_RATE * .640)
                    for start in range(0, len(result["wav"]), chunk):
                        audio_chunk = result["wav"][start:start + chunk]
                        await ws.send_json({"type": "response.audio.delta", "delta": _wav_b64(audio_chunk), "response_id": response_id})
                        sess.record_emitted_audio(token, max(1, int(round(len(audio_chunk) / (CODEC_SAMPLE_RATE * .08)))))
                await ws.send_json({"type": "response.done", "response_id": response_id})
                logger.info("[realtime %s %s] done total=%.2fs audio_chunks=%d", sess.session_id[:8], response_id, time.perf_counter() - started, streamed_chunks)
                queued_audio = sess.pending_audio.copy()
                sess.reset_turn()
                sess.pending_audio = queued_audio
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("realtime response failed")
                await ws.send_json({"type": "error", "error": {"code": "INFERENCE_ERROR", "message": str(exc)[:240]}})
            finally:
                sess.end_response(token)
                if sess.auto_response and sess.pending_audio.size and not sess.active_response:
                    await schedule_auto_response()

        auto_generation = 0
        async def schedule_auto_response() -> None:
            nonlocal auto_task, response_task, auto_generation
            auto_generation += 1
            generation = auto_generation
            if auto_task and not auto_task.done():
                auto_task.cancel()
            async def delayed() -> None:
                nonlocal response_task
                await asyncio.sleep(0.35)
                if generation != auto_generation or sess.active_response or not sess.pending_audio.size:
                    return
                committed = sess.pending_audio.copy()
                sess.pending_audio = np.empty(0, dtype=np.float32)
                response_task = asyncio.create_task(produce_response(committed))
            auto_task = asyncio.create_task(delayed())

        try:
            await ws.send_json({"type": "session.created", "session_id": sess.session_id})
            while True:
                try:
                    ev = json.loads(await ws.receive_text())
                    if not isinstance(ev, dict): raise ValueError("evento deve essere un oggetto JSON")
                except (json.JSONDecodeError, ValueError) as exc:
                    await ws.send_json({"type": "error", "error": {"code": "INVALID_EVENT", "message": str(exc)}})
                    continue
                et = ev.get("type")
                if et == "session.update":
                    cfg = ev.get("session", {})
                    sess.instructions = str(cfg.get("instructions", sess.instructions)); sess.voice = str(cfg.get("voice", sess.voice)); sess.tools = cfg.get("tools", sess.tools) or []
                    extra = cfg.get("extra_body", {}) or {}; sess.auto_response = bool(cfg.get("auto_response", extra.get("auto_response", sess.auto_response)))
                    policy_name = cfg.get("turn_policy") or extra.get("turn_policy")
                    if policy_name is not None:
                        sess.turn_policy = _make_policy(str(policy_name), cfg)
                    await ws.send_json({"type": "session.updated", "session_id": sess.session_id})
                elif et == "conversation.item.create":
                    item = ev.get("item", {}); sess.history.append({"role": item.get("role", "user"), "content": _message_text(item.get("content", ""))}); await ws.send_json({"type": "conversation.item.created"})
                elif et == "input_audio_buffer.append":
                    try:
                        raw = base64.b64decode(ev.get("audio", ""), validate=True)
                        if not raw or len(raw) % 2:
                            raise ValueError("audio PCM16 vuoto o con lunghezza dispari")
                        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
                    except (ValueError, TypeError) as exc:
                        await ws.send_json({"type": "error", "error": {"code": "INVALID_AUDIO", "message": str(exc)}})
                        continue
                    if sess.pending_audio.size + pcm.size > USER_SAMPLE_RATE * 30:
                        await ws.send_json({"type": "error", "error": {"code": "AUDIO_LIMIT", "message": "massimo 30 secondi per turno"}}); continue
                    sess.touch(); sess.pending_audio = np.concatenate((sess.pending_audio, pcm))
                    # Il WebSocket è solo transport: la queue è consumata e
                    # segmentata dal DuplexScheduler, che invoca la policy a
                    # cadenza 80 ms anche mentre la risposta è attiva.
                    sess.audio_ingress.append(pcm)
                    if sess.auto_response and not sess.active_response:
                        await schedule_auto_response()
                elif et in {"input_audio_buffer.commit", "response.create"}:
                    if response_task and not response_task.done():
                        await ws.send_json({"type": "error", "error": {"code": "RESPONSE_ACTIVE", "message": "response già attiva"}})
                        continue
                    if auto_task and not auto_task.done():
                        auto_task.cancel()
                    auto_generation += 1
                    # Congela il micro-turn per il worker: l'audio che arriva
                    # mentre la risposta parla appartiene già al turno successivo.
                    committed_audio = sess.pending_audio.copy()
                    sess.pending_audio = np.empty(0, dtype=np.float32)
                    # Non-blocking: the receive loop resta vivo durante
                    # Thinker/V9/Talker e può ricevere barge-in.
                    response_task = asyncio.create_task(produce_response(committed_audio))
                elif et == "session.close": break
        except (WebSocketDisconnect, asyncio.CancelledError): pass
        finally:
            if auto_task and not auto_task.done(): auto_task.cancel()
            if response_task and not response_task.done(): response_task.cancel()
            scheduler_events.put_nowait(None)
            if scheduler_event_task and not scheduler_event_task.done(): scheduler_event_task.cancel()
            sessions.close(sess.session_id)
    return app


def run(engine: Any, *, host: str = "0.0.0.0", port: int = 8092) -> None:
    import uvicorn
    uvicorn.run(create_app(engine), host=host, port=port, log_level="info")

__all__ = ["create_app", "run"]
