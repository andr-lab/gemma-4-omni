from __future__ import annotations

import queue
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.scheduler import AudioIngressQueue, DuplexScheduler, VirtualFrameClock
from duplex_kernel.engine import GenerationAdmissionAborted, ListenerBacklogAdmission, PipelineEngine
from duplex_kernel.server import _wait_for_listener_backlog_admission, create_app
from duplex_kernel.session import DuplexSession, SessionManager
from duplex_kernel.turn_policy import NoOpPolicy


FRAME_SAMPLES = 1280


def test_prefill_admits_at_one_queued_frame_even_while_callback_is_active() -> None:
    now = [0.0]
    callback_active = True
    admitted = _wait_for_listener_backlog_admission(
        lambda: 1,
        lambda: True,
        callbacks_active=callback_active,
        now=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    assert callback_active is True
    assert admitted is ListenerBacklogAdmission.ADMITTED
    assert now[0] == 0.0


def test_prefill_admission_waits_only_for_backlog_catchup_and_is_bounded() -> None:
    now = [0.0]
    depths = iter([3, 2, 1])
    admitted = _wait_for_listener_backlog_admission(
        lambda: next(depths),
        lambda: True,
        callbacks_active=True,
        max_wait_s=0.160,
        now=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    assert admitted is ListenerBacklogAdmission.ADMITTED
    assert 0.0 < now[0] <= 0.160


def test_prefill_admission_stops_on_stale_generation_or_deadline() -> None:
    now = [0.0]
    stale = _wait_for_listener_backlog_admission(
        lambda: 5,
        lambda: False,
        now=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    assert stale is ListenerBacklogAdmission.ABORTED
    assert now[0] == 0.0

    deferred = _wait_for_listener_backlog_admission(
        lambda: 1 if now[0] >= 0.20 else 2,
        lambda: True,
        callbacks_active=True,
        max_wait_s=0.160,
        poll_interval_s=0.040,
        now=lambda: now[0],
        sleep=lambda duration: now.__setitem__(0, now[0] + duration),
    )
    assert deferred is ListenerBacklogAdmission.ADMITTED
    assert now[0] == pytest.approx(0.240)


def test_stale_generation_aborts_before_thinker_audio_forward() -> None:
    import torch

    class _Model:
        calls = 0

        def __call__(self, **_kwargs):
            self.calls += 1
            raise AssertionError("stale generation must not reach the model")

    class _Thinker:
        def __init__(self):
            self.model = _Model()

        def __call__(self, **_kwargs):
            self.model.calls += 1
            raise AssertionError("stale generation must not reach the model")

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.thinker = _Thinker()
    engine.device = "cpu"
    audio_inputs = {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "input_features": torch.ones((1, 1, 2), dtype=torch.float32),
        "input_features_mask": torch.ones((1, 1), dtype=torch.long),
    }
    with pytest.raises(GenerationAdmissionAborted):
        engine._run_thinker_audio_prefill(
            audio_inputs,
            is_compiled=False,
            static_cache=None,
            past_length=0,
            cache_init_started=None,
            runtime_metrics={},
            listener_admission_wait=lambda: ListenerBacklogAdmission.ABORTED,
        )
    assert engine.thinker.model.calls == 0


def test_thinker_returns_cancelled_for_typed_stale_prefill_abort() -> None:
    import torch

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine.thinker_static_cache_max_len = 4096
    engine._prepare_thinker_audio_inputs = lambda _inputs, _ids: {
        "input_ids": torch.ones((1, 2), dtype=torch.long),
        "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "input_features": torch.ones((1, 1, 2), dtype=torch.float32),
        "input_features_mask": torch.ones((1, 1), dtype=torch.long),
    }

    def abort_prefill(*_args, **_kwargs):
        raise GenerationAdmissionAborted("stale test generation")

    engine._run_thinker_audio_prefill = abort_prefill
    result = engine.think([1], input_audio={})
    assert result["cancelled"] is True
    assert result["runtime_metrics"]["thinker_prefill_aborted_stale_generation"] is True


class _Tokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return messages[-1]["content"]

    def __call__(self, _text, **_kwargs):
        return {"input_ids": np.array([[1]])}


class _Engine:
    loaded = True
    enforce_production_listener = False
    tokenizer = _Tokenizer()
    audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    default_turn_policy_factory = NoOpPolicy


def test_direct_v2_websocket_advertises_six_frame_queue_but_legacy_stays_three(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_DIRECT_AUDIO_CANDIDATE", "1")
    v2_sessions = SessionManager()
    v2_client = TestClient(create_app(_Engine(), v2_sessions))
    health = v2_client.get("/health").json()
    assert health["response_protocol"] == AUDIO_RESPONSE_PROTOCOL_V2
    assert health["direct_audio_candidate"] is True
    assert health["thinker_audio_forward_active"] is False
    with v2_client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["input_audio_buffer_capacity_frames"] == 6
        session = v2_sessions.all()[0]
        assert session.audio_ingress.max_frames == 6
        ws.send_json({"type": "session.close"})

    class _LegacyEngine(_Engine):
        audio_response_protocol = None

    legacy_sessions = SessionManager()
    legacy_client = TestClient(create_app(_LegacyEngine(), legacy_sessions))
    with legacy_client.websocket_connect("/v1/realtime?duplex=1") as ws:
        created = ws.receive_json()
        assert created["input_audio_buffer_capacity_frames"] == 3
        assert legacy_sessions.all()[0].audio_ingress.max_frames == 3
        ws.send_json({"type": "session.close"})


def test_six_frame_v2_ingress_retains_every_frame_and_rejects_only_beyond_bound() -> None:
    session = DuplexSession("v2-ingress", audio_ingress_capacity_frames=6)
    for frame_id in range(6):
        pcm = np.full(FRAME_SAMPLES, frame_id / 8, dtype=np.float32)
        session.audio_ingress.put(pcm, frame_id=frame_id)

    with pytest.raises(queue.Full):
        session.audio_ingress.put(np.ones(FRAME_SAMPLES, dtype=np.float32), frame_id=6)
    assert [session.audio_ingress.get_nowait().frame_id for _ in range(6)] == list(range(6))
    assert session.audio_ingress.stats()["dropped_input_frames"] == 1


def test_scheduler_priority_indicator_covers_listener_and_policy_and_clears_on_error() -> None:
    ingress = AudioIngressQueue(max_frames=1)
    observed: list[tuple[str, bool]] = []
    scheduler = DuplexScheduler(
        ingress,
        clock=VirtualFrameClock(),
        on_listener=lambda _frame, _tick: observed.append(("listener", scheduler.callbacks_active)),
        on_policy=lambda _frame, _tick: observed.append(("policy", scheduler.callbacks_active)),
    )
    ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=0)

    scheduler.tick()

    assert observed == [("listener", True), ("policy", True)]
    assert not scheduler.callbacks_active

    failing = DuplexScheduler(
        AudioIngressQueue(max_frames=1),
        clock=VirtualFrameClock(),
        on_listener=lambda *_args: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    failing.ingress.put(np.zeros(FRAME_SAMPLES, dtype=np.float32), frame_id=0)
    with pytest.raises(RuntimeError, match="fixture"):
        failing.tick()
    assert not failing.callbacks_active


def test_listener_admission_callback_records_wait_without_holding_a_model_lock() -> None:
    from duplex_kernel.engine import PipelineEngine

    called = []
    metrics: dict[str, float] = {}
    PipelineEngine._wait_for_listener_admission(
        lambda: called.append("released"), metrics, "talker"
    )

    assert called == ["released"]
    assert metrics["listener_admission_talker_wait_ms"] >= 0
    assert metrics["listener_admission_talker_wait_max_ms"] >= 0


def test_talker_waits_for_listener_before_every_qwen_forward(monkeypatch) -> None:
    import torch

    import duplex_kernel.engine as engine_module
    from duplex_kernel.engine import PipelineEngine

    timeline: list[str] = []

    class _TalkerModel:
        def __call__(self, **_kwargs):
            timeline.append("qwen_forward")
            return SimpleNamespace(
                past_key_values=object(),
                last_hidden_state=torch.ones((1, 1, 2048), dtype=torch.float32),
            )

    class _Talker:
        model = _TalkerModel()
        tcfg = SimpleNamespace()

        def codec_head(self, hidden):
            timeline.append("codec_head")
            return torch.zeros((1, 1, 8), dtype=hidden.dtype)

        def get_input_embeddings(self):
            return lambda _token: torch.zeros((1, 1, 2048), dtype=torch.float32)

    class _Subtalker:
        def generate(self, *_args, **_kwargs):
            timeline.append("subtalker")
            return [1] * 15

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = SimpleNamespace(
        talker=_Talker(),
        tcfg=SimpleNamespace(),
        decode=lambda _codes: np.zeros(240, dtype=np.float32),
    )
    engine.subtalker_graph = _Subtalker()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.allow_talker_eos = False
    engine._stop_event = False
    engine._postprocess = lambda wav: wav
    engine._cb0_sample = lambda _logits, **_kwargs: 2

    monkeypatch.setattr(engine_module, "_estimate_talker_frame_budget", lambda *_a, **_k: (2, 0, 2))
    monkeypatch.setattr(
        engine_module,
        "_build_talker_prefill_parts",
        lambda *_a, **_k: (
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
        ),
    )
    monkeypatch.setattr(
        engine_module,
        "build_qwen_audio_conditioning",
        lambda *_a, **_k: torch.zeros((1, 1, 2048)),
    )

    def listener_admission_wait() -> None:
        timeline.append("listener_admitted")

    engine.talk(
        "Ciao",
        torch.ones((1, 1, 2048), dtype=torch.float32),
        n_words=1,
        min_frames=2,
        listener_admission_wait=listener_admission_wait,
    )

    forward_indexes = [i for i, event in enumerate(timeline) if event == "qwen_forward"]
    assert len(forward_indexes) == 3  # static prefill, say/V9 prefix, next-frame Qwen step
    previous_forward_index = -1
    for index in forward_indexes:
        assert "listener_admitted" in timeline[previous_forward_index + 1:index]
        previous_forward_index = index


def test_compiled_v2_talker_uses_priority_lock_without_per_frame_admission(monkeypatch) -> None:
    import threading
    import torch

    import duplex_kernel.engine as engine_module
    from duplex_kernel.engine import PipelineEngine

    timeline: list[str] = []
    lock = threading.Lock()

    class _Cache:
        def __init__(self, length: int = 0) -> None:
            self.length = length

        def get_seq_length(self, layer_idx: int = 0) -> int:
            del layer_idx
            return self.length

    class _TalkerModel:
        def __call__(self, *, inputs_embeds, past_key_values=None, use_cache):
            assert use_cache is True
            assert lock.locked(), "Qwen forward must retain listener-priority serialization"
            timeline.append("qwen_forward")
            cache = past_key_values if past_key_values is not None else _Cache()
            cache.length += int(inputs_embeds.shape[1])
            return SimpleNamespace(
                past_key_values=cache,
                last_hidden_state=torch.ones((1, int(inputs_embeds.shape[1]), 2048)),
            )

    class _Talker:
        model = _TalkerModel()
        tcfg = SimpleNamespace()

        def codec_head(self, hidden):
            assert lock.locked(), "codec head must retain listener-priority serialization"
            timeline.append("codec_head")
            return torch.zeros((1, 1, 8), dtype=hidden.dtype)

        def get_input_embeddings(self):
            def embed(_token):
                assert lock.locked(), "Talker embedding lookup must stay in the priority section"
                return torch.zeros((1, 1, 2048), dtype=torch.float32)
            return embed

    class _Subtalker:
        def generate(self, *_args, return_tensor=False, **_kwargs):
            assert lock.locked(), "SubTalker must retain listener-priority serialization"
            assert return_tensor is True
            timeline.append("subtalker")
            return torch.ones((15,), dtype=torch.long)

    talker = _Talker()
    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = SimpleNamespace(
        talker=talker,
        tcfg=SimpleNamespace(),
        decode=lambda _codes: np.zeros(240, dtype=np.float32),
    )
    engine.subtalker_graph = _Subtalker()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.audio_response_protocol = AUDIO_RESPONSE_PROTOCOL_V2
    engine._compiled_talker_q1 = lambda inputs, cache: (
        talker.model(
            inputs_embeds=inputs,
            past_key_values=cache,
            use_cache=True,
        ).last_hidden_state[:, -1:, :],
        cache,
    )
    engine.talker_q1_backend = "torch_compile_dynamic"
    engine.talker_q1_compile_status = "ready"
    engine._thinker_lock = lock
    engine.allow_talker_eos = False
    engine._stop_event = False
    engine._postprocess = lambda wav: wav
    engine._cb0_sample = lambda _logits, **_kwargs: 2

    monkeypatch.setattr(engine_module, "_estimate_talker_frame_budget", lambda *_a, **_k: (2, 0, 2))
    monkeypatch.setattr(
        engine_module,
        "_resolve_talker_voice_conditioning",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        engine_module,
        "_build_talker_prefill_parts",
        lambda *_a, **_k: (
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
            torch.zeros((1, 1, 2048)),
        ),
    )
    monkeypatch.setattr(
        engine_module,
        "build_qwen_audio_conditioning",
        lambda *_a, **_k: torch.zeros((1, 1, 2048)),
    )

    admissions: list[str] = []
    metrics: dict[str, object] = {}

    def listener_admission_wait() -> None:
        admissions.append("admission")

    engine.talk(
        "Ciao",
        torch.ones((1, 1, 2048), dtype=torch.float32),
        n_words=1,
        min_frames=2,
        listener_admission_wait=listener_admission_wait,
        metrics=metrics,
    )

    assert admissions == ["admission", "admission"]  # eager static/dynamic prefills only
    assert timeline.count("qwen_forward") == 3  # two eager prefills, one compiled Q1
    assert timeline.count("codec_head") == 2
    assert timeline.count("subtalker") == 2
    assert metrics["compiled_talker_q1_calls"] == 1
    assert metrics["talker_q1_backend"] == "torch_compile_dynamic"
    assert metrics["listener_priority_dynamic_prefill_lock_hold_ms"] >= 0
    assert metrics["listener_priority_subtalker_lock_hold_ms"] >= 0
    assert metrics["listener_priority_next_qwen_model_lock_hold_ms"] >= 0
    assert not lock.locked()


def test_direct_v2_startup_warmup_uses_validated_say_text_and_only_say_hiddens(monkeypatch) -> None:
    import torch
    import sys
    from pathlib import Path

    from duplex_kernel.audio_response_protocol import format_audio_response_v2_target
    from duplex_kernel.engine import PipelineEngine
    from duplex_kernel import voices

    observed: dict[str, object] = {}
    voice_profile = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(2048, dtype=torch.float32),
    }
    monkeypatch.setattr(
        voices,
        "resolve_voice_conditioning",
        lambda name: voice_profile if name == "gemma" else None,
    )

    class _Processor:
        audio_token = "<|audio|>"

        def __call__(self, **kwargs):
            observed["processor"] = kwargs
            return {"input_ids": torch.tensor([[10, 11, 12, 13]])}

    class _Tokenizer:
        audio_token = "<|audio|>"

        def __init__(self):
            self._ids_by_text = {}
            self._text_by_id = {}

        def apply_chat_template(self, messages, **kwargs):
            observed["messages"] = messages
            observed["template_kwargs"] = kwargs
            return "rendered generic audio prompt"

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            value = self._ids_by_text.setdefault(text, 1000 + len(self._ids_by_text))
            self._text_by_id[value] = text
            return [value]

        def decode(self, token_ids, *, skip_special_tokens):
            assert skip_special_tokens
            return "".join(self._text_by_id[int(value)] for value in token_ids)

    engine = PipelineEngine.__new__(PipelineEngine)
    engine._thinker_wrapper = SimpleNamespace(processor=_Processor())
    engine.tokenizer = _Tokenizer()
    engine.v9_stream = SimpleNamespace(raw_dim=1536)
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.direct_v2_audio_warmup_status = "pending"
    engine._compiled_talker_q1 = None
    engine.talker_q1_compile_status = "disabled"
    engine.talker_q1_backend = "eager"

    def _think(_prompt_ids, **kwargs):
        observed["think_kwargs"] = kwargs
        assert kwargs["max_new_tokens"] == 128
        assert kwargs["input_audio"]["input_ids"].shape == (1, 4)
        tokenizer = engine.tokenizer
        target = format_audio_response_v2_target(
            heard="Siamo in quattro persone.",
            action="SPEAK",
            say="Perfetto.",
        )
        pieces = [
            "{",
            '"heard":"',
            "Siamo in quattro persone.",
            '"',
            ',"action":"',
            "SPEAK",
            '"',
            ',"say":"',
            "Perfetto",
            ".",
            '"}',
        ]
        start_id = kwargs["on_generation_start"]()
        assert start_id == tokenizer.encode("{", add_special_tokens=False)[0]
        for index, piece in enumerate(pieces):
            token_id = tokenizer.encode(piece, add_special_tokens=False)[0]
            assert kwargs["on_candidate_token"](token_id, piece, False)
            forced = kwargs["on_generated_token"](
                index,
                token_id,
                piece,
                torch.full((1, 1, 1536), float(index), dtype=torch.float32),
            )
            if forced is not None and index + 1 < len(pieces):
                forced_id = forced[0] if isinstance(forced, tuple) else forced
                assert forced_id == tokenizer.encode(pieces[index + 1], add_special_tokens=False)[0]
        assert kwargs["on_candidate_token"](0, "", True)
        return {
            "text": target,
            "hidden": torch.full((1, len(pieces), 1536), -99.0),
            "hidden_prompt_len": 4,
            "gen_ids": [tokenizer.encode(piece, add_special_tokens=False)[0] for piece in pieces],
        }

    def _v9_prefix(hidden, prompt_len):
        assert prompt_len == 0
        observed["v9_input_shape"] = tuple(hidden.shape)
        observed["v9_input_values"] = hidden[0, :, 0].tolist()
        return torch.ones((1, hidden.shape[1], 2048), dtype=torch.float32)

    def _talk(text, prefix, **kwargs):
        observed["talk"] = (text, tuple(prefix.shape), kwargs)
        return np.zeros(240, dtype=np.float32)

    engine.think = _think
    engine.v9_prefix = _v9_prefix
    engine.talk = _talk

    fixture_name = "sample_1_native_qwen_reference.wav"
    original_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: path.name == fixture_name or original_is_file(path),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        SimpleNamespace(read=lambda *_args, **_kwargs: (np.ones(4800, dtype=np.float32) * 0.1, 24000)),
    )

    engine._warmup_direct_v2_audio_path()

    processor_kwargs = observed["processor"]
    assert processor_kwargs["sampling_rate"] == 16000
    assert processor_kwargs["audio"][0].shape == (3200,)
    assert observed["v9_input_shape"] == (1, 2, 1536)
    assert observed["v9_input_values"] == [8.0, 9.0]
    assert observed["talk"] == (
        "Perfetto.",
        (1, 2, 2048),
        {
            "n_words": 1,
            "voice_conditioning": voice_profile,
            "min_frames": 4,
        },
    )
    assert engine.direct_v2_audio_warmup_status == "complete"
