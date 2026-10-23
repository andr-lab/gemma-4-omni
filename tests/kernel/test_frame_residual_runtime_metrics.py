from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import duplex_kernel.engine as engine_module
from duplex_kernel.engine import PipelineEngine
from duplex_kernel.live_say import LiveQwenSayStream, TurnLocalQwenSayStream
from duplex_kernel.session import (
    SpanAlignedSpeechCursor,
    SpokenTokenPacket,
    TurnLocalSpeechFIFO,
    estimate_token_audio_frames_80ms,
)
from duplex_kernel.server import (
    _filter_frame_residual_runtime_metrics,
    _filter_native_thinker_lock_metrics,
    _filter_turn_local_speech_runtime_metrics,
    _format_turn_local_speech_summary,
)


def test_token_audio_frame_estimator_uses_bounded_token_surface_heuristic() -> None:
    assert estimate_token_audio_frames_80ms("  ") == 1
    assert estimate_token_audio_frames_80ms("hello") == 4
    assert estimate_token_audio_frames_80ms(".") == 2
    assert estimate_token_audio_frames_80ms(",") == 1
    assert estimate_token_audio_frames_80ms("abcdefghijkl") == 6


def test_spoken_packet_exposes_transition_provenance_with_legacy_default() -> None:
    hidden = torch.zeros((1, 1536))
    legacy = SpokenTokenPacket(1, "x", hidden, 1)
    assert legacy.hidden_provenance == "token"
    assert legacy.kv_committed is True
    assert SpokenTokenPacket(2, "y", hidden, 2, hidden_provenance="transition").hidden_provenance == "transition"
    with pytest.raises(ValueError, match="hidden_provenance"):
        SpokenTokenPacket(3, "z", hidden, 3, hidden_provenance="guessed")
    with pytest.raises(TypeError, match="kv_committed"):
        SpokenTokenPacket(4, "w", hidden, 4, kv_committed=1)


def test_span_cursor_sentence_has_35_to_45_frames_and_small_40_frame_tail() -> None:
    words = ("we", " can", " map", " each", " spoken", " word", " to", " frames",
             " and", " keep", " audio", " in", " sync")
    cursor = SpanAlignedSpeechCursor(generation_id=31)
    for index, text in enumerate(words):
        cursor.append(token_id=100 + index, token_text=text, hidden=torch.full((1, 1, 3), index))
    spans = cursor.frame_spans
    assert len(spans) == 13
    assert spans[0].start_frame == 0
    assert all(left.end_frame == right.start_frame for left, right in zip(spans, spans[1:]))
    assert 35 <= spans[-1].end_frame <= 45

    tail_holds = sum(cursor.select(frame, generation_id=31, segment_id=0).tail_hold for frame in range(40))
    assert tail_holds / 40 < 0.25
    assert cursor.select(0, generation_id=31, segment_id=0).span.token_id == 100
    assert cursor.select(0, generation_id=31, segment_id=0).tail_hold is False


def test_report_fixture_estimator_does_not_hide_historical_tail_hold_failure() -> None:
    # The report records this sentence and 11 hidden rows, but no token surfaces.
    # This reconstruction has 11 rows and the same exact source text; it records
    # the heuristic's likely tail miss without pretending the report had offsets.
    token_texts = ("Le", " ore", " son", "o", " le", " 1", "4", ":", "3", "0", ".")
    cursor = SpanAlignedSpeechCursor(generation_id=32)
    for index, text in enumerate(token_texts):
        cursor.append(token_id=index, token_text=text, hidden=torch.zeros((1, 1, 3)))
    tail_holds = sum(cursor.select(frame, generation_id=32, segment_id=0).tail_hold for frame in range(40))
    assert "".join(token_texts) == "Le ore sono le 14:30."
    assert cursor.frame_spans[-1].end_frame == 19
    assert tail_holds == 21
    assert tail_holds / 40 > 0.25


def test_live_span_arrival_eos_and_stale_identity_keep_frame_schedule_stable() -> None:
    cursor = SpanAlignedSpeechCursor(generation_id=7, segment_id=2)
    first = cursor.append(token_id=20, token_text="a", hidden=torch.full((1, 1, 2), 1.0))
    held = cursor.select(first.end_frame + 2, generation_id=7, segment_id=2)
    assert held.span is first
    assert held.tail_hold is True

    second = cursor.append(token_id=21, token_text="b", hidden=torch.full((1, 1, 2), 2.0))
    assert second.start_frame > first.end_frame + 2
    assert cursor.select(first.end_frame + 2, generation_id=7, segment_id=2) is held
    assert cursor.select(second.start_frame, generation_id=7, segment_id=2).span is second
    third = cursor.append(token_id=22, token_text="c", hidden=torch.full((1, 1, 2), 3.0))
    assert third.start_frame == second.end_frame
    assert third.token_sequence == second.token_sequence + 1
    cursor.close()
    assert cursor.select(third.end_frame + 1, generation_id=7, segment_id=2).tail_hold is True
    with pytest.raises(RuntimeError, match="stale generation"):
        cursor.ack_emitted_frame(0, generation_id=6, segment_id=2)
    with pytest.raises(RuntimeError, match="stale segment"):
        cursor.ack_emitted_frame(0, generation_id=7, segment_id=1)


def test_token_arriving_after_pcm_release_never_claims_an_already_emitted_frame() -> None:
    cursor = SpanAlignedSpeechCursor(generation_id=8, segment_id=3)
    first = cursor.append(token_id=30, token_text="a", hidden=torch.zeros((1, 1, 2)))
    assert (first.start_frame, first.end_frame) == (0, 1)
    cursor.ack_emitted_frame(0, generation_id=8, segment_id=3)
    cursor.ack_emitted_frame(1, generation_id=8, segment_id=3)

    second = cursor.append(token_id=31, token_text="b", hidden=torch.ones((1, 1, 2)))
    assert second.start_frame == 2
    assert cursor.select(2, generation_id=8, segment_id=3).span is second


def test_pcm_ack_watermark_preserves_only_fully_started_packets_on_rollback() -> None:
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=41)
    packets = (
        SpokenTokenPacket(1, "we can", torch.zeros((1, 1536)), 11),
        SpokenTokenPacket(2, " map each spoken word", torch.ones((1, 1536)), 12),
        SpokenTokenPacket(3, " to frames and keep audio", torch.full((1, 1536), 2.0), 13),
    )
    surfaces = (("we", " can"), (" map", " each", " spoken", " word"),
                (" to", " frames", " and", " keep", " audio"))
    token_id = 100
    for packet, token_texts in zip(packets, surfaces):
        ids = tuple(range(token_id, token_id + len(token_texts)))
        fifo.put(
            packet,
            qwen_token_ids=ids,
            qwen_token_texts=token_texts,
            generation_id=41,
        )
        token_id += len(token_texts)
    fifo.close()
    while True:
        token = fifo.take_qwen_token(generation_id=41, timeout_s=0)
        if token is None:
            break
        fifo.consume_qwen_token(generation_id=41)

    # Qwen ingestion alone must not move the rollback boundary.
    assert fifo.last_vocalized_kv_position is None
    assert fifo.vocalized_text == ""
    for frame_id in range(20):
        fifo.ack_emitted_audio_frame(
            frame_id,
            generation_id=41,
            segment_id=0,
        )

    cursor = fifo.clear_unspoken_on_bargein()
    assert cursor == 12
    assert cursor != 13
    assert fifo.vocalized_text == "we can map each spoken word"


def test_fifo_pcm_ack_is_monotonic_and_segment_scoped() -> None:
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=51)
    fifo.put(
        SpokenTokenPacket(1, "a", torch.zeros((1, 1536)), 1),
        qwen_token_ids=(1,),
        generation_id=51,
    )
    fifo.take_qwen_token(initial=True, generation_id=51, timeout_s=0)
    assert fifo.consume_qwen_token(generation_id=51) is None
    with pytest.raises(RuntimeError, match="contiguous"):
        fifo.ack_qwen_token(generation_id=51, segment_id=0, frame_id=1)
    fifo.ack_emitted_audio_frame(0, generation_id=51, segment_id=0)
    with pytest.raises(RuntimeError, match="contiguous"):
        fifo.ack_emitted_audio_frame(0, generation_id=51, segment_id=0)


def test_frame_residual_metrics_survive_typed_response_filter() -> None:
    values = {
        "frame_residual_adapter": "response_hidden_q1_input_residual_v1",
        "adapter_mode": "response_hidden_q1_query_film_v2",
        "frame_residual_checkpoint": "/workspace/checkpoints/adapter.pt",
        "frame_residual_calls": 17,
        "frame_residual_tail_hold_calls": 2,
        "frame_residual_qwen_q1_only": True,
        "frame_residual_prefix_tokens_added": 0,
        "frame_residual_learned_scale": 0.05,
        "frame_residual_delta_sq_sum": 0.003,
        "frame_residual_base_sq_sum": 3.2,
        "frame_residual_rms": 0.012,
        "frame_residual_base_rms": 0.31,
        "frame_residual_rms_to_base_ratio": 0.0387,
        "frame_residual_live_first_last_hidden_cosine": 0.519,
        "qwen_route": "plain_native_audio_q1",
        "qwen_prefix_slots_added": 0,
        "qwen_semantic_text_unchanged": True,
        "source": "gemma_assistant_response_posttoken_hidden",
        "frame_residual_live_hidden_trace": [
            {
                "qwen_frame_step": 0,
                "gemma_token_id": 1201,
                "gemma_kv_position": 261,
                "qwen_token_id": 991,
                "qwen_packet_token_index": 0,
                "qwen_packet_token_count": 1,
                "tail_hold": False,
                "gemma_hidden_norm": 38.25,
                "gemma_hidden_cosine_previous_packet": None,
            }
        ],
        "untrusted_extra": "drop me",
    }

    result = _filter_frame_residual_runtime_metrics(values)

    assert result == {key: value for key, value in values.items() if key != "untrusted_extra"}


def test_frame_residual_trace_is_scalar_bounded_and_rejects_invalid_entries() -> None:
    good = {
        "qwen_frame_step": 0,
        "gemma_token_id": 1201,
        "gemma_kv_position": 261,
        "qwen_token_id": 991,
        "qwen_packet_token_index": 0,
        "qwen_packet_token_count": 1,
        "tail_hold": False,
        "gemma_hidden_norm": 38.25,
        "gemma_hidden_cosine_previous_packet": None,
    }
    bad = {**good, "gemma_hidden_norm": float("inf")}
    result = _filter_frame_residual_runtime_metrics({
        "frame_residual_live_hidden_trace": [good, bad] * 200,
    })
    assert len(result["frame_residual_live_hidden_trace"]) == 128
    assert all(
        value is None or type(value) in {bool, int, float}
        for entry in result["frame_residual_live_hidden_trace"]
        for value in entry.values()
    )
    assert _filter_frame_residual_runtime_metrics({
        "frame_residual_live_hidden_trace": [{**good, "qwen_frame_step": True}],
    }) == {}


def test_frame_residual_metric_filter_rejects_wrong_types_and_nonfinite_values() -> None:
    values = {
        "frame_residual_calls": True,
        "frame_residual_qwen_q1_only": 1,
        "frame_residual_rms": float("nan"),
        "frame_residual_learned_scale": "0.05",
        "frame_residual_live_first_last_hidden_cosine": 1.5,
    }

    assert _filter_frame_residual_runtime_metrics(values) == {}


def test_native_thinker_lock_metrics_report_lock_topology_and_reject_bad_values() -> None:
    assert _filter_native_thinker_lock_metrics({
        "thinker_decode_lock_wait_ms": 0.0,
        "thinker_decode_lock_acquire_count": 0,
        "thinker_decode_lock_wait_provenance": "native_no_python_decode_lock",
        "untrusted_extra": 9,
    }) == {
        "thinker_decode_lock_wait_ms": 0.0,
        "thinker_decode_lock_acquire_count": 0,
        "thinker_decode_lock_wait_provenance": "native_no_python_decode_lock",
    }
    assert _filter_native_thinker_lock_metrics({
        "thinker_decode_lock_wait_ms": float("nan"),
        "thinker_decode_lock_acquire_count": True,
        "thinker_decode_lock_wait_provenance": "measured_from_total_turn_ms",
    }) == {}


def test_turn_local_speech_metrics_keep_bounded_fifo_and_pacer_evidence() -> None:
    values = {
        "turn_local_fifo_capacity": 4,
        "turn_local_fifo_remaining_packets": 0,
        "turn_local_fifo_peak_packets": 4,
        "turn_local_last_vocalized_kv_position": 278,
        "turn_local_vocalized_cursor_is_playback_ack": False,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 2,
        },
        "realtime_pcm_packet_queue_capacity": 4,
        "native_q1_import_audio_frame_capacity": 512,
        "native_q1_live_frame_cap": 512,
        "untrusted_extra": "drop me",
    }

    assert _filter_turn_local_speech_runtime_metrics(values) == {
        key: value for key, value in values.items() if key != "untrusted_extra"
    }


def test_turn_local_speech_metric_filter_rejects_wrong_types_and_over_cap() -> None:
    assert _filter_turn_local_speech_runtime_metrics({
        "turn_local_fifo_peak_packets": True,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 3,
        },
    }) == {}


def test_turn_local_speech_summary_formats_runtime_values_without_external_scope() -> None:
    line = _format_turn_local_speech_summary("session123", {
        "turn_local_fifo_capacity": 4,
        "turn_local_fifo_peak_packets": 4,
        "turn_local_fifo_remaining_packets": 0,
        "turn_local_last_vocalized_kv_position": 278,
        "live_gemma_token_pacer": {
            "tick_ms": 80,
            "max_tokens_per_tick": 2,
            "total_tokens": 34,
            "peak_tokens_in_tick": 2,
        },
        "realtime_pcm_packet_queue_capacity": 4,
        "frame_residual_calls": 23,
        "frame_residual_tail_hold_calls": 85,
        "native_q1_live_frame_cap": 512,
        "frame_residual_live_hidden_trace": [{"qwen_frame_step": 0}],
        "frame_residual_live_first_last_hidden_cosine": 0.51,
    })
    assert "session=session123" in line
    assert "fifo_peak=4" in line
    assert "pacer_tokens=34" in line
    assert "hidden_trace_entries=1" in line
    assert "first_last_hidden_cosine=0.51" in line


def test_packet_event_is_recorded_after_owned_clone_and_propagated_to_live_hidden(monkeypatch) -> None:
    order = []
    producer_stream = object()

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing is False

        def record(self, stream):
            order.append(("record", stream))

    class FakeCudaStream:
        def wait_event(self, event):
            order.append(("producer_wait", event))

    class FakeCudaHidden:
        shape = (1, 1536)
        device = torch.device("cuda:0")

        def detach(self):
            order.append("detach")
            return self

        def clone(self):
            order.append("clone")
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *, device=None: producer_stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    packet = SpokenTokenPacket(
        token_id=7,
        text_piece="a",
        gemma_hidden_1536=FakeCudaHidden(),
        exclusive_kv_cursor=9,
    )

    assert order == ["detach", "clone", ("record", producer_stream)]
    assert packet.producer_event is not None

    explicit_event = object()
    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=1)
    stream = TurnLocalQwenSayStream(
        qwen_tokenizer=_CharacterTokenizer(),
        text_eos_token_id=900,
        fifo=fifo,
        generation_id=1,
    )
    stream.feed_packet(SpokenTokenPacket(
        token_id=8,
        text_piece="b",
        gemma_hidden_1536=torch.ones((1, 1536)),
        exclusive_kv_cursor=10,
        producer_event=explicit_event,
    ))
    stream.finish(final_text="b")
    stream.take_initial_token(timeout_s=0)
    assert stream.hidden_for_frame(0).producer_event is explicit_event


def test_controller_guided_published_hidden_records_producer_event_before_visibility(monkeypatch) -> None:
    order = []
    producer_stream = object()

    class FakeEvent:
        def __init__(self, *, enable_timing):
            assert enable_timing is False

        def record(self, stream):
            order.append(("record", stream))

    class FakeCudaHidden(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

        def detach(self):
            return self

        def clone(self):
            order.append("clone")
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *, device=None: producer_stream)
    monkeypatch.setattr(torch.cuda, "Event", FakeEvent)
    hidden = FakeCudaHidden._make_subclass(
        FakeCudaHidden,
        torch.zeros((1, 1, 2)),
        require_grad=False,
    )
    stream = LiveQwenSayStream(
        qwen_tokenizer=SimpleNamespace(added_tokens_decoder={}),
        qwen_processor=object(),
        text_eos_token_id=900,
    )

    stream._publish_token(7, (0, 1), hidden, initial=True)
    published = stream.hidden_for_frame(0)

    assert order == ["clone", ("record", producer_stream)]
    assert published.producer_event is stream._published_events[0]


def test_cuda_hidden_consumer_waits_on_event_and_records_storage_without_cpu_sync(monkeypatch) -> None:
    order = []
    consumer_stream = object()
    producer_event = object()

    class FakeCudaStream:
        def wait_event(self, event):
            order.append(("wait_event", event))

    class FakeCudaTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda:0")

        def record_stream(self, stream):
            order.append(("record_stream", stream))

    consumer_stream = FakeCudaStream()
    tensor = FakeCudaTensor._make_subclass(
        FakeCudaTensor,
        torch.zeros((1, 1, 1536)),
        require_grad=False,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *, device=None: consumer_stream)

    engine_module._wait_for_cuda_producer_event(tensor, producer_event)

    assert order == [
        ("wait_event", producer_event),
        ("record_stream", consumer_stream),
    ]

    order.clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda **_kwargs: pytest.fail("CPU fallback must not touch CUDA streams"),
    )
    engine_module._wait_for_cuda_producer_event(torch.zeros((1, 1, 1536)), producer_event)
    assert order == []


class _TinyQwenModel:
    def __call__(self, *, inputs_embeds, past_key_values=None, use_cache):
        assert use_cache is True
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=object() if past_key_values is None else past_key_values,
        )


class _InPlaceFrameAdapter(torch.nn.Module):
    architecture_id = "response_hidden_q1_query_film_v2"
    preserve_frame0 = True

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))

    def forward(self, response_hidden: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        amount = self.scale * torch.sqrt(response_hidden.float().square().mean())
        return query.add_(amount.to(dtype=query.dtype))


class _CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return [ord(char) for char in text]

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(chr(token_id) for token_id in ids)


def test_live_fifo_q1_callback_measures_pre_adapter_base_for_in_place_adapter(monkeypatch) -> None:
    """Exercise talk's real Q1 callback with token-current FIFO hiddens and tail hold."""
    hidden_size = 2048

    class TinyTalker:
        model = _TinyQwenModel()

        def codec_head(self, hidden):
            return torch.zeros((*hidden.shape[:-1], 3072), dtype=hidden.dtype)

        def get_input_embeddings(self):
            return lambda token: torch.zeros((1, 1, hidden_size), dtype=torch.float32)

        def get_text_embeddings(self):
            return lambda token: torch.zeros((1, 1, hidden_size), dtype=torch.float32)

        def text_projection(self, hidden):
            return hidden

    class TinyTalkerBundle:
        talker = TinyTalker()
        tcfg = SimpleNamespace(codec_eos_token_id=2150)

        def decode(self, codes, *, preclamp_callback=None):
            del preclamp_callback
            count = len(codes)
            return np.zeros((count * 1920,), dtype=np.float32)

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = TinyTalkerBundle()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.allow_talker_eos = False
    engine._stop_event = False
    import threading
    engine._talker_lock = threading.RLock()
    engine._thinker_lock = threading.Lock()
    engine.subtalker_graph = SimpleNamespace(
        generate=lambda *_args, **_kwargs: torch.zeros((15,), dtype=torch.long)
    )
    engine.frame_residual_adapter = _InPlaceFrameAdapter()
    engine.frame_residual_adapter_resolved_path = "test-adapter"
    engine._postprocess = lambda wav: wav
    engine._cb0_sample = lambda *_args, **_kwargs: 42

    prefill = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    monkeypatch.setattr(
        engine_module,
        "_build_talker_prefill_parts",
        lambda *_args, **_kwargs: (prefill, prefill, prefill, prefill),
    )
    monkeypatch.setattr(
        engine_module,
        "build_qwen_audio_conditioning",
        lambda *_args, **_kwargs: torch.ones((1, 1, hidden_size), dtype=torch.float32),
    )

    fifo = TurnLocalSpeechFIFO()
    fifo.begin_turn(generation_id=3)
    live_stream = TurnLocalQwenSayStream(
        qwen_tokenizer=_CharacterTokenizer(),
        text_eos_token_id=900,
        fifo=fifo,
        generation_id=3,
    )
    packet_hiddens = []
    for token_id, text, cursor, half in (
        (100, "a", 10, slice(0, 768)),
        (101, "b", 11, slice(0, 768)),
        (102, "c", 12, slice(768, 1536)),
    ):
        hidden = torch.zeros((1, 1536), dtype=torch.float32)
        hidden[:, half] = 1.0
        packet_hiddens.append(hidden.clone())
        live_stream.feed_packet(SpokenTokenPacket(
            token_id=token_id,
            text_piece=text,
            gemma_hidden_1536=hidden,
            exclusive_kv_cursor=cursor,
        ))
    live_stream.finish(final_text="abc")
    initial = live_stream.take_initial_token(timeout_s=0)

    q1_inputs = []
    original_q1 = engine._qwen_q1_forward

    def observe_q1(model, inputs_embeds, past_key_values, *, metrics=None):
        q1_inputs.append(inputs_embeds.detach().clone())
        return original_q1(model, inputs_embeds, past_key_values, metrics=metrics)

    engine._qwen_q1_forward = observe_q1
    metrics: dict[str, object] = {}
    talk_errors: list[BaseException] = []

    def run_talker_while_thinker_lock_is_held() -> None:
        try:
            engine.talk(
                "abc",
                torch.empty((1, 0, hidden_size), dtype=torch.float32),
                n_words=1,
                min_frames=2,
                max_frames=5,
                raw_text_token_ids=[initial.token_id],
                include_text_eos=False,
                live_qwen_text_stream=live_stream,
                live_frame_residual_hidden_stream=live_stream,
                metrics=metrics,
            )
        except BaseException as exc:
            talk_errors.append(exc)

    engine._thinker_lock.acquire()
    talker_thread = threading.Thread(target=run_talker_while_thinker_lock_is_held)
    talker_thread.start()
    talker_thread.join(timeout=2.0)
    engine._thinker_lock.release()
    if talker_thread.is_alive():
        talker_thread.join(timeout=2.0)
    assert not talker_thread.is_alive(), "Talker Q1 must progress while Thinker lock is held"
    assert talk_errors == []

    assert len(q1_inputs) >= 3, "expected actual iterative Q1 forwards after live FIFO callbacks"
    assert metrics["frame_residual_q1_steps"] == len(q1_inputs)
    assert metrics["frame_residual_calls"] == metrics["frame_residual_q1_steps"] - 1
    assert metrics["frame_residual_calls"] >= 2
    assert metrics["frame_residual_tail_hold_calls"] >= 1
    trace = metrics["frame_residual_live_hidden_trace"]
    assert [entry["gemma_token_id"] for entry in trace[:4]] == [100, 101, 102, 102]
    assert trace[0]["tail_hold"] is False
    assert trace[3]["tail_hold"] is True
    assert metrics["frame_residual_live_first_last_hidden_cosine"] == pytest.approx(0.0, abs=1e-5)
    assert metrics["frame_residual_rms"] > 0.001
    assert metrics["frame_residual_delta_sq_sum"] > 0.0
    assert metrics["frame_residual_base_sq_sum"] > 0.0
    assert 0.01 <= metrics["frame_residual_rms_to_base_ratio"] <= 0.25

    exposed = _filter_frame_residual_runtime_metrics(metrics)
    assert exposed["frame_residual_delta_sq_sum"] == metrics["frame_residual_delta_sq_sum"]
    assert exposed["frame_residual_base_sq_sum"] == metrics["frame_residual_base_sq_sum"]


def test_controller_guided_live_qwen_stream_q1_traces_hidden_without_gemma_cursor(monkeypatch) -> None:
    """Run 13 real live Qwen spans through Talker's frame-indexed Q1 adapter."""
    hidden_size = 2048
    token_surfaces = (
        "we", " can", " map", " each", " spoken", " word", " to",
        " frames", " and", " keep", " audio", " in", " sync",
    )
    sentence = "".join(token_surfaces)

    class TinyTokenizer:
        added_tokens_decoder = {}
        prefix = "<|im_start|>assistant\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n"

        def __init__(self):
            self.pieces = dict(enumerate(token_surfaces, start=100))

            def split(text):
                spans = []
                cursor = 0
                while cursor < len(text):
                    match = next(
                        (piece for piece in self.pieces.values() if text.startswith(piece, cursor)),
                        None,
                    )
                    assert match is not None, (text, cursor)
                    spans.append((match, (cursor, cursor + len(match))))
                    cursor += len(match)
                return spans

            self.backend_tokenizer = SimpleNamespace(pre_tokenizer=SimpleNamespace(pre_tokenize_str=split))

        def __call__(self, text, *, return_offsets_mapping=False):
            assert return_offsets_mapping is True
            assert text.startswith(self.prefix) and text.endswith(self.suffix)
            raw = text[len(self.prefix):-len(self.suffix)]
            body = []
            cursor = 0
            while cursor < len(raw):
                match = next(
                    ((token_id, piece) for token_id, piece in self.pieces.items()
                     if raw.startswith(piece, cursor)),
                    None,
                )
                assert match is not None, (raw, cursor)
                token_id, piece = match
                body.append((token_id, cursor, cursor + len(piece)))
                cursor += len(piece)
            ids = [0, 1, 2] + [token_id for token_id, _start, _end in body] + [900, 901, 902, 903, 904]
            offsets = [(0, 0)] * 3 + [
                (len(self.prefix) + start, len(self.prefix) + end)
                for _token_id, start, end in body
            ] + [(0, 0)] * 5
            return {"input_ids": ids, "offset_mapping": offsets}

        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is False
            assert clean_up_tokenization_spaces is False
            return "".join(self.pieces[int(token_id)] for token_id in ids)

    class TinyProcessor:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, *, text, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([self.tokenizer(text, return_offsets_mapping=True)["input_ids"]])}

    class TinyTalker:
        model = _TinyQwenModel()

        def codec_head(self, hidden):
            return torch.zeros((*hidden.shape[:-1], 3072), dtype=hidden.dtype)

        def get_input_embeddings(self):
            return lambda token: torch.zeros((1, 1, hidden_size), dtype=torch.float32)

        def get_text_embeddings(self):
            return lambda token: torch.zeros((1, 1, hidden_size), dtype=torch.float32)

        def text_projection(self, hidden):
            return hidden

    class TinyTalkerBundle:
        talker = TinyTalker()
        tcfg = SimpleNamespace(codec_eos_token_id=2150)

        def decode(self, codes, *, preclamp_callback=None):
            del preclamp_callback
            return np.zeros((len(codes) * 1920,), dtype=np.float32)

    engine = PipelineEngine.__new__(PipelineEngine)
    engine.talker = TinyTalkerBundle()
    engine.device = "cpu"
    engine.dtype = torch.float32
    engine.allow_talker_eos = False
    engine._stop_event = False
    import threading
    engine._talker_lock = threading.RLock()
    engine._thinker_lock = threading.Lock()
    engine.subtalker_graph = SimpleNamespace(
        generate=lambda *_args, **_kwargs: torch.zeros((15,), dtype=torch.long)
    )
    engine.frame_residual_adapter = _InPlaceFrameAdapter()
    engine.frame_residual_adapter_resolved_path = "test-adapter"
    engine._postprocess = lambda wav: wav
    engine._cb0_sample = lambda *_args, **_kwargs: 42

    prefill = torch.zeros((1, 1, hidden_size), dtype=torch.float32)
    monkeypatch.setattr(
        engine_module,
        "_build_talker_prefill_parts",
        lambda *_args, **_kwargs: (prefill, prefill, prefill, prefill),
    )
    monkeypatch.setattr(
        engine_module,
        "build_qwen_audio_conditioning",
        lambda *_args, **_kwargs: torch.ones((1, 1, hidden_size), dtype=torch.float32),
    )

    tokenizer = TinyTokenizer()
    stream = LiveQwenSayStream(
        qwen_tokenizer=tokenizer,
        qwen_processor=TinyProcessor(tokenizer),
        text_eos_token_id=905,
    )
    for token_index, token_surface in enumerate(token_surfaces):
        hidden = torch.zeros((1, 1, 1536), dtype=torch.float32)
        hidden[0, 0, (token_index % 3) * 512:((token_index % 3) + 1) * 512] = 1.0
        stream.feed_piece(token_surface, hidden)
    stream.finish(final_text=sentence)
    initial = stream.take_initial_token(timeout_s=0)
    assert len(stream.span_cursor.frame_spans) == 13
    assert stream.span_cursor.frame_spans[-1].end_frame == 41

    q1_inputs = []
    original_q1 = engine._qwen_q1_forward

    def observe_q1(model, inputs_embeds, past_key_values, *, metrics=None):
        q1_inputs.append(inputs_embeds.detach().clone())
        return original_q1(model, inputs_embeds, past_key_values, metrics=metrics)

    engine._qwen_q1_forward = observe_q1
    metrics: dict[str, object] = {}
    engine.talk(
        sentence,
        torch.empty((1, 0, hidden_size), dtype=torch.float32),
        n_words=13,
        min_frames=2,
        max_frames=43,
        raw_text_token_ids=[initial.token_id],
        include_text_eos=False,
        live_qwen_text_stream=stream,
        live_frame_residual_hidden_stream=stream,
        metrics=metrics,
    )

    assert len(q1_inputs) >= 42
    trace = metrics["frame_residual_live_hidden_trace"]
    assert len({entry["qwen_token_id"] for entry in trace}) == 13
    assert all(entry["gemma_token_id"] is None for entry in trace)
    assert metrics["frame_residual_q1_steps"] == 42
    assert metrics["frame_residual_tail_hold_calls"] == 1
    assert metrics["frame_residual_tail_hold_calls"] / metrics["frame_residual_q1_steps"] < 0.25
    assert metrics["frame_residual_rms"] > 0.001
    assert 0.01 <= metrics["frame_residual_rms_to_base_ratio"] <= 0.25
    assert len(_filter_frame_residual_runtime_metrics(metrics)["frame_residual_live_hidden_trace"]) >= 13
