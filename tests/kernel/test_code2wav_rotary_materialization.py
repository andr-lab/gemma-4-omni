from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from duplex_kernel.models.talker import (
    _install_shims,
    _rebuild_code2wav_rotary_buffers,
)
from duplex_kernel.speaker_runtime import Code2WavWindowGraph


class Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(nn.Module):
    def __init__(self, rope_init_fn):
        super().__init__()
        self.config = SimpleNamespace(
            rope_theta=10000.0,
            head_dim=64,
            hidden_size=512,
            num_attention_heads=8,
            partial_rotary_factor=1.0,
        )
        self.rope_type = "default"
        self.rope_init_fn = rope_init_fn
        with torch.device("meta"):
            inv_freq, self.attention_scaling = rope_init_fn(self.config)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


class _Decoder(nn.Module):
    def __init__(self, rope_init_fn):
        super().__init__()
        self.rotary_emb = Qwen3TTSTokenizerV2DecoderRotatoryEmbedding(rope_init_fn)


class _Code2Wav(nn.Module):
    def __init__(self, rope_init_fn):
        super().__init__()
        self.decoder = _Decoder(rope_init_fn)


def test_meta_materialized_qwen_rope_buffers_are_rebuilt_from_config():
    repo_root = Path(__file__).resolve().parents[2]
    _install_shims(str(repo_root))

    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
    model = _Code2Wav(rope_init_fn)
    rotary = model.decoder.rotary_emb

    # Match HF's meta-to-device materialization: allocate non-persistent
    # buffers without loading them from the checkpoint. Poisoning makes the
    # uninitialized state deterministic for this CPU regression test.
    rotary.register_buffer(
        "inv_freq",
        torch.empty((32,), dtype=torch.float32).fill_(float("nan")),
        persistent=False,
    )
    assert rotary.original_inv_freq.is_meta
    assert not torch.isfinite(rotary.inv_freq).all()

    expected, expected_scaling = rope_init_fn(
        rotary.config,
        device=torch.device("cpu"),
    )
    count = _rebuild_code2wav_rotary_buffers(model)

    assert count == 1
    assert torch.equal(rotary.inv_freq, expected)
    assert rotary.original_inv_freq is rotary.inv_freq
    assert rotary.attention_scaling == expected_scaling
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.inv_freq.shape == (32,)
    assert torch.isfinite(rotary.inv_freq).all()
    assert float(rotary.inv_freq.max()) <= 1.0 + 1e-6
    assert "inv_freq" in rotary._non_persistent_buffers_set


def test_code2wav_graph_bridges_caller_and_capture_streams(monkeypatch):
    timeline = []
    codes = torch.ones((1, 2, 16), dtype=torch.long)
    static_codes = torch.zeros_like(codes)

    class _Stream:
        def __init__(self, handle, label):
            self.cuda_stream = handle
            self.device = torch.device("cpu")
            self.label = label

        def wait_stream(self, stream):
            assert stream is caller_stream
            assert torch.equal(static_codes, codes)
            timeline.append(("wait_stream", self.label, stream.label))

        def wait_event(self, event):
            assert event is completion_event
            timeline.append(("wait_event", self.label))

    class _Event:
        def record(self, stream):
            assert stream is capture_stream
            timeline.append(("record_event", stream.label))

    class _Graph:
        def replay(self):
            assert torch.equal(static_codes, codes)
            timeline.append(("replay",))

    class _Audio:
        def __getitem__(self, _slice):
            timeline.append(("slice_audio",))
            return self

        def clone(self):
            timeline.append(("clone_audio",))
            return "audio"

    caller_stream = _Stream(11, "caller")
    capture_stream = _Stream(22, "capture")
    completion_event = _Event()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller_stream)
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **_kwargs: completion_event,
    )

    graph = Code2WavWindowGraph.__new__(Code2WavWindowGraph)
    graph.captured = True
    graph.graph = _Graph()
    graph.static_codes = static_codes
    graph.static_audio = _Audio()
    graph.capture_stream = capture_stream
    graph.window_frames = 2
    graph.output_crop_frames = 2
    graph.frame_samples = graph.FRAME_SAMPLES
    graph.device = torch.device("cpu")
    graph.replays = 0

    assert graph.decode(codes) == "audio"
    assert timeline == [
        ("wait_stream", "capture", "caller"),
        ("replay",),
        ("record_event", "capture"),
        ("wait_event", "caller"),
        ("slice_audio",),
        ("clone_audio",),
    ]
    assert graph.replays == 1


def test_code2wav_graph_failure_still_bridges_capture_completion(monkeypatch):
    timeline = []
    codes = torch.ones((1, 2, 16), dtype=torch.long)
    caller_stream = SimpleNamespace(cuda_stream=11, device=torch.device("cpu"))
    capture_stream = SimpleNamespace(cuda_stream=22, device=torch.device("cpu"))

    def wait_stream(stream):
        assert stream is caller_stream
        timeline.append("wait_stream")

    capture_stream.wait_stream = wait_stream
    completion_event = SimpleNamespace()

    def record(stream):
        assert stream is capture_stream
        timeline.append("record_event")

    completion_event.record = record

    def wait_event(event):
        assert event is completion_event
        timeline.append("wait_event")

    caller_stream.wait_event = wait_event

    class _Graph:
        def replay(self):
            timeline.append("replay")
            raise RuntimeError("graph replay failed")

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda **_kwargs: completion_event)
    graph = Code2WavWindowGraph.__new__(Code2WavWindowGraph)
    graph.captured = True
    graph.graph = _Graph()
    graph.static_codes = torch.zeros_like(codes)
    graph.static_audio = torch.empty(1)
    graph.capture_stream = capture_stream
    graph.window_frames = 2
    graph.output_crop_frames = 2
    graph.frame_samples = graph.FRAME_SAMPLES
    graph.device = torch.device("cpu")
    graph.replays = 0

    with pytest.raises(RuntimeError, match="graph replay failed"):
        graph.decode(codes)
    assert timeline == ["wait_stream", "replay", "record_event", "wait_event"]
    assert graph.replays == 0
