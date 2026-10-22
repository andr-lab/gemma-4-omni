from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.qwen_conditioning import (
    build_qwen_audio_conditioning,
    build_qwen_audio_conditioning_sequence,
)
from duplex_kernel.talker_state import TalkerState


HIDDEN = 4


class _Embedding:
    def __init__(self, offset: float) -> None:
        self.offset = float(offset)

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        values = ids.to(torch.float32) + self.offset
        return values.unsqueeze(-1).expand(*ids.shape, HIDDEN)


class _RecordingModel:
    def __init__(self) -> None:
        self.inputs: list[torch.Tensor] = []

    def __call__(self, *, inputs_embeds, attention_mask, position_ids, use_cache):
        del attention_mask, position_ids, use_cache
        self.inputs.append(inputs_embeds.detach().clone())
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class _FakeTalker:
    def __init__(self) -> None:
        self.model = _RecordingModel()
        self.talker = SimpleNamespace(
            model=self.model,
            codec_head=lambda hidden: hidden,
            get_input_embeddings=lambda: _Embedding(0.0),
            code_predictor=SimpleNamespace(
                get_input_embeddings=lambda: [_Embedding(float(index + 1)) for index in range(15)]
            ),
        )

    def build_prefill_parts(self, text: str):
        assert text
        zero = torch.zeros((1, 1, HIDDEN), dtype=torch.float32)
        tts_pad = torch.full_like(zero, 7.0)
        return zero, zero.clone(), zero.clone(), tts_pad


def test_shared_conditioning_preserves_explicit_trailing_text_then_falls_back_to_pad() -> None:
    fake = _FakeTalker()
    codes = torch.arange(1, 17, dtype=torch.long).view(1, 1, 16)
    pad = torch.full((1, 1, HIDDEN), 7.0)
    trailing = torch.stack(
        (
            torch.full((HIDDEN,), 11.0),
            torch.full((HIDDEN,), 13.0),
        )
    ).view(1, 2, HIDDEN)

    code_sum = sum(float(index + 1 + (0.0 if index == 0 else index)) for index in range(16))
    for step, text_value in ((0, 11.0), (1, 13.0)):
        actual = build_qwen_audio_conditioning(
            fake.talker,
            codes,
            tts_pad_embed=pad,
            trailing_text_hidden=trailing,
            generation_step=step,
        )
        assert torch.equal(actual, torch.full((1, 1, HIDDEN), code_sum + text_value))

    exhausted = build_qwen_audio_conditioning(
        fake.talker,
        codes,
        tts_pad_embed=pad,
        trailing_text_hidden=trailing,
        generation_step=2,
    )
    assert torch.equal(exhausted, torch.full((1, 1, HIDDEN), code_sum + 7.0))


def test_conditioning_sequence_advances_trailing_text_per_frame() -> None:
    fake = _FakeTalker()
    codes = torch.arange(1, 33, dtype=torch.long).view(1, 2, 16)
    trailing = torch.stack(
        (
            torch.full((HIDDEN,), 11.0),
            torch.full((HIDDEN,), 13.0),
        )
    ).view(1, 2, HIDDEN)
    actual = build_qwen_audio_conditioning_sequence(
        fake.talker,
        codes,
        tts_pad_embed=torch.full((1, 1, HIDDEN), 7.0),
        trailing_text_hidden=trailing,
    )
    assert actual.shape == (1, 2, HIDDEN)
    assert torch.equal(actual[:, 0], torch.full((1, HIDDEN), 267.0))
    assert torch.equal(actual[:, 1], torch.full((1, HIDDEN), 525.0))
    shorthand = build_qwen_audio_conditioning_sequence(
        fake.talker,
        codes[:, :1],
        tts_pad_embed=torch.full((1, 1, HIDDEN), 7.0),
        trailing_text_hidden=trailing.squeeze(0),
    )
    assert torch.equal(shorthand, actual[:, :1])


def test_audio_conditioning_rejects_partial_rvq_codebooks() -> None:
    fake = _FakeTalker()
    with pytest.raises(ValueError, match="tutti i 16 codebook"):
        build_qwen_audio_conditioning(
            fake.talker,
            torch.zeros((1, 1, 1), dtype=torch.long),
            tts_pad_embed=torch.zeros((1, 1, HIDDEN)),
        )


def test_audio_conditioning_requires_pad_after_prefill() -> None:
    fake = _FakeTalker()
    with pytest.raises(ValueError, match="tts_pad_embed"):
        build_qwen_audio_conditioning(
            fake.talker,
            torch.zeros((1, 1, 16), dtype=torch.long),
        )


def test_future_prefix_uses_trailing_text_before_pad() -> None:
    fake = _FakeTalker()
    speaker = FuturePrefixQwen(
        fake,
        lookahead_frames=0,
        max_prefix_frames=64,
        max_audio_frames=2,
        device="cpu",
        dtype=torch.float32,
    )
    trailing = torch.full((1, 1, HIDDEN), 11.0)
    speaker.begin("ciao", trailing_text_hidden=trailing)
    speaker.push_v9_frame(0, torch.zeros(HIDDEN))
    speaker.push_v9_frame(1, torch.zeros(HIDDEN))
    speaker.step_frame()
    speaker.commit_codes(torch.arange(1, 17))
    speaker.step_frame()

    previous_codes = torch.arange(1, 17, dtype=torch.float32)
    code_sum = sum(previous_codes[index] + (0.0 if index == 0 else float(index)) for index in range(16))
    assert torch.allclose(
        fake.model.inputs[-1][:, speaker._audio_start + 1, :],
        torch.full((1, HIDDEN), code_sum + 11.0),
    )
    speaker.close()


def test_second_audio_decoder_input_includes_upstream_tts_pad() -> None:
    fake = _FakeTalker()
    speaker = FuturePrefixQwen(
        fake,
        lookahead_frames=0,
        max_prefix_frames=64,
        max_audio_frames=2,
        device="cpu",
        dtype=torch.float32,
    )
    state = TalkerState()
    speaker.open_segment("ciao", state, segment_id="seg", generation_id=1)
    speaker.push_v9_frame(0, torch.zeros(HIDDEN))
    speaker.push_v9_frame(1, torch.zeros(HIDDEN))

    first = speaker.step_frame()
    speaker.commit_codes(torch.arange(1, 17))
    speaker.step_frame()

    # The first step consumes BOS. The second step consumes the previous
    # 16-codebook audio embedding plus upstream tts_pad (Qwen source line
    # 1692 when all text is already in prefill).
    second_call = fake.model.inputs[-1]
    audio_position = speaker._audio_start + 1
    previous_codes = torch.arange(1, 17, dtype=torch.float32)
    expected_audio = sum(previous_codes[index] + (0.0 if index == 0 else float(index)) for index in range(16)) + 7.0
    assert first.last_hidden.shape == (1, 1, HIDDEN)
    assert torch.allclose(
        second_call[:, audio_position, :],
        torch.full((1, HIDDEN), expected_audio),
    ), "tts_pad manca dall'input del decoder dal secondo frame"
