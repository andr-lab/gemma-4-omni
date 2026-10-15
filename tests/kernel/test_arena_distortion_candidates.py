from types import SimpleNamespace

import pytest
import torch
from torch import nn

from duplex_kernel.frame_residual_adapter import (
    NormPreservingQueryFiLMAdapter,
    project_norm_preserving_query,
)
from duplex_kernel.models.talker import TalkerModel, rescale_code2wav_preclamp
from duplex_kernel.subtalker_graph import (
    BASELINE_SAMPLING_PROFILE,
    HRCR_SAMPLING_PROFILE,
    residual_sampling_parameters,
)


class _ScaleBlock(nn.Module):
    def forward(self, value):
        return value * 2.0


class _TokenizerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList([_ScaleBlock()])

    def forward(self, value):
        waveform = value
        for block in self.decoder:
            waveform = block(waveform)
        return waveform.clamp(min=-1.0, max=1.0)


class _TokenizerModel:
    def __init__(self):
        self.decoder = _TokenizerDecoder()

    def decode(self, codes):
        return SimpleNamespace(audio_values=[self.decoder(codes)])


def _talker_with_fake_qwen_decoder():
    talker = object.__new__(TalkerModel)
    talker.code2wav = _TokenizerModel()
    return talker


def test_code2wav_preclamp_hook_observes_and_can_replace_real_final_block_output():
    talker = _talker_with_fake_qwen_decoder()
    observed = []
    codes = torch.tensor([[0.2, 0.6]], dtype=torch.float32)

    def rescale(value):
        observed.append(value.detach().clone())
        return rescale_code2wav_preclamp(value)

    waveform = talker.decode(codes, preclamp_callback=rescale)

    assert len(observed) == 1
    assert float(observed[0].abs().max()) == pytest.approx(1.2, abs=1e-6)
    assert float(waveform.abs().max()) == pytest.approx(0.95, abs=1e-6)
    # The temporary hook must be gone after decode.
    assert len(talker.code2wav.decoder.decoder[-1]._forward_hooks) == 0


def test_code2wav_observation_callback_none_preserves_official_clamp_behavior():
    talker = _talker_with_fake_qwen_decoder()
    observed = []
    waveform = talker.decode(
        torch.tensor([[0.2, 0.6]], dtype=torch.float32),
        preclamp_callback=lambda value: observed.append(value.detach().clone()),
    )

    assert float(observed[0].abs().max()) == pytest.approx(1.2, abs=1e-6)
    assert float(waveform.abs().max()) == pytest.approx(1.0, abs=1e-6)
    assert len(talker.code2wav.decoder.decoder[-1]._forward_hooks) == 0


def test_code2wav_preclamp_hook_is_removed_when_capture_callback_raises():
    talker = _talker_with_fake_qwen_decoder()

    def fail_capture(_value):
        raise RuntimeError("capture failure")

    with pytest.raises(RuntimeError, match="capture failure"):
        talker.decode(torch.tensor([[0.2]], dtype=torch.float32), preclamp_callback=fail_capture)

    assert len(talker.code2wav.decoder.decoder[-1]._forward_hooks) == 0


def test_hrcr_sampling_is_codebook_indexed_and_baseline_profile_is_unchanged():
    baseline = (0.80, 20, 0.90)
    assert residual_sampling_parameters(
        1, profile=BASELINE_SAMPLING_PROFILE, baseline=baseline
    ) == baseline
    assert residual_sampling_parameters(
        15, profile=BASELINE_SAMPLING_PROFILE, baseline=baseline
    ) == baseline
    for codebook_index in (1, 2, 3):
        assert residual_sampling_parameters(
            codebook_index, profile=HRCR_SAMPLING_PROFILE, baseline=baseline
        ) == (0.35, 8, 0.85)
    for codebook_index in range(4, 16):
        assert residual_sampling_parameters(
            codebook_index, profile=HRCR_SAMPLING_PROFILE, baseline=baseline
        ) == (0.15, 3, 0.75)
    with pytest.raises(ValueError, match=r"\[1, 15\]"):
        residual_sampling_parameters(0, profile=HRCR_SAMPLING_PROFILE, baseline=baseline)
    with pytest.raises(ValueError, match="integer"):
        residual_sampling_parameters(1.5, profile=HRCR_SAMPLING_PROFILE, baseline=baseline)
    with pytest.raises(ValueError):
        residual_sampling_parameters(True, profile=HRCR_SAMPLING_PROFILE, baseline=baseline)


def test_subtalker_constructor_preserves_reference_sampler_defaults():
    class _Model:
        config = SimpleNamespace(hidden_size=2, vocab_size=4)

        def get_input_embeddings(self):
            return None

    from duplex_kernel.subtalker_graph import SubtalkerGraph

    graph = SubtalkerGraph(_Model(), device="cpu")

    assert (graph.temperature, graph.top_k, graph.top_p) == (0.80, 20, 0.90)
    assert graph.sampling_profile == BASELINE_SAMPLING_PROFILE


def test_np_film_projection_preserves_query_norm_and_candidate_is_opt_in():
    query = torch.tensor([[[3.0, 4.0]]])
    delta = torch.tensor([[[1.0, 0.0]]])
    projected = project_norm_preserving_query(query, delta)

    assert torch.linalg.vector_norm(projected, dim=-1).item() == pytest.approx(5.0)
    assert not torch.equal(projected, query)
    assert NormPreservingQueryFiLMAdapter.preserve_frame0 is True
    assert NormPreservingQueryFiLMAdapter.architecture_id.endswith("_np_candidate")


def test_har_headroom_rescaling_is_scalar_and_bounded():
    waveform = torch.tensor([[0.0, 0.5, -2.0, 1.0]], dtype=torch.float32)
    rescaled = rescale_code2wav_preclamp(waveform)

    assert float(rescaled.abs().max()) == pytest.approx(0.95, abs=1e-6)
    assert float(rescaled[0, 1] / waveform[0, 1]) == pytest.approx(0.475, abs=1e-6)
