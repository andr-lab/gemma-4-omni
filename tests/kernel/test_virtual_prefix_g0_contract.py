from types import SimpleNamespace

import torch

from duplex_kernel.future_prefix import FuturePrefixQwen


def _mask_speaker(mode: str, log_gate: float = -8.0) -> FuturePrefixQwen:
    speaker = object.__new__(FuturePrefixQwen)
    speaker._slot_cache = SimpleNamespace(max_cache_len=12)
    speaker._icl = torch.zeros((1, 1, 4), dtype=torch.float32)
    speaker._keys_arange = None
    speaker.device = torch.device("cpu")
    speaker.dtype = torch.float32
    speaker._prefix_start = 2
    speaker._audio_start = 8
    speaker.virtual_prefix_mode = mode
    speaker.prefix_key_log_gate = log_gate
    return speaker


def test_g0_masks_all_virtual_prefix_keys_from_audio_query_only():
    speaker = _mask_speaker("g0")
    positions = torch.tensor([2, 8], dtype=torch.long)
    mask = speaker._slot_mask(positions, visible_prefix_end=0)
    floor = torch.finfo(torch.float32).min

    # Prefix query can build its own causal KV row. Audio sees stable text and
    # its current audio position, but has no path to the virtual prefix slots.
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 1, 0].item() == 0.0
    assert mask[0, 0, 1, 2].item() == floor
    assert mask[0, 0, 1, 8].item() == 0.0


def test_gated_mode_uses_visible_prefix_count_normalization():
    speaker = _mask_speaker("gated", log_gate=-8.0)
    positions = torch.tensor([4, 10], dtype=torch.long)
    # prefix_start=2; frame 2 is visible, so there are exactly three keys.
    mask = speaker._slot_mask(positions, visible_prefix_end=2)
    expected = -8.0 - torch.log(torch.tensor(3.0)).item()
    assert abs(mask[0, 0, 1, 4].item() - expected) < 1e-6
    assert mask[0, 0, 1, 5].item() == torch.finfo(torch.float32).min


def test_virtual_prefix_keeps_audio_rope_at_plain_text_position():
    speaker = _mask_speaker("g0")
    assert speaker._audio_rope_position(frame_id=0, physical_position=8) == 2
    assert speaker._audio_rope_position(frame_id=3, physical_position=11) == 5


def test_legacy_audio_rope_keeps_physical_position():
    speaker = _mask_speaker("legacy")
    assert speaker._audio_rope_position(frame_id=3, physical_position=11) == 11


def test_stable_text_prefill_primes_slot_cache_before_prefix_frames():
    speaker = _mask_speaker("g0")
    speaker._slot_cache = SimpleNamespace(max_cache_len=12)
    speaker._icl = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    speaker._text_emb = torch.tensor([[[5.0, 6.0]]])
    speaker._response_text_conditioning = torch.tensor([[[7.0, 8.0]]])
    speaker._prefix_start = 4
    speaker._audio_start = 8
    speaker._keys_arange = None
    captured = {}

    class StubModel:
        def __call__(self, **kwargs):
            captured.update(kwargs)

    speaker.talker = SimpleNamespace(talker=SimpleNamespace(model=StubModel()))
    speaker._observe_qwen_input = lambda **kwargs: None

    speaker._prefill_stable()

    expected = torch.cat(
        [speaker._icl, speaker._text_emb, speaker._response_text_conditioning],
        dim=1,
    )
    assert torch.equal(captured["inputs_embeds"], expected)
    assert captured["cache_position"].tolist() == [0, 1, 2, 3]
    assert captured["position_ids"].shape == (3, 1, 4)
    assert captured["use_cache"] is True
