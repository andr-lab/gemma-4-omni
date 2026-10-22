from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from duplex_kernel.models.talker import (
    _install_shims,
    _rebuild_code2wav_rotary_buffers,
)


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
