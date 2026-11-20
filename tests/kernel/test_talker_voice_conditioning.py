from __future__ import annotations

import hashlib
import json
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from duplex_kernel.models.talker import TalkerModel
from duplex_kernel.voices import resolve_voice_conditioning


HIDDEN = 4
REFERENCE_SHA = "a" * 64


class _Processor:
    def __init__(self) -> None:
        self.seen_texts: list[str] = []
        self.tokenizer = self

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        assert add_special_tokens is False
        self.seen_texts.append(text)
        if text == "<|im_start|>assistant\n":
            return {"input_ids": [10, 11, 12]}
        return {"input_ids": list(range(20, 20 + len(text)))}


class _Embedding:
    def __init__(self, hidden: int = HIDDEN) -> None:
        self.hidden = hidden
        self.seen_ids: list[torch.Tensor] = []

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.seen_ids.append(ids.detach().clone())
        return torch.zeros((*ids.shape, self.hidden), dtype=torch.float32)


class _MarkedEmbedding:
    def __init__(self, offset: int) -> None:
        self.offset = offset

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return (ids.float() + self.offset).unsqueeze(-1).expand(-1, -1, HIDDEN)


class _Talker:
    def __init__(self, hidden: int = HIDDEN) -> None:
        self.embedding = _Embedding(hidden)

    def get_text_embeddings(self) -> _Embedding:
        return self.embedding

    def get_input_embeddings(self) -> _Embedding:
        return self.embedding

    def text_projection(self, values: torch.Tensor) -> torch.Tensor:
        return values


def _model(hidden: int = HIDDEN) -> tuple[TalkerModel, _Processor]:
    processor = _Processor()
    cfg = SimpleNamespace(
        tts_bos_token_id=1,
        tts_eos_token_id=2,
        tts_pad_token_id=3,
    )
    tcfg = SimpleNamespace(
        hidden_size=hidden,
        codec_language_id={"italian": 2070},
        codec_think_id=4,
        codec_think_bos_id=5,
        codec_think_eos_id=6,
        codec_pad_id=7,
        codec_bos_id=8,
    )
    return TalkerModel(
        _Talker(hidden),
        processor,
        None,
        cfg,
        tcfg,
        device="cpu",
    ), processor


def test_base_prefill_uses_official_assistant_framing_and_italian_id() -> None:
    model, processor = _model()

    icl, text, bos, _ = model.build_prefill_parts("ciao")

    assert processor.seen_texts == ["<|im_start|>assistant\n", "ciao"]
    assert icl.shape == (1, 8, HIDDEN)
    assert text.shape == (1, 5, HIDDEN)
    assert bos.shape == (1, 1, HIDDEN)


def test_live_prefill_uses_only_prefilled_raw_qwen_token_and_omits_text_eos() -> None:
    model, _processor = _model()

    _icl, text, _bos, _pad = model.build_prefill_parts(
        " ", raw_text_token_ids=[77], include_text_eos=False
    )

    assert text.shape == (1, 0, HIDDEN)
    assert _processor.seen_texts == ["<|im_start|>assistant\n"]
    assert any(torch.equal(ids, torch.tensor([[77]])) for ids in model.talker.embedding.seen_ids)


@pytest.mark.parametrize("with_voice", [False, True])
@pytest.mark.parametrize("first_text_id", [34, 13878])
def test_live_prefill_combines_first_text_id_with_codec_bos_as_upstream(
    with_voice: bool, first_text_id: int
) -> None:
    model, _ = _model()
    model.talker.get_text_embeddings = lambda: _MarkedEmbedding(100)
    model.talker.get_input_embeddings = lambda: _MarkedEmbedding(1000)
    voice = (
        {"mode": "x_vector_only", "speaker_embedding": torch.ones(HIDDEN)}
        if with_voice else None
    )

    icl, text, bos, pad = model.build_prefill_parts(
        "Certo!", raw_text_token_ids=[first_text_id], include_text_eos=False,
        voice_conditioning=voice,
    )

    assert text.shape == (1, 0, HIDDEN)
    assert icl.shape == (1, 9 if with_voice else 8, HIDDEN)
    assert torch.equal(bos, torch.full((1, 1, HIDDEN), first_text_id + 100 + 8 + 1000.0))
    assert torch.equal(pad, torch.full((1, 1, HIDDEN), 3 + 100.0))
    assert torch.cat((icl, text, bos), dim=1).shape[1] == icl.shape[1] + 1


def test_complete_prefill_keeps_text_pad_then_text_eos_then_pad_bos() -> None:
    model, _ = _model()
    model.talker.get_text_embeddings = lambda: _MarkedEmbedding(100)
    model.talker.get_input_embeddings = lambda: _MarkedEmbedding(1000)
    _, text, bos, _ = model.build_prefill_parts("Certo!", raw_text_token_ids=[34, 13878, 0])

    expected = torch.tensor([34, 13878, 0, 2], dtype=torch.float32) + 100 + 7 + 1000
    assert torch.equal(text, expected.view(1, -1, 1).expand(-1, -1, HIDDEN))
    assert torch.equal(bos, torch.full((1, 1, HIDDEN), 3 + 100 + 8 + 1000.0))


def test_streaming_prefill_rejects_multiple_initial_text_ids() -> None:
    model, _ = _model()
    with pytest.raises(ValueError, match="exactly one text token"):
        model.build_prefill_parts("Certo!", raw_text_token_ids=[34, 13878], include_text_eos=False)


@pytest.mark.parametrize("text", ["\nCiao", "\n\nDue", "  Ciao", "Ciao!", "🙂"])
def test_base_prefill_direct_text_tokens_accept_whitespace_punctuation_and_emoji(text: str) -> None:
    model, processor = _model()

    icl, text_part, bos, _pad = model.build_prefill_parts(text)

    assert processor.seen_texts == ["<|im_start|>assistant\n", text]
    assert icl.shape == (1, 8, HIDDEN)
    assert text_part.shape == (1, len(text) + 1, HIDDEN)
    assert bos.shape == (1, 1, HIDDEN)


def test_base_prefill_inserts_x_vector_between_codec_prefix_and_tail() -> None:
    model, _ = _model()
    voice = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(HIDDEN),
        "reference_wav_sha256": REFERENCE_SHA,
        "ref_code": None,
    }

    icl, _, _, _ = model.build_prefill_parts(
        "ciao",
        voice_conditioning=voice,
    )

    assert icl.shape == (1, 9, HIDDEN)
    embedding_sha = hashlib.sha256(
        torch.ones(HIDDEN).numpy().astype("float32").tobytes()
    ).hexdigest()
    assert model.voice_conditioning_signature(voice) == (
        "xvector:"
        f"{REFERENCE_SHA}:"
        f"{embedding_sha}"
    )


def test_voice_conditioning_rejects_reference_codec_in_x_vector_mode() -> None:
    model, _ = _model()
    voice = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(HIDDEN),
        "ref_code": [1, 2, 3],
    }

    with pytest.raises(ValueError, match="ref_code"):
        model.build_prefill_parts("ciao", voice_conditioning=voice)


def test_manifest_xvec_resolves_to_p7_payload_and_builds_prefill(tmp_path, monkeypatch) -> None:
    voice_root = tmp_path / "voices"
    voice_root.mkdir()
    (voice_root / "gemma.safetensors").write_bytes(b"fixture")
    (voice_root / "custom_voice_manifest.json").write_text(
        json.dumps({"voices": {"gemma": {"file": "gemma.safetensors", "mode": "xvec"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMMA4_OMNI_CUSTOM_VOICE_DIR", str(voice_root))

    safe_tensors = types.ModuleType("safetensors")
    safe_tensors_torch = types.ModuleType("safetensors.torch")
    safe_tensors_torch.load_file = lambda path: {
        "speaker_embedding": torch.ones(HIDDEN),
        # The x-vector resolver must not carry any clone-code payload into P7.
        "ref_code": torch.tensor([[1, 2, 3]]),
    }
    safe_tensors.torch = safe_tensors_torch
    monkeypatch.setitem(sys.modules, "safetensors", safe_tensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safe_tensors_torch)

    model, _ = _model()
    voice = resolve_voice_conditioning("gemma")

    assert voice is not None
    assert voice["mode"] == "x_vector_only"
    assert isinstance(voice["speaker_embedding"], torch.Tensor)
    assert voice["speaker_embedding"].shape == (HIDDEN,)
    assert "ref_code" not in voice
    assert "voice_clone_prompt" not in voice
    assert model.build_prefill_parts("ciao", voice_conditioning=voice)[0].shape == (
        1,
        9,
        HIDDEN,
    )


def test_installed_gemma_voice_profile_builds_p7_prefill_when_available() -> None:
    from pathlib import Path
    import os

    voice_root = Path(os.environ.get("GEMMA4_OMNI_CUSTOM_VOICE_DIR", "/workspace/custom_voices"))
    if not (voice_root / "custom_voice_manifest.json").is_file():
        pytest.skip("manifest voci installato non presente in questo ambiente")
    voice = resolve_voice_conditioning("gemma")
    assert voice is not None and voice["mode"] == "x_vector_only"
    model, _ = _model(int(voice["speaker_embedding"].numel()))
    parts = model.build_prefill_parts("ciao", voice_conditioning=voice)
    assert parts[0].shape[-1] == voice["speaker_embedding"].numel()
