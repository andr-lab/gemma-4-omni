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
        self.seen_text: str | None = None

    def __call__(self, *, text: str, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        self.seen_text = text
        return {"input_ids": torch.arange(10, dtype=torch.long).view(1, -1)}


class _Embedding:
    def __init__(self, hidden: int = HIDDEN) -> None:
        self.hidden = hidden

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*ids.shape, self.hidden), dtype=torch.float32)


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

    assert processor.seen_text == (
        "<|im_start|>assistant\n"
        "ciao"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    assert icl.shape == (1, 8, HIDDEN)
    assert text.shape == (1, 3, HIDDEN)
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
