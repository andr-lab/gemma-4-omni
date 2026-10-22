from types import SimpleNamespace

import pytest
import torch

import duplex_kernel.server as server


def test_enforced_voice_rejects_reference_file_with_stale_sha_metadata(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"replaced audio, stale metadata")
    embedding = torch.tensor([0.25, -0.5])
    default_conditioning = {
        "mode": "x_vector_only",
        "speaker_embedding": embedding,
        "reference_wav_path": str(reference),
        "reference_wav_sha256": server.V9_TRAINING_REFERENCE_SHA256,
    }
    engine = SimpleNamespace(
        talker=SimpleNamespace(
            default_voice_conditioning=default_conditioning,
            tcfg=SimpleNamespace(hidden_size=2),
        ),
        enforce_v9_training_voice=True,
        ablate_session_voice_conditioning=False,
    )
    monkeypatch.setattr(
        server,
        "resolve_voice_conditioning",
        lambda *_args, **_kwargs: {
            "mode": "x_vector_only",
            "speaker_embedding": embedding.clone(),
        },
    )

    with pytest.raises(ValueError, match="SHA|hash|reference"):
        server._resolve_p7_session_voice(engine, voice="gemma")


def test_voice_ablation_cannot_override_v9_identity_enforcement():
    engine = SimpleNamespace(
        talker=None,
        enforce_v9_training_voice=True,
        ablate_session_voice_conditioning=True,
    )

    with pytest.raises(ValueError, match="incompatibili|enforcement|ablation"):
        server._resolve_p7_session_voice(engine, voice="gemma")

