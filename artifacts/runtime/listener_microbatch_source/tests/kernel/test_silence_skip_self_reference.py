import numpy as np

from duplex_kernel.server import _silence_skip_inputs_are_exact_zero


def test_silence_skip_requires_exact_zero_user_and_self_reference_pcm() -> None:
    zeros = np.zeros(128, dtype=np.float32)
    self_audio = np.full(128, 0.2, dtype=np.float32)
    user_audio = np.full(128, 0.1, dtype=np.float32)

    assert not _silence_skip_inputs_are_exact_zero(zeros, self_audio)
    assert _silence_skip_inputs_are_exact_zero(zeros, zeros)
    assert not _silence_skip_inputs_are_exact_zero(user_audio, zeros)
    assert not _silence_skip_inputs_are_exact_zero(zeros, None)
