from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from duplex_kernel.listener import (
    AudioTowerRowTiming,
    DEFAULT_CONTEXT_FRAMES,
    GemmaAudioTowerBackend,
)
from duplex_kernel.listener_microbatch import ListenerMicrobatchCoordinator
from duplex_kernel.listener_microbatch_frontend import (
    MEL_WINDOW_ROWS,
    ListenerMicrobatchAudioFrontend,
    PreparedListenerAudioBatch,
)
from duplex_kernel.microbatch_contract import (
    ListenerFrame,
    ListenerSessionState,
    NativeBatchCapability,
    RuntimeSignature,
)


@pytest.fixture
def runtime_signature() -> RuntimeSignature:
    return RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        topology_sha256="c" * 64,
        native_library_sha256="d" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )


@pytest.fixture
def capability() -> NativeBatchCapability:
    return NativeBatchCapability(
        api_version=2,
        native_abi_version=3,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
        supports_arbitrary_batch_size=True,
        stream_priority=-1,
        stream_identity="listener-cuda-stream-17",
    )


@dataclass
class _RealMelFixture:
    frame_length: int = 321
    fft_length: int = 512
    preemphasis: float = 0.97
    preemphasis_htk_flavor: bool = False
    mel_floor: float = 1e-6
    per_bin_mean: np.ndarray | None = None
    per_bin_stddev: np.ndarray | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(90210)
        self.window = np.hanning(320).astype(np.float32)
        self.mel_filters = rng.uniform(0.0, 1.0, (257, 128)).astype(np.float32)


class _AudioTower:
    def __init__(self, *, valid_lengths: tuple[int, ...] | None = None) -> None:
        self.valid_lengths = valid_lengths
        self.calls: list[dict[str, Any]] = []
        self.output = torch.empty((4, 15, 1536), dtype=torch.float32)
        self.mask = torch.zeros((4, 15), dtype=torch.bool)
        self.raise_on_call: BaseException | None = None

    def get_audio_features(self, *, input_features, input_features_mask, return_dict):
        self.calls.append({
            "features_shape": tuple(input_features.shape),
            "mask_shape": tuple(input_features_mask.shape),
            "return_dict": return_dict,
            "features": input_features.detach().cpu().numpy().copy(),
            "mask": input_features_mask.detach().cpu().numpy().copy(),
        })
        if self.raise_on_call is not None:
            raise self.raise_on_call
        count = int(input_features.shape[0])
        output_rows = int(input_features.shape[1]) // 4
        if self.output.shape[0] < count or self.output.shape[1] != output_rows:
            self.output = torch.empty((max(4, count), output_rows, 1536), dtype=torch.float32)
            self.mask = torch.zeros((max(4, count), output_rows), dtype=torch.bool)
        input_mask = input_features_mask.to(device=input_features.device, dtype=torch.bool)
        if input_mask.shape != input_features.shape[:2]:
            raise AssertionError("feature mask must align with the batched feature rows")
        # Gemma 4 v5.5.0 subsamples the valid mask twice with stride 2 and
        # returns it alongside the audio soft tokens.
        output_mask = input_mask[:, ::2][:, ::2]
        self.mask.zero_()
        if self.valid_lengths is None:
            self.mask[:count].copy_(output_mask)
        else:
            for row in range(count):
                length = self.valid_lengths[row]
                if length:
                    self.mask[row, -length:] = True

        # A small CPU fixture: each output token depends on its four input
        # rows and the validity mask. With right padding + an all-true mask,
        # the last tokens would see only zero pads and ignore real PCM.
        row_features = input_features.float().reshape(count, output_rows, 4, 128)
        row_mask = input_mask.reshape(count, output_rows, 4)
        weighted = row_features * row_mask[:, :, :, None]
        denominator = row_mask.sum(dim=2).clamp_min(1).float() * 128.0
        token_values = weighted.sum(dim=(2, 3)) / denominator
        positions = torch.arange(output_rows, dtype=torch.float32, device=input_features.device)
        channels = torch.arange(1536, dtype=torch.float32, device=input_features.device)
        values = (
            token_values[:, :, None]
            + positions[None, :, None] / 128.0
            + channels[None, None, :] / 4096
        )
        self.output[:count].copy_(values)
        return SimpleNamespace(pooler_output=self.output[:count], attention_mask=self.mask[:count])


class _ElapsedCudaEvent:
    def __init__(self, elapsed_ms: float) -> None:
        self.elapsed_ms = elapsed_ms

    def elapsed_time(self, _end) -> float:
        return self.elapsed_ms


def _runtime_states_and_batch(
    n: int,
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
    *,
    mel_frames: int = 5,
    include_reference: bool = True,
    frame_id: int = 1,
    reference_frame_id: int | None = None,
) -> tuple[ListenerMicrobatchCoordinator, Any, tuple[ListenerSessionState, ...]]:
    coordinator = ListenerMicrobatchCoordinator(runtime_signature, capability)
    states: list[ListenerSessionState] = []
    rng = np.random.default_rng(1234 + n + mel_frames)
    for row in range(n):
        sid = f"sid-{row}"
        state = ListenerSessionState(
            session_id=sid,
            generation_id=3,
            state_epoch_id=f"epoch-{sid}",
            runtime_signature=runtime_signature,
            next_frame_id=frame_id,
            state_version=frame_id,
            sequence_position=frame_id,
            cache_position=38,
            valid_history_tokens=38,
            native_context_id=50_000 + row,
            user_pcm_tail=rng.standard_normal(160).astype(np.float32),
            assistant_pcm_tail=rng.standard_normal(160).astype(np.float32),
        )
        state.user_mel_history = deque(
            (rng.standard_normal((8, 128)).astype(np.float32) for _ in range(mel_frames)),
            maxlen=DEFAULT_CONTEXT_FRAMES,
        )
        state.assistant_mel_history = deque(
            (rng.standard_normal((8, 128)).astype(np.float32) for _ in range(mel_frames)),
            maxlen=DEFAULT_CONTEXT_FRAMES,
        )
        coordinator.register_session(state)
        states.append(state)

    for row, state in enumerate(states):
        reference = (
            np.full(1280, row + 0.25, dtype=np.float32) if include_reference else None
        )
        coordinator.enqueue(ListenerFrame(
            session_id=state.session_id,
            generation_id=state.generation_id,
            state_epoch_id=state.state_epoch_id,
            frame_id=frame_id,
            ready_at_ns=10_000 + row,
            pcm=np.linspace(row, row + 1, 1280, dtype=np.float32),
            reference_pcm=reference,
            reference_frame_id=(
                (frame_id - 1 if reference_frame_id is None else reference_frame_id)
                if reference is not None
                else None
            ),
        ))
    batch = coordinator.take_ready(2_010_000)
    assert batch is not None
    return coordinator, batch, tuple(states)


def _production_extractor() -> Any:
    backend = SimpleNamespace(feature_extractor=_RealMelFixture())
    return lambda wave, tail: GemmaAudioTowerBackend._extract_single_frame(backend, wave, tail)


def test_audio_tower_batch_cuda_span_is_distinct_from_replay_sum() -> None:
    class _Event:
        def __init__(self, elapsed_ms: float) -> None:
            self.elapsed_ms = elapsed_ms

        def elapsed_time(self, _end: Any) -> float:
            return self.elapsed_ms

    row_timing = SimpleNamespace(
        copy_cuda_events=None,
        tower_cuda_events=(_Event(1.25), _Event(0.0)),
        workspace_lock_wait_ms=0.0,
        workspace_lock_hold_ms=0.0,
    )
    prepared = PreparedListenerAudioBatch(
        manifest_id="timing-contract",
        input_shape=(1, 60, 128),
        audio_soft_tokens=(),
        snapshots=(),
        mel_prepare_ms_by_row=(),
        frontend_copy_ms=0.0,
        audio_tower_lock_wait_ms=0.0,
        audio_tower_lock_hold_ms=0.0,
        frontend_copy_cuda_events=None,
        audio_tower_cuda_events=(_Event(6.5), _Event(0.0)),
        audio_tower_row_timings=(row_timing,),
        audio_tower_graph_submit_ms=0.0,
        audio_tower_execution_mode="cuda_graph_n1",
        owner_thread_id=0,
        release_workspace=lambda _manifest_id: None,
        abort_workspace=lambda _manifest_id: None,
    )

    assert prepared.cuda_phase_timings_after_completion() == (None, None, 1.25)
    assert prepared.audio_tower_batch_cuda_span_ms_after_completion() == 6.5


def _expected_user_rows(batch, states, extractor) -> tuple[np.ndarray, ...]:
    expected: list[np.ndarray] = []
    for frame, state in zip(batch.frames, states, strict=True):
        user_mel, _ = extractor(np.asarray(frame.pcm), state.user_pcm_tail.copy())
        user_history = list(state.user_mel_history)
        user_history = user_history[-DEFAULT_CONTEXT_FRAMES:]
        user = (
            np.concatenate(user_history, axis=0)
            if user_history else np.empty((0, 128), dtype=np.float32)
        )
        context_rows = MEL_WINDOW_ROWS - 8
        user = user[-context_rows:] if context_rows else np.empty((0, 128), dtype=np.float32)
        expected.append(np.concatenate((user, user_mel), axis=0))
    return tuple(expected)


@pytest.mark.parametrize("n", (1, 2, 4))
@pytest.mark.parametrize("mel_frames", (0, 1, 62, 63, 64))
def test_batches_match_production_incremental_mel_history_and_encode_once(
    n: int,
    mel_frames: int,
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        n, runtime_signature, capability, mel_frames=mel_frames
    )
    extractor = _production_extractor()
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower,
        extractor,
        device="cpu",
        dtype=torch.float32,
    )

    expected_features = _expected_user_rows(batch, states, extractor)
    valid_rows = expected_features[0].shape[0]
    prior_user_tails = [state.user_pcm_tail.copy() for state in states]
    prior_assistant_tails = [state.assistant_pcm_tail.copy() for state in states]
    prepared = frontend.encode_batch(batch, states)

    assert len(tower.calls) == 1
    call = tower.calls[0]
    assert call["features_shape"] == (n, valid_rows, 128)
    assert call["mask_shape"] == (n, valid_rows)
    assert call["return_dict"] is True
    expected_mask = np.ones((n, valid_rows), dtype=np.bool_)
    np.testing.assert_array_equal(call["mask"], expected_mask)
    for index, expected in enumerate(expected_features):
        np.testing.assert_array_equal(call["features"][index], expected)
    assert all(token_rows.shape == (2, 1536) for token_rows in prepared.audio_soft_tokens)
    assert len(prepared.mel_prepare_ms_by_row) == n
    assert all(value >= 0.0 for value in prepared.mel_prepare_ms_by_row)
    assert prepared.frontend_copy_ms >= 0.0
    assert prepared.audio_tower_lock_wait_ms >= 0.0
    assert prepared.audio_tower_lock_hold_ms >= 0.0
    assert prepared.cuda_phase_timings_after_completion() == (None, None, None)
    current_frame_start = valid_rows // 4 - 2
    for index, token_rows in enumerate(prepared.audio_soft_tokens):
        assert torch.isfinite(token_rows).all()
        valid = tower.mask[index]
        torch.testing.assert_close(
            token_rows,
            tower.output[index][valid][current_frame_start:current_frame_start + 2].clone(),
            rtol=0,
            atol=0,
        )
    assert prepared.input_shape == (n, MEL_WINDOW_ROWS, 128)
    assert prepared.valid_mel_counts == (valid_rows,) * n
    assert not prepared.committed

    prepared.commit()
    assert prepared.committed
    assert all(isinstance(state.user_mel_history, deque) for state in states)
    assert all(state.user_mel_history.maxlen == DEFAULT_CONTEXT_FRAMES for state in states)
    assert all(isinstance(state.assistant_mel_history, deque) for state in states)
    assert all(state.assistant_mel_history.maxlen == DEFAULT_CONTEXT_FRAMES for state in states)
    for frame, state, prior_user_tail, prior_assistant_tail in zip(
        batch.frames,
        states,
        prior_user_tails,
        prior_assistant_tails,
        strict=True,
    ):
        expected_user_mel, expected_user_tail = extractor(frame.pcm, prior_user_tail)
        assistant_pcm = np.zeros(1280, dtype=np.float32) if frame.reference_pcm is None else frame.reference_pcm
        expected_assistant_mel, expected_assistant_tail = extractor(
            assistant_pcm, prior_assistant_tail
        )
        # The history has rolled independently for each state; only the staged
        # current frame and exact production tail are being asserted here.
        np.testing.assert_array_equal(state.user_mel_history[-1], expected_user_mel)
        np.testing.assert_array_equal(state.assistant_mel_history[-1], expected_assistant_mel)
        np.testing.assert_array_equal(state.user_pcm_tail, expected_user_tail)
        np.testing.assert_array_equal(state.assistant_pcm_tail, expected_assistant_tail)


def test_early_rows_match_user_only_unpadded_prefix_and_current_q2(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        2,
        runtime_signature,
        capability,
        mel_frames=0,
        include_reference=False,
    )
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )

    prepared = frontend.encode_batch(batch, states)
    call = tower.calls[0]
    expected_features = _expected_user_rows(
        batch, states, _production_extractor()
    )
    assert not torch.equal(prepared.audio_soft_tokens[0], prepared.audio_soft_tokens[1])
    np.testing.assert_array_equal(call["mask"], np.ones((2, 8), dtype=np.bool_))
    assert prepared.valid_mel_counts == (8, 8)
    for row, expected in enumerate(expected_features):
        np.testing.assert_array_equal(call["features"][row], expected)
        # The newest two output rows correspond to the current tick.
        torch.testing.assert_close(
            prepared.audio_soft_tokens[row], tower.output[row, -2:].clone(), rtol=0, atol=0
        )


def test_cuda_row_runner_uses_n1_inputs_and_selects_current_q2_without_vectorized_tower(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        3, runtime_signature, capability, mel_frames=2
    )
    tower = _AudioTower()
    calls: list[tuple[torch.Tensor, int]] = []

    def row_runner(features: torch.Tensor, q_start: int):
        calls.append((features.detach().clone(), q_start))
        values = torch.full(
            (1, 2, 1536), float(len(calls)), dtype=torch.float32
        )
        return values, AudioTowerRowTiming(
            graph_submit_ms=0.125,
            copy_cuda_events=None,
            tower_cuda_events=None,
            execution_mode="cuda_graph_n1",
            workspace_lock_wait_ms=0.025,
            workspace_lock_hold_ms=0.05,
        )

    frontend = ListenerMicrobatchAudioFrontend(
        tower,
        _production_extractor(),
        device="cpu",
        dtype=torch.float32,
        audio_tower_row_runner=row_runner,
    )

    prepared = frontend.encode_batch(batch, states)

    assert tower.calls == []
    assert len(calls) == 3
    assert all(tuple(features.shape) == (1, 24, 128) for features, _ in calls)
    assert [q_start for _, q_start in calls] == [4, 4, 4]
    assert prepared.valid_mel_counts == (24, 24, 24)
    assert [float(tokens[0, 0]) for tokens in prepared.audio_soft_tokens] == [1, 2, 3]
    assert prepared.audio_tower_graph_submit_ms == pytest.approx(0.375)
    assert prepared.audio_tower_execution_mode == "cuda_graph_n1"
    assert prepared.audio_tower_workspace_lock_wait_ms == pytest.approx(0.075)
    assert prepared.audio_tower_workspace_lock_hold_ms == pytest.approx(0.15)
    prepared.commit()


@pytest.mark.parametrize(
    ("prior_frames", "valid_rows", "q_start"),
    ((0, 8, 0), (1, 16, 2), (28, 232, 56), (29, 240, 58), (40, 240, 58)),
)
def test_startup_prefix_grows_without_padding_then_uses_full_window_graph(
    prior_frames: int,
    valid_rows: int,
    q_start: int,
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        1, runtime_signature, capability, mel_frames=prior_frames
    )
    tower = _AudioTower()
    calls: list[tuple[tuple[int, ...], int]] = []

    def row_runner(features: torch.Tensor, start: int):
        calls.append((tuple(features.shape), start))
        return torch.full((1, 2, 1536), float(start)), AudioTowerRowTiming(
            graph_submit_ms=0.1,
            copy_cuda_events=None,
            tower_cuda_events=None,
            execution_mode=("cuda_graph_n1" if features.shape[1] == 240 else "eager_n1_degraded"),
        )

    frontend = ListenerMicrobatchAudioFrontend(
        tower,
        _production_extractor(),
        device="cpu",
        dtype=torch.float32,
        audio_tower_row_runner=row_runner,
    )

    prepared = frontend.encode_batch(batch, states)

    assert calls == [((1, valid_rows, 128), q_start)]
    assert prepared.valid_mel_counts == (valid_rows,)
    assert prepared.audio_tower_execution_mode == (
        "cuda_graph_n1" if valid_rows == 240 else "eager_n1_degraded"
    )
    assert torch.all(prepared.audio_soft_tokens[0] == float(q_start))
    prepared.commit()


def test_mixed_startup_ages_keep_each_prefix_and_q2_output_in_row_order(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        2, runtime_signature, capability, mel_frames=29
    )
    states[0].user_mel_history = deque(maxlen=DEFAULT_CONTEXT_FRAMES)
    states[0].assistant_mel_history = deque(maxlen=DEFAULT_CONTEXT_FRAMES)
    tower = _AudioTower()
    calls: list[tuple[int, int]] = []

    def row_runner(features: torch.Tensor, q_start: int):
        calls.append((int(features.shape[1]), q_start))
        return torch.full((1, 2, 1536), float(q_start)), AudioTowerRowTiming(
            graph_submit_ms=0.1,
            copy_cuda_events=None,
            tower_cuda_events=None,
            execution_mode=("cuda_graph_n1" if features.shape[1] == 240 else "eager_n1_degraded"),
        )

    frontend = ListenerMicrobatchAudioFrontend(
        tower,
        _production_extractor(),
        device="cpu",
        dtype=torch.float32,
        audio_tower_row_runner=row_runner,
    )

    prepared = frontend.encode_batch(batch, states)

    assert calls == [(8, 0), (240, 58)]
    assert prepared.valid_mel_counts == (8, 240)
    assert prepared.audio_tower_execution_mode == "mixed_n1"
    assert [float(row[0, 0]) for row in prepared.audio_soft_tokens] == [0.0, 58.0]
    prepared.commit()


def test_absent_assistant_reference_matches_production_zero_pcm(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    # This is the production listener's `reference is None` branch: it appends
    # a real 1,280-sample zero frame before calling the frontend.
    _, batch, states = _runtime_states_and_batch(
        1,
        runtime_signature,
        capability,
        mel_frames=0,
        include_reference=False,
    )
    extractor = _production_extractor()
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, extractor, device="cpu", dtype=torch.float32
    )

    expected = _expected_user_rows(batch, states, extractor)[0]
    frontend.encode_batch(batch, states)

    np.testing.assert_array_equal(tower.calls[0]["features"][0], expected)


def test_encoder_failure_does_not_commit_any_frontend_state(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(2, runtime_signature, capability)
    before = [
        (
            state.user_pcm_tail.copy(),
            state.assistant_pcm_tail.copy(),
            tuple(item.copy() for item in state.user_mel_history),
            tuple(item.copy() for item in state.assistant_mel_history),
        )
        for state in states
    ]
    tower = _AudioTower()
    tower.raise_on_call = RuntimeError("injected encoder failure")
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )

    with pytest.raises(RuntimeError, match="injected encoder failure"):
        frontend.encode_batch(batch, states)
    for state, snapshot in zip(states, before, strict=True):
        np.testing.assert_array_equal(state.user_pcm_tail, snapshot[0])
        np.testing.assert_array_equal(state.assistant_pcm_tail, snapshot[1])
        for actual, expected in zip(state.user_mel_history, snapshot[2], strict=True):
            np.testing.assert_array_equal(actual, expected)
        for actual, expected in zip(state.assistant_mel_history, snapshot[3], strict=True):
            np.testing.assert_array_equal(actual, expected)


def test_invalid_second_row_output_fails_without_partial_state_commit(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(2, runtime_signature, capability)
    tower = _AudioTower(valid_lengths=(15, 1, 15, 15))
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )

    with pytest.raises(RuntimeError, match="fewer than 2 valid soft tokens"):
        frontend.encode_batch(batch, states)
    assert len(tower.calls) == 1
    assert all(len(state.user_mel_history) == 5 for state in states)
    assert all(len(state.assistant_mel_history) == 5 for state in states)


def test_frontend_stages_then_commits_all_rows_only_when_snapshots_unchanged(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(2, runtime_signature, capability)
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )
    prepared = frontend.encode_batch(batch, states)
    prior_lengths = [len(state.user_mel_history) for state in states]
    states[1].user_mel_history.append(np.zeros((8, 128), dtype=np.float32))

    with pytest.raises(RuntimeError, match="changed before frontend commit"):
        prepared.commit()
    assert not prepared.committed
    assert len(states[0].user_mel_history) == prior_lengths[0]
    assert len(states[1].user_mel_history) == prior_lengths[1] + 1
    assert all(state.user_pcm_tail is not None for state in states)
    _, retry_batch, retry_states = _runtime_states_and_batch(
        2, runtime_signature, capability
    )
    with pytest.raises(RuntimeError, match="poisoned after a failed transaction"):
        frontend.encode_batch(retry_batch, retry_states)


def test_outstanding_frontend_batch_exclusively_leases_workspace_until_commit(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )
    _, first_batch, first_states = _runtime_states_and_batch(
        1, runtime_signature, capability, frame_id=1
    )
    prepared = frontend.encode_batch(first_batch, first_states)
    _, second_batch, second_states = _runtime_states_and_batch(
        2, runtime_signature, capability, frame_id=1
    )

    with pytest.raises(RuntimeError, match="previous frontend batch has not committed"):
        frontend.encode_batch(second_batch, second_states)
    assert frontend.workspace_capacity == 1
    prepared.commit()

    second = frontend.encode_batch(second_batch, second_states)
    second.commit()
    assert frontend.workspace_capacity == 2
    assert len(tower.calls) == 2


def test_aborting_downstream_failure_releases_lease_and_fails_closed(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )
    _, batch, states = _runtime_states_and_batch(1, runtime_signature, capability)
    prepared = frontend.encode_batch(batch, states)

    prepared.abort()
    with pytest.raises(RuntimeError, match="transaction was aborted"):
        prepared.commit()
    _, retry_batch, retry_states = _runtime_states_and_batch(
        1, runtime_signature, capability
    )
    with pytest.raises(RuntimeError, match="poisoned after a failed transaction"):
        frontend.encode_batch(retry_batch, retry_states)


def test_frontend_reuses_one_grow_only_workspace_with_standard_all_true_mask(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )
    capacities: list[int] = []
    data_ptrs: list[int] = []
    saved_soft: torch.Tensor | None = None
    first_prepared = None
    for n in (1, 2, 3, 4, 5, 8, 12):
        _, batch, states = _runtime_states_and_batch(n, runtime_signature, capability)
        prepared = frontend.encode_batch(batch, states)
        capacities.append(frontend.workspace_capacity)
        data_ptrs.append(frontend.workspace_data_ptr(n))
        if n == 1:
            first_prepared = prepared
            saved_soft = prepared.audio_soft_tokens[0].clone()
        prepared.commit()
    assert capacities == [1, 2, 4, 4, 8, 8, 16]
    assert data_ptrs[2] == data_ptrs[3]
    assert data_ptrs[4] == data_ptrs[5]

    _, batch, states = _runtime_states_and_batch(
        1,
        runtime_signature,
        capability,
        mel_frames=0,
    )
    refreshed = frontend.encode_batch(batch, states)
    refreshed.commit()
    assert tower.calls[-1]["features_shape"] == (1, 8, 128)
    expected_mask = np.ones((1, 8), dtype=np.bool_)
    np.testing.assert_array_equal(tower.calls[-1]["mask"], expected_mask)
    assert saved_soft is not None and first_prepared is not None
    assert not torch.equal(saved_soft, tower.output[0, :4])
    torch.testing.assert_close(first_prepared.audio_soft_tokens[0], saved_soft, rtol=0, atol=0)


def test_reference_must_match_pinned_prior_frame_and_fail_without_state_mutation(
    runtime_signature: RuntimeSignature,
    capability: NativeBatchCapability,
) -> None:
    _, batch, states = _runtime_states_and_batch(
        1,
        runtime_signature,
        capability,
        mel_frames=0,
        frame_id=2,
        reference_frame_id=0,
    )
    # The frontend operates on the passed immutable row and checks the default
    # production delay of one frame rather than accepting merely any past ref.
    tower = _AudioTower()
    frontend = ListenerMicrobatchAudioFrontend(
        tower, _production_extractor(), device="cpu", dtype=torch.float32
    )
    with pytest.raises(ValueError, match="reference frame must be the immediate prior frame"):
        frontend.encode_batch(batch, states)
