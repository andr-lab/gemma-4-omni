"""CPU-testable shared listener frontend for causal Q=2 80 ms ticks.

This module stages per-session PCM/mel transitions and forms one user-only
``[N, W, 128]`` tensor: ``W - 8`` prior rows plus the current eight. CUDA keeps
the tower geometry at M=1 for each row. A caller commits staged PCM tails and
mel histories only after the complete listener transaction has succeeded.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import threading
import time
from typing import Any

import numpy as np
import torch

from duplex_kernel.listener import AudioTowerRowTiming
from duplex_kernel.listener_microbatch import ListenerBatch
from duplex_kernel.microbatch_contract import (
    CONTROL_HIDDEN_WIDTH,
    MEL_BINS,
    PCM_SAMPLES_PER_FRAME,
    PCM_TAIL_SAMPLES,
    ListenerFrame,
    ListenerSessionState,
)


# Source parity: GemmaListenerState defaults to 64 context frames and keeps
# eight mel rows per frame; the current default window is 240 user Mel rows.
CONTEXT_FRAMES = 64
MEL_ROWS_PER_FRAME = 8
MEL_HISTORY_ROWS = CONTEXT_FRAMES * MEL_ROWS_PER_FRAME
MEL_WINDOW_ROWS = 240
MEL_CONTEXT_ROWS = MEL_WINDOW_ROWS - MEL_ROWS_PER_FRAME
SOFT_TOKENS_PER_FRAME = 2
TOWER_OUTPUT_ROWS_PER_WINDOW = MEL_WINDOW_ROWS // 4
CURRENT_Q_START = TOWER_OUTPUT_ROWS_PER_WINDOW - SOFT_TOKENS_PER_FRAME
REFERENCE_DELAY_FRAMES = 1


@dataclass(frozen=True, slots=True)
class MelWindowGeometry:
    mel_window_rows: int
    context_rows: int
    tower_output_rows: int
    q_start: int
    required_history_frames: int


def validate_mel_window_rows(mel_window_rows: int) -> int:
    """Validate a fixed Gemma audio-tower Mel window geometry."""
    if (
        isinstance(mel_window_rows, bool)
        or not isinstance(mel_window_rows, int)
        or mel_window_rows < MEL_ROWS_PER_FRAME
        or mel_window_rows % 4 != 0
    ):
        raise ValueError("mel_window_rows must be a multiple of 4 and at least 8")
    return mel_window_rows


def mel_window_geometry(mel_window_rows: int) -> MelWindowGeometry:
    rows = validate_mel_window_rows(mel_window_rows)
    context_rows = rows - MEL_ROWS_PER_FRAME
    required_history_frames = (context_rows + MEL_ROWS_PER_FRAME - 1) // MEL_ROWS_PER_FRAME
    tower_output_rows = rows // 4
    return MelWindowGeometry(
        mel_window_rows=rows,
        context_rows=context_rows,
        tower_output_rows=tower_output_rows,
        q_start=tower_output_rows - SOFT_TOKENS_PER_FRAME,
        required_history_frames=required_history_frames,
    )

IncrementalMelExtractor = Callable[
    [np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray]
]
AudioTowerRowRunner = Callable[[torch.Tensor, int], tuple[torch.Tensor, "AudioTowerRowTiming"]]
@dataclass(slots=True)
class _Workspace:
    capacity: int
    cpu_features: np.ndarray
    features: torch.Tensor
    streaming_features: torch.Tensor | None
    streaming_mask: torch.Tensor | None
    manifest_id: str | None = None


@dataclass(frozen=True, slots=True)
class _StateSnapshot:
    state: ListenerSessionState
    frame_id: int
    generation_id: int
    epoch_id: str
    state_version: int
    sequence_position: int
    cache_position: int
    valid_history_tokens: int
    previous_user_tail: np.ndarray | None
    previous_assistant_tail: np.ndarray | None
    previous_user_mels: tuple[np.ndarray, ...]
    previous_assistant_mels: tuple[np.ndarray, ...]
    user_tail: np.ndarray | None
    assistant_tail: np.ndarray | None
    user_mels: tuple[np.ndarray, ...]
    assistant_mels: tuple[np.ndarray, ...]


class PreparedListenerAudioBatch:
    """Encoded rows plus an all-row frontend-state commit transaction.

    The returned soft-token tensors are cloned from the model output so a later
    batch call cannot overwrite them through a model-owned output workspace.
    The caller must commit only after native listener and policy work for this
    manifest has succeeded, or abort if downstream work fails.
    """

    def __init__(
        self,
        *,
        manifest_id: str,
        input_shape: tuple[int, int, int],
        audio_soft_tokens: tuple[torch.Tensor, ...],
        snapshots: tuple[_StateSnapshot, ...],
        mel_prepare_ms_by_row: tuple[float, ...],
        frontend_copy_ms: float,
        audio_tower_lock_wait_ms: float,
        audio_tower_lock_hold_ms: float,
        frontend_copy_cuda_events: tuple[Any, Any] | None,
        audio_tower_cuda_events: tuple[Any, Any] | None,
        audio_tower_row_timings: tuple[AudioTowerRowTiming, ...],
        audio_tower_graph_submit_ms: float,
        audio_tower_execution_mode: str,
        owner_thread_id: int,
        release_workspace: Callable[[str], None],
        abort_workspace: Callable[[str], None],
        valid_mel_counts: tuple[int, ...] | None = None,
        streaming_audio_batch: Any | None = None,
        audio_tower_raw_tokens: tuple[torch.Tensor, ...] = (),
        audio_stream_cursors: tuple[Any, ...] = (),
        streaming_audio_input_shape: tuple[int, int, int] | None = None,
    ) -> None:
        self.manifest_id = manifest_id
        self.input_shape = input_shape
        self.valid_mel_counts = valid_mel_counts or tuple(input_shape[1] for _ in audio_soft_tokens)
        self.audio_soft_tokens = audio_soft_tokens
        self._snapshots = snapshots
        self.mel_prepare_ms_by_row = mel_prepare_ms_by_row
        self.frontend_copy_ms = frontend_copy_ms
        self.audio_tower_lock_wait_ms = audio_tower_lock_wait_ms
        self.audio_tower_lock_hold_ms = audio_tower_lock_hold_ms
        self._frontend_copy_cuda_events = frontend_copy_cuda_events
        self._audio_tower_cuda_events = audio_tower_cuda_events
        self._audio_tower_row_timings = audio_tower_row_timings
        self.audio_tower_graph_submit_ms = audio_tower_graph_submit_ms
        self.audio_tower_execution_mode = audio_tower_execution_mode
        self.audio_tower_raw_tokens = audio_tower_raw_tokens
        self.audio_stream_cursors = audio_stream_cursors
        self.streaming_audio_input_shape = streaming_audio_input_shape
        self.audio_tower_workspace_lock_wait_ms = round(
            sum(item.workspace_lock_wait_ms for item in audio_tower_row_timings), 3
        )
        self.audio_tower_workspace_lock_hold_ms = round(
            sum(item.workspace_lock_hold_ms for item in audio_tower_row_timings), 3
        )
        self._owner_thread_id = owner_thread_id
        self._release_workspace = release_workspace
        self._abort_workspace = abort_workspace
        self._streaming_audio_batch = streaming_audio_batch
        self._committed = False
        self._aborted = False

    def cuda_phase_timings_after_completion(
        self,
    ) -> tuple[float | None, float | None, float | None]:
        """Read phase events after the worker's existing stream-completion wait.

        This method deliberately does not synchronize CUDA. Calling it before
        the owning listener stream has completed is unsupported.
        """
        row_copy_ms = _sum_cuda_event_pairs(
            timing.copy_cuda_events for timing in self._audio_tower_row_timings
        )
        tower_ms = (
            _sum_cuda_event_pairs(
                timing.tower_cuda_events for timing in self._audio_tower_row_timings
            )
            if self._audio_tower_row_timings
            else _elapsed_cuda_events(self._audio_tower_cuda_events)
        )
        return _elapsed_cuda_events(self._frontend_copy_cuda_events), row_copy_ms, tower_ms

    def audio_tower_batch_cuda_span_ms_after_completion(self) -> float | None:
        """Read the whole tower span after the worker's existing stream fence."""
        return _elapsed_cuda_events(self._audio_tower_cuda_events)

    @property
    def committed(self) -> bool:
        return self._committed

    def commit(self) -> None:
        """Atomically install all rows' staged tails and bounded mel histories."""
        if self._committed:
            raise RuntimeError("frontend transaction was already committed")
        if self._aborted:
            raise RuntimeError("frontend transaction was aborted")
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("frontend state must be committed by its single owner thread")

        mutation_started = False
        streaming_committed = False
        previous: list[tuple[Any, ...]] = []
        try:
            # Validate every row before mutating any row; this prevents a stale
            # second session from leaving the first session partially advanced.
            for snapshot in self._snapshots:
                _validate_snapshot_unchanged(snapshot)

            previous = [
                (
                    snapshot.state,
                    _copy_optional_tail(snapshot.state.user_pcm_tail),
                    _copy_optional_tail(snapshot.state.assistant_pcm_tail),
                    _copy_history_container(snapshot.state.user_mel_history),
                    _copy_history_container(snapshot.state.assistant_mel_history),
                )
                for snapshot in self._snapshots
            ]
            mutation_started = True
            if self._streaming_audio_batch is not None:
                self._streaming_audio_batch.commit()
                streaming_committed = True
            for snapshot in self._snapshots:
                state = snapshot.state
                state.user_pcm_tail = _copy_optional_tail(snapshot.user_tail)
                state.assistant_pcm_tail = _copy_optional_tail(snapshot.assistant_tail)
                state.user_mel_history = _history_container_like(
                    state.user_mel_history, snapshot.user_mels
                )
                state.assistant_mel_history = _history_container_like(
                    state.assistant_mel_history, snapshot.assistant_mels
                )
            self._release_workspace(self.manifest_id)
        except BaseException:
            try:
                if streaming_committed:
                    self._streaming_audio_batch.rollback_committed()
                elif self._streaming_audio_batch is not None:
                    abort_stream = getattr(self._streaming_audio_batch, "abort", None)
                    if callable(abort_stream) and not getattr(self._streaming_audio_batch, "_aborted", False):
                        abort_stream()
                if mutation_started:
                    for state, user_tail, assistant_tail, user_mels, assistant_mels in previous:
                        state.user_pcm_tail = user_tail
                        state.assistant_pcm_tail = assistant_tail
                        state.user_mel_history = user_mels
                        state.assistant_mel_history = assistant_mels
            finally:
                self._aborted = True
                self._abort_workspace(self.manifest_id)
            raise
        self._committed = True

    def abort(self) -> None:
        """Release this lease after downstream native/policy work fails.

        The frontend is poisoned because the caller cannot prove that the
        whole listener transaction is still at the state this batch observed.
        """
        if self._committed:
            raise RuntimeError("committed frontend transaction cannot be aborted")
        if self._aborted:
            raise RuntimeError("frontend transaction was already aborted")
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("frontend transaction must be aborted by its single owner thread")
        self._aborted = True
        try:
            if self._streaming_audio_batch is not None:
                self._streaming_audio_batch.abort()
        finally:
            self._abort_workspace(self.manifest_id)


class ListenerMicrobatchAudioFrontend:
    """Shared mel frontend with exact-geometry CUDA tower execution.

    ``extract_single_frame`` should be the bound production
    ``GemmaAudioTowerBackend._extract_single_frame`` method. The owner must call
    this object serially on its designated listener stream; one grow-only input
    and mask workspace is reused for every batch size. CUDA requires either a
    backend row runner. CUDA keeps the tower geometry at M=1 for each row.
    CPU-only tests may retain the vectorized tower call.
    """

    def __init__(
        self,
        gemma_model: Any,
        extract_single_frame: IncrementalMelExtractor,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        context_frames: int = CONTEXT_FRAMES,
        mel_window_rows: int = MEL_WINDOW_ROWS,
        audio_tower_lock: Any | None = None,
        audio_tower_row_runner: AudioTowerRowRunner | None = None,
        streaming_audio_tower: Any | None = None,
    ) -> None:
        if not callable(getattr(gemma_model, "get_audio_features", None)):
            raise TypeError("Gemma model must expose get_audio_features")
        if not callable(extract_single_frame):
            raise TypeError("the production incremental mel extractor is required")
        if (
            isinstance(context_frames, bool)
            or not isinstance(context_frames, int)
            or context_frames != CONTEXT_FRAMES
        ):
            raise ValueError(
                f"batched listener frontend is pinned to {CONTEXT_FRAMES} context frames"
            )
        geometry = mel_window_geometry(mel_window_rows)
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        self._gemma_model = gemma_model
        self._extract_single_frame = extract_single_frame
        self.device = torch.device(device)
        self.dtype = dtype
        # 64 remains the existing minimum retained context. Larger configured
        # windows grow Mel history only; LM/KV replay limits remain separate.
        self.context_frames = max(context_frames, geometry.required_history_frames)
        self.mel_window_rows = geometry.mel_window_rows
        self.mel_context_rows = geometry.context_rows
        self.tower_output_rows = geometry.tower_output_rows
        self.current_q_start = geometry.q_start
        self.audio_tower_lock = audio_tower_lock
        if audio_tower_row_runner is not None and not callable(audio_tower_row_runner):
            raise TypeError("audio_tower_row_runner must be callable")
        if self.device.type == "cuda" and audio_tower_row_runner is None:
            raise ValueError("CUDA microbatch frontend requires an N=1 audio tower row runner")
        self.audio_tower_row_runner = audio_tower_row_runner
        if streaming_audio_tower is not None and not callable(
            getattr(streaming_audio_tower, "prepare_batch", None)
        ):
            raise TypeError("streaming_audio_tower must expose prepare_batch")
        self.streaming_audio_tower = streaming_audio_tower
        self._workspace_buffer: _Workspace | None = None
        self._workspace_capacity = 0
        self._owner_thread_id: int | None = None
        self._call_lock = threading.Lock()
        self._active_manifest_id: str | None = None
        self._poisoned = False
        self._closed = False
        self.encoder_call_count = 0

    def close_session(self, session_id: str) -> None:
        """Release one session's incremental tower state while the frontend is idle."""
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("session_id must be a non-empty string")
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("cannot close streaming audio state while the frontend is active")
        try:
            self._ensure_open()
            if self._active_manifest_id is not None:
                raise RuntimeError("cannot close streaming audio state while a batch is prepared")
            if self.streaming_audio_tower is not None:
                self.streaming_audio_tower.reset(session_id)
        finally:
            self._call_lock.release()

    def close(self) -> None:
        """Release the incremental tower and reusable frontend workspace while idle."""
        if not self._call_lock.acquire(blocking=False):
            raise RuntimeError("cannot close listener audio frontend while it is active")
        try:
            if self._active_manifest_id is not None:
                raise RuntimeError("cannot close listener audio frontend with a prepared batch")
            if self._closed:
                return
            if self.streaming_audio_tower is not None:
                close = getattr(self.streaming_audio_tower, "close", None)
                if callable(close):
                    close()
                self.streaming_audio_tower = None
            self._workspace_buffer = None
            self._workspace_capacity = 0
            self._owner_thread_id = None
            self._closed = True
        finally:
            self._call_lock.release()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("listener audio frontend is closed")

    def workspace_data_ptr(self, batch_size: int) -> int:
        """Expose storage identity for CPU tests and preflight diagnostics."""
        return self._workspace(batch_size).features.data_ptr()

    @property
    def workspace_capacity(self) -> int:
        """Current high-water row capacity of the single reusable workspace."""
        return self._workspace_capacity

    def _cuda_event_pair(self) -> tuple[Any, Any] | None:
        if self.device.type != "cuda":
            return None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )

    def encode_batch(
        self,
        batch: ListenerBatch,
        states: Sequence[ListenerSessionState],
        *,
        listener_stream: Any | None = None,
    ) -> PreparedListenerAudioBatch:
        rows = tuple(states)
        with self._call_lock:
            self._ensure_open()
            current_thread = threading.get_ident()
            if self._owner_thread_id is None:
                self._owner_thread_id = current_thread
            elif self._owner_thread_id != current_thread:
                raise RuntimeError("listener audio frontend has a single owner thread")
            if self._poisoned:
                raise RuntimeError("listener audio frontend is poisoned after a failed transaction")
            if self._active_manifest_id is not None:
                raise RuntimeError("previous frontend batch has not committed")
            snapshots, feature_rows, mel_prepare_ms_by_row = self._stage_rows(batch, rows)
            timing_stream = listener_stream
            if self.device.type == "cuda" and timing_stream is None:
                timing_stream = torch.cuda.current_stream(device=self.device)
            n = batch.manifest.batch_size
            workspace = self._workspace(n)
            if workspace.manifest_id is not None:
                raise RuntimeError("frontend workspace is still leased")
            workspace.manifest_id = batch.manifest.manifest_id
            self._active_manifest_id = batch.manifest.manifest_id
            try:
                cpu_features = workspace.cpu_features[:n]
                features = workspace.features[:n]
                copy_started_ns = time.perf_counter_ns()
                cpu_features.fill(0.0)
                valid_mel_counts = tuple(int(row.shape[0]) for row in feature_rows)
                for index, (row, valid_rows) in enumerate(
                    zip(feature_rows, valid_mel_counts, strict=True)
                ):
                    cpu_features[index, :valid_rows] = row
                source_cpu = torch.from_numpy(cpu_features)
                frontend_copy_cuda_events = self._cuda_event_pair()
                if frontend_copy_cuda_events is not None:
                    frontend_copy_cuda_events[0].record(timing_stream)
                source = source_cpu.to(
                    device=self.device,
                    dtype=self.dtype,
                    non_blocking=True,
                )
                features.copy_(source, non_blocking=True)
                if frontend_copy_cuda_events is not None:
                    frontend_copy_cuda_events[1].record(timing_stream)
                frontend_copy_ms = max(
                    0, time.perf_counter_ns() - copy_started_ns
                ) / 1_000_000.0

                self.encoder_call_count += 1
                lock = self.audio_tower_lock
                acquired = False
                lock_wait_started_ns = time.perf_counter_ns()
                audio_tower_lock_wait_ms = 0.0
                audio_tower_lock_hold_ms = 0.0
                lock_acquired_ns: int | None = None
                audio_tower_cuda_events = self._cuda_event_pair()
                audio_tower_row_timings: tuple[AudioTowerRowTiming, ...] = ()
                audio_tower_graph_submit_ms = 0.0
                audio_tower_execution_mode = "vectorized_cpu"
                streaming_audio_batch: Any | None = None
                audio_tower_raw_tokens: tuple[torch.Tensor, ...] = ()
                try:
                    acquire_fair = getattr(lock, "acquire_fair", None)
                    if callable(acquire_fair):
                        acquired = bool(acquire_fair("listener"))
                        if not acquired:
                            raise RuntimeError("listener audio tower lock admission was cancelled")
                    elif lock is not None:
                        lock.acquire()
                        acquired = True
                    lock_acquired_ns = time.perf_counter_ns()
                    audio_tower_lock_wait_ms = max(
                        0, lock_acquired_ns - lock_wait_started_ns
                    ) / 1_000_000.0
                    if audio_tower_cuda_events is not None:
                        audio_tower_cuda_events[0].record(timing_stream)
                    if self.streaming_audio_tower is not None:
                        from duplex_kernel.streaming_conformer import AudioStreamCursor

                        graph_submit_started = time.perf_counter_ns()
                        chunks = workspace.streaming_features
                        stream_mask = workspace.streaming_mask
                        if chunks is None or stream_mask is None:
                            raise RuntimeError("streaming frontend workspace was not allocated")
                        chunks = chunks[:n]
                        stream_mask = stream_mask[:n]
                        for index, valid_rows in enumerate(valid_mel_counts):
                            chunks[index].copy_(
                                features[
                                    index,
                                    valid_rows - MEL_ROWS_PER_FRAME:valid_rows,
                                ],
                                non_blocking=True,
                            )
                        cursors = tuple(
                            AudioStreamCursor(
                                session_id=snapshot.state.session_id,
                                generation_id=snapshot.generation_id,
                                state_epoch_id=snapshot.epoch_id,
                                frame_id=snapshot.frame_id,
                                sequence_position=snapshot.sequence_position,
                            )
                            for snapshot in snapshots
                        )
                        streaming_audio_batch = self.streaming_audio_tower.prepare_batch(
                            chunks,
                            cursors,
                            input_mask=stream_mask,
                            use_cuda_graph=self.device.type == "cuda",
                        )
                        audio_tower_graph_submit_ms = max(
                            0, time.perf_counter_ns() - graph_submit_started
                        ) / 1_000_000.0
                        if audio_tower_cuda_events is not None:
                            audio_tower_cuda_events[1].record(timing_stream)
                        soft = streaming_audio_batch.soft_tokens
                        raw = streaming_audio_batch.raw_tokens
                        if tuple(soft.shape) != (n, SOFT_TOKENS_PER_FRAME, CONTROL_HIDDEN_WIDTH):
                            raise RuntimeError("streaming audio tower did not return [N,2,1536]")
                        audio_model = getattr(self.streaming_audio_tower, "audio_tower", None)
                        audio_config = getattr(audio_model, "config", None)
                        if audio_config is None:
                            raise RuntimeError("streaming audio tower does not expose its audio config")
                        expected_raw_width = int(audio_config.output_proj_dims)
                        if raw.ndim != 3 or tuple(raw.shape[:2]) != (n, SOFT_TOKENS_PER_FRAME):
                            raise RuntimeError("streaming audio tower raw output must be [N,2,D]")
                        if raw.shape[-1] != expected_raw_width:
                            raise RuntimeError("streaming audio tower raw width differs from output_proj_dims")
                        soft_tokens = tuple(soft[index].detach() for index in range(n))
                        audio_tower_raw_tokens = tuple(raw[index].detach() for index in range(n))
                        audio_tower_execution_mode = (
                            f"streaming_conformer_{streaming_audio_batch.execution_mode}_n{n}"
                        )
                    elif self.audio_tower_row_runner is not None:
                        row_outputs: list[torch.Tensor] = []
                        row_timings: list[AudioTowerRowTiming] = []
                        for index, snapshot in enumerate(snapshots):
                            valid_rows = valid_mel_counts[index]
                            q_start = valid_rows // 4 - SOFT_TOKENS_PER_FRAME
                            with torch.inference_mode():
                                row_output, row_timing = self.audio_tower_row_runner(
                                    features[index:index + 1, :valid_rows, :], q_start
                                )
                            if not isinstance(row_output, torch.Tensor) or tuple(row_output.shape) != (
                                1, SOFT_TOKENS_PER_FRAME, CONTROL_HIDDEN_WIDTH
                            ):
                                raise RuntimeError(
                                    "N=1 audio tower row runner must return [1,2,1536]"
                                )
                            if row_output.device != self.device or row_output.dtype != self.dtype:
                                raise RuntimeError(
                                    "N=1 audio tower row output device/dtype differs from its input"
                                )
                            if not isinstance(row_timing, AudioTowerRowTiming):
                                raise TypeError("audio tower row runner must return AudioTowerRowTiming")
                            if row_timing.execution_mode not in {
                                "cuda_graph_n1", "eager_n1_degraded"
                            }:
                                raise RuntimeError("audio tower row runner reported an unknown execution mode")
                            if not np.isfinite(float(row_timing.graph_submit_ms)) or row_timing.graph_submit_ms < 0:
                                raise RuntimeError("audio tower graph submit time must be finite and non-negative")
                            row_outputs.append(row_output[0].detach())
                            row_timings.append(row_timing)
                        soft_tokens = tuple(row_outputs)
                        audio_tower_row_timings = tuple(row_timings)
                        audio_tower_graph_submit_ms = round(
                            sum(item.graph_submit_ms for item in row_timings), 3
                        )
                        modes = {item.execution_mode for item in row_timings}
                        audio_tower_execution_mode = (
                            next(iter(modes)) if len(modes) == 1 else "mixed_n1"
                        )
                    else:
                        by_valid_rows: dict[int, list[int]] = {}
                        for index, valid_rows in enumerate(valid_mel_counts):
                            by_valid_rows.setdefault(valid_rows, []).append(index)
                        row_outputs_by_index: list[torch.Tensor | None] = [None] * n
                        for valid_rows, indices in by_valid_rows.items():
                            index_tensor = torch.tensor(indices, dtype=torch.long)
                            group_features = features.index_select(0, index_tensor)[:, :valid_rows, :]
                            group_mask = torch.ones(
                                (len(indices), valid_rows), dtype=torch.bool, device=self.device
                            )
                            with torch.inference_mode():
                                output = self._gemma_model.get_audio_features(
                                    input_features=group_features,
                                    input_features_mask=group_mask,
                                    return_dict=True,
                                )
                            group_outputs = self._select_current_frame_tokens(
                                output,
                                len(indices),
                                tuple(len(snapshots[index].user_mels) - 1 for index in indices),
                            )
                            for index, row_output in zip(indices, group_outputs, strict=True):
                                row_outputs_by_index[index] = row_output
                        if any(output is None for output in row_outputs_by_index):
                            raise RuntimeError("frontend did not encode every startup/window row")
                        soft_tokens = tuple(output for output in row_outputs_by_index if output is not None)
                    if audio_tower_cuda_events is not None:
                        audio_tower_cuda_events[1].record(timing_stream)
                finally:
                    if acquired:
                        assert lock_acquired_ns is not None
                        audio_tower_lock_hold_ms = max(
                            0, time.perf_counter_ns() - lock_acquired_ns
                        ) / 1_000_000.0
                        lock.release()
            except BaseException:
                if streaming_audio_batch is not None and not getattr(streaming_audio_batch, "_committed", False):
                    try:
                        streaming_audio_batch.abort()
                    except RuntimeError:
                        pass
                self._abort_workspace(batch.manifest.manifest_id)
                raise
            return PreparedListenerAudioBatch(
                manifest_id=batch.manifest.manifest_id,
                input_shape=(n, self.mel_window_rows, MEL_BINS),
                valid_mel_counts=valid_mel_counts,
                audio_soft_tokens=soft_tokens,
                snapshots=snapshots,
                mel_prepare_ms_by_row=mel_prepare_ms_by_row,
                frontend_copy_ms=round(frontend_copy_ms, 3),
                audio_tower_lock_wait_ms=round(audio_tower_lock_wait_ms, 3),
                audio_tower_lock_hold_ms=round(audio_tower_lock_hold_ms, 3),
                frontend_copy_cuda_events=frontend_copy_cuda_events,
                audio_tower_cuda_events=audio_tower_cuda_events,
                audio_tower_row_timings=audio_tower_row_timings,
                audio_tower_graph_submit_ms=audio_tower_graph_submit_ms,
                audio_tower_execution_mode=audio_tower_execution_mode,
                owner_thread_id=current_thread,
                release_workspace=self._release_workspace,
                abort_workspace=self._abort_workspace,
                streaming_audio_batch=streaming_audio_batch,
                audio_tower_raw_tokens=audio_tower_raw_tokens,
                audio_stream_cursors=(
                    tuple(streaming_audio_batch.cursors)
                    if streaming_audio_batch is not None
                    else ()
                ),
                streaming_audio_input_shape=(
                    (n, MEL_ROWS_PER_FRAME, MEL_BINS)
                    if streaming_audio_batch is not None
                    else None
                ),
            )

    def _release_workspace(self, manifest_id: str) -> None:
        if self._active_manifest_id != manifest_id:
            raise RuntimeError("frontend workspace release does not match its active manifest")
        workspace = self._workspace_buffer
        if workspace is None or workspace.manifest_id != manifest_id:
            raise RuntimeError("active frontend workspace is missing")
        workspace.manifest_id = None
        self._active_manifest_id = None

    def _abort_workspace(self, manifest_id: str) -> None:
        self._poisoned = True
        self._release_workspace(manifest_id)

    def _workspace(self, batch_size: int) -> _Workspace:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("audio tower batch size must be a positive integer")
        workspace = self._workspace_buffer
        if workspace is None or batch_size > workspace.capacity:
            if workspace is not None and workspace.manifest_id is not None:
                raise RuntimeError("cannot grow the frontend workspace while it is leased")
            capacity = max(batch_size, 1 if workspace is None else workspace.capacity * 2)
            workspace = _Workspace(
                capacity=capacity,
                cpu_features=np.empty(
                    (capacity, self.mel_window_rows, MEL_BINS), dtype=np.float32
                ),
                features=torch.empty(
                    (capacity, self.mel_window_rows, MEL_BINS),
                    dtype=self.dtype,
                    device=self.device,
                ),
                streaming_features=(
                    torch.empty(
                        (capacity, MEL_ROWS_PER_FRAME, MEL_BINS),
                        dtype=self.dtype,
                        device=self.device,
                    )
                    if self.streaming_audio_tower is not None
                    else None
                ),
                streaming_mask=(
                    torch.ones(
                        (capacity, MEL_ROWS_PER_FRAME),
                        dtype=torch.bool,
                        device=self.device,
                    )
                    if self.streaming_audio_tower is not None
                    else None
                ),
            )
            self._workspace_buffer = workspace
            self._workspace_capacity = capacity
        return workspace

    def _stage_rows(
        self,
        batch: ListenerBatch,
        states: tuple[ListenerSessionState, ...],
    ) -> tuple[tuple[_StateSnapshot, ...], tuple[np.ndarray, ...], tuple[float, ...]]:
        manifest = batch.manifest
        if len(states) != manifest.batch_size or len(batch.frames) != manifest.batch_size:
            raise ValueError("batch, manifest, and state row counts must match")
        if len({id(state) for state in states}) != len(states):
            raise ValueError("a frontend batch cannot alias a session state object")
        if len({state.session_id for state in states}) != len(states):
            raise ValueError("a frontend batch requires distinct session states")

        snapshots: list[_StateSnapshot] = []
        feature_rows: list[np.ndarray] = []
        mel_prepare_ms_by_row: list[float] = []
        for row, frame, state in zip(manifest.rows, batch.frames, states, strict=True):
            row_started_ns = time.perf_counter_ns()
            if (
                state.session_id != row.session_id
                or state.runtime_signature != manifest.runtime_signature
                or state.generation_id != row.session_generation
                or state.state_epoch_id != row.state_epoch_id
                or state.next_frame_id != row.frame_id
                or state.state_version != row.listener_state_version
                or state.sequence_position != row.sequence_position
                or state.cache_position != row.cache_position
                or state.valid_history_tokens != row.valid_history_tokens
                or state.native_context_id != row.native_context_id
            ):
                raise RuntimeError("frontend row does not match its current session snapshot")
            if frame.reference_pcm is not None:
                self.validate_reference_frame(frame)
            user_tail = _copy_optional_tail(state.user_pcm_tail)
            assistant_tail = _copy_optional_tail(state.assistant_pcm_tail)
            user_history = _snapshot_mel_history(
                state.user_mel_history, max_frames=self.context_frames
            )
            assistant_history = _snapshot_mel_history(
                state.assistant_mel_history, max_frames=self.context_frames
            )
            if len(user_history) != len(assistant_history):
                raise RuntimeError("user and assistant mel histories are not frame-aligned")

            assistant_pcm = (
                np.zeros(PCM_SAMPLES_PER_FRAME, dtype=np.float32)
                if frame.reference_pcm is None
                else np.asarray(frame.reference_pcm, dtype=np.float32)
            )
            user_mel, next_user_tail = self._extract_single_frame(
                np.asarray(frame.pcm, dtype=np.float32), user_tail
            )
            assistant_mel, next_assistant_tail = self._extract_single_frame(
                assistant_pcm, assistant_tail
            )
            user_mel = _validate_mel_rows(user_mel, "user mel")
            assistant_mel = _validate_mel_rows(assistant_mel, "assistant mel")
            next_user_tail = _validate_pcm_tail(next_user_tail, "user PCM tail")
            next_assistant_tail = _validate_pcm_tail(next_assistant_tail, "assistant PCM tail")

            next_user_history = (user_history + (user_mel,))[-self.context_frames:]
            next_assistant_history = (assistant_history + (assistant_mel,))[-self.context_frames:]
            feature_rows.append(
                _user_mel_window(user_history, user_mel, mel_window_rows=self.mel_window_rows)
            )
            snapshots.append(_StateSnapshot(
                state=state,
                frame_id=row.frame_id,
                generation_id=row.session_generation,
                epoch_id=row.state_epoch_id,
                state_version=row.listener_state_version,
                sequence_position=row.sequence_position,
                cache_position=row.cache_position,
                valid_history_tokens=row.valid_history_tokens,
                previous_user_tail=user_tail,
                previous_assistant_tail=assistant_tail,
                previous_user_mels=user_history,
                previous_assistant_mels=assistant_history,
                user_tail=next_user_tail,
                assistant_tail=next_assistant_tail,
                user_mels=next_user_history,
                assistant_mels=next_assistant_history,
            ))
            mel_prepare_ms_by_row.append(
                round(max(0, time.perf_counter_ns() - row_started_ns) / 1_000_000.0, 3)
            )

        return tuple(snapshots), tuple(feature_rows), tuple(mel_prepare_ms_by_row)

    @staticmethod
    def validate_reference_frame(frame: ListenerFrame) -> None:
        if frame.reference_pcm is not None and frame.reference_frame_id != (
            frame.frame_id - REFERENCE_DELAY_FRAMES
        ):
            raise ValueError(
                "reference frame must be the immediate prior frame for the production listener"
            )

    @staticmethod
    def _select_current_frame_tokens(
        output: Any,
        batch_size: int,
        current_frame_indices: Sequence[int],
    ) -> tuple[torch.Tensor, ...]:
        pooled = getattr(output, "pooler_output", None)
        valid_mask = getattr(output, "attention_mask", None)
        if not isinstance(pooled, torch.Tensor) or pooled.ndim != 3:
            raise RuntimeError("Gemma audio tower did not return rank-3 pooler_output")
        if not isinstance(valid_mask, torch.Tensor) or valid_mask.ndim != 2:
            raise RuntimeError("Gemma audio tower did not return a rank-2 output attention mask")
        if (
            pooled.shape[0] != batch_size
            or tuple(pooled.shape[:2]) != tuple(valid_mask.shape)
            or pooled.shape[-1] != CONTROL_HIDDEN_WIDTH
        ):
            raise RuntimeError("Gemma audio tower batch output shape is incompatible")
        if len(current_frame_indices) != batch_size:
            raise RuntimeError("audio tower current-frame indices do not match the batch")
        result: list[torch.Tensor] = []
        for index in range(batch_size):
            valid = pooled[index][valid_mask[index].to(device=pooled.device, dtype=torch.bool)]
            frame_index = current_frame_indices[index]
            if (
                isinstance(frame_index, bool)
                or not isinstance(frame_index, int)
                or frame_index < 0
            ):
                raise RuntimeError("audio tower current-frame index must be non-negative")
            if valid.shape[0] < SOFT_TOKENS_PER_FRAME:
                raise RuntimeError(
                    "Gemma audio tower returned fewer than 2 valid soft tokens for the current tick"
                )
            result.append(valid[-SOFT_TOKENS_PER_FRAME:].detach().clone())
        return tuple(result)


def _copy_optional_tail(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return _validate_pcm_tail(value, "PCM tail")


def _validate_pcm_tail(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.size != PCM_TAIL_SAMPLES or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly {PCM_TAIL_SAMPLES} finite samples")
    return np.array(result, dtype=np.float32, copy=True)


def _validate_mel_rows(value: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (MEL_ROWS_PER_FRAME, MEL_BINS) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain exactly 8 finite rows of 128 bins")
    return np.array(result, dtype=np.float32, copy=True)


def _copy_mel(value: np.ndarray) -> np.ndarray:
    return np.array(value, dtype=np.float32, copy=True)


def _copy_history_container(value: Any) -> list[np.ndarray] | deque[np.ndarray]:
    values = [_copy_mel(entry) for entry in value]
    if isinstance(value, deque):
        return deque(values, maxlen=value.maxlen)
    return values


def _history_container_like(
    template: Any,
    values: tuple[np.ndarray, ...],
) -> list[np.ndarray] | deque[np.ndarray]:
    copied = [_copy_mel(value) for value in values]
    if isinstance(template, deque):
        maxlen = template.maxlen
        if maxlen is not None and len(copied) > maxlen:
            maxlen = len(copied)
        return deque(copied, maxlen=maxlen)
    return copied


def _snapshot_mel_history(
    value: Any,
    *,
    max_frames: int = CONTEXT_FRAMES,
) -> tuple[np.ndarray, ...]:
    try:
        entries = tuple(value)
    except TypeError as exc:
        raise TypeError("mel history must be a finite iterable of frame arrays") from exc
    if len(entries) > max_frames:
        raise ValueError(f"mel history exceeds its configured {max_frames}-frame context")
    return tuple(_validate_mel_rows(entry, "mel history frame") for entry in entries)


def _pad_mel_history(history: tuple[np.ndarray, ...]) -> np.ndarray:
    if not history:
        return np.zeros((MEL_HISTORY_ROWS, MEL_BINS), dtype=np.float32)
    joined = np.concatenate(history, axis=0)
    if joined.shape[0] == MEL_HISTORY_ROWS:
        return joined
    if joined.shape[0] > MEL_HISTORY_ROWS:
        # Match the production `_pad_rows` truncation direction. Histories are
        # normally capped at 64; this branch is defensive, not a cache advance.
        return joined[:MEL_HISTORY_ROWS]
    # This mirrors the active standard listener `_pad_rows`: chronological
    # history starts at row zero and zeros fill the fixed-width tail. The
    # standard path also sends an all-true input mask, so the audio tower sees
    # those padded rows as valid context. Its current-frame Q2 is selected by
    # frame index, not by taking the last two outputs.
    return np.pad(
        joined,
        ((0, MEL_HISTORY_ROWS - joined.shape[0]), (0, 0)),
        mode="constant",
    )


def _user_mel_window(
    history: tuple[np.ndarray, ...],
    current: np.ndarray,
    *,
    mel_window_rows: int = MEL_WINDOW_ROWS,
) -> np.ndarray:
    """Build a causal ``W - 8`` prior + 8-current user-only Mel window."""
    geometry = mel_window_geometry(mel_window_rows)
    if tuple(current.shape) != (MEL_ROWS_PER_FRAME, MEL_BINS):
        raise ValueError("current user Mel must have shape [8,128]")
    prior_rows = tuple(_validate_mel_rows(entry, "user Mel history") for entry in history)
    current_rows = _validate_mel_rows(current, "current user Mel")
    joined_prior = (
        np.concatenate(prior_rows, axis=0)
        if prior_rows else np.empty((0, MEL_BINS), dtype=np.float32)
    )
    context = (
        joined_prior[-geometry.context_rows:]
        if geometry.context_rows
        else np.empty((0, MEL_BINS), dtype=np.float32)
    )
    result = np.concatenate((context, current_rows), axis=0)
    if result.shape[0] > geometry.mel_window_rows or result.shape[0] % 4:
        raise RuntimeError("user Mel window did not preserve its configured geometry")
    return np.asarray(result, dtype=np.float32)


def _validate_snapshot_unchanged(snapshot: _StateSnapshot) -> None:
    state = snapshot.state
    if (
        state.next_frame_id != snapshot.frame_id
        or state.generation_id != snapshot.generation_id
        or state.state_epoch_id != snapshot.epoch_id
        or state.state_version != snapshot.state_version
        or state.sequence_position != snapshot.sequence_position
        or state.cache_position != snapshot.cache_position
        or state.valid_history_tokens != snapshot.valid_history_tokens
    ):
        raise RuntimeError("session cursor changed before frontend commit")
    if not _optional_arrays_equal(state.user_pcm_tail, _snapshot_previous_tail(snapshot, "user")):
        raise RuntimeError("session user PCM tail changed before frontend commit")
    if not _optional_arrays_equal(
        state.assistant_pcm_tail,
        _snapshot_previous_tail(snapshot, "assistant"),
    ):
        raise RuntimeError("session assistant PCM tail changed before frontend commit")
    if not _histories_equal(state.user_mel_history, _snapshot_previous_mels(snapshot, "user")):
        raise RuntimeError("session user mel history changed before frontend commit")
    if not _histories_equal(
        state.assistant_mel_history,
        _snapshot_previous_mels(snapshot, "assistant"),
    ):
        raise RuntimeError("session assistant mel history changed before frontend commit")


def _snapshot_previous_tail(snapshot: _StateSnapshot, side: str) -> np.ndarray | None:
    value = getattr(snapshot, f"previous_{side}_tail")
    return None if value is None else np.array(value, dtype=np.float32, copy=True)


def _snapshot_previous_mels(snapshot: _StateSnapshot, side: str) -> tuple[np.ndarray, ...]:
    return tuple(_copy_mel(value) for value in getattr(snapshot, f"previous_{side}_mels"))


def _optional_arrays_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return np.array_equal(np.asarray(left), right)


def _histories_equal(left: Any, right: tuple[np.ndarray, ...]) -> bool:
    try:
        values = tuple(left)
    except TypeError:
        return False
    return len(values) == len(right) and all(
        np.array_equal(np.asarray(value), expected)
        for value, expected in zip(values, right, strict=True)
    )


def _elapsed_cuda_events(events: tuple[Any, Any] | None) -> float | None:
    if events is None:
        return None
    start, end = events
    return round(max(0.0, float(start.elapsed_time(end))), 3)


def _sum_cuda_event_pairs(groups: Any) -> float | None:
    elapsed = [
        value
        for value in (_elapsed_cuda_events(events) for events in groups)
        if value is not None
    ]
    return round(sum(elapsed), 3) if elapsed else None


__all__ = [
    "AudioTowerRowTiming",
    "ListenerMicrobatchAudioFrontend",
    "PreparedListenerAudioBatch",
]
