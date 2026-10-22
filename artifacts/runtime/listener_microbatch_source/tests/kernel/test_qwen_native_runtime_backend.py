from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from duplex_kernel.engine import PipelineEngine
from duplex_kernel.future_prefix import FuturePrefixQwen


class _FakeTalker:
    pass


def test_qwen_runtime_backend_is_explicit_and_closed_set() -> None:
    with pytest.raises(ValueError, match="runtime_backend"):
        FuturePrefixQwen(_FakeTalker(), 0, 4, runtime_backend="cuda")


def test_qwen_native_requires_joint_q2_horizon() -> None:
    with pytest.raises(ValueError, match="lookahead_frames=0"):
        FuturePrefixQwen(_FakeTalker(), 1, 4, runtime_backend="native")
    with pytest.raises(ValueError, match="max_audio_frames"):
        FuturePrefixQwen(_FakeTalker(), 0, 4, max_audio_frames=5, runtime_backend="native")
    with pytest.raises(ValueError, match="al massimo una terminal query Q=1"):
        FuturePrefixQwen(_FakeTalker(), 0, 4, terminal_query_budget=2, runtime_backend="native")
    with pytest.raises(ValueError, match="dtype=torch.bfloat16"):
        FuturePrefixQwen(_FakeTalker(), 0, 4, max_audio_frames=4, dtype=None, device="cuda", runtime_backend="native")
    with pytest.raises(ValueError, match="device CUDA"):
        FuturePrefixQwen(_FakeTalker(), 0, 4, max_audio_frames=4, dtype=torch.bfloat16, runtime_backend="native")


def test_qwen_pytorch_remains_default() -> None:
    speaker = FuturePrefixQwen(_FakeTalker(), 0, 4)
    assert speaker.runtime_info() == {
        "runtime_backend": "pytorch",
        "native_enabled": False,
        "segment_open": False,
        "segment_id": None,
        "generation_id": None,
        "conditioning_slots": 4,
        "prefix_capacity": 4,
        "acoustic_frame_budget": 64,
        "terminal_query_budget": 1,
        "terminal_queries_used": 0,
        "conditioning_available_through": -1,
        "conditioning_writes": 0,
        "conditioning_duplicate_writes": 0,
        "conditioning_out_of_order_writes": 0,
        "acoustic_kv_position": -1,
        "terminal_queries": 0,
        "terminal_query_position": -1,
        "terminal_query_frame_id": -1,
        "unexpected_cache_resets": 0,
        "duplicate_cache_appends": 0,
        "segment_opens": 0,
        "segment_closes": 0,
        "begin_calls": 0,
        "native_context_creates": 0,
        "native_context_reuses": 0,
        "slot_cache_creates": 0,
        "slot_cache_reuses": 0,
        "stable_prefill_runs": 0,
        "stable_prefill_reuses": 0,
        "native_cache_imports": 0,
    }
    assert PipelineEngine().runtime_backend == "pytorch"
    assert PipelineEngine(runtime_backend="native").runtime_backend == "native"
    with pytest.raises(ValueError, match="runtime_backend"):
        PipelineEngine(runtime_backend="cuda")


class _LifecycleNativeExecutor:
    instances: list["_LifecycleNativeExecutor"] = []
    fail_import = False
    fail_close = False

    def __init__(self, talker, *, max_cache_len: int, execute_layers: int) -> None:
        del talker
        self.max_cache_len = int(max_cache_len)
        self.execute_layers = int(execute_layers)
        self.imports: list[tuple[object, int, int, int]] = []
        self.close_calls = 0
        self.closed = False
        type(self).instances.append(self)

    def import_cache_async(self, cache, *, stable_length: int, prefix_start: int, audio_start: int) -> None:
        if type(self).fail_import:
            raise RuntimeError("synthetic cache import failure")
        self.imports.append((cache, int(stable_length), int(prefix_start), int(audio_start)))

    def runtime_info(self) -> dict[str, object]:
        return {
            "max_cache_len": self.max_cache_len,
            "execute_layers": self.execute_layers,
            "cache_imports": len(self.imports),
            "joint_steps": 0,
        }

    def close(self) -> None:
        if not self.closed:
            self.close_calls += 1
            if type(self).fail_close:
                raise RuntimeError("synthetic close failure")
            self.closed = True


class _LifecycleModel:
    def __init__(self) -> None:
        self.layers = [object() for _ in range(28)]
        self.rotary_emb = object()
        self.inputs: list[torch.Tensor] = []

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask,
        position_ids,
        cache_position,
        past_key_values,
        use_cache,
    ):
        del attention_mask, position_ids, cache_position, past_key_values, use_cache
        self.inputs.append(inputs_embeds.detach().clone())
        return SimpleNamespace(last_hidden_state=inputs_embeds)

    __call__ = forward


class _LifecycleTalker:
    def __init__(self) -> None:
        self.model = _LifecycleModel()
        self.talker = SimpleNamespace(model=self.model)
        self.build_calls: list[str] = []

    def build_prefill_parts(self, text: str):
        self.build_calls.append(text)
        text_width = 2 if text == "changed layout" else 1
        icl = torch.zeros((1, 1, 4))
        text_emb = torch.zeros((1, text_width, 4))
        bos = torch.zeros((1, 1, 4))
        return icl, text_emb, bos, bos.clone()


def _lifecycle_speaker(monkeypatch: pytest.MonkeyPatch) -> tuple[FuturePrefixQwen, _LifecycleTalker]:
    import duplex_kernel.native_qwen_talker as native_module

    _LifecycleNativeExecutor.instances.clear()
    _LifecycleNativeExecutor.fail_import = False
    _LifecycleNativeExecutor.fail_close = False
    monkeypatch.setattr(native_module, "NativeQwenTalker", _LifecycleNativeExecutor)
    talker = _LifecycleTalker()
    speaker = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=2,
        max_audio_frames=2,
        device="cpu",
        dtype=torch.float32,
        runtime_backend="pytorch",
    )
    # The fake model deliberately bypasses CUDA/native ABI; lifecycle behavior
    # is what this contract test isolates.
    speaker._runtime_backend = "native"
    speaker._slot_kv = True
    return speaker, talker


def test_native_runtime_info_disambiguates_frame_and_cache_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker, _talker = _lifecycle_speaker(monkeypatch)
    try:
        speaker.begin("terminal runtime metadata")
        native = speaker._native_qwen
        assert native is not None
        native.runtime_info = lambda: {
            "terminal_query_count": 1,
            "terminal_query_position": 11,
            "terminal_visible_prefix_position": 6,
        }
        state = speaker._active_talker_state
        assert state is not None
        state.note_terminal_query(2)

        info = speaker.runtime_info()
        assert info["terminal_query_position"] == 2
        assert info["terminal_query_frame_id"] == 2
        assert info["native_terminal_query_position"] == 11
        assert info["native_terminal_visible_prefix_position"] == 6
    finally:
        speaker.close()


def test_native_begin_identical_text_reuses_executor_cache_and_prefill(monkeypatch: pytest.MonkeyPatch) -> None:
    speaker, talker = _lifecycle_speaker(monkeypatch)
    prefill = Mock(wraps=speaker._prefill_stable)
    speaker._prefill_stable = prefill
    try:
        speaker.begin("Continuous conversational speech.")
        native = speaker._native_qwen
        cache = speaker._slot_cache
        speaker.begin("Continuous conversational speech.")

        assert speaker._native_qwen is native
        assert speaker._slot_cache is cache
        assert talker.build_calls == ["Continuous conversational speech."]
        assert prefill.call_count == 1
        info = speaker.runtime_info()
        assert info["begin_calls"] == 2
        assert info["native_context_creates"] == 1
        assert info["native_context_reuses"] == 1
        assert info["slot_cache_creates"] == 1
        assert info["slot_cache_reuses"] == 1
        assert info["stable_prefill_runs"] == 1
        assert info["stable_prefill_reuses"] == 1
        assert info["native_cache_imports"] == 2
    finally:
        speaker.close()


def test_native_begin_changed_text_reuses_compatible_layout_and_prefills(monkeypatch: pytest.MonkeyPatch) -> None:
    speaker, talker = _lifecycle_speaker(monkeypatch)
    prefill = Mock(wraps=speaker._prefill_stable)
    speaker._prefill_stable = prefill
    try:
        speaker.begin("first text")
        native = speaker._native_qwen
        cache = speaker._slot_cache
        speaker.begin("second text")

        assert speaker._native_qwen is native
        assert speaker._slot_cache is cache
        assert talker.build_calls == ["first text", "second text"]
        assert prefill.call_count == 2
        info = speaker.runtime_info()
        assert info["native_context_creates"] == 1
        assert info["native_context_reuses"] == 1
        assert info["slot_cache_creates"] == 1
        assert info["slot_cache_reuses"] == 1
        assert info["stable_prefill_runs"] == 2
        assert info["stable_prefill_reuses"] == 0
        assert info["native_cache_imports"] == 2
    finally:
        speaker.close()


def test_response_text_conditioning_is_stable_prefill_and_listener_slots_remain_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speaker, talker = _lifecycle_speaker(monkeypatch)
    response_prefix = torch.full((1, 2, 4), 22.0)
    changed_response_prefix = torch.full((1, 2, 4), 33.0)
    listener_frame = torch.full((1, 4), 7.0)
    try:
        speaker.begin("same text", response_text_conditioning=response_prefix)
        first_stable = talker.model.inputs[-1]
        assert torch.equal(first_stable[0, -2:, :], response_prefix[0])
        assert speaker._prefix_start == first_stable.shape[1]

        speaker.push_v9_frame(0, listener_frame)
        torch.testing.assert_close(speaker.buffer.frame(0), listener_frame.unsqueeze(1))

        # The same spoken string can have new generated hidden states. They
        # must replace the stable conditioning prefill instead of importing a
        # prior turn's cached Qwen prefix.
        speaker.begin("same text", response_text_conditioning=changed_response_prefix)
        second_stable = talker.model.inputs[-1]
        assert torch.equal(second_stable[0, -2:, :], changed_response_prefix[0])
        assert talker.build_calls == ["same text", "same text"]
    finally:
        speaker.close()


def test_native_executor_capacity_is_persistent_while_slot_cache_stays_exact_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import duplex_kernel.future_prefix as future_prefix_module

    monkeypatch.setattr(future_prefix_module, "QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN", 8)
    speaker, _ = _lifecycle_speaker(monkeypatch)
    try:
        speaker.begin("first text")
        native = speaker._native_qwen
        first_cache = speaker._slot_cache
        speaker.begin("changed layout")

        assert speaker._native_qwen is native
        assert native.max_cache_len == 8
        assert first_cache.max_cache_len == 7
        assert speaker._slot_cache is not first_cache
        assert speaker._slot_cache.max_cache_len == 8
        info = speaker.runtime_info()
        assert info["native_context_creates"] == 1
        assert info["native_context_reuses"] == 1
        assert info["slot_cache_creates"] == 2
        assert info["slot_cache_reuses"] == 0
    finally:
        speaker.close()


def test_native_begin_incompatible_layout_closes_and_recreates_once(monkeypatch: pytest.MonkeyPatch) -> None:
    import duplex_kernel.future_prefix as future_prefix_module

    monkeypatch.setattr(future_prefix_module, "QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN", 6)
    speaker, _ = _lifecycle_speaker(monkeypatch)
    try:
        speaker.begin("first text")
        old_native = speaker._native_qwen
        old_cache = speaker._slot_cache
        speaker.begin("changed layout")
        new_native = speaker._native_qwen
        new_cache = speaker._slot_cache

        assert new_native is not old_native
        assert new_cache is not old_cache
        assert old_native.close_calls == 1
        info = speaker.runtime_info()
        assert info["native_context_creates"] == 2
        assert info["native_context_reuses"] == 0
        assert info["slot_cache_creates"] == 2
        assert info["slot_cache_reuses"] == 0
        assert info["stable_prefill_runs"] == 2
        assert info["native_cache_imports"] == 2
    finally:
        speaker.close()


def test_native_executor_min_capacity_is_at_least_256() -> None:
    from duplex_kernel.config import QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN

    assert QWEN_NATIVE_EXECUTOR_MIN_CACHE_LEN >= 256


def test_native_close_is_idempotent_and_releases_segment_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    speaker, _ = _lifecycle_speaker(monkeypatch)
    speaker.begin("text")
    native = speaker._native_qwen
    segment = speaker._legacy_state.segment
    assert segment is not None
    speaker.close()
    speaker.close()

    assert native.close_calls == 1
    assert speaker._native_qwen is None
    assert speaker._slot_cache is None
    assert segment.closed is True
    assert segment.static_icl_kv is None
    assert segment.acoustic_kv is None
    assert speaker._started is False
    assert speaker.runtime_info()["native_enabled"] is False
    assert speaker.runtime_info()["segment_open"] is False
    assert speaker.runtime_info()["native_context_creates"] == 1


def test_native_close_failure_still_releases_python_state(monkeypatch: pytest.MonkeyPatch) -> None:
    speaker, _ = _lifecycle_speaker(monkeypatch)
    speaker.begin("text")
    native = speaker._native_qwen
    _LifecycleNativeExecutor.fail_close = True
    try:
        with pytest.raises(RuntimeError, match="synthetic close failure"):
            speaker.close()
    finally:
        _LifecycleNativeExecutor.fail_close = False

    assert native.close_calls == 1
    assert speaker._native_qwen is native
    assert speaker._slot_cache is None
    assert speaker._started is False
    assert speaker.runtime_info()["segment_open"] is False

    speaker.close()
    assert native.close_calls == 2
    assert speaker._native_qwen is None


def test_native_cache_import_failure_invalidates_without_pytorch_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    speaker, _ = _lifecycle_speaker(monkeypatch)
    speaker.begin("text")
    native = speaker._native_qwen
    _LifecycleNativeExecutor.fail_import = True
    try:
        with pytest.raises(RuntimeError, match="synthetic cache import failure"):
            speaker.begin("text")
    finally:
        _LifecycleNativeExecutor.fail_import = False
        speaker.close()

    assert native.close_calls == 1
    assert speaker._started is False
    assert speaker._native_qwen is None
    assert speaker.runtime_info()["native_enabled"] is False
