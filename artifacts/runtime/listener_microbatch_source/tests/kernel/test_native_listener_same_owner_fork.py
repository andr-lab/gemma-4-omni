from __future__ import annotations

import ctypes
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import duplex_kernel.native_listener_microbatch as binding
from duplex_kernel.microbatch_contract import BatchRowDescriptor


ROOT = Path(__file__).resolve().parents[2]
NATIVE_IMPL = ROOT / "native" / "gemma4_listener_microbatch" / "gemma4_listener_microbatch_impl.inc"


def _row() -> BatchRowDescriptor:
    return BatchRowDescriptor(
        session_id="owner-session",
        session_generation=7,
        state_epoch_id="epoch-owner-7",
        frame_id=23,
        listener_state_version=41,
        sequence_position=83,
        cache_position=295,
        valid_history_tokens=295,
        native_context_id=1001,
        input_frame_sha256="a" * 64,
        reference_pcm_sha256="b" * 64,
        frame_ready_at_ns=123456,
        reference_frame_id=22,
        logical_q_len=2,
    )


class _FakeNative:
    def __init__(self, owner, *, clone_status=0, state_status=0, release_status=0):
        self.owner = owner
        self.clone_status = clone_status
        self.state_status = state_status
        self.release_status = release_status
        self.clone_calls = []
        self.release_calls = []

    def gemma4_listener_batch_get_session_state(self, _ctx, slot, row_ptr, state_ptr):
        if self.state_status:
            return self.state_status
        slot = int(slot.value)
        state = ctypes.cast(state_ptr, ctypes.POINTER(binding.CBatchSessionState)).contents
        session = next(item for item in self.owner._session_by_id.values() if item.slot == slot)
        state.slot = slot
        state.native_context_id = session.native_context_id
        state.session_generation = session.generation
        state.state_epoch_id = session.epoch_numeric
        state.expected_frame_id = session.expected_frame_id
        state.expected_state_version = session.expected_state_version
        state.expected_sequence_position = session.expected_sequence_position
        state.prefix_len = session.prefix_len
        state.current_position = session.cache_position
        state.valid_kv_len = session.valid_history_tokens
        state.max_seq_len = 16384
        state.primed = 1
        return 0

    def gemma4_listener_batch_clone_session_slot_async(
        self, _ctx, destination_slot, destination_row_ptr, source_slot, source_row_ptr, stream,
    ):
        destination_row = ctypes.cast(
            destination_row_ptr, ctypes.POINTER(binding.CBatchRow)
        ).contents
        source_row = ctypes.cast(source_row_ptr, ctypes.POINTER(binding.CBatchRow)).contents
        self.clone_calls.append((
            int(destination_slot.value),
            tuple(int(getattr(destination_row, name)) for name, _ in binding.CBatchRow._fields_),
            int(source_slot.value),
            tuple(int(getattr(source_row, name)) for name, _ in binding.CBatchRow._fields_),
            int(stream.value),
        ))
        return self.clone_status

    def gemma4_listener_batch_release_session_async(self, _ctx, slot, row_ptr, _stream):
        row = ctypes.cast(row_ptr, ctypes.POINTER(binding.CBatchRow)).contents
        self.release_calls.append((int(slot.value), int(row.native_context_id)))
        return self.release_status


def _owner(monkeypatch, *, clone_status=0, state_status=0, release_status=0):
    native = object.__new__(binding.NativeGemmaListenerMicrobatch)
    native._owner_thread_id = __import__("threading").get_ident()
    native._closed = False
    native._poisoned = False
    native._ctx = ctypes.c_void_p(777)
    native._active_output_generation = None
    native._session_fork_api_available = True
    native.listener_stream = SimpleNamespace(cuda_stream=0xCAFE)
    native.max_batch_size = 2
    source = _row()
    source_binding = binding._SessionBinding(
        slot=0,
        session_id=source.session_id,
        native_context_id=source.native_context_id,
        generation=source.session_generation,
        epoch_text=source.state_epoch_id,
        epoch_numeric=binding._epoch_id_u64(source.state_epoch_id),
        expected_frame_id=source.frame_id,
        expected_state_version=source.listener_state_version,
        expected_sequence_position=source.sequence_position,
        cache_position=source.cache_position,
        valid_history_tokens=source.valid_history_tokens,
        prefix_len=291,
    )
    native._session_by_id = {source.session_id: source_binding}
    native._session_by_context = {source.native_context_id: source_binding}
    native._epoch_by_numeric = {source_binding.epoch_numeric: source.state_epoch_id}
    native._import_refs = {0: ("source-ref",)}
    native._lib = _FakeNative(
        native, clone_status=clone_status, state_status=state_status,
        release_status=release_status,
    )
    monkeypatch.setattr(binding.torch.cuda, "stream", lambda _stream: nullcontext())
    return native, source


def test_same_owner_fork_clones_identity_and_full_logical_boundary(monkeypatch):
    native, source = _owner(monkeypatch)

    fork = native.fork_session_slot(
        source, fork_session_id="bc-branch", fork_native_context_id=2002,
    )

    assert fork == replace(source, session_id="bc-branch", native_context_id=2002)
    assert native._session_by_id[source.session_id].slot == 0
    assert native._session_by_id[fork.session_id].slot == 1
    assert native._session_by_context == {1001: native._session_by_id[source.session_id],
                                          2002: native._session_by_id[fork.session_id]}
    assert native._session_by_id[fork.session_id].cache_position == 295
    assert native._session_by_id[fork.session_id].prefix_len == 291
    assert native._import_refs[0] == ("source-ref",)
    assert native._lib.clone_calls == [
        (1, binding._c_row_values(binding._row_to_c(fork)),
         0, binding._c_row_values(binding._row_to_c(source)), 0xCAFE)
    ]


def test_same_owner_fork_rejects_stale_source_before_reserving_slot(monkeypatch):
    native, source = _owner(monkeypatch)
    stale = replace(source, sequence_position=source.sequence_position + 1)

    with pytest.raises(RuntimeError, match="snapshot state mismatch"):
        native.fork_session_slot(
            stale, fork_session_id="bc-branch", fork_native_context_id=2002,
        )

    assert native._lib.clone_calls == []
    assert set(native._session_by_id) == {source.session_id}
    assert set(native._session_by_context) == {source.native_context_id}


def test_same_owner_fork_failure_does_not_publish_partial_python_binding(monkeypatch):
    native, source = _owner(monkeypatch, clone_status=-1)

    with pytest.raises(RuntimeError, match="native same-owner session fork"):
        native.fork_session_slot(
            source, fork_session_id="bc-branch", fork_native_context_id=2002,
        )

    assert set(native._session_by_id) == {source.session_id}
    assert set(native._session_by_context) == {source.native_context_id}
    assert native._session_by_id[source.session_id].cache_position == source.cache_position
    assert native._import_refs[0] == ("source-ref",)
    assert not native._poisoned


def test_same_owner_partial_copy_failure_poison_owner_and_does_not_publish_fork(monkeypatch):
    native, source = _owner(monkeypatch, clone_status=-3)

    with pytest.raises(RuntimeError, match="native same-owner session fork"):
        native.fork_session_slot(
            source, fork_session_id="bc-branch", fork_native_context_id=2002,
        )

    assert native._poisoned
    assert set(native._session_by_id) == {source.session_id}
    assert native._session_by_id[source.session_id].cache_position == source.cache_position


def test_fork_cleanup_releases_only_the_private_slot(monkeypatch):
    native, source = _owner(monkeypatch)
    fork = native.fork_session_slot(
        source, fork_session_id="bc-branch", fork_native_context_id=2002,
    )

    native.release_session_async(fork)

    assert native._lib.release_calls == [(1, 2002)]
    assert set(native._session_by_id) == {source.session_id}
    assert set(native._session_by_context) == {source.native_context_id}
    assert native._import_refs[0] == ("source-ref",)
    assert native._epoch_by_numeric[binding._epoch_id_u64(source.state_epoch_id)] == source.state_epoch_id


def test_native_clone_export_checks_unprimed_destination_and_copies_kv_and_prefix():
    source = NATIVE_IMPL.read_text(encoding="utf-8")
    clone = source.split("int gemma4_listener_batch_clone_session_slot_async(", 1)[1].split(
        "int gemma4_listener_batch_copy_session_cache_async(", 1
    )[0]

    assert "destination_slot == source_slot" in clone
    assert "destination_state.primed" in clone
    assert "gemma4_batch_snapshot_row_matches(batch, source_slot, source_row)" in clone
    assert "gemma4_cache_copy_async" in clone
    assert "destination_state.prefix_attention" in clone
    assert "destination_state.prefix_len = source_state.prefix_len" in clone
    assert "destination_state.current_position = source_state.current_position" in clone
    assert "destination_state.context_id = destination_row->native_context_id" in clone


def test_native_fork_feature_configures_its_export_only_when_advertised():
    class FakeFunction:
        def __init__(self, value=0):
            self.value = value
            self.restype = None
            self.argtypes = None

        def __call__(self, *_args):
            return self.value

    class FakeLibrary:
        def __init__(self, capabilities):
            self.functions = {
                "gemma4_listener_abi_version": FakeFunction(3),
                "gemma4_listener_batch_api_version": FakeFunction(binding.BATCH_API_VERSION),
                "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
            }

        def __getattr__(self, name):
            return self.functions.setdefault(name, FakeFunction())

    capabilities = (
        binding.BATCH_CAPABILITIES_V1
        | binding.BATCH_DYNAMIC_CAPABILITY
        | binding.BATCH_OUTPUT_HEALTH_CAPABILITY
        | binding.BATCH_LOGICAL_Q_CAPABILITY
        | binding.BATCH_TOKEN_EMBEDDING_CAPABILITY
        | binding.BATCH_TAIL_REWIND_CAPABILITY
        | binding.BATCH_QUEUED_AUDIO_REPLAY_CAPABILITY
        | binding.BATCH_SESSION_FORK_CAPABILITY
    )
    library = FakeLibrary(capabilities)

    features = binding._configure_native_api(library)

    clone = library.functions["gemma4_listener_batch_clone_session_slot_async"]
    assert features.session_fork
    assert clone.restype is ctypes.c_int
    assert clone.argtypes == [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(binding.CBatchRow),
        ctypes.c_uint32, ctypes.POINTER(binding.CBatchRow), ctypes.c_void_p,
    ]
