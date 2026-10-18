"""CPU-only checks for the ABI3-backed listener-microbatch v5 candidate.

These tests validate the ctypes/C layout and statically ensure that the batch
entry point dispatches the shared N-row layer path. They do not claim CUDA
compilation, numerical parity, or runtime performance.
"""
from __future__ import annotations

import ctypes
import importlib.util
import inspect
import re
import sys
import threading
import types
import unittest
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NATIVE_DIR = ROOT / "native" / "gemma4_listener_microbatch"
BINDING_PATH = ROOT / "duplex_kernel" / "native_listener_microbatch.py"


def _load_binding_with_cpu_stubs():
    """Load the real binding module without importing or initializing CUDA."""
    package = types.ModuleType("duplex_kernel")
    package.__path__ = [str(ROOT / "duplex_kernel")]

    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    torch_stub.bfloat16 = object()
    torch_stub.int64 = object()
    torch_stub.device = lambda value: value
    torch_stub.cuda = types.SimpleNamespace(
        Stream=type("Stream", (), {}),
        Event=type("Event", (), {}),
        current_stream=lambda **_kwargs: None,
    )

    contract_stub = types.ModuleType("duplex_kernel.microbatch_contract")

    @dataclass(frozen=True)
    class BatchRowDescriptor:
        session_id: str
        session_generation: int
        state_epoch_id: str
        frame_id: int
        listener_state_version: int
        sequence_position: int
        native_context_id: int
        cache_position: int
        valid_history_tokens: int
        logical_q_len: int = 2

    class ListenerBatchManifest:
        pass

    class NativeBatchCapability:
        pass

    class RuntimeSignature:
        pass

    contract_stub.BatchRowDescriptor = BatchRowDescriptor
    contract_stub.ListenerBatchManifest = ListenerBatchManifest
    contract_stub.NativeBatchCapability = NativeBatchCapability
    contract_stub.RuntimeSignature = RuntimeSignature
    contract_stub.MIN_BATCH_SIZE = 1
    contract_stub.SUPPORTED_LISTENER_STREAM_PRIORITIES = frozenset({-1, 0})

    listener_stub = types.ModuleType("duplex_kernel.native_listener")
    listener_stub.GEMMA4_ATTN_NATIVE_SPLITK_GQA = 3
    listener_stub.GEMMA4_HIDDEN_SIZE = 1536
    listener_stub.GEMMA4_RMS_NATIVE_EXACT = 2

    def _ctypes_struct(name):
        return type(name, (ctypes.Structure,), {"_fields_": []})

    listener_stub.Gemma4ImportedCache = _ctypes_struct("Gemma4ImportedCache")
    listener_stub.Gemma4ListenerConfig = _ctypes_struct("Gemma4ListenerConfig")
    listener_stub.Gemma4ListenerWeights = _ctypes_struct("Gemma4ListenerWeights")
    listener_stub.Gemma4TensorMeta = _ctypes_struct("Gemma4TensorMeta")

    class NativeGemmaListenerLM:
        pass

    listener_stub.NativeGemmaListenerLM = NativeGemmaListenerLM

    replacements = {
        "torch": torch_stub,
        "duplex_kernel": package,
        "duplex_kernel.microbatch_contract": contract_stub,
        "duplex_kernel.native_listener": listener_stub,
    }
    prior = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "native_listener_microbatch_cpu_test", BINDING_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, BatchRowDescriptor
    finally:
        sys.modules.pop("native_listener_microbatch_cpu_test", None)
        for name, old in prior.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


class NativeListenerMicrobatchBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binding, cls.Row = _load_binding_with_cpu_stubs()

    def base_capabilities(self):
        b = self.binding
        return (
            b.BATCH_CAPABILITIES_V1
            | b.BATCH_DYNAMIC_CAPABILITY
            | b.BATCH_OUTPUT_HEALTH_CAPABILITY
            | b.BATCH_LOGICAL_Q_CAPABILITY
            | b.BATCH_TOKEN_EMBEDDING_CAPABILITY
            | b.BATCH_TAIL_REWIND_CAPABILITY
            | b.BATCH_QUEUED_AUDIO_REPLAY_CAPABILITY
        )

    def test_layer0_trace_capability_is_optional_and_has_fixed_stage_inventory(self):
        b = self.binding
        self.assertEqual(b.BATCH_TRACE_CAPABILITY, 0x100)
        self.assertEqual(b.BATCH_TRACE_API_VERSION, 1)
        self.assertEqual(len(b.TRACE_STAGE_NAMES), 33)
        self.assertEqual(len(set(b.TRACE_STAGE_NAMES)), len(b.TRACE_STAGE_NAMES))
        self.assertEqual(b.TRACE_STAGE_NAMES[0], "AUDIO_PLE_GEMM")
        self.assertEqual(b.TRACE_STAGE_NAMES[2], "AUDIO_PLE_RMS")
        self.assertEqual(b.TRACE_STAGE_NAMES[13], "L0_ATTENTION")
        self.assertEqual(b.TRACE_STAGE_NAMES[-2], "L0_LAYER_SCALE")
        self.assertEqual(b.TRACE_STAGE_NAMES[-1], "FINAL_LAYER_OUTPUT")
        self.assertEqual(len(b.TRACE_STAGE_KINDS), len(b.TRACE_STAGE_NAMES))
        self.assertEqual(b.TRACE_STAGE_KINDS[13], b.TRACE_KIND_ATTENTION)

    def test_trace_reference_layout_uses_session_contiguous_q_and_ple_rows(self):
        b = self.binding
        self.assertEqual(
            b.trace_reference_layout("L0_Q_NORM", n=2),
            {"actual_rows": 4 * 8, "reference_rows": 2 * 8, "dim": 256},
        )
        self.assertEqual(
            b.trace_reference_layout("AUDIO_PLE_RMS", n=2),
            {"actual_rows": 2 * 2 * 35, "reference_rows": 2 * 35, "dim": 256},
        )
        self.assertEqual(
            b.trace_reference_layout("L0_INPUT_NORM", n=2),
            {"actual_rows": 4, "reference_rows": 2, "dim": 1536},
        )

    def test_batched_native_rms_keeps_reference_tree_and_launches_all_actual_rows(self):
        rms = (NATIVE_DIR / "gemma4_listener_rms_exact.cu").read_text(encoding="utf-8")
        rms_header = (NATIVE_DIR / "gemma4_listener_rms_exact.cuh").read_text(encoding="utf-8")
        runner = (NATIVE_DIR / "gemma4_listener.cu").read_text(encoding="utf-8")
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(
            encoding="utf-8"
        )

        self.assertIn("gemma4_rmsnorm_native_exact_geometry_async", rms_header)
        self.assertIn("geometry_rows <= 4", rms)
        self.assertIn("(rows + 3) / 4", rms)
        self.assertIn("launch_mean_exact<DIM>(squared, row_sum, mean, rows, geometry_rows, stream)", rms)
        self.assertIn("rows, rows, dim, eps, workspace, stream", rms)
        self.assertIn("const uint32_t geometry_rows = reduction_geometry_rows", runner)
        self.assertIn("? reduction_geometry_rows : rows;", runner)
        self.assertIn("GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS) != 0", implementation)
        self.assertIn("GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_LAYERS) != 0", implementation)

        batch_calls = implementation.split("static int gemma4_batch_execute_layer", 1)[1].split(
            "int gemma4_listener_batch_step_async(", 1
        )[0]
        self.assertGreaterEqual(batch_calls.count("GEMMA4_STREAM_Q_LEN) != 0"), 7)

    def test_trace_gemm_reference_uses_legacy_q4_dispatch_and_records_each_split(self):
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        branch = implementation.split(
            "if (record.reference_op == GEMMA4_BATCH_TRACE_REF_GEMM)", 1
        )[1].split("} else if (record.reference_op == GEMMA4_BATCH_TRACE_REF_RMS)", 1)[0]
        self.assertIn("run_linear_cublas(", branch)
        self.assertNotIn("run_linear_gemm_cublas(", branch)
        self.assertIn("gemma4_q4_gemv4_dispatch_eligible(", branch)
        for name in (
            "gemm_reference_wrapper", "q4_gemv4_opt_in",
            "q4_gemv4_split0_eligible", "q4_gemv4_split1_eligible",
            "q4_gemv4_split0_selected", "q4_gemv4_split1_selected",
        ):
            self.assertIn(name, (NATIVE_DIR / "gemma4_listener_microbatch.h").read_text(encoding="utf-8"))
            self.assertIn(name, {field_name for field_name, _ctype in self.binding.CBatchTraceInfo._fields_})
        probe = (ROOT / "duplex_kernel/microbatch/probe_native_listener_microbatch_parity.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"GEMMA4_NATIVE_Q4_GEMV4": os.environ.get("GEMMA4_NATIVE_Q4_GEMV4")', probe)

    def test_trace_frame_pad_captures_two_distinct_source_rows_and_guards_reference(self):
        contract = (NATIVE_DIR / "gemma4_listener_batch_contract.h").read_text(encoding="utf-8")
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        kernels = (NATIVE_DIR / "gemma4_listener_kernels.cuh").read_text(encoding="utf-8")
        self.assertIn("gemma4_batch_trace_frame_pad_source_row", contract)
        self.assertIn("gemma4_batch_trace_frame_pad_input_valid", contract)
        self.assertIn("return token % GEMMA4_BATCH_TRACE_Q_LEN", contract)
        self.assertIn("required_input_elements <= available_input_elements", contract)
        stage = implementation.split(
            "GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_FRAME_PAD", 1
        )[1].split("GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_PAD_RESIDUAL", 1)[0]
        self.assertIn("GEMMA4_STREAM_Q_LEN, ple_width", stage)
        reference = implementation.split(
            "case GEMMA4_BATCH_TRACE_REF_FRAME_PAD:", 1
        )[1].split("case GEMMA4_BATCH_TRACE_REF_PLE_LAYER:", 1)[0]
        self.assertIn("gemma4_batch_trace_frame_pad_input_valid(", reference)
        self.assertLess(reference.index("gemma4_batch_trace_frame_pad_input_valid("),
                        reference.index("kernel_repeat_frame_pad_ple_bf16<<<"))
        self.assertIn("token % GEMMA4_STREAM_Q_LEN", kernels)
        source_rows = (11, 23)
        mapped = [source_rows[token % self.binding.LISTENER_STREAM_Q_LEN] for token in range(8)]
        self.assertEqual(mapped, [11, 23, 11, 23, 11, 23, 11, 23])
        self.assertIn("input_rows != GEMMA4_BATCH_TRACE_Q_LEN", contract)
        self.assertIn("required_input_elements <= available_input_elements", contract)

    def test_trace_is_lazily_allocated_and_only_captures_after_explicit_arm(self):
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        create_source = implementation.split("int gemma4_listener_batch_create(", 1)[1].split(
            "int gemma4_listener_batch_destroy(", 1
        )[0]
        self.assertNotIn("trace_arena", create_source)
        arm_source = implementation.split("int gemma4_listener_batch_trace_arm(", 1)[1].split(
            "static int gemma4_batch_trace_update_attention_layout", 1
        )[0]
        self.assertIn("cudaMalloc((void **)&batch->trace_arena, bytes)", arm_source)
        self.assertIn("!batch->stream_bound || !batch->priority_verified", arm_source)
        contract = (NATIVE_DIR / "gemma4_listener_batch_contract.h").read_text(encoding="utf-8")
        for stage in self.binding.TRACE_STAGE_NAMES:
            self.assertIn(f"GEMMA4_BATCH_TRACE_{stage}", contract)
            self.assertIn(f"GEMMA4_BATCH_TRACE_{stage}", implementation)
        self.assertIn("gemma4_batch_trace_capture_complete", implementation)
        self.assertIn("cudaEventRecord(batch->trace_event, listener_stream)", implementation)
        self.assertIn("batch->trace_state = GEMMA4_BATCH_TRACE_CAPTURED", implementation)
        self.assertIn("run_linear_cublas(batch->model->cublas, record.weight", implementation)
        self.assertIn("run_rmsnorm_exact_or_custom(batch->model", implementation)

    def test_trace_reader_archives_all_raw_inputs_outputs_and_references(self):
        b = self.binding
        trace = object.__new__(b.NativeGemmaListenerMicrobatch)
        trace._closed = False
        trace._poisoned = False
        trace._owner_thread_id = threading.get_ident()
        trace._trace_api_available = True
        trace._ctx = ctypes.c_void_p(1)
        calls = []
        names = b.TRACE_STAGE_NAMES
        bad_frame_pad_capture = False

        def get_info(_ctx, stage_id, info_ptr):
            info = ctypes.cast(info_ptr, ctypes.POINTER(b.CBatchTraceInfo)).contents
            info.struct_size = ctypes.sizeof(b.CBatchTraceInfo)
            index = int(getattr(stage_id, "value", stage_id))
            info.stage_id = index
            info.kind = b.TRACE_STAGE_KINDS[index]
            info.flags = b.TRACE_FLAG_INPUTS_CAPTURED | b.TRACE_FLAG_ACTUAL_CAPTURED
            if info.kind == b.TRACE_KIND_ATTENTION:
                info.flags |= (b.TRACE_FLAG_REFERENCE_DEFERRED |
                               b.TRACE_FLAG_ATTENTION_OUTPUT_ROW0 |
                               b.TRACE_FLAG_ATTENTION_OUTPUT_ROW1)
                info.attention_mask_causal = 1
            else:
                info.flags |= b.TRACE_FLAG_REFERENCE_CAPTURED
            info.row_count = 2
            info.q_len = b.LISTENER_STREAM_Q_LEN
            info.input0_rows = 2
            info.input0_cols = 1
            info.actual_rows = 2
            info.actual_cols = 1
            if names[index] == "AUDIO_PLE_FRAME_PAD":
                info.input0_rows = b.LISTENER_STREAM_Q_LEN
            if info.kind == b.TRACE_KIND_GEMM:
                info.gemm_actual_wrapper = b.TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM
                info.gemm_reference_wrapper = b.TRACE_GEMM_WRAPPER_LEGACY_CUBLAS
                info.q4_gemv4_opt_in = 0
                info.q4_gemv4_split0_eligible = 0
                info.q4_gemv4_split1_eligible = 0
                info.q4_gemv4_split0_selected = 0
                info.q4_gemv4_split1_selected = 0
            if info.kind != b.TRACE_KIND_ATTENTION:
                info.reference_rows = 2
                info.reference_cols = 1
            info.input0_elements = 2
            if names[index] == "AUDIO_PLE_FRAME_PAD":
                info.input0_elements = 1 if bad_frame_pad_capture else b.LISTENER_STREAM_Q_LEN
                if bad_frame_pad_capture:
                    info.input0_rows = 1
            info.input1_elements = 0
            info.actual_elements = 2
            if info.kind != b.TRACE_KIND_ATTENTION:
                info.reference_elements = 2
            return 0

        def read_stage(_ctx, stage_id, input0, input0_bytes, input1, input1_bytes,
                       input2, input2_bytes, actual, actual_bytes, reference, reference_bytes):
            calls.append(int(getattr(stage_id, "value", stage_id)))
            for pointer, size_arg, pattern in ((input0, input0_bytes, b"I"),
                                               (actual, actual_bytes, b"A"),
                                               (reference, reference_bytes, b"R")):
                size = int(getattr(size_arg, "value", size_arg))
                if size == 0:
                    continue
                if size % 4:
                    return -2
                ctypes.memmove(pointer, pattern * size, size)
            if int(getattr(input1_bytes, "value", input1_bytes)) != 0 or \
                    int(getattr(input2_bytes, "value", input2_bytes)) != 0:
                return -3
            return 0

        trace._lib = types.SimpleNamespace(
            gemma4_listener_batch_trace_stage_count=lambda: len(names),
            gemma4_listener_batch_trace_get_stage_info=get_info,
            gemma4_listener_batch_trace_read_stage=read_stage,
        )
        trace._trace_stage_count = len(names)
        records = trace.probe_trace_read_all()
        self.assertEqual(calls, list(range(len(names))))
        self.assertEqual(len(records), len(names))
        self.assertEqual(records[0]["raw"], {
            "input0_bits": b"IIII", "input1_bits": b"", "input2_bits": b"",
            "actual_bits": b"AAAA", "reference_bits": b"RRRR",
        })
        bad_frame_pad_capture = True
        with self.assertRaisesRegex(RuntimeError, "both immutable source rows"):
            trace.probe_trace_read_all()

    def test_trace_feature_symbols_are_configured_only_when_advertised(self):
        api_version = self.binding.BATCH_API_VERSION
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
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                }

            def __getattr__(self, name):
                if name.startswith("gemma4_listener_batch_trace_"):
                    raise AttributeError(name)
                return self.functions.setdefault(name, FakeFunction())

        unadvertised = FakeLibrary(self.base_capabilities())
        features = self.binding._configure_native_api(unadvertised)
        self.assertFalse(features.trace)
        self.assertNotIn("gemma4_listener_batch_trace_api_version", unadvertised.functions)
        advertised = FakeLibrary(self.base_capabilities() | self.binding.BATCH_TRACE_CAPABILITY)
        with self.assertRaises(AttributeError):
            self.binding._configure_native_api(advertised)

    def test_ctypes_sizes_and_field_offsets_match_c_abi(self):
        b = self.binding
        self.assertEqual(ctypes.sizeof(b.CBatchRow), 56)
        self.assertEqual(ctypes.sizeof(b.CBatchConfig), 32)
        self.assertEqual(ctypes.sizeof(b.CBatchRuntimeInfo), 44)
        self.assertEqual(ctypes.sizeof(b.CBatchCounters), 56)
        self.assertEqual(b.CBatchRow.cache_position.offset, 48)
        self.assertEqual(b.CBatchRow.valid_history_tokens.offset, 52)
        self.assertEqual(b.CBatchConfig.listener_stream_priority.offset, 20)
        self.assertEqual(b.CBatchRuntimeInfo.listener_stream_priority.offset, 28)
        self.assertEqual(ctypes.sizeof(b.CBatchTraceInfo), 272)
        self.assertEqual(b.CBatchTraceInfo.attention_is_sliding.offset, 80)
        self.assertEqual(b.CBatchTraceInfo.launch_grid_x.offset, 144)
        self.assertEqual(b.CBatchTraceInfo.gemm_actual_wrapper.offset, 148)
        self.assertEqual(b.CBatchTraceInfo.gemm_reference_wrapper.offset, 152)
        self.assertEqual(b.CBatchTraceInfo.q4_gemv4_split1_selected.offset, 172)
        self.assertEqual(b.CBatchTraceInfo.input0_elements.offset, 208)
        self.assertEqual(b.CBatchRuntimeInfo.max_seq_len.offset, 32)

    def test_snapshot_state_ctypes_layout_is_versioned_and_complete(self):
        b = self.binding
        self.assertEqual(ctypes.sizeof(b.CBatchSessionState), 88)
        self.assertEqual(ctypes.sizeof(b.CBatchProbeInfo), 136)
        self.assertEqual(b.CBatchProbeInfo.forward_generation.offset, 96)
        self.assertEqual(b.CBatchProbeInfo.rms_eps.offset, 128)
        self.assertEqual(b.CBatchSessionState.native_context_id.offset, 8)
        self.assertEqual(b.CBatchSessionState.expected_frame_id.offset, 32)
        self.assertEqual(b.CBatchSessionState.current_position.offset, 60)
        self.assertEqual(b.BATCH_CAPABILITIES_V1, 0x7F)
        self.assertEqual(b.BATCH_DYNAMIC_CAPABILITY, 0x200)
        self.assertEqual(b.BATCH_OUTPUT_HEALTH_CAPABILITY, 0x400)
        self.assertEqual(b.BATCH_PROBE_CAPABILITY, 0x80)
        self.assertEqual(b.BATCH_PROBE_API_VERSION, 1)
        self.assertEqual(b.BATCH_GEMM_GRAPH_CAPABILITY, 0x800)
        self.assertEqual(b.BATCH_LOGICAL_Q_CAPABILITY, 0x2000)
        self.assertEqual(b.BATCH_TOKEN_EMBEDDING_CAPABILITY, 0x4000)

    def test_base_batch_api_accepts_seven_bit_capability_without_probe_exports(self):
        api_version = self.binding.BATCH_API_VERSION
        base_capabilities = self.base_capabilities()
        class FakeFunction:
            def __init__(self, value=0):
                self.value = value
                self.restype = None
                self.argtypes = None

            def __call__(self, *_args):
                return self.value

        class FakeLibrary:
            def __init__(self):
                self.functions = {
                    "gemma4_listener_abi_version": FakeFunction(3),
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(base_capabilities),
                }

            def __getattr__(self, name):
                if name.startswith("gemma4_listener_batch_probe_"):
                    raise AttributeError(name)
                return self.functions.setdefault(name, FakeFunction())

        library = FakeLibrary()
        features = self.binding._configure_native_api(library)
        self.assertFalse(features.probe)
        self.assertNotIn("gemma4_listener_batch_probe_api_version", library.functions)

        batch = object.__new__(self.binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._probe_api_available = features.probe
        calls = []
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_probe_arm=lambda *_args: calls.append("arm") or 0,
        )
        with self.assertRaisesRegex(RuntimeError, "does not advertise the optional"):
            batch.probe_arm()
        self.assertEqual(calls, [])

    def test_pre_v5_or_missing_logical_q_capability_fails_at_startup(self):
        class FakeFunction:
            def __init__(self, value=0):
                self.value = value
                self.restype = None
                self.argtypes = None

            def __call__(self, *_args):
                return self.value

        class FakeLibrary:
            def __init__(self, api_version, capabilities):
                self.functions = {
                    "gemma4_listener_abi_version": FakeFunction(3),
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                }

            def __getattr__(self, name):
                return self.functions.setdefault(name, FakeFunction())

        with self.assertRaisesRegex(RuntimeError, "API version mismatch"):
            self.binding._configure_native_api(FakeLibrary(self.binding.BATCH_API_VERSION - 1, self.base_capabilities()))
        with self.assertRaisesRegex(RuntimeError, "requires logical-Q"):
            self.binding._configure_native_api(
                FakeLibrary(self.binding.BATCH_API_VERSION, 0x67F)
            )

    def test_api5_requires_explicit_model_projected_token_ple_capability(self):
        api_version = self.binding.BATCH_API_VERSION
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
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                }

            def __getattr__(self, name):
                return self.functions.setdefault(name, FakeFunction())

        with self.assertRaisesRegex(RuntimeError, "requires model-projected token PLE"):
            self.binding._configure_native_api(FakeLibrary(
                self.base_capabilities() & ~self.binding.BATCH_TOKEN_EMBEDDING_CAPABILITY
            ))

    def test_probe_exports_are_configured_only_when_feature_bit_is_advertised(self):
        api_version = self.binding.BATCH_API_VERSION
        capabilities = self.base_capabilities() | self.binding.BATCH_PROBE_CAPABILITY
        class FakeFunction:
            def __init__(self, value=0):
                self.value = value
                self.restype = None
                self.argtypes = None

            def __call__(self, *_args):
                return self.value

        class FakeLibrary:
            def __init__(self):
                self.functions = {
                    "gemma4_listener_abi_version": FakeFunction(3),
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                    "gemma4_listener_batch_probe_api_version": FakeFunction(1),
                }

            def __getattr__(self, name):
                return self.functions.setdefault(name, FakeFunction())

        library = FakeLibrary()
        features = self.binding._configure_native_api(library)
        self.assertTrue(features.probe)
        self.assertIsNotNone(library.functions["gemma4_listener_batch_probe_arm"].argtypes)

    def test_gemm_graph_exports_are_optional_and_configured_only_when_advertised(self):
        api_version = self.binding.BATCH_API_VERSION
        base_capabilities = self.base_capabilities()
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
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                }

            def __getattr__(self, name):
                return self.functions.setdefault(name, FakeFunction())

        legacy = FakeLibrary(base_capabilities)
        self.assertFalse(self.binding._configure_native_api(legacy).gemm_graphs)
        self.assertNotIn("gemma4_listener_batch_prepare_gemm_graphs", legacy.functions)

        graph_library = FakeLibrary(base_capabilities | self.binding.BATCH_GEMM_GRAPH_CAPABILITY)
        features = self.binding._configure_native_api(graph_library)
        self.assertTrue(features.gemm_graphs)
        self.assertEqual(
            graph_library.functions["gemma4_listener_batch_prepare_gemm_graphs"].restype,
            ctypes.c_int,
        )
        self.assertEqual(
            graph_library.functions["gemma4_listener_batch_gemm_graph_count"].restype,
            ctypes.c_uint64,
        )

    def test_cache_copy_export_is_optional_and_configured_only_when_advertised(self):
        api_version = self.binding.BATCH_API_VERSION
        base_capabilities = self.base_capabilities()
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
                    "gemma4_listener_batch_api_version": FakeFunction(api_version),
                    "gemma4_listener_batch_capabilities": FakeFunction(capabilities),
                }

            def __getattr__(self, name):
                return self.functions.setdefault(name, FakeFunction())

        legacy = FakeLibrary(base_capabilities)
        self.assertFalse(self.binding._configure_native_api(legacy).cache_copy)
        self.assertNotIn("gemma4_listener_batch_copy_session_cache_async", legacy.functions)

        copy_library = FakeLibrary(base_capabilities | self.binding.BATCH_CACHE_COPY_CAPABILITY)
        features = self.binding._configure_native_api(copy_library)
        self.assertTrue(features.cache_copy)
        function = copy_library.functions["gemma4_listener_batch_copy_session_cache_async"]
        self.assertEqual(function.restype, ctypes.c_int)
        self.assertEqual(len(function.argtypes), 7)

    def test_cache_copy_rows_require_same_logical_boundary_but_allow_stale_cursor(self):
        source = self.Row("session", 4, "epoch-a", 17, 23, 9, 101, 153, 153)
        destination = self.Row("session", 4, "epoch-a", 17, 23, 9, 202, 41, 41)
        self.assertIsNone(self.binding.NativeGemmaListenerMicrobatch._validate_cache_copy_rows(
            source, destination,
        ))

        stale = self.Row("session", 4, "epoch-a", 18, 23, 9, 202, 41, 41)
        with self.assertRaisesRegex(ValueError, "logical boundary"):
            self.binding.NativeGemmaListenerMicrobatch._validate_cache_copy_rows(
                source, stale,
            )
        malformed_cursor = self.Row("session", 4, "epoch-a", 17, 23, 9, 202, 41, 40)
        with self.assertRaisesRegex(ValueError, "valid KV cursors"):
            self.binding.NativeGemmaListenerMicrobatch._validate_cache_copy_rows(
                source, malformed_cursor,
            )

    def test_cache_copy_source_token_is_immutable_and_retains_owner_fence(self):
        row = self.Row("session", 4, "epoch-a", 17, 23, 9, 101, 153, 153)
        state = self.binding.NativeBatchSessionState(
            slot=2,
            native_context_id=101,
            session_generation=4,
            state_epoch_id=11,
            expected_frame_id=17,
            expected_state_version=23,
            expected_sequence_position=9,
            prefix_len=141,
            current_position=153,
            valid_kv_len=153,
            max_seq_len=16_384,
            primed=True,
        )
        owner = object()
        ready_event = self.binding.torch.cuda.Event()
        source = self.binding.NativeCacheCopySource(
            row=row,
            state=state,
            prefix_len=141,
            ready_event=ready_event,
            _owner=owner,
        )
        self.assertIs(source._owner, owner)
        self.assertIs(source.ready_event, ready_event)
        self.assertEqual(source.state.current_position, row.cache_position)
        with self.assertRaises(FrozenInstanceError):
            source.prefix_len = 34

    def test_gemm_graph_preparation_reports_count_and_is_idempotent(self):
        batch = object.__new__(self.binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._gemm_graphs_api_available = True
        batch.gemm_graphs_enabled = False
        batch.gemm_graph_count = 0
        batch.gemm_graph_prepare_ms = 0.0
        batch._active_output_generation = None
        calls = []

        def prepare(_ctx, out_count):
            calls.append("prepare")
            ctypes.cast(out_count, ctypes.POINTER(ctypes.c_uint64))[0] = 17
            return 0

        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_prepare_gemm_graphs=prepare,
        )
        self.assertEqual(batch.prepare_gemm_graphs(), 17)
        self.assertEqual(batch.gemm_graph_count, 17)
        self.assertGreaterEqual(batch.gemm_graph_prepare_ms, 0.0)
        self.assertEqual(batch.prepare_gemm_graphs(), 17)
        self.assertEqual(calls, ["prepare"])

    def test_gemm_graph_preparation_is_explicitly_opt_in(self):
        parameter = inspect.signature(
            self.binding.NativeGemmaListenerMicrobatch.__init__
        ).parameters["enable_gemm_graphs"]
        self.assertIs(parameter.default, False)

    def test_probe_ctypes_layout_and_native_exports_are_additive(self):
        b = self.binding
        self.assertEqual(ctypes.sizeof(b.CBatchProbeInfo), 136)
        self.assertEqual(b.CBatchProbeInfo.forward_generation.offset, 96)
        self.assertEqual(b.CBatchProbeInfo.rms_weight_device_ptr.offset, 104)
        self.assertEqual(b.CBatchProbeInfo.rms_eps.offset, 128)
        header = (NATIVE_DIR / "gemma4_listener_microbatch.h").read_text(encoding="utf-8")
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        for symbol in (
            "gemma4_listener_batch_probe_arm",
            "gemma4_listener_batch_probe_read_rms",
            "gemma4_listener_batch_probe_mark_rms_exact",
            "gemma4_listener_batch_probe_qproj_async",
            "gemma4_listener_batch_probe_read_qproj",
        ):
            self.assertIn(symbol, header)
            self.assertIn(symbol, implementation)
        self.assertIn("UINT64_C(128)", implementation)

    def test_probe_read_copies_full_rms_capture_without_cuda_or_shared_alias(self):
        binding = self.binding
        batch = object.__new__(binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._probe_api_available = True
        batch._last_probe_info = None
        payloads = [bytes([0x10 + index]) * (4 * 1536 * 2) for index in range(3)]

        def read_rms(_ctx, input_ptr, input_len, actual_ptr, actual_len,
                     split_ptr, split_len, info_ptr):
            self.assertEqual((input_len, actual_len, split_len), tuple(map(len, payloads)))
            for ptr, payload in zip((input_ptr, actual_ptr, split_ptr), payloads, strict=True):
                ctypes.memmove(ptr, payload, len(payload))
            info = info_ptr._obj
            info.state = 2
            info.row_count = 4
            info.rms_dim = 1536
            info.rms_backend_id = 2
            info.rms_batched_block_x = 64
            info.rms_batched_block_y = 8
            info.rms_reference_block_x = 128
            info.rms_reference_block_y = 4
            info.forward_generation = 9
            info.rms_weight_device_ptr = 0xC0FFEE
            info.rms_eps = 1e-6
            return 0

        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_probe_read_rms=read_rms,
        )
        metadata, arrays = batch.probe_read_rms()
        self.assertEqual(metadata["row_count"], 4)
        self.assertEqual(metadata["rms_batched_block_x"], 64)
        self.assertEqual(metadata["rms_reference_block_x"], 128)
        self.assertEqual(metadata["forward_generation"], 9)
        self.assertEqual(arrays["input_bits"], payloads[0])
        self.assertEqual(arrays["batched_bits"], payloads[1])
        self.assertEqual(arrays["split_bits"], payloads[2])

    def test_probe_mutation_rejects_non_owner_before_native_call(self):
        binding = self.binding
        calls = []
        batch = object.__new__(binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._probe_api_available = True
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_probe_arm=lambda *_args: calls.append("arm") or 0,
        )
        errors = []

        def arm_from_other_thread():
            try:
                batch.probe_arm()
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=arm_from_other_thread)
        worker.start()
        worker.join()
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)

    def test_probe_is_opt_in_staged_and_does_not_touch_forward_counters(self):
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        layer_start = implementation.index("static int gemma4_batch_execute_layer(")
        layer_end = implementation.index("int gemma4_listener_batch_step_async(", layer_start)
        layer = implementation[layer_start:layer_end]
        self.assertIn("batch->probe_state == GEMMA4_BATCH_PROBE_ARMED", layer)
        self.assertIn("gemma4_batch_probe_mark_rms_captured", layer)
        self.assertIn("GEMMA4_BATCH_PROBE_Q_ROWS", layer)
        q_start = implementation.index("int gemma4_listener_batch_probe_qproj_async(")
        q_end = implementation.index("int gemma4_listener_batch_probe_read_qproj(", q_start)
        qprobe = implementation[q_start:q_end]
        self.assertIn("gemma4_batch_probe_can_run_qproj(batch->probe_state)", qprobe)
        self.assertEqual(qprobe.count("run_linear_cublas("), 3)
        self.assertNotIn("projection_gemm_calls", qprobe)
        self.assertNotIn("layers_executed", qprobe)
        self.assertNotIn("forward_calls", qprobe)

    def test_session_snapshot_rejects_a_different_valid_prefix_length(self):
        binding = self.binding
        row = self.Row(
            session_id="sid", session_generation=1, state_epoch_id="epoch",
            frame_id=0, listener_state_version=0, sequence_position=0,
            native_context_id=19, cache_position=295, valid_history_tokens=295,
        )

        def get_session_state(_ctx, _slot, _row_ptr, state_ptr):
            state = state_ptr._obj
            state.slot = 0
            state.native_context_id = 19
            state.session_generation = 1
            state.state_epoch_id = binding._epoch_id_u64("epoch")
            state.expected_frame_id = 0
            state.expected_state_version = 0
            state.expected_sequence_position = 0
            state.prefix_len = 291  # valid globally, but this session imported 34
            state.current_position = 295
            state.valid_kv_len = 295
            state.max_seq_len = 16384
            state.primed = 1
            return 0

        batch = object.__new__(binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._session_by_id = {
            "sid": binding._SessionBinding(
                slot=0, session_id="sid", native_context_id=19,
                generation=1, epoch_text="epoch",
                epoch_numeric=binding._epoch_id_u64("epoch"),
                expected_frame_id=0, expected_state_version=0,
                expected_sequence_position=0, cache_position=295,
                valid_history_tokens=295, prefix_len=34,
            ),
        }
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_get_session_state=get_session_state
        )

        with self.assertRaisesRegex(RuntimeError, "prefix"):
            batch.session_state(row)

    def test_python_row_mapping_preserves_provenance_and_hashes_epoch(self):
        row = self.Row(
            session_id="opaque-sid",
            session_generation=7,
            state_epoch_id="epoch-9",
            frame_id=0,
            listener_state_version=11,
            sequence_position=2,
            native_context_id=81,
            cache_position=291,
            valid_history_tokens=291,
        )
        mapped = self.binding._row_to_c(row)
        self.assertEqual(mapped.native_context_id, 81)
        self.assertEqual(mapped.session_generation, 7)
        self.assertEqual(mapped.state_epoch_id, self.binding._epoch_id_u64("epoch-9"))
        self.assertEqual(mapped.frame_id, 0)
        self.assertEqual(mapped.listener_state_version, 11)
        self.assertEqual(mapped.sequence_position, 2)
        self.assertEqual(mapped.cache_position, 291)
        self.assertEqual(mapped.valid_history_tokens, 291)
        self.assertNotEqual(mapped.state_epoch_id, 0)

    def test_row_mapping_rejects_boolean_or_overflowing_cursors(self):
        valid = dict(
            session_id="sid", session_generation=1, state_epoch_id="epoch",
            frame_id=1, listener_state_version=1, sequence_position=1,
            native_context_id=1, cache_position=34, valid_history_tokens=34,
        )
        with self.assertRaises(ValueError):
            self.binding._row_to_c(self.Row(**(valid | {"native_context_id": True})))
        with self.assertRaises(ValueError):
            self.binding._row_to_c(self.Row(**(valid | {"cache_position": 1 << 32})))

    def test_ctypes_binds_native_output_rows_as_versioned_provenance(self):
        class Function:
            restype = None
            argtypes = None

        class Library:
            def __init__(self):
                self.functions = {}

            def __getattr__(self, name):
                value = self.functions.setdefault(name, Function())
                return value

        library = Library()
        self.binding._configure_native_signatures(library)
        acquire = library.functions["gemma4_listener_batch_output_acquire"]
        self.assertEqual(len(acquire.argtypes), 6)
        self.assertEqual(
            acquire.argtypes[3], ctypes.POINTER(ctypes.POINTER(self.binding.CBatchRow))
        )
        self.assertEqual(acquire.argtypes[4], ctypes.POINTER(ctypes.c_uint32))
        with self.assertRaises(AttributeError):
            self.binding._configure_native_signatures(types.SimpleNamespace())

    def test_output_health_returns_immutable_row_bound_native_guard_snapshots(self):
        b = self.binding
        row = self.Row(
            session_id="sid", session_generation=7, state_epoch_id="epoch",
            frame_id=4, listener_state_version=9, sequence_position=5,
            native_context_id=81, cache_position=36, valid_history_tokens=36,
        )
        output = b.NativeBatchOutput(
            manifest_id="manifest", generation=1, rows=(row,), tensor=None,
            completion_event=None,
        )
        batch = object.__new__(b.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._last_health_generation = 0
        batch._session_by_id = {
            "sid": b._SessionBinding(
                slot=0, session_id="sid", native_context_id=81, generation=7,
                epoch_text="epoch", epoch_numeric=b._epoch_id_u64("epoch"),
                expected_frame_id=5, expected_state_version=10,
                expected_sequence_position=6, cache_position=38,
                valid_history_tokens=38, prefix_len=34,
            )
        }
        seen = []

        def output_health(_ctx, generation, rows, logical_q_lens, row_count, out_nonfinite):
            seen.append((int(generation.value), int(row_count.value),
                         b._c_row_values(rows[0]), int(logical_q_lens[0])))
            ctypes.cast(out_nonfinite, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0

        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_output_health=output_health,
        )
        snapshots = batch.verify_output_health(output)
        self.assertEqual(seen, [(1, 1, b._c_row_values(b._row_to_c(row)), 2)])
        self.assertEqual(len(snapshots), 1)
        self.assertIs(snapshots[0].row, row)
        self.assertEqual(snapshots[0].status, "healthy")
        self.assertTrue(snapshots[0].checked)
        self.assertFalse(snapshots[0].nonfinite)
        with self.assertRaises((AttributeError, TypeError)):
            snapshots[0].nonfinite = True

    def test_output_health_marks_every_row_nonfinite_on_aggregate_guard_failure(self):
        b = self.binding
        row = self.Row(
            session_id="sid", session_generation=7, state_epoch_id="epoch",
            frame_id=4, listener_state_version=9, sequence_position=5,
            native_context_id=81, cache_position=36, valid_history_tokens=36,
        )
        row2 = self.Row(
            session_id="sid-2", session_generation=8, state_epoch_id="epoch-2",
            frame_id=6, listener_state_version=11, sequence_position=7,
            native_context_id=82, cache_position=36, valid_history_tokens=36,
        )
        batch = object.__new__(b.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._last_health_generation = 0
        batch._session_by_id = {
            "sid": b._SessionBinding(
                slot=0, session_id="sid", native_context_id=81, generation=7,
                epoch_text="epoch", epoch_numeric=b._epoch_id_u64("epoch"),
                expected_frame_id=5, expected_state_version=10,
                expected_sequence_position=6, cache_position=38,
                valid_history_tokens=38, prefix_len=34,
            ),
            "sid-2": b._SessionBinding(
                slot=1, session_id="sid-2", native_context_id=82, generation=8,
                epoch_text="epoch-2", epoch_numeric=b._epoch_id_u64("epoch-2"),
                expected_frame_id=7, expected_state_version=12,
                expected_sequence_position=8, cache_position=38,
                valid_history_tokens=38, prefix_len=34,
            ),
        }
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_output_health=lambda *_args: 1,
        )
        output = b.NativeBatchOutput(
            manifest_id="manifest", generation=1, rows=(row, row2), tensor=None,
            completion_event=None,
        )
        snapshots = batch.verify_output_health(output)
        self.assertEqual(tuple(item.status for item in snapshots),
                         ("nonfinite_guard", "nonfinite_guard"))
        self.assertTrue(all(item.nonfinite for item in snapshots))
        self.assertTrue(batch._poisoned)

    def test_output_health_checks_q0_q1_and_q2_committed_positions(self):
        b = self.binding
        for logical_q_len in (0, 1, 2):
            with self.subTest(logical_q_len=logical_q_len):
                row = self.Row(
                    session_id="sid", session_generation=7, state_epoch_id="epoch",
                    frame_id=4, listener_state_version=9, sequence_position=5,
                    native_context_id=81, cache_position=36, valid_history_tokens=36,
                    logical_q_len=logical_q_len,
                )
                output = b.NativeBatchOutput(
                    manifest_id="manifest", generation=1, rows=(row,), tensor=None,
                    completion_event=None, logical_q_lens=(logical_q_len,),
                )
                batch = object.__new__(b.NativeGemmaListenerMicrobatch)
                batch._owner_thread_id = threading.get_ident()
                batch._closed = False
                batch._poisoned = False
                batch._ctx = ctypes.c_void_p(123)
                batch._active_output_generation = None
                batch._last_health_generation = 0
                batch._session_by_id = {
                    "sid": b._SessionBinding(
                        slot=0, session_id="sid", native_context_id=81, generation=7,
                        epoch_text="epoch", epoch_numeric=b._epoch_id_u64("epoch"),
                        expected_frame_id=5, expected_state_version=10,
                        expected_sequence_position=6,
                        cache_position=36 + logical_q_len,
                        valid_history_tokens=36 + logical_q_len, prefix_len=34,
                    )
                }
                seen = []

                def output_health(_ctx, generation, rows, logical_q_lens, row_count, out_nonfinite):
                    row_values = b._c_row_values(rows[0])
                    q_len = int(logical_q_lens[0])
                    seen.append((int(generation.value), row_values[6], q_len, int(row_count.value)))
                    ctypes.cast(out_nonfinite, ctypes.POINTER(ctypes.c_int))[0] = 0
                    return 0 if row_values[6] + q_len == 36 + logical_q_len else -4

                batch._lib = types.SimpleNamespace(
                    gemma4_listener_batch_output_health=output_health,
                )
                snapshots = batch.verify_output_health(output)
                self.assertEqual(seen, [(1, 36, logical_q_len, 1)])
                self.assertEqual(snapshots[0].status, "healthy")

    def test_embedded_text_output_health_preserves_audio_identity(self):
        b = self.binding
        row = self.Row(
            session_id="sid", session_generation=7, state_epoch_id="epoch",
            frame_id=4, listener_state_version=9, sequence_position=5,
            native_context_id=81, cache_position=36, valid_history_tokens=36,
            logical_q_len=1,
        )
        batch = object.__new__(b.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._last_health_generation = 0
        batch._session_by_id = {
            "sid": b._SessionBinding(
                slot=0, session_id="sid", native_context_id=81, generation=7,
                epoch_text="epoch", epoch_numeric=b._epoch_id_u64("epoch"),
                expected_frame_id=4, expected_state_version=9,
                expected_sequence_position=5, cache_position=37,
                valid_history_tokens=37, prefix_len=34,
            )
        }
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_output_health=lambda _ctx, _gen, _rows, _qlens, _count, out: (
                ctypes.cast(out, ctypes.POINTER(ctypes.c_int)).__setitem__(0, 0) or 0
            ),
        )
        output = b.NativeBatchOutput(
            manifest_id="manifest", generation=1, rows=(row,), tensor=None,
            completion_event=None, logical_q_lens=(1,), embedded_text_input=True,
            preserves_listener_cursors=True,
        )
        result = batch.verify_output_health(output)
        self.assertEqual(result[0].status, "healthy")

    def test_text_append_counter_contract_is_separate_from_audio_ticks(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        step = source.split("static int gemma4_listener_batch_step_core(", 1)[1].split(
            "int gemma4_listener_batch_step_async(", 1
        )[0]
        truncate = source.split("int gemma4_listener_batch_truncate_after_step_async(", 1)[1].split(
            "int gemma4_listener_batch_output_acquire(", 1
        )[0]
        health = source.split("int gemma4_listener_batch_output_health(", 1)[1].split(
            "int gemma4_listener_batch_output_abandon_async(", 1
        )[0]
        self.assertIn("if (embedded_text_input)", step)
        text_path = step.split("if (embedded_text_input) {", 1)[1].split("} else {", 1)[0]
        audio_path = step.split("} else {", 1)[1]
        self.assertIn("per_layer_inputs", text_path)
        self.assertNotIn("gemma4_batch_run_per_session_gemm", text_path)
        self.assertIn("gemma4_batch_run_per_session_gemm", audio_path)
        self.assertIn("output_embedded_text_input = embedded_text_input", step)
        self.assertIn("embedded_text_input && logical_q_len == 0u", truncate)
        self.assertIn("state.expected_frame_id != pre_step_row->frame_id", truncate)
        self.assertIn("preserve_listener_cursors ? preserved_identity_matches : audio_identity_matches", health)
        self.assertIn("batch->output_embedded_text_input = false", health)

    def test_output_health_c_api_is_latest_generation_and_exact_row_bound(self):
        header = (NATIVE_DIR / "gemma4_listener_microbatch.h").read_text(encoding="utf-8")
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        health = source.split("int gemma4_listener_batch_output_health(", 1)[1].split(
            "int gemma4_listener_batch_output_abandon_async(", 1
        )[0]
        self.assertIn("gemma4_listener_batch_output_health", header)
        self.assertIn("completed_output_generation", health)
        self.assertIn("completed_output_row_count", health)
        for field in (
            "native_context_id", "session_generation", "state_epoch_id", "frame_id",
            "listener_state_version", "sequence_position", "cache_position",
            "valid_history_tokens",
        ):
            self.assertIn(f"row.{field} != expected.{field}", health)
        self.assertIn("const uint32_t *logical_q_lens", health)
        self.assertIn("state.current_position != row.cache_position + logical_q_len", health)
        self.assertIn("gemma4_listener_output_health(batch->model, out_nonfinite)", health)
        self.assertIn("UINT64_C(1024)", source)

    def test_capacity_growth_updates_only_after_native_success_and_is_idempotent(self):
        batch = object.__new__(self.binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch.gemm_graphs_enabled = False
        batch.max_batch_size = 4
        calls = []

        def ensure_capacity(_ctx, requested, out_capacity):
            requested = int(requested.value)
            calls.append(requested)
            if requested == 9:
                return -4
            ctypes.cast(out_capacity, ctypes.POINTER(ctypes.c_uint32))[0] = 8
            return 0

        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_ensure_capacity=ensure_capacity,
        )
        self.assertEqual(batch.ensure_capacity(5), 8)
        self.assertEqual(batch.max_batch_size, 8)
        self.assertEqual(batch.ensure_capacity(7), 8)
        self.assertEqual(calls, [5])
        with self.assertRaisesRegex(RuntimeError, "previous capacity 8"):
            batch.ensure_capacity(9)
        self.assertEqual(batch.max_batch_size, 8)
        self.assertEqual(calls, [5, 9])
        batch._active_output_generation = 11
        with self.assertRaisesRegex(RuntimeError, "output lease"):
            batch.ensure_capacity(12)
        with self.assertRaisesRegex(ValueError, "unsigned 32-bit"):
            batch.ensure_capacity(1 << 32)
        self.assertEqual(batch.max_batch_size, 8)
        self.assertEqual(calls, [5, 9])

    def test_graph_growth_failure_preserves_reported_capacity_and_live_cache(self):
        batch = object.__new__(self.binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._poisoned = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch.gemm_graphs_enabled = True
        batch.gemm_graph_count = 17
        batch.max_batch_size = 4
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_ensure_capacity=lambda *_args: -1006,
        )

        with self.assertRaisesRegex(RuntimeError, "previous capacity 4"):
            batch.ensure_capacity(5)
        self.assertEqual(batch.max_batch_size, 4)
        self.assertEqual(batch.gemm_graph_count, 17)
        self.assertTrue(batch.gemm_graphs_enabled)
        self.assertFalse(batch._poisoned)

    def test_c_header_layout_and_new_exports_match_binding(self):
        header = (NATIVE_DIR / "gemma4_listener_microbatch.h").read_text(encoding="utf-8")
        expected = {
            "gemma4_listener_batch_row": [
                "native_context_id", "session_generation", "state_epoch_id", "frame_id",
                "listener_state_version", "sequence_position", "cache_position",
                "valid_history_tokens",
            ],
            "gemma4_listener_batch_runtime_info": [
                "struct_size", "native_abi_version", "batch_api_version", "max_batch_size",
                "supported_batch_mask", "attention_backend_id", "rmsnorm_backend_id",
                "listener_stream_priority", "max_seq_len", "stream_q_len",
                "stream_priority_verified",
            ],
        }
        for struct_name, field_names in expected.items():
            match = re.search(
                rf"typedef struct\s*\{{([^}}]*)\}}\s*{struct_name}\s*;", header, re.S
            )
            self.assertIsNotNone(match, struct_name)
            found = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^]]+\])?\s*;", match[1])
            self.assertEqual(found, field_names)
        exports = (
            "gemma4_listener_batch_create", "gemma4_listener_batch_ensure_capacity",
            "gemma4_listener_batch_destroy", "gemma4_listener_batch_prepare_gemm_graphs",
            "gemma4_listener_batch_gemm_graph_count",
            "gemma4_listener_batch_bind_stream", "gemma4_listener_batch_import_cache_async",
            "gemma4_listener_batch_step_async", "gemma4_listener_batch_output_acquire",
            "gemma4_listener_batch_output_clone_async", "gemma4_listener_batch_output_release",
            "gemma4_listener_batch_output_abandon_async",
            "gemma4_listener_batch_release_session_async", "gemma4_listener_batch_rollover_async",
            "gemma4_listener_batch_get_counters", "gemma4_listener_batch_get_runtime_info",
            "gemma4_listener_batch_get_session_state",
            "gemma4_listener_batch_copy_session_kv_async",
        )
        for name in exports:
            self.assertIn(name, header)
            self.assertIn(name, BINDING_PATH.read_text(encoding="utf-8"))
        self.assertIn("const gemma4_listener_batch_row **out_rows", header)
        self.assertIn("uint32_t *out_row_count", header)

    def test_optional_gemm_graph_path_uses_geometric_lane_buckets(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        listener = (NATIVE_DIR / "gemma4_listener.cu").read_text(encoding="utf-8")
        self.assertIn("UINT64_C(2048)", source)
        helper = source.split("static int gemma4_batch_run_per_session_gemm(", 1)[1].split(
            "static const uint64_t GEMMA4_BATCH_CAPABILITIES_V2", 1
        )[0]
        self.assertIn("cudaGraphLaunch(entry.executable, stream)", helper)
        self.assertIn("entry.weight != weight", helper)
        self.assertIn("operation_id >= GEMMA4_BATCH_GEMM_OP_COUNT", helper)
        self.assertIn("gemma4_batch_graph_bucket_width", helper)
        self.assertIn("gemm_graph_bucket_count", helper)
        capture = source.split("static int gemma4_batch_capture_gemm_graph(", 1)[1].split(
            "static int gemma4_batch_build_gemm_graphs(", 1
        )[0]
        self.assertIn("cudaStreamBeginCapture(", capture)
        self.assertIn("batch->listener_stream, cudaStreamCaptureModeThreadLocal", capture)
        self.assertIn("i < bucket_width", capture)
        self.assertIn("entry.lane_width = bucket_width", capture)
        self.assertIn("cudaStreamWaitEvent(lane.stream, batch->gemm_ready_event", capture)
        self.assertIn("cudaStreamWaitEvent(batch->listener_stream", capture)
        self.assertIn("gemma4_batch_graph_bucket_count(batch->capacity)", source)
        self.assertIn("bucket_index * GEMMA4_BATCH_GEMM_OP_COUNT", source)
        self.assertNotIn("lane->graphs", source)
        self.assertIn("gemma4_batch_clear_graphs(batch)", source)
        self.assertIn("gemma4_batch_prepare_gemm_graphs_internal(batch)", source)
        lane_create = source.split("static int gemma4_batch_create_lane(", 1)[1].split(
            "static int gemma4_batch_resize_capacity(", 1
        )[0]
        self.assertLess(lane_create.index("cublasSetStream"), lane_create.index("cublasSetWorkspace"))
        self.assertIn("GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES", lane_create)
        bound = listener.split("static int run_linear_cublas_bound(", 1)[1].split(
            "/* The microbatch ABI", 1
        )[0]
        self.assertNotIn("cublasSetStream", bound)
        self.assertIn("run_linear_cublas_impl(handle, d_w, d_x, d_y, M, N, K, stream, false)", bound)

    def test_snapshot_export_is_read_only_stream_ordered_and_fails_closed(self):
        header = (NATIVE_DIR / "gemma4_listener_microbatch.h").read_text(encoding="utf-8")
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        self.assertIn("gemma4_listener_batch_session_state", header)
        common = source.split("static bool gemma4_batch_snapshot_row_matches(", 1)[1]
        common = common.split("int gemma4_listener_batch_get_session_state(", 1)[0]
        state = source.split("int gemma4_listener_batch_get_session_state(", 1)[1]
        state = state.split("int gemma4_listener_batch_copy_session_kv_async(", 1)[0]
        copy = source.split("int gemma4_listener_batch_copy_session_kv_async(", 1)[1]
        copy = copy.split("int gemma4_listener_batch_get_counters(", 1)[0]
        self.assertIn("batch->output_lease_active", common)
        for segment in (common, state, copy):
            self.assertNotRegex(segment, r"cuda(?:Device|Stream)Synchronize")
        for segment in (common, state):
            self.assertIn("expected_frame_id", segment)
            self.assertIn("expected_state_version", segment)
            self.assertIn("expected_sequence_position", segment)
        self.assertIn("gemma4_batch_snapshot_row_matches", copy)
        self.assertIn("spec->is_kv_shared", copy)
        self.assertIn("return -1", copy)
        self.assertIn("cudaMemcpyAsync", copy)

    def test_python_manifest_preflight_precedes_native_forward(self):
        source = BINDING_PATH.read_text(encoding="utf-8")
        step = source.split("    def step_async(", 1)[1]
        native_call = step.index("self._lib.gemma4_listener_batch_step_async(")
        self.assertLess(step.index("c_rows = self._preflight_manifest(manifest)"), native_call)
        preflight = source.split("    def _preflight_manifest(", 1)[1]
        self.assertLess(preflight.index("self._assert_open()"), preflight.index("for row in manifest.rows:"))
        self.assertIn("if manifest.runtime_signature != self.runtime_signature", source)
        self.assertIn("row.session_generation != binding.generation", source)
        self.assertIn("row.state_epoch_id != binding.epoch_text", source)
        self.assertIn("manifest.stream_priority != self.capability.stream_priority", source)
        self.assertIn("listener_stream_priority=self._stream_priority", source)
        self.assertIn('"listener_stream_priority": self._stream_priority', source)

    def test_native_shadow_stream_priority_is_exactly_zero_or_primary_minus_one(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        self.assertRegex(
            source,
            r"batch_config->listener_stream_priority != -1\s*&&\s*batch_config->listener_stream_priority != 0",
        )
        self.assertIn("actual_priority != batch->configured_stream_priority", source)

    def test_native_forward_is_layer_wise_batched_and_never_calls_abi3_step(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        self.assertIn("for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer)", source)
        self.assertIn("gemma4_batch_execute_layer(batch, row_count, layer, listener_stream)", source)
        start = source.index("static int gemma4_batch_execute_layer(")
        end = source.index("static int gemma4_listener_batch_step_core(", start)
        layer_body = source[start:end]
        per_session_gemms = re.findall(
            r"gemma4_batch_run_per_session_gemm\((.*?)\)", layer_body, re.S,
        )
        self.assertEqual(len(per_session_gemms), 9)  # Q/K/V/O, gate/up/down, PLE gate/proj
        self.assertNotIn("run_linear_gemm_cublas(", layer_body)
        self.assertIn("run_linear_cublas(lane.handle", source)
        self.assertIn("rows % rows_per_session != 0", source)
        self.assertNotIn("gemma4_listener_step_async(", source)

    def test_embedded_text_path_uses_complete_ple_and_skips_audio_projection(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        core = source.split("static int gemma4_listener_batch_step_core(", 1)[1].split(
            "int gemma4_listener_batch_step_async(", 1
        )[0]
        self.assertIn("GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM", core)
        text_branch = core.split("if (embedded_text_input) {", 1)[1].split("} else {", 1)[0]
        self.assertIn("cudaMemcpyAsync(batch->projected_ple, per_layer_inputs", text_branch)
        self.assertNotIn("frame_pad_ple_ptr", text_branch)
        self.assertNotIn("per_layer_model_proj_w", text_branch)
        self.assertNotIn("per_layer_input_scale", text_branch)
        self.assertIn("run_rmsnorm_exact_or_custom(ctx, batch->hidden_a, ctx->weights.final_norm_w", core)
        self.assertIn("gemma4_listener_batch_step_embedded_async", source)
        self.assertIn("BATCH_TOKEN_EMBEDDING_CAPABILITY", BINDING_PATH.read_text(encoding="utf-8"))

    def test_native_capacity_growth_checks_overflow_and_commits_transactionally(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        resize = source.split("static int gemma4_batch_resize_capacity(", 1)[1].split(
            "static void gemma4_batch_destroy_allocations(", 1
        )[0]
        ensure = source.split("int gemma4_listener_batch_ensure_capacity(", 1)[1].split(
            "uint32_t gemma4_listener_batch_probe_api_version(", 1
        )[0]
        self.assertIn("SIZE_MAX / sizeof(gemma4_listener_batch_slot)", resize)
        self.assertIn("return -20", resize)
        self.assertIn("gemma4_batch_product3(rows", source)
        self.assertIn("gemma4_batch_mul_size((size_t)capacity", source)
        commit = resize.index("batch->slots = new_slots")
        for allocation in (
            "gemma4_batch_workspace_allocate(target_capacity",
            "gemma4_cache_create(&new_slots[i].cache",
            "gemma4_batch_create_lane(&new_lanes[i]",
        ):
            self.assertLess(resize.index(allocation), commit)
        self.assertLess(resize.index("memcpy(new_slots, batch->slots"), commit)
        self.assertLess(resize.index("gemma4_batch_build_gemm_graphs("), commit)
        self.assertLess(resize.index("gemma4_batch_clear_graphs(batch)"), commit)
        self.assertLess(resize.index("gemma4_batch_workspace_free(&old_workspace)", commit), len(resize))
        self.assertLess(ensure.index("batch->output_lease_active"), ensure.index("gemma4_batch_resize_capacity"))
        self.assertLess(ensure.index("cudaStreamSynchronize(batch->listener_stream)"), ensure.index("gemma4_batch_resize_capacity"))
        self.assertLess(ensure.index("cudaStreamSynchronize(batch->gemm_lanes[i].stream)"), ensure.index("gemma4_batch_resize_capacity"))
        self.assertIn("free(new_slots); free(new_active); free(new_rows); free(new_lanes)", resize)

    def test_output_lease_fences_clone_and_returns_native_row_provenance(self):
        source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        self.assertIn("*out_rows = batch->output_rows", source)
        self.assertIn("*out_row_count = batch->output_row_count", source)
        clone_source = source.split("int gemma4_listener_batch_output_clone_async(", 1)[1]
        clone_source = clone_source.split("int gemma4_listener_batch_output_release(", 1)[0]
        self.assertRegex(clone_source, r"row_count\s*!=\s*batch->output_row_count")
        self.assertRegex(
            source,
            r"batch->clone_enqueued\s*\?\s*batch->clone_event\s*:\s*batch->completion_event",
        )

    def test_import_orders_and_retains_original_and_canonical_cache_tensors(self):
        binding = self.binding
        events = []
        active_stream = [None]
        device = types.SimpleNamespace(type="cuda", index=0)

        class FakeStream:
            def __init__(self, handle):
                self.cuda_stream = handle
                self.device = device

            def wait_stream(self, producer):
                events.append(("wait", producer.cuda_stream))

        listener = FakeStream(0xBEEF)
        producer = FakeStream(0xCAFE)

        @contextmanager
        def stream_context(stream):
            active_stream[0] = stream
            events.append(("enter", stream.cuda_stream))
            try:
                yield
            finally:
                events.append(("exit", stream.cuda_stream))
                active_stream[0] = None

        binding.torch.cuda.stream = stream_context

        class FakeTensor:
            def __init__(self, name, dtype, shape, contiguous, strides=None, values=None):
                self.name = name
                self.device = device
                self.dtype = dtype
                self.shape = shape
                self._contiguous = contiguous
                self._strides = strides or (shape[1], 1)
                self.values = values if values is not None else tuple(range(shape[1]))

            def is_contiguous(self):
                return self._contiguous

            def stride(self):
                return self._strides

            def record_stream(self, stream):
                events.append(("record", self.name, stream.cuda_stream))

            def data_ptr(self):
                return id(self)

        prefix_len = 97
        source_k = FakeTensor("source-k", binding.torch.bfloat16, (1, 1, prefix_len, 8), False)
        source_v = FakeTensor("source-v", binding.torch.bfloat16, (1, 1, prefix_len, 8), False)
        canonical_k = FakeTensor("canonical-k", binding.torch.bfloat16, (1, 1, prefix_len, 8), True)
        canonical_v = FakeTensor("canonical-v", binding.torch.bfloat16, (1, 1, prefix_len, 8), True)
        prefix_values = tuple(range(prefix_len))
        # A sliced [1, L] view can be considered contiguous by PyTorch even
        # when the singleton-axis stride is still inherited from [1, 291].
        prefix = FakeTensor(
            "prefix-source", binding.torch.int64, (1, prefix_len), True,
            strides=(291, 1), values=prefix_values,
        )
        canonical_prefix = FakeTensor(
            "prefix-canonical", binding.torch.int64, (1, prefix_len), True,
            strides=(prefix_len, 1), values=prefix_values,
        )
        source_cache = types.SimpleNamespace(
            layers=[types.SimpleNamespace(keys=source_k, values=source_v)],
            shared_layers=None,
        )

        def prepare(_provider, cache, _position):
            self.assertIs(cache, source_cache)
            self.assertIs(active_stream[0], listener)
            events.append(("canonicalize",))
            return binding.Gemma4ImportedCache(), [], [canonical_k, canonical_v]

        def canonicalize_prefix(tensor, *, name, dtype):
            self.assertIs(tensor, prefix)
            self.assertEqual(name, "prefix_attention")
            self.assertEqual(dtype, binding.torch.int64)
            self.assertIs(active_stream[0], listener)
            events.append(("canonicalize-prefix", tensor.name, tensor.stride()))
            return canonical_prefix

        def tensor_meta(tensor, *, dtype_code):
            expected = (tensor.shape[1], 1) if len(tensor.shape) == 2 else None
            if tensor.stride() != expected:
                raise ValueError(
                    f"metadata nativo richiede layout row-major contiguous: "
                    f"{tensor.name} shape={tensor.shape} stride={tensor.stride()} expected={expected}"
                )
            events.append(("metadata", tensor.name, dtype_code))
            return binding.Gemma4TensorMeta()

        native_class = binding.NativeGemmaListenerLM
        had_prepare = hasattr(native_class, "_prepare_imported_cache")
        had_canonicalize = hasattr(native_class, "_canonicalize_cuda_tensor")
        had_meta = hasattr(native_class, "_tensor_meta")
        old_prepare = getattr(native_class, "_prepare_imported_cache", None)
        old_canonicalize = getattr(native_class, "_canonicalize_cuda_tensor", None)
        old_meta = getattr(native_class, "_tensor_meta", None)
        native_class._prepare_imported_cache = staticmethod(prepare)
        native_class._canonicalize_cuda_tensor = staticmethod(canonicalize_prefix)
        native_class._tensor_meta = staticmethod(tensor_meta)

        class Library:
            def gemma4_listener_batch_import_cache_async(self, *_args):
                self_outer = self
                self_outer.assert_stream = active_stream[0]
                events.append(("native-import", active_stream[0].cuda_stream))
                return 0

        batch = object.__new__(binding.NativeGemmaListenerMicrobatch)
        batch._ctx = ctypes.c_void_p(123)
        batch._closed = False
        batch._poisoned = False
        batch._owner_thread_id = threading.get_ident()
        batch._active_output_generation = None
        batch._session_by_id = {}
        batch._session_by_context = {}
        batch._epoch_by_numeric = {}
        batch._import_refs = {}
        batch._cache_provider = object()
        batch._model_device = device
        batch.listener_stream = listener
        batch.max_batch_size = 1
        batch._lib = Library()
        row = self.Row(
            session_id="sid", session_generation=1, state_epoch_id="epoch",
            frame_id=0, listener_state_version=0, sequence_position=0,
            native_context_id=19, cache_position=prefix_len, valid_history_tokens=prefix_len,
        )
        try:
            slot = batch.import_session(
                row, source_cache, prefix, prefix_len=prefix_len, producer_stream=producer,
            )
        finally:
            if had_prepare:
                native_class._prepare_imported_cache = old_prepare
            else:
                del native_class._prepare_imported_cache
            if had_canonicalize:
                native_class._canonicalize_cuda_tensor = old_canonicalize
            else:
                del native_class._canonicalize_cuda_tensor
            if had_meta:
                native_class._tensor_meta = old_meta
            else:
                del native_class._tensor_meta

        self.assertEqual(slot, 0)
        kinds = [entry[0] for entry in events]
        self.assertLess(kinds.index("wait"), kinds.index("enter"))
        self.assertLess(kinds.index("enter"), kinds.index("canonicalize"))
        self.assertLess(kinds.index("canonicalize"), kinds.index("native-import"))
        self.assertLess(kinds.index("wait"), kinds.index("canonicalize-prefix"))
        self.assertLess(kinds.index("enter"), kinds.index("canonicalize-prefix"))
        self.assertLess(kinds.index("canonicalize-prefix"), kinds.index("native-import"))
        self.assertIn(("metadata", "prefix-canonical", 2), events)
        self.assertTrue(prefix.is_contiguous())
        self.assertEqual(prefix.stride(), (291, 1))
        self.assertEqual(canonical_prefix.stride(), (prefix_len, 1))
        self.assertEqual(canonical_prefix.values, prefix.values)
        self.assertEqual(batch._lib.assert_stream, listener)
        retained = batch._import_refs[slot]
        for value in (
            source_cache, source_k, source_v, canonical_k, canonical_v,
            prefix, canonical_prefix,
        ):
            self.assertTrue(any(item is value for item in retained), value)
        recorded = {entry[1] for entry in events if entry[0] == "record"}
        self.assertTrue({
            "source-k", "source-v", "canonical-k", "canonical-v",
            "prefix-source", "prefix-canonical",
        }.issubset(recorded))

    def test_batch_prefix_contract_accepts_any_actual_length_up_to_291(self):
        binding = self.binding
        for prefix_len in (1, 34, 35, 97, 290, 291):
            self.assertIn(prefix_len, binding.BATCH_PREFIX_LENGTHS)
        self.assertNotIn(0, binding.BATCH_PREFIX_LENGTHS)
        self.assertNotIn(292, binding.BATCH_PREFIX_LENGTHS)
        implementation = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(
            encoding="utf-8"
        )
        import_body = implementation.split(
            "int gemma4_listener_batch_import_cache_async(", 1
        )[1].split("/* One session's attention", 1)[0]
        self.assertIn("prefix_len == 0", import_body)
        self.assertIn("prefix_len > GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN", import_body)
        self.assertNotIn("prefix_len != GEMMA4_CANONICAL_PREFIX_LEN", import_body)

    def test_stream_identity_uses_actual_device_and_stream_handle(self):
        identity = self.binding.NativeGemmaListenerMicrobatch._stream_identity

        class Stream:
            def __init__(self, device, handle):
                self.device = device
                self.cuda_stream = handle

        self.assertEqual(
            identity(Stream(types.SimpleNamespace(type="cuda", index=0), 0xBEEF)),
            "cuda:0:stream:0xbeef",
        )
        self.assertNotEqual(
            identity(Stream(types.SimpleNamespace(type="cuda", index=0), 0xBEEF)),
            identity(Stream(types.SimpleNamespace(type="cuda", index=1), 0xBEEF)),
        )
        self.assertNotEqual(
            identity(Stream(types.SimpleNamespace(type="cuda", index=0), 0xBEEF)),
            identity(Stream(types.SimpleNamespace(type="cuda", index=0), 0xCAFE)),
        )
        native = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(encoding="utf-8")
        self.assertRegex(native, r"cudaStreamGetPriority\(stream,\s*&actual_priority\)")
        self.assertIn('"stream_priority_verified": 1', BINDING_PATH.read_text(encoding="utf-8"))

    def test_non_owner_thread_is_rejected_before_native_close_call(self):
        binding = self.binding
        calls = []
        batch = object.__new__(binding.NativeGemmaListenerMicrobatch)
        batch._owner_thread_id = threading.get_ident()
        batch._closed = False
        batch._ctx = ctypes.c_void_p(123)
        batch._active_output_generation = None
        batch._lib = types.SimpleNamespace(
            gemma4_listener_batch_destroy=lambda *_args: calls.append("destroy") or 0
        )
        batch._import_refs = {}
        batch._session_by_id = {}
        batch._session_by_context = {}
        batch._weight_tensors = ()
        batch._language_model = None
        batch._frame_pad_ple = None
        errors = []

        def close_from_other_thread():
            try:
                batch.close()
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=close_from_other_thread)
        worker.start()
        worker.join()
        self.assertEqual(calls, [])
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)

    def test_full_repeat_capacity_guards_single_and_batch_routes(self):
        source = (NATIVE_DIR / "gemma4_listener.cu").read_text(encoding="utf-8")
        batch_source = (NATIVE_DIR / "gemma4_listener_microbatch_impl.inc").read_text(
            encoding="utf-8"
        )
        helper = (NATIVE_DIR / "full_repeat_capacity.h").read_text(encoding="utf-8")

        single_attention = source.split("static int execute_layer(", 1)[1].split(
            "\nstatic int ", 1
        )[0]
        batch_attention = batch_source.split("static int gemma4_batch_attention_one(", 1)[1].split(
            "\nstatic int ", 1
        )[0]
        for route in (single_attention, batch_attention):
            self.assertIn("gemma4_listener_ensure_full_repeat_capacity(", route)
            self.assertLess(
                route.index("gemma4_listener_ensure_full_repeat_capacity("),
                route.index("kernel_repeat_kv_bf16<<<"),
            )

        resize = source.split("static int gemma4_listener_resize_full_repeat_capacity(", 1)[1].split(
            "static int gemma4_listener_ensure_full_repeat_capacity(", 1
        )[0]
        self.assertIn("replace_pair_transactionally", resize)
        self.assertLess(
            resize.index("replace_pair_transactionally"),
            resize.index("ctx->d_sdpa_full_k_repeat[slot] ="),
        )
        self.assertLess(
            resize.index("replace_pair_transactionally"),
            resize.index("ctx->d_sdpa_full_v_repeat[slot] ="),
        )
        self.assertLess(
            resize.index("replace_pair_transactionally"),
            resize.index("ctx->full_repeat_capacity[slot] ="),
        )
        self.assertIn("initial_capacity", source)
        self.assertIn("next_capacity", helper)
        self.assertNotIn("full_repeat_capacity_mutex", source)

        batch_create = batch_source.split("int gemma4_listener_batch_create(", 1)[1].split(
            "int gemma4_listener_batch_destroy(", 1
        )[0]
        self.assertRegex(batch_create, r"gemma4_listener_create_impl\([\s\S]*?,\s*true\)")


if __name__ == "__main__":
    unittest.main()
