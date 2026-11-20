import ast
import math
from contextlib import nullcontext
from pathlib import Path


class _Scalar:
    def __init__(self, value):
        self.value = float(value)

    def __truediv__(self, other):
        return _Scalar(self.value / other)

    def detach(self):
        return self

    def float(self):
        return self

    def reshape(self, *_shape):
        return self

    def cpu(self):
        return self

    def __float__(self):
        return self.value


class _Adapter:
    architecture_id = "frame_residual_q1_v1"
    scale = _Scalar(0.05)
    affective_bank = None


class _StackedScalars:
    def __init__(self, state):
        self.state = state

    def detach(self):
        return self

    def to(self, *, device):
        assert device == "cpu"
        assert self.state["quantum_active"] is True
        self.state["d2h_transfers"] += 1
        return self

    def tolist(self):
        return [16.0, 64.0, 0.05]


class _Quantum:
    def __init__(self, state):
        self.state = state

    def __enter__(self):
        assert self.state["quantum_active"] is False
        self.state["quantum_active"] = True
        return self.state["permit"]

    def __exit__(self, *_exc):
        self.state["quantum_active"] = False


def _run_finalizer(*, permit):
    source_path = Path(__file__).parents[2] / "duplex_kernel" / "engine.py"
    parsed = ast.parse(source_path.read_text(encoding="utf-8"))
    finalizer = next(
        node
        for node in ast.walk(parsed)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "finalize_frame_residual_metrics"
    )
    isolated_module = ast.Module(body=[finalizer], type_ignores=[])
    isolated_module = ast.fix_missing_locations(isolated_module)

    state = {"quantum_active": False, "d2h_transfers": 0, "permit": permit}

    class _Torch:
        @staticmethod
        def stack(values):
            assert len(values) == 3
            return _StackedScalars(state)

        @staticmethod
        def sqrt(value):
            return _Scalar(math.sqrt(value.value))

    metrics = {}
    namespace = {
        "metrics": metrics,
        "frame_residual_adapter": _Adapter(),
        "frame_residual_response_hidden": object(),
        "frame_residual_element_count": 4,
        "frame_residual_delta_sq_sum": _Scalar(16),
        "frame_residual_base_sq_sum": _Scalar(64),
        "frame_residual_calls": 2,
        "frame_residual_q1_steps": 2,
        "frame_residual_tail_hold_calls": 0,
        "live_frame_residual_hidden_stream": None,
        "live_first_packet_hidden": None,
        "live_previous_packet_hidden": None,
        "live_first_packet_hidden_object": None,
        "live_previous_packet_hidden_object": None,
        "live_hidden_trace_tensors": [],
        "listener_compute_quantum": lambda: _Quantum(state),
        "nullcontext": nullcontext,
        "self": type("Engine", (), {"frame_residual_adapter_resolved_path": "candidate.pt"})(),
        "torch": _Torch,
    }
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    namespace["finalize_frame_residual_metrics"]()
    return metrics, state


def test_finalizer_uses_adapter_from_talk_scope_and_holds_compute_quantum():
    metrics, state = _run_finalizer(permit=True)

    assert metrics["frame_residual_adapter"] == "frame_residual_q1_v1"
    assert metrics["frame_residual_checkpoint"] == "candidate.pt"
    assert metrics["frame_residual_calls"] == 2
    assert metrics["frame_residual_rms"] == 2.0
    assert metrics["frame_residual_base_rms"] == 4.0
    assert metrics["frame_residual_rms_to_base_ratio"] == 0.5
    assert state == {"quantum_active": False, "d2h_transfers": 1, "permit": True}


def test_finalizer_skips_d2h_transfer_when_compute_quantum_is_denied():
    metrics, state = _run_finalizer(permit=False)

    assert metrics == {}
    assert state == {"quantum_active": False, "d2h_transfers": 0, "permit": False}
