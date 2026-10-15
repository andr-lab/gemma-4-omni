import ast
import math
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

    def cpu(self):
        return self

    def __float__(self):
        return self.value


class _Adapter:
    architecture_id = "frame_residual_q1_v1"
    scale = _Scalar(0.05)
    affective_bank = None


def test_finalizer_uses_adapter_from_talk_scope_without_implicit_local():
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

    class _Torch:
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
        "frame_residual_tail_hold_calls": 0,
        "self": type("Engine", (), {"frame_residual_adapter_resolved_path": "candidate.pt"})(),
        "torch": _Torch,
    }
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    namespace["finalize_frame_residual_metrics"]()

    assert metrics["frame_residual_adapter"] == "frame_residual_q1_v1"
    assert metrics["frame_residual_checkpoint"] == "candidate.pt"
    assert metrics["frame_residual_calls"] == 2
    assert metrics["frame_residual_rms"] == 2.0
    assert metrics["frame_residual_base_rms"] == 4.0
    assert metrics["frame_residual_rms_to_base_ratio"] == 0.5
