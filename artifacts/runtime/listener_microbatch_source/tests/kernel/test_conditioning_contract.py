from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duplex_kernel.conditioning_provenance import (
    CONDITIONING_PATHS,
    PRODUCTION_CONTRACT_ID,
    REQUIRED_TRACE_FIELDS,
    compare_distributions,
    load_conditioning_fixture,
    make_trace_row,
    pcm_frame_provenance,
    run_future_pcm_perturbation_probe,
    summarize_distribution,
    validate_trace_table,
    write_provenance_json,
)
from duplex_kernel.listener import GemmaListener, GemmaListenerState

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "conditioning_contract" / "p7_same_example.json"


def _row(source_path: str, *, hidden: np.ndarray | None = None) -> dict[str, object]:
    return make_trace_row(
        source_path=source_path,
        sample_id="fixture-p7-00",
        frame_id=0,
        timestamp_ns=80_000_000,
        available_at_ns=80_000_000,
        available_through_frame=0,
        token_type="audio_feature_token",
        mode="live_causal_listener" if source_path == CONDITIONING_PATHS["C"] else "offline_full_sequence",
        prompt_visibility="not_visible" if source_path == CONDITIONING_PATHS["C"] else "visible",
        text_visibility="separate_plan_not_in_listener_hidden" if source_path == CONDITIONING_PATHS["C"] else "full_text_visible",
        frontend="fixture_pcm_passthrough",
        hidden_layer="fixture.hidden",
        hidden_stage="pre_causal_conv",
        hidden_kind="h_raw",
        hidden=hidden if hidden is not None else np.ones(4, dtype=np.float32),
        causal_conv_stage="not_applied",
        anchor="none",
        availability_kind="listener_frame_end" if source_path == CONDITIONING_PATHS["C"] else "offline_complete",
        future_pcm_visible=False,
        source_availability_evidence=(
            "AudioFrame.capture_end_ns" if source_path == CONDITIONING_PATHS["C"]
            else "fixture.complete_input_declared_for_offline_diagnostic"
        ),
        evidence_role="fixture_causality_diagnostic" if source_path == CONDITIONING_PATHS["C"] else None,
        model_provenance=(
            {"runtime_profile": "compatibility", "fixture": True}
            if source_path == CONDITIONING_PATHS["C"] else None
        ),
        input_provenance=(
            {
                **pcm_frame_provenance(np.zeros(4, dtype=np.float32), sample_rate_hz=16000),
                "audio_frame_id": 0,
                "listener_output_frame_id": 0,
                "capture_start_ns": 0,
                "capture_end_ns": 80_000_000,
                "assistant_reference_frame_id": None,
                "assistant_reference_emitted_at_ns": None,
            }
            if source_path == CONDITIONING_PATHS["C"] else None
        ),
        generation_id=0 if source_path == CONDITIONING_PATHS["C"] else None,
        segment_id="fixture-segment" if source_path == CONDITIONING_PATHS["C"] else "offline",
    )


def test_production_contract_requires_separate_text_plan_and_zero_lookahead() -> None:
    assert PRODUCTION_CONTRACT_ID == "live_causal_listener_80ms_separate_text_plan_v1"
    assert set(REQUIRED_TRACE_FIELDS) >= {
        "source_path",
        "token_type",
        "mode",
        "prompt_visibility",
        "text_visibility",
        "frontend",
        "hidden_layer",
        "hidden_l2_norm",
        "causal_conv_stage",
        "anchor",
        "timestamp_ns",
        "sample_id",
        "frame_id",
        "availability",
    }


def test_trace_table_rejects_missing_provenance_instead_of_shape_only_acceptance() -> None:
    row = _row(CONDITIONING_PATHS["A"], hidden=np.ones(4, dtype=np.float32))
    validate_trace_table([row])
    incomplete = dict(row)
    incomplete.pop("text_visibility")
    with pytest.raises(ValueError, match="text_visibility"):
        validate_trace_table([incomplete])


def test_validator_rejects_forged_hashes_norms_and_non_hex_lineage() -> None:
    row = _row(CONDITIONING_PATHS["A"])
    forged_hash = dict(row)
    forged_hash["hidden_sha256"] = "g" * 64
    with pytest.raises(ValueError, match="statistiche hidden"):
        validate_trace_table([forged_hash])

    forged_norm = dict(row)
    forged_norm["hidden_l2_norm"] = -1.0
    with pytest.raises(ValueError, match="statistiche hidden"):
        validate_trace_table([forged_norm])


def test_provenance_output_is_no_clobber_and_outside_historical_p7() -> None:
    target = ROOT / ".tmp-conditioning-provenance-test.json"
    try:
        write_provenance_json(target, {"ok": True})
        with pytest.raises(FileExistsError, match="no-clobber"):
            write_provenance_json(target, {"ok": False})
        with pytest.raises(ValueError, match="reports/p7"):
            write_provenance_json(ROOT / "reports" / "p7" / "refuse.json", {"ok": True})
    finally:
        target.unlink(missing_ok=True)


def test_validator_rejects_path_role_timestamp_and_availability_type_mismatches() -> None:
    row = _row(CONDITIONING_PATHS["A"])
    bad_role = dict(row)
    bad_role["evidence_role"] = "microphone_causality_evidence"
    with pytest.raises(ValueError, match="diagnostic offline"):
        validate_trace_table([bad_role])

    bad_timestamp = dict(row)
    bad_timestamp["timestamp_ns"] = "80ms"
    with pytest.raises(ValueError, match="timestamp_ns"):
        validate_trace_table([bad_timestamp])

    bad_availability = dict(row)
    bad_availability["availability"] = dict(row["availability"])
    bad_availability["availability"]["future_pcm_visible"] = "false"
    with pytest.raises(ValueError, match="booleano"):
        validate_trace_table([bad_availability])


def test_same_shape_vectors_are_reported_as_distribution_shift_not_tensor_equality() -> None:
    left = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    right = np.array([[0.0, 2.0], [0.0, 2.0]], dtype=np.float32)
    result = compare_distributions(left, right)
    assert result["shape_equal"] is True
    assert result["tensor_equal"] is False
    assert result["assessment"] == "distribution_only"
    assert result["mean_l2_norm_delta"] > 0.0
    assert summarize_distribution(left)["count"] == 2


def test_fixture_has_one_identical_source_example_for_all_three_paths() -> None:
    fixture = load_conditioning_fixture(FIXTURE)
    assert fixture["sample_id"] == "fixture-p7-00"
    assert fixture["prompt"]
    assert fixture["reference_text"]
    assert len(fixture["pcm_frames"]) == 3
    assert all(len(frame["samples"]) == 1280 for frame in fixture["pcm_frames"])
    assert tuple(fixture["path_sample_ids"].values()) == ("fixture-p7-00",) * 3


class _PCMBackend:
    hidden_size = 4

    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []

    def encode(self, user_pcm, _assistant_reference, _state):
        user = np.asarray(user_pcm, dtype=np.float32).copy()
        self.inputs.append(user)
        return np.array(
            [
                float(user.mean()),
                float(user[0]),
                float(user[-1]),
                float(np.mean(user * user)),
            ],
            dtype=np.float32,
        )


def test_future_pcm_probe_perturbs_source_before_listener_frontend() -> None:
    frames = [
        np.zeros(1280, dtype=np.float32),
        np.full(1280, 0.25, dtype=np.float32),
        np.full(1280, 0.5, dtype=np.float32),
    ]
    backends: list[_PCMBackend] = []

    def factory() -> GemmaListener:
        backend = _PCMBackend()
        backends.append(backend)
        return GemmaListener(backend=backend)

    result = run_future_pcm_perturbation_probe(
        factory,
        frames,
        perturb_frame_id=2,
        perturbation=np.full(1280, 0.125, dtype=np.float32),
    )
    assert result["status"] == "PASS"
    assert result["perturbation_stage"] == "raw_pcm_before_frontend_and_gemma"
    assert result["h_raw_cache_reuse"] is False
    assert result["pre_perturbation_frames_unchanged"] is True
    assert result["future_source_input_changed"] is True
    assert np.array_equal(backends[0].inputs[0], backends[1].inputs[0])
    assert not np.array_equal(backends[0].inputs[2], backends[1].inputs[2])


def test_session_close_fences_outstanding_generation_writes() -> None:
    from duplex_kernel.session import SessionManager

    manager = SessionManager(session_id_factory=lambda: "close-fence-session")
    session = manager.create_session()
    token = session.begin_response("response-1")
    manager.close(session.session_id)
    assert session.is_generation_current(token) is False
    assert session.record_planned_text(token, "stale") is False


def test_production_listener_rejects_text_context_mixed_into_audio_hidden() -> None:
    listener = object.__new__(GemmaListener)
    listener.runtime_profile = "production"
    with pytest.raises(ValueError, match="piano testuale"):
        listener.set_text_context(GemmaListenerState(), [42])


def test_server_trace_binding_is_explicit_for_scheduler_listener() -> None:
    source = (ROOT / "duplex_kernel" / "server.py").read_text(encoding="utf-8")
    assert "scheduler_listener" in source
    assert "make_trace_row" in source
    assert "conditioning_trace_enabled" in source


def test_live_microphone_evidence_does_not_depend_on_git_clean_checkout() -> None:
    from duplex_kernel.server import _listener_evidence_role

    assert _listener_evidence_role("production", native_runtime_enabled=True) == "microphone_causality_evidence"
    assert _listener_evidence_role("production", native_runtime_enabled=False) == "fixture_causality_diagnostic"
    assert _listener_evidence_role("test", native_runtime_enabled=True) == "fixture_causality_diagnostic"


def test_server_fails_closed_when_production_listener_is_missing_or_incompatible() -> None:
    from fastapi.testclient import TestClient
    from duplex_kernel.server import create_app
    from duplex_kernel.session import SessionManager
    from duplex_kernel.turn_policy import NoOpPolicy

    class Engine:
        loaded = True
        device = "cpu"
        enforce_production_listener = True
        default_turn_policy_factory = NoOpPolicy

    with pytest.raises(RuntimeError, match="listener production"):
        with TestClient(create_app(Engine(), SessionManager())).websocket_connect("/v1/realtime?duplex=1"):
            pass

    engine = Engine()
    engine.listener = GemmaListener(backend=_PCMBackend())
    with pytest.raises(RuntimeError, match="profilo incompatibile"):
        with TestClient(create_app(engine, SessionManager())).websocket_connect("/v1/realtime?duplex=1"):
            pass


def test_server_scheduler_listener_emits_live_trace_for_same_fixture_frame() -> None:
    import base64
    import time
    from fastapi.testclient import TestClient
    from duplex_kernel.server import create_app
    from duplex_kernel.session import SessionManager
    from duplex_kernel.turn_policy import NoOpPolicy

    class Backend(_PCMBackend):
        pass

    backend = Backend()
    listener = GemmaListener(backend=backend)

    class Engine:
        loaded = True
        device = "cpu"
        enforce_production_listener = False

    engine = Engine()
    engine.listener = listener
    engine.default_turn_policy_factory = NoOpPolicy
    sessions = SessionManager()
    with TestClient(create_app(engine, sessions)).websocket_connect("/v1/realtime?duplex=1") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type": "session.update", "session": {"turn_policy": "none", "conditioning_trace_enabled": True, "conditioning_trace_sample_id": "fixture-p7-00", "conditioning_trace_stream_id": "fixture-stream-00"}})
        assert ws.receive_json()["type"] == "session.updated"
        session = sessions.all()[0]
        pcm = np.asarray(load_conditioning_fixture(FIXTURE)["pcm_frames"][0]["samples"], dtype="<i2").tobytes()
        ws.send_json({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
        deadline = time.monotonic() + 1.0
        while not session.conditioning_trace and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(session.conditioning_trace) == 1
        assert np.array_equal(backend.inputs[0], np.asarray(load_conditioning_fixture(FIXTURE)["pcm_frames"][0]["samples"], dtype=np.float32) / 32768.0)
        row = session.conditioning_trace[0]
        assert row["source_path"] == CONDITIONING_PATHS["C"]
        assert row["evidence_role"] == "fixture_causality_diagnostic"
        assert row["sample_id"] == "fixture-p7-00"
        assert row["stream_id"] == session.session_id
        assert row["frame_id"] == 0
        assert row["availability"]["future_pcm_visible"] is False
        validate_trace_table(session.conditioning_trace)
        bad_pcm = dict(row)
        bad_pcm["input_provenance"] = dict(row["input_provenance"])
        bad_pcm["input_provenance"]["sample_count"] = 1
        with pytest.raises(ValueError, match="1280"):
            validate_trace_table([bad_pcm])
        bad_state = dict(row)
        bad_state["state_version"] = None
        with pytest.raises(ValueError, match="state_version"):
            validate_trace_table([bad_state])
        ws.send_json({"type": "session.close"})



