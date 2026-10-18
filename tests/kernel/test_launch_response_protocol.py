from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_V2
from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_PROTOCOL_ENV, CONTROLLER_GUIDED_SAY_V1
from duplex_kernel.launch import configure_bundle_runtime_environment
from duplex_kernel.server import create_app


DIRECT_AUDIO_ENV = "DUPLEX_DIRECT_AUDIO_CANDIDATE"
PREFIX_REUSE_ENV = "DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE"
PARTIAL_ASR_ENV = "DUPLEX_NATIVE_PARTIAL_ASR"
TALKER_COMPILED_DECODE_ENV = "DUPLEX_TALKER_COMPILED_DECODE"
TALKER_NATIVE_AUDIO_Q1_ENV = "DUPLEX_TALKER_NATIVE_AUDIO_Q1"
QWEN_NATIVE_LIBRARY_ENV = "QWEN_NATIVE_LIBRARY_PATH"


def _bundle(tmp_path, protocol: str | None) -> str:
    root = tmp_path / "bundle"
    root.mkdir()
    manifest = {"bundle_format": "gemma4_duplex_response"}
    if protocol is not None:
        manifest["response_protocol"] = protocol
    (root / "bundle_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(root)


def test_v2_bundle_selects_direct_audio_and_safe_startup_defaults(tmp_path) -> None:
    env: dict[str, str] = {}

    protocol = configure_bundle_runtime_environment(
        _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2),
        workspace=str(tmp_path),
        environ=env,
    )

    assert protocol == AUDIO_RESPONSE_PROTOCOL_V2
    assert env == {
        AUDIO_RESPONSE_PROTOCOL_ENV: CONTROLLER_GUIDED_SAY_V1,
        DIRECT_AUDIO_ENV: "1",
        PREFIX_REUSE_ENV: "0",
        PARTIAL_ASR_ENV: "0",
        TALKER_COMPILED_DECODE_ENV: "0",
        TALKER_NATIVE_AUDIO_Q1_ENV: "0",
    }


def test_v2_bundle_allows_explicit_compiled_talker_decode(tmp_path) -> None:
    env = {TALKER_COMPILED_DECODE_ENV: "1"}

    protocol = configure_bundle_runtime_environment(
        _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2),
        workspace=str(tmp_path),
        environ=env,
    )

    assert protocol == AUDIO_RESPONSE_PROTOCOL_V2
    assert env[TALKER_COMPILED_DECODE_ENV] == "1"


def test_v2_bundle_native_q1_selects_native_and_library_path(tmp_path) -> None:
    env = {TALKER_NATIVE_AUDIO_Q1_ENV: "1"}

    protocol = configure_bundle_runtime_environment(
        _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2),
        workspace=str(tmp_path),
        environ=env,
    )

    assert protocol == AUDIO_RESPONSE_PROTOCOL_V2
    assert env[TALKER_NATIVE_AUDIO_Q1_ENV] == "1"
    assert env[QWEN_NATIVE_LIBRARY_ENV] == (
        "/workspace/.tmp/native_qwen_audio_q1/libqwen_talker_audio_q1.so"
    )


@pytest.mark.parametrize("value", ["true", "2", ""])
def test_native_q1_flag_is_closed_set(tmp_path, value: str) -> None:
    with pytest.raises(ValueError, match="DUPLEX_TALKER_NATIVE_AUDIO_Q1"):
        configure_bundle_runtime_environment(
            _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2),
            workspace=str(tmp_path),
            environ={TALKER_NATIVE_AUDIO_Q1_ENV: value},
        )


def test_native_q1_rejected_for_legacy_bundle(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires response protocol"):
        configure_bundle_runtime_environment(
            _bundle(tmp_path, None),
            workspace=str(tmp_path),
            environ={TALKER_NATIVE_AUDIO_Q1_ENV: "1"},
        )


@pytest.mark.parametrize("value", ["true", "2", ""])
def test_compiled_talker_decode_flag_is_closed_set(tmp_path, value: str) -> None:
    with pytest.raises(ValueError, match="DUPLEX_TALKER_COMPILED_DECODE"):
        configure_bundle_runtime_environment(
            _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2),
            workspace=str(tmp_path),
            environ={TALKER_COMPILED_DECODE_ENV: value},
        )


def test_compiled_talker_decode_rejected_for_legacy_bundle(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires response protocol"):
        configure_bundle_runtime_environment(
            _bundle(tmp_path, None),
            workspace=str(tmp_path),
            environ={TALKER_COMPILED_DECODE_ENV: "1"},
        )


def test_v2_bundle_rejects_incompatible_direct_or_prefix_cache_overrides(tmp_path) -> None:
    bundle = _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2)

    with pytest.raises(ValueError, match="requires DUPLEX_DIRECT_AUDIO_CANDIDATE=1"):
        configure_bundle_runtime_environment(
            bundle, workspace=str(tmp_path), environ={DIRECT_AUDIO_ENV: "0"},
        )

    with pytest.raises(ValueError, match="requires DUPLEX_AUDIO_TEXT_PREFIX_CACHE_REUSE=0"):
        configure_bundle_runtime_environment(
            bundle, workspace=str(tmp_path), environ={PREFIX_REUSE_ENV: "1"},
        )


def test_legacy_bundle_preserves_existing_runtime_mode(tmp_path) -> None:
    env = {DIRECT_AUDIO_ENV: "0", PREFIX_REUSE_ENV: "1"}

    protocol = configure_bundle_runtime_environment(
        _bundle(tmp_path, None), workspace=str(tmp_path), environ=env,
    )

    assert protocol is None
    assert env == {
        AUDIO_RESPONSE_PROTOCOL_ENV: CONTROLLER_GUIDED_SAY_V1,
        DIRECT_AUDIO_ENV: "0",
        PREFIX_REUSE_ENV: "1",
    }


def test_launch_normalizes_explicit_protocol_and_legacy_precedence(tmp_path) -> None:
    env = {
        AUDIO_RESPONSE_PROTOCOL_ENV: AUDIO_RESPONSE_PROTOCOL_V2,
        "DUPLEX_CONTROLLER_GUIDED_SAY_V1": "1",
    }
    configure_bundle_runtime_environment(
        _bundle(tmp_path, AUDIO_RESPONSE_PROTOCOL_V2), workspace=str(tmp_path), environ=env,
    )
    assert env[AUDIO_RESPONSE_PROTOCOL_ENV] == AUDIO_RESPONSE_PROTOCOL_V2


def test_health_reports_loaded_response_protocol_and_direct_audio_mode(monkeypatch) -> None:
    monkeypatch.setenv(DIRECT_AUDIO_ENV, "1")
    monkeypatch.setenv(AUDIO_RESPONSE_PROTOCOL_ENV, AUDIO_RESPONSE_PROTOCOL_V2)
    monkeypatch.setenv("DUPLEX_CONTROLLER_GUIDED_SAY_V1", "1")
    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH", "0")
    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS", "1")
    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW", "1")
    engine = SimpleNamespace(
        loaded=True,
        audio_response_protocol=AUDIO_RESPONSE_PROTOCOL_V2,
        talker_q1_backend="torch_compile_dynamic",
        talker_q1_compile_status="ready",
        talker_q1_native_status="disabled",
    )

    with TestClient(create_app(engine)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["response_protocol"] == AUDIO_RESPONSE_PROTOCOL_V2
    assert response.json()["direct_audio_candidate"] is True
    assert response.json()["talker_q1_backend"] == "torch_compile_dynamic"
    assert response.json()["talker_q1_compile_status"] == "ready"
    assert response.json()["talker_q1_native_status"] == "disabled"
    microbatch = response.json()["listener_microbatch"]
    assert microbatch["enabled"] is False
    assert microbatch["runtime_ready"] is False
    assert microbatch["gemm_graphs_setting"] == "1"
    assert microbatch["semantic_shadow_setting"] == "1"
    assert microbatch["semantic_shadow_status"] == {}
