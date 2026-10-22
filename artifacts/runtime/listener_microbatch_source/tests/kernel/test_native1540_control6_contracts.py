from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from duplex_kernel import native_control6_phase
from duplex_kernel.session import DuplexSession
from duplex_kernel.acoustic_phase_features import (
    ACOUSTIC_PHASE_FEATURE_CONTRACT,
    TICK_SAMPLES,
    acoustic_phase_features,
)
from training.duplex_qlora import native_control6, stage2


def test_native1540_phase_schema_is_distinct_and_uses_causal_pcm_frames() -> None:
    expected_names = (
        "user_rms_log",
        "user_rms_delta",
        "assistant_rms_log",
        "user_minus_assistant_log_ratio",
    )
    assert native_control6_phase.NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES == expected_names
    assert native_control6_phase.native_control6_native1540_phase_t is not (
        native_control6_phase.native_control6_phase_t
    )
    assert hasattr(stage2, "native_control6_native1540_phase_t"), (
        "Stage2 must import the shared native1540 training/runtime phase helper"
    )
    assert stage2.native_control6_native1540_phase_t is (
        native_control6_phase.native_control6_native1540_phase_t
    )

    current = np.full(TICK_SAMPLES, 0.1, dtype=np.float32)
    previous = np.full(TICK_SAMPLES, 0.05, dtype=np.float32)
    assistant_prior = np.full(TICK_SAMPLES, 0.025, dtype=np.float32)
    phase = native_control6_phase.native_control6_native1540_phase_t(
        listener_frame_id=11,
        user_pcm_current=current,
        user_pcm_previous=previous,
        user_previous_frame_id=10,
        assistant_pcm_prior=assistant_prior,
        assistant_prior_frame_id=10,
    )
    assert phase.shape == (4,)
    assert phase[0] == pytest.approx(np.log10(0.1))
    assert phase[2] == pytest.approx(np.log10(0.025))
    assert phase[3] == pytest.approx(np.log10(0.1 / 0.025))

    with pytest.raises(ValueError, match="previous|frame"):
        native_control6_phase.native_control6_native1540_phase_t(
            listener_frame_id=11,
            user_pcm_current=current,
            user_pcm_previous=previous,
            user_previous_frame_id=9,
            assistant_pcm_prior=assistant_prior,
            assistant_prior_frame_id=10,
        )
    with pytest.raises(ValueError, match="prior|frame"):
        native_control6_phase.native_control6_native1540_phase_t(
            listener_frame_id=11,
            user_pcm_current=current,
            user_pcm_previous=previous,
            user_previous_frame_id=10,
            assistant_pcm_prior=assistant_prior,
            assistant_prior_frame_id=9,
        )

    # The established Control6 phase_t ABI keeps its values, names, and causal
    # t-1/t-2 checks while native1540 uses a separate profile.
    old_phase = native_control6_phase.native_control6_phase_t(
        listener_frame_id=10,
        user_pcm_current=np.full(TICK_SAMPLES, 0.1, dtype=np.float32),
        assistant_pcm_prior=np.full(TICK_SAMPLES, 0.01, dtype=np.float32),
        assistant_prior_frame_id=9,
        assistant_pcm_preceding=np.full(TICK_SAMPLES, 0.001, dtype=np.float32),
        assistant_preceding_frame_id=8,
    )
    np.testing.assert_allclose(old_phase, [-2.0, 1.0, 1.0 / 6.0, -1.0], atol=1e-6)

    server_source = (Path(__file__).resolve().parents[2] / "duplex_kernel" / "server.py").read_text(
        encoding="utf-8"
    )
    preparation = server_source.split("def _prepare_policy_tick_state(", 1)[1].split(
        "def _shared_native_policy_context(", 1
    )[0]
    prehead = server_source.split("def _shared_native_policy_context(", 1)[1].split(
        "def scheduler_policy(", 1
    )[0]
    main = server_source.split("def scheduler_policy(", 1)[1]
    assert preparation.count("sess.observe_native_control6_user_pcm(") == 1, (
        "the scheduler preparation must update one shared snapshot at most once per frame"
    )
    for builder, source in (("pre-head", prehead), ("main", main)):
        assert "native_control6_native1540_phase_t(" in source, (
            f"native Control6 {builder} builder must use the shared phase helper"
        )
        assert "prepared[\"native_user_previous_pcm\"]" in source, (
            f"native Control6 {builder} builder must consume the shared causal snapshot"
        )
        assert "native_control_assistant_pcm_prior" in source, (
            f"native Control6 {builder} builder must expose the exact assistant t-1 PCM"
        )
        assert "native_control_assistant_prior_frame_id" in source, (
            f"native Control6 {builder} builder must expose the assistant t-1 frame ID"
        )
        assert "peer_turn_active=peer_turn_active" in source
        assert "sess.observe_native_control6_user_pcm(" not in source


@pytest.mark.parametrize(
    ("user_level", "assistant_level"),
    [
        (0.0, 0.1),
        (0.1, 0.0),
        (native_control6_phase.EPSILON, 0.2),
        (0.2, native_control6_phase.EPSILON),
        (1.0e-7, 0.2),
        (0.2, 1.0e-7),
    ],
)
def test_native1540_ratio_is_unavailable_for_silence_floor(
    user_level: float, assistant_level: float
) -> None:
    phase = native_control6_phase.native_control6_native1540_phase_t(
        listener_frame_id=3,
        user_pcm_current=np.full(1280, user_level, dtype=np.float32),
        user_pcm_previous=None,
        user_previous_frame_id=None,
        assistant_pcm_prior=np.full(1280, assistant_level, dtype=np.float32),
        assistant_prior_frame_id=2,
    )
    assert phase[3] == 0.0


def test_session_native_control6_pcm_snapshot_is_causal_at_most_once_and_resettable() -> None:
    session = DuplexSession(session_id="native1540-test")
    pcm0 = np.full(1280, 0.1, dtype=np.float32)
    pcm1 = np.full(1280, 0.2, dtype=np.float32)
    pcm3 = np.full(1280, 0.3, dtype=np.float32)
    observe = session.observe_native_control6_user_pcm

    previous, previous_frame = observe(0, pcm0)
    assert previous is None and previous_frame is None
    previous, previous_frame = observe(1, pcm1)
    assert previous_frame == 0
    np.testing.assert_array_equal(previous, pcm0)
    # A repeated builder observation of the same frame sees the original
    # previous frame and does not rotate the snapshot a second time.
    repeated_previous, repeated_frame = observe(1, pcm1.copy())
    assert repeated_frame == 0
    np.testing.assert_array_equal(repeated_previous, pcm0)

    previous, previous_frame = observe(3, pcm3)
    assert previous is None and previous_frame is None
    previous, previous_frame = observe(4, pcm1)
    assert previous_frame == 3
    np.testing.assert_array_equal(previous, pcm3)

    session.reset_native_control6_user_pcm()
    previous, previous_frame = observe(5, pcm0)
    assert previous is None and previous_frame is None
    session.close()
    try:
        previous, previous_frame = observe(6, pcm1)
    except RuntimeError as exc:
        assert "closed" in str(exc).lower()
    else:
        assert previous is None and previous_frame is None


def test_native1540_checkpoint_requires_exact_phase_schema(tmp_path) -> None:
    base_rows = torch.zeros((len(native_control6.ACTIONS), native_control6.HIDDEN_WIDTH))
    model = native_control6.NativeControl6LoRA(base_rows, rank=16, alpha=16)
    payload = {
        "schema_version": native_control6.SCHEMA_VERSION,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": native_control6.MODEL_ID,
        "action_to_token_id": dict(native_control6.ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in native_control6.TOKEN_ID_TO_STRING.items()},
        "actions": list(native_control6.ACTIONS),
        "hidden_contract": native_control6.HIDDEN_CONTRACT,
        "base_weight_source": native_control6.BASE_WEIGHT_SOURCE,
        "rank": 16,
        "alpha": 16,
        "base_rows": base_rows,
        "adapter_state_dict": model.adapter_state_dict(),
        "phase_schema": {
            "profile": "native1540",
            "width": 4,
            "names": [
                "user_rms_log",
                "user_rms_delta",
                "assistant_rms_log",
                "user_minus_assistant_log_ratio",
            ],
        },
    }
    checkpoint = tmp_path / "wrong-phase-schema.pt"
    torch.save(payload, checkpoint)
    native_control6.load_native_control6_adapter(checkpoint)

    invalid_schemas = (
        {**payload["phase_schema"], "names": ["legacy", "wrong", "order", "values"]},
        {**payload["phase_schema"], "profile": "legacy"},
        None,
    )
    for invalid_schema in invalid_schemas:
        if invalid_schema is None:
            payload.pop("phase_schema")
        else:
            payload["phase_schema"] = invalid_schema
        torch.save(payload, checkpoint)
        with pytest.raises(ValueError, match="phase|schema|contract"):
            native_control6.load_native_control6_adapter(checkpoint)


def test_stage2_exposes_native1540_profile_without_changing_candidate_c_default() -> None:
    parameters = inspect.signature(stage2.augment_control_features_with_acoustic_phase).parameters
    assert "phase_profile" in parameters
    assert parameters["phase_profile"].default == "candidate_c"


def _stage2_source(
    tmp_path: Path, *, native1540_source: bool = False
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    source = tmp_path / "source.wav"
    feature_dir = tmp_path / "native-features"
    prepared_dir = tmp_path / "prepared"
    feature_dir.mkdir()
    prepared_dir.mkdir()

    pcm = np.zeros((2 * TICK_SAMPLES, 2), dtype=np.float32)
    pcm[:TICK_SAMPLES, 0] = 0.01
    pcm[:TICK_SAMPLES, 1] = 0.2
    pcm[TICK_SAMPLES:, 0] = 0.03
    pcm[TICK_SAMPLES:, 1] = 0.1
    sf.write(source, pcm, 16_000, subtype="FLOAT")
    metadata = [{
        "conversation_id": "native1540-source",
        "source_group_id": "native1540-group",
        "session_id": "train:native1540-source:cluster-0",
        "frame_id": 1,
        "input_frame_id": 1,
        "target_frame_id": 1,
        "listener_frame_id": 1,
        "assistant_reference_frame_id": 0,
        "source": "recorded-test-source",
        "synthetic": False,
        "action": "LISTEN",
        "action_mask": True,
        "target_action_mask": True,
        "event_onset": False,
    }]
    hidden = np.arange(1536, dtype=np.float32).reshape(1, 1536)
    np.savez_compressed(
        feature_dir / "train.npz",
        features=hidden,
        labels=np.asarray([native_control6.ACTIONS.index("LISTEN")], dtype=np.int64),
        frame_ids=np.asarray([1], dtype=np.int64),
        listener_frame_ids=np.asarray([1], dtype=np.int64),
    )
    (feature_dir / "train.json").write_text(json.dumps(metadata), encoding="utf-8")
    source_manifest = {
        "feature_contract": (
            stage2.native_control6_feature_contract(1536)
            if native1540_source
            else stage2.FEATURE_CONTRACT
        ),
        "merged_before_native_listener": not native1540_source,
        "serving_base_only": native1540_source,
        "serving_base_loader": (
            {
                "loader": "duplex_kernel.models.thinker.load_thinker",
                "workspace": "/workspace",
                "model_dtype": "bfloat16",
                "response_lora_applied": False,
                "stage1_adapter_applied": False,
            }
            if native1540_source
            else None
        ),
        "adapter_dir": None if native1540_source else "/models/final-adapter",
        "base_model_path": "/models/base",
    }
    if native1540_source:
        source_manifest["native_control6_extraction"] = {
            "profile": "native1540",
            "hidden_source": "ListenerOutput.gemma_hidden_t",
            "gemma_audio_queries_per_tick": 2,
            "audio_tower_queries_per_tick": 2,
            "mel_window_rows_effective": 240,
            "serving_worker": {
                "worker": "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker",
                "prefix_builder": "duplex_kernel.server._build_unified_listener_prefix_token_ids",
                "runtime_source_root": "/workspace/artifacts/runtime/listener_microbatch_source",
                "worker_factory_source": "/workspace/artifacts/runtime/listener_microbatch_source/duplex_kernel/listener_microbatch_serving.py",
                "prefix_builder_source": "/workspace/artifacts/runtime/listener_microbatch_source/duplex_kernel/server.py",
                "native_library_path": "/workspace/artifacts/runtime/libgemma4_listener_microbatch_prefix512_20261019_build5.so",
                "runtime_info": {
                    "library_path": "/workspace/artifacts/runtime/libgemma4_listener_microbatch_prefix512_20261019_build5.so",
                    "stream_q_len": 2,
                },
                "runtime_signature": {"stream_q_len": 2},
                "prefix": {
                    "contract": "serving_unified_chat_prefix_v2",
                    "length": 190,
                },
            },
        }
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(source_manifest), encoding="utf-8"
    )
    source_row = {
        "conversation_id": "native1540-source",
        "source_group_id": "native1540-group",
        "split": "train",
        "audio_path": str(source),
        "audio_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "sample_rate_hz": 16_000,
        "channels": 2,
        "tick_ms": 80,
        "channel_map": {"0": "assistant", "1": "user"},
        "frames": [{
            "frame_id": 1,
            "start_ms": 80,
            "end_ms": 160,
            "target_action": "LISTEN",
            "action_mask": True,
        }],
        "segments": [],
    }
    (prepared_dir / "train.jsonl").write_text(
        json.dumps(source_row) + "\n", encoding="utf-8"
    )
    return feature_dir, prepared_dir, pcm, hidden


def test_default_stage2_candidate_c_keeps_augmented_feature_bytes_and_lineage(
    tmp_path: Path,
) -> None:
    feature_dir, prepared_dir, pcm, hidden = _stage2_source(tmp_path)
    source_manifest = (feature_dir / "feature_manifest.json").read_bytes()
    output = tmp_path / "candidate-c"
    report = stage2.augment_control_features_with_acoustic_phase(
        native_feature_dir=feature_dir,
        prepared_dir=prepared_dir,
        output_dir=output,
        splits=("train",),
    )

    history = np.zeros(5 * TICK_SAMPLES, dtype=np.float32)
    history[-2 * TICK_SAMPLES :] = pcm[:, 1]
    expected = acoustic_phase_features(
        history,
        pcm[:TICK_SAMPLES, 0],
        sample_rate_hz=16_000,
        tick_ms=80,
    )
    with np.load(output / "train.npz", allow_pickle=False) as values:
        features = values["features"]
    assert features[:, :1536].tobytes() == hidden.tobytes()
    assert features[0, 1536:].tobytes() == expected.tobytes()
    assert report["feature_contract"] == stage2.listener_feature_contract(1540)
    assert report["acoustic_phase_contract"] == ACOUSTIC_PHASE_FEATURE_CONTRACT
    assert report["source_feature_manifest_sha256"] == hashlib.sha256(
        source_manifest
    ).hexdigest()


def test_stage2_native1540_output_preserves_feature_lineage_and_phase_schema(
    tmp_path: Path,
) -> None:
    parameters = inspect.signature(stage2.augment_control_features_with_acoustic_phase).parameters
    assert "phase_profile" in parameters, (
        "Stage2 must expose native1540 as an explicit phase profile"
    )
    names = list(native_control6_phase.NATIVE_CONTROL6_NATIVE1540_PHASE_T_NAMES)
    feature_dir, prepared_dir, pcm, hidden = _stage2_source(
        tmp_path, native1540_source=True
    )
    source_manifest = (feature_dir / "feature_manifest.json").read_bytes()
    output = tmp_path / "native1540"
    report = stage2.augment_control_features_with_acoustic_phase(
        native_feature_dir=feature_dir,
        prepared_dir=prepared_dir,
        output_dir=output,
        splits=("train",),
        phase_profile="native1540",
    )

    expected = native_control6_phase.native_control6_native1540_phase_t(
        listener_frame_id=1,
        user_pcm_current=pcm[TICK_SAMPLES:, 1],
        user_pcm_previous=pcm[:TICK_SAMPLES, 1],
        user_previous_frame_id=0,
        assistant_pcm_prior=pcm[:TICK_SAMPLES, 0],
        assistant_prior_frame_id=0,
    )
    with np.load(output / "train.npz", allow_pickle=False) as values:
        features = values["features"]
    assert features[:, :1536].tobytes() == hidden.tobytes()
    assert features[0, 1536:].tobytes() == expected.tobytes()
    assert report["phase_schema"] == {"profile": "native1540", "width": 4, "names": names}
    assert report["feature_contract"]["input_dim"] == 1540
    assert report["source_feature_manifest_sha256"] == hashlib.sha256(
        source_manifest
    ).hexdigest()
    output_manifest = json.loads((output / "feature_manifest.json").read_text(encoding="utf-8"))
    assert output_manifest["phase_schema"] == report["phase_schema"]
    assert output_manifest["source_feature_manifest_sha256"] == report[
        "source_feature_manifest_sha256"
    ]


def test_legacy_1536_native_adapter_warmstart_zero_pads_phase_columns() -> None:
    model = native_control6.NativeControl6LoRA(
        torch.zeros((len(native_control6.ACTIONS), native_control6.HIDDEN_WIDTH)), rank=2
    )
    legacy_a = torch.ones((2, native_control6.HIDDEN_WIDTH), dtype=torch.float32)
    legacy_b = torch.zeros((len(native_control6.ACTIONS), 2), dtype=torch.float32)

    migrated = model.load_adapter_state_dict({"lora_A": legacy_a, "lora_B": legacy_b})

    assert migrated is True
    assert torch.equal(model.lora_A[:, : native_control6.HIDDEN_WIDTH], legacy_a)
    assert torch.equal(model.lora_A[:, native_control6.HIDDEN_WIDTH :], torch.zeros((2, 4)))
