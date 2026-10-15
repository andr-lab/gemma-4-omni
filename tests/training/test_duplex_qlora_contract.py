from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from training.duplex_qlora.control import ACTIONS, ControlHead, load_control_head
from training.duplex_qlora.features import last_audio_query
from training.duplex_qlora.schema import canonicalize_conversation, split_conversations
from duplex_kernel.duplex_control6 import DuplexControl6Policy
from duplex_kernel.turn_policy import ControlAction, PolicyInput


def test_last_audio_query_selects_final_ordered_query_and_rejects_v9_width() -> None:
    queries = torch.arange(4 * 1536, dtype=torch.float32).reshape(1, 4, 1536)

    actual = last_audio_query(queries)

    assert actual.shape == (1536,)
    assert torch.equal(actual, queries[0, 3])
    with pytest.raises(ValueError, match="1536"):
        last_audio_query(torch.zeros(1, 4, 2048))


def test_unknown_actions_and_out_of_range_events_are_masked_without_clipping() -> None:
    conversation = canonicalize_conversation(
        {
            "conversation_id": "conv-1",
            "source_group_id": "group-1",
            "source": "DuplexConv",
            "duration_ms": 160,
            "audio_path": "audio.wav",
            "sample_rate_hz": 16000,
            "channel_map": {"0": "user", "1": "assistant"},
            "segments": [
                {"speaker_id": "speaker-x", "channel": 1, "start_ms": 0, "end_ms": 80, "text": "ciao", "source": "human"}
            ],
            "frames": [
                {"frame_id": 0, "start_ms": 0, "end_ms": 80, "target_action": "UNRESOLVED", "text": "ciao"}
            ],
            "events": [
                {"action": "YIELD", "start_ms": -20, "end_ms": 90, "speaker_id": "speaker-x", "channel": 1, "source": "annotation"}
            ],
        }
    )

    assert conversation["frames"][0]["target_action"] is None
    assert conversation["frames"][0]["action_mask"] is False
    event = conversation["events"][0]
    assert event["start_ms"] == -20 and event["end_ms"] == 90
    assert event["time_mask"] is False
    assert event["action_mask"] is False
    assert conversation["segments"][0]["speaker_id"] == "speaker-x"
    assert conversation["segments"][0]["channel"] == 1
    assert conversation["frames"][0]["text"] == "ciao"


def test_split_keeps_source_groups_and_duplicate_audio_together() -> None:
    rows = [
        {"conversation_id": "a", "source_group_id": "shared", "audio_sha256": "same", "frames": []},
        {"conversation_id": "b", "source_group_id": "shared", "audio_sha256": "other", "frames": []},
        {"conversation_id": "c", "source_group_id": "elsewhere", "audio_sha256": "same", "frames": []},
        {"conversation_id": "d", "source_group_id": "independent", "audio_sha256": "unique", "frames": []},
    ]

    assignments = split_conversations(rows, seed=17)

    assert assignments["a"] == assignments["b"] == assignments["c"]
    assert assignments["d"] in {"train", "validation"}


def test_control_head_persists_exact_six_action_order_and_fails_closed(tmp_path) -> None:
    assert ACTIONS == ("LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN")
    head = ControlHead()
    path = tmp_path / "control_head.pt"
    head.save(path)

    loaded = load_control_head(path)

    assert loaded.input_dim == 1536
    assert loaded.actions == ACTIONS
    with pytest.raises(FileNotFoundError, match="control head"):
        load_control_head(tmp_path / "missing.pt")


def test_control_head_mlp_checkpoint_and_legacy_linear_load(tmp_path) -> None:
    architecture = "mlp_256_gelu_dropout_0.1"
    mlp = ControlHead(architecture=architecture)
    mlp_path = tmp_path / "control_head_mlp.pt"
    mlp.save(mlp_path)
    payload = torch.load(mlp_path, map_location="cpu", weights_only=True)
    assert payload["architecture"] == architecture
    loaded_mlp = load_control_head(mlp_path)
    assert loaded_mlp.architecture == architecture
    assert loaded_mlp(torch.zeros((2, 1536))).shape == (2, len(ACTIONS))

    # Existing linear files predate the explicit architecture field.
    linear = ControlHead()
    legacy_path = tmp_path / "legacy_linear.pt"
    linear.save(legacy_path)
    legacy_payload = torch.load(legacy_path, map_location="cpu", weights_only=True)
    legacy_payload.pop("architecture")
    torch.save(legacy_payload, legacy_path)
    loaded_legacy = load_control_head(legacy_path)
    assert loaded_legacy.architecture == "linear"
    assert loaded_legacy(torch.zeros((1, 1536))).shape == (1, len(ACTIONS))


def test_runtime_control_policy_uses_last_native_query_and_never_maps_hard_stop():
    head = ControlHead()
    with torch.no_grad():
        head.norm.weight.fill_(1.0)
        head.norm.bias.zero_()
        head.classifier.weight.zero_()
        head.classifier.bias.zero_()
        head.classifier.weight[1, 0] = 1.0  # SPEAK only if query index 3 is consumed
    policy = DuplexControl6Policy(
        {
            "run_id": "run-1",
            "base_path": "base",
            "adapter_path": "adapter",
            "control_head": "head",
            "actions": list(ACTIONS),
            "feature_contract": {"queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80, "source": "native_gemma_listener"},
            "head": head,
        }
    )
    queries = torch.zeros((1, 4, 1536))
    queries[0, 3, 0] = 3.0
    state = PolicyInput(
        session_id="session-1",
        frame_id=12,
        user_pcm=np.zeros(1280, dtype=np.float32),
        speaker_active=False,
        generation_id=2,
        listener_audio_queries=queries,
        listener_frame_id=12,
    )

    decision = policy.step(state)

    assert decision.action == ControlAction.SPEAK
    assert decision.listener_frame_id == decision.frame_id == 12
    assert set(ACTIONS).isdisjoint({ControlAction.HARD_STOP.value})


def test_runtime_control_policy_rejects_stale_listener_feature():
    policy = DuplexControl6Policy(
        {
            "run_id": "run-1",
            "base_path": "base",
            "adapter_path": "adapter",
            "control_head": "head",
            "actions": list(ACTIONS),
            "feature_contract": {"queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80, "source": "native_gemma_listener"},
            "head": ControlHead(),
        }
    )
    state = PolicyInput(
        session_id="session-1",
        frame_id=12,
        user_pcm=np.zeros(1280, dtype=np.float32),
        speaker_active=False,
        generation_id=2,
        listener_audio_queries=torch.zeros((4, 1536)),
        listener_frame_id=11,
    )

    with pytest.raises(RuntimeError, match="disallineati"):
        policy.step(state)


def test_calibrated_control_policy_maps_only_low_margin_event_classes() -> None:
    head = ControlHead()
    with torch.no_grad():
        head.norm.weight.fill_(1.0)
        head.norm.bias.zero_()
        head.classifier.weight.zero_()
        head.classifier.bias.copy_(torch.tensor([0.0, 1.0, 0.2, -1.0, -2.0, -3.0]))
    calibration = {
        "schema_version": 1,
        "kind": "shared_non_idle_logit_margin",
        "threshold": 1.365189790725708,
        "baseline_actions": ["LISTEN", "CONTINUE"],
        "eligible_actions": ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
        "source": "validation",
    }
    policy = DuplexControl6Policy({
        "adapter_path": "adapter",
        "control_head": "head",
        "actions": list(ACTIONS),
        "feature_contract": {"queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80, "source": "native_gemma_listener"},
        "head": head,
        "calibration": calibration,
    })
    state = PolicyInput(
        session_id="session-1",
        frame_id=7,
        user_pcm=np.zeros(1280, dtype=np.float32),
        speaker_active=False,
        generation_id=1,
        listener_audio_queries=torch.zeros((1, 4, 1536)),
        listener_frame_id=7,
    )

    decision = policy.step(state)

    assert decision.action == ControlAction.CONTINUE
    assert decision.confidence == pytest.approx(decision.diagnostics["selected_action_probability"])
    assert decision.diagnostics["native_action"] == "SPEAK"
    assert decision.diagnostics["selected_action"] == "CONTINUE"
    assert decision.diagnostics["calibration_applied"] is True
    assert decision.diagnostics["calibration_margin"] == pytest.approx(0.8)
    assert len(decision.diagnostics["native_action_probabilities"]) == len(ACTIONS)


def test_calibration_never_remaps_native_continue() -> None:
    head = ControlHead()
    with torch.no_grad():
        head.norm.weight.fill_(1.0)
        head.norm.bias.zero_()
        head.classifier.weight.zero_()
        head.classifier.bias.copy_(torch.tensor([0.0, 0.1, 0.5, -1.0, -2.0, -3.0]))
    policy = DuplexControl6Policy({
        "adapter_path": "adapter",
        "control_head": "head",
        "actions": list(ACTIONS),
        "feature_contract": {"queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80, "source": "native_gemma_listener"},
        "head": head,
        "calibration": {
            "schema_version": 1,
            "kind": "shared_non_idle_logit_margin",
            "threshold": 1.365189790725708,
            "baseline_actions": ["LISTEN", "CONTINUE"],
            "eligible_actions": ["SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"],
            "source": "validation",
        },
    })
    state = PolicyInput(
        session_id="session-1", frame_id=8, user_pcm=np.zeros(1280, dtype=np.float32),
        speaker_active=False, generation_id=1,
        listener_audio_queries=torch.zeros((1, 4, 1536)), listener_frame_id=8,
    )

    decision = policy.step(state)

    assert decision.action == ControlAction.CONTINUE
    assert decision.diagnostics["native_action"] == "CONTINUE"
    assert decision.diagnostics["calibration_applied"] is False
    assert decision.diagnostics["calibration_margin"] is None
