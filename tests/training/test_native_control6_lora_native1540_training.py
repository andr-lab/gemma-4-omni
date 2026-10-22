from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from duplex_kernel.native_control6_prompt import validate_native_control6_semantic_prefix_metadata
from training.duplex_qlora.native_control6 import (
    ACTIONS,
    ACTION_TO_TOKEN_ID,
    BASE_WEIGHT_SOURCE,
    HIDDEN_CONTRACT,
    MODEL_ID,
    NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
    NATIVE1540_PHASE_SCHEMA,
    SCHEMA_VERSION,
    TOKEN_ID_TO_STRING,
    NativeControl6LoRA,
)
from training.duplex_qlora.features import native_control6_feature_contract
from tools import train_native_control6_lora as trainer


def test_replan_alias_loss_matches_runtime_probability_pool_and_target_remap() -> None:
    logits = torch.tensor(
        [[-1.0, 0.3, 0.2, -2.0, -0.1, 1.0], [1.0, -0.5, 0.4, 0.1, -0.2, -0.7]],
        dtype=torch.float32,
    )
    raw_probabilities = torch.softmax(logits, dim=-1)
    physical_probabilities = torch.softmax(trainer._effective_logits(logits, True), dim=-1)
    expected = torch.stack(
        (
            raw_probabilities[:, ACTIONS.index("LISTEN")],
            raw_probabilities[:, ACTIONS.index("SPEAK")],
            raw_probabilities[:, ACTIONS.index("CONTINUE")],
            raw_probabilities[:, ACTIONS.index("YIELD")] + raw_probabilities[:, ACTIONS.index("REPLAN")],
            raw_probabilities[:, ACTIONS.index("BACKCHANNEL")],
        ),
        dim=-1,
    )
    torch.testing.assert_close(physical_probabilities, expected)

    target = torch.tensor([ACTIONS.index("REPLAN"), ACTIONS.index("BACKCHANNEL"), -100])
    assert trainer._effective_targets(target, True).tolist() == [
        ACTIONS.index("YIELD"),
        ACTIONS.index("BACKCHANNEL"),
        -100,
    ]
    torch.testing.assert_close(
        torch.nn.functional.cross_entropy(trainer._effective_logits(logits[:1], True), torch.tensor([3])),
        torch.nn.functional.cross_entropy(
            trainer._effective_logits(logits[:1], True),
            trainer._effective_targets(torch.tensor([ACTIONS.index("REPLAN")]), True),
        ),
    )


def test_runtime_effective_loss_uses_shared_context_and_full_sixway_denominator() -> None:
    logits = torch.tensor([[0.1, 1.2, -0.4, 0.7, 1.5, 0.3]], requires_grad=True)
    speaker = torch.tensor([True])
    user = torch.tensor([True])
    bucket_log_mass = trainer._runtime_effective_log_probs(logits, speaker, user)
    raw_probability = torch.softmax(logits, dim=-1)
    expected = torch.stack(
        (
            raw_probability[:, ACTIONS.index("CONTINUE")],
            torch.zeros(1),
            raw_probability[:, ACTIONS.index("LISTEN")]
            + raw_probability[:, ACTIONS.index("YIELD")]
            + raw_probability[:, ACTIONS.index("REPLAN")],
            torch.zeros(1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(bucket_log_mass.exp(), expected)
    assert float(bucket_log_mass.exp().sum().detach()) < 1.0

    target = trainer._runtime_effective_target_ids(
        torch.tensor([ACTIONS.index("YIELD")]), speaker, user
    )
    assert target.tolist() == [trainer.RUNTIME_BUCKETS.index("YIELD")]
    exact_loss = torch.logsumexp(logits, dim=-1) - torch.logsumexp(
        logits[:, [0, 3, 5]], dim=-1
    )
    observed_loss = -bucket_log_mass.gather(1, target.unsqueeze(1)).squeeze(1)
    torch.testing.assert_close(observed_loss, exact_loss)
    observed_loss.sum().backward()
    assert logits.grad is not None
    assert logits.grad[0, ACTIONS.index("SPEAK")] > 0
    assert logits.grad[0, ACTIONS.index("BACKCHANNEL")] > 0

    contexts_speaker = torch.tensor([False, False, True, True, True])
    contexts_user = torch.tensor([False, True, False, True, True])
    incompatible = torch.tensor([
        ACTIONS.index("YIELD"),
        ACTIONS.index("BACKCHANNEL"),
        ACTIONS.index("SPEAK"),
        ACTIONS.index("BACKCHANNEL"),
        ACTIONS.index("REPLAN"),
    ])
    mapped = trainer._runtime_effective_target_ids(incompatible, contexts_speaker, contexts_user)
    assert mapped.tolist() == [-100, 3, -100, -100, trainer.RUNTIME_BUCKETS.index("YIELD")]


def test_balanced_runtime_bucket_batches_use_exact_real_mix_without_inverse_weights() -> None:
    pools = {
        bucket: {
            "real": [(bucket, "real", index) for index in range(3)],
            "synthetic": [(bucket, "synthetic", index) for index in range(2)],
        }
        for bucket in trainer.RUNTIME_BUCKETS
    }
    rng = random.Random(44)
    aggregate = {"real": 0, "synthetic": 0}
    for batch_index in range(5):
        batch = trainer._sample_runtime_bucket_source_batch(
            pools, batch_size=128, rng=rng, batch_index=batch_index
        )
        assert len(batch) == 128
        per_bucket = {
            bucket: [entry for entry in batch if entry[0] == bucket]
            for bucket in trainer.RUNTIME_BUCKETS
        }
        assert {bucket: len(entries) for bucket, entries in per_bucket.items()} == {
            bucket: 32 for bucket in trainer.RUNTIME_BUCKETS
        }
        for bucket, entries in per_bucket.items():
            sources = [entry[1] for entry in entries]
            expected_real = [26, 25, 26, 25, 26][batch_index]
            assert sources.count("real") == expected_real
            assert sources.count("synthetic") == 32 - expected_real
        for _, source, _ in batch:
            aggregate[source] += 1
    assert aggregate == {"real": 512, "synthetic": 128}

    log_probs = torch.log_softmax(
        torch.tensor([[0.1, -0.3, 0.2, -0.1], [0.4, 0.2, -0.5, 0.0]]), dim=-1
    )
    targets = torch.tensor([0, 2])
    expected = -log_probs.gather(1, targets[:, None]).mean()
    torch.testing.assert_close(
        trainer._unweighted_runtime_bucket_nll(log_probs, targets), expected
    )


def test_hold_raw_aux_changes_listen_continue_split_without_changing_hold_mass() -> None:
    listen = ACTIONS.index("LISTEN")
    continue_ = ACTIONS.index("CONTINUE")
    first = torch.tensor([[1.0, -2.0, 0.0, -1.0, -3.0, -4.0]], requires_grad=True)
    second = torch.tensor([[0.0, -2.0, 1.0, -1.0, -3.0, -4.0]], requires_grad=True)
    target = torch.tensor([listen])
    supervised = torch.tensor([True])
    speaker = torch.tensor([False])
    user = torch.tensor([False])

    first_physical = -trainer._runtime_effective_log_probs(first, speaker, user)[
        0, trainer.RUNTIME_BUCKETS.index("HOLD")
    ]
    second_physical = -trainer._runtime_effective_log_probs(second, speaker, user)[
        0, trainer.RUNTIME_BUCKETS.index("HOLD")
    ]
    torch.testing.assert_close(
        torch.logsumexp(first[:, [listen, continue_]], dim=-1),
        torch.logsumexp(second[:, [listen, continue_]], dim=-1),
    )
    torch.testing.assert_close(first_physical, second_physical)

    first_aux, first_eligible = trainer._hold_raw_aux_loss(
        first, target, supervised, speaker, user
    )
    second_aux, second_eligible = trainer._hold_raw_aux_loss(
        second, target, supervised, speaker, user
    )
    assert first_eligible.tolist() == second_eligible.tolist() == [True]
    assert first_aux.item() != pytest.approx(second_aux.item())
    torch.testing.assert_close(first_aux, torch.nn.functional.cross_entropy(first, target))


def test_hold_raw_aux_excludes_yield_context_listen_yield_replan_pool() -> None:
    logits = torch.tensor(
        [[0.1, 1.2, -0.4, 0.7, 1.5, 0.3]] * 3,
        dtype=torch.float32,
        requires_grad=True,
    )
    target = torch.tensor([ACTIONS.index(action) for action in ("LISTEN", "YIELD", "REPLAN")])
    supervised = torch.ones(3, dtype=torch.bool)
    speaker = torch.ones(3, dtype=torch.bool)
    user = torch.ones(3, dtype=torch.bool)

    bucket_log_mass = trainer._runtime_effective_log_probs(logits, speaker, user)
    expected_yield_mass = torch.logsumexp(
        logits[:, [ACTIONS.index("LISTEN"), ACTIONS.index("YIELD"), ACTIONS.index("REPLAN")]],
        dim=-1,
    ) - torch.logsumexp(logits, dim=-1)
    torch.testing.assert_close(
        bucket_log_mass[:, trainer.RUNTIME_BUCKETS.index("YIELD")], expected_yield_mass
    )
    aux_loss, eligible = trainer._hold_raw_aux_loss(
        logits, target, supervised, speaker, user
    )
    assert eligible.tolist() == [False, False, False]
    assert torch.isfinite(aux_loss)
    assert aux_loss.item() == 0.0


def test_hold_raw_aux_without_eligible_rows_is_differentiable_zero() -> None:
    logits = torch.randn(2, len(ACTIONS), requires_grad=True)
    targets = torch.tensor([ACTIONS.index("YIELD"), ACTIONS.index("SPEAK")])
    supervised = torch.tensor([True, False])
    speaker = torch.tensor([True, False])
    user = torch.tensor([True, False])

    aux_loss, eligible = trainer._hold_raw_aux_loss(
        logits, targets, supervised, speaker, user
    )
    assert eligible.tolist() == [False, False]
    assert torch.isfinite(aux_loss)
    assert aux_loss.item() == 0.0
    aux_loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -0.1])
def test_hold_raw_aux_cli_rejects_nonfinite_or_negative_weight(invalid: float) -> None:
    parser = trainer._build_parser()
    default_args = parser.parse_args(["train", "--output", "unused.pt"])
    assert default_args.hold_raw_aux_weight == 0.0
    with pytest.raises(SystemExit):
        parser.parse_args([
            "train", "--output", "unused.pt", "--hold-raw-aux-weight", str(invalid)
        ])


def test_hold_raw_aux_cli_rejects_nonzero_weight_outside_supported_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        trainer.main([
            "train", "--output", str(tmp_path / "unused.pt"),
            "--hold-raw-aux-weight", "0.1",
        ])
    assert "requires --phase-profile native1540 and --loss-space runtime_effective" in capsys.readouterr().err


def _write_split(feature_dir: Path, split: str, group: str, conversation: str) -> None:
    rng = np.random.default_rng(21 if split == "train" else 22)
    actions = list(ACTIONS)
    hidden = rng.normal(0, 0.3, size=(len(actions), 1536)).astype(np.float32)
    phase = np.asarray(
        [[-1.0, -0.2, -2.0, 0.3], [-0.8, 0.1, -1.0, 0.5], [-0.5, 0.3, -3.0, -0.1],
         [-0.3, -0.1, -1.5, 0.2], [-1.2, 0.2, -0.7, -0.3], [-0.6, 0.0, -1.7, 0.4]],
        dtype=np.float32,
    )
    features = np.concatenate((hidden, phase), axis=1)
    labels = np.asarray([ACTIONS.index(action) for action in actions], dtype=np.int64)
    frame_ids = np.arange(len(actions), dtype=np.int64)
    metadata = [
        {
            "conversation_id": conversation,
            "source_group_id": group,
            "split": split,
            "frame_id": int(frame),
            "listener_frame_id": int(frame),
            "source": "authored_synthetic_audio_test",
            "training_state_context": NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
            "audio_sha256": f"{split}-sha-{frame}",
            "action": action,
            "action_mask": True,
            "target_action_mask": True,
            "speaker_active_at_frame_start": action in {"CONTINUE", "YIELD", "REPLAN"},
            "native_control6_user_activity_detected": action in {"LISTEN", "CONTINUE", "YIELD", "REPLAN"},
            "native_control6_context": {
                "schema": "native_control6_runtime_context_v1",
                "source_kind": "synthetic_trace",
            },
        }
        for frame, action in enumerate(actions)
    ]
    np.savez_compressed(
        feature_dir / f"{split}.npz",
        features=features,
        labels=labels,
        frame_ids=frame_ids,
        listener_frame_ids=frame_ids,
    )
    (feature_dir / f"{split}.json").write_text(json.dumps(metadata), encoding="utf-8")


def _write_legacy_warmstart(path: Path, *, rank: int = 2) -> None:
    base_rows = torch.zeros((len(ACTIONS), 1536), dtype=torch.float32)
    model = NativeControl6LoRA(base_rows, rank=rank, alpha=float(rank))
    with torch.no_grad():
        model.lora_A[:, :1536].normal_(0.0, 0.02)
        model.lora_B.normal_(0.0, 0.03)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "architecture": "native_gemma_selected_output_rows_lora",
        "model_id": MODEL_ID,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {str(k): v for k, v in TOKEN_ID_TO_STRING.items()},
        "actions": list(ACTIONS),
        "hidden_contract": HIDDEN_CONTRACT,
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "rank": rank,
        "alpha": float(rank),
        "base_rows": base_rows,
        "adapter_state_dict": {
            "lora_A": model.lora_A[:, :1536].detach().cpu(),
            "lora_B": model.lora_B.detach().cpu(),
        },
        "report": {"training_fixture": True},
    }
    torch.save(payload, path)


def test_native1540_training_warmstarts_legacy_tracks_provenance_and_learnability(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features-native1540"
    feature_dir.mkdir()
    _write_split(feature_dir, "train", "train-group", "train-conversation")
    _write_split(feature_dir, "validation", "validation-group", "validation-conversation")
    manifest = {
        "feature_contract": native_control6_feature_contract(1540),
        "merged_before_native_listener": False,
        "serving_base_only": True,
        "adapter_dir": None,
        "serving_base_loader": {
            "loader": "duplex_kernel.models.thinker.load_thinker",
            "workspace": "/workspace",
            "model_dtype": "bfloat16",
            "response_lora_applied": False,
            "stage1_adapter_applied": False,
        },
        "native_control6_extraction": {
            "profile": "native1540",
            "hidden_source": "ListenerOutput.gemma_hidden_t",
            "gemma_audio_queries_per_tick": 2,
            "audio_tower_queries_per_tick": 2,
            "mel_window_rows_effective": 240,
            "serving_worker": {
                "worker": "duplex_kernel.listener_microbatch_serving.make_listener_microbatch_worker",
                "prefix_builder": "duplex_kernel.server._build_unified_listener_prefix_token_ids",
                "runtime_source_root": "/runtime/listener_microbatch_source",
                "prefix_builder_source": "/runtime/listener_microbatch_source/duplex_kernel/server.py",
                "worker_factory_source": "/workspace/duplex_kernel/listener_microbatch_serving.py",
                "native_library_path": "/runtime/libnative-cap512.so",
                "runtime_info": {"stream_q_len": 2, "library_path": "/runtime/libnative-cap512.so"},
                "runtime_signature": {"stream_q_len": 2},
                "prefix": {
                    "contract": "serving_unified_chat_prefix_v2",
                    "history_messages": 0,
                    "session_instructions": "empty_session_default",
                    "token_ids": [1] * 189 + [256000],
                    "length": 190,
                },
            },
        },
        "phase_profile": "native1540",
        "phase_schema": NATIVE1540_PHASE_SCHEMA,
        "training_state_context": NATIVE_CONTROL6_TRAINING_STATE_CONTEXT,
        "test_data_read": False,
        "test_used_for_training_or_calibration": False,
    }
    (feature_dir / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    wrong_context_manifest = dict(manifest)
    wrong_context_manifest["training_state_context"] = "assistant_prefix_q2"
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(wrong_context_manifest), encoding="utf-8"
    )
    _, descriptive_context_info = trainer.load_stage2_feature_sequences(feature_dir, "train")
    assert descriptive_context_info["training_state_context"] == "assistant_prefix_q2"
    assert descriptive_context_info["training_state_context_row_values"] == [
        NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
    ]
    (feature_dir / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    manifest_without_context = dict(manifest)
    manifest_without_context.pop("training_state_context")
    metadata_path = feature_dir / "train.json"
    original_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_without_context = [dict(row) for row in original_metadata]
    for row in metadata_without_context:
        row.pop("training_state_context", None)
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(manifest_without_context), encoding="utf-8"
    )
    metadata_path.write_text(json.dumps(metadata_without_context), encoding="utf-8")
    _, unspecified_context_info = trainer.load_stage2_feature_sequences(feature_dir, "train")
    assert unspecified_context_info["training_state_context"] == "unspecified"
    assert unspecified_context_info["training_state_context_row_values"] == ["unspecified"]
    (feature_dir / "feature_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metadata_path.write_text(json.dumps(original_metadata), encoding="utf-8")

    derived_metadata = [dict(row) for row in original_metadata]
    speak_index = next(index for index, row in enumerate(derived_metadata) if row["action"] == "SPEAK")
    derived_metadata[speak_index].update({
        "speaker_active_at_frame_start": True,
        "native_control6_runtime_target_action": "CONTINUE",
        "native_control6_runtime_target_derived": True,
        "native_control6_runtime_target_reason": "same_active_speak_event_continues",
    })
    metadata_path.write_text(json.dumps(derived_metadata), encoding="utf-8")
    derived_train, derived_info = trainer.load_stage2_feature_sequences(feature_dir, "train")
    derived_sequence = derived_train[0]
    assert derived_sequence["target_ids"][speak_index].item() == ACTIONS.index("SPEAK")
    assert derived_sequence["runtime_target_ids"][speak_index].item() == ACTIONS.index("CONTINUE")
    assert derived_info["action_counts"]["SPEAK"] == 1
    assert derived_info["runtime_target_action_counts"]["SPEAK"] == 0
    assert derived_info["runtime_target_action_counts"]["CONTINUE"] == 2
    effective_target = trainer._runtime_effective_target_ids(
        derived_sequence["runtime_target_ids"],
        derived_sequence["speaker_active"],
        derived_sequence["user_activity_detected"],
    )
    assert effective_target[speak_index].item() == trainer.RUNTIME_BUCKETS.index("HOLD")
    derived_aux_counts = trainer._hold_raw_aux_eligible_counts(derived_train)
    assert derived_aux_counts["eligible_train_rows_by_source_and_raw_action"] == {
        "authored_synthetic_audio_test": {"LISTEN": 1, "CONTINUE": 2}
    }

    unknown_metadata = [dict(row) for row in original_metadata]
    unknown_metadata[speak_index]["speaker_active_at_frame_start"] = None
    unknown_metadata[speak_index]["native_control6_context_valid"] = False
    metadata_path.write_text(json.dumps(unknown_metadata), encoding="utf-8")
    unknown_train, unknown_info = trainer.load_stage2_feature_sequences(feature_dir, "train")
    unknown_sequence = unknown_train[0]
    assert unknown_sequence["target_ids"][speak_index].item() == ACTIONS.index("SPEAK")
    assert unknown_sequence["runtime_target_ids"][speak_index].item() == -100
    assert unknown_sequence["loss_mask"][speak_index].item() is True
    assert unknown_info["context_unsupported_targets_by_source_action"][
        "authored_synthetic_audio_test:SPEAK:pre_action_context_unknown"
    ] == 1
    metadata_path.write_text(json.dumps(original_metadata), encoding="utf-8")

    legacy_checkpoint = tmp_path / "native-control6-legacy.pt"
    _write_legacy_warmstart(legacy_checkpoint)
    output = tmp_path / "native-control6-native1540.pt"

    report = trainer.train_adapter(
        output_path=output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        replan_target_alias_to_yield=True,
        steps=4,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=0.02,
        seed=17,
        device_name="cpu",
    )

    assert report["native_control6_phase_profile"] == "native1540"
    assert report["phase_schema"] == NATIVE1540_PHASE_SCHEMA
    assert report["training_state_context"] == NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
    assert report["semantic_prefix_schema_version"] == 2
    assert report["semantic_prefix_contract"] == "gemma_native_control6_semantic_prefix_boa_v2"
    assert report["semantic_prefix_context_conditioned"] is True
    assert report["semantic_prefix_capacity"] == report["semantic_prefix_max_tokens"] == 190
    assert report["semantic_prefix_actual_prefix_tokens"] == 190
    assert report["semantic_prefix_runtime_import_capacity_tokens"] == 512
    assert report["semantic_prefix_canonical_prime_target_tokens"] == 291
    assert report["semantic_prefix_history_messages"] == 0
    assert report["semantic_prefix_boa_token_id"] == 256000
    assert report["initialization"]["phase_profile"] == "legacy"
    assert report["initialization"]["phase_t_checkpoint_migration_details"]["phase_columns_trained"] is False
    assert report["target_remapping"]["remapped_train_target_rows"] == 1
    assert report["target_remapping"]["dataset_action_provenance_preserved"] is True
    assert report["target_remapping"]["loss_action_names"] == [
        "LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL"
    ]
    assert report["validation_metrics_physical"]["action_names"] == report["target_remapping"]["loss_action_names"]
    assert len(report["validation_metrics_raw_six_class"]["action_names"]) == 6
    assert report["source_provenance"]["train"]["source_group_ids"] == ["train-group"]
    assert report["source_provenance"]["validation"]["source_group_ids"] == ["validation-group"]
    learnability = report["native1540_phase_learnability"]
    assert learnability["phase_a_gradient_max_l2"] > 0
    assert learnability["phase_a_update_l2"] > 0
    assert learnability["phase_a_weight_l2"] > 0
    assert learnability["phase_conditioned_logit_delta_rms"] > 0

    runtime_output = tmp_path / "native-control6-runtime-effective.pt"
    runtime_report = trainer.train_adapter(
        output_path=runtime_output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        loss_space="runtime_effective",
        steps=2,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=0.02,
        seed=18,
        device_name="cpu",
    )
    assert runtime_report["hold_raw_aux_weight"] == 0.0
    assert runtime_report["loss_space"] == "runtime_effective"
    assert runtime_report["training_state_context"] == NATIVE_CONTROL6_TRAINING_STATE_CONTEXT
    assert runtime_report["runtime_effective_loss"]["excluded_raw_rows_remain_in_six_way_denominator"] is True
    assert runtime_report["runtime_effective_loss"]["train_runtime_bucket_counts"] == {
        "HOLD": 2,
        "SPEAK": 1,
        "YIELD": 2,
        "BACKCHANNEL": 1,
    }
    assert runtime_report["validation_metrics"]["action_names"] == list(trainer.RUNTIME_BUCKETS)
    assert runtime_report["runtime_effective_loss"]["unsupported_train_targets_by_source_action"] == {}
    assert runtime_report["hold_raw_aux"]["eligible_train_rows"] == 2
    assert runtime_report["hold_raw_aux"]["eligible_train_rows_by_source"] == {
        "authored_synthetic_audio_test": 2
    }
    assert runtime_report["hold_raw_aux"]["eligible_train_rows_by_raw_action"] == {
        "LISTEN": 1,
        "CONTINUE": 1,
    }
    assert runtime_report["mean_physical_step_loss"] == runtime_report["mean_training_loss"]
    assert runtime_report["mean_auxiliary_step_loss"] == 0.0
    assert runtime_report["mean_combined_step_loss"] == runtime_report["mean_training_loss"]

    balanced_output = tmp_path / "native-control6-balanced-runtime-effective.pt"
    balanced_report = trainer.train_adapter(
        output_path=balanced_output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        loss_space="runtime_effective",
        sampling_strategy="balanced_runtime_bucket_source",
        steps=2,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=0.0002,
        seed=19,
        device_name="cpu",
    )
    assert balanced_report["sampling_strategy"] == "balanced_runtime_bucket_source"
    assert balanced_report["class_weights"] == {bucket: 1.0 for bucket in trainer.RUNTIME_BUCKETS}
    assert balanced_report["sampling_strategy_config"]["sampling_with_replacement"] is True
    assert balanced_report["sampling_strategy_config"]["loss_weights"] == "unweighted mean runtime-effective NLL"
    assert balanced_report["sampling_strategy_config"]["sampled_rows_by_bucket_and_source"] == {
        f"{bucket}:synthetic": 4 for bucket in trainer.RUNTIME_BUCKETS
    }
    assert balanced_report["checkpoint_selected_step"] == 2
    assert balanced_report["checkpoint_saved_steps"] == [0]
    assert [row["step"] for row in balanced_report["checkpoint_evaluation_curve"]] == [0]
    balanced_checkpoint = torch.load(balanced_output, map_location="cpu", weights_only=True)
    assert set(balanced_checkpoint["training_snapshots"]) == {"0"}
    balanced_initial = torch.load(legacy_checkpoint, map_location="cpu", weights_only=True)[
        "adapter_state_dict"
    ]
    balanced_trained = balanced_checkpoint["adapter_state_dict"]
    assert not torch.equal(balanced_trained["lora_A"][:, :1536], balanced_initial["lora_A"])
    assert torch.linalg.vector_norm(balanced_trained["lora_A"][:, 1536:]) > 0
    assert not torch.equal(balanced_trained["lora_B"], balanced_initial["lora_B"])

    phase_only_original = torch.load(legacy_checkpoint, map_location="cpu", weights_only=True)
    phase_only_output = tmp_path / "native-control6-phase-only.pt"
    phase_only_report = trainer.train_adapter(
        output_path=phase_only_output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        loss_space="runtime_effective",
        sampling_strategy="balanced_runtime_bucket_source",
        phase_only=True,
        steps=2,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=1e-3,
        seed=23,
        device_name="cpu",
    )
    phase_only_checkpoint = torch.load(phase_only_output, map_location="cpu", weights_only=True)
    phase_only_adapter = phase_only_checkpoint["adapter_state_dict"]
    assert phase_only_report["phase_only"] is True
    assert phase_only_report["phase_only_preservation"] == {
        "hidden_A_bitwise_unchanged": True,
        "lora_B_bitwise_unchanged": True,
        "base_rows_bitwise_unchanged": True,
    }
    torch.testing.assert_close(
        phase_only_adapter["lora_A"][:, :1536],
        phase_only_original["adapter_state_dict"]["lora_A"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        phase_only_adapter["lora_B"],
        phase_only_original["adapter_state_dict"]["lora_B"],
        rtol=0,
        atol=0,
    )
    assert torch.linalg.vector_norm(phase_only_adapter["lora_A"][:, 1536:]) > 0
    assert phase_only_report["native1540_phase_learnability"]["phase_a_gradient_max_l2"] > 0
    assert phase_only_report["checkpoint_saved_steps"] == [0]

    explicit_zero_output = tmp_path / "native-control6-runtime-effective-explicit-zero.pt"
    explicit_zero_report = trainer.train_adapter(
        output_path=explicit_zero_output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        loss_space="runtime_effective",
        hold_raw_aux_weight=0.0,
        steps=2,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=0.02,
        seed=18,
        device_name="cpu",
    )
    default_checkpoint = torch.load(runtime_output, map_location="cpu", weights_only=True)
    explicit_zero_checkpoint = torch.load(explicit_zero_output, map_location="cpu", weights_only=True)
    torch.testing.assert_close(
        default_checkpoint["adapter_state_dict"]["lora_A"],
        explicit_zero_checkpoint["adapter_state_dict"]["lora_A"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        default_checkpoint["adapter_state_dict"]["lora_B"],
        explicit_zero_checkpoint["adapter_state_dict"]["lora_B"],
        rtol=0,
        atol=0,
    )
    assert explicit_zero_report["mean_training_loss"] == runtime_report["mean_training_loss"]

    aux_output = tmp_path / "native-control6-runtime-effective-aux.pt"
    aux_report = trainer.train_adapter(
        output_path=aux_output,
        feature_dir=feature_dir,
        phase_profile="native1540",
        warmstart_checkpoint=legacy_checkpoint,
        loss_space="runtime_effective",
        hold_raw_aux_weight=0.25,
        steps=2,
        batch_frames=8,
        rank=2,
        alpha=2.0,
        learning_rate=0.02,
        seed=18,
        device_name="cpu",
    )
    assert aux_report["hold_raw_aux_weight"] == 0.25
    assert aux_report["hold_raw_aux"]["eligible_train_rows_by_source_and_raw_action"] == {
        "authored_synthetic_audio_test": {"LISTEN": 1, "CONTINUE": 1}
    }
    assert aux_report["mean_auxiliary_step_loss"] > 0
    assert aux_report["mean_combined_step_loss"] == pytest.approx(
        aux_report["mean_physical_step_loss"] + aux_report["mean_auxiliary_step_loss"]
    )
    assert all("step_losses" not in key for key in aux_report)

    merged_adapter_manifest = dict(manifest)
    merged_adapter_manifest["serving_base_only"] = False
    merged_adapter_manifest["merged_before_native_listener"] = True
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(merged_adapter_manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="requires serving base-only BF16 weights"):
        trainer.load_stage2_feature_sequences(feature_dir, "train")

    checkpoint = torch.load(output, map_location="cpu", weights_only=True)
    assert checkpoint["phase_schema"] == NATIVE1540_PHASE_SCHEMA
    assert checkpoint["native_control6_phase_profile"] == "native1540"
    assert validate_native_control6_semantic_prefix_metadata(checkpoint)["semantic_prefix_capacity"] == 190
    assert checkpoint["report"]["semantic_prefix_runtime_import_capacity_tokens"] == 512
    assert checkpoint["report"]["semantic_prefix_canonical_prime_target_tokens"] == 291
    assert tuple(checkpoint["adapter_state_dict"]["lora_A"].shape) == (2, 1540)
    assert torch.linalg.vector_norm(checkpoint["adapter_state_dict"]["lora_A"][:, 1536:]) > 0


def test_native1540_training_refuses_checkpoint_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing.pt"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="overwrite"):
        trainer.train_adapter(output_path=output, steps=1)
