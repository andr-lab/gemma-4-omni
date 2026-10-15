from __future__ import annotations

import json

import numpy as np
import pytest

from training.duplex_qlora.features import FEATURE_CONTRACT, listener_feature_contract
from training.duplex_qlora.schema import ACTIONS
from training.duplex_qlora.stage2 import (
    _v5_event_onset_selection_metrics,
    train_control_head,
)


def _feature_split(
    root,
    split,
    labels,
    group,
    *,
    source="synthetic_it_v3",
    synthetic=True,
    input_dim=1536,
):
    features = np.zeros((len(labels), input_dim), dtype=np.float32)
    for index, label in enumerate(labels):
        if label >= 0:
            features[index, label] = 1.0
    np.savez_compressed(
        root / f"{split}.npz",
        features=features,
        labels=np.asarray(labels, dtype=np.int64),
        frame_ids=np.arange(len(labels), dtype=np.int64),
        listener_frame_ids=np.arange(len(labels), dtype=np.int64),
    )
    metadata = [
        {
            "conversation_id": f"{group}-{index}",
            "source_group_id": group,
            "session_id": f"{split}:{group}",
            "frame_id": index,
            "listener_frame_id": index,
            "source": source,
            "synthetic": synthetic,
            "action": ACTIONS[label] if label >= 0 else None,
            "action_mask": label >= 0,
        }
        for index, label in enumerate(labels)
    ]
    (root / f"{split}.json").write_text(json.dumps(metadata), encoding="utf-8")


def _feature_dir(tmp_path, *, input_dim=1536):
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    _feature_split(
        feature_dir, "train", [0, 0, 0, 1, 1, 1, -100], "train-family", input_dim=input_dim
    )
    _feature_split(
        feature_dir, "validation", [0, 1, -100], "validation-family", input_dim=input_dim
    )
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "merged_before_native_listener": True,
                "feature_contract": listener_feature_contract(input_dim),
            }
        ),
        encoding="utf-8",
    )
    training_manifest = tmp_path / "training_manifest.json"
    training_manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    return feature_dir, training_manifest


def test_stage2_trains_native_phase_fusion_head_with_unweighted_empirical_objective(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path, input_dim=1540)
    train_path = feature_dir / "train.npz"
    with np.load(train_path, allow_pickle=False) as values:
        features = values["features"].copy()
        labels = values["labels"].copy()
        frame_ids = values["frame_ids"].copy()
        listener_ids = values["listener_frame_ids"].copy()
    features[:, 1536:] = np.asarray([-0.2, -0.4, -0.6, 0.25], dtype=np.float32)
    features[labels == 1, 1536:] = np.asarray([-0.9, -0.8, -0.2, -0.5], dtype=np.float32)
    np.savez_compressed(
        train_path,
        features=features,
        labels=labels,
        frame_ids=frame_ids,
        listener_frame_ids=listener_ids,
    )

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head_f.pt",
        training_manifest_path=training_manifest,
        max_steps=20,
        architecture="native_phase_fusion_mlp",
        sampling_strategy="empirical_rows",
        class_weighting="none",
        event_onset_loss_multiplier=1.0,
        seed=43,
    )

    report = result["stage2_training"]
    assert report["architecture"] == "native_phase_fusion_mlp"
    assert report["sampling_strategy"] == "empirical_rows"
    assert report["loss"] == {
        "function": "unweighted_cross_entropy",
        "class_weights": None,
    }
    assert report["event_onset_loss_weighting"]["multiplier"] == 1.0
    assert report["trainable_head_parameters"] > 0
    assert (tmp_path / "control_head_f.pt").is_file()


def test_stage2_trains_and_reports_class_support_and_validation_metrics(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head.pt",
        training_manifest_path=training_manifest,
        max_steps=20,
    )

    report = result["stage2_training"]
    assert report["examples"] == 6
    assert report["class_support"]["LISTEN"] == 3
    assert report["class_support"]["SPEAK"] == 3
    assert report["class_support"]["YIELD"] == 0
    assert report["validation"]["examples"] == 2
    assert 0 <= report["validation"]["accuracy"] <= 1
    assert (tmp_path / "control_head.pt").is_file()


def test_stage2_rejects_conversation_group_leakage_between_splits(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    val_metadata = json.loads((feature_dir / "validation.json").read_text(encoding="utf-8"))
    val_metadata[0]["source_group_id"] = "train-family"
    (feature_dir / "validation.json").write_text(json.dumps(val_metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="leakage"):
        train_control_head(
            feature_dir=feature_dir,
            output_path=tmp_path / "control_head.pt",
            training_manifest_path=training_manifest,
            max_steps=5,
        )


def test_stage2_mlp_selects_only_on_natural_validation_macro_f1(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    _feature_split(
        feature_dir,
        "validation",
        [0, 1, 2, 4],
        "validation-natural-family",
        source="SmoothConv",
        synthetic=False,
    )

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head_mlp.pt",
        training_manifest_path=training_manifest,
        max_steps=30,
        architecture="mlp_256_gelu_dropout_0.1",
        selection_metric="validation_natural_macro_f1",
        selection_interval=10,
    )

    report = result["stage2_training"]
    selection = report["checkpoint_selection"]
    assert report["architecture"] == "mlp_256_gelu_dropout_0.1"
    assert selection["metric"] == "validation_natural_macro_f1"
    assert selection["evaluated_steps"] == [10, 20, 30]
    assert selection["selected_step"] in {10, 20, 30}
    assert selection["selected_validation_examples"] == 4
    assert report["heldout_test"] is None
    natural = report["validation"]["subgroups"]["natural_four_class"]
    confusion = natural["confusion_matrix"]["values"]
    natural_actions = natural["actions"]
    correct = sum(
        confusion[index][ACTIONS.index(action)]
        for index, action in enumerate(natural_actions)
    )
    assert natural["accuracy"] == pytest.approx(correct / natural["examples"])
    backchannel_row = natural_actions.index("BACKCHANNEL")
    backchannel_column = ACTIONS.index("BACKCHANNEL")
    backchannel_predicted = sum(row[backchannel_column] for row in confusion)
    expected_backchannel_precision = (
        confusion[backchannel_row][backchannel_column] / backchannel_predicted
        if backchannel_predicted
        else 0.0
    )
    assert natural["per_class"]["BACKCHANNEL"]["precision"] == pytest.approx(
        expected_backchannel_precision
    )
    assert (tmp_path / "control_head_mlp.pt").is_file()


def test_stage2_reports_v5_event_onset_six_class_metrics_separately(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    _feature_split(
        feature_dir,
        "test",
        [0, 4, 5, 2],
        "heldout-v5-family",
        source="authored_synthetic_audio_candidate",
        synthetic=True,
    )
    metadata_path = feature_dir / "test.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for row, onset in zip(metadata, [True, True, True, False]):
        row.update(profile="audio_grounded_v5", event_onset=onset)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head_event_onset.pt",
        training_manifest_path=training_manifest,
        max_steps=20,
        architecture="mlp_256_gelu_dropout_0.1",
    )

    event_metrics = result["stage2_training"]["heldout_test"]["subgroups"][
        "audio_grounded_v5_event_onsets_six_class"
    ]
    assert event_metrics["examples"] == 3
    assert event_metrics["class_support"] == {
        "LISTEN": 1,
        "SPEAK": 0,
        "CONTINUE": 0,
        "YIELD": 0,
        "BACKCHANNEL": 1,
        "REPLAN": 1,
    }
    assert event_metrics["per_class"]["REPLAN"]["recall"] is not None
    background_false_event = result["stage2_training"]["heldout_test"]["subgroups"][
        "audio_grounded_v5_event_onset_background_false_event_rate"
    ]
    assert background_false_event["examples"] == 1
    assert background_false_event["event_action_outputs"] == [
        "SPEAK", "YIELD", "BACKCHANNEL", "REPLAN"
    ]
    assert 0 <= background_false_event["false_event_rate"] <= 1
    legacy_rate = result["stage2_training"]["heldout_test"]["subgroups"][
        "audio_grounded_v5_event_onset_legacy_off_onset_event_output_rate_not_false_action_rate"
    ]
    assert legacy_rate["examples"] == 1
    assert "not a false-action rate" in legacy_rate["note"]


def test_v5_selection_uses_background_false_actions_and_labels_legacy_rate(tmp_path):
    target = np.asarray(
        [
            ACTIONS.index("SPEAK"),
            ACTIONS.index("SPEAK"),
            ACTIONS.index("LISTEN"),
            ACTIONS.index("CONTINUE"),
            ACTIONS.index("CONTINUE"),
            ACTIONS.index("YIELD"),
            ACTIONS.index("BACKCHANNEL"),
            ACTIONS.index("BACKCHANNEL"),
            ACTIONS.index("REPLAN"),
            ACTIONS.index("LISTEN"),
        ],
        dtype=np.int64,
    )
    prediction = np.asarray(
        [
            ACTIONS.index("SPEAK"),
            ACTIONS.index("SPEAK"),
            ACTIONS.index("LISTEN"),
            ACTIONS.index("CONTINUE"),
            ACTIONS.index("CONTINUE"),
            ACTIONS.index("YIELD"),
            ACTIONS.index("BACKCHANNEL"),
            ACTIONS.index("BACKCHANNEL"),
            ACTIONS.index("REPLAN"),
            ACTIONS.index("YIELD"),
        ],
        dtype=np.int64,
    )
    metadata = [
        {
            "profile": "audio_grounded_v5",
            "source": "authored_synthetic_audio_candidate",
            "event_onset": onset,
        }
        for onset in (True, False, False, False, True, True, True, False, True, False)
    ]

    result = _v5_event_onset_selection_metrics(target, prediction, metadata)

    assert result["event_onset_macro_f1"] == pytest.approx(1.0)
    assert result["event_onset_examples"] == 5
    assert result["background_false_event_rate"] == pytest.approx(1 / 3)
    assert result["background_examples"] == 3
    assert result["background_false_event_predictions"] == 1
    assert result["legacy_off_onset_event_output_rate_not_false_action_rate"] == pytest.approx(3 / 5)
    assert result["legacy_off_onset_examples"] == 5


def test_empirical_rows_reports_actual_prior_without_class_uniform_sampling(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    _feature_split(
        feature_dir,
        "train",
        [0] * 90 + [1] * 10,
        "imbalanced-train-family",
    )

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head_empirical.pt",
        training_manifest_path=training_manifest,
        max_steps=10,
        sampling_strategy="empirical_rows",
        architecture="linear",
        seed=29,
    )

    report = result["stage2_training"]
    assert report["sampling"] == "uniform_over_all_supervised_training_rows"
    assert report["sampling_strategy"] == "empirical_rows"
    assert report["loss"] == {"function": "unweighted_cross_entropy", "class_weights": None}
    assert report["class_sampling_probabilities"] == pytest.approx(
        {"LISTEN": 0.9, "SPEAK": 0.1, "CONTINUE": 0.0, "YIELD": 0.0,
         "BACKCHANNEL": 0.0, "REPLAN": 0.0}
    )
    assert report["expected_updates_per_observed_class"]["LISTEN"] == pytest.approx(9.0)
    assert report["expected_updates_per_observed_class"]["SPEAK"] == pytest.approx(1.0)


def test_sqrt_inverse_weights_and_event_onset_weight_include_continue_transitions(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    _feature_split(
        feature_dir,
        "train",
        [0] * 9 + [1] * 4 + [2],
        "weighted-train-family",
    )
    train_metadata_path = feature_dir / "train.json"
    train_metadata = json.loads(train_metadata_path.read_text(encoding="utf-8"))
    train_metadata[-1].update(
        source="authored_synthetic_audio_candidate",
        profile="audio_grounded_v5",
        event_onset=True,
    )
    train_metadata_path.write_text(json.dumps(train_metadata), encoding="utf-8")

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=tmp_path / "control_head_weighted.pt",
        training_manifest_path=training_manifest,
        max_steps=1,
        architecture="linear",
        sampling_strategy="empirical_rows",
        class_weighting="sqrt_inverse_frequency",
        event_onset_loss_multiplier=2.0,
        seed=31,
    )

    report = result["stage2_training"]
    weights = report["loss"]["class_weights"]
    assert weights["LISTEN"] == pytest.approx(6 / 11)
    assert weights["SPEAK"] == pytest.approx(9 / 11)
    assert weights["CONTINUE"] == pytest.approx(18 / 11)
    assert weights["YIELD"] == weights["BACKCHANNEL"] == weights["REPLAN"] == 0.0
    onset_weighting = report["event_onset_loss_weighting"]
    assert onset_weighting["multiplier"] == 2.0
    assert onset_weighting["training_event_onset_examples_by_action"]["CONTINUE"] == 1
    assert onset_weighting["training_event_onset_examples_by_action"]["LISTEN"] == 0


def test_infeasible_v5_diagnostic_checkpoint_is_saved_but_not_promotion_eligible(tmp_path):
    feature_dir, training_manifest = _feature_dir(tmp_path)
    labels = [1, 2, 3, 4, 5, 0, 0, 2, 2, 0]
    _feature_split(
        feature_dir,
        "validation",
        labels,
        "validation-v5-family",
        source="authored_synthetic_audio_candidate",
        synthetic=True,
    )
    feature_path = feature_dir / "validation.npz"
    with np.load(feature_path, allow_pickle=False) as values:
        features = values["features"].copy()
        saved_labels = values["labels"].copy()
        frame_ids = values["frame_ids"].copy()
        listener_ids = values["listener_frame_ids"].copy()
    for event_index, background_index in ((0, 5), (2, 6), (3, 7), (4, 8)):
        features[background_index] = features[event_index]
    np.savez_compressed(
        feature_path,
        features=features,
        labels=saved_labels,
        frame_ids=frame_ids,
        listener_frame_ids=listener_ids,
    )
    metadata_path = feature_dir / "validation.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for index, row in enumerate(metadata):
        row.update(
            profile="audio_grounded_v5",
            event_onset=index < 5,
        )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output = tmp_path / "diagnostic_head.pt"

    result = train_control_head(
        feature_dir=feature_dir,
        output_path=output,
        training_manifest_path=training_manifest,
        max_steps=2,
        architecture="linear",
        selection_metric="validation_v5_event_onset_macro_f1",
        selection_interval=1,
        max_validation_v5_background_false_event_rate=0.0,
        save_infeasible_diagnostic_checkpoint=True,
        validation_checkpoint_dir=tmp_path / "validation_states",
        seed=37,
    )

    report = result["stage2_training"]
    selection = report["checkpoint_selection"]
    assert selection["selection_status"].startswith("infeasible_")
    assert selection["promotion_eligible"] is False
    assert selection["diagnostic_checkpoint_saved_for_validation_calibration_only"] is True
    assert report["diagnostic_checkpoint_role"] == (
        "best_infeasible_event_onset_checkpoint_saved_for_validation_calibration_only"
    )
    assert report["candidate_promotion_eligible"] is False
    assert report["heldout_test"] is None
    assert output.is_file()
    assert selection["validation_state_checkpoint_dir"] == str(
        (tmp_path / "validation_states").resolve()
    )
    assert (tmp_path / "validation_states" / "step_0001.pt").is_file()
    assert (tmp_path / "validation_states" / "step_0002.pt").is_file()
