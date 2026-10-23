from __future__ import annotations

import json

import pytest
import torch

from tools.train_native_control6_lora import (
    BASE_WEIGHT_SOURCE,
    HIDDEN_CONTRACT,
    HIDDEN_NORM,
    HIDDEN_SOURCE,
    HIDDEN_WIDTH,
    MODEL_ID,
    TICK_MS,
    _write_jsonl,
    assert_disjoint_source_groups,
    cpu_smoke,
    load_sequences,
    prepare_targets,
    train_adapter,
)
from training.duplex_qlora.native_control6 import (
    ACTION_TO_TOKEN_ID,
    ACTIONS,
    load_native_control6_adapter,
)


def _record(conversation_id: str, split: str, family: str) -> dict:
    return {
        "conversation_id": conversation_id,
        "split": split,
        "template_family_id": family,
        "human_annotated": False,
        "audio_path": "unused-test-fixture.wav",
        "duration_s": 0.48,
        "action_segments": [
            {"action": action, "start_ms": i * 80, "end_ms": (i + 1) * 80}
            for i, action in enumerate(ACTIONS)
        ],
        "label_origin": "authored_action_segments",
    }


def _write_manifest(path, records) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )


def test_native_output_row_lora_cpu_smoke_preserves_base_logits() -> None:
    result = cpu_smoke()
    assert result["status"] == "passed"
    assert result["logits_shape"] == [12, 6]
    assert result["base_rows_unchanged"] is True
    assert result["ordinary_vocabulary_rows_modified"] is False
    assert ACTION_TO_TOKEN_ID["LISTEN"] == 256001
    assert 256000 not in ACTION_TO_TOKEN_ID.values()


def test_prepare_targets_preserves_template_grouped_train_validation_split(tmp_path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    output = tmp_path / "targets"
    _write_manifest(
        manifest,
        [
            _record("synthetic-it-t6-train-01", "train", "train-family"),
            _record("synthetic-it-t6-val-01", "validation", "validation-family"),
        ],
    )
    summary = prepare_targets(manifest, output)
    train_rows = [json.loads(line) for line in (output / "train_targets.jsonl").read_text(encoding="utf-8").splitlines()]
    validation_rows = [json.loads(line) for line in (output / "validation_targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["target_action"] for row in train_rows] == list(ACTIONS)
    assert all(row["action_mask"] for row in train_rows + validation_rows)
    assert summary["source_group_split"]["template_families_train"] == 1
    assert summary["source_group_split"]["template_families_validation"] == 1
    assert {row["template_family_id"] for row in train_rows}.isdisjoint(
        {row["template_family_id"] for row in validation_rows}
    )


def test_split_leak_and_partial_tick_are_fail_closed(tmp_path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    records = [
        _record("synthetic-it-t6-train-01", "train", "shared-family"),
        _record("synthetic-it-t6-val-01", "validation", "shared-family"),
    ]
    _write_manifest(manifest, records)
    with pytest.raises(ValueError, match="leaks across"):
        prepare_targets(manifest, tmp_path / "out")

    records[1]["template_family_id"] = "validation-family"
    records[0]["duration_s"] = 0.45
    _write_manifest(manifest, records)
    summary = prepare_targets(manifest, tmp_path / "out")
    train_rows = [json.loads(line) for line in (tmp_path / "out" / "train_targets.jsonl").read_text(encoding="utf-8").splitlines()]
    assert train_rows[-1]["valid_ms"] == pytest.approx(50.0)
    assert train_rows[-1]["action_mask"] is False
    assert train_rows[-1]["target_action"] is None
    assert summary["counts"]["train"]["action_masked_frames"] == 1


def test_hidden_cache_requires_final_norm_contract_and_masks_stale_frame(tmp_path) -> None:
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    cache_dir = tmp_path / "cache"
    conversation_dir = cache_dir / "conversations"
    conversation_dir.mkdir(parents=True)
    metadata = {
        "schema_version": 1,
        "contract": HIDDEN_CONTRACT,
        "model_id": MODEL_ID,
        "hidden_source": HIDDEN_SOURCE,
        "normalization": HIDDEN_NORM,
            "hidden_width": HIDDEN_WIDTH,
            "tick_ms": TICK_MS,
            "selected_query_position": 0,
            "audio_queries_per_tick": 4,
            "per_frame_semantics": "newest production listener Gemma audio-query output after language_model.norm",
            "freshness_policy": "stale reused hidden retained with fresh_mask=false and excluded by trainer",
            "context_policy": "one GemmaListenerState per conversation; prime once; chronological user PCM and previous assistant track frame references; production bounded context/rebase",
        "qwen_loaded": False,
        "response_adapter": "OFF (base Gemma only)",
        "listener_profile": "production graph-enabled native LM; serving audio frontend/features",
        "native_cuda_graph_enabled": True,
        "native_library_path": "native/libgemma4_listener.so",
        "native_attention_backend": "CUSTOM",
        "native_rmsnorm_backend": "NATIVE_EXACT",
        "native_max_seq_len": 4096,
        "native_rebase_retain_frames": 64,
    }
    (cache_dir / "hidden_cache_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    train_id = "synthetic-it-t6-train-01"
    validation_id = "synthetic-it-t6-val-01"
    train_targets = [
        {
            "schema": "native_control6_action_targets_v1",
            "schema_version": 1,
            "conversation_id": train_id,
            "template_family_id": "train-family",
            "split": "train",
            "frame_id": i,
            "target_action": action,
            "action_mask": True,
        }
        for i, action in enumerate(ACTIONS)
    ]
    validation_targets = [
        {
            "schema": "native_control6_action_targets_v1",
            "schema_version": 1,
            "conversation_id": validation_id,
            "template_family_id": "validation-family",
            "split": "validation",
            "frame_id": i,
            "target_action": action,
            "action_mask": True,
        }
        for i, action in enumerate(ACTIONS)
    ]
    _write_jsonl(target_dir / "train.jsonl", train_targets)
    _write_jsonl(target_dir / "validation.jsonl", validation_targets)

    for conversation_id, split, family, stale in (
        (train_id, "train", "train-family", 2),
        (validation_id, "validation", "validation-family", -1),
    ):
        frame_ids = torch.arange(6, dtype=torch.int64)
        source_frame_ids = frame_ids.clone()
        fresh_mask = torch.ones(6, dtype=torch.bool)
        if stale >= 0:
            fresh_mask[stale] = False
            source_frame_ids[stale] = stale - 1
        torch.save(
            {
                "conversation_id": conversation_id,
                "template_family_id": family,
                "split": split,
                "frame_ids": frame_ids,
                "source_frame_ids": source_frame_ids,
                "assistant_reference_frame_ids": torch.arange(6, dtype=torch.int64) - 1,
                "fresh_mask": fresh_mask,
                "hidden_states": torch.randn(6, HIDDEN_WIDTH, dtype=torch.bfloat16),
            },
            conversation_dir / f"{conversation_id}.pt",
        )

    train, train_info = load_sequences(target_dir / "train.jsonl", cache_dir, "train")
    validation, validation_info = load_sequences(target_dir / "validation.jsonl", cache_dir, "validation")
    assert train[0]["runtime_context_available"] is False
    assert not bool(train[0]["speaker_active"].any())
    assert not bool(train[0]["user_activity_detected"].any())
    assert train_info["masked_stale_labeled_frames"] == 1
    assert train_info["action_counts"][ACTIONS[2]] == 0
    assert validation_info["feature_rows_aligned"] == 6
    assert_disjoint_source_groups(train, validation)

    metadata["normalization"] = "last_audio_query"
    (cache_dir / "hidden_cache_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="final-norm listener contract"):
        load_sequences(target_dir / "train.jsonl", cache_dir, "train")


def test_cpu_training_runner_updates_only_native_row_lora_and_saves_adapter(tmp_path) -> None:
    manifest = tmp_path / "synthetic.jsonl"
    _write_manifest(
        manifest,
        [
            _record("synthetic-it-t6-train-01", "train", "train-family"),
            _record("synthetic-it-t6-val-01", "validation", "validation-family"),
        ],
    )
    target_dir = tmp_path / "targets"
    prepare_targets(manifest, target_dir)
    cache_dir = tmp_path / "cache"
    conversation_dir = cache_dir / "conversations"
    conversation_dir.mkdir(parents=True)
    (cache_dir / "hidden_cache_metadata.json").write_text(json.dumps({
        "schema_version": 1,
        "contract": HIDDEN_CONTRACT,
        "model_id": MODEL_ID,
        "hidden_source": HIDDEN_SOURCE,
        "normalization": HIDDEN_NORM,
            "hidden_width": HIDDEN_WIDTH,
            "tick_ms": TICK_MS,
            "selected_query_position": 0,
            "audio_queries_per_tick": 4,
            "per_frame_semantics": "newest production listener Gemma audio-query output after language_model.norm",
            "freshness_policy": "stale reused hidden retained with fresh_mask=false and excluded by trainer",
            "context_policy": "one GemmaListenerState per conversation; prime once; chronological user PCM and previous assistant track frame references; production bounded context/rebase",
        "qwen_loaded": False,
        "response_adapter": "OFF (base Gemma only)",
        "listener_profile": "production graph-enabled native LM; serving audio frontend/features",
        "native_cuda_graph_enabled": True,
        "native_library_path": "native/libgemma4_listener.so",
        "native_attention_backend": "CUSTOM",
        "native_rmsnorm_backend": "NATIVE_EXACT",
        "native_max_seq_len": 4096,
        "native_rebase_retain_frames": 64,
    }), encoding="utf-8")
    for conversation_id, split, family in (
        ("synthetic-it-t6-train-01", "train", "train-family"),
        ("synthetic-it-t6-val-01", "validation", "validation-family"),
    ):
        torch.save({
            "conversation_id": conversation_id,
            "template_family_id": family,
            "split": split,
            "frame_ids": torch.arange(6, dtype=torch.int64),
            "source_frame_ids": torch.arange(6, dtype=torch.int64),
            "assistant_reference_frame_ids": torch.arange(6, dtype=torch.int64) - 1,
            "fresh_mask": torch.ones(6, dtype=torch.bool),
            "hidden_states": torch.randn(6, HIDDEN_WIDTH, dtype=torch.bfloat16),
        }, conversation_dir / f"{conversation_id}.pt")
    base_rows = tmp_path / "base_rows.pt"
    torch.save({
        "schema_version": 1,
        "architecture": "native_gemma_tied_output_rows",
        "model_id": MODEL_ID,
        "base_weight_source": BASE_WEIGHT_SOURCE,
        "hidden_size": HIDDEN_WIDTH,
        "vocab_size": 262144,
        "action_to_token_id": dict(ACTION_TO_TOKEN_ID),
        "token_id_to_string": {"256001": "<unused89>", "256002": "<unused90>", "256003": "<unused91>", "256004": "<unused92>", "256005": "<unused93>", "256006": "<unused94>"},
        "actions": list(ACTIONS),
        "base_rows": torch.randn(6, HIDDEN_WIDTH, dtype=torch.bfloat16),
    }, base_rows)

    output = tmp_path / "adapter.pt"
    report = train_adapter(
        train_targets=target_dir / "train_targets.jsonl",
        validation_targets=target_dir / "validation_targets.jsonl",
        hidden_cache=cache_dir,
        base_rows_path=base_rows,
        output_path=output,
        steps=2,
        batch_frames=6,
        device_name="cpu",
    )
    adapter = torch.load(output, map_location="cpu", weights_only=True)
    base_bundle = torch.load(base_rows, map_location="cpu", weights_only=True)
    loaded_adapter, _ = load_native_control6_adapter(output, base_bundle["base_rows"])
    assert report["steps"] == 2
    assert report["validation_metrics"]["examples"] == 6
    assert adapter["response_adapter"] == "OFF"
    assert set(adapter["adapter_state_dict"]) == {"lora_A", "lora_B"}
    assert adapter["ordinary_vocabulary_rows_modified"] is False
    assert adapter["base_gemma_weights_modified"] is False
    assert loaded_adapter(torch.randn(2, HIDDEN_WIDTH)).shape == (2, 6)


def test_legacy_training_rejects_runtime_effective_loss_without_native_context(tmp_path) -> None:
    with pytest.raises(ValueError, match="runtime_effective loss requires native1540 context"):
        train_adapter(
            output_path=tmp_path / "legacy-runtime-effective.pt",
            loss_space="runtime_effective",
        )
