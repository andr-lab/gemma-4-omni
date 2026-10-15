from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from duplex_kernel.native_say_v9_bridge import (
    build_native_q1_training_embeddings,
    compute_native_say_bridge_loss,
    differentiable_stateful_v9_prefix,
    prepare_native_say_bridge_pilot,
    say_hidden_input_positions,
    teacher_forced_residual_logits,
)
from duplex_kernel.models.v9 import CausalConv1D, DeepCausalTransformerProjectorV9, IncrementalV9Projector


def _v5_example(split: str, action: str, number: int, group: str) -> dict:
    conversation = f"{split}-{action.lower()}-{number:03d}"
    utterance_id = {
        "SPEAK": "assistant_reply",
        "BACKCHANNEL": "assistant_backchannel",
        "REPLAN": "assistant_replan",
    }[action]
    say = f"Risposta {number}."
    return {
        "conversation_id": conversation,
        "source_group_id": group,
        "split": split,
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "profile_version": "synthetic-it-t6-audio-grounded-v5-plan.1",
        "audio_response_protocol": "heard_action_say_v2",
        "audio_path": f"/workspace/data/{conversation}.wav",
        "audio_channel_index": 0,
        "sample_rate_hz": 16000,
        "audio_start_ms": 0.0,
        "audio_end_ms": 4000.0,
        "input_cutoff_ms": 4000.0,
        "target_start_ms": 4000.0,
        "target_end_ms": 5200.0,
        "target_utterance_id": utterance_id,
        "target_say_text": say,
        "target_text": json.dumps(
            {"heard": "testo", "action": action, "say": say},
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "action": action,
    }


def _raw_example(record: dict) -> dict:
    action = record["action"]
    role = {
        "SPEAK": "assistant_start",
        "BACKCHANNEL": "assistant_backchannel",
        "REPLAN": "assistant_replan",
    }[action]
    target_id = record["target_utterance_id"]
    target_text = record["target_say_text"]
    start = 64_000
    end = 83_200
    reference = {
        "utterance_id": "assistant_reference",
        "speaker": "assistant",
        "role": "assistant_start_interruptible",
        "start_sample": 0,
        "end_sample": 64_000,
        "text": "Una risposta di riferimento abbastanza lunga.",
    }
    target = {
        "utterance_id": target_id,
        "speaker": "assistant",
        "role": role,
        "start_sample": start,
        "end_sample": end,
        "text": target_text,
    }
    return {
        "conversation_id": record["conversation_id"],
        "topic_group_id": record["source_group_id"],
        "split": record["split"],
        "source": "authored_synthetic_audio_candidate",
        "profile": "audio_grounded_v5",
        "profile_version": "synthetic-it-t6-audio-grounded-v5-plan.1",
        "sample_rate": 16000,
        "channels": 2,
        "channel_map": {"0": "user", "1": "assistant"},
        "audio_path": record["audio_path"].replace("/workspace/", ""),
        "utterances": [reference, target],
    }


def _records():
    train = []
    validation = []
    raw = []
    for split, output, per_action in (
        ("train", train, {"SPEAK": 6, "BACKCHANNEL": 4, "REPLAN": 6}),
        ("validation", validation, {"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3}),
    ):
        actions = []
        for action, count in per_action.items():
            for number in range(count):
                actions.append((action, number))
        for ordinal, (action, number) in enumerate(actions):
            group = f"{split}-topic-{ordinal % (3 if split == 'train' else 2)}"
            record = _v5_example(split, action, number, group)
            output.append(record)
            raw.append(_raw_example(record))
    # Add a legacy row which must never enter the bridge pilot.
    legacy = _v5_example("train", "SPEAK", 99, "legacy-topic")
    legacy["source"] = "synthetic_it_t6"
    train.append(legacy)
    return {"train": train, "validation": validation, "test": []}, raw


def test_pilot_selects_v5_only_with_exact_assistant_windows_and_split_groups():
    records, raw = _records()
    result = prepare_native_say_bridge_pilot(
        records,
        raw,
        train_action_quota={"SPEAK": 6, "BACKCHANNEL": 4, "REPLAN": 6},
        validation_action_quota={"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3},
        seed=17,
    )

    assert result["report"]["counts_by_split_action"] == {
        "train": {"SPEAK": 6, "BACKCHANNEL": 4, "REPLAN": 6},
        "validation": {"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3},
    }
    rows = result["records"]
    assert len(rows) == 24
    assert {row["profile_version"] for row in rows} == {
        "synthetic-it-t6-audio-grounded-v5-plan.1"
    }
    assert not set(result["report"]["groups_by_split"]["train"]) & set(
        result["report"]["groups_by_split"]["validation"]
    )
    for row in rows:
        assert row["target_audio"]["channel_index"] == 1
        assert row["target_audio"]["start_sample"] == 64_000
        assert row["target_audio"]["end_sample"] == 83_200
        assert row["voice_reference"]["utterance_id"] == "assistant_reference"
        assert row["target_audio"]["estimated_rvq_frames"] == 15


def test_pilot_rejects_group_leakage_and_noncausal_target_window():
    records, raw = _records()
    leaked_conversation = raw[-1]["conversation_id"]
    raw[-1]["topic_group_id"] = "train-topic-0"
    next(row for row in records["validation"] if row["conversation_id"] == leaked_conversation)[
        "source_group_id"
    ] = "train-topic-0"
    with pytest.raises(ValueError, match="group-disjoint"):
        prepare_native_say_bridge_pilot(
            records,
            raw,
            train_action_quota={"SPEAK": 6, "BACKCHANNEL": 4, "REPLAN": 6},
            validation_action_quota={"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3},
            seed=17,
        )

    records, raw = _records()
    records["train"][0]["target_start_ms"] = 3999.0
    with pytest.raises(ValueError, match="target begins before input cutoff"):
        prepare_native_say_bridge_pilot(
            records,
            raw,
            train_action_quota={"SPEAK": 6, "BACKCHANNEL": 4, "REPLAN": 6},
            validation_action_quota={"SPEAK": 3, "BACKCHANNEL": 2, "REPLAN": 3},
            seed=17,
        )


def test_say_hidden_positions_use_postselected_token_input_positions():
    assert say_hidden_input_positions(prompt_length=37, say_token_indices=[0, 3, 7]) == [37, 40, 44]


def test_differentiable_v9_prefix_matches_production_causal_stream_and_trains_only_projector():
    torch.manual_seed(21)
    projector = DeepCausalTransformerProjectorV9(
        raw_dim=8, text_dim=8, hidden_dim=16, num_layers=2, num_heads=4, ffn_dim=32
    ).eval()
    causal_conv = CausalConv1D(hidden_size=8, kernel_size=3).eval()
    for parameter in causal_conv.parameters():
        parameter.requires_grad_(False)
    raw = torch.randn((1, 6, 8), requires_grad=True)

    batch = differentiable_stateful_v9_prefix(raw, projector, causal_conv)
    stream = IncrementalV9Projector(projector, causal_conv)
    state = stream.new_state(batch_size=1, device="cpu", dtype=torch.float32, max_seq_len=6)
    incremental = torch.stack(
        [stream.step(raw[:, index, :], raw[:, 0, :], state) for index in range(raw.shape[1])],
        dim=1,
    )
    torch.testing.assert_close(batch.detach(), incremental, rtol=2e-5, atol=2e-5)
    batch.square().mean().backward()
    assert any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in projector.parameters())
    assert all(parameter.grad is None for parameter in causal_conv.parameters())


class _Embedding(nn.Module):
    def __init__(self, vocab: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((vocab, dim), dtype=torch.float32))

    def forward(self, ids):
        return self.weight[ids]


class _FakeTalkerModel:
    def __init__(self):
        self.embedding = _Embedding(2048, 2048)
        self.code_predictor = SimpleNamespace(get_input_embeddings=lambda: [self.embedding] * 15)

    def get_input_embeddings(self):
        return self.embedding


class _FakeWrapper:
    def __init__(self):
        self.talker = _FakeTalkerModel()

    def build_prefill_parts(self, text, *, voice_conditioning=None):
        assert text == "Ciao."
        assert voice_conditioning == {"mode": "x_vector_only"}
        device = torch.device("cpu")
        return (
            torch.zeros((1, 2, 2048), device=device),
            torch.zeros((1, 3, 2048), device=device),
            torch.zeros((1, 1, 2048), device=device),
            torch.ones((1, 1, 2048), device=device),
        )


def test_native_q1_prefill_keeps_all_say_states_and_all_target_frames():
    wrapper = _FakeWrapper()
    h_pfx = torch.ones((1, 3, 2048))
    codes = torch.zeros((1, 48, 16), dtype=torch.long)
    embeddings, bos_position, frame_count = build_native_q1_training_embeddings(
        wrapper,
        h_pfx=h_pfx,
        codes=codes,
        text="Ciao.",
        voice_conditioning={"mode": "x_vector_only"},
    )

    # Two ICL + three text + three V9 + BOS, followed by all 48 real RVQ inputs.
    assert embeddings.shape == (1, 57, 2048)
    assert bos_position == 8
    assert frame_count == 48


class _FakeCodePredictor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        if kwargs.get("inputs_embeds") is not None:
            context = kwargs["inputs_embeds"].sum(dim=1)
            step = 0
        else:
            step = int(kwargs["generation_steps"])
            context = kwargs["past_key_values"] + kwargs["input_ids"].float()
        self.calls.append((step, kwargs.get("input_ids")))
        logits = context.sum(dim=-1, keepdim=True).unsqueeze(1).expand(-1, 1, 2048)
        return SimpleNamespace(logits=logits, past_key_values=context)


class _ResidualWrapper:
    def __init__(self):
        self.talker = _FakeTalkerModel()
        self.talker.code_predictor = _FakeCodePredictor()


def test_residual_teacher_forcing_uses_real_previous_books_and_keeps_input_gradient():
    wrapper = _ResidualWrapper()
    last_h = torch.ones((4, 2048), requires_grad=True)
    target = torch.arange(4 * 16, dtype=torch.long).view(4, 16) % 2048
    logits = teacher_forced_residual_logits(wrapper, last_h, target)

    assert logits.shape == (4, 15, 2048)
    assert [step for step, _ in wrapper.talker.code_predictor.calls] == list(range(15))
    for codebook_index, input_ids in wrapper.talker.code_predictor.calls[1:]:
        assert torch.equal(input_ids[:, 0], target[:, codebook_index])
    logits.sum().backward()
    assert last_h.grad is not None and bool(last_h.grad.abs().sum() > 0)


def test_bridge_loss_includes_cb0_residual_books_and_natural_eos():
    codes = torch.zeros((3, 16), dtype=torch.long)
    codes[:, 1:] = 1
    cb0_logits = torch.randn((4, 2050), requires_grad=True)
    residual_logits = torch.randn((3, 15, 2048), requires_grad=True)
    loss, metrics = compute_native_say_bridge_loss(
        cb0_logits,
        residual_logits,
        codes,
        eos_token_id=2049,
    )

    assert torch.isfinite(loss)
    assert metrics["cb0_frames"] == 3
    assert metrics["eos_targets"] == 1
    assert metrics["residual_targets"] == 45
    loss.backward()
    assert cb0_logits.grad is not None and bool(cb0_logits.grad.abs().sum() > 0)
    assert residual_logits.grad is not None and bool(residual_logits.grad.abs().sum() > 0)
