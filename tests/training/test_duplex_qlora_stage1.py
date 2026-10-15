from __future__ import annotations

import numpy as np
import soundfile as sf
import torch

from training.duplex_qlora.stage1 import (
    count_response_examples,
    _make_training_schedule,
    _make_audio_asr_mix_schedule,
    _language_model_lora_target_modules,
    _freeze_float_bridges,
    _freeze_float_towers,
    _freeze_input_embeddings,
    _quantization_runtime_report,
    _read_audio_for_record,
    _training_manifest_path,
    _training_config,
    _training_records,
    _validate_audio_asr_mix_records,
    _format_chat,
    _verify_lora_scope,
    _move_batch_to_device,
    build_training_inputs,
)


def _conversation(audio_path, *, segments, source="synthetic_it_v3", sample_rate=24000):
    return {
        "conversation_id": "conv-1",
        "source_group_id": "family-1",
        "source": source,
        "synthetic": source == "synthetic_it_v3",
        "audio_path": str(audio_path),
        "sample_rate_hz": sample_rate,
        "duration_ms": 1000,
        "channel_map": {"0": "user", "1": "assistant"},
        "segments": segments,
    }


def test_synthetic_response_records_mask_cropped_text_and_use_prior_user_audio(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    audio = np.zeros((24000, 2), dtype=np.float32)
    audio[1000:7000, 0] = 0.25
    audio[10000:15000, 1] = 0.4
    sf.write(audio_path, audio, 24000)
    conversation = _conversation(
        audio_path,
        segments=[
            {"role": "user", "text": "ciao", "start_ms": 40, "end_ms": 280, "time_mask": True, "text_mask": True},
            {"role": "assistant", "text": "risposta tagliata", "start_ms": 300, "end_ms": 420, "measured_duration_s": 0.6, "time_mask": True, "text_mask": False},
            {"role": "user", "text": "parole future in overlap", "start_ms": 450, "end_ms": 600, "time_mask": True, "text_mask": True},
            {"role": "assistant", "text": "risposta completa", "start_ms": 500, "end_ms": 750, "measured_duration_s": 0.25, "time_mask": True, "text_mask": True},
        ],
    )

    records = _training_records([conversation])

    assert [row["target_text"] for row in records] == ["risposta completa"]
    assert records[0]["task_kind"] == "conversation_response"
    assert records[0]["audio_channel_index"] == 0
    assert records[0]["audio_start_ms"] == 0
    assert records[0]["audio_end_ms"] == 500
    assert records[0]["audio_history_cap_ms"] == 5120
    assert "user: ciao" in records[0]["context_text"]
    assert "risposta tagliata" not in records[0]["context_text"]
    assert "parole future in overlap" not in records[0]["context_text"]


def test_masked_assistant_segment_is_not_target_but_remains_prior_context(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    audio = np.ones((24000, 2), dtype=np.float32) * 0.1
    sf.write(audio_path, audio, 24000, subtype="FLOAT")
    conversation = _conversation(
        audio_path,
        segments=[
            {"role": "assistant", "text": "frase da mascherare come target", "start_ms": 200, "end_ms": 400,
             "target_mask": False, "text_mask": True, "time_mask": True},
            {"role": "user", "text": "domanda successiva", "start_ms": 500, "end_ms": 650,
             "text_mask": True, "time_mask": True},
            {"role": "assistant", "text": "risposta successiva", "start_ms": 700, "end_ms": 950,
             "text_mask": True, "time_mask": True},
        ],
    )

    records = _training_records([conversation])

    assert count_response_examples([conversation]) == 1
    assert [row["target_text"] for row in records] == ["risposta successiva"]
    assert "assistant: frase da mascherare come target" in records[0]["context_text"]


def test_lora_target_names_are_full_paths_inside_language_model_only():
    from torch import nn

    def projections():
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(self, name, nn.Linear(2, 2, bias=False))

        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                for name in ("gate_proj", "up_proj", "down_proj"):
                    setattr(self, name, nn.Linear(2, 2, bias=False))

        class Layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = Attention()
                self.mlp = MLP()

        class Branch(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Layer()])

        return Branch

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            branch = projections()
            self.language_model = branch()
            self.audio_tower = branch()

    class Gemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    targets = _language_model_lora_target_modules(Gemma())

    assert len(targets) == 7
    assert all(name.startswith("model.language_model.layers.0.") for name in targets)
    assert {name.rsplit(".", 1)[-1] for name in targets} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    assert not any("audio_tower" in name for name in targets)


def test_audio_response_training_config_uses_recoverable_50_step_checkpoint_interval(tmp_path):
    config = _training_config(
        train_jsonl=tmp_path / "train.jsonl",
        validation_jsonl=tmp_path / "validation.jsonl",
        model_path="/cached/gemma",
        max_steps=300,
        learning_rate=1e-5,
        seed=20261002,
        conditioning_mode="baselineaudio_response",
        checkpoint_interval=50,
    )

    assert config["checkpoint_interval"] == 50
    assert config["use_cache_for_training"] is True
    assert config["kv_sharing_semantics"] == "transformers_gemma4_native_dynamic_cache"


def test_qlora_loader_injects_only_full_language_model_targets(monkeypatch):
    import sys
    import types
    from torch import nn
    from training.duplex_qlora import stage1

    class AdaptedLinear(nn.Linear):
        def __init__(self):
            super().__init__(2, 2, bias=False)
            self.lora_A = nn.ModuleDict()
            self.lora_B = nn.ModuleDict()

    class Linear4bit(AdaptedLinear):
        pass

    class Branch(nn.Module):
        def __init__(self, *, quantized=False):
            super().__init__()
            self.input_layernorm = nn.LayerNorm(2, dtype=torch.bfloat16)
            self.embed_tokens = nn.Embedding(8, 2, dtype=torch.bfloat16)
            self.ple_embeddings = nn.Embedding(16, 2, dtype=torch.bfloat16)
            self.layers = nn.ModuleList([nn.Module()])
            layer = self.layers[0]
            layer.self_attn = nn.Module()
            linear_type = Linear4bit if quantized else AdaptedLinear
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                setattr(layer.self_attn, name, linear_type())
            layer.mlp = nn.Module()
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(layer.mlp, name, linear_type())

    class FakeGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = types.SimpleNamespace(use_cache=True)
            self.model = nn.Module()
            self.model.language_model = Branch(quantized=True)
            self.model.audio_tower = Branch()
            self.model.vision_tower = Branch()
            self.model.embed_audio = nn.Module()
            self.model.embed_audio.embedding_projection = nn.Linear(2, 2, bias=False)
            self.model.embed_vision = nn.Module()
            self.model.embed_vision.embedding_projection = nn.Linear(2, 2, bias=False)
            self.input_embeddings = nn.Embedding(8, 2, dtype=torch.bfloat16)
            self.lm_head = nn.Linear(2, 8, bias=False)
            self.lm_head.weight = self.input_embeddings.weight
            self.gradient_checkpointing_kwargs = None

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

        def enable_input_require_grads(self):
            raise AssertionError("non-reentrant checkpointing must not add the input-grad hook")

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.lm_head

    class FakeProcessor:
        pass

    model = FakeGemma()
    observed = {}

    class AutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            observed["model_args"] = (args, kwargs)
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path):
            observed["processor_path"] = path
            return FakeProcessor()

    class FakeLoraConfig:
        def __init__(self, **kwargs):
            observed["lora_config"] = kwargs
            self.target_modules = kwargs["target_modules"]

    def get_peft_model(base, config):
        for name, module in list(base.named_modules()):
            if name in config.target_modules:
                module.lora_A["default"] = nn.Linear(2, 1, bias=False)
                module.lora_B["default"] = nn.Linear(1, 2, bias=False)
        return base

    fake_peft = types.ModuleType("peft")
    fake_peft.LoraConfig = FakeLoraConfig
    fake_peft.PeftModel = type("FakePeftModel", (), {})
    fake_peft.get_peft_model = get_peft_model
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = AutoModel
    fake_transformers.AutoProcessor = AutoProcessor
    def fake_bnb_config(**kwargs):
        observed["bnb_config"] = kwargs
        return kwargs

    fake_transformers.BitsAndBytesConfig = fake_bnb_config
    fake_bitsandbytes = types.ModuleType("bitsandbytes")
    monkeypatch.setitem(sys.modules, "peft", fake_peft)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bitsandbytes)

    loaded_model, processor, trainable, matched, runtime = stage1._load_qlora(
        "/cached/gemma", dtype=torch.bfloat16
    )

    assert processor.__class__ is FakeProcessor
    assert len(trainable) == 14
    assert len(matched) == 7
    assert observed["lora_config"]["target_modules"] == matched
    assert observed["bnb_config"]["llm_int8_skip_modules"] == [
        "model.audio_tower", "model.vision_tower",
        "model.embed_audio.embedding_projection", "model.embed_vision.embedding_projection",
        "lm_head",
    ]
    assert all("language_model" in name for name, _ in trainable)
    assert runtime["audio_tower"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["vision_tower"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["audio_tower"]["requires_grad"] is False
    assert runtime["vision_tower"]["requires_grad"] is False
    assert runtime["audio_tower"]["linear4bit_count"] == 0
    assert runtime["vision_tower"]["linear4bit_count"] == 0
    assert runtime["language_model_quantized_module_count"] == 7
    assert runtime["llm_int8_skip_modules"] == observed["bnb_config"]["llm_int8_skip_modules"]
    assert runtime["embedding_bridges"]["model.embed_audio.embedding_projection"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["embedding_bridges"]["model.embed_vision.embedding_projection"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["input_embeddings"]["input_embedding_dtype"] == "torch.bfloat16"
    assert runtime["input_embeddings"]["output_head_dtype"] == "torch.float32"
    assert runtime["input_embeddings"]["output_head_input_cast"] == "torch.float32"
    assert runtime["input_embeddings"]["output_head_was_tied"] is True
    assert runtime["input_embeddings"]["output_head_tied"] is False
    assert loaded_model.get_input_embeddings().weight is not loaded_model.get_output_embeddings().weight
    assert loaded_model.get_input_embeddings().weight.dtype == torch.bfloat16
    output_head = loaded_model.get_output_embeddings()
    assert output_head.weight.dtype == torch.float32
    assert not output_head.weight.requires_grad
    assert getattr(output_head, "_gemma4_lm_head_dtype_bridge_registered", False)
    assert len(output_head._forward_pre_hooks) == 1
    assert loaded_model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
    assert loaded_model.model.language_model.ple_embeddings.weight.dtype == torch.bfloat16
    assert loaded_model.config.use_cache is True
    assert loaded_model.gradient_checkpointing_kwargs is None
    assert runtime["training_preparation"]["cache_enabled_for_shared_kv"] is True
    assert runtime["training_preparation"]["gradient_checkpointing"] is False
    assert runtime["training_preparation"]["norms_cast_to_fp32"] == [
        "model.language_model.input_layernorm"
    ]
    lora_source = next(parameter for name, parameter in trainable if "lora_A" in name)
    head_input = lora_source.sum().to(dtype=torch.bfloat16).expand(1, 2)
    output_head(head_input).sum().backward()
    assert lora_source.grad is not None
    assert torch.isfinite(lora_source.grad).all()
    assert lora_source.grad.abs().sum() > 0
    assert output_head.weight.grad is None
    assert not any(
        hasattr(module, "lora_A") and len(module.lora_A)
        for name, module in loaded_model.named_modules()
        if "audio_tower" in name
    )


def test_lora_scope_verifier_rejects_adapter_on_duplicate_audio_projection():
    from torch import nn

    class AdaptedLinear(nn.Linear):
        def __init__(self):
            super().__init__(2, 2, bias=False)
            self.lora_A = nn.ModuleDict({"default": nn.Linear(2, 1, bias=False)})
            self.lora_B = nn.ModuleDict({"default": nn.Linear(1, 2, bias=False)})

    class Branch(nn.Module):
        def __init__(self, *, adapted):
            super().__init__()
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].q_proj = AdaptedLinear() if adapted else nn.Linear(2, 2, bias=False)

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Branch(adapted=True)
            self.audio_tower = Branch(adapted=False)

    class Gemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Gemma()
    targets = ["model.language_model.layers.0.q_proj"]
    assert _verify_lora_scope(model, targets) == targets
    model.model.audio_tower.layers[0].q_proj = AdaptedLinear()

    import pytest
    with pytest.raises(RuntimeError, match="outside the selected Gemma language_model targets"):
        _verify_lora_scope(model, targets)


def test_stage1_manifest_is_scoped_to_the_adapter_checkpoint_root(tmp_path):
    adapter_dir = tmp_path / "checkpoints" / "experiment-a" / "adapter-smoke"

    assert _training_manifest_path(adapter_dir) == tmp_path / "checkpoints" / "experiment-a" / "training_manifest.json"


def test_frozen_towers_are_float_bf16_and_reject_quantized_linear_modules():
    from torch import nn
    import pytest

    class Tower(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(2, 2, bias=False, dtype=torch.float32)

    class Gemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = Tower()
            self.vision_tower = Tower()

    model = Gemma()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runtime = _freeze_float_towers(model, dtype=torch.bfloat16)
    assert runtime["audio_tower"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["vision_tower"]["dtypes"] == ["torch.bfloat16"]
    assert runtime["audio_tower"]["requires_grad"] is False

    class Linear4bit(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones((2, 2), dtype=torch.uint8), requires_grad=False)

    model.audio_tower.projection = Linear4bit()
    with pytest.raises(RuntimeError, match="tower.*quantizzato"):
        _freeze_float_towers(model, dtype=torch.bfloat16)


def test_audio_and_vision_embedding_bridges_return_to_frozen_bf16():
    from torch import nn

    class Gemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_audio = nn.Module()
            self.model.embed_audio.embedding_projection = nn.Linear(2, 2, dtype=torch.float32)
            self.model.embed_vision = nn.Module()
            self.model.embed_vision.embedding_projection = nn.Linear(2, 2, dtype=torch.float32)

    model = Gemma()
    report = _freeze_float_bridges(model, dtype=torch.bfloat16)
    assert report["model.embed_audio.embedding_projection"]["dtypes"] == ["torch.bfloat16"]
    assert report["model.embed_vision.embedding_projection"]["dtypes"] == ["torch.bfloat16"]
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_input_embeddings_and_tied_output_head_match_audio_compute_dtype():
    from torch import nn

    class Gemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.input = nn.Embedding(8, 4, dtype=torch.bfloat16)
            self.output = nn.Linear(4, 8, bias=False, dtype=torch.bfloat16)
            self.output.weight = self.input.weight

        def get_input_embeddings(self):
            return self.input

        def get_output_embeddings(self):
            return self.output

    class PeftWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = Gemma()

        def get_base_model(self):
            return self.base

    model = PeftWrapper()
    keys_before = set(model.state_dict())
    report = _freeze_input_embeddings(model, dtype=torch.bfloat16)
    assert report["input_embedding_dtype"] == "torch.bfloat16"
    assert report["output_head_was_tied"] is True
    assert report["output_head_tied"] is False
    assert model.base.input.weight is not model.base.output.weight
    assert model.base.input.weight.dtype == torch.bfloat16
    assert model.base.output.weight.dtype == torch.float32
    assert not model.base.input.weight.requires_grad
    assert not model.base.output.weight.requires_grad
    keys_after_first_call = set(model.state_dict())
    assert keys_after_first_call == keys_before
    hook_count = len(model.base.output._forward_pre_hooks)
    second_report = _freeze_input_embeddings(model, dtype=torch.bfloat16)
    assert second_report["output_head_input_cast"] == "torch.float32"
    assert len(model.base.output._forward_pre_hooks) == hook_count == 1
    assert set(model.state_dict()) == keys_before

    activation = torch.randn(2, 4, dtype=torch.bfloat16, requires_grad=True)
    logits = model.base.output(activation)
    assert logits.dtype == torch.float32
    logits.sum().backward()
    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()
    assert activation.grad.abs().sum() > 0
    assert model.base.output.weight.grad is None


def test_selective_qlora_preparation_keeps_large_frozen_embeddings_and_multimodal_towers_bf16():
    from types import SimpleNamespace
    from torch import nn
    from training.duplex_qlora.stage1 import _prepare_gemma_for_kbit_training

    observed = {"float32_module_to_calls": [], "checkpoint_kwargs": None, "input_grad_hook": 0}

    class GuardedEmbedding(nn.Embedding):
        def __init__(self, num_embeddings=8, embedding_dim=4):
            super().__init__(num_embeddings, embedding_dim, dtype=torch.bfloat16)

        def to(self, *args, **kwargs):
            requested = kwargs.get("dtype")
            if args and isinstance(args[0], torch.dtype):
                requested = args[0]
            if requested == torch.float32:
                observed["float32_module_to_calls"].append(type(self).__name__)
                raise AssertionError("large frozen embedding must not be promoted to FP32")
            return super().to(*args, **kwargs)

    class Gemma4RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4, dtype=torch.bfloat16))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.input_layernorm = Gemma4RMSNorm()
            self.model.language_model.embed_tokens = GuardedEmbedding(32768, 256)
            self.model.language_model.ple_embeddings = GuardedEmbedding(65536, 128)
            self.model.audio_tower = GuardedEmbedding()
            self.model.vision_tower = GuardedEmbedding()
            self.model.embed_audio = nn.Module()
            self.model.embed_audio.embedding_projection = GuardedEmbedding()
            self.model.embed_vision = nn.Module()
            self.model.embed_vision.embedding_projection = GuardedEmbedding()

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            observed["checkpoint_kwargs"] = gradient_checkpointing_kwargs

        def enable_input_require_grads(self):
            observed["input_grad_hook"] += 1

    model = Model()
    report = _prepare_gemma_for_kbit_training(model)

    assert observed["float32_module_to_calls"] == []
    assert model.model.language_model.input_layernorm.weight.dtype == torch.float32
    assert model.model.language_model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.model.language_model.ple_embeddings.weight.dtype == torch.bfloat16
    assert model.model.audio_tower.weight.dtype == torch.bfloat16
    assert model.model.vision_tower.weight.dtype == torch.bfloat16
    assert model.model.embed_audio.embedding_projection.weight.dtype == torch.bfloat16
    assert model.model.embed_vision.embedding_projection.weight.dtype == torch.bfloat16
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert model.config.use_cache is True
    assert observed["checkpoint_kwargs"] is None
    assert report["cache_enabled_for_shared_kv"] is True
    assert report["gradient_checkpointing"] is False
    assert observed["input_grad_hook"] == 0
    assert report["norms_cast_to_fp32"] == ["model.language_model.input_layernorm"]


def test_native_shared_kv_training_rejects_cache_disabled_mode_with_actionable_error():
    from types import SimpleNamespace
    from torch import nn
    from training.duplex_qlora.stage1 import _prepare_gemma_for_kbit_training

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(use_cache=True)
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.input_layernorm = nn.LayerNorm(4, dtype=torch.bfloat16)
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.is_kv_shared_layer = True
            self.model.language_model.layer = layer

    try:
        _prepare_gemma_for_kbit_training(Model(), use_cache_for_training=False)
    except ValueError as exc:
        assert "shared-KV" in str(exc)
        assert "use_cache=True" in str(exc)
    else:
        raise AssertionError("Gemma4 shared-KV training must not silently use the non-shared no-cache path")


def test_native_shared_kv_cache_probe_checks_dynamic_cache_source_indices_from_topology():
    from types import SimpleNamespace
    from torch import nn
    from training.duplex_qlora.stage1 import _validate_native_shared_kv_cache

    model = nn.Module()
    model.language_model = nn.Module()
    for layer_idx, source_idx in ((15, 13), (16, 14)):
        layer = nn.Module()
        layer.self_attn = nn.Module()
        layer.self_attn.layer_idx = layer_idx
        layer.self_attn.is_kv_shared_layer = True
        layer.self_attn.kv_shared_layer_index = source_idx
        setattr(model.language_model, f"layer_{layer_idx}", layer)

    cache = SimpleNamespace(shared_layers={13: object(), 14: object()})
    report = _validate_native_shared_kv_cache(model, SimpleNamespace(past_key_values=cache))

    assert report["first_shared_layer_index"] == 15
    assert report["shared_attention_layer_count"] == 2
    assert report["expected_cache_source_indices"] == [13, 14]
    assert report["observed_cache_source_indices"] == [13, 14]


def test_batch_runtime_summary_reports_actual_task_source_action_and_resource_counts():
    from training.duplex_qlora.stage1 import _summarize_batch_runtime

    summary = _summarize_batch_runtime([
        {
            "task_kind": "asr", "source": "synthetic_v3_clean_asr", "action": None,
            "step_elapsed_seconds": 3.0, "cuda_peak_memory_allocated_bytes": 1000,
            "cuda_peak_memory_reserved_bytes": 1200,
        },
        {
            "task_kind": "conversation_response", "source": "authored_synthetic_audio_candidate",
            "action": "BACKCHANNEL", "step_elapsed_seconds": 5.0,
            "cuda_peak_memory_allocated_bytes": 2000, "cuda_peak_memory_reserved_bytes": 2400,
        },
    ])

    assert summary["examples_by_task"] == {"asr": 1, "conversation_response": 1}
    assert summary["examples_by_source"] == {
        "authored_synthetic_audio_candidate": 1,
        "synthetic_v3_clean_asr": 1,
    }
    assert summary["examples_by_action"] == {"BACKCHANNEL": 1}
    assert summary["mean_step_elapsed_seconds"] == 4.0
    assert summary["max_step_elapsed_seconds"] == 5.0
    assert summary["cuda_peak_memory_allocated_bytes"] == 2000
    assert summary["cuda_peak_memory_reserved_bytes"] == 2400


def test_stage1_entrypoint_rejects_use_cache_false_before_loading_data_or_model():
    from training.duplex_qlora.stage1 import train_qlora

    try:
        train_qlora(
            train_jsonl="missing-train.jsonl",
            model_path="google/gemma-4-e2b-it",
            adapter_dir="missing-output",
            max_steps=4,
            conditioning_mode="audio_asr_mix_v2",
            use_cache_for_training=False,
        )
    except ValueError as exc:
        assert "use_cache=False" in str(exc)
        assert "condivisione K/V nativa" in str(exc)
    else:
        raise AssertionError("Stage1 Gemma4 must reject no-cache training before reading the dataset")


def test_feature_batch_uses_compute_dtype_not_first_quantized_parameter():
    from torch import nn

    model = nn.Module()
    model.quantized_first = nn.Module()
    model.quantized_first.weight = nn.Parameter(torch.ones(1, dtype=torch.uint8), requires_grad=False)
    inputs = {
        "input_features": torch.ones((1, 3, 8), dtype=torch.float32),
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "labels": torch.tensor([[-100, 2, 3]], dtype=torch.int64),
    }
    moved, stats = _move_batch_to_device(
        inputs, device=torch.device("cpu"), compute_dtype=torch.bfloat16
    )
    assert moved["input_features"].dtype == torch.bfloat16
    assert moved["input_ids"].dtype == torch.int64
    assert stats["non_ignored_label_tokens"] == 2


def test_response_audio_history_is_limited_to_kernel_listener_context(tmp_path):
    audio_path = tmp_path / "long.wav"
    audio = np.zeros((240000, 2), dtype=np.float32)
    audio[7 * 24000 + 12000:7 * 24000 + 14000, 0] = 0.2
    sf.write(audio_path, audio, 24000, subtype="FLOAT")
    conversation = _conversation(
        audio_path,
        segments=[
            {"role": "user", "text": "domanda", "start_ms": 7800, "end_ms": 7950, "time_mask": True, "text_mask": True},
            {"role": "assistant", "text": "risposta", "start_ms": 8000, "end_ms": 8200, "time_mask": True, "text_mask": True},
        ],
    )
    conversation["duration_ms"] = 10000

    record = _training_records([conversation])[0]

    assert record["audio_start_ms"] == 2880
    assert record["audio_end_ms"] == 8000
    assert record["audio_end_ms"] - record["audio_start_ms"] == record["audio_history_cap_ms"]


def test_conversation_with_unusable_language_targets_is_excluded_from_stage1(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, np.ones((24000, 2), dtype=np.float32) * 0.1, 24000)
    conversation = _conversation(
        audio_path,
        source="synthetic_it_t6_v3",
        segments=[
            {"role": "assistant", "text": "risposta con testo non affidabile", "start_ms": 500, "end_ms": 900,
             "text_mask": True, "time_mask": True},
        ],
    )
    conversation["synthetic"] = True
    conversation["language_training_eligible"] = False
    conversation["frames"] = [{"frame_id": 0, "target_action": "YIELD", "action_mask": True}]

    assert _training_records([conversation]) == []
    assert count_response_examples([conversation]) == 0
    assert conversation["frames"][0]["action_mask"] is True


def test_role_free_asr_uses_verified_physical_channel_and_exact_span(tmp_path):
    audio_path = tmp_path / "stereo.wav"
    audio = np.zeros((24000, 2), dtype=np.float32)
    audio[7200:12000, 1] = 0.4
    sf.write(audio_path, audio, 24000)
    conversation = _conversation(
        audio_path,
        source="DuplexConv",
        segments=[
            {
                "role": None,
                "task_kind": "asr",
                "text": "parole originali",
                "start_ms": 300,
                "end_ms": 500,
                "channel_index": 1,
                "audio_channel_verified": True,
                "time_mask": True,
                "text_mask": True,
            }
        ],
    )

    records = _training_records([conversation])
    wave = _read_audio_for_record(records[0], target_sample_rate_hz=16000)

    assert len(records) == 1
    assert records[0]["task_kind"] == "asr"
    assert records[0]["audio_channel_index"] == 1
    assert records[0]["audio_start_ms"] == 300
    assert records[0]["audio_end_ms"] == 500
    assert wave.shape == (3200,)
    assert np.max(np.abs(wave)) > 0.2


def test_audio_example_cannot_fall_back_to_marker_only_processor_input(tmp_path):
    audio_path = tmp_path / "mono.wav"
    sf.write(audio_path, np.ones(16000, dtype=np.float32) * 0.1, 16000)
    record = {
        "audio_path": str(audio_path),
        "sample_rate_hz": 16000,
        "audio_channel_index": 0,
        "audio_start_ms": 0,
        "audio_end_ms": 1000,
        "audio_mask": True,
        "task_kind": "asr",
        "language": "it",
        "context_text": "",
        "target_text": "ciao mondo",
    }

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            prompt = str(messages[0]["content"])
            suffix = "" if add_generation_prompt else " ciao mondo"
            return prompt + suffix

        def __call__(self, text, *, add_special_tokens, return_tensors):
            return {"input_ids": torch.tensor([[ord(char) for char in text]], dtype=torch.long)}

    class Processor:
        tokenizer = Tokenizer()
        audio_token = "<|audio|>"

        def __init__(self):
            self.calls = []

        def __call__(self, *, text, audio, sampling_rate, return_tensors, padding):
            encoded_text = text[0]
            before, marker, after = encoded_text.partition(self.audio_token)
            assert marker == self.audio_token
            token_ids = [ord(char) for char in before] + [32001, 32002, 32003]
            token_ids.extend(ord(char) for char in after)
            ids = torch.tensor([token_ids], dtype=torch.long)
            self.calls.append({"audio": audio, "sampling_rate": sampling_rate, "input_ids": ids})
            return {"input_ids": ids, "input_features": torch.tensor([[float(np.max(np.abs(audio[0])))]])}

    processor = Processor()
    encoded = build_training_inputs(processor, record)

    assert len(processor.calls) == 2  # prompt-only and full-chat share processor/audio/config
    assert all(call["sampling_rate"] == 16000 for call in processor.calls)
    assert all(call["audio"][0].ndim == 1 for call in processor.calls)
    assert np.array_equal(processor.calls[0]["audio"][0], processor.calls[1]["audio"][0])
    assert torch.allclose(encoded["input_features"], torch.tensor([[0.1]]), atol=1e-4)
    prompt_length = processor.calls[0]["input_ids"].shape[1]
    expected_target = torch.tensor([ord(char) for char in " ciao mondo"], dtype=torch.long)
    assert torch.all(encoded["labels"][:, :prompt_length] == -100)
    assert torch.equal(encoded["labels"][0, prompt_length:], expected_target)


def test_source_schedule_enforces_synthetic_examples_and_target_token_share():
    records = [
        {"synthetic": True, "target_text": "sintetico breve"},
        {"synthetic": False, "target_text": "ausiliario molto lungo con molte parole"},
    ]

    schedule, report = _make_training_schedule(
        records,
        max_steps=20,
        target_token_counts=[2, 7],
        synthetic_fraction=0.8,
        seed=3,
    )

    assert len(schedule) == 20
    assert report["synthetic_example_fraction"] >= 0.7
    assert report["synthetic_target_token_fraction"] >= 0.7


def test_audio_asr_mix_samples_fixed_strata_and_routes_each_prompt_without_current_text():
    from duplex_kernel.audio_response_conditioning import causal_audio_window

    direct = []
    for action, target_start, user_end, active in (
        ("SPEAK", 1000, 800, False),
        ("BACKCHANNEL", 1600, 2400, True),
        ("REPLAN", 2000, 1200, False),
    ):
        window = causal_audio_window(
            action=action,
            target_start_ms=target_start,
            current_user_start_ms=0,
            current_user_end_ms=user_end,
            current_user_active=active,
        )
        direct.append({
            "task_mix_contract": "audio_asr_mix_v1",
            "audio_response_contract": "audio_response_conditioning_v1",
            "task_kind": "conversation_response",
            "conditioning_mode": "baselineaudio_response",
            "audio_mask": True,
            "action_is_prompt_conditioning": False,
            "action": action,
            "action_label": action.lower(),
            "sampling_stratum": {
                "SPEAK": "labelled_speak",
                "BACKCHANNEL": "labelled_backchannel",
                "REPLAN": "labelled_replan",
            }[action],
            "action_label_provenance": "measured_v5_event_schedule",
            "target_start_ms": target_start,
            "action_event_start_ms": target_start,
            "action_event_end_ms": target_start + 160,
            "current_user_start_ms": 0,
            "current_user_end_ms": user_end,
            "current_user_active_at_target": active,
            "max_audio_ms": 30000,
            **window,
            "audio_path": f"/{action}.wav",
            "source_audio_sha256": "a" * 64,
            "sample_rate_hz": 16000,
            "audio_channel_index": 0,
            "source_group_id": f"group-{action}",
            "source": "authored_synthetic_audio_candidate",
            "synthetic": True,
            "language": "it-IT",
            "history_messages": [{"role": "user", "content": "Domanda precedente."}],
            "context_text": "",
            "target_text": {"SPEAK": "La risposta completa.", "BACKCHANNEL": "Capito.", "REPLAN": "Rivedo il piano."}[action],
        })
    asr = {
        "task_mix_contract": "audio_asr_mix_v1",
        "task_kind": "asr",
        "conditioning_mode": "native_audio_asr",
        "audio_mask": True,
        "audio_path": "/ASR.wav",
        "audio_channel_index": 0,
        "sample_rate_hz": 16000,
        "audio_start_ms": 0,
        "audio_end_ms": 1000,
        "source_group_id": "group-ASR",
        "source": "synthetic_v3_clean_asr",
        "synthetic": False,
        "language": "it-IT",
        "history_messages": [],
        "context_text": "",
        "target_text": "Trascrizione fedele.",
    }

    validated, by_task = _validate_audio_asr_mix_records([*direct, asr], split="train")
    schedule, report = _make_audio_asr_mix_schedule(validated, max_steps=10, seed=9)
    assert len(schedule) == 10
    assert report["examples_by_task"] == {"asr": 2, "conversation_response": 8}
    assert report["audio_response_by_action"] == {"SPEAK": 4, "BACKCHANNEL": 2, "REPLAN": 2}
    assert report["synthetic_example_fraction"] == 0.8
    assert len(by_task["asr"]) == 1 and len(by_task["conversation_response"]) == 3

    class Tokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            self.messages = messages
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    class Processor:
        tokenizer = Tokenizer()
        audio_token = "<|audio|>"

    processor = Processor()
    direct_prompt = _format_chat(processor, direct[0], add_target=False)
    assert "Domanda precedente." in direct_prompt
    assert "<|audio|>" in direct_prompt
    assert "current user secret" not in direct_prompt
    assert "SPEAK" not in direct_prompt
    asr_prompt = _format_chat(processor, asr, add_target=False)
    assert "<|audio|>" in asr_prompt
    assert "Domanda precedente." not in asr_prompt


def test_protocol_v2_mix_routes_generic_audio_prompt_and_preserves_asr_targets():
    from duplex_kernel.audio_response_protocol import format_audio_response_v2_target
    from duplex_kernel.asr_prompt import asr_prompt_content

    direct = []
    for action, target_start, user_end, active, heard, say in (
        ("SPEAK", 1000, 800, False, "Prima richiesta completa.", "La risposta completa."),
        ("BACKCHANNEL", 1600, 2400, True, "", "Capito."),
        ("REPLAN", 2000, 1200, False, "Correzione completa.", "Rivedo il piano."),
    ):
        from duplex_kernel.audio_response_conditioning import causal_audio_window

        window = causal_audio_window(
            action=action,
            target_start_ms=target_start,
            current_user_start_ms=0,
            current_user_end_ms=user_end,
            current_user_active=active,
        )
        target = format_audio_response_v2_target(heard=heard, action=action, say=say)
        direct.append({
            "task_mix_contract": "audio_asr_mix_v2",
            "audio_response_protocol": "heard_action_say_v2",
            "task_kind": "conversation_response",
            "conditioning_mode": "baselineaudio_response",
            "audio_mask": True,
            "action_is_prompt_conditioning": False,
            "action": action,
            "action_label": action.lower(),
            "sampling_stratum": {
                "SPEAK": "labelled_speak",
                "BACKCHANNEL": "labelled_backchannel",
                "REPLAN": "labelled_replan",
            }[action],
            "action_label_provenance": "measured_v5_event_schedule",
            "target_start_ms": target_start,
            "action_event_start_ms": target_start,
            "action_event_end_ms": target_start + 160,
            "current_user_start_ms": 0,
            "current_user_end_ms": user_end,
            "current_user_active_at_target": active,
            "max_audio_ms": 30000,
            **window,
            "audio_path": f"/{action}.wav",
            "source_audio_sha256": "a" * 64,
            "sample_rate_hz": 16000,
            "audio_channel_index": 0,
            "source_group_id": f"group-{action}",
            "source": "authored_synthetic_audio_candidate",
            "synthetic": True,
            "language": "it-IT",
            "history_messages": [{"role": "user", "content": "Domanda precedente."}],
            "context_text": "",
            "target_text": target,
            "target_heard_text": heard,
            "target_say_text": say,
        })
    asr = {
        "task_mix_contract": "audio_asr_mix_v2",
        "task_kind": "asr",
        "conditioning_mode": "native_audio_asr",
        "audio_mask": True,
        "audio_path": "/ASR.wav",
        "audio_channel_index": 0,
        "sample_rate_hz": 16000,
        "audio_start_ms": 0,
        "audio_end_ms": 1000,
        "source_group_id": "group-ASR",
        "source": "synthetic_v3_clean_asr",
        "synthetic": False,
        "language": "it-IT",
        "history_messages": [],
        "context_text": "",
        "target_text": "Trascrizione fedele.",
    }

    validated, by_task = _validate_audio_asr_mix_records(
        [*direct, asr], split="train", response_protocol="v2",
    )
    schedule, report = _make_audio_asr_mix_schedule(
        validated, max_steps=5, seed=9, response_protocol="v2",
    )
    assert report["task_mix_contract"] == "audio_asr_mix_v2"
    assert report["examples_by_task"] == {"asr": 1, "conversation_response": 4}
    assert report["audio_response_by_action"] == {"SPEAK": 2, "BACKCHANNEL": 1, "REPLAN": 1}
    assert len(by_task["asr"]) == 1
    assert by_task["asr"][0]["target_text"] == asr["target_text"]

    test_rows, test_by_task = _validate_audio_asr_mix_records(
        direct, split="test", response_protocol="v2", require_asr=False,
    )
    assert len(test_rows) == 3
    assert test_by_task == {"conversation_response": test_rows, "asr": []}

    class Tokenizer:
        def __init__(self):
            self.messages = None

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert tokenize is False
            self.messages = messages
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    class Processor:
        tokenizer = Tokenizer()
        audio_token = "<|audio|>"

    processor = Processor()
    prompts = [_format_chat(processor, row, add_target=False) for row in direct]
    assert prompts[0] == prompts[1] == prompts[2]
    assert "Domanda precedente." in prompts[0]
    assert "<|audio|>" in prompts[0]
    assert "Prima richiesta completa." not in prompts[0]
    assert "Correzione completa." not in prompts[0]
    assert "BACKCHANNEL" in prompts[0]  # protocol vocabulary only; it is identical across classes

    asr_prompt = _format_chat(processor, asr, add_target=False)
    assert processor.tokenizer.messages == [{"role": "user", "content": asr_prompt_content("it-IT", "<|audio|>")}]
    assert "heard_action_say_v2" not in asr_prompt
    assert "Trascrizione fedele." not in asr_prompt


def test_audio_asr_mix_prioritizes_v5_speak_replan_and_falls_back_to_legacy():
    records = []
    for action in ("SPEAK", "REPLAN"):
        for source, count in (("authored_synthetic_audio_candidate", 4), ("synthetic_it_t6", 4)):
            records.extend({
                "task_mix_contract": "audio_asr_mix_v1",
                "task_kind": "conversation_response",
                "action": action,
                "source": source,
                "synthetic": True,
            } for _ in range(count))
    records.extend({
        "task_mix_contract": "audio_asr_mix_v1",
        "task_kind": "conversation_response",
        "action": "BACKCHANNEL",
        "source": "authored_synthetic_audio_candidate",
        "synthetic": True,
    } for _ in range(4))
    records.extend({
        "task_mix_contract": "audio_asr_mix_v1",
        "task_kind": "asr",
        "source": "synthetic_v3_clean_asr",
        "synthetic": False,
    } for _ in range(4))

    schedule, report = _make_audio_asr_mix_schedule(records, max_steps=20, seed=260928)
    by_action_source = {}
    for record in schedule:
        if record["task_kind"] == "conversation_response":
            key = (record["action"], record["source"])
            by_action_source[key] = by_action_source.get(key, 0) + 1
    assert report["examples_by_task"] == {"asr": 4, "conversation_response": 16}
    assert report["audio_response_by_action"] == {"SPEAK": 8, "BACKCHANNEL": 4, "REPLAN": 4}
    assert by_action_source[("SPEAK", "authored_synthetic_audio_candidate")] == 6
    assert by_action_source[("SPEAK", "synthetic_it_t6")] == 2
    assert by_action_source[("REPLAN", "authored_synthetic_audio_candidate")] == 3
    assert by_action_source[("REPLAN", "synthetic_it_t6")] == 1
    assert by_action_source[("BACKCHANNEL", "authored_synthetic_audio_candidate")] == 4
    assert report["audio_response_by_action_source"] == {
        "BACKCHANNEL": {"authored_synthetic_audio_candidate": 4},
        "REPLAN": {"authored_synthetic_audio_candidate": 3, "synthetic_it_t6": 1},
        "SPEAK": {"authored_synthetic_audio_candidate": 6, "synthetic_it_t6": 2},
    }

    legacy_only = [
        {"task_mix_contract": "audio_asr_mix_v1", "task_kind": "asr", "source": "synthetic_v3_clean_asr", "synthetic": False},
        {"task_mix_contract": "audio_asr_mix_v1", "task_kind": "conversation_response", "action": "SPEAK", "source": "synthetic_it_t6", "synthetic": True},
        {"task_mix_contract": "audio_asr_mix_v1", "task_kind": "conversation_response", "action": "BACKCHANNEL", "source": "authored_synthetic_audio_candidate", "synthetic": True},
        {"task_mix_contract": "audio_asr_mix_v1", "task_kind": "conversation_response", "action": "REPLAN", "source": "synthetic_it_t6", "synthetic": True},
    ]
    fallback_schedule, _ = _make_audio_asr_mix_schedule(legacy_only, max_steps=4, seed=1)
    assert {record.get("source") for record in fallback_schedule if record.get("task_kind") == "conversation_response"} == {
        "synthetic_it_t6", "authored_synthetic_audio_candidate",
    }
