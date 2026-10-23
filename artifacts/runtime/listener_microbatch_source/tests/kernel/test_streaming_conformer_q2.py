from __future__ import annotations

import pytest
import torch

from transformers.models.gemma4 import modeling_gemma4
from transformers.models.gemma4.configuration_gemma4 import (
    Gemma4AudioConfig,
    Gemma4TextConfig,
)

from duplex_kernel.streaming_conformer import (
    AudioStreamCursor,
    StreamingConformerAudioTower,
)


@pytest.fixture(autouse=True)
def _single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def _tiny_audio_tower(seed: int = 23):
    torch.manual_seed(seed)
    audio_config = Gemma4AudioConfig(
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        subsampling_conv_channels=(128, 4),
        conv_kernel_size=5,
        attention_chunk_size=12,
        attention_context_left=13,
        attention_context_right=0,
        output_proj_dims=24,
        rms_norm_eps=1e-6,
        use_clipped_linears=True,
    )
    text_config = Gemma4TextConfig(
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        layer_types=["full_attention"],
        rope_parameters={
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
            "rope_theta": 1_000_000.0,
        },
    )
    audio_tower = modeling_gemma4.Gemma4AudioModel(audio_config).eval()
    embed_audio = modeling_gemma4.Gemma4MultimodalEmbedder(audio_config, text_config).eval()
    return audio_config, audio_tower, embed_audio


def _offline_reference(audio_tower, mel, mask):
    hidden, output_mask = audio_tower.subsample_conv_projection(mel, mask)
    position_embeddings = audio_tower.rel_pos_enc(hidden)
    attention_mask = modeling_gemma4.create_bidirectional_mask(
        config=audio_tower.config,
        inputs_embeds=hidden,
        attention_mask=output_mask,
        and_mask_function=modeling_gemma4.sliding_window_mask_function(
            (audio_tower.config.attention_context_left - 1,
             audio_tower.config.attention_context_right)
        ),
    )
    attention_mask = audio_tower._convert_4d_mask_to_blocked_5d(attention_mask)
    layers = []
    for layer in audio_tower.layers:
        hidden = layer(
            hidden,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        layers.append(hidden)
    return audio_tower.output_proj(hidden), layers


def _cursor(session: str, frame: int, *, generation: int = 0, epoch: str = "epoch-0"):
    return AudioStreamCursor(session, generation, epoch, frame, frame)


def test_compute_step_does_not_mutate_layer_state_with_copy_ops():
    from torch.utils._python_dispatch import TorchDispatchMode

    class CopyOperationRecorder(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.copy_operations = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten.copy_.default:
                self.copy_operations.append(func)
            return func(*args, **(kwargs or {}))

    _, audio_tower, embed_audio = _tiny_audio_tower(85)
    streaming = StreamingConformerAudioTower(
        audio_tower,
        embed_audio,
        device="cpu",
        dtype=torch.float32,
        max_sessions=2,
    )
    tick = torch.randn((1, 8, 128), generator=torch.Generator().manual_seed(86))
    recorder = CopyOperationRecorder()
    with torch.inference_mode(), recorder:
        streaming._compute_step(
            tick,
            torch.ones((1, 8), dtype=torch.bool),
            torch.tensor([0]),
            torch.tensor([True]),
        )
    assert recorder.copy_operations == []


def test_benchmark_primary_and_w240_diagnostic_thresholds_are_separate():
    from tools import benchmark_streaming_conformer_q2 as benchmark

    summary = {
        "after_first_13_tokens": {"mean": 0.9998, "min": 0.9985},
    }
    primary = benchmark._cached_quality_thresholds(summary)
    diagnostic = benchmark._w240_diagnostic_thresholds(summary)

    assert primary["required_after_first_13_tokens"] == {
        "mean_minimum": 0.9995,
        "minimum_minimum": 0.9990,
    }
    assert primary["passed_after_first_13_tokens"] is False
    assert diagnostic["required_after_first_13_tokens"] == {"minimum_minimum": 0.9970}
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["passed_after_first_13_tokens"] is True
    assert benchmark.LATENCY_TARGET_MS == 1.90
    assert benchmark._latency_within_budget([1.90, 1.90]) is True
    assert benchmark._latency_within_budget([1.9001, 1.9001]) is False
    assert benchmark.REFERENCE_SM_CLOCK_MHZ == 1995


def test_benchmark_previous_report_loader_accepts_absent_and_empty_output(tmp_path):
    from tools import benchmark_streaming_conformer_q2 as benchmark

    absent = tmp_path / "absent.json"
    empty = tmp_path / "empty.json"
    empty.write_text("  \n", encoding="utf-8")

    assert benchmark._load_previous_benchmark_report(absent) == {}
    assert benchmark._load_previous_benchmark_report(empty) == {}


def test_python_runtime_source_mirror_sha256_matches():
    import hashlib
    from pathlib import Path

    root = next(
        parent for parent in Path(__file__).resolve().parents
        if (parent / "artifacts" / "runtime" / "listener_microbatch_source").is_dir()
    )
    mirror = root / "artifacts" / "runtime" / "listener_microbatch_source"
    source_paths = (
        Path("duplex_kernel/streaming_conformer.py"),
        Path("duplex_kernel/listener_microbatch_serving.py"),
        Path("duplex_kernel/listener_microbatch_frontend.py"),
    )
    for relative in source_paths:
        source_digest = hashlib.sha256((root / relative).read_bytes()).digest()
        mirror_digest = hashlib.sha256((mirror / relative).read_bytes()).digest()
        assert source_digest == mirror_digest, f"runtime Python source mirror differs: {relative}"


def test_streaming_tower_close_requires_idle_and_releases_cache():
    _, audio_tower, embed_audio = _tiny_audio_tower(78)
    streaming = StreamingConformerAudioTower(
        audio_tower,
        embed_audio,
        device="cpu",
        dtype=torch.float32,
        max_sessions=2,
    )
    assert streaming.optimize_cuda is False
    tick = torch.randn((1, 8, 128), dtype=torch.float32)
    prepared = streaming.prepare_batch(tick, (_cursor("close-session", 0),))
    with pytest.raises(RuntimeError, match="active batch"):
        streaming.close()
    prepared.abort()
    assert streaming.committed_positions == ()
    streaming.close()
    streaming.close()
    assert streaming.committed_positions == ()
    with pytest.raises(RuntimeError, match="is closed"):
        streaming.reset("close-session")


def test_streaming_conformer_kernel_module_does_not_import_triton_eagerly(monkeypatch):
    import builtins
    import importlib
    import sys

    module_name = "duplex_kernel.streaming_conformer_kernels"
    module = sys.modules.get(module_name)
    original_import = builtins.__import__
    attempted: list[str] = []

    def reject_triton(name, *args, **kwargs):
        if name == "triton" or name.startswith("triton."):
            attempted.append(name)
            raise AssertionError("Triton import must be lazy until CUDA opt-in")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(builtins, "__import__", reject_triton)
        if module is None:
            module = importlib.import_module(module_name)
        else:
            importlib.reload(module)
    assert attempted == []
    assert module_name in sys.modules


def _commit_tick(tower, chunks, cursors, mask=None):
    prepared = tower.prepare_batch(chunks, cursors, input_mask=mask)
    result = (
        prepared.raw_tokens.detach().clone(),
        prepared.soft_tokens.detach().clone(),
        prepared.subsampled.detach().clone(),
        tuple(value.detach().clone() for value in prepared.layer_tokens),
    )
    prepared.commit()
    return result


def test_cached_subsampler_every_layer_and_complete_output_match_125_causal_ticks():
    _, audio_tower, embed_audio = _tiny_audio_tower()
    generator = torch.Generator().manual_seed(9917)
    mel = torch.randn((1, 125 * 8, 128), generator=generator, dtype=torch.float32)
    mask = torch.ones((1, 125 * 8), dtype=torch.bool)

    with torch.inference_mode():
        reference, reference_layers = _offline_reference(audio_tower, mel, mask)
        reference_soft = embed_audio(reference)
        streaming = StreamingConformerAudioTower(
            audio_tower,
            embed_audio,
            device="cpu",
            dtype=torch.float32,
            max_sessions=4,
        )
        raw, soft, subsampled, layer_tokens = [], [], [], [[] for _ in reference_layers]
        for frame in range(125):
            tick = mel[:, frame * 8:(frame + 1) * 8].clone()
            values = _commit_tick(streaming, tick, (_cursor("long", frame),))
            raw.append(values[0])
            soft.append(values[1])
            subsampled.append(values[2])
            for index, value in enumerate(values[3]):
                layer_tokens[index].append(value)

    actual_raw = torch.cat(raw, dim=1)
    actual_soft = torch.cat(soft, dim=1)
    actual_subsampled = torch.cat(subsampled, dim=1)
    torch.testing.assert_close(actual_subsampled, audio_tower.subsample_conv_projection(mel, mask)[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual_raw, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual_soft, reference_soft, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(layer_tokens, reference_layers, strict=True):
        torch.testing.assert_close(torch.cat(actual, dim=1), expected, rtol=2e-5, atol=2e-6)


def test_streaming_subsampler_and_attention_preserve_nontrivial_input_masks():
    _, audio_tower, embed_audio = _tiny_audio_tower(31)
    generator = torch.Generator().manual_seed(712)
    mel = torch.randn((1, 7 * 8, 128), generator=generator)
    mask = torch.ones((1, 7 * 8), dtype=torch.bool)
    mask[:, (4, 10, 16, 31, 40)] = False
    streaming = StreamingConformerAudioTower(
        audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2
    )

    with torch.inference_mode():
        reference, reference_layers = _offline_reference(audio_tower, mel, mask)
        actual_raw, actual_layers = [], [[] for _ in reference_layers]
        for frame in range(7):
            prepared = streaming.prepare_batch(
                mel[:, frame * 8:(frame + 1) * 8].clone(),
                (_cursor("masked", frame),),
                input_mask=mask[:, frame * 8:(frame + 1) * 8],
            )
            actual_raw.append(prepared.raw_tokens.clone())
            for index, value in enumerate(prepared.layer_tokens):
                actual_layers[index].append(value.clone())
            prepared.commit()

    torch.testing.assert_close(torch.cat(actual_raw, dim=1), reference, rtol=2e-5, atol=2e-6)
    for actual, expected in zip(actual_layers, reference_layers, strict=True):
        torch.testing.assert_close(torch.cat(actual, dim=1), expected, rtol=2e-5, atol=2e-6)


def test_two_session_reordering_reset_and_gap_do_not_cross_contaminate():
    _, audio_tower, embed_audio = _tiny_audio_tower(41)
    generator = torch.Generator().manual_seed(125)
    chunks = torch.randn((9, 8, 128), generator=generator)
    shared = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=8)
    separate_a = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2)
    separate_b = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2)

    with torch.inference_mode():
        # Both independent sessions see the same sequence. Compare every tick
        # before exercising row reordering, reset, and cursor gaps below.
        for frame in range(4):
            pair = chunks[frame:frame + 1].expand(2, -1, -1).clone()
            actual = _commit_tick(
                shared, pair, (_cursor("A", frame), _cursor("B", frame))
            )[0]
            ref_a = _commit_tick(separate_a, chunks[frame:frame + 1], (_cursor("A", frame),))[0]
            ref_b = _commit_tick(separate_b, chunks[frame:frame + 1], (_cursor("B", frame),))[0]
            torch.testing.assert_close(actual[0:1], ref_a, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual[1:2], ref_b, rtol=2e-5, atol=2e-6)

        # The same next tick is submitted with reversed microbatch row order.
        ba = _commit_tick(
            shared,
            chunks[4:5].expand(2, -1, -1).clone(),
            (_cursor("B", 4), _cursor("A", 4)),
        )[0]
        b_ref = _commit_tick(separate_b, chunks[4:5], (_cursor("B", 4),))[0]
        a_ref = _commit_tick(separate_a, chunks[4:5], (_cursor("A", 4),))[0]
        torch.testing.assert_close(ba[0:1], b_ref, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(ba[1:2], a_ref, rtol=2e-5, atol=2e-6)

        # Reset A alone while B advances normally in the same reordered batch.
        shared.reset("A", 0, "epoch-0")
        reset_a_b = _commit_tick(
            shared,
            torch.cat((chunks[5:6], chunks[6:7])),
            (_cursor("B", 5), _cursor("A", 20)),
        )[0]
        reset_a = reset_a_b[1:2]
        b_after_a_reset = reset_a_b[0:1]
        b_ref_after_reset = _commit_tick(separate_b, chunks[5:6], (_cursor("B", 5),))[0]
        torch.testing.assert_close(b_after_a_reset, b_ref_after_reset, rtol=2e-5, atol=2e-6)
        fresh = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2)
        reset_ref = _commit_tick(fresh, chunks[6:7], (_cursor("A", 20),))[0]
        gap = _commit_tick(shared, chunks[7:8], (_cursor("A", 22),))[0]
        gap_fresh = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2)
        gap_ref = _commit_tick(gap_fresh, chunks[7:8], (_cursor("A", 22),))[0]
        epoch_change = _commit_tick(
            shared,
            chunks[8:9],
            (_cursor("A", 0, generation=1, epoch="epoch-1"),),
        )[0]
        epoch_fresh = StreamingConformerAudioTower(
            audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=2
        )
        epoch_ref = _commit_tick(
            epoch_fresh,
            chunks[8:9],
            (_cursor("A", 0, generation=1, epoch="epoch-1"),),
        )[0]

    torch.testing.assert_close(reset_a, reset_ref, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(gap, gap_ref, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(epoch_change, epoch_ref, rtol=2e-5, atol=2e-6)


def test_aborted_prepare_is_retryable_and_invalid_row_prevents_all_row_commit():
    _, audio_tower, embed_audio = _tiny_audio_tower(53)
    streaming = StreamingConformerAudioTower(audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=4)
    generator = torch.Generator().manual_seed(63)
    chunks = torch.randn((2, 8, 128), generator=generator)

    with torch.inference_mode():
        first = streaming.prepare_batch(chunks, (_cursor("A", 0), _cursor("B", 0)))
        discarded = first.raw_tokens.detach().clone()
        first.abort()
        retry = streaming.prepare_batch(chunks, (_cursor("A", 0), _cursor("B", 0)))
        torch.testing.assert_close(retry.raw_tokens, discarded, rtol=0, atol=0)
        retry.commit()

        pending = streaming.prepare_batch(chunks, (_cursor("A", 1), _cursor("B", 1)))
        before = pending.raw_tokens.detach().clone()
        b_key = ("B", 0, "epoch-0")
        streaming._expected_positions[b_key] = (99, 99)
        with pytest.raises(RuntimeError, match="cursor changed before streaming commit"):
            pending.commit()
        streaming._expected_positions[b_key] = (1, 1)
        rerun = streaming.prepare_batch(chunks, (_cursor("A", 1), _cursor("B", 1)))
        torch.testing.assert_close(rerun.raw_tokens, before, rtol=0, atol=0)
        rerun.abort()


def test_selected_row_staging_and_commit_touch_only_selected_slots_at_capacity_128():
    _, audio_tower, embed_audio = _tiny_audio_tower(59)
    streaming = StreamingConformerAudioTower(
        audio_tower,
        embed_audio,
        device="cpu",
        dtype=torch.float32,
        max_sessions=128,
    )
    for index, bank in enumerate(streaming._canonical_banks()):
        bank.fill_(False if bank.dtype is torch.bool else 100.25 + index)
    for bank in streaming._staging_banks():
        bank.fill_(False if bank.dtype is torch.bool else -100.25)

    canonical_before = tuple(bank.clone() for bank in streaming._canonical_banks())
    staging_before = tuple(bank.clone() for bank in streaming._staging_banks())
    chunks = torch.randn((2, 8, 128), generator=torch.Generator().manual_seed(60))

    def assert_selected_staging_advanced(batch):
        selected = batch._slot_tensor
        unselected_mask = torch.ones(128, dtype=torch.bool)
        unselected_mask[selected] = False
        for before, after in zip(
            staging_before, streaming._staging_banks(), strict=True
        ):
            assert not torch.equal(
                before.index_select(0, selected), after.index_select(0, selected)
            )
            assert torch.equal(before[unselected_mask], after[unselected_mask])

    with torch.inference_mode():
        prepared = streaming.prepare_batch(
            chunks, (_cursor("selected-A", 0), _cursor("selected-B", 0))
        )
        assert prepared.slot_ids == (0, 1)
        assert_selected_staging_advanced(prepared)
        for before, after in zip(
            canonical_before, streaming._canonical_banks(), strict=True
        ):
            assert torch.equal(before, after)

        prepared.abort()
        for before, after in zip(
            canonical_before, streaming._canonical_banks(), strict=True
        ):
            assert torch.equal(before, after)

        retry = streaming.prepare_batch(
            chunks, (_cursor("selected-A", 0), _cursor("selected-B", 0))
        )
        assert_selected_staging_advanced(retry)
        retry.commit()

    selected = torch.tensor(retry.slot_ids, dtype=torch.long)
    unselected_mask = torch.ones(128, dtype=torch.bool)
    unselected_mask[selected] = False
    for before, canonical, staged in zip(
        canonical_before,
        streaming._canonical_banks(),
        streaming._staging_banks(),
        strict=True,
    ):
        assert torch.equal(
            canonical.index_select(0, selected), staged.index_select(0, selected)
        )
        assert torch.equal(before[unselected_mask], canonical[unselected_mask])


def test_reset_session_releases_only_requested_slot_and_reuses_it():
    _, audio_tower, embed_audio = _tiny_audio_tower(67)
    streaming = StreamingConformerAudioTower(
        audio_tower, embed_audio, device="cpu", dtype=torch.float32, max_sessions=3
    )
    chunks = torch.randn((2, 8, 128), generator=torch.Generator().manual_seed(68))
    _commit_tick(streaming, chunks, (_cursor("slot-A", 0), _cursor("slot-B", 0)))
    slots = dict(streaming._slots_by_key)
    slot_a = slots[("slot-A", 0, "epoch-0")]
    slot_b = slots[("slot-B", 0, "epoch-0")]
    assert slot_a != slot_b

    pending = streaming.prepare_batch(chunks[:1], (_cursor("slot-A", 1),))
    before = streaming.committed_positions
    with pytest.raises(RuntimeError, match="while a batch is prepared"):
        streaming.reset_session(slot_b)
    assert streaming.committed_positions == before
    pending.abort()

    with pytest.raises(IndexError):
        streaming.reset_session(3)
    with pytest.raises(TypeError):
        streaming.reset_session(True)
    streaming.reset_session(slot_a)
    assert streaming.committed_positions == (("slot-B", 0, "epoch-0", 1, 1),)

    reused = streaming.prepare_batch(chunks[:1], (_cursor("slot-C", 0),))
    assert reused.slot_ids == (slot_a,)
    reused.commit()
    assert streaming._slots_by_key[("slot-B", 0, "epoch-0")] == slot_b
    assert streaming._slots_by_key[("slot-C", 0, "epoch-0")] == slot_a


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity requires CUDA")
@pytest.mark.parametrize("batch_size", (1, 2, 3, 4))
def test_cuda_graph_matches_eager_and_full_offline_with_reset(batch_size: int):
    _, audio_tower, embed_audio = _tiny_audio_tower(71)
    audio_tower = audio_tower.cuda().to(dtype=torch.float16)
    embed_audio = embed_audio.cuda().to(dtype=torch.float16)
    streaming = StreamingConformerAudioTower(
        audio_tower,
        embed_audio,
        device="cuda",
        dtype=torch.float16,
        max_sessions=4,
        capture_cuda_graphs=True,
    )
    assert streaming.device == next(audio_tower.parameters()).device
    chunks = torch.randn((3, batch_size, 8, 128), device="cuda", dtype=torch.float16)
    eager = StreamingConformerAudioTower(
        audio_tower,
        embed_audio,
        device="cuda",
        dtype=torch.float16,
        max_sessions=4,
    )
    for row in range(batch_size):
        streaming.reset(f"g{row}")
    assert streaming.committed_positions == ()
    streaming.capture_graph(batch_size)
    assert streaming.committed_positions == ()
    with torch.inference_mode():
        mel_by_row = [[] for _ in range(batch_size)]
        for frame in range(2):
            order = (
                tuple(reversed(range(batch_size)))
                if batch_size > 1 and frame % 2
                else tuple(range(batch_size))
            )
            positions = tuple(_cursor(f"g{row}", frame) for row in order)
            tick_input = chunks[frame, list(order)]
            actual = streaming.prepare_batch(tick_input, positions, use_cuda_graph=True)
            baseline = eager.prepare_batch(tick_input, positions)
            torch.testing.assert_close(actual.raw_tokens, baseline.raw_tokens, rtol=3e-3, atol=3e-3)
            torch.testing.assert_close(actual.soft_tokens, baseline.soft_tokens, rtol=3e-3, atol=3e-3)
            for output_row, row in enumerate(order):
                mel_by_row[row].append(chunks[frame, row:row + 1])
                offline, _ = _offline_reference(
                    audio_tower,
                    torch.cat(mel_by_row[row], dim=1),
                    torch.ones((1, (frame + 1) * 8), device="cuda", dtype=torch.bool),
                )
                torch.testing.assert_close(
                    actual.raw_tokens[output_row:output_row + 1],
                    offline[:, -2:],
                    rtol=5e-3,
                    atol=5e-3,
                )
            actual.commit()
            baseline.commit()

        # Reset only the first identity; row two continues uninterrupted.
        streaming.reset("g0")
        eager.reset("g0")
        after_reset_order = tuple(reversed(range(batch_size)))
        after_reset_rows = tuple(
            _cursor(
                f"g{row}",
                0 if row == 0 else 2,
                generation=1 if row == 0 else 0,
                epoch="reset-epoch" if row == 0 else "epoch-0",
            )
            for row in after_reset_order
        )
        reset_chunk = torch.stack(
            [chunks[0, row] if row == 0 else chunks[2, row] for row in after_reset_order]
        )
        after_reset = streaming.prepare_batch(reset_chunk, after_reset_rows, use_cuda_graph=True)
        eager_after_reset = eager.prepare_batch(reset_chunk, after_reset_rows)
        torch.testing.assert_close(after_reset.raw_tokens, eager_after_reset.raw_tokens, rtol=3e-3, atol=3e-3)
        torch.testing.assert_close(after_reset.soft_tokens, eager_after_reset.soft_tokens, rtol=3e-3, atol=3e-3)
        first_offline, _ = _offline_reference(
            audio_tower, chunks[0, 0:1], torch.ones((1, 8), device="cuda", dtype=torch.bool)
        )
        reset_row = after_reset_order.index(0)
        torch.testing.assert_close(
            after_reset.raw_tokens[reset_row:reset_row + 1],
            first_offline[:, -2:],
            rtol=5e-3,
            atol=5e-3,
        )
        for output_row, row in enumerate(after_reset_order):
            if row == 0:
                continue
            continued = torch.cat((*mel_by_row[row], chunks[2, row:row + 1]), dim=1)
            continued_offline, _ = _offline_reference(
                audio_tower,
                continued,
                torch.ones((1, 24), device="cuda", dtype=torch.bool),
            )
            torch.testing.assert_close(
                after_reset.raw_tokens[output_row:output_row + 1],
                continued_offline[:, -2:],
                rtol=5e-3,
                atol=5e-3,
            )
        after_reset.commit()
        eager_after_reset.commit()
    assert torch.isfinite(after_reset.soft_tokens).all()
    assert streaming.committed_positions == tuple(
        sorted(
            (
                "g0", 1, "reset-epoch", 1, 1
            ) if row == 0 else (f"g{row}", 0, "epoch-0", 3, 3)
            for row in range(batch_size)
        )
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 graph parity requires CUDA")
def test_cuda_bf16_graph_restores_global_flag_and_matches_offline_after_warmup():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    _, audio_tower, embed_audio = _tiny_audio_tower(79)
    audio_tower = audio_tower.cuda().to(dtype=torch.bfloat16)
    embed_audio = embed_audio.cuda().to(dtype=torch.bfloat16)
    clipped = next(module for module in audio_tower.modules() if isinstance(module, Gemma4ClippableLinear))
    original_forward = clipped.forward
    requires_grad_before = tuple(
        parameter.requires_grad
        for root in (audio_tower, embed_audio)
        for parameter in root.parameters()
    )
    matmul_backend = torch.backends.cuda.matmul
    original_flag = matmul_backend.allow_bf16_reduced_precision_reduction
    try:
        matmul_backend.allow_bf16_reduced_precision_reduction = False
        # Construction includes the one-time relative-R projection and must
        # share the same precision policy as warmup and graph capture.
        streaming = StreamingConformerAudioTower(
            audio_tower,
            embed_audio,
            device="cuda",
            dtype=torch.bfloat16,
            max_sessions=4,
            capture_cuda_graphs=True,
            optimize_cuda=True,
        )
        for batch_size in range(1, 5):
            streaming.capture_graph(batch_size)
    finally:
        matmul_backend.allow_bf16_reduced_precision_reduction = original_flag
    assert matmul_backend.allow_bf16_reduced_precision_reduction is original_flag

    chunks = torch.randn((10, 1, 8, 128), device="cuda", dtype=torch.bfloat16)
    cached_raw = []
    cached_soft = []
    try:
        # Captured kernels must replay under the restored/default True flag.
        matmul_backend.allow_bf16_reduced_precision_reduction = True
        with torch.inference_mode():
            streaming.reset("bf16-session")
            for frame in range(chunks.shape[0]):
                prepared = streaming.prepare_batch(
                    chunks[frame], (_cursor("bf16-session", frame),), use_cuda_graph=True
                )
                cached_raw.append(prepared.raw_tokens.detach().clone())
                cached_soft.append(prepared.soft_tokens.detach().clone())
                prepared.commit()
            mel = chunks.reshape(1, -1, 128)
            offline_raw, _ = _offline_reference(
                audio_tower,
                mel,
                torch.ones((1, mel.shape[1]), device="cuda", dtype=torch.bool),
            )
            offline_soft = embed_audio(offline_raw)
    finally:
        matmul_backend.allow_bf16_reduced_precision_reduction = original_flag
        streaming.close()

    raw_cos = torch.nn.functional.cosine_similarity(
        torch.cat(cached_raw, dim=1).float()[:, 13:], offline_raw.float()[:, 13:], dim=-1
    )
    soft_cos = torch.nn.functional.cosine_similarity(
        torch.cat(cached_soft, dim=1).float()[:, 13:], offline_soft.float()[:, 13:], dim=-1
    )
    assert float(raw_cos.mean()) >= 0.9995 and float(raw_cos.min()) >= 0.999
    assert float(soft_cos.mean()) >= 0.9995 and float(soft_cos.min()) >= 0.999
    assert clipped.forward.__func__ is original_forward.__func__
    assert "forward" not in clipped.__dict__
    assert tuple(
        parameter.requires_grad
        for root in (audio_tower, embed_audio)
        for parameter in root.parameters()
    ) == requires_grad_before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="optimized startup cleanup requires CUDA")
@pytest.mark.parametrize("failure_stage", ["compile", "capture"])
def test_optimized_startup_failure_restores_forward_and_global_flag(monkeypatch, failure_stage):
    from types import SimpleNamespace

    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    _, audio_tower, embed_audio = _tiny_audio_tower(80)
    audio_tower = audio_tower.cuda().to(dtype=torch.bfloat16)
    embed_audio = embed_audio.cuda().to(dtype=torch.bfloat16)
    linear = next(module for module in audio_tower.modules() if isinstance(module, Gemma4ClippableLinear))
    original_forward = linear.forward
    matmul_backend = torch.backends.cuda.matmul
    original_flag = matmul_backend.allow_bf16_reduced_precision_reduction

    if failure_stage == "compile":
        def fail_compile(*_args, **_kwargs):
            raise RuntimeError("compile cleanup sentinel")

        monkeypatch.setattr(torch, "compile", fail_compile)
    else:
        def fail_capture(*_args, **_kwargs):
            raise RuntimeError("capture cleanup sentinel")

        monkeypatch.setattr(StreamingConformerAudioTower, "capture_graph", fail_capture)

    with pytest.raises(RuntimeError, match=f"{failure_stage} cleanup sentinel"):
        StreamingConformerAudioTower.from_gemma_model_optimized_cuda(
            SimpleNamespace(audio_tower=audio_tower, embed_audio=embed_audio),
            device="cuda",
            dtype=torch.bfloat16,
            max_sessions=4,
            max_graph_batch_size=1,
        )
    assert matmul_backend.allow_bf16_reduced_precision_reduction is original_flag
    assert linear.forward.__func__ is original_forward.__func__
    assert "forward" not in linear.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton Q2 linear prototype requires CUDA")
def test_q2_triton_clipped_linear_matches_genuine_gemma_linear_and_bounds():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from duplex_kernel.streaming_conformer_kernels import triton_clipped_linear

    audio_config, _, _ = _tiny_audio_tower(83)
    audio_config.use_clipped_linears = True
    linear = Gemma4ClippableLinear(audio_config, 128, 64).cuda().to(torch.bfloat16).eval()
    linear.input_min.fill_(-0.25)
    linear.input_max.fill_(0.25)
    linear.output_min.fill_(-0.125)
    linear.output_max.fill_(0.125)
    for parameter in linear.parameters():
        parameter.requires_grad_(False)
    inputs = torch.randn((1, 2, 128), device="cuda", dtype=torch.bfloat16) * 4
    with torch.inference_mode():
        expected = linear(inputs)
        actual = triton_clipped_linear(linear, inputs, block_n=64, block_k=64)
    torch.cuda.synchronize()
    cosine = torch.nn.functional.cosine_similarity(
        expected.float(), actual.float(), dim=-1
    )
    assert float(cosine.min()) >= 0.9995
    assert float(actual.float().min()) >= -0.125
    assert float(actual.float().max()) <= 0.125


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton Q2 linear fallback requires CUDA")
def test_q2_triton_instance_wrapper_fallback_is_original_and_restores_forward():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from duplex_kernel.streaming_conformer_kernels import (
        install_gemma4_linear_wrappers,
        restore_gemma4_linear_wrappers,
    )

    audio_config, _, _ = _tiny_audio_tower(84)
    audio_config.use_clipped_linears = True
    linear = Gemma4ClippableLinear(audio_config, 128, 64).cuda().to(torch.bfloat16).eval()
    linear.input_min.fill_(-0.25)
    linear.input_max.fill_(0.25)
    linear.output_min.fill_(-0.125)
    linear.output_max.fill_(0.125)
    requires_grad_before = tuple(parameter.requires_grad for parameter in linear.parameters())
    original_forward = linear.forward
    wrappers = install_gemma4_linear_wrappers((linear,))
    try:
        assert not hasattr(linear, "_q2_triton_calls")
        # M=9 takes the exact original forward, including its native behavior.
        inputs = torch.randn((1, 9, 128), device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = original_forward(inputs)
            actual = linear(inputs)
        assert torch.equal(actual, expected)
        differentiable_input = torch.randn((1, 2, 128), device="cuda", dtype=torch.bfloat16)
        expected_grad_guard = original_forward(differentiable_input)
        actual_grad_guard = linear(differentiable_input)
        assert torch.equal(actual_grad_guard, expected_grad_guard)
        assert tuple(parameter.requires_grad for parameter in linear.parameters()) == requires_grad_before
    finally:
        restore_gemma4_linear_wrappers(wrappers)
    assert linear.forward.__func__ is original_forward.__func__
    assert "forward" not in linear.__dict__


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Inductor BF16 cast test requires CUDA")
def test_inductor_emulate_precision_casts_preserves_eager_bf16_pointwise_rounds():
    """Keep eager BF16 cast barriers when Inductor fuses pointwise ops."""
    value = torch.ones((1024,), device="cuda", dtype=torch.bfloat16)
    scale = 1.00390625

    def two_pointwise_rounds(tensor):
        intermediate = tensor * scale
        return intermediate * scale

    with torch.inference_mode():
        eager = two_pointwise_rounds(value)
        default_compiled = torch.compile(
            two_pointwise_rounds, fullgraph=True, dynamic=False, backend="inductor"
        )(value)
        cast_emulated = torch.compile(
            two_pointwise_rounds,
            fullgraph=True,
            dynamic=False,
            backend="inductor",
            options={"emulate_precision_casts": True},
        )(value)
    torch.cuda.synchronize()
    assert not torch.equal(default_compiled, eager)
    assert torch.equal(cast_emulated, eager)


def _q2_clipped_linear_config():
    return Gemma4AudioConfig(use_clipped_linears=True)


def test_q2_wrapper_install_remove_preserves_requires_grad_flags():
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear
    from duplex_kernel.streaming_conformer_kernels import (
        install_gemma4_linear_wrappers,
        restore_gemma4_linear_wrappers,
    )

    linear = Gemma4ClippableLinear(_q2_clipped_linear_config(), 32, 24).eval()
    before = tuple(parameter.requires_grad for parameter in linear.parameters())
    wrappers = install_gemma4_linear_wrappers((linear,))
    assert tuple(parameter.requires_grad for parameter in linear.parameters()) == before
    restore_gemma4_linear_wrappers(wrappers)
    assert tuple(parameter.requires_grad for parameter in linear.parameters()) == before
