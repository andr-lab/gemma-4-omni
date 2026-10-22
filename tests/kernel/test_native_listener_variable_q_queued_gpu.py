"""Opt-in RTX 3090 parity probe for variable-Q queued audio append.

Run with ``GEMMA4_RUN_VARIABLE_Q_QUEUED_GPU=1`` and
``DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH`` pointing at a build7 candidate DSO.
"""
from __future__ import annotations

import os
import json
import sys
import time
import uuid
from dataclasses import replace

import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_VARIABLE_Q_QUEUED_GPU") != "1",
    reason="requires the explicit RTX 3090 variable-Q runtime",
)
def test_variable_q_queued_append_matches_sequential_q2_on_private_forks() -> None:
    import torch

    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor,
        ListenerBatchManifest,
        RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import (
        NativeGemmaListenerLM,
        _find_library as find_native_core_library,
        _load_compiled_listener_topology,
    )
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from duplex_kernel.unified_token_adapter import UnifiedTokenAdapter
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    workspace = "/workspace"
    base_position = int(os.environ.get("GEMMA4_VARIABLE_Q_TEST_BASE_POSITION", "2050"))
    assert base_position > AUDIO_START_AFTER_TEXT and (base_position - AUDIO_START_AFTER_TEXT) % 2 == 0
    candidate = os.environ.get("DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH")
    assert candidate and os.path.isfile(candidate)
    result_json_path = os.environ.get("GEMMA4_VARIABLE_Q_TEST_RESULT_JSON")
    assert result_json_path, "set GEMMA4_VARIABLE_Q_TEST_RESULT_JSON to persist parity results"
    layer_trace_directory = os.environ.get("GEMMA4_BATCH_LAYER_TRACE_DIR")
    input_fixture_path = os.environ.get("GEMMA4_VARIABLE_Q_TEST_INPUT_BF16")
    if layer_trace_directory:
        os.makedirs(layer_trace_directory, exist_ok=True)
    core_library = os.environ.get("GEMMA4_NATIVE_LISTENER_LIBRARY_PATH", candidate)
    assert os.path.isfile(core_library)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = core_library
    provider_path = find_native_core_library()
    print(
        f"GPU_PREFLIGHT provider_core={provider_path} wrapper={candidate} "
        "input=BF16[1,Q,1536] q2_policy=2 attention=3 rmsnorm=2 max_seq_len=16384",
        flush=True,
    )
    assert os.path.samefile(provider_path, candidate)
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"
    os.environ["GEMMA4_NATIVE_LISTENER_CUDA_GRAPH"] = "0"

    device = torch.device("cuda:0")
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(device)
    counts_env = os.environ.get("GEMMA4_VARIABLE_Q_TEST_COUNTS")
    interleaving_only = os.environ.get("GEMMA4_VARIABLE_Q_TEST_INTERLEAVING_ONLY") == "1"
    frame_counts = () if interleaving_only else (
        tuple(int(value) for value in counts_env.split(",") if value.strip())
        if counts_env else (1, 2, 7, 32)
    )
    if layer_trace_directory:
        assert frame_counts == (2,), "layer trace is scoped to the four-query comparison"
        for trace_mode in ("sequential", "batch"):
            os.makedirs(os.path.join(layer_trace_directory, trace_mode), exist_ok=True)
    results = {
        "provider_core": provider_path,
        "wrapper_library": candidate,
        "device": torch.cuda.get_device_name(device),
        "base_position": base_position,
        "frame_counts": list(frame_counts),
        "interleaving_only": interleaving_only,
        "query_counts": [2 * count for count in frame_counts],
        "attention_backend_id": 3,
        "rmsnorm_backend_id": 2,
        "max_seq_len": 16_384,
        "input_seed": 20_261_021,
        "input_distribution": "torch.randn * 0.02, converted to BF16",
        "input_rng_fixture": ({
            "path": input_fixture_path,
            "shape": [4, 1536],
            "dtype": "bfloat16",
            "rng_device": "cuda:0",
            "draw_order": [
                "torch.manual_seed(20261021)",
                "torch.randn((2,1536), device=cuda:0, dtype=float32) for counts=(1,2) N=1",
                "N=1 q0 future probe clones and add_(0.25), no RNG draw",
                "torch.randn((4,1536), device=cuda:0, dtype=float32) for counts=(1,2) N=2",
                "multiply by 0.02 and convert to bfloat16",
            ],
        } if input_fixture_path else None),
        "cases": [],
        "layer_trace": None,
        "q0_future_probe_passed": None,
        "canonical_unchanged": False,
    }
    thinker = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    lm = backend.gemma_model.language_model
    tokenizer = thinker.tokenizer
    with torch.inference_mode():
        multimodal_ids, multimodal_attention = backend._build_multimodal_inputs(
            (), 1024, device=device,
        )
    prefix_ids_tensor = multimodal_ids[:, :AUDIO_START_AFTER_TEXT].contiguous()
    prefix_attention = multimodal_attention[:, :AUDIO_START_AFTER_TEXT].to(
        device=device, dtype=torch.long,
    ).contiguous()
    assert AUDIO_START_AFTER_TEXT == 34

    stream = torch.cuda.Stream(device=device, priority=-1)
    lock = backend._lock
    if not lock.acquire_fair("listener"):
        raise RuntimeError("variable-Q smoke could not acquire the Thinker prefix lock")
    try:
        with torch.cuda.stream(stream), torch.inference_mode():
            prefix_embeds = lm.embed_tokens(prefix_ids_tensor).contiguous()
            prefix_ple = lm.get_per_layer_inputs(prefix_ids_tensor, prefix_embeds).contiguous()
            prefix_positions = torch.arange(34, dtype=torch.long, device=device).view(1, -1)
            prefix_masks = create_masks_for_generate(
                lm.config, prefix_embeds, prefix_attention,
                past_key_values=None, position_ids=prefix_positions,
            )
            prefix_cache = lm(
                input_ids=None,
                inputs_embeds=prefix_embeds,
                per_layer_inputs=prefix_ple,
                attention_mask=prefix_masks,
                position_ids=prefix_positions,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            ).past_key_values

            # Default 2050 exercises ordered full attention plus ATen sliding
            # layers. The base-position override lets diagnosis isolate the
            # same append below the ordered-attention threshold.
            pad_id = int(tokenizer.pad_token_id or backend.bos_token_id)
            extend_len = base_position - 34
            extension_ids = torch.full(
                (1, extend_len), pad_id, dtype=torch.long, device=device,
            )
            extension_embeds = lm.embed_tokens(extension_ids).contiguous()
            extension_ple = lm.get_per_layer_inputs(
                extension_ids, extension_embeds,
            ).contiguous()
            positions = torch.arange(34, base_position, dtype=torch.long, device=device).view(1, -1)
            all_attention = torch.ones((1, base_position), dtype=torch.long, device=device)
            masks = create_masks_for_generate(
                lm.config, extension_embeds, all_attention,
                past_key_values=prefix_cache, position_ids=positions,
            )
            cache = lm(
                input_ids=None,
                inputs_embeds=extension_embeds,
                per_layer_inputs=extension_ple,
                attention_mask=masks,
                position_ids=positions,
                past_key_values=prefix_cache,
                use_cache=True,
                return_dict=True,
            ).past_key_values
            shared_layers = getattr(cache, "shared_layers", None)
            for layer_idx, layer in enumerate(cache.layers[:35]):
                attention = lm.layers[layer_idx].self_attn
                if bool(getattr(attention, "is_kv_shared_layer", False)):
                    continue
                source = (
                    shared_layers.get(layer_idx)
                    if shared_layers is not None and hasattr(shared_layers, "get")
                    else None
                )
                if source is None:
                    keys, values = layer.keys, layer.values
                else:
                    keys, values = source
                missing = base_position - int(keys.shape[2])
                if missing <= 0:
                    continue
                pad_shape = (keys.shape[0], keys.shape[1], missing, keys.shape[3])
                padded_keys = torch.cat((
                    torch.zeros(pad_shape, dtype=keys.dtype, device=device), keys,
                ), dim=2).contiguous()
                padded_values = torch.cat((
                    torch.zeros(pad_shape, dtype=values.dtype, device=device), values,
                ), dim=2).contiguous()
                if source is None:
                    layer.keys, layer.values = padded_keys, padded_values
                else:
                    shared_layers[layer_idx] = (padded_keys, padded_values)
            assert int(cache.get_seq_length(0)) == base_position
        stream.synchronize()
    finally:
        lock.release()

    signature = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256=_load_compiled_listener_topology()["manifest_sha256"],
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=5,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )
    provider = NativeGemmaListenerLM(
        lm,
        max_seq_len=16_384,
        stream_q_len=2,
        enable_cuda_graph=False,
        attention_backend=3,
        rmsnorm_backend=2,
    )
    owner = NativeGemmaListenerMicrobatch(
        provider,
        library_path=candidate,
        runtime_signature=signature,
        listener_stream=stream,
        max_batch_size=2,
    )
    prefix_len = 34
    canonical_row = BatchRowDescriptor(
        session_id=f"variable-q-canonical-{uuid.uuid4().hex}",
        session_generation=1,
        state_epoch_id=uuid.uuid4().hex,
        frame_id=17,
        listener_state_version=23,
        sequence_position=31,
        cache_position=base_position,
        valid_history_tokens=base_position,
        native_context_id=1,
        input_frame_sha256="0" * 64,
        reference_pcm_sha256=None,
        frame_ready_at_ns=time.monotonic_ns(),
        reference_frame_id=None,
    )
    owner.import_session(
        canonical_row,
        cache,
        prefix_attention,
        prefix_len=prefix_len,
        producer_stream=stream,
    )
    adapter = UnifiedTokenAdapter(thinker, tokenizer=tokenizer, require_cuda=True)

    def manifest(row: BatchRowDescriptor) -> ListenerBatchManifest:
        return ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=signature,
            stream_identity=owner.capability.stream_identity,
            stream_priority=owner.capability.stream_priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="carryover_ready",
            rows=(row,),
        )

    def fork() -> BatchRowDescriptor:
        return owner.fork_session_slot(
            canonical_row,
            fork_session_id=f"variable-q-fork-{uuid.uuid4().hex}",
            fork_native_context_id=int(uuid.uuid4().int & ((1 << 63) - 1)) or 2,
        )

    def run(row: BatchRowDescriptor, frames: torch.Tensor, *, batched: bool):
        previous_trace_directory = os.environ.get("GEMMA4_BATCH_LAYER_TRACE_DIR")
        if layer_trace_directory:
            trace_mode = "batch" if batched else "sequential"
            os.environ["GEMMA4_BATCH_LAYER_TRACE_DIR"] = os.path.join(
                layer_trace_directory, trace_mode,
            )
        started = time.perf_counter()
        already_verified = False
        try:
            if batched:
                output = owner.step_queued_audio_batch_async(
                    manifest(row), frames, producer_stream=stream,
                )
            else:
                output = None
                for index in range(0, frames.shape[0], 2):
                    output = owner.step_queued_audio_async(
                        manifest(row), frames[index:index + 2].unsqueeze(0).contiguous(),
                        producer_stream=stream,
                    )
                    output.completion_event.synchronize()
                    health = owner.verify_output_health(output)
                    assert len(health) == 1 and health[0].checked and not health[0].nonfinite
                    already_verified = True
                    row = replace(
                        row,
                        cache_position=row.cache_position + 2,
                        valid_history_tokens=row.valid_history_tokens + 2,
                    )
            assert output is not None
            output.completion_event.synchronize()
            if not already_verified:
                health = owner.verify_output_health(output)
                assert len(health) == 1 and health[0].checked and not health[0].nonfinite
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            final_row = replace(
                row,
                cache_position=row.cache_position + (frames.shape[0] if batched else 0),
                valid_history_tokens=row.valid_history_tokens + (frames.shape[0] if batched else 0),
            )
            return output.tensor[0, :2].detach().clone(), final_row, elapsed_ms
        finally:
            if previous_trace_directory is None:
                os.environ.pop("GEMMA4_BATCH_LAYER_TRACE_DIR", None)
            else:
                os.environ["GEMMA4_BATCH_LAYER_TRACE_DIR"] = previous_trace_directory

    def compare_tensor(name: str, left: torch.Tensor, right: torch.Tensor) -> tuple[float, float]:
        delta = (left.float() - right.float()).abs()
        max_abs = float(delta.max().item())
        max_rel = float((delta / right.float().abs().clamp_min(1e-6)).max().item())
        print(f"{name} max_abs={max_abs:.8g} max_rel={max_rel:.8g}", flush=True)
        return max_abs, max_rel

    def compare_snapshots(left, right, label: str) -> tuple[
        float, float, dict[int, float], dict[int, dict[int, float]],
    ]:
        assert left.state.current_position == right.state.current_position
        assert left.state.valid_kv_len == right.state.valid_kv_len
        assert left.state.expected_frame_id == right.state.expected_frame_id
        assert left.state.expected_state_version == right.state.expected_state_version
        assert left.state.expected_sequence_position == right.state.expected_sequence_position
        max_abs = 0.0
        max_rel = 0.0
        per_pair_abs: dict[int, float] = {}
        per_layer_pair: dict[int, dict[int, float]] = {}
        for layer in left.kv_by_layer:
            layer_pairs: dict[int, float] = {}
            for kind, a, b in zip(
                ("K", "V"), left.kv_by_layer[layer], right.kv_by_layer[layer], strict=True,
            ):
                delta = (a.float() - b.float()).abs()
                max_abs = max(max_abs, float(delta.max().item()))
                max_rel = max(
                    max_rel,
                    float((delta / b.float().abs().clamp_min(1e-6)).max().item()),
                )
                prefix_delta = float(delta[..., :base_position, :].max().item())
                if prefix_delta > 0:
                    per_pair_abs[-1] = max(per_pair_abs.get(-1, 0.0), prefix_delta)
                for pair_index, start in enumerate(range(base_position, delta.shape[-2], 2)):
                    end = min(start + 2, delta.shape[-2])
                    pair_delta = float(delta[..., start:end, :].max().item())
                    layer_pairs[pair_index] = max(
                        layer_pairs.get(pair_index, 0.0), pair_delta,
                    )
                    per_pair_abs[pair_index] = max(
                        per_pair_abs.get(pair_index, 0.0), pair_delta,
                    )
            per_layer_pair[layer] = layer_pairs
        print(
            f"{label} max_abs={max_abs:.8g} max_rel={max_rel:.8g} "
            f"layers={len(left.kv_by_layer)} position={left.state.current_position} "
            f"pair_max_abs={per_pair_abs} layer_pair_max_abs={per_layer_pair}",
            flush=True,
        )
        return max_abs, max_rel, per_pair_abs, per_layer_pair

    def fork_snapshot(row: BatchRowDescriptor):
        state_row = replace(row, logical_q_len=2)
        return owner.snapshot_session(state_row)

    def compare_layer_trace() -> dict:
        assert layer_trace_directory is not None
        stages = (
            "hidden_in", "input_norm", "k_projection", "k_keynorm",
            "k_rope", "attention_output", "hidden_next",
        )

        def read_stage(layer: int, stage: str, base: int, q: int) -> torch.Tensor:
            path = os.path.join(
                layer_trace_directory, "sequential",
                f"l{layer:02d}_{stage}_base{base}_q{q}.bf16",
            )
            with open(path, "rb") as trace_file:
                raw = bytearray(trace_file.read())
            assert len(raw) % (q * 2) == 0, f"bad BF16 trace size: {path}"
            return torch.frombuffer(raw, dtype=torch.bfloat16).reshape(q, -1)

        stage_metrics = []
        first_divergence = {0: None, 1: None}
        for layer in range(7):
            for stage in stages:
                batch_path = os.path.join(
                    layer_trace_directory, "batch",
                    f"l{layer:02d}_{stage}_base{base_position}_q4.bf16",
                )
                with open(batch_path, "rb") as trace_file:
                    batch_raw = bytearray(trace_file.read())
                assert len(batch_raw) % 8 == 0, f"bad BF16 trace size: {batch_path}"
                batched = torch.frombuffer(batch_raw, dtype=torch.bfloat16).reshape(4, -1)
                assert batched.shape[0] == 4
                for pair in range(2):
                    sequential = read_stage(layer, stage, base_position + pair * 2, 2)
                    actual = batched[pair * 2:pair * 2 + 2]
                    assert sequential.shape == actual.shape
                    delta = (sequential.float() - actual.float()).abs()
                    bitwise_equal_count = int(torch.count_nonzero(
                        sequential.view(torch.int16) == actual.view(torch.int16),
                    ).item())
                    metric = {
                        "layer": layer,
                        "stage": stage,
                        "pair": pair,
                        "max_abs": float(delta.max().item()),
                        "bitwise_equal_count": bitwise_equal_count,
                        "element_count": sequential.numel(),
                    }
                    stage_metrics.append(metric)
                    if bitwise_equal_count != sequential.numel() and first_divergence[pair] is None:
                        first_divergence[pair] = metric
        print(f"LAYER_TRACE first_divergence={first_divergence}", flush=True)
        return {
            "base_position": base_position,
            "sequential_bases": [base_position, base_position + 2],
            "batch_base": base_position,
            "layers": list(range(7)),
            "stages": list(stages),
            "first_divergence_by_pair": first_divergence,
            "stage_metrics": stage_metrics,
        }

    cleanup_rows: list[BatchRowDescriptor] = []
    try:
        canonical_before = owner.snapshot_session(canonical_row)
        torch.manual_seed(20261021)
        for frame_count in frame_counts:
            if input_fixture_path and frame_count == 2:
                with open(input_fixture_path, "rb") as fixture_file:
                    fixture_bytes = bytearray(fixture_file.read())
                frames = torch.frombuffer(
                    fixture_bytes, dtype=torch.bfloat16,
                ).reshape(2 * frame_count, 1536).to(device).contiguous()
            else:
                frames = (torch.randn(
                    (2 * frame_count, 1536), device=device, dtype=torch.float32,
                ) * 0.02).to(torch.bfloat16).contiguous()
            sequential_fork = fork()
            cleanup_rows.append(sequential_fork)
            seq_hidden, seq_final_row, seq_ms = run(sequential_fork, frames, batched=False)
            seq_snapshot = fork_snapshot(seq_final_row)
            owner.release_session_async(seq_final_row)
            cleanup_rows.clear()

            batch_fork = fork()
            cleanup_rows.append(batch_fork)
            batch_hidden, batch_final_row, batch_ms = run(batch_fork, frames, batched=True)
            batch_snapshot = fork_snapshot(batch_final_row)
            owner.release_session_async(batch_final_row)
            cleanup_rows.clear()
            hidden_abs, hidden_rel = compare_tensor(
                f"N={frame_count} hidden_last_pair", seq_hidden, batch_hidden,
            )
            kv_abs, kv_rel, pair_max_abs, layer_pair_max_abs = compare_snapshots(
                seq_snapshot, batch_snapshot, f"N={frame_count} full_KV",
            )
            seq_logits = adapter.logits_from_final_norm_hidden(seq_hidden.unsqueeze(0)).float()[0, -1]
            batch_logits = adapter.logits_from_final_norm_hidden(batch_hidden.unsqueeze(0)).float()[0, -1]
            seq_top = torch.topk(seq_logits, k=5).indices.tolist()
            batch_top = torch.topk(batch_logits, k=5).indices.tolist()
            logits_abs = float((seq_logits - batch_logits).abs().max().item())
            print(
                f"N={frame_count} logits_max_abs={logits_abs:.8g} "
                f"top1={seq_top[0]}/{batch_top[0]} top5={seq_top}/{batch_top} "
                f"sequential_ms={seq_ms:.3f} batch_ms={batch_ms:.3f}",
                flush=True,
            )
            results["cases"].append({
                "frame_count": frame_count,
                "query_count": frame_count * 2,
                "cache_position": seq_snapshot.state.current_position,
                "hidden_last_pair_max_abs": hidden_abs,
                "hidden_last_pair_max_rel": hidden_rel,
                "hidden_last_pair_bitwise_equal": hidden_abs == 0.0,
                "full_kv_max_abs": kv_abs,
                "full_kv_max_rel": kv_rel,
                "full_kv_bitwise_equal": kv_abs == 0.0,
                "kv_pair_max_abs": pair_max_abs,
                "kv_layer_pair_max_abs": layer_pair_max_abs,
                "logits_max_abs": logits_abs,
                "sequential_top5": seq_top,
                "batch_top5": batch_top,
                "top1_matches": seq_top[0] == batch_top[0],
                "sequential_ms": seq_ms,
                "batch_ms": batch_ms,
            })
            if frame_count == 1:
                q0a, q0b = frames.clone(), frames.clone()
                q0b[1].add_(0.25)
                probe_a_row = fork()
                cleanup_rows.append(probe_a_row)
                probe_a, probe_a_row, _ = run(probe_a_row, q0a, batched=True)
                owner.release_session_async(probe_a_row)
                cleanup_rows.clear()
                probe_b_row = fork()
                cleanup_rows.append(probe_b_row)
                probe_b, probe_b_row, _ = run(probe_b_row, q0b, batched=True)
                assert torch.equal(probe_a[0], probe_b[0]), "Q0 output depends on future Q1 input"
                results["q0_future_probe_passed"] = True
                owner.release_session_async(probe_b_row)
            cleanup_rows = []

        if layer_trace_directory:
            results["layer_trace"] = compare_layer_trace()

        # Exercise the exact output-health sequence that used to confuse an
        # embedded logical-Q append with the preceding physical Q2 advance.
        smoke_row = fork()
        cleanup_rows.append(smoke_row)
        quiet_q0 = torch.zeros((1, 2, 1536), dtype=torch.bfloat16, device=device)
        q0_output = owner.step_async(
            manifest(replace(
                smoke_row,
                logical_q_len=0,
                ephemeral_reason="prespeech_silence",
            )), quiet_q0,
            producer_stream=stream,
        )
        q0_output.completion_event.synchronize()
        q0_health = owner.verify_output_health(q0_output)
        assert len(q0_health) == 1 and q0_health[0].checked and not q0_health[0].nonfinite
        interleaving_health_statuses = [q0_health[0].status]
        smoke_row = replace(
            smoke_row,
            frame_id=smoke_row.frame_id + 1,
            listener_state_version=smoke_row.listener_state_version + 1,
            sequence_position=smoke_row.sequence_position + 1,
            logical_q_len=0,
            ephemeral_reason="prespeech_silence",
        )
        cleanup_rows[-1] = smoke_row
        q0_state = owner.session_state(smoke_row)
        assert q0_state.current_position == q0_state.valid_kv_len == base_position

        smoke_token_ids = tuple(int(value) for value in tokenizer.encode(
            "Certo va", add_special_tokens=False,
        ))
        assert smoke_token_ids
        token_a = smoke_token_ids[0]
        token_b = smoke_token_ids[1] if len(smoke_token_ids) > 1 else int(tokenizer.pad_token_id)

        def append_text(row: BatchRowDescriptor, ids: tuple[int, ...], logical_q_len: int):
            prepared = adapter.prepare(ids, logical_q_lens=[logical_q_len])
            output = owner.step_async(
                manifest(replace(
                    row,
                    logical_q_len=logical_q_len,
                    ephemeral_reason=None,
                )),
                prepared.inputs_embeds,
                per_layer_inputs=prepared.per_layer_inputs,
                producer_stream=stream,
            )
            output.completion_event.synchronize()
            health = owner.verify_output_health(output)
            assert len(health) == 1 and health[0].checked and not health[0].nonfinite
            interleaving_health_statuses.append(health[0].status)
            advanced = replace(
                row,
                cache_position=row.cache_position + logical_q_len,
                valid_history_tokens=row.valid_history_tokens + logical_q_len,
                logical_q_len=logical_q_len,
                ephemeral_reason=None,
            )
            state = owner.session_state(advanced)
            assert state.current_position == state.valid_kv_len == advanced.cache_position
            return advanced

        smoke_row = append_text(smoke_row, (token_a,), 1)
        cleanup_rows[-1] = smoke_row
        smoke_row = append_text(smoke_row, (token_a, token_b), 2)
        cleanup_rows[-1] = smoke_row
        queued_soft = (torch.randn(
            (2, 1536), dtype=torch.float32, device=device,
        ) * 0.02).to(torch.bfloat16).contiguous()
        queued_output = owner.step_queued_audio_async(
            manifest(replace(smoke_row, logical_q_len=2)),
            queued_soft.unsqueeze(0).contiguous(),
            producer_stream=stream,
        )
        queued_output.completion_event.synchronize()
        queued_health = owner.verify_output_health(queued_output)
        assert len(queued_health) == 1 and queued_health[0].checked and not queued_health[0].nonfinite
        interleaving_health_statuses.append(queued_health[0].status)
        smoke_row = replace(
            smoke_row,
            cache_position=smoke_row.cache_position + 2,
            valid_history_tokens=smoke_row.valid_history_tokens + 2,
            logical_q_len=2,
        )
        cleanup_rows[-1] = smoke_row
        smoke_row = append_text(smoke_row, (token_b,), 1)
        cleanup_rows[-1] = smoke_row
        smoke_state = owner.session_state(smoke_row)
        assert smoke_state.current_position == smoke_state.valid_kv_len == base_position + 6
        assert smoke_state.expected_frame_id == canonical_before.state.expected_frame_id + 1
        assert smoke_state.expected_state_version == canonical_before.state.expected_state_version + 1
        assert smoke_state.expected_sequence_position == canonical_before.state.expected_sequence_position + 1
        results["text_health_interleaving"] = {
            "sequence": ["Q0", "text1", "text2", "queuedQ2", "text1"],
            "health_statuses": interleaving_health_statuses,
            "final_cursor": smoke_state.current_position,
            "final_expected_frame_id": smoke_state.expected_frame_id,
            "canonical_context_id": canonical_row.native_context_id,
            "fork_context_id": smoke_row.native_context_id,
        }
        owner.release_session_async(smoke_row)
        cleanup_rows.pop()

        canonical_after = owner.snapshot_session(canonical_row)
        assert canonical_before.state == canonical_after.state
        assert set(canonical_before.kv_by_layer) == set(canonical_after.kv_by_layer)
        for layer in canonical_before.kv_by_layer:
            before_key, before_value = canonical_before.kv_by_layer[layer]
            after_key, after_value = canonical_after.kv_by_layer[layer]
            assert torch.equal(before_key, after_key), f"canonical K changed at layer {layer}"
            assert torch.equal(before_value, after_value), f"canonical V changed at layer {layer}"
        results["canonical_unchanged"] = True
    finally:
        exception_type, exception_value, _ = sys.exc_info()
        if exception_type is not None:
            results["error"] = f"{exception_type.__name__}: {exception_value}"
        result_directory = os.path.dirname(result_json_path)
        if result_directory:
            os.makedirs(result_directory, exist_ok=True)
        with open(result_json_path, "w", encoding="utf-8") as result_file:
            json.dump(results, result_file, indent=2, sort_keys=True)
            result_file.write("\n")
        if not owner._poisoned:
            for row in reversed(cleanup_rows):
                try:
                    owner.release_session_async(row)
                except Exception:
                    pass
        owner.close()
