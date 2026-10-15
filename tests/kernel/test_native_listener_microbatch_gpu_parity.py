"""Opt-in real RTX 3090 numerical comparison for native listener batching.

Set ``GEMMA4_RUN_LISTENER_MICROBATCH_GPU=1`` inside the configured GPU image.
Set ``DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS=1`` to additionally exercise the
explicitly prewarmed geometric-bucket, one-graph-per-operation GEMM cache,
including capacity growth.
The test uses the cached model and the ordinary ABI3 library as its reference;
it writes no reports or checkpoint attestations.
"""
from __future__ import annotations

import os
import time
import uuid

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_LISTENER_MICROBATCH_GPU") != "1",
    reason="requires explicit RTX 3090 runtime",
)
def test_listener_batch_matches_locked_abi3_for_arbitrary_positive_n() -> None:
    import torch
    from dataclasses import replace

    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.listener_microbatch import ListenerBatch as FrontendBatch
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch.probe_native_listener_microbatch_parity import (
        _bf16_bits,
        _bf16_bits_as_float32,
        _make_manifest,
        _make_pcm_rows,
        _make_probe_rows,
        _source_layer_head_dims,
    )
    from duplex_kernel.microbatch_contract import RuntimeSignature
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import NativeGemmaListenerLM, _load_compiled_listener_topology
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    graph_mode = os.environ.get("DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS", "0") == "1"
    from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    def graph_bucket_widths(capacity: int) -> list[int]:
        widths = [1]
        while widths[-1] < capacity:
            widths.append(capacity if widths[-1] > capacity // 2 else widths[-1] * 2)
        return widths

    def post_growth_probe(
        owner,
        *,
        lm,
        runtime,
        stream,
        audio_soft,
        rows,
        prefix_cache,
        provider,
        initial_reference,
    ) -> dict[str, object]:
        """Measure exact steady low-N ticks after growing the graph table to 16."""
        capacity_before = owner.max_batch_size
        free_before, total_bytes = torch.cuda.mem_get_info(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        owner.ensure_capacity(16)
        free_after, _ = torch.cuda.mem_get_info(device)
        allocated_after = torch.cuda.memory_allocated(device)
        reserved_after = torch.cuda.memory_reserved(device)
        if owner.max_batch_size != 16 or owner.gemm_graph_count <= 0:
            raise AssertionError("capacity-16 graph growth did not commit completely")

        # The outer test has already committed the first tick before calling
        # this probe, so continuation descriptors begin at frame/state 1.
        current_rows = [
            replace(
                row,
                frame_id=row.frame_id + 1,
                listener_state_version=row.listener_state_version + 1,
                sequence_position=row.sequence_position + 1,
                cache_position=row.cache_position + 4,
                valid_history_tokens=row.valid_history_tokens + 4,
            )
            for row in rows
        ]
        def descriptor_key(row) -> tuple[int, int, int, int]:
            return (
                row.frame_id,
                row.listener_state_version,
                row.sequence_position,
                row.cache_position,
            )

        candidate_by_descriptor: list[dict[tuple[int, int, int, int], np.ndarray]] = [
            {descriptor_key(row): initial_reference[index]}
            for index, row in enumerate(rows)
        ]
        timings: dict[str, list[dict[str, float]]] = {"4": [], "2": [], "1": []}
        # Two measured ticks at each active width; rows 2/3 remain owned but
        # inactive after the first two calls, then row 1 after the next two.
        for active_n in (4, 4, 2, 2, 1, 1):
            active_rows = tuple(current_rows[:active_n])
            manifest = _make_manifest(
                active_rows, runtime, stream, NativeGemmaListenerMicrobatch,
            )
            torch.cuda.synchronize(device)
            host_started = time.perf_counter_ns()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            with torch.cuda.stream(stream), torch.inference_mode():
                out = owner.step_async(
                    manifest, audio_soft[:active_n], producer_stream=stream,
                )
            end.record(stream)
            end.synchronize()
            out.completion_event.synchronize()
            timings[str(active_n)].append({
                "host_ms": (time.perf_counter_ns() - host_started) / 1_000_000.0,
                "cuda_ms": float(start.elapsed_time(end)),
            })
            for index in range(active_n):
                row = current_rows[index]
                key = descriptor_key(row)
                if key in candidate_by_descriptor[index]:
                    raise AssertionError(f"duplicate candidate frame descriptor: {key}")
                candidate_by_descriptor[index][key] = _bf16_bits(
                    out.tensor[index:index + 1],
                )[0].copy()
                current_rows[index] = replace(
                    row,
                    frame_id=row.frame_id + 1,
                    listener_state_version=row.listener_state_version + 1,
                    sequence_position=row.sequence_position + 1,
                    cache_position=row.cache_position + 4,
                    valid_history_tokens=row.valid_history_tokens + 4,
                )
            del out

        growth_report = {
            "capacity_before": capacity_before,
            "capacity_after": owner.max_batch_size,
            "graph_count": owner.gemm_graph_count,
            "total_device_bytes": total_bytes,
            "device_free_bytes_before": free_before,
            "device_free_bytes_after": free_after,
            "torch_allocated_delta_bytes": allocated_after - allocated_before,
            "torch_reserved_delta_bytes": reserved_after - reserved_before,
            "known_lane_workspace_bytes": 16 * 32 * 1024 * 1024,
            "active_width_timings": timings,
        }
        print("post_growth_capacity16_measurement", growth_report, flush=True)
        graph_count = owner.gemm_graph_count
        # Release the enlarged private slots/graph cache before constructing
        # the independent standard ABI3 references; the GPU test should not
        # measure parity by holding two full native contexts at once.
        owner.close()
        provider.close()
        provider = NativeGemmaListenerLM(
            lm,
            max_seq_len=16_384,
            stream_q_len=4,
            enable_cuda_graph=False,
            attention_backend=3,
            rmsnorm_backend=2,
        )
        standard_continuations = (6, 4, 2, 2)
        try:
            for row_index, continuation_count in enumerate(standard_continuations):
                provider.reset()
                provider.import_cache_async(
                    prefix_cache,
                    current_position=AUDIO_START_AFTER_TEXT,
                    stream=stream,
                    producer_stream=stream,
                )
                standard_row = rows[row_index]
                for tick in range(continuation_count + 1):
                    with torch.cuda.stream(stream), torch.inference_mode():
                        expected = provider.step_async(
                            audio_soft[row_index:row_index + 1],
                            stream=stream,
                            producer_stream=stream,
                        ).detach().clone()
                        done = torch.cuda.Event(enable_timing=False)
                        done.record(stream)
                    done.synchronize()
                    key = descriptor_key(standard_row)
                    expected_bits = candidate_by_descriptor[row_index].get(key)
                    if expected_bits is None:
                        raise AssertionError(
                            f"no candidate output recorded for standard frame descriptor {key}"
                        )
                    if not np.array_equal(
                        _bf16_bits(expected)[0], expected_bits,
                    ):
                        delta = np.abs(
                            _bf16_bits_as_float32(_bf16_bits(expected)[0])
                            - _bf16_bits_as_float32(expected_bits)
                        )
                        raise AssertionError(
                            "capacity-16 continuation diverged from standard ABI3: "
                            f"row={row_index} descriptor={key} max_abs={float(delta.max(initial=0.0))}"
                        )
                    standard_row = replace(
                        standard_row,
                        frame_id=standard_row.frame_id + 1,
                        listener_state_version=standard_row.listener_state_version + 1,
                        sequence_position=standard_row.sequence_position + 1,
                        cache_position=standard_row.cache_position + 4,
                        valid_history_tokens=standard_row.valid_history_tokens + 4,
                    )
        finally:
            provider.close()

        return {
            **growth_report,
            "graph_count": graph_count,
            "standard_abi3_continuation_exact": True,
        }

    workspace = "/workspace"
    baseline = (
        "/workspace/.tmp/listener_batch_source_overlay/native/gemma4_listener/"
        "libgemma4_listener.so"
    )
    candidate = os.environ.get(
        "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH",
        "/workspace/.tmp/listener_batch_source_overlay/native/gemma4_listener_microbatch/"
        "libgemma4_listener_microbatch.so",
    )
    assert os.path.isfile(baseline) and os.path.isfile(candidate)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = baseline
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(device)
    thinker = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    topology_id = _load_compiled_listener_topology()["manifest_sha256"]
    runtime = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256=topology_id,
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=4,
    )
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    frontend = ListenerMicrobatchAudioFrontend(
        backend.gemma_model,
        backend._extract_single_frame,
        device=device,
        dtype=torch.bfloat16,
        audio_tower_row_runner=backend.run_audio_tower_q4,
    )
    stream = torch.cuda.Stream(device=device, priority=-1)

    with torch.cuda.stream(stream), torch.inference_mode():
        input_ids, attention = backend._build_multimodal_inputs((), 1024, device=device)
        prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
        prefix_attention = attention[:, :AUDIO_START_AFTER_TEXT].to(
            device=device, dtype=torch.long,
        ).contiguous()
        prefix_embeds = lm.embed_tokens(prefix_ids).contiguous()
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds).contiguous()
        positions = torch.arange(AUDIO_START_AFTER_TEXT, device=device).view(1, -1)
        prefix_masks = create_masks_for_generate(
            lm.config,
            prefix_embeds,
            prefix_attention,
            past_key_values=None,
            position_ids=positions,
        )
        oracle = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
        prefix_cache = oracle.forward(
            inputs_embeds=prefix_embeds,
            per_layer_inputs=prefix_ple,
            attention_mask=prefix_masks,
            position_ids=positions,
            past_key_values=None,
            shared_kv_states=None,
        ).past_key_values
    torch.cuda.synchronize(device)

    results: dict[int, dict[str, float | bool]] = {}
    cases = tuple(int(value) for value in os.environ.get(
        "GEMMA4_LISTENER_MICROBATCH_CASES", "1,2,3,4",
    ).split(","))
    if not cases or any(value < 1 for value in cases) or len(set(cases)) != len(cases):
        raise ValueError("GEMMA4_LISTENER_MICROBATCH_CASES must contain unique positive row counts")
    for n in cases:
        run_id = uuid.uuid4().hex
        pcm_rows = _make_pcm_rows(n, phase_seed=n)
        states, frames, rows = _make_probe_rows(
            n, run_id=run_id, pcm_rows=pcm_rows, runtime_signature=runtime,
        )
        manifest = _make_manifest(rows, runtime, stream, NativeGemmaListenerMicrobatch)
        audio_batch = FrontendBatch(manifest=manifest, frames=frames)
        with torch.cuda.stream(stream), torch.inference_mode():
            prepared = frontend.encode_batch(audio_batch, states)
            audio_soft = torch.stack(prepared.audio_soft_tokens, dim=0).contiguous()
        torch.cuda.synchronize(device)

        reference: list[np.ndarray] = []
        reference_kv: list[dict[int, tuple[np.ndarray, np.ndarray]]] = []
        standard_step_host_ms: list[float] = []
        standard_step_cuda_ms: list[float] = []
        provider = NativeGemmaListenerLM(
            lm,
            max_seq_len=16_384,
            stream_q_len=4,
            enable_cuda_graph=False,
            attention_backend=3,
            rmsnorm_backend=2,
        )
        for row_idx in range(n):
            if row_idx:
                provider.reset()
            provider.import_cache_async(
                prefix_cache,
                current_position=AUDIO_START_AFTER_TEXT,
                stream=stream,
                producer_stream=stream,
            )
            standard_started_ns = time.perf_counter_ns()
            with torch.cuda.stream(stream), torch.inference_mode():
                standard_start = torch.cuda.Event(enable_timing=True)
                standard_start.record(stream)
                output = provider.step_async(
                    audio_soft[row_idx:row_idx + 1], stream=stream, producer_stream=stream,
                ).detach().clone()
                done = torch.cuda.Event(enable_timing=True)
                done.record(stream)
            done.synchronize()
            standard_step_host_ms.append((time.perf_counter_ns() - standard_started_ns) / 1_000_000.0)
            standard_step_cuda_ms.append(float(standard_start.elapsed_time(done)))
            reference.append(_bf16_bits(output)[0])
            layer_kv: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for layer_idx in _source_layer_head_dims(lm):
                key, value = provider.get_cached_kv(layer_idx)
                layer_kv[layer_idx] = (_bf16_bits(key)[0], _bf16_bits(value)[0])
                del key, value
            reference_kv.append(layer_kv)
        provider.close()

        debug_taps: list[dict[str, np.ndarray]] = []
        if n == 2:
            tap_stages = {
                "L0_INPUT_NORM": ("tap_input_norm", "hidden"),
                "L0_Q_PROJ": ("tap_q_proj", "q"),
                "L0_Q_NORM": ("tap_q_norm", "qnorm"),
                "L0_K_PROJ": ("tap_k_proj", "kv"),
                "L0_V_PROJ": ("tap_v_proj", "kv"),
                "L0_K_NORM": ("tap_k_norm", "kv"),
                "L0_V_NORM": ("tap_v_norm", "kv"),
                "L0_ATTENTION": ("tap_attn_output", "q"),
                "L0_O_PROJ": ("tap_o_proj", "hidden"),
                "L0_POST_ATTN_NORM": ("tap_post_attn_norm", "hidden"),
                "L0_POST_ATTN_RESIDUAL": ("tap_attn_residual", "hidden"),
                "L0_PRE_FF_NORM": ("tap_pre_ff_norm", "hidden"),
                "L0_GATE_PROJ": ("tap_gate_proj", "ffn"),
                "L0_UP_PROJ": ("tap_up_proj", "ffn"),
                "L0_GATE_GELU": ("tap_mlp_act", "ffn"),
                "L0_GATE_UP_MUL": ("tap_mlp_gate_up", "ffn"),
                "L0_DOWN_PROJ": ("tap_down_proj", "hidden"),
                "L0_POST_FF_NORM": ("tap_post_ff_norm", "hidden"),
                "L0_POST_FF_RESIDUAL": ("tap_ff_residual", "hidden"),
                "L0_PLE_GATE_PROJ": ("tap_ple_gate", "ple"),
                "L0_PLE_GELU": ("tap_ple_act", "ple"),
                "L0_PLE_LAYER_MULT": ("tap_ple_mult", "ple"),
                "L0_PLE_PROJ": ("tap_ple_proj", "hidden"),
                "L0_PLE_NORM": ("tap_ple_norm", "hidden"),
                "L0_PLE_RESIDUAL": ("tap_ple_residual", "hidden"),
                "L0_LAYER_SCALE": ("tap_layer_scaled", "hidden"),
            }
            debug_provider = NativeGemmaListenerLM(
                lm, max_seq_len=16_384, stream_q_len=4,
                enable_cuda_graph=False, attention_backend=3, rmsnorm_backend=2,
            )
            try:
                for row_idx in range(n):
                    if row_idx:
                        debug_provider.reset()
                    debug_provider.import_cache_async(
                        prefix_cache, current_position=AUDIO_START_AFTER_TEXT,
                        stream=stream, producer_stream=stream,
                    )
                    with torch.cuda.stream(stream), torch.inference_mode():
                        debug_provider.step_debug(
                            audio_soft[row_idx:row_idx + 1], num_layers=1,
                            apply_final_norm=False, stream=stream, producer_stream=stream,
                        )
                        done = torch.cuda.Event()
                        done.record(stream)
                    done.synchronize()
                    row_taps: dict[str, np.ndarray] = {}
                    for stage, (tap_name, kind) in tap_stages.items():
                        meta_cols = {
                            "hidden": 1536,
                            "q": int(lm.config.num_attention_heads) * 256,
                            "kv": 256,
                            "qnorm": 256,
                            "ffn": int(lm.config.intermediate_size),
                            "ple": 256,
                        }[kind]
                        row_count = 4
                        if kind == "qnorm":
                            row_count *= int(lm.config.num_attention_heads)
                        tap = debug_provider.copy_debug_tensor(tap_name, (row_count, meta_cols))
                        row_taps[stage] = _bf16_bits(tap).copy()
                    debug_provider.reset()
                    debug_provider.import_cache_async(
                        prefix_cache, current_position=AUDIO_START_AFTER_TEXT,
                        stream=stream, producer_stream=stream,
                    )
                    with torch.cuda.stream(stream), torch.inference_mode():
                        debug_provider.step_debug(
                            audio_soft[row_idx:row_idx + 1],
                            num_layers=35,
                            apply_final_norm=False, stream=stream, producer_stream=stream,
                        )
                        done = torch.cuda.Event()
                        done.record(stream)
                    done.synchronize()
                    all_layer_outputs = debug_provider.copy_debug_tensor(
                        "layer_outputs",
                        (35 * 4, 1536),
                    )
                    row_taps["FINAL_LAYER_OUTPUT"] = _bf16_bits(
                        all_layer_outputs[-4:]
                    ).copy()
                    debug_taps.append(row_taps)
            finally:
                debug_provider.close()

        provider = NativeGemmaListenerLM(
            lm,
            max_seq_len=16_384,
            stream_q_len=4,
            enable_cuda_graph=False,
            attention_backend=3,
            rmsnorm_backend=2,
        )
        free_before_owner, _ = torch.cuda.mem_get_info(device)
        owner = NativeGemmaListenerMicrobatch(
            provider,
            library_path=candidate,
            runtime_signature=runtime,
            listener_stream=stream,
            max_batch_size=min(n, 2),
            enable_gemm_graphs=graph_mode,
        )
        try:
            if graph_mode:
                if not owner.gemm_graphs_enabled or owner.gemm_graph_count <= 0:
                    raise AssertionError("opt-in graph parity run did not prepare its native graph cache")
                free_after_prepare, total_after_prepare = torch.cuda.mem_get_info(device)
                print(
                    "gemm_graph_cache",
                    {
                        "capacity": owner.max_batch_size,
                        "count": owner.gemm_graph_count,
                        "prepare_ms": owner.gemm_graph_prepare_ms,
                        "device_bytes_allocated_during_prepare": free_before_owner - free_after_prepare,
                        "total_device_bytes": total_after_prepare,
                    },
                )
            graph_count_before_growth = owner.gemm_graph_count
            capacity_before_growth = owner.max_batch_size
            owner.ensure_capacity(n)
            if (
                graph_mode
                and owner.max_batch_size > capacity_before_growth
                and owner.gemm_graph_count <= graph_count_before_growth
            ):
                raise AssertionError(
                    "capacity growth did not prepare additional geometric graph buckets"
                )
            if graph_mode:
                free_after_growth, _ = torch.cuda.mem_get_info(device)
                print(
                    "gemm_graph_cache_after_growth",
                    {
                        "capacity": owner.max_batch_size,
                        "bucket_widths": graph_bucket_widths(owner.max_batch_size),
                        "count": owner.gemm_graph_count,
                        "device_bytes_allocated_during_growth": free_after_prepare - free_after_growth,
                    },
                )
            for row in rows:
                owner.import_session(
                    row,
                    prefix_cache,
                    prefix_attention,
                    prefix_len=AUDIO_START_AFTER_TEXT,
                    producer_stream=stream,
                )
            if n == 2:
                owner.probe_arm()
                owner.probe_trace_arm()
            candidate_started_ns = time.perf_counter_ns()
            candidate_start = torch.cuda.Event(enable_timing=True)
            candidate_start.record(stream)
            output = owner.step_async(manifest, audio_soft, producer_stream=stream)
            candidate_end = torch.cuda.Event(enable_timing=True)
            candidate_end.record(stream)
            candidate_end.synchronize()
            output.completion_event.synchronize()
            candidate_host_ms = (time.perf_counter_ns() - candidate_started_ns) / 1_000_000.0
            candidate_cuda_ms = float(candidate_start.elapsed_time(candidate_end))
            first_trace_difference = None
            layer0_trace_differences: list[dict[str, object]] = []
            first_operator_probe = None
            if n == 2:
                trace_records = owner.probe_trace_read_all()
                for trace_record in trace_records:
                    metadata = trace_record["metadata"]
                    stage = str(metadata["stage_name"])
                    if stage not in debug_taps[0]:
                        continue
                    actual_rows = int(metadata["actual_rows"])
                    actual_cols = int(metadata["actual_cols"])
                    rows_per_session = actual_rows // n
                    actual_bits = np.frombuffer(
                        trace_record["raw"]["actual_bits"], dtype=np.uint16,
                    ).reshape(actual_rows, actual_cols)
                    for sid in range(n):
                        batched = actual_bits[
                            sid * rows_per_session:(sid + 1) * rows_per_session
                        ]
                        standard = debug_taps[sid][stage]
                        if batched.shape != standard.shape:
                            raise AssertionError(
                                f"{stage} row {sid} shape mismatch: {batched.shape} vs {standard.shape}"
                            )
                        delta = np.abs(
                            _bf16_bits_as_float32(batched) -
                            _bf16_bits_as_float32(standard)
                        )
                        if not np.array_equal(batched, standard):
                            reference_float = _bf16_bits_as_float32(standard)
                            actual_float = _bf16_bits_as_float32(batched)
                            item = {
                                "stage": stage,
                                "row": sid,
                                "max_abs": float(delta.max(initial=0.0)),
                                "rms": float(np.sqrt(np.square(delta).mean())),
                                "cosine": float(np.dot(actual_float.ravel(), reference_float.ravel()) /
                                                max(1e-30, np.linalg.norm(actual_float) * np.linalg.norm(reference_float))),
                                "first_index": tuple(int(v) for v in np.argwhere(batched != standard)[0]),
                            }
                            layer0_trace_differences.append(item)
                            if first_trace_difference is None:
                                first_trace_difference = item
                rms_meta, rms_raw = owner.probe_read_rms()
                rms_actual = np.frombuffer(rms_raw["batched_bits"], dtype=np.uint16).reshape(8, 1536)
                rms_split = np.frombuffer(rms_raw["split_bits"], dtype=np.uint16).reshape(8, 1536)
                rms_equal = bool(np.array_equal(rms_actual, rms_split))
                owner.probe_mark_rms_exact(rms_equal)
                rms_delta = np.abs(_bf16_bits_as_float32(rms_actual) - _bf16_bits_as_float32(rms_split))
                first_operator_probe = {
                    "rms_batch_vs_split_exact": rms_equal,
                    "rms_max_abs": float(rms_delta.max(initial=0.0)),
                    "rms_rms": float(np.sqrt(np.square(rms_delta).mean())),
                    "rms_batch_shape": [rms_meta.get("rms_batched_block_x"), rms_meta.get("rms_batched_block_y")],
                    "rms_split_shape": [rms_meta.get("rms_reference_block_x"), rms_meta.get("rms_reference_block_y")],
                }
                if rms_equal:
                    owner.probe_qproj_async()
                    stream.synchronize()
                    q_meta, q_raw = owner.probe_read_qproj()
                    qdim = int(q_meta["q_dim"])
                    q_actual = np.frombuffer(q_raw["actual_bits"], dtype=np.uint16).reshape(8, qdim)
                    q_batched = np.frombuffer(q_raw["batched_bits"], dtype=np.uint16).reshape(8, qdim)
                    q_split = np.frombuffer(q_raw["split_bits"], dtype=np.uint16).reshape(8, qdim)
                    q_delta = np.abs(_bf16_bits_as_float32(q_actual) - _bf16_bits_as_float32(q_split))
                    first_operator_probe.update({
                        "q_actual_vs_m8_exact": bool(np.array_equal(q_actual, q_batched)),
                        "q_actual_vs_m4_split_exact": bool(np.array_equal(q_actual, q_split)),
                        "q_batch_vs_split_max_abs": float(q_delta.max(initial=0.0)),
                        "q_batch_vs_split_rms": float(np.sqrt(np.square(q_delta).mean())),
                        "q_m": [q_meta.get("q_batched_m"), q_meta.get("q_split_m")],
                    })
            actual = _bf16_bits(output.tensor)
            max_hidden_error = 0.0
            max_hidden_relative = 0.0
            hidden_sq_error = 0.0
            hidden_dot = 0.0
            hidden_ref_sq = 0.0
            hidden_actual_sq = 0.0
            max_kv_error = 0.0
            max_kv_relative = 0.0
            kv_sq_error = 0.0
            kv_values = 0
            first_kv_difference = None
            state_ok = True
            for index, row in enumerate(rows):
                hidden_ref = _bf16_bits_as_float32(reference[index])
                hidden_actual = _bf16_bits_as_float32(actual[index])
                hidden_delta = np.abs(hidden_actual - hidden_ref)
                max_hidden_error = max(max_hidden_error, float(hidden_delta.max(initial=0.0)))
                max_hidden_relative = max(
                    max_hidden_relative,
                    float((hidden_delta / np.maximum(np.abs(hidden_ref), 1e-6)).max(initial=0.0)),
                )
                hidden_sq_error += float(np.square(hidden_delta).sum())
                hidden_dot += float(np.multiply(hidden_actual, hidden_ref).sum())
                hidden_ref_sq += float(np.square(hidden_ref).sum())
                hidden_actual_sq += float(np.square(hidden_actual).sum())
                next_row = replace(
                    row,
                    frame_id=1,
                    listener_state_version=1,
                    sequence_position=1,
                    cache_position=AUDIO_START_AFTER_TEXT + 4,
                    valid_history_tokens=AUDIO_START_AFTER_TEXT + 4,
                )
                snapshot = owner.snapshot_session(next_row)
                state_ok &= snapshot.state.current_position == AUDIO_START_AFTER_TEXT + 4
                for layer_idx, (expected_key, expected_value) in reference_kv[index].items():
                    got_key, got_value = snapshot.kv_by_layer[layer_idx]
                    for expected, got in ((expected_key, _bf16_bits(got_key)[0]),
                                          (expected_value, _bf16_bits(got_value)[0])):
                        expected_float = _bf16_bits_as_float32(expected)
                        got_float = _bf16_bits_as_float32(got)
                        delta = np.abs(got_float - expected_float)
                        max_kv_error = max(max_kv_error, float(delta.max(initial=0.0)))
                        max_kv_relative = max(
                            max_kv_relative,
                            float((delta / np.maximum(np.abs(expected_float), 1e-6)).max(initial=0.0)),
                        )
                        kv_sq_error += float(np.square(delta).sum())
                        kv_values += int(delta.size)
                        if first_kv_difference is None and not np.array_equal(got, expected):
                            first = tuple(int(v) for v in np.argwhere(got != expected)[0])
                            first_kv_difference = {
                                "row": index,
                                "layer": layer_idx,
                                "kind": "key" if expected is expected_key else "value",
                                "index": first,
                                "expected": float(expected_float[first]),
                                "actual": float(got_float[first]),
                            }
            results[n] = {
                "max_hidden_abs_error": max_hidden_error,
                "max_hidden_relative_error": max_hidden_relative,
                "rms_hidden_error": float(np.sqrt(hidden_sq_error / (n * actual[0].size))),
                "hidden_cosine": hidden_dot / np.sqrt(hidden_ref_sq * hidden_actual_sq),
                "hidden_relative_rms": float(np.sqrt(hidden_sq_error / (n * actual[0].size)) /
                                              np.sqrt(hidden_ref_sq / (n * actual[0].size))),
                "max_kv_abs_error": max_kv_error,
                "max_kv_relative_error": max_kv_relative,
                "rms_kv_error": float(np.sqrt(kv_sq_error / max(1, kv_values))),
                "first_kv_difference": first_kv_difference,
                "first_layer0_trace_difference": first_trace_difference,
                "layer0_trace_differences": layer0_trace_differences,
                "first_layer0_operator_probe": first_operator_probe,
                "state_cursor_ok": state_ok,
                "batch_rows": float(n),
                "standard_step_host_ms": standard_step_host_ms,
                "standard_step_cuda_ms": standard_step_cuda_ms,
                "standard_rows_host_ms": float(sum(standard_step_host_ms)),
                "candidate_batch_host_ms": candidate_host_ms,
                "candidate_batch_cuda_ms": candidate_cuda_ms,
            }
            if (
                graph_mode
                and n == 4
                and os.environ.get("GEMMA4_LISTENER_MICROBATCH_POST_GROWTH_CAPACITY") == "16"
            ):
                results[n]["post_capacity16_steady"] = post_growth_probe(
                    owner,
                    lm=lm,
                    runtime=runtime,
                    stream=stream,
                    audio_soft=audio_soft,
                    rows=rows,
                    prefix_cache=prefix_cache,
                    provider=provider,
                    initial_reference=reference,
                )
        finally:
            owner.close()
            provider.close()
            prepared.commit()
        print(f"listener-microbatch N={n}: {results[n]}", flush=True)

    assert all(result["state_cursor_ok"] for result in results.values())
    assert all(
        float(result["max_hidden_abs_error"]) <= 1e-2
        and float(result["max_kv_abs_error"]) <= 1e-2
        for result in results.values()
    ), results
