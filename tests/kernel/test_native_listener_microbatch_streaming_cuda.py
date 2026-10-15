"""Opt-in multi-tick state-parity test for the native listener microbatch.

Run inside the configured RTX 3090 image with
``GEMMA4_RUN_LISTENER_MICROBATCH_GPU=1``. This complements the one-frame
parity test by retaining real native KV state across row permutations,
capacity growth, and a session generation/epoch reset.
"""
from __future__ import annotations

import os
import time
import uuid

import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_LISTENER_MICROBATCH_GPU") != "1",
    reason="requires the explicit RTX 3090 listener-microbatch runtime",
)
def test_streaming_state_parity_growth_permutation_and_epoch_reset() -> None:
    import torch
    from dataclasses import replace

    from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.listener_microbatch import ListenerBatch as FrontendBatch
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch.probe_native_listener_microbatch_parity import (
        _bf16_bits,
        _make_manifest,
        _make_pcm_rows,
        _make_probe_rows,
        _source_layer_head_dims,
        assert_exact_array,
    )
    from duplex_kernel.microbatch_contract import ListenerFrame, ListenerSessionState, RuntimeSignature
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import NativeGemmaListenerLM, _load_compiled_listener_topology
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    workspace = "/workspace"
    baseline_library = (
        "/workspace/.tmp/listener_batch_source_overlay/native/gemma4_listener/"
        "libgemma4_listener.so"
    )
    candidate_library = (
        "/workspace/.tmp/listener_batch_source_overlay/native/gemma4_listener_microbatch/"
        "libgemma4_listener_microbatch.so"
    )
    assert os.path.isfile(baseline_library) and os.path.isfile(candidate_library)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = baseline_library
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"

    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(device)
    thinker = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    runtime = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256=_load_compiled_listener_topology()["manifest_sha256"],
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
    assert AUDIO_START_AFTER_TEXT == 34
    source_layers = _source_layer_head_dims(lm)

    def assert_same_snapshot(label: str, expected: object, actual: object) -> None:
        assert expected.state == actual.state, f"{label}: native cursor/state changed"
        assert set(expected.kv_by_layer) == set(actual.kv_by_layer)
        for layer_idx in sorted(expected.kv_by_layer):
            for kind, left, right in zip(
                ("key", "value"), expected.kv_by_layer[layer_idx],
                actual.kv_by_layer[layer_idx], strict=True,
            ):
                assert_exact_array(
                    f"{label} layer {layer_idx} {kind}",
                    _bf16_bits(left)[0], _bf16_bits(right)[0],
                )

    def make_current_row(base_row: object, state: ListenerSessionState, frame: ListenerFrame):
        return replace(
            base_row,
            session_generation=state.generation_id,
            state_epoch_id=state.state_epoch_id,
            frame_id=state.next_frame_id,
            listener_state_version=state.state_version,
            sequence_position=state.sequence_position,
            cache_position=state.cache_position,
            valid_history_tokens=state.valid_history_tokens,
            native_context_id=state.native_context_id,
            input_frame_sha256=frame.pcm_sha256,
            reference_pcm_sha256=frame.reference_pcm_sha256,
            frame_ready_at_ns=frame.ready_at_ns,
            reference_frame_id=frame.reference_frame_id,
        )

    def assert_final_kv_matches_reference(
        owner: NativeGemmaListenerMicrobatch,
        row: object,
        reference: NativeGemmaListenerLM,
        label: str,
    ) -> None:
        native = owner.snapshot_session(row)
        assert native.state.current_position == row.cache_position
        assert native.state.valid_kv_len == row.valid_history_tokens
        assert set(native.kv_by_layer) == set(source_layers)
        for layer_idx in sorted(source_layers):
            ref_key, ref_value = reference.get_cached_kv(layer_idx)
            got_key, got_value = native.kv_by_layer[layer_idx]
            assert_exact_array(
                f"{label} layer {layer_idx} key",
                _bf16_bits(ref_key)[0], _bf16_bits(got_key)[0],
            )
            assert_exact_array(
                f"{label} layer {layer_idx} value",
                _bf16_bits(ref_value)[0], _bf16_bits(got_value)[0],
            )

    for batch_size in (1, 2, 4):
        run_id = uuid.uuid4().hex
        initial_pcm = _make_pcm_rows(batch_size, phase_seed=100 + batch_size)
        states, _, rows = _make_probe_rows(
            batch_size,
            run_id=run_id,
            pcm_rows=initial_pcm,
            runtime_signature=runtime,
        )
        session_ids = [row.session_id for row in rows]
        state_by_sid = {state.session_id: state for state in states}
        base_row_by_sid = {row.session_id: row for row in rows}
        current_row_by_sid = dict(base_row_by_sid)

        references: list[NativeGemmaListenerLM] = []
        owner: NativeGemmaListenerMicrobatch | None = None
        try:
            for _ in range(batch_size):
                reference = NativeGemmaListenerLM(
                    lm,
                    max_seq_len=16_384,
                    stream_q_len=4,
                    enable_cuda_graph=False,
                    attention_backend=3,
                    rmsnorm_backend=2,
                )
                reference.reset()
                reference.import_cache_async(
                    prefix_cache,
                    current_position=AUDIO_START_AFTER_TEXT,
                    stream=stream,
                    producer_stream=stream,
                )
                references.append(reference)

            owner_provider = NativeGemmaListenerLM(
                lm,
                max_seq_len=16_384,
                stream_q_len=4,
                enable_cuda_graph=False,
                attention_backend=3,
                rmsnorm_backend=2,
            )
            owner = NativeGemmaListenerMicrobatch(
                owner_provider,
                library_path=candidate_library,
                runtime_signature=runtime,
                listener_stream=stream,
                max_batch_size=1,
            )
            owner.import_session(
                current_row_by_sid[session_ids[0]],
                prefix_cache,
                prefix_attention,
                prefix_len=AUDIO_START_AFTER_TEXT,
                producer_stream=stream,
            )

            total_ticks = 12 if batch_size == 1 else 15
            observed_orders: set[tuple[str, ...]] = set()
            for global_tick in range(total_ticks):
                if batch_size > 1 and global_tick == 3:
                    first_sid = session_ids[0]
                    row_before_growth = current_row_by_sid[first_sid]
                    before_growth = owner.snapshot_session(row_before_growth)
                    old_capacity = owner.max_batch_size
                    grown_capacity = owner.ensure_capacity(batch_size)
                    assert old_capacity == 1 and grown_capacity >= batch_size
                    after_growth = owner.snapshot_session(row_before_growth)
                    assert_same_snapshot(
                        f"N={batch_size} populated-cache growth", before_growth, after_growth,
                    )
                    for sid in session_ids[1:]:
                        owner.import_session(
                            current_row_by_sid[sid],
                            prefix_cache,
                            prefix_attention,
                            prefix_len=AUDIO_START_AFTER_TEXT,
                            producer_stream=stream,
                        )

                active_sids = (
                    session_ids if batch_size == 1 or global_tick >= 3
                    else session_ids[:1]
                )
                ordered_sids = list(active_sids)
                if global_tick % 2:
                    ordered_sids.reverse()
                observed_orders.add(tuple(ordered_sids))

                now_ns = time.monotonic_ns()
                ordered_frames: list[ListenerFrame] = []
                ordered_rows = []
                ordered_states: list[ListenerSessionState] = []
                for sid in ordered_sids:
                    state = state_by_sid[sid]
                    session_index = session_ids.index(sid)
                    phase_seed = state.generation_id * 1000 + state.next_frame_id
                    pcm = _make_pcm_rows(batch_size, phase_seed=phase_seed)[session_index]
                    frame = ListenerFrame(
                        session_id=sid,
                        generation_id=state.generation_id,
                        state_epoch_id=state.state_epoch_id,
                        frame_id=state.next_frame_id,
                        ready_at_ns=now_ns,
                        pcm=pcm,
                    )
                    row = make_current_row(base_row_by_sid[sid], state, frame)
                    ordered_frames.append(frame)
                    ordered_rows.append(row)
                    ordered_states.append(state)

                manifest = _make_manifest(
                    ordered_rows, runtime, stream, NativeGemmaListenerMicrobatch,
                )
                frontend_batch = FrontendBatch(
                    manifest=manifest,
                    frames=tuple(ordered_frames),
                )
                prepared = None
                try:
                    with torch.cuda.stream(stream), torch.inference_mode():
                        prepared = frontend.encode_batch(frontend_batch, ordered_states)
                        audio_soft = torch.stack(
                            prepared.audio_soft_tokens, dim=0,
                        ).contiguous()
                        reference_hidden: dict[str, object] = {}
                        for index, sid in enumerate(ordered_sids):
                            reference = references[session_ids.index(sid)]
                            hidden = reference.step_async(
                                audio_soft[index:index + 1],
                                stream=stream,
                                producer_stream=stream,
                            ).detach().clone()
                            reference_hidden[sid] = _bf16_bits(hidden)[0].copy()

                        output = owner.step_async(
                            manifest, audio_soft, producer_stream=stream,
                        )
                    output.completion_event.synchronize()
                    health = owner.verify_output_health(output)
                    assert tuple(row.session_id for row in output.rows) == tuple(ordered_sids)
                    assert all(item.checked and not item.nonfinite and item.status == "healthy"
                               for item in health)
                    actual_hidden = _bf16_bits(output.tensor)
                    next_rows: dict[str, object] = {}
                    for index, (sid, row) in enumerate(zip(ordered_sids, ordered_rows, strict=True)):
                        assert_exact_array(
                            f"N={batch_size} tick={global_tick} {sid} hidden",
                            reference_hidden[sid], actual_hidden[index],
                        )
                        next_row = replace(
                            row,
                            frame_id=row.frame_id + 1,
                            listener_state_version=row.listener_state_version + 1,
                            sequence_position=row.sequence_position + 1,
                            cache_position=row.cache_position + 4,
                            valid_history_tokens=row.valid_history_tokens + 4,
                        )
                        native_state = owner.session_state(next_row)
                        assert native_state.current_position == next_row.cache_position
                        assert native_state.valid_kv_len == next_row.valid_history_tokens
                        assert native_state.native_context_id == next_row.native_context_id
                        assert native_state.session_generation == next_row.session_generation
                        assert native_state.expected_frame_id == next_row.frame_id
                        assert native_state.expected_state_version == next_row.listener_state_version
                        assert native_state.expected_sequence_position == next_row.sequence_position
                        assert references[session_ids.index(sid)].current_position == next_row.cache_position
                        next_rows[sid] = next_row

                    prepared.commit()
                    for sid, next_row in next_rows.items():
                        state = state_by_sid[sid]
                        state.next_frame_id = next_row.frame_id
                        state.state_version = next_row.listener_state_version
                        state.sequence_position = next_row.sequence_position
                        state.cache_position = next_row.cache_position
                        state.valid_history_tokens = next_row.valid_history_tokens
                        current_row_by_sid[sid] = next_row
                    prepared = None
                finally:
                    if prepared is not None and not prepared.committed:
                        try:
                            prepared.abort()
                        except RuntimeError:
                            pass

            if batch_size > 1:
                assert len(observed_orders) > 1, "the test must exercise a real row permutation"

            # Reset one live SID by retiring and reimporting its native slot
            # under a new generation, epoch, and opaque context identity.
            reset_sid = session_ids[0]
            old_row = current_row_by_sid[reset_sid]
            owner.release_session_async(old_row)
            references[0].reset()
            references[0].import_cache_async(
                prefix_cache,
                current_position=AUDIO_START_AFTER_TEXT,
                stream=stream,
                producer_stream=stream,
            )
            old_state = state_by_sid[reset_sid]
            new_context_id = int(old_state.native_context_id) + 100_000
            new_generation = old_state.generation_id + 1
            new_epoch = f"{old_state.state_epoch_id}-reset-{uuid.uuid4().hex[:8]}"
            reset_state = ListenerSessionState(
                session_id=reset_sid,
                generation_id=new_generation,
                state_epoch_id=new_epoch,
                runtime_signature=runtime,
                next_frame_id=0,
                state_version=0,
                sequence_position=0,
                cache_position=AUDIO_START_AFTER_TEXT,
                valid_history_tokens=AUDIO_START_AFTER_TEXT,
                native_context_id=new_context_id,
            )
            reset_row = replace(
                old_row,
                session_generation=new_generation,
                state_epoch_id=new_epoch,
                frame_id=0,
                listener_state_version=0,
                sequence_position=0,
                cache_position=AUDIO_START_AFTER_TEXT,
                valid_history_tokens=AUDIO_START_AFTER_TEXT,
                native_context_id=new_context_id,
            )
            owner.import_session(
                reset_row,
                prefix_cache,
                prefix_attention,
                prefix_len=AUDIO_START_AFTER_TEXT,
                producer_stream=stream,
            )
            state_by_sid[reset_sid] = reset_state
            current_row_by_sid[reset_sid] = reset_row
            base_row_by_sid[reset_sid] = reset_row

            stale_manifest = _make_manifest(
                (old_row,), runtime, stream, NativeGemmaListenerMicrobatch,
            )
            with pytest.raises(RuntimeError, match="no bound native session"):
                owner.step_async(
                    stale_manifest,
                    torch.empty((1, 4, 1536), dtype=torch.bfloat16, device=device),
                    producer_stream=stream,
                )

            for reset_tick in range(2):
                now_ns = time.monotonic_ns()
                ordered_sids = session_ids[reset_tick % batch_size:] + session_ids[:reset_tick % batch_size]
                ordered_frames = []
                ordered_rows = []
                ordered_states = []
                for sid in ordered_sids:
                    state = state_by_sid[sid]
                    session_index = session_ids.index(sid)
                    phase_seed = state.generation_id * 1000 + state.next_frame_id
                    pcm = _make_pcm_rows(batch_size, phase_seed=phase_seed)[session_index]
                    frame = ListenerFrame(
                        session_id=sid,
                        generation_id=state.generation_id,
                        state_epoch_id=state.state_epoch_id,
                        frame_id=state.next_frame_id,
                        ready_at_ns=now_ns,
                        pcm=pcm,
                    )
                    row = make_current_row(base_row_by_sid[sid], state, frame)
                    ordered_frames.append(frame)
                    ordered_rows.append(row)
                    ordered_states.append(state)
                manifest = _make_manifest(
                    ordered_rows, runtime, stream, NativeGemmaListenerMicrobatch,
                )
                prepared = None
                try:
                    with torch.cuda.stream(stream), torch.inference_mode():
                        prepared = frontend.encode_batch(
                            FrontendBatch(manifest=manifest, frames=tuple(ordered_frames)),
                            ordered_states,
                        )
                        audio_soft = torch.stack(prepared.audio_soft_tokens, dim=0).contiguous()
                        reference_hidden = {}
                        for index, sid in enumerate(ordered_sids):
                            hidden = references[session_ids.index(sid)].step_async(
                                audio_soft[index:index + 1],
                                stream=stream,
                                producer_stream=stream,
                            ).detach().clone()
                            reference_hidden[sid] = _bf16_bits(hidden)[0].copy()
                        output = owner.step_async(
                            manifest, audio_soft, producer_stream=stream,
                        )
                    output.completion_event.synchronize()
                    health = owner.verify_output_health(output)
                    assert all(item.checked and not item.nonfinite and item.status == "healthy"
                               for item in health)
                    actual_hidden = _bf16_bits(output.tensor)
                    next_rows = {}
                    for index, (sid, row) in enumerate(zip(ordered_sids, ordered_rows, strict=True)):
                        assert_exact_array(
                            f"N={batch_size} reset tick={reset_tick} {sid} hidden",
                            reference_hidden[sid], actual_hidden[index],
                        )
                        next_row = replace(
                            row,
                            frame_id=row.frame_id + 1,
                            listener_state_version=row.listener_state_version + 1,
                            sequence_position=row.sequence_position + 1,
                            cache_position=row.cache_position + 4,
                            valid_history_tokens=row.valid_history_tokens + 4,
                        )
                        native_state = owner.session_state(next_row)
                        assert native_state.current_position == next_row.cache_position
                        assert native_state.valid_kv_len == next_row.valid_history_tokens
                        assert native_state.native_context_id == next_row.native_context_id
                        assert native_state.session_generation == next_row.session_generation
                        assert native_state.expected_frame_id == next_row.frame_id
                        assert native_state.expected_state_version == next_row.listener_state_version
                        assert native_state.expected_sequence_position == next_row.sequence_position
                        assert references[session_ids.index(sid)].current_position == next_row.cache_position
                        next_rows[sid] = next_row
                    prepared.commit()
                    for sid, next_row in next_rows.items():
                        state = state_by_sid[sid]
                        state.next_frame_id = next_row.frame_id
                        state.state_version = next_row.listener_state_version
                        state.sequence_position = next_row.sequence_position
                        state.cache_position = next_row.cache_position
                        state.valid_history_tokens = next_row.valid_history_tokens
                        current_row_by_sid[sid] = next_row
                    prepared = None
                finally:
                    if prepared is not None and not prepared.committed:
                        try:
                            prepared.abort()
                        except RuntimeError:
                            pass

            for sid in session_ids:
                assert_final_kv_matches_reference(
                    owner,
                    current_row_by_sid[sid],
                    references[session_ids.index(sid)],
                    f"N={batch_size} final {sid}",
                )
        finally:
            if owner is not None:
                owner.close()
            for reference in references:
                reference.close()
            stream.synchronize()
