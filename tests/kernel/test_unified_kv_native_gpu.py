"""Opt-in actual-model smoke for unified native KV and same-owner fork contracts.

Run inside the Mondaini GPU container with
``GEMMA4_RUN_UNIFIED_KV_GPU=1`` and
``DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH`` set to the API5 candidate DSO.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace

import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_UNIFIED_KV_GPU") != "1",
    reason="requires explicit RTX 3090 runtime",
)
def test_unified_native_q0_q1_and_queued_audio_keep_one_kv_cursor() -> None:
    import torch

    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor,
        ListenerBatchManifest,
        RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import (
        NativeGemmaListenerLM,
        _load_compiled_listener_topology,
    )
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from duplex_kernel.unified_token_adapter import UnifiedTokenAdapter

    workspace = "/workspace"
    candidate = os.environ.get("DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH")
    assert candidate and os.path.isfile(candidate)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = candidate
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"
    os.environ["GEMMA4_NATIVE_LISTENER_CUDA_GRAPH"] = "0"

    device = torch.device("cuda:0")
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(device)
    thinker = load_thinker(workspace, device=device, dtype=torch.bfloat16)
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    tokenizer = thinker.tokenizer
    marker = (
        getattr(backend.processor, "audio_token", None)
        or getattr(tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    messages = [
        {"role": "system", "content": "Rispondi in modo breve e resta in ascolto."},
        {"role": "user", "content": f"{marker}"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    ids = list(tokenizer.encode(str(rendered), add_special_tokens=False))
    marker_ids = tokenizer.encode(marker, add_special_tokens=False)
    assert len(marker_ids) == 1
    marker_id = int(marker_ids[0])
    marker_positions = [index for index, value in enumerate(ids) if int(value) == marker_id]
    assert len(marker_positions) == 1
    prefix_ids = tuple(int(value) for value in ids[:marker_positions[0]]) + (
        int(backend.boa_token_id),
    )
    assert 1 <= len(prefix_ids) <= 291 and prefix_ids[0] == backend.bos_token_id

    stream = torch.cuda.Stream(device=device, priority=-1)
    lock = backend._lock
    if not lock.acquire_fair("listener"):
        raise RuntimeError("unified GPU smoke could not acquire the Thinker prefix lock")
    try:
        with torch.cuda.stream(stream), torch.inference_mode():
            prefix = backend.build_semantic_prefix_cache(prefix_ids)
        stream.synchronize()
    finally:
        lock.release()

    topology_sha256 = _load_compiled_listener_topology()["manifest_sha256"]
    signature = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256=topology_sha256,
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=5,
        attention_backend_id=3,
        rmsnorm_backend_id=2,
        max_seq_len=16_384,
        stream_q_len=2,
    )
    provider = NativeGemmaListenerLM(
        backend.gemma_model.language_model,
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
    session_id = f"unified-kv-gpu-{uuid.uuid4().hex}"
    epoch_id = uuid.uuid4().hex
    prefix_len = len(prefix_ids)
    row = BatchRowDescriptor(
        session_id=session_id,
        session_generation=1,
        state_epoch_id=epoch_id,
        frame_id=0,
        listener_state_version=0,
        sequence_position=0,
        cache_position=prefix_len,
        valid_history_tokens=prefix_len,
        native_context_id=1,
        input_frame_sha256="0" * 64,
        reference_pcm_sha256=None,
        frame_ready_at_ns=time.monotonic_ns(),
        reference_frame_id=None,
    )
    fork_cleanup_row: BatchRowDescriptor | None = None

    def make_manifest(row_value: BatchRowDescriptor) -> ListenerBatchManifest:
        return ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=signature,
            stream_identity=owner.capability.stream_identity,
            stream_priority=owner.capability.stream_priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="carryover_ready",
            rows=(row_value,),
        )

    def step(
        row_value: BatchRowDescriptor,
        inputs: torch.Tensor,
        *,
        per_layer_inputs: torch.Tensor | None = None,
        queued: bool = False,
    ):
        manifest = make_manifest(row_value)
        if queued:
            result = owner.step_queued_audio_async(
                manifest, inputs, producer_stream=stream,
            )
        else:
            result = owner.step_async(
                manifest, inputs, per_layer_inputs=per_layer_inputs,
                producer_stream=stream,
            )
        result.completion_event.synchronize()
        health = owner.verify_output_health(result)
        assert len(health) == 1 and health[0].checked and not health[0].nonfinite
        assert torch.isfinite(result.tensor).all().item()
        return result

    try:
        owner.import_session(
            row,
            prefix["past_key_values"],
            prefix["prefix_attention"],
            prefix_len=prefix_len,
            producer_stream=stream,
        )
        adapter = UnifiedTokenAdapter(thinker, tokenizer=tokenizer, require_cuda=True)
        lm = backend.gemma_model.language_model
        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

        parity_deltas: dict[str, tuple[float, float]] = {}
        parity_cosines: dict[str, float] = {}
        logits_top1_matches: dict[str, bool] = {}

        def make_reference_prefix_cache():
            reference_lock = backend._lock
            if not reference_lock.acquire_fair("listener"):
                raise RuntimeError("unified GPU smoke could not acquire the Thinker reference lock")
            try:
                with torch.cuda.stream(stream), torch.inference_mode():
                    reference_prefix = backend.build_semantic_prefix_cache(prefix_ids)
                stream.synchronize()
                return reference_prefix["past_key_values"]
            finally:
                reference_lock.release()

        def reference_forward(inputs_embeds, token_identity_ple, past_key_values, position_start):
            q_len = int(inputs_embeds.shape[1])
            position_ids = torch.arange(
                position_start, position_start + q_len, dtype=torch.long, device=device,
            ).view(1, q_len)
            attention = torch.ones(
                (1, position_start + q_len), dtype=torch.long, device=device,
            )
            masks = create_masks_for_generate(
                lm.config, inputs_embeds, attention,
                past_key_values=past_key_values, position_ids=position_ids,
            )
            reference_lock = backend._lock
            if not reference_lock.acquire_fair("listener"):
                raise RuntimeError("unified GPU smoke could not acquire the Thinker reference lock")
            try:
                with torch.cuda.stream(stream), torch.inference_mode():
                    # Use Transformers' installed Gemma4TextModel forward as the
                    # reference. `per_layer_inputs` here is the raw token identity
                    # tensor: the model itself applies projection/RMSNorm/scaling.
                    reference_output = lm(
                        input_ids=None,
                        inputs_embeds=inputs_embeds,
                        per_layer_inputs=token_identity_ple,
                        attention_mask=masks,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                stream.synchronize()
                return reference_output.last_hidden_state.detach().clone(), reference_output.past_key_values
            finally:
                reference_lock.release()

        def compare_hidden(name, native_hidden, reference_hidden):
            delta = (native_hidden.float() - reference_hidden.float()).abs()
            max_abs = float(delta.max().item())
            rms = float(delta.square().mean().sqrt().item())
            native_f = native_hidden.float()
            reference_f = reference_hidden.float()
            cosine = float(torch.nn.functional.cosine_similarity(
                native_f.reshape(-1, native_f.shape[-1]),
                reference_f.reshape(-1, reference_f.shape[-1]), dim=-1,
            ).mean().item())
            native_norm = float(torch.linalg.vector_norm(native_f, dim=-1).mean().item())
            reference_norm = float(torch.linalg.vector_norm(reference_f, dim=-1).mean().item())
            parity_deltas[name] = (max_abs, rms)
            parity_cosines[name] = cosine
            print(
                f"{name}_HIDDEN max_abs={max_abs:.8g} rms={rms:.8g} "
                f"cosine={cosine:.8g} native_norm={native_norm:.8g} "
                f"reference_norm={reference_norm:.8g}", flush=True,
            )

        def compare_logits(name, native_hidden, reference_hidden):
            native_logits = adapter.logits_from_final_norm_hidden(native_hidden).float()
            reference_logits = adapter.logits_from_final_norm_hidden(reference_hidden).float()
            native_logits = native_logits[:, -1, :]
            reference_logits = reference_logits[:, -1, :]
            native_top = torch.topk(native_logits, k=5, dim=-1).indices[0]
            reference_top = torch.topk(reference_logits, k=5, dim=-1).indices[0]
            greedy_native = int(native_top[0].item())
            greedy_reference = int(reference_top[0].item())
            matches = greedy_native == greedy_reference
            logits_top1_matches[name] = matches
            print(
                f"{name}_LOGITS greedy_native={greedy_native} "
                f"greedy_reference={greedy_reference} "
                f"top5_native={native_top.tolist()} top5_reference={reference_top.tolist()} "
                f"top1_match={matches}",
                flush=True,
            )
            return matches

        # First isolate token-only Q2 at the true imported chat prefix.
        text_token_ids = (
            int(tokenizer.encode("Sì", add_special_tokens=False)[0]),
            int(tokenizer.encode("Certo", add_special_tokens=False)[0]),
        )
        text_ids = torch.tensor([text_token_ids], dtype=torch.long, device=device)
        text_prepared = adapter.prepare(text_token_ids, logical_q_lens=[2])
        with torch.inference_mode():
            text_embeds = lm.embed_tokens(text_ids)
            text_identity_ple = lm.get_per_layer_inputs(text_ids, text_embeds)
            text_projected_ple = lm.project_per_layer_inputs(text_embeds, text_identity_ple)
        assert torch.equal(text_prepared.inputs_embeds, text_embeds)
        assert torch.equal(text_prepared.per_layer_inputs, text_projected_ple)
        reference_text_hidden, _ = reference_forward(
            text_embeds, text_identity_ple, make_reference_prefix_cache(), prefix_len,
        )
        text_q2 = step(row, text_prepared.inputs_embeds, per_layer_inputs=text_prepared.per_layer_inputs)
        compare_hidden("TEXT_Q2", text_q2.tensor[:, :2, :], reference_text_hidden)
        compare_logits("TEXT_Q2_LAST", text_q2.tensor[:, 1:2, :], reference_text_hidden[:, 1:2, :])
        row = owner.rewind_session_async(
            replace(row, cache_position=prefix_len + 2, valid_history_tokens=prefix_len + 2),
            target_position=prefix_len,
        )

        # Q1 logical commit must match a one-token HF forward and ignore physical slot 1.
        token_id = text_token_ids[0]
        token_ids_1 = torch.tensor([[token_id]], dtype=torch.long, device=device)
        text_q1_prepared = adapter.prepare([token_id], logical_q_lens=[1])
        with torch.inference_mode():
            text_q1_embeds = lm.embed_tokens(token_ids_1)
            text_q1_identity_ple = lm.get_per_layer_inputs(token_ids_1, text_q1_embeds)
            text_q1_projected_ple = lm.project_per_layer_inputs(text_q1_embeds, text_q1_identity_ple)
        assert torch.equal(text_q1_prepared.inputs_embeds[:, :1, :], text_q1_embeds)
        assert torch.equal(text_q1_prepared.per_layer_inputs[:, :1, :, :], text_q1_projected_ple)
        assert int(text_q1_prepared.padded_input_ids[0, 1]) == int(lm.config.pad_token_id)
        reference_q1_hidden, _ = reference_forward(
            text_q1_embeds, text_q1_identity_ple, make_reference_prefix_cache(), prefix_len,
        )
        # Distinguish a logical-Q1 cache contract from backend arithmetic
        # changes caused solely by running Q=1 versus the listener's physical
        # Q2 GEMM. Both rows use the same official HF prefix/cache source.
        text_q1_padded_ids = torch.tensor(
            [[token_id, int(lm.config.pad_token_id)]], dtype=torch.long, device=device,
        )
        with torch.inference_mode():
            text_q1_padded_embeds = lm.embed_tokens(text_q1_padded_ids)
            text_q1_padded_identity = lm.get_per_layer_inputs(
                text_q1_padded_ids, text_q1_padded_embeds,
            )
        reference_q1_padded_hidden, _ = reference_forward(
            text_q1_padded_embeds,
            text_q1_padded_identity,
            make_reference_prefix_cache(),
            prefix_len,
        )
        compare_hidden(
            "HF_Q1_VS_HF_PHYSICAL_Q2_ROW0",
            reference_q1_hidden,
            reference_q1_padded_hidden[:, :1, :],
        )
        q1_row = replace(row, logical_q_len=1)
        text_q1 = step(q1_row, text_q1_prepared.inputs_embeds,
                       per_layer_inputs=text_q1_prepared.per_layer_inputs)
        q1_hidden = text_q1.tensor[:, :1, :]
        compare_hidden("TEXT_Q1", q1_hidden, reference_q1_hidden)
        compare_hidden(
            "NATIVE_Q1_VS_HF_PHYSICAL_Q2_ROW0",
            q1_hidden,
            reference_q1_padded_hidden[:, :1, :],
        )
        compare_logits("TEXT_Q1", q1_hidden, reference_q1_hidden)
        compare_logits("TEXT_Q1_PHYSICAL_Q2", q1_hidden, reference_q1_padded_hidden[:, :1, :])
        q1_logits = adapter.logits_from_final_norm_hidden(q1_hidden)
        assert q1_logits.ndim == 3 and q1_logits.shape[:2] == (1, 1)
        assert torch.isfinite(q1_logits).all().item()
        row = owner.rewind_session_async(
            replace(q1_row, cache_position=prefix_len + 1, valid_history_tokens=prefix_len + 1),
            target_position=prefix_len,
        )
        altered_embeds = text_q1_prepared.inputs_embeds.clone()
        altered_ple = text_q1_prepared.per_layer_inputs.clone()
        altered_embeds[:, 1, :].fill_(0.375)
        altered_ple[:, 1, :, :].fill_(-0.625)
        altered_q1 = step(q1_row, altered_embeds, per_layer_inputs=altered_ple)
        assert torch.equal(q1_hidden, altered_q1.tensor[:, :1, :]), (
            "Q1 slot 0 changed when only the non-logical physical slot was altered"
        )
        row = owner.rewind_session_async(
            replace(q1_row, cache_position=prefix_len + 1, valid_history_tokens=prefix_len + 1),
            target_position=prefix_len,
        )
        row = replace(row, logical_q_len=2)

        # Compare native audio-Q2 and following text-Q1 against the installed
        # Transformers Gemma4TextModel forward.
        audio_soft = torch.randn(
            (1, 2, 1536), device=device, dtype=torch.bfloat16,
        ).mul_(0.02).contiguous()
        reference_audio_cache = make_reference_prefix_cache()
        pad_ids = torch.full(
            (1, 2), int(lm.config.pad_token_id), dtype=torch.long, device=device,
        )
        with torch.inference_mode():
            pad_embeds = lm.embed_tokens(pad_ids)
            audio_identity_ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)
        reference_audio_hidden, reference_audio_cache = reference_forward(
            audio_soft, audio_identity_ple, reference_audio_cache, prefix_len,
        )
        first_audio = step(row, audio_soft)
        compare_hidden("AUDIO_Q2", first_audio.tensor[:, :2, :], reference_audio_hidden)
        compare_logits("AUDIO_Q2_LAST", first_audio.tensor[:, 1:2, :], reference_audio_hidden[:, 1:2, :])
        row = replace(
            row,
            frame_id=row.frame_id + 1,
            listener_state_version=row.listener_state_version + 1,
            sequence_position=row.sequence_position + 1,
            cache_position=prefix_len + 2,
            valid_history_tokens=prefix_len + 2,
        )
        reference_after_audio_hidden, reference_after_audio_cache = reference_forward(
            text_q1_embeds, text_q1_identity_ple, reference_audio_cache, prefix_len + 2,
        )
        audio_following_q1 = step(
            replace(row, logical_q_len=1), text_q1_prepared.inputs_embeds,
            per_layer_inputs=text_q1_prepared.per_layer_inputs,
        )
        audio_following_hidden = audio_following_q1.tensor[:, :1, :]
        compare_hidden("AUDIO_THEN_TEXT_Q1", audio_following_hidden, reference_after_audio_hidden)
        compare_logits("AUDIO_THEN_TEXT_Q1", audio_following_hidden, reference_after_audio_hidden)
        assert torch.isfinite(adapter.logits_from_final_norm_hidden(audio_following_hidden)).all().item()
        row = replace(row, cache_position=prefix_len + 3, valid_history_tokens=prefix_len + 3)

        # Q0 must not change any earlier K/V; queued Q2 replay advances only KV.
        before_ephemeral = owner.snapshot_session(row)
        ephemeral_audio = torch.randn(
            (1, 2, 1536), device=device, dtype=torch.bfloat16,
        ).mul_(0.02).contiguous()
        queued_row = replace(
            row, logical_q_len=0, ephemeral_reason="assistant_turn_open",
            queue_audio_for_next_turn=True,
        )
        ephemeral = step(queued_row, ephemeral_audio)
        assert ephemeral.kv_rollback_tokens_by_row == (2,)
        after_ephemeral_row = replace(
            row,
            frame_id=row.frame_id + 1,
            listener_state_version=row.listener_state_version + 1,
            sequence_position=row.sequence_position + 1,
        )
        after_ephemeral = owner.snapshot_session(after_ephemeral_row)
        assert after_ephemeral.state.current_position == before_ephemeral.state.current_position
        assert after_ephemeral.state.valid_kv_len == before_ephemeral.state.valid_kv_len
        assert before_ephemeral.kv_by_layer.keys() == after_ephemeral.kv_by_layer.keys()
        for layer in before_ephemeral.kv_by_layer:
            before_key, before_value = before_ephemeral.kv_by_layer[layer]
            after_key, after_value = after_ephemeral.kv_by_layer[layer]
            assert torch.equal(before_key, after_key), f"Q0 changed prior K for layer {layer}"
            assert torch.equal(before_value, after_value), f"Q0 changed prior V for layer {layer}"

        replay = step(replace(after_ephemeral_row, logical_q_len=2), ephemeral_audio, queued=True)
        assert replay.preserves_listener_cursors and replay.logical_q_lens == (2,)
        replay_row = replace(
            after_ephemeral_row,
            cache_position=prefix_len + 5,
            valid_history_tokens=prefix_len + 5,
        )
        replay_state = owner.snapshot_session(replay_row)
        assert replay_state.state.current_position == prefix_len + 5
        assert replay_state.state.expected_frame_id == after_ephemeral_row.frame_id
        assert replay_state.state.expected_state_version == after_ephemeral_row.listener_state_version
        assert replay_state.state.expected_sequence_position == after_ephemeral_row.sequence_position
        reference_replay_hidden, reference_replay_cache = reference_forward(
            ephemeral_audio,
            audio_identity_ple,
            reference_after_audio_cache,
            prefix_len + 3,
        )
        compare_hidden("QUEUED_AUDIO_REPLAY_Q2", replay.tensor[:, :2, :], reference_replay_hidden)

        # Clone the complete canonical state into a second slot of the same
        # owner, append one divergent token there, and prove that neither the
        # source cursor nor any source K/V changes. Then append a different
        # token to the canonical row and compare the private token's hidden
        # state with a fresh Transformers prefix replay using the same weights.
        source_before_fork = owner.snapshot_session(replay_row)
        fork_session_id = f"unified-kv-fork-{uuid.uuid4().hex}"
        fork_context_id = int(uuid.uuid4().int & ((1 << 63) - 1)) or 2
        fork_row = owner.fork_session_slot(
            replay_row,
            fork_session_id=fork_session_id,
            fork_native_context_id=fork_context_id,
        )
        fork_cleanup_row = fork_row
        fork_before_append = owner.snapshot_session(fork_row)
        assert fork_before_append.state.current_position == source_before_fork.state.current_position
        assert fork_before_append.state.valid_kv_len == source_before_fork.state.valid_kv_len
        assert fork_before_append.state.prefix_len == source_before_fork.state.prefix_len
        assert fork_row.native_context_id != replay_row.native_context_id
        assert fork_row.session_id != replay_row.session_id
        assert fork_before_append.state.expected_frame_id == source_before_fork.state.expected_frame_id
        assert fork_before_append.state.expected_state_version == source_before_fork.state.expected_state_version
        assert fork_before_append.state.expected_sequence_position == source_before_fork.state.expected_sequence_position
        assert fork_before_append.kv_by_layer.keys() == source_before_fork.kv_by_layer.keys()
        for layer in source_before_fork.kv_by_layer:
            source_key, source_value = source_before_fork.kv_by_layer[layer]
            fork_key, fork_value = fork_before_append.kv_by_layer[layer]
            assert torch.equal(source_key, fork_key), f"fork changed cloned K for layer {layer}"
            assert torch.equal(source_value, fork_value), f"fork changed cloned V for layer {layer}"

        private_token_id = int(tokenizer.encode("No", add_special_tokens=False)[0])
        canonical_token_id = int(tokenizer.encode("Sì", add_special_tokens=False)[0])
        assert private_token_id != canonical_token_id

        def prepare_single_token(token_id: int):
            prepared = adapter.prepare([token_id], logical_q_lens=[1])
            with torch.inference_mode():
                token_embeds = lm.embed_tokens(prepared.padded_input_ids)
                token_identity_ple = lm.get_per_layer_inputs(
                    prepared.padded_input_ids, token_embeds,
                )
                projected_ple = lm.project_per_layer_inputs(token_embeds, token_identity_ple)
            assert torch.equal(prepared.inputs_embeds, token_embeds)
            assert torch.equal(prepared.per_layer_inputs, projected_ple)
            return prepared, token_identity_ple

        private_prepared, private_identity_ple = prepare_single_token(private_token_id)
        private_output = step(
            replace(fork_row, logical_q_len=1),
            private_prepared.inputs_embeds,
            per_layer_inputs=private_prepared.per_layer_inputs,
        )
        fork_after_private = replace(
            fork_row,
            cache_position=fork_row.cache_position + 1,
            valid_history_tokens=fork_row.valid_history_tokens + 1,
            logical_q_len=1,
        )
        fork_cleanup_row = fork_after_private
        source_after_private = owner.snapshot_session(replay_row)
        assert source_after_private.state == source_before_fork.state
        assert source_after_private.kv_by_layer.keys() == source_before_fork.kv_by_layer.keys()
        for layer in source_before_fork.kv_by_layer:
            source_key, source_value = source_before_fork.kv_by_layer[layer]
            after_key, after_value = source_after_private.kv_by_layer[layer]
            assert torch.equal(source_key, after_key), f"private append changed source K for layer {layer}"
            assert torch.equal(source_value, after_value), f"private append changed source V for layer {layer}"

        private_snapshot = owner.snapshot_session(fork_after_private)
        reference_private_hidden, _ = reference_forward(
            private_prepared.inputs_embeds,
            private_identity_ple,
            reference_replay_cache,
            prefix_len + 5,
        )
        compare_hidden(
            "FORK_DIVERGENT_Q1",
            private_output.tensor[:, :1, :],
            reference_private_hidden[:, :1, :],
        )
        compare_logits(
            "FORK_DIVERGENT_Q1",
            private_output.tensor[:, :1, :],
            reference_private_hidden[:, :1, :],
        )
        assert parity_cosines["FORK_DIVERGENT_Q1"] >= 0.9997, (
            "same-owner fork append differs from fresh Transformers prefix replay: "
            f"cosine={parity_cosines['FORK_DIVERGENT_Q1']}"
        )
        assert logits_top1_matches["FORK_DIVERGENT_Q1"], (
            "same-owner fork append greedy token differs from fresh Transformers prefix replay"
        )

        canonical_prepared, canonical_identity_ple = prepare_single_token(canonical_token_id)
        canonical_output = step(
            replace(replay_row, logical_q_len=1),
            canonical_prepared.inputs_embeds,
            per_layer_inputs=canonical_prepared.per_layer_inputs,
        )
        canonical_after_append = replace(
            replay_row,
            cache_position=replay_row.cache_position + 1,
            valid_history_tokens=replay_row.valid_history_tokens + 1,
            logical_q_len=1,
        )
        canonical_snapshot = owner.snapshot_session(canonical_after_append)
        fork_after_canonical = owner.snapshot_session(fork_after_private)
        assert fork_after_canonical.state == private_snapshot.state
        divergent_layers = [
            layer for layer in private_snapshot.kv_by_layer
            if not (
                torch.equal(private_snapshot.kv_by_layer[layer][0], canonical_snapshot.kv_by_layer[layer][0])
                and torch.equal(private_snapshot.kv_by_layer[layer][1], canonical_snapshot.kv_by_layer[layer][1])
            )
        ]
        assert divergent_layers, "different private/canonical token appends left every K/V layer identical"
        assert not torch.equal(
            private_output.tensor[:, :1, :], canonical_output.tensor[:, :1, :],
        ), "different private/canonical token appends produced identical hidden states"
        owner.release_session_async(fork_cleanup_row)
        fork_cleanup_row = None

        assert first_audio.logical_q_lens == (2,)
        assert text_q2.logical_q_lens == (2,)
        assert text_q1.logical_q_lens == (1,)
        assert altered_q1.logical_q_lens == (1,)
        assert ephemeral.logical_q_lens == (0,)
        physical_q1_max, physical_q1_rms = parity_deltas[
            "NATIVE_Q1_VS_HF_PHYSICAL_Q2_ROW0"
        ]
        assert physical_q1_max <= 0.25 and physical_q1_rms <= 0.02, (
            "native logical-Q1 slot0 differs from official HF physical-Q2 [token,PAD] slot0: "
            f"max_abs={physical_q1_max}, rms={physical_q1_rms}"
        )
        for name in ("TEXT_Q1", "AUDIO_THEN_TEXT_Q1"):
            assert parity_cosines[name] >= 0.9997, (
                f"native/HF logical-Q1 cosine too low for {name}: {parity_cosines[name]}"
            )
        for name, matches in logits_top1_matches.items():
            assert matches, f"native/HF greedy token differs for {name}"
    finally:
        try:
            if fork_cleanup_row is not None and not owner._poisoned:
                owner.release_session_async(fork_cleanup_row)
        finally:
            owner.close()
