"""Opt-in end-to-end numerical parity from real PCM through native Control6.

Run on Mondaini with ``GEMMA4_RUN_LISTENER_PIPELINE_GPU=1``. The test keeps one
loaded Thinker, compares the standard per-session audio-tower Q4 and ABI3 LM
against the shared frontend/native LM, and refreshes the semantic prefix once
after a completed user/assistant exchange. The opt-in
``GEMMA4_LISTENER_NATIVE_SHADOW_REPLAY_PROTOTYPE=1`` branch compares text-prefix
import plus native M4 replay against the full-cache and ABI3 references.
"""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _derive_native_replay_base_cursor(
    *,
    replay_count: int,
    current_frame_id: int,
    listener_state_version: int,
    sequence_position: int,
    prefix_len: int,
    request_cache_position: int,
    historical_frame_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Derive and validate the native replay cursor without importing GPU code."""
    values = {
        "replay_count": replay_count,
        "current_frame_id": current_frame_id,
        "listener_state_version": listener_state_version,
        "sequence_position": sequence_position,
        "prefix_len": prefix_len,
        "request_cache_position": request_cache_position,
    }
    if any(type(value) is not int for value in values.values()):
        raise ValueError("native replay cursor fields must be integers")
    if replay_count < 0 or prefix_len < 0:
        raise ValueError("replay_count and prefix_len must be nonnegative")
    if current_frame_id < replay_count:
        raise ValueError("replay base frame_id would be negative")
    if listener_state_version < replay_count:
        raise ValueError("replay base listener_state_version would be negative")
    if sequence_position < replay_count:
        raise ValueError("replay base sequence_position would be negative")

    frame_ids = tuple(historical_frame_ids)
    if any(type(frame_id) is not int for frame_id in frame_ids):
        raise ValueError("historical frame IDs must be integers")
    expected_ids = tuple(range(current_frame_id - replay_count, current_frame_id))
    if frame_ids != expected_ids:
        raise ValueError("historical frame IDs must be contiguous and end at current_frame_id - 1")
    target_cache_position = prefix_len + 4 * replay_count
    if request_cache_position != target_cache_position:
        raise ValueError("request cache_position must equal prefix_len + 4 * replay_count")

    return {
        "frame_id": current_frame_id - replay_count,
        "listener_state_version": listener_state_version - replay_count,
        "sequence_position": sequence_position - replay_count,
        "cache_position": prefix_len,
        "target_frame_id": current_frame_id,
        "target_listener_state_version": listener_state_version,
        "target_sequence_position": sequence_position,
        "target_cache_position": target_cache_position,
        "historical_frame_ids": frame_ids,
    }


@pytest.mark.parametrize("replay_count", (0, 1, 64))
def test_native_replay_base_cursor_is_exact_and_contiguous(replay_count: int) -> None:
    prefix_len = 23
    current_frame_id = 64
    frame_ids = tuple(range(current_frame_id - replay_count, current_frame_id))
    for version_extra, sequence_extra in ((0, 0), (7, 11)):
        cursor = _derive_native_replay_base_cursor(
            replay_count=replay_count,
            current_frame_id=current_frame_id,
            listener_state_version=replay_count + version_extra,
            sequence_position=replay_count + sequence_extra,
            prefix_len=prefix_len,
            request_cache_position=prefix_len + 4 * replay_count,
            historical_frame_ids=frame_ids,
        )

        assert cursor["frame_id"] == current_frame_id - replay_count
        assert cursor["listener_state_version"] == version_extra
        assert cursor["sequence_position"] == sequence_extra
        assert cursor["cache_position"] == prefix_len
        assert cursor["target_frame_id"] == current_frame_id
        assert cursor["target_listener_state_version"] == replay_count + version_extra
        assert cursor["target_sequence_position"] == replay_count + sequence_extra
        assert cursor["target_cache_position"] == prefix_len + 4 * replay_count
        assert cursor["historical_frame_ids"] == frame_ids


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"current_frame_id": 63}, "frame_id"),
        ({"listener_state_version": 63}, "listener_state_version"),
        ({"sequence_position": 63}, "sequence_position"),
        ({"request_cache_position": 23 + 4 * 63}, "cache_position"),
        ({"historical_frame_ids": tuple(range(1, 65))}, "frame IDs"),
    ),
)
def test_native_replay_base_cursor_rejects_invalid_inputs(
    overrides: dict[str, Any], message: str,
) -> None:
    values: dict[str, Any] = {
        "replay_count": 64,
        "current_frame_id": 64,
        "listener_state_version": 64,
        "sequence_position": 64,
        "prefix_len": 23,
        "request_cache_position": 23 + 4 * 64,
        "historical_frame_ids": tuple(range(64)),
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        _derive_native_replay_base_cursor(**values)


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_LISTENER_PIPELINE_GPU") != "1",
    reason="requires the configured RTX 3090 runtime and cached Thinker",
)
def test_real_pcm_to_native_control6_scores_match_standard_rows() -> None:
    import torch

    import duplex_kernel.listener as listener_module
    from duplex_kernel.gemma_native_control6_batch_policy import GemmaNativeControl6BatchPolicy
    from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
    from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListener, GemmaListenerState
    from duplex_kernel.listener_microbatch import ListenerBatch as FrontendBatch
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor,
        ListenerBatchManifest,
        ListenerFrame,
        ListenerSessionState,
        RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_control6_prompt import (
        build_native_control6_prefix_ids,
        validate_native_control6_semantic_prefix_metadata,
    )
    from duplex_kernel.native_listener import NativeGemmaListenerLM, _load_compiled_listener_topology
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from training.duplex_qlora.native_control6 import (
        ACTIONS,
        load_native_control6_adapter,
        select_tied_output_rows,
    )
    from duplex_kernel.turn_policy import PolicyInput

    source_root = Path("/workspace/.tmp/listener_batch_source_overlay").resolve()
    imported_listener = Path(listener_module.__file__).resolve()
    assert source_root in imported_listener.parents, (
        f"expected persistent overlay source {source_root}, imported {imported_listener}"
    )
    assert hasattr(GemmaAudioTowerBackend, "run_audio_tower_q4")
    print(
        f"prototype_source={imported_listener} "
        f"run_audio_tower_q4={hasattr(GemmaAudioTowerBackend, 'run_audio_tower_q4')}",
        flush=True,
    )

    workspace = Path("/workspace")
    candidate = Path(os.environ.get(
        "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH",
        "/workspace/.tmp/listener_batch_source_overlay/native/gemma4_listener_microbatch/"
        "libgemma4_listener_microbatch.so",
    ))
    checkpoint = Path(os.environ.get(
        "DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH",
        "checkpoints/native_control6_semantic_prefix_mixed_candidate600.pt",
    ))
    if not checkpoint.is_absolute():
        checkpoint = workspace / checkpoint
    assert candidate.is_file() and checkpoint.is_file()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Both ABI3 reference rows and the shared batch API must come from one DSO;
    # otherwise native source/version drift can masquerade as a parity failure.
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = str(candidate)
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"
    device = torch.device("cuda:0")
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(device)

    thinker = load_thinker(str(workspace), device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)

    # Keep the ordinary GemmaListener frontend transaction, while capturing
    # the exact Q4 slice it passes into the ordinary ABI3 native listener.
    def capture_standard_q4(
        self: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        audio_soft: Any,
        state: GemmaListenerState,
        text_ids: tuple[int, ...],
    ) -> tuple[Any, Any]:
        del input_ids, attention_mask, text_ids
        first = (len(state.user_pcm_history) - 1) * 4
        current = audio_soft[:, first:first + 4, :]
        return current[:, 0, :], current

    backend._run_incremental_lm = types.MethodType(capture_standard_q4, backend)
    standard_listener = GemmaListener(backend=backend)
    frontend = ListenerMicrobatchAudioFrontend(
        backend.gemma_model,
        backend._extract_single_frame,
        device=device,
        dtype=torch.bfloat16,
        audio_tower_lock=backend._audio_tower_workspace_lock,
        audio_tower_row_runner=backend.run_audio_tower_q4,
    )
    stream = torch.cuda.Stream(device=device, priority=-1)

    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = validate_native_control6_semantic_prefix_metadata(checkpoint_payload)
    assert metadata["semantic_prefix_context_conditioned"] is True
    adapter, _report = load_native_control6_adapter(
        checkpoint,
        base_rows=select_tied_output_rows(thinker.model.lm_head.weight),
        device=device,
    )
    adapter.eval()
    checkpoint_path = str(checkpoint.resolve())
    batch_scorer = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path=checkpoint_path)
    single_policy = GemmaNativeControl6Policy(adapter, checkpoint_path=checkpoint_path)

    marker = (
        getattr(backend.processor, "audio_token", None)
        or getattr(thinker.tokenizer, "audio_token", None)
        or "<|audio|>"
    )
    context_a = (
        {"role": "user", "content": "Qual è il punto centrale della conversazione?"},
        {"role": "assistant", "content": "Sto ascoltando il contesto."},
    )
    context_b = context_a + (
        {"role": "user", "content": "Puoi chiarire il passaggio precedente?"},
        {"role": "assistant", "content": "Sì, aggiungo il dettaglio richiesto."},
    )
    prefix_ids_by_tick: list[tuple[int, ...]] = []
    for messages in (context_a, context_b):
        ids, length = build_native_control6_prefix_ids(
            thinker.tokenizer,
            instructions="Rispondi in modo conciso e resta attento alle interruzioni.",
            completed_messages=messages,
            audio_marker=str(marker),
            boa_token_id=backend.boa_token_id,
        )
        assert length == len(ids) and 1 <= length <= 291
        prefix_ids_by_tick.append(tuple(ids))

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

    def pcm(session_index: int, frame_id: int, channel: int) -> np.ndarray:
        seed = 910_003 + session_index * 10_007 + frame_id * 7919 + channel * 101
        return (np.random.default_rng(seed).standard_normal(1280) * 0.035).astype(np.float32)

    def make_cache(prefix_ids: tuple[int, ...], replay: Any | None = None) -> dict[str, Any]:
        with torch.cuda.stream(stream), torch.inference_mode():
            if replay is None:
                cache = backend.build_semantic_prefix_cache(prefix_ids)
            else:
                cache = backend.build_semantic_prefix_cache(
                    prefix_ids, replay_q4=(replay,), replay_frame_ids=(0,),
                )
        return cache

    def make_rows(
        states: list[ListenerSessionState],
        frames: list[ListenerFrame],
    ) -> tuple[BatchRowDescriptor, ...]:
        result = []
        for state, frame in zip(states, frames, strict=True):
            result.append(BatchRowDescriptor(
                session_id=state.session_id,
                session_generation=state.generation_id,
                state_epoch_id=state.state_epoch_id,
                frame_id=frame.frame_id,
                listener_state_version=state.state_version,
                sequence_position=state.sequence_position,
                cache_position=state.cache_position,
                valid_history_tokens=state.valid_history_tokens,
                native_context_id=int(state.native_context_id),
                input_frame_sha256=frame.pcm_sha256,
                reference_pcm_sha256=frame.reference_pcm_sha256,
                frame_ready_at_ns=frame.ready_at_ns,
                reference_frame_id=frame.reference_frame_id,
            ))
        return tuple(result)

    def make_manifest(
        rows: tuple[BatchRowDescriptor, ...],
        *,
        listener_stream: Any = stream,
        stream_priority: int = -1,
    ) -> ListenerBatchManifest:
        return ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=runtime,
            stream_identity=NativeGemmaListenerMicrobatch._stream_identity(listener_stream),
            stream_priority=stream_priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="coalesce_deadline" if len(rows) == 1 else "multiple_ready",
            rows=rows,
        )

    results: list[dict[str, Any]] = []
    requested_n = os.environ.get("GEMMA4_LISTENER_PIPELINE_N_VALUES", "1,2,4")
    batch_sizes = tuple(int(item.strip()) for item in requested_n.split(",") if item.strip())
    if not batch_sizes or any(size < 1 or size > 12 for size in batch_sizes):
        raise ValueError("GEMMA4_LISTENER_PIPELINE_N_VALUES must contain sizes in [1,12]")
    for n in batch_sizes:
        standard_states = [GemmaListenerState() for _ in range(n)]
        batch_states: list[ListenerSessionState] = []
        session_ids = [f"pipeline-{uuid.uuid4().hex[:12]}-{i}" for i in range(n)]
        for i, sid in enumerate(session_ids):
            batch_states.append(ListenerSessionState(
                session_id=sid,
                generation_id=1,
                state_epoch_id=f"epoch-{sid}",
                runtime_signature=runtime,
                next_frame_id=0,
                state_version=0,
                sequence_position=0,
                cache_position=len(prefix_ids_by_tick[0]),
                valid_history_tokens=len(prefix_ids_by_tick[0]),
                native_context_id=90_000 + n * 100 + i,
            ))

        reference_providers = [
            NativeGemmaListenerLM(
                lm, max_seq_len=16_384, stream_q_len=4, enable_cuda_graph=False,
                attention_backend=3, rmsnorm_backend=2,
            )
            for _ in range(n)
        ]
        batch_provider = NativeGemmaListenerLM(
            lm, max_seq_len=16_384, stream_q_len=4, enable_cuda_graph=False,
            attention_backend=3, rmsnorm_backend=2,
        )
        owner = NativeGemmaListenerMicrobatch(
            batch_provider,
            library_path=candidate,
            runtime_signature=runtime,
            listener_stream=stream,
            max_batch_size=1,
        )
        owner.ensure_capacity(n)
        old_rows: tuple[BatchRowDescriptor, ...] = ()
        q4_replay: list[torch.Tensor] = []
        try:
            for tick in (0, 1):
                prefix_ids = prefix_ids_by_tick[tick]
                if tick == 0:
                    cache_rows = [make_cache(prefix_ids) for _ in range(n)]
                    for provider, cache in zip(reference_providers, cache_rows, strict=True):
                        provider.import_cache_async(
                            cache["past_key_values"], current_position=cache["cache_position"],
                            stream=stream, producer_stream=stream,
                            prefix_len=cache["prefix_len"],
                            prefix_mask=cache["prefix_attention"],
                        )
                else:
                    cache_rows = [
                        make_cache(prefix_ids, replay=q4_replay[index])
                        for index in range(n)
                    ]
                    for row in old_rows:
                        owner.release_session_async(row)
                    for state, cache in zip(batch_states, cache_rows, strict=True):
                        state.cache_position = int(cache["cache_position"])
                        state.valid_history_tokens = int(cache["cache_position"])
                    for provider, cache in zip(reference_providers, cache_rows, strict=True):
                        provider.reset()
                        provider.import_cache_async(
                            cache["past_key_values"], current_position=cache["cache_position"],
                            stream=stream, producer_stream=stream,
                            prefix_len=cache["prefix_len"],
                            prefix_mask=cache["prefix_attention"],
                        )

                frames = []
                for row_index, state in enumerate(batch_states):
                    user_wave = pcm(row_index, tick, 0)
                    reference_wave = None if tick == 0 else pcm(row_index, 0, 1)
                    frames.append(ListenerFrame(
                        session_id=state.session_id,
                        generation_id=state.generation_id,
                        state_epoch_id=state.state_epoch_id,
                        frame_id=state.next_frame_id,
                        ready_at_ns=time.monotonic_ns(),
                        pcm=user_wave,
                        reference_pcm=reference_wave,
                        reference_frame_id=None if tick == 0 else 0,
                    ))
                frame_rows = make_rows(batch_states, frames)
                manifest = make_manifest(frame_rows)
                batch = FrontendBatch(manifest=manifest, frames=tuple(frames))

                standard_q4: list[torch.Tensor] = []
                for i, frame in enumerate(frames):
                    standard_output = standard_listener.step(
                        frame.pcm,
                        frame.reference_pcm,
                        standard_states[i],
                    )
                    tower = standard_output.audio_tower_hidden
                    assert isinstance(tower, torch.Tensor)
                    start = tick * 4
                    standard_q4.append(tower[start:start + 4].detach().clone())

                prepared = frontend.encode_batch(batch, batch_states)
                candidate_q4 = torch.stack(prepared.audio_soft_tokens, dim=0).contiguous()
                standard_q4_batch = torch.stack(standard_q4, dim=0).contiguous()
                q4_delta = (candidate_q4.float() - standard_q4_batch.float()).abs()
                q4_exact = torch.equal(candidate_q4, standard_q4_batch)
                print(
                    f"pipeline Q4 N={n} tick={tick} prefix_len={len(prefix_ids)} "
                    f"exact={q4_exact} max_abs={float(q4_delta.max().item()):.8g} "
                    f"rms={float(q4_delta.square().mean().sqrt().item()):.8g}",
                    flush=True,
                )
                assert q4_exact, (n, tick, float(q4_delta.max().item()))

                if tick == 1:
                    for row, cache in zip(frame_rows, cache_rows, strict=True):
                        owner.import_session(
                            row,
                            cache["past_key_values"],
                            cache["prefix_attention"],
                            prefix_len=int(cache["prefix_len"]),
                            producer_stream=stream,
                        )
                elif tick == 0:
                    for row, cache in zip(frame_rows, cache_rows, strict=True):
                        owner.import_session(
                            row,
                            cache["past_key_values"],
                            cache["prefix_attention"],
                            prefix_len=int(cache["prefix_len"]),
                            producer_stream=stream,
                        )

                reference_hidden: list[torch.Tensor] = []
                with torch.cuda.stream(stream), torch.inference_mode():
                    for provider, q4 in zip(reference_providers, standard_q4, strict=True):
                        hidden = provider.step_async(
                            q4.unsqueeze(0).contiguous(),
                            stream=stream,
                            producer_stream=stream,
                        ).detach().clone()
                        reference_hidden.append(hidden)
                    output = owner.step_async(
                        manifest, candidate_q4, producer_stream=stream,
                    )
                output.completion_event.synchronize()
                health = owner.verify_output_health(output)
                assert len(health) == n and all(item.status == "healthy" for item in health)
                expected_hidden = torch.cat(reference_hidden, dim=0)
                hidden_delta = (output.tensor.float() - expected_hidden.float()).abs()
                hidden_exact = torch.equal(output.tensor, expected_hidden)
                token_max = hidden_delta.amax(dim=(0, 2)).detach().cpu().tolist()
                print(
                    f"pipeline shapes N={n} tick={tick} actual={tuple(output.tensor.shape)} "
                    f"expected={tuple(expected_hidden.shape)} prefix_len={cache_rows[0]['prefix_len']} "
                    f"cache_position={cache_rows[0]['cache_position']} "
                    f"mask_sum={int(cache_rows[0]['prefix_attention'].sum().item())} "
                    f"token_max={token_max}",
                    flush=True,
                )
                snapshots = batch_scorer.score_raw_batch(output, manifest)
                batch_scores = torch.tensor(
                    [snapshot.raw_logits for snapshot in snapshots],
                    dtype=torch.float32,
                    device=device,
                )
                standard_scores = adapter(expected_hidden[:, 0, :]).float()
                score_delta = (batch_scores - standard_scores).abs()
                score_close = torch.allclose(batch_scores, standard_scores, atol=1e-5, rtol=1e-5)
                actions_match = True
                for i, (row, frame, snapshot) in enumerate(zip(frame_rows, frames, snapshots, strict=True)):
                    hidden = expected_hidden[i:i + 1, 0, :]
                    common = dict(
                        session_id=row.session_id,
                        frame_id=row.frame_id,
                        user_pcm=frame.pcm,
                        speaker_active=False,
                        generation_id=1,
                        native_control_hidden=hidden,
                        native_control_hidden_frame_id=row.frame_id,
                        native_control_hidden_at_ns=time.monotonic_ns(),
                        legal_actions=tuple(ACTIONS),
                        native_listener_generation=1,
                    )
                    standard_decision = single_policy.step(PolicyInput(**common))
                    batch_decision = single_policy.step(PolicyInput(
                        **common, native_control6_raw_scores=snapshot,
                    ))
                    actions_match &= standard_decision.action == batch_decision.action

                print(
                    f"pipeline LM/Control6 N={n} tick={tick} hidden_exact={hidden_exact} "
                    f"hidden_max_abs={float(hidden_delta.max().item()):.8g} "
                    f"scores_max_abs={float(score_delta.max().item()):.8g} "
                    f"scores_rms={float(score_delta.square().mean().sqrt().item()):.8g} "
                    f"actions_match={actions_match}",
                    flush=True,
                )
                assert hidden_exact, (n, tick, float(hidden_delta.max().item()))
                assert score_close and actions_match, (
                    n, tick, float(score_delta.max().item()), actions_match,
                )
                results.append({
                    "n": n,
                    "tick": tick,
                    "prefix_len": len(prefix_ids),
                    "q4_exact": q4_exact,
                    "hidden_exact": hidden_exact,
                    "score_max_abs": float(score_delta.max().item()),
                    "actions_match": actions_match,
                })

                prepared.commit()
                q4_replay = [q4.detach().clone() for q4 in standard_q4]
                old_rows = tuple(replace(
                    row,
                    frame_id=row.frame_id + 1,
                    listener_state_version=row.listener_state_version + 1,
                    sequence_position=row.sequence_position + 1,
                    cache_position=row.cache_position + 4,
                    valid_history_tokens=row.valid_history_tokens + 4,
                ) for row in frame_rows)
                for state in batch_states:
                    state.next_frame_id += 1
                    state.state_version += 1
                    state.sequence_position += 1
                    state.cache_position += 4
                    state.valid_history_tokens += 4
        finally:
            owner.close()
            for provider in reference_providers:
                provider.close()
        torch.cuda.synchronize(device)

    assert len(results) == 2 * len(batch_sizes)

    if os.environ.get("GEMMA4_LISTENER_NATIVE_REPLAY_EXPERIMENT") == "1":
        prefix_ids = prefix_ids_by_tick[1]
        observed_q4: list[Any] = []
        history_state = GemmaListenerState()
        for frame_id in range(17):
            output = standard_listener.step(pcm(17, frame_id, 0), None, history_state)
            tower = output.audio_tower_hidden
            assert isinstance(tower, torch.Tensor)
            frame_start = (len(history_state.user_pcm_history) - 1) * 4
            observed_q4.append(tower[frame_start:frame_start + 4].detach().clone())
        torch.cuda.synchronize(device)
        if len(observed_q4) < 17 or len(prefix_ids) != 169:
            raise RuntimeError("native replay probe requires L=169, R=16, and one next Q4")
        replay_rows = tuple(observed_q4[:16])
        next_q4 = observed_q4[16].unsqueeze(0).contiguous()
        replay_ids = tuple(range(16))

        torch_replay_started = time.perf_counter()
        with torch.cuda.stream(stream), torch.inference_mode():
            torch_reference = backend.build_semantic_prefix_cache(
                prefix_ids,
                replay_q4=replay_rows,
                replay_frame_ids=replay_ids,
            )
        torch.cuda.synchronize(device)
        torch_replay_wall_ms = (time.perf_counter() - torch_replay_started) * 1000.0
        reference_cache = torch_reference["past_key_values"]
        ref_position = int(torch_reference["cache_position"])

        native_prefix_started = time.perf_counter()
        with torch.cuda.stream(stream), torch.inference_mode():
            native_prefix = backend.build_semantic_prefix_cache(prefix_ids)
        torch.cuda.synchronize(device)
        native_prefix_wall_ms = (time.perf_counter() - native_prefix_started) * 1000.0

        native_replay = NativeGemmaListenerLM(
            lm, max_seq_len=16_384, stream_q_len=4, enable_cuda_graph=False,
            attention_backend=3, rmsnorm_backend=2,
        )
        try:
            with torch.cuda.stream(stream):
                native_replay.import_cache_async(
                    native_prefix["past_key_values"],
                    current_position=len(prefix_ids),
                    stream=stream,
                    producer_stream=stream,
                    prefix_len=len(prefix_ids),
                    prefix_mask=native_prefix["prefix_attention"],
                )
                replay_start_event = torch.cuda.Event(enable_timing=True)
                replay_end_event = torch.cuda.Event(enable_timing=True)
                replay_start_event.record(stream)
                native_submit_started = time.perf_counter()
                native_replay_wall_started = native_submit_started
                last_native_output = None
                for q4 in replay_rows:
                    last_native_output = native_replay.step_async(
                        q4.unsqueeze(0).contiguous(),
                        stream=stream,
                        producer_stream=stream,
                    )
                native_submit_ms = (time.perf_counter() - native_submit_started) * 1000.0
                replay_end_event.record(stream)
            if last_native_output is None:
                raise RuntimeError("native history replay produced no output")
            stream.synchronize()
            native_replay_wall_ms = (time.perf_counter() - native_replay_wall_started) * 1000.0
            native_history_hidden = last_native_output.detach().clone()
            native_position = native_replay.current_position
            if native_position != ref_position:
                raise AssertionError(
                    f"native history cursor {native_position} != Torch cursor {ref_position}"
                )

            first_cache_diff = None
            cache_max_abs = 0.0
            cache_rms_sq = 0.0
            cache_numel = 0
            compared_layers = 0
            for layer_index, model_layer in enumerate(lm.layers):
                attention = model_layer.self_attn
                if bool(getattr(attention, "is_kv_shared_layer", False)):
                    continue
                expected_pair = getattr(reference_cache, "shared_layers", {}).get(layer_index)
                if expected_pair is None:
                    cache_layer = reference_cache.layers[layer_index]
                    expected_pair = (cache_layer.keys, cache_layer.values)
                native_key, native_value = native_replay.get_cached_kv(layer_index)
                for name, expected_tensor, actual_tensor in (
                    ("K", expected_pair[0], native_key),
                    ("V", expected_pair[1], native_value),
                ):
                    if tuple(expected_tensor.shape) != tuple(actual_tensor.shape):
                        first_cache_diff = first_cache_diff or (
                            f"layer{layer_index}.{name}:shape "
                            f"{tuple(expected_tensor.shape)} != {tuple(actual_tensor.shape)}"
                        )
                        continue
                    delta = (expected_tensor.float() - actual_tensor.float()).abs()
                    current_max = float(delta.max().item()) if delta.numel() else 0.0
                    cache_max_abs = max(cache_max_abs, current_max)
                    cache_rms_sq += float(delta.square().sum().item())
                    cache_numel += int(delta.numel())
                    if current_max and first_cache_diff is None:
                        per_position = delta.amax(dim=(0, 1, 3))
                        differing = torch.nonzero(per_position > 0, as_tuple=False)
                        position_index = int(differing[0].item()) if differing.numel() else -1
                        if position_index < len(prefix_ids):
                            location = f"prefix_token={position_index}"
                        else:
                            replay_offset = position_index - len(prefix_ids)
                            location = f"replay_frame={replay_offset // 4}_q4_row={replay_offset % 4}"
                        first_cache_diff = f"layer{layer_index}.{name}@{location} max_abs={current_max:.8g}"
                compared_layers += 1
            cache_rms = (cache_rms_sq / max(1, cache_numel)) ** 0.5

            position = ref_position
            frame_audio = next_q4.to(dtype=lm.embed_tokens.weight.dtype)
            positions = torch.arange(
                position, position + 4, dtype=torch.long, device=device,
            ).view(1, -1)
            attention = torch.ones((1, position + 4), dtype=torch.long, device=device)
            masks = create_masks_for_generate(
                lm.config,
                frame_audio,
                attention,
                past_key_values=reference_cache,
                position_ids=positions,
            )
            with torch.cuda.stream(stream), torch.inference_mode():
                expected_next_hidden = shape_lm.forward(
                    inputs_embeds=frame_audio,
                    per_layer_inputs=backend._frame_pad_per_layer_inputs(lm),
                    attention_mask=masks,
                    position_ids=positions,
                    past_key_values=reference_cache,
                    shared_kv_states=getattr(reference_cache, "shared_layers", None),
                ).last_hidden_state.detach().clone()
                native_next_hidden = native_replay.step_async(
                    next_q4,
                    stream=stream,
                    producer_stream=stream,
                ).detach().clone()
            stream.synchronize()

            hidden_delta = (expected_next_hidden.float() - native_next_hidden.float()).abs()
            hidden_max_abs = float(hidden_delta.max().item()) if hidden_delta.numel() else 0.0
            expected_scores = adapter(expected_next_hidden[:, 0, :]).float()
            native_scores = adapter(native_next_hidden[:, 0, :]).float()
            score_delta = (expected_scores - native_scores).abs()
            score_max_abs = float(score_delta.max().item()) if score_delta.numel() else 0.0
            score_rms = float(score_delta.square().mean().sqrt().item())
            probe_common = dict(
                frame_id=30_016,
                user_pcm=pcm(17, 16, 0),
                speaker_active=False,
                generation_id=1,
                native_control_hidden_frame_id=30_016,
                native_control_hidden_at_ns=time.monotonic_ns(),
                legal_actions=tuple(ACTIONS),
                native_listener_generation=1,
            )
            torch_decision = single_policy.step(PolicyInput(
                session_id="native-replay-torch-reference",
                native_control_hidden=expected_next_hidden[:, 0, :],
                **probe_common,
            ))
            native_decision = single_policy.step(PolicyInput(
                session_id="native-replay-native-candidate",
                native_control_hidden=native_next_hidden[:, 0, :],
                **probe_common,
            ))
            hidden_exact = torch.equal(expected_next_hidden, native_next_hidden)
            score_exact = torch.equal(expected_scores, native_scores)
            action_match = torch_decision.action == native_decision.action
            print(
                f"native history replay R=16 prefix_len={len(prefix_ids)} "
                f"torch_replay_wall_ms={torch_replay_wall_ms:.3f} "
                f"native_prefix_build_wall_ms={native_prefix_wall_ms:.3f} "
                f"native_replay_submit_ms={native_submit_ms:.3f} "
                f"native_replay_cuda_event_ms={replay_start_event.elapsed_time(replay_end_event):.3f} "
                f"native_replay_wall_ms={native_replay_wall_ms:.3f} "
                f"native_cache_position={native_position} torch_cache_position={ref_position} "
                f"compared_kv_layers={compared_layers} first_cache_diff={first_cache_diff} "
                f"cache_max_abs={cache_max_abs:.8g} cache_rms={cache_rms:.8g} "
                f"next_hidden_exact={hidden_exact} next_hidden_max_abs={hidden_max_abs:.8g} "
                f"scores_exact={score_exact} scores_max_abs={score_max_abs:.8g} "
                f"scores_rms={score_rms:.8g} actions_match={action_match} "
                f"torch_action={torch_decision.action.value} native_action={native_decision.action.value}",
                flush=True,
            )
            if first_cache_diff is not None or not hidden_exact or not score_exact or not action_match:
                raise AssertionError("native historical replay differs from the Torch-sequential reference")
        finally:
            native_replay.close()

    if os.environ.get("GEMMA4_LISTENER_NATIVE_SHADOW_REPLAY_PROTOTYPE") == "1":
        replay_count = 64
        prefix_ids = prefix_ids_by_tick[1]
        prefix_len = len(prefix_ids)
        replay_session_id = f"native-shadow-replay-{uuid.uuid4().hex[:12]}"
        standard_session_id = f"native-standard-replay-{uuid.uuid4().hex[:12]}"

        # Capture actual tower Q4 rows from the ordinary listener. The 65th
        # frame is reserved for the next-step comparison after the 64-row replay.
        replay_q4: list[Any] = []
        replay_pcm: list[np.ndarray] = []
        replay_frame_ids: list[int] = []
        history_state = GemmaListenerState()
        capture_cursor = ListenerSessionState(
            session_id=replay_session_id,
            generation_id=1,
            state_epoch_id=f"epoch-{replay_session_id}",
            runtime_signature=runtime,
            next_frame_id=0,
            state_version=0,
            sequence_position=0,
            cache_position=prefix_len,
            valid_history_tokens=prefix_len,
            native_context_id=97_064,
        )
        for frame_id in range(replay_count + 1):
            assert capture_cursor.next_frame_id == frame_id
            frame_pcm = pcm(37, frame_id, 0)
            output = standard_listener.step(frame_pcm, None, history_state)
            tower = output.audio_tower_hidden
            assert isinstance(tower, torch.Tensor)
            frame_start = (len(history_state.user_pcm_history) - 1) * 4
            q4 = tower[frame_start:frame_start + 4].detach().clone()
            assert tuple(q4.shape) == (4, int(getattr(backend, "hidden_size", q4.shape[-1])))
            replay_q4.append(q4)
            replay_pcm.append(frame_pcm)
            replay_frame_ids.append(frame_id)
            if frame_id < replay_count:
                capture_cursor.next_frame_id += 1
                capture_cursor.state_version += 1
                capture_cursor.sequence_position += 1
                capture_cursor.cache_position += 4
                capture_cursor.valid_history_tokens += 4
        torch.cuda.synchronize(device)
        assert len(replay_q4) == replay_count + 1

        historical_frame_ids = tuple(replay_frame_ids[:replay_count])
        current_frame_id = replay_frame_ids[replay_count]
        assert capture_cursor.next_frame_id == current_frame_id
        request_listener_state_version = capture_cursor.state_version
        stage_sequence_position = capture_cursor.sequence_position
        request_cache_position = capture_cursor.cache_position
        assert request_cache_position == prefix_len + 4 * replay_count
        replay_cursor = _derive_native_replay_base_cursor(
            replay_count=replay_count,
            current_frame_id=current_frame_id,
            listener_state_version=request_listener_state_version,
            sequence_position=stage_sequence_position,
            prefix_len=prefix_len,
            request_cache_position=request_cache_position,
            historical_frame_ids=historical_frame_ids,
        )

        # The backend's ordinary BOAv2 builder performs this short text-only
        # prefix prefill. Fence it before either native owner imports the same
        # DynamicCache and mask; the historical native steps do not hold a model lock.
        torch.cuda.synchronize(device)
        prefix_started = time.perf_counter()
        with torch.cuda.stream(stream), torch.inference_mode():
            text_prefix = backend.build_semantic_prefix_cache(prefix_ids)
        stream.synchronize()
        text_prefix_build_wall_ms = (time.perf_counter() - prefix_started) * 1000.0
        assert int(text_prefix["prefix_len"]) == prefix_len
        assert int(text_prefix["cache_position"]) == prefix_len
        assert int(text_prefix["past_key_values"].get_seq_length(0)) == prefix_len

        shadow_stream = torch.cuda.Stream(device=device, priority=0)
        standard_replay: Any | None = None
        shadow_provider: Any | None = None
        shadow_owner: Any | None = None
        try:
            standard_replay = NativeGemmaListenerLM(
                lm, max_seq_len=16_384, stream_q_len=4, enable_cuda_graph=False,
                attention_backend=3, rmsnorm_backend=2,
            )
            shadow_provider = NativeGemmaListenerLM(
                lm, max_seq_len=16_384, stream_q_len=4, enable_cuda_graph=False,
                attention_backend=3, rmsnorm_backend=2,
            )
            replay_state = ListenerSessionState(
                session_id=replay_session_id,
                generation_id=1,
                state_epoch_id=f"epoch-{replay_session_id}",
                runtime_signature=runtime,
                next_frame_id=int(replay_cursor["frame_id"]),
                state_version=int(replay_cursor["listener_state_version"]),
                sequence_position=int(replay_cursor["sequence_position"]),
                cache_position=int(replay_cursor["cache_position"]),
                valid_history_tokens=prefix_len,
                native_context_id=97_064,
            )
            base_frame = ListenerFrame(
                session_id=replay_session_id,
                generation_id=1,
                state_epoch_id=replay_state.state_epoch_id,
                frame_id=int(replay_cursor["frame_id"]),
                ready_at_ns=time.monotonic_ns(),
                pcm=replay_pcm[0],
            )
            base_row = make_rows([replay_state], [base_frame])[0]
            shadow_owner = NativeGemmaListenerMicrobatch(
                shadow_provider,
                library_path=candidate,
                runtime_signature=runtime,
                listener_stream=shadow_stream,
                max_batch_size=1,
                stream_priority=0,
            )
            shadow_owner.ensure_capacity(1)

            torch.cuda.synchronize(device)
            standard_import_started = time.perf_counter()
            with torch.cuda.stream(stream):
                standard_replay.import_cache_async(
                    text_prefix["past_key_values"],
                    current_position=prefix_len,
                    stream=stream,
                    producer_stream=stream,
                    prefix_len=prefix_len,
                    prefix_mask=text_prefix["prefix_attention"],
                )
            stream.synchronize()
            standard_native_import_wall_ms = (
                time.perf_counter() - standard_import_started
            ) * 1000.0

            shadow_import_started = time.perf_counter()
            shadow_owner.import_session(
                base_row,
                text_prefix["past_key_values"],
                text_prefix["prefix_attention"],
                prefix_len=prefix_len,
                producer_stream=stream,
            )
            shadow_stream.synchronize()
            shadow_native_import_wall_ms = (
                time.perf_counter() - shadow_import_started
            ) * 1000.0

            standard_hidden_rows: list[Any] = []
            torch.cuda.synchronize(device)
            standard_replay_started = time.perf_counter()
            with torch.cuda.stream(stream), torch.inference_mode():
                for q4 in replay_q4[:replay_count]:
                    standard_hidden_rows.append(standard_replay.step_async(
                        q4.unsqueeze(0).contiguous(),
                        stream=stream,
                        producer_stream=stream,
                    ).detach().clone())
            stream.synchronize()
            standard_replay_64_wall_ms = (
                time.perf_counter() - standard_replay_started
            ) * 1000.0

            shadow_hidden_rows: list[Any] = []
            shadow_replay_rows: list[BatchRowDescriptor] = []
            torch.cuda.synchronize(device)
            shadow_replay_started = time.perf_counter()
            for replay_index in range(replay_count):
                frame = ListenerFrame(
                    session_id=replay_session_id,
                    generation_id=replay_state.generation_id,
                    state_epoch_id=replay_state.state_epoch_id,
                    frame_id=replay_frame_ids[replay_index],
                    ready_at_ns=time.monotonic_ns(),
                    pcm=replay_pcm[replay_index],
                )
                row = make_rows([replay_state], [frame])[0]
                manifest = make_manifest(
                    (row,), listener_stream=shadow_stream, stream_priority=0,
                )
                with torch.cuda.stream(shadow_stream), torch.inference_mode():
                    shadow_output = shadow_owner.step_async(
                        manifest,
                        replay_q4[replay_index].unsqueeze(0).contiguous(),
                        producer_stream=stream,
                    )
                shadow_output.completion_event.synchronize()
                with torch.cuda.stream(shadow_stream), torch.inference_mode():
                    shadow_hidden_rows.append(shadow_output.tensor.detach().clone())
                shadow_stream.synchronize()
                shadow_replay_rows.append(row)
                replay_state.next_frame_id += 1
                replay_state.state_version += 1
                replay_state.sequence_position += 1
                replay_state.cache_position += 4
                replay_state.valid_history_tokens += 4
            shadow_replay_64_wall_ms = (
                time.perf_counter() - shadow_replay_started
            ) * 1000.0

            assert replay_state.next_frame_id == replay_cursor["target_frame_id"]
            assert replay_state.state_version == replay_cursor["target_listener_state_version"]
            assert replay_state.sequence_position == replay_cursor["target_sequence_position"]
            assert replay_state.cache_position == replay_cursor["target_cache_position"]
            assert replay_state.valid_history_tokens == replay_cursor["target_cache_position"]
            assert standard_replay.current_position == replay_cursor["target_cache_position"]

            native_next_frame = ListenerFrame(
                session_id=replay_session_id,
                generation_id=replay_state.generation_id,
                state_epoch_id=replay_state.state_epoch_id,
                frame_id=current_frame_id,
                ready_at_ns=time.monotonic_ns(),
                pcm=replay_pcm[replay_count],
            )
            native_next_row = make_rows([replay_state], [native_next_frame])[0]
            assert native_next_row.frame_id == replay_cursor["target_frame_id"]
            assert native_next_row.listener_state_version == replay_cursor["target_listener_state_version"]
            assert native_next_row.sequence_position == replay_cursor["target_sequence_position"]
            assert native_next_row.cache_position == replay_cursor["target_cache_position"]

            standard_hidden = torch.cat(standard_hidden_rows, dim=0)
            shadow_hidden = torch.cat(shadow_hidden_rows, dim=0)
            historical_hidden_delta = (standard_hidden.float() - shadow_hidden.float()).abs()
            historical_hidden_exact = torch.equal(standard_hidden, shadow_hidden)
            standard_scores = adapter(standard_hidden[:, 0, :]).float()
            shadow_scores = adapter(shadow_hidden[:, 0, :]).float()
            historical_score_delta = (standard_scores - shadow_scores).abs()
            historical_scores_exact = torch.equal(standard_scores, shadow_scores)
            history_actions_match = True
            for replay_index in range(replay_count):
                frame_id = replay_frame_ids[replay_index]
                common = dict(
                    frame_id=frame_id,
                    user_pcm=replay_pcm[replay_index],
                    speaker_active=False,
                    generation_id=1,
                    native_control_hidden_frame_id=frame_id,
                    native_control_hidden_at_ns=time.monotonic_ns(),
                    legal_actions=tuple(ACTIONS),
                    native_listener_generation=1,
                )
                standard_decision = single_policy.step(PolicyInput(
                    session_id=standard_session_id,
                    native_control_hidden=standard_hidden[replay_index:replay_index + 1, 0, :],
                    **common,
                ))
                shadow_decision = single_policy.step(PolicyInput(
                    session_id=replay_session_id,
                    native_control_hidden=shadow_hidden[replay_index:replay_index + 1, 0, :],
                    **common,
                ))
                history_actions_match &= standard_decision.action == shadow_decision.action

            kv_exact = True
            kv_first_difference: str | None = None
            kv_max_abs = 0.0
            kv_tensors_compared = 0
            kv_unique_layers = 0
            # The diagnostic snapshot API validates a row at the current
            # cursor, so use the next frame's target row after all R replays.
            shadow_snapshot = shadow_owner.snapshot_session(native_next_row)
            for layer_index, model_layer in enumerate(lm.layers):
                if bool(getattr(model_layer.self_attn, "is_kv_shared_layer", False)):
                    continue
                kv_unique_layers += 1
                standard_key, standard_value = standard_replay.get_cached_kv(layer_index)
                shadow_key, shadow_value = shadow_snapshot.kv_by_layer[layer_index]
                for name, standard_tensor, shadow_tensor in (
                    ("K", standard_key.cpu(), shadow_key),
                    ("V", standard_value.cpu(), shadow_value),
                ):
                    kv_tensors_compared += 1
                    assert int(standard_tensor.shape[-2]) == replay_cursor["target_cache_position"]
                    assert int(shadow_tensor.shape[-2]) == replay_cursor["target_cache_position"]
                    if tuple(standard_tensor.shape) != tuple(shadow_tensor.shape):
                        kv_exact = False
                        kv_first_difference = kv_first_difference or (
                            f"layer{layer_index}.{name}:shape "
                            f"{tuple(standard_tensor.shape)} != {tuple(shadow_tensor.shape)}"
                        )
                        continue
                    delta = (standard_tensor.float() - shadow_tensor.float()).abs()
                    current_max = float(delta.max().item()) if delta.numel() else 0.0
                    kv_max_abs = max(kv_max_abs, current_max)
                    if current_max != 0.0:
                        kv_exact = False
                        kv_first_difference = kv_first_difference or f"layer{layer_index}.{name}"
            assert kv_unique_layers > 0
            assert kv_tensors_compared == 2 * kv_unique_layers

            with torch.cuda.stream(stream), torch.inference_mode():
                standard_next_hidden = standard_replay.step_async(
                    replay_q4[replay_count].unsqueeze(0).contiguous(),
                    stream=stream,
                    producer_stream=stream,
                ).detach().clone()
            stream.synchronize()
            native_next_manifest = make_manifest(
                (native_next_row,), listener_stream=shadow_stream, stream_priority=0,
            )
            with torch.cuda.stream(shadow_stream), torch.inference_mode():
                shadow_next_output = shadow_owner.step_async(
                    native_next_manifest,
                    replay_q4[replay_count].unsqueeze(0).contiguous(),
                    producer_stream=stream,
                )
            shadow_next_output.completion_event.synchronize()
            with torch.cuda.stream(shadow_stream), torch.inference_mode():
                shadow_next_hidden = shadow_next_output.tensor.detach().clone()
            shadow_stream.synchronize()
            standard_next_scores = adapter(standard_next_hidden[:, 0, :]).float()
            shadow_next_scores = adapter(shadow_next_hidden[:, 0, :]).float()
            next_hidden_exact = torch.equal(standard_next_hidden, shadow_next_hidden)
            next_scores_exact = torch.equal(standard_next_scores, shadow_next_scores)
            next_hidden_max_abs = float(
                (standard_next_hidden.float() - shadow_next_hidden.float()).abs().max().item()
            )
            next_score_max_abs = float(
                (standard_next_scores - shadow_next_scores).abs().max().item()
            )
            next_policy_common = dict(
                frame_id=current_frame_id,
                user_pcm=replay_pcm[replay_count],
                speaker_active=False,
                generation_id=1,
                native_control_hidden_frame_id=current_frame_id,
                native_control_hidden_at_ns=time.monotonic_ns(),
                legal_actions=tuple(ACTIONS),
                native_listener_generation=1,
            )
            standard_next_decision = single_policy.step(PolicyInput(
                session_id=f"{standard_session_id}-next",
                native_control_hidden=standard_next_hidden[:, 0, :],
                **next_policy_common,
            ))
            shadow_next_decision = single_policy.step(PolicyInput(
                session_id=f"{replay_session_id}-next",
                native_control_hidden=shadow_next_hidden[:, 0, :],
                **next_policy_common,
            ))
            next_actions_match = standard_next_decision.action == shadow_next_decision.action

            # Diagnostic only: reconstruct the historical cache with the old
            # sequential text-prefix + Q4 teacher builder, then run the next
            # real Q4 through the ordinary ABI3 NativeGemmaListenerLM.
            teacher_build_started = time.perf_counter()
            with torch.cuda.stream(stream), torch.inference_mode():
                teacher_cache = backend.build_semantic_prefix_cache(
                    prefix_ids,
                    replay_q4=tuple(replay_q4[:replay_count]),
                    replay_frame_ids=historical_frame_ids,
                )
            stream.synchronize()
            teacher_cache_build_wall_ms = (
                time.perf_counter() - teacher_build_started
            ) * 1000.0
            teacher_cache_position = int(teacher_cache["cache_position"])
            assert teacher_cache_position == replay_cursor["target_cache_position"]

            teacher_import_started = time.perf_counter()
            standard_replay.reset()
            with torch.cuda.stream(stream):
                standard_replay.import_cache_async(
                    teacher_cache["past_key_values"],
                    current_position=teacher_cache_position,
                    stream=stream,
                    producer_stream=stream,
                    prefix_len=prefix_len,
                    prefix_mask=teacher_cache["prefix_attention"],
                )
            stream.synchronize()
            teacher_native_import_wall_ms = (
                time.perf_counter() - teacher_import_started
            ) * 1000.0
            teacher_next_started = time.perf_counter()
            with torch.cuda.stream(stream), torch.inference_mode():
                teacher_next_hidden = standard_replay.step_async(
                    replay_q4[replay_count].unsqueeze(0).contiguous(),
                    stream=stream,
                    producer_stream=stream,
                ).detach().clone()
            stream.synchronize()
            teacher_next_wall_ms = (time.perf_counter() - teacher_next_started) * 1000.0
            teacher_next_scores = adapter(teacher_next_hidden[:, 0, :]).float()
            teacher_decision = single_policy.step(PolicyInput(
                session_id=f"{replay_session_id}-teacher-next",
                native_control_hidden=teacher_next_hidden[:, 0, :],
                **next_policy_common,
            ))
            teacher_hidden_vs_standard = (
                teacher_next_hidden.float() - standard_next_hidden.float()
            ).abs()
            teacher_hidden_vs_shadow = (
                teacher_next_hidden.float() - shadow_next_hidden.float()
            ).abs()
            teacher_scores_vs_standard = (teacher_next_scores - standard_next_scores).abs()
            teacher_scores_vs_shadow = (teacher_next_scores - shadow_next_scores).abs()

            proposal_ms = text_prefix_build_wall_ms + shadow_replay_64_wall_ms
            print(
                f"native shadow historical replay R={replay_count} prefix_len={prefix_len} "
                f"text_prefix_build_wall_ms={text_prefix_build_wall_ms:.3f} "
                f"standard_native_import_wall_ms={standard_native_import_wall_ms:.3f} "
                f"shadow_native_import_wall_ms={shadow_native_import_wall_ms:.3f} "
                f"standard_replay_64_wall_ms={standard_replay_64_wall_ms:.3f} "
                f"shadow_replay_64_wall_ms={shadow_replay_64_wall_ms:.3f} "
                f"shadow_prefix_plus_64_ms={proposal_ms:.3f} "
                f"proposal_under_5120ms={proposal_ms < 5120.0} "
                f"historical_hidden_exact={historical_hidden_exact} "
                f"historical_hidden_max_abs={float(historical_hidden_delta.max().item()):.8g} "
                f"historical_raw6_exact={historical_scores_exact} "
                f"historical_raw6_max_abs={float(historical_score_delta.max().item()):.8g} "
                f"historical_actions_match={history_actions_match} "
                f"native_kv_layers_compared={kv_unique_layers} "
                f"native_kv_tensors_compared={kv_tensors_compared} kv_exact={kv_exact} "
                f"kv_first_difference={kv_first_difference} kv_max_abs={kv_max_abs:.8g} "
                f"cursor_base={replay_cursor['frame_id']}/{replay_cursor['listener_state_version']}/"
                f"{replay_cursor['sequence_position']}/{replay_cursor['cache_position']} "
                f"cursor_target={replay_state.next_frame_id}/{replay_state.state_version}/"
                f"{replay_state.sequence_position}/{replay_state.cache_position} "
                f"next_hidden_exact={next_hidden_exact} next_hidden_max_abs={next_hidden_max_abs:.8g} "
                f"next_raw6_exact={next_scores_exact} next_raw6_max_abs={next_score_max_abs:.8g} "
                f"next_actions_match={next_actions_match} "
                f"standard_action={standard_next_decision.action.value} "
                f"shadow_action={shadow_next_decision.action.value} "
                f"teacher_cache_build_wall_ms={teacher_cache_build_wall_ms:.3f} "
                f"teacher_native_import_wall_ms={teacher_native_import_wall_ms:.3f} "
                f"teacher_next_wall_ms={teacher_next_wall_ms:.3f} "
                f"teacher_hidden_vs_standard_max_abs={float(teacher_hidden_vs_standard.max().item()):.8g} "
                f"teacher_hidden_vs_shadow_max_abs={float(teacher_hidden_vs_shadow.max().item()):.8g} "
                f"teacher_raw6_vs_standard_max_abs={float(teacher_scores_vs_standard.max().item()):.8g} "
                f"teacher_raw6_vs_shadow_max_abs={float(teacher_scores_vs_shadow.max().item()):.8g} "
                f"teacher_action={teacher_decision.action.value} "
                f"teacher_action_vs_standard_match="
                f"{teacher_decision.action == standard_next_decision.action} "
                f"teacher_action_vs_shadow_match="
                f"{teacher_decision.action == shadow_next_decision.action}",
                flush=True,
            )
            assert historical_hidden_exact, float(historical_hidden_delta.max().item())
            assert historical_scores_exact, float(historical_score_delta.max().item())
            assert history_actions_match
            assert kv_exact, kv_first_difference
            assert next_hidden_exact, next_hidden_max_abs
            assert next_scores_exact, next_score_max_abs
            assert next_actions_match
        finally:
            try:
                if shadow_owner is not None:
                    shadow_owner.close()
            finally:
                try:
                    if standard_replay is not None:
                        standard_replay.close()
                finally:
                    if shadow_owner is None and shadow_provider is not None:
                        shadow_provider.close()
