"""Opt-in GPU parity for batched raw NativeControl6 scores.

Run inside the configured RTX 3090 image with
``GEMMA4_RUN_CONTROL6_RAW_GPU=1``.  The test uses real Gemma listener outputs
and the trained semantic-prefix 600 checkpoint, then compares one [N,1536]
adapter call with the same adapter applied row by row.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_CONTROL6_RAW_GPU") != "1",
    reason="requires explicit RTX 3090 runtime",
)
def test_native_control6_raw_batch_scores_match_single_rows_on_gpu() -> None:
    import torch

    from duplex_kernel.gemma_native_control6_batch_policy import GemmaNativeControl6BatchPolicy
    from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.listener_microbatch import ListenerBatch
    from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor,
        ListenerBatchManifest,
        ListenerFrame,
        ListenerSessionState,
        RuntimeSignature,
    )
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_listener import NativeGemmaListenerLM, _load_compiled_listener_topology
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from duplex_kernel.native_control6_prompt import build_native_control6_prefix_ids
    from duplex_kernel.native_control6_prompt import validate_native_control6_semantic_prefix_metadata
    from training.duplex_qlora.native_control6 import (
        ACTIONS,
        load_native_control6_adapter,
        select_tied_output_rows,
    )
    from duplex_kernel.turn_policy import ControlAction, PolicyInput

    workspace = Path("/workspace")
    baseline = workspace / ".tmp/listener_batch_source_overlay/native/gemma4_listener/libgemma4_listener.so"
    candidate = Path(os.environ.get(
        "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH",
        str(workspace / ".tmp/listener_batch_source_overlay/native/gemma4_listener_microbatch/libgemma4_listener_microbatch.so"),
    ))
    checkpoint = Path(os.environ.get(
        "DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH",
        "checkpoints/native_control6_semantic_prefix_mixed_candidate600.pt",
    ))
    if not checkpoint.is_absolute():
        checkpoint = workspace / checkpoint
    assert baseline.is_file() and candidate.is_file()
    assert checkpoint.is_file(), f"trained semantic-600 adapter not found: {checkpoint}"
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    semantic_metadata = validate_native_control6_semantic_prefix_metadata(checkpoint_payload)
    assert semantic_metadata["semantic_prefix_context_conditioned"] is True

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = str(baseline)
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"
    device = torch.device("cuda:0")
    assert torch.cuda.is_available()
    assert "3090" in torch.cuda.get_device_name(device)

    thinker = load_thinker(str(workspace), device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    output_weight = thinker.model.lm_head.weight
    live_rows = select_tied_output_rows(output_weight)
    adapter, _report = load_native_control6_adapter(
        checkpoint,
        base_rows=live_rows,
        device=device,
    )
    adapter.eval()
    checkpoint_path = str(checkpoint.resolve())
    batch_scorer = GemmaNativeControl6BatchPolicy(adapter, checkpoint_path=checkpoint_path)
    single_policy = GemmaNativeControl6Policy(adapter, checkpoint_path=checkpoint_path)

    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    stream = torch.cuda.Stream(device=device, priority=-1)
    processor = backend.processor
    marker = getattr(processor, "audio_token", None) or getattr(thinker.tokenizer, "audio_token", None) or "<|audio|>"
    prefix_ids, prefix_len = build_native_control6_prefix_ids(
        thinker.tokenizer,
        instructions="Rispondi in modo conciso e resta attento alle interruzioni.",
        completed_messages=(
            {"role": "user", "content": "Puoi riassumere il punto principale?"},
            {"role": "assistant", "content": "Certo, ascolto e rispondo al punto."},
        ),
        audio_marker=str(marker),
        boa_token_id=backend.boa_token_id,
    )
    with torch.cuda.stream(stream), torch.inference_mode():
        prefix_cache = backend.build_semantic_prefix_cache(prefix_ids)
    stream.synchronize()
    assert prefix_len == len(prefix_ids) and 1 <= prefix_len <= 291
    assert prefix_cache["prefix_len"] == prefix_len
    assert prefix_cache["cache_position"] == prefix_len
    prefix_attention = prefix_cache["prefix_attention"]
    assert tuple(prefix_attention.shape) == (1, prefix_len)

    frontend = ListenerMicrobatchAudioFrontend(
        backend.gemma_model,
        backend._extract_single_frame,
        device=device,
        dtype=torch.bfloat16,
        audio_tower_row_runner=backend.run_audio_tower_q4,
    )
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
    settings = {
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "allow_bf16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
        ),
    }
    cases = (1, 2, 4, 8, 12)
    results: dict[int, dict[str, object]] = {}
    enable_gemm_graphs = os.environ.get("DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS") == "1"

    for n in cases:
        run_id = uuid.uuid4().hex
        now = time.monotonic_ns()
        pcm_rows = [
            (0.05 * np.random.default_rng(20261013 + n * 1009 + index * 7919).standard_normal(1280))
            .astype(np.float32)
            for index in range(n)
        ]
        states: list[ListenerSessionState] = []
        frames: list[ListenerFrame] = []
        rows: list[BatchRowDescriptor] = []
        for index, pcm in enumerate(pcm_rows):
            session_id = f"control6-score-{run_id[:8]}-{index}"
            epoch = f"control6-score-{run_id}-{index}"
            native_context_id = 80_000 + index
            frame = ListenerFrame(
                session_id=session_id,
                generation_id=1,
                state_epoch_id=epoch,
                frame_id=0,
                ready_at_ns=now,
                pcm=pcm,
            )
            state = ListenerSessionState(
                session_id=session_id,
                generation_id=1,
                state_epoch_id=epoch,
                runtime_signature=runtime,
                next_frame_id=0,
                state_version=0,
                sequence_position=0,
                cache_position=prefix_len,
                valid_history_tokens=prefix_len,
                native_context_id=native_context_id,
            )
            row = BatchRowDescriptor(
                session_id=session_id,
                session_generation=1,
                state_epoch_id=epoch,
                frame_id=0,
                listener_state_version=0,
                sequence_position=0,
                cache_position=prefix_len,
                valid_history_tokens=prefix_len,
                native_context_id=native_context_id,
                input_frame_sha256=frame.pcm_sha256,
                reference_pcm_sha256=None,
                frame_ready_at_ns=now,
                reference_frame_id=None,
            )
            frames.append(frame)
            states.append(state)
            rows.append(row)
        manifest = ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=runtime,
            stream_identity=NativeGemmaListenerMicrobatch._stream_identity(stream),
            stream_priority=-1,
            dispatch_at_ns=now,
            max_coalesce_wait_ns=0,
            dispatch_reason="coalesce_deadline" if n == 1 else "multiple_ready",
            rows=tuple(rows),
        )
        batch = ListenerBatch(manifest=manifest, frames=tuple(frames))

        prepared = None
        provider = NativeGemmaListenerLM(
            lm,
            max_seq_len=16_384,
            stream_q_len=4,
            enable_cuda_graph=False,
            attention_backend=3,
            rmsnorm_backend=2,
        )
        owner = None
        try:
            with torch.cuda.stream(stream), torch.inference_mode():
                prepared = frontend.encode_batch(batch, states)
                audio_soft = torch.stack(prepared.audio_soft_tokens, dim=0).contiguous()
            owner = NativeGemmaListenerMicrobatch(
                provider,
                library_path=candidate,
                runtime_signature=runtime,
                listener_stream=stream,
                max_batch_size=1,
                enable_gemm_graphs=enable_gemm_graphs,
            )
            owner.ensure_capacity(n)
            for row in rows:
                owner.import_session(
                    row,
                    prefix_cache["past_key_values"],
                    prefix_attention,
                    prefix_len=prefix_len,
                    producer_stream=stream,
                )
            output = owner.step_async(manifest, audio_soft, producer_stream=stream)
            output.completion_event.synchronize()
            health = owner.verify_output_health(output)
            assert len(health) == n and all(item.status == "healthy" for item in health)

            snapshots = batch_scorer.score_raw_batch(output, manifest)
            batched_logits = torch.tensor(
                [snapshot.raw_logits for snapshot in snapshots],
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode():
                singles = torch.cat([
                    adapter(output.tensor[index:index + 1, 0, :]).float()
                    for index in range(n)
                ], dim=0)
            delta = (batched_logits - singles).abs()
            max_abs = float(delta.max().item())
            rms = float(delta.square().mean().sqrt().item())
            cosine = float(torch.nn.functional.cosine_similarity(
                batched_logits.reshape(1, -1), singles.reshape(1, -1), dim=1,
            ).item())
            batch_argmax = batched_logits.argmax(dim=1)
            single_argmax = singles.argmax(dim=1)
            argmax_stable = bool(torch.equal(batch_argmax, single_argmax))
            top_two = torch.topk(singles, k=2, dim=1).values
            minimum_top_gap = float((top_two[:, 0] - top_two[:, 1]).min().item())

            action_matches = True
            masked_matches = True
            for index, (row, snapshot) in enumerate(zip(rows, snapshots, strict=True)):
                hidden = output.tensor[index:index + 1, 0, :]
                base_state = PolicyInput(
                    session_id=row.session_id,
                    frame_id=row.frame_id,
                    user_pcm=pcm_rows[index],
                    speaker_active=False,
                    # Turn generation deliberately differs from the stable native-row generation.
                    generation_id=41 + index,
                    native_control_hidden=hidden,
                    native_control_hidden_frame_id=row.frame_id,
                    native_control_hidden_at_ns=time.monotonic_ns(),
                    legal_actions=tuple(ACTIONS),
                    native_control6_raw_scores=snapshot,
                    native_listener_generation=row.session_generation,
                )
                standard = single_policy.step(
                    PolicyInput(**{
                        **base_state.__dict__,
                        "native_control6_raw_scores": None,
                    })
                )
                batched = single_policy.step(base_state)
                action_matches &= standard.action == batched.action

                top_order = torch.argsort(singles[index], descending=True).tolist()
                legal_second = (ACTIONS[top_order[1]], ACTIONS[top_order[2]])
                masked_state = PolicyInput(**{
                    **base_state.__dict__,
                    "legal_actions": legal_second,
                })
                standard_masked = single_policy.step(PolicyInput(**{
                    **masked_state.__dict__,
                    "native_control6_raw_scores": None,
                }))
                batched_masked = single_policy.step(masked_state)
                masked_matches &= standard_masked.action == batched_masked.action

            results[n] = {
                "max_abs_error": max_abs,
                "rms_error": rms,
                "cosine": cosine,
                "argmax_stable": argmax_stable,
                "action_matches": action_matches,
                "masked_action_matches": masked_matches,
                "minimum_top_two_gap": minimum_top_gap,
                "checkpoint": checkpoint_path,
                "semantic_prefix_metadata": semantic_metadata,
                "tf32": settings,
                "gemm_graphs_enabled": owner.gemm_graphs_enabled,
                "gemm_graph_count": owner.gemm_graph_count,
            }
            if enable_gemm_graphs:
                assert owner.gemm_graphs_enabled and owner.gemm_graph_count > 0, results[n]
            assert argmax_stable, results[n]
            assert action_matches and masked_matches, results[n]
            assert max_abs <= 1.0, results[n]
            prepared.commit()
        finally:
            if (
                prepared is not None
                and not prepared.committed
                and not getattr(prepared, "_aborted", False)
            ):
                prepared.abort()
            if owner is not None:
                owner.close()
            else:
                provider.close()
        print(f"native-control6-raw N={n}: {results[n]}", flush=True)

    assert tuple(results) == cases
