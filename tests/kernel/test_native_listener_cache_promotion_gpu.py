"""Focused semantic-prefix cache promotion parity on the configured CUDA image.

Opt in with ``GEMMA4_RUN_NATIVE_CACHE_PROMOTION_GPU=1``. One loaded Gemma is
shared by a standard ABI3 continuation, a priority-0 shadow owner, and a
priority--1 primary owner. The shadow replays two Q4 frames, promotes its
cache at the matching logical watermark, and the primary continues twice.
"""
from __future__ import annotations

import hashlib
import os
import queue
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import pytest


def _snapshot_kv_mismatches(
    source: dict[int, tuple[Any, Any]],
    destination: dict[int, tuple[Any, Any]],
    *,
    prefix_len: int | None = None,
) -> list[dict[str, Any]]:
    import torch

    if set(source) != set(destination):
        return [{"layers": (sorted(source), sorted(destination))}]
    mismatches: list[dict[str, Any]] = []
    for layer, (source_key, source_value) in source.items():
        destination_key, destination_value = destination[layer]
        layer_detail: dict[str, Any] = {"layer": layer}
        for name, source_tensor, destination_tensor in (
            ("key", source_key, destination_key),
            ("value", source_value, destination_value),
        ):
            compared_source = (
                source_tensor if prefix_len is None else source_tensor[:, :, :prefix_len]
            )
            if torch.equal(compared_source, destination_tensor):
                continue
            delta = (compared_source - destination_tensor).abs()
            nonzero = torch.nonzero(delta, as_tuple=False)
            layer_detail[name] = {
                "shape_source": tuple(compared_source.shape),
                "shape_destination": tuple(destination_tensor.shape),
                "max_abs": float(delta.max().item()) if delta.numel() else None,
                "first_index": (
                    tuple(int(value) for value in nonzero[0].tolist())
                    if nonzero.numel() else None
                ),
            }
        if len(layer_detail) > 1:
            mismatches.append(layer_detail)
            if len(mismatches) == 2:
                break
    return mismatches


def test_snapshot_comparison_pairs_key_and_value_across_contexts() -> None:
    torch = pytest.importorskip("torch")
    source = {
        0: (
            torch.ones((1, 1, 5, 2), dtype=torch.bfloat16),
            torch.full((1, 1, 5, 2), 2, dtype=torch.bfloat16),
        )
    }
    destination = {
        0: (
            torch.ones((1, 1, 3, 2), dtype=torch.bfloat16),
            torch.full((1, 1, 3, 2), 2, dtype=torch.bfloat16),
        )
    }
    assert not _snapshot_kv_mismatches(source, destination, prefix_len=3)
    destination[0][0][0, 0, 2, 1] = 3
    mismatch = _snapshot_kv_mismatches(source, destination, prefix_len=3)
    assert mismatch[0]["key"]["first_index"] == (0, 0, 2, 1)
    assert "value" not in mismatch[0]


@pytest.mark.skipif(
    os.environ.get("GEMMA4_RUN_NATIVE_CACHE_PROMOTION_GPU") != "1",
    reason="requires explicit RTX 3090 runtime",
)
def test_shadow_cache_promotion_preserves_native_continuation_and_scores() -> None:
    import torch

    from duplex_kernel.microbatch_contract import (
        BatchRowDescriptor,
        ListenerBatchManifest,
        RuntimeSignature,
    )
    from duplex_kernel.listener import GemmaAudioTowerBackend
    from duplex_kernel.models.thinker import load_thinker
    from duplex_kernel.native_control6_prompt import (
        build_native_control6_prefix_ids,
        validate_native_control6_semantic_prefix_metadata,
    )
    from duplex_kernel.native_listener import (
        GEMMA4_ATTN_NATIVE_SPLITK_GQA,
        GEMMA4_RMS_NATIVE_EXACT,
        NativeGemmaListenerLM,
        _load_compiled_listener_topology,
    )
    from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
    from training.duplex_qlora.native_control6 import load_native_control6_adapter, select_tied_output_rows

    workspace = Path("/workspace")
    batch_library = Path(os.environ.get(
        "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH",
        str(workspace / ".tmp/listener_batch_source_overlay/native/gemma4_listener_microbatch/libgemma4_listener_microbatch_cachecopy.so"),
    ))
    # Use the same compiled source for both paths: the older standalone ABI3
    # library does not export semantic-prefix import, so it is not a valid
    # oracle for an actual L=132-style prefix.
    standard_library = batch_library
    checkpoint = Path(os.environ.get(
        "DUPLEX_NATIVE_CONTROL6_ADAPTER_RELATIVE_PATH",
        "checkpoints/native_control6_semantic_prefix_mixed_candidate600.pt",
    ))
    if not checkpoint.is_absolute():
        checkpoint = workspace / checkpoint
    assert standard_library.is_file() and batch_library.is_file() and checkpoint.is_file()

    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["GEMMA4_NATIVE_LISTENER_LIBRARY_PATH"] = str(standard_library)
    os.environ["GEMMA4_RMSNORM_BACKEND"] = "NATIVE_EXACT"
    os.environ["GEMMA4_ATEN_RMSNORM_BRIDGE"] = "0"
    device = torch.device("cuda:0")
    assert torch.cuda.is_available() and "3090" in torch.cuda.get_device_name(device)

    metadata = validate_native_control6_semantic_prefix_metadata(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    assert metadata["semantic_prefix_context_conditioned"] is True
    thinker = load_thinker(str(workspace), device=device, dtype=torch.bfloat16)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device), use_native_lm=False)
    adapter, _ = load_native_control6_adapter(
        checkpoint,
        base_rows=select_tied_output_rows(thinker.model.lm_head.weight),
        device=device,
    )
    adapter.eval()

    topology_id = _load_compiled_listener_topology()["manifest_sha256"]
    runtime = RuntimeSignature(
        model_id="google/gemma-4-E2B-it",
        model_config_sha256="0" * 64,
        checkpoint_sha256="0" * 64,
        topology_sha256=topology_id,
        native_library_sha256="0" * 64,
        native_abi_version=3,
        batch_api_version=2,
        attention_backend_id=GEMMA4_ATTN_NATIVE_SPLITK_GQA,
        rmsnorm_backend_id=GEMMA4_RMS_NATIVE_EXACT,
        max_seq_len=16_384,
        stream_q_len=4,
    )
    session_id = f"cache-promotion-{uuid.uuid4().hex[:10]}"
    epoch = uuid.uuid4().hex
    now = time.monotonic_ns()

    def row(frame_id: int, version: int, sequence: int, context_id: int, position: int) -> BatchRowDescriptor:
        return BatchRowDescriptor(
            session_id=session_id,
            session_generation=7,
            state_epoch_id=epoch,
            frame_id=frame_id,
            listener_state_version=version,
            sequence_position=sequence,
            cache_position=position,
            valid_history_tokens=position,
            native_context_id=context_id,
            input_frame_sha256=hashlib.sha256(f"promotion-frame-{frame_id}".encode()).hexdigest(),
            reference_pcm_sha256=None,
            frame_ready_at_ns=now + frame_id,
            reference_frame_id=None,
        )

    marker = getattr(backend.processor, "audio_token", None) or getattr(
        thinker.tokenizer, "audio_token", None
    ) or "<|audio|>"
    prefix_ids, prefix_len = build_native_control6_prefix_ids(
        thinker.tokenizer,
        instructions="Ascolta e rispondi in modo conciso.",
        completed_messages=(
            {"role": "user", "content": "Qual è il punto principale?"},
            {"role": "assistant", "content": "Il punto principale è chiaro."},
        ),
        audio_marker=str(marker),
        boa_token_id=backend.boa_token_id,
    )
    assert 1 <= prefix_len <= 291 and prefix_len == len(prefix_ids)
    prefix_stream = torch.cuda.Stream(device=device, priority=0)
    with torch.cuda.stream(prefix_stream), torch.inference_mode():
        prefix_cache = backend.build_semantic_prefix_cache(prefix_ids)
    prefix_stream.synchronize()
    assert prefix_cache["prefix_len"] == prefix_len
    assert prefix_cache["cache_position"] == prefix_len
    prefix_attention = prefix_cache["prefix_attention"]

    generator = torch.Generator(device=device).manual_seed(20261014)
    q4_inputs = [
        torch.randn((1, 4, 1536), generator=generator, device=device, dtype=torch.float32)
        .to(torch.bfloat16).contiguous()
        for _ in range(4)
    ]

    standard_stream = torch.cuda.Stream(device=device, priority=-1)
    standard = NativeGemmaListenerLM(
        lm,
        max_seq_len=16_384,
        stream_q_len=4,
        enable_cuda_graph=False,
        attention_backend=GEMMA4_ATTN_NATIVE_SPLITK_GQA,
        rmsnorm_backend=GEMMA4_RMS_NATIVE_EXACT,
    )
    shared_frame_pad_ple = standard.frame_pad_ple
    standard_outputs: list[torch.Tensor] = []
    try:
        standard.import_cache_async(
            prefix_cache["past_key_values"], current_position=prefix_len,
            stream=standard_stream, producer_stream=prefix_stream,
            prefix_len=prefix_len, prefix_mask=prefix_attention,
        )
        with torch.cuda.stream(standard_stream), torch.inference_mode():
            for q4 in q4_inputs:
                output = standard.step_async(
                    q4, stream=standard_stream, producer_stream=standard_stream,
                )
                standard_outputs.append(output.clone())
        standard_stream.synchronize()
    finally:
        standard.close()

    class NativeOwner:
        def __init__(self, priority: int, context_id: int):
            self.priority = priority
            self.context_id = context_id
            self.commands: queue.Queue[Any] = queue.Queue()
            self.started = threading.Event()
            self.failure: BaseException | None = None
            self.stream: torch.cuda.Stream | None = None
            self.thread = threading.Thread(target=self._run, name=f"cache-owner-{priority}", daemon=True)
            self.thread.start()
            if not self.started.wait(60):
                raise TimeoutError(f"native owner priority {priority} did not initialize")
            if self.failure is not None:
                raise RuntimeError(f"native owner priority {priority} failed to initialize") from self.failure

        def _run(self) -> None:
            native = None
            try:
                torch.cuda.set_device(device)
                self.stream = torch.cuda.Stream(device=device, priority=self.priority)
                provider = NativeGemmaListenerLM(
                    lm,
                    max_seq_len=16_384,
                    stream_q_len=4,
                    enable_cuda_graph=False,
                    attention_backend=GEMMA4_ATTN_NATIVE_SPLITK_GQA,
                    rmsnorm_backend=GEMMA4_RMS_NATIVE_EXACT,
                    frame_pad_ple_tensor=shared_frame_pad_ple,
                )
                native = NativeGemmaListenerMicrobatch(
                    provider,
                    library_path=batch_library,
                    runtime_signature=runtime,
                    listener_stream=self.stream,
                    max_batch_size=1,
                    stream_priority=self.priority,
                )
            except BaseException as exc:
                self.failure = exc
                self.started.set()
                return
            self.started.set()
            while True:
                task = self.commands.get()
                if task is None:
                    break
                fn, result_queue = task
                try:
                    result_queue.put((True, fn(native, self.stream)))
                except BaseException as exc:
                    result_queue.put((False, exc))
            try:
                native.close()
                self.stream.synchronize()
            except BaseException as exc:
                self.failure = exc

        def call(self, fn: Callable[[NativeGemmaListenerMicrobatch, torch.cuda.Stream], Any]) -> Any:
            result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
            self.commands.put((fn, result_queue))
            success, value = result_queue.get(timeout=60)
            if not success:
                raise value
            return value

        def close(self) -> None:
            self.commands.put(None)
            self.thread.join(timeout=60)
            if self.thread.is_alive():
                raise TimeoutError(f"native owner priority {self.priority} did not close")
            if self.failure is not None:
                raise RuntimeError(f"native owner priority {self.priority} failed during close") from self.failure

    def manifest_for(owner: NativeOwner, descriptor: BatchRowDescriptor) -> ListenerBatchManifest:
        assert owner.stream is not None
        return ListenerBatchManifest(
            manifest_id=uuid.uuid4().hex,
            runtime_signature=runtime,
            stream_identity=NativeGemmaListenerMicrobatch._stream_identity(owner.stream),
            stream_priority=owner.priority,
            dispatch_at_ns=time.monotonic_ns(),
            max_coalesce_wait_ns=0,
            dispatch_reason="multiple_ready",
            rows=(descriptor,),
        )

    def step(owner: NativeOwner, descriptor: BatchRowDescriptor, q4: torch.Tensor) -> tuple[torch.Tensor, BatchRowDescriptor, Any]:
        def invoke(native: NativeGemmaListenerMicrobatch, _stream: torch.cuda.Stream):
            output = native.step_async(
                manifest_for(owner, descriptor), q4, producer_stream=_stream,
            )
            output.completion_event.synchronize()
            health = native.verify_output_health(output)
            assert len(health) == 1 and health[0].status == "healthy"
            with torch.cuda.stream(_stream):
                output_row = output.tensor[0:1].clone()
            next_row = replace(
                descriptor,
                frame_id=descriptor.frame_id + 1,
                listener_state_version=descriptor.listener_state_version + 1,
                sequence_position=descriptor.sequence_position + 1,
                cache_position=descriptor.cache_position + 4,
                valid_history_tokens=descriptor.valid_history_tokens + 4,
            )
            return output_row, next_row, output
        return owner.call(invoke)

    shadow = NativeOwner(0, 910_002)
    primary = NativeOwner(-1, 910_001)
    try:
        shadow_row0 = row(0, 0, 0, shadow.context_id, prefix_len)
        primary_row2 = row(2, 2, 2, primary.context_id, prefix_len)
        for owner, descriptor in ((shadow, shadow_row0), (primary, primary_row2)):
            owner.call(lambda native, _stream, d=descriptor: native.import_session(
                d,
                prefix_cache["past_key_values"],
                prefix_attention,
                prefix_len=prefix_len,
                producer_stream=prefix_stream,
            ))

        shadow_row = shadow_row0
        shadow_hidden: list[torch.Tensor] = []
        for index in range(2):
            hidden, shadow_row, _ = step(shadow, shadow_row, q4_inputs[index])
            shadow_hidden.append(hidden)
        assert shadow_row.frame_id == 2 and shadow_row.cache_position == prefix_len + 8

        token = shadow.call(lambda native, _stream: native.prepare_cache_copy_source(shadow_row))
        before = primary.call(lambda native, _stream: native.session_state(primary_row2))
        stale_destination = replace(primary_row2, frame_id=3)
        with pytest.raises(ValueError, match="logical boundary"):
            primary.call(lambda native, _stream: native.copy_session_cache_from(token, stale_destination))
        after = primary.call(lambda native, _stream: native.session_state(primary_row2))
        assert before == after, "stale promotion modified the active primary context"

        source_snapshot = shadow.call(lambda native, _stream: native.snapshot_session(shadow_row))
        destination_before = primary.call(
            lambda native, _stream: native.snapshot_session(primary_row2)
        )
        assert source_snapshot.state.current_position == prefix_len + 8
        assert destination_before.state.current_position == prefix_len
        prefix_mismatches = _snapshot_kv_mismatches(
            source_snapshot.kv_by_layer,
            destination_before.kv_by_layer,
            prefix_len=prefix_len,
        )
        assert not prefix_mismatches, f"pre-copy imported prefix differs: {prefix_mismatches}"

        copy_start = torch.cuda.Event(enable_timing=True)
        copy_end = torch.cuda.Event(enable_timing=True)
        host_started = time.perf_counter_ns()

        def promote(native: NativeGemmaListenerMicrobatch, stream: torch.cuda.Stream):
            copy_start.record(stream)
            result = native.copy_session_cache_from(token, primary_row2)
            copy_end.record(stream)
            return result

        promoted = primary.call(promote)
        copy_host_ms = (time.perf_counter_ns() - host_started) / 1_000_000.0
        # The source owner fences reuse asynchronously; no device-wide sync is needed.
        shadow.call(lambda _native, stream: stream.wait_event(promoted.completion_event))
        copy_end.synchronize()
        copy_cuda_ms = float(copy_start.elapsed_time(copy_end))
        assert promoted.prefix_len == prefix_len
        assert promoted.destination_row.cache_position == prefix_len + 8

        destination_snapshot = primary.call(
            lambda native, _stream: native.snapshot_session(promoted.destination_row)
        )
        assert source_snapshot.state.current_position == destination_snapshot.state.current_position
        assert set(source_snapshot.kv_by_layer) == set(destination_snapshot.kv_by_layer)
        kv_mismatches = _snapshot_kv_mismatches(
            source_snapshot.kv_by_layer,
            destination_snapshot.kv_by_layer,
        )
        assert not kv_mismatches, f"promoted K/V differed at layers {kv_mismatches}"

        primary_row = promoted.destination_row
        primary_hidden2, primary_row, _ = step(primary, primary_row, q4_inputs[2])
        primary_hidden3, primary_row, _ = step(primary, primary_row, q4_inputs[3])
        standard_hidden2, standard_hidden3 = standard_outputs[2], standard_outputs[3]
        hidden2_error = float((primary_hidden2.float() - standard_hidden2.float()).abs().max().item())
        hidden3_error = float((primary_hidden3.float() - standard_hidden3.float()).abs().max().item())
        standard_scores = adapter(standard_hidden2[:, 0, :]).float()
        promoted_scores = adapter(primary_hidden2[:, 0, :]).float()
        score_error = float((standard_scores - promoted_scores).abs().max().item())
        actions_match = bool(torch.equal(standard_scores.argmax(dim=-1), promoted_scores.argmax(dim=-1)))
        memory = {
            "allocated_gib": round(torch.cuda.memory_allocated(device) / 1024**3, 3),
            "reserved_gib": round(torch.cuda.memory_reserved(device) / 1024**3, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
        }
        result = {
            "prefix_len": prefix_len,
            "source_cursor": source_snapshot.state.current_position,
            "unique_kv_layers_exact": len(source_snapshot.kv_by_layer),
            "stale_rejection_preserved_primary": before == after,
            "copy_host_enqueue_ms": round(copy_host_ms, 4),
            "copy_cuda_event_ms": round(copy_cuda_ms, 4),
            "hidden_after_promotion_max_abs": hidden2_error,
            "continued_hidden_max_abs": hidden3_error,
            "raw_six_score_max_abs": score_error,
            "raw_action_match": actions_match,
            "memory": memory,
        }
        print(f"native-cache-promotion: {result}", flush=True)
        assert hidden2_error == 0.0 and hidden3_error == 0.0
        assert score_error <= 1e-5 and actions_match
    finally:
        shadow.close()
        primary.close()

