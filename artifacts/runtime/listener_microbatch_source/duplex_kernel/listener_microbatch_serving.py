"""Serving adapters for the opt-in shared listener tick runtime."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from duplex_kernel.listener_microbatch_worker import (
    InitialPrefixAdmission,
    ListenerMicrobatchRuntime,
    ListenerMicrobatchWorker,
    PreparedInitialPrefix,
    PrefixCacheImport,
    SemanticPrefixShadowRuntime,
)
from duplex_kernel.gemma_native_control6_batch_policy import (
    GemmaNativeControl6BatchPolicy,
    PolicyPreHeadContext,
)
from duplex_kernel.gemma_native_control6_policy import GemmaNativeControl6Policy
from duplex_kernel.microbatch_contract import (
    CANONICAL_PREFIX_TOKENS,
    LISTENER_MAX_SEQ_LEN,
    LISTENER_Q_PER_FRAME,
    ListenerSessionState,
    RuntimeSignature,
)
from duplex_kernel.native_control6_runtime import NativeControl6SemanticPrefixRequest


MICROBATCH_LIBRARY_ENV = "DUPLEX_LISTENER_MICROBATCH_LIBRARY_PATH"
GEMM_GRAPHS_ENV = "DUPLEX_LISTENER_MICROBATCH_GEMM_GRAPHS"
SEMANTIC_SHADOW_ENV = "DUPLEX_LISTENER_MICROBATCH_SEMANTIC_SHADOW"
STABLE_TURN_POLICY_Q2_CONTEXT_ENV = "DUPLEX_NATIVE_CONTROL6_STABLE_TURN_POLICY_Q2_CONTEXT"
STREAMING_CONFORMER_Q2_EXPERIMENTAL_ENV = "DUPLEX_STREAMING_CONFORMER_Q2_EXPERIMENTAL"
_EMPTY_SHA256 = "0" * 64


def _capture_experimental_streaming_q2(
    gemma_model: Any,
    *,
    device: Any,
    dtype: Any,
    max_sessions: int,
    max_graph_batch_size: int,
) -> Any:
    """Build the explicit opt-in tower using its single startup precision scope."""
    from duplex_kernel.streaming_conformer import StreamingConformerAudioTower

    return StreamingConformerAudioTower.from_gemma_model_optimized_cuda(
        gemma_model,
        device=device,
        dtype=dtype,
        max_sessions=max_sessions,
        max_graph_batch_size=max_graph_batch_size,
    )


def _build_streaming_q2_for_serving(
    gemma_model: Any,
    *,
    device: Any,
    dtype: Any,
    max_sessions: int,
    max_graph_batch_size: int,
) -> tuple[Any | None, str]:
    """Capture supported Q2 graph shapes, retaining W240 on opt-out/failure."""
    if not _strict_env_flag(STREAMING_CONFORMER_Q2_EXPERIMENTAL_ENV, default=True):
        return None, "disabled_explicit_opt_out_w240"
    try:
        tower = _capture_experimental_streaming_q2(
            gemma_model,
            device=device,
            dtype=dtype,
            max_sessions=max_sessions,
            max_graph_batch_size=max_graph_batch_size,
        )
    except Exception as exc:
        return None, f"unsupported_fallback_w240: {type(exc).__name__}: {exc}"
    return tower, f"experimental_graphs_n1_to_n{max_graph_batch_size}_captured"


def _strict_env_flag(name: str, *, default: bool = False) -> bool:
    """Read an explicit 0/1 serving switch without accepting ambiguous values."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value == "0":
        return False
    if value == "1":
        return True
    raise ValueError(f"{name} must be exactly '0' or '1'")


def _semantic_prefix_request_matches_applied_row(
    request: Any, request_generation_id: Any, row_output: Any
) -> bool:
    """Return whether a stored request names this row's committed prefix identity."""
    if request is None or request_generation_id is None:
        return False
    try:
        generation_matches = int(request_generation_id) == int(row_output.generation_id)
        revision_matches = int(request.semantic_context_revision) == int(
            row_output.semantic_context_revision
        )
    except (AttributeError, TypeError, ValueError):
        return False
    return bool(
        generation_matches
        and revision_matches
        and str(getattr(request, "session_id", "")) == str(row_output.session_id)
        and str(getattr(request, "state_epoch_id", "")) == str(row_output.state_epoch_id)
    )


def _policy_contexts_for_batch(
    batch: Any,
    states: tuple[ListenerSessionState, ...],
    *,
    policy_adapter: Any,
) -> tuple[PolicyPreHeadContext, ...]:
    """Bind each queued frame's immutable scheduler snapshot to its exact row."""
    frames = tuple(getattr(batch, "frames", ()))
    rows = tuple(getattr(getattr(batch, "manifest", None), "rows", ()))
    if not frames or len(frames) != len(states) or len(rows) != len(states):
        raise RuntimeError("native Control6 policy context batch rows are incomplete")
    contexts: list[PolicyPreHeadContext] = []
    for index, (frame, row, listener_state) in enumerate(
        zip(frames, rows, states, strict=True)
    ):
        context = getattr(frame, "policy_context", None)
        if not isinstance(context, PolicyPreHeadContext):
            raise RuntimeError(f"native Control6 frame {index} lacks PolicyPreHeadContext")
        policy = context.policy
        if getattr(policy, "adapter", None) is not policy_adapter:
            raise RuntimeError(f"native Control6 frame {index} uses a different loaded adapter")
        if context.listener_generation_id != row.session_generation:
            raise RuntimeError(f"native Control6 frame {index} listener generation differs from row")
        if (
            listener_state.session_id != row.session_id
            or listener_state.generation_id != row.session_generation
            or listener_state.state_epoch_id != row.state_epoch_id
        ):
            raise RuntimeError(f"native Control6 frame {index} listener state differs from row")
        if (
            context.policy_input.session_id != row.session_id
            or context.policy_input.frame_id != row.frame_id
            or context.tick_id
            != (
                context.policy_input.session_id,
                context.policy_input.generation_id,
                context.policy_input.frame_id,
            )
            or (frame.session_id, frame.generation_id, frame.frame_id)
            != (row.session_id, row.session_generation, row.frame_id)
        ):
            raise RuntimeError(f"native Control6 frame {index} turn snapshot differs from row")
        contexts.append(context)
    if len({context.policy_input.session_id for context in contexts}) != len(contexts):
        raise RuntimeError("native Control6 policy contexts must belong to distinct sessions")
    return tuple(contexts)


def _build_native_control6_batch_policy(
    native_control6_policy: Any | None,
) -> tuple[GemmaNativeControl6BatchPolicy | None, Any | None]:
    """Create the shared one-forward scorer and its frame-snapshot callback."""
    if native_control6_policy is None:
        return None, None
    if not isinstance(native_control6_policy, GemmaNativeControl6Policy):
        raise TypeError("shared native Control6 requires GemmaNativeControl6Policy")
    batch_policy = GemmaNativeControl6BatchPolicy(
        native_control6_policy.adapter,
        checkpoint_path=native_control6_policy.checkpoint_path,
        capture_debug_inputs=bool(
            os.environ.get("DUPLEX_NATIVE_CONTROL6_DEBUG_INPUT_DIRECTORY", "").strip()
        ),
    )

    def make_policy_inputs(
        batch: Any,
        states: tuple[ListenerSessionState, ...],
    ) -> tuple[PolicyPreHeadContext, ...]:
        return _policy_contexts_for_batch(
            batch,
            states,
            policy_adapter=batch_policy.adapter,
        )

    return batch_policy, make_policy_inputs


def make_listener_microbatch_worker(
    listener: Any,
    *,
    library_path: str | os.PathLike[str] | None = None,
    native_control6_policy: Any | None = None,
    semantic_prefix_enabled: bool = True,
    unified_kv_enabled: bool = False,
) -> ListenerMicrobatchWorker:
    """Create a single-owner worker from the already loaded serving listener.

    The CUDA stream, native weight provider, private batch runtime, and audio
    frontend are constructed on the worker's owner thread. Legacy mode also
    creates the canonical 34-token prefix. Unified mode requires an exact
    per-session text prefix importer and never builds the canonical audio
    prefix. There is deliberately no per-session fallback.
    """
    gemm_graphs_enabled = _strict_env_flag(GEMM_GRAPHS_ENV)
    stable_turn_policy_q2_context = bool(
        unified_kv_enabled
        and _strict_env_flag(
            STABLE_TURN_POLICY_Q2_CONTEXT_ENV,
            default=True,
        )
    )
    if type(unified_kv_enabled) is not bool:
        raise TypeError("unified_kv_enabled must be a bool")
    semantic_shadow_enabled = (
        False if unified_kv_enabled else _strict_env_flag(SEMANTIC_SHADOW_ENV)
    )
    if not unified_kv_enabled and semantic_shadow_enabled and (
        native_control6_policy is None or not semantic_prefix_enabled
    ):
        raise ValueError(
            f"{SEMANTIC_SHADOW_ENV}=1 requires native Control6 and semantic prefixes"
        )
    if not unified_kv_enabled and (
        native_control6_policy is not None
        and semantic_prefix_enabled
        and not semantic_shadow_enabled
    ):
        raise RuntimeError(
            "shared listener batching with semantic Control6 requires semantic shadow staging; "
            "set DUPLEX_LISTENER_MICROBATCH=0 for the inline compatibility path or enable "
            f"{SEMANTIC_SHADOW_ENV}=1"
        )

    backend = getattr(listener, "backend", None)
    if backend is None or not callable(getattr(backend, "_extract_single_frame", None)):
        raise TypeError("shared listener batching requires the loaded Gemma audio backend")
    if library_path is None:
        library_path = os.environ.get(MICROBATCH_LIBRARY_ENV)
    if not library_path:
        library_path = (
            Path(__file__).resolve().parents[1]
            / "native"
            / "gemma4_listener_microbatch"
            / "libgemma4_listener_microbatch.so"
        )
    resolved_library_path = Path(library_path).expanduser()

    def runtime_factory() -> ListenerMicrobatchRuntime:
        from duplex_kernel.listener_microbatch_frontend import ListenerMicrobatchAudioFrontend
        from duplex_kernel.native_listener import NativeGemmaListenerLM
        from duplex_kernel.native_listener_microbatch import NativeGemmaListenerMicrobatch
        from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

        language_model = backend.gemma_model.language_model
        device = torch.device(backend.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("shared native listener microbatch requires a CUDA model")
        if backend._shape_stable_lm is None:
            raise RuntimeError("shared listener needs the canonical ShapeStable Gemma LM")
        stream = torch.cuda.Stream(device=device, priority=-1)

        model_lock = getattr(backend, "_lock", None)
        if model_lock is None or not callable(getattr(model_lock, "acquire_fair", None)):
            raise RuntimeError("shared listener requires the source backend's fair model lock")
        configure_host_timeline = getattr(model_lock, "configure_host_timeline", None)
        if not callable(configure_host_timeline):
            raise RuntimeError("shared listener requires bounded fair-lock host timeline support")
        # The worker builds this runtime once on its owner thread, before any
        # listener registration or tick. Keep host contention telemetry scoped
        # to shared-listener serving; do not enable the global CUDA profiler.
        configure_host_timeline(enabled=True, capacity=256)
        prefix_cache = None
        prefix_attention = None
        if not unified_kv_enabled:
            # Legacy mode builds the historical BOS + 32 audio slots + BOA
            # prefix. Unified mode imports its actual chat prefix per session.
            input_ids, attention = backend._build_multimodal_inputs(
                (), 1024, device=device
            )
            prefix_ids = input_ids[:, :CANONICAL_PREFIX_TOKENS]
            prefix_attention = attention[:, :CANONICAL_PREFIX_TOKENS].to(
                device=device, dtype=torch.long
            )
            prefix_positions = torch.arange(
                CANONICAL_PREFIX_TOKENS, device=device, dtype=torch.long
            ).view(1, -1)
            model_lock_acquired = False
            try:
                acquire_fair = getattr(model_lock, "acquire_fair", None)
                if callable(acquire_fair):
                    model_lock_acquired = bool(acquire_fair("listener"))
                    if not model_lock_acquired:
                        raise RuntimeError("canonical listener prefix lock admission was cancelled")
                elif model_lock is not None:
                    model_lock.acquire()
                    model_lock_acquired = True
                with torch.cuda.stream(stream), torch.inference_mode():
                    prefix_embeds = language_model.embed_tokens(prefix_ids)
                    prefix_per_layer = language_model.get_per_layer_inputs(prefix_ids, prefix_embeds)
                    prefix_masks = create_masks_for_generate(
                        language_model.config,
                        prefix_embeds,
                        prefix_attention,
                        past_key_values=None,
                        position_ids=prefix_positions,
                    )
                    prefix_output = backend._shape_stable_lm.forward(
                        inputs_embeds=prefix_embeds,
                        per_layer_inputs=prefix_per_layer,
                        attention_mask=prefix_masks,
                        position_ids=prefix_positions,
                        past_key_values=None,
                        shared_kv_states=None,
                    )
            finally:
                if model_lock_acquired:
                    model_lock.release()
            prefix_cache = prefix_output.past_key_values
            stream.synchronize()

        provider = NativeGemmaListenerLM(
            language_model,
            max_seq_len=LISTENER_MAX_SEQ_LEN,
            stream_q_len=LISTENER_Q_PER_FRAME,
            enable_cuda_graph=False,
            attention_backend=3,
            rmsnorm_backend=2,
        )
        topology_sha256 = str(getattr(provider, "_manifest_sha256", ""))
        if len(topology_sha256) != 64:
            raise RuntimeError("native listener provider did not expose its topology identity")
        signature = RuntimeSignature(
            model_id="google/gemma-4-E2B-it",
            model_config_sha256=_EMPTY_SHA256,
            checkpoint_sha256=_EMPTY_SHA256,
            topology_sha256=topology_sha256,
            native_library_sha256=_EMPTY_SHA256,
            native_abi_version=3,
            batch_api_version=5,
            attention_backend_id=3,
            rmsnorm_backend_id=2,
            max_seq_len=LISTENER_MAX_SEQ_LEN,
            stream_q_len=LISTENER_Q_PER_FRAME,
        )
        native = NativeGemmaListenerMicrobatch(
            provider,
            library_path=resolved_library_path,
            runtime_signature=signature,
            listener_stream=stream,
            max_batch_size=4,
            enable_gemm_graphs=gemm_graphs_enabled,
            enable_variable_q_gemm_graphs=unified_kv_enabled,
        )
        with torch.cuda.stream(stream):
            primary_capacity = native.ensure_capacity(4)
        if primary_capacity < 4:
            raise RuntimeError(
                f"primary listener capacity prewarm returned {primary_capacity}, expected >=4"
            )
        streaming_audio_tower, streaming_q2_status = _build_streaming_q2_for_serving(
            backend.gemma_model,
            device=device,
            dtype=backend.dtype,
            max_sessions=128,
            max_graph_batch_size=int(native.max_batch_size),
        )
        frontend = ListenerMicrobatchAudioFrontend(
            backend.gemma_model,
            backend._extract_single_frame,
            device=device,
            dtype=backend.dtype,
            mel_window_rows=backend.mel_window_rows,
            # The shared LM lock is reserved for semantic-prefix ShapeStable work.
            audio_tower_row_runner=backend.run_audio_tower_q2,
            streaming_audio_tower=streaming_audio_tower,
        )
        # The mixed-source policy wrapper runs one head forward after the
        # server has captured each row's exact pre-head scheduler snapshot.
        control6_scorer = None
        batch_policy, policy_inputs = _build_native_control6_batch_policy(
            native_control6_policy
        )
        token_adapter = None
        if unified_kv_enabled:
            from duplex_kernel.unified_token_adapter import UnifiedTokenAdapter

            tokenizer = getattr(getattr(backend, "processor", None), "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("unified text decode requires the official Thinker tokenizer")
            token_adapter = UnifiedTokenAdapter(
                backend.model,
                tokenizer=tokenizer,
                require_cuda=True,
            )
        backend._listener_microbatch_runtime_info = {
            "enabled": True,
            "mode": "shared_fixed_80ms_microbatch",
            "streaming_conformer_q2_opt_in": streaming_audio_tower is not None,
            "streaming_conformer_q2_status": streaming_q2_status,
            "audio_frontend_default": (
                "streaming_conformer_q2_experimental"
                if streaming_audio_tower is not None
                else "w240"
            ),
            "unified_kv_enabled": unified_kv_enabled,
            "native_abi_version": signature.native_abi_version,
            "batch_api_version": signature.batch_api_version,
            "attention_backend_id": signature.attention_backend_id,
            "rmsnorm_backend_id": signature.rmsnorm_backend_id,
            "max_seq_len": signature.max_seq_len,
            "stream_q_len": signature.stream_q_len,
            "supports_arbitrary_batch_size": native.capability.supports_arbitrary_batch_size,
            "initial_capacity_hint": 4,
            "allocated_capacity": native.max_batch_size,
            "gemm_graphs_enabled": native.gemm_graphs_enabled,
            "gemm_graph_count": native.gemm_graph_count,
            "gemm_graph_prepare_ms": native.gemm_graph_prepare_ms,
            "variable_q_gemm_graphs_enabled": native.variable_q_gemm_graphs_enabled,
            "variable_q_gemm_graph_count": native.variable_q_gemm_graph_count,
            "variable_q_gemm_graph_prepare_ms": native.variable_q_gemm_graph_prepare_ms,
            "variable_q_gemm_graph_prepare_error": native.variable_q_gemm_graph_prepare_error,
            "stream_priority": native.capability.stream_priority,
            "topology_sha256": signature.topology_sha256,
            "library_path": str(native.library_path),
            "native_control6_raw_batch_scoring": control6_scorer is not None,
            "native_control6_mixed_source_policy": batch_policy is not None,
            "native_control6_checkpoint_path": (
                None if batch_policy is None else batch_policy.checkpoint_path
            ),
            "semantic_shadow_enabled": semantic_shadow_enabled,
            "semantic_shadow_ready": False,
        }

        import_prefix = None
        if not unified_kv_enabled:
            def import_prefix(_state: ListenerSessionState, _row: Any) -> PrefixCacheImport:
                assert prefix_cache is not None and prefix_attention is not None
                return PrefixCacheImport(
                    past_key_values=prefix_cache,
                    prefix_attention=prefix_attention,
                    prefix_len=CANONICAL_PREFIX_TOKENS,
                )

            # This importer returns an immutable cache already built during
            # worker initialization; no Thinker work is needed during import.
            setattr(import_prefix, "requires_shared_lm_lock", False)

        shadow_factory = None
        shadow_prefix_builder = None
        if semantic_shadow_enabled:
            def create_shadow_runtime():
                shadow_stream = torch.cuda.Stream(device=device, priority=0)
                shadow_provider = NativeGemmaListenerLM(
                    language_model,
                    max_seq_len=LISTENER_MAX_SEQ_LEN,
                    stream_q_len=LISTENER_Q_PER_FRAME,
                    enable_cuda_graph=False,
                    attention_backend=3,
                    rmsnorm_backend=2,
                    frame_pad_ple_tensor=provider.frame_pad_ple,
                )
                shadow_native = NativeGemmaListenerMicrobatch(
                    shadow_provider,
                    library_path=resolved_library_path,
                    runtime_signature=signature,
                    listener_stream=shadow_stream,
                    max_batch_size=1,
                    enable_gemm_graphs=gemm_graphs_enabled,
                    enable_variable_q_gemm_graphs=False,
                    stream_priority=0,
                )
                if shadow_native.max_batch_size < 1:
                    raise RuntimeError("semantic shadow capacity prewarm returned less than one")
                backend._listener_microbatch_runtime_info.update({
                    "semantic_shadow_ready": True,
                    "semantic_shadow_capacity": shadow_native.max_batch_size,
                    "semantic_shadow_gemm_graphs_enabled": shadow_native.gemm_graphs_enabled,
                    "semantic_shadow_gemm_graph_count": shadow_native.gemm_graph_count,
                    "semantic_shadow_gemm_graph_prepare_ms": shadow_native.gemm_graph_prepare_ms,
                })
                return SemanticPrefixShadowRuntime(
                    native=shadow_native,
                    listener_stream=shadow_stream,
                )

            def build_shadow_prefix(request: Any, shadow_stream: Any) -> PrefixCacheImport:
                from duplex_kernel.native_control6_runtime import NativeControl6SemanticPrefixRequest

                if not isinstance(request, NativeControl6SemanticPrefixRequest):
                    raise TypeError("shadow staging requires a frozen native Control6 prefix request")
                acquire_fair = getattr(model_lock, "acquire_fair", None)
                if not callable(acquire_fair):
                    raise RuntimeError("semantic shadow builder requires the shared fair model lock")
                fair_lock_wait_started_ns = time.perf_counter_ns()
                acquired = acquire_fair("listener")
                fair_lock_wait_ms = round(
                    max(0, time.perf_counter_ns() - fair_lock_wait_started_ns)
                    / 1_000_000.0,
                    3,
                )
                if not acquired:
                    raise RuntimeError("semantic shadow prefill lock admission was cancelled")
                teacher_prefix_fenced_wall_ms: float | None = None
                try:
                    with torch.cuda.stream(shadow_stream), torch.inference_mode():
                        teacher_prefix_started_ns = time.perf_counter_ns()
                        cache = backend.build_semantic_prefix_cache(
                            request.prefix_token_ids,
                        )
                        if (
                            int(cache["prefix_len"]) != int(request.prefix_len)
                            or int(cache["cache_position"]) != int(request.prefix_len)
                            or tuple(cache["replay_frame_ids"])
                        ):
                            raise RuntimeError("shadow teacher builder did not return a text-only prefix cache")
                        ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
                        ready_event.record(shadow_stream)
                        # ShapeStable uses mutable backend workspaces. Keep the
                        # fair lock until this stream has consumed those buffers;
                        # primary native ticks do not acquire this lock.
                        shadow_stream.synchronize()
                        teacher_prefix_fenced_wall_ms = round(
                            max(0, time.perf_counter_ns() - teacher_prefix_started_ns)
                            / 1_000_000.0,
                            3,
                        )
                finally:
                    model_lock.release()
                return PrefixCacheImport(
                    past_key_values=cache["past_key_values"],
                    prefix_attention=cache["prefix_attention"],
                    prefix_len=int(cache["prefix_len"]),
                    cache_position=int(request.prefix_len),
                    semantic_context_revision=int(request.semantic_context_revision),
                    semantic_replay_fallback_reason=request.fallback_reason,
                    prefix_build_ms=float(cache["prefix_build_ms"]),
                    replay_ms=None,
                    ready_event=ready_event,
                    replay_frame_ids=tuple(int(value) for value in request.replay_frame_ids),
                    fair_lock_wait_ms=fair_lock_wait_ms,
                    teacher_prefix_fenced_wall_ms=teacher_prefix_fenced_wall_ms,
                )

            shadow_factory = create_shadow_runtime
            shadow_prefix_builder = build_shadow_prefix

        return ListenerMicrobatchRuntime(
            listener_stream=stream,
            frontend=frontend,
            native=native,
            policy=batch_policy,
            policy_inputs=policy_inputs,
            listener_model_lock=model_lock,
            prefix_importer=import_prefix,
            control6_scorer=control6_scorer,
            shadow_factory=shadow_factory,
            shadow_prefix_builder=shadow_prefix_builder,
            token_adapter=token_adapter,
            close_resources=frontend.close,
        )

    return ListenerMicrobatchWorker(
        None,
        runtime_factory=runtime_factory,
        stable_turn_policy_q2_context=stable_turn_policy_q2_context,
    )


def make_unified_prefix_import(
    backend: Any,
    prefix_token_ids: tuple[int, ...] | list[int],
) -> Any:
    """Build and cache one real chat-prefix KV for a unified listener session.

    ``prefix_token_ids`` must be the exact system + opening user-header + BOA
    sequence produced from the active Thinker chat template.  In particular,
    it must not contain the legacy BOS/PAD audio prefix or audio placeholders.
    The returned owner-thread importer runs a single text-only ShapeStable
    prefill, then returns the same immutable cache for native import.  All
    later audio and generated-text appends use the native session cache.
    """
    token_ids = tuple(int(token) for token in prefix_token_ids)
    if not token_ids:
        raise ValueError("unified prefix token IDs must not be empty")
    if any(isinstance(token, bool) for token in prefix_token_ids):
        raise TypeError("unified prefix token IDs must be integers, not booleans")
    bos_token_id = getattr(backend, "bos_token_id", None)
    if bos_token_id is None:
        config = getattr(getattr(backend, "gemma_model", None), "config", None)
        bos_token_id = getattr(config, "bos_token_id", None)
    if bos_token_id is not None and token_ids[0] != int(bos_token_id):
        raise ValueError("unified prefix must begin with the Thinker BOS token")
    boa_token_id = getattr(backend, "boa_token_id", None)
    if boa_token_id is None:
        config = getattr(getattr(backend, "gemma_model", None), "config", None)
        boa_token_id = getattr(config, "boa_token_id", None)
    if boa_token_id is None or token_ids[-1] != int(boa_token_id):
        raise ValueError("unified prefix must end with the Thinker BOA token")
    audio_token_id = getattr(backend, "audio_token_id", None)
    eoa_token_id = getattr(backend, "eoa_token_id", None)
    forbidden = {
        int(value) for value in (audio_token_id, eoa_token_id)
        if value is not None
    }
    if any(token in forbidden for token in token_ids[:-1]):
        raise ValueError("unified prefix cannot contain audio placeholders or EOA")
    if token_ids.count(int(boa_token_id)) != 1:
        raise ValueError("unified prefix must contain exactly one final BOA token")
    if not callable(getattr(backend, "build_semantic_prefix_cache", None)):
        raise TypeError("unified prefix requires the serving backend's ShapeStable cache builder")
    model_lock = getattr(backend, "_lock", None)
    if model_lock is None:
        raise TypeError("unified prefix requires the serving Thinker model lock")

    cached_prefix: PrefixCacheImport | None = None

    def import_prefix(state: ListenerSessionState, row: Any) -> PrefixCacheImport:
        nonlocal cached_prefix
        if (
            state.session_id != row.session_id
            or state.generation_id != row.session_generation
            or state.state_epoch_id != row.state_epoch_id
            or state.next_frame_id != 0
            or state.sequence_position != 0
            or row.frame_id != 0
            or row.sequence_position != 0
            or row.cache_position != len(token_ids)
            or row.valid_history_tokens != len(token_ids)
        ):
            raise RuntimeError("unified prefix import requires the pristine frame-zero session cursor")
        if cached_prefix is None:
            with torch.inference_mode():
                cache = backend.build_semantic_prefix_cache(
                    token_ids,
                    replay_q2=(),
                    replay_frame_ids=(),
                )
            prefix_len = int(cache["prefix_len"])
            cache_position = int(cache["cache_position"])
            replay_frame_ids = tuple(cache.get("replay_frame_ids", ()))
            if (
                prefix_len != len(token_ids)
                or cache_position != len(token_ids)
                or replay_frame_ids
            ):
                raise RuntimeError("unified prefix builder returned a non-prefix cache")
            cached_prefix = PrefixCacheImport(
                past_key_values=cache["past_key_values"],
                prefix_attention=cache["prefix_attention"],
                prefix_len=prefix_len,
                cache_position=cache_position,
                prefix_build_ms=float(cache["prefix_build_ms"]),
                replay_ms=0.0,
            )
        return cached_prefix

    # ShapeStable reads the same mutable Thinker workspaces as ordinary
    # thinker inference.  The worker takes a short model lock while it builds
    # this initial prefix and releases it before native listener execution.
    setattr(import_prefix, "requires_shared_lm_lock", True)
    return import_prefix


def initial_listener_session_state(
    session: Any,
    runtime_signature: RuntimeSignature,
    *,
    native_context_id: int,
    generation_id: int = 1,
    cache_position: int = CANONICAL_PREFIX_TOKENS,
) -> ListenerSessionState:
    """Map an empty serving listener epoch into a private batch row state.

    A supplied cache position may reflect the actual-length semantic prefix
    captured for this session before its first real listener frame.
    """
    state = session.listener_state
    listener_backend = getattr(getattr(session, "listener", None), "backend", None)
    with state._lock:
        if (
            state.next_frame_id != 0
            or state.listener_input_frames != 0
            or state.listener_state_version != 0
            or state.gemma_lm_audio_frames != 0
            or state.gemma_lm_cache is not None
            or state.gemma_lm_native is not None
            or state.gemma_lm_text_context
            or state.semantic_context_revision is not None
            or bool(state.semantic_audio_soft_history)
            or bool(state.text_token_ids)
            or bool(getattr(listener_backend, "text_token_ids", ()))
            or state.gemma_lm_terminal_error is not None
            or state.native_last_hidden is not None
            or state.user_pcm_history
            or state.assistant_pcm_history
            or state.user_mel_history
            or state.assistant_mel_history
        ):
            raise RuntimeError(
                "shared listener requires an empty listener epoch; canonical replay/reset "
                "for an existing semantic/audio prefix is not implemented"
            )
        return ListenerSessionState(
            session_id=str(session.session_id),
            generation_id=int(generation_id),
            state_epoch_id=str(state.state_epoch_id),
            runtime_signature=runtime_signature,
            next_frame_id=int(state.next_frame_id),
            state_version=int(state.listener_state_version),
            sequence_position=int(state.listener_input_frames),
            cache_position=int(cache_position),
            valid_history_tokens=int(cache_position),
            native_context_id=int(native_context_id),
            user_pcm_tail=_copy_tail(state.user_pcm_tail),
            assistant_pcm_tail=_copy_tail(state.assistant_pcm_tail),
            user_mel_history=[np.array(value, dtype=np.float32, copy=True) for value in state.user_mel_history],
            assistant_mel_history=[np.array(value, dtype=np.float32, copy=True) for value in state.assistant_mel_history],
        )


def semantic_prefix_importer(
    backend: Any,
    request_supplier: Any,
) -> Any:
    """Create an owner-thread importer bound to one immutable frame request."""
    if not callable(request_supplier):
        raise TypeError("request_supplier must return an immutable semantic prefix request")

    def import_prefix(state: ListenerSessionState, row: Any) -> PrefixCacheImport:
        request = request_supplier()
        if not isinstance(request, NativeControl6SemanticPrefixRequest):
            raise RuntimeError("semantic prefix request was not prepared before the tick")
        if (
            request.session_id != row.session_id
            or request.state_epoch_id != row.state_epoch_id
            or request.current_frame_id != row.frame_id
            or request.listener_state_version != row.listener_state_version
            or state.next_frame_id != row.frame_id
            or state.state_version != row.listener_state_version
            or state.cache_position != row.cache_position
        ):
            raise RuntimeError("semantic prefix request lost its current-frame compare-and-swap")
        cache = backend.build_semantic_prefix_cache(
            request.prefix_token_ids,
            replay_q2=request.replay_q2,
            replay_frame_ids=request.replay_frame_ids,
        )
        if (
            cache["prefix_len"] != request.prefix_len
            or cache["cache_position"] != request.cache_position
            or tuple(cache["replay_frame_ids"]) != request.replay_frame_ids
        ):
            raise RuntimeError("semantic prefix cache did not preserve its frozen cursor")
        return PrefixCacheImport(
            past_key_values=cache["past_key_values"],
            prefix_attention=cache["prefix_attention"],
            prefix_len=request.prefix_len,
            cache_position=request.cache_position,
            semantic_context_revision=request.semantic_context_revision,
            semantic_replay_fallback_reason=request.fallback_reason,
            prefix_build_ms=cache["prefix_build_ms"],
            replay_ms=cache["replay_ms"],
            replay_frame_ids=tuple(int(value) for value in request.replay_frame_ids),
        )

    # The semantic builder runs ShapeStable against the serving model's shared
    # buffers; the worker holds FairModelLock only while invoking this closure.
    setattr(import_prefix, "requires_shared_lm_lock", True)
    return import_prefix


def prepare_initial_semantic_prefix(
    backend: Any,
    model_lock: Any,
    request: NativeControl6SemanticPrefixRequest,
    *,
    listener_generation_id: int,
    expected_cache_position: int,
) -> PreparedInitialPrefix:
    """Build frame-zero semantic KV off-owner and return its stream event.

    ShapeStable Gemma uses shared mutable workspaces, so preparation and its
    stream completion are protected by the backend's fair model lock.  The
    worker owner later waits on ``ready_event`` and imports the result into the
    session's private native slot; it never runs this expensive model forward.
    """
    if not isinstance(request, NativeControl6SemanticPrefixRequest):
        raise TypeError("request must be an immutable native Control6 prefix snapshot")
    if request.current_frame_id != 0 or request.replay_q2 or request.replay_frame_ids:
        raise ValueError("initial semantic admission requires frame zero and no audio replay")
    if not callable(getattr(backend, "build_semantic_prefix_cache", None)):
        raise TypeError("backend must expose build_semantic_prefix_cache")
    if not callable(getattr(model_lock, "acquire_fair", None)):
        raise TypeError("model_lock must expose acquire_fair for semantic preparation")
    if (
        isinstance(expected_cache_position, bool)
        or not isinstance(expected_cache_position, int)
        or expected_cache_position < 0
    ):
        raise ValueError("expected_cache_position must be a non-negative integer")

    device = torch.device(backend.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("initial semantic-prefix admission requires the serving CUDA device")
    prepare_stream = torch.cuda.Stream(device=device, priority=0)
    acquired = bool(model_lock.acquire_fair("listener"))
    if not acquired:
        raise RuntimeError("semantic prefix preparation was cancelled by the fair model lock")
    try:
        with torch.cuda.stream(prepare_stream), torch.inference_mode():
            cache = backend.build_semantic_prefix_cache(
                request.prefix_token_ids,
                replay_q2=(),
                replay_frame_ids=(),
            )
            if (
                int(cache["prefix_len"]) != request.prefix_len
                or int(cache["cache_position"]) != request.prefix_len
                or tuple(cache["replay_frame_ids"])
            ):
                raise RuntimeError("prepared initial cache differs from the frozen semantic prefix")
            ready_event = torch.cuda.Event(enable_timing=False, blocking=False)
            ready_event.record(prepare_stream)
            # ShapeStable reuses mutable buffers after this call returns.  The
            # event alone does not stop another lock user from overwriting
            # those buffers before this stream has consumed them, so drain
            # only the preparation stream before releasing the shared lock.
            prepare_stream.synchronize()
    finally:
        model_lock.release()

    prefix = PrefixCacheImport(
        past_key_values=cache["past_key_values"],
        prefix_attention=cache["prefix_attention"],
        prefix_len=request.prefix_len,
        cache_position=request.prefix_len,
        semantic_context_revision=request.semantic_context_revision,
        semantic_replay_fallback_reason=request.fallback_reason,
        prefix_build_ms=float(cache["prefix_build_ms"]),
        replay_ms=float(cache["replay_ms"]),
        replay_frame_ids=tuple(int(value) for value in request.replay_frame_ids),
    )
    return PreparedInitialPrefix(
        prefix=prefix,
        ready_event=ready_event,
        session_id=request.session_id,
        listener_generation_id=int(listener_generation_id),
        state_epoch_id=request.state_epoch_id,
        semantic_context_revision=request.semantic_context_revision,
        expected_cache_position=int(expected_cache_position),
        frame_id=0,
        state_version=request.listener_state_version,
        sequence_position=0,
    )


def commit_initial_semantic_prefix_state(
    session: Any,
    request: NativeControl6SemanticPrefixRequest,
    prepared: PreparedInitialPrefix,
) -> bool:
    """Atomically apply an imported initial prefix if its snapshot is current.

    Lock order matches request construction: session generation, then listener
    state.  The owner import may succeed just as a newer semantic revision is
    committed, so this final check prevents the listener state from claiming a
    prefix that no longer matches the session context.
    """
    state = session.listener_state
    if (
        str(session.session_id) != request.session_id
        or prepared.session_id != request.session_id
        or prepared.semantic_context_revision != request.semantic_context_revision
        or prepared.state_epoch_id != request.state_epoch_id
        or prepared.frame_id != request.current_frame_id
        or prepared.state_version != request.listener_state_version
        or prepared.expected_cache_position != request.cache_position
        or prepared.prefix.prefix_len != request.prefix_len
        or prepared.prefix.cache_position != request.cache_position
        or request.current_frame_id != 0
        or request.replay_q2
        or request.replay_frame_ids
    ):
        return False

    with session._generation_lock:
        if int(session.semantic_context_revision) != request.semantic_context_revision:
            return False
        with state._lock:
            if (
                state.state_epoch_id != request.state_epoch_id
                or state.next_frame_id != request.current_frame_id
                or state.listener_state_version != request.listener_state_version
                or state.listener_input_frames != 0
                or state.gemma_lm_audio_frames != 0
                or state.semantic_audio_soft_history
            ):
                return False
            state.semantic_context_revision = request.semantic_context_revision
            state.semantic_replay_fallback_reason = request.fallback_reason
            state.gemma_lm_prefix_len = request.prefix_len
            state.gemma_lm_prefix_attention = prepared.prefix.prefix_attention.detach()
            state.gemma_lm_audio_frames = 0
            state.gemma_lm_source_frame_ids.clear()
            state.gemma_lm_context_start_frame = 0
            state.gemma_lm_last_source_frame_id = None
            state.gemma_lm_cache = None
            state.gemma_lm_shared_kv_states = None
            state.gemma_lm_native = None
            return True


async def prepare_initial_semantic_prefix_async(*args: Any, **kwargs: Any) -> PreparedInitialPrefix:
    """Run the expensive shared-model prefill without blocking the server loop."""
    return await asyncio.to_thread(prepare_initial_semantic_prefix, *args, **kwargs)


def _copy_tail(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    return np.array(value, dtype=np.float32, copy=True)


__all__ = [
    "InitialPrefixAdmission",
    "MICROBATCH_LIBRARY_ENV",
    "STABLE_TURN_POLICY_Q2_CONTEXT_ENV",
    "commit_initial_semantic_prefix_state",
    "initial_listener_session_state",
    "make_listener_microbatch_worker",
    "make_unified_prefix_import",
    "prepare_initial_semantic_prefix",
    "prepare_initial_semantic_prefix_async",
    "semantic_prefix_importer",
]
