import base64, json
import numpy as np
import torch
from types import SimpleNamespace
from fastapi.testclient import TestClient

from duplex_kernel.audio_stream import AUDIO_FRAME_SAMPLES
from duplex_kernel.server import create_app

class FakeTokenizer:
    def apply_chat_template(self, messages, **_): return messages[-1]["content"]
    def __call__(self, text, **_): return {"input_ids": np.array([[1]])}

class FakeEngine:
    loaded = True
    enforce_production_listener = False
    tokenizer = FakeTokenizer()
    def think(self, ids, **_): return {"text": "ciao mondo", "hidden": None, "prompt_len": 1}
    def v9_prefix(self, hidden, prompt_len): return None
    def talk(self, text, h_pfx, **kwargs):
        self.last_voice_conditioning = kwargs.get("voice_conditioning")
        return np.zeros(2 * AUDIO_FRAME_SAMPLES, dtype=np.float32)

def client(): return TestClient(create_app(FakeEngine()))

def test_models_and_validation():
    c = client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "gemma-4-omni-duplex"
    assert c.post("/v1/chat/completions", json={}).status_code == 422


def test_realtime_ingress_queue_holds_sixteen_frames():
    from duplex_kernel.scheduler import REALTIME_INGRESS_QUEUE_FRAMES

    assert REALTIME_INGRESS_QUEUE_FRAMES == 16


def test_shared_listener_runtime_is_ready_before_health_and_stops_at_shutdown(monkeypatch):
    from concurrent.futures import Future
    import duplex_kernel.server as server

    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH", "1")
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "0")
    backend = SimpleNamespace()
    engine = FakeEngine()
    engine.listener = SimpleNamespace(backend=backend)
    worker_instances = []
    dispatcher_instances = []

    class Worker:
        def __init__(self, listener, **_):
            self.listener = listener
            self._thread = SimpleNamespace(is_alive=lambda: True)
            self.owner_thread_id = 123
            self.fatal_error = None
            self.runtime_signature = object()
            self.stop_calls = 0
            worker_instances.append(self)

        def start(self):
            backend._listener_microbatch_runtime_info = {"enabled": True}
            future = Future()
            future.set_result(None)
            return future

        def stop(self, **_):
            self.stop_calls += 1
            future = Future()
            future.set_result(None)
            return future

    class Dispatcher:
        def __init__(self, *_args, **_kwargs):
            self.is_running = False
            self.fatal_error = None
            self.stop_calls = 0
            dispatcher_instances.append(self)

        def start(self):
            self.is_running = True

        def stop(self, **_):
            self.stop_calls += 1
            self.is_running = False

    monkeypatch.setattr(server, "make_listener_microbatch_worker", Worker)
    monkeypatch.setattr(server, "SharedListenerTickDispatcher", Dispatcher)

    with TestClient(create_app(engine)) as test_client:
        health = test_client.get("/health").json()
        assert health["ready"] is True
        assert health["listener_microbatch"]["runtime_ready"] is True
        assert health["listener_microbatch"]["error"] is None
        assert len(worker_instances) == len(dispatcher_instances) == 1
        assert worker_instances[0].listener is engine.listener

    assert worker_instances[0].stop_calls == 1
    assert dispatcher_instances[0].stop_calls == 1


def test_shared_listener_startup_error_is_reported_as_unready(monkeypatch):
    import duplex_kernel.server as server

    monkeypatch.setenv("DUPLEX_LISTENER_MICROBATCH", "1")
    monkeypatch.setenv("DUPLEX_UNIFIED_KV", "0")
    engine = FakeEngine()
    engine.listener = SimpleNamespace(backend=SimpleNamespace())

    def fail_worker_start(*_args, **_kwargs):
        raise RuntimeError("listener init failed")

    monkeypatch.setattr(server, "make_listener_microbatch_worker", fail_worker_start)
    with TestClient(create_app(engine)) as test_client:
        health = test_client.get("/health").json()
        assert health["ready"] is False
        assert health["listener_microbatch"]["runtime_ready"] is False
        assert "listener init failed" in health["listener_microbatch"]["error"]

def test_chat_audio_and_reusable_session():
    c = client()
    r = c.post("/v1/chat/completions", json={"messages":[{"role":"user","content":"ciao"}], "modalities":["text","audio"]})
    assert r.status_code == 200
    body = r.json(); assert body["choices"][0]["message"]["audio"]["data"]
    sid = body["session_id"]
    r2 = c.post("/v1/chat/completions", json={"session_id":sid,"messages":[{"role":"user","content":"ancora"}]})
    assert r2.status_code == 200
    assert r2.json()["choices"][0]["message"]["audio"]["data"]

def test_chat_default_is_gemma_voice_and_text_audio(monkeypatch):
    import duplex_kernel.server as server

    engine = FakeEngine()
    engine.talker = SimpleNamespace(tcfg=SimpleNamespace(hidden_size=4))
    calls = []
    conditioning = {
        "mode": "x_vector_only",
        "speaker_embedding": torch.ones(4),
        "ref_code": None,
    }

    def resolve_voice(voice, voice_style=None, ref_audio=None, ref_text=None):
        calls.append(voice)
        return dict(conditioning)

    monkeypatch.setattr(server, "resolve_voice_conditioning", resolve_voice)
    c = TestClient(create_app(engine))
    response = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "ciao"}]},
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["audio"]["data"]
    assert calls == ["gemma"]
    assert torch.equal(engine.last_voice_conditioning["speaker_embedding"], torch.ones(4))

    text_only = c.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "solo testo"}], "modalities": ["text"]},
    )
    assert text_only.status_code == 200
    assert "audio" not in text_only.json()["choices"][0]["message"]

def test_sse_emits_text_audio_and_done():
    r = client().post("/v1/chat/completions", json={"messages":[{"role":"user","content":"ciao"}],"modalities":["text","audio"],"stream":True})
    assert "data: [DONE]" in r.text
    assert '"audio"' in r.text


def test_server_diagnostics_helpers_remain_reexported():
    import duplex_kernel.server as server
    from duplex_kernel import server_diagnostics

    names = (
        "_attach_native_control6_trace_diagnostic",
        "_native_control6_event_annotations",
        "_native_control6_debug_input_row",
        "_write_native_control6_debug_input_sidecar",
        "_native_control6_selection_fields",
        "scheduler_tick_trace_metadata",
        "_acquire_shared_transition_lock_with_timing",
        "_queued_audio_drain_metric_fields",
        "_unified_native_finalizer_event",
        "_send_unified_native_finalizer_event",
        "_filter_unified_native_event_metrics",
        "native_q4_cuda_profile_host_context",
        "attach_native_q4_lock_metrics",
        "_filter_frame_residual_live_hidden_trace",
        "_filter_turn_local_speech_runtime_metrics",
        "_format_turn_local_speech_summary",
        "_submit_or_defer_audio_history_archive",
        "_filter_frame_residual_runtime_metrics",
        "_filter_native_thinker_lock_metrics",
        "_retain_archive_transcript_commitment",
        "native_listener_cuda_profile_effective_directory",
        "_quiesce_scheduler_for_native_profile",
        "write_native_listener_cuda_profile_sidecar",
        "_filter_native_talker_q1_graph_runtime",
    )
    assert all(getattr(server, name) is getattr(server_diagnostics, name) for name in names)


def test_server_realtime_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_realtime_policy, server_realtime_scheduler

    scheduler_names = (
        "_prepare_listener_tick_reference",
        "_evaluate_listener_silence_skip",
        "_record_listener_commit_timing",
        "_publish_listener_conditioning_row",
        "_build_input_audio_consumed_event",
    )
    policy_names = (
        "_prepare_policy_tick_state_impl",
        "_shared_native_policy_context_impl",
        "_classify_control6_policy_action",
        "_control6_transition_rejection",
    )
    assert all(
        getattr(server, name) is getattr(server_realtime_scheduler, name)
        for name in scheduler_names
    )
    assert all(
        getattr(server, name) is getattr(server_realtime_policy, name)
        for name in policy_names
    )

def test_tool_parser_contract():
    from duplex_kernel.tool_parser import extract_tool_calls
    tools = [{"type":"function", "function":{"name":"clock"}}]
    calls = extract_tool_calls('call:clock{"tz":"Europe/Rome"}', tools)
    assert json.loads(calls[0]["function"]["arguments"])["tz"] == "Europe/Rome"

def test_server_state_symbols_are_reexported():
    import duplex_kernel.server as server
    from duplex_kernel import server_state

    names = (
        "_ResponseAudioAdmissionState",
        "_configured_speak_endpoint_frames",
        "_CausalUserSpeechTracker",
        "_SchedulerEventBuffer",
        "_RealtimeResponseControl",
        "_EngineSynthesisOwner",
        "_InferenceExecutionGate",
        "POLICY_FRAME_MS",
        "DEFAULT_SPEAK_ENDPOINT_QUIET_MS",
    )
    assert all(getattr(server, name) is getattr(server_state, name) for name in names)

def test_server_unified_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_unified

    names = (
        "_strict_env_switch",
        "_listener_frame_is_voiced",
        "_unified_listener_frame_contract",
        "_unified_response_owns_shared_audio_queue",
        "_shared_listener_response_branch",
        "_shared_listener_frame_queue_contract",
        "_clear_shared_listener_response_branches",
        "_attach_shared_listener_response_branches",
        "_validate_private_transition_chunk",
        "_configured_audio_response_protocol",
        "_require_unified_stable_audio_protocol",
        "_tokenize_template_text",
        "_unified_audio_template_ids",
        "_unified_turn_end_token_id",
        "_validate_unified_audio_turn_end_suffix",
        "_build_unified_audio_response_suffix",
        "_prewarm_unified_v1_transition",
        "_unified_v1_transition_prewarm_key",
        "_prewarm_unified_v1_transition_after_import",
        "_build_unified_backchannel_response_suffix",
        "_build_unified_next_user_header",
        "_unified_end_of_turn_token_id",
        "_finalize_unified_native_cache_branch",
        "_consume_unified_native_text_append_result",
        "_run_unified_native_thinker",
    )
    assert all(getattr(server, name) is getattr(server_unified, name) for name in names)


def test_thinker_sampler_is_reexported_by_identity():
    import duplex_kernel.engine as engine
    from duplex_kernel import thinker_sampler

    assert getattr(engine, "ThinkerSampler") is getattr(thinker_sampler, "ThinkerSampler")


def test_server_speech_ack_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_speech_ack

    names = (
        "_validate_unified_say_packet_frontier",
        "_publish_thinker_metrics_event",
        "_record_unified_native_profile_timestamp",
        "_wait_for_qwen_dynamic_prefill_model_entry",
        "_mark_unified_say_packet_committed",
        "_feed_unified_v1_say_packet",
        "_publish_acknowledged_speech_prefix",
        "_record_client_consumed_pcm",
        "_commit_client_consumed_segment",
        "_record_emitted_pcm_packet",
        "_complete_client_pcm_segment",
        "_require_closed_tts_codec_eos",
        "_turn_local_vocalized_prefix_snapshot",
        "_wait_for_listener_backlog_admission",
        "_ack_emitted_audio_packet",
        "_response_speech_ack_lock",
        "_register_response_speech_ack_streams",
        "_response_speech_ack_resources",
        "_make_finish_audio_segment_callback",
        "_retire_response_speech_ack_streams",
        "_ack_response_speech_packet",
    )
    assert all(getattr(server, name) is getattr(server_speech_ack, name) for name in names)


def test_server_response_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_response, server_response_builder

    builder_names = (
        "_empty_controller_say_think_from_eos",
        "_claim_audio_response_v2_action_probe",
        "_is_v2_no_speech_result",
        "_record_audio_response_v2_private_failure",
        "_resolve_request_voice_conditioning",
        "_diagnostic_talker_rng_scope",
        "_append_native_action_token",
        "_prepare_direct_audio_processor_inputs",
        "_direct_audio_prompt_history",
        "_align_generated_response_hidden_with_diagnostics",
        "_qwen_text_token_surfaces",
        "_backchannel_budget_without_eos",
        "_publish_backchannel_tokens_after_eos",
        "_message_text",
    )
    assert all(
        getattr(server_response_builder, name)
        is getattr(server_response, name)
        is getattr(server, name)
        for name in builder_names
    )
    assert server.run_response is server_response.run_response

def test_split_server_helpers_retain_identity_across_modules():
    from duplex_kernel import (
        server_diagnostics,
        server_metrics,
        server_pcm_tracker,
        server_speech_ack,
        server_unified,
        server_unified_cache,
        server_unified_templates,
    )

    pcm_names = (
        "_record_client_consumed_pcm",
        "_commit_client_consumed_segment",
        "_record_emitted_pcm_packet",
        "_complete_client_pcm_segment",
        "_require_closed_tts_codec_eos",
        "_turn_local_vocalized_prefix_snapshot",
        "_wait_for_listener_backlog_admission",
    )
    assert all(
        getattr(server_speech_ack, name) is getattr(server_pcm_tracker, name)
        for name in pcm_names
    )

    template_names = (
        "_strict_env_switch",
        "_listener_frame_is_voiced",
        "_unified_listener_frame_contract",
        "_unified_response_owns_shared_audio_queue",
        "_shared_listener_response_branch",
        "_shared_listener_frame_queue_contract",
        "_clear_shared_listener_response_branches",
        "_attach_shared_listener_response_branches",
        "_validate_private_transition_chunk",
        "_configured_audio_response_protocol",
        "_require_unified_stable_audio_protocol",
        "_tokenize_template_text",
        "_unified_audio_template_ids",
        "_unified_turn_end_token_id",
        "_validate_unified_audio_turn_end_suffix",
        "_build_unified_audio_response_suffix",
        "_prewarm_unified_v1_transition",
        "_unified_v1_transition_prewarm_key",
        "_prewarm_unified_v1_transition_after_import",
        "_build_unified_backchannel_response_suffix",
        "_build_unified_next_user_header",
        "_unified_end_of_turn_token_id",
    )
    assert all(
        getattr(server_unified, name) is getattr(server_unified_templates, name)
        for name in template_names
    )
    assert (
        server_unified._finalize_unified_native_cache_branch
        is server_unified_cache._finalize_unified_native_cache_branch
    )

    metric_names = (
        "_acquire_shared_transition_lock_with_timing",
        "_queued_audio_drain_metric_fields",
        "_unified_native_finalizer_event",
        "_filter_unified_native_event_metrics",
        "native_q4_cuda_profile_host_context",
        "attach_native_q4_lock_metrics",
        "_filter_frame_residual_live_hidden_trace",
        "_filter_turn_local_speech_runtime_metrics",
        "_format_turn_local_speech_summary",
        "_submit_or_defer_audio_history_archive",
        "_filter_frame_residual_runtime_metrics",
        "_filter_native_thinker_lock_metrics",
        "_retain_archive_transcript_commitment",
        "native_listener_cuda_profile_effective_directory",
        "write_native_listener_cuda_profile_sidecar",
        "_filter_native_talker_q1_graph_runtime",
        "_alignment_failure_diagnostics",
        "_wait_for_emitted_tts_segment",
        "_signal_emitted_tts_segment",
        "_cancel_tts_segment_emission_waiters",
    )
    assert all(
        getattr(server_diagnostics, name) is getattr(server_metrics, name)
        for name in metric_names
    )


def test_server_audio_snapshot_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_audio_snapshot

    names = (
        "_response_admission_allowed_locked",
        "_controller_guided_say_generation_options",
        "_response_uses_direct_audio_candidate",
        "_require_direct_audio_context",
        "_native_control6_policy_hidden_for_tick",
        "_native_control6_speaker_active",
        "_apply_native_control6_pending_pcm_yield_gate",
        "_response_branch_has_pending_pcm",
        "_response_branch_has_pending_pcm_locked",
        "_pending_pcm_response_branches",
        "_cancel_response_branch_playback",
        "_wait_for_response_playback_drain",
        "_native_control6_endpoint_bypass_enabled",
        "_native_control6_decision_has_valid_input",
        "_native_control6_listener_source_frame_id",
        "_native_control6_listener_output_is_fresh_for_frame",
        "_normalize_chat_modalities",
        "ResponseOutputQueueFull",
        "_put_response_queue_bounded",
        "_response_projector_mode",
        "_module_inference_freeze_status",
        "_realtime_inference_failure_done_event",
        "_record_turn_local_vocalized_prefix",
        "_snapshot_direct_policy_audio",
        "_snapshot_speaking_branch_context",
        "_manual_commit_speak_owner_for_span",
        "DirectAudioSnapshotError",
        "_direct_audio_snapshot_error",
        "_direct_audio_snapshot_event_fields",
        "_resolve_p7_session_voice",
    )
    assert all(
        getattr(server, name) is getattr(server_audio_snapshot, name)
        for name in names
    )


def test_server_silence_skip_helpers_are_reexported_by_identity():
    import duplex_kernel.server as server
    from duplex_kernel import server_silence_skip

    names = (
        "_build_unified_listener_prefix_token_ids",
        "_unified_instruction_refresh_ineligible_reason",
        "_refresh_unified_prefix_for_instruction_update",
        "_apply_direct_replan_transaction",
        "_claim_response_cancel_notification",
        "_claim_pending_yielded_v2_pcm_cancel",
        "_listener_evidence_role",
        "_yield_admission_rejection",
        "_advance_pending_speak_on_new_onset",
        "_silence_skip_inputs_are_exact_zero",
        "_user_turn_is_unanswered",
        "_silence_skip_is_true_idle",
        "_silence_skip_transition_is_still_idle",
        "_scheduler_cuda_stream_context",
        "_listener_v9_stream_for_engine",
        "_silence_skip_query_is_coherent",
        "_wav_b64",
        "_raw_source_audio_quality",
        "_audio_packet_matches_emission_owner",
        "_release_pcm_packet_then_ack",
    )
    assert all(
        getattr(server, name) is getattr(server_silence_skip, name)
        for name in names
    )


def test_server_and_engine_compatibility_exports():
    import duplex_kernel.server as server
    import duplex_kernel.engine as engine
    from duplex_kernel.engine_types import ListenerBacklogAdmission
    from duplex_kernel.tts_quality_diagnostics import TTSQualityDiagnosticCapture
    from duplex_kernel.listener_microbatch_serving import _semantic_prefix_request_matches_applied_row
    from duplex_kernel.server_response_builder import (
        _ACTION_TOKEN_MAP,
        _AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK,
        _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED,
    )
    from duplex_kernel.audio_response_protocol import AUDIO_RESPONSE_ACTIONS

    assert server.ListenerBacklogAdmission is ListenerBacklogAdmission
    assert server.TTSQualityDiagnosticCapture is TTSQualityDiagnosticCapture
    assert server._semantic_prefix_request_matches_applied_row is _semantic_prefix_request_matches_applied_row
    assert server._ACTION_TOKEN_MAP is _ACTION_TOKEN_MAP
    assert server._AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK is _AUDIO_RESPONSE_V2_ACTION_PROBE_LOCK
    assert server._AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED is _AUDIO_RESPONSE_V2_ACTION_PROBE_CLAIMED
    assert engine.AUDIO_RESPONSE_ACTIONS is AUDIO_RESPONSE_ACTIONS


def test_assistant_open_queue_keeps_all_frames_in_canonical_q2():
    from duplex_kernel.server_unified_templates import _unified_listener_frame_contract

    silent = np.zeros(1280, dtype=np.float32)
    voiced = np.full(1280, 0.02, dtype=np.float32)
    assert _unified_listener_frame_contract(
        silent, assistant_audio_queue_open=True, user_input_pending=False,
    ) == (2, None, False)
    assert _unified_listener_frame_contract(
        voiced, assistant_audio_queue_open=True, user_input_pending=False,
    ) == (2, None, False)
    # A pending onset does not change the continuous canonical audio contract.
    assert _unified_listener_frame_contract(
        silent, assistant_audio_queue_open=True, user_input_pending=True,
    ) == (2, None, False)


def test_submit_tick_normalizes_a_valid_session_stale_assistant_q0():
    from concurrent.futures import Future

    from duplex_kernel.listener_microbatch_worker import (
        ListenerMicrobatchWorker,
        _Command,
    )
    from duplex_kernel.microbatch_contract import ListenerFrame

    enqueued = []
    state = SimpleNamespace(
        generation_id=3,
        state_epoch_id="epoch",
        next_frame_id=0,
        cache_position=9,
        closed=False,
        poisoned_reason=None,
    )
    worker = object.__new__(ListenerMicrobatchWorker)
    worker._tick_future = None
    worker._assistant_audio_queue_open = set()
    worker._session_states = {"s1": state}
    worker._coordinator = SimpleNamespace(
        coalesce_window_ns=80,
        ready_depth=0,
        capability=SimpleNamespace(max_seq_len=16_384),
        enqueue=enqueued.append,
    )
    worker._shadow_worker = None
    worker._outstanding_frames = {}
    worker._outstanding_logical_tokens = {}
    worker._trace_control6_queue_handoff = lambda *_args, **_kwargs: None
    frame = ListenerFrame(
        session_id="s1",
        generation_id=3,
        state_epoch_id="epoch",
        frame_id=0,
        ready_at_ns=1,
        pcm=np.zeros(1280, dtype=np.float32),
        logical_q_len=0,
        ephemeral_reason="assistant_turn_open",
        queue_audio_for_next_turn=True,
    )
    future = Future()

    worker._execute_command(_Command("submit_tick", ((frame,),), future))

    assert len(enqueued) == 1
    assert enqueued[0].queue_audio_for_next_turn is False
    assert enqueued[0].logical_q_len == 2
    assert enqueued[0].ephemeral_reason is None
    assert worker._tick_expected == {"s1": 0}


def test_closed_assistant_q0_rows_are_downgraded_and_warned(caplog):
    from duplex_kernel.listener_microbatch_worker import (
        _normalize_stale_assistant_queue_rows,
    )
    from duplex_kernel.microbatch_contract import BatchRowDescriptor

    row = BatchRowDescriptor(
        session_id="s1",
        session_generation=3,
        state_epoch_id="epoch",
        frame_id=7,
        listener_state_version=2,
        sequence_position=4,
        cache_position=9,
        valid_history_tokens=9,
        native_context_id=1,
        input_frame_sha256="0" * 64,
        reference_pcm_sha256=None,
        frame_ready_at_ns=1,
        reference_frame_id=None,
        logical_q_len=0,
        ephemeral_reason="assistant_turn_open",
        queue_audio_for_next_turn=True,
    )

    with caplog.at_level("WARNING"):
        normalized, stale_sessions = _normalize_stale_assistant_queue_rows(
            (row,), queue_open_sessions=set(), valid_sessions={"s1"},
        )

    assert stale_sessions == {"s1"}
    assert normalized[0].queue_audio_for_next_turn is False
    assert normalized[0].logical_q_len == 0
    assert "stale assistant-phase Q0" in caplog.text


def test_shared_listener_row_callback_error_does_not_stop_other_sessions():
    from duplex_kernel.shared_listener_scheduler import SharedListenerTickDispatcher

    outputs_seen = []
    errors_seen = []
    failure = RuntimeError("single-session callback failure")
    dispatcher = object.__new__(SharedListenerTickDispatcher)
    dispatcher._fatal_error = None
    dispatcher._on_error = lambda row, error, _receipt: errors_seen.append(
        (row.session_id, error)
    )

    def on_result(row, output, _receipt):
        outputs_seen.append(row.session_id)
        if row.session_id == "s1":
            raise failure

    dispatcher._on_result = on_result
    rows = (SimpleNamespace(session_id="s1"), SimpleNamespace(session_id="s2"))
    dispatcher._fanout_results(rows, {"s1": 1, "s2": 2}, receipt=object())

    assert outputs_seen == ["s1", "s2"]
    assert errors_seen == [("s1", failure)]
    assert dispatcher._fatal_error is None


def test_shared_listener_marks_only_irrecoverable_gpu_errors_process_fatal():
    from duplex_kernel.shared_listener_scheduler import _is_irrecoverable_gpu_error

    assert _is_irrecoverable_gpu_error(RuntimeError("session row validation failed")) is False
    assert _is_irrecoverable_gpu_error(
        RuntimeError("CUDA error: an illegal memory access was encountered")
    ) is True


def test_shared_listener_result_error_isolates_session_and_keeps_dispatcher_usable():
    import queue

    from duplex_kernel.scheduler import AudioFrame
    from duplex_kernel.shared_listener_scheduler import SharedListenerTickDispatcher

    class Clock:
        def __init__(self) -> None:
            self.now = 0

        def now_ns(self) -> int:
            return self.now

        def wait_until(self, deadline_ns: int) -> None:
            self.now = max(self.now, deadline_ns)

    def frame(frame_id: int, enqueue_ns: int) -> AudioFrame:
        return AudioFrame(
            frame_id=frame_id,
            pcm=np.zeros(1280, dtype=np.float32),
            capture_start_ns=max(0, enqueue_ns - 80_000_000),
            capture_end_ns=enqueue_ns,
            enqueue_ns=enqueue_ns,
        )

    clock = Clock()
    ingresses = {session_id: queue.Queue() for session_id in ("s1", "s2")}
    for frame_id, enqueue_ns in enumerate((70_000_000, 150_000_000, 230_000_000)):
        for ingress in ingresses.values():
            ingress.put(frame(frame_id, enqueue_ns))

    session_error = RuntimeError("session-local batch failure")
    session_error.session_id = "s1"
    batch_error = RuntimeError("batch-wide listener failure")
    batch_calls = 0

    def batch_listener(rows):
        nonlocal batch_calls
        batch_calls += 1
        if batch_calls == 1:
            raise session_error
        if batch_calls == 2:
            raise batch_error
        return {
            row.session_id: SimpleNamespace(frame_id=row.frame.frame_id)
            for row in rows
        }

    errors_seen = []
    results_seen = []
    dispatcher = SharedListenerTickDispatcher(
        batch_listener,
        on_result=lambda row, _output, _receipt: results_seen.append(row.session_id),
        on_error=lambda row, error, _receipt: errors_seen.append((row.session_id, error)),
        now_ns=clock.now_ns,
        wait_until=clock.wait_until,
    )
    for session_id, ingress in ingresses.items():
        dispatcher.register_session(session_id, ingress)

    fanout_calls = []
    original_fanout_errors = dispatcher._fanout_errors

    def record_fanout(rows, error, receipt):
        fanout_calls.append((tuple(row.session_id for row in rows), error))
        original_fanout_errors(rows, error, receipt)

    dispatcher._fanout_errors = record_fanout

    first = dispatcher.dispatch_next_tick()

    assert first.batch_size == 2
    assert first.error == "RuntimeError: session-local batch failure"
    assert fanout_calls == [(('s1',), session_error)]
    assert errors_seen == [("s1", session_error)]
    assert dispatcher._fatal_error is None

    second = dispatcher.dispatch_next_tick()

    assert second.batch_size == 2
    assert second.error == "RuntimeError: batch-wide listener failure"
    assert fanout_calls[1] == (("s1", "s2"), batch_error)
    assert errors_seen[1:] == [("s1", batch_error), ("s2", batch_error)]
    assert dispatcher._fatal_error is None

    third = dispatcher.dispatch_next_tick()

    assert third.batch_size == 2
    assert third.error is None
    assert results_seen == ["s1", "s2"]
    assert dispatcher._fatal_error is None
