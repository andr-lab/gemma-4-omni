import asyncio
import base64
from collections import deque
import io
import json
import time
import threading
import wave

import numpy as np
import pytest

from client import duplex_stream_e2e
from client.duplex_stream_e2e import RealtimePairProbe
from client.duplex_stream_e2e import SingleStreamProbe


def test_followup_audio_delay_starts_from_first_response_done():
    done_ns = 9_000_000_000
    delay_ms = 1_000

    assert not duplex_stream_e2e._followup_after_response_done_due(
        None, done_ns + 2_000_000_000, delay_ms,
    )
    assert not duplex_stream_e2e._followup_after_response_done_due(
        done_ns, done_ns + 999_999_999, delay_ms,
    )
    assert duplex_stream_e2e._followup_after_response_done_due(
        done_ns, done_ns + 1_000_000_000, delay_ms,
    )


def test_isolated_bargein_observation_waits_from_last_injected_sample():
    due = duplex_stream_e2e._bargein_observation_due

    assert not due("enqueued", 10_000_000_000, 18_000_000_000, 8.0)
    assert not due("sent", None, 18_000_000_000, 8.0)
    assert not due("sent", 10_000_000_000, 17_999_999_999, 8.0)
    assert due("sent", 10_000_000_000, 18_000_000_000, 8.0)


def test_isolated_observation_sender_stops_with_nonterminal_active_response(tmp_path):
    async def run():
        probe = RealtimePairProbe(
            "ws://unused", tmp_path / "isolated-cleanup", max_turns=1,
            max_duration_s=30, bargein_wav=tmp_path / "correction.wav",
            bargein_isolated_observation_s=8,
        )
        probe.isolated_observation_complete = True
        peer_a = {
            "speaker": "A", "ws": object(), "queue": asyncio.Queue(),
            "ack_pending": deque([(1, time.monotonic_ns())]), "ack_wakeup": asyncio.Event(),
            "stop_sender": True, "active": {"A-long-response"}, "route_pending": 0,
        }
        peer_b = {
            "speaker": "B", "ws": object(), "queue": asyncio.Queue(),
            "ack_pending": deque(), "ack_wakeup": asyncio.Event(),
            "stop_sender": True, "active": set(), "route_pending": 0,
        }
        probe.peers = {"A": peer_a, "B": peer_b}

        async def acknowledge_after_drain_starts():
            await asyncio.sleep(0.01)
            peer_a["ack_pending"].clear()
            peer_a["ack_wakeup"].set()

        ack_task = asyncio.create_task(acknowledge_after_drain_starts())
        await asyncio.wait_for(probe._send_loop(peer_a), timeout=0.1)
        await ack_task
        assert not peer_a["ack_pending"]

    asyncio.run(run())


def test_isolated_observation_completion_disables_auto_response_after_window(tmp_path):
    probe = RealtimePairProbe(
        "ws://unused", tmp_path / "isolated-observation-deadline", max_turns=1,
        max_duration_s=30, bargein_wav=tmp_path / "correction.wav",
        bargein_isolated_observation_s=8,
    )
    peer_a = {
        "speaker": "A", "ws": object(), "queue": asyncio.Queue(),
        "ack_pending": deque(), "ack_wakeup": asyncio.Event(),
        "stop_sender": False, "active": {"A-long-response"}, "route_pending": 0,
    }
    peer_b = {
        "speaker": "B", "ws": object(), "queue": asyncio.Queue(),
        "ack_pending": deque(), "ack_wakeup": asyncio.Event(),
        "stop_sender": False, "active": set(), "route_pending": 0,
    }
    probe.peers = {"A": peer_a, "B": peer_b}
    disable_requests = []
    probe._request_auto_response_disable = lambda reason: disable_requests.append(
        (reason, time.monotonic_ns())
    )
    last_sent_ns = time.monotonic_ns() - 8_000_000_001
    injection = {"status": "sent", "last_sent_at_ns": last_sent_ns}

    assert not duplex_stream_e2e._bargein_observation_due(
        "sent", last_sent_ns, last_sent_ns + 7_999_999_999, 8.0,
    )
    assert duplex_stream_e2e._bargein_observation_due(
        "sent", last_sent_ns, last_sent_ns + 8_000_000_000, 8.0,
    )
    probe._complete_isolated_bargein_observation(injection)

    assert probe.isolated_observation_complete
    assert probe.isolated_observation_completed_at_ns >= last_sent_ns + 8_000_000_000
    assert probe.bridge_stopped
    assert probe.stop_senders_requested
    assert all(peer["stop_sender"] for peer in (peer_a, peer_b))
    assert disable_requests[0][0] == "isolated_bargein_observation_complete"
    assert disable_requests[0][1] >= last_sent_ns + 8_000_000_000


def test_raw_audio_quality_reports_original_pcm_rails_without_clipping():
    pcm = np.asarray([-1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25], dtype=np.float32)

    quality = duplex_stream_e2e.raw_audio_quality(pcm)

    assert quality == {
        "finite": True,
        "sample_count": 7,
        "peak": 1.25,
        "rms": round(float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))))), 8),
        "clipping_rail_samples": 4,
    }


def test_concurrent_tick_summary_keeps_warmup_and_full_deadline_spans_separate(tmp_path):
    events = [
        {
            "type": "input_audio_buffer.consumed", "frame_id": 0,
            "listener_batch_id": "batch-0",
            "scheduler_deadline_ns": 100_000_000,
            "scheduler_listener_end_ns": 120_000_000,
            "scheduler_deadline_lateness_ms": 2.0,
            "listener_batch_size": 1,
            "listener_batch_compute_ms": 14.0,
            "listener_batch_mel_prepare_ms": 3.0,
            "listener_batch_frontend_copy_cuda_ms": 1.25,
            "listener_batch_audio_tower_workspace_lock_wait_ms": 0.1,
            "listener_batch_audio_tower_workspace_lock_hold_ms": 0.9,
            "listener_batch_audio_tower_row_copy_cuda_ms": 0.4,
            "listener_batch_audio_tower_graph_submit_ms": 2.2,
            "listener_batch_audio_tower_cuda_ms": 9.5,
            "listener_batch_audio_tower_execution_mode": "cuda_graph_n1",
            "listener_batch_semantic_prefix_request_ms": 0.5,
        },
        {
            "type": "duplex_control6.decision", "frame_id": 0,
            "listener_batch_id": "batch-0",
            "policy_decision_at_ns": 125_000_000,
        },
        {
            "type": "input_audio_buffer.consumed", "frame_id": 1,
            "listener_batch_id": "batch-1",
            "scheduler_deadline_ns": 200_000_000,
            "scheduler_listener_end_ns": 310_000_000,
            "scheduler_deadline_lateness_ms": 8.0,
            "listener_batch_size": 1,
            "listener_batch_compute_ms": 91.0,
        },
        {
            "type": "duplex_control6.decision", "frame_id": 1,
            "listener_batch_id": "batch-1",
            "policy_decision_at_ns": 320_000_000,
        },
    ]
    event_path = tmp_path / "events.jsonl"
    event_path.write_text("\n".join(json.dumps(row) for row in events), encoding="utf-8")

    summary = duplex_stream_e2e.summarize_listener_tick_events(
        [event_path], warmup_ticks_per_session=1,
    )

    assert summary["total_ticks"] == 2
    assert summary["batch_size_histogram"] == {"1": 2}
    assert summary["warmup"]["ticks"] == 1
    assert summary["warmup"]["metrics"]["tick_deadline_to_listener_end_ms"]["max_ms"] == 20.0
    steady = summary["steady_state"]["metrics"]
    assert steady["tick_deadline_to_listener_end_ms"]["max_ms"] == 110.0
    assert steady["tick_deadline_to_listener_end_ms"]["over_80ms"] == 1
    assert steady["tick_deadline_to_policy_decision_ms"]["max_ms"] == 120.0
    warmup_metrics = summary["batch_timing"]["warmup"]["metrics"]
    assert warmup_metrics["listener_batch_mel_prepare_ms"]["p50_ms"] == 3.0
    assert warmup_metrics["listener_batch_frontend_copy_cuda_ms"]["p50_ms"] == 1.25
    assert warmup_metrics["listener_batch_audio_tower_row_copy_cuda_ms"]["p50_ms"] == 0.4
    assert warmup_metrics["listener_batch_audio_tower_graph_submit_ms"]["p50_ms"] == 2.2
    assert warmup_metrics["listener_batch_audio_tower_cuda_ms"]["p50_ms"] == 9.5
    assert warmup_metrics["listener_batch_semantic_prefix_request_ms"]["p50_ms"] == 0.5


def test_batch_summary_uses_last_row_decision_and_counts_shared_phases_once(tmp_path):
    session_a = tmp_path / "session_a.jsonl"
    session_b = tmp_path / "session_b.jsonl"
    session_c = tmp_path / "session_c.jsonl"
    ack_common = {
        "type": "input_audio_buffer.consumed",
        "listener_batch_id": "tick-7",
        "listener_batch_size": 2,
        "scheduler_deadline_ns": 100_000_000,
        "scheduler_observed_ns": 112_000_000,
        "scheduler_deadline_lateness_ms": 12.0,
        "listener_batch_native_submit_ms": 5.0,
        "listener_batch_audio_tower_cuda_ms": 9.0,
        "listener_batch_native_control6_raw_score_ms": 6.5,
        "listener_batch_native_output_health_ms": 0.4,
        "listener_batch_semantic_prefix_build_ms": 3.0,
        "listener_batch_semantic_prefix_replay_ms": 1.5,
        "listener_batch_semantic_prefix_import_ms": 0.8,
        "listener_batch_semantic_prefix_refresh_event": True,
        "listener_batch_semantic_prefix_refresh_event_id": "tick-7:session-a:epoch-a:4:9",
        "listener_batch_semantic_prefix_refresh_revision": 9,
        "listener_batch_semantic_prefix_len": 141,
        "listener_batch_native_output_health_checked": True,
        "listener_batch_native_output_health_status": "healthy",
        "listener_batch_native_output_health_nonfinite": False,
        "listener_batch_native_output_health_generation": 21,
        "listener_batch_native_output_health_session_generation": 3,
        "listener_batch_native_output_health_state_epoch_id": "epoch-x",
        "listener_batch_native_output_health_row_frame_id": 37,
        "listener_batch_native_output_health_row_state_version": 8,
        "listener_batch_native_output_health_row_sequence_position": 4,
        "listener_batch_native_output_health_row_cache_position": 141,
        "listener_batch_native_output_health_native_context_id": 9001,
        "logical_q_len": 0,
        "kv_rollback_tokens": 2,
        "kv_rollback_enqueue_ms": 0.05,
        "listener_cache_position_before": 141,
        "listener_cache_position_after": 141,
        "listener_valid_history_tokens_before": 141,
        "listener_valid_history_tokens_after": 141,
        "native_cache_current_position_after": 141,
        "native_cache_valid_kv_len_after": 141,
        "frame_id": 37,
    }
    decision_common = {
        "type": "duplex_control6.decision",
        "listener_batch_id": "tick-7",
        "turn_policy_id": "gemma_native_control6",
        "native_control6_step_called": True,
        "native_control6_head_scored": True,
        "native_control6_raw_token": "SPEAK",
        "native_control6_selected_token": "SPEAK",
        "native_action": "SPEAK",
        "selected_action": "SPEAK",
        "gate_applied_action": "SPEAK",
        "accepted": True,
        "checkpoint_path": "control6.pt",
    }
    session_a.write_text("\n".join(json.dumps(row) for row in (
        {
            **ack_common,
            "frame_id": 4,
            "listener_batch_host_lock_timeline": {
                "enabled": True,
                "status": "complete",
                "batch_id": "tick-7",
                "capacity": 256,
                "events": [
                    {"trace_seq": 41, "event": "acquire", "priority": "compute", "ticket": 12},
                    {"trace_seq": 42, "event": "release", "priority": "compute", "ticket": 12, "hold_ns": 239_000_000},
                ],
            },
            "listener_batch_native_output_health_row_frame_id": 4,
            "listener_batch_native_output_health_row_state_version": 12,
            "listener_batch_native_output_health_session_generation": 2,
            "listener_batch_native_output_health_state_epoch_id": "epoch-a",
            "listener_batch_native_output_health_row_sequence_position": 5,
            "listener_batch_native_output_health_row_cache_position": 161,
            "listener_batch_native_output_health_native_context_id": 101,
        },
        {**decision_common, "frame_id": 4, "policy_decision_at_ns": 130_000_000},
        {
            "type": "response.created", "response_id": "auto-1", "action": "SPEAK",
            "intent_source": "direct_audio_policy", "policy_decision_frame_id": 4,
            "policy_decision_at_ns": 130_000_000,
        },
        {"type": "response.audio.delta", "response_id": "auto-1"},
        {"type": "response.created", "response_id": "api-1", "action": "SPEAK"},
        {"type": "response.audio.delta", "response_id": "api-1"},
    )), encoding="utf-8")
    session_b.write_text("\n".join(json.dumps(row) for row in (
        {
            **ack_common,
            "frame_id": 9,
            "listener_batch_semantic_prefix_build_ms": None,
            "listener_batch_semantic_prefix_replay_ms": None,
            "listener_batch_semantic_prefix_import_ms": None,
            "listener_batch_semantic_prefix_refresh_event": False,
            "listener_batch_semantic_prefix_refresh_event_id": None,
            "listener_batch_semantic_prefix_refresh_revision": None,
            "listener_batch_semantic_prefix_len": 181,
            "listener_batch_native_output_health_row_frame_id": 9,
            "listener_batch_native_output_health_row_state_version": 25,
            "listener_batch_native_output_health_session_generation": 8,
            "listener_batch_native_output_health_state_epoch_id": "epoch-b",
            "listener_batch_native_output_health_row_sequence_position": 1,
            "listener_batch_native_output_health_row_cache_position": 34,
            "listener_batch_native_output_health_native_context_id": 202,
        },
        {**decision_common, "frame_id": 9, "policy_decision_at_ns": 147_000_000},
    )), encoding="utf-8")
    session_c.write_text(json.dumps({
        "type": "input_audio_buffer.consumed",
        "frame_id": 2,
        "listener_batch_id": "tick-incomplete",
        "listener_batch_size": 2,
        "scheduler_deadline_ns": 180_000_000,
        "listener_batch_error": "policy callback failed",
    }), encoding="utf-8")

    summary = duplex_stream_e2e.summarize_listener_tick_events(
        [session_a, session_b, session_c], warmup_ticks_per_session=0,
    )

    batch_timing = summary["batch_timing"]
    assert batch_timing["total_batches"] == 2
    assert batch_timing["batch_size_histogram"] == {"2": 2}
    assert batch_timing["all"]["incomplete_batches"] == 1
    batch = batch_timing["batches"][0]
    assert batch["observed_ack_rows"] == 2
    assert batch["complete"] is True
    assert batch["decision_rows"] == 2
    assert batch["tick_deadline_to_last_policy_decision_ms"] == 47.0
    assert batch["scheduler_deadline_lateness_ms"] == 12.0
    assert batch["late_rows_at_dispatch"] == 2
    assert batch_timing["steady_state"]["metrics"]["listener_batch_audio_tower_cuda_ms"]["samples"] == 1
    assert batch_timing["steady_state"]["metrics"]["listener_batch_native_submit_ms"]["p50_ms"] == 5.0
    assert batch_timing["steady_state"]["metrics"]["listener_batch_native_control6_raw_score_ms"]["samples"] == 1
    assert batch_timing["steady_state"]["metrics"]["listener_batch_native_control6_raw_score_ms"]["p50_ms"] == 6.5
    assert batch_timing["steady_state"]["metrics"]["listener_batch_native_output_health_ms"]["p50_ms"] == 0.4
    assert batch_timing["steady_state"]["metrics"]["listener_batch_semantic_prefix_build_ms"]["p50_ms"] == 3.0
    assert batch_timing["steady_state"]["metrics"]["listener_batch_semantic_prefix_replay_ms"]["p50_ms"] == 1.5
    assert batch_timing["steady_state"]["metrics"]["listener_batch_semantic_prefix_import_ms"]["p50_ms"] == 0.8
    assert batch["semantic_prefix_refresh_events"] == [{
        "event_file": str(session_a),
        "frame_id": 4,
        "event_id": "tick-7:session-a:epoch-a:4:9",
        "revision": 9,
        "prefix_len": 141,
    }]
    assert batch["native_output_health"] == {
        "checked_rows": 2,
        "unchecked_rows": 0,
        "nonfinite_rows": 0,
        "valid_provenance_rows": 2,
        "invalid_provenance_rows": 0,
        "statuses": ["healthy"],
        "generations": [21],
        "consistent": True,
    }
    assert batch["host_lock_timeline_snapshot_count"] == 1
    assert batch["host_lock_timeline_snapshot"]["batch_id"] == "tick-7"
    assert batch["host_lock_timeline_snapshot"]["events"][-1]["hold_ns"] == 239_000_000
    assert batch["policy_decisions"][1]["native_control6_head_scored"] is True
    assert batch["policy_decisions"][1]["native_control6_selected_token"] == "SPEAK"
    assert batch["policy_decisions"][1]["gate_applied_action"] == "SPEAK"
    incomplete = batch_timing["batches"][1]
    assert incomplete["complete"] is False
    assert incomplete["missing_ack_rows"] == 1
    assert incomplete["tick_deadline_to_last_policy_decision_ms"] is None
    assert {row["response_id"]: row["trigger"] for row in summary["response_audio_provenance"]} == {
        "auto-1": "automatic_policy",
        "api-1": "explicit_api",
    }


def test_concurrent_runner_surfaces_session_setup_failure_before_start_barrier(monkeypatch, tmp_path):
    class FailedProbe:
        def __init__(self, *_args, **_kwargs):
            pass

        async def run(self, *_args, **_kwargs):
            raise RuntimeError("session update rejected: adapter not loaded")

    monkeypatch.setattr(duplex_stream_e2e, "SingleStreamProbe", FailedProbe)

    with pytest.raises(RuntimeError, match="session update rejected: adapter not loaded"):
        asyncio.run(duplex_stream_e2e.run_concurrent_single_sessions(
            "ws://unused", tmp_path / "concurrent", tmp_path / "unused.wav", None, 1.0,
            session_count=1, warmup_ticks_per_session=0,
        ))


def _write_two_frame_wav(path):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((np.ones(2 * 1280, dtype="<i2") * 1200).tobytes())


def test_concurrent_runner_keeps_receiver_open_until_delayed_final_ack(tmp_path, monkeypatch):
    probes = []

    class RecordingProbe(SingleStreamProbe):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            probes.append(self)

    class FakeWebSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.append_count = 0
            self.closed_pending_acks = None
            self.last_ack_received = False

        async def __aenter__(self):
            self.incoming.put_nowait({
                "type": "session.created",
                "session_id": "concurrent-final-ack",
                "input_audio_buffer_capacity_frames": 6,
            })
            return self

        async def __aexit__(self, *_args):
            return None

        async def recv(self):
            return json.dumps(await self.incoming.get())

        async def send(self, payload):
            event = json.loads(payload)
            if event.get("type") == "session.update":
                self.incoming.put_nowait({
                    "type": "session.updated",
                    "turn_policy": "gemma_native_control6",
                    "voice_conditioning_mode": "x_vector_only",
                })
            elif event.get("type") == "input_audio_buffer.append":
                self.append_count += 1
                frame_id = self.append_count - 1
                if self.append_count == 1:
                    self.incoming.put_nowait({"type": "response.created", "response_id": "r1"})
                    self.incoming.put_nowait({"type": "response.done", "response_id": "r1"})

                async def deliver_ack():
                    if frame_id == 1:
                        await asyncio.sleep(0.12)
                    self.incoming.put_nowait({
                        "type": "input_audio_buffer.consumed",
                        "frame_id": frame_id,
                    })
                    if frame_id == 1:
                        self.last_ack_received = True

                asyncio.create_task(deliver_ack())
            elif event.get("type") == "session.close":
                self.closed_pending_acks = len(probes[0].ack_sent)

    websocket = FakeWebSocket()
    wav_path = tmp_path / "two-frames.wav"
    _write_two_frame_wav(wav_path)
    monkeypatch.setattr(duplex_stream_e2e, "SingleStreamProbe", RecordingProbe)
    monkeypatch.setattr(duplex_stream_e2e.websockets, "connect", lambda *_a, **_k: websocket)

    report = asyncio.run(duplex_stream_e2e.run_concurrent_single_sessions(
        "ws://unused", tmp_path / "concurrent", wav_path, None, 0.16,
        session_count=1, warmup_ticks_per_session=0,
    ))

    row = report["session_reports"][0]
    assert websocket.last_ack_received is True
    assert websocket.closed_pending_acks == 0
    assert row["input_frames_sent"] == row["input_frames_acked"] == 2
    assert row["final_ack_drain_complete"] is True
    assert row["pending_after_close"] == 0
    assert report["final_ack_drain_complete"] is True
    assert report["pending_after_close"] == 0
    assert report["all_sessions_healthy"] is True


def test_concurrent_runner_reports_final_ack_drain_timeout_as_unhealthy(tmp_path, monkeypatch):
    class FastDrainProbe(SingleStreamProbe):
        async def _wait_for_final_ack_drain(
            self, pending_acks_drained, *, timeout_s=None, stop_event=None,
        ):
            del timeout_s
            return await super()._wait_for_final_ack_drain(
                pending_acks_drained, timeout_s=0.01, stop_event=stop_event,
            )

    class NoAckWebSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.closed_pending_acks = None
            self.probe = None

        async def __aenter__(self):
            self.incoming.put_nowait({
                "type": "session.created",
                "session_id": "concurrent-ack-timeout",
                "input_audio_buffer_capacity_frames": 6,
            })
            return self

        async def __aexit__(self, *_args):
            return None

        async def recv(self):
            return json.dumps(await self.incoming.get())

        async def send(self, payload):
            event = json.loads(payload)
            if event.get("type") == "session.update":
                self.incoming.put_nowait({
                    "type": "session.updated",
                    "turn_policy": "gemma_native_control6",
                    "voice_conditioning_mode": "x_vector_only",
                })
            elif event.get("type") == "input_audio_buffer.append":
                self.incoming.put_nowait({"type": "response.created", "response_id": "r1"})
                self.incoming.put_nowait({"type": "response.done", "response_id": "r1"})
            elif event.get("type") == "session.close":
                self.closed_pending_acks = len(self.probe.ack_sent)

    websocket = NoAckWebSocket()
    wav_path = tmp_path / "one-frame.wav"
    _write_two_frame_wav(wav_path)
    probes = []

    def make_probe(*args, **kwargs):
        probe = FastDrainProbe(*args, **kwargs)
        probes.append(probe)
        websocket.probe = probe
        return probe

    monkeypatch.setattr(duplex_stream_e2e, "SingleStreamProbe", make_probe)
    monkeypatch.setattr(duplex_stream_e2e.websockets, "connect", lambda *_a, **_k: websocket)

    report = asyncio.run(duplex_stream_e2e.run_concurrent_single_sessions(
        "ws://unused", tmp_path / "concurrent-timeout", wav_path, None, 0.08,
        session_count=1, warmup_ticks_per_session=0,
    ))

    row = report["session_reports"][0]
    assert websocket.closed_pending_acks == 1
    assert row["final_ack_drain_complete"] is False
    assert row["final_ack_drain_timed_out"] is True
    assert row["pending_after_close"] == 1
    assert row["sender_error"] == "final ACK drain timed out with 1 pending frame(s)"
    assert report["final_ack_drain_complete"] is False
    assert report["pending_after_close"] == 1
    assert report["all_sessions_healthy"] is False


def test_pair_receive_preserves_server_ack_timings_without_fabricating_client_latency(tmp_path):
    # Mirrors the input_audio_buffer.consumed fields emitted by
    # duplex_kernel/server.py's listener ACK builder and websocket writer.
    server_ack = {
        "type": "input_audio_buffer.consumed",
        "frame_id": 37,
        "server_listener_queue_ms": 0.0,
        "server_listener_step_ms": None,
        "server_ack_queue_ms": 0.0,
        "listener_batch_mel_prepare_ms": 3.0,
        "listener_batch_frontend_copy_cuda_ms": 1.25,
        "listener_batch_audio_tower_workspace_lock_wait_ms": 0.1,
        "listener_batch_audio_tower_workspace_lock_hold_ms": 0.9,
        "listener_batch_audio_tower_row_copy_cuda_ms": 0.4,
        "listener_batch_audio_tower_graph_submit_ms": 2.2,
        "listener_batch_audio_tower_cuda_ms": 9.5,
        "listener_batch_audio_tower_execution_mode": "cuda_graph_n1",
        "listener_batch_semantic_prefix_request_ms": 0.5,
        "scheduler_dispatch_start_ns": 100,
        "scheduler_dispatch_end_ns": 200,
        "scheduler_row_callback_entry_ns": 110,
        "scheduler_timeline_request_ns": 120,
        "scheduler_timeline_capture_lock_acquired_ns": 121,
        "scheduler_timeline_model_lock_acquired_ns": 122,
        "scheduler_timeline_snapshot_ns": 123,
        "scheduler_timeline_age_ms": 1.5,
        "scheduler_timeline_status": "complete",
        "scheduler_timeline_reason": "snapshot",
        "scheduler_commit_microbatch_output_start_ns": 130,
        "scheduler_commit_microbatch_output_end_ns": 140,
        "scheduler_commit_state_lock_acquired_ns": 141,
        "scheduler_commit_state_lock_released_ns": 142,
        "scheduler_commit_state_lock_wait_ms": 0.2,
        "scheduler_commit_state_lock_hold_ms": 0.3,
        "scheduler_policy_decision_at_ns": 150,
        "scheduler_deadline_to_policy_decision_ms": 0.5,
    }

    probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=1, max_duration_s=10)

    class FakeWebSocket:
        async def recv(self):
            probe.stop.set()
            return json.dumps(server_ack)

    async def run():
        peer = {
            "ws": FakeWebSocket(),
            "speaker": "A",
            "ack_ids": [],
            "ack_duplicates": 0,
            "ack_out_of_order": 0,
            "ack_pending": deque(),
            "ack_latency_ms": [],
            "ack_latency_by_frame": {},
            "ack_wakeup": asyncio.Event(),
            "responses": [],
            "active": set(),
            "session_errors": [],
            "auto_response_disable_requested": False,
            "auto_response_disable_confirmed": False,
            "auto_response_disable_ack": asyncio.Event(),
        }
        probe.peers = {"A": peer, "B": {"active": set()}}
        logged = []
        probe.log = lambda _speaker, event: logged.append(event)

        await asyncio.wait_for(probe._receive_loop(peer), timeout=2.0)
        return probe, peer, logged

    probe, peer, logged = asyncio.run(run())

    received_ack = next(event for event in logged if event.get("type") == "input_audio_buffer.consumed")
    probe_ack = next(event for event in logged if event.get("type") == "probe.input_frame_ack")
    for event in (received_ack, probe_ack):
        assert event["server_listener_queue_ms"] == 0.0
        assert event["server_listener_step_ms"] is None
        assert event["server_ack_queue_ms"] == 0.0
        assert event["listener_batch_mel_prepare_ms"] == 3.0
        assert event["listener_batch_frontend_copy_cuda_ms"] == 1.25
        assert event["listener_batch_audio_tower_workspace_lock_wait_ms"] == 0.1
        assert event["listener_batch_audio_tower_workspace_lock_hold_ms"] == 0.9
        assert event["listener_batch_audio_tower_row_copy_cuda_ms"] == 0.4
        assert event["listener_batch_audio_tower_graph_submit_ms"] == 2.2
        assert event["listener_batch_audio_tower_cuda_ms"] == 9.5
        assert event["listener_batch_audio_tower_execution_mode"] == "cuda_graph_n1"
        assert event["listener_batch_semantic_prefix_request_ms"] == 0.5
        assert event["scheduler_dispatch_start_ns"] == 100
        assert event["scheduler_dispatch_end_ns"] == 200
        assert event["scheduler_row_callback_entry_ns"] == 110
        assert event["scheduler_timeline_request_ns"] == 120
        assert event["scheduler_timeline_capture_lock_acquired_ns"] == 121
        assert event["scheduler_timeline_model_lock_acquired_ns"] == 122
        assert event["scheduler_timeline_snapshot_ns"] == 123
        assert event["scheduler_timeline_age_ms"] == 1.5
        assert event["scheduler_timeline_status"] == "complete"
        assert event["scheduler_timeline_reason"] == "snapshot"
        assert event["scheduler_commit_microbatch_output_start_ns"] == 130
        assert event["scheduler_commit_microbatch_output_end_ns"] == 140
        assert event["scheduler_commit_state_lock_acquired_ns"] == 141
        assert event["scheduler_commit_state_lock_released_ns"] == 142
        assert event["scheduler_commit_state_lock_wait_ms"] == 0.2
        assert event["scheduler_commit_state_lock_hold_ms"] == 0.3
        assert event["scheduler_policy_decision_at_ns"] == 150
        assert event["scheduler_deadline_to_policy_decision_ms"] == 0.5
    assert received_ack["frame_id"] == 37
    assert probe_ack["ack_frame_id"] == 37
    assert probe_ack["client_frame_id"] is None
    assert probe_ack["client_ack_latency_ms"] is None
    assert peer["ack_latency_by_frame"] == {}


def test_pair_peer_session_update_includes_configured_conversation_instructions(monkeypatch, tmp_path):
    class FakeWebSocket:
        def __init__(self):
            self.sent = []
            self.incoming = deque([
                json.dumps({"type": "session.created", "session_id": "session-A", "input_audio_buffer_capacity_frames": 6}),
                json.dumps({"type": "session.updated", "turn_policy": duplex_stream_e2e.TURN_POLICY, "voice_conditioning_mode": "x_vector_only"}),
            ])

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        async def recv(self):
            return self.incoming.popleft()

        async def close(self):
            return None

    ws = FakeWebSocket()

    async def fake_connect(*_args, **_kwargs):
        return ws

    monkeypatch.setattr(duplex_stream_e2e.websockets, "connect", fake_connect)

    async def run():
        probe = RealtimePairProbe(
            "ws://unused", tmp_path, max_turns=2, max_duration_s=10,
            instructions_a="Stiamo organizzando una cena con amici.",
            instructions_b="Aiuta a scegliere insieme un ristorante e un orario.",
        )
        await probe._open_peer("A")

    asyncio.run(run())
    assert ws.sent[0]["session"]["instructions"] == "Stiamo organizzando una cena con amici."


def test_pair_manual_seed_bootstrap_is_opt_in_and_disables_auto_response_only_for_a(monkeypatch, tmp_path):
    class FakeWebSocket:
        def __init__(self, session_id):
            self.session_id = session_id
            self.sent = []
            self.incoming = deque([
                json.dumps({"type": "session.created", "session_id": session_id,
                            "input_audio_buffer_capacity_frames": 6}),
                json.dumps({"type": "session.updated", "turn_policy": duplex_stream_e2e.TURN_POLICY,
                            "voice_conditioning_mode": "x_vector_only"}),
            ])

        async def send(self, payload):
            self.sent.append(json.loads(payload))

        async def recv(self):
            return self.incoming.popleft()

        async def close(self):
            return None

    created_sockets = [FakeWebSocket("default"), FakeWebSocket("A"), FakeWebSocket("B")]
    sockets = deque(created_sockets)

    async def fake_connect(*_args, **_kwargs):
        return sockets.popleft()

    monkeypatch.setattr(duplex_stream_e2e.websockets, "connect", fake_connect)

    async def run():
        default_probe = RealtimePairProbe("ws://unused", tmp_path / "default", max_turns=2,
                                          max_duration_s=10)
        await default_probe._open_peer("A")
        bootstrap_probe = RealtimePairProbe(
            "ws://unused", tmp_path / "bootstrap", max_turns=2, max_duration_s=10,
            manual_seed_bootstrap=True,
        )
        await bootstrap_probe._open_peer("A")
        await bootstrap_probe._open_peer("B")

    asyncio.run(run())
    assert not sockets
    configs = [socket.sent[0]["session"] for socket in created_sockets]
    assert [config["auto_response"] for config in configs] == [True, False, True]


def test_pair_manual_seed_bootstrap_commits_once_after_seed_and_tail_acks(tmp_path):
    async def run():
        probe = RealtimePairProbe(
            "ws://unused", tmp_path, max_turns=3, max_duration_s=5,
            endpoint_silence_frames=1, manual_seed_bootstrap=True,
        )
        sent = []
        peer = {
            "speaker": "A",
            "ws": None,
            "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque(),
            "ack_capacity": 6,
            "ack_wakeup": asyncio.Event(),
            "ack_latency_by_frame": {},
            "ack_latency_ms": [],
            "ack_ids": [],
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "queue_frames_sent": 0,
            "queue_frames_enqueued": 2,
            "queued_audio_frames_sent": 0,
            "endpoint_silence_frames_sent": 0,
            "live_silence_frames_sent": 0,
            "queue_high_water": 2,
            "sender_backpressure_ms": [],
            "frames_discarded_after_response_limit": 0,
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "input_frame_send_records": [],
            "pending_high_water": 0,
            "stop_sender": False,
            "active": set(),
            "route_pending": 0,
            "manual_seed_bootstrap": True,
            "bootstrap_phase": "seed",
            "bootstrap_frame_ids": [],
            "bootstrap_seed_frame_count": 0,
            "bootstrap_seed_marker_seen": False,
            "bootstrap_tail_frames_required": 0,
            "bootstrap_tail_frames_scheduled": 0,
            "bootstrap_tail_frames_sent": 0,
            "bootstrap_tail_sent": False,
            "bootstrap_commit_sent": False,
        }

        class FakeWebSocket:
            async def send(self, payload):
                event = json.loads(payload)
                sent.append(event)
                if event["type"] == "input_audio_buffer.append":
                    local_frame_id, _ = peer["ack_pending"].popleft()
                    peer["ack_latency_by_frame"][local_frame_id] = {
                        "frame_id": local_frame_id,
                        "ack_frame_id": local_frame_id,
                        "ack_received_at_ns": time.monotonic_ns(),
                        "latency_ms": 1.0,
                    }
                    peer["ack_wakeup"].set()
                    if local_frame_id == 2:
                        peer["stop_sender"] = True
                elif event["type"] == "input_audio_buffer.commit":
                    # A later end marker must not rearm the one-shot bootstrap.
                    peer["queue"].put_nowait((probe._END_UTTERANCE, 1, "later"))

        peer["ws"] = FakeWebSocket()
        peer_b = {
            "speaker": "B", "ws": FakeWebSocket(), "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque(), "ack_capacity": 6, "ack_wakeup": asyncio.Event(),
            "input_samples": [], "recent_user_frames": deque(maxlen=5), "queue_frames_sent": 0,
            "stop_sender": True, "active": set(), "route_pending": 0,
            "manual_seed_bootstrap": False, "bootstrap_phase": "disabled",
        }
        probe.peers = {"A": peer, "B": peer_b}
        logged = []
        probe.log = lambda speaker, event: logged.append({"speaker": speaker, **event})
        await peer["queue"].put((np.full(1280, 0.1, dtype=np.float32), "seed", (), time.monotonic_ns()))
        await peer["queue"].put((probe._END_UTTERANCE, 1, None, "manual_seed_bootstrap"))

        await asyncio.wait_for(probe._send_loop(peer), timeout=1.0)

        kinds = [event["type"] for event in sent]
        assert kinds.count("input_audio_buffer.commit") == 1
        assert kinds.count("response.create") == 0
        commit_index = kinds.index("input_audio_buffer.commit")
        assert kinds[:commit_index].count("input_audio_buffer.append") == 2
        assert kinds[commit_index + 1:].count("input_audio_buffer.append") == 1
        commit_log = next(event for event in logged if event["type"] == "probe.manual_seed_bootstrap_commit")
        assert commit_log["frame_ids"] == [0, 1]
        assert [ack["frame_id"] for ack in commit_log["acknowledgements"]] == [0, 1]
        assert commit_log["sent_at_ns"] > max(ack["ack_received_at_ns"] for ack in commit_log["acknowledgements"])
        sent_before_b = len(sent)
        await probe._send_loop(peer_b)
        assert peer_b["manual_seed_bootstrap"] is False
        assert len(sent) == sent_before_b

    asyncio.run(run())


def test_pair_manual_seed_bootstrap_enables_auto_response_after_audible_speak_done(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=3, max_duration_s=10,
                                  manual_seed_bootstrap=True)
        incoming = asyncio.Queue()
        order = []
        commit_send_completed = False

        class FakeWebSocket:
            async def recv(self):
                event = await incoming.get()
                order.append(("in", event["type"]))
                if event["type"] == "session.updated":
                    probe.stop.set()
                return json.dumps(event)

            async def send(self, payload):
                nonlocal commit_send_completed
                event = json.loads(payload)
                order.append(("out", event["type"], event.get("session", {}).get("auto_response")))
                if event["type"] == "session.update":
                    assert commit_send_completed
                    incoming.put_nowait({"type": "session.updated"})
                elif event["type"] == "input_audio_buffer.commit":
                    assert peer["bootstrap_phase"] == "commit_inflight"
                    while not peer.get("bootstrap_response_eligible"):
                        await asyncio.sleep(0.001)
                    # A response was created and completed while this send was
                    # suspended; the sender marks commit successful only after
                    # this method returns.
                    commit_send_completed = True

        peer = {
            "speaker": "A", "ws": FakeWebSocket(), "ack_ids": [], "ack_duplicates": 0,
            "ack_out_of_order": 0, "ack_pending": deque(), "ack_latency_ms": [],
            "ack_latency_by_frame": {}, "ack_wakeup": asyncio.Event(), "responses": [],
            "active": set(), "session_errors": [], "auto_response_disable_requested": False,
            "auto_response_disable_confirmed": False, "auto_response_disable_ack": asyncio.Event(),
            "manual_seed_bootstrap": True, "bootstrap_phase": "commit_inflight",
            "bootstrap_commit_sent": False, "bootstrap_response_id": None,
            "bootstrap_auto_response_enable_requested": False,
            "bootstrap_auto_response_enable_confirmed": False,
            "bootstrap_auto_response_enable_ack": asyncio.Event(),
        }
        probe.peers = {"A": peer, "B": {"active": set(), "route_pending": 0}}
        probe.log = lambda *_args, **_kwargs: None
        probe._enqueue_route_item = lambda *_args, **_kwargs: None
        audio = io.BytesIO()
        with wave.open(audio, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24_000)
            wav.writeframes(np.ones(480, dtype="<i2").tobytes())
        for event in [
            {"type": "response.created", "response_id": "bootstrap-1", "action": "SPEAK"},
            {"type": "response.audio.delta", "response_id": "bootstrap-1",
             "delta": base64.b64encode(audio.getvalue()).decode("ascii")},
            {"type": "response.done", "response_id": "bootstrap-1",
             "runtime_metrics": {
                 "audio_response_protocol": {"status": "controller_action_fixed",
                                              "response_action": "SPEAK"},
                 "talker": {"terminated_by": "eos", "active_pcm_samples": 480},
             }},
        ]:
            incoming.put_nowait(event)

        receiver_task = asyncio.create_task(probe._receive_loop(peer))
        await asyncio.wait_for(
            peer["ws"].send(json.dumps({"type": "input_audio_buffer.commit"})),
            timeout=1.0,
        )
        assert peer["bootstrap_response_id"] == "bootstrap-1"
        assert peer["bootstrap_response_eligible"] is True
        assert peer["bootstrap_auto_response_enable_requested"] is False
        peer["bootstrap_commit_sent"] = True
        peer["bootstrap_phase"] = "awaiting_response"
        probe._schedule_manual_seed_auto_response(peer)
        await asyncio.wait_for(receiver_task, timeout=1.0)
        await asyncio.wait_for(peer["bootstrap_auto_response_enable_task"], timeout=1.0)

        outbound_updates = [row for row in order if row[0] == "out" and row[1] == "session.update"]
        assert outbound_updates == [("out", "session.update", True)]
        assert order.index(("in", "response.done")) < order.index(outbound_updates[0])
        assert peer["bootstrap_phase"] == "complete"
        assert peer["bootstrap_auto_response_enable_confirmed"] is True
        assert not probe.errors
        assert all(not (row[0] == "out" and row[1] == "response.create") for row in order)

        budget_probe = RealtimePairProbe(
            "ws://unused", tmp_path / "budget", max_turns=3, max_created_responses=1,
            max_duration_s=10, manual_seed_bootstrap=True,
        )
        budget_probe.created_budget_exhausted = True
        budget_probe.log = lambda *_args, **_kwargs: None
        budget_peer = {
            "speaker": "A", "manual_seed_bootstrap": True,
            "bootstrap_phase": "awaiting_response", "bootstrap_response_id": "budget-1",
            "bootstrap_response_eligible": True, "bootstrap_commit_sent": True,
            "auto_response_disable_requested": True,
        }
        budget_probe._schedule_manual_seed_auto_response(budget_peer)
        assert budget_peer["bootstrap_phase"] == "failed"
        assert not budget_peer.get("bootstrap_auto_response_enable_requested", False)
        assert budget_peer.get("bootstrap_auto_response_enable_task") is None
        assert budget_probe.errors

    asyncio.run(run())


def test_pair_manual_seed_bootstrap_missing_frame_ack_fails_without_waiting_for_session_deadline(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=300,
                                  manual_seed_bootstrap=True)

        class FakeWebSocket:
            async def send(self, _payload):
                raise AssertionError("commit must not be sent before the missing frame ACK")

        peer = {
            "speaker": "A", "ws": FakeWebSocket(), "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque([(0, time.monotonic_ns())]), "ack_capacity": 6,
            "ack_wakeup": asyncio.Event(), "ack_latency_by_frame": {},
            "manual_seed_bootstrap": True, "bootstrap_phase": "awaiting_ack",
            "bootstrap_frame_ids": [0], "bootstrap_seed_frame_count": 1,
            "bootstrap_ack_deadline_at": asyncio.get_running_loop().time() + 0.01,
            "bootstrap_auto_response_enable_ack": asyncio.Event(), "stop_sender": False,
        }
        probe.peers = {"A": peer}
        probe.log = lambda *_args, **_kwargs: None

        await asyncio.wait_for(probe._send_loop(peer), timeout=0.5)

        assert peer["bootstrap_phase"] == "failed"
        assert "ACKs were not received" in peer["bootstrap_failure_reason"]
        assert probe.errors

    asyncio.run(run())


def test_pair_manual_seed_bootstrap_failure_does_not_enable_or_force_retry(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=3, max_duration_s=10,
                                  manual_seed_bootstrap=True)
        incoming = asyncio.Queue()
        sent = []

        class FakeWebSocket:
            async def recv(self):
                event = await incoming.get()
                if event["type"] == "response.done":
                    probe.stop.set()
                return json.dumps(event)

            async def send(self, payload):
                sent.append(json.loads(payload))

        peer = {
            "speaker": "A", "ws": FakeWebSocket(), "ack_ids": [], "ack_duplicates": 0,
            "ack_out_of_order": 0, "ack_pending": deque(), "ack_latency_ms": [],
            "ack_latency_by_frame": {}, "ack_wakeup": asyncio.Event(), "responses": [],
            "active": set(), "session_errors": [], "auto_response_disable_requested": False,
            "auto_response_disable_confirmed": False, "auto_response_disable_ack": asyncio.Event(),
            "manual_seed_bootstrap": True, "bootstrap_phase": "awaiting_response",
            "bootstrap_commit_sent": True, "bootstrap_response_id": None,
            "bootstrap_auto_response_enable_requested": False,
            "bootstrap_auto_response_enable_confirmed": False,
            "bootstrap_auto_response_enable_ack": asyncio.Event(),
        }
        probe.peers = {"A": peer, "B": {"active": set(), "route_pending": 0}}
        probe.log = lambda *_args, **_kwargs: None
        probe._enqueue_route_item = lambda *_args, **_kwargs: None
        incoming.put_nowait({"type": "response.created", "response_id": "bootstrap-failed", "action": "SPEAK"})
        incoming.put_nowait({"type": "response.done", "response_id": "bootstrap-failed", "status": "failed"})

        await asyncio.wait_for(probe._receive_loop(peer), timeout=1.0)

        assert peer["bootstrap_phase"] == "failed"
        assert probe.errors
        assert not peer["bootstrap_auto_response_enable_requested"]
        assert not any(event["type"] in {"session.update", "response.create", "input_audio_buffer.commit"}
                       for event in sent)

    asyncio.run(run())


def test_single_session_diagnostic_marker_is_opt_in_and_seed_bound():
    normal = duplex_stream_e2e.build_single_session_update("Rispondi in italiano.")
    assert "tts_quality_diagnostic_fixture" not in normal
    assert "tts_quality_diagnostic_seed_sha256" not in normal

    digest = "a" * 64
    diagnostic = duplex_stream_e2e.build_single_session_update(
        "Rispondi in italiano.",
        tts_quality_diagnostics=True,
        diagnostic_seed_sha256=digest,
    )
    assert diagnostic["tts_quality_diagnostic_fixture"] == duplex_stream_e2e.SYNTHETIC_DIALOGUE_FIXTURE_ID
    assert diagnostic["tts_quality_diagnostic_seed_sha256"] == digest


def test_single_sender_waits_for_ack_capacity_and_records_backpressure(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    probe.ack_sent.extend((0, time.monotonic_ns()) for _ in range(2))
    capacity_available = asyncio.Event()

    async def run():
        task = asyncio.create_task(probe._wait_for_ack_capacity(2, capacity_available))
        await asyncio.sleep(0)
        assert not task.done()
        probe.ack_sent.popleft()
        capacity_available.set()
        waited_ms = await task
        return waited_ms

    waited_ms = asyncio.run(run())
    assert waited_ms >= 0.0
    assert probe.ack_capacity_wait_ms == [waited_ms]


def test_single_sender_registers_ack_before_socket_send_and_rolls_back_failure(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)

    class ImmediateAckSocket:
        async def send(self, _payload):
            assert probe.ack_sent
            probe.ack_sent.popleft()

    class FailedSocket:
        async def send(self, _payload):
            raise RuntimeError("socket send failed")

    async def run():
        await probe._send_with_ack_tracking(ImmediateAckSocket(), "frame", 0, 1)
        assert not probe.ack_sent
        with pytest.raises(RuntimeError, match="socket send failed"):
            await probe._send_with_ack_tracking(FailedSocket(), "frame", 1, 2)
        assert not probe.ack_sent

    asyncio.run(run())


def test_single_sender_ack_wait_times_out_without_dropping_credit(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    probe.ack_sent.append((0, time.monotonic_ns()))

    async def run():
        with pytest.raises(TimeoutError, match="ACK credit"):
            await probe._wait_for_ack_capacity(1, asyncio.Event(), timeout_s=0.01)

    asyncio.run(run())
    assert len(probe.ack_sent) == 1
    assert probe.ack_credit_timeouts == 1
    assert not probe.overflow


def test_single_sender_ack_wait_is_cancellable(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    probe.ack_sent.append((0, time.monotonic_ns()))

    async def run():
        task = asyncio.create_task(
            probe._wait_for_ack_capacity(1, asyncio.Event(), timeout_s=10.0)
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert len(probe.ack_sent) == 1


def test_single_sender_ack_wait_stops_when_receiver_closes(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    probe.ack_sent.append((0, time.monotonic_ns()))
    stop_event = asyncio.Event()
    capacity_available = asyncio.Event()

    async def run():
        task = asyncio.create_task(probe._wait_for_ack_capacity(
            1,
            capacity_available,
            timeout_s=10.0,
            stop_event=stop_event,
        ))
        await asyncio.sleep(0)
        stop_event.set()
        capacity_available.set()
        with pytest.raises(RuntimeError, match="receiver ended"):
            await task

    asyncio.run(run())
    assert len(probe.ack_sent) == 1
    assert probe.ack_credit_timeouts == 0


def test_single_sender_ack_arrival_during_clear_does_not_lose_credit_wakeup(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    probe.ack_sent.extend(((0, time.monotonic_ns()), (1, time.monotonic_ns())))

    class AckArrivesDuringClear:
        def __init__(self):
            self.event = asyncio.Event()

        def clear(self):
            self.event.clear()
            probe.ack_sent.popleft()

        async def wait(self):
            await self.event.wait()

    async def run():
        waited_ms = await probe._wait_for_ack_capacity(2, AckArrivesDuringClear())
        assert 0.0 <= waited_ms < 1.0

    asyncio.run(run())
    assert len(probe.ack_sent) == 1


def test_single_mode_exit_fails_if_source_stops_after_all_sent_frames_were_acked():
    report = {
        "capacity_overflow": False,
        "missing_ack_ids": [],
        "errors": [],
        "sender_error": None,
        "input_frames_sent": 7,
        "input_frames_acked": 7,
        "input_frames_not_dispatched": 3,
        "source_playback_complete": False,
        "final_ack_drain_complete": False,
    }

    assert duplex_stream_e2e._single_mode_report_failed(report) is True


def test_single_mode_exit_accepts_only_complete_error_free_source_playback():
    report = {
        "capacity_overflow": False,
        "missing_ack_ids": [],
        "errors": [],
        "sender_error": None,
        "input_frames_sent": 10,
        "input_frames_acked": 10,
        "input_frames_not_dispatched": 0,
        "source_playback_complete": True,
        "final_ack_drain_complete": True,
    }

    assert duplex_stream_e2e._single_mode_report_failed(report) is False


def test_single_mode_exit_fails_on_sender_error_even_if_source_report_is_complete():
    report = {
        "capacity_overflow": False,
        "missing_ack_ids": [],
        "errors": [],
        "sender_error": "WebSocket sender failed",
        "input_frames_sent": 10,
        "input_frames_acked": 10,
        "input_frames_not_dispatched": 0,
        "source_playback_complete": True,
        "final_ack_drain_complete": True,
    }

    assert duplex_stream_e2e._single_mode_report_failed(report) is True


def test_final_ack_drain_waits_for_late_ack_before_websocket_close(tmp_path, monkeypatch):
    class FakeWebSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.probe = None
            self.ack_received = False
            self.closed_with_pending_ack = None

        async def __aenter__(self):
            self.incoming.put_nowait({
                "type": "session.created",
                "session_id": "session-final-ack",
                "input_audio_buffer_capacity_frames": 6,
            })
            return self

        async def __aexit__(self, *_args):
            return None

        async def recv(self):
            return json.dumps(await self.incoming.get())

        async def send(self, payload):
            event = json.loads(payload)
            if event.get("type") == "session.update":
                self.incoming.put_nowait({
                    "type": "session.updated",
                    "turn_policy": "gemma_native_control6",
                    "voice_conditioning_mode": "x_vector_only",
                })
            elif event.get("type") == "input_audio_buffer.append":
                async def send_response_then_delayed_ack():
                    self.incoming.put_nowait({
                        "type": "response.created",
                        "response_id": "response-1",
                        "action": "SPEAK",
                    })
                    self.incoming.put_nowait({
                        "type": "response.done",
                        "response_id": "response-1",
                    })
                    await asyncio.sleep(0.03)
                    self.ack_received = True
                    self.incoming.put_nowait({
                        "type": "input_audio_buffer.consumed",
                        "frame_id": 0,
                    })

                asyncio.create_task(send_response_then_delayed_ack())
            elif event.get("type") == "session.close":
                self.closed_with_pending_ack = bool(self.probe.ack_sent)

    wav_path = tmp_path / "one-frame.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes((np.ones(1280, dtype="<i2") * 1000).tobytes())

    websocket = FakeWebSocket()
    probe = SingleStreamProbe("ws://unused", tmp_path / "probe")
    websocket.probe = probe
    monkeypatch.setattr(duplex_stream_e2e.websockets, "connect", lambda *_a, **_k: websocket)
    report = asyncio.run(probe.run(wav_path, None, 0.08))

    assert websocket.ack_received is True
    assert websocket.closed_with_pending_ack is False
    assert report["input_frames_sent"] == report["input_frames_acked"] == 1
    assert report["final_ack_drain_complete"] is True
    assert report["final_ack_drain_timed_out"] is False
    assert report["pending_after_close"] == 0


def test_final_ack_drain_timeout_is_explicit_and_preserves_pending_frames(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path / "timeout")
    probe.ack_sent.append((4, 40))
    pending_acks_drained = asyncio.Event()
    pending_acks_drained.clear()

    async def run():
        return await probe._wait_for_final_ack_drain(
            pending_acks_drained,
            timeout_s=0.01,
        )

    assert asyncio.run(run()) is False
    assert len(probe.ack_sent) == 1
    assert probe.final_ack_drain_timeouts == 1
    assert duplex_stream_e2e._single_mode_report_failed({
        "capacity_overflow": False,
        "missing_ack_ids": [4],
        "errors": [],
        "sender_error": "final ACK drain timed out with 1 pending frame(s)",
        "final_ack_drain_complete": False,
        "source_playback_complete": True,
        "input_frames_not_dispatched": 0,
    }) is True


def test_scheduled_dispatch_never_bursts_after_capture_backlog():
    target = duplex_stream_e2e._input_frame_dispatch_deadline(
        stream_start_loop=10.0,
        frame_index=2,
        previous_dispatch_loop=10.31,
    )
    assert target == pytest.approx(10.39)
    assert target >= 10.31 + duplex_stream_e2e.FRAME_MS / 1000.0


def test_single_probe_waits_for_automatic_response_completion(tmp_path):
    probe = SingleStreamProbe("ws://unused", tmp_path)
    first_created = asyncio.Event()
    response_done = asyncio.Event()

    async def run():
        task = asyncio.create_task(probe._wait_for_response_completion(
            first_created,
            response_done,
            trigger_timeout_s=0.2,
            completion_timeout_s=0.2,
        ))
        await asyncio.sleep(0)
        first_created.set()
        await asyncio.sleep(0)
        response_done.set()
        return await task

    assert asyncio.run(run()) is False


def test_pair_initial_seed_uses_bounded_backpressure_instead_of_preload_overflow(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=10)
        peer = {
            "queue": asyncio.Queue(maxsize=1),
            "queue_frames_enqueued": 0,
            "queue_high_water": 0,
        }
        frames = [np.full(1280, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
        consumed = []

        async def consume():
            await asyncio.sleep(0.01)
            while len(consumed) < len(frames):
                item = await peer["queue"].get()
                consumed.append(item)
                peer["queue"].task_done()
                await asyncio.sleep(0.005)

        await asyncio.gather(probe._queue_seed_frames(peer, frames), consume())
        assert len(consumed) == 3
        assert peer["queue_frames_enqueued"] == 3
        assert peer["queue_high_water"] == 1
        assert [item[1] for item in consumed] == ["seed"] * 3

    asyncio.run(run())


def test_pair_routes_pcm_before_source_response_finishes(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=6, max_duration_s=10)
        sender = {"speaker": "A", "resample_remainder": np.empty(0, dtype=np.float32), "input_routed_to_peer_frames": 0}
        receiver = {"speaker": "B", "queue": asyncio.Queue(maxsize=8), "queue_frames_enqueued": 0, "queue_high_water": 0}

        # Two 40 ms codec packets complete one 80 ms client frame. The frame is
        # already on B's ingress queue without waiting for a response.done event.
        packet = np.full(960, 0.2, dtype=np.float32)
        await probe._route_packet(sender, receiver, "response-A-1", packet)
        assert receiver["queue"].qsize() == 0
        await probe._route_packet(sender, receiver, "response-A-1", packet)
        assert receiver["queue"].qsize() == 1
        frame, response_id, *_ = receiver["queue"].get_nowait()
        assert frame.shape == (1280,)
        assert response_id == "response-A-1"
        assert np.max(np.abs(frame)) > 0.01

    asyncio.run(run())


def test_pair_cancel_fence_drops_old_queued_and_inflight_pcm_but_keeps_live_and_next_response(monkeypatch, tmp_path):
    original = duplex_stream_e2e.scipy.signal.resample_poly
    resample_started = threading.Event()
    release_resample = threading.Event()

    def held_first_resample(samples, up, down):
        if not resample_started.is_set():
            resample_started.set()
            assert release_resample.wait(timeout=2)
        return original(samples, up, down)

    monkeypatch.setattr(duplex_stream_e2e.scipy.signal, "resample_poly", held_first_resample)

    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=10)
        probe.started_ns = time.monotonic_ns()
        probe.events_path.write_text("", encoding="utf-8")
        old_id, next_id = "response-cancelled", "response-next"
        sender = {
            "speaker": "A",
            "route_queue": asyncio.Queue(maxsize=8),
            "route_pending": 1,
            "resample_remainder": np.full(80, 0.1, dtype=np.float32),
            "resample_remainder_chunk_ids": np.array([1], dtype=np.int64),
            "resample_remainder_response_id": old_id,
            "input_routed_to_peer_frames": 0,
            "active": set(),
        }
        receiver = {
            "speaker": "B",
            "ws": None,
            "queue": asyncio.Queue(maxsize=8),
            "queue_frames_enqueued": 2,
            "queue_frames_sent": 0,
            "queue_high_water": 4,
            "queue_frames_cancelled_dropped": 0,
            "ack_pending": deque([(-1, time.monotonic_ns())]),
            "ack_capacity": 1,
            "ack_wakeup": asyncio.Event(),
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "sender_backpressure_ms": [],
            "frames_discarded_after_response_limit": 0,
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "pending_high_water": 0,
            "input_frame_send_records": [],
            "ack_latency_by_frame": {},
            "active": set(),
            "stop_sender": False,
            "send_inflight_response_id": None,
        }
        sender["route_queue"].put_nowait(("audio", old_id, 1, np.ones(10, dtype=np.float32), time.monotonic_ns()))
        receiver["queue"].put_nowait((np.full(1280, 0.4, dtype=np.float32), old_id, (1,), time.monotonic_ns()))
        receiver["queue"].put_nowait((np.full(1280, 0.3, dtype=np.float32), old_id, (2,), time.monotonic_ns()))
        receiver["queue"].put_nowait((probe._END_UTTERANCE, 0, old_id))
        receiver["queue"].put_nowait((np.zeros(1280, dtype=np.float32), "live_silence", (), time.monotonic_ns()))
        probe.peers = {"A": sender, "B": receiver}

        class FakeWebSocket:
            def __init__(self):
                self.payloads = []

            async def send(self, payload):
                self.payloads.append(json.loads(payload))
                receiver["ack_pending"].popleft()
                receiver["ack_wakeup"].set()

        receiver["ws"] = FakeWebSocket()
        send_task = asyncio.create_task(probe._send_loop(receiver))
        await asyncio.sleep(0)
        assert receiver["send_inflight_response_id"] == old_id

        route_task = asyncio.create_task(
            probe._route_packet(sender, receiver, old_id, 3, np.full(1920, 0.2, dtype=np.float32))
        )
        assert await asyncio.to_thread(resample_started.wait, 1)

        # Fence while both an old resample and an old send are in flight.
        receiver["ack_pending"].clear()
        fence = probe._fence_cancelled_response(sender, receiver, old_id)
        sender["route_pending"] = 0
        # A fresh response remains routable after the fence; live silence also
        # remains in the peer queue so the input cadence continues.
        await probe._route_packet(sender, receiver, next_id, 4, np.full(1920, 0.25, dtype=np.float32))
        receiver["stop_sender"] = True
        release_resample.set()
        await route_task
        await asyncio.wait_for(send_task, timeout=2)

        sent_sources = [item["source_response_id"] for item in receiver["input_frame_send_records"]]
        assert old_id not in sent_sources
        assert next_id in sent_sources
        assert "live_silence" in sent_sources
        assert fence["route_queue_audio_items_dropped"] == 1
        assert fence["peer_queue_audio_frames_dropped"] == 1
        assert fence["inflight_peer_frames_dropped"] == 1
        assert fence["inflight_route_chunks_dropped"] == 1
        assert fence["resample_remainder_samples_dropped"] == 80
        assert receiver["queue_frames_cancelled_dropped"] == 2
        assert sender["resample_remainder_response_id"] is None
        assert sender["route_pending"] == 0

    asyncio.run(run())


def test_pair_resampler_runs_off_loop_and_keeps_packet_order(monkeypatch, tmp_path):
    original = duplex_stream_e2e.scipy.signal.resample_poly
    worker_threads: list[int] = []

    def tracked_resample(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(duplex_stream_e2e.scipy.signal, "resample_poly", tracked_resample)

    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=6, max_duration_s=10)
        event_loop_thread = threading.get_ident()
        sender = {"speaker": "A", "resample_remainder": np.empty(0, dtype=np.float32), "input_routed_to_peer_frames": 0}
        receiver = {"speaker": "B", "queue": asyncio.Queue(maxsize=8), "queue_frames_enqueued": 0, "queue_high_water": 0}
        for value in (0.1, 0.2):
            await probe._route_packet(sender, receiver, "response-A-1", np.full(960, value, dtype=np.float32))
        queued = [receiver["queue"].get_nowait() for _ in range(receiver["queue"].qsize())]
        return event_loop_thread, queued

    loop_thread, queued = asyncio.run(run())
    assert len(worker_threads) == 2
    assert all(thread_id != loop_thread for thread_id in worker_threads)
    assert len(queued) == 1
    assert queued[0][1] == "response-A-1"
    assert queued[0][0].shape == (1280,)
    assert abs(float(np.mean(queued[0][0])) - 0.15) < 0.01


def test_pair_event_log_writer_batches_without_reordering(tmp_path):
    probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=10)
    probe.started_ns = time.monotonic_ns()
    probe.events_path.write_text("", encoding="utf-8")
    probe._start_log_writer()
    for index in range(100):
        probe.log("A", {"type": "ordered", "sequence": index})
    probe._stop_log_writer()
    rows = [json.loads(line) for line in probe.events_path.read_text(encoding="utf-8").splitlines()]
    assert [row["sequence"] for row in rows] == list(range(100))


def test_pair_flushes_partial_pcm_before_endpoint_marker(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=6, max_duration_s=10)
        sender = {"speaker": "A", "resample_remainder": np.empty(0, dtype=np.float32), "input_routed_to_peer_frames": 0}
        receiver = {"speaker": "B", "queue": asyncio.Queue(maxsize=8), "queue_frames_enqueued": 0, "queue_high_water": 0}
        await probe._route_packet(sender, receiver, "response-A-2", np.full(120, 0.1, dtype=np.float32))
        assert receiver["queue"].empty()

        await probe._finish_peer_utterance(sender, receiver, "response-A-2")
        padded, response_id, *_ = receiver["queue"].get_nowait()
        assert padded.shape == (1280,)
        assert response_id == "response-A-2"
        assert np.any(np.abs(padded[:80]) > 0.01)
        assert np.all(padded[80:] == 0)
        marker = receiver["queue"].get_nowait()
        assert marker[0] is probe._END_UTTERANCE
        assert marker[1] == probe.endpoint_silence_frames

    asyncio.run(run())


def test_pair_can_close_at_target_without_retriggering_peer_silence(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=10,
                                  endpoint_silence_frames=25)
        sender = {"speaker": "A", "resample_remainder": np.empty(0, dtype=np.float32), "input_routed_to_peer_frames": 0}
        receiver = {"speaker": "B", "queue": asyncio.Queue(maxsize=8), "queue_frames_enqueued": 0, "queue_high_water": 0}
        await probe._finish_peer_utterance(sender, receiver, "last-response", quiet_frames=0)
        marker = receiver["queue"].get_nowait()
        assert marker[:2] == (probe._END_UTTERANCE, 0)

    asyncio.run(run())


def test_pair_sender_suppresses_queued_endpoint_after_turn_target(tmp_path):
    class DummyWebSocket:
        async def send(self, _payload):
            raise AssertionError("target cutoff must not send queued silence")

    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=2, max_duration_s=10,
                                  endpoint_silence_frames=25)
        probe.audible_completed = 2
        peer = {
            "ws": DummyWebSocket(),
            "speaker": "B",
            "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque(),
            "ack_capacity": 6,
            "ack_wakeup": asyncio.Event(),
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "queue_frames_sent": 0,
            "queue_frames_enqueued": 1,
            "queue_high_water": 1,
            "sender_backpressure_ms": [],
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "pending_high_water": 0,
            "stop_sender": True,
        }
        peer["queue"].put_nowait((probe._END_UTTERANCE, 25))
        await asyncio.wait_for(probe._send_loop(peer), timeout=0.2)
        assert peer["queue"].empty()
        assert peer["queue_frames_sent"] == 0

    asyncio.run(run())


def test_pair_sender_keeps_live_silence_ticks_and_services_queued_voice(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=1, max_duration_s=2)
        probe.started_ns = __import__("time").monotonic_ns()
        peer = {
            "speaker": "B",
            "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque(),
            "ack_capacity": 6,
            "ack_wakeup": asyncio.Event(),
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "queue_frames_sent": 0,
            "queue_frames_enqueued": 0,
            "queue_high_water": 0,
            "sender_backpressure_ms": [],
            "frames_discarded_after_response_limit": 0,
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "pending_high_water": 0,
            "stop_sender": False,
        }

        class FakeWebSocket:
            def __init__(self):
                self.payloads = []

            async def send(self, payload):
                self.payloads.append(json.loads(payload))
                if len(self.payloads) == 2:
                    peer["queue"].put_nowait((np.full(1280, 0.25, dtype=np.float32), "peer-speech", __import__("time").monotonic_ns()))
                if len(self.payloads) == 3:
                    peer["stop_sender"] = True

                    def acknowledge_all():
                        peer["ack_pending"].clear()
                        peer["ack_wakeup"].set()

                    asyncio.get_running_loop().call_soon(acknowledge_all)

        peer["ws"] = FakeWebSocket()
        await asyncio.wait_for(probe._send_loop(peer), timeout=0.6)

        assert len(peer["input_samples"]) == 3
        np.testing.assert_array_equal(peer["input_samples"][0], np.zeros(1280, dtype=np.float32))
        np.testing.assert_array_equal(peer["input_samples"][1], np.zeros(1280, dtype=np.float32))
        np.testing.assert_array_equal(peer["input_samples"][2], np.full(1280, 0.25, dtype=np.float32))
        assert peer["queue_frames_sent"] == 3
        assert len(peer["input_send_gaps_ms"]) == 2
        assert all(60 <= gap <= 120 for gap in peer["input_send_gaps_ms"])

    asyncio.run(run())


def test_pair_sender_services_relay_pcm_while_ack_is_pending_during_stop(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=1, max_duration_s=2)
        probe.started_ns = __import__("time").monotonic_ns()
        frame = np.full(1280, 0.25, dtype=np.float32)

        class FakeWebSocket:
            def __init__(self):
                self.payloads = []

            async def send(self, payload):
                self.payloads.append(json.loads(payload))
                peer["ack_pending"].clear()
                peer["ack_wakeup"].set()

        peer = {
            "speaker": "B",
            "queue": asyncio.Queue(maxsize=8),
            "ack_pending": deque([(0, __import__("time").monotonic_ns())]),
            "ack_capacity": 6,
            "ack_wakeup": asyncio.Event(),
            "input_samples": [],
            "recent_user_frames": deque(maxlen=5),
            "queue_frames_sent": 0,
            "queue_frames_enqueued": 1,
            "queued_audio_frames_sent": 0,
            "queue_high_water": 1,
            "sender_backpressure_ms": [],
            "frames_discarded_after_response_limit": 0,
            "input_send_times": [],
            "input_send_gaps_ms": [],
            "input_frame_send_records": [],
            "ack_latency_by_frame": {},
            "pending_high_water": 1,
            "stop_sender": True,
            "active": set(),
            "route_pending": 0,
        }
        peer["ws"] = FakeWebSocket()
        probe.peers = {
            "A": {"active": set(), "route_pending": 0},
            "B": peer,
        }
        probe.log = lambda *_args, **_kwargs: None

        sender = asyncio.create_task(probe._send_loop(peer))
        await asyncio.sleep(0.01)
        peer["queue"].put_nowait((frame, "peer-speech", (), __import__("time").monotonic_ns()))
        await asyncio.wait_for(sender, timeout=0.5)

        assert len(peer["ws"].payloads) == 1
        np.testing.assert_array_equal(peer["input_samples"][0], frame)
        assert peer["queued_audio_frames_sent"] == 1
        assert not peer["ack_pending"]

    asyncio.run(run())


def test_pair_response_done_preserves_complete_runtime_metrics(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=3, max_duration_s=90,
                                  max_created_responses=5)
        metrics = {
            "talker": {"calls": 7, "gpu_event_ms": {"p50": 4.2}},
            "frame_residual_adapter": {"calls": 7, "rms_mean": 0.03125},
            "future_field": {"kept": True},
        }
        control_context = {
            "legal_actions": ["LISTEN", "SPEAK"],
            "user_turn_pending": True,
            "pending_user_onset_frame_id": 41,
            "last_answered_onset_frame_id": 12,
            "audio_history": {"first_frame_id": 41, "last_frame_id": 44, "frame_count": 4},
            "semantic_prefix": {"length": 91, "capacity": 291, "revision_age": 0},
        }
        events = [
            {"type": "input_audio_buffer.consumed", "frame_id": 0,
             "server_listener_step_start_ns": 220000000,
             "server_listener_step_end_ns": 260000000,
             "scheduler_clock": "time.monotonic_ns",
             "scheduler_enqueue_ns": 100000000,
             "scheduler_deadline_ns": 120000000,
             "scheduler_observed_ns": 125000000,
             "scheduler_listener_start_ns": 130000000,
             "scheduler_listener_end_ns": None,
             "scheduler_enqueue_to_deadline_ms": 20.0,
             "scheduler_deadline_lateness_ms": 5.0,
             "scheduler_observed_to_listener_start_ms": 5.0,
             "scheduler_listener_callback_ms": None},
            {"type": "duplex_control6.decision", "action": "LISTEN",
             "control6_context": control_context,
             "turn_policy_id": "gemma_native_control6",
             "turn_policy_class": "duplex_kernel.gemma_native_control6_policy.GemmaNativeControl6Policy",
             "native_control6_class_match": True,
             "native_control6_step_called": True,
             "native_control6_head_scored": True,
             "policy_decision_at_ns": 123456789,
             "listener_hidden_at_ns": 123400000,
             "listener_hidden_age_frames": 2,
             "listener_hidden_age_ms": 5.679,
             "listener_query_reused": True,
             "listener_query_origin_frame_id": 40,
             "listener_query_age_frames": 1,
             "source": "gemma_native_control6",
             "checkpoint_path": "checkpoints/native_control6.pt",
             "native_action_order": ["LISTEN", "SPEAK", "CONTINUE", "YIELD", "BACKCHANNEL", "REPLAN"],
             "native_action_base_logits": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
             "native_action_logits": [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
             "native_control6_raw_row_index": 3,
             "native_control6_raw_token_id": 256004,
             "native_control6_raw_token": "<unused92>",
             "native_control6_selected_row_index": 2,
             "native_control6_selected_token_id": 256003,
             "native_control6_selected_token": "<unused91>",
             "native_control6_token_selection_scope": "six_projected_action_rows_not_full_vocabulary_autoregressive_sample",
             "gate_applied_action": "LISTEN",
             "gate_application_scope": "control_plane_acceptance_not_audible_output",
             "scheduler_clock": "time.monotonic_ns",
             "scheduler_enqueue_ns": 100000000,
             "scheduler_deadline_ns": 120000000,
             "scheduler_observed_ns": 125000000,
             "scheduler_listener_start_ns": 130000000,
             "scheduler_listener_end_ns": 190000000,
             "scheduler_enqueue_to_deadline_ms": 20.0,
             "scheduler_deadline_lateness_ms": 5.0,
             "scheduler_observed_to_listener_start_ms": 5.0,
             "scheduler_listener_callback_ms": 60.0},
            {"type": "response.created", "response_id": "r1"},
            {"type": "response.done", "response_id": "r1", "status": "completed",
             "runtime_metrics": metrics},
        ]

        class FakeWebSocket:
            def __init__(self):
                self.index = 0

            async def recv(self):
                event = events[self.index]
                self.index += 1
                if self.index == len(events):
                    probe.stop.set()
                return json.dumps(event)

        peer = {
            "ws": FakeWebSocket(),
            "speaker": "A",
            "ack_ids": [],
            "ack_duplicates": 0,
            "ack_out_of_order": 0,
            "ack_pending": deque(),
            "ack_latency_ms": [],
            "ack_latency_by_frame": {},
            "ack_wakeup": asyncio.Event(),
            "responses": [],
            "active": set(),
            "session_errors": [],
            "auto_response_disable_requested": False,
            "auto_response_disable_confirmed": False,
            "auto_response_disable_ack": asyncio.Event(),
        }
        probe.peers = {"A": peer, "B": {"active": set(), "route_pending": 0}}
        logged = []
        probe.log = lambda _speaker, event: logged.append(event)
        probe._enqueue_route_item = lambda *_args: None

        await asyncio.wait_for(probe._receive_loop(peer), timeout=2.0)

        done_event = next(event for event in logged if event.get("type") == "response.done")
        ack_event = next(event for event in logged if event.get("type") == "probe.input_frame_ack")
        consumed_event = next(event for event in logged if event.get("type") == "input_audio_buffer.consumed")
        decision_event = next(event for event in logged if event.get("type") == "duplex_control6.decision")
        assert decision_event["control6_context"] == control_context
        assert decision_event["turn_policy_id"] == "gemma_native_control6"
        assert decision_event["native_control6_head_scored"] is True
        assert decision_event["policy_decision_at_ns"] == 123456789
        assert decision_event["listener_hidden_at_ns"] == 123400000
        assert decision_event["listener_query_origin_frame_id"] == 40
        assert decision_event["checkpoint_path"] == "checkpoints/native_control6.pt"
        assert len(decision_event["native_action_order"]) == 6
        assert decision_event["native_action_base_logits"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        assert decision_event["native_action_logits"] == [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
        assert decision_event["native_control6_raw_row_index"] == 3
        assert decision_event["native_control6_raw_token_id"] == 256004
        assert decision_event["native_control6_raw_token"] == "<unused92>"
        assert decision_event["native_control6_selected_row_index"] == 2
        assert decision_event["native_control6_selected_token_id"] == 256003
        assert decision_event["native_control6_selected_token"] == "<unused91>"
        assert decision_event["native_control6_token_selection_scope"] == (
            "six_projected_action_rows_not_full_vocabulary_autoregressive_sample"
        )
        assert decision_event["gate_applied_action"] == "LISTEN"
        assert decision_event["scheduler_listener_end_ns"] == 190000000
        assert decision_event["scheduler_listener_callback_ms"] == 60.0
        assert consumed_event["scheduler_enqueue_ns"] == 100000000
        assert consumed_event["scheduler_listener_end_ns"] is None
        assert consumed_event["server_listener_step_end_ns"] == 260000000
        assert ack_event["scheduler_enqueue_ns"] == 100000000
        assert ack_event["scheduler_listener_end_ns"] is None
        assert ack_event["server_listener_step_end_ns"] == 260000000
        assert done_event["runtime_metrics"] == metrics
        assert probe.responses["r1"]["runtime_metrics"] == metrics

    asyncio.run(run())


def test_pair_counts_audible_responses_not_deferred_created_events_and_routes_active_audio(tmp_path):
    async def run():
        probe = RealtimePairProbe("ws://unused", tmp_path, max_turns=1, max_created_responses=4, max_duration_s=10)

        audio_bytes = io.BytesIO()
        pcm16 = np.full(1920, int(0.2 * 32767), dtype="<i2")
        with wave.open(audio_bytes, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(pcm16.tobytes())

        class FakeWebSocket:
            def __init__(self):
                self.events = iter([
                    {"type": "response.created", "response_id": "deferred-1", "action": "BACKCHANNEL"},
                    {
                        "type": "response.action_selected", "response_id": "deferred-1",
                        "wake_action": "BACKCHANNEL", "response_action": "SPEAK",
                        "deferred_until_endpoint": True,
                    },
                    {"type": "response.done", "response_id": "deferred-1", "status": "deferred_until_endpoint"},
                    {"type": "response.created", "response_id": "audible-2", "action": "SPEAK"},
                    {
                        "type": "response.action_selected", "response_id": "audible-2",
                        "wake_action": "SPEAK", "response_action": "SPEAK",
                        "deferred_until_endpoint": False,
                    },
                    {
                        "type": "response.audio.delta", "response_id": "audible-2",
                        "delta": base64.b64encode(audio_bytes.getvalue()).decode("ascii"),
                        "sample_rate": 24000,
                        "packet_id": 17,
                        "generation_id": 3,
                        "first_frame_id": 7,
                        "last_frame_id": 8,
                        "start_sample": 1920,
                        "end_sample": 3840,
                        "server_emit_started_at_ns": 555000000,
                        "raw_source_audio_quality": {
                            "finite": True, "sample_count": 1920, "peak": 0.2,
                            "rms": 0.2, "clipping_rail_samples": 0,
                        },
                    },
                    {"type": "response.done", "response_id": "audible-2", "status": "completed"},
                ])

            async def recv(self):
                try:
                    return json.dumps(next(self.events))
                except StopIteration:
                    probe.stop.set()
                    return json.dumps({"type": "session.closed"})

        def peer(speaker, ws):
            return {
                "speaker": speaker,
                "ws": ws,
                "queue": asyncio.Queue(maxsize=8),
                "route_queue": asyncio.Queue(maxsize=8),
                "route_pending": 0,
                "route_queue_high_water": 0,
                "queue_frames_enqueued": 0,
                "queue_high_water": 0,
                "ack_wakeup": asyncio.Event(),
                "responses": [],
                "active": set(),
                "stop_sender": False,
                "frames_discarded_after_response_limit": 0,
                "session_errors": [],
                "ack_ids": [],
                "ack_pending": deque(),
                "ack_duplicates": 0,
                "ack_out_of_order": 0,
                "resample_remainder": np.empty(0, dtype=np.float32),
                "input_routed_to_peer_frames": 0,
            }

        peer_a = peer("A", FakeWebSocket())
        peer_b = peer("B", FakeWebSocket())
        probe.peers = {"A": peer_a, "B": peer_b}
        probe.started_ns = time.monotonic_ns()
        logged = []
        probe.log = lambda _speaker, event: logged.append(event)

        await asyncio.wait_for(probe._receive_loop(peer_a), timeout=2.0)
        await asyncio.sleep(0.05)

        assert probe.created == 2
        assert probe.completed == 2
        assert probe.audible_completed == 1
        assert probe.created_target_event.is_set()
        assert not probe.bridge_stopped
        assert probe.target_drain_deadline is not None
        audio_chunk_event = next(event for event in logged if event.get("type") == "probe.response_audio_chunk")
        assert audio_chunk_event["server_packet_id"] == 17
        assert audio_chunk_event["server_emit_started_at_ns"] == 555000000
        assert audio_chunk_event["raw_source_audio_quality"]["clipping_rail_samples"] == 0
        assert peer_a["active"] == set()
        queued = [peer_b["queue"].get_nowait() for _ in range(peer_b["queue"].qsize())]
        audio_frames = [item for item in queued if item[0] is not probe._END_UTTERANCE]
        assert len(audio_frames) == 1
        assert audio_frames[0][1] == "audible-2"
        frame_rms = float(np.sqrt(np.mean(audio_frames[0][0] ** 2)))
        assert 0.195 <= frame_rms <= 0.205

    asyncio.run(run())

def test_pair_client_event_projection_keeps_control6_candidates_and_scheduler_timestamps():
    event = {
        "type": "duplex_control6.decision",
        "native_control6_head_scored": True,
        "native_control6_raw_row_index": 3,
        "native_control6_raw_token_id": 256004,
        "native_control6_raw_token": "<unused92>",
        "native_control6_selected_row_index": 2,
        "native_control6_selected_token_id": 256003,
        "native_control6_selected_token": "<unused91>",
        "native_control6_token_selection_scope": "six_projected_action_rows_not_full_vocabulary_autoregressive_sample",
        "gate_applied_action": "CONTINUE",
        "gate_application_scope": "control_plane_acceptance_not_audible_output",
        "scheduler_clock": "time.monotonic_ns",
        "scheduler_enqueue_ns": 100,
        "scheduler_deadline_ns": 120,
        "scheduler_observed_ns": 125,
        "scheduler_listener_start_ns": 130,
        "scheduler_listener_end_ns": 190,
        "scheduler_enqueue_to_deadline_ms": 0.00002,
        "scheduler_deadline_lateness_ms": 0.000005,
        "scheduler_observed_to_listener_start_ms": 0.000005,
        "scheduler_listener_callback_ms": 0.00006,
    }
    projected = duplex_stream_e2e._project_server_event_for_log(event)

    for field in (
        "native_control6_raw_row_index", "native_control6_raw_token_id", "native_control6_raw_token",
        "native_control6_selected_row_index", "native_control6_selected_token_id",
        "native_control6_selected_token", "native_control6_token_selection_scope",
        "gate_applied_action", "gate_application_scope", "scheduler_clock",
        "scheduler_enqueue_ns", "scheduler_deadline_ns", "scheduler_observed_ns",
        "scheduler_listener_start_ns", "scheduler_listener_end_ns",
        "scheduler_enqueue_to_deadline_ms", "scheduler_deadline_lateness_ms",
        "scheduler_observed_to_listener_start_ms", "scheduler_listener_callback_ms",
    ):
        assert projected[field] == event[field]

    fallback = duplex_stream_e2e._project_server_event_for_log({
        "type": "duplex_control6.decision",
        "native_control6_head_scored": False,
        "native_control6_raw_row_index": 3,
        "native_control6_raw_token_id": 256004,
        "native_control6_raw_token": "<unused92>",
    })
    assert "native_control6_raw_row_index" not in fallback
    assert "native_control6_raw_token_id" not in fallback
    assert "native_control6_raw_token" not in fallback


def test_client_response_projection_keeps_policy_vs_explicit_creation_origin():
    event = {
        "type": "response.created",
        "response_id": "auto-1",
        "action": "SPEAK",
        "effective_action": "SPEAK",
        "intent_source": "direct_audio_policy",
        "intent_source_frame_id": 18,
        "intent_endpoint_frame_id": 20,
        "policy_decision_frame_id": 20,
        "policy_decision_at_ns": 123456789,
    }

    projected = duplex_stream_e2e._project_server_event_for_log(event)

    for field in (
        "response_id", "action", "effective_action", "intent_source",
        "intent_source_frame_id", "intent_endpoint_frame_id",
        "policy_decision_frame_id", "policy_decision_at_ns",
    ):
        assert projected[field] == event[field]


def test_pair_client_ack_projection_retains_truthful_unmeasured_listener_end():
    event = {
        "server_listener_queue_ms": 1.5,
        "server_listener_step_ms": 2.5,
        "server_listener_step_start_ns": 210,
        "server_listener_step_end_ns": 260,
        "server_ack_queue_ms": 0.5,
        "scheduler_clock": "time.monotonic_ns",
        "scheduler_enqueue_ns": 100,
        "scheduler_deadline_ns": 120,
        "scheduler_observed_ns": 125,
        "scheduler_listener_start_ns": 130,
        "scheduler_listener_end_ns": None,
        "scheduler_enqueue_to_deadline_ms": 0.00002,
        "scheduler_deadline_lateness_ms": 0.000005,
        "scheduler_observed_to_listener_start_ms": 0.000005,
        "scheduler_listener_callback_ms": None,
        "listener_batch_native_control6_raw_score_ms": 6.5,
        "listener_batch_native_output_health_ms": 0.4,
        "listener_batch_semantic_prefix_build_ms": 3.0,
        "listener_batch_semantic_prefix_replay_ms": 1.5,
        "listener_batch_semantic_prefix_import_ms": 0.8,
        "listener_batch_semantic_prefix_refresh_event": True,
        "listener_batch_semantic_prefix_refresh_event_id": "batch-x:session-x:epoch-x:37:9",
        "listener_batch_semantic_prefix_refresh_revision": 9,
        "listener_batch_semantic_prefix_len": 141,
        "listener_batch_native_output_health_checked": True,
        "listener_batch_native_output_health_status": "healthy",
        "listener_batch_native_output_health_nonfinite": False,
        "listener_batch_native_output_health_generation": 21,
        "listener_batch_native_output_health_session_generation": 3,
        "listener_batch_native_output_health_state_epoch_id": "epoch-x",
        "listener_batch_native_output_health_row_frame_id": 37,
        "listener_batch_native_output_health_row_state_version": 8,
        "listener_batch_native_output_health_row_sequence_position": 4,
        "listener_batch_native_output_health_row_cache_position": 141,
        "listener_batch_native_output_health_native_context_id": 9001,
        "logical_q_len": 0,
        "kv_rollback_tokens": 2,
        "kv_rollback_enqueue_ms": 0.05,
        "listener_cache_position_before": 141,
        "listener_cache_position_after": 141,
        "listener_valid_history_tokens_before": 141,
        "listener_valid_history_tokens_after": 141,
        "native_cache_current_position_after": 141,
        "native_cache_valid_kv_len_after": 141,
        "frame_id": 37,
    }
    projected = duplex_stream_e2e._project_input_ack_fields(event)

    assert projected["server_listener_step_end_ns"] == 260
    assert projected["scheduler_enqueue_ns"] == 100
    assert projected["scheduler_listener_end_ns"] is None
    assert projected["scheduler_listener_callback_ms"] is None
    assert projected["logical_q_len"] == 0
    assert projected["kv_rollback_tokens"] == 2
    assert projected["kv_rollback_enqueue_ms"] == 0.05
    assert projected["listener_cache_position_before"] == projected["listener_cache_position_after"] == 141
    assert projected["listener_valid_history_tokens_before"] == projected["listener_valid_history_tokens_after"] == 141
    assert projected["native_cache_current_position_after"] == 141
    assert projected["native_cache_valid_kv_len_after"] == 141
    assert projected["listener_batch_native_control6_raw_score_ms"] == 6.5
    assert projected["listener_batch_native_output_health_ms"] == 0.4
    assert projected["listener_batch_semantic_prefix_build_ms"] == 3.0
    assert projected["listener_batch_semantic_prefix_replay_ms"] == 1.5
    assert projected["listener_batch_semantic_prefix_import_ms"] == 0.8
    assert projected["listener_batch_semantic_prefix_refresh_event"] is True
    assert projected["listener_batch_semantic_prefix_refresh_event_id"] == "batch-x:session-x:epoch-x:37:9"
    assert projected["listener_batch_semantic_prefix_refresh_revision"] == 9
    assert projected["listener_batch_semantic_prefix_len"] == 141
    assert projected["listener_batch_native_output_health_checked"] is True
    assert projected["listener_batch_native_output_health_status"] == "healthy"
    assert projected["listener_batch_native_output_health_nonfinite"] is False
    assert projected["listener_batch_native_output_health_generation"] == 21
    assert projected["listener_batch_native_output_health_session_generation"] == 3
    assert projected["listener_batch_native_output_health_state_epoch_id"] == "epoch-x"
    assert projected["listener_batch_native_output_health_row_frame_id"] == 37
    assert projected["listener_batch_native_output_health_row_state_version"] == 8
    assert projected["listener_batch_native_output_health_row_sequence_position"] == 4
    assert projected["listener_batch_native_output_health_row_cache_position"] == 141
    assert projected["listener_batch_native_output_health_native_context_id"] == 9001


def test_pair_bargein_wav_is_opt_in_one_shot_local_to_target_and_delayed(tmp_path, monkeypatch):
    async def run():
        fake_now_ns = [100_000_000]
        monkeypatch.setattr(duplex_stream_e2e.time, "monotonic_ns", lambda: fake_now_ns[0])

        def response_audio_event(response_id, generation_id, pcm16):
            data = io.BytesIO()
            with wave.open(data, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24_000)
                wav.writeframes(np.asarray(pcm16, dtype="<i2").tobytes())
            return {
                "type": "response.audio.delta",
                "response_id": response_id,
                "generation_id": generation_id,
                "delta": base64.b64encode(data.getvalue()).decode("ascii"),
            }

        class FakeWebSocket:
            def __init__(self, probe, peer, events, *, wait_after_first_audio=False):
                self.probe = probe
                self.peer = peer
                self.events = list(events)
                self.index = 0
                self.wait_after_first_audio = wait_after_first_audio
                self.waiting_for_injection = False
                self.payloads = []
                self.auto_ack = True

            async def recv(self):
                if self.waiting_for_injection and self.probe.bargein_task is not None:
                    await self.probe.bargein_task
                    self.waiting_for_injection = False
                if self.index == len(self.events):
                    self.probe.stop.set()
                    return json.dumps({"type": "session.closed"})
                event, at_ns = self.events[self.index]
                self.index += 1
                fake_now_ns[0] = at_ns
                if (
                    self.wait_after_first_audio
                    and event.get("type") == "response.audio.delta"
                    and base64.b64decode(event["delta"])
                ):
                    self.waiting_for_injection = True
                return json.dumps(event)

            async def send(self, payload):
                self.payloads.append(json.loads(payload))
                if self.auto_ack:
                    self.peer["ack_pending"].clear()
                    self.peer["ack_wakeup"].set()

        def make_peer(speaker):
            return {
                "speaker": speaker,
                "queue": asyncio.Queue(maxsize=8),
                "route_queue": asyncio.Queue(maxsize=8),
                "route_queue_high_water": 0,
                "queue_frames_enqueued": 0,
                "queue_high_water": 0,
                "ack_capacity": 6,
                "ack_pending": deque(),
                "ack_wakeup": asyncio.Event(),
                "ack_ids": [],
                "ack_latency_ms": [],
                "ack_latency_by_frame": {},
                "ack_duplicates": 0,
                "ack_out_of_order": 0,
                "responses": [],
                "active": set(),
                "session_errors": [],
                "auto_response_disable_requested": False,
                "auto_response_disable_ack": asyncio.Event(),
                "resample_remainder": np.empty(0, dtype=np.float32),
                "input_samples": [],
                "recent_user_frames": deque(maxlen=5),
                "queue_frames_sent": 0,
                "queued_audio_frames_sent": 0,
                "frames_discarded_after_response_limit": 0,
                "sender_backpressure_ms": [],
                "input_send_times": [],
                "input_send_gaps_ms": [],
                "input_frame_send_records": [],
                "pending_high_water": 0,
                "stop_sender": False,
                "route_pending": 0,
            }

        no_op_probe = RealtimePairProbe("ws://unused", tmp_path / "no-op", max_turns=5, max_duration_s=5)
        no_op_peer = make_peer("B")
        no_op_probe.peers = {"A": make_peer("A"), "B": no_op_peer}
        no_op_probe.started_ns = 0
        no_op_logs = []
        no_op_probe.log = lambda speaker, event: no_op_logs.append({"speaker": speaker, **event})
        no_op_probe._enqueue_route_item = lambda *_args: None
        no_op_peer["ws"] = FakeWebSocket(no_op_probe, no_op_peer, [
            ({"type": "response.created", "response_id": "no-op-response"}, 100_000_000),
            (response_audio_event("no-op-response", 3, np.ones(1920, dtype="<i2")), 200_000_000),
            ({"type": "response.done", "response_id": "no-op-response"}, 300_000_000),
        ])
        await no_op_probe._receive_loop(no_op_peer)
        assert no_op_probe.bargein_task is None
        assert no_op_peer["queue"].empty()
        assert not any(event.get("source_kind") == "injection" for event in no_op_logs)

        injection_wav = tmp_path / "bargein.wav"
        _write_two_frame_wav(injection_wav)
        sleep_calls = []

        async def fake_sleep(delay_s):
            sleep_calls.append(delay_s)
            fake_now_ns[0] += round(delay_s * 1_000_000_000)

        async def immediate_to_thread(function, *args, **kwargs):
            return function(*args, **kwargs)

        monkeypatch.setattr(duplex_stream_e2e.asyncio, "sleep", fake_sleep)
        monkeypatch.setattr(duplex_stream_e2e.asyncio, "to_thread", immediate_to_thread)

        probe = RealtimePairProbe(
            "ws://unused", tmp_path / "injected", max_turns=1, max_duration_s=5,
            bargein_wav=injection_wav,
            bargein_isolated_observation_s=8,
        )
        peer_a, peer_b = make_peer("A"), make_peer("B")
        probe.peers = {"A": peer_a, "B": peer_b}
        probe.started_ns = 0
        logs = []
        probe.log = lambda speaker, event: logs.append({"speaker": speaker, **event})

        peer_a["ws"] = FakeWebSocket(probe, peer_a, [
            ({"type": "response.created", "response_id": "A-response"}, 400_000_000),
            (response_audio_event("A-response", 4, np.ones(1920, dtype="<i2")), 500_000_000),
            ({"type": "response.done", "response_id": "A-response"}, 600_000_000),
        ])
        await probe._receive_loop(peer_a)
        assert peer_a["queue"].empty()
        assert peer_b["queue"].empty()

        probe.stop = asyncio.Event()
        trigger_ns = 1_000_000_000
        peer_b["ws"] = FakeWebSocket(probe, peer_b, [
            ({"type": "response.created", "response_id": "B-response"}, 700_000_000),
            (response_audio_event("B-response", 6, np.empty(0, dtype="<i2")), 800_000_000),
            (response_audio_event("B-response", 7, np.ones(1920, dtype="<i2")), trigger_ns),
            (response_audio_event("B-response", 8, np.ones(1920, dtype="<i2")), 1_700_000_000),
            ({"type": "response.done", "response_id": "B-response"}, 1_800_000_000),
        ], wait_after_first_audio=True)
        await probe._receive_loop(peer_b)
        assert probe.bargein_task is not None and probe.bargein_task.done()
        assert probe.isolated_observation_active is True
        assert probe.auto_response_disable_task is None
        assert probe.auto_response_disable_reason is None
        assert peer_a["route_queue"].empty()
        assert peer_b["route_queue"].empty()
        assert len(sleep_calls) == 1
        assert sleep_calls[0] == pytest.approx(0.5)
        assert fake_now_ns[0] == 1_800_000_000
        assert peer_a["queue"].empty()
        assert peer_b["queue"].qsize() == 2

        queued = [peer_b["queue"].get_nowait() for _ in range(peer_b["queue"].qsize())]
        injection_id = "bargein:B:B-response:generation-7"
        assert all(item[1] == injection_id for item in queued)
        assert all(item[4]["source_kind"] == "injection" for item in queued)
        assert all(item[4]["trigger_response_id"] == "B-response" for item in queued)
        assert all(item[4]["generation_id"] == 7 for item in queued)
        assert all(item[4]["trigger_received_at_ns"] == trigger_ns for item in queued)
        assert all(item[4]["delay_ms"] == 500 for item in queued)
        assert [item[4]["frame_index"] for item in queued] == [0, 1]
        assert [(item[4]["source_sample_start"], item[4]["source_sample_end"])
                for item in queued] == [(0, 1280), (1280, 2560)]

        peer_b["queue"].put_nowait(queued[0])
        peer_b["queue"].put_nowait(queued[1])
        peer_b["queue_frames_enqueued"] = 2
        peer_b["queue_high_water"] = 2
        peer_b["stop_sender"] = True
        peer_b["ws"].auto_ack = False
        peer_b["ws"].events = [
            ({"type": "input_audio_buffer.consumed", "frame_id": 0}, 1_900_000_000),
            ({"type": "input_audio_buffer.consumed", "frame_id": 1}, 2_000_000_000),
        ]
        peer_b["ws"].index = 0
        probe.stop = asyncio.Event()
        await asyncio.wait_for(asyncio.gather(
            probe._send_loop(peer_b),
            probe._receive_loop(peer_b),
        ), timeout=2)

        sent_events = [
            event for event in logs
            if event.get("type") == "probe.input_frame_sent"
            and event.get("source_kind") == "injection"
        ]
        assert len(sent_events) == 2
        assert all(event["injection_id"] == injection_id for event in sent_events)
        assert all(event["trigger_response_id"] == "B-response" for event in sent_events)
        assert len(peer_b["ws"].payloads) == 2
        assert all(payload["type"] == "input_audio_buffer.append" for payload in peer_b["ws"].payloads)
        ack_events = [
            event for event in logs
            if event.get("type") == "probe.input_frame_ack"
            and event.get("source_kind") == "injection"
        ]
        assert len(ack_events) == 2
        assert all(event["injection_id"] == injection_id for event in ack_events)
        assert all(event["trigger_response_id"] == "B-response" for event in ack_events)
        assert probe.bargein_injections[injection_id]["status"] == "sent"
        assert probe.bargein_injections[injection_id]["source_kind"] == "injection"
        sent_summary = next(event for event in logs if event.get("type") == "probe.bargein_injection_sent")
        assert sent_summary["source_kind"] == "injection"
        assert sent_summary["trigger_received_at_ns"] == trigger_ns
        assert sent_summary["generation_id"] == 7
        assert sent_summary["delay_ms"] == 500
        assert sent_summary["enqueued_at_ns"] == trigger_ns + 500_000_000
        assert sent_summary["frame_count"] == 2
        assert sent_summary["first_sent_at_ns"] is not None
        assert sent_summary["last_sent_at_ns"] is not None

        # A send already inside the target WebSocket is allowed to finish, but
        # the single sender must place the correction after it and suppress any
        # stale relay that reaches the queue after isolation starts.
        isolated_probe = RealtimePairProbe(
            "ws://unused", tmp_path / "inflight-isolation", max_turns=1,
            max_duration_s=5, bargein_wav=injection_wav,
            bargein_isolated_observation_s=8,
        )
        isolated_peer = make_peer("B")
        isolated_probe.peers = {"A": make_peer("A"), "B": isolated_peer}
        isolated_probe.stop = asyncio.Event()
        isolation_logs = []
        isolated_probe.log = lambda speaker, event: isolation_logs.append(
            {"speaker": speaker, **event}
        )

        class BlockingFirstSendWebSocket:
            def __init__(self, peer):
                self.peer = peer
                self.payloads = []
                self.first_send_started = asyncio.Event()
                self.release_first_send = asyncio.Event()

            async def send(self, payload):
                self.payloads.append(json.loads(payload))
                if len(self.payloads) == 1:
                    self.first_send_started.set()
                    await self.release_first_send.wait()
                self.peer["ack_pending"].clear()
                self.peer["ack_wakeup"].set()

        barrier_ws = BlockingFirstSendWebSocket(isolated_peer)
        isolated_peer["ws"] = barrier_ws
        isolated_peer["stop_sender"] = True
        isolated_peer["queue"].put_nowait((
            np.ones(duplex_stream_e2e.FRAME_SAMPLES, dtype=np.float32),
            "prior-response", (), time.monotonic_ns(), {"source_kind": "relay"},
        ))
        isolated_peer["queue"].put_nowait((
            np.ones(duplex_stream_e2e.FRAME_SAMPLES, dtype=np.float32),
            "queued-response", (), time.monotonic_ns(), {"source_kind": "relay"},
        ))
        sender_task = asyncio.create_task(isolated_probe._send_loop(isolated_peer))
        await asyncio.wait_for(barrier_ws.first_send_started.wait(), timeout=1)
        isolated_probe._begin_isolated_bargein_observation("trigger-response")
        isolation_event = next(
            event for event in isolation_logs
            if event["type"] == "probe.isolated_bargein_observation_started"
        )
        assert isolation_event["preexisting_websocket_sends"] == [{
            "speaker": "B",
            "response_id": "prior-response",
            "source_kind": "relay",
            "send_started_at_ns": isolation_event["preexisting_websocket_sends"][0]["send_started_at_ns"],
        }]
        assert isolation_event["preexisting_websocket_sends"][0]["send_started_at_ns"] is not None
        assert isolated_peer["queue"].empty()
        queue_clear_event = next(
            event for event in isolation_logs
            if event["type"] == "probe.isolated_bargein_relay_queues_cleared"
        )
        assert queue_clear_event["dropped_queued_relay_items"] == 1
        injection_meta = {
            "source_kind": "injection", "injection_id": "isolated-injection",
            "trigger_response_id": "trigger-response", "generation_id": 1,
            "frame_index": 0, "frame_count": 1,
            "source_sample_start": 0, "source_sample_end": duplex_stream_e2e.FRAME_SAMPLES,
            "source_sample_rate_hz": duplex_stream_e2e.INPUT_RATE,
        }
        isolated_peer["queue"].put_nowait((
            np.ones(duplex_stream_e2e.FRAME_SAMPLES, dtype=np.float32),
            "isolated-injection", (), time.monotonic_ns(), injection_meta,
        ))
        isolated_peer["queue"].put_nowait((
            np.ones(duplex_stream_e2e.FRAME_SAMPLES, dtype=np.float32),
            "late-response", (), time.monotonic_ns(), {"source_kind": "relay"},
        ))
        barrier_ws.release_first_send.set()
        await asyncio.wait_for(sender_task, timeout=2)
        send_records = isolated_peer["input_frame_send_records"]
        assert [record["source_response_id"] for record in send_records] == [
            "prior-response", "isolated-injection",
        ]
        assert send_records[0]["sent_at_ns"] <= send_records[1]["send_started_at_ns"]
        injection_sent = next(
            event for event in isolation_logs
            if event["type"] == "probe.input_frame_sent"
            and event.get("source_kind") == "injection"
        )
        assert injection_sent["sent_at_ns"] >= isolation_event["started_at_ns"]
        assert isolated_peer["isolated_relay_frames_dropped"] == 2
        assert len(barrier_ws.payloads) == 2

    asyncio.run(run())
