import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import ssl
import time
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from tools.real_dialogue_simulator import (
    DEFAULT_MAX_GAP_MS,
    DEFAULT_PACING_TOLERANCE_MS,
    BROWSER_PLAYBACK_MAX_SECONDS,
    BROWSER_PLAYBACK_MAX_SAMPLES,
    EvidenceError,
    FixtureSpec,
    PacingMonitor,
    RealDialogueSimulator,
    ResponseRecord,
    TURN_POLICY,
    _extract_kv_evidence_from_rows,
    _long_response_input_window_metrics,
    _offline_finalize_report,
    _playback_capacity_samples,
    _json_safe,
    _device_samples_due_since_epoch,
    load_fixture_manifest,
    make_noise_frame_with_metrics,
    prepare_fixture,
)
from client.openai_duplex_client import BoundedPCMPlayback, CODEC_FRAME_SAMPLES, CODEC_SAMPLE_RATE


def test_noise_floor_is_seeded_and_pcm16_safe():
    first = make_noise_frame_with_metrics(np.random.default_rng(19), 0.002)
    second = make_noise_frame_with_metrics(np.random.default_rng(19), 0.002)

    assert np.array_equal(first[0], second[0])
    assert first[1] == pytest.approx(0.002)
    assert first[2] == 0
    assert first[0].dtype == np.dtype("<i2")


def test_fixture_prequantization_clipping_fails_closed(tmp_path):
    audio_path = tmp_path / "rail.wav"
    samples = np.full(1280, 32767, dtype="<i2")
    with wave.open(str(audio_path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(samples.tobytes())
    spec = FixtureSpec(
        key="rail",
        path=audio_path,
        asr_path=tmp_path / "asr.json",
        transcript="speech",
        asr_model="small",
        asr_device="cpu",
        language="it",
        transcript_verified=True,
    )

    with pytest.raises(EvidenceError, match="clip before PCM16 quantization"):
        prepare_fixture(spec, seed=19, noise_rms=0.002)


def test_manifest_reads_offline_whisper_segments_and_checks_exact_stop_phrase(tmp_path):
    names = {
        "long_request": "long.wav",
        "interruption": "stop.wav",
        "memory_followup": "followup.wav",
        "closure": "closure.wav",
    }
    fixtures = {}
    for key, filename in names.items():
        (tmp_path / filename).write_bytes(b"fixture placeholder")
        asr_name = f"{key}.asr.json"
        text = "Aspetta, fermati un momento." if key == "interruption" else f"testo {key}"
        (tmp_path / asr_name).write_text(json.dumps({
            "model": "faster-whisper small",
            "device": "cpu",
            "compute_type": "int8",
            "results": [{
                "file": filename,
                "language": "it",
                "segments": [{"text": text}],
            }],
        }), encoding="utf-8")
        fixtures[key] = {
            "path": filename,
            "asr_file": asr_name,
            "asr_model": "faster-whisper small",
            "asr_device": "cpu",
            "language": "it",
            "transcript_verified": True,
            "transcript": text,
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"fixtures": fixtures}), encoding="utf-8")

    specs, _ = load_fixture_manifest(manifest)

    assert specs["interruption"].transcript == "Aspetta, fermati un momento."
    assert specs["interruption"].asr_model == "faster-whisper small"
    assert specs["interruption"].asr_device == "cpu"
    assert specs["interruption"].asr_compute_type == "int8"


@pytest.mark.parametrize("phase", ["session.created", "session.updated"])
def test_handshake_timeout_reports_phase_elapsed_time_and_task(phase):
    class SilentSocket:
        async def recv(self):
            await asyncio.Future()

    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = argparse.Namespace(handshake_timeout_s=0.005)
    simulator.socket = SilentSocket()

    with pytest.raises(TimeoutError) as caught:
        asyncio.run(simulator._recv_initial(phase=phase))

    message = str(caught.value)
    assert f"timed out waiting for {phase}" in message
    assert re.search(r"after \d+\.\d{3}s", message)
    assert re.search(r"\(task='[^']+'\)$", message)


def test_pacing_tolerance_and_gap_are_independent_explicit_gates():
    monitor = PacingMonitor(
        tolerance_ms=DEFAULT_PACING_TOLERANCE_MS,
        max_gap_ms=DEFAULT_MAX_GAP_MS,
    )
    monitor.record_send(0)
    monitor.record_send(100_000_000)  # 20 ms late, but gap remains below 160 ms.
    assert monitor.report()["max_lateness_ms"] == pytest.approx(20.0)

    late = PacingMonitor(tolerance_ms=20, max_gap_ms=160)
    late.record_send(0)
    with pytest.raises(EvidenceError, match="scheduling lateness"):
        late.record_send(100_000_001)

    gap = PacingMonitor(tolerance_ms=80, max_gap_ms=160)
    gap.record_send(0)
    with pytest.raises(EvidenceError, match="frame gap"):
        gap.record_send(160_000_000)


def test_pacing_backpressure_wait_is_excluded_from_the_frame_gap_limit():
    monitor = PacingMonitor(tolerance_ms=1000, max_gap_ms=160)
    monitor.record_send(0)
    # The sender rebases its deadline after the ACK wait; 320 ms of this
    # 396 ms inter-send interval is attributable to backpressure.
    monitor.deadline_ns = 376_000_000

    monitor.record_send(396_000_000, backpressure_wait_ns=320_000_000)

    assert monitor.samples[-1]["gap_ns"] == 396_000_000
    assert monitor.samples[-1]["backpressure_wait_ns"] == 320_000_000
    assert monitor.samples[-1]["net_gap_ns"] == 76_000_000
    assert monitor.max_gap_observed_ns == 76_000_000
    assert monitor.report()["max_interframe_gap_ms"] == pytest.approx(396.0)
    assert monitor.report()["max_net_interframe_gap_ms"] == pytest.approx(76.0)


def test_pacing_backpressure_does_not_mask_an_over_limit_net_gap():
    monitor = PacingMonitor(tolerance_ms=1000, max_gap_ms=160)
    monitor.record_send(0)
    monitor.deadline_ns = 596_000_000

    with pytest.raises(EvidenceError, match="frame gap"):
        monitor.record_send(796_000_000, backpressure_wait_ns=40_000_000)


def test_pacing_without_backpressure_still_enforces_raw_frame_gap_limit():
    monitor = PacingMonitor(tolerance_ms=1000, max_gap_ms=160)
    monitor.record_send(0)

    with pytest.raises(EvidenceError, match="frame gap"):
        monitor.record_send(160_000_000)


def _long_window_sends_with_one_backpressure_gap(backpressure_wait_ns):
    start_ns = 5_000_000_000
    sends = []
    acks = []
    for frame_id in range(503):
        after_stall = frame_id >= 250
        sent_ns = start_ns + frame_id * 80_000_000 + (316_000_000 if after_stall else 0)
        wait_ns = backpressure_wait_ns if frame_id == 250 else 0
        sends.append({
            "local_frame_id": frame_id,
            "sent_ns": sent_ns,
            "backpressure_wait_ns": wait_ns,
        })
        # Model ACKs continuing through the sender's wait as earlier frames
        # leave the bounded in-flight queue.
        ack_delay_ns = max(0, frame_id - 244) * 75_000_000 if 245 <= frame_id <= 249 else 0
        acks.append({
            "local_frame_id": frame_id,
            "received_ns": sent_ns + 10_000_000 + ack_delay_ns,
        })
    return start_ns, sends, acks


def test_long_response_continuity_subtracts_backpressure_but_keeps_raw_gap():
    start_ns, sends, acks = _long_window_sends_with_one_backpressure_gap(320_000_000)

    metrics = _long_response_input_window_metrics(
        pcm_samples=41 * CODEC_SAMPLE_RATE,
        first_audio_ns=start_ns - 500_000_000,
        done_ns=start_ns + 500_000_000,
        playout_started_ns=start_ns,
        playout_completed_ns=start_ns + 41_000_000_000,
        final_observation_ns=start_ns + 41_240_000_000,
        input_frame_sends=sends,
        input_frame_acks=acks,
        fifo_peak_samples=32_000,
    )

    assert metrics["max_input_send_gap_ms"] == pytest.approx(396.0)
    assert metrics["max_input_send_net_gap_ms"] == pytest.approx(80.0)


def test_long_response_continuity_fails_when_net_backpressure_gap_is_over_limit():
    start_ns, sends, acks = _long_window_sends_with_one_backpressure_gap(200_000_000)

    with pytest.raises(EvidenceError, match="send gap after backpressure accounting"):
        _long_response_input_window_metrics(
            pcm_samples=41 * CODEC_SAMPLE_RATE,
            first_audio_ns=start_ns - 500_000_000,
            done_ns=start_ns + 500_000_000,
            playout_started_ns=start_ns,
            playout_completed_ns=start_ns + 41_000_000_000,
            final_observation_ns=start_ns + 41_240_000_000,
            input_frame_sends=sends,
            input_frame_acks=acks,
            fifo_peak_samples=32_000,
        )


def test_long_response_reports_pcm_production_and_playout_separately():
    start_ns = 5_000_000_000
    sent = [
        {"local_frame_id": frame_id, "sent_ns": start_ns + frame_id * 80_000_000}
        for frame_id in range(503)
    ]
    acked = [
        {"local_frame_id": frame_id, "received_ns": start_ns + frame_id * 80_000_000 + 10_000_000}
        for frame_id in range(503)
    ]

    metrics = _long_response_input_window_metrics(
        pcm_samples=40 * CODEC_SAMPLE_RATE,
        first_audio_ns=start_ns - 500_000_000,
        done_ns=start_ns + 500_000_000,
        playout_started_ns=start_ns,
        playout_completed_ns=start_ns + 40_000_000_000,
        final_observation_ns=start_ns + 40_240_000_000,
        input_frame_sends=sent,
        input_frame_acks=acked,
        fifo_peak_samples=32_000,
    )

    assert metrics["pcm_duration_seconds"] == 40.0
    assert metrics["production_window_seconds"] == 1.0
    assert metrics["playout_window_seconds"] == 40.0
    assert metrics["input_ack_cutoff_ns"] == start_ns + 40_240_000_000
    # Frame 500 was sent exactly at playout completion and its ACK arrived
    # during final observation; frames 501-502 were sent after playback.
    assert metrics["input_frames_acked"] == 501
    assert metrics["max_input_send_gap_ms"] == 80.0
    assert metrics["max_input_ack_gap_ms"] == 80.0


def test_long_response_counts_inflight_ack_after_final_observation_within_cutoff():
    start_ns = 5_000_000_000
    playout_end_ns = start_ns + 40_000_000_000
    sent = [
        {"local_frame_id": frame_id, "sent_ns": start_ns + frame_id * 80_000_000}
        for frame_id in range(501)
    ]
    acked = [
        {
            "local_frame_id": item["local_frame_id"],
            "received_ns": item["sent_ns"] + (
                75_000_000 if item["local_frame_id"] == 500 else 10_000_000
            ),
        }
        for item in sent
    ]
    ack_cutoff_ns = playout_end_ns + 500_000_000

    metrics = _long_response_input_window_metrics(
        pcm_samples=40 * CODEC_SAMPLE_RATE,
        first_audio_ns=start_ns - 500_000_000,
        done_ns=start_ns + 500_000_000,
        playout_started_ns=start_ns,
        playout_completed_ns=playout_end_ns,
        final_observation_ns=playout_end_ns + 45_000_000,
        input_ack_cutoff_ns=ack_cutoff_ns,
        input_frame_sends=sent,
        input_frame_acks=acked,
        fifo_peak_samples=32_000,
    )

    assert metrics["input_frames_acked"] == 501
    assert metrics["ack_ratio"] == 1.0
    assert metrics["input_ack_cutoff_ns"] == ack_cutoff_ns


def test_post_playout_microphone_frames_do_not_extend_continuity_window():
    start_ns = 5_000_000_000
    playout_end_ns = start_ns + 5_000_000_000
    # Forty seconds of input are present overall, but only five seconds overlap
    # playback; the remaining frames arrive during post-playout observation.
    sent = [
        {"local_frame_id": frame_id, "sent_ns": start_ns + frame_id * 80_000_000}
        for frame_id in range(501)
    ]
    acked = [
        {"local_frame_id": item["local_frame_id"], "received_ns": item["sent_ns"] + 10_000_000}
        for item in sent
    ]

    with pytest.raises(EvidenceError, match="carried only"):
        _long_response_input_window_metrics(
            pcm_samples=40 * CODEC_SAMPLE_RATE,
            first_audio_ns=start_ns - 500_000_000,
            done_ns=start_ns + 500_000_000,
            playout_started_ns=start_ns,
            playout_completed_ns=playout_end_ns,
            final_observation_ns=playout_end_ns + 35_000_000_000,
            input_frame_sends=sent,
            input_frame_acks=acked,
            fifo_peak_samples=32_000,
            minimum_frames=500,
            minimum_playout_seconds=5.0,
        )


def _tls_simulator(*, url, mode="acceptance", insecure=False, ca_cert=None, tls_server_name=None):
    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = argparse.Namespace(
        url=url,
        mode=mode,
        insecure_loopback=insecure,
        ca_cert=ca_cert,
        tls_server_name=tls_server_name,
    )
    simulator.evidence_gaps = []
    simulator.tls_evidence = {
        "scheme": None,
        "peer_certificate_verified": None,
        "ca_certificate": None,
        "insecure_loopback_opt_in": insecure,
    }
    return simulator


def test_tls_is_verified_by_default_and_insecure_transport_is_diagnostic_loopback_only():
    secure = _tls_simulator(url="wss://127.0.0.1:8443/v1/realtime?duplex=1")
    options = secure._websocket_tls_options()
    assert options["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert options["ssl"].check_hostname
    assert secure.tls_evidence["peer_certificate_verified"] is True

    named = _tls_simulator(
        url="wss://127.0.0.1:8443/v1/realtime?duplex=1",
        tls_server_name="mondaini",
    )
    named_options = named._websocket_tls_options()
    assert named_options["server_hostname"] == "mondaini"
    assert named_options["ssl"].verify_mode == ssl.CERT_REQUIRED
    assert named_options["ssl"].check_hostname
    assert named.tls_evidence["server_hostname_override"] == "mondaini"

    with pytest.raises(EvidenceError, match="unencrypted ws"):
        _tls_simulator(url="ws://127.0.0.1:8443")._websocket_tls_options()

    diagnostic = _tls_simulator(
        url="wss://localhost:8443/v1/realtime?duplex=1",
        mode="diagnostic",
        insecure=True,
    )
    assert diagnostic._websocket_tls_options()["ssl"].verify_mode == ssl.CERT_NONE
    assert diagnostic.evidence_gaps

    with pytest.raises(EvidenceError, match="cannot be combined"):
        _tls_simulator(
            url="wss://127.0.0.1:8443",
            mode="diagnostic",
            insecure=True,
            tls_server_name="mondaini",
        )._websocket_tls_options()

    with pytest.raises(EvidenceError, match="diagnostic loopback"):
        _tls_simulator(
            url="wss://127.0.0.1:8443",
            mode="acceptance",
            insecure=True,
        )._websocket_tls_options()
    with pytest.raises(EvidenceError, match="diagnostic loopback"):
        _tls_simulator(
            url="wss://example.com:8443",
            mode="diagnostic",
            insecure=True,
        )._websocket_tls_options()


def _quality_simulator():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.runtime_audio_telemetry_complete = True
    return simulator


def _record():
    return ResponseRecord(response_id="r1", created_event={}, created_ns=0)


def test_audio_telemetry_rejects_missing_nonfinite_and_wrong_sample_count():
    simulator = _quality_simulator()
    record = _record()
    simulator._record_audio_quality({}, record, expected_samples=8)
    assert not record.audio_telemetry_complete
    assert not simulator.runtime_audio_telemetry_complete

    simulator = _quality_simulator()
    record = _record()
    with pytest.raises(EvidenceError, match="NaN or Inf"):
        simulator._record_audio_quality({
            "raw_source_audio_quality": {
                "finite": True,
                "rms": float("nan"),
                "peak": 0.2,
                "clipping_rail_samples": 0,
                "sample_count": 8,
            },
        }, record, expected_samples=8)

    simulator = _quality_simulator()
    with pytest.raises(EvidenceError, match="NaN or Inf"):
        simulator._record_audio_quality({
            "raw_source_audio_quality": {
                "finite": False,
                "rms": None,
                "peak": None,
                "clipping_rail_samples": 0,
                "sample_count": 8,
            },
        }, _record(), expected_samples=8)

    simulator = _quality_simulator()
    with pytest.raises(EvidenceError, match="clipped rail"):
        simulator._record_audio_quality({
            "raw_source_audio_quality": {
                "finite": True,
                "rms": 0.1,
                "peak": 1.0,
                "clipping_rail_samples": 1,
                "sample_count": 8,
            },
        }, _record(), expected_samples=8)

    simulator = _quality_simulator()
    record = _record()
    simulator._record_audio_quality({
        "raw_source_audio_quality": {
            "finite": True,
            "rms": 0.1,
            "peak": 0.2,
            "clipping_rail_samples": 0,
            "sample_count": 7,
        },
    }, record, expected_samples=8)
    assert not record.audio_telemetry_complete


def _telemetry_args(**overrides):
    values = {
        "vram_peak_over_baseline_budget_mib": 500.0,
        "vram_post_drain_growth_budget_mib": 100.0,
        "vram_reserved_peak_growth_budget_mib": 500.0,
        "vram_reserved_post_drain_growth_budget_mib": 100.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _telemetry_simulator(args=None):
    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = args or _telemetry_args()
    simulator.session_created = {"session_id": "session-test"}
    simulator.run_started_at_utc = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
    simulator.run_finished_at_utc = simulator.run_started_at_utc + timedelta(seconds=10)
    simulator.evidence_gaps = []
    simulator.runtime_kv_evidence = [{"values": {"confirmed_exclusive_kv_cursor": 4}}]
    simulator.runtime_audio_telemetry_complete = True
    simulator.responses = {
        tag: ResponseRecord(response_id=f"response-{tag}", created_event={}, created_ns=0, scenario_tag=tag)
        for tag in (
            "long_response_completed",
            "long_response_interrupted",
            "memory_followup_resume",
            "closure",
        )
    }
    return simulator


def _valid_runtime_telemetry():
    tags = (
        "long_response_completed",
        "long_response_interrupted",
        "memory_followup_resume",
        "closure",
    )
    return {
        "session_id": "session-test",
        "run_window_utc": {
            "started_at": "2026-10-25T12:00:00+00:00",
            "finished_at": "2026-10-25T12:00:10+00:00",
        },
        "cuda_allocator": {
            "status": "ok",
            "device": "cuda:0",
            "allocated_bytes": 1000,
            "reserved_bytes": 1200,
            "peak_allocated_bytes": 1350,
            "peak_reserved_bytes": 1600,
            "sampled_at_monotonic_ns": 500,
        },
        "kv": {"cursor": 4},
        "vram": {
            "post_warmup_baseline": {"allocated_mib": 1000, "reserved_mib": 1200},
            "active": {"allocated_mib": 1300, "reserved_mib": 1500},
            "peak": {"allocated_mib": 1350, "reserved_mib": 1600},
            "turns": [
                {
                    "response_id": f"response-{tag}",
                    "scenario_tag": tag,
                    "post_drain": {"allocated_mib": 1010 + index, "reserved_mib": 1210 + index},
                }
                for index, tag in enumerate(tags)
            ],
        },
    }


def test_telemetry_requires_budgets_and_reports_short_trend_without_leak_claim():
    simulator = _telemetry_simulator(_telemetry_args(
        vram_peak_over_baseline_budget_mib=None,
        vram_post_drain_growth_budget_mib=None,
        vram_reserved_peak_growth_budget_mib=None,
        vram_reserved_post_drain_growth_budget_mib=None,
    ))
    assessment = simulator._assess_telemetry(_valid_runtime_telemetry())
    assert assessment["status"] == "INCOMPLETE"
    assert any("budgets" in item for item in simulator.evidence_gaps)

    simulator = _telemetry_simulator()
    assessment = simulator._assess_telemetry(_valid_runtime_telemetry())
    assert assessment["status"] == "PASS"
    assert assessment["within_budget"] is True
    assert "4 samples do not certify absence of a leak" in assessment["trend_interpretation"]

    simulator = _telemetry_simulator(_telemetry_args(vram_peak_over_baseline_budget_mib=100.0))
    assessment = simulator._assess_telemetry(_valid_runtime_telemetry())
    assert assessment["status"] == "FAIL"
    assert assessment["within_budget"] is False

    simulator = _telemetry_simulator(_telemetry_args(vram_peak_over_baseline_budget_mib=float("nan")))
    assessment = simulator._assess_telemetry(_valid_runtime_telemetry())
    assert assessment["status"] == "INCOMPLETE"
    assert any("finite and non-negative" in item for item in simulator.evidence_gaps)


def test_telemetry_requires_per_turn_post_drain_and_session_identity():
    simulator = _telemetry_simulator()
    runtime = _valid_runtime_telemetry()
    runtime["vram"]["turns"].pop()
    assessment = simulator._assess_telemetry(runtime)
    assert assessment["status"] == "INCOMPLETE"

    simulator = _telemetry_simulator()
    runtime = _valid_runtime_telemetry()
    runtime["session_id"] = "stale-session"
    assessment = simulator._assess_telemetry(runtime)
    assert assessment["status"] == "INCOMPLETE"
    assert any("session_id" in item for item in simulator.evidence_gaps)


def _server_log_simulator(tmp_path, raw_log):
    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = argparse.Namespace(
        server_log=str(tmp_path / "server.log"),
        server_log_window=str(tmp_path / "server-log-window.json"),
    )
    simulator.session_created = {"session_id": "session-abc12345"}
    simulator.responses = {"resp-long-1": object(), "resp-stop-2": object()}
    simulator.evidence_gaps = []
    simulator.run_started_at_utc = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
    simulator.run_finished_at_utc = simulator.run_started_at_utc + timedelta(seconds=10)
    since = simulator.run_started_at_utc.isoformat()
    until = simulator.run_finished_at_utc.isoformat()
    raw_bytes = raw_log.encode("utf-8")
    (tmp_path / "server.log").write_bytes(raw_bytes)
    (tmp_path / "server-log-window.json").write_text(json.dumps({
        "schema": "duplex_server_log_window_v1",
        "session_id": "session-abc12345",
        "since_utc": since,
        "until_utc": until,
        "source": "docker logs --since/--until",
    }), encoding="utf-8")
    return simulator


def test_server_log_ignores_historical_global_markers_outside_run_window(tmp_path):
    simulator = _server_log_simulator(tmp_path, "\n".join((
        "2026-10-25T11:59:40Z historical KV cache_position=900 memory_allocated=9000",
        "2026-10-25T12:00:01Z [realtime session-abc resp-long-1] websocket session opened",
        "2026-10-25T12:00:03Z [realtime session-abc resp-stop-2] response cancelled",
    )))

    evidence = simulator._read_server_log()

    assert evidence is not None
    assert evidence["capture_window_valid"] is False
    assert evidence["correlated_to_this_run"] is False
    assert evidence["contains_kv_telemetry"] is False
    assert evidence["contains_cuda_memory_telemetry"] is False
    assert evidence["out_of_window_records"] == 1
    assert any("outside the run window" in item for item in evidence["capture_errors"])


def test_server_log_accepts_current_process_window_and_multiline_unidentified_cuda_record(tmp_path):
    simulator = _server_log_simulator(tmp_path, "\n".join((
        "2026-10-25T12:00:01Z [realtime session-abc resp-long-1] cache_position=420",
        "2026-10-25T12:00:02Z response_talk_context_pool memory_allocated=14544 memory_reserved=15744",
        "  diagnostic stack continuation has no repeated session or response id",
        "2026-10-25T12:00:09Z [realtime session-abc resp-stop-2] response cancelled",
    )))

    evidence = simulator._read_server_log()

    assert evidence is not None
    assert evidence["capture_window_valid"] is True
    assert evidence["correlated_to_this_run"] is True
    assert evidence["contains_kv_telemetry"] is True
    assert evidence["contains_cuda_memory_telemetry"] is True
    assert evidence["exception_markers"] == []
    assert evidence["timestamped_records"] == 3
    assert evidence["out_of_window_records"] == 0
    assert evidence["capture_errors"] == []


def test_server_log_requires_matching_current_session_window(tmp_path):
    simulator = _server_log_simulator(tmp_path, "\n".join((
        "2026-10-25T12:00:01Z [realtime session-abc resp-long-1] cache_position=420",
        "2026-10-25T12:00:02Z response_talk_context_pool memory_allocated=14544",
    )))
    window_path = tmp_path / "server-log-window.json"
    window = json.loads(window_path.read_text(encoding="utf-8"))
    window["session_id"] = "historical-session"
    window_path.write_text(json.dumps(window), encoding="utf-8")

    evidence = simulator._read_server_log()

    assert evidence is not None
    assert evidence["capture_window_valid"] is False
    assert evidence["correlated_to_this_run"] is False
    assert evidence["contains_kv_telemetry"] is False
    assert evidence["contains_cuda_memory_telemetry"] is False
    assert any("session_id does not match" in item for item in evidence["capture_errors"])


def test_server_log_needs_scoped_session_and_zero_errors_not_cuda_keyword(tmp_path):
    simulator = _server_log_simulator(tmp_path, "\n".join((
        "2026-10-25T12:00:01Z [realtime session-abc resp-long-1] websocket session opened",
        "2026-10-25T12:00:09Z [realtime session-abc resp-stop-2] response cancelled",
    )))

    evidence = simulator._read_server_log()

    assert evidence is not None
    assert evidence["correlated_to_this_run"] is True
    assert evidence["capture_window_valid"] is True
    assert evidence["contains_cuda_memory_telemetry"] is False
    assert evidence["exception_markers"] == []
    assert simulator.evidence_gaps == []

    (tmp_path / "server.log").write_text(
        "2026-10-25T12:00:02Z [realtime session-abc resp-long-1] ERROR CUDA allocator failed\n",
        encoding="utf-8",
    )
    simulator.evidence_gaps.clear()
    failed_evidence = simulator._read_server_log()
    assert failed_evidence is not None
    assert failed_evidence["exception_markers"]
    assert any("exception/error marker" in item for item in simulator.evidence_gaps)


def test_telemetry_requires_cuda_allocator_device_and_matching_run_window():
    simulator = _telemetry_simulator()
    runtime = _valid_runtime_telemetry()
    runtime["cuda_allocator"]["device"] = "cpu"
    assessment = simulator._assess_telemetry(runtime)
    assert assessment["status"] == "INCOMPLETE"
    assert any("CUDA device allocator" in item for item in simulator.evidence_gaps)

    simulator = _telemetry_simulator()
    runtime = _valid_runtime_telemetry()
    runtime["run_window_utc"]["started_at"] = "2026-10-25T11:59:00+00:00"
    assessment = simulator._assess_telemetry(runtime)
    assert assessment["status"] == "INCOMPLETE"
    assert any("run window" in item for item in simulator.evidence_gaps)


def test_absolute_playout_clock_carries_timer_jitter_and_keeps_response_fences():
    epoch_ns = 1_000_000_000
    total_device_samples = 0
    callback_block_ns = 10_000_000
    wake_jitter_ns = 600_000
    for callback_index in range(1, 1001):
        deadline_ns = epoch_ns + callback_index * callback_block_ns
        wake_ns = deadline_ns + wake_jitter_ns
        due = _device_samples_due_since_epoch(wake_ns, epoch_ns, total_device_samples)
        assert due >= 0
        total_device_samples += due
    assert total_device_samples == 240_014
    elapsed_ns = 1000 * callback_block_ns + wake_jitter_ns
    assert total_device_samples / (elapsed_ns / 1e9) == pytest.approx(CODEC_SAMPLE_RATE, abs=1.0)

    playback = BoundedPCMPlayback(max_samples=14_400)
    next_callback_index = 1
    clock_samples = 0

    def consume_through(until_ns):
        nonlocal next_callback_index, clock_samples
        while epoch_ns + next_callback_index * callback_block_ns + wake_jitter_ns <= until_ns:
            wake_ns = epoch_ns + next_callback_index * callback_block_ns + wake_jitter_ns
            due = _device_samples_due_since_epoch(wake_ns, epoch_ns, clock_samples)
            if due:
                playback.read_into(np.zeros(due, dtype=np.float32))
                clock_samples += due
            next_callback_index += 1

    def enqueue_packet(response_id, generation_id, packet_index):
        start_sample = packet_index * 3_840
        return playback.enqueue_packet(
            response_id=response_id,
            generation_id=generation_id,
            speaker_segment_id=f"segment-{response_id}",
            packet_id=packet_index,
            first_frame_id=packet_index * 2,
            last_frame_id=packet_index * 2 + 1,
            start_sample=start_sample,
            end_sample=start_sample + 3_840,
            pcm=np.full(3_840, 0.08, dtype=np.float32),
            sample_rate=CODEC_SAMPLE_RATE,
        )

    playback.begin_response("resp-backchannel")
    backchannel_start_ns = epoch_ns + 2_000_000_000
    for packet_index in range(5):
        arrival_ns = backchannel_start_ns + packet_index * 120_000_000
        consume_through(arrival_ns)
        assert enqueue_packet("resp-backchannel", 1, packet_index)
    consume_through(backchannel_start_ns + 500_000_000)
    playback.cancel_response("resp-backchannel", 1)
    assert playback.pending_samples == 0
    assert playback.drain_client_consumed_pcm_acks() == []

    playback.begin_response("resp-speak")
    assert enqueue_packet("resp-backchannel", 1, 5) is False
    speak_start_ns = backchannel_start_ns + 1_000_000_000
    producer_interval_ns = 153_000_000  # 25.1 kHz server PCM rate > 24 kHz device clock.
    with pytest.raises(BufferError, match="capacity exceeded"):
        for packet_index in range(100):
            arrival_ns = speak_start_ns + packet_index * producer_interval_ns
            consume_through(arrival_ns)
            enqueue_packet("resp-speak", 2, packet_index)
    assert playback.pending_samples <= 14_400
    assert playback.drain_client_consumed_pcm_acks() == []  # no segment_done, so no consumption claim


def test_playback_fifo_matches_bounded_browser_horizon(tmp_path):
    browser_source = (Path(__file__).resolve().parents[2] / "server" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    match = re.search(r"const MAX_PLAYBACK_AHEAD_SECONDS = ([0-9.]+);", browser_source)

    assert match is not None
    assert float(match.group(1)) == BROWSER_PLAYBACK_MAX_SECONDS == 4.0
    assert BROWSER_PLAYBACK_MAX_SAMPLES == 96_000
    assert _playback_capacity_samples("acceptance", 4.0) == BROWSER_PLAYBACK_MAX_SAMPLES
    assert _playback_capacity_samples("diagnostic", 0.6) == 14_400
    with pytest.raises(EvidenceError, match="acceptance mode must match"):
        _playback_capacity_samples("acceptance", 0.6)
    with pytest.raises(EvidenceError, match="cannot exceed the browser"):
        _playback_capacity_samples("diagnostic", 4.1)


def _write_synthetic_finalize_bundle(output_dir, *, allocator_device="cuda:0"):
    output_dir.mkdir(parents=True)
    start = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(seconds=52)
    session_id = "session-offline-123456"
    tls = {"scheme": "wss", "peer_certificate_verified": True, "insecure_loopback_opt_in": False}
    fixture_manifest = {"fixtures": {"interruption": {
        "transcript": "Aspetta, fermati un momento.", "transcript_verified": True,
    }}}
    (output_dir / "fixture_manifest.json").write_text(json.dumps(fixture_manifest), encoding="utf-8")
    rows = []

    def add(seconds, event_type, payload, direction=None):
        row = {
            "utc": (start + timedelta(seconds=seconds)).isoformat(),
            "monotonic_ns": int(seconds * 1_000_000_000),
            "type": event_type,
            "payload": payload,
        }
        if direction:
            row["direction"] = direction
        rows.append(row)

    add(0, "run_started", {"started_at_utc": start.isoformat()})
    add(0.1, "session.created", {"session_id": session_id, "turn_policy": TURN_POLICY, "voice": "Aoede"})
    add(0.2, "session.updated", {
        "turn_policy": TURN_POLICY, "voice": "Aoede", "response_profile": "long",
    })
    add(0.3, "transport_verified", tls)
    add(0.4, "client_event", {
        "type": "session.update",
        "session": {
            "auto_response": True,
            "turn_policy": TURN_POLICY,
            "input_audio_transcription": False,
            "input_audio_buffer_acknowledgements": True,
            "response_profile": "long",
        },
    }, direction="client")
    delivery_id = "barge-delivery-1"
    add(0.5, "fixture_scheduled", {"scenario_tag": "barge_in", "delivery_id": delivery_id})

    tags = (
        ("response-long-done", "long_response_completed", 960_000, True, 0.6, 41.0, "length"),
        ("response-memory", "memory_followup_resume", 1_920, True, 41.2, 41.5, "stop"),
        ("response-long-cancel", "long_response_interrupted", 1_920, False, 42.0, None, None),
        ("response-closure", "closure", 1_920, True, 49.5, 49.8, "stop"),
    )
    report_responses = []
    for index, (response_id, tag, sample_count, normally_done, created_at, done_at, finish_reason) in enumerate(tags):
        segment_id = f"segment-{index}"
        text = f"risposta {tag}"
        add(created_at, "response.created", {
            "response_id": response_id,
            "effective_action": "SPEAK",
            "response_profile": "long",
        })
        add(created_at + 0.01, "scenario_response_assigned", {
            "response_id": response_id, "scenario_tag": tag, "action": "SPEAK",
        })
        quality = {
            "finite": True,
            "rms": 0.09,
            "peak": 0.1,
            "clipping_rail_samples": 0,
            "sample_count": sample_count,
        }
        add(created_at + 0.1, "response.audio.delta", {
            "response_id": response_id,
            "speaker_segment_id": segment_id,
            "sample_count": sample_count,
            "raw_source_audio_quality": quality,
        })
        first_audio_ns = int(round((created_at + 0.1) * 1e9))
        playout_start_ns = first_audio_ns
        playout_complete_ns = playout_start_ns + sample_count * 1_000_000_000 // CODEC_SAMPLE_RATE
        add(created_at + 0.2, "response.audio.segment_done", {
            "response_id": response_id, "speaker_segment_id": segment_id,
        })
        if normally_done:
            playout_start = playout_start_ns / 1e9
            playout_complete = playout_complete_ns / 1e9
            add(playout_start, "playout_started", {
                "response_id": response_id,
                "scenario_tag": tag,
                "playout_started_ns": playout_start_ns,
            })
            add(playout_complete, "playout_completed", {
                "response_id": response_id,
                "scenario_tag": tag,
                "playout_started_ns": playout_start_ns,
                "playout_completed_ns": playout_complete_ns,
                "pcm_samples_consumed": sample_count,
            })
        ack_ids = []
        if normally_done:
            sequence = sum(len(item["client_consumed_ack_ids"]) for item in report_responses)
            ack_time = playout_complete_ns / 1e9 + 0.01
            add(ack_time, "client_consumed_pcm", {
                "response_id": response_id,
                "speaker_segment_id": segment_id,
                "sequence": sequence,
                "consumption_semantics": "logical_client_device_buffer",
            })
            ack_ids = [segment_id]
            add(done_at, "response.done", {
                "response_id": response_id, "status": "completed", "finish_reason": finish_reason,
            })
            done_status, cancel_reason = "completed", None
        else:
            add(43.04, "playout_onset", {
                "response_id": response_id,
                "playout_onset_ns": 43_040_000_000,
            })
            done_status, cancel_reason = None, "yield"
        add(created_at + 0.15, "response.output_text.delta", {"response_id": response_id, "delta": text})
        if not normally_done:
            add(46.12, "response.cancelled", {
                "response_id": response_id, "reason": "yield", "finish_reason": "cancelled",
            })
        add(done_at + 0.1 if normally_done else 46.2, "response.unified_native_finalized", {
            "type": "response.unified_native_finalized",
            "response_id": response_id,
            "status": "completed",
            "runtime_metrics": {"confirmed_exclusive_kv_cursor": index + 1},
        })
        safe_id = response_id
        pcm = np.full(sample_count, 2949, dtype="<i2")
        with wave.open(str(output_dir / f"response_{safe_id}_received.wav"), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(CODEC_SAMPLE_RATE)
            target.writeframes(pcm.tobytes())
        vocal_frames = (sample_count // CODEC_FRAME_SAMPLES)
        report_responses.append({
            "response_id": response_id,
            "action": "SPEAK",
            "audio_samples": sample_count,
            "vocal_frames_rms_ge_0_015": vocal_frames,
            "segment_ids": [segment_id],
            "segment_done_ids": [segment_id],
            "client_consumed_ack_ids": ack_ids,
            "done_status": done_status,
            "finish_reason": finish_reason,
            "cancel_reason": cancel_reason,
            "text": text,
            "response_profile": "long",
            "first_audio_ns": first_audio_ns,
            "done_ns": int(done_at * 1e9) if normally_done else None,
            "final_observation_ns": int((done_at + 0.1) * 1e9) if normally_done else None,
            "pcm_duration_seconds": sample_count / CODEC_SAMPLE_RATE,
            "production_window_seconds": (
                done_at - created_at - 0.1 if normally_done else None
            ),
            "playout_started_ns": (
                playout_start_ns if normally_done else None
            ),
            "playout_completed_ns": (
                playout_complete_ns if normally_done else None
            ),
            "playout_window_seconds": (
                sample_count / CODEC_SAMPLE_RATE if normally_done else None
            ),
        })

    first_input_ns = 46_040_000_000
    input_sent_payloads = []
    input_ack_payloads = []
    for index in range(618):
        seconds = 0.68 + index * 0.08
        sent_ns = int(seconds * 1_000_000_000)
        is_vocal = 567 <= index <= 607
        payload = {
            "local_frame_id": index,
            "sent_ns": sent_ns,
            "inflight_after_reservation": 1,
            "fixture": "barge_in" if is_vocal else "idle_noise_floor",
            "delivery_id": delivery_id if is_vocal else None,
            "frame_rms_before_quantization": 0.08 if is_vocal else 0.002,
            "prequantization_pcm16_rail_samples": 0,
            "sender_scheduler_delay_ns": 0,
            "backpressure_wait_ns": 0,
            "waited_for_backpressure": False,
        }
        add(seconds, "input_frame_sent", payload)
        input_sent_payloads.append(payload)
        ack_payload = {"local_frame_id": index}
        add(seconds + 0.01, "input_frame_ack_received", ack_payload)
        input_ack_payloads.append({
            **ack_payload,
            "received_ns": int((seconds + 0.01) * 1_000_000_000),
        })
    add(46.12, "response_cancel_observed", {
        "response_id": "response-long-cancel",
        "reason": "yield",
        "cancelled_ns": first_input_ns + 50_000_000,
        "flush_ns": first_input_ns + 80_000_000,
    })
    for index in range(len(tags)):
        add(50.5, "runtime_kv_snapshot", {"response_id": f"response-{index}"})
    add(50.7, "scenario_completed", {
        "responses": len(tags),
        "input_frames_sent": len(input_sent_payloads),
        "input_frames_acked": len(input_ack_payloads),
        "playback_pending_samples": 0,
        "async_tasks_stopped": True,
    })
    add(52.0, "run_finished", {"finished_at_utc": end.isoformat(), "fatal_error": None})
    rows.sort(key=lambda row: (row["monotonic_ns"], row["type"]))
    event_path = output_dir / "events.jsonl"
    event_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    fifo_peak = 80_000
    fifo_trace_rows = [
        {"monotonic_ns": 20_000_000_000, "pending_before_samples": 70_000, "pending_after_samples": fifo_peak},
        {"monotonic_ns": 39_000_000_000, "pending_before_samples": fifo_peak, "pending_after_samples": 65_000},
        {"monotonic_ns": 41_100_000_000, "pending_before_samples": 0, "pending_after_samples": 0},
    ]
    (output_dir / "playback_queue_trace.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in fifo_trace_rows), encoding="utf-8"
    )
    long_window = _long_response_input_window_metrics(
        pcm_samples=960_000,
        first_audio_ns=700_000_000,
        done_ns=41_000_000_000,
        playout_started_ns=700_000_000,
        playout_completed_ns=40_700_000_000,
        final_observation_ns=41_100_000_000,
        input_frame_sends=input_sent_payloads,
        input_frame_acks=input_ack_payloads,
        fifo_peak_samples=fifo_peak,
    )

    server_log = output_dir / "server.log"
    server_log.write_text(
        f"{start.isoformat()} [realtime {session_id}] session ready\n"
        f"{end.isoformat()} [realtime {session_id}] session drained\n",
        encoding="utf-8",
    )
    server_window = output_dir / "server-log-window.json"
    server_window.write_text(json.dumps({
        "schema": "duplex_server_log_window_v1",
        "session_id": session_id,
        "since_utc": start.isoformat(),
        "until_utc": end.isoformat(),
    }), encoding="utf-8")
    runtime_path = output_dir / "runtime-telemetry.json"
    runtime_path.write_text(json.dumps({
        "session_id": session_id,
        "run_window_utc": {"started_at": start.isoformat(), "finished_at": end.isoformat()},
        "cuda_allocator": {
            "status": "ok",
            "device": allocator_device,
            "allocated_bytes": 1_000_000,
            "reserved_bytes": 1_200_000,
            "peak_allocated_bytes": 1_350_000,
            "peak_reserved_bytes": 1_600_000,
            "sampled_at_monotonic_ns": 20_000_000_000,
        },
        "kv": {"cursor": 4},
        "vram": {
            "post_warmup_baseline": {"allocated_mib": 100, "reserved_mib": 120},
            "active": {"allocated_mib": 130, "reserved_mib": 150},
            "peak": {"allocated_mib": 135, "reserved_mib": 160},
            "turns": [
                {
                    "response_id": response_id,
                    "scenario_tag": tag,
                    "post_drain": {"allocated_mib": 101 + index, "reserved_mib": 121 + index},
                }
                for index, (response_id, tag, *_rest) in enumerate(tags)
            ],
        },
    }), encoding="utf-8")
    kv_events = _extract_kv_evidence_from_rows(rows)
    report = {
        "schema_version": 1,
        "overall_status": "INCOMPLETE",
        "run_window_utc": {"started_at": start.isoformat(), "finished_at": end.isoformat()},
        "event_log": {"path": str(event_path), "rows": len(rows)},
        "fixture_manifest": fixture_manifest,
        "connection": {
            "session_id": session_id,
            "tls": tls,
            "response_profile_updated": "long",
        },
        "acceptance_parameters": {
            "mode": "acceptance",
            "response_profile": "long",
            "playback_buffer_seconds": 4.0,
            "playback_buffer_max_samples": 96_000,
            "pacing_tolerance_ms": 20,
            "max_input_gap_ms": 160,
            "cancellation_budget_ms": 250,
            "interrupt_after_playout_s": 3,
            "vram_peak_over_baseline_budget_mib": 500,
            "vram_post_drain_growth_budget_mib": 100,
            "vram_reserved_peak_growth_budget_mib": 500,
            "vram_reserved_post_drain_growth_budget_mib": 100,
        },
        "input": {
            "frames_sent": len(input_sent_payloads),
            "frames_acked": len(input_ack_payloads),
            "frames_inflight_at_end": 0,
            "inflight_high_water": 1,
            "sender_timing": {
                "frames_waiting_for_backpressure": 0,
                "total_backpressure_wait_ms": 0,
                "max_backpressure_wait_ms": 0,
                "max_sender_scheduler_delay_ms": 0,
                "max_gap_ms": 160,
                "backpressure_and_scheduler_delay_are_separate": True,
            },
        },
        "playout": {"fifo_peak_pending_samples": fifo_peak},
        "scenario": {
            "errors": [],
            "responses": report_responses,
            "long_response_input_window": long_window,
            "required_response_order": [
                "long_response_completed", "memory_followup_resume",
                "long_response_interrupted", "closure",
            ],
        },
        "error_categories": {
            "runtime_error": [],
            "transport_error": [],
            "producer_limit": [],
            "acoustic_limit": [],
            "harness_interruption": [],
        },
        "runtime_evidence": {
            "kv_events": kv_events,
            "raw_source_audio_telemetry_complete": True,
            "sources": {
                "server_log": str(server_log),
                "server_log_window": str(server_window),
                "runtime_telemetry": str(runtime_path),
            },
        },
        "errors": [],
    }
    (output_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return report


def test_offline_finalization_revalidates_evidence_before_pass(tmp_path):
    output_dir = tmp_path / "run"
    preliminary = _write_synthetic_finalize_bundle(output_dir)
    args = argparse.Namespace(
        server_log=None,
        server_log_window=None,
        runtime_telemetry=None,
        vram_peak_over_baseline_budget_mib=None,
        vram_post_drain_growth_budget_mib=None,
        vram_reserved_peak_growth_budget_mib=None,
        vram_reserved_post_drain_growth_budget_mib=None,
    )

    finalized = _offline_finalize_report(output_dir, args)

    assert preliminary["overall_status"] == "INCOMPLETE"
    assert finalized["overall_status"] == "PASS"
    assert finalized["runtime_evidence"]["server_log"]["contains_cuda_memory_telemetry"] is False
    assert json.loads((output_dir / "report.json").read_text(encoding="utf-8"))["overall_status"] == "INCOMPLETE"
    assert (output_dir / "report.finalized.json").is_file()

    bad_dir = tmp_path / "bad-run"
    _write_synthetic_finalize_bundle(bad_dir, allocator_device="cpu")
    failed_closed = _offline_finalize_report(bad_dir, args)
    assert failed_closed["overall_status"] == "INCOMPLETE"


def test_long_response_offline_reconstruction_accepts_legacy_window_without_cutoff(tmp_path):
    output_dir = tmp_path / "legacy-run"
    report = _write_synthetic_finalize_bundle(output_dir)
    report["scenario"]["long_response_input_window"].pop("input_ack_cutoff_ns")
    (output_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    args = argparse.Namespace(
        server_log=None,
        server_log_window=None,
        runtime_telemetry=None,
        vram_peak_over_baseline_budget_mib=None,
        vram_post_drain_growth_budget_mib=None,
        vram_reserved_peak_growth_budget_mib=None,
        vram_reserved_post_drain_growth_budget_mib=None,
    )

    finalized = _offline_finalize_report(output_dir, args)

    assert finalized["overall_status"] == "PASS"


def test_long_response_offline_reconstruction_rejects_cutoff_beyond_grace_window(tmp_path):
    output_dir = tmp_path / "out-of-range-cutoff-run"
    report = _write_synthetic_finalize_bundle(output_dir)
    long_window = report["scenario"]["long_response_input_window"]
    long_window["input_ack_cutoff_ns"] = long_window["playout_completed_ns"] + 2_000_000_001
    (output_dir / "report.json").write_text(json.dumps(report), encoding="utf-8")
    args = argparse.Namespace(
        server_log=None,
        server_log_window=None,
        runtime_telemetry=None,
        vram_peak_over_baseline_budget_mib=None,
        vram_post_drain_growth_budget_mib=None,
        vram_reserved_peak_growth_budget_mib=None,
        vram_reserved_post_drain_growth_budget_mib=None,
    )

    finalized = _offline_finalize_report(output_dir, args)

    assert any(
        "input ACK cutoff exceeds the two-second grace window" in error
        for error in finalized["scenario"]["errors"]
    )


def test_cancelled_response_is_terminal_when_completion_was_requested():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.fatal_error = None
    record = ResponseRecord(
        response_id="early",
        created_event={"response_id": "early"},
        created_ns=time.monotonic_ns(),
    )
    record.cancelled_event = {"reason": "yield"}
    record.terminal_signal.set()

    asyncio.run(asyncio.wait_for(
        simulator._await_response_terminal(record, want_done=True, timeout_s=10.0),
        timeout=0.1,
    ))


def test_complete_fixture_recovers_after_yield_while_prompt_speech_is_streaming():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = argparse.Namespace(response_timeout_s=1.0, playback_timeout_s=1.0)
    simulator.current_response_id = None
    logged = []
    simulator.log = SimpleNamespace(
        write=lambda direction, kind, payload, **kwargs: logged.append((kind, dict(payload)))
    )
    delivery = SimpleNamespace(
        delivery_id=8,
        scheduled_ns=time.monotonic_ns() - 1_000_000,
        first_sent_ns=time.monotonic_ns() - 750_000,
        first_vocal_sent_ns=time.monotonic_ns() - 500_000,
        last_sent_ns=time.monotonic_ns() - 250_000,
        last_vocal_sent_ns=time.monotonic_ns() - 250_000,
        frames_sent=3,
        fixture=SimpleNamespace(frame_count=10, vocal_end_frame=8),
    )
    early = ResponseRecord(
        response_id="attempt-a",
        created_event={"response_id": "attempt-a"},
        created_ns=time.monotonic_ns(),
        action="SPEAK",
        playout_started_ns=time.monotonic_ns() - 200_000,
        cancelled_event={"reason": "yield"},
        cancelled_ns=time.monotonic_ns() - 100_000,
        flush_ns=time.monotonic_ns() - 50_000,
    )
    completed = ResponseRecord(
        response_id="attempt-b",
        created_event={"response_id": "attempt-b"},
        created_ns=time.monotonic_ns(),
        action="SPEAK",
        done_event={"status": "completed"},
        audio_samples=1,
        segment_ids={"segment-b"},
        marker_ids={"segment-b"},
        audio_telemetry_complete=True,
        audio_quality_events=1,
    )
    queued = iter((early, completed))
    selection_budgets = []
    finalizers = []

    simulator._assign_fixture = lambda fixture_key, scenario_tag: delivery

    async def next_response(timeout_s, *, after_ns=None):
        selection_budgets.append(timeout_s)
        assert after_ns == delivery.scheduled_ns
        record = next(queued)
        simulator._set_active_response(record)
        return record

    async def await_terminal(record, *, want_done, timeout_s):
        assert want_done is True
        assert timeout_s > 0

    async def drain(record, timeout_s):
        assert record.response_id == "attempt-b"

    async def await_finalizer(record):
        finalizers.append(record.response_id)

    simulator._next_speaking_response = next_response
    simulator._await_response_terminal = await_terminal
    simulator._wait_playback_drain = drain
    simulator._await_response_finalizer = await_finalizer

    result = asyncio.run(simulator._complete_fixture(
        "long_request", "long_response_completed"
    ))

    assert result is completed
    assert early.scenario_tag == "long_response_early_interruption_attempt"
    assert early.delivery_attempt_status == "early_interruption"
    assert early.prompt_speech_active_at_cancel is True
    assert early.audible_overlap_with_prompt_speech is True
    assert early.recovered_by_response_id == "attempt-b"
    assert completed.scenario_tag == "long_response_completed"
    assert completed.delivery_attempt_status == "completed"
    assert selection_budgets[1] <= selection_budgets[0]
    assert finalizers == ["attempt-b", "attempt-a"]
    assert "delivery_early_interruption_attempt" in [kind for kind, _ in logged]
    assert "delivery_attempt_recovered" in [kind for kind, _ in logged]


def test_late_cleanup_of_early_attempt_does_not_clear_new_active_response():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.current_response_id = "attempt-b"
    early = ResponseRecord(
        response_id="attempt-a",
        created_event={},
        created_ns=time.monotonic_ns(),
    )

    simulator._clear_active_response_if_current(early)

    assert simulator.current_response_id == "attempt-b"


def test_next_speaking_response_skips_stale_and_non_speaking_responses():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.fatal_error = None
    simulator.response_created_queue = asyncio.Queue()
    logged = []
    simulator.log = SimpleNamespace(
        write=lambda direction, kind, payload, **kwargs: logged.append((kind, dict(payload)))
    )
    stale = ResponseRecord("stale", {}, created_ns=1, action="SPEAK")
    backchannel = ResponseRecord(
        "backchannel", {}, created_ns=time.monotonic_ns(), action="BACKCHANNEL"
    )
    speak = ResponseRecord("current", {}, created_ns=time.monotonic_ns(), action="SPEAK")
    simulator.response_created_queue.put_nowait(stale)
    simulator.response_created_queue.put_nowait(backchannel)
    simulator.response_created_queue.put_nowait(speak)

    selected = asyncio.run(simulator._next_speaking_response(1.0, after_ns=2))

    assert selected is speak
    assert [kind for kind, _ in logged] == [
        "stale_response_rejected",
        "non_speaking_response_skipped",
    ]


def test_complete_fixture_fails_immediately_for_non_yield_cancellation():
    simulator = object.__new__(RealDialogueSimulator)
    simulator.args = argparse.Namespace(response_timeout_s=1.0)
    delivery = SimpleNamespace(
        delivery_id=9,
        scheduled_ns=time.monotonic_ns() - 1_000_000,
        first_sent_ns=None,
        first_vocal_sent_ns=None,
        frames_sent=10,
        fixture=SimpleNamespace(frame_count=10, vocal_end_frame=8),
    )
    response = ResponseRecord(
        response_id="error-response",
        created_event={"response_id": "error-response"},
        created_ns=time.monotonic_ns(),
        action="SPEAK",
        cancelled_event={"reason": "error"},
    )
    next_calls = []

    simulator._assign_fixture = lambda fixture_key, scenario_tag: delivery

    async def next_response(timeout_s, *, after_ns=None):
        next_calls.append(timeout_s)
        return response

    async def await_terminal(record, *, want_done, timeout_s):
        return None

    simulator._next_speaking_response = next_response
    simulator._await_response_terminal = await_terminal

    with pytest.raises(EvidenceError, match="cancelled unexpectedly"):
        asyncio.run(simulator._complete_fixture(
            "long_request", "long_response_completed"
        ))

    assert len(next_calls) == 1
