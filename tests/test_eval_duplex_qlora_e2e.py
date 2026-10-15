from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import sys
import types
import wave
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import tools.eval_duplex_qlora_e2e as e2e
from dialogue_eval.scenarios import DialogueScenario, ScenarioStep
from tools.eval_duplex_qlora_e2e import (
    ACTIONS,
    action_coverage,
    action_lifecycle_coverage,
    audio_quality,
    calculate_cer,
    compare_asr_metrics,
    compute_event_metrics,
    live_policy_semantics,
    load_checkpoint_manifest,
    session_isolation_evidence,
    validate_loaded_identity,
)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "run_id": "run-123",
        "base_model_id": "org/gemma-4-e2b-it",
        "base_model_path": "models/gemma",
        "adapter_dir": "lora_adapters",
        "control_head_path": "duplex_control_head.safetensors",
        "control_head": {
            "input_dim": 1536,
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(ACTIONS),
        },
        "versions": {"torch": "2.12.1", "transformers": "5.5.0", "peft": "0.20.0"},
        "training": {"qlora_steps": 1, "control_steps": 1, "train_examples": 2, "validation_examples": 1},
        "sources": [{"id": "cpu-fixture", "train_conversations": 2, "validation_conversations": 1}],
    }


def _write_manifest(tmp_path, manifest: dict) -> None:
    import json

    (tmp_path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _loaded(manifest: dict) -> dict:
    return {
        "type": "model_loaded",
        "run_id": manifest["run_id"],
        "base_path": manifest["base_model_path"],
        "adapter_path": "lora_adapters",
        "control_head": "duplex_control_head.safetensors",
        "actions": list(ACTIONS),
        "feature_contract": {
            "input_dim": 1536,
            "pooling": "last_audio_query",
            "tick_ms": 80,
        },
        "policy_id": "duplex_control6",
    }


def test_manifest_requires_control_contract_and_resolves_checkpoint_paths(tmp_path) -> None:
    _write_manifest(tmp_path, _manifest())
    loaded = load_checkpoint_manifest(tmp_path)
    assert loaded["run_id"] == "run-123"
    assert loaded["adapter_path"] == (tmp_path / "lora_adapters").resolve()
    assert loaded["control_head_path"] == (tmp_path / "duplex_control_head.safetensors").resolve()

    invalid = _manifest()
    invalid["control_head"]["actions"] = list(reversed(ACTIONS))
    _write_manifest(tmp_path, invalid)
    with pytest.raises(ValueError, match="actions"):
        load_checkpoint_manifest(tmp_path)


def test_manifest_accepts_minimal_runtime_identity_without_historical_certifications(tmp_path) -> None:
    minimal = _manifest()
    for key in ("schema_version", "versions", "training", "sources"):
        minimal.pop(key)
    minimal["data_sources"] = {"synthetic_it": {"train": 2, "validation": 1}}
    _write_manifest(tmp_path, minimal)
    loaded = load_checkpoint_manifest(tmp_path)
    assert loaded["run_id"] == "run-123"
    assert loaded["base_model_path"] == "models/gemma"


def test_response_bundle_manifest_and_live_identity_need_no_training_run_attestation(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    (tmp_path / "control_head.pt").write_bytes(b"head")
    (tmp_path / "backchannel_lexicon.json").write_text("{}", encoding="utf-8")
    bundle = {
        "bundle_format": "gemma4_duplex_response",
        "schema_version": 1,
        "bundle_id": "lm-head-v5",
        "model": {"adapter_format": "peft", "base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter"},
        "policy": {
            "id": "duplex_control6", "head_path": "control_head.pt", "actions": list(ACTIONS),
            "feature_contract": {
                "queries_per_tick": 4, "input_dim": 1536, "pooling": "last_audio_query",
                "tick_ms": 80, "source": "native_gemma_listener",
            },
            "abi": {
                "hidden_dim": 1536, "query_index": 3, "classes": list(ACTIONS),
                "sample_rate_hz": 16000, "channels": 1, "tick_ms": 80,
            },
            "feature_source": {
                "base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter",
            },
        },
        "backchannel": {"language": "it", "lexicon_path": "backchannel_lexicon.json"},
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(bundle), encoding="utf-8")
    manifest = load_checkpoint_manifest(tmp_path)
    loaded = {
        "type": "model_loaded", "bundle_id": "lm-head-v5", "policy_id": "duplex_control6",
        "adapter_path": str(adapter.resolve()), "control_head": str((tmp_path / "control_head.pt").resolve()),
        "actions": list(ACTIONS), "feature_contract": {
            "input_dim": 1536, "pooling": "last_audio_query", "tick_ms": 80,
        },
        "base_model_id": "google/gemma-4-E2B-it",
        "feature_source": {"base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter"},
        "runtime_abi": {
            "hidden_dim": 1536, "query_index": 3, "classes": list(ACTIONS),
            "sample_rate_hz": 16000, "channels": 1, "tick_ms": 80,
        },
    }
    decision = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 1, "listener_frame_id": 1, "session": "s1", "source": "native_gemma_listener:last_audio_query",
    }
    assert manifest["bundle_id"] == "lm-head-v5"
    assert "run_id" not in manifest
    assert validate_loaded_identity(manifest, [{"event": loaded}, {"event": decision}], "duplex_control6")["status"] == "matched"
    loaded["bundle_id"] = "different-bundle"
    mismatch = validate_loaded_identity(manifest, [{"event": loaded}, {"event": decision}], "duplex_control6")
    assert mismatch["status"] == "mismatch"
    assert mismatch["mismatches"] == ["bundle_id"]


def test_response_bundle_accepts_canonical_causal_acoustic_phase_contract(tmp_path) -> None:
    from duplex_kernel.acoustic_phase_features import ACOUSTIC_PHASE_FEATURE_CONTRACT

    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}', encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    (tmp_path / "control_head.pt").write_bytes(b"head")
    (tmp_path / "backchannel_lexicon.json").write_text("{}", encoding="utf-8")
    feature_contract = {
        "queries_per_tick": 4, "input_dim": 1540, "pooling": "last_audio_query",
        "tick_ms": 80, "source": "native_gemma_listener",
        "acoustic_phase": ACOUSTIC_PHASE_FEATURE_CONTRACT,
    }
    bundle = {
        "bundle_format": "gemma4_duplex_response", "schema_version": 1,
        "bundle_id": "lm-head-v5-phase",
        "model": {"adapter_format": "peft", "base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter"},
        "policy": {
            "id": "duplex_control6", "head_path": "control_head.pt", "actions": list(ACTIONS),
            "feature_contract": feature_contract,
            "abi": {
                "hidden_dim": 1536, "query_index": 3, "classes": list(ACTIONS),
                "sample_rate_hz": 16000, "channels": 1, "tick_ms": 80,
            },
            "feature_source": {"base_model_id": "google/gemma-4-E2B-it", "adapter_dir": "adapter"},
        },
        "backchannel": {"language": "it", "lexicon_path": "backchannel_lexicon.json"},
    }
    (tmp_path / "bundle_manifest.json").write_text(json.dumps(bundle), encoding="utf-8")

    manifest = load_checkpoint_manifest(tmp_path)

    assert manifest["control_head"]["input_dim"] == 1540
    assert manifest["control_head"]["feature_contract"] == feature_contract


def test_loaded_identity_fails_closed_on_mismatch_or_missing_policy_decision(tmp_path) -> None:
    _write_manifest(tmp_path, _manifest())
    manifest = load_checkpoint_manifest(tmp_path)
    valid = [{"event": _loaded(manifest), "session": "s1"}]
    decisions = [{
        "event": {"type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
                  "frame_id": 1, "listener_frame_id": 1, "session": "s1",
                  "source": "duplex_control6"},
        "session": "s1",
    }]
    assert validate_loaded_identity(manifest, valid + decisions, "duplex_control6")["status"] == "matched"
    assert validate_loaded_identity(manifest, valid, "duplex_control6")["status"] == "unavailable"
    reordered = copy.deepcopy(valid + decisions)
    reordered[0]["event"]["actions"] = list(reversed(ACTIONS))
    reordered_identity = validate_loaded_identity(manifest, reordered, "duplex_control6")
    assert reordered_identity["status"] == "mismatch"
    assert "actions" in reordered_identity["mismatches"]
    wrong = copy.deepcopy(valid + decisions)
    wrong[0]["event"]["run_id"] = "other-run"
    assert validate_loaded_identity(manifest, wrong, "duplex_control6")["status"] == "mismatch"


def test_live_policy_semantics_correlates_continue_ack_backchannel_overlap_and_replan_resume() -> None:
    def row(kind: str, timestamp: int, *, event: dict, direction: str = "server", frame_id: int | None = None) -> dict:
        result = {
            "direction": direction,
            "session_id": "s1",
            "received_at_monotonic_ns": timestamp,
            "sent_at_monotonic_ns": timestamp,
            "event": {"type": kind, **event},
        }
        if frame_id is not None:
            result["event"]["frame_id"] = frame_id
        return result

    def decision(action: str, frame: int) -> dict:
        return {
            "action": action, "confidence": 0.9, "frame_id": frame,
            "listener_frame_id": frame, "session": "s1", "source": "native_gemma_listener:last_audio_query",
            "accepted": True,
        }

    rows = [
        row("duplex_control6.decision", 10, event=decision("CONTINUE", 3)),
        row("input_audio_buffer.consumed", 11, event={"listener_output_frame_id": 3}, frame_id=3),
        row("duplex_control6.decision", 20, event=decision("BACKCHANNEL", 4)),
        row("response.created", 30, event={"response_id": "bc", "effective_action": "BACKCHANNEL"}),
        row("input_audio_buffer.append", 40, event={"audio_samples": 1280}, direction="client"),
        row("response.audio.delta", 45, event={"response_id": "bc"}),
        row("response.done", 50, event={"response_id": "bc"}),
        row("duplex_control6.decision", 60, event=decision("REPLAN", 8)),
        row("response.cancelled", 70, event={"response_id": "old", "reason": "replan"}),
        row("response.created", 80, event={"response_id": "new", "action": "SPEAK", "intent_endpoint_frame_id": 9}),
    ]

    metrics = live_policy_semantics(rows)

    assert metrics["continue"] == {
        "accepted_decisions": 1,
        "same_frame_listener_consumed_ack": 1,
        "ack_without_same_frame_replan": 1,
    }
    assert metrics["backchannel"] == {
        "accepted_decisions": 1,
        "responses_created": 1,
        "responses_with_user_audio_continuing_during_playout": 1,
    }
    assert metrics["replan"] == {"accepted_decisions": 1, "cancel_then_endpoint_scoped_resume": 1}


def test_loaded_identity_accepts_the_live_server_event_names(tmp_path) -> None:
    _write_manifest(tmp_path, _manifest())
    manifest = load_checkpoint_manifest(tmp_path)
    loaded = _loaded(manifest)
    loaded["policy_id"] = "duplex_control6"
    decision = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 1, "listener_frame_id": 1, "session": "s1", "source": "duplex_control6",
    }
    result = validate_loaded_identity(
        manifest, [{"event": loaded}, {"event": decision, "session": "s1"}], "duplex_control6"
    )
    assert result["status"] == "matched"
    assert result["decision_events"] == 1


def test_live_native_listener_decisions_count_raw_and_accepted_separately(tmp_path) -> None:
    _write_manifest(tmp_path, _manifest())
    manifest = load_checkpoint_manifest(tmp_path)
    loaded = _loaded(manifest)
    loaded["policy_id"] = "duplex_control6"
    decision_base = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 1, "listener_frame_id": 1, "session": "s1",
        "source": "native_gemma_listener:last_audio_query",
    }
    rows = [
        {"event": loaded},
        {"event": {**decision_base, "accepted": True}},
        {"event": {**decision_base, "action": "YIELD", "frame_id": 2,
                    "listener_frame_id": 2, "accepted": False,
                    "rejection_reason": "assistant_audio_not_started"}},
    ]

    identity = validate_loaded_identity(manifest, rows, "duplex_control6")
    raw = action_coverage(rows, ACTIONS, "duplex_control6")
    accepted = action_coverage(rows, ACTIONS, "duplex_control6", accepted_only=True)

    assert identity["status"] == "matched"
    assert identity["decision_events"] == 2
    assert raw["decision_count"] == 2
    assert raw["count_by_action"]["YIELD"] == 1
    assert accepted["decision_count"] == 1
    assert accepted["observed"] == ["LISTEN"]


def test_loaded_identity_accepts_local_huggingface_cache_for_manifest_model_id(tmp_path) -> None:
    data = _manifest()
    data["base_model_id"] = "org/gemma-base"
    data["base_model_path"] = "org/gemma-base"
    _write_manifest(tmp_path, data)
    manifest = load_checkpoint_manifest(tmp_path)
    loaded = _loaded(data)
    loaded["base_path"] = "/workspace/.hf_cache/models--org--gemma-base/full"
    decision = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 1, "listener_frame_id": 1, "session": "s1", "source": "duplex_control6",
    }
    result = validate_loaded_identity(manifest, [{"event": loaded}, {"event": decision}], "duplex_control6")
    assert result["status"] == "matched"


def test_qualitative_prompts_are_opt_in_and_do_not_expand_canonical_coverage(tmp_path) -> None:
    canonical_default, qualitative_default = e2e._evaluation_scenarios(False)
    canonical_requested, qualitative_requested = e2e._evaluation_scenarios(True)

    assert [scenario.scenario_id for scenario in canonical_requested] == [
        scenario.scenario_id for scenario in canonical_default
    ]
    assert qualitative_default == []
    assert [scenario.scenario_id for scenario in qualitative_requested] == ["qualitative_schedule_replan"]
    scenario = qualitative_requested[0]
    assert len([step for step in scenario.steps if step.speaker == "user"]) == 3
    assert all(not step.deterministic_checks for step in scenario.steps)
    assert any("martedì" in step.expected_text and "nuoto" in step.expected_text for step in scenario.steps)
    assert any("chiarire" in step.expected_text for step in scenario.steps)
    assert any("Correggo" in step.expected_text and "venerdì" in step.expected_text for step in scenario.steps)

    audio_dir = tmp_path / "qualitative_audio"
    pcm = 0.08 * np.sin(2 * np.pi * 440 * np.arange(1280) / 16_000).astype(np.float32)
    for step in scenario.steps:
        e2e._write_wav(audio_dir / f"{scenario.scenario_id}__{step.step_id}.wav", pcm, 16_000)
    available, skipped = e2e._scenario_assets_for(audio_dir, qualitative_requested)
    assert not skipped
    assert len(available) == 1
    assert set(available[0][1]) == {"turn_1_user", "turn_2_user", "turn_3_user"}


def test_targeted_scenario_selection_keeps_canonical_order_and_rejects_unknown_ids(tmp_path) -> None:
    canonical, _ = e2e._evaluation_scenarios(False)
    parsed = e2e._parser().parse_args([
        "--scenario-id", "05_barge_in_interruption",
        "--scenario-id", "01_name_number_memory",
    ])
    assert parsed.scenario_ids == ["05_barge_in_interruption", "01_name_number_memory"]

    selected = e2e._select_canonical_scenarios(
        canonical,
        ["05_barge_in_interruption", "01_name_number_memory", "05_barge_in_interruption"],
    )

    assert [scenario.scenario_id for scenario in selected] == [
        "01_name_number_memory",
        "05_barge_in_interruption",
    ]
    with pytest.raises(ValueError, match="unknown scenario"):
        e2e._select_canonical_scenarios(canonical, ["not_a_canonical_scenario"])

    targeted_report = {
        "identity": {"status": "matched"},
        "behavior": {
            "status": "observed",
            "action_coverage": {"missing": []},
            "barge_in": {"status": "observed"},
        },
        "scenario_coverage": {"scope": "targeted_subset", "status": "partial"},
    }
    report_dir = tmp_path / "targeted-report"
    report_dir.mkdir()
    e2e._write_reports(targeted_report, report_dir)
    assert targeted_report["evaluation_status"] == "incomplete"


def test_requested_qualitative_probe_captures_each_turn_without_changing_six_scenario_gate(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    _write_manifest(tmp_path, manifest)
    audio_dir = tmp_path / "audio"
    pcm = 0.08 * np.sin(2 * np.pi * 440 * np.arange(1280) / 16_000).astype(np.float32)
    canonical = DialogueScenario(
        scenario_id="canonical_fixture", title="Canonical", description="One canonical fixture.",
        system_prompt_a="Utente.", system_prompt_b="Assistente.",
        steps=[ScenarioStep("turn_1_user", "user", "Ciao."), ScenarioStep("turn_1_assistant", "assistant", "Salve.")],
    )
    monkeypatch.setattr(e2e, "ALL_SCENARIOS", [canonical])
    for scenario in [canonical, *e2e.QUALITATIVE_SCENARIOS]:
        for step in scenario.steps:
            if step.speaker == "user":
                e2e._write_wav(audio_dir / f"{scenario.scenario_id}__{step.step_id}.wav", pcm, 16_000)

    class MultiTurnSocket:
        def __init__(self, index: int):
            self.index = index
            self.incoming = asyncio.Queue()
            self.sent = []
            self.turn = 0
            self.session_id = f"cpu-session-{index}"

        async def __aenter__(self):
            self.incoming.put_nowait({"type": "session.created", "session_id": self.session_id})
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if event.get("type") == "session.update":
                self.incoming.put_nowait({**_loaded(manifest), "session_id": self.session_id})
                self.incoming.put_nowait({
                    "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
                    "frame_id": 0, "listener_frame_id": 0, "session": self.session_id,
                    "source": "duplex_control6", "accepted": True,
                })
                self.incoming.put_nowait({"type": "session.updated", "session_id": self.session_id})
            elif event.get("type") == "input_audio_buffer.append":
                pcm_payload = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                if not np.any(pcm_payload):
                    return
                response_id = f"{self.session_id}-response-{self.turn}"
                self.incoming.put_nowait({"type": "input_audio_buffer.consumed", "frame_id": self.turn})
                self.incoming.put_nowait({"type": "response.created", "response_id": response_id,
                                          "action": "SPEAK", "effective_action": "SPEAK"})
                self.incoming.put_nowait({
                    "type": "response.output_text.delta", "response_id": response_id,
                    "delta": f"Risposta {self.turn + 1}.",
                })
                self.incoming.put_nowait({
                    "type": "response.audio.delta", "response_id": response_id,
                    "delta": _wav_payload(0.06 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24_000), 24_000),
                    "sample_rate": 24_000,
                })
                self.incoming.put_nowait({"type": "response.done", "response_id": response_id})
                self.turn += 1

        async def recv(self):
            return json.dumps(await self.incoming.get())

    socket_index = 0
    sockets = []

    def connect(*_args, **_kwargs):
        nonlocal socket_index
        socket = MultiTurnSocket(socket_index)
        socket_index += 1
        sockets.append(socket)
        return socket

    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=connect))
    args = Namespace(
        checkpoint_root=tmp_path, server="ws://cpu.invalid/realtime", output_dir=tmp_path / "evaluation",
        turn_policy="duplex_control6", expected_run_id="run-123", scenario_audio_dir=audio_dir,
        asr_model_path=None, asr_device="cpu", asr_backend="auto", frame_pacing=False,
        include_qualitative_prompts=True,
    )

    report = asyncio.run(e2e._build_report(args))

    assert report["scenario_coverage"]["expected"] == ["canonical_fixture"]
    assert report["scenario_coverage"]["run"] == ["canonical_fixture"]
    assert report["scenario_coverage"]["status"] == "complete"
    assert report["qualitative_probes"]["status"] == "captured"
    assert report["qualitative_probes"]["assessment"] == "manual_review_only"
    probe = report["qualitative_probes"]["probes"][0]
    assert probe["capture_status"] == "captured"
    assert [turn["response_text"] for turn in probe["turns"]] == ["Risposta 1.", "Risposta 2.", "Risposta 3."]
    assert all(Path(turn["input_wav"]).is_file() and Path(turn["response_wav"]).is_file() for turn in probe["turns"])
    assert Path(report["artifacts"]["qualitative_trace_jsonl"]).is_file()
    assert all("input_audio_buffer.commit" not in [event["type"] for event in socket.sent] for socket in sockets)
    for socket in sockets:
        appended = [event for event in socket.sent if event.get("type") == "input_audio_buffer.append"]
        pcm_frames = [np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2") for event in appended]
        assert sum(not np.any(frame) for frame in pcm_frames) >= 8


def test_offline_asr_enrichment_transcribes_captured_canonical_and_qualitative_wavs(tmp_path, monkeypatch) -> None:
    canonical_in = tmp_path / "canonical_input.wav"
    canonical_out = tmp_path / "canonical_output.wav"
    qualitative_in = tmp_path / "qualitative_input.wav"
    qualitative_out = tmp_path / "qualitative_output.wav"
    for path, rate in ((canonical_in, 16_000), (canonical_out, 24_000),
                       (qualitative_in, 16_000), (qualitative_out, 24_000)):
        e2e._write_wav(path, np.full(160, 0.1, dtype=np.float32), rate)
    report_path = tmp_path / "report.json"
    report = {
        "behavior": {
            "turns": [{
                "session_id": "canonical-session", "scenario_id": "canonical-fixture",
                "reference_text": "parola utente", "response_text": "risposta assistente",
                "asr_metrics": {"status": "unavailable"},
            }],
            "asr": {"status": "unavailable", "reason": "live capture excluded ASR"},
        },
        "qualitative_probes": {"probes": [{
            "scenario_id": "qualitative_schedule_replan", "turns": [{
                "input_reference_text": "preferenza parlata", "response_text": "risposta qualitativa",
                "input_wav": str(qualitative_in), "response_wav": str(qualitative_out),
                "asr_metrics": {"status": "unavailable"},
            }],
        }]},
        "artifacts": {"session_audio": {"canonical-session": {
            "input_wav": str(canonical_in), "output_wav": str(canonical_out),
        }}},
    }
    transcript_calls = []

    def fake_loader(model_path, device, backend):
        assert model_path == str(tmp_path / "local-whisper")
        assert device == "cuda"
        assert backend == "transformers"

        def transcribe(pcm, sample_rate):
            transcript_calls.append((len(pcm), sample_rate))
            return "parola utente" if sample_rate == 16_000 else "risposta assistente"

        return transcribe, None, "transformers"

    monkeypatch.setattr(e2e, "_load_local_asr", fake_loader)
    enriched = e2e.enrich_report_with_offline_asr(
        report, report_path, model_path=str(tmp_path / "local-whisper"),
        device="cuda", backend="transformers",
    )

    canonical = enriched["behavior"]["turns"][0]
    assert canonical["asr_input_text"] == "parola utente"
    assert canonical["asr_output_text"] == "risposta assistente"
    assert canonical["asr_metrics"]["status"] == "available"
    qualitative = enriched["qualitative_probes"]["probes"][0]["turns"][0]
    assert qualitative["input_asr_text"] == "parola utente"
    assert qualitative["response_asr_text"] == "risposta assistente"
    assert qualitative["asr_metrics"]["status"] == "available"
    assert enriched["behavior"]["asr"]["execution"] == "offline_after_server_stop"
    assert len(transcript_calls) == 4


def test_live_capture_does_not_initialize_asr_even_when_a_path_is_supplied(tmp_path, monkeypatch) -> None:
    audio_dir = tmp_path / "scenario_audio"
    audio_dir.mkdir()
    scenario = DialogueScenario(
        scenario_id="cpu_capture_without_asr", title="CPU capture", description="Capture fixture.",
        system_prompt_a="test", system_prompt_b="test",
        steps=[ScenarioStep(step_id="turn_1_user", speaker="user", expected_text="ciao")], max_turns=1,
    )
    e2e._write_wav(audio_dir / "cpu_capture_without_asr__turn_1_user.wav",
                   np.full(e2e.FRAME_SAMPLES, 0.1, dtype=np.float32), 16_000)
    monkeypatch.setattr(e2e, "ALL_SCENARIOS", [scenario])
    monkeypatch.setattr(e2e, "_evaluation_scenarios", lambda include: ([scenario], []))

    async def fake_run_session(**kwargs):
        return {"scenario_id": scenario.scenario_id, "session_id": "capture-session", "label": "capture",
                "inputs": [np.full(e2e.FRAME_SAMPLES, 0.1, dtype=np.float32)], "input_references": ["ciao"],
                "responses": [], "text_by_response": {}, "output_parts": [], "output_parts_by_response": {},
                "output_packet_boundaries": [], "output_rate": 24_000, "errors": []}

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("live server capture must not initialize ASR")

    monkeypatch.setattr(e2e, "_run_session", fake_run_session)
    monkeypatch.setattr(e2e, "_load_local_asr", forbidden_loader)
    args = Namespace(
        checkpoint_root=tmp_path, server="ws://cpu.invalid/realtime", output_dir=tmp_path / "capture",
        turn_policy="duplex_control6", expected_run_id="run-123", scenario_audio_dir=audio_dir,
        asr_model_path=str(tmp_path / "large-whisper"), asr_device="cuda", asr_backend="transformers",
        frame_pacing=False, include_qualitative_prompts=False,
    )
    _write_manifest(tmp_path, _manifest())
    monkeypatch.setattr(e2e, "load_checkpoint_manifest", lambda _: load_checkpoint_manifest(tmp_path))

    report = asyncio.run(e2e._build_report(args))

    assert report["system"]["status"] != "not_run_missing_audio", report["scenario_coverage"]
    assert report["behavior"]["asr"]["status"] == "unavailable"
    assert "offline" in report["behavior"]["asr"]["reason"]
    assert "error" not in report


def _wav_payload(pcm: np.ndarray, sample_rate: int) -> str:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes((np.asarray(pcm, dtype=np.float32).clip(-1, 1) * 32767).astype("<i2").tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_e2e_runner_records_live_policy_audio_and_never_commits(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    _write_manifest(tmp_path, manifest)
    scenario_audio = tmp_path / "scenario_audio"
    scenario_audio.mkdir()
    scenario = DialogueScenario(
        scenario_id="cpu_live_policy_probe", title="Probe CPU", description="Probe WebSocket live policy.",
        system_prompt_a="Utente italiano.", system_prompt_b="Assistente italiano.",
        steps=[
            ScenarioStep("turn_1_user", "user", "Ciao, mi senti?"),
            ScenarioStep("turn_1_assistant", "assistant", "Ciao, ti sento.",
                         deterministic_checks=[lambda text: "ciao" in text.casefold()]),
        ],
    )
    input_pcm = 0.08 * np.sin(2 * np.pi * 440 * np.arange(1280) / 16_000).astype(np.float32)
    e2e._write_wav(scenario_audio / "cpu_live_policy_probe__turn_1_user.wav", input_pcm, 16_000)
    monkeypatch.setattr(e2e, "ALL_SCENARIOS", [scenario])

    loaded_event = {
        "type": "model_loaded", "policy_id": "duplex_control6", "run_id": manifest["run_id"],
        "base_path": manifest["base_model_path"], "adapter_path": manifest["adapter_dir"],
        "control_head": manifest["control_head_path"], "actions": list(ACTIONS),
        "feature_contract": {key: manifest["control_head"][key] for key in ("input_dim", "pooling", "tick_ms", "actions")},
    }
    decision_event = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 0, "listener_frame_id": 0, "session": "cpu-session", "source": "duplex_control6",
        "accepted": True,
    }
    runtime_metrics = {
        "asr": {"audio_prefill_ms": 12.0, "shared_lm_lock_wait_ms": 3.0},
        "thinker": {"thinker_prefill_ms": 18.0, "thinker_decode_ms": 9.0},
        "talker": {"code2wav_decode_ms": 5.0},
        "scheduler": {"tick_jitter_p99_ms": 2.0, "deadline_misses": 1},
    }

    class FakeSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.sent = []
            self.response_emitted = False

        async def __aenter__(self):
            self.incoming.put_nowait({
                "type": "session.created", "session_id": "cpu-session",
                "runtime_metrics": {"listener": {"canonical_prime_calls": 1, "canonical_prime_ms": 8123.0}},
            })
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if event.get("type") == "session.update":
                assert event["session"]["auto_response"] is True
                assert event["session"]["turn_policy"] == "duplex_control6"
                self.incoming.put_nowait(loaded_event)
                self.incoming.put_nowait(decision_event)
                self.incoming.put_nowait({"type": "session.updated", "session_id": "cpu-session"})
            elif event.get("type") == "input_audio_buffer.append" and not self.response_emitted:
                self.response_emitted = True
                self.incoming.put_nowait({"type": "input_audio_buffer.consumed", "frame_id": 0})
                self.incoming.put_nowait({"type": "response.created", "response_id": "cpu-response",
                                          "action": "SPEAK", "effective_action": "SPEAK"})
                self.incoming.put_nowait({
                    "type": "response.output_text.delta", "response_id": "cpu-response", "delta": "Ciao, ti sento."
                })
                self.incoming.put_nowait({
                    "type": "response.audio.delta", "response_id": "cpu-response",
                    "delta": _wav_payload(0.06 * np.sin(2 * np.pi * 440 * np.arange(2400) / 24_000), 24_000),
                    "sample_rate": 24_000,
                })
                self.incoming.put_nowait({"type": "response.done", "response_id": "cpu-response",
                                          "runtime_metrics": runtime_metrics})

        async def recv(self):
            return json.dumps(await self.incoming.get())

    socket = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=lambda *_args, **_kwargs: socket))
    args = Namespace(
        checkpoint_root=tmp_path, server="ws://cpu.invalid/realtime", output_dir=tmp_path / "evaluation",
        turn_policy="duplex_control6", expected_run_id="run-123", scenario_audio_dir=scenario_audio,
        asr_model_path=None, asr_device="cpu", frame_pacing=False,
    )

    report = asyncio.run(e2e._build_report(args))

    assert report["identity"]["status"] == "matched"
    assert report["identity"]["observed_model_loaded"]["policy_id"] == "duplex_control6"
    assert report["behavior"]["action_coverage"]["observed"] == ["LISTEN"]
    assert report["behavior"]["status"] == "incomplete"
    assert report["scenario_coverage"]["status"] == "complete"
    captured_metrics_event = next(
        row["event"] for row in map(json.loads, (args.output_dir / "trace.jsonl").read_text().splitlines())
        if row["event"].get("type") == "response.done"
    )
    assert captured_metrics_event["runtime_metrics"] == runtime_metrics
    assert report["system"]["runtime_metrics"]["status"] == "available"
    assert report["system"]["runtime_metrics"]["response_events"][0]["metrics"] == runtime_metrics
    assert report["system"]["runtime_metrics"]["session_events"][0]["metrics"] == {
        "listener": {"canonical_prime_calls": 1, "canonical_prime_ms": 8123.0}
    }
    turn = report["behavior"]["turns"][0]
    assert turn["response_text"] == "Ciao, ti sento."
    assert turn["input_audio"]["status"] == "available"
    assert turn["output_audio"]["status"] == "available"
    assert turn["output_audio"]["metrics"]["samples"] == 2400
    saved = report["artifacts"]["session_audio"]["cpu-session"]
    assert Path(saved["input_wav"]).is_file()
    assert Path(saved["output_wav"]).is_file()
    playout = report["artifacts"]["session_playout"]["cpu-session"]
    assert playout["endpoint"] == "simulated_headless_pcm_playout"
    assert playout["physical_device_playback"] is False
    response_playout = playout["responses"]["cpu-response"]
    assert Path(response_playout["played_wav"]).is_file()
    assert response_playout["metrics"]["audio_samples_received"] == 2400
    assert response_playout["metrics"]["audio_samples_consumed"] == 2400
    assert response_playout["metrics"]["queued_samples_at_terminal"] == 2400
    assert response_playout["metrics"]["done_before_playback_complete"] is True
    playback_rows = [
        row for row in map(json.loads, (args.output_dir / "trace.jsonl").read_text().splitlines())
        if row["direction"] == "playback_simulator"
    ]
    assert playback_rows
    playback_timestamps = [row["at_monotonic_ns"] for row in playback_rows]
    assert playback_timestamps == sorted(playback_timestamps)
    sent_types = [event["type"] for event in socket.sent]
    assert "input_audio_buffer.commit" not in sent_types
    sent_audio = [event for event in socket.sent if event["type"] == "input_audio_buffer.append"]
    decoded_audio = [np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2") for event in sent_audio]
    assert any(np.any(frame) for frame in decoded_audio)
    assert any(not np.any(frame) for frame in decoded_audio)
    e2e._write_reports(report, args.output_dir)
    assert report["evaluation_status"] == "incomplete"
    assert (args.output_dir / "report.json").is_file()

    loaded_event = {**loaded_event, "run_id": "wrong-run"}
    rejected_socket = FakeSocket()
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=lambda *_args, **_kwargs: rejected_socket))
    rows = []
    result = asyncio.run(e2e._run_session(
        scenario=scenario, assets={"turn_1_user": scenario_audio / "cpu_live_policy_probe__turn_1_user.wav"},
        label="identity_rejection", ws_url=args.server, policy="duplex_control6",
        manifest=load_checkpoint_manifest(tmp_path), output_dir=args.output_dir, trace_rows=rows, pace=False,
    ))
    assert any("identity unavailable or mismatched" in error for error in result["errors"])
    assert all(event["type"] != "input_audio_buffer.append" for event in rejected_socket.sent)


def test_run_session_normal_branch_streams_endpoint_silence_and_waits_for_terminal(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    manifest["_checkpoint_root"] = str(tmp_path)
    manifest["adapter_path"] = manifest["adapter_dir"]
    manifest["control_head_path_resolved"] = manifest["control_head_path"]
    scenario = DialogueScenario(
        scenario_id="cpu_endpoint_silence", title="Endpoint silence", description="Normal live capture branch.",
        system_prompt_a="Utente.", system_prompt_b="Assistente.",
        steps=[ScenarioStep("turn_1_user", "user", "Ciao."),
               ScenarioStep("turn_1_assistant", "assistant", "Salve.")],
    )
    audio_dir = tmp_path / "audio"
    user_pcm = np.full(e2e.FRAME_SAMPLES, 0.08, dtype=np.float32)
    wav_path = audio_dir / f"{scenario.scenario_id}__turn_1_user.wav"
    e2e._write_wav(wav_path, user_pcm, 16_000)
    loaded_event = _loaded(manifest)
    decision_event = {
        "type": "duplex_control6.decision", "action": "LISTEN", "confidence": 0.9,
        "frame_id": 0, "listener_frame_id": 0, "session": "cpu-endpoint-session",
        "source": "duplex_control6", "accepted": True,
    }

    class EndpointSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.sent = []
            self.user_frames = 0
            self.silence_frames = 0
            self.emitted = False
            self.terminal_task = None
            self.speak_created = False
            self.five_post_speak_silence_frames = asyncio.Event()

        async def __aenter__(self):
            self.incoming.put_nowait({"type": "session.created", "session_id": "cpu-endpoint-session"})
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if event.get("type") == "session.update":
                self.incoming.put_nowait(loaded_event)
                self.incoming.put_nowait(decision_event)
                self.incoming.put_nowait({"type": "session.updated", "session_id": "cpu-endpoint-session"})
            elif event.get("type") == "input_audio_buffer.append":
                pcm16 = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
                if np.any(pcm16):
                    self.user_frames += 1
                    return
                if self.user_frames:
                    self.silence_frames += 1
                    if self.speak_created and self.silence_frames >= 15:
                        self.five_post_speak_silence_frames.set()
                    self.incoming.put_nowait({
                        "type": "input_audio_buffer.consumed", "frame_id": self.user_frames + self.silence_frames - 1,
                    })
                    if self.silence_frames == 8 and not self.emitted:
                        self.emitted = True
                        self.incoming.put_nowait({"type": "response.created", "response_id": "cpu-endpoint-backchannel",
                                                  "action": "BACKCHANNEL", "effective_action": "BACKCHANNEL"})
                        self.incoming.put_nowait({
                            "type": "response.done", "response_id": "cpu-endpoint-backchannel",
                        })
                    if self.silence_frames == 10:
                        self.speak_created = True
                        self.incoming.put_nowait({"type": "response.created", "response_id": "cpu-endpoint-response",
                                                  "action": "LISTEN", "effective_action": "SPEAK",
                                                  "intent_source_frame_id": 0, "intent_endpoint_frame_id": 8})
                        self.incoming.put_nowait({
                            "type": "response.output_text.delta", "response_id": "cpu-endpoint-response",
                            "delta": "Ciao, ti sento.",
                        })
                        self.incoming.put_nowait({
                            "type": "response.audio.delta", "response_id": "cpu-endpoint-response",
                            "delta": _wav_payload(np.full(2400, 0.04, dtype=np.float32), 24_000),
                            "sample_rate": 24_000,
                        })
                        self.terminal_task = asyncio.create_task(self._finish_speak_after_playout_window())

        async def _finish_speak_after_playout_window(self):
            await asyncio.sleep(0.4)
            await asyncio.wait_for(self.five_post_speak_silence_frames.wait(), timeout=0.5)
            self.incoming.put_nowait({
                "type": "response.done", "response_id": "cpu-endpoint-response",
            })

        async def recv(self):
            return json.dumps(await self.incoming.get())

    socket = EndpointSocket()
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(
        connect=lambda *_args, **_kwargs: socket,
    ))
    trace_rows = []
    result = asyncio.run(e2e._run_session(
        scenario=scenario,
        assets={"turn_1_user": wav_path},
        label=scenario.scenario_id,
        ws_url="ws://cpu.invalid/realtime",
        policy="duplex_control6",
        manifest=manifest,
        output_dir=tmp_path / "run",
        trace_rows=trace_rows,
        pace=True,
    ))

    assert result["errors"] == []
    assert socket.user_frames == 1
    assert socket.silence_frames >= 15
    assert result["input_audio_stream"][0].any()
    assert all(not frame.any() for frame in result["input_audio_stream"][1:16])
    assert "cpu-endpoint-response" in result["responses"]
    assert "cpu-endpoint-backchannel" in result["responses"]
    assert any(row.get("event", {}).get("type") == "response.done"
               and row.get("event", {}).get("response_id") == "cpu-endpoint-response"
               for row in trace_rows if row.get("direction") == "server")


def test_qualitative_wait_ignores_speak_ending_inside_user_pause(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    manifest["_checkpoint_root"] = str(tmp_path)
    manifest["adapter_path"] = manifest["adapter_dir"]
    manifest["control_head_path_resolved"] = manifest["control_head_path"]
    scenario = DialogueScenario(
        scenario_id="cpu_causal_endpoint_gate", title="Causal endpoint gate", description="Keep streaming through an intra-utterance pause.",
        system_prompt_a="Utente.", system_prompt_b="Assistente.",
        steps=[ScenarioStep("turn_1_user", "user", "Preferisco venerdì."),
               ScenarioStep("turn_1_assistant", "assistant", "Segno la preferenza.")],
    )
    audio_dir = tmp_path / "audio"
    first_phrase = np.full(4 * e2e.FRAME_SAMPLES, 0.08, dtype=np.float32)
    intra_utterance_pause = np.zeros(9 * e2e.FRAME_SAMPLES, dtype=np.float32)
    final_clause = np.full(3 * e2e.FRAME_SAMPLES, 0.06, dtype=np.float32)
    user_pcm = np.concatenate((first_phrase, intra_utterance_pause, final_clause))
    wav_path = audio_dir / f"{scenario.scenario_id}__turn_1_user.wav"
    e2e._write_wav(wav_path, user_pcm, 16_000)
    loaded_event = _loaded(manifest)

    class EndpointSocket:
        def __init__(self):
            self.incoming = asyncio.Queue()
            self.input_frame_id = 0
            self.user_frames = 0
            self.internal_pause_frames = 0
            self.endpoint_silence_frames = 0
            self.early_endpoint = 11
            self.last_voiced_endpoint = 15
            self.late_endpoint = 25

        async def __aenter__(self):
            self.incoming.put_nowait({"type": "session.created", "session_id": "cpu-causal-gate"})
            return self

        async def __aexit__(self, *_args):
            return False

        async def send(self, raw):
            event = json.loads(raw)
            if event.get("type") == "session.update":
                self.incoming.put_nowait(loaded_event)
                self.incoming.put_nowait({"type": "session.updated", "session_id": "cpu-causal-gate"})
                return
            if event.get("type") != "input_audio_buffer.append":
                return
            frame_id = self.input_frame_id
            self.input_frame_id += 1
            pcm16 = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
            if np.any(pcm16):
                self.user_frames += 1
                return
            if frame_id < 16:
                self.internal_pause_frames += 1
                if frame_id == self.early_endpoint:
                    self.incoming.put_nowait({
                        "type": "response.created", "response_id": "early-speak",
                        "action": "SPEAK", "effective_action": "SPEAK",
                        "intent_source_frame_id": 4, "intent_endpoint_frame_id": self.early_endpoint,
                    })
                    self.incoming.put_nowait({"type": "response.done", "response_id": "early-speak"})
                    await asyncio.sleep(0)
            else:
                self.endpoint_silence_frames += 1
                if frame_id == self.late_endpoint:
                    self.incoming.put_nowait({
                        "type": "response.created", "response_id": "complete-speak",
                        "action": "SPEAK", "effective_action": "SPEAK",
                        "intent_source_frame_id": 13, "intent_endpoint_frame_id": self.late_endpoint,
                    })
                    self.incoming.put_nowait({"type": "response.done", "response_id": "complete-speak"})
                    await asyncio.sleep(0)

        async def recv(self):
            return json.dumps(await self.incoming.get())

    socket = EndpointSocket()
    monkeypatch.setitem(sys.modules, "websockets", types.SimpleNamespace(connect=lambda *_args, **_kwargs: socket))
    trace_rows = []
    result = asyncio.run(e2e._run_session(
        scenario=scenario,
        assets={"turn_1_user": wav_path},
        label=scenario.scenario_id,
        ws_url="ws://cpu.invalid/realtime",
        policy="duplex_control6",
        manifest=manifest,
        output_dir=tmp_path / "run",
        trace_rows=trace_rows,
        pace=False,
    ))

    client_appends = [row for row in trace_rows if row.get("direction") == "client"
                      and row.get("event", {}).get("type") == "input_audio_buffer.append"]
    early_done = next(row for row in trace_rows if row.get("direction") == "server"
                      and row.get("event", {}).get("type") == "response.done"
                      and row.get("event", {}).get("response_id") == "early-speak")
    last_voice_append = next(row for row in client_appends if row.get("input_frame_id") == 15)
    gate = next(row["event"] for row in trace_rows if row.get("direction") == "local"
                and row.get("event", {}).get("type") == "turn_completion_gate")

    assert result["errors"] == []
    assert socket.user_frames == 7
    assert socket.internal_pause_frames == 9
    assert socket.endpoint_silence_frames >= 10
    assert early_done["received_at_monotonic_ns"] < last_voice_append["sent_at_monotonic_ns"]
    assert gate["response_id"] == "complete-speak"
    assert gate["last_voiced_input_frame_id"] == 15
    assert gate["intent_endpoint_frame_id"] == 25
    assert "complete-speak" in result["responses"]
    assert result["input_audio_stream"][13:16] and all(frame.any() for frame in result["input_audio_stream"][13:16])

def test_response_wait_streams_real_zero_pcm_frames_until_terminal(tmp_path) -> None:
    trace_rows = []
    collector = e2e._SessionCollector("silence-fixture", tmp_path, trace_rows)

    class SilenceEndpoint:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            event = json.loads(raw)
            self.sent.append(event)
            if event.get("type") != "input_audio_buffer.append":
                return
            payload = np.frombuffer(base64.b64decode(event["audio"]), dtype="<i2")
            assert payload.size == e2e.FRAME_SAMPLES
            assert not np.any(payload)
            if len(self.sent) == 8:
                collector.record("server", {"type": "response.created", "response_id": "silence-response"},
                                 e2e.time.monotonic_ns())
                collector.record("server", {"type": "response.done", "response_id": "silence-response"},
                                 e2e.time.monotonic_ns())

    ws = SilenceEndpoint()
    created, end = asyncio.run(e2e._wait_response_with_silence(
        ws, collector, pace=False, timeout_s=1.0,
    ))

    appends = [event for event in ws.sent if event.get("type") == "input_audio_buffer.append"]
    assert created and created["response_id"] == "silence-response"
    assert end and end["type"] == "response.done"
    # The unpaced CPU fixture may race a few extra frames before cancellation;
    # the real capture path paces each frame to 80 ms.
    assert len(appends) >= 8  # At least 640 ms of actual microphone silence.
    client_appends = [row for row in trace_rows if row.get("direction") == "client"]
    assert len(client_appends) >= 8
    assert all(row.get("audio_samples") == e2e.FRAME_SAMPLES for row in client_appends)
    assert all(row.get("silence_frame") is True for row in client_appends)


def test_barge_wait_ignores_backchannel_and_requires_speak_playout(tmp_path) -> None:
    async def run() -> None:
        trace_rows = []
        collector = e2e._SessionCollector("barge-trigger-fixture", tmp_path, trace_rows)
        await collector.playout.start()

        def record_response(response_id: str, action: str) -> None:
            now = e2e.time.monotonic_ns()
            if action == "SPEAK":
                collector.record("server", {
                    "type": "duplex_control6.decision", "action": "SPEAK",
                    "accepted": False, "rejection_reason": "deferred_speak_dispatched",
                }, now)
                created = {
                    "type": "response.created", "response_id": response_id,
                    "action": "SPEAK", "effective_action": "SPEAK",
                    "intent_source_frame_id": 0, "intent_endpoint_frame_id": 8,
                    "intent_source": "scripted_fixture",
                }
            else:
                collector.record("server", {
                    "type": "duplex_control6.decision", "action": action, "accepted": True,
                }, now)
                created = {"type": "response.created", "response_id": response_id}
            collector.record("server", created, now + 1)
            pcm = 0.1 * np.sin(2 * np.pi * 440 * np.arange(e2e.FRAME_SAMPLES) / 24_000).astype(np.float32)
            collector.record("server", {
                "type": "response.audio.delta", "response_id": response_id,
                "delta": _wav_payload(pcm, 24_000), "sample_rate": 24_000,
            }, now + 2)
            collector.record("server", {
                "type": "response.done", "response_id": response_id,
            }, now + 3)

        record_response("backchannel-response", "BACKCHANNEL")
        # Give the simulator time to start and consume the control response.
        await asyncio.sleep(0.01)

        class SpeakEndpoint:
            def __init__(self):
                self.silence_frames = 0

            async def send(self, raw):
                event = json.loads(raw)
                if event.get("type") == "input_audio_buffer.append":
                    self.silence_frames += 1
                    if self.silence_frames == 1:
                        record_response("speak-response", "SPEAK")

        ws = SpeakEndpoint()
        trigger = await e2e._wait_for_speak_playout_active(
            ws, collector, pace=False, timeout_s=1.0,
        )
        assert trigger["response_id"] == "speak-response"
        assert collector.response_actions["backchannel-response"] == "BACKCHANNEL"
        assert collector.response_actions["speak-response"] == "SPEAK"
        assert collector.playout.is_response_playout_active("speak-response")
        await collector.playout.finish(timeout_s=1.0)

    asyncio.run(run())


def test_asr_metrics_are_null_without_explicit_backend_and_use_real_transcript_when_enabled() -> None:
    unavailable = compare_asr_metrics("Ciao mondo", "ciao mondo", "Ciao mondo", "ciao mondo", configured=False)
    assert unavailable["status"] == "unavailable"
    assert unavailable["input_wer"] is unavailable["output_wer"] is None
    measured = compare_asr_metrics("Ciao mondo", "ciao mondo", "ciao mondo", "ciao mondo", configured=True)
    assert measured["status"] == "available"
    assert measured["input_wer"] == measured["output_wer"] == 0.0
    assert measured["output_cer"] == 0.0
    assert calculate_cer("abc", "adc") == pytest.approx(1 / 3, abs=0.0001)


def test_asr_backend_preflight_dispatches_only_to_the_matching_local_format(tmp_path) -> None:
    transformers_model = tmp_path / "transformers-whisper"
    transformers_model.mkdir()
    (transformers_model / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (transformers_model / "model.safetensors").touch()
    faster_whisper_model = tmp_path / "faster-whisper"
    faster_whisper_model.mkdir()
    (faster_whisper_model / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (faster_whisper_model / "model.bin").touch()

    assert e2e._resolve_asr_backend(transformers_model, "auto") == "transformers"
    assert e2e._resolve_asr_backend(faster_whisper_model, "auto") == "faster-whisper"
    with pytest.raises(ValueError, match="local Transformers Whisper checkpoint"):
        e2e._resolve_asr_backend(faster_whisper_model, "transformers")


def test_transformers_asr_loader_transcribes_audio_without_falling_back(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "whisper-large-v3-turbo"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type":"whisper"}', encoding="utf-8")
    (model_path / "model.safetensors").touch()
    calls = {}

    class FakeModel:
        def to(self, device):
            calls["model_device"] = str(device)
            return self

        def eval(self):
            calls["model_eval"] = True

    class FakeProcessor:
        tokenizer = object()
        feature_extractor = object()

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"

        @staticmethod
        def device(value):
            return value

        class cuda:
            @staticmethod
            def is_available():
                return True

    class FakePipeline:
        def __call__(self, audio, *, generate_kwargs):
            calls["audio"] = audio
            calls["generate_kwargs"] = generate_kwargs
            return {"text": " trascrizione indipendente "}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["model_load"] = (path, kwargs)
            return FakeModel()

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls["processor_load"] = (path, kwargs)
            return FakeProcessor()

    def fake_pipeline(task, **kwargs):
        calls["pipeline"] = (task, kwargs)
        return FakePipeline()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoModelForSpeechSeq2Seq=FakeAutoModel,
        AutoProcessor=FakeAutoProcessor,
        pipeline=fake_pipeline,
    ))

    transcribe, error, backend = e2e._load_local_asr(str(model_path), "cuda", "auto")

    assert error is None
    assert backend == "transformers"
    assert transcribe is not None
    pcm = np.full(160, 0.1, dtype=np.float32)
    assert transcribe(pcm, 16_000) == "trascrizione indipendente"
    assert calls["model_load"][1]["local_files_only"] is True
    assert calls["processor_load"][1]["local_files_only"] is True
    assert calls["pipeline"][0] == "automatic-speech-recognition"
    assert calls["pipeline"][1]["device"] == "cuda:0"
    assert calls["audio"]["sampling_rate"] == 16_000
    assert calls["generate_kwargs"]["language"] == "italian"
    assert calls["generate_kwargs"]["task"] == "transcribe"


def test_event_metrics_count_transport_responses_and_real_timing() -> None:
    rows = [
        {"direction": "client", "session_id": "s1", "sent_at_monotonic_ns": 1_000_000_000, "event": {"type": "input_audio_buffer.append", "audio_samples": 1280}},
        {"direction": "server", "received_at_monotonic_ns": 1_030_000_000, "session": "s1", "event": {"type": "input_audio_buffer.consumed", "frame_id": 3}},
        {"direction": "client", "session_id": "s1", "sent_at_monotonic_ns": 1_100_000_000, "event": {"type": "input_audio_buffer.commit"}},
        {"direction": "server", "received_at_monotonic_ns": 1_200_000_000, "session": "s1", "event": {"type": "response.created", "response_id": "r1"}},
        {"direction": "server", "received_at_monotonic_ns": 1_500_000_000, "session": "s1", "event": {"type": "response.audio.delta", "response_id": "r1", "audio_samples": 2400, "sample_rate": 24000}},
        {"direction": "server", "received_at_monotonic_ns": 2_200_000_000, "session": "s1", "event": {"type": "response.done", "response_id": "r1"}},
        {"direction": "server", "received_at_monotonic_ns": 2_300_000_000, "session": "s1", "event": {"type": "error", "error": {"code": "X"}}},
    ]
    metrics = compute_event_metrics(rows)
    assert metrics["input_frames_sent"] == 1
    assert metrics["input_ack_events"] == 1
    assert metrics["server_errors"] == 1
    assert metrics["responses_completed"] == 1
    assert metrics["frame_to_response_start_latency_ms"][0] == pytest.approx(100.0)
    assert metrics["frame_to_first_audio_latency_ms"] == [400.0]
    assert metrics["mean_frame_to_first_audio_latency_ms"] == pytest.approx(400.0)
    assert metrics["output_audio_samples"] == 2400
    assert metrics["output_rtf"] == pytest.approx(0.1)


def test_input_transcription_metrics_measure_revisions_and_snapshot_latency() -> None:
    rows = [
        {"direction": "client", "session_id": "s1", "sent_at_monotonic_ns": 1_000_000_000,
         "input_frame_id": 0, "event": {"type": "input_audio_buffer.append"}},
        {"direction": "client", "session_id": "s1", "sent_at_monotonic_ns": 1_500_000_000,
         "input_frame_id": 5, "event": {"type": "input_audio_buffer.append"}},
        {"direction": "client", "session_id": "s1", "sent_at_monotonic_ns": 1_900_000_000,
         "input_frame_id": 9, "event": {"type": "input_audio_buffer.append"}},
        {"direction": "server", "session_id": "s1", "received_at_monotonic_ns": 2_000_000_000,
         "event": {"type": "input_audio_transcription.partial", "utterance_id": "utt-1", "revision": 1,
                   "transcript": "ciao", "first_frame_id": 0, "last_frame_id": 5}},
        {"direction": "server", "session_id": "s1", "received_at_monotonic_ns": 2_200_000_000,
         "event": {"type": "input_audio_transcription.partial", "utterance_id": "utt-1", "revision": 2,
                   "transcript": "ciao mondo", "first_frame_id": 0, "last_frame_id": 9}},
        {"direction": "server", "session_id": "s1", "received_at_monotonic_ns": 2_300_000_000,
         "event": {"type": "input_audio_transcription.completed", "utterance_id": "utt-1", "revision": 3,
                   "transcript": "ciao mondo", "first_frame_id": 0, "last_frame_id": 9}},
    ]

    metrics = e2e.summarize_input_audio_transcription(rows, requested=True, enabled=True)

    assert metrics["status"] == "observed"
    assert metrics["partial_events"] == 2
    assert metrics["completed_events"] == 1
    assert metrics["utterances"] == 1
    assert metrics["revision_monotonic"] is True
    assert metrics["first_partial_from_utterance_onset_ms"] == [1000.0]
    assert metrics["first_partial_delivery_lag_ms"] == [500.0]


def test_input_transcription_metrics_fail_closed_on_nonmonotonic_revisions() -> None:
    rows = [
        {"direction": "server", "session_id": "s1", "received_at_monotonic_ns": 10,
         "event": {"type": "input_audio_transcription.partial", "utterance_id": "utt-1", "revision": 2,
                   "transcript": "ciao", "first_frame_id": 0, "last_frame_id": 1}},
        {"direction": "server", "session_id": "s1", "received_at_monotonic_ns": 20,
         "event": {"type": "input_audio_transcription.completed", "utterance_id": "utt-1", "revision": 2,
                   "transcript": "ciao", "first_frame_id": 0, "last_frame_id": 1}},
    ]

    metrics = e2e.summarize_input_audio_transcription(rows, requested=True, enabled=True)

    assert metrics["status"] == "invalid_events"
    assert metrics["revision_monotonic"] is False


def test_session_setup_requests_native_transcription_and_checks_server_echo() -> None:
    class Collector:
        session_id = "s1"

        def record(self, *_args, **_kwargs):
            return None

    class Socket:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(json.loads(raw))

    socket = Socket()
    collector = Collector()

    async def run():
        calls = 0

        async def ordered_wait(types_, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "session.created"}
            return {"type": "session.updated", "session_id": "s1", "input_audio_transcription_enabled": True}

        collector.wait_for = ordered_wait
        return await e2e._setup(socket, collector, "instructions", "duplex_control6", input_audio_transcription=True)

    assert asyncio.run(run()) is True
    assert socket.sent[0]["session"]["input_audio_transcription"] is True


def test_parser_native_input_transcription_is_optin() -> None:
    assert e2e._parser().parse_args([]).input_audio_transcription is False
    assert e2e._parser().parse_args(["--input-audio-transcription"]).input_audio_transcription is True


def test_session_setup_rejects_unavailable_native_transcription_optin() -> None:
    class Collector:
        session_id = "s1"

        async def wait_for(self, _types, _timeout):
            return {"type": "session.created"}

        def record(self, *_args, **_kwargs):
            return None

    class Socket:
        async def send(self, _raw):
            return None

    async def run():
        collector = Collector()
        calls = 0

        async def ordered_wait(types_, timeout):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {"type": "session.created"}
            return {"type": "session.updated", "session_id": "s1", "input_audio_transcription_enabled": False}

        collector.wait_for = ordered_wait
        return await e2e._setup(Socket(), collector, "instructions", "duplex_control6", input_audio_transcription=True)

    with pytest.raises(RuntimeError, match="did not enable native input transcription"):
        asyncio.run(run())


def test_action_coverage_and_session_isolation_require_observed_evidence() -> None:
    rows = [
        {"session": "s1", "event": {"type": "duplex_control6.decision", "action": "LISTEN", "confidence": 1,
          "frame_id": 1, "listener_frame_id": 1, "session": "s1", "source": "duplex_control6", "accepted": True}},
        {"session": "s1", "event": {"type": "response.created", "response_id": "r1"}},
        {"session": "s2", "event": {"type": "duplex_control6.decision", "action": "SPEAK", "confidence": 1,
          "frame_id": 2, "listener_frame_id": 2, "session": "s2", "source": "duplex_control6", "accepted": False}},
        {"session": "s2", "event": {"type": "response.created", "response_id": "r2"}},
    ]
    coverage = action_coverage(rows, ACTIONS, "duplex_control6")
    assert coverage["observed"] == ["LISTEN", "SPEAK"]
    assert coverage["missing"] == list(ACTIONS[2:])
    accepted = action_coverage(rows, ACTIONS, "duplex_control6", accepted_only=True)
    assert accepted["observed"] == ["LISTEN"]
    assert accepted["missing"] == ["SPEAK", *ACTIONS[2:]]
    assert session_isolation_evidence(rows)["status"] == "observed"
    assert session_isolation_evidence(rows[:2])["status"] == "unavailable"


def test_action_lifecycle_keeps_deferred_and_executed_responses_separate() -> None:
    base = {
        "type": "duplex_control6.decision", "confidence": 0.9,
        "frame_id": 3, "listener_frame_id": 3, "session": "s1",
        "source": "native_gemma_listener:last_audio_query",
    }
    rows = [
        {"event": {**base, "action": "SPEAK", "accepted": False,
                    "rejection_reason": "deferred_speak_dispatched"}},
        {"event": {**base, "action": "SPEAK", "accepted": False,
                    "rejection_reason": "deferred_until_backchannel_done"}},
        {"event": {**base, "action": "BACKCHANNEL", "accepted": True}},
        {"event": {"type": "response.created", "response_id": "bc",
                    "effective_action": "BACKCHANNEL"}},
        {"event": {"type": "response.created", "response_id": "speak",
                    "action": "SPEAK"}},
        {"event": {"type": "response.created", "response_id": "unknown"}},
    ]

    accepted = action_coverage(rows, ACTIONS, "duplex_control6", accepted_only=True)
    lifecycle = action_lifecycle_coverage(rows, ACTIONS, "duplex_control6")

    assert accepted["count_by_action"] == {"LISTEN": 0, "SPEAK": 0, "CONTINUE": 0,
                                            "YIELD": 0, "BACKCHANNEL": 1, "REPLAN": 0}
    assert lifecycle["deferred_decisions"]["count"] == 2
    assert lifecycle["deferred_decisions"]["count_by_action"]["SPEAK"] == 2
    assert lifecycle["deferred_decisions"]["count_by_reason"] == {
        "deferred_speak_dispatched": 1,
        "deferred_until_backchannel_done": 1,
    }
    assert lifecycle["executed_responses"]["count"] == 2
    assert lifecycle["executed_responses"]["count_by_action"]["BACKCHANNEL"] == 1
    assert lifecycle["executed_responses"]["count_by_action"]["SPEAK"] == 1
    assert lifecycle["executed_responses"]["unattributed_response_count"] == 1


def test_empty_and_nonfinite_audio_are_unavailable_or_invalid() -> None:
    assert audio_quality(np.empty(0, dtype=np.float32), 24000)["status"] == "unavailable"
    invalid = audio_quality(np.asarray([0.1, np.nan], dtype=np.float32), 24000)
    assert invalid["status"] == "invalid"
    assert invalid["metrics"]["finite"] is False


def test_audio_quality_flags_silence_and_clipping() -> None:
    silent = audio_quality(np.zeros(24_000, dtype=np.float32), 24_000)
    assert silent["status"] == "silent"
    assert silent["metrics"]["silence_frame_ratio"] == 1.0
    clipped = audio_quality(np.tile(np.asarray([1.0, -1.0], dtype=np.float32), 12_000), 24_000)
    assert clipped["status"] == "degraded"
    assert clipped["metrics"]["clipping_pct"] == 100.0
    discontinuous = audio_quality(np.asarray([-0.2, 0.2, -0.2, 0.8], dtype=np.float32), 1000, [2])
    assert discontinuous["metrics"]["packet_discontinuities_count"] == 1


def test_pcm_playout_simulator_retains_yield_tail_after_response_done() -> None:
    async def run() -> dict:
        simulator = e2e._PCMPlaybackSimulator(label="yield", trace_rows=[])
        await simulator.start()
        response_id = "yield-response"
        now_ns = e2e.time.monotonic_ns()
        simulator.mark_yield(response_id, now_ns)
        simulator.enqueue(response_id, pcm, 24_000, now_ns)
        simulator.finish_response(response_id, "response.done", now_ns)
        return await simulator.finish(timeout_s=2.0)

    pcm = np.linspace(-0.3, 0.3, 2400, dtype=np.float32)
    result = asyncio.run(run())
    response = result["responses"]["yield-response"]

    assert response["terminal_event"] == "response.done"
    assert response["queued_samples_at_terminal"] == 2400
    assert response["done_before_playback_complete"] is True
    assert response["audio_samples_consumed"] == 2400
    assert response["audio_samples_after_yield_consumed"] == 2400
    assert response["playback_end_ns"] > response["terminal_ns"]
    np.testing.assert_array_equal(result["played_pcm_by_response"]["yield-response"], pcm)
    assert not response["audio_samples_cancelled"]


def test_pcm_playout_cancel_clears_only_matching_response_and_records_pcm() -> None:
    cancelled_pcm = np.linspace(-0.4, 0.4, 4800, dtype=np.float32)
    retained_pcm = np.linspace(0.4, -0.4, 1920, dtype=np.float32)

    async def run() -> dict:
        simulator = e2e._PCMPlaybackSimulator(label="cancel", trace_rows=[])
        await simulator.start()
        now_ns = e2e.time.monotonic_ns()
        simulator.enqueue("cancel-response", cancelled_pcm, 24_000, now_ns)
        simulator.enqueue("retained-response", retained_pcm, 24_000, now_ns)
        await asyncio.sleep(0.02)
        simulator.finish_response("cancel-response", "response.cancelled", e2e.time.monotonic_ns())
        simulator.finish_response("retained-response", "response.done", e2e.time.monotonic_ns())
        return await simulator.finish(timeout_s=2.0)

    result = asyncio.run(run())
    cancelled = result["responses"]["cancel-response"]
    retained = result["responses"]["retained-response"]

    assert cancelled["terminal_event"] == "response.cancelled"
    assert cancelled["audio_samples_consumed"] > 0
    assert cancelled["audio_samples_cancelled"] == 4800 - cancelled["audio_samples_consumed"]
    assert cancelled["queue_cleared_samples"] > 0
    assert retained["audio_samples_consumed"] == 1920
    assert retained["audio_samples_cancelled"] == 0
    np.testing.assert_array_equal(
        result["played_pcm_by_response"]["retained-response"], retained_pcm
    )
    np.testing.assert_array_equal(
        result["cancelled_pcm_by_response"]["cancel-response"],
        cancelled_pcm[cancelled["audio_samples_consumed"]:],
    )
