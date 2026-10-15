from __future__ import annotations

import asyncio

import tools.capture_duplex_qlora_sequential as capture
import tools.eval_duplex_qlora_e2e as e2e


def test_capture_applies_requested_scenario_before_building_asset_scope(tmp_path, monkeypatch) -> None:
    args = e2e._parser().parse_args([
        "--server", "ws://localhost:8124/v1/realtime?duplex=1",
        "--checkpoint-root", str(tmp_path / "checkpoint"),
        "--output-dir", str(tmp_path / "capture"),
        "--expected-run-id", "run-test",
        "--scenario-id", "01_name_number_memory",
    ])
    manifest = {
        "run_id": "run-test",
        "_manifest_path": tmp_path / "checkpoint" / "manifest.json",
        "base_model_id": "fixture/model",
        "base_model_path": "fixture/model",
        "adapter_path": tmp_path / "adapter",
        "control_head_path_resolved": tmp_path / "head.pt",
        "control_head": {
            "input_dim": 1536,
            "pooling": "last_audio_query",
            "tick_ms": 80,
            "actions": list(e2e.ACTION_ORDER),
        },
    }
    asset_scope: list[str] = []
    monkeypatch.setattr(e2e, "load_checkpoint_manifest", lambda _root: manifest)

    def record_asset_scope(_audio_dir, scenarios):
        asset_scope.extend(scenario.scenario_id for scenario in scenarios)
        return [], []

    monkeypatch.setattr(e2e, "_scenario_assets_for", record_asset_scope)
    monkeypatch.setattr(e2e, "validate_loaded_identity", lambda *_args: {"status": "unavailable"})
    monkeypatch.setattr(e2e, "compute_event_metrics", lambda _rows: {"server_errors": [], "responses_created": 0})
    monkeypatch.setattr(
        e2e, "action_coverage",
        lambda _rows, actions, _policy: {"missing": list(actions)},
    )
    monkeypatch.setattr(e2e, "live_policy_semantics", lambda _rows: {})

    report = asyncio.run(capture._capture(args))

    assert asset_scope == ["01_name_number_memory"]
    assert report["scenario_coverage"]["expected"] == ["01_name_number_memory"]
    assert report["requested"]["scenario_ids"] == ["01_name_number_memory"]
