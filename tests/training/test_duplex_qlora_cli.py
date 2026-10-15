from __future__ import annotations

from pathlib import Path

from training.duplex_qlora import cli


def test_prepare_dispatches_from_synthetic_root_argument(monkeypatch, tmp_path, capsys):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "prepared"
    rows = [{"conversation_id": "synthetic-1", "segments": [], "frames": []}]
    seen: list[Path] = []

    monkeypatch.setattr(cli, "read_synthetic_v3_manifest", lambda path: seen.append(path) or rows)
    monkeypatch.setattr(cli, "prepare_splits", lambda conversations, output_dir, seed: {"rows": len(conversations)})
    monkeypatch.setattr(cli, "summarize_preparation_counts", lambda conversations: {"synthetic": len(conversations)})

    assert cli.main(["prepare", "--synthetic-root", str(manifest), "--output-dir", str(output)]) == 0
    assert seen == [manifest]
    assert '"rows": 1' in capsys.readouterr().out


def test_prepare_dispatches_clean_v3_asr_as_auxiliary_source(monkeypatch, tmp_path, capsys):
    source = tmp_path / "synthetic-v3"
    output = tmp_path / "prepared"
    rows = [{"conversation_id": "v3-clean-1", "synthetic": False, "segments": [], "frames": []}]
    seen = []

    monkeypatch.setattr(cli, "read_synthetic_v3_clean_asr", lambda path: seen.append(path) or rows)
    monkeypatch.setattr(cli, "prepare_splits", lambda conversations, output_dir, seed: {"rows": len(conversations)})
    monkeypatch.setattr(cli, "summarize_preparation_counts", lambda conversations: {"aux": len(conversations)})

    assert cli.main([
        "prepare", "--synthetic-v3-clean-asr-root", str(source), "--output-dir", str(output)
    ]) == 0
    assert seen == [source]
    assert '"aux": 1' in capsys.readouterr().out


def test_prepare_real_conversational_views_requires_opt_in_and_transforms_external_rows(monkeypatch, tmp_path, capsys):
    source = tmp_path / "smooth"
    output = tmp_path / "prepared"
    raw = [{"conversation_id": "raw", "source": "SmoothConv"}]
    views = [{
        "conversation_id": "SmoothConv:raw:local-channel-0", "source": "SmoothConv",
        "source_group_id": "raw-group", "source_view_eligible_conversations_seen": 1,
        "source_view_selected_conversations": 1,
        "frames": [{"target_action": "SPEAK", "action_mask": True}],
    }]
    seen = {}

    def transform(rows, *, max_conversations):
        seen["input"] = rows
        seen["max_conversations"] = max_conversations
        return views

    def prepare(rows, output_dir, seed):
        seen["prepared"] = rows
        return {"rows": len(rows)}

    monkeypatch.setattr(cli, "read_smoothconv", lambda path, evidence_path=None: raw)
    monkeypatch.setattr(cli, "build_real_conversational_views", transform)
    monkeypatch.setattr(cli, "prepare_splits", prepare)
    monkeypatch.setattr(cli, "summarize_preparation_counts", lambda rows: {"rows": len(rows)})

    assert cli.main([
        "prepare", "--smoothconv-root", str(source), "--real-conversational-views",
        "--real-conversational-max-conversations", "64",
        "--output-dir", str(output),
    ]) == 0
    assert seen["input"] == raw
    assert seen["max_conversations"] == 64
    assert seen["prepared"] == views
    output_text = capsys.readouterr().out
    assert '"rows": 1' in output_text
    assert '"supervised_frames_by_action": {\n      "SPEAK": 1' in output_text


def test_prepare_real_conversational_views_rejects_missing_external_root(tmp_path):
    try:
        cli.main(["prepare", "--synthetic-root", str(tmp_path / "synthetic.jsonl"),
                  "--real-conversational-views", "--output-dir", str(tmp_path / "out")])
    except SystemExit as exc:
        assert "richiede --smoothconv-root" in str(exc)
    else:
        raise AssertionError("flag opt-in deve richiedere una sorgente SmoothConv/DuplexConv")


def test_smoke_dispatch_requests_four_optimizer_updates(monkeypatch, tmp_path, capsys):
    from training.duplex_qlora import stage1

    seen: dict[str, object] = {}

    def fake_train_qlora(**kwargs):
        seen.update(kwargs)
        return {"steps": kwargs["max_steps"]}

    monkeypatch.setattr(stage1, "train_qlora", fake_train_qlora)
    assert cli.main([
        "train", "--train-jsonl", str(tmp_path / "train.jsonl"),
        "--conditioning-mode", "baselineaudio_response", "--smoke",
    ]) == 0
    assert seen["max_steps"] == 4
    assert seen["conditioning_mode"] == "baselineaudio_response"
    assert seen["use_cache_for_training"] is True
    assert '"steps": 4' in capsys.readouterr().out


def test_train_dispatches_ordered_schedule_jsonl(monkeypatch, tmp_path, capsys):
    from training.duplex_qlora import stage1

    seen = {}

    def fake_train_qlora(**kwargs):
        seen.update(kwargs)
        return {"steps": kwargs["max_steps"]}

    monkeypatch.setattr(stage1, "train_qlora", fake_train_qlora)
    schedule = tmp_path / "ordered_schedule.jsonl"
    assert cli.main([
        "train", "--train-jsonl", str(tmp_path / "train.jsonl"),
        "--validation-jsonl", str(tmp_path / "validation.jsonl"),
        "--conditioning-mode", "audio_asr_mix_v2", "--schedule-jsonl", str(schedule), "--smoke",
    ]) == 0

    assert seen["schedule_jsonl"] == schedule
    assert seen["conditioning_mode"] == "audio_asr_mix_v2"
    assert seen["max_steps"] == 4
    assert '"steps": 4' in capsys.readouterr().out


def test_train_requires_explicit_conditioning_mode(tmp_path):
    try:
        cli.build_parser().parse_args(["train", "--train-jsonl", str(tmp_path / "train.jsonl")])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("conditioning mode deve essere esplicito per ogni run Stage1")
