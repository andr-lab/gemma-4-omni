from __future__ import annotations

import pytest
import torch

from duplex_kernel.dataset_eos_contract import (
    DATASET_EOS_CONTRACT_SCHEMA_VERSION,
    DEFAULT_WINDOW_FRAMES,
    SegmentWindowRecord,
    UtteranceRecord,
    assert_voice_conditioning_compatible,
    build_segment_windows,
    split_utterances_by_time,
    contract_document,
    validate_window_records,
)


def _utterance(length: int, *, start: int = 0, utterance_id: str = "utt-1", speaker_id: str = "spk-1", text: str = "ciao") -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=utterance_id,
        speaker_id=speaker_id,
        frame_start=start,
        frame_end=start + length,
        text=text,
        text_token_ids=(11, 12),
        mel_provenance={"source": "mel-a", "sha256": "m" * 64},
        rvq_provenance={"source": "rvq-a", "sha256": "r" * 64, "codebooks": 16},
        voice_conditioning_signature="voice:spk-1",
    )


def test_intermediate_window_has_no_terminal_label() -> None:
    records = build_segment_windows([_utterance(65)], total_frames=65)

    assert len(records) == 2
    assert records[0].frame_start == 0
    assert records[0].frame_end == 64
    assert records[0].valid_frame_count == 64
    assert records[0].is_utterance_end is False
    assert records[1].frame_start == 64
    assert records[1].frame_end == 65
    assert records[1].is_utterance_end is True
    assert records[1].valid_frame_mask == (True,) + (False,) * 63


def test_contiguous_utterances_never_share_a_window_or_eos() -> None:
    records = build_segment_windows(
        [_utterance(1, utterance_id="utt-a"), _utterance(63, start=1, utterance_id="utt-b")],
        total_frames=64,
    )

    assert [(item.utterance_id, item.frame_start, item.frame_end, item.is_utterance_end) for item in records] == [
        ("utt-a", 0, 1, True),
        ("utt-b", 1, 64, True),
    ]
    assert records[0].valid_frame_count == 1
    assert records[1].valid_frame_count == 63


@pytest.mark.parametrize("length", [1, 63, 64])
def test_boundary_lengths_keep_terminal_query_reachable(length: int) -> None:
    records = build_segment_windows([_utterance(length)], total_frames=length)

    assert len(records) == 1
    assert records[0].is_utterance_end is True
    assert records[0].valid_frame_count == length
    assert len(records[0].valid_frame_mask) == DEFAULT_WINDOW_FRAMES


def test_missing_tokenization_disables_semantic_text_supervision() -> None:
    record = UtteranceRecord("utt-noids", "spk-1", 0, 1, "ciao", ())
    window = build_segment_windows([record], total_frames=1)[0]
    assert window.semantic_text_valid is False
    assert window.input_text_token_ids == ()


def test_text_capacity_reserves_a_target_shift_slot() -> None:
    record = UtteranceRecord(
        utterance_id="utt-48",
        speaker_id="spk-1",
        frame_start=0,
        frame_end=1,
        text="testo",
        text_token_ids=tuple(range(48)),
    )
    window = build_segment_windows([record], total_frames=1, text_block_len=48)[0]
    assert window.text_truncated is True
    assert window.input_text_token_ids == ()
    assert window.semantic_text_valid is False


def test_long_text_is_retained_but_not_used_as_a_truncated_conditioning_prefix() -> None:
    long_ids = tuple(range(49))
    record = UtteranceRecord(
        utterance_id="utt-long",
        speaker_id="spk-1",
        frame_start=0,
        frame_end=1,
        text="una frase molto lunga",
        text_token_ids=long_ids,
    )

    window = build_segment_windows([record], total_frames=1)[0]

    assert window.text_truncated is True
    assert window.text_token_ids == long_ids
    assert window.input_text_token_ids == ()
    assert window.semantic_text_valid is False
    assert window.effective_text == "una frase molto lunga"


def test_padding_and_provenance_are_serialized_without_losing_target_identity() -> None:
    window = build_segment_windows([_utterance(1)], total_frames=1)[0]
    payload = window.to_dict()

    assert payload["valid_frame_mask"] == [True] + [False] * 63
    assert payload["is_utterance_end"] is True
    assert payload["mel_provenance"]["sha256"] == "m" * 64
    assert payload["rvq_provenance"]["codebooks"] == 16
    assert payload["utterance_id"] == "utt-1"


def test_validation_rejects_a_fake_terminal_on_an_intermediate_window() -> None:
    records = build_segment_windows([_utterance(65)], total_frames=65)
    fake = SegmentWindowRecord(**{**records[0].__dict__, "is_utterance_end": True})

    with pytest.raises(ValueError, match="termina prima"):
        validate_window_records([fake, records[1]], DEFAULT_WINDOW_FRAMES)


def test_split_rejects_partial_utterance() -> None:
    with pytest.raises(ValueError, match="taglia un utterance"):
        build_segment_windows([_utterance(65)], total_frames=65, frame_start=64, frame_end=65)


def test_split_is_by_complete_utterance_and_optional_speaker_group() -> None:
    records = [
        _utterance(4, utterance_id="a-1", speaker_id="a"),
        _utterance(4, start=4, utterance_id="a-2", speaker_id="a"),
        _utterance(4, start=8, utterance_id="b-1", speaker_id="b"),
    ]
    train, validation = split_utterances_by_time(records, validation_fraction=0.34)
    assert {item.utterance_id for item in train + validation} == {"a-1", "a-2", "b-1"}
    assert not ({item.utterance_id for item in train} & {item.utterance_id for item in validation})
    train_spk, validation_spk = split_utterances_by_time(records, validation_fraction=0.34, by_speaker=True)
    assert {item.speaker_id for item in validation_spk} <= {"a", "b"}
    assert not ({item.speaker_id for item in train_spk} & {item.speaker_id for item in validation_spk})


def test_voice_conditioning_mismatch_fails_closed() -> None:
    record = build_segment_windows([_utterance(1)], total_frames=1)[0]

    with pytest.raises(ValueError, match="incompatibile"):
        assert_voice_conditioning_compatible(
            record,
            {"signature": "voice:another-speaker"},
        )
    with pytest.raises(ValueError, match="mancante"):
        assert_voice_conditioning_compatible(
            {"speaker_id": "alice", "voice_conditioning_signature": None},
            None,
        )


def test_contract_document_is_versioned_and_matches_normative_json() -> None:
    document = contract_document()
    assert document["schema_version"] == DATASET_EOS_CONTRACT_SCHEMA_VERSION
    assert document["eos"]["terminal_query_budget"] == 1
    assert document["continuation"]["max_acoustic_frames_per_segment"] == 64
    assert document["continuation"]["terminal_query_does_not_consume_acoustic_budget"] is True



