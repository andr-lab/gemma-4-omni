import pytest
import torch

from duplex_kernel.response_intent_local import (
    align_generated_response_hidden_to_qwen_tokens,
    align_hidden_states_by_char_overlap,
    build_local_response_prefix,
    build_token_char_spans,
    load_local_cursor_projector,
    select_frame_hidden_by_qwen_cursor,
)
from duplex_kernel.response_intent import ResponseIntentProjector


def test_unicode_piece_spans_trim_only_outer_space_and_keep_token_indices():
    pieces = {1: " L", 2: "’al", 3: "bero "}
    spans, kept = build_token_char_spans(
        [1, 2, 3], lambda token_id: pieces[token_id], expected_text="L’albero"
    )
    assert spans == [(0, 1), (1, 4), (4, 8)]
    assert kept == [0, 1, 2]


def test_qwen_tokens_receive_mean_of_overlapping_gemma_hidden_states():
    gemma_offsets = [(0, 1), (1, 4), (4, 8)]
    qwen_offsets = [(0, 2), (2, 6), (6, 8)]
    gemma_hidden = torch.tensor([[[1.0], [3.0], [5.0]]])
    aligned = align_hidden_states_by_char_overlap(
        gemma_hidden, gemma_offsets, qwen_offsets
    )
    torch.testing.assert_close(aligned, torch.tensor([[[2.0], [4.0], [5.0]]]))


def test_qwen_frame_cursor_uses_first_token_for_bos_then_tail_holds_last():
    qwen_tokens = torch.tensor([[[10.0], [20.0], [30.0]]])
    frames, provenance = select_frame_hidden_by_qwen_cursor(qwen_tokens, frame_count=5)
    torch.testing.assert_close(frames, torch.tensor([[[10.0], [20.0], [30.0], [30.0], [30.0]]]))
    assert provenance == ["qwen_text_cursor"] * 3 + ["tail_hold"] * 2


def test_mapping_fails_closed_when_text_or_character_coverage_differs():
    with pytest.raises(ValueError, match="reconstruct"):
        build_token_char_spans([1, 2], lambda token_id: {1: "ca", 2: "ne"}[token_id], expected_text="cane!")
    with pytest.raises(ValueError, match="overlap"):
        align_hidden_states_by_char_overlap(
            torch.ones(1, 1, 4), [(0, 1)], [(1, 2)]
        )


def test_local_prefix_builder_uses_qwen_processor_ids_and_holds_tail():
    class GemmaTokenizer:
        def decode(self, ids, **_kwargs):
            return {1: "a", 2: "b"}[int(ids[0])]

    class QwenTokenizer:
        def __call__(self, text, **_kwargs):
            prefix = "<|im_start|>assistant\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n"
            assert text == prefix + "ab" + suffix
            ids = [0, 1, 2, 10, 11, 20, 21, 22, 23, 24]
            offsets = [(0, 0)] * 3 + [(len(prefix), len(prefix) + 1), (len(prefix) + 1, len(prefix) + 2)] + [(0, 0)] * 5
            return {"input_ids": ids, "offset_mapping": offsets}

        def decode(self, ids, **_kwargs):
            return "".join({10: "a", 11: "b"}[int(value)] for value in ids)

    class QwenProcessor:
        tokenizer = QwenTokenizer()

        def __call__(self, *, text, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}

    projector = ResponseIntentProjector()
    with torch.no_grad():
        projector.linear.weight.zero_()
        projector.linear.bias.zero_()
        projector.linear.weight[0, 0] = 1.0
    hidden = torch.zeros(1, 2, 1536)
    hidden[0, 0, 0] = 1.0
    hidden[0, 1, 0] = 2.0
    prefix, metrics = build_local_response_prefix(
        hidden,
        [1, 2],
        gemma_tokenizer=GemmaTokenizer(),
        qwen_tokenizer=QwenProcessor.tokenizer,
        qwen_processor=QwenProcessor(),
        text="ab",
        projector=projector,
    )
    assert prefix.shape == (1, 256, 2048)
    torch.testing.assert_close(prefix[0, :3, 0], torch.tensor([1.0, 2.0, 2.0]))
    assert metrics["qwen_raw_text_tokens"] == 2
    assert metrics["tail_hold_frames"] == 254


def test_generated_response_hidden_aligns_exactly_to_qwen_raw_text_tokens():
    class GemmaTokenizer:
        def decode(self, ids, **_kwargs):
            return {1: " a", 2: "b "}[int(ids[0])]

    class QwenTokenizer:
        def __call__(self, text, **_kwargs):
            prefix = "<|im_start|>assistant\n"
            suffix = "<|im_end|>\n<|im_start|>assistant\n"
            assert text == prefix + "ab" + suffix
            ids = [0, 1, 2, 10, 11, 20, 21, 22, 23, 24]
            offsets = [(0, 0)] * 3 + [
                (len(prefix), len(prefix) + 1),
                (len(prefix) + 1, len(prefix) + 2),
            ] + [(0, 0)] * 5
            return {"input_ids": ids, "offset_mapping": offsets}

        def decode(self, ids, **_kwargs):
            return "".join({10: "a", 11: "b"}[int(token_id)] for token_id in ids)

    class QwenProcessor:
        tokenizer = QwenTokenizer()

        def __call__(self, *, text, return_tensors):
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([self.tokenizer(text)["input_ids"]])}

    hidden = torch.tensor([[[1.0, 3.0], [5.0, 7.0]]])
    aligned, metrics = align_generated_response_hidden_to_qwen_tokens(
        hidden,
        [1, 2],
        gemma_tokenizer=GemmaTokenizer(),
        qwen_tokenizer=QwenProcessor.tokenizer,
        qwen_processor=QwenProcessor(),
        text="ab",
    )
    torch.testing.assert_close(aligned, hidden)
    assert metrics["qwen_raw_text_tokens"] == 2
    assert metrics["alignment"] == "character_overlap_mean"


def test_local_checkpoint_loader_requires_the_trained_cursor_contract(tmp_path):
    source = ResponseIntentProjector()
    checkpoint = tmp_path / "local.pt"
    torch.save({
        "state_dict": source.state_dict(),
        "training_contract": {
            "prefix_capacity": 256,
            "qwen_cursor": "frame 0 uses prefilled raw token 0; frame j selects min(j,Q-1); tail holds final token state",
        },
    }, checkpoint)
    loaded = load_local_cursor_projector(checkpoint, device="cpu", dtype=torch.float32)
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
