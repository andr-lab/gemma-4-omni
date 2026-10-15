from __future__ import annotations

from duplex_kernel.native_control6_prompt import (
    NATIVE_CONTROL6_INSTRUCTIONS_MAX_TOKENS,
    NATIVE_CONTROL6_MESSAGE_MAX_TOKENS,
    NATIVE_CONTROL6_PREFIX_MAX_TOKENS,
    build_native_control6_prefix_ids,
    build_native_control6_prefix_messages,
    select_native_control6_replay_suffix,
    validate_native_control6_semantic_prefix_metadata,
)


class _TokenizerStub:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {"<|audio|>": 2}
        self.inverse = {2: "<|audio|>"}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "<|audio|>":
            return [2]
        ids = []
        for word in text.split():
            if word not in self.vocab:
                index = max(self.vocab.values(), default=2) + 1
                self.vocab[word] = index
                self.inverse[index] = word
            ids.append(self.vocab[word])
        return ids

    def decode(
        self,
        ids: list[int],
        *,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return " ".join(self.inverse.get(int(token), f"t{token}") for token in ids)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        ids = [1]
        for message in messages:
            role = message["role"]
            ids.append({"system": 4, "user": 5, "assistant": 6}[role])
            ids.extend(self.encode(message["content"], add_special_tokens=False))
            ids.append(3)
        if add_generation_prompt:
            ids.append(6)
        return ids


def test_native_control6_prefix_uses_recent_history_and_ends_at_audio_boa():
    tokenizer = _TokenizerStub()
    history = []
    for index in range(8):
        history.append({
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"turn{index} " + "dettaglio " * 79,
        })

    ids, prefix_len = build_native_control6_prefix_ids(
        tokenizer,
        instructions="istruzione " * 100,
        completed_messages=history,
        audio_marker="<|audio|>",
        boa_token_id=256000,
    )
    messages = build_native_control6_prefix_messages(
        tokenizer,
        instructions="istruzione " * 100,
        completed_messages=history,
        audio_marker="<|audio|>",
    )

    assert len(ids) == prefix_len <= NATIVE_CONTROL6_PREFIX_MAX_TOKENS
    assert ids[-1] == 256000  # Real BOA replaces, rather than embeds, the marker token.
    assert ids[-2] != 2
    assert messages[-2]["role"] == "assistant"
    assert "turn7" in messages[-2]["content"]
    assert len(tokenizer.encode(messages[0]["content"].split("Istruzioni della sessione: ", 1)[-1])) <= NATIVE_CONTROL6_INSTRUCTIONS_MAX_TOKENS
    assert all(
        len(tokenizer.encode(message["content"])) <= NATIVE_CONTROL6_MESSAGE_MAX_TOKENS
        for message in messages[1:-1]
    )
    assert messages[-1] == {"role": "user", "content": "<|audio|>"}


def test_native_control6_prefix_fails_closed_without_single_boa_token():
    tokenizer = _TokenizerStub()
    tokenizer.encode = lambda text, **kwargs: [2, 7] if text == "<|audio|>" else []

    try:
        build_native_control6_prefix_ids(
            tokenizer,
            instructions="",
            completed_messages=[],
            audio_marker="<|audio|>",
            boa_token_id=256000,
        )
    except ValueError as exc:
        assert "exactly one placeholder token" in str(exc)
    else:
        raise AssertionError("multi-token audio marker must fail closed")


def test_replay_suffix_requires_observed_endpoint_and_keeps_only_contiguous_tail():
    observed = [(10, "a"), (11, "b"), (14, "c"), (15, "d")]
    assert select_native_control6_replay_suffix(
        observed, expected_last_frame_id=15, max_frames=64
    ) == ((14, 15), ("c", "d"))
    assert select_native_control6_replay_suffix(
        observed, expected_last_frame_id=13, max_frames=64
    ) is None
    assert select_native_control6_replay_suffix(
        observed, expected_last_frame_id=-1, max_frames=64
    ) == ((), ())


def test_semantic_prefix_checkpoint_metadata_matches_runtime_schema_v2():
    metadata = {
        "architecture": "native_gemma_selected_output_rows_lora",
        "semantic_prefix_schema_version": 2,
        "semantic_prefix_contract": "gemma_native_control6_semantic_prefix_boa_v2",
        "semantic_prefix_boa_token_id": 256000,
        "semantic_prefix_capacity": 291,
        "semantic_prefix_max_tokens": 291,
        "semantic_prefix_context_conditioned": True,
        "response_adapter": "OFF",
        "ordinary_vocabulary_rows_modified": False,
        "base_gemma_weights_modified": False,
    }
    report = {
        key: value for key, value in metadata.items()
        if key.startswith("semantic_prefix_")
    }
    result = validate_native_control6_semantic_prefix_metadata({
        **metadata,
        "report": report,
    })
    assert result == {
        key: value for key, value in metadata.items()
        if key.startswith("semantic_prefix_")
    }

    stale = {**metadata, "semantic_prefix_schema_version": 1}
    try:
        validate_native_control6_semantic_prefix_metadata({
            **stale,
            "report": report,
        })
    except ValueError as exc:
        assert "semantic_prefix_schema_version" in str(exc)
    else:
        raise AssertionError("a mismatched prefix checkpoint schema must fail closed")
