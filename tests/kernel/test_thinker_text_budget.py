from duplex_kernel.thinker_text_budget import (
    ends_at_sentence_boundary,
    graceful_budget_action,
    last_complete_word_prefix,
)


def test_sentence_boundary_allows_complete_token_sequence():
    pieces = {1: "Cosa", 2: " ti", 3: " interessa", 4: "?"}
    decode = lambda ids: "".join(pieces[token_id] for token_id in ids)

    assert ends_at_sentence_boundary(decode([1, 2, 3, 4]))
    assert last_complete_word_prefix([1, 2, 3, 4], decode) == (
        4,
        "Cosa ti interessa?",
    )


def test_incomplete_final_word_is_removed_at_hard_budget():
    pieces = {1: "Cosa", 2: " ti", 3: " affasc"}
    decode = lambda ids: "".join(pieces[token_id] for token_id in ids)

    assert not ends_at_sentence_boundary(decode([1, 2, 3]))
    assert last_complete_word_prefix([1, 2, 3], decode) == (2, "Cosa ti")


def test_single_unterminated_word_fails_closed():
    pieces = {1: "affasc"}
    decode = lambda ids: "".join(pieces[token_id] for token_id in ids)

    assert last_complete_word_prefix([1], decode) == (0, "")


def test_budget_extends_only_for_unfinished_sentence_and_then_marks_partial():
    assert graceful_budget_action("Una frase completa.", 64, 64, 96) == "sentence_complete"
    assert graceful_budget_action("La spiegazione continua", 64, 64, 96) == "extend"
    assert graceful_budget_action("La spiegazione continua", 96, 64, 96) == "partial"
    assert graceful_budget_action("Inizio", 63, 64, 96) == "continue"
