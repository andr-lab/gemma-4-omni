import pytest

from duplex_kernel.server_response_control import should_emit_final_text_fallback


@pytest.mark.parametrize(
    ("published_delta_count", "result_text", "expected"),
    [
        (0, "final answer", True),
        (1, "final answer", False),
        (16, "final answer", False),
        (16, "", False),
    ],
)
def test_final_text_fallback_depends_on_published_deltas_not_live_segment_state(
    published_delta_count: int,
    result_text: str,
    expected: bool,
) -> None:
    assert should_emit_final_text_fallback(
        published_delta_count=published_delta_count,
        result_text=result_text,
    ) is expected
