from __future__ import annotations

from duplex_kernel.server import (
    _controller_guided_say_generation_options,
    _empty_controller_say_think_from_eos,
)


def test_only_natural_empty_eos_becomes_no_speech_think_result() -> None:
    error = RuntimeError("Thinker ha prodotto una risposta vuota")
    error.empty_eos = True
    error.terminal_reason = "eos"
    error.runtime_metrics = {"thinker_decode_ms": 12.5, "terminal_reason": "eos"}

    result = _empty_controller_say_think_from_eos(error, prompt_len=7)

    assert result == {
        "text": "",
        "gen_ids": [],
        "hidden": None,
        "hidden_prompt_len": 7,
        "prompt_len": 7,
        "runtime_metrics": {"thinker_decode_ms": 12.5, "terminal_reason": "eos"},
        "terminal_reason": "eos",
        "cancelled": False,
    }


def test_budget_or_unmarked_empty_errors_remain_failures() -> None:
    error = RuntimeError("Thinker ha prodotto una risposta vuota")
    error.empty_eos = False
    error.terminal_reason = "budget"

    assert _empty_controller_say_think_from_eos(error, prompt_len=3) is None


class _RecordingThinker:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def think(self, _ids, *, max_new_tokens: int, min_new_tokens: int):
        self.calls.append({"max_new_tokens": max_new_tokens, "min_new_tokens": min_new_tokens})
        return {"terminal_reason": "eos", "text": "Risposta completa."}


def test_production_controller_generation_options_reach_thinker_for_speak_and_backchannel() -> None:
    from duplex_kernel.audio_response_protocol import CONTROLLER_GUIDED_SAY_V1
    from duplex_kernel.turn_policy import ControlAction

    engine = _RecordingThinker()
    for action in (ControlAction.SPEAK, ControlAction.BACKCHANNEL):
        request_options = _controller_guided_say_generation_options(
            action=action,
            controller_guided_say=True,
            direct_audio_candidate=True,
            response_protocol=CONTROLLER_GUIDED_SAY_V1,
        )
        engine.think(
            [],
            max_new_tokens=int(request_options["max_tokens"]),
            min_new_tokens=int(request_options["_min_new_tokens"]),
        )
        assert engine.calls[-1] == {"max_new_tokens": 256, "min_new_tokens": 1}
