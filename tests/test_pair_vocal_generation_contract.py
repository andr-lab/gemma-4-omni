from __future__ import annotations

import ast
from enum import Enum
from pathlib import Path
from typing import Any


class _Action(str, Enum):
    SPEAK = "SPEAK"
    BACKCHANNEL = "BACKCHANNEL"
    REPLAN = "REPLAN"


def _production_generation_options():
    source_path = Path(__file__).resolve().parents[1] / "duplex_kernel" / "server.py"
    source = ast.parse(source_path.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_controller_guided_say_generation_options"
    )
    isolated_module = ast.fix_missing_locations(ast.Module(body=[helper], type_ignores=[]))
    namespace = {
        "ControlAction": _Action,
        "Any": Any,
        "AUDIO_RESPONSE_PROTOCOL_V2": "audio_response_v2",
        "AUDIO_RESPONSE_V2_MAX_NEW_TOKENS": 128,
    }
    exec(compile(isolated_module, str(source_path), "exec"), namespace)
    return namespace["_controller_guided_say_generation_options"]


def test_controller_guided_vocal_actions_reserve_a_non_eos_token_for_enum_and_string_inputs() -> None:
    options_for = _production_generation_options()
    for action in (_Action.SPEAK, _Action.BACKCHANNEL, " speak ", " backchannel "):
        options = options_for(
            action=action,
            controller_guided_say=True,
            direct_audio_candidate=True,
            response_protocol="controller_guided_say_v1",
        )

        assert options == {"max_tokens": 256, "_min_new_tokens": 1}


def test_non_vocal_and_v2_routes_keep_their_existing_generation_floor() -> None:
    options_for = _production_generation_options()
    assert options_for(
        action=_Action.REPLAN,
        controller_guided_say=True,
        direct_audio_candidate=True,
        response_protocol="controller_guided_say_v1",
    ) == {"max_tokens": 64, "_min_new_tokens": 0}
    assert options_for(
        action=_Action.SPEAK,
        controller_guided_say=False,
        direct_audio_candidate=True,
        response_protocol="audio_response_v2",
    ) == {"max_tokens": 128, "_min_new_tokens": 0}
