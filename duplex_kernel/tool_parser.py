"""Small, dependency-free tool-call parser for thinker output."""
from __future__ import annotations
import json
import re
from typing import Any

_CALL = re.compile(r"(?:call:|<tool_call>\s*)?([A-Za-z_][\w.-]*)\s*(\{)")


def extract_tool_calls(text: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {t.get("function", {}).get("name") for t in tools if isinstance(t, dict)}
    out = []
    for match in _CALL.finditer(text or ""):
        name, start = match.group(1), match.start(2)
        if allowed and name not in allowed:
            continue
        depth, quoted, escape = 0, False, False
        end = None
        for i in range(start, len(text)):
            ch = text[i]
            if quoted:
                if escape: escape = False
                elif ch == "\\": escape = True
                elif ch == '"': quoted = False
            elif ch == '"': quoted = True
            elif ch == "{": depth += 1
            elif ch == "}" and (depth := depth - 1) == 0:
                end = i + 1; break
        if end is None:
            continue
        try:
            args = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        out.append({"id": f"call_{len(out)}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return out


def tool_system_prompt(tools: list[dict[str, Any]]) -> str:
    specs = []
    for tool in tools:
        fn = tool.get("function", {})
        specs.append(json.dumps({"name": fn.get("name"), "description": fn.get("description", ""),
                                 "parameters": fn.get("parameters", {})}, ensure_ascii=False))
    return "\nUse a tool when needed. Emit exactly call:name{JSON}.\n" + "\n".join(specs)
