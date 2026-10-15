# -*- coding: utf-8 -*-
"""
server/tool_parser.py — Parser e Formattatore per Tool / Function Calling compatibile OpenAI.

Formatta gli schemi JSON dei tool nel system prompt e analizza l'output di Gemma 4
per estrarre le `tool_calls` nel formato standard OpenAI:
{
    "id": "call_...",
    "type": "function",
    "function": {
        "name": "nome_funzione",
        "arguments": "{\"arg\": \"val\"}"
    }
}
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def format_tools_system_prompt(tools: List[Dict[str, Any]]) -> str:
    """Formatta le definizioni dei tool nel prompt di sistema per Gemma 4."""
    if not tools:
        return ""

    tools_desc = []
    for t in tools:
        if t.get("type") == "function":
            fn = t.get("function", {})
            tools_desc.append(
                f"- **{fn.get('name')}**: {fn.get('description', '')}\n"
                f"  Parametri JSON Schema: {json.dumps(fn.get('parameters', {}), ensure_ascii=False)}"
            )

    prompt = (
        "\n\n# Disponibilità degli Strumenti (Tool / Function Calling)\n"
        "Hai a disposizione le seguenti funzioni da chiamare quando necessario per rispondere all'utente:\n"
        + "\n".join(tools_desc)
        + "\n\nQuando decidi di chiamare una funzione, formatta la chiamata esattamente come:\n"
        "```tool_call\n"
        '{"name": "nome_funzione", "arguments": {"param1": "val1"}}\n'
        "```\n"
        "Oppure nel formato compatto: call:nome_funzione{\"param1\": \"val1\"}\n"
        "Non aggiungere testo superfluo prima della chiamata se devi eseguire solo l'azione.\n"
    )
    return prompt


def extract_tool_calls_from_text(
    text: str, tools: Optional[List[Dict[str, Any]]] = None
) -> Optional[List[Dict[str, Any]]]:
    """Estrae le chiamate a funzione dal testo generato dal modello, gestendo anche JSON annidati."""
    if not text:
        return None

    valid_tool_names = set()
    if tools:
        for t in tools:
            if t.get("type") == "function":
                valid_tool_names.add(t.get("function", {}).get("name"))

    tool_calls = []
    decoder = json.JSONDecoder()

    # Pattern 1: call:func_name{...}
    pattern_compact = re.compile(r'call:([a-zA-Z0-9_\-]+)\s*(\{)')
    for match in pattern_compact.finditer(text):
        fn_name = match.group(1)
        start_brace_idx = match.start(2)
        if not valid_tool_names or fn_name in valid_tool_names:
            try:
                args_dict, _ = decoder.raw_decode(text[start_brace_idx:])
                if isinstance(args_dict, dict):
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": json.dumps(args_dict, ensure_ascii=False),
                        },
                    })
            except Exception:
                pass

    # Pattern 2: ```tool_call / ```json {"name": "...", "arguments": {...}} ```
    # oppure blocchi JSON isolati con "name" e "arguments"
    pattern_block = re.compile(r'```(?:tool_call|json)?\s*(\{)', re.DOTALL)
    for match in pattern_block.finditer(text):
        start_brace_idx = match.start(1)
        try:
            parsed, _ = decoder.raw_decode(text[start_brace_idx:])
            if isinstance(parsed, dict) and "name" in parsed:
                fn_name = parsed.get("name")
                args = parsed.get("arguments", {})
                if isinstance(args, dict):
                    args_str = json.dumps(args, ensure_ascii=False)
                else:
                    args_str = str(args)
                if fn_name and (not valid_tool_names or fn_name in valid_tool_names):
                    # Evita duplicati se già catturato da call:
                    if not any(tc["function"]["name"] == fn_name and tc["function"]["arguments"] == args_str for tc in tool_calls):
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "type": "function",
                            "function": {
                                "name": fn_name,
                                "arguments": args_str,
                            },
                        })
        except Exception:
            pass

    return tool_calls if tool_calls else None
