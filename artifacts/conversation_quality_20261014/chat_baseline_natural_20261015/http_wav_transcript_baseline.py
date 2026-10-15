"""Replay the exact independent ASR transcripts from the paired WAV fixtures.

The two calls are stateless: each request carries the complete history for its
turn, including the actual first assistant answer, so there is no server-side
session history to duplicate or omit.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
import urllib.request


URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "gemma-4-omni-duplex"
SYSTEM = (
    "Rispondi in italiano naturale e conciso, con al massimo due frasi. "
    "Usa solo le informazioni dette nella conversazione e rispetta i vincoli espliciti."
)
USER_1 = (
    "Mi chiamo Elena. Il mio codice è 7-3-1. Vorrei visitare Torino con 120 euro. "
    "Rispondi in italiano, in non più di due frasi."
)
USER_2 = (
    "Il baggettore è 80 euro. Quali sono i miei nome, codice, città e baggetta attuale?"
)


def call(messages: list[dict[str, str]], turn: int) -> dict:
    payload = {
        "model": MODEL,
        "modalities": ["text"],
        "messages": messages,
        "max_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.9,
    }
    started = time.monotonic()
    request = urllib.request.Request(
        URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read())
    answer = body["choices"][0]["message"].get("content", "")
    return {
        "turn": turn,
        "request": payload,
        "response": body,
        "assistant_text": answer,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def main() -> None:
    first = call(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_1}],
        1,
    )
    second = call(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_1},
            {"role": "assistant", "content": first["assistant_text"]},
            {"role": "user", "content": USER_2},
        ],
        2,
    )
    result = {
        "endpoint": URL,
        "model": MODEL,
        "session_mode": "stateless_each_call_full_history_in_request",
        "system_prompt": SYSTEM,
        "source_transcripts": {
            "clean_initial.wav": USER_1,
            "clean_followup.wav": USER_2,
        },
        "calls": [first, second],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    output_path_value = os.environ.get("CONVERSATION_BASELINE_OUTPUT")
    output_path = (
        Path(output_path_value)
        if output_path_value
        else Path(__file__).with_name("http_wav_transcript_baseline.json")
    )
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "output_path": str(output_path),
        "turns": [
            {"turn": item["turn"], "assistant_text": item["assistant_text"],
             "elapsed_s": item["elapsed_s"]}
            for item in result["calls"]
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
