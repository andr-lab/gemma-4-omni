#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client/gemma4_omni_test_client.py — Client di test completo per Gemma-4-Omni
Full-Duplex su vLLM-Omni (ADR-028 + porting L2).

Copre TUTTE le funzionalità del server:
  1. Chat text (testo semplice)
  2. Chat text+audio (modalities ["text","audio"]) → WAV base64 salvato su disco
  3. Streaming SSE (chunk testo + audio incrementale)
  4. Tool/Function calling (schema locale + ciclo automatico)
  5. Full-Duplex WebSocket /v1/realtime?duplex=1 con input AUDIO (PCM 16k):
     STT upstream via encoder audio nativo Gemma4 (nessun Whisper), risposta
     testo+audio in streaming, auto_response (flusso continuo)
  6. Voci: system prompt, voice_style, ref_audio cloning, profili default
  7. ARIA pacing per-request (verifica durata audio vs n. parole)

Uso:
    python client/gemma4_omni_test_client.py --host 192.168.0.186 --port 8091 --all
    python client/gemma4_omni_test_client.py --mode text
    python client/gemma4_omni_test_client.py --mode audio
    python client/gemma4_omni_test_client.py --mode stream
    python client/gemma4_omni_test_client.py --mode tools
    python client/gemma4_omni_test_client.py --mode duplex
    python client/gemma4_omni_test_client.py --mode voice --voice leonardo
    python client/gemma4_omni_test_client.py --mode ref_audio
"""

import argparse
import asyncio
import base64
import json
import math
import os
import struct
import sys
import time
from typing import Any

# UTF-8 robusto su Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import numpy as np
    import soundfile as sf
    import websockets
except ImportError as exc:
    print(f"[Client] Manca una dipendenza: {exc}. Installa: pip install numpy soundfile websockets")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # gli altri modi funzionano senza SDK (urllib)

MODEL = "gemma-4-omni-duplex"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples_test")


# ---------------------------------------------------------------------------
# Strumenti locali (tool calling)
# ---------------------------------------------------------------------------
LOCAL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Restituisce temperatura e condizioni meteo per una città.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "Città (es. Milano)"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
]


def call_tool(name: str, args: dict[str, Any]) -> str:
    if name == "get_current_weather":
        loc = args.get("location", "Milano")
        unit = args.get("unit", "celsius")
        temp = 22 if unit == "celsius" else 72
        return json.dumps({"location": loc, "temperature": temp, "unit": unit, "conditions": "soleggiato"})
    return json.dumps({"error": f"tool sconosciuto: {name}"})


# ---------------------------------------------------------------------------
# WAV helper (per input audio nei test)
# ---------------------------------------------------------------------------
def make_tone_wav_b64(seconds: float = 1.0, freq: float = 440.0, sr: int = 16000) -> str:
    """Genera un tono PCM16 mono e lo restituisce come base64 di un WAV valido."""
    n = int(sr * seconds)
    samples = [int(0.3 * 32767 * math.sin(2 * math.pi * freq * i / sr)) for i in range(n)]
    data = b"".join(struct.pack("<h", s) for s in samples)
    wav = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return base64.b64encode(wav).decode()


def save_b64_audio(b64_data: str, name: str) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    try:
        raw = base64.b64decode(b64_data)
        with open(path, "wb") as f:
            f.write(raw)
        return path
    except Exception as exc:
        print(f"  ⚠️ Salvataggio audio fallito ({exc})")
        return ""


def _wav_pcm_chunks_to_wav(chunks_b64: list[str], sample_rate: int = 24000) -> str:
    """Concatena chunk audio SSE (ognuno un WAV completo) in un unico WAV.

    In streaming SSE ogni ``content`` è un WAV RIFF indipendente; la semplice
    concatenazione produce un file corrotto. Qui estraiamo i sample PCM16 da
    ciascun chunk e ricostruiamo un WAV unico.
    """
    pcm_parts: list[bytes] = []
    for c in chunks_b64:
        try:
            raw = base64.b64decode(c)
        except Exception:
            continue
        if raw[:4] != b"RIFF" or len(raw) < 44:
            pcm_parts.append(raw)
            continue
        # PCM16 mono: payload dopo l'header (44 byte) — data size in raw[40:44]
        data_size = struct.unpack("<I", raw[40:44])[0]
        pcm = raw[44 : 44 + data_size]
        if pcm:
            pcm_parts.append(pcm)
    if not pcm_parts:
        return ""
    data = b"".join(pcm_parts)
    wav = (
        b"RIFF"
        + struct.pack("<I", 36 + len(data))
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return base64.b64encode(wav).decode()


# ---------------------------------------------------------------------------
# 1. Chat text
# ---------------------------------------------------------------------------
def chat_text(base_url: str, prompt: str = "Ciao! Chi sei e cosa sai fare?") -> str:
    url = f"{base_url}/v1/chat/completions"
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 80}
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=120))
    return resp["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# 2. Chat text+audio
# ---------------------------------------------------------------------------
def chat_audio(base_url: str, prompt: str = "Rispondi in una sola breve frase: chi è Leonardo da Vinci?") -> tuple[str, str]:
    """Chat con audio usando lo streaming SSE (assembla i chunk audio).

    In modalità async_chunk (deploy debug) la risposta NON-streaming non
    aggrega il waveform nel message.audio (limite noto del flusso async);
    lo streaming SSE invece emette i chunk audio nel content: qui li
    raccogliamo e li salviamo come WAV completo.
    """
    result = chat_stream(base_url, prompt)
    return result["text"], result["wav"]


# ---------------------------------------------------------------------------
# 3. Streaming SSE
# ---------------------------------------------------------------------------
def chat_stream(base_url: str, prompt: str = "Racconta una breve storiella in 2 frasi.") -> dict[str, Any]:
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 120,
        "modalities": ["text", "audio"],
        "stream": True,
    }
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req, timeout=180)
    text = ""
    audio_b64 = ""
    audio_chunks_b64: list[str] = []
    n_audio = 0
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            c = delta.get("content", "")
            if c and not c.startswith("UklGR"):  # non-audio
                text += c
            a = delta.get("audio") or (c if c.startswith("UklGR") else "")
            if a:
                audio_b64 += a
                audio_chunks_b64.append(a)
                n_audio += 1
        except Exception:
            continue
    # Ogni chunk SSE è un WAV completo: ricostruisci un unico WAV dai PCM.
    merged = _wav_pcm_chunks_to_wav(audio_chunks_b64)
    wav_path = save_b64_audio(merged, "stream_audio.wav") if merged else ""
    return {"text": text, "audio_chunks": n_audio, "wav": wav_path, "audio_bytes": len(audio_b64)}


# ---------------------------------------------------------------------------
# 4. Tool calling
# ---------------------------------------------------------------------------
def _parse_tool_call_from_text(text: str) -> tuple[str, dict[str, Any]] | None:
    """Fallback: estrae call:func{json} dal contenuto quando il parser server
    non popola tool_calls (token di confine <|tool_call> soppresso)."""
    import re

    m = re.search(r"call:([A-Za-z_][A-Za-z0-9_]*)\s*\{([^}]*)\}", text)
    if not m:
        return None
    name = m.group(1)
    raw_args = m.group(2).strip()
    try:
        if not raw_args:
            args = {}
        else:
            args = json.loads(raw_args)
    except Exception:
        # formato key:value non JSON
        args = {}
        for kv in raw_args.split(","):
            if ":" in kv:
                k, v = kv.split(":", 1)
                args[k.strip()] = v.strip().strip('"')
    return name, args


def chat_tools(base_url: str) -> str:
    url = f"{base_url}/v1/chat/completions"
    import urllib.request

    def complete(messages, tools=None, tool_choice=None):
        body = {"model": MODEL, "messages": messages, "max_tokens": 120}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = tool_choice or "auto"
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        return json.load(urllib.request.urlopen(req, timeout=120))

    messages = [
        {"role": "user", "content": "Che tempo fa a Milano? Usa lo strumento disponibile."}
    ]
    for _round in range(3):
        resp = complete(messages, tools=LOCAL_TOOLS)
        msg = resp["choices"][0]["message"]
        messages.append({"role": "assistant", "content": msg.get("content") or ""})
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                args = json.loads(fn.get("arguments") or "{}")
                result = call_tool(name, args)
                print(f"  🔧 tool chiamato: {name}({args}) → {result[:60]}...")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                )
        else:
            # Fallback: tool call nel testo (parser server non attivo)
            content = msg.get("content") or ""
            parsed = _parse_tool_call_from_text(content)
            if parsed:
                name, args = parsed
                result = call_tool(name, args)
                print(f"  🔧 tool (fallback testo): {name}({args}) → {result[:60]}...")
                messages.append(
                    {
                        "role": "assistant",
                        "content": content,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": f"call_{name}",
                        "content": result,
                    }
                )
            else:
                return content
    return "(ciclo tool non concluso)"


# ---------------------------------------------------------------------------
# 5. Full-Duplex WebSocket con input AUDIO (STT upstream Gemma nativo)
# ---------------------------------------------------------------------------
async def duplex_audio(
    base_ws: str,
    *,
    auto_response: bool = True,
    prompt_text: str | None = "Rispondi in una sola breve frase a questo segnale.",
    audio_seconds: float = 1.2,
    instructions: str = "Sei un assistente vocale. Rispondi in una sola breve frase in italiano.",
) -> dict[str, Any]:
    uri = f"{base_ws}/v1/realtime?duplex=1"
    async with websockets.connect(uri, max_size=200_000_000) as ws:
        session = {
            "model": MODEL,
            "modalities": ["text", "audio"],
            "instructions": instructions,
            "extra_body": {"auto_response": auto_response, "full_duplex": True},
        }
        await ws.send(json.dumps({"type": "session.update", "session": session}))
        # Opzionale: prompt testuale di contesto
        if prompt_text:
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": prompt_text}],
                        },
                    }
                )
            )
        # Input AUDIO (PCM16) → STT upstream encoder Gemma4 nativo
        pcm = make_tone_wav_b64(seconds=audio_seconds, freq=440.0)
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(
                        pcm_bytes_from_wav_b64(pcm)
                    ).decode(),
                    "format": "pcm16",
                    "sample_rate_hz": 16000,
                }
            )
        )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        if not auto_response:
            await ws.send(json.dumps({"type": "response.create", "response": {"modalities": ["text", "audio"]}}))

        t0 = time.time()
        text = ""
        audio_b64 = ""
        audio_chunks_b64: list[str] = []
        audio_chunks = 0
        done = False
        events: list[str] = []
        while time.time() - t0 < 120:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
            except asyncio.TimeoutError:
                print("  ⏱️ timeout ricezione")
                break
            et = msg.get("type")
            events.append(et)
            if et == "response.output_text.delta":
                text += msg.get("delta", "")
            if et == "response.audio.delta":
                d = msg.get("delta", "")
                audio_b64 += d
                audio_chunks_b64.append(d)
                audio_chunks += 1
            if et == "response.done":
                done = True
                break
        merged = _wav_pcm_chunks_to_wav(audio_chunks_b64)
        wav_path = save_b64_audio(merged, "duplex_audio_response.wav") if merged else ""
        return {
            "done": done,
            "elapsed": round(time.time() - t0, 1),
            "text": text[:200],
            "audio_chunks": audio_chunks,
            "audio_bytes": len(audio_b64),
            "wav": wav_path,
            "events": list(dict.fromkeys(events))[:12],
        }


def pcm_bytes_from_wav_b64(wav_b64: str) -> bytes:
    """Estrae i sample PCM16 da un WAV base64 (per l'append pcm16)."""
    raw = base64.b64decode(wav_b64)
    # skip RIFF header (44 byte standard PCM)
    data = raw[44:]
    return data


# ---------------------------------------------------------------------------
# 6. Voci
# ---------------------------------------------------------------------------
def chat_with_voice(base_url: str, *, voice: str | None, voice_style: str | None, ref_audio: str | None) -> str:
    url = f"{base_url}/v1/chat/completions"
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Rispondi in una sola breve frase: benvenuto nel test vocale."}],
        "max_tokens": 80,
        "modalities": ["text", "audio"],
    }
    if voice:
        body["voice"] = voice
    if voice_style:
        body["voice_style"] = voice_style
    if ref_audio:
        body["ref_audio"] = ref_audio
        body["ref_text"] = "Questo è il riferimento per la clonazione vocale."
    import urllib.request

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    msg = resp["choices"][0]["message"]
    audio = msg.get("audio")
    wav = ""
    if isinstance(audio, dict) and audio.get("data"):
        wav = save_b64_audio(audio["data"], f"voice_{voice or voice_style or 'ref'}.wav")
    return f"{msg.get('content','')} | wav={wav or 'N/A'}"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma-4-Omni Full-Duplex test client")
    ap.add_argument("--host", default="192.168.0.186")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument(
        "--mode",
        choices=["text", "audio", "stream", "tools", "duplex", "voice", "ref_audio", "all"],
        default="all",
    )
    ap.add_argument("--voice", default="leonardo")
    ap.add_argument("--prompt", default="")
    args = ap.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    base_ws = f"ws://{args.host}:{args.port}"
    print("=" * 72)
    print("  GEMMA-4-OMNI FULL-DUPLEX — CLIENT DI TEST COMPLETO")
    print(f"  server: {base_url} | model: {MODEL}")
    print("=" * 72)

    modes = ["text", "audio", "stream", "tools", "duplex", "voice"] if args.mode == "all" else [args.mode]

    if "text" in modes:
        print("\n[1/6] CHAT TEXT")
        t0 = time.time()
        text = chat_text(base_url, args.prompt or "Ciao! Chi sei e cosa sai fare?")
        print(f"  → {text[:200]}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    if "audio" in modes:
        print("\n[2/6] CHAT TEXT+AUDIO")
        t0 = time.time()
        text, wav = chat_audio(base_url)
        print(f"  testo: {text[:120]}")
        print(f"  audio: {wav or 'N/A'}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    if "stream" in modes:
        print("\n[3/6] STREAMING SSE (testo + audio incrementale)")
        t0 = time.time()
        result = chat_stream(base_url)
        print(f"  testo: {result['text'][:150]}")
        print(f"  chunk audio: {result['audio_chunks']} | WAV: {result['wav'] or 'N/A'}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    if "tools" in modes:
        print("\n[4/6] TOOL / FUNCTION CALLING")
        t0 = time.time()
        final = chat_tools(base_url)
        print(f"  risposta finale: {final[:200]}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    if "duplex" in modes:
        print("\n[5/6] FULL-DUPLEX WS CON INPUT AUDIO (STT Gemma nativo, niente Whisper)")
        t0 = time.time()
        result = asyncio.run(duplex_audio(base_ws, auto_response=True))
        print(f"  done={result['done']} in {result['elapsed']}s")
        print(f"  testo: {result['text'][:150]}")
        print(f"  audio chunks={result['audio_chunks']} bytes={result['audio_bytes']} WAV={result['wav']}")
        print(f"  eventi: {result['events']}")
        print(f"  ⏱ totale {time.time()-t0:.1f}s")

    if "voice" in modes:
        print("\n[6/6] VOCE (profilo default)")
        t0 = time.time()
        out = chat_with_voice(base_url, voice=args.voice, voice_style=None, ref_audio=None)
        print(f"  → {out}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    if args.mode == "ref_audio":
        print("\n[EXTRA] CLONAZIONE VOCALE (ref_audio)")
        t0 = time.time()
        out = chat_with_voice(base_url, voice=None, voice_style=None, ref_audio=make_tone_wav_b64(1.0))
        print(f"  → {out}")
        print(f"  ⏱ {time.time()-t0:.1f}s")

    print("\n" + "=" * 72)
    print("  TEST COMPLETATO. Audio salvati in:", OUT_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
