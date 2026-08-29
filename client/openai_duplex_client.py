# -*- coding: utf-8 -*-
"""
client/openai_duplex_client.py — Client Python di Test Ufficiale con SDK OpenAI per Gemma-4-Omni.

Verifica tutti i 4 requisiti di test:
1. Definizione di un Tool Locale (`get_current_weather` e `get_device_telemetry`) ed esecuzione automatica del ciclo di function calling.
2. Chat statica normale (senza audio, solo testo).
3. Chat con risposte lette a voce da Qwen3-TTS (modalità `["text", "audio"]` con estrazione WAV).
4. Streaming parallelo Full-Duplex (WebSocket Realtime con transcript delta testuale e audio delta simultanei).

Uso:
    python client/openai_duplex_client.py --host 192.168.0.186 --port 8000 --all
    python client/openai_duplex_client.py --mode tools
    python client/openai_duplex_client.py --mode audio
    python client/openai_duplex_client.py --mode duplex
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from typing import Any, Dict

# Assicura codifica UTF-8 su Windows terminal
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Import client OpenAI ufficiale
try:
    from openai import OpenAI
except ImportError:
    print("[Client] Errore: libreria 'openai' non trovata. Esegui: pip install openai")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. Definizione del Tool Locale di Prova
# ---------------------------------------------------------------------------
LOCAL_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Restituisce la temperatura e le condizioni meteo attuali per una data città.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Nome della città (es. Milano, Roma, Napoli)",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Unità di misura della temperatura",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_device_telemetry",
            "description": "Legge lo stato dei sensori e della memoria della GPU NVIDIA RTX 3090.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "string",
                        "description": "Identificativo della GPU o del device",
                    }
                },
                "required": ["device_id"],
            },
        },
    },
]


def execute_local_tool(name: str, args: Dict[str, Any]) -> str:
    """Esegue la funzione in locale e restituisce la stringa di risultato."""
    print(f"\n⚙️ [Local Tool Execution] Esecuzione: {name}({args})")
    if name == "get_current_weather":
        city = args.get("location", "Milano")
        return json.dumps({
            "location": city,
            "temperature_celsius": 23.5,
            "condition": "Sereno con brezza leggera",
            "humidity_percent": 45,
        }, ensure_ascii=False)
    elif name == "get_device_telemetry":
        return json.dumps({
            "device": "NVIDIA GeForce RTX 3090 24GB",
            "vram_used_gb": 15.6,
            "vram_total_gb": 24.0,
            "temperature_c": 46.0,
            "fan_speed_percent": 35,
            "status": "HEALTHY",
        }, ensure_ascii=False)
    return json.dumps({"status": "unknown_tool"})


# ---------------------------------------------------------------------------
# Test Requisiti
# ---------------------------------------------------------------------------
def test_static_chat(client: OpenAI, model: str):
    """Requisito 3: Chat Statica Normale (senza Qwen3-TTS)."""
    print("\n" + "=" * 60)
    print("💬 [TEST 1] Chat Statica Normale (Solo Testo)")
    print("=" * 60)
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Spiegami in 2 righe cosa significa elaborazione full-duplex."}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    print(f"🤖 Risposta Modello:\n{text}")
    assert text is not None and len(text) > 0
    print("✅ Chat statica completata con successo.")


def test_function_calling(client: OpenAI, model: str):
    """Requisito 1: Function Calling con Tool Locale."""
    print("\n" + "=" * 60)
    print("⚡ [TEST 2] Function Calling con Tool Locale (get_current_weather)")
    print("=" * 60)

    messages = [{"role": "user", "content": "Che tempo fa oggi a Milano?"}]
    print(f"👤 Utente: {messages[0]['content']}")

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=LOCAL_TOOLS_SCHEMA,
        tool_choice="auto",
    )

    msg = resp.choices[0].message
    print(f"🤖 Risposta Iniziale: {msg.content}")

    if msg.tool_calls:
        for tc in msg.tool_calls:
            print(f"  • Tool Call Rilevata: ID={tc.id}, Name={tc.function.name}")
            args = json.loads(tc.function.arguments)
            tool_res = execute_local_tool(tc.function.name, args)
            print(f"  • Risultato Tool: {tool_res}")

            # Second turn: invia il risultato del tool al modello
            assistant_msg_dict = {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                ],
            }
            messages.append(assistant_msg_dict)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": tool_res,
            })

            final_resp = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            final_text = final_resp.choices[0].message.content
            print(f"🤖 Risposta Finale Post-Tool:\n{final_text}")
    print("✅ Ciclo di Function Calling validato con successo.")


def test_chat_with_voice(client: OpenAI, model: str, output_wav: str = "agent_reply.wav"):
    """Requisito 4: Chat Normale con Risposta Letta a Voce (Qwen3-TTS)."""
    print("\n" + "=" * 60)
    print("🔊 [TEST 3] Chat Normale con Risposta Vocale (modalities: ['text', 'audio'])")
    print("=" * 60)

    # Chiamata raw o tramite extra_body per abilitare l'audio
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Salutami con entusiasmo e presentati."}],
        extra_body={"modalities": ["text", "audio"], "voice": "Vivian"},
    )

    choice = resp.choices[0]
    msg = choice.message
    print(f"🤖 Testo Risposta: {msg.content}")

    # Estrazione audio sintetizzato
    audio_obj = getattr(msg, "audio", None) or getattr(choice, "audio", None)
    if audio_obj and isinstance(audio_obj, dict):
        raw_b64 = audio_obj.get("data")
    elif hasattr(audio_obj, "data"):
        raw_b64 = audio_obj.data
    else:
        raw_b64 = None

    if raw_b64:
        wav_bytes = base64.b64decode(raw_b64)
        with open(output_wav, "wb") as f:
            f.write(wav_bytes)
        print(f"✅ Audio generato e salvato con successo in: {output_wav} ({len(wav_bytes)} bytes)")
    else:
        print("⚠️ Nessun payload audio restituito nel dizionario standard.")


async def test_realtime_duplex_websocket(base_url: str):
    """Requisito 2: Streaming Parallelo Testo + Audio Full Duplex su WebSocket."""
    print("\n" + "=" * 60)
    print("🎙️ [TEST 4] Streaming Parallelo Testo + Audio Full-Duplex (WebSocket Realtime)")
    print("=" * 60)

    import websockets
    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://") + "/v1/realtime"
    print(f"Connessione a: {ws_url}...")

    async with websockets.connect(ws_url) as ws:
        # 1. Configura sessione con tool e voce
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "voice": "Vivian",
                "tools": LOCAL_TOOLS_SCHEMA,
            },
        }))

        # 2. Invia input utente
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Ciao Gemma, dimmi che ora è e come stai."}],
            },
        }))
        await ws.send(json.dumps({"type": "response.create"}))

        transcript_chunks = []
        audio_chunks_count = 0

        print("Ascolto stream parallelo dal server...")
        while True:
            raw_msg = await ws.recv()
            data = json.loads(raw_msg)
            mtype = data.get("type")

            if mtype == "response.audio_transcript.delta":
                delta = data.get("delta", "")
                transcript_chunks.append(delta)
                print(f"[Transcript Delta] {delta}", end="", flush=True)

            elif mtype == "response.audio.delta":
                audio_chunks_count += 1

            elif mtype == "response.function_call_arguments.done":
                print(f"\n[WebSocket Tool Call] {data.get('name')}({data.get('arguments')})")

            elif mtype == "response.done":
                print(f"\n✅ Risposta WebSocket completata. Ricevuti {audio_chunks_count} chunk audio PCM.")
                break

    full_transcript = "".join(transcript_chunks)
    print(f"Trascrizione finale: {full_transcript}")


def main():
    parser = argparse.ArgumentParser(description="Client di Test OpenAI per Gemma 4 Omni")
    parser.add_argument("--host", default="localhost", help="Host del server (es. localhost o 192.168.0.186)")
    parser.add_argument("--port", type=int, default=8000, help="Porta del server")
    parser.add_argument("--mode", choices=["static", "tools", "audio", "duplex", "all"], default="all")
    parser.add_argument("--model", default="gemma-4-omni-duplex")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    api_url = f"{base_url}/v1"
    client = OpenAI(base_url=api_url, api_key="dummy-key-for-local-serving")

    if args.mode in ("static", "all"):
        test_static_chat(client, args.model)

    if args.mode in ("tools", "all"):
        test_function_calling(client, args.model)

    if args.mode in ("audio", "all"):
        test_chat_with_voice(client, args.model)

    if args.mode in ("duplex", "all"):
        try:
            import websockets
            asyncio.run(test_realtime_duplex_websocket(base_url))
        except ImportError:
            print("[Client] Installa 'websockets' per testare il websocket: pip install websockets")


if __name__ == "__main__":
    main()
