# -*- coding: utf-8 -*-
"""
tests/test_openai_runtime_tdd.py — Suite TDD per la verifica del Runtime OpenAI-Compatible.

Verifica:
1. Endpoint `/v1/models` (lista modelli compatibili).
2. Chat Statica solo Testo (`modalities: ["text"]`) con e senza Tool/Function Calling.
3. Chat con Sintesi Vocale (`modalities: ["text", "audio"]`, estrazione audio base64).
4. Streaming SSE (`stream=True`) per testo e delta tool calls.
5. Condizionamento Vocale Qwen3-TTS (Voice cloning con ref_audio e Voice prompt).
6. Protocollo WebSocket Full Duplex (`/v1/realtime?duplex=1`) con streaming parallelo Testo (transcript delta) + Audio (audio delta) e supporto Tool.
"""

import base64
import json
import unittest
import numpy as np
from fastapi.testclient import TestClient

from server.openai_omni_server import app
from server.tool_parser import extract_tool_calls_from_text, format_tools_system_prompt
from server.voice_manager import VoiceProfileManager
from server.duplex_session import DuplexSessionManager


class TestOpenAIRuntimeTDD(unittest.TestCase):
    """Test unitari e di conformità per il runtime OpenAI di Gemma-4-Omni."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_tool_calling_schema_and_parser(self):
        """Test 1: Verifica che il parser dei Tool interpreti formato compatto, markdown e JSON annidati."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Ottiene le previsioni meteo attuali.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "details": {"type": "object", "properties": {"hourly": {"type": "boolean"}}},
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        # 1. Prompt formatting
        system_prompt = format_tools_system_prompt(tools)
        self.assertIn("get_current_weather", system_prompt)
        self.assertIn("JSON Schema", system_prompt)

        # 2. Parsing formato compatto con JSON annidato
        output_nested = 'call:get_current_weather{"location": "Milano", "details": {"hourly": true}}'
        parsed_nested = extract_tool_calls_from_text(output_nested, tools)
        self.assertIsNotNone(parsed_nested)
        self.assertEqual(len(parsed_nested), 1)
        self.assertEqual(parsed_nested[0]["function"]["name"], "get_current_weather")
        args = json.loads(parsed_nested[0]["function"]["arguments"])
        self.assertEqual(args["location"], "Milano")
        self.assertEqual(args["details"]["hourly"], True)

        # 3. Parsing formato Markdown JSON block
        output_block = '```json\n{"name": "get_current_weather", "arguments": {"location": "Roma"}}\n```'
        parsed_block = extract_tool_calls_from_text(output_block, tools)
        self.assertIsNotNone(parsed_block)
        self.assertEqual(len(parsed_block), 1)
        self.assertEqual(parsed_block[0]["function"]["name"], "get_current_weather")

    def test_02_voice_conditioning_config(self):
        """Test 2: Verifica preset, VoiceDesign, clonazione zero-shot e gestione input nulli."""
        mgr = VoiceProfileManager()

        # Preset standard
        preset = mgr.get_voice("Vivian")
        self.assertEqual(preset.name, "Vivian")
        self.assertEqual(preset.variant, "CustomVoice")

        # Gestione None/Vuoto di sicurezza
        fallback = mgr.get_voice(None)
        self.assertEqual(fallback.name, "Vivian")

        # VoiceDesign
        design = mgr.get_voice("Una conduttrice radiofonica con voce vivace e tono brillante")
        self.assertEqual(design.variant, "VoiceDesign")

        # Prompt formatting
        prompt = mgr.format_conditioning_prompt("Benvenuti all'ascolto.", design)
        self.assertIn("<|instruct|>", prompt)

        # Clonazione zero-shot
        dummy_pcm = np.zeros(24000 * 3, dtype=np.float32)
        clone = mgr.create_clone_profile("speaker_x", dummy_pcm, ref_text="Frase guida.")
        self.assertEqual(clone.variant, "Base")
        self.assertEqual(len(clone.speaker_embedding), 512)

    def test_03_parallel_text_and_audio_streaming_session(self):
        """Test 3: Verifica sessione Duplex, accumulo cronologia e coerenza transcript."""
        manager = DuplexSessionManager()
        session = manager.create_session("session_123", voice="Ryan")

        session.add_conversation_item({"role": "user", "content": "Ciao Gemma!"})
        self.assertEqual(session.get_last_user_text(), "Ciao Gemma!")

        events = []
        session.start_response(lambda ev: events.append(ev))
        session.emit_text_delta("Buon", lambda ev: events.append(ev))
        session.emit_audio_chunk(b"\x00" * 480, lambda ev: events.append(ev))
        session.emit_text_delta("giorno!", lambda ev: events.append(ev))
        session.finish_response(lambda ev: events.append(ev))

        types = [e["type"] for e in events]
        self.assertIn("response.created", types)
        self.assertIn("response.audio_transcript.delta", types)
        self.assertIn("response.audio.delta", types)
        self.assertIn("response.audio_transcript.done", types)
        self.assertIn("response.done", types)

        full_tx = next(e["transcript"] for e in events if e["type"] == "response.audio_transcript.done")
        self.assertEqual(full_tx, "Buongiorno!")

    def test_04_models_catalog_and_health(self):
        """Test 4: Verifica conformità degli endpoint /health e /v1/models."""
        # Health check
        h_resp = self.client.get("/health")
        self.assertEqual(h_resp.status_code, 200)
        h_data = h_resp.json()
        self.assertEqual(h_data["status"], "healthy")
        self.assertEqual(h_data["sample_rate"], 24000)

        # Models catalog
        m_resp = self.client.get("/v1/models")
        self.assertEqual(m_resp.status_code, 200)
        m_data = m_resp.json()
        self.assertEqual(m_data["object"], "list")
        self.assertTrue(any(m["id"] == "gemma-4-omni-duplex" for m in m_data["data"]))

    def test_05_chat_completions_static_and_tools(self):
        """Test 5: Verifica chat completions testuale standard e ciclo di function calling a 2 turni."""
        # 1. Chat statica standard
        payload_static = {
            "model": "gemma-4-omni-duplex",
            "messages": [{"role": "user", "content": "Raccontami una barzelletta."}],
        }
        res_static = self.client.post("/v1/chat/completions", json=payload_static)
        self.assertEqual(res_static.status_code, 200)
        data_s = res_static.json()
        self.assertEqual(data_s["object"], "chat.completion")
        self.assertEqual(data_s["choices"][0]["finish_reason"], "stop")
        self.assertTrue(len(data_s["choices"][0]["message"]["content"]) > 0)

        # 2. Function calling (Turno 1: Richiesta tool)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Meteo attuale",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ]
        payload_tool_req = {
            "model": "gemma-4-omni-duplex",
            "messages": [{"role": "user", "content": "Che tempo fa oggi a Milano?"}],
            "tools": tools,
        }
        res_tool = self.client.post("/v1/chat/completions", json=payload_tool_req)
        self.assertEqual(res_tool.status_code, 200)
        data_t = res_tool.json()
        self.assertEqual(data_t["choices"][0]["finish_reason"], "tool_calls")
        tc = data_t["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(tc["function"]["name"], "get_current_weather")

        # 3. Function calling (Turno 2: Invio risultato tool)
        messages_t2 = [
            {"role": "user", "content": "Che tempo fa oggi a Milano?"},
            data_t["choices"][0]["message"],
            {
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["function"]["name"],
                "content": '{"location": "Milano", "temp": 24.0}',
            },
        ]
        res_t2 = self.client.post("/v1/chat/completions", json={"model": "gemma-4-omni-duplex", "messages": messages_t2})
        self.assertEqual(res_t2.status_code, 200)
        data_t2 = res_t2.json()
        self.assertIn("consultato lo strumento con successo", data_t2["choices"][0]["message"]["content"])

    def test_06_chat_completions_sse_streaming(self):
        """Test 6: Verifica streaming SSE e conformità schema dei delta tool calls con indice."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Meteo attuale",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
                },
            }
        ]
        payload_stream = {
            "model": "gemma-4-omni-duplex",
            "messages": [{"role": "user", "content": "Meteo Roma"}],
            "tools": tools,
            "stream": True,
        }
        resp = self.client.post("/v1/chat/completions", json=payload_stream)
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/event-stream", resp.headers["content-type"])

        lines = resp.text.split("\n")
        data_lines = [l.replace("data: ", "").strip() for l in lines if l.startswith("data: ")]
        self.assertIn("[DONE]", data_lines)

        # Verifica che il chunk di tool calls contenga il campo "index" obbligatorio per OpenAI SDK
        tool_call_found = False
        for dl in data_lines:
            if dl == "[DONE]":
                continue
            chunk = json.loads(dl)
            delta = chunk["choices"][0]["delta"]
            if "tool_calls" in delta and delta["tool_calls"]:
                tool_call_found = True
                self.assertIn("index", delta["tool_calls"][0])
                self.assertEqual(delta["tool_calls"][0]["index"], 0)
        self.assertTrue(tool_call_found)

    def test_07_chat_completions_audio_modality(self):
        """Test 7: Verifica output vocale (modalities: ['text', 'audio']) e validità WAV Base64."""
        payload_audio = {
            "model": "gemma-4-omni-duplex",
            "messages": [{"role": "user", "content": "Ciao Gemma!"}],
            "modalities": ["text", "audio"],
            "voice": "Vivian",
        }
        res = self.client.post("/v1/chat/completions", json=payload_audio)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        msg = data["choices"][0]["message"]
        self.assertIn("audio", msg)
        self.assertIn("data", msg["audio"])
        self.assertIn("transcript", msg["audio"])

        # Decodifica e verifica header WAV (RIFF...WAVE)
        wav_bytes = base64.b64decode(msg["audio"]["data"])
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertIn(b"WAVE", wav_bytes[:16])

    def test_08_realtime_websocket_full_duplex_session(self):
        """Test 8: Verifica protocollo WebSocket Full-Duplex, streaming parallelo testo/audio e tool calls."""
        with self.client.websocket_connect("/v1/realtime") as ws:
            # 1. Riceve session.created
            init_ev = ws.receive_json()
            self.assertEqual(init_ev["type"], "session.created")

            # 2. Configura sessione (session.update)
            ws.send_json({
                "type": "session.update",
                "session": {"voice": "Vivian", "tools": [{"type": "function", "function": {"name": "get_device_telemetry"}}]},
            })
            up_ev = ws.receive_json()
            self.assertEqual(up_ev["type"], "session.updated")

            # 3. Aggiunge messaggio utente
            ws.send_json({
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Stato sensori GPU"}],
                },
            })
            item_created_ev = ws.receive_json()
            self.assertEqual(item_created_ev["type"], "conversation.item.created")

            # 4. Richiede risposta (response.create)
            ws.send_json({"type": "response.create"})

            events = []
            while True:
                ev = ws.receive_json()
                events.append(ev)
                if ev.get("type") == "response.done":
                    break

            types = [e["type"] for e in events]
            self.assertEqual(types.count("response.created"), 1)
            self.assertEqual(types.count("response.done"), 1)
            self.assertIn("response.audio_transcript.delta", types)
            self.assertIn("response.audio.delta", types)
            self.assertIn("response.function_call_arguments.done", types)

    def test_09_realtime_websocket_barge_in_cancellation(self):
        """Test 9: Verifica che l'interruzione Barge-In (audio in ingresso) cancelli atomicamente lo stream attivo."""
        with self.client.websocket_connect("/v1/realtime") as ws:
            ws.receive_json()  # session.created

            ws.send_json({
                "type": "conversation.item.create",
                "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Parla a lungo"}]},
            })
            ws.receive_json()  # conversation.item.created
            ws.send_json({"type": "response.create"})

            resp_created = ws.receive_json()
            self.assertEqual(resp_created["type"], "response.created")

            # Simula interruzione utente al microfono
            dummy_pcm_interruption = base64.b64encode(b"\x00" * 2400).decode("ascii")
            ws.send_json({"type": "input_audio_buffer.append", "audio": dummy_pcm_interruption})

            cancelled_received = False
            for _ in range(15):
                ev = ws.receive_json()
                if ev.get("type") == "audio.cancelled":
                    cancelled_received = True
                    self.assertEqual(ev["reason"], "user_barge_in")
                    break

            self.assertTrue(cancelled_received)


if __name__ == "__main__":
    unittest.main()
