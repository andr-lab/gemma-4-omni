import base64, json
import numpy as np
from fastapi.testclient import TestClient
from duplex_kernel.server import create_app

class FakeTokenizer:
    def apply_chat_template(self, messages, **_): return messages[-1]["content"]
    def __call__(self, text, **_): return {"input_ids": np.array([[1]])}

class FakeEngine:
    loaded = True
    tokenizer = FakeTokenizer()
    def think(self, ids, **_): return {"text": "ciao mondo", "hidden": None, "prompt_len": 1}
    def v9_prefix(self, hidden, prompt_len): return None
    def talk(self, text, h_pfx, **_): return np.zeros(2400, dtype=np.float32)

def client(): return TestClient(create_app(FakeEngine()))

def test_models_and_validation():
    c = client()
    assert c.get("/v1/models").json()["data"][0]["id"] == "gemma-4-omni-duplex"
    assert c.post("/v1/chat/completions", json={}).status_code == 422

def test_chat_audio_and_reusable_session():
    c = client()
    r = c.post("/v1/chat/completions", json={"messages":[{"role":"user","content":"ciao"}], "modalities":["text","audio"]})
    assert r.status_code == 200
    body = r.json(); assert body["choices"][0]["message"]["audio"]["data"]
    sid = body["session_id"]
    r2 = c.post("/v1/chat/completions", json={"session_id":sid,"messages":[{"role":"user","content":"ancora"}]})
    assert r2.status_code == 200

def test_sse_emits_text_audio_and_done():
    r = client().post("/v1/chat/completions", json={"messages":[{"role":"user","content":"ciao"}],"modalities":["text","audio"],"stream":True})
    assert "data: [DONE]" in r.text
    assert '"audio"' in r.text

def test_tool_parser_contract():
    from duplex_kernel.tool_parser import extract_tool_calls
    tools = [{"type":"function", "function":{"name":"clock"}}]
    calls = extract_tool_calls('call:clock{"tz":"Europe/Rome"}', tools)
    assert json.loads(calls[0]["function"]["arguments"])["tz"] == "Europe/Rome"
