#!/usr/bin/env python3
"""Lightweight smoke client for the standalone duplex kernel.

Examples:
  python client/kernel_client.py --mode all --host 127.0.0.1 --port 8092
  python client/kernel_client.py --mode duplex
"""
from __future__ import annotations
import argparse, asyncio, base64, json, struct, sys, urllib.request

MODEL = "gemma-4-omni-duplex"

def post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as r: return json.load(r)

def chat(base, audio=False):
    body = {"model": MODEL, "messages":[{"role":"user","content":"Rispondi in una frase breve: saluta il test."}]}
    if audio: body["modalities"] = ["text", "audio"]
    out = post(base + "/v1/chat/completions", body)
    msg = out["choices"][0]["message"]
    if audio and msg.get("audio", {}).get("data"):
        with open("kernel_client_output.wav", "wb") as f: f.write(base64.b64decode(msg["audio"]["data"]))
    print(msg.get("content", "")); return out

def stream(base):
    body = {"model":MODEL,"messages":[{"role":"user","content":"Di ciao in due parole."}],"modalities":["text","audio"],"stream":True}
    req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        for line in r:
            if line.startswith(b"data: "): print(line.decode().strip()[6:])

async def duplex(ws_base):
    import websockets
    async with websockets.connect(ws_base + "/v1/realtime?duplex=1", max_size=50_000_000) as ws:
        await ws.send(json.dumps({"type":"session.update","session":{"model":MODEL,"instructions":"Rispondi brevemente in italiano."}}))
        await ws.send(json.dumps({"type":"conversation.item.create","item":{"role":"user","content":[{"type":"input_text","text":"Ciao"}]}}))
        await ws.send(json.dumps({"type":"response.create"}))
        while True:
            ev = json.loads(await ws.recv()); print(ev.get("type"))
            if ev.get("type") == "response.done": break

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",type=int,default=8092); ap.add_argument("--mode",choices=["health","text","audio","stream","duplex","all"],default="all"); a=ap.parse_args()
    base=f"http://{a.host}:{a.port}"; ws=f"ws://{a.host}:{a.port}"
    if a.mode in ("health","all"): print(post(base+"/health", {}) if False else urllib.request.urlopen(base+"/health").read().decode())
    if a.mode in ("text","all"): chat(base)
    if a.mode in ("audio","all"): chat(base, True)
    if a.mode in ("stream","all"): stream(base)
    if a.mode in ("duplex","all"): asyncio.run(duplex(ws))

if __name__ == "__main__": main()
