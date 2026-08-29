import asyncio, json, base64, struct, math, time
import websockets

def make_wav_b64(seconds=1.0, freq=440.0, sr=16000):
    samples = [int(0.3*32767*math.sin(2*math.pi*freq*i/sr)) for i in range(int(sr*seconds))]
    data = b''.join(struct.pack('<h', s) for s in samples)
    wav = b'RIFF' + struct.pack('<I', 36+len(data)) + b'WAVEfmt ' + struct.pack('<IHHIIHH',16,1,1,sr,sr*2,2,16) + b'data' + struct.pack('<I', len(data)) + data
    return base64.b64encode(wav).decode()

async def main():
    uri = "ws://172.17.0.1:8091/v1/realtime?duplex=1"
    async with websockets.connect(uri, max_size=100_000_000) as ws:
        await ws.send(json.dumps({"type":"session.update","session":{"model":"gemma-4-omni-duplex","modalities":["text","audio"]}}))
        # item con AUDIO (STT upstream)
        await ws.send(json.dumps({"type":"conversation.item.create","item":{"type":"message","role":"user","content":[{"type":"input_audio","audio":make_wav_b64(),"format":"wav"}]}}))
        await ws.send(json.dumps({"type":"response.create","response":{"modalities":["text","audio"]}}))
        t0=time.time(); text=""; audio_chunks=0; audio_b64=0; done=False
        while time.time()-t0 < 120:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=40))
            except asyncio.TimeoutError:
                print("TIMEOUT"); break
            et = msg.get("type")
            if et == "response.output_text.delta": text += msg.get("delta","")
            if et == "response.audio.delta": audio_chunks += 1; audio_b64 += len(msg.get("delta",""))
            if et == "response.done": done=True; break
        print(f"DONE={done} in {time.time()-t0:.1f}s chunks={audio_chunks} b64={audio_b64}")
        print(f"text: {text[:150]}")
asyncio.run(main())
