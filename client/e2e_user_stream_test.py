#!/usr/bin/env python3
"""Stream del canale utente sintetico al kernel e raccolta risposta audio."""
from __future__ import annotations
import argparse, asyncio, base64, io, json, threading, time
import numpy as np
import soundfile as sf


def metrics(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
    return {"sample_rate": sr, "samples": int(x.size), "duration_s": round(x.size / sr, 3),
            "peak": round(peak, 6), "rms": round(rms, 6),
            "crest_factor": round(peak / max(rms, 1e-12), 4),
            "clipping_pct": round(float(np.mean(np.abs(x) >= 0.999)) * 100, 5) if x.size else 0.0,
            "finite": bool(np.isfinite(x).all())}


async def stream(host: str, port: int, pcm: np.ndarray, *, chunk_ms: int = 640) -> tuple[list[np.ndarray], list[dict]]:
    import aiohttp
    chunks, events = [], []
    async with aiohttp.ClientSession() as http:
      async with http.ws_connect(f"ws://{host}:{port}/v1/realtime?duplex=1", max_msg_size=50_000_000) as ws:
        await ws.receive()  # session.created
        await ws.send_str(json.dumps({"type": "session.update", "session": {"instructions": "Rispondi in italiano, brevemente."}}))
        await ws.receive()
        n = int(16000 * chunk_ms / 1000)
        for start in range(0, len(pcm), n):
            raw = (np.clip(pcm[start:start+n], -1, 1) * 32767).astype("<i2").tobytes()
            await ws.send_str(json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode()}))
        await ws.send_str(json.dumps({"type": "input_audio_buffer.commit"}))
        while True:
            msg = await ws.receive(); ev = json.loads(msg.data); events.append(ev)
            if ev.get("type") == "response.audio.delta":
                data, _ = sf.read(io.BytesIO(base64.b64decode(ev["delta"])), dtype="float32")
                chunks.append(np.asarray(data))
            if ev.get("type") in {"response.done", "error"}: break
    return chunks, events


def run_asr(pcm: np.ndarray) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")
    segments, _ = model.transcribe(pcm, language="it", beam_size=5, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("wav"); ap.add_argument("--channel", type=int, default=0)
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--output", default="e2e_user_stream_response.wav"); ap.add_argument("--asr", action="store_true")
    a = ap.parse_args()
    x, sr = sf.read(a.wav, always_2d=True, dtype="float32")
    pcm = x[:, a.channel]
    if sr != 16000:
        from scipy.signal import resample_poly
        pcm = resample_poly(pcm, 16000, sr).astype("float32"); sr = 16000
    # Gemma4AudioFeatureExtractor ha limite nativo di 30s per chiamata.
    pcm = pcm[:16000 * 30]
    chunks, events = asyncio.run(stream(a.host, a.port, pcm))
    response = np.concatenate(chunks) if chunks else np.empty(0, dtype="float32")
    if response.size: sf.write(a.output, response, 24000)
    result = {"input": metrics(pcm, sr), "response": metrics(response, 24000),
              "event_types": [e.get("type") for e in events], "audio_chunks": len(chunks), "output": a.output}
    if a.asr: result["asr_input_it"] = run_asr(pcm)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
