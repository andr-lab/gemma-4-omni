# -*- coding: utf-8 -*-
"""
serve_duplex.py — Server di inferenza per Gemma 4 Full-Duplex (coupling Qwen3-TTS, ADR-005)

Carica il sistema addestrato (LoRA + teste custom + sub-talker congelato + decoder Code2Wav)
e lo serve come: 1) demo CLI, 2) API FastAPI (REST), 3) app Gradio (opzionale).

⚠️ Eseguire su una MACCHINA CON GPU fuori Kaggle (kernel Kaggle è effimero). Requisiti:
    pip install unsloth transformers accelerate bitsandbytes fastapi uvicorn python-multipart \
                soundfile numpy pandas torch safetensors
    export HF_TOKEN=<token HF>   # repo gated: google/gemma-4-E2B-it, Qwen3-TTS
    export KAGGLE_BIG_DISK=/tmp  # opzionale, usato solo da Fase 1

Uso:
    1) Risposta singola a un wav utente:
       python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota \
           --demo samples/user_input.wav -o samples/agent_response.wav

    2) Server REST (Streamlit-free):
       python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota --serve --port 8000
       curl -F "file=@parlato.wav" http://localhost:8000/talk -o risposta.wav
       curl http://localhost:8000/health

    3) App Gradio (opzionale se `gradio` è installato):
       python serve_duplex.py --checkpoint checkpoints/gemma4_duplex_sota --gradio

Il coupling Qwen3-TTS si attiva automaticamente se la dir del sub-talker esiste
(come in Fase 2); per forzarlo/evitarlo: --coupling 1 / --coupling 0.

Per il vero serving in produzione ad alta concorrenza (vLLM-Omni) vedi README:
il porting richiede la registrazione del modello multi-stadio (Thinker = Gemma4 +
heads custom, Talker = sub-talker, Code2Wav già presente in vLLM-Omni).
"""
import argparse
import base64
import io
import os
import sys
import tempfile

import numpy as np

# La pipeline (costanti, loader, demo) — pensata anche per l'import esterno
import kaggle_pipeline as kp


# ---------------------------------------------------------------------------
# Caricamento
# ---------------------------------------------------------------------------
def load_served_model(checkpoint_dir: str):
    """Carica Gemma 4 (Unsloth) + teste custom + sub-talker + decoder Code2Wav.
    Architettura COUPLING obbligatoria (ADR-005): serve la dir del sub-talker.
    """
    duplex, tokenizer = kp.load_gemma4_duplex_system()
    if duplex is None:
        raise RuntimeError(
            "Caricamento fallito. Verifica GPU, unsloth/transformers>=5.x e HF_TOKEN. "
            "Nota: serve_duplex.py NON va eseguito dentro il kernel Kaggle."
        )
    kp.load_gemma4_duplex_checkpoint(duplex, checkpoint_dir)
    decoder = kp.Code2WavDecoder()
    duplex.eval()
    if kp.DEVICE == "cuda":
        import torch
        torch.cuda.empty_cache()
    print(f"[Serve] Pronto: device={kp.DEVICE} coupling=OBBLIGATORIO (ADR-005) "
          f"checkpoint={checkpoint_dir}")
    return duplex, decoder


# ---------------------------------------------------------------------------
# Inferenza (riusa demo_talk, già verificato: wav -> mel nativo -> generate -> Code2Wav)
# ---------------------------------------------------------------------------
def respond_to_wav(duplex, decoder, wav_bytes: bytes, suffix: str = "req") -> bytes:
    """wav utente (bytes, mono o stereo, 8-48kHz) -> wav risposta agente (bytes, 24kHz)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_bytes)
        tmp_in = f.name
    tmp_out = os.path.join(tempfile.gettempdir(), f"duplex_resp_{suffix}_{os.getpid()}.wav")
    try:
        out = kp.demo_talk(duplex, decoder=decoder, input_wav=tmp_in, output_wav=tmp_out)
        if not out or not os.path.isfile(out):
            raise RuntimeError("La demo non ha prodotto audio.")
        with open(out, "rb") as f:
            return f.read()
    finally:
        for p in (tmp_in, tmp_out):
            try:
                os.remove(p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# App FastAPI (REST) — opzionale, solo se --serve
# ---------------------------------------------------------------------------
def build_app(duplex, decoder):
    try:
        from fastapi import FastAPI, UploadFile, File
        from fastapi.responses import Response, JSONResponse
        import soundfile as sf
    except ImportError:
        print("[Serve] FastAPI non installato: `pip install fastapi uvicorn python-multipart`.")
        sys.exit(2)

    app = FastAPI(title="Gemma 4 Full-Duplex", version="1.0")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "device": kp.DEVICE,
            "coupling": "qwen3-tts-subtalker (ADR-005)",
            "sample_rate": kp.SYNTH_SAMPLE_RATE,
        }

    @app.post("/talk")
    async def talk(file: UploadFile = File(...)):
        """Upload wav utente -> risposta agente wav (audio/wav, 24kHz)."""
        data = await file.read()
        try:
            resp = respond_to_wav(duplex, decoder, data)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        return Response(content=resp, media_type="audio/wav",
                        headers={"Content-Disposition": "attachment; filename=agent_response.wav"})

    @app.post("/talk_json")
    async def talk_json(payload: dict):
        """JSON: {'wav_b64': <base64 wav>, 'sr': <sample_rate attuale>} -> wav risposta b64."""
        try:
            raw = base64.b64decode(payload["wav_b64"])
            resp = respond_to_wav(duplex, decoder, raw)
            return {"wav_b64": base64.b64encode(resp).decode(), "sr": kp.SYNTH_SAMPLE_RATE}
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


# ---------------------------------------------------------------------------
# App Gradio — opzionale
# ---------------------------------------------------------------------------
def build_gradio(duplex, decoder):
    try:
        import gradio as gr
    except ImportError:
        print("[Serve] Gradio non installato: `pip install gradio`.")
        sys.exit(2)

    def fn(wav):  # (sr, array) o None
        if wav is None:
            return None
        import soundfile as sf
        sr, arr = wav
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        buf = io.BytesIO()
        sf.write(buf, arr, int(sr), format="WAV")
        resp = respond_to_wav(duplex, decoder, buf.getvalue())
        sr_out, arr_out = sf.read(io.BytesIO(resp), dtype="float32")
        return (int(sr_out), arr_out)

    demo = gr.Interface(
        fn=fn,
        inputs=gr.Audio(label="Parla tu (microfono o file)", sources=["microphone", "upload"]),
        outputs=gr.Audio(label="Risposta agente (24kHz)"),
        title="Gemma 4 Full-Duplex — parla col modello",
        description=(
            "Modello full-duplex addestrato (coupling Qwen3-TTS, ADR-005). "
            "Il wav utente viene ri-campionato, convertito in mel nativo Gemma 4 E2B e "
            "processato; la risposta agente (16 codebook RVQ) è decodificata con Code2Wav."
        ),
    )
    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Server di inferenza Gemma 4 Full-Duplex")
    ap.add_argument("--checkpoint", default="checkpoints/gemma4_duplex_sota",
                    help="Dir del checkpoint (LoRA + custom_duplex_heads.pt)")
    ap.add_argument("--demo", metavar="WAV", help="Esegue una risposta singola su un wav utente")
    ap.add_argument("-o", "--output", default="samples/agent_response.wav",
                    help="Output wav per --demo")
    ap.add_argument("--serve", action="store_true", help="Avvia il server REST FastAPI")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--gradio", action="store_true", help="Avvia l'app Gradio")
    args = ap.parse_args()

    duplex, decoder = load_served_model(args.checkpoint)

    if args.demo:
        out = kp.demo_talk(duplex, decoder=decoder, input_wav=args.demo, output_wav=args.output)
        if out:
            print(f"\n✅ Risposta agente salvata in: {out}")
        return

    if args.gradio:
        build_gradio(duplex, decoder).launch(share=False)
        return

    if args.serve:
        import uvicorn
        uvicorn.run(build_app(duplex, decoder), host="0.0.0.0", port=args.port)
        return

    print("[Serve] Specifica --demo <wav>, --serve o --gradio.")


if __name__ == "__main__":
    main()
