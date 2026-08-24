#!/usr/bin/env python3
"""gen_ar_semantic.py — Generazione LIBERA: cb0 dall'AR-SemanticDecoder (senza ancoraggio
GT), residui dal sub-talker frozen, decode ufficiale, blind ASR."""
import os, sys, argparse

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import numpy as np
import torch
import soundfile as sf

import kaggle_pipeline as KP
sys.path.insert(0, "/workspace/tools")
from train_ar_semantic import ARSemanticDecoder, generate_ar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/workspace/cache_ars_full.pt")
    ap.add_argument("--ar-ckpt", default="/workspace/checkpoints/ar_semantic_final.pt")
    ap.add_argument("--adapter-ckpt", default="/workspace/checkpoints/ht_adapter_temporal2_final.pt")
    ap.add_argument("--out", default="/workspace/samples_ars")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args()

    print("[GEN-ARS] 1/5 Sistema + adapter temporale...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(args.adapter_ckpt))
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)

    print("[GEN-ARS] 2/5 Cache val h_t...", flush=True)
    blob = torch.load(args.cache, weights_only=False)
    va_h = blob["va_h"]

    print("[GEN-ARS] 3/5 AR-SemanticDecoder...", flush=True)
    blob_ar = torch.load(args.ar_ckpt, weights_only=False)
    V = blob_ar.get("vocab", 2048)
    model = ARSemanticDecoder(h_dim=va_h[0].shape[-1], vocab=V).to(device)
    model.load_state_dict(blob_ar["model"])
    model.eval()

    decoder = KP.Code2WavDecoder()
    os.makedirs(args.out, exist_ok=True)

    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    sub_dev = next(duplex_model.subtalker.parameters()).device
    sub_dtype = next(duplex_model.subtalker.parameters()).dtype

    print(f"[GEN-ARS] 4/5 Generazione libera su {args.n} finestre val...", flush=True)
    with torch.inference_mode():
        for i in range(min(args.n, len(va_h))):
            h_seq = va_h[i].unsqueeze(0).to(device).float()[:, :64, :]
            # GENERAZIONE LIBERA: nessun codice GT nell'ancoraggio
            cb0_free = generate_ar(model, h_seq, temperature=0.9, top_k=50)
            T = len(cb0_free)

            h_sub = va_h[i][:T].unsqueeze(0).to(sub_dev).to(sub_dtype)
            codes_anchor = torch.zeros((1, T, 16), dtype=torch.long, device=sub_dev)
            codes_anchor[:, :, 0] = torch.from_numpy(cb0_free).to(sub_dev)
            logits = duplex_model.subtalker.forward_sub_talker_finetune(h_sub, codes_anchor)
            pred_rest = torch.argmax(logits, dim=-1)

            full = torch.zeros((1, T, 16), dtype=torch.long)
            full[0, :, 0] = torch.from_numpy(cb0_free)
            full[0, :, 1:] = pred_rest.cpu()

            wav = decoder.decode(full[0].cpu())
            out_f = f"{args.out}/free_{i:03d}.wav"
            sf.write(out_f, wav, 24000)

            segs, _ = asr.transcribe(out_f, language="it", beam_size=5)
            blind = " ".join(s.text.strip() for s in segs)
            rms = float(np.sqrt(np.mean(np.square(wav))))
            print(f"[GEN] free_{i:03d} rms={rms:.4f} blind_asr: {blind[:130]}", flush=True)

    print("[GEN-ARS] FATTO.", flush=True)


if __name__ == "__main__":
    main()
