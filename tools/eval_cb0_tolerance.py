#!/usr/bin/env python3
"""eval_cb0_tolerance.py — Tolleranza del sub-talker agli errori su cb0.
Ancorato (cb0=GT) con corruzione p in {0, .1, .3, .5}: se la trascrizione blind
resta identica a p=0, il sistema tollera quell'errore; se collassa, il predittore
cb0 deve essere quasi esatto. Il riferimento di qualita' e' p=0 (adapter ancorato)."""
import os, sys
sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
import numpy as np
import torch
import soundfile as sf

import kaggle_pipeline as KP


def main():
    cache = "/workspace/cache_ars_full.pt"
    adapter_ckpt = "/workspace/checkpoints/ht_adapter_temporal2_final.pt"
    out = "/workspace/samples_tol"

    print("[TOL] 1/4 Sistema + adapter...", flush=True)
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, os.path.dirname(adapter_ckpt))
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)

    blob = torch.load(cache, weights_only=False)
    va_h, va_c = blob["va_h"], blob["va_c"]
    print(f"[TOL] 2/4 val windows: {len(va_h)}", flush=True)

    decoder = KP.Code2WavDecoder()
    os.makedirs(out, exist_ok=True)
    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    sub_dev = next(duplex_model.subtalker.parameters()).device
    sub_dtype = next(duplex_model.subtalker.parameters()).dtype
    V = 2048

    n_win = min(3, len(va_h))
    for p in [0.0, 0.1, 0.3, 0.5]:
        print(f"[TOL] 3/4 corruzione cb0 p={p}", flush=True)
        with torch.inference_mode():
            for i in range(n_win):
                T = min(64, va_c[i].shape[0], va_h[i].shape[0])
                gt = va_c[i][:T].clone()
                cb0 = gt.clone()
                if p > 0:
                    rng = np.random.default_rng(42 + i)
                    mask = rng.random(T) < p
                    for j in np.where(mask)[0]:
                        cand = int(rng.integers(0, V))
                        while cand == int(cb0[j]):
                            cand = int(rng.integers(0, V))
                        cb0[j] = cand
                h_sub = va_h[i][:T].unsqueeze(0).to(sub_dev).to(sub_dtype)
                codes_anchor = torch.zeros((1, T, 16), dtype=torch.long, device=sub_dev)
                codes_anchor[:, :, 0] = cb0.unsqueeze(0).to(sub_dev)
                logits = duplex_model.subtalker.forward_sub_talker_finetune(h_sub, codes_anchor)
                pred_rest = torch.argmax(logits, dim=-1)
                full = torch.zeros((1, T, 16), dtype=torch.long)
                full[0, :, 0] = cb0
                full[0, :, 1:] = pred_rest.cpu()
                wav = decoder.decode(full[0].cpu())
                f = f"{out}/tol_p{int(p*100):03d}_w{i}.wav"
                sf.write(f, wav, 24000)
                segs, _ = asr.transcribe(f, language="it", beam_size=5)
                blind = " ".join(s.text.strip() for s in segs)
                rms = float(np.sqrt(np.mean(np.square(wav))))
                print(f"[TOL] p={p:.1f} w{i} rms={rms:.3f} asr: {blind[:130]}", flush=True)
    print("[TOL] FATTO.", flush=True)


if __name__ == "__main__":
    main()
