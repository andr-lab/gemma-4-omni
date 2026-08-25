#!/usr/bin/env python3
"""
prep_synth_cache.py — Cache di training per il trainer v5 (token speciali).

I dialoghi sintetici (data/synth_it/out) hanno wav stereo + GT frame-level con
eventi (backchannel/yield/barge_in/overlap). Per addestrare il predittore cb0
con i token di controllo servono, per ogni finestra del canale AGENTE:
  - h_t / h_sub: hidden di Gemma (stessa estrazione di extract_ars_cache_v2)
  - h_text: ancora testuale (testo del turno agente)
  - cb0_target: tokenizzazione del wav agente con Qwen3-TTS Tokenizer 12Hz
  - event: etichetta frame-level dal GT (none/backchannel/yield/barge_in/overlap)

Il vocabolario del predittore viene esteso con token speciali:
  V_special = [<backchannel>, <yield>, <barge_in>, <overlap>, <silence>] (5)
posti DOPO i 2048 codebook. Nei frame con evento, il target e' il token
speciale corrispondente (il predittore impara a emettere il token invece del
codebook); nei frame di silenzio -> <silence>.

Output: data/synth_it/synth_cache.pt con {tr_h, tr_tx, tr_cb0, tr_ev, ...}
"""
import os, sys, json, glob, argparse
import numpy as np

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

EVENT_TOKENS = {"backchannel": 2048, "yield": 2049, "barge_in": 2050,
                "overlap": 2051, "none": None}
SILENCE_TOKEN = 2052


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/data/synth_it/out")
    ap.add_argument("--output", default="/workspace/data/synth_it/synth_cache.pt")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoProcessor
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor

    # --- 1. tokenizer per wav -> cb0 (il modello Base include il tokenizer) ---
    snaps = sorted(glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*"))
    model_id = snaps[-1] if snaps else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"[PREP] carico tokenizer da {model_id}...", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev)
    if hasattr(model, "speech_tokenizer") and model.speech_tokenizer is not None:
        st = model.speech_tokenizer
        if hasattr(st, "model") and st.model is not None:
            st.model = st.model.to(dev)
            st.device = torch.device(dev)
    # encoder del tokenizer per wav -> codici
    tok = model.speech_tokenizer

    # --- 2. sistema Gemma per h_t / h_text (come extract_ars_cache_v2) ---
    print("[PREP] carico Gemma + adapter...", flush=True)
    import kaggle_pipeline as KP
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, "/workspace/checkpoints")
    for p in duplex_model.parameters():
        p.requires_grad = False
    duplex_model.eval()
    device = next(duplex_model.base_model.parameters()).device
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)

    dialogs = sorted(glob.glob(os.path.join(args.out_dir, "*_mix.wav")))
    if args.smoke:
        dialogs = dialogs[:2]
    print(f"[PREP] {len(dialogs)} dialoghi", flush=True)

    tr_h, tr_tx, tr_cb0, tr_ev = [], [], [], []
    for di, d in enumerate(dialogs):
        base = d.replace("_mix.wav", "")
        gt = json.load(open(base + "_gt.json"))
        meta = json.load(open(base + "_meta.json"))
        # wav agente (canale 1)
        import soundfile as sf
        mix, sr = sf.read(d, dtype="float32")
        ch1 = mix[:, 1] if mix.ndim == 2 else mix
        # 1) tokenizza wav agente -> codici (T, 16)
        codes = tok.encode(ch1, sample_rate=sr)[0] if hasattr(tok, "encode") else None
        if codes is None:
            print(f"[PREP] {os.path.basename(base)}: encode non disponibile, skip", flush=True)
            continue
        codes = np.asarray(codes)
        if codes.ndim == 2 and codes.shape[0] == 16:
            codes = codes.T
        cb0 = codes[:, 0] if codes.ndim == 2 else codes
        T = min(len(cb0), len(gt))
        cb0 = cb0[:T]

        # 2) eventi frame-level
        ev = np.array([EVENT_TOKENS.get(g["event"], None) for g in gt[:T]])

        # 3) h_t/h_text: finestra 64 frame come nel training AR
        #    (riuso il dataset MDC? no: costruisco input da mel del wav)
        #    NOTA: per il v5 iniziale usiamo SOLO cb0+eventi (la cache h_t/h_text
        #    arriva da extract_ars_cache_v2 per le finestre MDC; qui servirebbe
        #    Gemma sul mel del dialogo — rimandato al runtime, vedi doc)
        print(f"[PREP] {os.path.basename(base)}: T={T} cb0 range "
              f"[{cb0.min()},{cb0.max()}] eventi={dict(zip(*np.unique(ev, return_counts=True)))}", flush=True)
        # (per ora salviamo cb0/ev; h_t/h_text si agganceranno dalla cache v2
        #  o da estrazione dedicata nel passo successivo)
        tr_cb0.append(torch.from_numpy(cb0).long())
        tr_ev.append(torch.from_numpy(ev).long())

    torch.save({"tr_cb0": tr_cb0, "tr_ev": tr_ev}, args.output)
    print(f"[PREP] salvato {args.output}: {len(tr_cb0)} dialoghi (cb0+eventi). "
          "h_t/h_text: da agganciare (v5 full richiede estrazione dedicata)", flush=True)
    print("[PREP] FATTO.", flush=True)


if __name__ == "__main__":
    main()
