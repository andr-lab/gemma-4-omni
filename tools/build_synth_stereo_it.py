#!/usr/bin/env python3
"""
build_synth_stereo_it.py — Sintesi dello STEREO IT SINTETICO (livello 2 dati).

Design evidence-based (docs/DESIGN_FULLDUPLEX_LETTERATURA.md §2.8):
- CONTENUTO: dialoghi multi-turno italiani (SerFabio89/italian-open-sft-chat-dataset)
- PARTITURA TEMPORALE: griglie reali di SmoothConv (annotazione UMANA: turn
  complete/incomplete/backchannel + timings float) → time-structure transfer
  cross-lingua (la FORMA degli eventi di turno è universale).
- SINTESI: Qwen3-TTS Base, voice clone dual-role:
    * canale 0 UTENTE  = ref_audio voce umana (oracle GT italiano)
    * canale 1 AGENTE  = ref_audio timbro sub-talker (la voce del nostro sistema)
    x_vector_only_mode=True → solo timbro, niente ref_text.
- GT BY CONSTRUCTION: turn state frame-level (12.5Hz) USER/AGENT/BOTH/NONE +
  etichette evento (overlap) per i token di controllo del predittore cb0.

Output per dialogo in out_dir/:
  <id>_mix.wav (stereo 24k: [utente, agente]), _ch0.wav, _ch1.wav,
  <id>_gt.json (frame-level), <id>_meta.json
"""
import os, sys, json, glob, argparse, random
import numpy as np

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/workspace/qwen3tts_src")

SR = 24000
FPS = 12.5
FRAME = 1.0 / FPS

BACKCHANNEL_IT = ["mh-mh", "sì", "capito", "ok", "ah", "ecco", "mh", "già", "certo", "eh"]


def clean_text(t: str, max_len: int = 220) -> str:
    import re
    t = t.replace("**", "").replace("##", "").replace("#", "").replace("`", "")
    t = t.replace("---", " ").replace("###", "")
    t = re.sub(r"\[.*?\]\(.*?\)", "", t)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^\w\sàèéìòùáíóúäöü.,;:!?\"'()\-–—]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) > max_len:
        cut = t[:max_len]
        last = max(cut.rfind(". "), cut.rfind("? "), cut.rfind("! "), cut.rfind(", "))
        if last > max_len // 2:
            cut = cut[: last + 1]
        t = cut
    return t


def load_score(json_path: str):
    x = json.load(open(json_path, encoding="utf-8"))
    slots = []
    for i in x.get("instances", []):
        a = i.get("attributes", {})
        tt = a.get("turn", "?")
        if isinstance(tt, list):
            tt = tt[0] if tt else "?"
        slots.append({"ch": int(i.get("channelIndex", 0)),
                      "start": float(i.get("start", 0)),
                      "end": float(i.get("end", 0)),
                      "turn": str(tt)})
    slots.sort(key=lambda s: s["start"])
    return slots


def build_script(slots, dialogue_msgs, rng):
    user_turns = [clean_text(m["content"]) for m in dialogue_msgs
                  if m.get("role") == "user" and clean_text(m["content"])]
    asst_turns = [clean_text(m["content"]) for m in dialogue_msgs
                  if m.get("role") == "assistant" and clean_text(m["content"])]
    ui, ai = 0, 0
    script = []
    for s in slots:
        ch = s["ch"]
        tt = s["turn"]
        if tt.lower().startswith("backchannel"):
            txt = rng.choice(BACKCHANNEL_IT)
        elif ch == 0:
            txt = user_turns[ui % len(user_turns)] if user_turns else "mh"
            ui += 1
        else:
            txt = asst_turns[ai % len(asst_turns)] if asst_turns else "sì"
            ai += 1
        script.append((s, txt))
    return script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/workspace/data/synth_it/it_sft_train.parquet")
    ap.add_argument("--scores-dir", default="/workspace/data/smoothconv_json")
    ap.add_argument("--out", default="/workspace/data/synth_it/out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--ref-user", default="/workspace/samples_temp_0.7_robust/oracle_000.wav",
                    help="voce UTENTE (umana italiana)")
    ap.add_argument("--ref-agent", default="/workspace/samples_ars/free_000.wav",
                    help="voce AGENTE (timbro sub-talker Qwen)")
    ap.add_argument("--smoke", action="store_true", help="1 dialogo, 6 turni, niente salvataggio GT esteso")
    args = ap.parse_args()
    if args.smoke:
        args.n = 1
        args.max_turns = 6

    import pandas as pd
    df = pd.read_parquet(args.sft)
    multi = df[df["category"] == "multi_turno"].reset_index(drop=True)
    print(f"[SYNTH] {len(multi)} dialoghi multi-turno", flush=True)

    scores = sorted(glob.glob(os.path.join(args.scores_dir, "*.json")))
    print(f"[SYNTH] {len(scores)} partiture SmoothConv", flush=True)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from transformers import AutoModel, AutoProcessor
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    snaps = sorted(glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*"))
    model_id = snaps[-1] if snaps else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"[SYNTH] modello: {model_id}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(dev)
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    tts = Qwen3TTSModel(model=model, processor=proc)

    print("[SYNTH] genero reference voices pulite (voce canonica Qwen3-TTS, zero clipping)...", flush=True)
    import soundfile as sf

    # VOICE REF: generata dal modello stesso (timbro canonico Qwen3-TTS =
    # timbro del sub-talker del nostro sistema), normalizzata per evitare
    # il clipping sistematico dei wav storici (peak 1.0).
    REF_TXT_AGENT = ("Questa è una frase di riferimento per la voce dell'assistente. "
                     "Il cielo è azzurro e il mare è calmo.")
    REF_TXT_USER = ("Salve, come stai? Vorrei chiederti una cosa importante "
                    "a proposito del progetto di domani.")

    def _gen_ref(text, ref_audio=None):
        out, sr = tts.generate_voice_clone(
            text=text, language="it",
            ref_audio=ref_audio,          # None -> voce di default del modello
            x_vector_only_mode=ref_audio is not None,
            non_streaming_mode=True,
            do_sample=True, temperature=0.9, top_k=50, top_p=0.9,
        )
        w = np.asarray(out[0], dtype=np.float32)
        if sr != SR:
            import librosa
            w = librosa.resample(w, orig_sr=sr, target_sr=SR)
        # normalizzazione anti-clipping (headroom 0.85)
        peak = float(np.abs(w).max())
        if peak > 0.85:
            w = w * (0.85 / peak)
        return w

    # vo ci utente: voce umana diversa dall'agente — se --ref-user è un file
    # esistente lo usa (pulito), altrimenti voce di default con testo diverso.
    if args.ref_user and os.path.isfile(args.ref_user):
        ref_user_wav, _ = sf.read(args.ref_user, dtype="float32")
        ref_user_wav = ref_user_wav / max(float(np.abs(ref_user_wav).max()), 1e-6)
    else:
        ref_user_wav = None
    ref_agent_wav = None

    def synth(text, ch):
        out, sr = tts.generate_voice_clone(
            text=text, language="it",
            ref_audio=ref_user_wav if ch == 0 else None,
            x_vector_only_mode=(ch == 0 and ref_user_wav is not None),
            non_streaming_mode=True,
            do_sample=True, temperature=0.9, top_k=50, top_p=0.9,
        )
        wav = np.asarray(out[0], dtype=np.float32)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav

    done = 0
    for d_idx in range(min(args.n, len(multi))):
        row = multi.iloc[d_idx]
        msgs = row["messages"]
        sc = scores[d_idx % len(scores)]
        slots = load_score(sc)
        script = build_script(slots, msgs, rng)[: args.max_turns]
        if len(script) < 3:
            continue

        ch0, ch1 = [], []   # (start_sec, wav)
        t_cur = 0.0
        for (s, txt) in script:
            ch = s["ch"]
            try:
                wav = synth(txt, ch)
            except Exception as e:
                print(f"[SYNTH] ERR {txt[:40]}: {e}", flush=True)
                continue
            dur = len(wav) / SR
            (ch0 if ch == 0 else ch1).append((t_cur, wav))
            # gap dello slot reale (partitura) se >0, altrimenti 0.25s
            gap = max(0.0, s["end"] - s["start"] - dur) if (s["end"] - s["start"]) > 0 else 0.25
            t_cur = t_cur + dur + (gap * 0.3 + 0.2)   # gap attenuato + minimo

        if not ch0 and not ch1:
            continue

        total = max((s + len(w) / SR for (s, w) in ch0 + ch1), default=0.0) + 0.5
        n = int(total * SR)
        mix = np.zeros((n, 2), dtype=np.float32)
        for (s, w) in ch0:
            i0, i1 = int(s * SR), min(int(s * SR) + len(w), n)
            mix[i0:i1, 0] += w[: i1 - i0]
        for (s, w) in ch1:
            i0, i1 = int(s * SR), min(int(s * SR) + len(w), n)
            mix[i0:i1, 1] += w[: i1 - i0]
        mix = np.clip(mix, -1.0, 1.0)

        # GT frame-level 12.5Hz
        nf = int(total * FPS)
        gt = []
        for f in range(nf):
            t0 = f * FRAME
            u = any(s <= t0 < s + len(w) / SR for (s, w) in ch0)
            a = any(s <= t0 < s + len(w) / SR for (s, w) in ch1)
            gt.append({"t": round(t0, 3), "user": int(u), "agent": int(a),
                       "event": "overlap" if (u and a) else "none"})

        base = os.path.join(args.out, f"synth_{d_idx:04d}")
        sf.write(base + "_mix.wav", mix, SR)
        sf.write(base + "_ch0.wav", mix[:, 0], SR)
        sf.write(base + "_ch1.wav", mix[:, 1], SR)
        with open(base + "_gt.json", "w") as f:
            json.dump(gt, f)
        with open(base + "_meta.json", "w") as f:
            json.dump({"sft_id": row["id"], "score": os.path.basename(sc),
                       "n_turns": len(script), "dur_s": round(total, 2)}, f)
        done += 1
        n_ov = sum(1 for g in gt if g["event"] == "overlap")
        print(f"[SYNTH] {done}: {base} ({total:.1f}s, {len(script)} turni, "
              f"overlap={n_ov} frame)", flush=True)

    print(f"[SYNTH] FATTO: {done} dialoghi in {args.out}", flush=True)


if __name__ == "__main__":
    main()
