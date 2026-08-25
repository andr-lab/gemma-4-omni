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

# Mock torchaudio/sox (il tokenizer 25Hz viene importato anche se inutilizzato;
# stesso fix di decode_official_worker.py per il 12Hz)
import types, importlib.machinery
try:
    import torchaudio  # noqa
except Exception:
    mock_ta = types.ModuleType("torchaudio")
    mock_ta.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
    sys.modules["torchaudio"] = mock_ta
    sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
    sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
    mock_kaldi = types.ModuleType("torchaudio.compliance.kaldi")
    sys.modules["torchaudio.compliance.kaldi"] = mock_kaldi
for _m in ("sox",):
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

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
    """Assegna i turni ai canali per RUOLO (user->ch0, assistant->ch1),
    usando la partitura solo per timing/overlap. I backchannel della
    partitura diventano frasi brevi IT sul canale dell'ascoltatore."""
    user_turns = [clean_text(m["content"]) for m in dialogue_msgs
                  if m.get("role") == "user" and clean_text(m["content"])]
    asst_turns = [clean_text(m["content"]) for m in dialogue_msgs
                  if m.get("role") == "assistant" and clean_text(m["content"])]
    # fallback: se manca un ruolo, dividi i turni disponibili
    if not user_turns and asst_turns:
        user_turns = asst_turns[: len(asst_turns) // 2]
        asst_turns = asst_turns[len(asst_turns) // 2:]
    if not asst_turns and user_turns:
        asst_turns = user_turns[: len(user_turns) // 2]
        user_turns = user_turns[len(user_turns) // 2:]
    ui, ai = 0, 0
    script = []
    for s in slots:
        ch = s["ch"]
        tt = s["turn"]
        if tt.lower().startswith("backchannel"):
            # il backchannel va sull'ALTRO canale rispetto al turno corrente
            txt = rng.choice(BACKCHANNEL_IT)
        elif ch == 0:
            txt = user_turns[ui % len(user_turns)] if user_turns else rng.choice(BACKCHANNEL_IT)
            ui += 1
        else:
            txt = asst_turns[ai % len(asst_turns)] if asst_turns else rng.choice(BACKCHANNEL_IT)
            ai += 1
        script.append((s, txt))
    return script


def count_roles(msgs):
    n_u = sum(1 for m in msgs if m.get("role") == "user" and clean_text(m.get("content", "")))
    n_a = sum(1 for m in msgs if m.get("role") == "assistant" and clean_text(m.get("content", "")))
    return n_u, n_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="/workspace/data/synth_it/it_sft_train.parquet")
    ap.add_argument("--scores-dir", default="/workspace/data/smoothconv_json")
    ap.add_argument("--out", default="/workspace/data/synth_it/out")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--start-idx", type=int, default=0, help="primo dialogo da processare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-turns", type=int, default=24)
    ap.add_argument("--ref-user", default="/workspace/_rankfix_verify/eval_temporal2_gt_ref_a.wav",
                    help="voce UTENTE (parlato GT reale italiano, run rankfix r1_v3 — audio ottimo, no clipping)")
    ap.add_argument("--ref-agent", default="/workspace/checkpoints/eval_temporal2_rvq_oracle_a_N7.wav",
                    help="voce AGENTE (oracle N7: 7 codebook GT ancorati = timbro sub-talker PULITO; blind ASR esatto; greedy era incomprensibile)")
    ap.add_argument("--smoke", action="store_true", help="1 dialogo, 6 turni, niente salvataggio GT esteso")
    args = ap.parse_args()
    if args.smoke:
        args.n = 1
        args.max_turns = 6

    import pandas as pd
    df = pd.read_parquet(args.sft)
    multi = df[df["category"] == "multi_turno"].reset_index(drop=True)
    # filtra dialoghi con turni bilanciati (>=3 user e >=3 assistant)
    mask = multi["messages"].apply(
        lambda msgs: count_roles(msgs)[0] >= 3 and count_roles(msgs)[1] >= 3)
    multi = multi[mask].reset_index(drop=True)
    print(f"[SYNTH] {len(multi)} dialoghi multi-turno bilanciati (>=3 turni per ruolo)", flush=True)

    scores = sorted(glob.glob(os.path.join(args.scores_dir, "*.json")))
    # seleziona partiture con overlap reale (>=1 coppia sovrapposta >150ms)
    def _has_overlap(path):
        try:
            x = json.load(open(path, encoding="utf-8"))
            inst = x.get("instances", [])
            by_ch = {}
            for i in inst:
                by_ch.setdefault(int(i.get("channelIndex", 0)), []).append(i)
            if len(by_ch) < 2:
                return False
            chs = list(by_ch.keys())
            for i0 in by_ch[chs[0]]:
                s0, e0 = i0.get("start", 0), i0.get("end", 0)
                for i1 in by_ch[chs[1]]:
                    if min(e0, i1.get("end", 0)) - max(s0, i1.get("start", 0)) > 0.15:
                        return True
            return False
        except Exception:
            return False
    scores_ov = [s for s in scores if _has_overlap(s)]
    if scores_ov:
        scores = scores_ov
    print(f"[SYNTH] {len(scores)} partiture SmoothConv (con overlap: {len(scores_ov)})", flush=True)
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    import torch
    from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    snaps = sorted(glob.glob("/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*"))
    model_id = snaps[-1] if snaps else "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    print(f"[SYNTH] modello: {model_id}", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16).to(dev)
    proc = Qwen3TTSProcessor.from_pretrained(model_id, trust_remote_code=True)
    tts = Qwen3TTSModel(model=model, processor=proc)

    print("[SYNTH] ref voices: utente=gt_ref rankfix (voce umana IT), agente=oracle rankfix (timbro sub-talker)...", flush=True)
    import soundfile as sf

    def _load_ref(path):
        w, sr = sf.read(path, dtype="float32")
        pk = float(np.abs(w).max())
        if pk > 0.85:
            w = w * (0.85 / pk)
        return w, sr

    # ref UTENTE: parlato GT reale italiano (voce umana, run rankfix r1_v3)
    # ref AGENTE: oracle sub-talker con cb0 GT = timbro esatto del nostro sistema
    ref_user_wav = _load_ref(args.ref_user) if os.path.isfile(args.ref_user) else None
    ref_agent_wav = _load_ref(args.ref_agent) if os.path.isfile(args.ref_agent) else None
    if ref_user_wav is None or ref_agent_wav is None:
        raise SystemExit(f"[SYNTH] ref mancanti: user={args.ref_user} agent={args.ref_agent}")

    def synth(text, ch):
        out, sr = tts.generate_voice_clone(
            text=text, language="italian",
            ref_audio=ref_agent_wav if ch == 1 else ref_user_wav,
            x_vector_only_mode=True,
            non_streaming_mode=True,
            do_sample=True, temperature=0.9, top_k=50, top_p=0.9,
            max_new_tokens=700,   # ~56s audio max per turno (evita runaway 8192)
        )
        wav = np.asarray(out[0], dtype=np.float32)
        if sr != SR:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        return wav

    done = 0
    for d_idx in range(args.start_idx, min(args.start_idx + args.n, len(multi))):
        base = os.path.join(args.out, f"synth_{d_idx:04d}")
        if os.path.isfile(base + "_mix.wav"):
            print(f"[SYNTH] skip {d_idx} (gia' fatto)", flush=True)
            done += 1
            continue
        row = multi.iloc[d_idx]
        msgs = row["messages"]
        sc = scores[d_idx % len(scores)]
        slots = load_score(sc)
        script = build_script(slots, msgs, rng)[: args.max_turns]
        if len(script) < 3:
            continue

        ch0, ch1 = [], []   # (start_sec, wav)
        # TIME-STRUCTURE TRANSFER: i turni si piazzano agli start REALI della
        # partitura SmoothConv; dove la griglia ha overlap (slot successivi
        # sovrapposti), i canali si mixano davvero (sovrapposizione fisica).
        # Regola: start_turno_i = max(start_slot_i, end_wav_precedente - overlap_grq)
        # con overlap_grq = sovrapposizione reale tra slot i-1 e i nella griglia.
        prev_end_wav = -1.0
        for idx, (s, txt) in enumerate(script):
            ch = s["ch"]
            print(f"[SYNTH] turno {idx}/{len(script)} ch={ch} '{txt[:50]}'", flush=True)
            try:
                wav = synth(txt, ch)
            except Exception as e:
                print(f"[SYNTH] ERR {txt[:40]}: {e}", flush=True)
                continue
            print(f"[SYNTH]   -> {len(wav)/SR:.1f}s", flush=True)
            import torch as _t
            _t.cuda.empty_cache()
            dur = len(wav) / SR
            # start: posizione reale dello slot nella partitura
            start = s["start"]
            # se il wav precedente sta ancora suonando, calcola overlap della
            # griglia: overlap_grq = durata sovrapposizione tra slot i-1 e i
            # (da griglia reale). Il nuovo turno parte al suo start slot, che
            # può essere < end del precedente -> overlap fisico nel mix.
            if prev_end_wav > start and (idx > 0):
                pass  # overlap reale: il turno parte comunque allo start slot
            (ch0 if ch == 0 else ch1).append((start, wav))
            prev_end_wav = max(prev_end_wav, start + dur)

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

        # GT frame-level 12.5Hz: USER/AGENT/BOTH/NONE + eventi
        # (overlap = entrambi attivi; backchannel = turno breve agente < 1.2s)
        bc_windows = [(s, s + len(w) / SR) for (s, w) in ch1 if len(w) / SR < 1.2]
        nf = int(total * FPS)
        gt = []
        for f in range(nf):
            t0 = f * FRAME
            u = any(s <= t0 < s + len(w) / SR for (s, w) in ch0)
            a = any(s <= t0 < s + len(w) / SR for (s, w) in ch1)
            ev = "none"
            if u and a:
                ev = "overlap"
            elif a and any(bs <= t0 < be for (bs, be) in bc_windows):
                ev = "backchannel"
            gt.append({"t": round(t0, 3), "user": int(u), "agent": int(a),
                       "event": ev})

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
