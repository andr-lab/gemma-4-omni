#!/usr/bin/env python3
"""
inject_dynamics.py — Arricchimento post-hoc dei dialoghi sintetici (livello 2).

Obiettivo (docs/DESIGN_FULLDUPLEX_LETTERATURA.md §2.8 + verifica GT 2026-08-25):
Il GT dei dialoghi sintetici ha backchannel TROPPO rari (0.1% frame vs 2-4%
umani) e manca l'etichettatura esplicita yield/barge-in. Questo script:
1. INIETTA backchannel:
   - canale AGENTE: clip sintetizzate con Qwen3-TTS (timbro del sistema) --
     sono i backchannel che il modello deve IMPARARE A PRODURRE
   - canale UTENTE: clip REALI da SmoothConv (cross-lingua, non-lessicali) --
     contesto che Gemma ascolta
   Target: ~2-4% dei frame (1 backchannel ogni 10-20s, letteratura).
2. ETICHETTA yield/barge-in nel GT (derivati dal posizionamento):
   - barge_in: l'AGENTE inizia durante turno utente attivo (overlap con intento)
   - yield: l'UTENTE inizia durante turno agente e l'agente si ferma entro 0.8s
3. Aggiorna mix.wav + ch0/ch1 + gt.json (event: backchannel/overlap/yield/barge_in/none).

Uso: python inject_dynamics.py --out-dir data/synth_it/out [--bc-per-dialogo 4] [--dry]
"""
import os, sys, json, glob, wave, random, argparse
import numpy as np

SR = 24000
FPS = 12.5
FRAME = 1.0 / FPS

BACKCHANNEL_AGENT_TXT = ["mh-mh", "sì", "capito", "ok", "ah", "ecco", "già", "certo", "eh", "mmm"]

def read_wav(path):
    w = wave.open(path)
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
    if w.getnchannels() == 2:
        x = x[::2]
    return x, sr

def write_wav(path, x, sr):
    x16 = (np.clip(x, -1, 1)*32767).astype(np.int16)
    w = wave.open(path, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(x16.tobytes()); w.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/data/synth_it/out")
    ap.add_argument("--bc-user-dir", default="/workspace/data/backchannels",
                    help="clip backchannel REALI SmoothConv (canale utente)")
    ap.add_argument("--bc-agent-dir", default="/workspace/data/synth_it/bc_agent",
                    help="clip backchannel sintetizzate Qwen (canale agente)")
    ap.add_argument("--bc-per-dialogo", type=int, default=4,
                    help="backchannel per dialogo (target ~2-4% frame)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="solo statistiche, senza scrivere")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    bc_user = sorted(glob.glob(os.path.join(args.bc_user_dir, "*.wav")))
    bc_agent = sorted(glob.glob(os.path.join(args.bc_agent_dir, "*.wav")))
    print(f"[INJ] clip backchannel: utente(reali)={len(bc_user)} agente(sint)={len(bc_agent)}", flush=True)

    dialogs = sorted(glob.glob(os.path.join(args.out_dir, "*_mix.wav")))
    print(f"[INJ] {len(dialogs)} dialoghi", flush=True)

    tot_f = tot_bc = tot_ov = tot_y = tot_bi = 0
    for d in dialogs:
        base = d.replace("_mix.wav", "")
        gt_f = base + "_gt.json"
        if not os.path.isfile(gt_f):
            continue
        gt = json.load(open(gt_f))
        n = len(gt)
        # durata totale
        dur = gt[-1]["t"] + FRAME
        # seleziona punti di inserimento: DURANTE turni utente (per BC agente)
        # e DURANTE turni agente (per BC utente), non in overlap
        user_windows, agent_windows = [], []
        i = 0
        while i < n:
            st = gt[i]
            if st["user"] == 1 and st["agent"] == 0:
                j = i
                while j < n and gt[j]["user"] == 1 and gt[j]["agent"] == 0:
                    j += 1
                user_windows.append((gt[i]["t"], gt[j-1]["t"] + FRAME))
                i = j
            elif st["agent"] == 1 and st["user"] == 0:
                j = i
                while j < n and gt[j]["agent"] == 1 and gt[j]["user"] == 0:
                    j += 1
                agent_windows.append((gt[i]["t"], gt[j-1]["t"] + FRAME))
                i = j
            else:
                i += 1
        # filtra finestre utili (>= 1.5s)
        user_windows = [w for w in user_windows if w[1]-w[0] >= 1.5]
        agent_windows = [w for w in agent_windows if w[1]-w[0] >= 1.5]

        # ---- composizione audio attuale ----
        w = wave.open(d); nch = w.getnchannels()
        mix = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32)/32768
        if nch == 2:
            ch0 = mix[::2].copy(); ch1 = mix[1::2].copy()
        else:
            ch0 = mix.copy(); ch1 = np.zeros_like(mix)
        del mix

        inserts = []  # (time_sec, channel, clip_wav)
        # BC agente (sintetizzati) durante turni utente: meta' delle clip
        n_bc_agent = max(1, args.bc_per_dialogo // 2)
        n_bc_agent = min(n_bc_agent, len(user_windows), len(bc_agent))
        for k in range(n_bc_agent):
            t0, t1 = rng.choice(user_windows)
            pos = rng.uniform(t0 + 0.3, max(t0 + 0.3, t1 - 1.2))
            clip, csr = read_wav(bc_agent[k % len(bc_agent)])
            if csr != SR:
                import librosa
                clip = librosa.resample(clip, orig_sr=csr, target_sr=SR)
            clip = clip[: int(0.6 * SR)]   # tronca a 0.6s (backchannel breve)
            if len(clip) < int(0.15 * SR):
                continue
            inserts.append((pos, 1, clip))
        # BC utente (reali SmoothConv) durante turni agente: meta' delle clip
        n_bc_user = args.bc_per_dialogo - n_bc_agent
        for k in range(n_bc_user):
            if not agent_windows:
                break
            t0, t1 = rng.choice(agent_windows)
            pos = rng.uniform(t0 + 0.3, max(t0 + 0.3, t1 - 1.0))
            clip, csr = read_wav(bc_user[k % len(bc_user)])
            if csr != SR:
                import librosa
                clip = librosa.resample(clip, orig_sr=csr, target_sr=SR)
            clip = clip[: int(0.6 * SR)]
            if len(clip) < int(0.15 * SR):
                continue
            inserts.append((pos, 0, clip))

        if args.dry:
            tot_f += n
            tot_bc += len(inserts)
            continue

        # ---- applica inserimenti su ch0/ch1 ----
        for pos, ch, clip in inserts:
            i0 = int(pos * SR)
            if ch == 0:
                n1 = min(len(ch0), i0 + len(clip))
                ch0[i0:n1] += clip[: n1 - i0]
            else:
                n1 = min(len(ch1), i0 + len(clip))
                ch1[i0:n1] += clip[: n1 - i0]

        # ---- mix normalizzato ----
        n = max(len(ch0), len(ch1))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[:len(ch0), 0] = ch0
        mix[:len(ch1), 1] = ch1
        pk = float(np.abs(mix).max())
        if pk > 0.85:
            mix *= 0.85 / pk
        mix = np.clip(mix, -1, 1)

        write_wav(d, mix, SR)             # stereo
        write_wav(base + "_ch0.wav", mix[:, 0], SR)
        write_wav(base + "_ch1.wav", mix[:, 1], SR)

        # ---- GT aggiornato: eventi + yield/barge_in/backchannel ----
        nf = int(dur * FPS)
        # finestre dei backchannel iniettati (clip brevi, canale, intervallo)
        bc_wins = [(p, p + len(cl)/SR, ch) for (p, ch, cl) in inserts]
        new_gt = []
        prev = (0, 0)
        for f in range(nf):
            t0 = f * FRAME
            u = any(p <= t0 < p + len(cl)/SR for (p, ch_, cl) in inserts if ch_ == 0)
            a = any(p <= t0 < p + len(cl)/SR for (p, ch_, cl) in inserts if ch_ == 1)
            # stato base dal GT originale
            if f < len(gt):
                u = u or bool(gt[f]["user"])
                a = a or bool(gt[f]["agent"])
            ev = "none"
            # PRIORITA' 1: frame coperto da clip backchannel iniettata (breve)
            if any(bs <= t0 < be for (bs, be, _) in bc_wins):
                ev = "backchannel"
            # PRIORITA' 2: overlap con intento (yield/barge_in) o overlap puro
            elif u and a:
                if prev == (1, 0) and a:
                    ev = "barge_in"
                elif prev == (0, 1) and u:
                    ev = "yield"
                else:
                    ev = "overlap"
            new_gt.append({"t": round(t0, 3), "user": int(u), "agent": int(a), "event": ev})
            prev = (int(u), int(a))
        json.dump(new_gt, open(gt_f, "w"))

        # statistiche
        from collections import Counter
        c = Counter(g["event"] for g in new_gt)
        tot_f += len(new_gt)
        tot_bc += c.get("backchannel", 0) + c.get("yield", 0) + c.get("barge_in", 0)
        tot_ov += c.get("overlap", 0)
        tot_y += c.get("yield", 0)
        tot_bi += c.get("barge_in", 0)
        print(f"[INJ] {os.path.basename(base)}: {dict(c)}", flush=True)

    print(f"[INJ] TOT: frame={tot_f} "
          f"backchannel/yield/barge={tot_bc} ({100*tot_bc/max(tot_f,1):.1f}%) "
          f"overlap={tot_ov} ({100*tot_ov/max(tot_f,1):.1f}%)", flush=True)
    print("[INJ] FATTO.", flush=True)

if __name__ == "__main__":
    main()
