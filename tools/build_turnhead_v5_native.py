#!/usr/bin/env python3
"""
build_turnhead_v5_native.py — Dataset TurnHead v5 con GROUND TRUTH NATIVO DuplexConv
=====================================================================================
ORCHESTRATORE (Task 11H, fase 2). Sostituisce le euristiche energetiche di v3/v4
con le annotazioni native del dataset (README ufficiale HF):
  - wavInfo.vadFlagPerFrmPerTrack : VAD frame-level per canale (nFrm x nTrack)
  - asr[t][i].state               : <|complete|>/<|incomplete|> per utterance
  - asr[t][i].speaker.segments    : diarizzazione

5 classi a 12.5 Hz (80 ms/frame), canale 0 = utente, canale 1 = agente:
  0 SPEAK       : agente attivo, utente silente
  1 YIELD       : ultime 2 frame di un'utteranza agente <|complete|> seguita da silenzio agente >= 400ms
  2 BARGE_IN    : overlap con utteranza utente >= 400ms che tronca l'agente
                  (l'utteranza agente sovrapposta finisce entro 1.0s dall'onset utente
                   e il suo state e' <|incomplete|> oppure non segue parlato agente per > 1.5s)
  3 USER_BC     : overlap con utteranza utente < 400ms (feedback breve) o agente prosegue > 1s
  4 EMIT_BC     : utteranza agente breve (< 800ms) durante tratto utente-only lungo (> 2s)
Frame non validi (nessuna attività): turn_valid=False.

Output (in --out-dir, default /workspace/data):
  thv5_user_mel.parquet, thv5_agent_mel.parquet  (matrici piatte float16 @100fps, come 20.0h_ext)
  thv5_turnlabels.parquet (turn_label int8, turn_valid bool per frame agente @12.5Hz)
  thv5_hyperparams.json (SEGMENT_ENDS + statistiche classi)

Uso CPU per mel/label (niente contesa GPU col training adapter v3 in corso).
"""
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # FORCE CPU: GPU occupata dal training v3

import kaggle_pipeline as KP


def load_wavinfo(jpath):
    import json as _json
    with open(jpath, encoding="utf-8") as f:
        return _json.load(f)


def vad_at_frame(wavinfo, track, t_ms_center):
    """Legge vadFlagPerFrmPerTrack al tempo t (ms, centro frame 12.5Hz). Fallback: None."""
    try:
        vad = np.asarray(wavinfo["vadFlagPerFrmPerTrack"])
        frm_len = int(wavinfo.get("vadFrmLenInMs", 30))
        idx = int(t_ms_center // frm_len)
        if idx < 0 or idx >= vad.shape[0]:
            return None
        return bool(vad[idx, track] > 0) if vad.ndim == 2 else None
    except Exception:
        return None


def energy_vad(mel_block, thr=0.012):
    """Fallback se VAD nativo assente nel JSON: energia mel (come v3 ma SOLO fallback)."""
    e = np.mean(np.abs(np.asarray(mel_block, dtype=np.float32)), axis=-1)  # (T*MPF,)
    # downsample a 12.5Hz
    mpf = e.shape[0] // max(1, (e.shape[0] * 10 // 80))
    return e


def build_labels_for_segment(wavinfo, n_frames, frame_s=0.08):
    """Etichette native per n_frames a 12.5Hz. Ritorna (labels, valid)."""
    labels = np.zeros(n_frames, dtype=np.int64)
    valid = np.zeros(n_frames, dtype=bool)

    def utts(track):
        out = []
        for u in (wavinfo.get("asr") or [])[track] if len(wavinfo.get("asr") or []) > track else []:
            s = int(u.get("startInMs", 0)) / 1000.0
            e = int(u.get("endInMs", 0)) / 1000.0
            if e <= s:
                continue
            out.append({"s": s, "e": e, "state": u.get("state", ""), "dur": e - s})
        return sorted(out, key=lambda x: x["s"])

    agent_u, user_u = utts(1), utts(0)

    # VAD nativo pre-campionato a 12.5Hz (se disponibile)
    n_vad_frm = None
    try:
        vad = np.asarray(wavinfo["vadFlagPerFrmPerTrack"])
        frm_len = int(wavinfo.get("vadFrmLenInMs", 30))
        n_vad_frm = vad  # (nFrm, nTrack)
    except Exception:
        vad = None

    def is_active(track, t_sec):
        if vad is not None:
            idx = int((t_sec * 1000) // frm_len)
            if 0 <= idx < vad.shape[0]:
                return bool(vad[idx, track] > 0)
        # fallback: dentro un'utteranza ASR?
        for u in (agent_u if track == 1 else user_u):
            if u["s"] <= t_sec < u["e"]:
                return True
        return False

    centers = (np.arange(n_frames) + 0.5) * frame_s
    a_act = np.array([is_active(1, t) for t in centers], dtype=bool)
    u_act = np.array([is_active(0, t) for t in centers], dtype=bool)
    valid = a_act | u_act

    # helper: silenzio agente dopo la fine dell'utteranza i
    def agent_gap_after(e_sec, horizon=1.5):
        nxt = [u for u in agent_u if u["s"] >= e_sec - 1e-6]
        if not nxt:
            return True
        return (nxt[0]["s"] - e_sec) >= horizon

    for i, u in enumerate(agent_u):
        s_f = int(u["s"] / frame_s); e_f = min(int(np.ceil(u["e"] / frame_s)), n_frames)
        # YIELD: ultime 2 frame delle utteranze complete con gap dopo
        if u["state"].endswith("complete|>") or "<|complete|>" in u["state"]:
            if agent_gap_after(u["e"], 0.4):
                for t in range(max(s_f, e_f - 2), e_f):
                    if 0 <= t < n_frames and a_act[t]:
                        labels[t] = 1
        # EMIT_BC: utteranza agente breve in tratto utente lungo
        if u["dur"] < 0.8:
            u_span = [v for v in user_u if v["s"] < u["e"] and v["e"] > u["s"]]
            if u_span and any(v["dur"] > 2.0 for v in u_span):
                for t in range(max(0, s_f), min(e_f, n_frames)):
                    labels[t] = 4

    for u in user_u:
        s_f = int(u["s"] / frame_s); e_f = min(int(np.ceil(u["e"] / frame_s)), n_frames)
        overlap_agent = [a for a in agent_u if a["s"] < u["e"] and a["e"] > u["s"]]
        if overlap_agent:
            truncated = (
                ("<|incomplete|>" in overlap_agent[0]["state"]) or
                (overlap_agent[0]["e"] - u["s"] < 1.0 and agent_gap_after(overlap_agent[0]["e"], 1.5))
            )
            cls = 2 if (u["dur"] >= 0.4 and truncated) else 3
            for t in range(max(0, s_f), min(e_f, n_frames)):
                if a_act[t] and u_act[t]:  # solo sulle frame di effettiva overlap
                    labels[t] = cls

    # dove solo agente parla e non etichettato -> SPEAK
    speak_mask = a_act & ~u_act & (labels == 0)
    labels[speak_mask] = 0
    return labels, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-hours", type=float, default=float(os.environ.get("THV5_MAX_HOURS", "20")))
    ap.add_argument("--max-samples", type=int, default=int(os.environ.get("THV5_MAX_SAMPLES", "6000")))
    ap.add_argument("--out-dir", default="/workspace/data")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()

    print("[THV5] Carico annotazioni DuplexConv (93.709 JSON su disco)...", flush=True)
    annotations = KP.load_duplex_annotations(download_dir="/workspace/data/duplex_jsons")
    print(f"[THV5] {len(annotations)} segmenti indicizzati.", flush=True)

    # FIX torchcodec: datasets 4.3.0 prova a decodificare l'audio con torchcodec (assente).
    # Disabilitiamo la decodifica automatica: item['wav'] = {'path','bytes'} e il
    # decoding lo fa _decode_duplex_audio con soundfile (ramo 'esempio grezzo').
    from datasets import load_dataset as _hf_load, Audio as _hf_Audio
    ds = _hf_load("qualialabsAI/DuplexConv", split="train", streaming=True)
    try:
        ds = ds.cast_column("wav", _hf_Audio(decode=False))
        print("[THV5] cast_column wav -> Audio(decode=False) OK", flush=True)
    except Exception as e:
        print(f"[THV5] cast_column fallita ({e}), procedo senza.", flush=True)
    max_samples = args.max_samples
    subset = ds.take(max_samples)
    print(f"[THV5] Subset: ~{max_samples} segmenti streaming.", flush=True)
    if subset is None:
        print("[THV5] ERRORE: dataset non disponibile."); sys.exit(2)

    mel_extractor = KP.load_mel_extractor()  # AutoFeatureExtractor, CPU-only

    blocks_um, blocks_am, seg_ends = [], [], []
    label_blocks, valid_blocks = [], []
    total_frames = 0
    t0 = time.time(); processed = 0; used = 0
    chunk_um, chunk_am, chunk_lb, chunk_vd = [], [], [], []

    def flush_chunk(last=False):
        nonlocal chunk_um, chunk_am, chunk_lb, chunk_vd, total_frames
        if not chunk_um:
            return
        um = np.concatenate(chunk_um); am = np.concatenate(chunk_am)
        lb = np.concatenate(chunk_lb); vd = np.concatenate(chunk_vd)
        n_mel = um.shape[0] // 128
        # schema come 20.0h_ext: righe = frame mel @100fps, 128 colonne mel bins
        pd.DataFrame(um[: n_mel * 128].reshape(n_mel, 128).astype(np.float16)).to_parquet(
            f"{args.out_dir}/thv5_user_mel_part{processed:05d}.parquet", index=False)
        n_mel_a = am.shape[0] // 128
        pd.DataFrame(am[: n_mel_a * 128].reshape(n_mel_a, 128).astype(np.float16)).to_parquet(
            f"{args.out_dir}/thv5_agent_mel_part{processed:05d}.parquet", index=False)
        pd.DataFrame({"turn_label": lb.astype(np.int8), "turn_valid": vd}).to_parquet(
            f"{args.out_dir}/thv5_turnlabels_part{processed:05d}.parquet", index=False)
        total_frames += len(lb)
        chunk_um, chunk_am, chunk_lb, chunk_vd = [], [], [], []

    for item in subset:
        processed += 1
        key = item.get("__key__")
        jpath = annotations.get(str(key)) if key is not None else None
        raw_audio = item.get("wav") or item.get("audio") or item.get("user_audio")
        decoded = KP._decode_duplex_audio(raw_audio)
        if decoded is None or jpath is None or not os.path.exists(jpath):
            continue
        arr, sr = decoded
        if arr.ndim < 2 or arr.shape[1] < 2:
            continue
        u_mel = KP.extract_mel_features(mel_extractor, arr[:, 0], sr=sr)
        a_mel = KP.extract_mel_features(mel_extractor, arr[:, 1], sr=sr)
        if u_mel is None or a_mel is None:
            continue
        # allineo entrambe alle frame agente stesse (senza RVQ: uso mel agent come riferimento)
        n_agent_mel = min(len(u_mel), len(a_mel))
        MPF = getattr(KP, "MEL_PER_AGENT_FRAME", 8)
        n_code_frames = n_agent_mel // MPF
        if n_code_frames < 8:
            continue
        u_mel = u_mel[: n_code_frames * MPF]; a_mel = a_mel[: n_code_frames * MPF]

        wavinfo = load_wavinfo(jpath)
        dur_sec = arr.shape[0] / sr
        n_lbl = min(n_code_frames, int(dur_sec / 0.08))
        labels, valid = build_labels_for_segment(wavinfo, n_lbl)
        if len(labels) < n_code_frames:
            pad = n_code_frames - len(labels)
            labels = np.concatenate([labels, np.zeros(pad, dtype=np.int64)])
            valid = np.concatenate([valid, np.zeros(pad, dtype=bool)])

        chunk_um.append(np.asarray(u_mel, dtype=np.float32).reshape(-1))
        chunk_am.append(np.asarray(a_mel, dtype=np.float32).reshape(-1))
        seg_ends.append(total_frames + sum(len(x) for x in chunk_lb) + n_code_frames)
        chunk_lb.append(labels); chunk_vd.append(valid)
        used += 1

        if used % args.chunk == 0:
            flush_chunk()
            el = time.time() - t0
            print(f"[THV5] {used} segmenti ({total_frames} frame) {el:.0f}s "
                  f"({used/el:.2f} seg/s)", flush=True)
        if used >= args.max_samples:
            break

    flush_chunk(last=True)

    # merge parti + stats
    import glob
    parts_lb = sorted(glob.glob(f"{args.out_dir}/thv5_turnlabels_part*.parquet"))
    dfs = [pd.read_parquet(p) for p in parts_lb]
    all_lb = pd.concat(dfs, ignore_index=True)
    all_lb.to_parquet(f"{args.out_dir}/train_tokens_thv5_turnlabels.parquet", index=False)
    dist = {int(c): int(((all_lb.turn_label == c) & all_lb.turn_valid).sum())
            for c in sorted(all_lb.turn_label.unique())}
    tot = sum(dist.values())
    hyper = {
        "DATASET": "DuplexConv native-GT TurnHead v5",
        "TOTAL_FRAMES": int(len(all_lb)),
        "SEGMENT_ENDS": seg_ends,
        "CLASS_NAMES": ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"],
        "CLASS_DIST": dist,
        "CLASS_PCT": {k: round(100*v/max(1,tot), 2) for k, v in dist.items()},
        "SOURCE": "vadFlagPerFrmPerTrack + asr.state + speaker.segments (README HF)",
        "N_SEGMENTS_USED": used,
    }
    with open(f"{args.out_dir}/train_tokens_thv5_hyperparams.json", "w") as f:
        json.dump(hyper, f, indent=2)
    print("[THV5] DIST classi:", hyper["CLASS_PCT"], flush=True)
    print(f"[THV5] FATTO: {used} segmenti, {len(all_lb)} frame. Parti: {len(parts_lb)}", flush=True)


if __name__ == "__main__":
    main()
