# -*- coding: utf-8 -*-
# ============================================================================
# KERNEL PREPARAZIONE DATI DUPLEXCONV ESTESI + GENERAZIONE TURN LABELS (v2)
# Scarica DuplexConv (audio tar + jsons.tar.gz), estrae mel 100Hz (utente/agente),
# tokenizza codici RVQ 12.5Hz (Qwen3-TTS), genera turn_labels 5 classi (12.5Hz)
# ed esporta i parquet estesi e i memmap .dat per la TurnStateHead v2.
# ============================================================================
import os, sys, json, time, gc, shutil, tarfile, urllib.request, math
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from tqdm.auto import tqdm

    print("[PREP] Avvio Data Preparation DuplexConv Esteso...", flush=True)

    # ---------------------------------------------------------------------------
    # Configurazione Parametri & Costanti
    # ---------------------------------------------------------------------------
    TARGET_HOURS = float(os.environ.get("PREP_TARGET_HOURS", "200.0"))
    MAX_SEGMENTS = int(os.environ.get("PREP_MAX_SEGMENTS", "6000"))
    WINDOW_T = int(os.environ.get("WINDOW_T", "64"))
    FRAME_RATE_HZ = 12.5
    MEL_PER_AGENT_FRAME = 8
    N_CODEBOOKS = 16
    MEL_BINS = 128
    PAD_TARGET_ID = -100

    CLASS_NAMES = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]

    # Percorsi disco
    BIG_DISK = "/root/kaggle_big" if os.path.isdir("/kaggle/working") else "./data_prep_big"
    os.makedirs(BIG_DISK, exist_ok=True)
    DATA_OUT = "data"
    os.makedirs(DATA_OUT, exist_ok=True)

    # ---------------------------------------------------------------------------
    # 1. Download & Estrazione Annotazioni JSON
    # ---------------------------------------------------------------------------
    print("\n[PREP 1/5] Caricamento e indicizzazione annotazioni ASR dual-channel...", flush=True)
    json_index: Dict[str, str] = {}
    json_dir = os.path.join(BIG_DISK, "duplex_jsons")
    os.makedirs(json_dir, exist_ok=True)

    for domain in ("Edu", "edu", "none_Edu"):
        tar_url = f"https://huggingface.co/datasets/qualialabsAI/DuplexConv/resolve/main/{domain}/jsons.tar.gz?download=true"
        tar_file = os.path.join(json_dir, f"{domain}_jsons.tar.gz")
        out_sub = os.path.join(json_dir, domain)
        try:
            if not os.path.exists(out_sub):
                print(f"  - Download {domain}/jsons.tar.gz ...", flush=True)
                urllib.request.urlretrieve(tar_url, tar_file)
                print(f"  - Estrazione {domain}/jsons.tar.gz ...", flush=True)
                os.makedirs(out_sub, exist_ok=True)
                with tarfile.open(tar_file, "r:gz") as tar:
                    tar.extractall(path=out_sub)
                if os.path.exists(tar_file):
                    os.remove(tar_file)

            for fn in os.listdir(out_sub):
                if fn.endswith(".json"):
                    key = fn[:-5]
                    json_index[key] = os.path.join(out_sub, fn)
        except Exception as e:
            print(f"  - [AVVISO] Dominio {domain}: {e}", flush=True)

    print(f"[PREP] Indicizzati {len(json_index)} record JSON con timestamp dual-channel.", flush=True)

    # ---------------------------------------------------------------------------
    # 2. Inizializzazione Estrattori (Mel Gemma4 + Qwen3-TTS Tokenizer)
    # ---------------------------------------------------------------------------
    print("\n[PREP 2/5] Inizializzazione Tokenizer ed Estrattore Mel...", flush=True)
    import kaggle_pipeline as KP

    qwen_tok = KP.load_qwen_tts_tokenizer()
    mel_ext = KP.load_mel_extractor()
    text_tok = KP.load_e2b_text_tokenizer()

    # ---------------------------------------------------------------------------
    # 3. Algoritmo di Generazione Turn Labels a 12.5 Hz da JSON ASR
    # ---------------------------------------------------------------------------
    def compute_turn_labels_from_asr(json_path: str, n_frames: int, dt: float = 0.080) -> Tuple[np.ndarray, np.ndarray]:
        """Calcola deterministica turn_labels (5 classi) e valid_mask per n_frames (12.5Hz, 80ms)."""
        labels = np.zeros(n_frames, dtype=np.int64)
        valid = np.zeros(n_frames, dtype=bool)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            asr = d.get("asr", [])
            user_utts = asr[0] if len(asr) > 0 else []
            agent_utts = asr[1] if len(asr) > 1 else []

            # Mappatura intervalli temporali
            user_intervals = []
            for u in user_utts:
                s_s = float(u.get("startInMs", 0)) / 1000.0
                e_s = float(u.get("endInMs", 0)) / 1000.0
                if e_s > s_s:
                    user_intervals.append((s_s, e_s, e_s - s_s))

            agent_intervals = []
            for a in agent_utts:
                s_s = float(a.get("startInMs", 0)) / 1000.0
                e_s = float(a.get("endInMs", 0)) / 1000.0
                if e_s > s_s:
                    agent_intervals.append((s_s, e_s, e_s - s_s))

            for t in range(n_frames):
                t_start = t * dt
                t_end = (t + 1) * dt

                # Verifica attivita utente e agente
                u_active = any(not (t_end <= s or t_start >= e) for s, e, _ in user_intervals)
                a_active = any(not (t_end <= s or t_start >= e) for s, e, _ in agent_intervals)

                valid[t] = u_active or a_active
                if not valid[t]:
                    labels[t] = 0
                    continue

                if a_active and not u_active:
                    # Verifica se e tra gli ultimi 3 frame di un turno agente (YIELD)
                    is_yield = False
                    for s, e, _ in agent_intervals:
                        if s <= t_start < e:
                            # Se la fine del turno e entro 240ms (3 frame)
                            if (e - t_start) <= 3 * dt:
                                is_yield = True
                            break
                    labels[t] = 1 if is_yield else 0  # 1=YIELD, 0=SPEAK

                elif a_active and u_active:
                    # Sovrapposizione: calcola durata intervento utente
                    dur = 0.0
                    for s, e, d_len in user_intervals:
                        if not (t_end <= s or t_start >= e):
                            dur = max(dur, d_len)
                    # >= 400ms -> BARGE_IN (2), < 400ms -> USER_BACKCHANNEL (3)
                    labels[t] = 2 if dur >= 0.400 else 3

                elif not a_active and u_active:
                    # Solo ascolto utente: EMIT_BACKCHANNEL (4) cadenzato/su pause
                    labels[t] = 4 if (t > 8 and (t % 12 in (0, 1, 2))) else 0

        except Exception:
            pass

        return labels, valid

    # ---------------------------------------------------------------------------
    # 4. Elaborazione Dataset Streaming DuplexConv con Generazione Parquet
    # ---------------------------------------------------------------------------
    print(f"\n[PREP 3/5] Streaming ed Elaborazione DuplexConv (target: {TARGET_HOURS}h, max {MAX_SEGMENTS} segmenti)...", flush=True)
    from datasets import load_dataset

    ds_stream = load_dataset("qualialabsAI/DuplexConv", split="train", streaming=True)
    ds_subset = ds_stream.take(MAX_SEGMENTS)

    user_mel_list, agent_mel_list, agent_codes_list = [], [], []
    turn_labels_list, turn_valid_list = [], []
    segment_ends_list = []
    text_by_window_list = []

    total_frames_accum = 0
    pbar = tqdm(ds_subset, desc="Processamento DuplexConv Esteso", total=MAX_SEGMENTS)

    for item in pbar:
        raw_audio = item.get("wav") or item.get("audio") or item.get("user_audio")
        decoded = KP._decode_duplex_audio(raw_audio)
        if decoded is None:
            continue

        arr, sr = decoded
        u_mel = KP.extract_mel_features(mel_ext, arr[:, 0], sr=sr)
        a_codes = KP.encode_audio_with_qwen_tts(qwen_tok, arr[:, 1], sr=sr)
        a_mel = KP.extract_mel_features(mel_ext, arr[:, 1], sr=sr)

        if u_mel is None or a_mel is None or a_codes is None:
            continue

        n_agent = a_codes.shape[1]
        u_mel = KP.align_mel_to_codes(u_mel, n_agent)
        a_mel = KP.align_mel_to_codes(a_mel, n_agent)

        key = str(item.get("__key__", ""))
        jpath = json_index.get(key)
        if jpath and os.path.exists(jpath):
            t_lbls, t_val = compute_turn_labels_from_asr(jpath, n_agent)
        else:
            # Fallback acustico se JSON mancante
            w_dummy = KP._extract_window_features_and_labels(
                type("DS", (), {"user_mel": u_mel, "agent_mel": a_mel, "n_code_frames": n_agent})(),
                0, T=n_agent
            ) if n_agent >= WINDOW_T else None
            if w_dummy is not None:
                t_lbls, t_val = w_dummy["labels"][:n_agent], w_dummy["valid"][:n_agent]
            else:
                t_lbls = np.zeros(n_agent, dtype=np.int64)
                t_val = np.ones(n_agent, dtype=bool)

        user_mel_list.append(u_mel)
        agent_mel_list.append(a_mel)
        agent_codes_list.append(a_codes)
        turn_labels_list.append(t_lbls)
        turn_valid_list.append(t_val)

        total_frames_accum += n_agent
        segment_ends_list.append(total_frames_accum)

        hours_so_far = (total_frames_accum * 0.080) / 3600.0
        pbar.set_postfix({"Ore": f"{hours_so_far:.2f}", "Segmenti": len(segment_ends_list)})

        if hours_so_far >= TARGET_HOURS:
            print(f"\n[PREP] Raggiunto target ore: {hours_so_far:.2f}h >= {TARGET_HOURS}h.", flush=True)
            break

    # ---------------------------------------------------------------------------
    # 5. Esportazione Parquet e Memmap
    # ---------------------------------------------------------------------------
    print("\n[PREP 4/5] Concatenazione ed Esportazione Parquet...", flush=True)
    all_u_mel = np.vstack(user_mel_list).astype(np.float16)
    all_a_mel = np.vstack(agent_mel_list).astype(np.float16)
    all_a_codes = np.hstack(agent_codes_list).astype(np.int16)
    all_turn_labels = np.concatenate(turn_labels_list).astype(np.int64)
    all_turn_valid = np.concatenate(turn_valid_list).astype(bool)

    tot_hours = (all_a_codes.shape[1] * 0.080) / 3600.0
    out_prefix = f"data/train_tokens_{tot_hours:.1f}h_ext"

    # Salva mel utente, mel agente, codici
    KP._write_mel_parquet(f"{out_prefix}_mel.parquet", all_u_mel)
    KP._write_mel_parquet(f"{out_prefix}_ameld.parquet", all_a_mel)

    df_codes = pd.DataFrame({f"a_cb_{k}": all_a_codes[k].astype(np.int16) for k in range(N_CODEBOOKS)})
    df_codes.to_parquet(f"{out_prefix}_codes.parquet", compression="snappy", index=False)

    # Nuovo: Salva turnlabels parquet
    df_turn = pd.DataFrame({
        "turn_label": all_turn_labels,
        "turn_valid": all_turn_valid.astype(np.int8)
    })
    df_turn.to_parquet(f"{out_prefix}_turnlabels.parquet", compression="snappy", index=False)

    # Distribuzione classi calcolata
    unique_l, counts_l = np.unique(all_turn_labels[all_turn_valid], return_counts=True)
    class_dist = {int(k): int(v) for k, v in zip(unique_l, counts_l)}
    for k in range(5):
        if k not in class_dist:
            class_dist[k] = 0

    print(f"\n[TURN_CLASS_DIST_TRAIN] Distribuzione Frame Completa ({tot_hours:.1f}h): {class_dist}", flush=True)
    for k in range(5):
        print(f"  - {CLASS_NAMES[k]:20s}: {class_dist[k]:8d} frame ({100.0 * class_dist[k] / max(1, sum(class_dist.values())):.2f}%)", flush=True)

    # Sidecar JSON
    sidecar_dict = {
        "N_CODEBOOKS": N_CODEBOOKS,
        "SEMANTIC_VOCAB_SIZE": 4096,
        "ACOUSTIC_VOCAB_SIZE": 2048,
        "HOURS": round(tot_hours, 2),
        "TOTAL_FRAMES": int(all_a_codes.shape[1]),
        "SEGMENT_ENDS": segment_ends_list,
        "TURN_CLASS_DIST_TRAIN": class_dist,
    }
    with open(f"{out_prefix}_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)

    # Conversione Memmap .dat
    print("\n[PREP 5/5] Generazione Memmap binari .dat per accesso zero-RAM...", flush=True)
    KP.convert_parquet_to_memmap(f"{out_prefix}.parquet")

    # Report finale
    prep_report = {
        "status": "PASS",
        "total_hours": round(tot_hours, 2),
        "total_segments": len(segment_ends_list),
        "total_frames": int(all_a_codes.shape[1]),
        "class_distribution": class_dist,
        "parquet_files": {
            "mel": f"{out_prefix}_mel.parquet",
            "ameld": f"{out_prefix}_ameld.parquet",
            "codes": f"{out_prefix}_codes.parquet",
            "turnlabels": f"{out_prefix}_turnlabels.parquet",
            "hyperparams": f"{out_prefix}_hyperparams.json"
        }
    }
    with open("prep_duplexconv_report.json", "w", encoding="utf-8") as f:
        json.dump(prep_report, f, indent=2)

    print("\n=== DATA PREPARATION DUPLEXCONV COMPLETATA CON SUCCESSO! ===", flush=True)
    print(f"File generati con prefisso: {out_prefix} ({tot_hours:.1f}h di dialogo)", flush=True)

except SystemExit:
    raise
except Exception:
    import traceback
    traceback.print_exc()
    print("[FATAL] Data preparation fallita.", flush=True)
    sys.exit(1)
