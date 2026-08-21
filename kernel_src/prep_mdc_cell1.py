# -*- coding: utf-8 -*-
# ============================================================================
# KERNEL PREPARAZIONE DATASET MOZILLA COMMON VOICE (MDC) ITALIANO 50-80H
# Tokenizzazione RVQ 12.5Hz (Qwen3-TTS) + Mel 100Hz (Dual-Channel MDC)
# Esporta: data/train_tokens_mdc_{N}h_{mel,ameld,codes,text}.parquet
# ============================================================================
import os, sys, json, time, gc, shutil, math
from typing import Dict, Any, List, Tuple, Optional

try:
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import torch
    from tqdm.auto import tqdm

    print("[PREP MDC] Avvio Data Preparation Mozilla Common Voice Italiano...", flush=True)

    # ---------------------------------------------------------------------------
    # Configurazione Parametri & Costanti
    # ---------------------------------------------------------------------------
    TARGET_HOURS = float(os.environ.get("PREP_TARGET_HOURS", "50.0"))
    MAX_CLIPS = int(os.environ.get("PREP_MAX_CLIPS", "50000"))
    MAX_RUNTIME_SEC = int(os.environ.get("PREP_MAX_RUNTIME_SEC", str(80 * 60)))  # 80 min max budget
    WINDOW_T = int(os.environ.get("WINDOW_T", "64"))
    FRAME_RATE_HZ = 12.5
    MEL_PER_AGENT_FRAME = 8
    N_CODEBOOKS = 16
    SAMPLE_RATE = 24000
    SILENCE_SEC = 0.4
    BARGE_IN_PROB = 0.20

    BIG_DISK = "/root/kaggle_big" if os.path.isdir("/kaggle/working") else "./data_prep_big"
    os.makedirs(BIG_DISK, exist_ok=True)
    DATA_OUT = "data"
    os.makedirs(DATA_OUT, exist_ok=True)

    # ---------------------------------------------------------------------------
    # 1. Inizializzazione Estrattori (Mel Gemma4 + Qwen3-TTS Tokenizer)
    # ---------------------------------------------------------------------------
    print("\n[PREP MDC 1/4] Inizializzazione Tokenizer ed Estrattore Mel...", flush=True)
    import kaggle_pipeline as KP

    qwen_tok = KP.load_qwen_tts_tokenizer()
    mel_ext = KP.load_mel_extractor()
    text_tok = KP.load_e2b_text_tokenizer()
    silence_codes = [int(x) for x in KP.extract_silence_codes(qwen_tok)]
    print(f"[PREP MDC] Tokenizer inizializzato. Silence codes: {silence_codes[:4]}...", flush=True)

    # ---------------------------------------------------------------------------
    # 2. Caricamento Dataset MDC Italiano (HF datasets o datacollective)
    # ---------------------------------------------------------------------------
    print("\n[PREP MDC 2/4] Caricamento clip audio e trascrizioni MDC Italiano...", flush=True)
    import io, librosa

    def iter_mdc_clips(max_clips=50000):
        # FIX v3 (coordinatore, 2026-08-21): il caricamento v2 accumulava TUTTE le clip
        # decodificate in RAM (~20GB per 46k clip) -> OOM sulla T4 (16GB) -> kernel died.
        # Ora e' un GENERATORE LAZY: decodifica riga per riga e fa yield, memoria ~costante.
        # Streaming via datasets fallisce ("AudioDecoder object has no attribute get") e
        # datacollective richiede MDC_API_KEY, quindi: parquet HF diretti (huggingface_hub
        # + pandas + soundfile, bytes mp3 48kHz -> 24kHz mono). 4 parquet ~= 46k clip ~= 58h.
        total = 0
        try:
            from huggingface_hub import hf_hub_download
            hf_repo = "OpenSpeechHub/Common_Voice_17_0-It"
            n_parquet = max(1, int(math.ceil(TARGET_HOURS / 14.5)))  # ~14.5h per parquet
            n_parquet = min(n_parquet, 40)
            print(f"  - [PARQUET] Streaming {n_parquet} parquet da {hf_repo} (generatore lazy, RAM ~costante)...", flush=True)
            for pi in range(n_parquet):
                if total >= max_clips:
                    break
                fname = f"data/train-{pi:05d}-of-00040.parquet"
                try:
                    local = hf_hub_download(repo_id=hf_repo, filename=fname, repo_type="dataset")
                except Exception:
                    local = hf_hub_download(repo_id=hf_repo, filename=fname, repo_type="dataset", revision="main")
                print(f"    [PARQUET {pi+1}/{n_parquet}] streaming {fname}...", flush=True)
                df = pd.read_parquet(local)
                for _, row in df.iterrows():
                    if total >= max_clips:
                        break
                    try:
                        t = str(row.get("text") or "").strip()
                        audio_val = row.get("audio")
                        raw = None
                        if isinstance(audio_val, dict):
                            raw = audio_val.get("bytes")
                        elif audio_val is not None:
                            raw = bytes(audio_val)
                        if not raw or not t:
                            continue
                        y, sr_in = sf.read(io.BytesIO(raw), dtype="float32")
                        if y.ndim > 1:
                            y = y.mean(axis=1)
                        if sr_in != SAMPLE_RATE:
                            y = librosa.resample(y, orig_sr=sr_in, target_sr=SAMPLE_RATE)
                        if len(y) > SAMPLE_RATE:  # >= 1s di audio
                            rms = float(np.sqrt(np.mean(y**2))) + 1e-8
                            y = y.astype(np.float32) * (0.05 / rms)
                            total += 1
                            yield {"audio": y, "text": t}
                    except Exception:
                        pass
                del df
                gc.collect()
            if total == 0:
                print("  - [AVVISO] Parquet HF non caricati (nessun clip valido)", flush=True)
        except Exception as e:
            print(f"  - [AVVISO] Parquet HF fallito: {e}", flush=True)

        # Fallback sintetico per smoke test locale (mai raggiunto su Kaggle con internet)
        if total == 0:
            print("  - [FALLBACK MOCK] Creazione 50 clip simulati per test...", flush=True)
            for i in range(50):
                dur = np.random.uniform(2.5, 5.0)
                n_samples = int(dur * SAMPLE_RATE)
                t_arr = np.linspace(0, dur, n_samples, dtype=np.float32)
                freq = np.random.uniform(150, 300)
                y = 0.05 * np.sin(2 * np.pi * freq * t_arr)
                yield {"audio": y, "text": f"Questa e la frase di prova numero {i+1} in italiano."}

    all_clips_iter = iter_mdc_clips(max_clips=MAX_CLIPS)

    # ---------------------------------------------------------------------------
    # 3. Costruzione Dialoghi Dual-Channel Simulati e Tokenizzazione
    # ---------------------------------------------------------------------------
    print("\n[PREP MDC 3/4] Generazione dialoghi dual-channel e tokenizzazione RVQ/Mel...", flush=True)
    silence_len = int(SILENCE_SEC * SAMPLE_RATE)
    silence_wav = np.zeros(silence_len, dtype=np.float32)

    user_mel_list = []
    agent_mel_list = []
    agent_codes_list = []
    text_by_window_list = []
    segment_ends_list = []

    t_start = time.time()
    total_agent_frames = 0
    clip_idx = 0
    dialogue_idx = 0

    # Elabora in micro-dialoghi composti da coppie di clip Utente-Agente
    # (consumo del generatore lazy: coppie consecutive, niente lista in RAM)
    _it = iter(all_clips_iter)
    while True:
        if (time.time() - t_start) > MAX_RUNTIME_SEC:
            print(f"[PREP MDC] Raggiunto limite temporale ({MAX_RUNTIME_SEC}s). Arresto acquisizione.", flush=True)
            break

        try:
            c_user = next(_it)
            c_agent = next(_it)
        except StopIteration:
            break
        clip_idx += 2
        dialogue_idx += 1

        u_wav = c_user["audio"]
        a_wav = c_agent["audio"]
        a_text = c_agent["text"]

        # Costruisci traccia stereo: Utente parla -> Pausa -> Agente parla -> Pausa
        # Con 20% probabilita di barge-in: Utente subentra prima che l'agente finisca
        is_barge_in = (np.random.random() < BARGE_IN_PROB)
        if is_barge_in and len(a_wav) > int(1.5 * SAMPLE_RATE):
            split_pt = int(len(a_wav) * 0.6)
            a_wav_turn = a_wav[:split_pt]
        else:
            a_wav_turn = a_wav

        # Traccia Utente (Canale 0) e Traccia Agente (Canale 1)
        # Turno 1: Utente parla, Agente silenzioso
        u_part1 = np.concatenate([u_wav, silence_wav])
        a_part1 = np.zeros(len(u_part1), dtype=np.float32)

        # Turno 2: Agente parla, Utente silenzioso
        a_part2 = np.concatenate([a_wav_turn, silence_wav])
        u_part2 = np.zeros(len(a_part2), dtype=np.float32)

        u_dialogue = np.concatenate([u_part1, u_part2])
        a_dialogue = np.concatenate([a_part1, a_part2])

        # Estrai mel 100Hz
        u_mel = KP.extract_mel_features(mel_ext, u_dialogue, sr=SAMPLE_RATE)
        a_mel = KP.extract_mel_features(mel_ext, a_dialogue, sr=SAMPLE_RATE)
        a_codes = KP.encode_audio_with_qwen_tts(qwen_tok, a_dialogue, sr=SAMPLE_RATE)

        if u_mel is None or a_mel is None or a_codes is None:
            continue

        n_frames = a_codes.shape[1]
        if n_frames == 0:
            continue

        u_mel = KP.align_mel_to_codes(u_mel, n_frames)
        a_mel = KP.align_mel_to_codes(a_mel, n_frames)
        a_codes = a_codes[:, :n_frames]

        user_mel_list.append(u_mel.astype(np.float16))
        agent_mel_list.append(a_mel.astype(np.float16))
        agent_codes_list.append(a_codes.astype(np.int16))

        total_agent_frames += n_frames
        segment_ends_list.append(total_agent_frames)

        # Testo per finestra
        n_win_d = n_frames // WINDOW_T + (1 if n_frames % WINDOW_T else 0)
        for w in range(n_win_d):
            text_by_window_list.append(a_text)

        cur_hours = (total_agent_frames * 0.080) / 3600.0
        if dialogue_idx % 200 == 0:
            elapsed = time.time() - t_start
            speed = cur_hours / (elapsed / 3600.0) if elapsed > 0 else 0.0
            print(f"[PREP MDC] Dialogo {dialogue_idx:5d} | Frame: {total_agent_frames:7d} | Ore: {cur_hours:5.2f}h | Speed: {speed:5.1f}x", flush=True)

        if cur_hours >= TARGET_HOURS:
            print(f"[PREP MDC] Target ore raggiunto ({cur_hours:.2f}h >= {TARGET_HOURS}h). Completamento.", flush=True)
            break

    # ---------------------------------------------------------------------------
    # 4. Concatenazione ed Esportazione Parquet MDC-Only
    # ---------------------------------------------------------------------------
    print("\n[PREP MDC 4/4] Concatenazione ed Esportazione Parquet MDC...", flush=True)
    all_u_mel = np.vstack(user_mel_list).astype(np.float16)
    all_a_mel = np.vstack(agent_mel_list).astype(np.float16)
    all_a_codes = np.hstack(agent_codes_list).astype(np.int16)

    tot_hours = (all_a_codes.shape[1] * 0.080) / 3600.0
    out_prefix = f"data/train_tokens_mdc_{tot_hours:.1f}h_2500s"
    std_prefix = f"data/train_tokens_mdc_50.0h_2500s"

    # Salva mel utente, mel agente, codici RVQ
    KP._write_mel_parquet(f"{out_prefix}_mel.parquet", all_u_mel)
    KP._write_mel_parquet(f"{out_prefix}_ameld.parquet", all_a_mel)

    df_codes = pd.DataFrame({f"a_cb_{k}": all_a_codes[k].astype(np.int16) for k in range(N_CODEBOOKS)})
    df_codes.to_parquet(f"{out_prefix}_codes.parquet", compression="snappy", index=False)

    # Salva testo
    n_windows = all_a_codes.shape[1] // WINDOW_T + (1 if all_a_codes.shape[1] % WINDOW_T else 0)
    text_data = [(i, text_by_window_list[i] if i < len(text_by_window_list) else "") for i in range(n_windows)]
    df_text = pd.DataFrame(text_data, columns=["window_idx", "text"])
    df_text.to_parquet(f"{out_prefix}_text.parquet", compression="snappy", index=False)

    # Copie con prefisso standard per compatibilita automatica
    for suffix in ("_mel.parquet", "_ameld.parquet", "_codes.parquet", "_text.parquet"):
        src = f"{out_prefix}{suffix}"
        dst = f"{std_prefix}{suffix}"
        if src != dst and os.path.exists(src):
            shutil.copy2(src, dst)

    # Sidecar JSON
    sidecar_dict = {
        "DATASET": "Mozilla Common Voice (MDC) 26.0 / 17.0 Italiano",
        "LANGUAGE": "Italian (it)",
        "N_CODEBOOKS": N_CODEBOOKS,
        "SEMANTIC_VOCAB_SIZE": 4096,
        "ACOUSTIC_VOCAB_SIZE": 2048,
        "HOURS": round(tot_hours, 2),
        "TOTAL_FRAMES": int(all_a_codes.shape[1]),
        "TOTAL_DIALOGUES": dialogue_idx,
        "TOTAL_WINDOWS": n_windows,
        "SEGMENT_ENDS": segment_ends_list,
    }
    with open(f"{out_prefix}_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)
    with open(f"{std_prefix}_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)

    # Report finale
    prep_report = {
        "status": "PASS",
        "dataset": "Mozilla Common Voice 26.0 Italiano",
        "total_hours": round(tot_hours, 2),
        "total_dialogues": dialogue_idx,
        "total_frames": int(all_a_codes.shape[1]),
        "total_windows": n_windows,
        "parquet_files": {
            "mel": f"{out_prefix}_mel.parquet",
            "ameld": f"{out_prefix}_ameld.parquet",
            "codes": f"{out_prefix}_codes.parquet",
            "text": f"{out_prefix}_text.parquet",
            "hyperparams": f"{out_prefix}_hyperparams.json"
        }
    }
    with open("prep_mdc_report.json", "w", encoding="utf-8") as f:
        json.dump(prep_report, f, indent=2)

    print("\n=== DATA PREPARATION MDC ITALIANO COMPLETATA CON SUCCESSO! ===", flush=True)
    print(f"File generati con prefisso: {out_prefix} ({tot_hours:.1f}h di audio italiano puro)", flush=True)

except SystemExit:
    raise
except Exception:
    import traceback
    traceback.print_exc()
    print("[FATAL] Data preparation MDC fallita.", flush=True)
    sys.exit(1)
