#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prep_mdc_v2_3090.py -- High-Quality Italian Speech Data Preparation v2 on RTX 3090.
Builds full-duplex Italian conversational dataset from Common Voice 17-it with:
  1. Energy VAD silence trimming (start & end) + clip filtering (>0.8s speech, RMS > 0.01)
  2. Pure PyTorch resampling (24kHz for neural codec, 16kHz for Gemma mel extractor)
  3. Qwen3-TTS neural tokenizer on GPU (16 RVQ codebooks @ 12.5 Hz)
  4. Gemma4 Log-Mel Feature Extractor (128 bins @ 100 Hz)
  5. Compact natural dialogue turn packing (0.15-0.25s pauses, <5% constant frames)
  6. Streaming Parquet export with zero-copy memmap conversion
"""

import argparse
import gc
import glob
import io
import json
import math
import os
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

sys.path.insert(0, "/workspace/.pylibs")
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import paths
import kaggle_pipeline as KP


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MDC Italian Data Preparation v2 on RTX 3090."
    )
    parser.add_argument(
        "--target-hours",
        type=float,
        default=float(os.environ.get("MDC_TARGET_HOURS", "15.0")),
        help="Target total dialogue hours to preprocess (default: 15.0)",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=int(os.environ.get("MDC_MAX_CLIPS", "25000")),
        help="Maximum dataset clips to process (default: 25000)",
    )
    parser.add_argument(
        "--max-runtime-sec",
        type=int,
        default=int(os.environ.get("MDC_MAX_RUNTIME_SEC", str(90 * 60))),
        help="Maximum wall-clock execution time in seconds (default: 5400 / 90m)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=int(os.environ.get("MDC_CHUNK_SIZE", "150")),
        help="Number of dialogue segments per disk chunk (default: 150)",
    )
    parser.add_argument(
        "--barge-in-prob",
        type=float,
        default=0.20,
        help="Probability of user interrupting agent (default: 0.20)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke data preparation (0.05h / 25 clips)",
    )
    return parser.parse_args()


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D float32 audio using pure PyTorch linear interpolation."""
    if orig_sr == target_sr:
        return y.astype(np.float32)
    t = torch.from_numpy(y).view(1, 1, -1).float()
    new_len = int(round(len(y) * target_sr / orig_sr))
    out = F.interpolate(t, size=new_len, mode="linear", align_corners=False)
    return out.view(-1).numpy().astype(np.float32)


def apply_energy_vad(y: np.ndarray, sr: int, top_db: float = 30.0) -> Optional[np.ndarray]:
    """Trim leading and trailing silence using energy envelope."""
    if len(y) == 0:
        return None
    frame_len = int(sr * 0.020)  # 20ms
    hop_len = int(sr * 0.010)    # 10ms
    if len(y) < frame_len:
        return None

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_len)[::hop_len]
    frame_rms = np.sqrt(np.mean(frames**2, axis=-1)) + 1e-9
    peak_rms = np.max(frame_rms)
    if peak_rms < 0.005:
        return None

    threshold = max(0.002, peak_rms * (10 ** (-top_db / 20.0)))
    active_idx = np.where(frame_rms > threshold)[0]
    if len(active_idx) == 0:
        return None

    start_sample = max(0, int(active_idx[0] * hop_len))
    end_sample = min(len(y), int(active_idx[-1] * hop_len + frame_len))

    trimmed = y[start_sample:end_sample]
    if len(trimmed) < int(0.8 * sr):  # Minimum 0.8s of real voice
        return None

    # Normalize RMS to 0.05
    rms = np.sqrt(np.mean(trimmed**2)) + 1e-8
    trimmed = trimmed * (0.05 / rms)
    return trimmed.astype(np.float32)


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.target_hours = 0.05
        args.max_clips = 30
        args.max_runtime_sec = 180
        args.chunk_size = 5
        print(f"[PREP MDC v2] === SMOKE TEST ATTIVO: {args.target_hours}h, max {args.max_clips} clip ===")

    print("=" * 78)
    print("=== DATA PREPARATION MOZILLA COMMON VOICE v2 (VOCI REALI) SU RTX 3090 ===")
    print("=" * 78)
    print(f"Target Ore: {args.target_hours:.1f}h")
    print(f"Max Clip: {args.max_clips}")
    print(f"Max Runtime: {args.max_runtime_sec // 60}m ({args.max_runtime_sec}s)")
    print(f"Barge-In Prob: {int(args.barge_in_prob * 100)}%")
    print(f"Chunk Size: {args.chunk_size} turni")
    sys.stdout.flush()

    # 1. Environment & Paths
    dirs = paths.ensure_all_dirs()
    data_out = dirs["data"]
    chunks_dir = os.path.join(data_out, "chunks_mdc_v2")
    os.makedirs(chunks_dir, exist_ok=True)

    # 2. Tokenizers Initialization (Once!)
    print("\n[PREP 1/4] Inizializzazione estrattori e tokenizer...", flush=True)
    qwen_tok = KP.load_qwen_tts_tokenizer()
    mel_ext = KP.load_mel_extractor()
    txt_tok = KP.load_e2b_text_tokenizer()

    if qwen_tok is None or mel_ext is None:
        raise RuntimeError("[PREP FATAL] Impossibile caricare Qwen3-TTS tokenizer o Mel extractor.")

    # 3. Stream & Process Common Voice Shards
    print("\n[PREP 2/4] Elaborazione streaming clip italiani e dialogo dual-channel...", flush=True)
    repo_id = "OpenSpeechHub/Common_Voice_17_0-It"
    total_shards = 40
    # Natural pause: 0.15s
    pause_24k = np.zeros(int(0.15 * 24000), dtype=np.float32)
    pause_16k = np.zeros(int(0.15 * 16000), dtype=np.float32)

    user_mel_list: List[np.ndarray] = []
    agent_mel_list: List[np.ndarray] = []
    agent_codes_list: List[np.ndarray] = []
    turn_labels_list: List[np.ndarray] = []
    turn_valid_list: List[np.ndarray] = []
    text_by_window_list: List[Any] = []
    segment_ends_list: List[int] = []

    total_frames_accum = 0
    total_clips_processed = 0
    barge_in_count = 0
    chunk_counter = 0

    prev_clip_24k: Optional[np.ndarray] = None
    prev_clip_16k: Optional[np.ndarray] = None
    prev_text: str = ""

    t_start_loop = time.time()
    pbar = tqdm(total=args.max_clips, desc="Elaborazione MDC v2 Italiano", unit="clip")

    def _flush_current_chunk(c_idx: int) -> None:
        nonlocal user_mel_list, agent_mel_list, agent_codes_list
        nonlocal turn_labels_list, turn_valid_list
        if not user_mel_list:
            return

        c_u_mel = np.vstack(user_mel_list).astype(np.float16)
        c_a_mel = np.vstack(agent_mel_list).astype(np.float16)
        c_codes = np.hstack(agent_codes_list).astype(np.int16)
        c_tl = np.concatenate(turn_labels_list).astype(np.int64)
        c_tv = np.concatenate(turn_valid_list).astype(bool)

        df_u = pd.DataFrame({f"mel_{k}": c_u_mel[:, k] for k in range(c_u_mel.shape[1])})
        df_a = pd.DataFrame({f"mel_{k}": c_a_mel[:, k] for k in range(c_a_mel.shape[1])})
        df_c = pd.DataFrame({f"a_cb_{k}": c_codes[k, :] for k in range(c_codes.shape[0])})
        df_t = pd.DataFrame({"turn_label": c_tl, "turn_valid": c_tv})

        df_u.to_parquet(os.path.join(chunks_dir, f"chunk_{c_idx:05d}_mel.parquet"), compression="snappy", index=False)
        df_a.to_parquet(os.path.join(chunks_dir, f"chunk_{c_idx:05d}_ameld.parquet"), compression="snappy", index=False)
        df_c.to_parquet(os.path.join(chunks_dir, f"chunk_{c_idx:05d}_codes.parquet"), compression="snappy", index=False)
        df_t.to_parquet(os.path.join(chunks_dir, f"chunk_{c_idx:05d}_turnlabels.parquet"), compression="snappy", index=False)

        user_mel_list.clear()
        agent_mel_list.clear()
        agent_codes_list.clear()
        turn_labels_list.clear()
        turn_valid_list.clear()
        del c_u_mel, c_a_mel, c_codes, c_tl, c_tv, df_u, df_a, df_c, df_t
        gc.collect()

    stop_processing = False
    for shard_idx in range(total_shards):
        if stop_processing:
            break

        shard_filename = f"data/train-{shard_idx:05d}-of-{total_shards:05d}.parquet"
        print(f"\n[PREP v2] Download/Accesso shard HF: {shard_filename} ...", flush=True)
        try:
            local_shard_path = hf_hub_download(
                repo_id=repo_id,
                filename=shard_filename,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"[PREP v2 ERROR] Impossibile scaricare {shard_filename}: {e}", flush=True)
            break

        shard_table = pq.read_table(local_shard_path)
        shard_rows = len(shard_table)
        print(f"[PREP v2] Shard {shard_idx} caricato ({shard_rows} righe).", flush=True)

        audio_col = shard_table.column("audio").to_pylist()
        text_col = shard_table.column("text").to_pylist()
        del shard_table
        gc.collect()

        for row_idx in range(shard_rows):
            raw_item = audio_col[row_idx]
            raw_bytes = raw_item.get("bytes") if isinstance(raw_item, dict) else None
            clip_text = str(text_col[row_idx] or "").strip()

            if not raw_bytes:
                continue

            # Decode raw audio
            try:
                y_raw, sr = sf.read(io.BytesIO(raw_bytes))
                if y_raw.ndim > 1:
                    y_raw = np.mean(y_raw, axis=-1)
            except Exception:
                continue

            # Energy VAD at native SR
            y_trimmed = apply_energy_vad(y_raw, sr=sr, top_db=30.0)
            if y_trimmed is None:
                continue

            # Resample to 24kHz (for Qwen3 Tokenizer) and 16kHz (for Gemma Mel Extractor)
            y_24k = resample_audio(y_trimmed, orig_sr=sr, target_sr=24000)
            y_16k = resample_audio(y_trimmed, orig_sr=sr, target_sr=16000)

            total_clips_processed += 1
            is_barge_in = (
                (np.random.rand() < args.barge_in_prob)
                and (total_clips_processed > 1)
                and (prev_clip_24k is not None)
            )

            # Build dual-channel dialogue turn
            if is_barge_in:
                barge_in_count += 1
                overlap_samples_24k = int(len(y_24k) * 0.5)
                overlap_samples_16k = int(len(y_16k) * 0.5)

                user_24k = np.concatenate([y_24k, pause_24k])
                user_16k = np.concatenate([y_16k, pause_16k])

                agent_24k = np.zeros(len(user_24k), dtype=np.float32)
                agent_16k = np.zeros(len(user_16k), dtype=np.float32)

                prev_piece_24k = prev_clip_24k[:overlap_samples_24k] if len(prev_clip_24k) >= overlap_samples_24k else np.resize(prev_clip_24k, overlap_samples_24k)
                prev_piece_16k = prev_clip_16k[:overlap_samples_16k] if len(prev_clip_16k) >= overlap_samples_16k else np.resize(prev_clip_16k, overlap_samples_16k)

                agent_24k[:overlap_samples_24k] = prev_piece_24k * 0.8
                agent_16k[:overlap_samples_16k] = prev_piece_16k * 0.8
                turn_agent_text = prev_text
            else:
                agent_24k = np.concatenate([y_24k, pause_24k])
                agent_16k = np.concatenate([y_16k, pause_16k])

                if prev_clip_24k is not None:
                    user_piece_24k = prev_clip_24k if len(prev_clip_24k) <= len(agent_24k) else prev_clip_24k[:len(agent_24k)]
                    user_piece_16k = prev_clip_16k if len(prev_clip_16k) <= len(agent_16k) else prev_clip_16k[:len(agent_16k)]
                    user_24k = np.zeros(len(agent_24k), dtype=np.float32)
                    user_16k = np.zeros(len(agent_16k), dtype=np.float32)
                    user_24k[:len(user_piece_24k)] = user_piece_24k * 0.7
                    user_16k[:len(user_piece_16k)] = user_piece_16k * 0.7
                else:
                    user_24k = np.zeros(len(agent_24k), dtype=np.float32)
                    user_16k = np.zeros(len(agent_16k), dtype=np.float32)

                turn_agent_text = clip_text

            # Feature extraction
            u_mel = KP.extract_mel_features(mel_ext, user_16k, sr=16000)
            a_mel = KP.extract_mel_features(mel_ext, agent_16k, sr=16000)
            a_codes = KP.encode_audio_with_qwen_tts(qwen_tok, agent_24k, sr=24000)

            if u_mel is None or a_mel is None or a_codes is None:
                continue

            n_agent = a_codes.shape[1]
            u_mel = KP.align_mel_to_codes(u_mel, n_agent)
            a_mel = KP.align_mel_to_codes(a_mel, n_agent)

            # Turn labels
            labels = np.zeros(n_agent, dtype=np.int64)
            valid = np.ones(n_agent, dtype=bool)
            if is_barge_in:
                overlap_f = max(1, int(n_agent * 0.5))
                labels[:overlap_f] = 2  # BARGE_IN
                labels[overlap_f:] = 0  # SPEAK
            else:
                labels[:] = 0  # SPEAK

            # Tokenize text
            if txt_tok and turn_agent_text:
                try:
                    t_enc = txt_tok.encode(turn_agent_text, add_special_tokens=False)
                    text_ids = list(t_enc)
                except Exception:
                    text_ids = None
            else:
                text_ids = None

            w_idx_start = total_frames_accum // 64
            w_idx_end = (total_frames_accum + n_agent) // 64
            while len(text_by_window_list) <= w_idx_end + 1:
                text_by_window_list.append(None)
            for widx in range(w_idx_start, w_idx_end + 1):
                if text_by_window_list[widx] is None and text_ids:
                    text_by_window_list[widx] = text_ids

            user_mel_list.append(u_mel)
            agent_mel_list.append(a_mel)
            agent_codes_list.append(a_codes)
            turn_labels_list.append(labels)
            turn_valid_list.append(valid)

            total_frames_accum += n_agent
            segment_ends_list.append(total_frames_accum)

            prev_clip_24k = y_24k
            prev_clip_16k = y_16k
            prev_text = clip_text

            hours_so_far = (total_frames_accum * 0.080) / 3600.0
            elapsed_so_far = time.time() - t_start_loop
            pbar.update(1)
            pbar.set_postfix({
                "Ore": f"{hours_so_far:.2f}",
                "Clip": total_clips_processed,
                "BargeIn": barge_in_count,
                "Min": f"{elapsed_so_far/60.0:.1f}",
            })

            # Periodic disk flush
            if len(user_mel_list) >= args.chunk_size:
                _flush_current_chunk(chunk_counter)
                chunk_counter += 1

            if hours_so_far >= args.target_hours:
                print(f"\n[PREP v2] Raggiunto target ore: {hours_so_far:.2f}h >= {args.target_hours:.2f}h.", flush=True)
                stop_processing = True
                break

            if total_clips_processed >= args.max_clips:
                print(f"\n[PREP v2] Raggiunto limite clip: {total_clips_processed} >= {args.max_clips}.", flush=True)
                stop_processing = True
                break

            if elapsed_so_far >= args.max_runtime_sec:
                print(f"\n[PREP v2] Raggiunto limite runtime ({elapsed_so_far/60.0:.1f}m >= {args.max_runtime_sec//60}m).", flush=True)
                stop_processing = True
                break

    pbar.close()

    if user_mel_list:
        _flush_current_chunk(chunk_counter)
        chunk_counter += 1

    if chunk_counter == 0 or total_frames_accum == 0:
        raise RuntimeError("[PREP v2 FATAL] Nessun dato generato. Controlla parametri o accesso ai dati.")

    # 4. Merge Parquet Chunks
    tot_hours = (total_frames_accum * 0.080) / 3600.0
    print(f"\n[PREP v2 3/4] Streaming Merge Parquet da {chunk_counter} chunk ({tot_hours:.2f}h totali)...", flush=True)
    out_prefix = os.path.join(data_out, f"train_tokens_mdc_v2_{tot_hours:.1f}h_{total_clips_processed}s")

    def _merge_chunks_streaming(pattern: str, out_path: str):
        cfiles = sorted(glob.glob(pattern))
        if not cfiles:
            raise RuntimeError(f"Nessun file chunk trovato per pattern: {pattern}")
        first_t = pq.read_table(cfiles[0])
        with pq.ParquetWriter(out_path, first_t.schema, compression="snappy") as writer:
            writer.write_table(first_t)
            del first_t
            gc.collect()
            for cf in cfiles[1:]:
                tbl = pq.read_table(cf)
                writer.write_table(tbl)
                del tbl
                gc.collect()
        size_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"  - Scritto: {out_path} ({size_mb:.1f} MB)", flush=True)

    mel_path = f"{out_prefix}_mel.parquet"
    ameld_path = f"{out_prefix}_ameld.parquet"
    codes_path = f"{out_prefix}_codes.parquet"
    turnlabels_path = f"{out_prefix}_turnlabels.parquet"
    text_path = f"{out_prefix}_text.parquet"

    _merge_chunks_streaming(os.path.join(chunks_dir, "chunk_*_mel.parquet"), mel_path)
    _merge_chunks_streaming(os.path.join(chunks_dir, "chunk_*_ameld.parquet"), ameld_path)
    _merge_chunks_streaming(os.path.join(chunks_dir, "chunk_*_codes.parquet"), codes_path)
    _merge_chunks_streaming(os.path.join(chunks_dir, "chunk_*_turnlabels.parquet"), turnlabels_path)

    # Text Parquet
    n_windows = total_frames_accum // 64 + (1 if total_frames_accum % 64 else 0)
    aligned_text = (text_by_window_list[:n_windows] + [None] * n_windows)[:n_windows]
    df_text = pd.DataFrame([(i, aligned_text[i]) for i in range(n_windows)], columns=["window_idx", "text"])
    df_text.to_parquet(text_path, compression="snappy", index=False)
    print(f"  - Scritto: {text_path} ({n_windows} finestre, {sum(1 for t in aligned_text if t is not None)} con testo)", flush=True)

    # Cleanup temp chunks
    shutil.rmtree(chunks_dir, ignore_errors=True)

    # 5. Quality Verification & Sidecar JSON
    print("\n[PREP v2 4/4] Analisi salute target, codebook e copie standard...", flush=True)
    tbl_codes = pq.read_table(codes_path)
    cb0 = tbl_codes["a_cb_0"].to_numpy()
    u_vals, u_counts = np.unique(cb0, return_counts=True)
    top_share = float(u_counts.max() / max(1, u_counts.sum()))
    print(f"\n[MDC_v2_CB0_HEALTH] Codebook 0: {len(u_vals)} token unici su {len(cb0)} frame, top token al {top_share:.1%}", flush=True)

    tbl_a_mel = pq.read_table(ameld_path)
    a_mel_arr = tbl_a_mel.to_pandas().values
    a_stds = np.std(a_mel_arr, axis=1)
    const_ratio = float(np.mean(a_stds < 1e-4))
    print(f"[MDC_v2_MEL_HEALTH] Frazione Frame Mel Costanti: {const_ratio * 100:.2f}% (Target: < 5.0%)")

    # Sidecar JSON
    sidecar_dict = {
        "N_CODEBOOKS": 16,
        "SEMANTIC_VOCAB_SIZE": 4096,
        "ACOUSTIC_VOCAB_SIZE": 2048,
        "HOURS": round(tot_hours, 2),
        "TOTAL_FRAMES": int(total_frames_accum),
        "TOTAL_CLIPS": int(total_clips_processed),
        "BARGE_IN_CLIPS": int(barge_in_count),
        "SEGMENT_ENDS": segment_ends_list,
        "TEXT_BLOCK_LEN": 32,
        "TEXT_LOSS_WEIGHT": 0.5,
        "VAL_FRACTION": 0.1,
        "TOKENIZER_REAL_ENCODING_ACTIVE": True,
        "CONSTANT_MEL_RATIO": round(const_ratio, 4),
    }
    sidecar_file = f"{out_prefix}_hyperparams.json"
    with open(sidecar_file, "w", encoding="utf-8") as f:
        json.dump(sidecar_dict, f, indent=2)
    print(f"[PREP v2] Sidecar JSON salvato: {sidecar_file}", flush=True)

    # Standard prefix copies (train_tokens_mdc_50.0h_2500s_*)
    std_prefix = os.path.join(data_out, "train_tokens_mdc_50.0h_2500s")
    for ext in ("_mel.parquet", "_ameld.parquet", "_codes.parquet", "_turnlabels.parquet", "_text.parquet", "_hyperparams.json"):
        src = f"{out_prefix}{ext}"
        dst = f"{std_prefix}{ext}"
        if os.path.exists(src) and src != dst:
            shutil.copy2(src, dst)
            print(f"  - Copia standard: {os.path.basename(dst)}", flush=True)

    # REMOVE OLD .dat CACHE
    for dat_f in ["mel_user.dat", "mel_agent.dat", "codes.dat"]:
        dat_p = os.path.join(data_out, dat_f)
        if os.path.exists(dat_p):
            os.remove(dat_p)
            print(f"  - Rimosso vecchio memmap cache: {dat_f}", flush=True)

    print("\n" + "=" * 78)
    print(f"=== DATA PREPARATION MDC v2 COMPLETATA ({tot_hours:.2f}h, const mel {const_ratio*100:.2f}%) ===")
    print("=" * 78)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FATAL] MDC v2 Data preparation failed: {e}")
        sys.exit(1)
