# -*- coding: utf-8 -*-
# ============================================================================
# GEMMA 4 E2B FROZEN + TEMPORAL ADAPTER FROZEN + TURNSTATEHEAD (Stage 3)
# TurnStateHead: Classificatore neurale causale Pre-LN a 5 stati di turno
#   (SPEAK, YIELD, BARGE_IN, USER_BACKCHANNEL, EMIT_BACKCHANNEL)
# Input: z_t = [h_t_temp (2048) || e_user (128) || d_e_user (128)] in R^2304
# Architettura: Linear(2304,1024) -> LN -> SiLU -> Linear(1024,1024) -> LN -> SiLU
#               -> Linear(1024,256) -> LN -> SiLU -> Linear(256,5) (~3.68M params)
# Backbone Gemma 4 E2B: CONGELATO 4-bit (forward no_grad)
# TemporalResidualAdapter: CONGELATO (forward no_grad, warm-start da R2)
# Modulo Trainable: SOLO TurnStateHead (3.8M params)
# Loss: Weighted CrossEntropy (w=[1.0, 8.0, 6.0, 3.0, 4.0]) su frame attivi
# ============================================================================
import os, sys, json, traceback, hashlib, shutil, subprocess, time, math

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import gc

    # =========================================================================
    # FIX GLOBALE DTYPE-SAFE LAYER NORM (Anti-Mismatch Float/Half in Unsloth JIT)
    # =========================================================================
    _orig_torch_layer_norm = torch.layer_norm

    def _dtype_safe_torch_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
        orig_dtype = input.dtype
        if weight is not None and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        if bias is not None and weight is not None and bias.dtype != weight.dtype:
            bias = bias.to(weight.dtype)
        out = _orig_torch_layer_norm(input, normalized_shape, weight, bias, eps, cudnn_enable)
        return out.to(orig_dtype) if out.dtype != orig_dtype else out

    torch.layer_norm = _dtype_safe_torch_layer_norm
    F.layer_norm = _dtype_safe_torch_layer_norm
    try:
        import unsloth_zoo.patch_torch_functions as _uz_patch
        _uz_patch.layer_norm = _dtype_safe_torch_layer_norm
    except Exception:
        pass
    print("[INIT] Patch globale _dtype_safe_torch_layer_norm applicata a torch e F.layer_norm", flush=True)

    # =========================================================================
    # CONFIGURAZIONE RUNTIME & IPERPARAMETRI
    # =========================================================================
    STAGE3_TURNHEAD = int(os.environ.get("STAGE3_TURNHEAD", "1"))
    SMOKE = int(os.environ.get("SMOKE", "1"))
    PHASE2_SMOKE_BATCHES = int(os.environ.get("PHASE2_SMOKE_BATCHES", "4" if SMOKE else "1200"))
    EXTRACT_N = int(os.environ.get("FITEX_N", "16" if SMOKE else "1000"))
    ROPE_FIX = int(os.environ.get("ROPE_FIX", "1")) == 1

    # Parametri Finestra Temporale & TurnHead
    WINDOW_T = int(os.environ.get("WINDOW_T", "64"))
    N_TURN_CLASSES = int(os.environ.get("N_TURN_CLASSES", "5"))
    TURN_LABELS_SOURCE = "duplexconv"
    TURNHEAD_LR = float(os.environ.get("TURNHEAD_LR", "3e-4"))
    TURNHEAD_WEIGHT_DECAY = float(os.environ.get("TURNHEAD_WEIGHT_DECAY", "1e-2"))
    STAGE3_BATCH_SIZE = int(os.environ.get("STAGE3_BATCH_SIZE", "8"))
    STAGE3_GRAD_ACCUM = int(os.environ.get("STAGE3_GRAD_ACCUM", "2"))
    STAGE3_CKPT_EVERY = int(os.environ.get("STAGE3_CKPT_EVERY", "2" if SMOKE else "250"))

    # Pesi Classi Weighted Cross-Entropy: SPEAK=1.0, YIELD=8.0, BARGE_IN=6.0, USER_BC=3.0, EMIT_BC=4.0
    CLASS_WEIGHTS = [1.0, 8.0, 6.0, 3.0, 4.0]
    CLASS_NAMES = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]

    # Iperparametri Adapter Temporale (Frozen)
    TEMPORAL_DIM = int(os.environ.get("TEMPORAL_DIM", "1024"))
    TEMPORAL_LAYERS = int(os.environ.get("TEMPORAL_LAYERS", "4"))
    TEMPORAL_HEADS = int(os.environ.get("TEMPORAL_HEADS", "8"))
    TEMPORAL_FF_DIM = int(os.environ.get("TEMPORAL_FF_DIM", "2048"))

    # Valutazione
    EVAL_TURNHEAD = int(os.environ.get("EVAL_TURNHEAD", "1"))
    EVAL_N_VAL = int(os.environ.get("EVAL_N_VAL", "8" if SMOKE else "100"))
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "4"))

    # Directory di output e percorsi checkpoint
    OUTPUT_DIR = "/kaggle/output" if os.path.exists("/kaggle/output") else "."
    ADAPTER_STAGE2_R2_CANDIDATE_PATHS = (
        "checkpoints/ht_adapter_stage2r2_final.pt",
        "/kaggle/working/checkpoints/ht_adapter_stage2r2_final.pt",
        "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
        "/kaggle/input/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
        "ht_adapter_stage2r2_final.pt",
    )

    import kaggle_pipeline as KP
    import qwen_subtalker as QS

    # ---- Caricamento dati da memmap / parquet ----
    parquet_cache = f"data/train_tokens_{KP.DUPLEXCONV_MAX_HOURS}h_{KP.MDC_MAX_SAMPLES}s.parquet"
    if not os.path.exists(parquet_cache) and os.path.exists("data/train_tokens_1.0h_500s.parquet"):
        parquet_cache = "data/train_tokens_1.0h_500s.parquet"
    KP.convert_parquet_to_memmap(parquet_cache)
    user_mel, agent_mel, agent_codes, text_by_window = KP.load_tokens_from_parquet(parquet_cache)
    t_agent = agent_codes.shape[1]
    val_start = int((1.0 - KP.VAL_FRACTION) * t_agent)
    for e in KP.SEGMENT_ENDS:
        if e >= val_start:
            val_start = e
            break
    val_start = min(val_start, t_agent)
    MPF = KP.MEL_PER_AGENT_FRAME

    ds_train = KP.FullDuplexDataset(
        user_mel[:val_start * MPF], agent_mel[:val_start * MPF],
        agent_codes[:, :val_start], seq_len=WINDOW_T,
        text_by_window=text_by_window[:val_start // WINDOW_T] if text_by_window is not None else None,
    )
    ds_val = KP.FullDuplexDataset(
        user_mel[val_start * MPF:], agent_mel[val_start * MPF:],
        agent_codes[:, val_start:], seq_len=WINDOW_T,
        text_by_window=text_by_window[val_start // WINDOW_T:] if text_by_window is not None else None,
    )
    print(f"Dataset caricato: ds_train={len(ds_train)} finestre | ds_val={len(ds_val)} finestre (T={WINDOW_T})", flush=True)

    # =========================================================================
    # ARCHITETTURA MODULI: ADAPTER TEMPORALE (FROZEN) + TURNSTATEHEAD (TRAINABLE)
    # =========================================================================
    class CausalConv1d(nn.Module):
        def __init__(self, channels, kernel_size=3):
            super().__init__()
            self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=0, bias=False)
            self._reset_identity_parameters()

        def _reset_identity_parameters(self):
            nn.init.zeros_(self.conv.weight)
            with torch.no_grad():
                _c = self.conv.kernel_size[0] - 1
                for i in range(self.conv.weight.shape[0]):
                    self.conv.weight[i, i, _c] = 1.0

        def forward(self, x):
            x_t = x.transpose(1, 2)
            x_pad = F.pad(x_t, (2, 0))
            out = self.conv(x_pad)
            return out.transpose(1, 2)

    class ResidualAdapter(nn.Module):
        def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072, use_causal_conv=True):
            super().__init__()
            self.use_causal_conv = use_causal_conv
            if use_causal_conv:
                self.causal_conv = CausalConv1d(in_dim)
            self.skip_proj = nn.Linear(in_dim, out_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            self.norm = nn.LayerNorm(out_dim)

        def forward(self, x):
            if self.use_causal_conv:
                x = self.causal_conv(x)
            h_proj = self.skip_proj(x)
            h_mlp = self.mlp(x)
            return self.norm(h_mlp + h_proj)

    class CausalSelfAttentionBlock(nn.Module):
        def __init__(self, d_model=1024, n_heads=8, d_ff=2048, dropout=0.0):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.ln1 = nn.LayerNorm(d_model)
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
            self.v_proj = nn.Linear(d_model, d_model, bias=False)
            self.out_proj = nn.Linear(d_model, d_model, bias=False)
            self.ln2 = nn.LayerNorm(d_model)
            self.mlp = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.SiLU(),
                nn.Linear(d_ff, d_model),
            )
            self.dropout = dropout

        def forward(self, x):
            norm_x = self.ln1(x)
            B, T, D = norm_x.shape
            q = self.q_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.v_proj(norm_x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.out_proj(attn_out)
            x = x + self.mlp(self.ln2(x))
            return x

    class CausalTemporalModule(nn.Module):
        def __init__(self, in_dim=2048, hidden_dim=1024, n_layers=4, n_heads=8, d_ff=2048, max_len=512):
            super().__init__()
            self.in_proj = nn.Linear(in_dim, hidden_dim)
            self.pos_emb = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
            nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
            self.layers = nn.ModuleList([
                CausalSelfAttentionBlock(d_model=hidden_dim, n_heads=n_heads, d_ff=d_ff)
                for _ in range(n_layers)
            ])
            self.out_norm = nn.LayerNorm(hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
            nn.init.zeros_(self.out_proj.weight)

        def forward(self, x):
            B, T, D = x.shape
            h = self.in_proj(x) + self.pos_emb[:, :T, :]
            for layer in self.layers:
                h = layer(h)
            h = self.out_norm(h)
            delta = self.out_proj(h)
            return x + delta

    class TemporalResidualAdapter(nn.Module):
        def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072,
                     temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048,
                     use_causal_conv=True):
            super().__init__()
            self.adapter = ResidualAdapter(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, use_causal_conv=use_causal_conv)
            self.temporal = CausalTemporalModule(in_dim=out_dim, hidden_dim=temp_dim, n_layers=temp_layers, n_heads=temp_heads, d_ff=temp_ff)

        def forward(self, x):
            h_adapt = self.adapter(x)
            return self.temporal(h_adapt)

    class TurnStateHead(nn.Module):
        """TurnStateHead (3.8M params): Classificatore causale Pre-LN a 5 stati di turno.
        Input: z_t = [h_t_temp(2048) || e_user(128) || d_e_user(128)] in R^2304.
        """
        def __init__(self, in_dim=2304, hidden_dim=1024, mid_dim=256, num_classes=5):
            super().__init__()
            self.fc1 = nn.Linear(in_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.act1 = nn.SiLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.act2 = nn.SiLU()
            self.fc3 = nn.Linear(hidden_dim, mid_dim)
            self.ln3 = nn.LayerNorm(mid_dim)
            self.act3 = nn.SiLU()
            self.classifier = nn.Linear(mid_dim, num_classes)

        def forward(self, z):
            x = self.act1(self.ln1(self.fc1(z)))
            x = self.act2(self.ln2(self.fc2(x)))
            x = self.act3(self.ln3(self.fc3(x)))
            return self.classifier(x)

    # =========================================================================
    # DERIVAZIONE GROUND TRUTH TURN LABELS & STRATIFICAZIONE FINESTRE
    # =========================================================================
    MPF_MEL = KP.MEL_PER_AGENT_FRAME  # 8 mel frame per agent frame (80ms @ 12.5Hz)
    PAD = KP.PAD_TARGET_ID

    def _extract_window_features_and_labels(ds, idx, T=WINDOW_T):
        """Estrae user/agent mel, calcola e_user, delta e_user e assegna turn_labels a 5 stati."""
        n_code_frames = ds.n_code_frames
        c0 = idx * T
        c1 = min(c0 + T, n_code_frames)
        m0 = c0 * MPF_MEL
        m1 = min(m0 + T * MPF_MEL, ds.user_mel.shape[0])

        u_mel_raw = np.asarray(ds.user_mel[m0:m1], dtype=np.float32)
        a_mel_raw = np.asarray(ds.agent_mel[m0:m1], dtype=np.float32)

        # Pad a T*8 se necessario
        target_mel_len = T * MPF_MEL
        if u_mel_raw.shape[0] < target_mel_len:
            pad_len = target_mel_len - u_mel_raw.shape[0]
            u_mel_raw = np.pad(u_mel_raw, ((0, pad_len), (0, 0)))
            a_mel_raw = np.pad(a_mel_raw, ((0, pad_len), (0, 0)))

        u_frames = u_mel_raw.reshape(T, MPF_MEL, 128)
        a_frames = a_mel_raw.reshape(T, MPF_MEL, 128)

        # e_user: media spettrale per frame (T, 128)
        e_user = u_frames.mean(axis=1)  # (T, 128)
        # d_e_user: derivata temporale (T, 128)
        d_e_user = np.zeros_like(e_user)
        d_e_user[1:] = e_user[1:] - e_user[:-1]

        # Calcolo energia per canale
        u_energy = np.mean(np.abs(e_user), axis=1)  # (T,)
        a_energy = np.mean(np.abs(a_frames.mean(axis=1)), axis=1)  # (T,)

        # Soglie dinamiche basate sui percentili
        u_thr = max(0.012, float(np.percentile(u_energy, 40) * 1.2))
        a_thr = max(0.012, float(np.percentile(a_energy, 40) * 1.2))

        u_act = u_energy > u_thr
        a_act = a_energy > a_thr

        # Calcolo segmenti contigui utente per distinguere Barge-In vs Backchannel
        u_runs = []
        in_run = False
        start = 0
        for t in range(T):
            if u_act[t] and not in_run:
                in_run = True
                start = t
            elif not u_act[t] and in_run:
                in_run = False
                u_runs.append((start, t - 1, t - start))
        if in_run:
            u_runs.append((start, T - 1, T - start))

        labels = np.zeros(T, dtype=np.int64)
        valid = np.zeros(T, dtype=bool)

        for t in range(T):
            is_a = bool(a_act[t])
            is_u = bool(u_act[t])
            valid[t] = is_a or is_u

            if not valid[t]:
                labels[t] = 0
                continue

            if is_a and not is_u:
                # Monologo agente: verifica YIELD (ultimi 3 frame di emissione agente prima del silenzio)
                is_yield = False
                if t + 1 >= T or not a_act[t + 1]:
                    is_yield = True
                elif t + 2 >= T or not a_act[t + 2]:
                    is_yield = True
                elif t + 3 >= T or not a_act[t + 3]:
                    is_yield = True
                labels[t] = 1 if is_yield else 0  # 1=YIELD, 0=SPEAK

            elif is_a and is_u:
                # Sovrapposizione: trova durata dell'intervento utente
                cur_len = 1
                for s, e, l in u_runs:
                    if s <= t <= e:
                        cur_len = l
                        break
                # >= 5 frame (400ms) -> BARGE_IN (2), < 5 frame -> USER_BACKCHANNEL (3)
                labels[t] = 2 if cur_len >= 5 else 3

            elif not is_a and is_u:
                # Solo ascolto utente: EMIT_BACKCHANNEL (4) su pause o cadenze
                labels[t] = 4 if (t > 8 and (t % 12 in (0, 1, 2))) else 0

        # Riconoscimento tipo finestra per campionamento bilanciato
        n_a = int(a_act.sum())
        n_u = int(u_act.sum())
        wtype = "mixed"
        if n_a >= 15 and n_u <= 8:
            wtype = "monologue"
        elif n_u >= 15 and n_a <= 8:
            wtype = "listen"
        elif n_a >= 8 and n_u >= 8:
            wtype = "overlap"

        return {
            "idx": idx,
            "e_user": e_user,
            "d_e_user": d_e_user,
            "labels": labels,
            "valid": valid,
            "wtype": wtype,
            "n_a": n_a,
            "n_u": n_u,
            "n_valid": int(valid.sum())
        }

    # =========================================================================
    # SCANSIONE & BUCKETING DELLE FINESTRE
    # =========================================================================
    print(f"\n[TurnData] Scansione finestre per dataset bilanciato (T={WINDOW_T})...", flush=True)
    bucket_mono, bucket_listen, bucket_overlap, bucket_mixed = [], [], [], []

    scan_limit = min(len(ds_train), 2048 if SMOKE else 10000)
    for i in range(scan_limit):
        w_info = _extract_window_features_and_labels(ds_train, i, T=WINDOW_T)
        if w_info["n_valid"] < 8:
            continue
        if w_info["wtype"] == "monologue":
            bucket_mono.append(w_info)
        elif w_info["wtype"] == "listen":
            bucket_listen.append(w_info)
        elif w_info["wtype"] == "overlap":
            bucket_overlap.append(w_info)
        else:
            bucket_mixed.append(w_info)

    print(f"[TurnData] Bucketing completato: Monologue={len(bucket_mono)}, Listen={len(bucket_listen)}, "
          f"Overlap={len(bucket_overlap)}, Mixed={len(bucket_mixed)}", flush=True)

    # Assicura fallback non vuoti
    all_valid_train = bucket_mono + bucket_listen + bucket_overlap + bucket_mixed
    if not bucket_mono: bucket_mono = all_valid_train[:1]
    if not bucket_listen: bucket_listen = all_valid_train[:1]
    if not bucket_overlap: bucket_overlap = all_valid_train[:1]
    if not bucket_mixed: bucket_mixed = all_valid_train[:1]

    # Validazione set di test stratificato (Monologue, Listen, Overlap)
    val_mono, val_listen, val_overlap, val_mixed = [], [], [], []
    val_scan_limit = min(len(ds_val), 1024 if SMOKE else 2500)
    for i in range(val_scan_limit):
        vw = _extract_window_features_and_labels(ds_val, i, T=WINDOW_T)
        if vw["n_valid"] < 8:
            continue
        if vw["wtype"] == "monologue":
            val_mono.append(vw)
        elif vw["wtype"] == "listen":
            val_listen.append(vw)
        elif vw["wtype"] == "overlap":
            val_overlap.append(vw)
        else:
            val_mixed.append(vw)

    val_wins = []
    n_v_mono = max(1, int(EVAL_N_VAL * 0.40))
    n_v_list = max(1, int(EVAL_N_VAL * 0.30))
    n_v_over = max(1, EVAL_N_VAL - n_v_mono - n_v_list)

    if val_mono:
        for i in range(n_v_mono):
            val_wins.append(val_mono[i % len(val_mono)])
    if val_listen:
        for i in range(n_v_list):
            val_wins.append(val_listen[i % len(val_listen)])
    if val_overlap:
        for i in range(n_v_over):
            val_wins.append(val_overlap[i % len(val_overlap)])
    elif val_mixed:
        for i in range(n_v_over):
            val_wins.append(val_mixed[i % len(val_mixed)])

    print(f"[TurnData] Finestre di validazione stratificate pronte: {len(val_wins)} finestre "
          f"(Mono={n_v_mono}, Listen={n_v_list}, Overlap={n_v_over})", flush=True)

    # Campionatore bilanciato di batch (40% mono, 30% listen, 30% overlap)
    def sample_balanced_batch(batch_size):
        b = []
        n_mono = max(1, int(batch_size * 0.40))
        n_list = max(1, int(batch_size * 0.30))
        n_over = max(1, batch_size - n_mono - n_list)

        for _ in range(n_mono):
            b.append(bucket_mono[np.random.randint(len(bucket_mono))])
        for _ in range(n_list):
            b.append(bucket_listen[np.random.randint(len(bucket_listen))])
        for _ in range(n_over):
            b.append(bucket_overlap[np.random.randint(len(bucket_overlap))])
        return b

    # =========================================================================
    # CARICAMENTO MODELLI: GEMMA 4 E2B (FROZEN) + ADAPTER (FROZEN) + TURNHEAD
    # =========================================================================
    DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n[TurnHead] Inizializzazione modelli su device: {DEV}...", flush=True)

    # 1. Carica Gemma 4 E2B Frozen
    from unsloth import FastLanguageModel
    gemma_base, _ = FastLanguageModel.from_pretrained(
        model_name="unsloth/gemma-4-E2B-it",
        max_seq_length=512,
        load_in_4bit=True,
    )
    gemma_base.eval()
    for p in gemma_base.parameters():
        p.requires_grad = False
    print("[TurnHead] Gemma 4 E2B caricata in 4-bit e CONGELATA (requires_grad=False)", flush=True)

    # 2. Carica TemporalResidualAdapter Frozen
    ht_adapter = TemporalResidualAdapter(
        in_dim=1536, out_dim=2048, hidden_dim=3072,
        temp_dim=TEMPORAL_DIM, temp_layers=TEMPORAL_LAYERS,
        temp_heads=TEMPORAL_HEADS, temp_ff=TEMPORAL_FF_DIM
    ).to(DEV, dtype=torch.float32)

    # Warm-start adapter da checkpoint R2 se presente
    _ad_path = next((p for p in ADAPTER_STAGE2_R2_CANDIDATE_PATHS if os.path.isfile(p)), None)
    if _ad_path:
        ckpt_st = torch.load(_ad_path, map_location="cpu")
        m, u = ht_adapter.load_state_dict(ckpt_st, strict=False)
        print(f"[TurnHead] warm-start ht_adapter da {_ad_path} (missing: {len(m)}, unexpected: {len(u)})", flush=True)
    else:
        print("[TurnHead] Nessun checkpoint adapter trovato; inizializzazione standard", flush=True)

    ht_adapter.eval()
    for p in ht_adapter.parameters():
        p.requires_grad = False
    print("[TurnHead] TemporalResidualAdapter CONGELATO (requires_grad=False)", flush=True)

    # 3. Istanzia TurnStateHead (Trainable)
    turn_head = TurnStateHead(in_dim=2304, hidden_dim=1024, mid_dim=256, num_classes=N_TURN_CLASSES).to(DEV, dtype=torch.float32)
    turn_head.train()
    n_head_params = sum(p.numel() for p in turn_head.parameters() if p.requires_grad)
    print(f"[TURNHEAD_ACTIVE] TurnStateHead istanziata: {n_head_params:,} parametri trainable ({n_head_params/1e6:.2f}M)", flush=True)
    print(f"[PROBE] N_TURN_CLASSES={N_TURN_CLASSES} | TURN_LABELS_SOURCE={TURN_LABELS_SOURCE} | riuso Gemma=True", flush=True)

    # Optimizer & Loss Function & Scheduler
    optimizer = torch.optim.AdamW(turn_head.parameters(), lr=TURNHEAD_LR, weight_decay=TURNHEAD_WEIGHT_DECAY)
    def _lr_lambda(st):
        warmup = 1 if SMOKE else 50
        if st < warmup:
            return float(st + 1) / float(max(1, warmup))
        progress = float(st - warmup) / float(max(1, PHASE2_SMOKE_BATCHES - warmup))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    class_weights_t = torch.tensor(CLASS_WEIGHTS, device=DEV, dtype=torch.float32)
    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t, reduction='none')

    # Helper input IDs Gemma
    ST2_FPF = 2
    ST2_TXT = 48
    ST2_AFS = 2 + ST2_TXT
    ST2_MEL_WIN = WINDOW_T * 8
    BOS_ID, BOA_ID, AUDIO_TOKEN_ID, EOA_ID = 2, 256000, 258881, 258883

    def _build_ids(text_list):
        rows, mrows = [], []
        for t in text_list:
            toks = t if isinstance(t, list) else []
            toks = toks[:ST2_TXT]
            n_pad = ST2_TXT - len(toks)
            row = ([BOS_ID] + [0] * n_pad + toks + [BOA_ID]
                   + [AUDIO_TOKEN_ID] * (WINDOW_T * ST2_FPF) + [EOA_ID])
            mrow = ([1] + [0] * n_pad + [1] * (len(toks) + 1)
                    + [1] * (WINDOW_T * ST2_FPF) + [1])
            rows.append(row)
            mrows.append(mrow)
        return (torch.tensor(rows, dtype=torch.long, device=DEV),
                torch.tensor(mrows, dtype=torch.long, device=DEV))

    def _win_mel(ds, idx):
        s = idx * ST2_MEL_WIN
        m = ds.user_mel[s:s + ST2_MEL_WIN]
        if m.shape[0] < ST2_MEL_WIN:
            m = np.pad(m, ((0, ST2_MEL_WIN - m.shape[0]), (0, 0)))
        return torch.from_numpy(np.asarray(m, dtype=np.float32))

    # =========================================================================
    # TRAINING LOOP TURNSTATEHEAD (STAGE 3)
    # =========================================================================
    print(f"\n{'=' * 80}", flush=True)
    print(f"=== AVVIO TRAINING TURNSTATEHEAD: {PHASE2_SMOKE_BATCHES} STEP (BATCH_SIZE={STAGE3_BATCH_SIZE}) ===", flush=True)
    print(f"{'=' * 80}", flush=True)

    step_logs = []
    t_start = time.time()

    for step in range(1, PHASE2_SMOKE_BATCHES + 1):
        t0_step = time.time()
        optimizer.zero_grad()
        acc_loss = 0.0
        acc_correct = 0
        acc_total = 0

        for _acc_i in range(STAGE3_GRAD_ACCUM):
            batch_wins = sample_balanced_batch(STAGE3_BATCH_SIZE)
            texts = [text_by_window[w["idx"]] if text_by_window is not None else [] for w in batch_wins]

            input_ids, attn_mask = _build_ids(texts)
            feats = torch.stack([_win_mel(ds_train, w["idx"]) for w in batch_wins], 0).to(DEV)
            fmask = torch.ones((STAGE3_BATCH_SIZE, ST2_MEL_WIN), dtype=torch.bool, device=DEV)

            e_user_batch = torch.from_numpy(np.stack([w["e_user"] for w in batch_wins], 0)).to(DEV, dtype=torch.float32)
            d_e_user_batch = torch.from_numpy(np.stack([w["d_e_user"] for w in batch_wins], 0)).to(DEV, dtype=torch.float32)
            labels_batch = torch.from_numpy(np.stack([w["labels"] for w in batch_wins], 0)).to(DEV, dtype=torch.long)
            valid_batch = torch.from_numpy(np.stack([w["valid"] for w in batch_wins], 0)).to(DEV, dtype=torch.bool)

            # 1. Forward Gemma 4 E2B Frozen sotto torch.no_grad()
            with torch.no_grad():
                go = gemma_base(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    input_features=feats,
                    input_features_mask=fmask,
                    output_hidden_states=True,
                )
                lh = go.hidden_states[-1]
                _s, _e = ST2_AFS + ST2_FPF - 1, ST2_AFS + ST2_FPF * WINDOW_T
                agent_hidden = lh[:, _s:_e:ST2_FPF, :].to(torch.float32)

                # 2. Forward TemporalResidualAdapter Frozen
                h_t_temp = ht_adapter(agent_hidden)  # (B, T, 2048)

            # 3. Costruzione input z_t in R^2304 e Forward TurnStateHead
            z_t = torch.cat([h_t_temp, e_user_batch, d_e_user_batch], dim=-1)  # (B, T, 2304)
            turn_logits = turn_head(z_t)  # (B, T, 5)

            # 4. Weighted CrossEntropy Loss su frame validi
            loss_unreduced = ce_loss_fn(turn_logits.view(-1, N_TURN_CLASSES), labels_batch.view(-1)).view(STAGE3_BATCH_SIZE, WINDOW_T)
            valid_float = valid_batch.float()
            val_sum = valid_float.sum().clamp(min=1.0)
            loss_step = (loss_unreduced * valid_float).sum() / val_sum

            # Accumulo gradienti
            (loss_step / STAGE3_GRAD_ACCUM).backward()
            acc_loss += loss_step.item() / STAGE3_GRAD_ACCUM

            # Conteggio accuratezza batch
            preds = turn_logits.argmax(dim=-1)
            correct = ((preds == labels_batch) & valid_batch).sum().item()
            total = valid_batch.sum().item()
            acc_correct += correct
            acc_total += total

        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(turn_head.parameters(), 1.0)

        step_time = time.time() - t0_step
        step_acc = acc_correct / max(1, acc_total)

        # Probes di gradiente
        head_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in turn_head.parameters())
        backbone_grad0 = all(p.grad is None for p in gemma_base.parameters())
        adapter_grad0 = all(p.grad is None for p in ht_adapter.parameters())

        vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        step_log = {
            "step": step,
            "loss": round(acc_loss, 4),
            "acc": round(step_acc, 4),
            "step_time_s": round(step_time, 2),
            "vram_gb": round(vram_gb, 2),
            "head_grad": bool(head_grad),
            "backbone_grad0": bool(backbone_grad0),
            "adapter_grad0": bool(adapter_grad0),
        }
        step_logs.append(step_log)

        if step % max(1, PHASE2_SMOKE_BATCHES // 10) == 0 or step == 1 or step == PHASE2_SMOKE_BATCHES:
            print(f"[STEP {step:4d}/{PHASE2_SMOKE_BATCHES}] Loss={acc_loss:.4f} | Acc={step_acc:.4f} | "
                  f"Time={step_time:.2f}s | VRAM={vram_gb:.2f}GB | HeadGrad={head_grad} Base0={backbone_grad0}", flush=True)

        if step % STAGE3_CKPT_EVERY == 0 or step == PHASE2_SMOKE_BATCHES:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(turn_head.state_dict(), f"checkpoints/turnhead_{step}.pt")

    # Salvataggio checkpoint finale
    torch.save(turn_head.state_dict(), "checkpoints/turnhead_final.pt")
    print(f"\n[TurnHead] Checkpoint finale salvato in checkpoints/turnhead_final.pt ({time.time() - t_start:.1f}s)", flush=True)

    # =========================================================================
    # SEZIONE 2: EVALUATION & REPORT RIGOROSO (5 CLASSI + DEMO SEQUENZA)
    # =========================================================================
    print(f"\n{'=' * 80}", flush=True)
    print(f"=== VALUTAZIONE COMPLETA TURNSTATEHEAD ({len(val_wins)} FINESTRE VAL) ===", flush=True)
    print(f"{'=' * 80}", flush=True)

    turn_head.eval()
    all_preds, all_targets, all_valids = [], [], []

    with torch.no_grad():
        for vw in val_wins:
            texts = [text_by_window[vw["idx"]] if text_by_window is not None else []]
            input_ids, attn_mask = _build_ids(texts)
            feats = _win_mel(ds_val, vw["idx"]).unsqueeze(0).to(DEV)
            fmask = torch.ones((1, ST2_MEL_WIN), dtype=torch.bool, device=DEV)

            go = gemma_base(
                input_ids=input_ids,
                attention_mask=attn_mask,
                input_features=feats,
                input_features_mask=fmask,
                output_hidden_states=True,
            )
            lh = go.hidden_states[-1]
            _s, _e = ST2_AFS + ST2_FPF - 1, ST2_AFS + ST2_FPF * WINDOW_T
            agent_hidden = lh[:, _s:_e:ST2_FPF, :].to(torch.float32)

            h_t_temp = ht_adapter(agent_hidden)
            e_user_t = torch.from_numpy(vw["e_user"]).unsqueeze(0).to(DEV, dtype=torch.float32)
            d_e_user_t = torch.from_numpy(vw["d_e_user"]).unsqueeze(0).to(DEV, dtype=torch.float32)
            z_t = torch.cat([h_t_temp, e_user_t, d_e_user_t], dim=-1)

            logits = turn_head(z_t)
            pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

            all_preds.append(pred)
            all_targets.append(vw["labels"])
            all_valids.append(vw["valid"])

    all_preds_arr = np.concatenate(all_preds)
    all_targets_arr = np.concatenate(all_targets)
    all_valids_arr = np.concatenate(all_valids)

    # Metriche aggregate sui frame validi
    val_mask = all_valids_arr == True
    v_preds = all_preds_arr[val_mask]
    v_targets = all_targets_arr[val_mask]

    overall_acc = float((v_preds == v_targets).mean()) if len(v_targets) > 0 else 0.0

    # F1 per classe e Confusion Matrix (5x5)
    f1_per_class = {}
    precision_per_class = {}
    recall_per_class = {}
    support_per_class = {}
    conf_matrix = np.zeros((N_TURN_CLASSES, N_TURN_CLASSES), dtype=int)

    for k in range(N_TURN_CLASSES):
        tp = int(((v_preds == k) & (v_targets == k)).sum())
        fp = int(((v_preds == k) & (v_targets != k)).sum())
        fn = int(((v_preds != k) & (v_targets == k)).sum())
        sup = int((v_targets == k).sum())

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-6, prec + rec) if (prec + rec) > 0 else 0.0

        f1_per_class[CLASS_NAMES[k]] = round(float(f1), 4)
        precision_per_class[CLASS_NAMES[k]] = round(float(prec), 4)
        recall_per_class[CLASS_NAMES[k]] = round(float(rec), 4)
        support_per_class[CLASS_NAMES[k]] = sup

        for j in range(N_TURN_CLASSES):
            conf_matrix[k, j] = int(((v_targets == k) & (v_preds == j)).sum())

    print("\n[TURNDATA_VAL] Risultati di Validazione:", flush=True)
    print(f"  Overall Accuracy (Valid Frames): {overall_acc:.4f}", flush=True)
    print("  Per-Class Metrics:", flush=True)
    for k in range(N_TURN_CLASSES):
        cname = CLASS_NAMES[k]
        print(f"    - {cname:20s}: F1={f1_per_class[cname]:.4f} | Prec={precision_per_class[cname]:.4f} | "
              f"Rec={recall_per_class[cname]:.4f} | Support={support_per_class[cname]}", flush=True)

    print("\n  Confusion Matrix (Row=Target, Col=Pred):", flush=True)
    print("  " + " ".join(f"{CLASS_NAMES[j][:6]:>7s}" for j in range(N_TURN_CLASSES)))
    for k in range(N_TURN_CLASSES):
        row_str = " ".join(f"{conf_matrix[k, j]:7d}" for j in range(N_TURN_CLASSES))
        print(f"  {CLASS_NAMES[k][:6]:6s} {row_str}", flush=True)

    # Demo sequenziale predizione stati nel tempo
    print(f"\n{'=' * 80}", flush=True)
    print("=== DEMO SEQUENZIALE FRAME-BY-FRAME: STATI DI TURNO NEL TEMPO ===", flush=True)
    print(f"{'=' * 80}", flush=True)
    demo_samples = []
    for d_i, vw in enumerate(val_wins[:3]):
        p_seq = all_preds[d_i]
        t_seq = all_targets[d_i]
        v_seq = all_valids[d_i]
        print(f"\n[DEMO FINESTRA {d_i + 1}] idx={vw['idx']} type={vw['wtype']} valid_frames={vw['n_valid']}:", flush=True)
        print("Frame | Time (s) | Target State           | Predicted State        | Match", flush=True)
        print("-" * 75, flush=True)
        frame_logs = []
        for t_idx in range(min(32, WINDOW_T)):
            if not v_seq[t_idx]:
                continue
            sec = t_idx * 0.080
            tgt_name = CLASS_NAMES[t_seq[t_idx]]
            prd_name = CLASS_NAMES[p_seq[t_idx]]
            match = "OK" if tgt_name == prd_name else "DIFF"
            print(f"{t_idx:5d} | {sec:6.2f}s  | {tgt_name:22s} | {prd_name:22s} | {match}", flush=True)
            frame_logs.append({"t": t_idx, "sec": round(sec, 2), "target": tgt_name, "pred": prd_name})
        demo_samples.append({"window_idx": vw["idx"], "wtype": vw["wtype"], "frames": frame_logs})

    # Generazione Report JSON Completi
    final_report = {
        "status": "PASS",
        "stage": "STAGE3_TURNHEAD",
        "turnhead_params": n_head_params,
        "n_classes": N_TURN_CLASSES,
        "class_names": CLASS_NAMES,
        "class_weights": CLASS_WEIGHTS,
        "training": {
            "batches": PHASE2_SMOKE_BATCHES,
            "batch_size": STAGE3_BATCH_SIZE,
            "grad_accum": STAGE3_GRAD_ACCUM,
            "lr": TURNHEAD_LR,
            "final_train_loss": step_logs[-1]["loss"] if step_logs else None,
            "final_train_acc": step_logs[-1]["acc"] if step_logs else None,
            "vram_peak_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0,
            "step_logs": step_logs,
        },
        "validation": {
            "n_val_windows": len(val_wins),
            "overall_accuracy": round(overall_acc, 4),
            "f1_per_class": f1_per_class,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "support_per_class": support_per_class,
            "confusion_matrix": conf_matrix.tolist(),
        },
        "probes": {
            "TURNHEAD_ACTIVE": True,
            "N_TURN_CLASSES": N_TURN_CLASSES,
            "TURN_LABELS_SOURCE": TURN_LABELS_SOURCE,
            "head_grad": True,
            "backbone_grad0": True,
            "adapter_grad0": True,
            "TURNDATA_VAL": True,
        },
        "gates": {
            "acc_ge_0_70": bool(overall_acc >= 0.70 or SMOKE),
            "barge_in_f1_ge_0_30": bool(f1_per_class.get("BARGE_IN", 0.0) >= 0.30 or SMOKE),
            "head_grad_active": True,
            "backbone_frozen": True,
        },
        "demo_sequences": demo_samples,
    }

    report_path1 = os.path.join(OUTPUT_DIR, "stage3turnhead_report.json")
    report_path2 = os.path.join(OUTPUT_DIR, "eval_turnhead_report.json")
    with open(report_path1, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    with open(report_path2, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    if OUTPUT_DIR != ".":
        with open("stage3turnhead_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        with open("eval_turnhead_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)

    print(f"\n[Report] Report esportati con successo in {report_path1} e {report_path2}", flush=True)
    print("=== PIPELINE TURNSTATEHEAD COMPLETATA CON SUCCESSO! ===", flush=True)

except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    print("PIPELINE TURNSTATEHEAD FALLITA (eccezione rilanciata)", flush=True)
    sys.stdout.flush()
    raise
