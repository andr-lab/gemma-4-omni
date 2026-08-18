# -*- coding: utf-8 -*-
# ============================================================================
# FITNETS EXTRACT: estrazione h_t NATIVO dal teacher Qwen3-TTS (main talker)
# Il target di FitNets e' l'input esatto che il code_predictor (nostro
# sub-talker) ha visto in training: hidden finale (layer 28, post-RMSNorm,
# talker_hidden=2048) ai frame audio 12.5Hz.
# Formato input: replica ESATTA del collate_fn ufficiale del finetuning Qwen3-TTS
# (github.com/QwenLM/Qwen3-TTS -> finetuning/dataset.py + sft_12hz.py).
# Verdetto CHECK-E (17/08): KO (gap informativo) -> il piano B FitNets richiede
# questo target. Valutazione agy: layer 28 obbligatorio, sanity 1-sample prima.
# ============================================================================
import os, sys, json, traceback, hashlib, shutil, subprocess, time

try:
    sys.path.insert(0, "/kaggle/working")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    import gc

    EXTRACT_N = int(os.environ.get("FITEX_N", "1000"))
    SANITY = os.environ.get("FITEX_SANITY", "1") == "1"
    FIDELITY = os.environ.get("FIDELITY", "0") == "1" or os.environ.get("FIDELITY_PROBE", "0") == "1"
    START_IDX = int(os.environ.get("FITEX_START", "0"))
    SMOKE_B = os.environ.get("SMOKE_B", "0") == "1"
    SMOKE_B_STEPS = int(os.environ.get("SMOKE_B_STEPS", "300"))
    SMOKE_B_LR = float(os.environ.get("SMOKE_B_LR", "5e-4"))
    SMOKE_B_BATCH = int(os.environ.get("SMOKE_B_BATCH", "8"))
    STAGE2 = int(os.environ.get("STAGE2", "0"))
    PHASE2_SMOKE_BATCHES = int(os.environ.get("PHASE2_SMOKE_BATCHES", "4"))
    STAGE2_BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", "8"))
    STAGE2_GRAD_ACCUM = int(os.environ.get("STAGE2_GRAD_ACCUM", "2"))
    STAGE2_LR = float(os.environ.get("STAGE2_LR", "1.5e-4"))
    STAGE2_LAMBDA_COUPLING = float(os.environ.get("STAGE2_LAMBDA_COUPLING", "0.2"))
    STAGE2_WEIGHT_DECAY = float(os.environ.get("STAGE2_WEIGHT_DECAY", "1e-2"))
    STAGE2_GRAD_CLIP = float(os.environ.get("STAGE2_GRAD_CLIP", "1.0"))
    STAGE2_CKPT_EVERY = int(os.environ.get("STAGE2_CKPT_EVERY", "250"))

    import kaggle_pipeline as KP
    import qwen_subtalker as QS

    # ---- FIX RoPE runtime (bug storico: position_ids=[0] invece di 0..15) ----
    # Il dataset montato contiene qwen_subtalker.py con il bug: _forward_flat
    # calcola S=1 da h_t_flat (singolo frame) e applica il RoPE con fase 0 su
    # tutte le 16 posizioni. Patch della classe: S = lunghezza reale della
    # sequenza concatenata (h_t, cb0, cb1..cb14). Attivo con ROPE_FIX=1.
    if os.environ.get("ROPE_FIX", "0") == "1" and hasattr(QS, "Qwen3TTSSubTalker"):
        def _fixed_forward_flat(self, h_t_flat, codec_ids_flat):
            inputs = [h_t_flat]  # posizione 0: h_t (senza embedding)
            inputs.append(self.cb0_embedding(codec_ids_flat[:, :1].clamp(min=0)))
            for g in range(self.cfg["num_code_groups"] - 2):
                inputs.append(self.group_embeddings[g](codec_ids_flat[:, g + 1 : g + 2].clamp(min=0)))
            inputs_embeds = torch.cat(inputs, dim=1)  # (B, 16, talker_hidden)
            S = inputs_embeds.shape[1]  # FIX: 16 posizioni reali (prima S=1)
            hidden_states = self.small_to_mtp_projection(inputs_embeds)
            position_ids = torch.arange(0, S, device=h_t_flat.device).unsqueeze(0)
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            causal_mask = self._causal_mask.to(h_t_flat.device).to(h_t_flat.dtype)
            for layer in self.layers:
                hidden_states = layer(hidden_states, position_embeddings, causal_mask)
            hidden_states = self.norm(hidden_states)
            logits = torch.stack(
                [self.lm_head[j](hidden_states[:, j + 1]) for j in range(self.cfg["num_code_groups"] - 1)], dim=1
            )
            return logits
        QS.Qwen3TTSSubTalker._forward_flat = _fixed_forward_flat
        print("ROPE_FIX applicato: _forward_flat usa S=16 posizioni reali", flush=True)

        # ---- THETA FIX (secondo bug di fedelta'): il config.json del dataset ha
        # rope_theta=10000.0 (bug di extract_qwen_subtalker.py: SLIM_CONFIG
        # hardcoded) mentre il teacher reale usa rope_theta=1000000.0 -> frequenze
        # RoPE diverse -> logits divergenti (probe v13: teacher=1000000, sub=10000).
        # Patch della funzione load: dopo il load, se theta != 1e6, ricrea la
        # rotary_emb con theta 1e6 e aggiorna cfg. Copre tutti i load successivi
        # (SMOKE_B, probe FIDELITY, training).
        _orig_load_qs = QS.load_qwen_subtalker

        def _load_with_theta(subtalker_dir, device=None, dtype=torch.bfloat16):
            model, cfg = _orig_load_qs(subtalker_dir, device=device, dtype=dtype)
            if abs(float(cfg.get("rope_theta", 10000.0)) - 1000000.0) > 1e-7:
                cfg["rope_theta"] = 1000000.0
                model.rotary_emb = QS.Qwen3TTSRotaryEmbedding(
                    head_dim=cfg["head_dim"],
                    max_position_embeddings=cfg["max_position_embeddings"],
                    theta=1000000.0,
                ).to(device=device, dtype=dtype)
                print("THETA_FIX applicato: rope_theta 10000 -> 1000000 (rotary_emb ricreata)", flush=True)
            return model, cfg

        QS.load_qwen_subtalker = _load_with_theta
        print("THETA_FIX installato: load_qwen_subtalker patcha rope_theta a 1e6", flush=True)

    # ---- dati (identico a checke) ----
    parquet_cache = f"data/train_tokens_{KP.DUPLEXCONV_MAX_HOURS}h_{KP.MDC_MAX_SAMPLES}s.parquet"
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
        agent_codes[:, :val_start], seq_len=64,
        text_by_window=None,
    )
    print(f"split train=[0,{val_start}) | ds_train: {len(ds_train)} finestre", flush=True)

    # ---- clone repo ufficiale Qwen3-TTS (modeling + processor) ----
    _SRC = "/kaggle/working/qwen3tts_src"
    if not os.path.isdir(_SRC):
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/QwenLM/Qwen3-TTS.git", _SRC],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print("git clone FAIL:", r.stderr[-500:], flush=True)
            raise RuntimeError("git clone Qwen3-TTS fallito")

    # --- bypassa gli __init__ dei package che importano sox/onnxruntime ---
    # qwen_tts/__init__.py e qwen_tts/core/__init__.py importano i tokenizer
    # 25Hz/12Hz (tokenizer_25hz/vq/speech_vq.py -> `import sox`) che NON ci
    # servono: i codici GT arrivano dal dataset, il teacher serve solo per il
    # forward del talker (h_t nativo). Stub dei package padre + stub del
    # modulo tokenizer, poi carico modeling_qwen3_tts.py direttamente.
    import types as _types

    def _stub_pkg(_name, _path):
        _p = _types.ModuleType(_name)
        _p.__path__ = [_path]
        _p.__package__ = _name
        sys.modules[_name] = _p

    _stub_pkg("qwen_tts", f"{_SRC}/qwen_tts")
    _stub_pkg("qwen_tts.core", f"{_SRC}/qwen_tts/core")
    _stub_pkg("qwen_tts.core.models", f"{_SRC}/qwen_tts/core/models")
    _stub_pkg("qwen_tts.inference", f"{_SRC}/qwen_tts/inference")
    _tok = _types.ModuleType("qwen_tts.inference.qwen3_tts_tokenizer")
    class _StubTokenizer:
        def __init__(self, *a, **k):
            raise RuntimeError("stub Qwen3TTSTokenizer: non istanziare (percorso senza speech_tokenizer)")
    _tok.Qwen3TTSTokenizer = _StubTokenizer
    sys.modules["qwen_tts.inference.qwen3_tts_tokenizer"] = _tok
    sys.path.insert(0, _SRC)
    import qwen_tts.core.models.modeling_qwen3_tts as _M
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
    from transformers import AutoTokenizer
    print("repo Qwen3-TTS clonato e importato (tokenizer sox bypassati)", flush=True)

    # ---- carica teacher (bf16; forward-only) ----
    # transformers 5.5.0 in Kaggle: from_pretrained chiama
    # mark_tied_weights_as_initialized -> property `all_tied_weights_keys`
    # (introdotta in transformers 5.7+) che NON esiste sulla classe del repo
    # -> AttributeError. Carichiamo i pesi A MANO via safetensors (stesso
    # risultato, nessuna magia dei tied weights): istanzia con la config
    # (init random), poi load_state_dict(strict=False) con i pesi veri.
    # model.safetensors contiene solo talker.* + speaker_encoder.*.
    t0 = time.time()
    import safetensors.torch as _st
    from huggingface_hub import hf_hub_download
    # transformers 5.5.0 ha RIMOSSO la chiave "default" da ROPE_INIT_FUNCTIONS
    # (ripristinata nelle versioni >=5.6 su cui gira il repo ufficiale). Il
    # config del talker usa rope_scaling={"rope_type": "default",
    # mrope_section:[24,20,20], interleaved} -> ripristiniamo
    # _default_rope_init (inv_freq classica theta^{-2i/d} + scaling 1.0).
    # Il modeling importa ROPE_INIT_FUNCTIONS (stesso dict) -> mutation visibile.
    import transformers.modeling_rope_utils as _ropeu
    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        _dim = int(_head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
        return inv_freq, 1.0
    _ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    _cfg = _M.Qwen3TTSForConditionalGeneration.config_class.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    # pad_token_id manca nel config.json del talker (transformers >=5.7 lo
    # defaulta a None; 5.5.0 no). Non e' mai usato negli embedding (solo in
    # _init_weights con guardia `is not None`) -> None = comportamento ufficiale.
    _cfg.talker_config.pad_token_id = None
    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _ckpt = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors")
    _sd = _st.load_file(_ckpt, device="cpu")
    _missing, _unexpected = teacher.load_state_dict(_sd, strict=False)
    del _sd
    torch.cuda.empty_cache()
    if _missing:
        raise RuntimeError(f"pesi teacher mancanti ({len(_missing)}): {_missing[:8]}")
    if _unexpected:
        print(f"pesi inattesi ignorati ({len(_unexpected)}): {_unexpected[:6]}", flush=True)
    teacher = teacher.to(device="cuda", dtype=torch.bfloat16)  # cast rigido post-load
    teacher.eval()
    talker = teacher.talker
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", use_fast=True)
    proc = Qwen3TTSProcessor(tokenizer=tok)
    print(f"teacher caricato in {time.time()-t0:.1f}s | talker hidden={talker.config.hidden_size} "
          f"layers={talker.config.num_hidden_layers}", flush=True)

    CFG = teacher.config
    TCFG = CFG.talker_config
    TTS_PAD, TTS_BOS, TTS_EOS = CFG.tts_pad_token_id, CFG.tts_bos_token_id, CFG.tts_eos_token_id
    CN, CTB, CTE, CPAD, CBOS, CEOS = (
        TCFG.codec_nothink_id, TCFG.codec_think_bos_id, TCFG.codec_think_eos_id,
        TCFG.codec_pad_id, TCFG.codec_bos_id, TCFG.codec_eos_token_id,
    )
    print(f"token: tts_pad={TTS_PAD} bos={TTS_BOS} eos={TTS_EOS} | codec nothink={CN} "
          f"think_bos={CTB} think_eos={CTE} pad={CPAD} bos={CBOS} eos={CEOS}", flush=True)

    # ---- sub-talker locale (per sanity / smoke B: h_t nativo -> cb1-15 -> wav) ----
    if SANITY or SMOKE_B:
        PAD = KP.PAD_TARGET_ID
        SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
        sub, _sub_cfg = QS.load_qwen_subtalker("/kaggle/input/datasets/andrlab/gemma-4-omni-module/qwen_subtalker",
                                     device="cuda", dtype=torch.float32)
        decoder = KP.Code2WavDecoder()
        sub.eval()

    # ---- replica ESATTA del collate_fn ufficiale (batch 1, niente padding) ----
    def build_inputs(text: str, audio_codes: torch.Tensor):
        """audio_codes: (T_audio, 16) long, valori del tokenizer 12Hz GT.
        Ritorna input_ids (1,t,2), codec_ids (1,t,16), text_mask, codec_emb_mask, codec_mask,
        attn (1,t), pos_cbos. t = 8 + Tt + Ta."""
        assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = proc(text=assistant, return_tensors="pt")["input_ids"]  # (1, T_text_tot)
        text_ids = text_ids[:, :-5]  # come dataset ufficiale
        Tt = text_ids.shape[1]
        Ta = audio_codes.shape[0]
        t = 8 + Tt + Ta
        pos_cbos = 8 + Tt - 2
        input_ids = torch.zeros((1, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((1, t, 16), dtype=torch.long)
        text_mask = torch.zeros((1, t), dtype=torch.bool)
        codec_emb_mask = torch.zeros((1, t), dtype=torch.bool)
        codec_mask = torch.zeros((1, t), dtype=torch.bool)
        attn = torch.zeros((1, t), dtype=torch.long)
        # ---- text channel ----
        input_ids[0, :3, 0] = text_ids[0, :3]
        input_ids[0, 3:7, 0] = TTS_PAD
        input_ids[0, 7, 0] = TTS_BOS
        input_ids[0, 8:8 + Tt - 3, 0] = text_ids[0, 3:]
        input_ids[0, 8 + Tt - 3, 0] = TTS_EOS
        input_ids[0, 8 + Tt - 2:8 + Tt + Ta, 0] = TTS_PAD
        text_mask[0, :8 + Tt + Ta] = True
        # ---- codec channel ----
        input_ids[0, 3:8, 1] = torch.tensor([CN, CTB, CTE, 0, CPAD], dtype=torch.long)
        input_ids[0, 8:8 + Tt - 3, 1] = CPAD
        input_ids[0, 8 + Tt - 3, 1] = CPAD
        input_ids[0, 8 + Tt - 2, 1] = CBOS
        input_ids[0, 8 + Tt - 1:8 + Tt - 1 + Ta, 1] = audio_codes[:, 0]
        input_ids[0, 8 + Tt - 1 + Ta, 1] = CEOS
        codec_emb_mask[0, 3:8 + Tt + Ta] = True
        codec_emb_mask[0, 6] = False  # slot speaker embedding (niente voce clonata)
        codec_mask[0, 8 + Tt - 1:8 + Tt - 1 + Ta] = True
        attn[0, :8 + Tt + Ta] = True
        codec_ids[0, 8 + Tt - 1:8 + Tt - 1 + Ta, :] = audio_codes
        return input_ids, codec_ids, text_mask, codec_emb_mask, codec_mask, attn, pos_cbos

    # ---- forward talker + h_t target (come sft_12hz.py) ----
    def extract_ht(item, n_real, text, return_shifted: bool = False):
        audio_codes = item["audio_targets"][:n_real].long()  # (T_audio, 16)
        input_ids, codec_ids, text_mask, codec_emb_mask, codec_mask, attn, pos_cbos = build_inputs(text, audio_codes)
        input_ids = input_ids.to("cuda")
        codec_ids = codec_ids.to("cuda")
        text_mask = text_mask.to("cuda")
        codec_emb_mask = codec_emb_mask.to("cuda")
        codec_mask = codec_mask.to("cuda")
        attn = attn.to("cuda")
        with torch.inference_mode():
            # 1) canale testo: proiezione non-lineare UFFICIALE (ResizeMLP, SiLU)
            raw_text_emb = talker.model.text_embedding(input_ids[:, :, 0])
            input_text_emb = talker.text_projection(raw_text_emb) * text_mask.unsqueeze(-1)
            input_codec_emb = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_emb_mask.unsqueeze(-1)
            input_embeddings = input_text_emb + input_codec_emb
            for i in range(1, 16):
                ci = talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                input_embeddings = input_embeddings + ci * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings.to(dtype=talker.dtype)  # bf16 difensivo
            outputs = talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attn[:, :-1],
                output_hidden_states=True,
                use_cache=False,
            )
            # outputs.hidden_states[0][-1] e' gia' POST-RMSNorm finale (dim=2048)
            hidden_states = outputs.hidden_states[0][-1]  # (1, t-1, 2048)
            # ALLINEAMENTO TEMPORALE AUTOREGRESSIVO:
            # Per predire audio_codes[k] (frame k), il talker usa lo stato generato alla posizione precedente:
            # - Frame 0 (cb0..15): stato generato alla posizione CBOS (pos_cbos = 8 + Tt - 2)
            # - Frame k: stato generato a pos_cbos + k
            # Slicing corretto: [pos_cbos : pos_cbos + Ta] (lunghezza esatta Ta)
            Ta = audio_codes.shape[0]
            ht_target = hidden_states[0, pos_cbos : pos_cbos + Ta, :]  # (T_audio, 2048)
            if return_shifted:
                cm = codec_mask[:, :-1]
                ht_shifted = hidden_states[cm]  # vecchio slicing errato (pos_cbos+1 : pos_cbos+1+Ta)
                return ht_target.float().cpu().numpy(), ht_shifted.float().cpu().numpy()
        return ht_target.float().cpu().numpy()

    # ---- SCAN finestre attive (come checke, per una sanity comparabile) ----
    def _rms(x):
        x = np.asarray(x, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0

    def _frame_active_mask(wav, n_real):
        FRAME_SAMPLES = int(round(KP.SYNTH_SAMPLE_RATE / KP.FRAME_RATE_HZ))
        n = min(len(wav), n_real * FRAME_SAMPLES)
        w = np.asarray(wav[:n], dtype=np.float32)
        nf = n // FRAME_SAMPLES
        if nf == 0:
            return np.zeros(n_real, dtype=bool), np.zeros(n_real, dtype=np.float32)
        rms_per = np.array([_rms(w[t * FRAME_SAMPLES:(t + 1) * FRAME_SAMPLES]) for t in range(nf)])
        thr = max(0.02, 0.25 * (rms_per.max() if rms_per.max() > 0 else 1.0))
        mask = np.zeros(n_real, dtype=bool)
        mask[:nf] = rms_per > thr
        return mask, rms_per

    PAD = KP.PAD_TARGET_ID
    SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
    MPF_MEL = KP.MEL_PER_AGENT_FRAME

    def _mel_active_mask(ds, idx, n_real):
        """Mask attivo dai mel agent GT (proxy energia, senza decode wav).
        I mel agent NON sono per-item: sono nell'attributo globale
        ds.agent_mel, allineato per finestra (stessi indici di __getitem__)."""
        m0 = idx * ds.seq_len * MPF_MEL
        n_mel = n_real * MPF_MEL
        m1 = min(m0 + n_mel, ds.agent_mel.shape[0])
        mel = np.asarray(ds.agent_mel[m0:m1], dtype=np.float32)
        if mel.shape[0] == 0:
            return np.zeros(n_real, dtype=bool)
        nf = mel.shape[0] // MPF_MEL
        energy = np.array([float(np.mean(np.abs(mel[t * MPF_MEL:(t + 1) * MPF_MEL]))) for t in range(nf)])
        thr = max(0.01, 0.25 * (energy.max() if energy.max() > 0 else 1.0))
        mask = np.zeros(n_real, dtype=bool)
        mask[:nf] = energy > thr
        return mask

    def _scan_window(ds, idx):
        item = ds[idx]
        n_real = int((item["audio_targets"][:, 0] != PAD).sum().item())
        if n_real <= 0:
            return None
        if SANITY or FIDELITY:
            active = _mel_active_mask(ds, idx, n_real)
            if int(active.sum()) == 0:
                return None
            return {"idx": idx, "n_real": n_real, "active": active}
        return {"idx": idx, "n_real": n_real}

    train_wins = []
    for i in range(min(1024, len(ds_train))):
        w = _scan_window(ds_train, i)
        if w is None:
            continue
        train_wins.append(w)
        if len(train_wins) >= (1 if FIDELITY else EXTRACT_N):
            break
    print(f"raccolte {len(train_wins)} finestre train (EXTRACT_N={EXTRACT_N}, FIDELITY={FIDELITY})", flush=True)

    # ---- SEZIONE OPZIONALE: STAGE2_SMOKE (FitNets adapter Gemma -> h_t) ----
    # Obiettivo: validare forward/backward dell'adapter (Gemma E2B congelata 4bit +
    # ht_adapter trainable 3072->2048 + sub-talker Qwen congelato bf16) con loss
    # L_latent = (1-cos_sim) + 0.5*MSE su frame attivi + lambda*CE coupling.
    # Gate run completo: cos_sim>=0.88, CE_val<=5.95, rank SVD>=256 (revisione agy).
    if STAGE2 == 1 and train_wins:
        print("\n" + "=" * 80, flush=True)
        print("=== STAGE 2 SMOKE: ADAPTER GEMMA -> h_t (L_latent + lambda*CE, sub congelato) ===", flush=True)
        print("=" * 80, flush=True)
        try:
            ST2_T = 64  # frame agente per finestra
            ST2_FPF = 2  # feature audio 25fps per frame agente 12.5fps
            ST2_TXT = 48  # TEXT_BLOCK_LEN
            ST2_AFS = 2 + ST2_TXT  # audio_feat_start
            ST2_MEL_WIN = ST2_T * 8  # 512 frame mel 100fps per finestra
            BOS_ID, BOA_ID, AUDIO_TOKEN_ID, EOA_ID = 2, 256000, 258881, 258883
            DEV2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"[ST2] cfg: batches={PHASE2_SMOKE_BATCHES} bs={STAGE2_BATCH_SIZE} "
                  f"accum={STAGE2_GRAD_ACCUM} lr={STAGE2_LR} lambda_c={STAGE2_LAMBDA_COUPLING}", flush=True)

            # --- sub-talker bf16 (se sub_bf16 gia' caricato, altrimenti load deterministico) ---
            # NOTA: la variabile globale "sub" del cell0 e' una STRINGA (residuo di un
            # for loop) -> mai usarla; si carica sempre il modello bf16 dedicato.
            _v0 = globals().get("sub_bf16")
            if _v0 is not None and not isinstance(_v0, (str, bytes)):
                sub_st2 = _v0
            else:
                _sub_paths = [
                    "/kaggle/input/datasets/andrlab/gemma-4-omni-module/qwen_subtalker",
                    "/kaggle/input/andrlab/gemma-4-omni-module/qwen_subtalker",
                    "qwen_subtalker",
                ]
                _sp = next((p for p in _sub_paths if os.path.isdir(p)), _sub_paths[0])
                sub_st2, _ = QS.load_qwen_subtalker(_sp, device="cuda", dtype=torch.bfloat16)
                print(f"[ST2] sub-talker caricato bf16 da {_sp}", flush=True)
            sub_st2.eval()
            for _p in sub_st2.parameters():
                _p.requires_grad = False

            # --- Gemma E2B 4bit congelata (forward no_grad) ---
            # VRAM T4: il load unsloth materializza i pesi bf16 transitori (~5.2GB)
            # prima della quantizzazione 4bit -> picco OOM col teacher in GPU (v18).
            # Spostiamo il teacher (3.4GB bf16) su CPU durante il load e lo
            # riportiamo su GPU nel finally. Il sub bf16 (1.3GB) resta.
            print("[ST2] load Gemma E2B (unsloth 4bit, frozen; teacher su CPU durante il load)...", flush=True)
            _tdev = next(talker.parameters()).device
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            gemma_base = gemma_tok = None
            try:
                from unsloth import FastLanguageModel
                gemma_base, gemma_tok = FastLanguageModel.from_pretrained(
                    model_name="unsloth/gemma-4-E2B-it",
                    max_seq_length=2048,
                    load_in_4bit=True,
                    device_map="cuda:0",
                )
                for _nm2, _mod in gemma_base.named_modules():
                    if "norm" in _nm2.lower():
                        _mod.to(torch.float32)
                gemma_base.eval()
                for _p in gemma_base.parameters():
                    _p.requires_grad = False
            finally:
                teacher.to(_tdev)
                gc.collect()
                torch.cuda.empty_cache()
            if gemma_base is None:
                raise RuntimeError("load Gemma E2B fallito")
            print("[ST2] Gemma E2B caricata e congelata; teacher rispostato su GPU", flush=True)

            # --- adapter (UNICO trainable) ---
            # Le hidden del LLM Gemma 4 E2B hanno dim 1536 (text_config.hidden_size,
            # verificato su config unsloth e google; l'hardcode 3072 era errato).
            _HID = int(gemma_base.config.text_config.hidden_size) if hasattr(gemma_base.config, "text_config") else 1536
            ht_adapter = nn.Sequential(
                nn.Linear(_HID, 2048),
                nn.LayerNorm(2048),
            ).to(DEV2, dtype=torch.float32)
            print(f"[ST2] hidden_size LLM rilevato: {_HID}", flush=True)
            trainable_params = [p for p in ht_adapter.parameters()]
            n_tr = sum(p.numel() for p in trainable_params)
            print(f"[ST2] parametri trainable adapter: {n_tr:,}", flush=True)
            optimizer = torch.optim.AdamW(
                trainable_params, lr=STAGE2_LR, betas=(0.9, 0.98), eps=1e-6,
                weight_decay=STAGE2_WEIGHT_DECAY,
            )

            # --- costruzione input ---
            # gemma_tok di unsloth e' un Gemma4Processor (niente .encode): si usa
            # AutoTokenizer del repo google/gemma-4-E2B-it (pattern progetto riga 3616).
            print("[ST2] load tokenizer testo Gemma E2B (AutoTokenizer)...", flush=True)
            from transformers import AutoTokenizer
            gemma_txt = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
            E2B_PAD = 0  # pad_token_id tokenizer testo Gemma 4

            def _build_ids(texts):
                rows, mrows = [], []
                for t in texts:
                    toks = gemma_txt.encode(str(t) if t else "", add_special_tokens=False)
                    if len(toks) > ST2_TXT:
                        toks = toks[-ST2_TXT:]
                    n_pad = ST2_TXT - len(toks)
                    row = ([BOS_ID] + [E2B_PAD] * n_pad + toks + [BOA_ID]
                           + [AUDIO_TOKEN_ID] * (ST2_T * ST2_FPF) + [EOA_ID])
                    mrow = ([1] + [0] * n_pad + [1] * (len(toks) + 1)
                            + [1] * (ST2_T * ST2_FPF) + [1])
                    rows.append(row)
                    mrows.append(mrow)
                return (torch.tensor(rows, dtype=torch.long, device=DEV2),
                        torch.tensor(mrows, dtype=torch.long, device=DEV2))

            def _win_mel(idx):
                s = idx * ST2_MEL_WIN
                m = user_mel[s:s + ST2_MEL_WIN]
                if m.shape[0] < ST2_MEL_WIN:
                    m = np.pad(m, ((0, ST2_MEL_WIN - m.shape[0]), (0, 0)))
                return torch.from_numpy(np.asarray(m, dtype=np.float32))

            # --- estrazione target h_t UNA TANTUM (teacher in GPU, poi su CPU) ---
            # I target non dipendono dall'adapter: si estraggono prima del training e il
            # teacher (3.4GB bf16) esce dalla GPU -> evita OOM nel loop (v22: OOM 48MiB
            # nel forward del sub con teacher+Gemma+sub tutti in GPU).
            print(f"[ST2] estrazione target h_t per {len(train_wins)} finestre (teacher, no_grad)...", flush=True)
            ht_map = {}
            _t0 = time.time()
            for _w in train_wins:
                _it = ds_train[_w["idx"]]
                _tx = text_by_window[_w["idx"]] if text_by_window is not None else ""
                _h = extract_ht(_it, _w["n_real"], _tx, return_shifted=False)  # (n_real,2048)
                _h64 = np.zeros((ST2_T, 2048), dtype=np.float32)
                _h64[: min(_w["n_real"], ST2_T)] = _h[: min(_w["n_real"], ST2_T)]
                ht_map[_w["idx"]] = _h64
            print(f"[ST2] target estratti: {len(ht_map)} in {time.time()-_t0:.1f}s; teacher su CPU", flush=True)
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            step_logs = []
            win_cursor = 0
            avail = len(train_wins)
            optimizer.zero_grad()
            for step in range(1, PHASE2_SMOKE_BATCHES + 1):
                t0s = time.time()
                acc_loss = acc_lat = acc_coup = acc_cos = 0.0
                for _mi in range(STAGE2_GRAD_ACCUM):
                    wins = [train_wins[(win_cursor + i) % avail] for i in range(STAGE2_BATCH_SIZE)]
                    win_cursor = (win_cursor + STAGE2_BATCH_SIZE) % avail
                    items = [ds_train[w["idx"]] for w in wins]
                    texts = [text_by_window[w["idx"]] if text_by_window is not None else "" for w in wins]
                    nrs = [w["n_real"] for w in wins]

                    # target h_t gia' estratti pre-loop (teacher su CPU)
                    ht_target = torch.stack(
                        [torch.from_numpy(ht_map[w["idx"]]) for w in wins], 0
                    ).to(DEV2, dtype=torch.float32)

                    # codes + mask attiva
                    audio_targets = torch.stack(
                        [torch.from_numpy(np.asarray(it["audio_targets"], dtype=np.int64)) for it in items], 0
                    ).to(DEV2)
                    am = []
                    for w in wins:
                        if "active" in w and w["active"] is not None:
                            a = np.asarray(w["active"], dtype=np.float32)
                            if a.shape[0] < ST2_T:
                                a = np.pad(a, (0, ST2_T - a.shape[0]))
                        else:
                            a = np.zeros(ST2_T, dtype=np.float32)
                            a[: min(w["n_real"], ST2_T)] = 1.0
                        am.append(a)
                    active_mask = torch.from_numpy(np.stack(am, 0)).to(DEV2)  # (B, 64)

                    # forward Gemma (no_grad) + adapter (grafo)
                    input_ids, attn_mask = _build_ids(texts)
                    feats = torch.stack([_win_mel(w["idx"]) for w in wins], 0).to(DEV2)
                    fmask = torch.ones((STAGE2_BATCH_SIZE, ST2_MEL_WIN), dtype=torch.bool, device=DEV2)
                    with torch.no_grad():
                        go = gemma_base(
                            input_ids=input_ids,
                            attention_mask=attn_mask,
                            input_features=feats,
                            input_features_mask=fmask,
                            output_hidden_states=True,
                        )
                        lh = go.hidden_states[-1]
                        _s, _e = ST2_AFS + ST2_FPF - 1, ST2_AFS + ST2_FPF * ST2_T
                        agent_hidden = lh[:, _s:_e:ST2_FPF, :].to(torch.float32).detach()  # (B,64,3072)

                    h_t_pred = ht_adapter(agent_hidden)  # (B,64,2048) fp32
                    ms = active_mask.sum().clamp(min=1e-8)
                    cos_m = F.cosine_similarity(h_t_pred, ht_target, dim=-1)
                    cos_loss = 1.0 - (cos_m * active_mask).sum() / ms
                    mean_cos = (cos_m * active_mask).sum() / ms
                    mse_m = F.mse_loss(h_t_pred, ht_target, reduction="none").mean(dim=-1)
                    L_latent = cos_loss + 0.5 * ((mse_m * active_mask).sum() / ms)

                    # coupling CE (sub congelato, h_t castato bf16)
                    h_t_sub = h_t_pred.to(torch.bfloat16)
                    sub_logits = sub_st2.forward_sub_talker_finetune(h_t_sub, audio_targets)
                    L_coup = sub_st2.coupling_loss(sub_logits, audio_targets, frame_weights=active_mask)

                    micro_loss = (L_latent + STAGE2_LAMBDA_COUPLING * L_coup) / STAGE2_GRAD_ACCUM
                    micro_loss.backward()
                    acc_loss += micro_loss.item() * STAGE2_GRAD_ACCUM
                    acc_lat += L_latent.item()
                    acc_coup += L_coup.item()
                    acc_cos += mean_cos.item()

                gn = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=STAGE2_GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()
                base_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in gemma_base.parameters())
                sub_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in sub_st2.parameters())
                info = {
                    "step": step,
                    "loss": round(acc_loss / STAGE2_GRAD_ACCUM, 4),
                    "latent": round(acc_lat / STAGE2_GRAD_ACCUM, 4),
                    "coupling": round(acc_coup / STAGE2_GRAD_ACCUM, 4),
                    "cos_sim": round(acc_cos / STAGE2_GRAD_ACCUM, 4),
                    "grad_norm": round(float(gn.item()), 4) if hasattr(gn, "item") else float(gn),
                    "base_grad_zero": not base_g,
                    "sub_grad_zero": not sub_g,
                    "time_s": round(time.time() - t0s, 2),
                }
                step_logs.append(info)
                print(f"[ST2] step {step}/{PHASE2_SMOKE_BATCHES} | loss {info['loss']:.4f} "
                      f"(latent {info['latent']:.4f} coup {info['coupling']:.4f}) | cos {info['cos_sim']:.4f} "
                      f"| grad {info['grad_norm']:.4f} | base0 {info['base_grad_zero']} sub0 {info['sub_grad_zero']} "
                      f"| {info['time_s']}s", flush=True)
                if step % STAGE2_CKPT_EVERY == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save(ht_adapter.state_dict(), f"checkpoints/ht_adapter_stage2_{step}.pt")
                    print(f"[ST2] checkpoint periodico: checkpoints/ht_adapter_stage2_{step}.pt", flush=True)

            os.makedirs("checkpoints", exist_ok=True)
            _ck_smoke = PHASE2_SMOKE_BATCHES <= 8  # smoke vs run completo
            ck = "checkpoints/ht_adapter_stage2_smoke.pt" if _ck_smoke else "checkpoints/ht_adapter_stage2_final.pt"
            torch.save(ht_adapter.state_dict(), ck)
            report = {
                "status": "PASS",
                "phase": "STAGE2_FITNETS_SMOKE",
                "batches": PHASE2_SMOKE_BATCHES,
                "trainable_params": n_tr,
                "base_frozen_verified": all(s["base_grad_zero"] for s in step_logs),
                "sub_frozen_verified": all(s["sub_grad_zero"] for s in step_logs),
                "final": step_logs[-1] if step_logs else {},
                "all_steps": step_logs,
            }
            _outd = "/kaggle/output" if os.path.exists("/kaggle/output") else "."
            with open(os.path.join(_outd, "stage2_smoke_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"[ST2] checkpoint salvato: {ck}", flush=True)
            print(f"[ST2] report: {os.path.join(_outd, 'stage2_smoke_report.json')}", flush=True)
            print(">>> STAGE 2 SMOKE COMPLETATO CON SUCCESSO <<<", flush=True)
        except Exception:
            print("[ST2] ERRORE durante lo smoke Stage 2", flush=True)
            traceback.print_exc()

    # ---- SEZIONE OPZIONALE: FIDELITY_PROBE (Confronto rigoroso Teacher CodePredictor vs QwenSubTalker) ----
    if FIDELITY and train_wins:
        print("\n" + "=" * 78, flush=True)
        print("=== FIDELITY PROBE DECISIVO: CONFRONTO TEACHER CODE_PREDICTOR vs SUB-TALKER ===", flush=True)
        print("=" * 78, flush=True)

        sub_candidates = [
            "/kaggle/input/datasets/andrlab/gemma-4-omni-module/qwen_subtalker",
            "/kaggle/input/andrlab/gemma-4-omni-module/qwen_subtalker",
            "qwen_subtalker",
            "/kaggle/working/qwen_subtalker",
        ]
        sub_path = next((p for p in sub_candidates if os.path.isdir(p)), sub_candidates[0])
        sub_bf16, sub_cfg = QS.load_qwen_subtalker(sub_path, device="cuda", dtype=torch.bfloat16)
        sub_bf16.eval()
        print(f"[Probe] Sub-talker caricato in bf16 da: {sub_path}", flush=True)

        # 1b. Teacher EAGER per il confronto forward: il teacher di default gira
        # con SDPA (transformers 5.x: _attn_implementation="sdpa" assegnato al
        # load), il sub e' eager manuale -> eager-vs-SDPA introduce un residuo
        # di precisione/interfaccia. Il confronto DECISIVO a parita' di
        # implementazione (eager-vs-eager) isola i bug strutturali del wrapper.
        _attn_teacher_orig = getattr(talker.code_predictor.config, "_attn_implementation", None)
        print(f"[Probe] attn_implementation teacher (default): {_attn_teacher_orig} -> forzo 'eager' per il confronto", flush=True)
        import copy as _copy
        _cp_cfg = _copy.deepcopy(TCFG.code_predictor_config)
        _cp_cfg._attn_implementation = "eager"
        cp_eager = _M.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration(_cp_cfg, TCFG).to(device="cuda", dtype=torch.bfloat16)
        _miss, _unexp = cp_eager.load_state_dict(talker.code_predictor.state_dict(), strict=False)
        if _miss:
            raise RuntimeError(f"code_predictor eager: pesi mancanti {_miss[:5]}")
        cp_eager.eval()
        print(f"[Probe] code_predictor EAGER istanziato (pesi da teacher: {len(_unexp) if _unexp else 0} inattesi)", flush=True)

        # 2. Confronto Configurazione
        tch_cp_cfg = getattr(TCFG, "code_predictor_config", None)
        if tch_cp_cfg is None:
            tch_cp_cfg = getattr(CFG, "code_predictor_config", None)

        keys_to_compare = [
            ("hidden_size", 1024),
            ("num_hidden_layers", 5),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 8),
            ("head_dim", 128),
            ("intermediate_size", 3072),
            ("rms_norm_eps", 1e-6),
            ("vocab_size", 2048),
            ("num_code_groups", 16),
            ("max_position_embeddings", 65536),
            ("rope_theta", 10000.0),
            ("cb0_vocab_size", 3072),
        ]

        config_report = {}
        mismatches_cfg = []
        print("\n--- 1. CONFRONTO CONFIGURAZIONE ---", flush=True)
        for k, default_val in keys_to_compare:
            tch_val = getattr(tch_cp_cfg, k, None) if tch_cp_cfg is not None else None
            if tch_val is None:
                if k == "cb0_vocab_size":
                    tch_val = getattr(TCFG, "vocab_size", 3072)
                elif k == "talker_hidden_size":
                    tch_val = getattr(TCFG, "hidden_size", 2048)
                else:
                    tch_val = getattr(TCFG, k, default_val)
            sub_val = sub_cfg.get(k, None)
            match = (tch_val == sub_val) or (
                isinstance(tch_val, (int, float)) and isinstance(sub_val, (int, float)) and abs(float(tch_val) - float(sub_val)) < 1e-7
            )
            config_report[k] = {"teacher": tch_val, "sub": sub_val, "match": bool(match)}
            status = "OK" if match else "MISMATCH"
            print(f"  [{status:8s}] {k:24s}: teacher={tch_val} | sub={sub_val}", flush=True)
            if not match:
                mismatches_cfg.append(f"{k}: teacher={tch_val} != sub={sub_val}")

        # 3. Mappatura e Confronto Pesi per Modulo (89 tensori)
        print("\n--- 2. CONFRONTO PESI PER MODULO (89 TENSIORI ATTESI) ---", flush=True)
        mapping = {
            "cb0_embedding.weight": "talker.model.codec_embedding.weight",
            "small_to_mtp_projection.weight": "talker.code_predictor.small_to_mtp_projection.weight",
            "small_to_mtp_projection.bias": "talker.code_predictor.small_to_mtp_projection.bias",
            "norm.weight": "talker.code_predictor.model.norm.weight",
        }
        for i in range(15):
            mapping[f"group_embeddings.{i}.weight"] = f"talker.code_predictor.model.codec_embedding.{i}.weight"
            mapping[f"lm_head.{i}.weight"] = f"talker.code_predictor.lm_head.{i}.weight"
        for l in range(5):
            mapping[f"layers.{l}.input_layernorm.weight"] = f"talker.code_predictor.model.layers.{l}.input_layernorm.weight"
            mapping[f"layers.{l}.self_attn.q_proj.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.q_proj.weight"
            mapping[f"layers.{l}.self_attn.k_proj.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.k_proj.weight"
            mapping[f"layers.{l}.self_attn.v_proj.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.v_proj.weight"
            mapping[f"layers.{l}.self_attn.o_proj.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.o_proj.weight"
            mapping[f"layers.{l}.self_attn.q_norm.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.q_norm.weight"
            mapping[f"layers.{l}.self_attn.k_norm.weight"] = f"talker.code_predictor.model.layers.{l}.self_attn.k_norm.weight"
            mapping[f"layers.{l}.post_attention_layernorm.weight"] = f"talker.code_predictor.model.layers.{l}.post_attention_layernorm.weight"
            mapping[f"layers.{l}.mlp.gate_proj.weight"] = f"talker.code_predictor.model.layers.{l}.mlp.gate_proj.weight"
            mapping[f"layers.{l}.mlp.up_proj.weight"] = f"talker.code_predictor.model.layers.{l}.mlp.up_proj.weight"
            mapping[f"layers.{l}.mlp.down_proj.weight"] = f"talker.code_predictor.model.layers.{l}.mlp.down_proj.weight"

        tch_sd = teacher.state_dict()
        sub_sd = sub_bf16.state_dict()

        weight_diffs = {}
        max_w_diff_global = 0.0
        total_params_compared = 0
        weights_mismatch = []

        module_groups = {
            "cb0_embedding (1x)": ["cb0_embedding.weight"],
            "group_embeddings (15x)": [f"group_embeddings.{i}.weight" for i in range(15)],
            "small_to_mtp_projection (2x)": ["small_to_mtp_projection.weight", "small_to_mtp_projection.bias"],
            "layers (5 x 11 = 55x)": [k for k in mapping if k.startswith("layers.")],
            "norm (1x)": ["norm.weight"],
            "lm_head (15x)": [f"lm_head.{i}.weight" for i in range(15)],
        }

        for group_name, keys in module_groups.items():
            group_max_diff = 0.0
            for sub_k in keys:
                tch_k = mapping[sub_k]
                if sub_k not in sub_sd:
                    weights_mismatch.append(f"CHIAVE MANCANTE in SUB: {sub_k}")
                    continue
                if tch_k not in tch_sd:
                    weights_mismatch.append(f"CHIAVE MANCANTE in TEACHER: {tch_k}")
                    continue
                w_sub = sub_sd[sub_k]
                w_tch = tch_sd[tch_k]
                if tuple(w_sub.shape) != tuple(w_tch.shape):
                    weights_mismatch.append(f"SHAPE MISMATCH per {sub_k}: sub={tuple(w_sub.shape)} vs tch={tuple(w_tch.shape)}")
                    continue
                diff = (w_sub.to(torch.float32) - w_tch.to(torch.float32)).abs().max().item()
                weight_diffs[sub_k] = {"teacher_key": tch_k, "shape": list(w_sub.shape), "max_diff": diff}
                if diff > group_max_diff:
                    group_max_diff = diff
                if diff > max_w_diff_global:
                    max_w_diff_global = diff
                if diff > 1e-4:
                    weights_mismatch.append(f"{sub_k} <-> {tch_k}: max_diff={diff:.6e} (shape {list(w_sub.shape)})")
                total_params_compared += w_sub.numel()

            status_grp = "IDENTICI (max diff=0)" if group_max_diff < 1e-6 else f"DIFF={group_max_diff:.2e}"
            print(f"  Modulo: {group_name:28s} | {len(keys):2d} tensori | max |w_sub - w_tch| = {group_max_diff:.6e} -> {status_grp}", flush=True)

        print(f"\n  TOTALE TENSIORI CONFRONTATI: {len(mapping)}/89 ({total_params_compared/1e6:.2f}M parametri)", flush=True)
        print(f"  MAX WEIGHT DIFF GLOBALE    : {max_w_diff_global:.6e} (soglia identita: <1e-4)", flush=True)

        # 4. Confronto Forward Pass in BF16
        print("\n--- 3. CONFRONTO FORWARD PASS (STESSO DTYPE BF16, STESSO INPUT) ---", flush=True)
        w0 = train_wins[0]
        idx0 = w0["idx"]
        n_real0 = w0["n_real"]
        item0 = ds_train[idx0]
        text0 = text_by_window[idx0] if text_by_window is not None else ""
        gt0 = item0["audio_targets"][:n_real0].to("cuda").long()  # (T_audio, 16)
        act0 = torch.from_numpy(w0["active"]).to("cuda") if "active" in w0 else torch.ones(n_real0, dtype=torch.bool, device="cuda")

        ht_corr_np = extract_ht(item0, n_real0, text0, return_shifted=False)
        ht_corr_bf16 = torch.from_numpy(ht_corr_np).to("cuda", dtype=torch.bfloat16)

        with torch.inference_mode():
            # Teacher forward -- STESSO percorso di forward_sub_talker_finetune ma su
            # code_predictor EAGER (parita' di implementazione col sub): costruzione
            # inputs_embeds = [h_t] + [cb0 (codec_embedding talker)] + [cb1..cb14
            # (codec_embedding code_predictor)], poi forward_finetune.
            _emb_list = [ht_corr_bf16.unsqueeze(1)]
            _emb_list.append(talker.model.codec_embedding(gt0[:, :1]))
            for _i in range(1, 15):
                _emb_list.append(cp_eager.get_input_embeddings()[_i - 1](gt0[:, _i:_i + 1]))
            _inputs_tch = torch.cat(_emb_list, dim=1)  # (T, 16, 2048)
            _out_tch = cp_eager.forward_finetune(inputs_embeds=_inputs_tch, labels=gt0[:, 1:])
            logits_tch = _out_tch.logits
            # Sub-talker forward (bf16)
            logits_sub = sub_bf16.forward_sub_talker_finetune(ht_corr_bf16.unsqueeze(0), gt0.unsqueeze(0)).squeeze(0)

            # Delta logits
            diff_logits = (logits_tch.float() - logits_sub.float()).abs()
            max_logit_diff = float(diff_logits.max().item())
            mean_logit_diff = float(diff_logits.mean().item())

            max_coord = torch.nonzero(diff_logits == diff_logits.max())[0].tolist()
            max_frame, max_channel, max_vocab = max_coord[0], max_coord[1], max_coord[2]

            # Cross entropy
            per_tch = F.cross_entropy(
                logits_tch.reshape(-1, 2048), gt0[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(n_real0, 15)
            ce_tch_act = float(per_tch[act0].mean().item()) if int(act0.sum()) else float(per_tch.mean().item())
            ce_tch_all = float(per_tch.mean().item())

            per_sub = F.cross_entropy(
                logits_sub.reshape(-1, sub_bf16.cfg["vocab_size"]), gt0[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(n_real0, 15)
            ce_sub_act = float(per_sub[act0].mean().item()) if int(act0.sum()) else float(per_sub.mean().item())
            ce_sub_all = float(per_sub.mean().item())

        print(f"  Logits Teacher shape: {list(logits_tch.shape)} (dtype: {logits_tch.dtype})", flush=True)
        print(f"  Logits Sub     shape: {list(logits_sub.shape)} (dtype: {logits_sub.dtype})", flush=True)
        print(f"  MAX  |Delta logits| : {max_logit_diff:.6f}", flush=True)
        print(f"  MEAN |Delta logits| : {mean_logit_diff:.6f}", flush=True)
        print(f"  Max diff coordinate : Frame {max_frame}/{n_real0}, Canale/Gruppo {max_channel}/15 (cb{max_channel+1}), Vocab Token {max_vocab}/2048", flush=True)
        print(f"  CE Teacher (active) : {ce_tch_act:.4f} nats | (all): {ce_tch_all:.4f} nats", flush=True)
        print(f"  CE Sub     (active) : {ce_sub_act:.4f} nats | (all): {ce_sub_all:.4f} nats", flush=True)
        print(f"  Gap CE (Sub - Tch)  : {ce_sub_act - ce_tch_act:+.4f} nats", flush=True)

        # 5. Verdetto e Gate Quantitativo
        print("\n" + "=" * 78, flush=True)
        print("=== 4. VERDETTO & DECISION GATE ===", flush=True)
        print("=" * 78, flush=True)

        gate_weights_identical = (max_w_diff_global < 1e-4 and len(weights_mismatch) == 0)
        gate_forward_identical = (max_logit_diff < 0.05)

        if gate_weights_identical and gate_forward_identical:
            verdict_code = "SUB_EQUALS_TEACHER"
            verdict_msg = (
                "PROMOSSO AL 100%: SUB-TALKER == TEACHER CODE_PREDICTOR (Pesi e Forward identici in bf16).\n"
                "-> Il gap di 7.03 nello smoke v9 era un artefatto di precisione (bf16 vs fp32) o test sfasato.\n"
                "-> AZIONE: Procedere direttamente con FitNets Stage 2 con il sub-talker congelato (tetto CE ~5.66, NO Stage 1)."
            )
        elif not gate_weights_identical:
            verdict_code = "WEIGHTS_MISMATCH"
            verdict_msg = (
                f"BOCCIATO (PESI DIVERSI): Il dataset qwen_subtalker NON contiene i pesi esatti del checkpoint Base (max diff={max_w_diff_global:.4e}).\n"
                f"-> {len(weights_mismatch)} discrepanze rilevate.\n"
                "-> AZIONE: Rigenerare il dataset qwen_subtalker eseguendo extract_qwen_subtalker.py con download pulito dal checkpoint Base."
            )
        else:
            verdict_code = "FORWARD_WRAPPER_BUG"
            verdict_msg = (
                f"BOCCIATO (BUG NEL FORWARD WRAPPER): I pesi sono identici (max diff={max_w_diff_global:.2e}) ma i logits divergono (max delta_logits={max_logit_diff:.4f} > 0.05).\n"
                "-> Causa principale: RoPE position_ids (S=1 collassa le posizioni a 0 in _forward_flat), causal mask o proiezioni.\n"
                "-> AZIONE: Correggere qwen_subtalker.py e rieseguire il probe."
            )

        print(f"ESITO PROBE : [{verdict_code}]", flush=True)
        print(verdict_msg, flush=True)
        print("=" * 78, flush=True)

        fidelity_report = {
            "verdict_code": verdict_code,
            "verdict_message": verdict_msg,
            "gate_weights_identical": gate_weights_identical,
            "gate_forward_identical": gate_forward_identical,
            "max_weight_diff_global": max_w_diff_global,
            "num_weight_mismatches": len(weights_mismatch),
            "weight_mismatches_sample": weights_mismatch[:10],
            "max_logit_diff": max_logit_diff,
            "mean_logit_diff": mean_logit_diff,
            "max_logit_diff_coord": {"frame": max_frame, "channel": max_channel, "vocab": max_vocab},
            "ce_teacher_active": ce_tch_act,
            "ce_teacher_all": ce_tch_all,
            "ce_subtalker_active": ce_sub_act,
            "ce_subtalker_all": ce_sub_all,
            "ce_gap_active": ce_sub_act - ce_tch_act,
            "config_report": config_report,
            "config_mismatches": mismatches_cfg,
        }
        os.makedirs("/kaggle/output", exist_ok=True)
        with open("/kaggle/output/fidelity_report.json", "w", encoding="utf-8") as f:
            json.dump(fidelity_report, f, indent=2)
        print("Report Fidelity salvato in: /kaggle/output/fidelity_report.json", flush=True)
        print("\n=== FIDELITY PROBE COMPLETATO: skip estrazione e training ===", flush=True)
        sys.stdout.flush()
        sys.exit(0)

    # ---- estrazione (saltata nello smoke STAGE2: i target sono calcolati in batch) ----
    if STAGE2:
        print("estrazione h_t massiva saltata (STAGE2 smoke attivo)", flush=True)
    OUT = "/kaggle/output/fitnets_targets"
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    t0 = time.time()
    for k, w in enumerate(train_wins if not STAGE2 else []):
        idx = w["idx"]
        item = ds_train[idx]
        text = text_by_window[idx] if text_by_window is not None else ""
        ht = extract_ht(item, w["n_real"], text)  # (T_audio, 2048) fp32
        if ht.shape[0] != w["n_real"]:
            print(f"  WARN win{idx}: ht {ht.shape[0]} != n_real {w['n_real']}", flush=True)
        np.save(os.path.join(OUT, f"ht_{idx:05d}.npy"), ht.astype(np.float16))
        manifest.append({"idx": idx, "n_real": w["n_real"]})
        if (k + 1) % 100 == 0 or k == len(train_wins) - 1:
            print(f"  [{k+1}/{len(train_wins)}] win{idx}: ht={ht.shape} "
                  f"rms={float(np.sqrt(np.mean(ht.astype(np.float32)**2))):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    with open(os.path.join(OUT, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    print(f"TARGET SALVATI: {len(manifest)} finestre in {OUT}", flush=True)

    # ---- SANITY & TEST DISCRIMINANTE 4-VIE: h_t corretto vs shifted x Teacher vs Sub-Talker ----
    if SANITY and train_wins:
        w = train_wins[0]
        idx = w["idx"]
        item = ds_train[idx]
        text = text_by_window[idx] if text_by_window is not None else ""
        gt = item["audio_targets"][:w["n_real"]].to("cuda").long()  # (T_audio, 16)
        cb0 = gt[:, 0]
        active = torch.from_numpy(w["active"]).to("cuda")

        # Estrai sia ht corretto che ht shifted (per comparazione diretta del bug)
        ht_corr_np, ht_shift_np = extract_ht(item, w["n_real"], text, return_shifted=True)
        ht_corr = torch.from_numpy(ht_corr_np).to("cuda", dtype=talker.dtype)
        ht_shift = torch.from_numpy(ht_shift_np).to("cuda", dtype=talker.dtype)

        print("\n" + "=" * 70, flush=True)
        print("=== TEST DISCRIMINANTE 4-VIE (TEACHER vs SUB-TALKER, CORRETTO vs SHIFTED) ===", flush=True)
        print("=" * 70, flush=True)

        with torch.inference_mode():
            # A) TEACHER UFFICIALE con h_t CORRETTO
            logits_tch_corr, _ = talker.forward_sub_talker_finetune(gt, ht_corr)
            per_tch_corr = F.cross_entropy(
                logits_tch_corr.reshape(-1, 2048), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(w["n_real"], 15)
            ce_tch_corr_act = float(per_tch_corr[active].mean()) if int(active.sum()) else float("nan")
            ce_tch_corr_all = float(per_tch_corr.mean())

            # B) TEACHER UFFICIALE con h_t SHIFTED (vecchio bug)
            logits_tch_shift, _ = talker.forward_sub_talker_finetune(gt, ht_shift)
            per_tch_shift = F.cross_entropy(
                logits_tch_shift.reshape(-1, 2048), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(w["n_real"], 15)
            ce_tch_shift_act = float(per_tch_shift[active].mean()) if int(active.sum()) else float("nan")
            ce_tch_shift_all = float(per_tch_shift.mean())

            # C) NOSTRO SUB-TALKER STANDALONE con h_t CORRETTO
            logits_sub_corr = sub.forward_sub_talker_finetune(ht_corr.float().unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
            per_sub_corr = F.cross_entropy(
                logits_sub_corr.reshape(-1, sub.cfg["vocab_size"]), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(w["n_real"], 15)
            ce_sub_corr_act = float(per_sub_corr[active].mean()) if int(active.sum()) else float("nan")
            ce_sub_corr_all = float(per_sub_corr.mean())

            # D) NOSTRO SUB-TALKER STANDALONE con h_t SHIFTED (vecchio bug)
            logits_sub_shift = sub.forward_sub_talker_finetune(ht_shift.float().unsqueeze(0), gt.unsqueeze(0)).squeeze(0)
            per_sub_shift = F.cross_entropy(
                logits_sub_shift.reshape(-1, sub.cfg["vocab_size"]), gt[:, 1:].reshape(-1),
                reduction="none", ignore_index=-100,
            ).reshape(w["n_real"], 15)
            ce_sub_shift_act = float(per_sub_shift[active].mean()) if int(active.sum()) else float("nan")
            ce_sub_shift_all = float(per_sub_shift.mean())

            # Discrepanza matematica esatta tra Teacher e Sub-Talker
            max_logit_diff = float((logits_tch_corr.float() - logits_sub_corr.float()).abs().max())

        print(f"1. TEACHER UFFICIALE + h_t CORRETTO : ce_act={ce_tch_corr_act:.3f} | ce_all={ce_tch_corr_all:.3f}", flush=True)
        print(f"2. TEACHER UFFICIALE + h_t SHIFTED  : ce_act={ce_tch_shift_act:.3f} | ce_all={ce_tch_shift_all:.3f}", flush=True)
        print(f"3. SUB-TALKER NOSTRO + h_t CORRETTO : ce_act={ce_sub_corr_act:.3f} | ce_all={ce_sub_corr_all:.3f}", flush=True)
        print(f"4. SUB-TALKER NOSTRO + h_t SHIFTED  : ce_act={ce_sub_shift_act:.3f} | ce_all={ce_sub_shift_all:.3f}", flush=True)
        print(f"5. FEDELTA' IMPLEMENTAZIONE (|Tch - Sub| max diff): {max_logit_diff:.6f}", flush=True)
        print("=" * 70, flush=True)

        # Generazione audio greedy con h_t corretto
        with torch.inference_mode():
            ac = sub.sample_acoustic_codes(ht_corr.float().unsqueeze(0), cb0.unsqueeze(0), temperature=0.0)
        full_codes = torch.cat([cb0.unsqueeze(-1), ac.squeeze(0)], dim=-1)  # (T_audio, 16) long

        # Decodifica Code2Wav passando LongTensor (nessun warning FloatTensor)
        wav = decoder.decode(full_codes.cpu().long())
        wav_path = "/kaggle/output/sanity_ht_native_corr.wav"
        import scipy.io.wavfile as wavfile
        wavfile.write(wav_path, SAMPLE_RATE,
                      (wav.astype(np.float32) * 32767).astype(np.int16))
        wav_gt_path = "/kaggle/output/sanity_gt.wav"
        wav_gt = decoder.decode(torch.from_numpy(KP._trim_trailing_silence(gt.cpu().numpy())).long())
        wavfile.write(wav_gt_path, SAMPLE_RATE,
                      (wav_gt.astype(np.float32) * 32767).astype(np.int16))
        print(f"SANITY WAV: {wav_path} e {wav_gt_path}", flush=True)

        # Accuratezza greedy vs oracle cb1-15
        oracle = gt[:, 1:].cpu().numpy()
        gen = full_codes[:, 1:].cpu().numpy()
        acc = float((gen == oracle).mean())
        print(f"SANITY accuratezza cb1-15 con h_t CORRETTO (greedy vs oracle): {acc:.4f}", flush=True)

        report = {
            "idx": idx, "n_real": w["n_real"],
            "ce_teacher_correct_act": ce_tch_corr_act, "ce_teacher_correct_all": ce_tch_corr_all,
            "ce_teacher_shifted_act": ce_tch_shift_act, "ce_teacher_shifted_all": ce_tch_shift_all,
            "ce_subtalker_correct_act": ce_sub_corr_act, "ce_subtalker_correct_all": ce_sub_corr_all,
            "ce_subtalker_shifted_act": ce_sub_shift_act, "ce_subtalker_shifted_all": ce_sub_shift_all,
            "max_logit_diff_tch_vs_sub": max_logit_diff,
            "acc_cb115_greedy_correct": acc,
            "ht_shape": list(ht_corr.shape),
        }
        with open("/kaggle/output/sanity_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("SANITY REPORT COMPLETO: /kaggle/output/sanity_report.json", flush=True)

    # ---- SEZIONE OPZIONALE: SMOKE_B (Verifica Two-Stage: Training Sub-Talker su h_t nativi) ----
    if SMOKE_B and train_wins:
        print("\n" + "=" * 70, flush=True)
        print("=== SMOKE B: TWO-STAGE VERIFICATION (TRAINING SUB-TALKER SU H_T NATIVI) ===", flush=True)
        print("=" * 70, flush=True)

        # 1. Preparazione Dataset In-Memory per Smoke B
        smoke_items = []
        for w in train_wins:
            idx = w["idx"]
            n_real = w["n_real"]
            ht_path = os.path.join(OUT, f"ht_{idx:05d}.npy")
            if not os.path.exists(ht_path):
                continue
            ht_arr = np.load(ht_path).astype(np.float32)  # (n_real, 2048)
            item = ds_train[idx]
            gt_arr = item["audio_targets"][:n_real].long().numpy()  # (n_real, 16)
            active_arr = w.get("active", _mel_active_mask(ds_train, idx, n_real))
            smoke_items.append({
                "idx": idx,
                "n_real": n_real,
                "ht": torch.from_numpy(ht_arr),
                "gt": torch.from_numpy(gt_arr),
                "active": torch.from_numpy(active_arr).bool(),
            })

        n_tot = len(smoke_items)
        if n_tot == 0:
            print("  WARN SMOKE B: nessun file h_t trovato in cache, skip smoke B.", flush=True)
        else:
            n_val = max(2, min(16, int(n_tot * 0.2))) if n_tot > 4 else 1
            n_train = max(1, n_tot - n_val)
            smoke_train = smoke_items[:n_train]
            smoke_val = smoke_items[n_train:] if n_tot > 1 else smoke_items[:1]
            print(f"Smoke B dataset: {n_tot} finestre -> {len(smoke_train)} train, {len(smoke_val)} val", flush=True)
            print(f"Iperparametri: steps={SMOKE_B_STEPS} | lr={SMOKE_B_LR} | batch_size={SMOKE_B_BATCH}", flush=True)

            # 2. Inizializzazione Sub-Talker & Ottimizzatore
            # Float32 per massima stabilita' numerica e velocita' nativa su T4
            sub = sub.to(device="cuda", dtype=torch.float32)
            sub.train()
            sub.requires_grad_(True)

            optimizer = torch.optim.AdamW(
                sub.parameters(),
                lr=SMOKE_B_LR,
                weight_decay=1e-2,
                betas=(0.9, 0.98),
                eps=1e-8,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=SMOKE_B_STEPS,
                eta_min=1e-5,
            )

            def collate_smoke_batch(batch_list, device="cuda"):
                B = len(batch_list)
                T_max = max(item["n_real"] for item in batch_list)
                batch_ht = torch.zeros((B, T_max, 2048), dtype=torch.float32, device=device)
                batch_gt = torch.full((B, T_max, 16), PAD, dtype=torch.long, device=device)
                batch_act = torch.zeros((B, T_max), dtype=torch.bool, device=device)
                for b_idx, item in enumerate(batch_list):
                    nr = item["n_real"]
                    batch_ht[b_idx, :nr] = item["ht"].to(device)
                    batch_gt[b_idx, :nr] = item["gt"].to(device)
                    batch_act[b_idx, :nr] = item["active"].to(device)
                return batch_ht, batch_gt, batch_act

            val_ht, val_gt, val_act = collate_smoke_batch(smoke_val, device="cuda")

            def evaluate_smoke_val():
                sub.eval()
                with torch.inference_mode():
                    logits = sub.forward_sub_talker_finetune(val_ht, val_gt)  # (B, T, 15, V)
                    per_frame = F.cross_entropy(
                        logits.reshape(-1, sub.cfg["vocab_size"]),
                        val_gt[:, :, 1:].reshape(-1),
                        reduction="none",
                        ignore_index=PAD,
                    ).reshape(val_gt.shape[0], val_gt.shape[1], 15)

                    per_frame_mean = per_frame.mean(dim=-1)  # (B, T)
                    act_losses = per_frame_mean[val_act]
                    ce_val_act = float(act_losses.mean().item()) if act_losses.numel() > 0 else float("nan")

                    valid_mask = (val_gt[:, :, 0] != PAD)
                    all_losses = per_frame_mean[valid_mask]
                    ce_val_all = float(all_losses.mean().item()) if all_losses.numel() > 0 else float("nan")

                    greedy_pred = logits.argmax(dim=-1)  # (B, T, 15)
                    correct = (greedy_pred == val_gt[:, :, 1:])
                    acc_act = float(correct[val_act].float().mean().item()) if val_act.any() else 0.0
                sub.train()
                return ce_val_act, ce_val_all, acc_act

            # Baseline pre-training (Step 0)
            ce_0_act, ce_0_all, acc_0 = evaluate_smoke_val()
            print(f"[Step   0/{SMOKE_B_STEPS}] BASELINE PRE-TRAIN: ce_val_act={ce_0_act:.4f} | ce_val_all={ce_0_all:.4f} | acc_val_act={acc_0:.4f}", flush=True)

            history = []
            t_start_smoke = time.time()

            # 3. Training Loop
            for step in range(1, SMOKE_B_STEPS + 1):
                cur_batch_size = min(SMOKE_B_BATCH, len(smoke_train))
                indices = np.random.choice(len(smoke_train), size=cur_batch_size, replace=False)
                batch_list = [smoke_train[i] for i in indices]
                batch_ht, batch_gt, _ = collate_smoke_batch(batch_list, device="cuda")

                optimizer.zero_grad(set_to_none=True)
                logits = sub.forward_sub_talker_finetune(batch_ht, batch_gt)
                loss = sub.coupling_loss(logits, batch_gt)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(sub.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                if step % 25 == 0 or step == SMOKE_B_STEPS:
                    ce_v_act, ce_v_all, acc_v = evaluate_smoke_val()
                    lr_curr = scheduler.get_last_lr()[0]
                    print(f"[Step {step:3d}/{SMOKE_B_STEPS}] loss_train={loss.item():.4f} | ce_val_act={ce_v_act:.4f} | ce_val_all={ce_v_all:.4f} | acc_val_act={acc_v:.4f} | lr={lr_curr:.2e}", flush=True)
                    history.append({
                        "step": step,
                        "loss_train": float(loss.item()),
                        "ce_val_act": ce_v_act,
                        "ce_val_all": ce_v_all,
                        "acc_val_act": acc_v,
                        "lr": lr_curr,
                    })

            t_smoke_elapsed = time.time() - t_start_smoke
            print(f"\nSMOKE B TRAINING COMPLETATO in {t_smoke_elapsed:.1f}s ({t_smoke_elapsed/SMOKE_B_STEPS*1000:.1f} ms/step)", flush=True)

            # 4. Verdetto & Gate Quantitativo
            final_ce_act = history[-1]["ce_val_act"] if history else ce_0_act
            final_acc = history[-1]["acc_val_act"] if history else acc_0
            delta_ce = ce_0_act - final_ce_act
            gate_passed = final_ce_act < 6.2

            print("=" * 70, flush=True)
            print(f"RISULTATO SMOKE B: CE 0 -> {ce_0_act:.4f} | CE FINALE -> {final_ce_act:.4f} (Delta: -{delta_ce:.4f} nats)", flush=True)
            print(f"ACCURACY CB1-15 : {acc_0:.4f} -> {final_acc:.4f}", flush=True)
            print(f"DECISION GATE   : {'[PROMOSSO] (CE < 6.2) -> Procedere con Stage 2 FitNets' if gate_passed else '[BOCCIATO] (CE >= 6.2) -> Passare ad Opzione A (Teacher Code Predictor) o D'}", flush=True)
            print("=" * 70, flush=True)

            # 5. Salvataggio Pesi e Configurazione
            sub.eval()
            sub.requires_grad_(False)
            trained_weights_path = os.path.join(OUT, "qwen_subtalker_stage1.safetensors")
            trained_cfg_path = os.path.join(OUT, "qwen_subtalker_stage1_config.json")
            _st.save_file(sub.state_dict(), trained_weights_path)
            with open(trained_cfg_path, "w", encoding="utf-8") as f:
                json.dump(sub.cfg, f, indent=2)
            print(f"Pesi Sub-Talker Stage 1 salvati in: {trained_weights_path}", flush=True)

            # 6. Sintesi Audio Sample Post-Training (Window 0 di Val)
            val_w0 = smoke_val[0]
            ht_sample = val_w0["ht"].to("cuda", dtype=torch.float32).unsqueeze(0)
            cb0_sample = val_w0["gt"][:, 0].to("cuda").unsqueeze(0)
            with torch.inference_mode():
                ac_sample = sub.sample_acoustic_codes(ht_sample, cb0_sample, temperature=0.0)
            full_sample_codes = torch.cat([cb0_sample.unsqueeze(-1), ac_sample], dim=-1).squeeze(0)  # (T, 16)

            wav_trained = decoder.decode(full_sample_codes.cpu().long())
            wav_trained_path = "/kaggle/output/smoke_b_trained_sample.wav"
            import scipy.io.wavfile as wavfile
            wavfile.write(wav_trained_path, SAMPLE_RATE, (wav_trained.astype(np.float32) * 32767).astype(np.int16))
            print(f"WAV Sintetizzato Post-Training: {wav_trained_path}", flush=True)

            smoke_report = {
                "n_train_windows": len(smoke_train),
                "n_val_windows": len(smoke_val),
                "steps": SMOKE_B_STEPS,
                "lr": SMOKE_B_LR,
                "batch_size": SMOKE_B_BATCH,
                "time_seconds": t_smoke_elapsed,
                "ce_initial_act": ce_0_act,
                "ce_initial_all": ce_0_all,
                "acc_initial": acc_0,
                "ce_final_act": final_ce_act,
                "ce_final_all": history[-1]["ce_val_all"] if history else ce_0_all,
                "acc_final": final_acc,
                "delta_ce": delta_ce,
                "gate_passed": gate_passed,
                "history": history,
            }
            with open("/kaggle/output/smoke_b_report.json", "w", encoding="utf-8") as f:
                json.dump(smoke_report, f, indent=2)
            print("Report Smoke B salvato in: /kaggle/output/smoke_b_report.json", flush=True)

    print("=== FITNETS EXTRACT COMPLETATO ===", flush=True)
    sys.stdout.flush()

except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    print("FITNETS EXTRACT FALLITA (eccezione rilanciata)", flush=True)
    sys.stdout.flush()
    raise
