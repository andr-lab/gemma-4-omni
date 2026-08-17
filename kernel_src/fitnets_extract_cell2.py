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
    import torch.nn.functional as F
    import numpy as np

    EXTRACT_N = int(os.environ.get("FITEX_N", "1000"))
    SANITY = os.environ.get("FITEX_SANITY", "1") == "1"
    START_IDX = int(os.environ.get("FITEX_START", "0"))
    SMOKE_B = os.environ.get("SMOKE_B", "0") == "1"
    SMOKE_B_STEPS = int(os.environ.get("SMOKE_B_STEPS", "300"))
    SMOKE_B_LR = float(os.environ.get("SMOKE_B_LR", "5e-4"))
    SMOKE_B_BATCH = int(os.environ.get("SMOKE_B_BATCH", "8"))

    import kaggle_pipeline as KP
    import qwen_subtalker as QS

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
        if SANITY:
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
        if len(train_wins) >= EXTRACT_N:
            break
    print(f"raccolte {len(train_wins)} finestre train (EXTRACT_N={EXTRACT_N})", flush=True)

    # ---- estrazione ----
    OUT = "/kaggle/output/fitnets_targets"
    os.makedirs(OUT, exist_ok=True)
    manifest = []
    t0 = time.time()
    for k, w in enumerate(train_wins):
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
