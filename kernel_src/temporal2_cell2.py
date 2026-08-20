# -*- coding: utf-8 -*-
# ============================================================================
# GEMMA 4 E2B FROZEN + CAUSAL TEMPORAL RESIDUAL ADAPTER V2 (Stage 2 Temporal 2)
# Modulo Temporale Causale v2: 4 blocchi Causal Multi-Head Self-Attention
# (d=1024, 8 teste, d_ff=2048, ReZero / Identity-Init su out_proj).
# + ResidualAdapter (CausalConv1D k=3 identity-init + Skip + MLP 1536->3072->SiLU->2048).
# Backbone Gemma 4 E2B: CONGELATO 4-bit (zero trainable params, forward no_grad).
# Sub-talker Qwen3-TTS: CONGELATO bf16 (coupling loss lambda=0.2).
# Teacher Qwen3-TTS: su CPU per estrazione target h_t (2048D, 12.5Hz).
# Obiettivo: estendere la profondita temporale a 4 blocchi causali (59.51M params)
# per superare il record di similarita coseno (cos val > 0.64).
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
    # CONFIGURAZIONE RUNTIME & IPERPARAMETRI (Environment Variables Centralizzate)
    # =========================================================================
    STAGE2_TEMPORAL2 = int(os.environ.get("STAGE2_TEMPORAL2", os.environ.get("STAGE2_TEMPORAL", "1")))
    SMOKE = int(os.environ.get("SMOKE", "1"))
    PHASE2_SMOKE_BATCHES = int(os.environ.get("PHASE2_SMOKE_BATCHES", "4" if SMOKE else "1200"))
    EXTRACT_N = int(os.environ.get("FITEX_N", "16" if SMOKE else "1000"))
    SANITY = os.environ.get("FITEX_SANITY", "0") == "1"
    ROPE_FIX = int(os.environ.get("ROPE_FIX", "1")) == 1

    # --- Iperparametri Modulo Temporale Causale v2 (4 blocchi) ---
    TEMPORAL_DIM = int(os.environ.get("TEMPORAL_DIM", "1024"))
    TEMPORAL_LAYERS = int(os.environ.get("TEMPORAL_LAYERS", "4"))
    TEMPORAL_HEADS = int(os.environ.get("TEMPORAL_HEADS", "8"))
    TEMPORAL_FF_DIM = int(os.environ.get("TEMPORAL_FF_DIM", "2048"))
    TEMPORAL_LR_ADAPTER = float(os.environ.get("TEMPORAL_LR_ADAPTER", "1.5e-4"))
    TEMPORAL_LR_MODULE = float(os.environ.get("TEMPORAL_LR_MODULE", "1.5e-4"))
    TEMPORAL_WARMUP_STEPS = int(os.environ.get("TEMPORAL_WARMUP", "1" if SMOKE else "100"))
    TEMPORAL_MIN_LR_RATIO = float(os.environ.get("TEMPORAL_MIN_LR_RATIO", "0.1"))
    STAGE2_WEIGHT_DECAY = float(os.environ.get("STAGE2_WEIGHT_DECAY", "1e-2"))
    STAGE2_GRAD_CLIP = float(os.environ.get("STAGE2_GRAD_CLIP", "1.0"))
    STAGE2_BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", "8"))
    STAGE2_GRAD_ACCUM = int(os.environ.get("STAGE2_GRAD_ACCUM", "2"))
    STAGE2_LAMBDA_COUPLING = float(os.environ.get("STAGE2_LAMBDA_COUPLING", "0.2"))
    STAGE2_CKPT_EVERY = int(os.environ.get("STAGE2_CKPT_EVERY", "2" if SMOKE else "250"))

    # --- Valutazione Offline & Sintesi Audio (EVAL TEMPORAL 2) ---
    EVAL_TEMPORAL2 = int(os.environ.get("EVAL_TEMPORAL2", os.environ.get("EVAL_TEMPORAL", "1")))
    EVAL_N_VAL = int(os.environ.get("EVAL_N_VAL", "4" if SMOKE else "100"))
    EVAL_BATCH = int(os.environ.get("EVAL_BATCH", "4"))

    # --- Dataset & Checkpoint Paths ---
    OUTPUT_DIR = "/kaggle/output" if os.path.exists("/kaggle/output") else "."
    TARGETS_DIR = os.path.join(OUTPUT_DIR, "fitnets_targets")
    QWEN3_TTS_SRC_DIR = "/kaggle/working/qwen3tts_src"

    QWEN_SUBTALKER_CANDIDATE_PATHS = (
        "/kaggle/input/datasets/andrlab/gemma-4-omni-module/qwen_subtalker",
        "/kaggle/input/andrlab/gemma-4-omni-module/qwen_subtalker",
        "qwen_subtalker",
        "/kaggle/working/qwen_subtalker",
    )

    ADAPTER_STAGE2_R2_CANDIDATE_PATHS = (
        "checkpoints/ht_adapter_stage2r2_final.pt",
        "/kaggle/working/checkpoints/ht_adapter_stage2r2_final.pt",
        "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
        "/kaggle/input/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
        "ht_adapter_stage2r2_final.pt",
    )

    import kaggle_pipeline as KP
    import qwen_subtalker as QS

    # ---- PATCH sample_acoustic_codes aggiornata (top_k/top_p/seed/per-level) ----
    try:
        import inspect as _inspect_p
        _sig_p = _inspect_p.signature(QS.Qwen3TTSSubTalker.sample_acoustic_codes)
        if "top_k" not in _sig_p.parameters:
            print("[PATCH sample_acoustic_codes] upgrade (top_k/top_p/seed/per-level)", flush=True)
            def _sample_acoustic_codes_v2(self, h_t, cb0, temperature=0.0, top_k=50, top_p=1.0,
                    seed=None, device=None, warn_cb0_clamp=True, temperatures_per_level=None):
                if temperatures_per_level is not None:
                    expected_len = self.cfg["num_code_groups"] - 1
                    if len(temperatures_per_level) != expected_len:
                        raise ValueError(
                            f"temperatures_per_level ha lunghezza {len(temperatures_per_level)}, "
                            f"attesa {expected_len} (num_code_groups - 1)."
                        )

                if seed is not None:
                    torch.manual_seed(seed)

                B, T = cb0.shape
                dev = device if device is not None else cb0.device
                cb0 = cb0.to(dev).long()
                if cb0.numel() > 0 and cb0.max() >= self.cfg["vocab_size"]:
                    if warn_cb0_clamp:
                        print(
                            f"[QwenSubTalker] [WARN] clamp cb0 campionato: max={int(cb0.max().item())} -> "
                            f"{self.cfg['vocab_size'] - 1} (semantic_head 4096 vs range GT 2048)."
                        )
                    cb0 = cb0.clamp(max=self.cfg["vocab_size"] - 1)

                h_flat = h_t.reshape(B * T, 1, self.cfg["talker_hidden_size"])
                full = torch.zeros(B, T, self.cfg["num_code_groups"], dtype=torch.long, device=dev)
                full[:, :, 0] = cb0
                seq = cb0.unsqueeze(-1)
                V = self.cfg["vocab_size"]
                for g in range(1, self.cfg["num_code_groups"]):
                    if g > 1:
                        full[:, :, 1:g] = seq[:, :, 1:]
                    logits = self._forward_flat(h_flat, full.reshape(B * T, self.cfg["num_code_groups"]))
                    lg = logits[:, g - 1, :].reshape(B, T, V)

                    lvl_temp = (
                        float(temperatures_per_level[g - 1])
                        if temperatures_per_level is not None
                        else float(temperature)
                    )

                    if lvl_temp <= 0.0:
                        sampled = lg.argmax(dim=-1)
                    else:
                        logits_step = lg / lvl_temp
                        if top_k is not None and 0 < top_k < V:
                            topk_vals, _ = torch.topk(logits_step, min(int(top_k), V), dim=-1)
                            kth_val = topk_vals[..., -1, None]
                            logits_step = torch.where(
                                logits_step < kth_val,
                                torch.tensor(float("-inf"), device=logits_step.device, dtype=logits_step.dtype),
                                logits_step,
                            )

                        if top_p is not None and 0.0 < top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits_step, descending=True, dim=-1)
                            sorted_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = False

                            indices_to_remove = sorted_indices_to_remove.scatter(
                                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                            )
                            logits_step = logits_step.masked_fill(indices_to_remove, float("-inf"))

                        probs = torch.nn.functional.softmax(logits_step, dim=-1)
                        flat_probs = probs.reshape(-1, V)
                        sampled = torch.multinomial(flat_probs, num_samples=1).reshape(B, T)

                    seq = torch.cat([seq, sampled.unsqueeze(-1)], dim=-1)
                return seq[:, :, 1:]
            QS.Qwen3TTSSubTalker.sample_acoustic_codes = _sample_acoustic_codes_v2
            print("[PATCH sample_acoustic_codes] applicata con successo", flush=True)
    except Exception:
        traceback.print_exc()

    # ---- FIX RoPE & Theta runtime ----
    if ROPE_FIX and hasattr(QS, "Qwen3TTSSubTalker"):
        def _fixed_forward_flat(self, h_t_flat, codec_ids_flat):
            inputs = [h_t_flat]
            inputs.append(self.cb0_embedding(codec_ids_flat[:, :1].clamp(min=0)))
            for g in range(self.cfg["num_code_groups"] - 2):
                inputs.append(self.group_embeddings[g](codec_ids_flat[:, g + 1 : g + 2].clamp(min=0)))
            inputs_embeds = torch.cat(inputs, dim=1)
            S = inputs_embeds.shape[1]
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

    # ---- Caricamento dati da memmap / parquet ----
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

    # ---- Clone repo ufficiale Qwen3-TTS per teacher ----
    _SRC = QWEN3_TTS_SRC_DIR
    if not os.path.isdir(_SRC):
        r = subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/QwenLM/Qwen3-TTS.git", _SRC],
            capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            print("git clone FAIL:", r.stderr[-500:], flush=True)
            raise RuntimeError("git clone Qwen3-TTS fallito")

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
            raise RuntimeError("stub Qwen3TTSTokenizer: non istanziare")
    _tok.Qwen3TTSTokenizer = _StubTokenizer
    sys.modules["qwen_tts.inference.qwen3_tts_tokenizer"] = _tok
    sys.path.insert(0, _SRC)
    import qwen_tts.core.models.modeling_qwen3_tts as _M
    from qwen_tts.core.models.processing_qwen3_tts import Qwen3TTSProcessor
    from transformers import AutoTokenizer
    print("repo Qwen3-TTS clonato e importato (tokenizer sox bypassati)", flush=True)

    # ---- Caricamento Teacher Qwen3-TTS (bf16; forward-only) ----
    t0 = time.time()
    import safetensors.torch as _st
    from huggingface_hub import hf_hub_download
    import transformers.modeling_rope_utils as _ropeu
    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = config.rope_theta
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        _dim = int(_head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
        return inv_freq, 1.0
    _ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    _cfg = _M.Qwen3TTSForConditionalGeneration.config_class.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    _cfg.talker_config.pad_token_id = None
    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _ckpt = hf_hub_download("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "model.safetensors")
    _sd = _st.load_file(_ckpt, device="cpu")
    _missing, _unexpected = teacher.load_state_dict(_sd, strict=False)
    del _sd
    torch.cuda.empty_cache()
    if _missing:
        raise RuntimeError(f"pesi teacher mancanti ({len(_missing)}): {_missing[:8]}")
    teacher = teacher.to(device="cuda", dtype=torch.bfloat16)
    teacher.eval()
    talker = teacher.talker
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", use_fast=True)
    proc = Qwen3TTSProcessor(tokenizer=tok)
    print(f"teacher caricato in {time.time()-t0:.1f}s | talker hidden={talker.config.hidden_size} layers={talker.config.num_hidden_layers}", flush=True)

    CFG = teacher.config
    TCFG = CFG.talker_config
    TTS_PAD, TTS_BOS, TTS_EOS = CFG.tts_pad_token_id, CFG.tts_bos_token_id, CFG.tts_eos_token_id
    CN, CTB, CTE, CPAD, CBOS, CEOS = (
        TCFG.codec_nothink_id, TCFG.codec_think_bos_id, TCFG.codec_think_eos_id,
        TCFG.codec_pad_id, TCFG.codec_bos_id, TCFG.codec_eos_token_id,
    )

    def build_inputs(text: str, audio_codes: torch.Tensor):
        assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = proc(text=assistant, return_tensors="pt")["input_ids"]
        text_ids = text_ids[:, :-5]
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

        input_ids[0, :3, 0] = text_ids[0, :3]
        input_ids[0, 3:7, 0] = TTS_PAD
        input_ids[0, 7, 0] = TTS_BOS
        input_ids[0, 8:8 + Tt - 3, 0] = text_ids[0, 3:]
        input_ids[0, 8 + Tt - 3, 0] = TTS_EOS
        input_ids[0, 8 + Tt - 2:8 + Tt + Ta, 0] = TTS_PAD
        text_mask[0, :8 + Tt + Ta] = True

        input_ids[0, 3:8, 1] = torch.tensor([CN, CTB, CTE, 0, CPAD], dtype=torch.long)
        input_ids[0, 8:8 + Tt - 3, 1] = CPAD
        input_ids[0, 8 + Tt - 3, 1] = CPAD
        input_ids[0, 8 + Tt - 2, 1] = CBOS
        input_ids[0, 8 + Tt - 1:8 + Tt - 1 + Ta, 1] = audio_codes[:, 0]
        input_ids[0, 8 + Tt - 1 + Ta, 1] = CEOS
        codec_emb_mask[0, 3:8 + Tt + Ta] = True
        codec_emb_mask[0, 6] = False
        codec_mask[0, 8 + Tt - 1:8 + Tt - 1 + Ta] = True
        attn[0, :8 + Tt + Ta] = True
        codec_ids[0, 8 + Tt - 1:8 + Tt - 1 + Ta, :] = audio_codes
        return input_ids, codec_ids, text_mask, codec_emb_mask, codec_mask, attn, pos_cbos

    def extract_ht(item, n_real, text, return_shifted: bool = False):
        audio_codes = item["audio_targets"][:n_real].long()
        input_ids, codec_ids, text_mask, codec_emb_mask, codec_mask, attn, pos_cbos = build_inputs(text, audio_codes)
        _tdev = next(talker.parameters()).device
        input_ids = input_ids.to(_tdev)
        codec_ids = codec_ids.to(_tdev)
        text_mask = text_mask.to(_tdev)
        codec_emb_mask = codec_emb_mask.to(_tdev)
        codec_mask = codec_mask.to(_tdev)
        attn = attn.to(_tdev)
        with torch.inference_mode():
            raw_text_emb = talker.model.text_embedding(input_ids[:, :, 0])
            input_text_emb = talker.text_projection(raw_text_emb) * text_mask.unsqueeze(-1)
            input_codec_emb = talker.model.codec_embedding(input_ids[:, :, 1]) * codec_emb_mask.unsqueeze(-1)
            input_embeddings = input_text_emb + input_codec_emb
            for i in range(1, 16):
                ci = talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                input_embeddings = input_embeddings + ci * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings.to(dtype=talker.dtype)
            outputs = talker(
                inputs_embeds=input_embeddings[:, :-1, :],
                attention_mask=attn[:, :-1],
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = outputs.hidden_states[0][-1]
            Ta = audio_codes.shape[0]
            ht_target = hidden_states[0, pos_cbos : pos_cbos + Ta, :]
        return ht_target.float().cpu().numpy()

    # ---- SCAN finestre attive ----
    PAD = KP.PAD_TARGET_ID
    SAMPLE_RATE = KP.SYNTH_SAMPLE_RATE
    MPF_MEL = KP.MEL_PER_AGENT_FRAME

    def _mel_active_mask(ds, idx, n_real):
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
        active = _mel_active_mask(ds, idx, n_real)
        if int(active.sum()) == 0:
            return None
        return {"idx": idx, "n_real": n_real, "active": active}

    train_wins = []
    for i in range(min(1024, len(ds_train))):
        w = _scan_window(ds_train, i)
        if w is None:
            continue
        train_wins.append(w)
        if len(train_wins) >= EXTRACT_N:
            break
    print(f"raccolte {len(train_wins)} finestre train (EXTRACT_N={EXTRACT_N}, SMOKE={SMOKE})", flush=True)

    # =========================================================================
    # ARCHITETTURA MODULO TEMPORALE CAUSALE V2 & RESIDUAL ADAPTER
    # =========================================================================
    class CausalConv1d(nn.Module):
        """Conv1D causale k=3 con left-pad (2,0): contesto [t-2,t-1,t]=240ms, 0 look-ahead.
        Identity-init: peso centrale=I, laterali=0 -> forward(x)==x a passo 0.
        """
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
        """Adapter MLP residuale: CausalConv1D -> LayerNorm( MLP(h) + skip(h) ).
        skip_proj: Linear(in_dim, out_dim)
        mlp: Linear(in_dim, hidden_dim) -> SiLU -> Linear(hidden_dim, out_dim)
        norm: LayerNorm(out_dim)
        """
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
        """Blocco Causal Multi-Head Self-Attention Pre-LN con SDPA nativo PyTorch."""
        def __init__(self, d_model=1024, n_heads=8, d_ff=2048, dropout=0.0):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            assert d_model % n_heads == 0, "d_model deve essere divisibile per n_heads"

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

            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.out_proj(attn_out)

            x = x + self.mlp(self.ln2(x))
            return x

    class CausalTemporalModule(nn.Module):
        """Modulo Temporale Causale v2 (4 blocchi Transformer causali Pre-LN).
        Mappa lo stato latente h_adapt (2048D) -> h_t (2048D) estendendo il campo
        recettivo e la capacita di rappresentazione a tutta la finestra storica T=64 (5.12s a 12.5Hz).
        Identity-Init su out_proj (ReZero): delta iniziale nullo -> warm-start esatto a passo 0.
        """
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

            # ReZero / Identity-Init: out_proj azzerata a passo 0 per warm-start esatto
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
        """Wrapper composito: ResidualAdapter (1536->2048) + CausalTemporalModule (2048->2048).
        Permette warm-start trasparente da checkpoint R2/R3 con strict=False (missing solo temporal.*).
        """
        def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072,
                     temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048,
                     use_causal_conv=True):
            super().__init__()
            self.adapter = ResidualAdapter(in_dim=in_dim, out_dim=out_dim, hidden_dim=hidden_dim, use_causal_conv=use_causal_conv)
            self.temporal = CausalTemporalModule(in_dim=out_dim, hidden_dim=temp_dim, n_layers=temp_layers, n_heads=temp_heads, d_ff=temp_ff)

        def forward(self, x):
            h_adapt = self.adapter(x)
            return self.temporal(h_adapt)

    def build_ht_temporal_adapter(_hid, temp_dim=1024, temp_layers=4, temp_heads=8, temp_ff=2048):
        return TemporalResidualAdapter(
            in_dim=_hid, out_dim=2048, hidden_dim=3072,
            temp_dim=temp_dim, temp_layers=temp_layers, temp_heads=temp_heads, temp_ff=temp_ff,
            use_causal_conv=True
        )

    # =========================================================================
    # SEZIONE 1: STAGE 2 TEMPORAL 2 TRAINING (Gemma Frozen + TemporalResidualAdapter v2)
    # =========================================================================
    if STAGE2_TEMPORAL2 == 1 and train_wins:
        print("\n" + "=" * 80, flush=True)
        print("=== STAGE 2 TEMPORAL 2: CAUSAL TEMPORAL RESIDUAL ADAPTER V2 -> h_t ===", flush=True)
        print("=" * 80, flush=True)
        try:
            ST2_T = 64
            ST2_FPF = 2
            ST2_TXT = 48
            ST2_AFS = 2 + ST2_TXT
            ST2_MEL_WIN = ST2_T * 8
            BOS_ID, BOA_ID, AUDIO_TOKEN_ID, EOA_ID = 2, 256000, 258881, 258883
            DEV2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            print(f"[TEMPORAL2_ACTIVE] Modulo Temporale Causale v2 attivo: layers={TEMPORAL_LAYERS} dim={TEMPORAL_DIM} heads={TEMPORAL_HEADS} ff_dim={TEMPORAL_FF_DIM} window={ST2_T} (5.12s) ReZero=1", flush=True)
            print(f"[TEMPORAL2] cfg: batches={PHASE2_SMOKE_BATCHES} bs={STAGE2_BATCH_SIZE} "
                  f"accum={STAGE2_GRAD_ACCUM} lr_adapter={TEMPORAL_LR_ADAPTER} lr_module={TEMPORAL_LR_MODULE} "
                  f"warmup={TEMPORAL_WARMUP_STEPS} min_ratio={TEMPORAL_MIN_LR_RATIO} lambda_c={STAGE2_LAMBDA_COUPLING} "
                  f"temp_dim={TEMPORAL_DIM} temp_layers={TEMPORAL_LAYERS} temp_heads={TEMPORAL_HEADS} temp_ff={TEMPORAL_FF_DIM}", flush=True)

            # 1. Carica sub-talker bf16 (congelato)
            _sp = next((p for p in QWEN_SUBTALKER_CANDIDATE_PATHS if os.path.isdir(p)), QWEN_SUBTALKER_CANDIDATE_PATHS[0])
            sub_st2, _ = QS.load_qwen_subtalker(_sp, device="cuda", dtype=torch.bfloat16)
            sub_st2.eval()
            for _p in sub_st2.parameters():
                _p.requires_grad = False
            print(f"[TEMPORAL2] sub-talker caricato bf16 da {_sp} e CONGELATO ({sum(p.numel() for p in sub_st2.parameters()):,} params)", flush=True)

            # 2. Carica Gemma 4 E2B via Unsloth (COMPLETAMENTE CONGELATA)
            print("[TEMPORAL2] Sposto teacher su CPU durante il caricamento di Gemma 4 E2B...", flush=True)
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            from unsloth import FastLanguageModel
            gemma_base, gemma_tok = FastLanguageModel.from_pretrained(
                model_name="unsloth/gemma-4-E2B-it",
                max_seq_length=2048,
                load_in_4bit=True,
                device_map="cuda:0",
            )
            gemma_base.eval()
            for p in gemma_base.parameters():
                p.requires_grad = False
            print(f"[TEMPORAL2] Gemma 4 E2B caricata in 4-bit su {DEV2} e CONGELATA (requires_grad=False)", flush=True)

            # 3. Costruisci TemporalResidualAdapter v2 e applica warm-start da R2/R3
            _HID = int(gemma_base.config.text_config.hidden_size) if hasattr(gemma_base.config, "text_config") else 1536
            ht_model = build_ht_temporal_adapter(
                _hid=_HID, temp_dim=TEMPORAL_DIM, temp_layers=TEMPORAL_LAYERS,
                temp_heads=TEMPORAL_HEADS, temp_ff=TEMPORAL_FF_DIM
            ).to(DEV2, dtype=torch.float32)
            print(f"[TEMPORAL2] hidden_size LLM rilevato: {_HID} | TemporalResidualAdapter v2 istanziato", flush=True)

            _ck_warm = next((p for p in ADAPTER_STAGE2_R2_CANDIDATE_PATHS if os.path.exists(p)), None)
            if _ck_warm:
                _sd_loaded = torch.load(_ck_warm, map_location="cpu")
                if "skip_proj.weight" in _sd_loaded:
                    _missing_c, _unexp_c = ht_model.adapter.load_state_dict(_sd_loaded, strict=False)
                    print(f"[TEMPORAL2] warm-start ht_adapter_stage2r2_final da {_ck_warm} (in ht_model.adapter, missing keys: {_missing_c})", flush=True)
                elif "adapter.skip_proj.weight" in _sd_loaded:
                    _missing_c, _unexp_c = ht_model.load_state_dict(_sd_loaded, strict=False)
                    print(f"[TEMPORAL2] warm-start ht_adapter_stage2r2_final da {_ck_warm} (in ht_model, missing keys: {_missing_c})", flush=True)
                else:
                    _missing_c, _unexp_c = ht_model.adapter.load_state_dict(_sd_loaded, strict=False)
                    print(f"[TEMPORAL2] warm-start ht_adapter_stage2r2_final da {_ck_warm} (fallback adapter load, missing keys: {_missing_c})", flush=True)
            else:
                print("[TEMPORAL2] warm-start adapter: checkpoint non trovato, adapter da zero", flush=True)

            # 4. Parametri Trainable & Ottimizzatore Differenziale
            adapter_params = [p for p in ht_model.adapter.parameters() if p.requires_grad]
            temporal_params = [p for p in ht_model.temporal.parameters() if p.requires_grad]
            base_frozen_params = [p for p in gemma_base.parameters()]
            sub_frozen_params = [p for p in sub_st2.parameters()]

            n_adapt = sum(p.numel() for p in adapter_params)
            n_temp = sum(p.numel() for p in temporal_params)
            n_base_frozen = sum(p.numel() for p in base_frozen_params)
            n_sub_frozen = sum(p.numel() for p in sub_frozen_params)

            print(f"[TEMPORAL2] PARAMETRI: Adapter={n_adapt:,} | Modulo_Temporale={n_temp:,} | Base_Frozen={n_base_frozen:,} | Sub_Frozen={n_sub_frozen:,}", flush=True)
            print(f"[TEMPORAL2] TOTALE TRAINABLE: {n_adapt + n_temp:,}", flush=True)

            optimizer = torch.optim.AdamW(
                [
                    {"params": adapter_params, "lr": TEMPORAL_LR_ADAPTER, "name": "adapter"},
                    {"params": temporal_params, "lr": TEMPORAL_LR_MODULE, "name": "temporal_module"},
                ],
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=STAGE2_WEIGHT_DECAY,
            )

            # LR Scheduler con Warmup Lineare e Cosine Decay
            _T_total = max(1, PHASE2_SMOKE_BATCHES)
            _warmup = TEMPORAL_WARMUP_STEPS
            _min_ratio = TEMPORAL_MIN_LR_RATIO

            def _lr_lambda(step_idx):
                s = step_idx + 1
                if _warmup > 0 and s <= _warmup:
                    return float(s) / float(_warmup)
                if s >= _T_total:
                    return _min_ratio
                decay_steps = max(1, _T_total - _warmup)
                progress = min(max(float(s - _warmup) / float(decay_steps), 0.0), 1.0)
                cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return _min_ratio + (1.0 - _min_ratio) * cos_decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

            # 5. Tokenizer testo Gemma 4
            from transformers import AutoTokenizer
            gemma_txt = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
            E2B_PAD = 0

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

            # 6. Estrazione target h_t UNA TANTUM (teacher su GPU temporaneamente, poi CPU)
            print(f"[TEMPORAL2] estrazione target h_t per {len(train_wins)} finestre (teacher, no_grad)...", flush=True)
            _tdev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            teacher.to(_tdev)
            ht_map = {}
            _t0 = time.time()
            for _w in train_wins:
                _it = ds_train[_w["idx"]]
                _tx = text_by_window[_w["idx"]] if text_by_window is not None else ""
                _h = extract_ht(_it, _w["n_real"], _tx, return_shifted=False)
                _h64 = np.zeros((ST2_T, 2048), dtype=np.float32)
                _h64[: min(_w["n_real"], ST2_T)] = _h[: min(_w["n_real"], ST2_T)]
                ht_map[_w["idx"]] = _h64
            print(f"[TEMPORAL2] target estratti: {len(ht_map)} in {time.time()-_t0:.1f}s; teacher riportato su CPU (zero VRAM)", flush=True)
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            # 7. Training Loop Temporal Stage 2 v2
            gemma_base.eval()
            ht_model.train()
            step_logs = []
            win_cursor = 0
            avail = len(train_wins)
            optimizer.zero_grad()

            print(f"[TEMPORAL2] Inizio training loop: {PHASE2_SMOKE_BATCHES} step...", flush=True)
            for step in range(1, PHASE2_SMOKE_BATCHES + 1):
                t0s = time.time()
                acc_loss = acc_lat = acc_coup = acc_cos = 0.0
                for _mi in range(STAGE2_GRAD_ACCUM):
                    wins = [train_wins[(win_cursor + i) % avail] for i in range(STAGE2_BATCH_SIZE)]
                    win_cursor = (win_cursor + STAGE2_BATCH_SIZE) % avail
                    items = [ds_train[w["idx"]] for w in wins]
                    texts = [text_by_window[w["idx"]] if text_by_window is not None else "" for w in wins]

                    ht_target = torch.stack(
                        [torch.from_numpy(ht_map[w["idx"]]) for w in wins], 0
                    ).to(DEV2, dtype=torch.float32)

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
                    active_mask = torch.from_numpy(np.stack(am, 0)).to(DEV2)

                    input_ids, attn_mask = _build_ids(texts)
                    feats = torch.stack([_win_mel(w["idx"]) for w in wins], 0).to(DEV2)
                    fmask = torch.ones((STAGE2_BATCH_SIZE, ST2_MEL_WIN), dtype=torch.bool, device=DEV2)

                    # Forward Gemma Frozen sotto no_grad (risparmio VRAM e compute)
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
                        agent_hidden = lh[:, _s:_e:ST2_FPF, :].to(torch.float32)

                    # Forward TemporalResidualAdapter v2 (gradient tracking attivo!)
                    h_t_pred = ht_model(agent_hidden)  # (B, 64, 2048) fp32

                    # Latent Loss (cos_sim + 0.5*MSE) su frame attivi
                    ms = active_mask.sum().clamp(min=1e-8)
                    cos_m = F.cosine_similarity(h_t_pred, ht_target, dim=-1)
                    cos_loss = 1.0 - (cos_m * active_mask).sum() / ms
                    mean_cos = (cos_m * active_mask).sum() / ms
                    mse_m = F.mse_loss(h_t_pred, ht_target, reduction="none").mean(dim=-1)
                    L_latent = cos_loss + 0.5 * ((mse_m * active_mask).sum() / ms)

                    # Coupling CE Loss (sub-talker congelato)
                    h_t_sub = h_t_pred.to(torch.bfloat16)
                    sub_logits = sub_st2.forward_sub_talker_finetune(h_t_sub, audio_targets)
                    L_coup = sub_st2.coupling_loss(sub_logits, audio_targets, frame_weights=active_mask)

                    micro_loss = (L_latent + STAGE2_LAMBDA_COUPLING * L_coup) / STAGE2_GRAD_ACCUM
                    micro_loss.backward()

                    acc_loss += micro_loss.item() * STAGE2_GRAD_ACCUM
                    acc_lat += L_latent.item()
                    acc_coup += L_coup.item()
                    acc_cos += mean_cos.item()

                # Gradient Verification Gates
                adapt_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in adapter_params)
                temp_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in temporal_params)
                base_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in base_frozen_params)
                sub_g = any(p.grad is not None and bool((p.grad != 0).any()) for p in sub_frozen_params)

                all_trainable = adapter_params + temporal_params
                gn = torch.nn.utils.clip_grad_norm_(all_trainable, max_norm=STAGE2_GRAD_CLIP)
                cur_lr_adapt = optimizer.param_groups[0]["lr"]
                cur_lr_temp = optimizer.param_groups[1]["lr"]

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                info = {
                    "step": step,
                    "lr_adapter": round(cur_lr_adapt, 8),
                    "lr_temporal": round(cur_lr_temp, 8),
                    "loss": round(acc_loss / STAGE2_GRAD_ACCUM, 4),
                    "latent": round(acc_lat / STAGE2_GRAD_ACCUM, 4),
                    "coupling": round(acc_coup / STAGE2_GRAD_ACCUM, 4),
                    "cos_sim": round(acc_cos / STAGE2_GRAD_ACCUM, 4),
                    "grad_norm": round(float(gn.item()), 4) if hasattr(gn, "item") else float(gn),
                    "adapter_grad_active": adapt_g,
                    "temporal_grad_active": temp_g,
                    "base_grad_zero": not base_g,
                    "sub_grad_zero": not sub_g,
                    "time_s": round(time.time() - t0s, 2),
                }
                step_logs.append(info)

                print(f"[TEMPORAL2] step {step}/{PHASE2_SMOKE_BATCHES} | lr_a {info['lr_adapter']:.2e} lr_t {info['lr_temporal']:.2e} "
                      f"| loss {info['loss']:.4f} (lat {info['latent']:.4f} coup {info['coupling']:.4f}) "
                      f"| cos {info['cos_sim']:.4f} | grad {info['grad_norm']:.4f} "
                      f"| adapt_g {info['adapter_grad_active']} temp_g {info['temporal_grad_active']} "
                      f"base0 {info['base_grad_zero']} sub0 {info['sub_grad_zero']} "
                      f"| {info['time_s']}s", flush=True)

                if step % STAGE2_CKPT_EVERY == 0:
                    os.makedirs("checkpoints", exist_ok=True)
                    _ckpt_ad_name = f"checkpoints/ht_adapter_temporal2_{step}.pt"
                    torch.save(ht_model.state_dict(), _ckpt_ad_name)
                    print(f"[TEMPORAL2] checkpoint periodico: {_ckpt_ad_name}", flush=True)

            # 8. Salvataggio Pesi e Report Finale Training
            os.makedirs("checkpoints", exist_ok=True)
            _is_smoke_run = PHASE2_SMOKE_BATCHES <= 8
            ck_model = "checkpoints/ht_adapter_temporal2_smoke.pt" if _is_smoke_run else "checkpoints/ht_adapter_temporal2_final.pt"
            torch.save(ht_model.state_dict(), ck_model)
            torch.save(ht_model.state_dict(), "checkpoints/ht_adapter_stage2r2_final.pt")

            temporal2_report = {
                "status": "PASS" if (all(s["base_grad_zero"] for s in step_logs) and all(s["sub_grad_zero"] for s in step_logs) and all(s["adapter_grad_active"] for s in step_logs) and all(s["temporal_grad_active"] for s in step_logs) and not any(math.isnan(s["loss"]) for s in step_logs)) else "FAIL",
                "phase": "STAGE2_TEMPORAL2",
                "batches": PHASE2_SMOKE_BATCHES,
                "trainable_params": {
                    "adapter": n_adapt,
                    "temporal_module": n_temp,
                    "total": n_adapt + n_temp,
                },
                "frozen_params": {
                    "base": n_base_frozen,
                    "subtalker": n_sub_frozen,
                },
                "hyperparameters": {
                    "temporal_dim": TEMPORAL_DIM,
                    "temporal_layers": TEMPORAL_LAYERS,
                    "temporal_heads": TEMPORAL_HEADS,
                    "temporal_ff_dim": TEMPORAL_FF_DIM,
                    "lr_adapter": TEMPORAL_LR_ADAPTER,
                    "lr_temporal": TEMPORAL_LR_MODULE,
                    "warmup_steps": TEMPORAL_WARMUP_STEPS,
                    "weight_decay": STAGE2_WEIGHT_DECAY,
                    "grad_clip": STAGE2_GRAD_CLIP,
                    "lambda_coupling": STAGE2_LAMBDA_COUPLING,
                },
                "gates": {
                    "base_frozen_verified": all(s["base_grad_zero"] for s in step_logs),
                    "sub_frozen_verified": all(s["sub_grad_zero"] for s in step_logs),
                    "adapter_grad_active_verified": all(s["adapter_grad_active"] for s in step_logs),
                    "temporal_grad_active_verified": all(s["temporal_grad_active"] for s in step_logs),
                    "loss_finite": not any(math.isnan(s["loss"]) or math.isinf(s["loss"]) for s in step_logs),
                },
                "final": step_logs[-1] if step_logs else {},
                "all_steps": step_logs,
            }

            with open(os.path.join(OUTPUT_DIR, "stage2temporal2_report.json"), "w", encoding="utf-8") as f:
                json.dump(temporal2_report, f, indent=2)
            with open(os.path.join(OUTPUT_DIR, "stage2temporal_report.json"), "w", encoding="utf-8") as f:
                json.dump(temporal2_report, f, indent=2)

            print(f"[TEMPORAL2] Checkpoint salvato in: {ck_model}", flush=True)
            print(f"[TEMPORAL2] Report salvato in: {os.path.join(OUTPUT_DIR, 'stage2temporal2_report.json')}", flush=True)
            print(f">>> STAGE 2 TEMPORAL 2 COMPLETATO CON STATUS: {temporal2_report['status']} <<<", flush=True)
        except Exception:
            print("[TEMPORAL2] ERRORE durante lo Stage 2 Temporal 2 training", flush=True)
            traceback.print_exc()

    # =========================================================================
    # SEZIONE 2: EVAL & SINTESI AUDIO (Val split idx 1000+, mai viste)
    # =========================================================================
    if EVAL_TEMPORAL2 == 1 and train_wins:
        print("\n" + "=" * 80, flush=True)
        print("=== EVAL_TEMPORAL2: VALUTAZIONE ADAPTER TEMPORALE V2 SU VAL SPLIT ===", flush=True)
        print("=" * 80, flush=True)
        try:
            ST2_T = 64
            ST2_FPF = 2
            ST2_TXT = 48
            ST2_AFS = 2 + ST2_TXT
            ST2_MEL_WIN = ST2_T * 8
            BOS_ID, BOA_ID, AUDIO_TOKEN_ID, EOA_ID = 2, 256000, 258881, 258883
            N_VAL = EVAL_N_VAL
            DEV2 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            _ck_candidates = (
                "checkpoints/ht_adapter_temporal2_final.pt",
                "checkpoints/ht_adapter_temporal2_smoke.pt",
                "checkpoints/ht_adapter_temporal_final.pt",
                "checkpoints/ht_adapter_stage2r2_final.pt",
                "/kaggle/working/checkpoints/ht_adapter_stage2r2_final.pt",
                "/kaggle/input/datasets/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
                "/kaggle/input/andrlab/gemma4-ht-adapter-r2/ht_adapter_stage2r2_final.pt",
            )
            _ck = next((p for p in _ck_candidates if os.path.exists(p)), None)
            if _ck is None:
                raise RuntimeError("checkpoint adapter per eval non trovato")
            print(f"[EVAL_TEMPORAL2] checkpoint model: {_ck}", flush=True)

            n_avail = len(ds_train)
            val_idx = list(range(min(1000, n_avail), min(1000 + N_VAL, n_avail)))
            if len(val_idx) < (2 if SMOKE else 10):
                val_idx = list(range(min(N_VAL, n_avail)))
            print(f"[EVAL_TEMPORAL2] finestre val: {len(val_idx)} (idx {val_idx[0]}..{val_idx[-1]}, n_avail={n_avail})", flush=True)

            print(f"[EVAL_TEMPORAL2] estrazione target h_t val ({len(val_idx)} finestre, teacher no_grad)...", flush=True)
            _tdev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            teacher.to(_tdev)
            ht_map_val, act_map_val, nreal_map_val = {}, {}, {}
            _t0 = time.time()
            for _i in val_idx:
                _it = ds_train[_i]
                _nr = int((_it["audio_targets"][:, 0] != KP.PAD_TARGET_ID).sum().item())
                _tx = text_by_window[_i] if text_by_window is not None else ""
                _h = extract_ht(_it, _nr, _tx, return_shifted=False)
                _h64 = np.zeros((ST2_T, 2048), dtype=np.float32)
                _h64[: min(_nr, ST2_T)] = _h[: min(_nr, ST2_T)]
                ht_map_val[_i] = _h64
                nreal_map_val[_i] = _nr
                _a = np.zeros(ST2_T, dtype=np.float32)
                _a[: min(_nr, ST2_T)] = 1.0
                act_map_val[_i] = _a
            print(f"[EVAL_TEMPORAL2] target val estratti in {time.time()-_t0:.1f}s; teacher su CPU", flush=True)
            teacher.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

            _g = globals().get("gemma_base")
            if _g is not None and not isinstance(_g, (str, bytes)):
                print("[EVAL_TEMPORAL2] riuso Gemma E2B dello STAGE2 (in VRAM)", flush=True)
                gemma_base = _g
            else:
                print("[EVAL_TEMPORAL2] load Gemma 4 E2B...", flush=True)
                from unsloth import FastLanguageModel
                gemma_base, _ = FastLanguageModel.from_pretrained(
                    model_name="unsloth/gemma-4-E2B-it",
                    max_seq_length=2048,
                    load_in_4bit=True,
                    device_map="cuda:0",
                )
                gemma_base.eval()
                for p in gemma_base.parameters():
                    p.requires_grad = False

            _HID = int(gemma_base.config.text_config.hidden_size) if hasattr(gemma_base.config, "text_config") else 1536
            ht_model = build_ht_temporal_adapter(
                _hid=_HID, temp_dim=TEMPORAL_DIM, temp_layers=TEMPORAL_LAYERS,
                temp_heads=TEMPORAL_HEADS, temp_ff=TEMPORAL_FF_DIM
            ).to(DEV2, dtype=torch.float32)
            _sd_eval = torch.load(_ck, map_location="cpu")
            if "adapter.skip_proj.weight" in _sd_eval or "skip_proj.weight" in _sd_eval:
                if "adapter.skip_proj.weight" in _sd_eval:
                    ht_model.load_state_dict(_sd_eval, strict=False)
                else:
                    ht_model.adapter.load_state_dict(_sd_eval, strict=False)
            else:
                ht_model.load_state_dict(_sd_eval, strict=False)
            ht_model.eval()

            _v0 = globals().get("sub_st2")
            if _v0 is not None and not isinstance(_v0, (str, bytes)):
                sub_st2 = _v0
            else:
                _sp = next((p for p in QWEN_SUBTALKER_CANDIDATE_PATHS if os.path.isdir(p)), QWEN_SUBTALKER_CANDIDATE_PATHS[0])
                sub_st2, _ = QS.load_qwen_subtalker(_sp, device="cuda", dtype=torch.bfloat16)
            sub_st2.eval()

            from transformers import AutoTokenizer
            gemma_txt = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
            E2B_PAD = 0

            def _build_ids_eval(texts):
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

            def _win_mel_eval(idx):
                s = idx * ST2_MEL_WIN
                m = user_mel[s:s + ST2_MEL_WIN]
                if m.shape[0] < ST2_MEL_WIN:
                    m = np.pad(m, ((0, ST2_MEL_WIN - m.shape[0]), (0, 0)))
                return torch.from_numpy(np.asarray(m, dtype=np.float32))

            decoder = KP.Code2WavDecoder()
            cos_all, mse_all, ce_all = [], [], []
            h_pred_map, h_tgt_map = {}, {}

            for _b in range(0, len(val_idx), EVAL_BATCH):
                _idxs = val_idx[_b:_b + EVAL_BATCH]
                texts = [text_by_window[i] if text_by_window is not None else "" for i in _idxs]
                input_ids, attn_mask = _build_ids_eval(texts)
                feats = torch.stack([_win_mel_eval(i) for i in _idxs], 0).to(DEV2)
                fmask = torch.ones((len(_idxs), ST2_MEL_WIN), dtype=torch.bool, device=DEV2)

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
                    agent_hidden = lh[:, _s:_e:ST2_FPF, :].to(torch.float32)
                    h_pred = ht_model(agent_hidden)

                ht_tgt = torch.stack([torch.from_numpy(ht_map_val[i]) for i in _idxs], 0).to(DEV2)
                am = torch.stack([torch.from_numpy(act_map_val[i]) for i in _idxs], 0).to(DEV2)
                ms = am.sum().clamp(min=1e-8)

                cos_m = F.cosine_similarity(h_pred, ht_tgt, dim=-1)
                cos_all.append(((cos_m * am).sum() / ms).item())
                mse_m = F.mse_loss(h_pred, ht_tgt, reduction="none").mean(dim=-1)
                mse_all.append(((mse_m * am).sum() / ms).item())

                audio_targets = torch.stack(
                    [torch.from_numpy(np.asarray(ds_train[i]["audio_targets"], dtype=np.int64)) for i in _idxs], 0
                ).to(DEV2)
                sub_logits = sub_st2.forward_sub_talker_finetune(h_pred.to(torch.bfloat16), audio_targets)
                ce_all.append(sub_st2.coupling_loss(sub_logits, audio_targets, frame_weights=am).item())

                for _k, _i in enumerate(_idxs):
                    h_pred_map[_i] = h_pred[_k].detach().float().cpu().numpy()
                    h_tgt_map[_i] = ht_tgt[_k].float().cpu().numpy()

            cos_v = float(np.mean(cos_all))
            mse_v = float(np.mean(mse_all))
            ce_v = float(np.mean(ce_all))
            print(f"[EVAL_TEMPORAL2] val: cos_sim={cos_v:.4f} | mse={mse_v:.4f} | coupling_ce={ce_v:.4f}", flush=True)

            _P = np.concatenate([h_pred_map[i] for i in val_idx], 0).astype(np.float32)
            _T = np.concatenate([h_tgt_map[i] for i in val_idx], 0).astype(np.float32)
            def _rank_eff(M):
                _sv = np.linalg.svd(M, compute_uv=False)
                _thr = 1e-3 * float(_sv[0])
                return int((_sv > _thr).sum()), float(_sv[0]), float(_sv[_sv > _thr].sum() / _sv.sum())
            rk_p, s0_p, cum_p = _rank_eff(_P)
            rk_t, s0_t, cum_t = _rank_eff(_T)
            print(f"[EVAL_TEMPORAL2] rank SVD effettivo: h_t_pred={rk_p} (cum {cum_p:.3f}) | h_t_target={rk_t} (cum {cum_t:.3f})", flush=True)

            import scipy.io.wavfile as wavfile
            _outd = OUTPUT_DIR
            _sub_dtype = next(sub_st2.parameters()).dtype

            EVAL_SAMPLING_TEMP = 0.9
            EVAL_SAMPLING_TOP_K = 50
            EVAL_SAMPLING_TOP_P = 1.0
            EVAL_SAMPLING_SEED = 1234

            _cand_idxs = val_idx[:min(len(val_idx), 15)]
            _scored_cands = []
            for _ci in _cand_idxs:
                try:
                    _nr = nreal_map_val.get(_ci, 0)
                    _na = min(_nr, ST2_T)
                    if _na <= 0:
                        continue
                    _gt_c = np.asarray(ds_train[_ci]["audio_targets"], dtype=np.int64)[:_na]
                    _w_c = decoder.decode(torch.from_numpy(_gt_c).long())
                    _w_arr = np.asarray(_w_c, dtype=np.float32)
                    _rms = float(np.sqrt(np.mean(np.square(_w_arr)))) if _w_arr.size > 0 else 0.0
                    _scored_cands.append((_ci, _rms, _nr, _na))
                except Exception:
                    pass

            _energetic = [c for c in _scored_cands if c[1] > 0.01]
            _energetic.sort(key=lambda x: x[1], reverse=True)
            if len(_energetic) >= 3:
                _chosen_cands = _energetic[:3]
            else:
                _scored_cands.sort(key=lambda x: x[1], reverse=True)
                _chosen_cands = _scored_cands[:min(3, len(_scored_cands))]
                if not _chosen_cands and val_idx:
                    _chosen_cands = [(val_idx[0], 0.0, nreal_map_val.get(val_idx[0], 0), min(nreal_map_val.get(val_idx[0], 0), ST2_T))]

            print("[EVAL_TEMPORAL2] finestre selezionate per sintesi:", flush=True)
            for _tag_c, _cand_info in zip(["a", "b", "c"], _chosen_cands):
                print(f"  [{_tag_c}] idx={_cand_info[0]} | n_real={_cand_info[2]} | n_active={_cand_info[3]} | rms_gt={_cand_info[1]:.6f}", flush=True)

            _VARIANTS = [
                ("greedy", {"temperature": 0.0}),
                ("t01",    {"temperature": 0.1}),
                ("t03",    {"temperature": 0.3}),
                ("t09",    {"temperature": 0.9}),
                ("tdecay", {"temperatures_per_level": [round(0.8 * (0.75 ** (g - 1)), 6) for g in range(1, 16)]}),
            ]

            def _sample_variant(sub, h, cb0, var_cfg):
                return sub.sample_acoustic_codes(h, cb0, top_k=EVAL_SAMPLING_TOP_K,
                                                 top_p=EVAL_SAMPLING_TOP_P, seed=EVAL_SAMPLING_SEED, **var_cfg)

            synth_results = []
            _rvq_results = []
            for _tag, _cand_info in zip(["a", "b", "c"], _chosen_cands):
                _i, _rms_cand, _nr, _na = _cand_info
                try:
                    _gt_full = np.asarray(ds_train[_i]["audio_targets"], dtype=np.int64)[:_na]
                    _oracle = _gt_full[:, 1:]
                    cb0 = torch.from_numpy(_gt_full[:, 0])
                    _hp = torch.from_numpy(h_pred_map[_i][:_na]).to(DEV2, dtype=_sub_dtype).unsqueeze(0)
                    _ht = torch.from_numpy(h_tgt_map[_i][:_na]).to(DEV2, dtype=_sub_dtype).unsqueeze(0)
                    _cb0 = cb0.unsqueeze(0).to(DEV2)
                    _gt_full_t = torch.from_numpy(_gt_full).long()

                    wav_gt = decoder.decode(_gt_full_t)
                    wav_gt_arr = np.asarray(wav_gt, dtype=np.float32)
                    rms_gt = float(np.sqrt(np.mean(np.square(wav_gt_arr)))) if wav_gt_arr.size > 0 else 0.0
                    _f_gt_name = f"eval_temporal2_gt_ref_{_tag}.wav"
                    wavfile.write(os.path.join(_outd, _f_gt_name), SAMPLE_RATE,
                                  (wav_gt_arr * 32767).astype(np.int16))

                    _sampled = {}
                    with torch.inference_mode():
                        for _vname, _vcfg in _VARIANTS:
                            _sampled[("oracle", _vname)] = _sample_variant(sub_st2, _ht, _cb0, _vcfg)
                            _sampled[("pred", _vname)] = _sample_variant(sub_st2, _hp, _cb0, _vcfg)

                    _acc = {}
                    _wav_names = {}
                    for _src in ("oracle", "pred"):
                        for _vname, _vcfg in _VARIANTS:
                            _ac = _sampled[(_src, _vname)].squeeze(0).cpu().long()
                            _full = torch.cat([cb0.unsqueeze(-1), _ac], dim=-1)
                            _gen = _full[:, 1:].cpu().numpy()
                            _acc[f"{_src}_{_vname}"] = float((_gen == _oracle).mean()) if _na > 0 else 0.0
                            _wv = decoder.decode(_full)
                            _fname = f"eval_temporal2_{_src}_{_tag}_{_vname}.wav"
                            wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                          (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                            _wav_names[f"{_src}_{_vname}"] = _fname

                    _zeros = torch.zeros_like(_gt_full_t)
                    for _N in (0, 1, 3, 7, 15):
                        _mix = _gt_full_t.clone()
                        if _N < 15:
                            _mix[:, _N + 1:] = _zeros[:, _N + 1:]
                        _wv = decoder.decode(_mix)
                        _fname = f"eval_temporal2_rvq_gtz_{_tag}_N{_N}.wav"
                        wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                      (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                        _rvq_results.append({"tag": _tag, "kind": "gt+zeros", "N": _N, "file": _fname})

                    _gr_t = _sampled[("oracle", "greedy")].squeeze(0).cpu().long()
                    _gr_p = _sampled[("pred", "greedy")].squeeze(0).cpu().long()
                    for _src, _gr in (("oracle", _gr_t), ("pred", _gr_p)):
                        for _N in (0, 1, 3, 7):
                            _mix = _gt_full_t.clone()
                            _mix[:, _N + 1:] = _gr[:, _N:]
                            _wv = decoder.decode(_mix)
                            _fname = f"eval_temporal2_rvq_{_src}_{_tag}_N{_N}.wav"
                            wavfile.write(os.path.join(_outd, _fname), SAMPLE_RATE,
                                          (np.asarray(_wv, dtype=np.float32) * 32767).astype(np.int16))
                            _rvq_results.append({"tag": _tag, "kind": f"gt+{_src}_greedy", "N": _N, "file": _fname})

                    print(f"[EVAL_TEMPORAL2] synth {_tag}: idx={_i} n_active={_na} rms_gt={rms_gt:.4f}", flush=True)
                    for _src in ("oracle", "pred"):
                        _acc_line = " | ".join(f"{_v}={_acc[f'{_src}_{_v}']:.4f}" for _v, _ in _VARIANTS)
                        print(f"  {_src}: {_acc_line}", flush=True)

                    synth_results.append({
                        "tag": _tag,
                        "idx": int(_i),
                        "n_active": int(_na),
                        "rms_gt": round(rms_gt, 6),
                        "acc": {k: round(v, 4) for k, v in _acc.items()},
                        "sampling": {"top_k": EVAL_SAMPLING_TOP_K, "top_p": EVAL_SAMPLING_TOP_P, "seed": EVAL_SAMPLING_SEED},
                        "variants": [v for v, _ in _VARIANTS],
                        "files": {"gt_ref": _f_gt_name, **_wav_names},
                        "status": "OK",
                    })
                except Exception as _e_win:
                    print(f"[EVAL_TEMPORAL2] ERRORE sintesi finestra {_tag} (idx={_i}): {_e_win}", flush=True)
                    synth_results.append({"tag": _tag, "idx": int(_i), "status": "ERROR", "error": str(_e_win)})

            eval_report = {
                "status": "OK",
                "phase": "EVAL_TEMPORAL2",
                "ckpt": _ck,
                "n_val": len(val_idx),
                "val_idx_range": [val_idx[0], val_idx[-1]],
                "cos_sim": round(cos_v, 4),
                "mse": round(mse_v, 4),
                "coupling_ce": round(ce_v, 4),
                "rank_svd_pred": rk_p,
                "rank_svd_target": rk_t,
                "cum_energy_pred": round(cum_p, 4),
                "cum_energy_target": round(cum_t, 4),
                "sampling_protocol": {"temperature": EVAL_SAMPLING_TEMP, "top_k": EVAL_SAMPLING_TOP_K,
                                      "top_p": EVAL_SAMPLING_TOP_P, "seed": EVAL_SAMPLING_SEED},
                "variants": [v for v, _ in _VARIANTS],
                "synth_samples": synth_results,
                "rvq_isolation": _rvq_results,
                "mean_rms_gt": round(float(np.mean([s["rms_gt"] for s in synth_results if "rms_gt" in s])), 6) if any("rms_gt" in s for s in synth_results) else None,
                "gates": {
                    "cos_sim_ge_0.80": bool(cos_v >= 0.80),
                    "ce_val_le_5.95": bool(ce_v <= 5.95),
                    "rank_ge_1000": bool(rk_p >= 1000),
                },
            }

            with open(os.path.join(_outd, "eval_temporal2_report.json"), "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)
            with open(os.path.join(_outd, "eval_temporal_report.json"), "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)
            with open(os.path.join(_outd, "eval_r2_report.json"), "w", encoding="utf-8") as f:
                json.dump(eval_report, f, indent=2)

            print(json.dumps(eval_report, indent=2), flush=True)
            print(">>> EVAL TEMPORAL 2 COMPLETATO <<<", flush=True)
        except Exception:
            print("[EVAL_TEMPORAL2] ERRORE durante la valutazione temporale v2", flush=True)
            traceback.print_exc()
            try:
                _err_rep = {"status": "ERROR", "phase": "EVAL_TEMPORAL2", "error": traceback.format_exc()}
                with open(os.path.join(OUTPUT_DIR, "eval_temporal2_report.json"), "w", encoding="utf-8") as f:
                    json.dump(_err_rep, f, indent=2)
            except Exception:
                pass

    print("=== PIPELINE TEMPORAL 2 COMPLETATA ===", flush=True)
    sys.stdout.flush()

except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    print("PIPELINE TEMPORAL 2 FALLITA (eccezione rilanciata)", flush=True)
    sys.stdout.flush()
    raise
