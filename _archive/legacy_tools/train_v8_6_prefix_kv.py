#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training V8.6: Prefix KV-Cache Projector (ADR-022)
Trains the Text-Anchored Projector to construct Prefix KV-Cache representations for Qwen3-TTS Talker.
All Qwen3-TTS components (Talker 28L, CodePredictor 12L, Vocoder 24kHz) are 100% FROZEN.
Audio tokens use 100% canonical codebook embeddings during autoregression.
"""

import sys, os, glob, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

import types, importlib.machinery
try:
    import torchaudio
except Exception:
    _m = types.ModuleType('torchaudio'); _m.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
    sys.modules['torchaudio'] = _m
    sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
    sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
    sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
sys.modules['sox'] = types.ModuleType('sox')
sys.modules['soxr'] = types.ModuleType('soxr')
_l = types.ModuleType('librosa'); _l.__spec__ = importlib.machinery.ModuleSpec('librosa', None)
_l.filters = types.ModuleType('librosa.filters')
_l.filters.mel = lambda *args, **kwargs: None
sys.modules['librosa'] = _l
sys.modules['librosa.filters'] = _l.filters

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
sys.path.append('/workspace/.extra_pkgs')
import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = getattr(config, "rope_theta", 10000.0)
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
    _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    _dim = int(_head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
    return inv_freq, 1.0
_ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
_ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init

from transformers import AutoTokenizer
import qwen_tts.core.models as _M
import safetensors.torch as st
import kaggle_pipeline as KP

class TextAnchoredProjectionV8(nn.Module):
    def __init__(self, raw_dim=1536, text_dim=1536, hidden_dim=2048, out_dim=2048, initial_gamma=0.25):
        super().__init__()
        self.audio_mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.audio_skip = nn.Linear(raw_dim, out_dim, bias=False)
        self.audio_norm = nn.LayerNorm(out_dim)

        self.text_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.text_norm = nn.LayerNorm(out_dim)
        self.out_norm = nn.LayerNorm(out_dim)
        self.gamma = nn.Parameter(torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        h_a = self.audio_norm(self.audio_mlp(h_raw) + self.audio_skip(h_raw))
        h_t = self.text_norm(self.text_mlp(h_tx))
        return (self.gamma.to(h_a.dtype)) * self.out_norm(h_a + h_t)

def main():
    TOTAL_STEPS = int(os.environ.get("V86_STEPS", "10000"))
    BATCH_SIZE = int(os.environ.get("V86_BATCH", "4"))
    GRAD_ACCUM = int(os.environ.get("V86_ACCUM", "2"))
    LR = float(os.environ.get("V86_LR", "1.5e-4"))
    INIT_CKPT = os.environ.get("V86_INIT_CKPT", "/workspace/checkpoints/proj_v8_step25000.pt")

    os.makedirs("/workspace/checkpoints", exist_ok=True)
    os.makedirs("/workspace/logs", exist_ok=True)
    log_file = open("/workspace/logs/v8_6_prefix_kv_training.log", "w", encoding="utf-8")

    def log_print(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log_print("==================================================================")
    log_print(f"  TRAINING V8.6: PREFIX KV-CACHE PROJECTOR (ADR-022)")
    log_print(f"  Total Steps: {TOTAL_STEPS} | Batch Size: {BATCH_SIZE} | Accum: {GRAD_ACCUM} | LR: {LR}")
    log_print(f"  Initial Checkpoint: {INIT_CKPT}")
    log_print("==================================================================")

    # 1. Caricamento Teacher Qwen3-TTS (100% Frozen)
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = _cfg.talker_config
    if not hasattr(_tc, 'pad_token_id'): object.__setattr__(_tc, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    teacher.load_state_dict(_sd, strict=False)
    del _sd; torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker
    for p in teacher.parameters(): p.requires_grad = False

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    # 2. Inizializzazione Proiettore
    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    if os.path.exists(INIT_CKPT):
        sd = torch.load(INIT_CKPT, map_location='cpu')
        proj.load_state_dict(sd.get('proj', sd), strict=False)
        log_print(f"  [CKPT CARICATO] Proiettore inizializzato da {INIT_CKPT} (Gamma={proj.gamma.item():.4f})")
    else:
        log_print(f"  [AVVISO] Checkpoint {INIT_CKPT} non trovato, addestramento da zero.")

    proj.train()
    optimizer = torch.optim.AdamW(proj.parameters(), lr=LR, weight_decay=1e-2, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1.5e-5)

    # 3. Caricamento Dataset e Cache
    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    tr_raw = ck['tr_raw']
    tr_htx = ck['tr_htx']
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']

    N_train = len(tr_raw)
    N_val = len(va_raw)
    log_print(f"  [DATASET] Finestre Train: {N_train} | Finestre Val: {N_val}")

    def build_official_prefill_single(text_str):
        raw_ids = proc_qwen(text=text_str, return_tensors='pt')['input_ids'].to('cuda')
        tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor([[CFG.tts_bos_token_id, CFG.tts_eos_token_id, CFG.tts_pad_token_id]], device='cuda', dtype=raw_ids.dtype)
            )
        ).chunk(3, dim=1)

        codec_prefill = [[TCFG.codec_think_id, TCFG.codec_think_bos_id, lang_id, TCFG.codec_think_eos_id]]
        c_emb_0 = talker.get_input_embeddings()(torch.tensor(codec_prefill, device='cuda', dtype=raw_ids.dtype))
        c_emb_1 = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id, TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))
        codec_input_emebdding = torch.cat([c_emb_0, c_emb_1], dim=1)

        role_ids = proc_qwen(text="<|im_start|>assistant\n", return_tensors='pt')['input_ids'].to('cuda')
        role_embed = talker.text_projection(talker.get_text_embeddings()(role_ids))
        _talker_input_embed = torch.cat((tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1), tts_bos_embed), dim=1) + codec_input_emebdding[:, :-1]
        talker_input_embed = torch.cat((role_embed, _talker_input_embed), dim=1)

        text_proj_emb = talker.text_projection(talker.get_text_embeddings()(raw_ids))
        c_pad_full = talker.get_input_embeddings()(torch.tensor([[TCFG.codec_pad_id] * (raw_ids.shape[1] + 1)], device='cuda', dtype=raw_ids.dtype))
        text_part = torch.cat([text_proj_emb, tts_eos_embed], dim=1) + c_pad_full
        bos_part = tts_pad_embed + talker.get_input_embeddings()(torch.tensor([[TCFG.codec_bos_id]], device='cuda', dtype=raw_ids.dtype))

        # Output text part + BOS part
        return torch.cat([talker_input_embed, text_part], dim=1), bos_part

    # 4. Training Loop V8.6
    log_print("\n--- INIZIO TRAINING V8.6 PREFIX KV-CACHE ---")
    start_time = time.time()
    step = 0
    accum_step = 0
    optimizer.zero_grad()

    while step < TOTAL_STEPS:
        batch_indices = np.random.randint(0, N_train, size=BATCH_SIZE)
        batch_loss = 0.0
        batch_cb0_loss = 0.0

        for b_idx in batch_indices:
            c0 = b_idx * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1) # [64, 16]
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            if n_real < 16: n_real = 64
            
            raw_len = len(tr_raw[b_idx])
            T = min(raw_len, n_real, 64)
            if T < 16: continue

            tbw_entry = tbw[b_idx] if b_idx < len(tbw) else None
            if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
                text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
            else:
                text_str = str(tbw_entry).strip() if tbw_entry is not None else "Risposta dell'assistente."

            h_raw = torch.from_numpy(np.asarray(tr_raw[b_idx][:T], dtype=np.float32)).to('cuda')
            h_tx = torch.from_numpy(np.asarray(tr_htx[b_idx], dtype=np.float32)).to('cuda')

            # Gemma projection
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16)) # [1, T, 2048]
            text_icl, bos_part = build_official_prefill_single(text_str)

            # PREFIX KV SEQUENCE: [ Text_ICL || Projector(h_gemma) || BOS_Part ]
            prefix_embeds = torch.cat([text_icl, h_cond, bos_part], dim=1) # [1, L_pfx, 2048]
            L_pfx = prefix_embeds.shape[1]

            # CANONICAL AUDIO EMBEDDINGS (Teacher-forced target codes)
            cur_codes = codes[:T, :] # [T, 16]
            agg_codes = talker.get_input_embeddings()(cur_codes[:, 0:1].transpose(0, 1)) # [1, T, 2048]
            for g in range(1, 16):
                agg_codes = agg_codes + talker.code_predictor.get_input_embeddings()[g-1](cur_codes[:, g:g+1].transpose(0, 1))

            # Full input passed to Talker: [ Prefix || Canonical_Audio ]
            full_inputs = torch.cat([prefix_embeds, agg_codes], dim=1) # [1, L_pfx + T, 2048]

            # Forward through Frozen Talker
            talker_out = talker.model(inputs_embeds=full_inputs)
            hidden_states = talker_out.last_hidden_state # [1, L_pfx + T, 2048]

            # Audio hidden states corresponding to prediction of target frames 0..T-1
            audio_hidden = hidden_states[:, L_pfx - 1 : L_pfx + T - 1, :] # [1, T, 2048]

            # Loss CB0 (Semantic Codebook 0)
            cb0_logits = talker.codec_head(audio_hidden)[0] # [T, 4096]
            cb0_targets = cur_codes[:, 0] # [T]
            loss_cb0 = F.cross_entropy(cb0_logits, cb0_targets, ignore_index=KP.PAD_TARGET_ID)

            loss_total = (loss_cb0) / (BATCH_SIZE * GRAD_ACCUM)
            loss_total.backward()

            batch_loss += loss_total.item() * (BATCH_SIZE * GRAD_ACCUM)
            batch_cb0_loss += loss_cb0.item()

        accum_step += 1
        if accum_step % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % 50 == 0:
                cur_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                s_per_step = elapsed / step
                log_print(f"  [STEP {step:5d}/{TOTAL_STEPS}] Loss CB0: {batch_cb0_loss/BATCH_SIZE:.4f} | Gamma: {proj.gamma.item():.4f} | LR: {cur_lr:.2e} | Speed: {s_per_step:.2f}s/step")

            # Checkpoint & Validation
            if step % 1000 == 0 or step == TOTAL_STEPS:
                val_losses = []
                proj.eval()
                val_start = N_train * 64
                with torch.no_grad():
                    for v_i in range(min(15, N_val)):
                        c0 = val_start + v_i * 64
                        v_codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
                        v_codes = torch.from_numpy(v_codes_np).long().to('cuda').transpose(0, 1)
                        n_real = int((v_codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
                        raw_len = len(va_raw[v_i])
                        T = min(raw_len, n_real, 64)
                        if T < 16: continue

                        gi = N_train + v_i
                        tbw_entry = tbw[gi] if gi < len(tbw) else None
                        if isinstance(tbw_entry, (list, np.ndarray)) and tok_gemma is not None:
                            text_str = tok_gemma.decode(tbw_entry, skip_special_tokens=True).strip()
                        else:
                            text_str = str(tbw_entry).strip() if tbw_entry is not None else "Validazione."

                        h_raw = torch.from_numpy(np.asarray(va_raw[v_i][:T], dtype=np.float32)).to('cuda')
                        h_tx = torch.from_numpy(np.asarray(va_htx[v_i], dtype=np.float32)).to('cuda')

                        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
                        text_icl, bos_part = build_official_prefill_single(text_str)
                        prefix_embeds = torch.cat([text_icl, h_cond, bos_part], dim=1)
                        L_pfx = prefix_embeds.shape[1]

                        cur_codes = v_codes[:T, :]
                        agg_codes = talker.get_input_embeddings()(cur_codes[:, 0:1].transpose(0, 1))
                        for g in range(1, 16):
                            agg_codes = agg_codes + talker.code_predictor.get_input_embeddings()[g-1](cur_codes[:, g:g+1].transpose(0, 1))

                        full_inputs = torch.cat([prefix_embeds, agg_codes], dim=1)
                        talker_out = talker.model(inputs_embeds=full_inputs)
                        audio_hidden = talker_out.last_hidden_state[:, L_pfx - 1 : L_pfx + T - 1, :]
                        v_logits = talker.codec_head(audio_hidden)[0]
                        v_loss = F.cross_entropy(v_logits, cur_codes[:, 0], ignore_index=KP.PAD_TARGET_ID)
                        val_losses.append(v_loss.item())

                avg_val_loss = float(np.mean(val_losses))
                log_print(f"\n  >>> [VALIDAZIONE STEP {step}] Val Loss CB0: {avg_val_loss:.4f} <<< \n")

                ckpt_file = f"/workspace/checkpoints/proj_v8_6_step{step}.pt"
                torch.save({'proj': proj.state_dict(), 'step': step, 'val_loss': avg_val_loss, 'gamma': proj.gamma.item()}, ckpt_file)
                torch.save({'proj': proj.state_dict(), 'step': step, 'val_loss': avg_val_loss, 'gamma': proj.gamma.item()}, "/workspace/checkpoints/proj_v8_6_latest.pt")
                log_print(f"  [SALVATO CHECKPOINT] {ckpt_file}\n")
                proj.train()

    log_print(f"\n[TRAINING V8.6 COMPLETATO] Pesi salvati in /workspace/checkpoints/proj_v8_6_latest.pt")
    log_file.close()

if __name__ == '__main__':
    main()
