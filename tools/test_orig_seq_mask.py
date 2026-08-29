#!/usr/bin/env python3
import sys, types, importlib.machinery
try:
    import torchaudio
except Exception:
    _m = types.ModuleType('torchaudio'); _m.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
    sys.modules['torchaudio'] = _m
    sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
    sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
    sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
sys.modules['sox'] = types.ModuleType('sox')

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
import os, glob, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from qwen_tts.core.models import Qwen3TTSProcessor
import safetensors.torch as st
import kaggle_pipeline as KP
import qwen_subtalker as QS
from tools.train_loop_coupling_v7 import TextAnchoredProjectionV7

snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
_cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
_tc = _cfg.talker_config
if not hasattr(_tc, 'pad_token_id'):
    object.__setattr__(_tc, 'pad_token_id', None)

teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
_sd = st.load_file(snap + '/model.safetensors', device='cpu')
teacher.load_state_dict(_sd, strict=False)
del _sd; torch.cuda.empty_cache()
teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
talker = teacher.talker

tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
proc_qwen = Qwen3TTSProcessor(tokenizer=tok_qwen)

gemma_dirs = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True) + glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True) if len(gemma_dirs) > 0 else None

sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)
sd = torch.load('/workspace/checkpoints/proj_v7_step800.pt', map_location='cpu')
proj.load_state_dict(sd.get('proj', sd), strict=False)
proj.eval()

CFG = teacher.config
TCFG = CFG.talker_config
lang_map = getattr(TCFG, 'codec_language_id', {})
lang_id = lang_map.get('italian', 2070)

# Carica campioni
_, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
tr_raw = ck['tr_raw']; tr_htx = ck['tr_htx']

def build_official_prefill(text_str):
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

    return torch.cat([talker_input_embed, text_part, bos_part], dim=1)

print("=" * 70)
print("TEST CON ATTENTION_MASK=[1, 1] IDENTICO AL TRAINER V7 ORIGINALE")
print("=" * 70)

for idx in range(3):
    c0 = idx * 64
    codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
    n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
    if n_real < 16: continue
    codes = codes[:n_real]
    text_str = tok_gemma.decode(tbw[idx], skip_special_tokens=True).strip() if (idx < len(tbw) and isinstance(tbw[idx], list)) else 'Buongiorno.'
    h_raw_t = torch.from_numpy(np.asarray(tr_raw[idx][:n_real], dtype=np.float32)).to('cuda')
    h_tx_t = torch.from_numpy(np.asarray(tr_htx[idx], dtype=np.float32)).to('cuda')

    with torch.no_grad():
        T = codes.shape[0]
        h_cond = proj(h_raw_t.unsqueeze(0).to(torch.bfloat16), h_tx_t.unsqueeze(0).to(torch.bfloat16))

        # SEQUENTIAL con attention_mask=torch.ones(1, 1)
        icl = build_official_prefill(text_str)
        out_seq_prefill = talker.model(inputs_embeds=icl, attention_mask=torch.ones(1, icl.shape[1], device='cuda'), use_cache=True)
        kv = out_seq_prefill.past_key_values
        past_h = out_seq_prefill.last_hidden_state[:, -1:, :]

        h_list = [past_h]
        for t in range(1, T):
            cp = codes[t-1].to('cuda')
            agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
            in_t = agg + h_cond[:, t:t+1, :]
            out_s = talker.model(inputs_embeds=in_t, attention_mask=torch.ones(1, 1, device='cuda'), use_cache=True, past_key_values=kv)
            kv = out_s.past_key_values
            past_h = out_s.last_hidden_state[:, -1:, :]
            h_list.append(past_h)

        h_seq = torch.cat(h_list, dim=1)
        cb0_lg_seq = talker.codec_head(h_seq)
        loss_cb0_seq = F.cross_entropy(cb0_lg_seq.view(-1, cb0_lg_seq.shape[-1]), codes[:, 0].contiguous().view(-1)).item()
        sub_lg, loss_res_seq = talker.forward_sub_talker_finetune(codes, h_seq[0])
        acc = float((cb0_lg_seq.argmax(dim=-1)[0] == codes[:, 0]).float().mean().item())

    print(f"Sample {idx}: n_real={n_real} | CB0 Loss={loss_cb0_seq:.3f} | Res Loss={loss_res_seq.item():.3f} | Acc={acc*100:.1f}%")
