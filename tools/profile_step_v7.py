#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Profiler di precisione millisecondo per il loop di training V7.
Identifica esattamente dove vengono spesi i millisecondi in ogni batch.
"""
import os, sys, glob, time
import numpy as np

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

import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
sys.path.append('/workspace/.extra_pkgs')

import kaggle_pipeline as KP
import qwen_subtalker as QS
import transformers.modeling_rope_utils as _ropeu
def _default_rope_init(config, device=None, seq_len=None, **kwargs):
    base = config.rope_theta
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
tok_gemma = AutoTokenizer.from_pretrained(os.path.dirname(gemma_dirs[0]), use_fast=True)

sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)
_, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
tr_raw = ck['tr_raw']
tr_htx = ck['tr_htx']

from tools.train_loop_coupling_v7 import TextAnchoredProjectionV7
proj = TextAnchoredProjectionV7().to(device='cuda', dtype=torch.bfloat16)

def profile():
    print("=== PROFILING MILLISECONDO V7 STEP ===")
    times = {'tok_decode': [], 'qwen_proc': [], 'prefill_emb': [], 'prefill_fwd': [], 'loop_64': [], 'backward': []}
    
    # Warmup
    torch.cuda.synchronize()
    
    for step in range(5):
        t_batch_start = time.time()
        for i in range(4):
            idx = step * 4 + i
            c0 = idx * 64
            codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
            codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
            n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
            codes = codes[:n_real]
            
            t0 = time.time()
            t_item = tbw[idx]
            text_str = tok_gemma.decode(t_item, skip_special_tokens=True).strip() if (t_item and isinstance(t_item, list)) else 'Buongiorno.'
            torch.cuda.synchronize()
            t_tok = time.time() - t0
            times['tok_decode'].append(t_tok)
            
            t0 = time.time()
            assistant = f"<|im_start|>assistant\n{text_str}<|im_end|>\n<|im_start|>assistant\n"
            tids = proc_qwen(text=assistant, return_tensors='pt')['input_ids'][:, :-5].to('cuda')
            torch.cuda.synchronize()
            t_proc = time.time() - t0
            times['qwen_proc'].append(t_proc)
            
            t0 = time.time()
            text_embed = talker.text_projection(talker.get_text_embeddings()(tids))
            tts_eos_embed = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[teacher.config.tts_eos_token_id]], device='cuda', dtype=tids.dtype)))
            text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
            codec_pad_embed = talker.get_input_embeddings()(torch.tensor([[teacher.config.talker_config.codec_pad_id]], device='cuda', dtype=tids.dtype))
            tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[teacher.config.tts_pad_token_id]], device='cuda', dtype=tids.dtype)))
            icl = text_embed + codec_pad_embed.expand(text_embed.shape[0], text_embed.shape[1], -1)
            cbos_emb = talker.get_input_embeddings()(torch.tensor([[teacher.config.talker_config.codec_bos_id]], device='cuda', dtype=tids.dtype))
            icl = torch.cat([icl, cbos_emb + tts_pad_embed], dim=1)
            torch.cuda.synchronize()
            times['prefill_emb'].append(time.time() - t0)
            
            t0 = time.time()
            attn_pre = torch.ones(1, icl.shape[1], dtype=torch.long, device='cuda')
            out = talker.model(inputs_embeds=icl, attention_mask=attn_pre, use_cache=True)
            past_kv = out.past_key_values
            past_h = out.last_hidden_state[:, -1:, :]
            torch.cuda.synchronize()
            times['prefill_fwd'].append(time.time() - t0)
            
            t0 = time.time()
            h_raw_b = torch.from_numpy(np.asarray(tr_raw[idx][:n_real], dtype=np.float32)).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
            h_tx_b = torch.from_numpy(np.asarray(tr_htx[idx], dtype=np.float32)).unsqueeze(0).to('cuda', dtype=torch.bfloat16)
            h_cond = proj(h_raw_b, h_tx_b)
            
            attn1 = torch.ones(1, 1, dtype=torch.long, device='cuda')
            loss_total = torch.zeros((), device='cuda', dtype=torch.bfloat16)
            for t in range(1, n_real):
                cp = codes[t-1].to('cuda')
                agg = sum(talker.get_input_embeddings()(cp[g:g+1].reshape(1,1)) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1,1)) for g in range(16))
                inputs = agg + h_cond[:, t:t+1, :]
                out = talker.model(inputs_embeds=inputs, attention_mask=attn1, use_cache=True, past_key_values=past_kv)
                past_kv = out.past_key_values
                past_h = out.last_hidden_state[:, -1:, :]
                lg = talker.codec_head(past_h)
                l_cb0 = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), codes[t:t+1, 0])
                _, l_res = talker.forward_sub_talker_finetune(codes[t:t+1], past_h[0])
                loss_total = loss_total + l_cb0 + 0.5 * l_res
            torch.cuda.synchronize()
            times['loop_64'].append(time.time() - t0)
            
            t0 = time.time()
            (loss_total / (4 * n_real)).backward()
            torch.cuda.synchronize()
            times['backward'].append(time.time() - t0)
            
        t_batch_total = time.time() - t_batch_start
        print(f"Step {step}: Total Batch Time = {t_batch_total:.2f}s")
        print(f"  • tok_decode   : {sum(times['tok_decode'][-4:])*1000:.1f}ms")
        print(f"  • qwen_proc    : {sum(times['qwen_proc'][-4:])*1000:.1f}ms")
        print(f"  • prefill_emb  : {sum(times['prefill_emb'][-4:])*1000:.1f}ms")
        print(f"  • prefill_fwd  : {sum(times['prefill_fwd'][-4:])*1000:.1f}ms")
        print(f"  • loop_64 (fwd): {sum(times['loop_64'][-4:])*1000:.1f}ms")
        print(f"  • backward     : {sum(times['backward'][-4:])*1000:.1f}ms")

if __name__ == '__main__':
    profile()
