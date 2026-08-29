import sys, os, glob, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
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
    print("=== TEST PER-LEVEL RESIDUAL ACCURACY & CE ===", flush=True)
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

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    CFG = teacher.config
    TCFG = CFG.talker_config
    lang_map = getattr(TCFG, 'codec_language_id', {})
    lang_id = lang_map.get('italian', 2070)

    proj = TextAnchoredProjectionV8().to(device='cuda', dtype=torch.bfloat16)
    ckpt_path = '/workspace/checkpoints/proj_v8_final.pt'
    sd = torch.load(ckpt_path, map_location='cpu')
    proj.load_state_dict(sd.get('proj', sd), strict=False)
    proj.eval()

    _, _, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
    ck = torch.load('/workspace/cache_ars_full_v2.pt', map_location='cpu', weights_only=False, mmap=True)
    va_raw = ck['va_raw']
    va_htx = ck['va_htx']

    N_all = agent_codes.shape[1]
    val_start = int(0.90 * N_all)
    seg_ends = json.load(open('/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json')).get('SEGMENT_ENDS', [])
    for e in seg_ends:
        if e >= val_start: val_start = e; break

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

    sample_idx = 1
    c0 = val_start + sample_idx * 64
    codes_np = np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)
    codes = torch.from_numpy(codes_np).long().to('cuda').transpose(0, 1)
    codes = torch.clamp(codes, 0, 2047)
    n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
    if n_real < 16: n_real = 64
    codes = codes[:n_real]
    T = n_real

    text_str = "Si tratta del secondo singolo promozionale nella storia della band."
    h_raw = torch.from_numpy(np.asarray(va_raw[sample_idx][:n_real], dtype=np.float32)).to('cuda')
    h_tx = torch.from_numpy(np.asarray(va_htx[sample_idx], dtype=np.float32)).to('cuda')

    with torch.no_grad():
        h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
        icl = build_official_prefill(text_str)
        L_prefill = icl.shape[1]

        codes_prev = codes[:-1]
        agg_audio = sum(talker.get_input_embeddings()(codes_prev[:, 0:1]) if g==0 else talker.code_predictor.get_input_embeddings()[g-1](codes_prev[:, g:g+1]) for g in range(16)).transpose(0, 1)

        # Omni forward
        audio_embeds_omni = agg_audio + h_cond[:, 1:T, :]
        inp_omni = torch.cat([icl, audio_embeds_omni], dim=1)
        out_omni = talker.model(inputs_embeds=inp_omni, use_cache=False)
        h_omni = out_omni.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

        # Native forward
        inp_nat = torch.cat([icl, agg_audio], dim=1)
        out_nat = talker.model(inputs_embeds=inp_nat, use_cache=False)
        h_nat = out_nat.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

        # Eval subtalker across 15 levels
        lg_omni, _ = talker.forward_sub_talker_finetune(codes, h_omni[0])
        lg_nat, _ = talker.forward_sub_talker_finetune(codes, h_nat[0])

        print(f"\n--- PER-CODEBOOK CROSS-ENTROPY & ACCURACY (CB1..CB15) ---")
        print(f"{'Level':<6} | {'Omni CE':<8} | {'Omni Acc':<8} | {'Nat CE':<8} | {'Nat Acc':<8}")
        print("-" * 50)
        for g in range(15):
            ce_o = F.cross_entropy(lg_omni[:, g, :], codes[:, g+1]).item()
            acc_o = float((lg_omni[:, g, :].argmax(dim=-1) == codes[:, g+1]).float().mean().item()) * 100
            ce_n = F.cross_entropy(lg_nat[:, g, :], codes[:, g+1]).item()
            acc_n = float((lg_nat[:, g, :].argmax(dim=-1) == codes[:, g+1]).float().mean().item()) * 100
            print(f"CB{g+1:<4} | {ce_o:<8.3f} | {acc_o:<7.1f}% | {ce_n:<8.3f} | {acc_n:<7.1f}%")

if __name__ == '__main__':
    main()
