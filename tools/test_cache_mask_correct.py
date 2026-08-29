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
import torch, glob
import qwen_tts.core.models as _M
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
teacher = teacher.cuda().eval()
talker = teacher.talker

L_prefill = 30
T = 16
torch.manual_seed(42)
inputs = torch.randn(1, L_prefill + T - 1, 2048, device='cuda', dtype=torch.bfloat16)

# 1. Sequential with attention_mask=None (full past attention)
icl = inputs[:, :L_prefill]
out_prefill = talker.model(inputs_embeds=icl, use_cache=True)
kv = out_prefill.past_key_values
h_seq_list = [out_prefill.last_hidden_state[:, -1:, :]]

for t in range(1, T):
    in_t = inputs[:, L_prefill + t - 1 : L_prefill + t]
    out_t = talker.model(inputs_embeds=in_t, use_cache=True, past_key_values=kv)
    kv = out_t.past_key_values
    h_seq_list.append(out_t.last_hidden_state)

h_seq = torch.cat(h_seq_list, dim=1)

# 2. Parallel with use_cache=False
out_par = talker.model(inputs_embeds=inputs, use_cache=False)
h_par = out_par.last_hidden_state[:, L_prefill - 1 : L_prefill - 1 + T, :]

diff = (h_seq - h_par).abs().max().item()
cos = torch.nn.functional.cosine_similarity(h_seq.float(), h_par.float(), dim=-1).mean().item()

print(f"=== TEST ATTENZIONE PIENA: SEQUENTIAL (attention_mask=None) vs PARALLEL ===")
print(f"Sequence length T={T}: max diff = {diff:.6f} | Cosine Similarity = {cos:.6f}")
