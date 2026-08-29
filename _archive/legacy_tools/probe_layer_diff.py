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
T = 4
torch.manual_seed(0)
inputs = torch.randn(1, L_prefill + T - 1, 2048, device='cuda', dtype=torch.bfloat16)

# Test layer 0 in sequential vs parallel
layer0 = talker.model.layers[0]

# Sequential
icl = inputs[:, :L_prefill]
out0_prefill = talker.model(inputs_embeds=icl, use_cache=True, output_hidden_states=True)
kv = out0_prefill.past_key_values

in_1 = inputs[:, L_prefill : L_prefill + 1]
out0_step1 = talker.model(inputs_embeds=in_1, use_cache=True, past_key_values=kv, output_hidden_states=True)

h_seq_layer0_step1 = out0_step1.hidden_states[1] # Layer 0 output for step 1

# Parallel
out0_par = talker.model(inputs_embeds=inputs, use_cache=False, output_hidden_states=True)
h_par_layer0_step1 = out0_par.hidden_states[1][:, L_prefill : L_prefill + 1, :]

print(f"Layer 0 output Step 1 max diff: {(h_seq_layer0_step1 - h_par_layer0_step1).abs().max().item():.6f}")
print(f"Layer 0 output Step 1 cosine  : {torch.nn.functional.cosine_similarity(h_seq_layer0_step1.float(), h_par_layer0_step1.float(), dim=-1).mean().item():.6f}")

# Check final layer (layer 28)
h_seq_final_step1 = out0_step1.hidden_states[-1]
h_par_final_step1 = out0_par.hidden_states[-1][:, L_prefill : L_prefill + 1, :]
print(f"Final Layer output Step 1 max diff: {(h_seq_final_step1 - h_par_final_step1).abs().max().item():.6f}")
print(f"Final Layer output Step 1 cosine  : {torch.nn.functional.cosine_similarity(h_seq_final_step1.float(), h_par_final_step1.float(), dim=-1).mean().item():.6f}")
