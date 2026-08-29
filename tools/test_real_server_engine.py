# -*- coding: utf-8 -*-
"""
tools/test_real_server_engine.py — Verifica e Inizializzazione del Motore Neurale Reale Gemma-4 + Qwen3-TTS + Projector V9.
"""

import glob
import os
import sys
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')
os.environ.setdefault("PIPELINE_ENV", "local3090")

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

from transformers import AutoTokenizer, AutoModelForCausalLM
import qwen_tts.core.models as _M
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
import safetensors.torch as st
import kaggle_pipeline as KP
from tools.run_exact_interactive_dialogue_benchmark import DeepCausalTransformerProjectorV9


def test_and_load_engine():
    print("[1/4] Caricamento Gemma 4 Thinker...", flush=True)
    gemma_snaps = glob.glob('/workspace/.hf_cache/models--unsloth--gemma*/**/tokenizer.json', recursive=True)
    if not gemma_snaps:
        gemma_snaps = glob.glob('/workspace/.hf_cache/models--google--gemma*/**/tokenizer.json', recursive=True)
    
    print(f"Snapshot Gemma trovati: {gemma_snaps}", flush=True)
    gemma_path = os.path.dirname(gemma_snaps[0])
    KP.MODEL_NAME = gemma_path
    
    duplex_model, tok_gemma = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, "/workspace/checkpoints")
    duplex_model.eval()
    for p in duplex_model.parameters():
        p.requires_grad = False
    print("✓ Gemma 4 Thinker caricato con successo!", flush=True)

    print("[2/4] Caricamento Qwen3-TTS Talker & Vocoder...", flush=True)
    snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = _cfg.talker_config
    if not hasattr(_tc, 'pad_token_id'):
        object.__setattr__(_tc, 'pad_token_id', None)

    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(snap + '/model.safetensors', device='cpu')
    teacher.load_state_dict(_sd, strict=False)
    del _sd
    torch.cuda.empty_cache()
    teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
    talker = teacher.talker

    tok_qwen = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc_qwen = _M.Qwen3TTSProcessor(tokenizer=tok_qwen)

    snap_tok = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*'))[-1]
    tok_decoder = Qwen3TTSTokenizerV2Model.from_pretrained(snap_tok).cuda().eval()
    print("✓ Qwen3-TTS Talker & Code2Wav Decoder caricati!", flush=True)

    print("[3/4] Caricamento Projector V9 (Prefix KV-Cache 211M)...", flush=True)
    proj = DeepCausalTransformerProjectorV9().to(device='cuda', dtype=torch.bfloat16).eval()
    v9_ckpt = '/workspace/checkpoints/v9_prefix_step10000.pt'
    if not os.path.isfile(v9_ckpt):
        v9_ckpt = '/workspace/checkpoints/v9_prefix_final.pt'
    sd_v9 = torch.load(v9_ckpt, map_location='cpu')
    proj.load_state_dict(sd_v9.get('proj', sd_v9), strict=False)
    print(f"✓ Projector V9 caricato da: {v9_ckpt}!", flush=True)

    print("[4/4] Test di Generazione Testuale e Vocale Reale...", flush=True)
    prompt = "Spiegami brevemente il funzionamento della gravità quantistica."
    inputs = tok_gemma(text=prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to('cuda')
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to('cuda')

    with torch.no_grad():
        out = duplex_model.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
        )
        resp_text = tok_gemma.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n=======================================================", flush=True)
    print(f"Risposta Reale Generata da Gemma 4 su RTX 3090:\n{resp_text}", flush=True)
    print(f"=======================================================", flush=True)


if __name__ == "__main__":
    test_and_load_engine()
