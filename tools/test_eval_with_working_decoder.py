import sys, os, time, json
import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel
import types, importlib.machinery

# Mock torchaudio and sox for venv_phase1 compatibility
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
import kaggle_pipeline as KP
import qwen_subtalker as QS
import qwen_tts.core.models as _M
import safetensors.torch as st
from transformers import AutoTokenizer
from qwen_tts.core.models import Qwen3TTSProcessor

print("=== VERIFICA GENERAZIONE CON DECODER FUNZIONANTE (STEP 1000) ===", flush=True)

# 1. Carica teacher e subtalker
snap = sorted(__import__('glob').glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
_cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
_sd = st.load_file(snap + '/model.safetensors', device='cpu')
teacher.load_state_dict(_sd, strict=False)
del _sd; torch.cuda.empty_cache()
teacher = teacher.to(device='cuda', dtype=torch.bfloat16).eval()
talker = teacher.talker
for p in talker.parameters(): p.requires_grad = False
tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
proc = Qwen3TTSProcessor(tokenizer=tok)
CFG = teacher.config
TCFG = CFG.talker_config
CBOS = TCFG.codec_bos_id
sub, _ = QS.load_qwen_subtalker('/workspace/qwen_subtalker', device='cuda', dtype=torch.bfloat16)

# 2. Carica Proiezione step 1000
import torch.nn as nn
class Projection(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072):
        super().__init__()
        self.skip_proj = nn.Linear(in_dim, out_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, out_dim))
        self.norm = nn.LayerNorm(out_dim)
    def forward(self, x):
        return self.norm(self.mlp(x) + self.skip_proj(x))

proj = Projection().to('cuda', dtype=torch.bfloat16).eval()
ckpt_path = '/workspace/checkpoints/proj_v5_step1000.pt'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cuda')
    proj.load_state_dict(ckpt['proj'])
    print(f"✅ Checkpoint caricato: {ckpt_path} (step {ckpt.get('step', 'N/A')})", flush=True)
else:
    print(f"⚠️ Checkpoint {ckpt_path} non trovato, cerco step 800 o simili...")
    ckpts = sorted(__import__('glob').glob('/workspace/checkpoints/proj_v5_step*.pt'))
    if ckpts:
        ckpt_path = ckpts[-1]
        ckpt = torch.load(ckpt_path, map_location='cuda')
        proj.load_state_dict(ckpt['proj'])
        print(f"✅ Checkpoint alternativo caricato: {ckpt_path}", flush=True)

# 3. Carica cache v2 e Parquet
cache_path = '/workspace/cache_ars_full_v2.pt'
ck = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet('/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet')
t_agent = agent_codes.shape[1]
val_start = int(0.90 * t_agent)
sidecar = '/workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json'
seg_ends = json.load(open(sidecar)).get('SEGMENT_ENDS', [])
for e in seg_ends:
    if e >= val_start:
        val_start = e
        break

# Inizializza Decoder e ASR
decoder = KP.Code2WavDecoder()
asr = WhisperModel('small', device='cuda', compute_type='float16')
tdecay = [round(0.8 * (0.75 ** (g - 1)), 6) for g in range(1, 16)]

def build_prefill_text(text):
    assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = proc(text=assistant, return_tensors='pt')['input_ids']
    return text_ids[:, :-5].to('cuda')

# 4. Generazione e Test ASR su 3 finestre di validazione
print("\n--- Test Generazione con Decoder Ufficiale ---", flush=True)
with torch.inference_mode():
    for i in range(3):
        c0 = i * 64
        codes_gt = torch.from_numpy(np.asarray(agent_codes[:, val_start+c0:val_start+c0+64], dtype=np.int64)).long().to('cuda')
        gi = (val_start // 64) + i
        text = tbw[gi] if tbw and gi < len(tbw) and tbw[gi] else ''
        h_raw = torch.from_numpy(np.asarray(ck['va_raw'][i], dtype=np.float32)).to('cuda')
        T = codes_gt.shape[0]
        h_proj = proj(h_raw.unsqueeze(0).to(torch.bfloat16))
        
        # Prefill
        tids = build_prefill_text(text or ' ')
        text_embed = talker.text_projection(talker.get_text_embeddings()(tids))
        tts_eos_embed = talker.text_projection(talker.get_text_embeddings()(
            torch.tensor([[CFG.tts_eos_token_id]], device='cuda', dtype=tids.dtype)))
        text_embed = torch.cat([text_embed, tts_eos_embed], dim=1)
        codec_pad_embed = talker.get_input_embeddings()(
            torch.tensor([[TCFG.codec_pad_id]], device='cuda', dtype=tids.dtype))
        tts_pad_embed = talker.text_projection(talker.get_text_embeddings()(
            torch.tensor([[CFG.tts_pad_token_id]], device='cuda', dtype=tids.dtype)))
        icl = text_embed + codec_pad_embed.expand(text_embed.shape[0], text_embed.shape[1], -1)
        cbos_emb = talker.get_input_embeddings()(
            torch.tensor([[CBOS]], device='cuda', dtype=tids.dtype))
        icl = torch.cat([icl, cbos_emb + tts_pad_embed], dim=1)
        attn_pre = torch.ones(1, icl.shape[1], dtype=torch.long, device='cuda')
        attn1 = torch.ones(1, 1, dtype=torch.long, device='cuda')
        
        out = talker.model(inputs_embeds=icl, attention_mask=attn_pre, use_cache=True)
        past_kv = out.past_key_values
        past_hidden = out.last_hidden_state[:, -1:, :]
        
        resid_all, cb0_all = [], []
        cb0_gt = codes_gt[:, 0]
        resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                            cb0_gt[0:1].reshape(1, 1).to('cuda'),
                                            top_k=50, top_p=1.0, temperatures_per_level=tdecay)
        resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0_gt[0].item())
        
        for t in range(1, T):
            agg = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device='cuda')
            cp = codes_gt[t-1].to('cuda')
            for g in range(16):
                if g == 0:
                    e = talker.get_input_embeddings()(cp[g:g+1].reshape(1, 1))
                else:
                    e = talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1, 1))
                agg = agg + e
            inputs = agg + h_proj[:, t:t+1, :]
            out = talker.model(inputs_embeds=inputs, attention_mask=attn1, use_cache=True,
                               past_key_values=past_kv)
            past_kv = out.past_key_values
            past_hidden = out.last_hidden_state[:, -1:, :]
            resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                                cb0_gt[t:t+1].reshape(1, 1).to('cuda'),
                                                top_k=50, top_p=1.0, temperatures_per_level=tdecay)
            resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0_gt[t].item())
            
        R = torch.stack(resid_all, dim=0)
        C = torch.tensor(cb0_all).unsqueeze(-1)
        full = torch.cat([C, R], dim=-1)
        
        wav = decoder.decode(full)
        wav_file = f'/workspace/samples_ars_v4/test_working_eval_{i:03d}.wav'
        sf.write(wav_file, wav, 24000)
        
        segs, _ = asr.transcribe(wav_file, language='it', beam_size=5)
        txt = ' '.join(s.text.strip() for s in segs)
        fft_peak = np.argmax(np.abs(np.fft.rfft(wav))) * (24000 / len(wav))
        print(f"\n[Finestra {i}]")
        print(f"  Testo GT Atteso : \"{text}\"")
        print(f"  ASR Trascrizione: \"{txt}\"")
        print(f"  Caratteristiche : max={np.max(wav):.4f}, min={np.min(wav):.4f}, peak_freq={fft_peak:.1f}Hz")
