#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# TRAINER V6 — STRADA B (Qwen3-Omni Thinker-Talker con Ancora Testuale + h_proj Espressivo)
# Architettura:
#   h_raw (64, 1536) + h_tx (1536) --[TextAnchoredProjection]--> h_cond (64, 2048)
#   Talker Qwen3-TTS (congelato):
#     Frame t: inputs = codec_agg_{t-1} + h_cond[t]
#     Loss = CE(cb0 da codec_head) + lambda_r * CE(residui da sub_talker) + lambda_d * Distill(past_hidden, h_sub_gt)
# ============================================================================
import os, sys, json, glob, time, math, argparse, traceback
import numpy as np

# Mock torchaudio and sox for venv_phase1 compatibility
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
sys.path.append('/workspace/.extra_pkgs')

import torch
import torch.nn as nn
import torch.nn.functional as F
import qwen_tts.core.models as _M
import safetensors.torch as st
from transformers import AutoTokenizer
from qwen_tts.core.models import Qwen3TTSProcessor
import kaggle_pipeline as KP
import qwen_subtalker as QS

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument('--smoke', type=int, default=int(os.environ.get('V6_SMOKE', '1')))
ap.add_argument('--steps', type=int, default=int(os.environ.get('V6_STEPS', '2000')))
ap.add_argument('--batch', type=int, default=int(os.environ.get('V6_BATCH', '4')))
ap.add_argument('--lr', type=float, default=float(os.environ.get('V6_LR', '1e-4')))
ap.add_argument('--lambda-r', type=float, default=float(os.environ.get('V6_LAMBDA_R', '0.5')),
                help='peso loss residui vs loss cb0')
ap.add_argument('--lambda-d', type=float, default=float(os.environ.get('V6_LAMBDA_D', '0.2')),
                help='peso loss distillazione past_hidden vs teacher')
ap.add_argument('--eval-every', type=int, default=int(os.environ.get('V6_EVAL_EVERY', '100')))
ap.add_argument('--out', default=os.environ.get('V6_OUT', '/workspace/checkpoints'))
ap.add_argument('--cache', default=os.environ.get('V6_CACHE', '/workspace/cache_ars_full_v2.pt'))
ap.add_argument('--prefix', default=os.environ.get('V6_PREFIX',
                '/workspace/data/train_tokens_cv26_it_40.8h_31916s'))
ap.add_argument('--warmstart', default=os.environ.get('V6_WARMSTART',
                '/workspace/_archive_ckpts/temporal2/ht_adapter_temporal2_1000.pt'))
args = ap.parse_args()

print(f"[V6] smoke={args.smoke} steps={args.steps} batch={args.batch} lr={args.lr} "
      f"lambda_r={args.lambda_r} lambda_d={args.lambda_d} warmstart={os.path.basename(args.warmstart)}", flush=True)

# ---------------------------------------------------------------------------
# 1. Teacher (congelato) + processor + sub-talker eval + decoder
# ---------------------------------------------------------------------------
print('[V6 1/6] teacher...', flush=True)
snap = sorted(glob.glob('/workspace/.hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*'))[-1]
_cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
_sd = st.load_file(snap + '/model.safetensors', device='cpu')
_missing, _unexpected = teacher.load_state_dict(_sd, strict=False)
print(f'  teacher missing={len(_missing)} unexpected={len(_unexpected)}', flush=True)
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

# ---------------------------------------------------------------------------
# 2. Cache v2 (h_raw, h_tx, h_sub) + parquet (codici GT + testo)
# ---------------------------------------------------------------------------
print('[V6 2/6] dati...', flush=True)
ck = torch.load(args.cache, map_location='cpu', weights_only=False, mmap=True)
user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(args.prefix + '.parquet')
sidecar = args.prefix + '_hyperparams.json'
seg_ends = json.load(open(sidecar)).get('SEGMENT_ENDS', [])
t_agent = agent_codes.shape[1]
val_start = int(0.90 * t_agent)
for e in seg_ends:
    if e >= val_start:
        val_start = e
        break
n_tr = len(ck['tr_raw'])
n_va = len(ck['va_raw'])
print(f'  cache: {n_tr} train / {n_va} val | parquet frames={t_agent} val_start={val_start}', flush=True)

def codes_for(i, split='train'):
    c0 = i * 64
    if split == 'train':
        c = torch.from_numpy(np.asarray(agent_codes[:, c0:c0+64], dtype=np.int64)).long().to('cuda')
    else:
        c = torch.from_numpy(np.asarray(agent_codes[:, val_start+c0:val_start+c0+64], dtype=np.int64)).long().to('cuda')
    return c.transpose(0, 1)  # (T, 16)

def text_for(i, split='train'):
    gi = i if split == 'train' else (val_start // 64) + i
    if tbw is not None and gi < len(tbw) and tbw[gi]:
        return tbw[gi]
    return ''

# ---------------------------------------------------------------------------
# 3. PROIEZIONE: TextAnchoredProjection (Audio Projection + Text Projection)
# ---------------------------------------------------------------------------
print('[V6 3/6] TextAnchoredProjection (warm-start audio + zero-init text)...', flush=True)

class TextAnchoredProjection(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, hidden_dim=3072):
        super().__init__()
        # 1. Ramo Audio (modulatore espressivo e prosodico)
        self.audio_skip = nn.Linear(in_dim, out_dim)
        self.audio_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.audio_norm = nn.LayerNorm(out_dim)
        
        # 2. Ramo Testo (ancora semantica e fonetica dal Thinker)
        self.text_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.text_norm = nn.LayerNorm(out_dim)
        
        # Zero-init sul layer finale del testo per warm-start fluido
        nn.init.zeros_(self.text_mlp[2].weight)
        nn.init.zeros_(self.text_mlp[2].bias)
        
        self.out_norm = nn.LayerNorm(out_dim)

    def forward(self, h_raw, h_tx):
        # h_raw: (B, T, 1536), h_tx: (B, 1536) o (B, 1, 1536)
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)  # (B, 1, 1536)
        
        h_a = self.audio_norm(self.audio_mlp(h_raw) + self.audio_skip(h_raw))  # (B, T, 2048)
        h_t = self.text_norm(self.text_mlp(h_tx))                              # (B, 1, 2048)
        
        return self.out_norm(h_a + h_t)

proj = TextAnchoredProjection().to('cuda', dtype=torch.bfloat16)

# Carica warm-start sul ramo audio
if args.warmstart and os.path.exists(args.warmstart):
    ws = torch.load(args.warmstart, map_location='cpu', weights_only=False)
    own = proj.state_dict()
    mapping = {
        'adapter.skip_proj.weight': 'audio_skip.weight',
        'adapter.skip_proj.bias': 'audio_skip.bias',
        'adapter.mlp.0.weight': 'audio_mlp.0.weight',
        'adapter.mlp.0.bias': 'audio_mlp.0.bias',
        'adapter.mlp.2.weight': 'audio_mlp.2.weight',
        'adapter.mlp.2.bias': 'audio_mlp.2.bias',
        'adapter.norm.weight': 'audio_norm.weight',
        'adapter.norm.bias': 'audio_norm.bias',
    }
    ok = 0
    for k_src, k_dst in mapping.items():
        if k_src in ws:
            own[k_dst] = ws[k_src]
            ok += 1
    proj.load_state_dict(own)
    print(f'  warm-start audio: {ok}/8 chiavi caricate', flush=True)

n_params = sum(p.numel() for p in proj.parameters())
print(f'  TextAnchoredProjection params: {n_params:,}', flush=True)

opt = torch.optim.AdamW(proj.parameters(), lr=args.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

# ---------------------------------------------------------------------------
# 4. Loop forward (Thinker-Talker) con loss congiunta
# ---------------------------------------------------------------------------
def build_prefill_text(text):
    if isinstance(text, list):
        # Se è una lista di token, converti in stringa decodificata o usa fallback
        text = ' '
    assistant = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    text_ids = proc(text=assistant, return_tensors='pt')['input_ids']
    return text_ids[:, :-5].to('cuda')

def loop_forward(codes, text, h_raw_t, h_tx_t, h_teacher_t=None):
    """
    Qwen3-Omni Thinker-Talker loop:
    Per frame t: inputs = codec_agg_{t-1} + h_cond[t]
    Loss: CE(cb0) + lambda_r * CE(residui) + lambda_d * Distill(past_hidden, h_teacher)
    """
    T = codes.shape[0]
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
    icl = torch.cat([icl, cbos_emb + tts_pad_embed], dim=1)  # (1, Tt+2, 2048)

    attn_pre = torch.ones(1, icl.shape[1], dtype=torch.long, device='cuda')
    attn1 = torch.ones(1, 1, dtype=torch.long, device='cuda')
    
    # Calcolo h_cond congiunto (Audio + Text)
    h_raw_b = h_raw_t.unsqueeze(0).to('cuda', dtype=torch.bfloat16)  # (1, T, 1536)
    h_tx_b = h_tx_t.unsqueeze(0).to('cuda', dtype=torch.bfloat16)    # (1, 1536)
    h_cond = proj(h_raw_b, h_tx_b)                                   # (1, T, 2048)

    loss_total = torch.zeros((), device='cuda', dtype=torch.bfloat16)
    loss_cb0_accum = 0.0
    loss_res_accum = 0.0
    loss_dis_accum = 0.0
    n_frames = 0
    
    with torch.enable_grad():
        out = talker.model(inputs_embeds=icl, attention_mask=attn_pre, use_cache=True)
        past_kv = out.past_key_values
        past_hidden = out.last_hidden_state[:, -1:, :]
        
        # Frame 0
        lg0 = talker.codec_head(out.last_hidden_state)[:, -1:, :].reshape(1, 1, -1)
        l_cb0_0 = F.cross_entropy(lg0.reshape(-1, lg0.shape[-1]), codes[0:1, 0].to('cuda'))
        sub_logits, l_sub_0 = talker.forward_sub_talker_finetune(
            codes[:1].to('cuda'), past_hidden[0].to(torch.bfloat16))
        
        l_dis_0 = torch.zeros((), device='cuda', dtype=torch.bfloat16)
        if h_teacher_t is not None and args.lambda_d > 0:
            h_tgt_0 = h_teacher_t[0:1].to('cuda', dtype=torch.bfloat16)
            l_dis_0 = F.mse_loss(past_hidden[0], h_tgt_0) + 0.5 * (1.0 - F.cosine_similarity(past_hidden[0], h_tgt_0).mean())
            
        loss_total = loss_total + l_cb0_0 + args.lambda_r * l_sub_0.to(torch.bfloat16) + args.lambda_d * l_dis_0
        loss_cb0_accum += float(l_cb0_0.detach().item())
        loss_res_accum += float(l_sub_0.detach().item())
        loss_dis_accum += float(l_dis_0.detach().item())
        n_frames += 1

        for t in range(1, T):
            agg = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device='cuda')
            codes_prev = codes[t-1].to('cuda')
            for g in range(16):
                if g == 0:
                    e = talker.get_input_embeddings()(codes_prev[g:g+1].reshape(1, 1))
                else:
                    e = talker.code_predictor.get_input_embeddings()[g-1](codes_prev[g:g+1].reshape(1, 1))
                agg = agg + e
            
            inputs = agg + h_cond[:, t:t+1, :]
            out = talker.model(inputs_embeds=inputs, attention_mask=attn1, use_cache=True,
                               past_key_values=past_kv)
            past_kv = out.past_key_values
            past_hidden = out.last_hidden_state[:, -1:, :]
            
            lg = talker.codec_head(out.last_hidden_state).reshape(1, 1, -1)
            l_cb0_t = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), codes[t:t+1, 0].to('cuda'))
            
            sub_logits, l_sub_t = talker.forward_sub_talker_finetune(
                codes[t:t+1].to('cuda'), past_hidden[0].to(torch.bfloat16))
            
            l_dis_t = torch.zeros((), device='cuda', dtype=torch.bfloat16)
            if h_teacher_t is not None and args.lambda_d > 0:
                h_tgt_t = h_teacher_t[t:t+1].to('cuda', dtype=torch.bfloat16)
                l_dis_t = F.mse_loss(past_hidden[0], h_tgt_t) + 0.5 * (1.0 - F.cosine_similarity(past_hidden[0], h_tgt_t).mean())
            
            loss_total = loss_total + l_cb0_t + args.lambda_r * l_sub_t.to(torch.bfloat16) + args.lambda_d * l_dis_t
            loss_cb0_accum += float(l_cb0_t.detach().item())
            loss_res_accum += float(l_sub_t.detach().item())
            loss_dis_accum += float(l_dis_t.detach().item())
            n_frames += 1

    return loss_total / max(n_frames, 1), {
        'cb0': loss_cb0_accum / max(n_frames, 1),
        'res': loss_res_accum / max(n_frames, 1),
        'dis': loss_dis_accum / max(n_frames, 1)
    }

# ---------------------------------------------------------------------------
# 5. Eval ASR (loop con decoder riparato)
# ---------------------------------------------------------------------------
def eval_asr(n_win=3):
    tdecay = [round(0.8 * (0.75 ** (g - 1)), 6) for g in range(1, 16)]
    from faster_whisper import WhisperModel
    asr = WhisperModel('small', device='cuda', compute_type='float16')
    decoder = KP.Code2WavDecoder()
    import soundfile as sf
    proj.eval()
    with torch.inference_mode():
        for i in range(min(n_win, n_va)):
            h_raw = torch.from_numpy(np.asarray(ck['va_raw'][i], dtype=np.float32)).to('cuda')
            h_tx = torch.from_numpy(np.asarray(ck['va_htx'][i], dtype=np.float32)).to('cuda')
            codes = codes_for(i, 'val')
            text = text_for(i, 'val')
            T = codes.shape[0]
            
            h_cond = proj(h_raw.unsqueeze(0).to(torch.bfloat16), h_tx.unsqueeze(0).to(torch.bfloat16))
            
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
            cb0 = codes[:, 0]
            resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                                cb0[0:1].reshape(1, 1).to('cuda'),
                                                top_k=50, top_p=1.0, temperatures_per_level=tdecay)
            resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0[0].item())
            
            for t in range(1, T):
                agg = torch.zeros(1, 1, 2048, dtype=torch.bfloat16, device='cuda')
                cp = codes[t-1].to('cuda')
                for g in range(16):
                    if g == 0:
                        e = talker.get_input_embeddings()(cp[g:g+1].reshape(1, 1))
                    else:
                        e = talker.code_predictor.get_input_embeddings()[g-1](cp[g:g+1].reshape(1, 1))
                    agg = agg + e
                
                inputs = agg + h_cond[:, t:t+1, :]
                out = talker.model(inputs_embeds=inputs, attention_mask=attn1, use_cache=True,
                                   past_key_values=past_kv)
                past_kv = out.past_key_values
                past_hidden = out.last_hidden_state[:, -1:, :]
                resid_t = sub.sample_acoustic_codes(past_hidden.to(torch.bfloat16),
                                                    cb0[t:t+1].reshape(1, 1).to('cuda'),
                                                    top_k=50, top_p=1.0, temperatures_per_level=tdecay)
                resid_all.append(resid_t[0, 0].cpu().long()); cb0_all.append(cb0[t].item())
                
            R = torch.stack(resid_all, dim=0)
            C = torch.tensor(cb0_all).unsqueeze(-1)
            full = torch.cat([C, R], dim=-1)
            
            wav = decoder.decode(full)
            f = f'/workspace/samples_v6/v6_eval_{i:03d}.wav'
            sf.write(f, wav, 24000)
            segs, _ = asr.transcribe(f, language='it', beam_size=5)
            txt = ' '.join(s.text.strip() for s in segs)
            print(f'  [V6-EVAL {i}] asr: {txt[:90]}', flush=True)
    proj.train()

# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------
print('[V6 4/6] training...', flush=True)
n_train = n_tr if not args.smoke else min(64, n_tr)
steps = args.steps if not args.smoke else 12
os.makedirs(args.out, exist_ok=True)
os.makedirs('/workspace/samples_v6', exist_ok=True)
proj.train()
step = 0
t0 = time.time()
logs = []

while step < steps:
    for i in range(n_train):
        if step >= steps:
            break
        codes = codes_for(i, 'train')
        n_real = int((codes[:, 0] != KP.PAD_TARGET_ID).sum().item())
        if n_real < 16:
            continue
        codes = codes[:n_real]
        h_raw = torch.from_numpy(np.asarray(ck['tr_raw'][i], dtype=np.float32))[:n_real]
        h_tx = torch.from_numpy(np.asarray(ck['tr_htx'][i], dtype=np.float32))
        h_sub = torch.from_numpy(np.asarray(ck['tr_sub'][i], dtype=np.float32))[:n_real]
        text = text_for(i, 'train')
        
        try:
            loss, loss_dict = loop_forward(codes, text, h_raw, h_tx, h_sub)
            loss.backward()
        except Exception as e:
            print(f'  [V6] ERRORE finestra {i}: {e}', flush=True)
            traceback.print_exc()
            opt.zero_grad()
            continue
            
        if (step + 1) % args.batch == 0:
            torch.nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            
        step += 1
        if step % 20 == 0 or step == 1:
            lv = float(loss.detach().float())
            print(f'  [V6] step {step}/{steps} loss={lv:.4f} [cb0={loss_dict["cb0"]:.3f} res={loss_dict["res"]:.3f} dis={loss_dict["dis"]:.3f}] lr={sched.get_last_lr()[0]:.2e} '
                  f'({(time.time()-t0)/max(step,1):.2f}s/step)', flush=True)
            logs.append({'step': step, 'loss': lv, **loss_dict})
            
        if step % args.eval_every == 0:
            print(f'  [V6] EVAL @ step {step}', flush=True)
            try:
                eval_asr(3)
            except Exception as e:
                print(f'  [V6] eval error: {e}', flush=True)
            torch.save({'proj': proj.state_dict(), 'step': step, 'logs': logs},
                       f'{args.out}/proj_v6_step{step}.pt')
            proj.train()

torch.save({'proj': proj.state_dict(), 'step': step, 'logs': logs},
           f'{args.out}/proj_v6_final.pt')
print(f'[V6] checkpoint: {args.out}/proj_v6_final.pt ({step} step)', flush=True)
print('[V6] FATTO.', flush=True)
