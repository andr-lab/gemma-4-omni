#!/usr/bin/env python3
from __future__ import annotations
import sys
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")
import numpy as np
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM


def m(a, b):
    d=(a.float()-b.float()).abs()
    af=a.float().flatten(); bf=b.float().flatten()
    c=torch.dot(af,bf)/(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
    return float(d.mean()),float(c),float(d.max())


def rms(x,w,eps):
    z=x.float(); y=z*torch.pow(z.pow(2).mean(-1,keepdim=True)+eps,-0.5)
    return (y*w.float()).to(x.dtype)


def main():
    dev=torch.device("cuda:0"); dt=torch.bfloat16
    th=load_thinker("/workspace",device=dev,dtype=dt); lm=th.model.model.language_model
    nat=NativeGemmaListenerLM(lm,max_seq_len=2048,stream_q_len=4)
    x=torch.tensor(0.05*np.random.default_rng(2026).standard_normal((1,4,1536)),device=dev,dtype=dt)
    nat.step_debug(x,num_layers=1,apply_final_norm=False); torch.cuda.synchronize()
    qn=nat.copy_debug_tensor("q_buf",(4,8,256)); kn=nat.copy_debug_tensor("k_buf",(4,256)); vn=nat.copy_debug_tensor("v_buf",(4,256)); an=nat.copy_debug_tensor("attn_out",(4,8,256))
    a=lm.layers[0].self_attn; l=lm.layers[0]
    xn=rms(x,l.input_layernorm.weight,l.input_layernorm.eps)
    q=F.linear(xn,a.q_proj.weight).view(1,4,8,256); q=rms(q,a.q_norm.weight,a.q_norm.eps)
    k=F.linear(xn,a.k_proj.weight).view(1,4,1,256); k=rms(k,a.k_norm.weight,a.k_norm.eps)
    v=F.linear(xn,a.v_proj.weight).view(1,4,1,256); v=rms(v,torch.ones(256,device=dev,dtype=dt),a.v_norm.eps)
    cos,sin=lm.rotary_emb(x,torch.arange(4,device=dev).view(1,4),"sliding_attention")
    from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb
    q=apply_rotary_pos_emb(q,cos,sin,unsqueeze_dim=2).transpose(1,2)
    k=apply_rotary_pos_emb(k,cos,sin,unsqueeze_dim=2).transpose(1,2)
    v=v.transpose(1,2)
    scores=torch.matmul(q.float(),k.transpose(-1,-2).float())*a.scaling
    scores=scores.masked_fill(~torch.tril(torch.ones(4,4,device=dev,dtype=torch.bool)).view(1,1,4,4), -1e20)
    w=F.softmax(scores,dim=-1).to(dt)
    out=torch.matmul(w,v).transpose(1,2).reshape(1,4,2048)
    # Native layout is [q, heads, D], while q/k/v references are [1, heads, q, D].
    q_ref=q.transpose(1,2).reshape(4,8,256); k_ref=k.transpose(1,2).reshape(4,256); v_ref=v.transpose(1,2).reshape(4,256); attn_ref=torch.matmul(w,v).transpose(1,2).reshape(4,8,256)
    for name,ref,cand in [("q",q_ref,qn),("k",k_ref,kn),("v",v_ref,vn),("attn",attn_ref,an)]:
        print(name,m(ref,cand))
    print("attn_ref_head0_q0", attn_ref[0,0,:8].float().tolist())
    print("attn_nat_head0_q0", an[0,0,:8].float().tolist())
    print("weights_q0", w[0,0,0,:].float().tolist())

if __name__=="__main__": main()
