#!/usr/bin/env python3
from __future__ import annotations
import sys
if "/workspace" not in sys.path: sys.path.insert(0,"/workspace")
import numpy as np
import torch
import torch.nn.functional as F
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM


def metric(a,b):
 d=(a.float()-b.float()).abs(); af=a.float().flatten(); bf=b.float().flatten()
 c=torch.dot(af,bf)/(torch.linalg.vector_norm(af)*torch.linalg.vector_norm(bf))
 return float(d.mean()),float(c),float(d.max())

def rms(x,w,eps):
 z=x.float(); y=z*torch.pow(z.pow(2).mean(-1,keepdim=True)+float(eps),-0.5)
 if w is not None: y=y*w.float()
 return y.to(x.dtype)

def reference(lm,x,target):
 dt=x.dtype; dev=x.device; hidden=x.clone(); saved=None; own_k={}; own_v={}
 pos=torch.arange(4,device=dev).view(1,4)
 cos_sin={t:lm.rotary_emb(x,pos,t) for t in lm.unique_layer_types}
 causal=torch.tril(torch.ones(4,4,device=dev,dtype=torch.bool)).view(1,1,4,4)
 pad_ids=torch.zeros((1,4),device=dev,dtype=torch.long)
 ple_all=lm.get_per_layer_inputs(pad_ids,x)
 # Exact frame PLE projection, using the same raw operations as the progressive gate.
 proj=F.linear(x,lm.per_layer_model_projection.weight)*lm.per_layer_model_projection_scale
 proj=proj.reshape(1,4,lm.config.num_hidden_layers,lm.hidden_size_per_layer_input)
 proj=rms(proj,lm.per_layer_projection_norm.weight,lm.per_layer_projection_norm.eps)
 proj=(proj+ple_all)*lm.per_layer_input_scale
 for i,layer in enumerate(lm.layers[:target+1]):
  attn=layer.self_attn; mlp=layer.mlp; D=attn.head_dim; typ=lm.config.layer_types[i]
  record={"input":hidden.clone()}
  norm=rms(hidden,layer.input_layernorm.weight,layer.input_layernorm.eps); record["norm_buf_input"]=norm.clone()
  shape=(1,4,-1,D); q=rms(F.linear(norm,attn.q_proj.weight).view(shape),attn.q_norm.weight,attn.q_norm.eps)
  cos,sin=cos_sin[typ]; q1,q2=q[...,:D//2],q[...,D//2:]
  q=torch.cat([q1*cos[:,:,:D//2].unsqueeze(2)-q2*sin[:,:,:D//2].unsqueeze(2),q2*cos[:,:,:D//2].unsqueeze(2)+q1*sin[:,:,:D//2].unsqueeze(2)],-1).to(dt).transpose(1,2)
  record["q_buf"]=q.transpose(1,2).reshape(4,8,D).clone()
  if not attn.is_kv_shared_layer:
   k=rms(F.linear(norm,attn.k_proj.weight).view(shape),attn.k_norm.weight,attn.k_norm.eps)
   k=torch.cat([k[...,:D//2]*cos[:,:,:D//2].unsqueeze(2)-k[...,D//2:]*sin[:,:,:D//2].unsqueeze(2),k[...,D//2:]*cos[:,:,:D//2].unsqueeze(2)+k[...,:D//2]*sin[:,:,:D//2].unsqueeze(2)],-1).to(dt).transpose(1,2)
   v=rms(F.linear(norm,attn.v_proj.weight).view(shape),None,attn.v_norm.eps).transpose(1,2)
   own_k[i]=k; own_v[i]=v
  else:
   src=attn.kv_shared_layer_index; k=own_k[src]; v=own_v[src]
  record["k_buf"]=k.transpose(1,2).reshape(4,1,D).clone(); record["v_buf"]=v.transpose(1,2).reshape(4,D).clone()
  scores=torch.matmul(q.float(),k.transpose(-1,-2).float())*attn.scaling
  scores=scores.masked_fill(~causal,-1e20); weights=F.softmax(scores,dim=-1).to(dt)
  aout=torch.matmul(weights,v).transpose(1,2).reshape(1,4,8,D)
  record["attn_out"]=aout.clone()
  o=F.linear(aout.reshape(1,4,-1),attn.o_proj.weight)
  o_norm=rms(o,layer.post_attention_layernorm.weight,layer.post_attention_layernorm.eps)
  hidden=hidden+o_norm
  ff=rms(hidden,layer.pre_feedforward_layernorm.weight,layer.pre_feedforward_layernorm.eps); record["norm_buf_ff"]=ff.clone()
  gate=F.linear(ff,mlp.gate_proj.weight); up=F.linear(ff,mlp.up_proj.weight); act=mlp.act_fn(gate)*up; down=F.linear(act,mlp.down_proj.weight)
  record["gate_buf"]=gate.clone(); record["up_buf"]=up.clone(); record["mlp_act_buf"]=act.clone(); record["mlp_down"]=down.clone()
  down_norm=rms(down,layer.post_feedforward_layernorm.weight,layer.post_feedforward_layernorm.eps); hidden=hidden+down_norm
  ple_gate=F.linear(hidden,layer.per_layer_input_gate.weight); ple_act=layer.act_fn(ple_gate)*proj[:,:,i,:]; ple_proj=F.linear(ple_act,layer.per_layer_projection.weight); ple_norm=rms(ple_proj,layer.post_per_layer_input_norm.weight,layer.post_per_layer_input_norm.eps)
  record["ple_gate_buf"]=ple_gate.clone(); record["ple_act_buf"]=ple_act.clone(); record["ple_proj_out"]=ple_norm.clone()
  hidden=(hidden+ple_norm)*layer.layer_scalar; record["hidden_a"]=hidden.clone()
  if i==target: saved=record
 return saved

def main():
 dev=torch.device("cuda:0"); dt=torch.bfloat16; th=load_thinker("/workspace",device=dev,dtype=dt); lm=th.model.model.language_model
 x=torch.tensor(.05*np.random.default_rng(2026).standard_normal((1,4,1536)),device=dev,dtype=dt)
 for target in (0,2,3,4,13,14,15,24,34):
  nat=NativeGemmaListenerLM(lm,max_seq_len=2048,stream_q_len=4); nat.step_debug(x,num_layers=target+1,apply_final_norm=False); torch.cuda.synchronize(); ref=reference(lm,x,target)
  comps=[("norm_buf",ref["norm_buf_ff"],nat.copy_debug_tensor("norm_buf",(4,1536))), ("q_buf",ref["q_buf"],nat.copy_debug_tensor("q_buf",(4,8,ref["q_buf"].shape[-1]))), ("attn_out",ref["attn_out"],nat.copy_debug_tensor("attn_out",(4,8,ref["attn_out"].shape[-1]))), ("gate",ref["gate_buf"],nat.copy_debug_tensor("gate_buf",(4,ref["gate_buf"].shape[-1]))), ("up",ref["up_buf"],nat.copy_debug_tensor("up_buf",(4,ref["up_buf"].shape[-1]))), ("act",ref["mlp_act_buf"],nat.copy_debug_tensor("mlp_act_buf",(4,ref["mlp_act_buf"].shape[-1]))), ("ple_gate",ref["ple_gate_buf"],nat.copy_debug_tensor("ple_gate_buf",(4,256))), ("ple_act",ref["ple_act_buf"],nat.copy_debug_tensor("ple_act_buf",(4,256))), ("ple_proj",ref["ple_proj_out"],nat.copy_debug_tensor("ple_proj_out",(4,1536))), ("hidden",ref["hidden_a"],nat.copy_debug_tensor("hidden_a",(4,1536)))]
  print("TARGET",target)
  for n,a,b in comps:
   values=metric(a,b)
   print(f" {n:10s} mae={values[0]:.4e} cos={values[1]:.6f} max={values[2]:.4f}")

if __name__=="__main__": main()
