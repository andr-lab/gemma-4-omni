#!/usr/bin/env python3
from __future__ import annotations
import sys
if "/workspace" not in sys.path: sys.path.insert(0,"/workspace")
import numpy as np
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT

def main():
    dev=torch.device("cuda:0"); dt=torch.bfloat16
    th=load_thinker("/workspace",device=dev,dtype=dt); lm=th.model.model.language_model
    backend=GemmaAudioTowerBackend(th,device=str(dev)); native=NativeGemmaListenerLM(lm,max_seq_len=2048,stream_q_len=4)
    ids,mask=backend._build_multimodal_inputs((),1024,device=dev)
    stable=ids[:,:AUDIO_START_AFTER_TEXT]; attn=mask[:,:AUDIO_START_AFTER_TEXT].to(device=dev,dtype=torch.int64).contiguous()
    embeds=lm.embed_tokens(stable).contiguous(); ple=lm.get_per_layer_inputs(stable,embeds).contiguous()
    native.prime_async(embeds,ple,attn); torch.cuda.synchronize()
    x=torch.tensor(.05*np.random.default_rng(2026).standard_normal((1,4,1536)),device=dev,dtype=dt).contiguous()
    for _ in range(10): native.step_async(x)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
    for start, end in zip(starts, ends):
        start.record()
        native.step_async(x)
        end.record()
    ends[-1].synchronize()
    samples = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)])
    print({
        "mean_ms": float(samples.mean()),
        "p95_ms": float(np.percentile(samples, 95)),
        "frames": len(samples),
        "final_position": native.current_position,
    })
if __name__=="__main__": main()
