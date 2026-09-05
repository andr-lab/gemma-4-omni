#!/usr/bin/env python3
"""Smoke gate for the opt-in native listener backend."""
from __future__ import annotations
import sys
if "/workspace" not in sys.path: sys.path.insert(0,"/workspace")
import numpy as np
import torch
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.listener import GemmaAudioTowerBackend, GemmaListenerState
from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT

def main():
    dev=torch.device("cuda:0"); dt=torch.bfloat16
    th=load_thinker("/workspace",device=dev,dtype=dt)
    backend=GemmaAudioTowerBackend(th,device=str(dev),use_native_lm=True)
    state=GemmaListenerState()
    backend.prime_steady_state(state)
    assert state.gemma_lm_native is not None
    assert state.gemma_lm_cache is state.gemma_lm_native
    assert state.gemma_lm_native.current_position == AUDIO_START_AFTER_TEXT
    x=torch.tensor(.05*np.random.default_rng(2026).standard_normal((1,4,1536)),device=dev,dtype=dt).contiguous()
    out=backend._forward_incremental_lm(
        embeds=x,
        per_layer=backend._frame_pad_per_layer_inputs(backend.gemma_model.language_model),
        attention=torch.ones((1,AUDIO_START_AFTER_TEXT+4),device=dev,dtype=torch.long),
        positions=torch.arange(AUDIO_START_AFTER_TEXT,AUDIO_START_AFTER_TEXT+4,device=dev),
        state=state,
    )
    torch.cuda.synchronize()
    assert tuple(out.shape)==(1,4,1536)
    assert state.gemma_lm_native.current_position == AUDIO_START_AFTER_TEXT+4
    assert state.gemma_lm_native.verify_output_health()
    state.invalidate_gemma_lm_state()
    assert state.gemma_lm_cache is None
    assert state.gemma_lm_native.current_position == 0
    print({"native_backend":True,"prefix_position":AUDIO_START_AFTER_TEXT,"step_position":AUDIO_START_AFTER_TEXT+4,"reset_position":state.gemma_lm_native.current_position})
if __name__=="__main__": main()
