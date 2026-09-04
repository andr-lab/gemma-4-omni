#!/usr/bin/env python3
import sys
sys.path.insert(0, "/workspace")
import torch
from duplex_kernel.models.talker import load_talker

t = load_talker("/workspace", device="cuda:0", load_code2wav=True)
print("Code2wav config:", t.code2wav.config)
q = t.code2wav.decoder.quantizer
cb = q.rvq_first.vq.layers[0]._codebook
print("cb attributes:", [a for a in dir(cb) if not a.startswith("__")])
print("cb params:", {name: p.shape for name, p in cb.named_parameters()})
print("cb buffers:", {name: b.shape for name, b in cb.named_buffers()})
print("Talker vocab size / lm_head:", t.talker.codec_head[0] if hasattr(t.talker, "codec_head") else getattr(t.talker, "lm_head", None))
if hasattr(t.talker, "talker"):
    print("Inner talker lm_head:", getattr(t.talker.talker, "lm_head", None))
    print("Inner talker codec_head:", getattr(t.talker.talker, "codec_head", None))
