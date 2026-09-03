import time
import torch
from pathlib import Path
from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph

device = "cuda:0"
dtype = torch.bfloat16
talker = load_talker("/workspace", device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)

cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

# Try compiling talker.talker.model
try:
    print("Testing torch.compile on talker...", flush=True)
    # Check if talker has model attribute
    model_to_compile = talker.talker.model if hasattr(talker.talker, "model") else talker.talker
    compiled_model = torch.compile(model_to_compile, mode="default")
    print("torch.compile succeeded on model definition.", flush=True)
except Exception as e:
    print(f"torch.compile failed: {e}", flush=True)
