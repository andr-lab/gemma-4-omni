import time
import torch
from duplex_kernel.models.thinker import load_thinker

device = "cuda:0"
dtype = torch.bfloat16
th = load_thinker(workspace="/workspace", device=device, dtype=dtype)
lm = th.model.model.language_model
layer = lm.layers[0]

print("Testing torch.compile on single Gemma layer...", flush=True)
compiled_layer = torch.compile(layer, mode="default")

x = torch.randn(1, 4, 1536, device=device, dtype=dtype)
pos = torch.arange(4, device=device).unsqueeze(0)
rot = lm.rotary_emb(x, pos, "full_attention")

# Warmup
for _ in range(5):
    _ = layer(x, position_embeddings=rot)
    _ = compiled_layer(x, position_embeddings=rot)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(28):
    _ = layer(x, position_embeddings=rot)
torch.cuda.synchronize()
t_eager = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
for _ in range(28):
    _ = compiled_layer(x, position_embeddings=rot)
torch.cuda.synchronize()
t_compiled = (time.perf_counter() - t0) * 1000.0

print(f"28 layers: Eager={t_eager:.2f} ms vs Compiled={t_compiled:.2f} ms (saved = {t_eager - t_compiled:.2f} ms)")
