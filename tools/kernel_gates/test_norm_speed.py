import time
import torch
from duplex_kernel.models.thinker import load_thinker
device = "cuda:0"
dtype = torch.bfloat16
th = load_thinker(workspace="/workspace", device=device, dtype=dtype)
print("th.model.model attributes:", [k for k in dir(th.model.model) if not k.startswith("_")])
norm = th.model.model.language_model.layers[0].input_layernorm
x = torch.randn(1, 4, 1536, device=device, dtype=dtype)
out1 = norm(x)
ms = x.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
out2 = (x.float() * torch.pow(ms, -0.5) * norm.weight.float()).type_as(x)
print(f"Max diff: {(out1 - out2).abs().max().item():.8f}")

for _ in range(100):
    _ = norm(x)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(112):
    _ = norm(x)
torch.cuda.synchronize()
t_native = (time.perf_counter() - t0) * 1000.0

for _ in range(100):
    ms = x.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
    _ = (x.float() * torch.pow(ms, -0.5) * norm.weight.float()).type_as(x)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(112):
    ms = x.float().pow(2).mean(-1, keepdim=True) + float(norm.eps)
    _ = (x.float() * torch.pow(ms, -0.5) * norm.weight.float()).type_as(x)
torch.cuda.synchronize()
t_custom = (time.perf_counter() - t0) * 1000.0

print(f"112 RMSNorms: Native={t_native:.2f} ms vs Custom={t_custom:.2f} ms (diff = {t_custom - t_native:.2f} ms)")

