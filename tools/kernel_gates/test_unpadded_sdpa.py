import time
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward

device = "cuda:0"
dtype = torch.bfloat16

# 28 layers of Gemma attention
# query: (1, 4, 16, 256) -> transpose (1, 16, 4, 256)
# key/val: (1, 64, 4, 256) -> transpose (1, 4, 64, 256) (GQA)
q = torch.randn(1, 16, 4, 256, device=device, dtype=dtype)
k_64 = torch.randn(1, 4, 64, 256, device=device, dtype=dtype)
v_64 = torch.randn(1, 4, 64, 256, device=device, dtype=dtype)

k_291 = torch.zeros(1, 4, 291, 256, device=device, dtype=dtype)
v_291 = torch.zeros(1, 4, 291, 256, device=device, dtype=dtype)
k_291[:, :, :64, :].copy_(k_64)
v_291[:, :, :64, :].copy_(v_64)

# Warmup
for _ in range(50):
    _ = torch.nn.functional.scaled_dot_product_attention(q, k_64.repeat_interleave(4, dim=1), v_64.repeat_interleave(4, dim=1))
    _ = torch.nn.functional.scaled_dot_product_attention(q, k_291.repeat_interleave(4, dim=1), v_291.repeat_interleave(4, dim=1))
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(28):
    # unpadded SDPA
    _ = torch.nn.functional.scaled_dot_product_attention(q, k_64.repeat_interleave(4, dim=1), v_64.repeat_interleave(4, dim=1))
torch.cuda.synchronize()
t_unpadded = (time.perf_counter() - t0) * 1000.0

t0 = time.perf_counter()
for _ in range(28):
    # padded SDPA + copies
    k_291[:, :, :64, :].copy_(k_64)
    v_291[:, :, :64, :].copy_(v_64)
    _ = torch.nn.functional.scaled_dot_product_attention(q, k_291.repeat_interleave(4, dim=1), v_291.repeat_interleave(4, dim=1))
torch.cuda.synchronize()
t_padded = (time.perf_counter() - t0) * 1000.0

print(f"28 SDPA layers: unpadded (64) = {t_unpadded:.2f} ms vs padded (291 + copies) = {t_padded:.2f} ms (saved = {t_padded - t_unpadded:.2f} ms)")
