import time
import torch
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph

device = "cuda:0"
dtype = torch.bfloat16
talker = load_talker("/workspace", device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)

cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

last_h = torch.randn(1, 1, 2048, device=device, dtype=dtype)
c0 = 100

c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))

# Warmup
for _ in range(5):
    _ = subtalker_graph.generate(last_h, c0, c0_embed=c0_embed, do_sample=True)
    _ = subtalker_graph.generate(last_h, c0, c0_embed=c0_embed, do_sample=False)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(20):
    _ = subtalker_graph.generate(last_h, c0, c0_embed=c0_embed, do_sample=True)
torch.cuda.synchronize()
t_sample = (time.perf_counter() - t0) * 1000.0 / 20

t0 = time.perf_counter()
for _ in range(20):
    _ = subtalker_graph.generate(last_h, c0, c0_embed=c0_embed, do_sample=False)
torch.cuda.synchronize()
t_greedy = (time.perf_counter() - t0) * 1000.0 / 20

print(f"Subtalker generate: do_sample=True: {t_sample:.2f} ms vs do_sample=False: {t_greedy:.2f} ms")
