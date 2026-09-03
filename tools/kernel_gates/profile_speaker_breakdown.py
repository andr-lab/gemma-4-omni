import time
import torch
from pathlib import Path
from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.subtalker_graph import SubtalkerGraph

device = "cuda:0"
dtype = torch.bfloat16
talker = load_talker("/workspace", device=device, load_code2wav=True)
talker.talker.to(dtype=dtype)

cp = talker.talker.code_predictor
subtalker_graph = SubtalkerGraph(cp, talker_embed=talker.talker.get_input_embeddings()).capture()

talker.talker.model = torch.compile(talker.talker.model, mode="default")

speaker = FuturePrefixQwen(
    talker,
    lookahead_frames=0,
    max_prefix_frames=64,
    device=device,
    dtype=dtype,
    max_audio_frames=64,
)

dummy_cond = torch.randn((1, 2048), device=device, dtype=dtype)
dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

speaker.begin("Profile session.")

# Warmup
for t in range(5):
    speaker.push_v9_frame(t, dummy_cond)
    step_out = speaker.step_frame()
    c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
    c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
    subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
    speaker.commit_codes(dummy_codes)
torch.cuda.synchronize()

times_qwen = []
times_subtalker = []
times_total = []

for t in range(5, 25):
    t0 = time.perf_counter()
    speaker.push_v9_frame(t, dummy_cond)
    t1 = time.perf_counter()
    step_out = speaker.step_frame()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    c0 = int(step_out.logits[:, :1, :].argmax(-1).item())
    c0_embed = talker.talker.get_input_embeddings()(torch.tensor([[c0]], device=device))
    subtalker_graph.generate(step_out.last_hidden, c0, c0_embed=c0_embed)
    speaker.commit_codes(dummy_codes)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    times_qwen.append((t2 - t1) * 1000.0)
    times_subtalker.append((t3 - t2) * 1000.0)
    times_total.append((t3 - t0) * 1000.0)

print(f"Qwen step_frame:     mean={sum(times_qwen)/len(times_qwen):.2f} ms")
print(f"Subtalker + commit:  mean={sum(times_subtalker)/len(times_subtalker):.2f} ms")
print(f"Speaker Total Step:  mean={sum(times_total)/len(times_total):.2f} ms")
