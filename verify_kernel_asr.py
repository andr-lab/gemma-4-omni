import sys, glob
sys.path.insert(0, "/workspace")
from faster_whisper import WhisperModel

wd = sorted(glob.glob("/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/*"))
m = WhisperModel(wd[0] if wd else "small", device="cuda", compute_type="float16", download_root="/workspace/.hf_cache")
segs, _ = m.transcribe("/workspace/samples_kernel/kernel_smoke.wav", language="it", temperature=0.0)
print("ASR:", " ".join(s.text for s in segs).strip())
