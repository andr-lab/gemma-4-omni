import sys, os, traceback
import torch
import soundfile as sf
import transformers

print(f"Python: {sys.version}")
print(f"Transformers: {transformers.__version__}")

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/qwen3tts_src')

from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

tok_model = Qwen3TTSTokenizerV2Model.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    cache_dir="/workspace/.hf_cache"
).to("cuda").eval()

dummy_codes = torch.randint(0, 2048, (1, 64, 16), device="cuda").long()
print("Calling tok_model.decode(dummy_codes)...")
out = tok_model.decode(dummy_codes)
wav = out.audio_values[0].cpu().numpy()
print(f"✅ Success! Generated audio shape: {wav.shape}, duration: {len(wav)/24000:.2f}s")
