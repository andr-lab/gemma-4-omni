import sys, os, glob, inspect, types, importlib.machinery
try:
    import torchaudio
except Exception:
    _m = types.ModuleType('torchaudio'); _m.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
    sys.modules['torchaudio'] = _m
    sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
    sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
    sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
sys.modules['sox'] = types.ModuleType('sox')

sys.path.insert(0, '/workspace/qwen3tts_src')
import qwen_tts.core.models.modeling_qwen3_tts as mq

print("=== Qwen3TTSTalkerForConditionalGeneration.generate ===")
src = inspect.getsource(mq.Qwen3TTSTalkerForConditionalGeneration.generate)
print(src[:4000])
