import sys, os, types, importlib.util, importlib.machinery
sys.path = [p for p in sys.path if '.pylibs' not in p]
mock_ta = types.ModuleType('torchaudio')
mock_ta.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
sys.modules['torchaudio'] = mock_ta
sys.modules['torchaudio._extension'] = types.ModuleType('torchaudio._extension')
sys.modules['torchaudio.compliance'] = types.ModuleType('torchaudio.compliance')
sys.modules['torchaudio.compliance.kaldi'] = types.ModuleType('torchaudio.compliance.kaldi')
try:
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    print('IMPORT qwen_tts OK:', Qwen3TTSTokenizerV2Model)
except Exception as e:
    print('IMPORT qwen_tts FAIL:', str(e)[:200])
    p = '/workspace/.pylibs/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py'
    spec = importlib.util.spec_from_file_location('modeling_qwen3_tts_tokenizer_v2', p)
    m = importlib.util.module_from_spec(spec)
    sys.modules['modeling_qwen3_tts_tokenizer_v2'] = m
    try:
        spec.loader.exec_module(m)
        print('CARICAMENTO DIRETTO OK:', getattr(m, 'Qwen3TTSTokenizerV2Model', None))
    except Exception as e2:
        print('CARICAMENTO DIRETTO FAIL:', str(e2)[:300])
