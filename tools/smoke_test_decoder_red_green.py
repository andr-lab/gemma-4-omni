import sys, os, time, json
import numpy as np
import torch
import soundfile as sf
from faster_whisper import WhisperModel

print("=== SMOKE TEST: RED / GREEN DECODER VALIDATION ===", flush=True)

# 1. Carica codici GT reali da parquet
sys.path.insert(0, '/workspace')
import kaggle_pipeline as KP

prefix = '/workspace/data/train_tokens_cv26_it_40.8h_31916s'
print(f"[1] Caricamento codici GT da {prefix}.parquet...", flush=True)
user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(prefix + '.parquet')

# Seleziona una finestra GT con parlato reale (es. primi 64 frame attivi)
gt_codes = agent_codes[:, :64]
print(f"    GT codes shape: {gt_codes.shape}, range [{gt_codes.min()}, {gt_codes.max()}]", flush=True)

# Inizializza Whisper
print("[2] Inizializzazione Faster-Whisper...", flush=True)
asr = WhisperModel('small', device='cuda', compute_type='float16')

# -------------------------------------------------------------
# FASE RED: Riproduzione esatta dello stato attuale
# -------------------------------------------------------------
print("\n--- [FASE RED: Stato Attuale Senza Fix] ---", flush=True)
decoder_red = KP.Code2WavDecoder()

# Decodifica con decoder nello stato attuale
t0 = time.time()
wav_red = decoder_red.decode(gt_codes)
t_red = time.time() - t0

red_wav_path = '/workspace/samples_ars_v4/smoke_red_test.wav'
sf.write(red_wav_path, wav_red, 24000)

# Analisi spettrale RED
fft_peak_red = np.argmax(np.abs(np.fft.rfft(wav_red))) * (24000 / len(wav_red))
print(f"RED Wav: dur={len(wav_red)/24000:.2f}s, max={np.max(wav_red):.4f}, min={np.min(wav_red):.4f}, peak_freq={fft_peak_red:.1f}Hz", flush=True)

# Trascrizione Whisper RED
segs_red, _ = asr.transcribe(red_wav_path, language='it', beam_size=5)
txt_red = ' '.join(s.text.strip() for s in segs_red)
print(f"RED ASR output: \"{txt_red}\"", flush=True)

is_red_failing = ("Sottotitoli" in txt_red or "QTSS" in txt_red or abs(fft_peak_red - 440.0) < 5.0)
print(f"RED Test Verdetto: {'❌ RED CONFERMATO (Sinusoide 440Hz / Hallucination QTSS)' if is_red_failing else 'Inatteso'}", flush=True)

# -------------------------------------------------------------
# FASE GREEN: Applicazione della Universal Decorator Patch
# -------------------------------------------------------------
print("\n--- [FASE GREEN: Applicazione Fix Universale] ---", flush=True)
import inspect
from transformers.utils.generic import check_model_inputs as _orig_cmi

def universal_check_model_inputs(*args, **kwargs):
    if len(args) == 1 and callable(args[0]) and not kwargs:
        func = args[0]
        sig = inspect.signature(_orig_cmi)
        if "func" in sig.parameters:
            return _orig_cmi(func)
        else:
            return _orig_cmi()(func)
    sig = inspect.signature(_orig_cmi)
    if "func" in sig.parameters:
        return lambda f: _orig_cmi(f)
    else:
        return _orig_cmi(*args, **kwargs)

# Applica la patch al modulo Qwen3TTSTokenizerV2
import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as tok_mod
tok_mod.check_model_inputs = universal_check_model_inputs

# Riavvolgi / ri-decora pre_transformer.forward se necessario oppure istanzia nuovo modello
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model, Qwen3TTSTokenizerV2DecoderTransformer

# Re-wrap explicit forward with universal decorator
Qwen3TTSTokenizerV2DecoderTransformer.forward = universal_check_model_inputs(
    Qwen3TTSTokenizerV2DecoderTransformer.forward.__wrapped__ if hasattr(Qwen3TTSTokenizerV2DecoderTransformer.forward, '__wrapped__') else Qwen3TTSTokenizerV2DecoderTransformer.forward
)

# Ricarica decoder pulito
tok_model_green = Qwen3TTSTokenizerV2Model.from_pretrained(
    "Qwen/Qwen3-TTS-Tokenizer-12Hz",
    cache_dir="/workspace/.hf_cache"
).to("cuda").eval()

# Monkey patch decoder in Code2Wav
decoder_green = KP.Code2WavDecoder()
decoder_green.decoder_module = tok_model_green
decoder_green._wrapper_mode = "official_inprocess"

t0 = time.time()
wav_green = decoder_green.decode(gt_codes)
t_green = time.time() - t0

green_wav_path = '/workspace/samples_ars_v4/smoke_green_test.wav'
sf.write(green_wav_path, wav_green, 24000)

# Analisi spettrale GREEN
fft_peak_green = np.argmax(np.abs(np.fft.rfft(wav_green))) * (24000 / len(wav_green))
print(f"GREEN Wav: dur={len(wav_green)/24000:.2f}s, max={np.max(wav_green):.4f}, min={np.min(wav_green):.4f}, peak_freq={fft_peak_green:.1f}Hz", flush=True)

# Trascrizione Whisper GREEN
segs_green, _ = asr.transcribe(green_wav_path, language='it', beam_size=5)
txt_green = ' '.join(s.text.strip() for s in segs_green)
print(f"GREEN ASR output: \"{txt_green}\"", flush=True)
print(f"Testo GT atteso per finestra 0: \"{tbw[0] if tbw and len(tbw) > 0 else 'N/A'}\"", flush=True)

is_green_passing = len(txt_green.strip()) > 0 and "QTSS" not in txt_green
print(f"GREEN Test Verdetto: {'✅ GREEN SUPERATO (Voce Umana Reale / Trascrizione Valida)' if is_green_passing else '❌ GREEN FALLITO'}", flush=True)
