import json
import wave
from pathlib import Path
from faster_whisper import WhisperModel

root = Path('/workspace/artifacts/conversation_quality_20261014/pair_long_20261015_owner_pin_fix')
model_path = '/workspace/.hf_cache/models--Systran--faster-whisper-small/snapshots/536b0662742c02347bc0e980a01041f333bce120'
paths = [root / 'A_input.wav', root / 'B_input.wav']
with wave.open(str(root / 'B_input.wav'), 'rb') as src:
    sr = src.getframerate()
    nframes = src.getnframes()
    prefix_frames = min(nframes, int(146 * 0.08 * sr))
    prefix = src.readframes(prefix_frames)
prefix_path = root / 'B_input_prefix_first_B_SPEAK.wav'
with wave.open(str(prefix_path), 'wb') as dst:
    with wave.open(str(root / 'B_input.wav'), 'rb') as src:
        dst.setparams(src.getparams())
    dst.writeframes(prefix)
paths.append(prefix_path)
paths.extend(sorted(root.glob('response_*.wav')))
print(f'ASR model cache: {model_path}; CPU int8; prefix_frames={prefix_frames}; sample_rate={sr}', flush=True)
model = WhisperModel(model_path, device='cpu', compute_type='int8', cpu_threads=4)
results = []
for path in paths:
    with wave.open(str(path), 'rb') as wav:
        meta = {'sample_rate': wav.getframerate(), 'channels': wav.getnchannels(), 'frames': wav.getnframes(), 'duration_s': round(wav.getnframes()/wav.getframerate(),3)}
    segments, info = model.transcribe(str(path), language='it', word_timestamps=True, vad_filter=True)
    rows = []
    for segment in segments:
        rows.append({'start_s':segment.start,'end_s':segment.end,'text':segment.text,'words':[{'start_s':w.start,'end_s':w.end,'text':w.word,'probability':w.probability} for w in (segment.words or [])]})
    row = {'file':path.name,'path':str(path),'media':meta,'language':info.language,'language_probability':info.language_probability,'segments':rows,'text':' '.join(x['text'].strip() for x in rows).strip()}
    results.append(row)
    print(f"{path.name} ({meta['duration_s']}s): {row['text']}", flush=True)
out = {'model':'faster-whisper small cached snapshot 536b0662742c02347bc0e980a01041f333bce120','device':'cpu','compute_type':'int8','cpu_threads':4,'local_cache_only':True,'results':results}
(root/'asr_faster_whisper_small_cpu.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
