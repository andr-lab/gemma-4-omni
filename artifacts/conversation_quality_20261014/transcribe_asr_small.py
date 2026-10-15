import argparse, json, wave
from pathlib import Path
from faster_whisper import WhisperModel

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('files', nargs='+')
a = p.parse_args()
model = WhisperModel('small', device='cpu', compute_type='int8', cpu_threads=4, download_root='/workspace/.hf_cache')
results = []
for filename in a.files:
    path = Path(filename)
    with wave.open(str(path), 'rb') as wav:
        media = {'sample_rate': wav.getframerate(), 'channels': wav.getnchannels(), 'frames': wav.getnframes(), 'duration_s': wav.getnframes() / wav.getframerate()}
    segments, info = model.transcribe(str(path), language='it', word_timestamps=True, vad_filter=True)
    rows = []
    for segment in segments:
        rows.append({'start_s': segment.start, 'end_s': segment.end, 'text': segment.text, 'words': [{'start_s': word.start, 'end_s': word.end, 'text': word.word, 'probability': word.probability} for word in (segment.words or [])]})
    results.append({'file': path.name, 'path': str(path), 'language': info.language, 'language_probability': info.language_probability, 'media': media, 'segments': rows, 'text': ''.join(item['text'] for item in rows).strip()})
out = {'model': 'faster-whisper small', 'device': 'cpu', 'compute_type': 'int8', 'results': results}
Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
for row in results:
    print(row['file'] + ': ' + row['text'])
