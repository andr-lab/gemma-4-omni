import sys, numpy as np, pyarrow.parquet as pq, torch
sys.path.insert(0, '.')
import kaggle_pipeline as KP

d = KP.Code2WavDecoder()
print('WRAPPER_MODE:', d._wrapper_mode, flush=True)
assert d._wrapper_mode in ('official_subprocess', 'official_inprocess'), f"MODO SBAGLIATO: {d._wrapper_mode}"

codes = pq.read_table('data/train_tokens_mdc_v2_15.0h_11599s_codes.parquet').to_pandas().values[:64]
wav = d.decode(torch.from_numpy(codes.astype(np.int64)))
wav = np.asarray(wav).reshape(-1)
lag = int(24000/12.5)
x = wav[:lag*8]
ac = np.correlate(x, x, 'full')[len(x)-1:]; ac = ac/(ac[0]+1e-9)
seg = 12000; sub=[]
for i in range(0, len(wav)-seg, seg):
    xx = wav[i:i+seg]*np.hanning(seg)
    spec = np.abs(np.fft.rfft(xx))**2; fr = np.fft.rfftfreq(seg,1/24000)
    sub.append(100*spec[fr<100].sum()/(spec.sum()+1e-8))
corr = float(ac[lag])
print(f'DECODE PIPELINA: len={len(wav)} corr12.5={corr:+.3f} sub<100={np.median(sub):.1f}% rms={np.sqrt((wav**2).mean()):.4f} peak={np.abs(wav).max():.3f}', flush=True)
assert abs(corr) < 0.1, f"ELICA: corr12.5={corr}"
assert np.median(sub) < 2.0, f"SUB-BASS ANCORA: {np.median(sub):.1f}%"
print('GATE PASS: decoder ufficiale via subprocess nella pipeline ✓', flush=True)
d._cleanup_worker()
