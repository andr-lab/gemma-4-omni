import numpy as np, pyarrow.parquet as pq, subprocess, json, time, os

codes = pq.read_table('data/train_tokens_mdc_v2_15.0h_11599s_codes.parquet').to_pandas().values[:64]
np.save('/tmp/codes_test.npy', codes.astype(np.int64))

p = subprocess.Popen(
    ['/opt/venv_phase1/bin/python', '/workspace/decode_official_worker.py', '--daemon'],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    text=True, bufsize=1)
# read READY con timeout
import select
ready = p.stdout.readline().strip()
print('handshake:', ready, flush=True)
if ready != 'READY':
    err = p.stderr.read()[:2000]
    print('STDERR:', err, flush=True)
    p.kill()
    sys.exit(1)
req = json.dumps({'command': 'decode', 'input_file': '/tmp/codes_test.npy', 'output_file': '/tmp/wav_test.npy'})
p.stdin.write(req + '\n'); p.stdin.flush()
t0 = time.time()
resp = json.loads(p.stdout.readline())
print(f'decode resp ({time.time()-t0:.1f}s):', resp, flush=True)
p.stdin.write(json.dumps({'command': 'exit'}) + '\n'); p.stdin.flush()
p.wait(timeout=10)
wav = np.load('/tmp/wav_test.npy')
lag = int(24000/12.5)
x = wav[:lag*8]
ac = np.correlate(x, x, 'full')[len(x)-1:]; ac = ac/(ac[0]+1e-9)
seg = 12000; sub=[]
for i in range(0, len(wav)-seg, seg):
    xx = wav[i:i+seg]*np.hanning(seg)
    spec = np.abs(np.fft.rfft(xx))**2; fr = np.fft.rfftfreq(seg,1/24000)
    sub.append(100*spec[fr<100].sum()/(spec.sum()+1e-8))
print(f'RISULTATO: len={len(wav)} corr12.5={float(ac[lag]):+.3f} sub<100={np.median(sub):.1f}% rms={np.sqrt((wav**2).mean()):.4f} peak={np.abs(wav).max():.3f}')
