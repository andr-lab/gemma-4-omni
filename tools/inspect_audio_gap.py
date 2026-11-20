import wave
import numpy as np

wav_path = "work/gpu_forensics/fixtures_16k_final/long_backpressure_prompt.wav"
with wave.open(wav_path, "rb") as w:
    frames = w.getnframes()
    pcm = np.frombuffer(w.readframes(frames), dtype=np.int16).astype(np.float32) / 32768.0

frame_size = 1280 # 80ms @ 16kHz
n_frames = len(pcm) // frame_size

print(f"Total 80ms frames: {n_frames}")
for i in range(260, min(n_frames, 342)):
    chunk = pcm[i*frame_size : (i+1)*frame_size]
    rms = np.sqrt(np.mean(chunk**2))
    voiced = rms >= 0.002
    print(f"Frame {i:3d}: rms={rms:.5f} {'[VOICED]' if voiced else '[SILENCE]'}")
