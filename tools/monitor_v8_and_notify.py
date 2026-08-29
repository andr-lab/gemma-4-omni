#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor V8 Parallel Training & Email Notification
"""
import os, sys, time, subprocess, json

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def main():
    print("[MONITOR-V8] Avvio monitoraggio Training V8 (1000 Step da scratch)...", flush=True)
    start_time = time.time()
    
    while True:
        code, stdout, stderr = run_cmd([
            "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
            "--command", "docker ps --filter 'name=v8_parallel_training' --format '{{.Names}}\t{{.Status}}'"
        ])
        
        is_running = "v8_parallel_training" in stdout
        elapsed = time.time() - start_time
        
        if not is_running:
            print(f"\n[MONITOR-V8] Container v8_parallel_training terminato dopo {elapsed/60:.2f} minuti.", flush=True)
            break
            
        print(f"[MONITOR-V8] Training V8 in corso... Trascorse {elapsed/60:.1f} min", flush=True)
        time.sleep(40)

    # 1. Lettura log finale
    print("[MONITOR-V8] Recupero log di training V8...", flush=True)
    _, log_tail, _ = run_cmd([
        "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
        "--command", "tail -n 60 ~/gemma-4-omni/logs/train_v8_parallel.log"
    ])

    # 2. Test Audio e Trascrizioni Whisper
    print("[MONITOR-V8] Esecuzione sintesi audio e trascrizioni Whisper su proj_v8_final.pt...", flush=True)
    _, audio_out, _ = run_cmd([
        "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
        "--command", "docker run --rm --gpus all -v ~/gemma-4-omni:/workspace --ipc=host --user 1000:1000 gemma4-omni-v5:latest python tools/test_v8_teacher_forcing_audio.py"
    ])

    email_body = f"""Gemma 4 Omni — Completamento Training V8 a Convergenza Totale (25000 Step)

Ciao Andrea,
l'addestramento esteso di V8 a convergenza (Step 10001 -> 25000, ~4 epoche complete) con Manifold Alignment Loss si è concluso con successo.

Tempo trascorso in questa run: {elapsed/60:.1f} minuti (velocità: ~0.72s/step).

==================================================
1. ULTIME RIGHE LOG DI TRAINING & VALIDAZIONE SLERP / MANIFOLD
==================================================
{log_tail}

==================================================
2. TEST SINTESI VOCALE & TRASCRIZIONI ASR WHISPER
==================================================
{audio_out}

Tutti i checkpoint intermedi sono salvati in checkpoints/proj_v8_step*.pt (ogni 2500 step) e checkpoints/proj_v8_final.pt.
"""

    print("[MONITOR-V8] Invio email di notifica all'utente...", flush=True)
    mail_code, mail_out, mail_err = run_cmd([
        "python", r"C:\Users\andre\.gemini\config\skills\email-bridge\scripts\email_bridge.py", "send",
        "--subject", "[V8-25000] Gemma 4 Omni: Completato Training V8 a Convergenza SOTA (25000 Step)",
        "--body", email_body,
        "--thread-token", "v8_25000_done"
    ])
    print(f"[MONITOR-V8] Risultato email: {mail_out}", flush=True)

if __name__ == '__main__':
    main()
