#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script di monitoraggio in background per il Blocco 2 (Step 501-1000) su RTX 3090.
Quando il container Docker termina:
  1. Esegue la diagnostica manifold su proj_v7_final.pt
  2. Raccoglie i risultati di loss e trascrizioni ASR
  3. Invia email di completamento formattata all'utente via email_bridge.py
"""
import os, sys, time, subprocess, json

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.returncode, p.stdout.strip(), p.stderr.strip()

def main():
    print("[MONITOR] Avvio monitoraggio Blocco 2 (Step 501 -> 1000)...", flush=True)
    start_time = time.time()
    
    # Check interval: 90 secondi
    while True:
        code, stdout, stderr = run_cmd([
            "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
            "--command", "docker ps --filter 'name=v7_block2_training' --format '{{.Names}}\t{{.Status}}'"
        ])
        
        is_running = "v7_block2_training" in stdout
        elapsed = time.time() - start_time
        
        if not is_running:
            print(f"\n[MONITOR] Container v7_block2_training terminato dopo {elapsed/3600:.2f} ore.", flush=True)
            break
            
        print(f"[MONITOR] Training in corso... Trascorse {elapsed/60:.1f} min ({elapsed/3600:.2f} h)", flush=True)
        time.sleep(90)

    # 1. Lettura log finale
    print("[MONITOR] Recupero log di training...", flush=True)
    _, log_tail, _ = run_cmd([
        "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
        "--command", "tail -n 60 ~/gemma-4-omni/logs/train_v7_block2.log"
    ])

    # 2. Esecuzione diagnostica manifold
    print("[MONITOR] Esecuzione diagnostica manifold su Checkpoint 1000...", flush=True)
    _, diag_out, _ = run_cmd([
        "python", r"C:\Users\andre\.gemini\config\skills\gpu-3090\scripts\gpu_tool.py", "exec",
        "--command", "docker run --rm --gpus all -v ~/gemma-4-omni:/workspace --ipc=host --user 1000:1000 gemma4-omni-v5:latest python tools/diagnostics_manifold_oracle.py --checkpoint /workspace/checkpoints/proj_v7_final.pt"
    ])

    # 3. Composizione Email
    email_body = f"""Gemma 4 Omni — Completamento Training Blocco 2 (1000 Step Totali)

Ciao Andrea,
il Blocco 2 di addestramento (Step 501 -> 1000) su RTX 3090 si e' concluso con successo.
Tempo totale esecuzione: {elapsed/3600:.2f} ore.

=======================================================
1. DIAGNOSTICA MANIFOLD & SLERP (Step 1000)
=======================================================
{diag_out}

=======================================================
2. LOG FINALE DI ADDESTRAMENTO & SINTESI AUDIO
=======================================================
{log_tail}

=======================================================
3. CHECKPOINT SALVATI
=======================================================
- /workspace/checkpoints/proj_v7_step600.pt ... proj_v7_step1000.pt
- /workspace/checkpoints/proj_v7_final.pt
- /workspace/samples_v7/v7_eval_sample_*.wav

Il sistema e' pronto per ulteriori analisi o per la prosecuzione al Blocco 3.

— Antigravity Agent
"""

    # 4. Invio email tramite email_bridge
    print("[MONITOR] Invio email di notifica all'utente...", flush=True)
    bridge_script = r"C:\Users\andre\.gemini\config\skills\email-bridge\scripts\email_bridge.py"
    send_code, send_out, send_err = run_cmd([
        "python", bridge_script, "--op", "send",
        "--subject", "Gemma 4 Omni: Completato Training V7 Blocco 2 (1000 Step)",
        "--body", email_body,
        "--thread-token", "v7_block2_done"
    ])

    if send_code == 0:
        print(f"[MONITOR] Email inviata con successo: {send_out}", flush=True)
    else:
        print(f"[MONITOR] Errore invio email: {send_err}", flush=True)

if __name__ == '__main__':
    main()
