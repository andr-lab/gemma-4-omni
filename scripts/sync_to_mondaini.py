#!/usr/bin/env python3
"""
sync_to_mondaini.py - Sincronizzazione automatica del codice verso Mondaini (RTX 3090).

Garantisce che:
1. I file pesanti (checkpoints/, data/, *.pt, *.safetensors, *.wav, *.parquet)
   restano ESCLUSIVAMENTE su Mondaini.
2. La repository locale e GitHub rimangono leggerissime.
3. Le modifiche al codice (.py, .cu, .h, .cuh, .sh, .json, .md, .toml)
   vengono sincronizzate in tempo reale o su richiesta.

Modalità d'uso:
  python scripts/sync_to_mondaini.py           # Sincronizzazione incrementale one-shot
  python scripts/sync_to_mondaini.py --watch   # Watcher continuo in tempo reale (Ctrl+C per fermare)
  python scripts/sync_to_mondaini.py --git     # Allinea Mondaini all'ultimo commit GitHub
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Set

SSH_HOST = "mondaini"
REMOTE_BASE_DIR = "/home/administrator/gemma-4-omni"
LOCAL_BASE_DIR = Path(__file__).resolve().parents[1]

# Cartelle ed estensioni da escludere CATEGORICAMENTE (pesanti o runtime locale)
EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv_e2e_runtime",
    "__pycache__",
    ".tmp",
    ".cache",
    ".antigravity",
    "checkpoints",
    "data",
    "samples",
    "artifacts",
    "build",
    "dist",
}

EXCLUDE_EXTENSIONS = {
    ".pt",
    ".pth",
    ".safetensors",
    ".bin",
    ".gguf",
    ".onnx",
    ".ckpt",
    ".parquet",
    ".npy",
    ".npz",
    ".arrow",
    ".wav",
    ".mp3",
    ".flac",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".obj",
    ".pyc",
    ".log",
}

# Estensioni codice da monitorare attivamente
CODE_EXTENSIONS = {
    ".py",
    ".cu",
    ".h",
    ".cuh",
    ".inc",
    ".cpp",
    ".sh",
    ".bash",
    ".json",
    ".jsonl",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".txt",
    ".csv",
    ".sql",
}


def is_excluded(path: Path) -> bool:
    """Verifica se il percorso deve essere escluso dalla sincronizzazione."""
    rel = path.relative_to(LOCAL_BASE_DIR)
    
    # Controlla parti del path
    for part in rel.parts:
        if part in EXCLUDE_DIRS or part.startswith(".tmp_"):
            return True
            
    # Controlla estensione
    ext = path.suffix.lower()
    if ext in EXCLUDE_EXTENSIONS:
        return True
        
    return False


def is_code_file(path: Path) -> bool:
    """Verifica se è un file di codice o configurazione da sincronizzare."""
    if is_excluded(path):
        return False
    return path.suffix.lower() in CODE_EXTENSIONS or path.name in {
        "Dockerfile",
        ".dockerignore",
        ".gitignore",
        "README.md",
        "AGENTS.md",
    }


def sync_single_file(local_path: Path) -> bool:
    """Invia un singolo file verso Mondaini via SCP."""
    if not local_path.is_file() or is_excluded(local_path):
        return False

    rel_path = local_path.relative_to(LOCAL_BASE_DIR).as_posix()
    remote_path = f"{REMOTE_BASE_DIR}/{rel_path}"
    remote_dir = os.path.dirname(remote_path)

    # Assicura che la directory remota esista
    mkdir_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", SSH_HOST, f"mkdir -p '{remote_dir}'"]
    subprocess.run(mkdir_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

    scp_cmd = [
        "scp",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        str(local_path),
        f"{SSH_HOST}:{remote_path}",
    ]
    res = subprocess.run(scp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if res.returncode == 0:
        size_kb = local_path.stat().st_size / 1024
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] -> {rel_path} ({size_kb:.1f} KB) [OK]")
        return True
    else:
        now = time.strftime("%H:%M:%S")
        print(f"[{now}] ERR {rel_path}: {res.stderr.strip()}", file=sys.stderr)
        return False


def sync_all_incremental() -> int:
    """Scansiona tutti i file di codice locali e sincronizza quelli modificati."""
    print(f"Scansione codice locale in {LOCAL_BASE_DIR}...")
    synced_count = 0
    total_files = 0

    for root, dirs, files in os.walk(LOCAL_BASE_DIR):
        # Filtra directory escluse in-place per non scendere inutilmente
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".tmp")]
        for f in files:
            p = Path(root) / f
            if is_code_file(p):
                total_files += 1

    print(f"Trovati {total_files} file di codice tracciabili (esclusi checkpoint e data pesanti).")
    print(f"Avvio sincronizzazione verso {SSH_HOST}:{REMOTE_BASE_DIR}...")

    # Raccogli tutti i file idonei
    for root, dirs, files in os.walk(LOCAL_BASE_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".tmp")]
        for f in files:
            p = Path(root) / f
            if is_code_file(p):
                if sync_single_file(p):
                    synced_count += 1

    print(f"Sincronizzazione completata: {synced_count}/{total_files} file aggiornati.")
    return synced_count


def sync_git_remote() -> bool:
    """Allinea Mondaini all'ultimo commit su GitHub."""
    print(f"Allineamento git su {SSH_HOST} verso origin/main da GitHub...")
    cmd = (
        f"cd {REMOTE_BASE_DIR} && "
        f"git remote add github https://github.com/andr-lab/gemma-4-omni.git 2>/dev/null || "
        f"git remote set-url github https://github.com/andr-lab/gemma-4-omni.git ; "
        f"git fetch github main && "
        f"git merge --ff-only github/main 2>/dev/null || "
        f"git reset --soft github/main"
    )
    res = subprocess.run(["ssh", "-o", "BatchMode=yes", SSH_HOST, cmd], text=True, capture_output=True)
    if res.returncode == 0:
        print("Mondaini allineato con successo al commit GitHub:")
        print(res.stdout.strip())
        return True
    else:
        print(f"Errore durante l'allineamento git: {res.stderr.strip()}", file=sys.stderr)
        return False


def run_watcher(poll_interval: float = 1.0) -> None:
    """Monitor continuo di modifiche locali con debouncing."""
    print("=" * 65)
    print("  WATCHER AUTOMATICO ATTIVO: Sincronizzazione continua su Mondaini")
    print(f"  Locale:  {LOCAL_BASE_DIR}")
    print(f"  Remoto:  {SSH_HOST}:{REMOTE_BASE_DIR}")
    print("  Filtro:  Solo codice sorgente (checkpoints/ e data/ ESCLUSI)")
    print("  Premi Ctrl+C per interrompere.")
    print("=" * 65)

    mtimes: dict[Path, float] = {}
    
    # Inizializza snapshot mtimes
    for root, dirs, files in os.walk(LOCAL_BASE_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".tmp")]
        for f in files:
            p = Path(root) / f
            if is_code_file(p):
                try:
                    mtimes[p] = p.stat().st_mtime
                except OSError:
                    pass

    try:
        while True:
            time.sleep(poll_interval)
            changed_files: list[Path] = []

            for root, dirs, files in os.walk(LOCAL_BASE_DIR):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".tmp")]
                for f in files:
                    p = Path(root) / f
                    if is_code_file(p):
                        try:
                            cur_mtime = p.stat().st_mtime
                            last_mtime = mtimes.get(p)
                            if last_mtime is None or cur_mtime > last_mtime:
                                mtimes[p] = cur_mtime
                                changed_files.append(p)
                        except OSError:
                            continue

            if changed_files:
                for p in changed_files:
                    sync_single_file(p)

    except KeyboardInterrupt:
        print("\nWatcher arrestato dall'utente.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sincronizzatore automatico codice verso Mondaini (RTX 3090).")
    parser.add_argument("--watch", "-w", action="store_true", help="Avvia il demone watcher in tempo reale.")
    parser.add_argument("--git", "-g", action="store_true", help="Allinea Mondaini all'ultimo commit di GitHub.")
    parser.add_argument("--file", "-f", type=str, help="Sincronizza un singolo file specifico.")
    args = parser.parse_args()

    if args.git:
        sync_git_remote()
    elif args.file:
        p = Path(args.file).resolve()
        sync_single_file(p)
    elif args.watch:
        run_watcher()
    else:
        sync_all_incremental()


if __name__ == "__main__":
    main()
