# gpu-3090 — estensione pi per run remote su RTX 3090 (SSH + Docker)

Estensione pi che permette all'agente di eseguire comandi, trasferire file e
lanciare container Docker con GPU sulla macchina Linux con RTX 3090
(`192.168.0.186`), via SSH. Niente credenziali nell'estensione: si usa la
config SSH del sistema.

## Setup una tantum (a mano, sul terminale locale)

1. Generare la chiave (già fatto):

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "pi-agent-3090"
   ```

2. Installare la chiave sulla 3090 (serve la password UNA volta):

   ```bash
   ssh-copy-id <UTENTE>@192.168.0.186
   ```

   Sostituire `<UTENTE>` con l'username reale della macchina (es. `andrea`).
   Se l'utente non è `andrea`, aggiornare `~/.ssh/config`.

3. Verificare:

   ```bash
   ssh gpu3090 "hostname && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && docker --version"
   ```

4. (Opzionale) Copiare `config.example.json` → `config.json` e adattare
   `remoteUser`, `remoteBaseDir`, `dockerImage`.

## Sicurezza

- **Credenziali MAI in git**: la chiave privata e `~/.ssh/config` non vanno
  committati. L'estensione legge solo `~/.ssh/config` (`Host gpu3090`).
- **Docker**: l'estensione rifiuta `--privileged` e `--network host`. Default
  utente non-root (uid/gid da config), `--gpus all`, `--ipc=host` per
  DataLoader, mount del progetto remoto in `/workspace` (read-write).
- **BatchMode**: ssh senza prompt password; se la chiave manca, fallisce
  subito con istruzioni, non si blocca.
- **Timeout espliciti** su ogni chiamata SSH/Docker.
- Non loggare mai chiavi o segreti; i comandi vanno tenuti minimali.

## Tools

| Tool | Scopo |
|------|-------|
| `gpu_connect` | Verifica SSH + GPU + Docker + disco + progetto remoto |
| `gpu_exec` | Comando shell remoto (cwd opzionale, timeout) |
| `gpu_pull` | scp -r dalla 3090 al locale (checkpoint, wav, report) |
| `gpu_push` | scp -r dal locale alla 3090 (codice, dati) |
| `gpu_docker_run` | build/run container con GPU (default pytorch/pytorch) |
| `gpu_docker_logs` | tail log di un container |
| `gpu_status` | Stato sintetico GPU/proc/docker/disco/RAM |

Comando manuale: `/gpu-status`.

## Workflow tipico

1. `gpu_push` del repo/script sulla 3090 (o `git clone` in `remoteBaseDir`)
2. `gpu_docker_run` con `image` e `command` (es. `python train.py`)
3. `gpu_status` durante il run per monitorare GPU/RAM
4. `gpu_pull` degli artefatti (checkpoint, wav, report) al termine
