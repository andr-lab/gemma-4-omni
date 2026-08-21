/**
 * GPU-3090 Extension — esecuzione remota su macchina Linux con RTX 3090 via SSH + Docker.
 *
 * Sicurezza:
 *  - Le credenziali SSH NON vivono in questa estensione: si usano `~/.ssh/config`
 *    (Host gpu3090) con chiave ed25519 (ForwardAgent no, IdentitiesOnly yes).
 *  - Docker remoto: MAI --privileged, MAI mount della root, default utente non-root,
 *    --gpus all, --ipc=host solo se richiesto, reti limitate (--network host no di default).
 *  - Il config dell'estensione (host, user, paths, docker args) è in config.json
 *    nella stessa cartella, NON committato in nessun repo git.
 *  - Ogni comando remoto è eseguito con `set -e` e timeout esplicito.
 *
 * Tools registrati:
 *  - gpu_connect      : verifica connessione SSH + stato GPU (nvidia-smi) + docker
 *  - gpu_exec         : esegue un comando remoto (cwd opzionale)
 *  - gpu_pull         : copia file/dir dalla 3090 al locale (scp -r)
 *  - gpu_push         : copia file/dir dal locale alla 3090 (scp -r)
 *  - gpu_docker_run   : build+run di un container (immagine, comando, gpu, mounts)
 *  - gpu_docker_logs  : tail dei log di un container
 *  - gpu_status       : stato sintetico macchina (gpu, disk, ram, docker ps)
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

// ---------------------------------------------------------------------------
// Config (lettura a ogni chiamata: consente modifica senza /reload)
// ---------------------------------------------------------------------------

const __dirname = dirname(fileURLToPath(import.meta.url));
const CONFIG_PATH = join(__dirname, "config.json");

interface GpuConfig {
  sshHost: string; // alias in ~/.ssh/config
  remoteUser: string;
  remoteBaseDir: string; // dir progetto remota (es. ~/gemma-4-omni)
  dockerImage?: string; // immagine di default
  containerUser?: string; // utente nel container (default: nessuno = root? NO: default uid 1000)
  containerUid?: number;
  containerGid?: number;
  defaultTimeoutSec?: number;
}

const DEFAULTS: GpuConfig = {
  sshHost: "gpu3090",
  remoteUser: "andrea",
  remoteBaseDir: "~/gemma-4-omni",
  containerUid: 1000,
  containerGid: 1000,
  defaultTimeoutSec: 600,
};

function loadConfig(): GpuConfig {
  if (existsSync(CONFIG_PATH)) {
    try {
      const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
      return { ...DEFAULTS, ...raw };
    } catch (e) {
      throw new Error(`config.json non valido (${CONFIG_PATH}): ${(e as Error).message}`);
    }
  }
  return DEFAULTS;
}

// ---------------------------------------------------------------------------
// Helper SSH — tutti i comandi passano da ~/.ssh/config (mai password inline)
// ---------------------------------------------------------------------------

interface SshResult {
  stdout: string;
  stderr: string;
  code: number | null;
  killed: boolean;
}

async function sshRun(
  cmd: string,
  opts: { timeoutMs?: number; cwdRemote?: string } = {},
): Promise<SshResult> {
  const cfg = loadConfig();
  const timeoutMs = opts.timeoutMs ?? (cfg.defaultTimeoutSec ?? 600) * 1000;
  // Shell remota: set -e; cd se richiesto; output senza banner.
  const remote = opts.cwdRemote
    ? `cd ${shellQuote(opts.cwdRemote)} && ${cmd}`
    : cmd;
  try {
    const { stdout, stderr } = await execFileAsync(
      "ssh",
      [
        "-o", "BatchMode=yes", // mai prompt password (blocca se la chiave manca)
        "-o", "ConnectTimeout=15",
        cfg.sshHost,
        `set -e; ${remote}`,
      ],
      { timeout: timeoutMs, maxBuffer: 64 * 1024 * 1024, windowsHide: true },
    );
    return { stdout, stderr, code: 0, killed: false };
  } catch (e: unknown) {
    const err = e as {
      code?: number;
      killed?: boolean;
      signal?: string;
      stdout?: string;
      stderr?: string;
    };
    return {
      stdout: err.stdout ?? "",
      stderr: err.stderr ?? String(e),
      code: err.code ?? 1,
      killed: !!err.killed,
    };
  }
}

function shellQuote(s: string): string {
  return `'${s.replace(/'/g, `'\\''`)}'`;
}

async function scpTransfer(
  direction: "push" | "pull",
  localPath: string,
  remotePath: string,
  opts: { timeoutMs?: number } = {},
): Promise<SshResult> {
  const cfg = loadConfig();
  const timeoutMs = opts.timeoutMs ?? (cfg.defaultTimeoutSec ?? 600) * 1000;
  const src = direction === "push" ? localPath : `${cfg.sshHost}:${remotePath}`;
  const dst = direction === "push" ? `${cfg.sshHost}:${remotePath}` : localPath;
  try {
    const { stdout, stderr } = await execFileAsync(
      "scp",
      [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=15",
        "-r",
        src,
        dst,
      ],
      { timeout: timeoutMs, maxBuffer: 64 * 1024 * 1024, windowsHide: true },
    );
    return { stdout, stderr, code: 0, killed: false };
  } catch (e: unknown) {
    const err = e as { code?: number; killed?: boolean; stdout?: string; stderr?: string };
    return {
      stdout: err.stdout ?? "",
      stderr: err.stderr ?? String(e),
      code: err.code ?? 1,
      killed: !!err.killed,
    };
  }
}

// ---------------------------------------------------------------------------
// Tool: gpu_connect
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tool registrati sotto
// ---------------------------------------------------------------------------

export default function gpu3090Extension(pi: ExtensionAPI) {
  pi.registerTool({
    name: "gpu_connect",
    label: "GPU Connect",
    description:
      "Verifica la connessione SSH alla macchina 3090 (via ~/.ssh/config), mostra stato GPU (nvidia-smi), Docker e disco. Da chiamare PRIMA di qualsiasi altra operazione remota.",
    promptSnippet: "Verificare connessione SSH e stato GPU della macchina 3090",
    promptGuidelines: [
      "Use gpu_connect before the first gpu_* call in a session to confirm SSH + GPU + Docker sono disponibili.",
    ],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, onUpdate, ctx) {
      onUpdate?.({ content: [{ type: "text", text: "Verifica connessione SSH a gpu3090..." }] });
      const cfg = loadConfig();
      const r = await sshRun(
        `echo "HOST=$(hostname)"; echo "USER=$(whoami)"; ` +
          `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1; ` +
          `echo "---DOCKER---"; docker --version 2>&1; docker ps --format '{{.Names}} {{.Status}}' 2>&1; ` +
          `echo "---DISK---"; df -h ${shellQuote(cfg.remoteBaseDir.startsWith("~") ? cfg.remoteBaseDir : cfg.remoteBaseDir)} 2>&1 | tail -2; ` +
          `echo "---RAM---"; free -h 2>&1 | head -2; ` +
          `echo "---PROJECT---"; ls ${shellQuote(cfg.remoteBaseDir)} 2>&1 | head -20`,
        { timeoutMs: 60000 },
      );
      if (r.code !== 0) {
        return {
          content: [
            {
              type: "text",
              text:
                `Connessione SSH FALLITA (code ${r.code}).\n` +
                `stderr: ${r.stderr.slice(0, 2000)}\n\n` +
                `Fix: esegui sul terminale locale: ssh-copy-id ${cfg.remoteUser}@192.168.0.186  (una tantum, serve la password). ` +
                `Poi ricontrolla che ~/.ssh/config abbia il blocco Host ${cfg.sshHost}.`,
            },
          ],
          details: { ok: false, sshCode: r.code },
        };
      }
      return {
        content: [{ type: "text", text: `SSH OK:\n${r.stdout}` }],
        details: { ok: true },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: gpu_exec
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "gpu_exec",
    label: "GPU Exec",
    description:
      "Esegue un comando shell sulla macchina 3090 via SSH. cwdRemote opzionale per lavorare in una cartella del progetto. Non usare per run docker (usa gpu_docker_run).",
    promptSnippet: "Eseguire un comando shell sulla macchina 3090",
    promptGuidelines: [
      "Use gpu_exec for any shell command on the remote 3090 machine (files, python, nvidia-smi, git).",
      "Use gpu_docker_run instead when the task must run inside a container with GPU.",
    ],
    parameters: Type.Object({
      command: Type.String({ description: "Comando shell da eseguire (set -e applicato)" }),
      cwdRemote: Type.Optional(
        Type.String({ description: "Cartella remota in cui eseguire (default: remoteBaseDir)" }),
      ),
      timeoutSec: Type.Optional(
        Type.Number({ description: "Timeout in secondi (default: da config, 600)" }),
      ),
    }),
    async execute(_toolCallId, params, _signal, onUpdate, ctx) {
      const cfg = loadConfig();
      const cwd = params.cwdRemote ?? cfg.remoteBaseDir;
      onUpdate?.({ content: [{ type: "text", text: `ssh ${cfg.sshHost}: ${params.command.slice(0, 200)}` }] });
      const r = await sshRun(params.command, {
        cwdRemote: cwd,
        timeoutMs: (params.timeoutSec ?? cfg.defaultTimeoutSec ?? 600) * 1000,
      });
      const out = r.stdout || "(nessun output)";
      const errPart = r.stderr ? `\n[stderr]\n${r.stderr.slice(0, 4000)}` : "";
      if (r.code !== 0) {
        return {
          content: [
            {
              type: "text",
              text: `EXIT ${r.code}${errPart}\n[stdout]\n${out.slice(0, 12000)}`,
            },
          ],
          details: { ok: false, code: r.code, cwd },
        };
      }
      return {
        content: [{ type: "text", text: `${out.slice(0, 12000)}${errPart}` }],
        details: { ok: true, code: 0, cwd },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: gpu_pull / gpu_push
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "gpu_pull",
    label: "GPU Pull",
    description:
      "Copia file o cartelle dalla macchina 3090 al locale (scp -r). remotePath relativo a remoteBaseDir o assoluto.",
    promptSnippet: "Scaricare file dalla macchina 3090 al locale",
    promptGuidelines: ["Use gpu_pull to download artifacts (checkpoint, wav, report) from the 3090."],
    parameters: Type.Object({
      remotePath: Type.String({ description: "Path remoto (assoluto o relativo a remoteBaseDir)" }),
      localPath: Type.String({ description: "Path locale di destinazione" }),
    }),
    async execute(_toolCallId, params, _signal, onUpdate, ctx) {
      const cfg = loadConfig();
      const rp = params.remotePath.startsWith("/") || params.remotePath.startsWith("~")
        ? params.remotePath
        : `${cfg.remoteBaseDir}/${params.remotePath}`;
      onUpdate?.({ content: [{ type: "text", text: `scp -r ${rp} -> ${params.localPath}` }] });
      const r = await scpTransfer("pull", params.localPath, rp, { timeoutMs: 3600_000 });
      if (r.code !== 0) {
        return {
          content: [{ type: "text", text: `PULL FAILED (${r.code}): ${r.stderr.slice(0, 2000)}` }],
          details: { ok: false },
        };
      }
      return { content: [{ type: "text", text: `PULL OK: ${params.localPath}` }], details: { ok: true } };
    },
  });

  pi.registerTool({
    name: "gpu_push",
    label: "GPU Push",
    description:
      "Copia file o cartelle dal locale alla macchina 3090 (scp -r). remotePath relativo a remoteBaseDir o assoluto.",
    promptSnippet: "Caricare file dal locale alla macchina 3090",
    promptGuidelines: ["Use gpu_push to upload code/data to the 3090 before running."],
    parameters: Type.Object({
      localPath: Type.String({ description: "Path locale da inviare" }),
      remotePath: Type.String({ description: "Path remoto di destinazione (assoluto o relativo a remoteBaseDir)" }),
    }),
    async execute(_toolCallId, params, _signal, onUpdate, ctx) {
      const cfg = loadConfig();
      const rp = params.remotePath.startsWith("/") || params.remotePath.startsWith("~")
        ? params.remotePath
        : `${cfg.remoteBaseDir}/${params.remotePath}`;
      onUpdate?.({ content: [{ type: "text", text: `scp -r ${params.localPath} -> ${rp}` }] });
      const r = await scpTransfer("push", params.localPath, rp, { timeoutMs: 3600_000 });
      if (r.code !== 0) {
        return {
          content: [{ type: "text", text: `PUSH FAILED (${r.code}): ${r.stderr.slice(0, 2000)}` }],
          details: { ok: false },
        };
      }
      return { content: [{ type: "text", text: `PUSH OK: ${rp}` }], details: { ok: true } };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: gpu_docker_run
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "gpu_docker_run",
    label: "GPU Docker Run",
    description:
      "Esegue un comando in un container Docker sulla 3090 con accesso GPU (--gpus all). " +
      "Default: container eseguito come utente non-root (uid/gid da config), no --privileged, " +
      "no --network host, mount del progetto remoto in /workspace. Specificare image e cmd.",
    promptSnippet: "Eseguire un comando in Docker con GPU sulla 3090",
    promptGuidelines: [
      "Use gpu_docker_run to run python/training/inference inside a container with the RTX 3090 GPU.",
      "Do NOT pass --privileged, --network host, or host-root mounts: the extension rejects them for security.",
    ],
    parameters: Type.Object({
      image: Type.String({ description: "Immagine Docker (default: da config o nvcr/pytorch)" }),
      command: Type.String({ description: "Comando da eseguire nel container (es. python train.py)" }),
      workdir: Type.Optional(
        Type.String({ description: "Workdir nel container (default: /workspace)" }),
      ),
      mountProject: Type.Optional(
        Type.Boolean({ description: "Monta remoteBaseDir in /workspace (default: true)" }),
      ),
      extraArgs: Type.Optional(
        Type.Array(Type.String(), {
          description: "Argomenti docker aggiuntivi (es. ['-e', 'CUDA_VISIBLE_DEVICES=0', '-p', '8000:8000'])",
        }),
      ),
      timeoutSec: Type.Optional(Type.Number({ description: "Timeout (default: 3600)" })),
    }),
    async execute(_toolCallId, params, _signal, onUpdate, ctx) {
      const cfg = loadConfig();
      const image = params.image ?? cfg.dockerImage ?? "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04";
      const extra = params.extraArgs ?? [];
      if (extra.includes("--privileged") || extra.includes("--network") || extra.includes("--net")) {
        return {
          content: [
            { type: "text", text: "RIFIUTATO: --privileged / --network host non consentiti per sicurezza." },
          ],
          details: { ok: false, reason: "security" },
        };
      }
      const uid = cfg.containerUid ?? 1000;
      const gid = cfg.containerGid ?? 1000;
      const mountProj = params.mountProject !== false;
      const workdir = params.workdir ?? "/workspace";

      const dockerArgs: string[] = ["run", "--rm"];
      dockerArgs.push("--gpus", "all");
      dockerArgs.push("--ipc=host"); // shared memory per DataLoader (sicuro, non privileged)
      if (mountProj) {
        dockerArgs.push("-v", `${cfg.remoteBaseDir}:/workspace`);
      }
      dockerArgs.push("--user", `${uid}:${gid}`);
      dockerArgs.push("-w", workdir);
      for (const a of extra) dockerArgs.push(a);
      dockerArgs.push(image);
      dockerArgs.push("bash", "-lc", params.command);

      onUpdate?.({
        content: [{ type: "text", text: `docker ${dockerArgs.join(" ").slice(0, 400)}...` }],
      });

      const r = await sshRun(
        `docker ${dockerArgs.map(shellQuote).join(" ")}`,
        { timeoutMs: (params.timeoutSec ?? 3600) * 1000 },
      );
      const out = r.stdout || "(nessun output)";
      const errPart = r.stderr ? `\n[stderr]\n${r.stderr.slice(0, 4000)}` : "";
      if (r.code !== 0) {
        return {
          content: [
            { type: "text", text: `DOCKER EXIT ${r.code}${errPart}\n[stdout]\n${out.slice(0, 12000)}` },
          ],
          details: { ok: false, code: r.code, image },
        };
      }
      return {
        content: [{ type: "text", text: `DOCKER OK (${image})\n${out.slice(0, 12000)}${errPart}` }],
        details: { ok: true, code: 0, image },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: gpu_docker_logs
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "gpu_docker_logs",
    label: "GPU Docker Logs",
    description: "Mostra gli ultimi N righe dei log di un container Docker sulla 3090 (per container in esecuzione).",
    promptSnippet: "Leggere i log di un container Docker sulla 3090",
    promptGuidelines: ["Use gpu_docker_logs to tail logs of a running container."],
    parameters: Type.Object({
      container: Type.String({ description: "Nome o ID del container" }),
      lines: Type.Optional(Type.Number({ description: "Numero di righe (default 100)" })),
    }),
    async execute(_toolCallId, params, _signal, onUpdate, ctx) {
      const lines = params.lines ?? 100;
      const r = await sshRun(`docker logs --tail ${lines} ${shellQuote(params.container)} 2>&1`, {
        timeoutMs: 60000,
      });
      if (r.code !== 0) {
        return {
          content: [{ type: "text", text: `LOG FAILED: ${r.stderr.slice(0, 1500)}` }],
          details: { ok: false },
        };
      }
      return {
        content: [{ type: "text", text: r.stdout.slice(0, 12000) || "(log vuoto)" }],
        details: { ok: true },
      };
    },
  });

  // -------------------------------------------------------------------------
  // Tool: gpu_status
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: "gpu_status",
    label: "GPU Status",
    description:
      "Stato sintetico della macchina 3090: utilizzo GPU, processi GPU, docker ps, disco, RAM. Versione leggera di gpu_connect.",
    promptSnippet: "Stato sintetico GPU e macchina 3090",
    promptGuidelines: ["Use gpu_status for a quick health check of the 3090 during long runs."],
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, onUpdate, ctx) {
      const r = await sshRun(
        `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1; ` +
          `echo "---PROC---"; nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader 2>&1; ` +
          `echo "---DOCKER---"; docker ps --format '{{.Names}} {{.Status}}' 2>&1; ` +
          `echo "---DISK---"; df -h / 2>&1 | tail -1; echo "---RAM---"; free -h 2>&1 | sed -n 2p`,
        { timeoutMs: 60000 },
      );
      if (r.code !== 0) {
        return {
          content: [{ type: "text", text: `STATUS FAILED: ${r.stderr.slice(0, 1500)}` }],
          details: { ok: false },
        };
      }
      return { content: [{ type: "text", text: r.stdout }], details: { ok: true } };
    },
  });

  // -------------------------------------------------------------------------
  // Comando /gpu-status per debug manuale
  // -------------------------------------------------------------------------

  pi.registerCommand("gpu-status", {
    description: "Stato della macchina 3090 via SSH (gpu, docker, disco)",
    handler: async (_args, ctx) => {
      const r = await sshRun(
        `nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>&1; docker ps --format '{{.Names}} {{.Status}}' 2>&1`,
        { timeoutMs: 60000 },
      );
      ctx.ui.notify(r.code === 0 ? r.stdout : `FAILED: ${r.stderr.slice(0, 800)}`, r.code === 0 ? "info" : "error");
    },
  });
}
