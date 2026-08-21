# run_3090.py - Unified execution wrapper for RTX 3090 (SSH + Docker)
# Pure ASCII encoding - no non-ASCII characters, no emojis.

import argparse
import datetime
import os
import subprocess
import sys
import time
from typing import Optional

# Ensure project root and local modules are in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import paths


class DualLogger:
    """Tee logger that writes simultaneously to stdout and a log file."""

    def __init__(self, log_filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(log_filepath)), exist_ok=True)
        self.log_file = open(log_filepath, "a", encoding="ascii", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        # Filter non-ascii characters if any slip through
        clean_msg = message.encode("ascii", "replace").decode("ascii")
        self.log_file.write(clean_msg)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()


def print_banner(title: str) -> None:
    """Print a clean ASCII banner."""
    line = "=" * 78
    print(f"\n{line}")
    print(f"=== {title.upper()} ===")
    print(f"{line}\n")
    sys.stdout.flush()


def print_gpu_telemetry(device_str: str = "cuda:0") -> None:
    """Print GPU device information, driver telemetry, and VRAM limits."""
    try:
        import torch

        if not torch.cuda.is_available():
            print(f"[Telemetry] CUDA not available. Running on CPU.")
            sys.stdout.flush()
            return

        dev_idx = 0
        if ":" in device_str:
            try:
                dev_idx = int(device_str.split(":")[1])
            except ValueError:
                dev_idx = 0

        dev_name = torch.cuda.get_device_name(dev_idx)
        cap = torch.cuda.get_device_capability(dev_idx)
        total_vram_gb = torch.cuda.get_device_properties(dev_idx).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(dev_idx) / (1024**3)
        reserved_gb = torch.cuda.memory_reserved(dev_idx) / (1024**3)

        print(f"[Telemetry] Device: {device_str} ({dev_name})")
        print(f"[Telemetry] Compute Capability: {cap[0]}.{cap[1]}")
        print(f"[Telemetry] Total VRAM: {total_vram_gb:.2f} GB")
        print(f"[Telemetry] VRAM Allocated: {allocated_gb:.2f} GB | Reserved: {reserved_gb:.2f} GB")
        print(f"[Telemetry] PyTorch Version: {torch.__version__} (CUDA {torch.version.cuda})")
        sys.stdout.flush()
    except Exception as e:
        print(f"[Telemetry] Warning: Could not query GPU telemetry ({e})")
        sys.stdout.flush()


def run_phase_prep(args: argparse.Namespace) -> int:
    """Execute Phase 1: data preprocessing, RVQ tokenization, and Parquet caching."""
    print_banner("Phase 1: Preprocessing and RVQ Tokenization")

    # Check if we should delegate to phase1 virtual environment (ADR-003)
    phase1_venv_py = "/opt/venv_phase1/bin/python"
    current_py = sys.executable
    if (
        os.path.isfile(phase1_venv_py)
        and os.path.abspath(current_py) != os.path.abspath(phase1_venv_py)
        and os.environ.get("_SUB_INVOKED_PHASE1") != "1"
    ):
        print(f"[Phase 1] Delegating to Phase 1 virtualenv ({phase1_venv_py})...")
        cmd = [phase1_venv_py, os.path.abspath(__file__), "--phase", "prep"]
        if args.smoke:
            cmd.append("--smoke")
        cmd.extend(["--hours", str(args.hours)])
        cmd.extend(["--samples", str(args.samples)])
        cmd.extend(["--device", args.device])
        env = dict(os.environ)
        env["_SUB_INVOKED_PHASE1"] = "1"
        res = subprocess.run(cmd, env=env)
        return res.returncode

    # Set parameters in environment for kaggle_pipeline
    if args.smoke:
        print("[Phase 1] SMOKE MODE ACTIVE: DUPLEXCONV_MAX_HOURS=0.05, MDC_MAX_SAMPLES=8")
        os.environ["DUPLEXCONV_MAX_HOURS"] = "0.05"
        os.environ["MDC_MAX_SAMPLES"] = "8"
    else:
        os.environ["DUPLEXCONV_MAX_HOURS"] = str(args.hours)
        os.environ["MDC_MAX_SAMPLES"] = str(args.samples)

    import kaggle_pipeline as kp

    kp.DUPLEXCONV_MAX_HOURS = float(os.environ["DUPLEXCONV_MAX_HOURS"])
    kp.MDC_MAX_SAMPLES = int(os.environ["MDC_MAX_SAMPLES"])

    t0 = time.time()
    try:
        kp.execute_phase_1()
        elapsed = time.time() - t0
        print(f"[Phase 1] SUCCESS: Preprocessing completed in {elapsed:.2f}s")
        return 0
    except Exception as e:
        print(f"[Phase 1] ERROR: Phase 1 execution failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_phase_train(args: argparse.Namespace) -> int:
    """Execute Phase 2: Gemma 4 E2B QLoRA co-adaptation and coupling training."""
    print_banner("Phase 2: Gemma 4 Training")
    print_gpu_telemetry(args.device)

    # Apply smoke flags
    if args.smoke:
        print("[Phase 2] SMOKE MODE ACTIVE: PHASE2_SMOKE_BATCHES=4, PHASE2_CKPT_EVERY=2")
        os.environ["PHASE2_SMOKE_BATCHES"] = "4"
        os.environ["PHASE2_CKPT_EVERY"] = "2"
        os.environ["SMOKE"] = "1"

    # Set run log destination inside logs dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(paths.get_logs_dir(), f"phase2_run_log_{timestamp}.tsv")
    os.environ["PHASE2_RUN_LOG"] = run_log_path

    import kaggle_pipeline as kp

    t0 = time.time()
    try:
        print("[Phase 2] Loading Gemma 4 duplex system...")
        sys.stdout.flush()
        duplex_model, tokenizer = kp.load_gemma4_duplex_system()
        if duplex_model is None:
            raise RuntimeError(
                "Could not initialize Gemma4FullDuplexModel. Check VRAM and dependencies."
            )

        print("[Phase 2] Starting training pipeline...")
        sys.stdout.flush()
        kp.run_training_pipeline(duplex_model, num_epochs=args.epochs)
        elapsed = time.time() - t0
        print(f"[Phase 2] SUCCESS: Training completed in {elapsed:.2f}s")
        return 0
    except Exception as e:
        print(f"[Phase 2] ERROR: Training pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_phase_eval(args: argparse.Namespace) -> int:
    """Execute evaluation and sample synthesis on validation split."""
    print_banner("Phase 2: Evaluation and Synthesis")
    print_gpu_telemetry(args.device)

    import kaggle_pipeline as kp

    t0 = time.time()
    try:
        print("[Phase 2 - Eval] Initializing system...")
        duplex_model, tokenizer = kp.load_gemma4_duplex_system()
        if duplex_model is None:
            raise RuntimeError("Could not load Gemma 4 model for evaluation.")

        ckpt_dir = paths.resolve_path(args.checkpoint)
        if os.path.isdir(ckpt_dir):
            print(f"[Phase 2 - Eval] Loading checkpoint from {ckpt_dir}...")
            kp.load_gemma4_duplex_checkpoint(duplex_model, ckpt_dir)
        else:
            print(
                f"[Phase 2 - Eval] Warning: Checkpoint dir '{ckpt_dir}' not found. Evaluating base weights."
            )

        parquet_cache = os.path.join(
            paths.get_data_dir(),
            f"train_tokens_{kp.DUPLEXCONV_MAX_HOURS}h_{kp.MDC_MAX_SAMPLES}s.parquet",
        )
        print(f"[Phase 2 - Eval] Loading dataset tokens from {parquet_cache}...")
        user_mel, agent_mel, agent_codes, text_by_window = kp.load_tokens_from_parquet(
            parquet_cache
        )

        dataset = kp.FullDuplexDataset(
            user_mel,
            agent_mel,
            agent_codes,
            seq_len=64,
            text_by_window=text_by_window,
        )

        print(f"[Phase 2 - Eval] Generating audio synthesis examples in {paths.get_samples_dir()}...")
        kp.synthesize_examples(duplex_model, dataset)

        elapsed = time.time() - t0
        print(f"[Phase 2 - Eval] SUCCESS: Evaluation completed in {elapsed:.2f}s")
        return 0
    except Exception as e:
        print(f"[Phase 2 - Eval] ERROR: Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def run_phase_serve(args: argparse.Namespace) -> int:
    """Execute inference server or single-wav demo."""
    print_banner("Inference Serving")

    if args.demo_wav:
        # CLI single wav mode
        input_wav = paths.resolve_path(args.demo_wav)
        output_wav = paths.resolve_path(args.output_wav)
        if not os.path.isfile(input_wav):
            print(f"[Serve] Error: Input wav file not found at '{input_wav}'")
            return 1

        import kaggle_pipeline as kp

        print(f"[Serve] Loading model for demo talk on {input_wav}...")
        duplex_model, _ = kp.load_gemma4_duplex_system()
        ckpt_dir = paths.resolve_path(args.checkpoint)
        if os.path.isdir(ckpt_dir):
            kp.load_gemma4_duplex_checkpoint(duplex_model, ckpt_dir)
        decoder = kp.Code2WavDecoder()

        out_path = kp.demo_talk(
            duplex_model,
            decoder=decoder,
            input_wav=input_wav,
            output_wav=output_wav,
        )
        print(f"[Serve] Response wav generated at: {out_path}")
        return 0

    # REST API Serving mode
    import serve_duplex

    sys.argv = [
        "serve_duplex.py",
        "--checkpoint",
        paths.resolve_path(args.checkpoint),
        "--serve",
        "--port",
        str(args.port),
    ]
    try:
        duplex, decoder = serve_duplex.load_served_model(paths.resolve_path(args.checkpoint))
        app = serve_duplex.build_app(duplex, decoder)
        if app is None:
            print("[Serve] Error: FastAPI/Uvicorn not installed.")
            return 1
        import uvicorn

        print(f"[Serve] Starting FastAPI server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
        return 0
    except Exception as e:
        print(f"[Serve] ERROR: Serving failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified execution wrapper for Gemma 4 Full-Duplex on RTX 3090 (SSH + Docker)."
    )
    parser.add_argument(
        "--phase",
        choices=["prep", "train", "eval", "serve", "all"],
        default="train",
        help="Pipeline phase to execute (default: train)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Enable smoke test mode (reduced batches / samples for fast verification)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/gemma4_duplex_sota",
        help="Checkpoint directory or path (default: checkpoints/gemma4_duplex_sota)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=2.0,
        help="Max hours for DuplexConv preprocessing in Phase 1 (default: 2.0)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=500,
        help="Max samples for MDC Italian dataset in Phase 1 (default: 500)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for FastAPI inference server (default: 8000)",
    )
    parser.add_argument(
        "--demo-wav",
        type=str,
        default=None,
        help="Path to user input wav file for single-turn demo inference",
    )
    parser.add_argument(
        "--output-wav",
        type=str,
        default="samples/agent_response.wav",
        help="Path to save agent response wav (default: samples/agent_response.wav)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory to write execution logs (default: ~/gemma-4-omni/logs)",
    )

    args = parser.parse_args()

    # Environment setup
    os.environ["PIPELINE_ENV"] = "local3090"
    if "cuda" in args.device:
        dev_num = args.device.split(":")[-1] if ":" in args.device else "0"
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_num
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Initialize paths and directories
    dirs = paths.ensure_all_dirs()
    log_dir = args.log_dir or dirs["logs"]
    os.makedirs(log_dir, exist_ok=True)

    # Set up dual logging to stdout and file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "smoke" if args.smoke else "full"
    log_filename = f"run_3090_{args.phase}_{mode_str}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    dual_logger = DualLogger(log_filepath)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    sys.stdout = dual_logger
    sys.stderr = dual_logger

    print(f"[run_3090] Logging initialized: {log_filepath}")
    print(f"[run_3090] Active PIPELINE_ENV: {paths.get_pipeline_env()}")
    print(f"[run_3090] Project Root: {paths.get_project_root()}")
    print(f"[run_3090] Arguments: phase={args.phase}, smoke={args.smoke}, device={args.device}")
    sys.stdout.flush()

    status = 0
    try:
        if args.phase == "prep":
            status = run_phase_prep(args)
        elif args.phase == "train":
            status = run_phase_train(args)
        elif args.phase == "eval":
            status = run_phase_eval(args)
        elif args.phase == "serve":
            status = run_phase_serve(args)
        elif args.phase == "all":
            status = run_phase_prep(args)
            if status == 0:
                status = run_phase_train(args)
            if status == 0:
                status = run_phase_eval(args)
    finally:
        print(f"[run_3090] Execution finished with exit code {status}.")
        sys.stdout.flush()
        dual_logger.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return status


if __name__ == "__main__":
    sys.exit(main())
