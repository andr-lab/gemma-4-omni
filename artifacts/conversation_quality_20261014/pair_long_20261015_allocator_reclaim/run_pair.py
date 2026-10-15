from pathlib import Path
import runpy
import sys

bundle = Path(__file__).resolve().parent
instructions_a = (bundle / "instructions_a.txt").read_text(encoding="utf-8-sig")
instructions_b = (bundle / "instructions_b.txt").read_text(encoding="utf-8-sig")
sys.argv = [
    "client.duplex_stream_e2e",
    "--mode", "pair",
    "--turn-policy", "gemma_native_control6",
    "--ws-url", "ws://127.0.0.1:8000/v1/realtime?duplex=1",
    "--audio", "/workspace/artifacts/conversation_quality_20261014/inputs/pair_seed_it.wav",
    "--max-turns", "16",
    "--max-created-responses", "64",
    "--max-duration-s", "240",
    "--instructions-a", instructions_a,
    "--instructions-b", instructions_b,
    "--output-dir", str(bundle),
]
runpy.run_module("client.duplex_stream_e2e", run_name="__main__")
