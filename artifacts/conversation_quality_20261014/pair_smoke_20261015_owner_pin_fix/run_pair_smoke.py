from pathlib import Path
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
    "--max-turns", "4",
    "--max-created-responses", "12",
    "--max-duration-s", "90",
    "--manual-seed-bootstrap",
    "--instructions-a", instructions_a,
    "--instructions-b", instructions_b,
    "--output-dir", str(bundle),
]
sys.path.insert(0, "/workspace")
import client.duplex_stream_e2e as harness
print(f"pair client source: {harness.__file__}; manual_seed_bootstrap={'--manual-seed-bootstrap' in sys.argv}; max_turns=4", flush=True)
raise SystemExit(harness.main())
