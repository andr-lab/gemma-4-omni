#!/usr/bin/env bash
set -euo pipefail

# Canonical two-peer E2E runner. Invoke from the serving container with a
# unique label, for example: bash /workspace/scripts/run_coherent_native_pair_e2e.sh run8
readonly RUN_LABEL="${1:?provide a unique run label}"
[[ "$RUN_LABEL" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid run label" >&2; exit 2; }
cd /workspace
export PYTHONPATH="/workspace${PYTHONPATH:+:$PYTHONPATH}"

exec /workspace/.venv_e2e_runtime/bin/python client/duplex_stream_e2e.py \
  --mode pair \
  --turn-policy gemma_native_control6 \
  --ws-url 'ws://127.0.0.1:8000/v1/realtime?duplex=1' \
  --audio /workspace/samples/canonical_live/sample_leonardo_live_perfect.wav \
  --duration-s 7.44 \
  --instructions-a 'Ti interessa Leonardo come punto d’incontro tra arte e osservazione della natura. Condividi ciò che sai, distingui i fatti documentati dalle leggende e ragiona con curiosità sui suoi taccuini e sulle opere.' \
  --instructions-b 'Sei curioso ma scettico sulle storie più ripetute su Leonardo. Porta esempi verificabili di pittura, anatomia o ingegneria, chiedi chiarimenti se servono e non presentare ipotesi come fatti.' \
  --max-turns 6 \
  --max-created-responses 12 \
  --max-duration-s 180 \
  --voice-conditioning-mode x_vector_only \
  --cpu-audio-history-archive \
  --output-dir "/workspace/.tmp/coherent_leonardo_native_control6_${RUN_LABEL}"
