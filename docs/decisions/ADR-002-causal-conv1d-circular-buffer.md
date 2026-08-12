# ADR-002: 1D Causal Convolution with Circular Buffer for Streaming

## Status
Accepted

## Date
2026-08-05

## Context
Standard 1D convolutions with symmetric padding (e.g. `padding=1` on `kernel_size=3`) cause temporal leakage (non-causal future token leakage) during real-time frame-by-frame streaming generation.

## Decision
Implement a 1D Causal Convolution adapter (`kernel_size=3, padding=0`) with explicit left-padding:
1. In batch training (`forward`): Left-pad input hidden states by 2 frames (`F.pad(h, (2, 0))`).
2. In streaming (`stream_speech`): Maintain a 2-frame `conv_state` circular buffer on device to perform zero-latency causal convolution per step without re-computing past states or invalidating the LLM KV-cache.

## Alternatives Considered
- **Symmetric Conv1d**: Causes future frame leakage during training that cannot be replicated during real-time inference.
- **Linear Layer without Conv**: Lacks temporal smoothing between adjacent 80ms audio frames.

## Consequences
- Guaranteed zero temporal leakage during streaming.
- Maintains past KV-cache intact during simultaneous user audio input and agent audio output.
