# ADR-031: Continuous Gemma Audio Listener with Delayed Self-Reference

## Status
Accepted for P3 implementation; P4 incremental V9 remains separate.

## Decision

The realtime scheduler drives one `GemmaListener.step()` for every complete
16 kHz, 80 ms PCM frame. The listener uses the native Transformers 5 Gemma
`get_audio_features()` audio tower directly; ASR, tokenizer decode, Thinker
text generation, V9, and Qwen are not prerequisites for this path.

`GemmaListenerState` owns a bounded causal rolling context, accounting
counters, the monotonic listener-state version, and the last hidden output.
`AssistantReferenceBuffer` stores only audio that has already been sent to the
client. Its default visibility rule is exact-frame `t - 1`: a missing frame is
silence/absence, never a substitute from a later frame. Emitted frame IDs are
anchored to the latest listener-timeline frame consumed, so a response started
mid-stream cannot accidentally expose its frame 0 to microphone frame 1.
Assistant output is normalized from Code2Wav's 24 kHz to listener 16 kHz before
storage.

The listener returns the final valid Gemma audio-tower embedding (1536D in the
canonical E2B checkpoint). It preserves that native dimension; the existing
TurnHead v4 adapter ignores this embedding until P4 supplies the trained V9
2048D representation.

## Consequences

- Microphone accounting and listener execution are independent of request /
  response lifetime and continue while the speaker is active.
- Self-reference is causally timestamped and thread-safe, with no future-frame
  substitution on gaps.
- The rolling window bounds compute and memory, but P4 must replace the window
  recomputation with a truly incremental V9/listener state if profiling shows
  it is necessary.
- A deterministic backend remains available for CPU contract tests only. P3
  closure requires the real `GemmaAudioTowerBackend` replay and an independent
  raw-trace verifier.
