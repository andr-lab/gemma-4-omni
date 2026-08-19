# ADR-009: Conditional Runtime Monkey-Patching of Sub-Talker Acoustic Sampling

## Status
Accepted

## Date
2026-08-19

## Context
Kaggle execution environments rely on pre-packaged input datasets (e.g. `andrlab/gemma-4-omni-module`, ~5.45GB) containing static snapshots of supporting modules including `qwen_subtalker.py`.
In run v6, audio synthesis crashed with `TypeError: sample_acoustic_codes() got an unexpected keyword argument 'top_k'` because the mounted snapshot contained an earlier version of the function without `top_k`, `top_p`, `seed`, and `temperatures_per_level` parameters.
Re-uploading and re-indexing the entire 5.45GB dataset for a minor python function update creates significant workflow friction and delay.

## Decision
1. Implement a conditional runtime monkey-patch at notebook cell startup.
2. Use `inspect.signature` to check whether `qwen_subtalker.sample_acoustic_codes` accepts `top_k`.
3. If `top_k` is missing, dynamically monkey-patch `qwen_subtalker.sample_acoustic_codes` with the modern implementation supporting tiered temperatures, seed control, and top-k filtering.
4. If already modern, execute as a no-op.
5. Validate via unit tests (TDD).

## Alternatives Considered
- **Forced Re-Upload of 5.45GB Dataset**: Heavy overhead and bandwidth delay on every minor sampling refactor.
- **Legacy Fallback Sampling**: Reverts to suboptimal greedy sampling which causes acoustic buzz and artifacts.

## Consequences
- Immediate runtime compatibility across diverse Kaggle dataset versions.
- Enables advanced sampling techniques (`tdecay`, per-level temperature schedules) without dataset rebuilds.
- Clean execution validated in run v7 (72 synth wavs generated successfully).
