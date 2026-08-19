# ADR-008: Strict Precedence of Fresh Local Checkpoints over Static Input Dataset Weights

## Status
Accepted

## Date
2026-08-19

## Context
In multi-stage Kaggle runs executing training (`STAGE2=1`) followed by evaluation (`EVAL_R2=1`), static checkpoints from prior runs are often mounted as read-only Kaggle input datasets (e.g. `/kaggle/input/andrlab/gemma4-ht-adapter-r2/`).
In runs v4/v5, the evaluation routine defaulted to loading the input dataset path first, thereby evaluating older R2 weights rather than the newly trained checkpoint in the local working directory.

## Decision
Establish an explicit checkpoint resolution priority:
1. **Priority 1 (Fresh Local Checkpoint)**: Check for locally written checkpoints produced in the current execution session (`checkpoints/ht_adapter_stage2r2_final.pt` or `checkpoints/ht_adapter_stage2r2_smoke.pt`).
2. **Priority 2 (Mounted Dataset Fallback)**: Fall back to `/kaggle/input/...` paths only when executing in evaluation-only mode (`STAGE2=0`) where no local training occurred.
3. Log the resolved absolute path and checkpoint file size explicitly in the execution output.

## Alternatives Considered
- **Unconditional Dataset Loading**: Evaluates stale weights whenever a dataset is attached.
- **Manual Path Flag Configuration**: Error-prone and fragile when switching between smoke, full, and eval-only runs.

## Consequences
- Evaluation always evaluates the active model weights trained during the current execution.
- Eliminates silent evaluation bugs and false reporting of performance regressions.
