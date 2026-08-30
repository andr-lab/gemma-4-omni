# Kernel gates

Ogni fase produce un raw trace e un report JSON sotto:

```text
artifacts/kernel_gates/<git_sha>/<phase>/<run_id>/
```

## Source attestation

Prima di eseguire un gate il checkout deve essere pulito. L'attestation
content-addressed lega i byte del worktree agli oggetti Git del commit:

```bash
python tools/kernel_gates/source_attestation.py \
  --repo . --output /tmp/gemma4_source_attestation.json
```

Il verifier ricontrolla indipendentemente `git_sha`, `git_dirty`, SHA-256 e Git
blob ID di ogni file attestato.

## P0

```bash
python tools/kernel_gates/run_phase.py --phase P0 \
  --manifest tools/kernel_gates/manifests/p0_inputs.json --workspace /workspace \
  --source-attestation /tmp/gemma4_source_attestation.json --run-id run1 \
  --git-sha <git_sha> --git-dirty false
python tools/kernel_gates/run_phase.py --phase P0 \
  --manifest tools/kernel_gates/manifests/p0_inputs.json --workspace /workspace \
  --source-attestation /tmp/gemma4_source_attestation.json --run-id run2 \
  --git-sha <git_sha> --git-dirty false
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P0/run1/P0_report.json \
  --compare artifacts/kernel_gates/<git_sha>/P0/run2/P0_report.json
```

`run_phase.py` esegue i contract test CPU e il benchmark con il modello reale;
non accetta `FakeEngine`. Il verifier ricalcola dal raw JSONL i conteggi di
successo, NaN, Inf, audio vuoto ed eccezioni e verifica gli hash.

## P1

```bash
python tools/kernel_gates/run_phase.py --phase P1 --workspace . \
  --source-attestation /tmp/gemma4_source_attestation.json --run-id run1 \
  --git-sha <git_sha> --git-dirty false
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P1/run1/P1_report.json
```

P1 scrive eventi elementari nel raw trace: 10.000 sequenze FSM, 1.000 race
integrate A→invalidate→B, tentativi stale per audio/testo/KV/history/final
state, dipendenze e contratto PCM→mel con `Gemma4AudioFeatureExtractor`.
Il verifier ricomputa le metriche dal trace senza fidarsi del summary.
