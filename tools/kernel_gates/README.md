# Kernel gates

Ogni fase produce un raw trace e un report JSON sotto:

```text
artifacts/kernel_gates/<git_sha>/<phase>/<run_id>/
```

## P0

```bash
python tools/kernel_gates/build_manifest.py \
  --input tools/kernel_gates/manifests/p0_inputs.json \
  --output /tmp/p0_inputs.json --phase P0
python tools/kernel_gates/run_phase.py --phase P0 \
  --manifest /tmp/p0_inputs.json --workspace /workspace --run-id run1
python tools/kernel_gates/run_phase.py --phase P0 \
  --manifest /tmp/p0_inputs.json --workspace /workspace --run-id run2
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P0/run1/P0_report.json \
  --compare artifacts/kernel_gates/<git_sha>/P0/run2/P0_report.json
```

`run_phase.py` esegue i contract test CPU e il benchmark con il modello reale;
non accetta `FakeEngine`. Il verifier ricalcola dal raw JSONL i conteggi di
successo, NaN, Inf, audio vuoto ed eccezioni e verifica gli hash.
