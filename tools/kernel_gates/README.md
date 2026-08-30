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

## P2

```bash
python tools/kernel_gates/run_phase.py --phase P2 --workspace . \
  --source-attestation /tmp/gemma4_p2_source_attestation.json --run-id run1 \
  --git-sha <git_sha> --git-dirty false --real-duration-s 60
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P2/run1/P2_report.json
```

P2 verifica 125 tick virtuali in 10 s, 750 tick virtuali in 60 s, ingress
PCM lossless da 1280 sample, input durante speaker e 60 s di real clock.
`P2_ACTIVE_SHA` identifica il bundle pubblico corrente; il verifier ricalcola
tick, trasporto, jitter e deadline dal raw trace.

## P3

P3 usa il solo audio tower nativo di Gemma (non ASR e non testo intermedio).
Il gate reale va eseguito sull'host GPU con almeno 10.000 frame da 1280 sample;
la prima porzione copre almeno cinque minuti logici e il replay mantiene un
riferimento assistant esattamente a un frame di distanza:

```bash
python tools/kernel_gates/run_phase.py --phase P3 --workspace . \
  --source-attestation /tmp/gemma4_p3_source_attestation.json \
  --run-id run1 --git-sha <git_sha> --git-dirty false --listener-frames 10000
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P3/run1/P3_report.json
```

Il verifier ricalcola dal raw trace accounting PCM, coverage durante speaking,
hidden health, causalità della reference e p95 della step. L'evidence pubblica
va in `validation/kernel_gates/<git_sha>/P3/` solo dopo il PASS indipendente.

## P4

P4 usa il checkpoint V9 reale in BF16 e FP32 e la validation cache reale sul
3090. Verifica equivalenza batch/step su almeno 100 sequenze, future-prefix in
FP32, 10.000 step GPU con p95 <= 10 ms e causal-prefix ablation Qwen CB0 su
almeno 5.000 frame. Il trace contiene eventi per frame; il verifier ricalcola
MAE, cosine, future-prefix, percentili, CE, top-1/top-5, KL e quartili.

```bash
python tools/kernel_gates/run_phase.py --phase P4 --workspace /workspace \
  --source-attestation /workspace/p4_source_attestation.json \
  --run-id run1 --git-sha <git_sha> --git-dirty false \
  --v9-cache /workspace/cache_ars_full_v2.pt \
  --codes /workspace/data/train_tokens_cv26_it_40.8h_31916s_codes.parquet \
  --codes-metadata /workspace/data/train_tokens_cv26_it_40.8h_31916s_hyperparams.json
python tools/kernel_gates/verify_report.py \
  artifacts/kernel_gates/<git_sha>/P4/run1/P4_report.json
```

P4-T4 è obbligatorio: se l'ablation fallisce, QLoRA resta NO-GO. L'evidence
pubblica va in `validation/kernel_gates/<git_sha>/P4/` solo dopo il PASS
indipendente.
