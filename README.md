# Gemma 4 Full-Duplex Omni Pipeline

Real-time, dual-channel, synchronous Full-Duplex Speech-to-Speech conversational AI pipeline combining Google's **Gemma 4 E2B** multimodal backbone with Alibaba's **Qwen3-TTS 12Hz** neural audio codec and neural rendering sub-talker in a Thinker-Talker architecture.

---

## Architecture Overview (V9 SOTA + CUDA Graph Accelerated Engine)

The system enables native simultaneous listening, speaking, and turn-taking (barge-in) via a joint Thinker-Talker pipeline with **Prefix KV-Cache Conditioning** and **CUDA Graph Sub-Talker Acceleration**:

```
[User Audio Mel (Ch 0)] ──┐
                          ├──> Gemma 4 E2B Conformer ──> [Style Sequence h_raw] ──┐
[Agent Mel (Self-Listen)] ┘                             [Monologue Text h_tx]   ──┴──> [Deep Causal Transformer V9 (211M)] ──> [Prefix KV-Cache]
                                                                                                                                │
[24kHz Audio] <── [Code2Wav 24kHz] <── [CUDA Graph Sub-Talker (12ms/fr)] <── [Talker Codec Head (CB0)] <── [Qwen3-TTS Talker 1.7B] <──┘
```

- **Perception (Listening & Self-Listening)**: Ingests 100 fps log-mel spectrograms natively through Gemma 4 E2B's 305M USM-Conformer audio encoder.
- **Cognition & Latent Projection**: Gemma 4 E2B generates inner monologue text tokens ($h_{\text{tx}}$) and adapted style sequences ($h_{\text{raw}}$). **Deep Causal Transformer Projector V9** (211.83M parameters, 4 blocks, RoPE, SwiGLU, RMSNorm) projects multimodal representations into a continuous 16-frame prefix KV-cache.
- **Synthesis Engine (Speaking at RTF ~1.0, kernel RTX 3090)**:
  - **Talker 1.7B**: Autoregressively generates semantic token $CB_0$ at $45\text{ms/frame}$.
  - **CUDA Graph Accelerated Sub-Talker** (ADR-030): Generates 15 residual codebooks ($CB_1 \dots CB_{15}$) via 15 fused CUDA graph replays on a StaticCache at **$15.5\text{ms/frame}$** ($6.4\times$ vs eager), achieving **RTF $\approx 1.0$** end-to-end (0.97–1.09 measured, ASR 3/3) with bit-exact acoustic quality.
  - **Custom Duplex Kernel** (ADR-029): `duplex_kernel/` narrow inference engine (DwarfStar-inspired), self-contained (no unsloth/vLLM), loads in ~36s.
- **Next Milestone**: **TurnDetectorHead** (Real-time conversational turn-taking, barge-in / user interruption detection, and dynamic backchanneling).

---

## Calibrated Loss Formulation (Moshi Kyutai, arXiv:2410.00037, Eq. 7)

$$\mathcal{L}_{\text{total}} = \mathbf{1.0} \cdot \mathcal{L}_{\text{cb0}} + \mathbf{0.05} \cdot \mathcal{L}_{\text{res}} + \mathbf{0.20} \cdot \Big(1.0 - \text{CosineSimilarity}(\mathbf{h}_{\text{omni}}, \mathbf{h}_{\text{nat}})\Big) + \mathbf{0.10} \cdot \left(\frac{\|\mathbf{h}_{\text{omni}}\|}{\|\mathbf{h}_{\text{nat}}\| + 10^{-6}} - 1.0\right)^2$$

- **93.5% Gradient Allocation on $CB_0$**: Concentrates optimization on phonetic intelligibility and syllable accuracy.
- **Acoustic Residual Regularization ($\lambda_r = 0.05$)**: Smoothly anchors Sub-Talker formants without gradient distraction.
- **Manifold & Norm Alignment**: Continuous angular pull ($\cos \approx 0.60 - 0.65$) and Euclidean norm lock to $1.0000$.

---

## Repository Structure

```
gemma-4-omni/
|-- README.md                         # Main repository documentation and quickstart
|-- AGENTS.md                         # Engineering guidelines, DDD/SDD/TDD protocols, and Kaggle/3090 rules
|-- NEXT_STEP.md                      # Live project status, experimental verdicts, and technical roadmap
|-- gemma4_training_report.md         # Comprehensive diagnostic training report and run comparison table
|-- kaggle_pipeline.py                # Standalone 2-Phase Kaggle execution pipeline (tokenization & training)
|-- qwen_subtalker.py                 # Pure PyTorch implementation of the Qwen3-TTS sub-talker code predictor
|-- code2wav_decoder.py               # Standalone bit-exact PyTorch decoder for Qwen3-TTS 24kHz audio synthesis
|-- affective_steering_module.py      # ADR-015 Affective Prosody & Emotional Steering Vectors
|-- phase2_support.py                 # Dataset structures, batch collation, and evaluation helper routines
|-- serve_duplex.py                   # Local serving interface (CLI single-file demo, REST API, Gradio UI)
|-- docs/                             # Architecture documentation, ADRs, and academic sources
|   |-- ADRS.md                       # Master index of Architecture Decision Records (ADR-001 through ADR-017)
|   |-- ARCHITECTURE_STATE.md         # Live state of canonical models, metrics, and empirical verdicts
|   |-- decisions/                    # Formal ADR markdown records
|   |-- sources/                      # Reference papers, extracted texts, and official model source snapshots
|-- tools/                            # Operational tools, metric evaluation, and diagnostic tests
|   |-- train_loop_coupling_v8.py     # Active Trainer V8.5 (Dense Causal 3D MRoPE Parallel Engine)
|   |-- run_train_v8.sh               # 3090 Docker training launcher script
|   |-- test_v8_teacher_forcing_audio.py # Audio synthesis and local faster-whisper transcription tester
|   |-- monitor_v8_and_notify.py      # Background training watcher and email notification bridge
|   |-- find_layer0_exact_op.py       # Bit-exact proof of 3D MRoPE causal parallel forward
|   |-- diagnostics_manifold_oracle.py # SVD spectral rank, cosine similarity, and norm ratio probe
|   |-- train_turn_dynamics.py        # ADR-014 Full-Duplex Turn Dynamics training loop
|   -- tests/                        # Unit tests (causal convolution, sampling routines, wrapper fidelity)
-- _archive/                         # Historical logs, exploratory scripts, and legacy artifacts
```

---

## Local Execution on RTX 3090 (Docker)

```bash
# 1. Smoke test validation
./docker_run_3090.sh smoke

# 2. V8.5 Training (Dense Causal 3D MRoPE with Moshi Loss Weighting)
docker exec -it v8_parallel_training /bin/bash /workspace/tools/run_train_v8.sh

# 3. Audio synthesis & Whisper transcription evaluation
docker run --rm --gpus all -v ~/gemma-4-omni:/workspace --ipc=host --user 1000:1000 gemma4-omni-v5:latest python tools/test_v8_teacher_forcing_audio.py

# 4. REST API Serving (FastAPI :8000)
./docker_run_3090.sh serve
```

---

## Architecture Decision Records (ADRs)

Key architectural decisions are indexed in [docs/ADRS.md](file:///c:/Users/andre/Documents/gemma-4-omni/docs/ADRS.md):

- [ADR-001: Multi-Token Prediction with Qwen3-TTS Code2Wav](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-001-mtp-qwen3-tts.md) *(Superseded by ADR-012)*
- [ADR-002: 1D Causal Convolution with Identity-Init and Circular Buffer](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-002-causal-conv1d-circular-buffer.md)
- [ADR-003: Discoupled 2-Phase Kaggle Pipeline Architecture](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-003-discoupled-2phase-kaggle-pipeline.md)
- [ADR-004: Option A: Native Gemma E2B Conformer Input + Qwen3-TTS Output](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-004-option-a-gemma4-e2b-native-input.md)
- [ADR-005: Qwen3-TTS Teacher Selection for Sub-Talker Coupling](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-005-qwen3-tts-teacher-base-variant.md) *(Superseded by ADR-012)*
- [ADR-006: CPU Execution of Qwen3-TTS Teacher during Evaluation](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-006-eval-teacher-cpu-vram-isolation.md)
- [ADR-007: Reuse of Gemma 4 E2B Base Instance in Evaluation](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-007-reuse-gemma-instance-unsloth-eval.md)
- [ADR-008: Strict Precedence of Fresh Local Checkpoints](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-008-fresh-adapter-checkpoint-precedence.md)
- [ADR-009: Conditional Runtime Monkey-Patching of Sub-Talker Acoustic Sampling](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-009-runtime-monkey-patch-subtalker-topk.md)
- [ADR-010: Recognition of Informational Asymptote and QLoRA](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-010-informational-asymptote-qlora-coadaptation.md) *(Superseded by ADR-012)*
- [ADR-011: Unified Environment Gemma + Qwen3-TTS in Single Process](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-011-unified-env-gemma-qwen3tts.md)
- [ADR-012: TextAnchoredProjection (Strada B / Thinker-Talker Joint Conditioning)](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-012-text-anchored-projection-v6.md)
- [ADR-013: Post-Training Diagnostic Protocol — Manifold Alignment & SLERP Oracle Probes](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-013-post-train-manifold-oracle-diagnostics.md)
- [ADR-014: Decoupled Full-Duplex Conversational Dynamics Module](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-014-decoupled-full-duplex-dynamics.md)
- [ADR-015: Affective Prosody & Emotional Steering Vectors for Talker Conditioning](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-015-affective-prosody-steering-vectors.md)
- [ADR-016: Dense Causal 3D MRoPE Parallel Engine for Talker Conditioning (V8)](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-016-dense-causal-3d-mrope-parallel-engine.md)
- [ADR-028: Official vLLM-Omni Native Docker Serving Runtime](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-028-vllm-omni-native-docker-serving.md)
- [ADR-029: Custom Optimized Duplex Kernel (DwarfStar-Inspired)](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-029-custom-duplex-kernel.md)
- [ADR-030: CUDA Graph Sub-Talker nel Duplex Kernel — Ricostruzione e Gotcha](file:///c:/Users/andre/Documents/gemma-4-omni/docs/decisions/ADR-030-cuda-graph-subtalker-kernel.md)
