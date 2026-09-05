<USER_REQUEST>
segui direttamente questo piano corretto:

# Implementation Plan: P4-NATIVE — DS4-Style Native Gemma 4 Listener Executor

## 1. Goal

Replace the current generic / shape-stable PyTorch Gemma listener LM execution path with a narrow, model-specific, deterministic native GPU executor inspired by the design principles of `antirez/ds4`:

* model-specific rather than generic;
* device-resident activations and KV state;
* fixed memory ownership;
* minimal host API;
* CUDA graph islands where execution topology is stable;
* correctness before speed.

Current approximate steady-state budget on RTX 3090:

```text
Gemma listener total        ~64.2 ms
  audio frontend+tower       ~9.1 ms
  Gemma LM                  ~55.1 ms

V9                            ~2.7 ms
Speaker + Code2Wav           ~48.4 ms

Current total               ~115.5 ms
```

A compiled experimental Gemma LM has already demonstrated approximately:

```text
Gemma LM ~13.2 ms
```

but does not satisfy canonical BF16 parity.

Therefore the main objective is:

> reproduce the canonical listener computation with native fixed-state execution at approximately the performance demonstrated by the invalid-fast compiled path.

### Performance gates

Hard P4 gate:

```text
semantic tick p95 <= 80 ms
deadline miss rate <= 1%
speaker throughput >= 12.5 fps
final ingress backlog == 0
```

Engineering targets with margin:

```text
Gemma native LM p95       <= 15 ms preferred
                         <= 18 ms intermediate

full listener p95         <= 24 ms preferred

V9 p95                     <= 3 ms

speaker + Code2Wav p95    <= 49 ms

full semantic tick p95    <= 75 ms preferred
                         <= 80 ms mandatory
```

Note the arithmetic:

```text
9.1 frontend/tower
+ 15 native LM
+ 2.7 V9
+ 48.4 speaker
≈ 75.2 ms
```

Therefore an 18 ms LM is a useful intermediate result, but roughly 14–15 ms is the preferred target for comfortable P4 headroom.

---

# 2. Non-Negotiable Architectural Invariants

1. Models remain frozen:

   * Gemma 4 E2B
   * causal V9
   * Qwen3-TTS
   * SubTalker
   * Code2Wav

2. No retraining and no change to learned tensors.

3. V9 remains:

```text
L = 0
```

4. Assistant self-reference remains exactly:

```text
assistant_pcm[t] -> listener[t+1]
delay_frames = 1
```

5. Semantic listener cadence remains:

```text
80 ms
12.5 Hz
```

6. No microphone frame may be skipped, merged, semantically deferred, or processed after the assistant finishes speaking.

7. No future conditioning or future self-reference.

8. Production hot LM path must cross the Python/native boundary exactly once per listener frame:

```text
Python
  ↓
gemma4_listener_step_async(...)
  ↓
native executor
  ↓
Python
```

There must be no Python loop over Gemma decoder layers.

9. Steady-state native execution must perform:

```text
cudaMalloc = 0
cudaFree = 0
CPU tensor readback = 0
.item() = 0
device synchronize = 0
```

10. The native call must be asynchronous with respect to the CPU and enqueue work onto the supplied CUDA stream.

---

# 3. Correct Canonical LM Workload

The current canonical representation has:

```text
TEXT_BLOCK_LEN = 32
AUDIO_FEATURES_PER_FRAME = 4
AUDIO_START_AFTER_TEXT = 34
```

The multimodal audio tower receives both:

```text
user mel
assistant-reference mel
```

in canonical block-interleaved form before producing Gemma audio soft tokens.

Therefore the final Gemma language model advances by:

```text
Q = 4 soft tokens / 80 ms frame
```

NOT Q=8.

The native steady-state execution path must therefore specialize for:

```text
batch = 1
q_len = 4
hidden_size = checkpoint value
dtype = BF16
```

Q=8 is not a P4 native-listener certification requirement unless the production architecture later introduces a real Q=8 LM call.

---

# 4. Phase 0 — Attest the Actual Checkpoint Topology

Do not hardcode architectural assumptions from notes.

Before implementing native kernels, create:

```text
tools/kernel_gates/export_gemma4_native_topology.py
```

Output:

```text
validation/kernel_gates/P4_NATIVE/
GEMMA4_NATIVE_TOPOLOGY.json
```

The exporter must inspect the exact loaded checkpoint and record:

```text
checkpoint hash
transformers version

hidden_size
intermediate_size
num_hidden_layers

max_position_embeddings

hidden_size_per_layer_input

MLP activation type
PLE activation type

layer_scalar semantics

for every decoder layer:
    index
    layer_type

    num_attention_heads
    num_key_value_heads
    head_dim

    sliding_window

    attention scaling

    RoPE implementation
    RoPE base
    RoPE scaling parameters

    is_kv_shared_layer
    kv_shared_layer_index
    store_full_length_kv

    q_proj shape
    k_proj shape or null
    v_proj shape or null
    o_proj shape

    gate_proj shape
    up_proj shape
    down_proj shape

    normalization eps values
```

The topology listed in the current engineering notes should be treated as the expected topology, not as the source of truth.

If the live checkpoint differs from the expected topology:

```text
FAIL CLOSED
```

Do not silently adapt a partially incorrect hardcoded executor.

---

# 5. Generated Native Topology

Generate a versioned header from the attested JSON:

```text
native/gemma4_listener/generated_topology.h
```

Example:

```c
static const GemmaLayerSpec gemma_layers[NUM_LAYERS] = {
    ...
};
```

Embed into the library:

```text
topology_manifest_sha256
checkpoint_sha256
ABI_VERSION
```

At context creation, the Python adapter verifies that the native library topology hash matches the loaded checkpoint topology manifest.

Mismatch:

```text
native listener initialization FAIL
```

---

# 6. Native Executor Lifecycle

Do not make `gemma4_listener_step()` responsible for every lifecycle state.

Use explicit phases:

```text
create
  ↓
bind weights
  ↓
prime canonical text/BOA prefix
  ↓
step Q=4
  ↓
step Q=4
  ↓
...
  ↓
reset/reprime only when canonical listener state requires it
```

C ABI:

```c
typedef struct gemma4_listener_ctx gemma4_listener_ctx;

int gemma4_listener_create(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx
);

int gemma4_listener_prime_async(
    gemma4_listener_ctx *ctx,
    const void *prefix_embeds,
    const void *prefix_ple,
    uint32_t prefix_len,
    cudaStream_t stream
);

int gemma4_listener_step_async(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    cudaStream_t stream
);

const void *gemma4_listener_output_ptr(
    gemma4_listener_ctx *ctx
);

int gemma4_listener_reset(
    gemma4_listener_ctx *ctx
);

void gemma4_listener_destroy(
    gemma4_listener_ctx *ctx
);
```

Steady-state `step_async()` must own:

```text
position advance
KV update
shared-KV update
mask/window bounds
RoPE positions
decoder execution
final hidden selection
```

Python must not build per-frame `position_ids`, masks, caches, or per-layer objects.

---

# 7. Weight Ownership

## Phase A rule: zero weight duplication

Initially use the existing PyTorch parameter storage directly.

For every registered tensor record:

```text
CUDA pointer
dtype
shape
stride
storage offset
device id
alignment
semantic tensor id
```

Python must retain strong references to every tensor while the native context exists.

After native context creation, the Gemma model must not be:

```text
.to(...)
reloaded
repacked
freed
device-migrated
```

without invalidating the context.

Required initial metrics:

```text
duplicated_weight_bytes = 0
native_trainable_parameters = 0
```

## Optional Phase B

If profiling proves that a specific weight layout prevents the native executor from reaching the performance target, a derived packed weight cache may be evaluated separately.

This is NOT retraining and must not alter learned values.

However it requires an explicit benchmark showing:

```text
meaningful latency win
numerical equivalence
VRAM budget acceptable
```

Do not introduce packed mirrors speculatively.

---

# 8. Fixed-Address Runtime Arena

Allocate all persistent native scratch at context creation.

Examples:

```text
frame input staging
hidden state A
hidden state B

normalized hidden

Q
K
V

attention output

gate
up
activation product
FF output

PLE temporary

position scalar

KV metadata

native output hidden

CUDA graph workspaces
cuBLASLt workspace
```

No allocations in steady state.

---

# 9. Dynamic Input with Fixed CUDA Graph Addresses

The PyTorch audio-tower output pointer is not guaranteed to remain identical every frame.

CUDA Graph nodes require stable addresses.

Therefore:

```text
dynamic PyTorch frame_audio pointer
        ↓
native cudaMemcpyAsync
        ↓
fixed native input staging buffer
        ↓
CUDA graph islands
```

This D2D copy occurs inside the single native call.

Do not require the caller's frame tensor to have a permanent address.

The output should live in a stable registered CUDA buffer.

Python may expose this persistent output as a Tensor once at initialization and reuse the same tensor every tick.

---

# 10. Fixed DynamicCache-Equivalent State

Do NOT reproduce Transformers `StaticCache` semantics blindly.

Implement the logical semantics of the currently certified DynamicCache using stable native memory.

The state must track:

```text
absolute token position
frames processed
valid KV length

per-layer KV source state

shared-KV source ownership
shared-KV consumer mapping

sliding attention logical window
global attention history

full-length KV storage where required
```

Shared-KV consumers must alias the correct source storage logically.

A sliding attention source that also uses:

```text
store_full_length_kv = true
```

must preserve full source history for later shared-KV consumers while exposing only the correct sliding window to its own attention computation.

Do not conflate:

```text
stored KV history
```

with:

```text
attention-visible KV window
```

---

# 11. 64-Frame Context Is NOT the LM KV Limit

The current listener has a rolling 64-frame audio context.

That does NOT mean the LM KV cache is reset every 64 frames.

The continuous LM currently advances persistently across frame 64.

Therefore native certification must explicitly cross:

```text
frame 63
frame 64
frame 65
```

without resetting LM state.

Do not implement an artificial 64-frame LM rollover.

---

# 12. Session KV Capacity

Derive maximum legal LM positions from the actual Gemma config:

```text
max_position_embeddings
```

Compute:

```text
max_audio_frames =
floor((max_position_embeddings - canonical_prefix_length) / 4)
```

Native allocation must support the required production session horizon without reallocation.

For sliding layers:

```text
fixed ring buffer / bounded window
```

is permitted only if mathematically identical to the canonical model.

For global/full-history source layers:

```text
preserve full legal history
```

up to the attested maximum context.

Do not silently truncate global attention.

A preferred 1800-second soak is only valid if it fits inside the checkpoint's legal positional context.

If it does not, do not invent a new reset policy merely to run the soak.

---

# 13. Phase 1 — Correct Tensor-by-Tensor Bisect

Rewrite:

```text
tools/kernel_gates/p4_listener_cuda_graph_bisect.py
```

The current diagnostic must become a real oracle/candidate bisect over actual listener calls.

Do not recreate an approximate decoder equation inside the test.

Instrument the actual canonical:

```text
ShapeStableGemmaListenerLM
```

and the candidate.

Required frames:

```text
0
1
2

20
21
22
23
24
25
26
27
28

63
64
65

selected long-context frames
```

At each layer compare the first divergence after:

```text
input hidden

input normalization

Q projection
K projection if present
V projection if present

Q norm
K norm if present
V norm if present

Q RoPE
K RoPE if present

KV write
shared-KV state

attention mask/window metadata
attention output

O projection
post-attention norm
attention residual

pre-FF norm

gate projection
up projection
activation
gate*up
down projection

post-FF norm
FF residual

PLE gate
PLE activation
PLE multiplication
PLE projection
PLE norm
PLE residual

layer scalar
layer output

final norm
final hidden
```

The diagnostic must print/store:

```text
first divergent frame
first divergent layer
first divergent operation

MAE
max abs
cosine
```

---

# 14. Do Not Hardcode SwiGLU / SiLU

The native executor must use the exact activation functions reported by the live modules/config.

The current Python oracle uses:

```text
mlp.act_fn(...)
layer.act_fn(...)
```

Therefore the generated topology must contain activation enums.

Example:

```c
GEMMA_ACT_GELU_TANH
GEMMA_ACT_SILU
...
```

Implement only the activation actually used by the attested checkpoint.

Unexpected activation:

```text
FAIL CLOSED
```

---

# 15. RoPE

Likewise, do not hardcode guessed local/global RoPE parameters.

The topology exporter must attest them.

For Phase A, prefer minimizing numerical risk.

One valid strategy is to create canonical precomputed RoPE tables for the legal native context range using the same checkpoint/config semantics and make them persistent device buffers.

This avoids introducing a different trigonometric implementation inside the native hot path.

Later, if memory or startup time requires it, native RoPE generation may replace the tables after parity certification.

---

# 16. Phase 2 — Native Runtime Skeleton Before Custom Math

Create:

```text
native/gemma4_listener/
    gemma4_listener.h
    gemma4_listener.cu
    gemma4_listener_cache.cu
    gemma4_listener_graph.cu
    gemma4_listener_kernels.cuh
    gemma4_listener_test.cu
    generated_topology.h
    Makefile
```

Initial objective:

> reproduce canonical arithmetic and cache state before optimizing individual mathematical operations.

Do not begin by simultaneously changing:

```text
cache implementation
RMSNorm math
attention implementation
GEMM algorithm
activation implementation
```

Otherwise a divergence cannot be localized.

---

# 17. Linear Layers

The production matrices are fixed-shape BF16 small-M operations.

Benchmark exact production dimensions with:

```text
current PyTorch baseline
direct cuBLAS/cuBLASLt
CUTLASS candidate if justified
Triton prototype if useful
```

For each matrix shape report:

```text
M
N
K

latency
algorithm id
workspace
achieved bandwidth

MAE vs canonical
max abs
cosine
```

Pin the selected algorithm at context creation.

Do not perform cuBLASLt heuristic search inside the hot loop.

---

# 18. Attention Requires Special Care

Do NOT blindly replace the canonical SDPA path.

First determine, with profiling, which CUDA attention implementation the canonical RTX 3090 execution actually uses for:

```text
Q = 4
current KV length
sliding layers
global layers
GQA topology
```

Record:

```text
kernel/backend identity
Q/K/V layout
scaling
mask semantics
window semantics
```

The first native implementation should reproduce those semantics as closely as possible.

If a new native attention kernel introduces unacceptable drift:

```text
REJECT IT
```

Do not retain a faster path with unexplained attention divergence.

---

# 19. Phase 3 — CUDA Graph Islands

Do not attempt one monolithic 35-layer CUDA kernel.

Use stable graph islands.

Potential units:

```text
attention projection island
attention post-processing island
FFN island
PLE island
whole layer where practical
```

The native graph layer should follow DS4-style semantics:

```text
captured graph available
    → replay

first stable execution
    → capture + execute

unsupported state
    → exact native fallback
```

Graph topology must not depend on absolute frame id.

Dynamic runtime values such as:

```text
absolute position
valid KV length
window start
```

should live in fixed device-side metadata read by kernels.

---

# 20. Phase 4 — Custom Pointwise / Fused Kernels

Only after cache and graph replay are numerically correct should custom fusion begin.

Candidate operations:

### RMSNorm

Replicate the canonical formula and cast points.

Do NOT compile with approximate fast-math transformations unless independently certified.

### Activation × Up Projection Result

Fuse only the pointwise:

```text
activation(gate) * up
```

using the activation exported by the checkpoint.

### Residual / Norm

Candidate:

```text
residual add
+ exact normalization
```

only if end-to-end parity passes.

### Q/K normalization + RoPE + KV store

Potentially valuable because it removes intermediate traffic.

Again, promote only after canonical parity.

---

# 21. Compiler Rules

Production build target:

```text
RTX 3090
SM 8.6
```

Build inside the SAME CUDA/Docker environment used for runtime certification.

Do not compile with a random host NVCC and then load the library into a different CUDA userspace.

Example:

```text
nvcc
-O3
-shared
-Xcompiler -fPIC
-gencode arch=compute_86,code=sm_86
-gencode arch=compute_86,code=compute_86
-lineinfo
```

Link required CUDA libraries explicitly.

Do NOT use:

```text
--use_fast_math
```

for the parity-certified build.

---

# 22. Python Adapter

Create:

```text
duplex_kernel/native_listener.py
```

Class:

```python
class NativeGemmaListenerLM:
    ...
```

Responsibilities:

```text
load versioned native .so

validate native ABI version

validate topology hash
validate checkpoint hash

hold strong references to all borrowed weights

register weight pointers once

create native context

prime canonical prefix

pass current CUDA stream handle

invoke one native step per frame

expose persistent output Tensor
```

The native call must use:

```python
torch.cuda.current_stream().cuda_stream
```

or an explicitly managed listener stream with CUDA-event handoff.

Do not make assumptions about CUDA default-stream semantics.

---

# 23. Production Integration

Replace only the final LM backend initially.

Keep:

```text
incremental mel frontend
Gemma native audio tower
canonical interleaving
V9
speaker
```

unchanged.

Target:

```text
mic frame
 ↓
existing audio frontend
 ↓
existing audio tower
 ↓
4 new audio soft tokens
 ↓
NativeGemmaListenerLM.step_async()
 ↓
canonical first-frame hidden
 ↓
V9
```

The canonical V9-facing output remains:

```text
final Gemma LM hidden
at the first audio feature of frame t
before exactly one V9 causal_conv
```

---

# 24. Session Semantics

Normal conversational speaker turns must NOT reset the Gemma listener KV.

The listener remains persistent.

Test separately:

```text
normal continuous session
explicit session reset
text-context invalidation / reprime
shutdown / recreate
```

Do not call `reset()` on every Qwen utterance.

---

# 25. Numerical Certification

Create:

```text
tools/kernel_gates/test_native_listener_parity.py
```

Minimum:

```text
>= 100 independent sequences
>= 5000 total listener frames
```

Required scenarios:

```text
first frame

steady-state Q=4

frame 63/64/65 boundary

shared-KV source/consumer paths

sliding-window boundaries

global-attention long context

explicit reset/reprime

long continuous KV growth
```

No synthetic Q=8 requirement unless production starts using Q=8.

Canonical output gates:

```text
MAE <= 1e-4
cosine min >= 0.9999
future leak <= 1e-5

NaN = 0
Inf = 0
```

Also record:

```text
max_abs_diff
p99_abs_diff
```

even if they are not currently hard gates.

Do not call this "bit-exact" unless tensors are literally bit-identical.

The formal requirement is:

```text
canonical numerical parity
```

under the established tolerance.

---

# 26. Future-Leak Gate

Do not infer causality solely from parity.

For random frame t:

```text
preserve all inputs <= t
randomly perturb frames > t
```

Require:

```text
hidden_t difference <= 1e-5
```

over >=100 sequences.

---

# 27. Long-Context Performance

Native runtime performance must be measured at different KV lengths.

At minimum benchmark near:

```text
frame 1
frame 64
frame 512

mid-session
near the 300-second certification context length
```

If legal under model max context, also benchmark longer session lengths.

Report:

```text
LM p50/p95/p99 by context length
attention duration by context length
DRAM bandwidth
KV memory footprint
```

Do not certify a 15 ms LM measured only with a 384-token StaticCache.

---

# 28. Native Profiling

Generate:

```text
validation/kernel_gates/P4_NATIVE/
P4_NATIVE_PROFILE.json
```

Compare:

```text
current ShapeStable eager
invalid-fast torch.compile candidate
native executor
```

Record:

```text
kernel launches/frame

CUDA graph replay count

cuBLAS/cuBLASLt time

attention time

pointwise time

DRAM throughput

L2 throughput

SM utilization

tensor-core utilization

native LM p50/p95/p99
```

The report should explain where the ~40 ms improvement came from.

---

# 29. Triton Role

Triton is allowed as a development/microbenchmark tool.

Use it to prototype:

```text
RMSNorm
activation fusion
residual fusion
small-M GEMM
RoPE/cache-store fusion
```

Production architecture remains the native model-specific executor.

A Triton implementation should only become part of production if it clearly beats the corresponding CUDA/cuBLAS path and satisfies parity.

Do not rewrite the whole model in Triton.

---

# 30. Stop Condition Before Large Custom-Kernel Work

If the tensor bisect reveals that the ~13 ms path can be made canonical simply by correcting fixed-cache / mask / graph semantics while continuing to replay the existing canonical CUDA kernels:

```text
STOP
```

Do not write unnecessary custom GEMMs or attention kernels.

Closing P4 is the objective.

Writing custom kernels is not itself the objective.

---

# 31. P4 Final Verifier Hardening

Update:

```text
tools/kernel_gates/verify_p4_final.py
```

It must independently verify:

```text
git_sha == exact tested HEAD
git_dirty == false

source_manifest_sha256 valid
trace_sha256 valid

duration >= 300 s

self_reference_delay_frames == 1

for every valid listener frame t >= 1:
    ref_frame_id == t - 1

future_reference_violations == 0
wrong_reference_frames == 0

expected microphone frames
captured microphone frames
listener backend calls
V9 frames
speaker frames

no dropped frames
no duplicates
no out-of-order frames

final ingress backlog == 0

listener p95 <= 80 ms

TOTAL SEMANTIC TICK p95 <= 80 ms
TOTAL deadline miss rate <= 1%

speaker effective throughput >= 12.5 fps
speaker output interval miss rate <= 1%
```

Do not confuse:

```text
speaker compute latency <80 ms
```

with:

```text
speaker actually produces >=12.5 frames/s
```

Both must be measured.

---

# 32. Expected Frame Accounting

For every wall-clock certification run derive the expected frame count from the absolute 80 ms capture clock.

Do not allow a slow system to report success merely because it processed fewer frames.

Report:

```text
expected_capture_frames
actual_captured_frames
actual_listener_frames
missing_listener_frames
final_ingress_backlog
```

The current P4 architecture requires lossless listener processing.

---

# 33. Regression Gates

After native integration rerun on exact final source:

```text
Gemma canonical representation parity
Gemma future leak

V9 T1
V9 T2
V9 T3

P4D L=0 quality
slot-KV parity
speaker causal runtime

Thinker + listener concurrency
speaker + listener concurrency
```

No previous-source PASS may be reused as final evidence after the listener backend changes.

---

# 34. Source Binding

Before certification:

```text
git rev-parse HEAD
git status --porcelain
```

Require:

```text
git_dirty = false
```

Generate a new source manifest from that exact commit.

Native artifacts must include:

```text
native source SHA256
libgemma4_listener.so SHA256
topology manifest SHA256
checkpoint SHA256
```

The verifier must bind all of them.

---

# 35. Test / Build Procedure

## A. Export topology

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/export_gemma4_native_topology.py
'
```

## B. Tensor bisect

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/p4_listener_cuda_graph_bisect.py
'
```

## C. Build native library INSIDE runtime container

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
bash -lc "cd native/gemma4_listener && make clean && make"
'
```

Do not build against an unrelated host CUDA toolkit.

## D. Native primitive unit tests

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
bash -lc "cd native/gemma4_listener && ./gemma4_listener_test"
'
```

The standalone executable tests:

```text
native cache logic
pointwise kernels
graph lifecycle
synthetic GEMM fixtures
error handling
```

It does not certify the full production model because production weights are borrowed from the PyTorch runtime.

## E. Full native parity

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/test_native_listener_parity.py
'
```

## F. 60-second development full-duplex gate

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/p4_speaker_listener_concurrency.py --duration 60
'
```

## G. Commit and source-bind candidate

Only after the development gates pass:

```text
commit
clean checkout
generate manifest
```

## H. 300-second certification

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/p4_speaker_listener_concurrency.py --duration 300
'
```

## I. Master verifier

```bash
ssh gpu3090 '
cd ~/gemma-4-omni &&
./docker_run_3090.sh \
python3 tools/kernel_gates/verify_p4_final.py
'
```

## J. Extended soak

Run 1800 seconds only if that continuous token count is legal under the attested Gemma maximum context.

Otherwise use the maximum legally continuous context without inventing a new cache-reset policy.

---

# 36. Final Verdict Rules

During development:

```text
P4 OPEN — NATIVE EXECUTOR REMEDIATION IN PROGRESS
```

If native parity fails:

```text
P4 OPEN — NATIVE LISTENER PARITY FAIL
```

If parity passes but 80 ms full-duplex does not:

```text
P4 OPEN — SINGLE-GPU REALTIME CAPACITY FAIL
```

Only after all current-source hard gates pass:

```text
P4 PASS — NATIVE REALTIME CAUSAL FULL-DUPLEX KERNEL VERIFIED
```

Only then proceed to P5.

</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-04T23:21:05+02:00.
</ADDITIONAL_METADATA>