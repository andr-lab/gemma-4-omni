# Native listener microbatch candidate

This directory is an isolated overlay of the recovered 22-file listener source
snapshot. It builds a separate `libgemma4_listener_microbatch.so`; it does not
replace the canonical ABI3 library. The new API is versioned independently as
batch API 1 and keeps the existing ABI3 weight descriptor unchanged.

## Execution and ownership

One `gemma4_listener_batch_ctx` belongs to one Python Listener worker and one
CUDA stream bound at priority `-1`. The context owns one set of shared
35-layer model workspaces and one private KV/cache-position/prefix-mask slot
per admitted session. Frozen model weights and the PAD-frame PLE tensor are
borrowed from the existing Python model and held strongly for the context
lifetime. Batch sizes are exactly 1, 2, or 4; every forward consumes contiguous
BF16 `[N,4,1536]` input rows and returns an internal `[N,4,1536]` output slot.

The Python owner is confined to its creating host thread for context mutation,
cache import, forward/lease handling, rollover, retirement, counters, and close.
Cache import queues the producer-stream dependency before making any
canonicalized tensors, performs canonicalization in the Listener stream
context, records both source and canonical tensors with that stream, and holds
the source DynamicCache plus imported tensors until the native slot is retired.
The runtime capability names the actual CUDA device and stream handle; startup
also requires the native priority query to attest the bound stream's configured
priority `-1`.

The forward loop visits the 35 layers once. Dense projections use one
`cublasGemmEx` call over the combined `N*4` rows per projection; session-specific
KV writes and attention use each row's private cache, prefix mask, and absolute
cache position. No ABI3 single-session step is called from the batch API.
Attention remains on the existing backend-3 policy and RMSNorm on backend 2.
The code preserves Q=4, the 16384-token cache capacity, prefixes of 34 or 291,
and the existing fixed 35-layer topology. It does not opt into CUDA graphs or
alter model weights, training, templates, or audio roles.

The row descriptor carries native context ID, session generation, a stable
numeric epoch identity, frame ID, listener state version, sequence position,
cache position, and total valid-history tokens. Every row is checked before
input-copy or layer work. Cache position and valid-history tokens must match;
both advance by exactly four only after all layers and final normalization are
queued successfully. Any partial enqueue error poisons the whole batch context,
so callers cannot retry or fall back with possibly mutated caches.

Native output is a borrowed slot with a generation, completion event, and
ordered copy of the exact input row descriptors. The Python owner verifies the
native row provenance against the manifest before consuming output, enqueues a
device-to-device clone onto the bound Listener stream, and retains the clone
completion event with the owned tensor. Slot reuse is rejected while a lease
is open. Failure cleanup fences either the forward event or the clone event
before retiring the private lease. Session rollover and retirement also run on
that stream; resources are not freed while queued work can still reference
them.

## Implementation sources

The layer-wise BF16 GEMM call uses `CUBLAS_COMPUTE_32F` with BF16 inputs and
outputs, a combination documented by NVIDIA's [CUDA 13.1 cuBLAS `cublasGemmEx`
type table](https://docs.nvidia.com/cuda/archive/13.1.0/cublas/index.html). The
input, cache-import, clone, and lease operations use explicit stream ordering.
PyTorch 2.12's [CUDA semantics documentation](https://docs.pytorch.org/docs/2.12/notes/cuda.html)
states that cross-stream work requires explicit ordering, `wait_stream()`
orders a consumer stream after producer work, and `record_stream()` prevents
allocator reuse before side-stream use completes. Its
[Tensor.record_stream API](https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.record_stream.html)
documents that it tells the caching allocator which streams used a tensor.
CUDA 13.1's
[event documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-runtime-api/group__CUDART__EVENT.html)
defines a recorded event as capturing the stream contents at record time and
allows a stream wait to depend on that captured work. The wrapper therefore
uses stream waits/events instead of per-frame host synchronization; native
destroy is a lifecycle boundary and drains the owned stream before freeing
allocations.

Version-specific documentation anchors: cuBLAS 13.1 lists
`CUBLAS_COMPUTE_32F` with BF16 matrix inputs and outputs; PyTorch 2.12 says
`wait_stream()` “ensures that the normal_() execution has finished” before
work on a side stream and `record_stream()` “ensures that we do not deallocate
A before sum(A) has completed”; CUDA 13.1 says a recorded event “captures in
event the contents of stream at the time of this call.” These passages support
the type combination and ordering/lifetime primitives, not numerical parity
or performance.

Build settings are inherited from the recovered target Makefile: CUDA 13.1,
SM86, C++17, the existing project PyTorch bridge, and no `--use_fast_math`.
The target runtime is PyTorch 2.12.1+cu130 per the deployment contract; the
ATen bridge remains compiled against the exact target container installation.
PyTorch v2.12.1's [`torch.utils.cpp_extension` source](https://github.com/pytorch/pytorch/blob/v2.12.1/torch/utils/cpp_extension.py)
raises on a CUDA major-version mismatch and emits a warning, rather than an
error, for a minor-version mismatch. Thus CUDA 13.1 nvcc with a cu130 build is
within that check's supported same-major path, but compilation/link success
must still be measured in the target container with the 13.1 headers/libraries
and target PyTorch paths; the source rule is not a runtime or ABI certification.

## Counters and proof boundary

The candidate exports a versioned capability mask, runtime facts, and counters
for forward calls, layers executed, projection GEMM calls/rows, session rows,
and output generations. CPU-only tests check ABI layouts, row mapping, exported
names, and the structural shared-row forward path.

This source candidate has not yet been built with CUDA or run on the RTX 3090.
It is therefore not yet demonstrated to compile, preserve exact N=1/N=2/N=4
hidden-state and full-KV outputs, fit the live GPU memory budget, avoid drops,
or meet p95 latency. `cublasGemmEx` may select numerically different algorithms
for M=4, 8, or 16, so strict parity remains a required measured gate; this
implementation does not loosen numerical tolerances or declare parity by
construction.
