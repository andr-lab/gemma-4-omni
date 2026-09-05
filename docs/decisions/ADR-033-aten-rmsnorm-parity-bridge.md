# ADR-033: Literal ATen RMSNorm parity bridge for P4-NATIVE diagnosis

## Status
Accepted (diagnostic opt-in; native production path remains NO-GO)

## Context
After the ATen SDPA bridge removed the layer-0 attention mismatch, the exact
layer-local gate identified a new layer-2 `pre_ff_norm` mismatch. The exact
BF16 input and weight were equal; the first difference was the custom FP32
row reduction (`row_sum_fp32`, 1–4 ULP), followed by `mean`, reciprocal square
root, and final BF16 rounding. This is a numerical reduction-order issue, not
an FFN GEMM or activation issue.

## Decision
Add `gemma4_aten_rmsnorm_parity_async(...)` in a separate ATen translation
unit, compiled against the target PyTorch runtime (`2.12.1+cu130`). The bridge
borrows BF16 input, weight, and output storage and enqueues this literal
sequence on the caller CUDA stream:

```text
BF16 -> float -> pow(2) -> mean(-1, keepdim=True) -> + eps -> pow(-0.5)
-> input * scale -> weight -> BF16 output
```

The ATen views preserve the logical Python shape `[1, rows, dim]` for both
prefill and Q=4 execution. The bridge is enabled only by
`GEMMA4_ATEN_RMSNORM_BRIDGE=1` and is selected only for
`pre_feedforward_layernorm`; input, query/key/value, post-attention,
post-feedforward, and PLE norms remain on the custom kernels. It is therefore
not a production zero-allocation implementation.

The direct FFN diagnostic entrypoints (`debug_linear`, `debug_gelu_tanh`, and
`debug_mul`) use exact oracle tensors but do not change production behavior.
They exist to prove whether a GEMM or activation bridge is necessary.

## Evidence

- `P4_NATIVE_RMSNORM_ISOLATION_3D.json`: exact layer-2 canonical-prefix plus
  deterministic-frame input; custom kernel first differs at `row_sum_fp32`
  (`MAE=1.2207e-4`, max `2.4414e-4`), while the literal ATen bridge is exact
  (`MAE=0`, max `0`, exact BF16 stream).
- `P4_NATIVE_RMSNORM_LAYER4_POST_ATTN.json`: the next isolated custom norm
  boundary also differs first at `row_sum_fp32`; its ATen bridge is exact.
- `P4_NATIVE_FFN_PRIMITIVES.json`: layer-2 `gate_proj`, `up_proj`, GELU-tanh,
  gate-times-up, and `down_proj` all match exact BF16 streams (`MAE=0`), so no
  ATen GEMM/activation replacement is justified.
- `P4_NATIVE_LAYER_LOCAL_SCAN_ALL_ATEN_RMSNORM.json`: all 35 layers and 945
  local boundaries were executed. With the pre-FF bridge, layers 0–3 are
  exact; the first nonzero is layer 4 `post_attn_norm` (`MAE≈4.97e-9`), and
  the first formal failure is layer 9 `post_ff_norm` (`MAE≈3.94e-4`).
  Future-leak remains `0.0`.
- `P4_NATIVE_ATEN_BRIDGE_MULTIFRAME_TENSORS.pt` / report: the full primed
  3-frame path remains NO-GO (`MAE=0.3115, 0.6501, 0.6796`; cosine
  `0.9890, 0.9768, 0.9736`). This bridge does not certify native production
  parity.

## Consequences

The remaining mismatch is localized beyond layer 2: post-attention and other
RMSNorm reductions, plus an unresolved primed/prefill end-to-end discrepancy.
The default custom path is unchanged when the flag is absent. CUTLASS work and
any production bridge promotion remain deferred until strict multi-frame
parity, prefill parity, stream/lifetime safety, and zero-allocation evidence
pass.

## Official references

- PyTorch C++ ATen creation API:
  https://docs.pytorch.org/cppdocs/api/aten/creation.html
  (the `from_blob` documentation states that it views externally allocated
  data and that tensors created this way cannot be resized because ATen does
  not own the memory).
- PyTorch C++ ATen tensor API:
  https://docs.pytorch.org/cppdocs/api/aten/tensor.html
- PyTorch RMSNorm reference:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.rms_norm.html

The exact generated ATen overloads used here (`add`, `mul`, `pow`, and
`mean`) are verified by compilation and runtime parity against the installed
`2.12.1+cu130` library. The public documentation does not define this custom
C ABI/borrowed-storage stream contract; that part remains project-specific and
is validated by the native metadata and stream gates.
