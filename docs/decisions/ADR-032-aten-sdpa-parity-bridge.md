# ADR-032: ATen SDPA parity bridge for P4-NATIVE diagnosis

## Status
Accepted (diagnostic bridge; native production path unchanged)

## Context
The first strict P4-NATIVE layer-local failure was `layer 0 / attn_output`:
SDPA CUTLASS in Transformers differed from the scalar native attention kernel.
The native path must not be certified or replaced by a faster kernel before the
attention contract is isolated.

## Decision
Add `gemma4_listener_debug_attention(...)`, compiled against the exact PyTorch
installed in the target container (`2.12.1+cu130`). It:

- borrows CUDA Q/K/V/mask/output storage and uses the caller's CUDA stream;
- repeats K/V using Transformers' literal expand/reshape `repeat_kv` layout;
- calls public ATen `scaled_dot_product_attention` with the explicit mask,
  `dropout_p=0`, `is_causal=false`, `scale=1.0`, and `enable_gqa=false`;
- materializes non-contiguous native scratch/cache views before SDPA so the
  dispatch contract matches the isolated contiguous oracle;
- is enabled only with `GEMMA4_ATEN_SDPA_BRIDGE=1`; the default remains the
  native custom SDPA kernel.

The PyTorch build flags are exported by
`tools/kernel_gates/export_torch_build_config.py` into
`native/gemma4_listener/torch_build_config.mk`.

## Evidence

- `P4_NATIVE_SDPA_CONTRACT.json`: layer 0 Q `[1,8,4,256]`, K/V
  `[1,1,38,256]`, BOOL mask `[1,1,4,38]`, all contiguous, explicit
  `enable_gqa=false` contract.
- `P4_NATIVE_SDPA_ISOLATION.json`: public bridge matches the exact oracle on
  3 consecutive Q=4 frames (MAE `0`, max diff `0`, identical BF16 element
  streams).
- With the opt-in bridge, the layer-local first formal failure moves from
  `layer 0 / attn_output` to `layer 2 / down_proj` (MAE `1.0587e-4`), proving
  the original attention failure is removed without implying whole-model
  parity.
- Full 3-frame layer output remains `NO-GO`; non-SDPA drift persists.

## Consequences

The bridge is a diagnostic reference, not a production implementation: its
ATen contiguous materialization can allocate through the PyTorch caching
allocator and therefore does not satisfy the native zero-allocation
steady-state contract. Native CUTLASS work is deferred until the remaining
non-SDPA parity failures and allocation audits are closed.

Official references used by the gate:

- https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
