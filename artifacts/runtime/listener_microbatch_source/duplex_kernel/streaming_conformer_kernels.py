"""CUDA-only kernels used by the opt-in streaming Conformer implementation.

Importing this module does not import Triton.  The JIT kernel is created only
when the CUDA optimization is explicitly initialized.
"""
from __future__ import annotations

from typing import Any


_CLIPPED_LINEAR_KERNEL: Any | None = None


def initialize_triton_kernels() -> None:
    """Initialize Triton's JIT definition before entering Dynamo tracing."""
    _get_clipped_linear_kernel()


def _get_clipped_linear_kernel():
    global _CLIPPED_LINEAR_KERNEL
    if _CLIPPED_LINEAR_KERNEL is not None:
        return _CLIPPED_LINEAR_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def clipped_linear_kernel(
        X, W, B, IMIN, IMAX, OMIN, OMAX, Y,
        M: tl.constexpr, O: tl.constexpr, K: tl.constexpr,
        XS0: tl.constexpr, XS1: tl.constexpr, WS0: tl.constexpr, WS1: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr, CLIPPED: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        ks = tl.arange(0, BLOCK_K)
        accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if CLIPPED:
            input_min = tl.load(IMIN).to(tl.float32)
            input_max = tl.load(IMAX).to(tl.float32)
        for k_start in range(tl.cdiv(K, BLOCK_K)):
            k = k_start * BLOCK_K + ks
            x = tl.load(
                X + rows[:, None] * XS0 + k[None, :] * XS1,
                mask=(rows[:, None] < M) & (k[None, :] < K),
                other=0,
            )
            if CLIPPED:
                x = tl.minimum(tl.maximum(x.to(tl.float32), input_min), input_max)
            x = x.to(tl.bfloat16)
            w = tl.load(
                W + cols[:, None] * WS0 + k[None, :] * WS1,
                mask=(cols[:, None] < O) & (k[None, :] < K),
                other=0,
            )
            accum = tl.dot(x, tl.trans(w), accum)
        if HAS_BIAS:
            bias = tl.load(B + cols, mask=cols < O, other=0).to(tl.float32)
            accum = accum + bias[None, :]
        # Gemma materializes its BF16 linear result before the output clamp.
        output = accum.to(tl.bfloat16).to(tl.float32)
        if CLIPPED:
            output_min = tl.load(OMIN).to(tl.float32)
            output_max = tl.load(OMAX).to(tl.float32)
            output = tl.minimum(tl.maximum(output, output_min), output_max)
        tl.store(
            Y + rows[:, None] * O + cols[None, :],
            output.to(tl.bfloat16),
            mask=(rows[:, None] < M) & (cols[None, :] < O),
        )

    _CLIPPED_LINEAR_KERNEL = (clipped_linear_kernel, triton)
    return _CLIPPED_LINEAR_KERNEL


def _supported_input(
    module: Any, hidden_states: Any, *, require_frozen: bool
) -> bool:
    import torch

    linear = getattr(module, "linear", None)
    if (
        not isinstance(hidden_states, torch.Tensor)
        or hidden_states.device.type != "cuda"
        or hidden_states.dtype is not torch.bfloat16
        or hidden_states.ndim not in {2, 3}
        or not hidden_states.is_contiguous()
        or module.training
        or torch.is_grad_enabled()
        or linear is None
        or linear.weight.dtype is not torch.bfloat16
        or linear.weight.device != hidden_states.device
        or (
            require_frozen
            and any(parameter.requires_grad for parameter in module.parameters())
        )
    ):
        return False
    input_features = int(linear.weight.shape[1])
    return (
        input_features > 0
        and int(hidden_states.shape[-1]) == input_features
        and 1 <= hidden_states.numel() // input_features <= 8
    )


def triton_clipped_linear(
    module: Any,
    hidden_states: Any,
    *,
    block_n: int = 64,
    block_k: int = 64,
    require_frozen: bool = True,
):
    """Run one frozen Gemma4ClippableLinear with original BF16 parameters."""
    import torch

    if block_n not in {16, 32, 64, 128} or block_k not in {64, 128}:
        raise ValueError("Triton tiles require BLOCK_N 16/32/64/128 and BLOCK_K 64/128")
    if not _supported_input(module, hidden_states, require_frozen=require_frozen):
        raise ValueError("unsupported input for frozen CUDA BF16 Gemma linear")

    linear = module.linear
    output_features, input_features = map(int, linear.weight.shape)
    rows = int(hidden_states.numel() // input_features)
    clipped = bool(getattr(module, "use_clipped_linears", False))
    bias = linear.bias
    output = torch.empty(
        (*hidden_states.shape[:-1], output_features),
        dtype=torch.bfloat16,
        device=hidden_states.device,
    )
    kernel, triton = _get_clipped_linear_kernel()
    if clipped:
        imin, imax = module.input_min, module.input_max
        omin, omax = module.output_min, module.output_max
    else:
        # These pointers are compile-time dead when CLIPPED=False.
        imin = imax = omin = omax = linear.weight
    bias_pointer = bias if bias is not None else linear.weight
    flat_input = hidden_states.view(rows, input_features)
    kernel[(triton.cdiv(rows, 16), triton.cdiv(output_features, block_n))](
        flat_input,
        linear.weight,
        bias_pointer,
        imin,
        imax,
        omin,
        omax,
        output.view(rows, output_features),
        rows,
        output_features,
        input_features,
        int(flat_input.stride(0)),
        int(flat_input.stride(1)),
        int(linear.weight.stride(0)),
        int(linear.weight.stride(1)),
        16,
        block_n,
        block_k,
        bias is not None,
        clipped,
        num_warps=4,
    )
    return output


def install_gemma4_linear_wrappers(
    modules: tuple[Any, ...], *, block_n: int = 64, block_k: int = 64
) -> tuple[tuple[Any, Any, bool, Any], ...]:
    """Wrap frozen Gemma4ClippableLinear instances and retain exact originals."""
    import types

    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    installed: list[tuple[Any, Any, bool, Any]] = []
    seen: set[int] = set()
    try:
        for root in modules:
            for module in root.modules():
                if id(module) in seen or not isinstance(module, Gemma4ClippableLinear):
                    continue
                seen.add(id(module))
                if module.training:
                    continue
                had_instance_forward = "forward" in module.__dict__
                old_instance_forward = module.__dict__.get("forward")
                original_forward = module.forward

                def wrapped_forward(
                    self, hidden_states, *args, _original=original_forward, **kwargs
                ):
                    if not args and not kwargs and _supported_input(
                        self, hidden_states, require_frozen=False
                    ):
                        return triton_clipped_linear(
                            self,
                            hidden_states,
                            block_n=block_n,
                            block_k=block_k,
                            require_frozen=False,
                        )
                    return _original(hidden_states, *args, **kwargs)

                bound_wrapper = types.MethodType(wrapped_forward, module)
                module.forward = bound_wrapper
                installed.append(
                    (module, bound_wrapper, had_instance_forward, old_instance_forward)
                )
    except BaseException:
        restore_gemma4_linear_wrappers(tuple(installed))
        raise
    if not installed:
        raise RuntimeError("no frozen Gemma4ClippableLinear instances in optimized audio modules")
    return tuple(installed)


def restore_gemma4_linear_wrappers(
    wrappers: tuple[tuple[Any, Any, bool, Any], ...]
) -> None:
    """Restore the exact prior instance/class forward lookup for each module."""
    for module, bound_wrapper, had_instance_forward, old_instance_forward in reversed(wrappers):
        if module.forward is not bound_wrapper:
            continue
        if had_instance_forward:
            module.forward = old_instance_forward
        else:
            del module.forward
