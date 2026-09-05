/*
 * gemma4_listener_aten_norm.h — diagnostic literal ATen RMSNorm bridge.
 *
 * Compiled against the exact PyTorch C++ runtime installed in the target
 * container. This is a parity oracle, not a production allocation contract.
 */
#ifndef GEMMA4_LISTENER_ATEN_NORM_H
#define GEMMA4_LISTENER_ATEN_NORM_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Execute the literal ShapeStable sequence on the supplied CUDA stream:
 * BF16 input/weight -> FP32 -> pow(2) -> mean -> +eps -> pow(-0.5) ->
 * input*scale -> weight -> BF16 output. All buffers are contiguous flat
 * storage: input/output [1, rows, dim], weight [dim]. */
int gemma4_aten_rmsnorm_parity_async(
    const void *input,
    const void *weight,
    void *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_LISTENER_ATEN_NORM_H */
