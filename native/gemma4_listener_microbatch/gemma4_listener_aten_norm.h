/*
 * gemma4_listener_aten_norm.h — diagnostic literal ATen RMSNorm bridge.
 *
 * Compiled against the exact PyTorch C++ runtime installed in the target
 * container. This is a parity oracle, not a production allocation contract.
 */
#ifndef GEMMA4_LISTENER_ATEN_NORM_H
#define GEMMA4_LISTENER_ATEN_NORM_H

#include <stdint.h>
#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *input_fp32;
    void *squared_fp32;
    void *mean_fp32;
    void *mean_plus_eps_fp32;
    void *scale_fp32;
    void *normalized_fp32;
    void *weighted_fp32;
    void *weight_fp32;
    size_t max_elements;
    size_t max_rows;
    size_t max_dim;
} gemma4_aten_rms_workspace;

/* Execute the literal ShapeStable sequence on the supplied CUDA stream:
 * BF16 input/weight -> FP32 -> pow(2) -> mean -> +eps -> pow(-0.5) ->
 * input*scale -> optional weight -> BF16 output. All buffers are contiguous
 * flat storage: input/output [1, rows, dim], optional weight [dim]. A null
 * weight computes the unweighted RMSNorm used by Gemma V norms. */
int gemma4_aten_rmsnorm_parity_async(
    const void *input,
    const void *weight,
    void *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    cudaStream_t stream
);

/* Exact same sequence using caller-owned persistent CUDA float workspace.
 * The workspace is context-owned and reused serially on the listener stream;
 * no CUDA storage is allocated by this function. */
int gemma4_aten_rmsnorm_preallocated_async(
    const void *input,
    const void *weight,
    void *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    const gemma4_aten_rms_workspace *workspace,
    const void *cached_weight_fp32,
    cudaStream_t stream
);

/* Convert one frozen BF16 norm weight into context-owned FP32 storage on the
 * supplied stream. The conversion is the same ATen copy_ used by the literal
 * bridge and is performed only during first-use initialization. */
int gemma4_aten_rmsnorm_cache_weight_async(
    const void *weight_bf16,
    void *weight_fp32,
    uint32_t dim,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_LISTENER_ATEN_NORM_H */
