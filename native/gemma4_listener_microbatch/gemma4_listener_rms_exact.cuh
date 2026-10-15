/*
 * gemma4_listener_rms_exact.cuh — model-specialized native RMSNorm.
 *
 * The reduction mirrors the CUDA TensorIterator reduction shipped by the
 * certified PyTorch runtime (2.12.1+cu130, git 7269437d...).  In particular:
 * - pow(2) is the elementwise float multiply;
 * - mean uses four independent float accumulators per lane, then the ATen
 *   block-x tree (shared-memory halving followed by descending warp shuffles);
 * - pow(-0.5) is CUDA rsqrt, as in PyTorch's special-case dispatch.
 *
 * Source references (exact runtime revision):
 * https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/Reduce.cuh
 * https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/cuda/PowKernel.cu
 * https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/SharedReduceOps.h
 */
#ifndef GEMMA4_LISTENER_RMS_EXACT_CUH
#define GEMMA4_LISTENER_RMS_EXACT_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float *squared_fp32; /* [max_elements] */
    float *mean_fp32;    /* [max_rows] */
    float *scale_fp32;   /* [max_rows] */
    size_t max_elements;
    size_t max_rows;
    size_t max_dim;
} gemma4_rms_exact_workspace;

/* Production native exact path. The caller owns a persistent workspace and
 * must serialize calls on the supplied stream, exactly like the listener
 * context serializes all frame work. No allocation is performed here. */
int gemma4_rmsnorm_native_exact_async(
    const void *input,
    const float *weight_fp32,
    void *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    gemma4_rms_exact_workspace *workspace,
    cudaStream_t stream
);

/* Internal shape-stable entrypoint for batched rows. `rows` controls the
 * contiguous input/output extent and launch grid; `geometry_rows` selects the
 * reduction tree used by the equivalent standard per-call shape. */
int gemma4_rmsnorm_native_exact_geometry_async(
    const void *input,
    const float *weight_fp32,
    void *output,
    uint32_t rows,
    uint32_t geometry_rows,
    uint32_t dim,
    float eps,
    gemma4_rms_exact_workspace *workspace,
    cudaStream_t stream
);

/* Diagnostic entrypoint used by the staged parity gate. It writes the exact
 * native stage boundaries into caller-owned CUDA float buffers. `weight_fp32`
 * may be null for the non-affine V norm. */
int gemma4_rmsnorm_native_exact_trace_async(
    const void *input,
    const float *weight_fp32,
    void *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    float *squared_fp32,
    float *row_sum_fp32,
    float *mean_fp32,
    float *mean_plus_eps_fp32,
    float *scale_fp32,
    float *normalized_fp32,
    float *weighted_fp32,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_LISTENER_RMS_EXACT_CUH */
