/*
 * gemma4_listener_rms_exact.cu — exact native RMSNorm stages.
 *
 * This is intentionally a small, model-specialized adaptation of the
 * installed PyTorch CUDA reduction topology. It is not the old warp reduction:
 * PyTorch uses four vectorized input accumulators per lane and a descending
 * block-x reduction (shared memory, then warp shuffles).
 */
#include "gemma4_listener_rms_exact.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

__global__ void gemma4_rms_square_fp32(
    const __nv_bfloat16 *input,
    float *squared,
    uint32_t elements
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const float x = __bfloat162float(input[i]);
    squared[i] = x * x;
}

/* Equivalent to ReduceOp::block_x_reduce for the certified PyTorch runtime.
 * The block-y dimension maps independent output rows; it never participates
 * in this reduction for the contiguous [rows, dim] mean. */
template <int BLOCK_X, int BLOCK_Y>
__device__ __forceinline__ float gemma4_rms_block_x_reduce(
    float value,
    float (&shared)[BLOCK_Y][BLOCK_X]
) {
    if constexpr (BLOCK_X > 32) {
        const int address = threadIdx.y * BLOCK_X + threadIdx.x;
        shared[threadIdx.y][threadIdx.x] = value;
        for (int offset = BLOCK_X / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (threadIdx.x < offset && threadIdx.x + offset < BLOCK_X) {
                value += shared[threadIdx.y][threadIdx.x + offset];
                shared[threadIdx.y][threadIdx.x] = value;
            }
        }
        (void)address;
    }

    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

/* The exact TensorIterator configuration for contiguous FP32 [rows, dim]
 * means, with vt0=4 and input_vec_size=4:
 *
 *   input_idx = lane, step_input = BLOCK_X
 *   values[i] += input[row, (lane + n*BLOCK_X)*4 + i]
 *   values[0] += values[1] + values[2] + values[3]
 *
 * BLOCK_X/BLOCK_Y are selected by the same set_block_dimension and
 * split_input/split_output rules used by Reduce.cuh for the six production
 * shapes. */
template <int DIM, int BLOCK_X, int BLOCK_Y>
__global__ void gemma4_rms_mean_exact(
    const float *squared,
    float *row_sum,
    float *mean,
    uint32_t rows
) {
    const uint32_t row = blockIdx.x * BLOCK_Y + threadIdx.y;
    float value_list[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    if (row < rows) {
        const float *row_data = squared + static_cast<size_t>(row) * DIM;
        int idx = threadIdx.x;
        while (idx * 4 + 3 < DIM) {
            value_list[0] += row_data[idx * 4 + 0];
            value_list[1] += row_data[idx * 4 + 1];
            value_list[2] += row_data[idx * 4 + 2];
            value_list[3] += row_data[idx * 4 + 3];
            idx += BLOCK_X;
        }
    }

    value_list[0] += value_list[1];
    value_list[0] += value_list[2];
    value_list[0] += value_list[3];

    __shared__ float shared[BLOCK_Y][BLOCK_X];
    const float reduced = gemma4_rms_block_x_reduce<BLOCK_X, BLOCK_Y>(
        value_list[0], shared);
    if (threadIdx.x == 0 && row < rows) {
        /* MeanOps::project performs this FP32 multiplication after the
         * reduction; factor is the FP32 reciprocal of the reduced dimension. */
        if (row_sum) row_sum[row] = reduced;
        const float factor = 1.0f / static_cast<float>(DIM);
        mean[row] = reduced * factor;
    }
}

__global__ void gemma4_rms_finalize_exact(
    const __nv_bfloat16 *input,
    const float *mean,
    const float *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    float *mean_plus_eps,
    float *scale,
    float *normalized,
    float *weighted
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t elements = rows * dim;
    if (index >= elements) return;

    const uint32_t row = index / dim;
    const float with_eps = mean[row] + eps;
    const float rsqrt = rsqrtf(with_eps);
    const float normalized_value = __bfloat162float(input[index]) * rsqrt;
    const float weighted_value = weight ? normalized_value * weight[index % dim]
                                        : normalized_value;

    /* The optional traces are used only by the staged gate. Keeping these
     * writes in the same kernel preserves the exact scalar operation order. */
    if (index % dim == 0) {
        if (mean_plus_eps) mean_plus_eps[row] = with_eps;
        if (scale) scale[row] = rsqrt;
    }
    if (normalized) normalized[index] = normalized_value;
    if (weighted) weighted[index] = weighted_value;
    output[index] = __float2bfloat16(weighted_value);
}

template <int DIM>
int launch_mean_exact(
    const float *squared,
    float *row_sum,
    float *mean,
    uint32_t rows,
    uint32_t geometry_rows,
    cudaStream_t stream
) {
    /* set_block_dimension<scalar_t>(dim0=ceil(DIM/4), dim1=rows), followed by
     * split_input/split_output in Reduce.cuh. These are the resulting fixed
     * configurations for scalar FP32 and a 512-thread CUDA reduction. */
    if constexpr (DIM == 1536) {
        if (geometry_rows <= 1) {
            constexpr int bx = 256;
            gemma4_rms_mean_exact<DIM, bx, 1><<<
                rows, dim3(bx, 1, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 2) {
            constexpr int bx = 256;
            gemma4_rms_mean_exact<DIM, bx, 2><<<
                (rows + 1) / 2, dim3(bx, 2, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 4) {
            constexpr int bx = 128;
            gemma4_rms_mean_exact<DIM, bx, 4><<<
                (rows + 3) / 4, dim3(bx, 4, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 8) {
            constexpr int bx = 64;
            const uint32_t blocks = (rows + 7) / 8;
            gemma4_rms_mean_exact<DIM, bx, 8><<<
                blocks, dim3(bx, 8, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else {
            constexpr int bx = 32;
            const uint32_t blocks = (rows + 15) / 16;
            gemma4_rms_mean_exact<DIM, bx, 16><<<
                blocks, dim3(bx, 16, 1), 0, stream>>>(squared, row_sum, mean, rows);
        }
    } else if constexpr (DIM == 512) {
        if (geometry_rows <= 1) {
            constexpr int bx = 128;
            gemma4_rms_mean_exact<DIM, bx, 1><<<
                rows, dim3(bx, 1, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 2) {
            constexpr int bx = 128;
            gemma4_rms_mean_exact<DIM, bx, 2><<<
                (rows + 1) / 2, dim3(bx, 2, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 4) {
            constexpr int bx = 128;
            const uint32_t blocks = (rows + 3) / 4;
            gemma4_rms_mean_exact<DIM, bx, 4><<<
                blocks, dim3(bx, 4, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 8) {
            constexpr int bx = 64;
            const uint32_t blocks = (rows + 7) / 8;
            gemma4_rms_mean_exact<DIM, bx, 8><<<
                blocks, dim3(bx, 8, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else {
            constexpr int bx = 32;
            const uint32_t blocks = (rows + 15) / 16;
            gemma4_rms_mean_exact<DIM, bx, 16><<<
                blocks, dim3(bx, 16, 1), 0, stream>>>(squared, row_sum, mean, rows);
        }
    } else {
        if (geometry_rows <= 1) {
            constexpr int bx = 64;
            gemma4_rms_mean_exact<DIM, bx, 1><<<
                rows, dim3(bx, 1, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 2) {
            constexpr int bx = 64;
            gemma4_rms_mean_exact<DIM, bx, 2><<<
                (rows + 1) / 2, dim3(bx, 2, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 4) {
            constexpr int bx = 64;
            const uint32_t blocks = (rows + 3) / 4;
            gemma4_rms_mean_exact<DIM, bx, 4><<<
                blocks, dim3(bx, 4, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else if (geometry_rows <= 8) {
            constexpr int bx = 64;
            const uint32_t blocks = (rows + 7) / 8;
            gemma4_rms_mean_exact<DIM, bx, 8><<<
                blocks, dim3(bx, 8, 1), 0, stream>>>(squared, row_sum, mean, rows);
        } else {
            constexpr int bx = 32;
            const uint32_t blocks = (rows + 15) / 16;
            gemma4_rms_mean_exact<DIM, bx, 16><<<
                blocks, dim3(bx, 16, 1), 0, stream>>>(squared, row_sum, mean, rows);
        }
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

template <int DIM>
int launch_native_exact(
    const __nv_bfloat16 *input,
    const float *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t geometry_rows,
    float eps,
    float *squared,
    float *row_sum,
    float *mean,
    float *mean_plus_eps,
    float *scale,
    float *normalized,
    float *weighted,
    cudaStream_t stream
) {
    const uint64_t element_count64 = static_cast<uint64_t>(rows) * DIM;
    if (element_count64 > UINT32_MAX) return -1;
    const uint32_t elements = static_cast<uint32_t>(element_count64);

    gemma4_rms_square_fp32<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        input, squared, elements);
    if (cudaPeekAtLastError() != cudaSuccess) return -3;
    if (launch_mean_exact<DIM>(squared, row_sum, mean, rows, geometry_rows, stream) != 0) return -3;

    gemma4_rms_finalize_exact<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        input, mean, weight, output, rows, DIM, eps,
        mean_plus_eps, scale, normalized, weighted);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

} // namespace

extern "C" int gemma4_rmsnorm_native_exact_geometry_async(
    const void *input,
    const float *weight_fp32,
    void *output,
    uint32_t rows,
    uint32_t geometry_rows,
    uint32_t dim,
    float eps,
    gemma4_rms_exact_workspace *workspace,
    cudaStream_t stream
) {
    if (!input || !output || !workspace || rows == 0 || geometry_rows == 0 || dim == 0 ||
        !isfinite(eps) || eps < 0.0f ||
        rows > workspace->max_rows || dim > workspace->max_dim ||
        static_cast<size_t>(rows) > SIZE_MAX / static_cast<size_t>(dim) ||
        static_cast<size_t>(rows) * dim > workspace->max_elements ||
        !workspace->squared_fp32 || !workspace->mean_fp32 || !workspace->scale_fp32) {
        return -1;
    }
    if (dim == 256) {
        return launch_native_exact<256>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, geometry_rows, eps,
            workspace->squared_fp32, nullptr, workspace->mean_fp32, nullptr,
            workspace->scale_fp32, nullptr, nullptr, stream);
    }
    if (dim == 512) {
        return launch_native_exact<512>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, geometry_rows, eps,
            workspace->squared_fp32, nullptr, workspace->mean_fp32, nullptr,
            workspace->scale_fp32, nullptr, nullptr, stream);
    }
    if (dim == 1536) {
        return launch_native_exact<1536>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, geometry_rows, eps,
            workspace->squared_fp32, nullptr, workspace->mean_fp32, nullptr,
            workspace->scale_fp32, nullptr, nullptr, stream);
    }
    return -2;
}

extern "C" int gemma4_rmsnorm_native_exact_async(
    const void *input,
    const float *weight_fp32,
    void *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    gemma4_rms_exact_workspace *workspace,
    cudaStream_t stream
) {
    return gemma4_rmsnorm_native_exact_geometry_async(
        input, weight_fp32, output, rows, rows, dim, eps, workspace, stream);
}

extern "C" int gemma4_rmsnorm_native_exact_trace_async(
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
) {
    if (!input || !output || !squared_fp32 || !mean_fp32 ||
        !mean_plus_eps_fp32 || !scale_fp32 || !normalized_fp32 ||
        !weighted_fp32 || rows == 0 || dim == 0 || !isfinite(eps) || eps < 0.0f) {
        return -1;
    }
    if (dim == 256) {
        return launch_native_exact<256>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, rows, eps,
            squared_fp32, row_sum_fp32, mean_fp32, mean_plus_eps_fp32, scale_fp32,
            normalized_fp32, weighted_fp32, stream);
    }
    if (dim == 512) {
        return launch_native_exact<512>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, rows, eps,
            squared_fp32, row_sum_fp32, mean_fp32, mean_plus_eps_fp32, scale_fp32,
            normalized_fp32, weighted_fp32, stream);
    }
    if (dim == 1536) {
        return launch_native_exact<1536>(
            static_cast<const __nv_bfloat16 *>(input), weight_fp32,
            static_cast<__nv_bfloat16 *>(output), rows, rows, eps,
            squared_fp32, row_sum_fp32, mean_fp32, mean_plus_eps_fp32, scale_fp32,
            normalized_fp32, weighted_fp32, stream);
    }
    return -2;
}
