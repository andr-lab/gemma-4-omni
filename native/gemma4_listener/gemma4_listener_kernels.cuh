/*
 * gemma4_listener_kernels.cuh — Custom CUDA kernels for Gemma 4 Native Listener.
 *
 * Implements exact FP32-accumulation RMSNorm, GeGLU GELUTanh, RoPE,
 * residual addition with layer scaling, and PLE gating.
 * Conforms strictly to P4-NATIVE Implementation Plan §20.
 */
#ifndef GEMMA4_LISTENER_KERNELS_CUH
#define GEMMA4_LISTENER_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "generated_topology.h"

#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_COEFF 0.044715f

/* -------------------------------------------------------------------------
 * Warp / Block Reduction Helpers (FP32)
 * ------------------------------------------------------------------------- */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float b_sum = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        b_sum = warp_reduce_sum(b_sum);
    }
    __syncthreads();
    return b_sum;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_max(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    float b_max = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : -1e20f;
    if (wid == 0) {
        b_max = warp_reduce_max(b_max);
    }
    __syncthreads();
    return b_max;
}

/* -------------------------------------------------------------------------
 * Exact FP32-Accumulation RMSNorm
 * ------------------------------------------------------------------------- */
__global__ void kernel_finite_guard_bf16(
    __nv_bfloat16 *values,
    int count,
    int *status
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = __bfloat162float(values[index]);
    if (!isfinite(value)) {
        values[index] = __float2bfloat16(0.0f);
        atomicExch(status, 1);
    }
}

template <int D>
__global__ void kernel_rmsnorm_bf16(
    const __nv_bfloat16 * __restrict__ x,
    const __nv_bfloat16 * __restrict__ w,
    __nv_bfloat16 * __restrict__ y,
    int num_rows,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const __nv_bfloat16 *row_x = x + row * D;
    __nv_bfloat16 *row_y = y + row * D;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __bfloat162float(row_x[i]);
        sum_sq += val * val;
    }

    __shared__ float s_rsqrt;
    float block_sum = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) {
        float mean_sq = block_sum / (float)D;
        /* Gemma4RMSNorm uses torch.pow(mean_squared, -0.5), not rsqrt.
         * Keep the function and exponent explicit to match the checkpoint's
         * canonical BF16 boundary. */
        s_rsqrt = powf(mean_sq + eps, -0.5f);
    }
    __syncthreads();

    float rsq = s_rsqrt;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float val = __bfloat162float(row_x[i]);
        float weight = w ? __bfloat162float(w[i]) : 1.0f;
        row_y[i] = __float2bfloat16(val * rsq * weight);
    }
}

/* -------------------------------------------------------------------------
 * Fused GeGLU Activation with GELUTanh
 *   out = gelu_tanh(gate) * up
 * ------------------------------------------------------------------------- */
__device__ __forceinline__ float gelu_tanh(float x) {
    float cube = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * cube);
    return 0.5f * x * (1.0f + tanhf(inner));
}

__global__ void kernel_gelu_tanh_bf16(
    const __nv_bfloat16 * __restrict__ input,
    __nv_bfloat16 * __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = __float2bfloat16(gelu_tanh(__bfloat162float(input[idx])));
    }
}

__global__ void kernel_mul_bf16(
    const __nv_bfloat16 * __restrict__ lhs,
    const __nv_bfloat16 * __restrict__ rhs,
    __nv_bfloat16 * __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = __hmul(lhs[idx], rhs[idx]);
    }
}

/* -------------------------------------------------------------------------
 * Rotary Position Embedding (RoPE)
 *   In-place rotation on Q or K tensors: [q_len, num_heads, head_dim]
 * ------------------------------------------------------------------------- */
__global__ void kernel_rope_bf16(
    __nv_bfloat16 * __restrict__ x,
    const float * __restrict__ inv_freq,
    int num_heads,
    int q_len,
    int head_dim,
    uint32_t base_position,
    float scaling
) {
    int half_dim = head_dim / 2;
    int pair_idx = threadIdx.x;
    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (pair_idx >= half_dim || q_idx >= q_len || head_idx >= num_heads) return;

    /* Memory layout is [q_len, num_heads, head_dim] */
    int offset = (q_idx * num_heads + head_idx) * head_dim;
    float pos = (float)(base_position + q_idx);
    float freq = pos * inv_freq[pair_idx];

    float cos_val = cosf(freq) * scaling;
    float sin_val = sinf(freq) * scaling;

    float x1 = __bfloat162float(x[offset + pair_idx]);
    float x2 = __bfloat162float(x[offset + pair_idx + half_dim]);

    float rot1 = x1 * cos_val - x2 * sin_val;
    float rot2 = x2 * cos_val + x1 * sin_val;

    x[offset + pair_idx] = __float2bfloat16(rot1);
    x[offset + pair_idx + half_dim] = __float2bfloat16(rot2);
}

/* -------------------------------------------------------------------------
 * Elementwise Residual Add + Layer Scale
 *   out = (residual + input) * layer_scalar
 * ------------------------------------------------------------------------- */
__global__ void kernel_residual_add_and_scale_bf16(
    const __nv_bfloat16 * __restrict__ residual,
    const __nv_bfloat16 * __restrict__ input,
    __nv_bfloat16 * __restrict__ out,
    int total_elements,
    float layer_scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        float r = __bfloat162float(residual[idx]);
        float in = __bfloat162float(input[idx]);
        out[idx] = __float2bfloat16((r + in) * layer_scalar);
    }
}

/* -------------------------------------------------------------------------
 * Elementwise Residual Add (scale = 1.0)
 * ------------------------------------------------------------------------- */
__global__ void kernel_residual_add_bf16(
    const __nv_bfloat16 * __restrict__ residual,
    const __nv_bfloat16 * __restrict__ input,
    __nv_bfloat16 * __restrict__ out,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        out[idx] = __hadd(residual[idx], input[idx]);
    }
}

/* Elementwise BF16 scale, matching the canonical BF16 layer-scalar multiply. */
__global__ void kernel_scale_bf16(
    const __nv_bfloat16 * __restrict__ input,
    __nv_bfloat16 * __restrict__ out,
    int total_elements,
    float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const __nv_bfloat16 scale = __float2bfloat16(scalar);
        out[idx] = __hmul(input[idx], scale);
    }
}

/* -------------------------------------------------------------------------
 * PLE projection normalization.  Token identity addition and input scale
 * are deliberately separate BF16 operations, matching the canonical model.
 *   Processes [q_len, 35, 256] elements
 * ------------------------------------------------------------------------- */
__global__ void kernel_ple_project_and_norm_bf16(
    const __nv_bfloat16 * __restrict__ raw_proj,      /* [q_len, 35 * 256] */
    const __nv_bfloat16 * __restrict__ norm_weight,   /* [256] */
    const __nv_bfloat16 * __restrict__ token_id,      /* [q_len, 35 * 256] or NULL */
    __nv_bfloat16 * __restrict__ out,                 /* [q_len, 35 * 256] */
    int num_rows,                                     /* q_len * 35 */
    float proj_scale,
    float norm_eps,
    float input_scale
) {
    (void)input_scale;
    const __nv_bfloat16 proj_scale_bf16 = __float2bfloat16(proj_scale);
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const __nv_bfloat16 *row_x = raw_proj + row * 256;
    (void)token_id;
    __nv_bfloat16 *row_out = out + row * 256;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float val = __bfloat162float(__hmul(row_x[i], proj_scale_bf16));
        sum_sq += val * val;
    }

    __shared__ float s_rsqrt;
    float block_sum = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) {
        float mean_sq = block_sum / 256.0f;
        /* Match Gemma4RMSNorm's torch.pow(mean_squared, -0.5). */
        s_rsqrt = powf(mean_sq + norm_eps, -0.5f);
    }
    __syncthreads();

    float rsq = s_rsqrt;
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        float val = __bfloat162float(__hmul(row_x[i], proj_scale_bf16));
        float w = norm_weight ? __bfloat162float(norm_weight[i]) : 1.0f;
        row_out[i] = __float2bfloat16(val * rsq * w);
    }
}

/* -------------------------------------------------------------------------
 * Layer PLE Multiplier: out = ple_gate_activation * projected_ple[:, layer_idx, :]
 * ------------------------------------------------------------------------- */
__global__ void kernel_ple_layer_mult_bf16(
    const __nv_bfloat16 * __restrict__ ple_gate_activation, /* [q_len, 256] */
    const __nv_bfloat16 * __restrict__ projected_ple, /* [q_len, 35 * 256] */
    __nv_bfloat16 * __restrict__ ple_act,             /* [q_len, 256] */
    int q_len,
    int layer_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = q_len * 256;
    if (idx >= total_elems) return;

    int q = idx / 256;
    int d = idx % 256;

    int ple_offset = (q * GEMMA4_NUM_LAYERS + layer_idx) * 256 + d;
    ple_act[idx] = __hmul(ple_gate_activation[idx], projected_ple[ple_offset]);
}

/* -------------------------------------------------------------------------
 * Small-Q Stream SDPA Kernel (Q_LEN = 4, num_heads = 8, num_kv_heads = 1)
 *   Q:   [4, num_heads, D]
 *   K:   [total_k_len, D]
 *   V:   [total_k_len, D]
 *   out: [4, num_heads, D] (ready for o_proj without permutation)
 * ------------------------------------------------------------------------- */
template <int D>
__global__ void kernel_stream_sdpa_q4_bf16(
    const __nv_bfloat16 * __restrict__ q,
    const __nv_bfloat16 * __restrict__ k,
    const __nv_bfloat16 * __restrict__ v,
    __nv_bfloat16 * __restrict__ out,
    int total_k_len,
    int score_capacity,     /* row stride in dynamic shared-memory scores */
    int k_start_offset,     /* for sliding window */
    uint32_t base_position, /* absolute position of q[0] */
    uint32_t sliding_window,
    const int64_t * __restrict__ prefix_attention,
    uint32_t prefix_len,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int k_visible_len = total_k_len - k_start_offset;
    if (k_visible_len <= 0 || score_capacity < k_visible_len) return;

    /* Shared memory for query vectors and attention scores */
    __shared__ float q_smem[4][D];
    extern __shared__ float s_mem[];
    float *scores = s_mem;

    /* 1. Load Query into shared memory from [q_len, num_heads, D] */
    for (int i = threadIdx.x; i < 4 * D; i += blockDim.x) {
        int qi = i / D;
        int di = i % D;
        int q_offset = (qi * GEMMA4_NUM_ATTN_HEADS + head_idx) * D + di;
        q_smem[qi][di] = __bfloat162float(q[q_offset]);
    }
    __syncthreads();

    /* 2. Compute Q * K^T dot products for each query */
    const __nv_bfloat16 *k_base = k + k_start_offset * D;
    for (int k_idx = threadIdx.x; k_idx < k_visible_len; k_idx += blockDim.x) {
        const __nv_bfloat16 *curr_k = k_base + k_idx * D;
        uint32_t k_pos = (uint32_t)(k_start_offset + k_idx);

        float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        #pragma unroll 4
        for (int d = 0; d < D; ++d) {
            float k_val = __bfloat162float(curr_k[d]);
            dot[0] += q_smem[0][d] * k_val;
            dot[1] += q_smem[1][d] * k_val;
            dot[2] += q_smem[2][d] * k_val;
            dot[3] += q_smem[3][d] * k_val;
        }

        #pragma unroll 4
        for (int qi = 0; qi < 4; ++qi) {
            uint32_t q_pos = base_position + qi;
            bool prefix_valid = (k_pos >= prefix_len) ||
                (prefix_attention == NULL) || (prefix_attention[k_pos] != 0);
            if (!prefix_valid || k_pos > q_pos) {
                scores[qi * score_capacity + k_idx] = -1e20f; /* Padding + causal mask */
            } else if (sliding_window > 0 && (int32_t)k_pos < (int32_t)q_pos - (int32_t)sliding_window + 1) {
                scores[qi * score_capacity + k_idx] = -1e20f; /* Sliding window mask */
            } else {
                scores[qi * score_capacity + k_idx] = dot[qi] * sm_scale;
            }
        }
    }
    __syncthreads();

    /* 3. Softmax per query */
    __shared__ float s_max[4];
    __shared__ float s_sum[4];

    #pragma unroll 4
    for (int qi = 0; qi < 4; ++qi) {
        float my_max = -1e20f;
        for (int ki = threadIdx.x; ki < k_visible_len; ki += blockDim.x) {
            my_max = fmaxf(my_max, scores[qi * score_capacity + ki]);
        }
        my_max = block_reduce_max(my_max);
        if (threadIdx.x == 0) s_max[qi] = my_max;
        __syncthreads();
    }

    #pragma unroll 4
    for (int qi = 0; qi < 4; ++qi) {
        float my_sum = 0.0f;
        float m = s_max[qi];
        for (int ki = threadIdx.x; ki < k_visible_len; ki += blockDim.x) {
            float e = expf(scores[qi * score_capacity + ki] - m);
            scores[qi * score_capacity + ki] = e;
            my_sum += e;
        }
        my_sum = block_reduce_sum(my_sum);
        if (threadIdx.x == 0) s_sum[qi] = my_sum;
        __syncthreads();
    }

    #pragma unroll 4
    for (int qi = 0; qi < 4; ++qi) {
        float inv_s = (s_sum[qi] > 0.0f) ? (1.0f / s_sum[qi]) : 0.0f;
        for (int ki = threadIdx.x; ki < k_visible_len; ki += blockDim.x) {
            /* The attested Gemma BF16 path rounds softmax probabilities
             * before the weighted V reduction. */
            scores[qi * score_capacity + ki] = __bfloat162float(
                __float2bfloat16(scores[qi * score_capacity + ki] * inv_s)
            );
        }
    }
    __syncthreads();

    /* 4. Weighted sum: O = scores * V -> write to [4, num_heads, D] */
    const __nv_bfloat16 *v_base = v + k_start_offset * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int ki = 0; ki < k_visible_len; ++ki) {
            float v_val = __bfloat162float(v_base[ki * D + d]);
            acc[0] += scores[0 * score_capacity + ki] * v_val;
            acc[1] += scores[1 * score_capacity + ki] * v_val;
            acc[2] += scores[2 * score_capacity + ki] * v_val;
            acc[3] += scores[3 * score_capacity + ki] * v_val;
        }

        #pragma unroll 4
        for (int qi = 0; qi < 4; ++qi) {
            int out_offset = (qi * GEMMA4_NUM_ATTN_HEADS + head_idx) * D + d;
            out[out_offset] = __float2bfloat16(acc[qi]);
        }
    }
}

/* -------------------------------------------------------------------------
 * General Prefill SDPA Kernel (q_len <= 64, num_heads = 8, num_kv_heads = 1)
 *   Grid: (GEMMA4_NUM_ATTN_HEADS, q_len)
 *   Block: 256 threads
 * ------------------------------------------------------------------------- */
template <int D>
__global__ void kernel_prefill_sdpa_bf16(
    const __nv_bfloat16 * __restrict__ q,
    const __nv_bfloat16 * __restrict__ k,
    const __nv_bfloat16 * __restrict__ v,
    __nv_bfloat16 * __restrict__ out,
    int q_len,
    int total_k_len,
    int k_start_offset,     /* for sliding window */
    uint32_t base_position, /* absolute position of q[0] */
    uint32_t sliding_window,
    const int64_t * __restrict__ prefix_attention,
    uint32_t prefix_len,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    int q_idx = blockIdx.y;
    if (head_idx >= GEMMA4_NUM_ATTN_HEADS || q_idx >= q_len) return;

    uint32_t q_pos = base_position + q_idx;
    int max_k_pos = (int)q_pos;
    int k_limit = (max_k_pos + 1 < total_k_len) ? (max_k_pos + 1) : total_k_len;
    int visible_keys = k_limit - k_start_offset;
    if (visible_keys <= 0) return;

    __shared__ float q_vec[D];
    __shared__ float s_scores[512];

    /* 1. Load Query into shared memory from [q_len, num_heads, D] */
    int q_offset = (q_idx * GEMMA4_NUM_ATTN_HEADS + head_idx) * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        q_vec[i] = __bfloat162float(q[q_offset + i]);
    }
    __syncthreads();

    /* 2. Compute Q * K^T dot products for all visible keys */
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    int num_warps = blockDim.x >> 5;

    const __nv_bfloat16 *k_base = k + k_start_offset * D;
    for (int ki = wid; ki < visible_keys; ki += num_warps) {
        const __nv_bfloat16 *curr_k = k_base + ki * D;
        uint32_t k_pos = (uint32_t)(k_start_offset + ki);

        float dot = 0.0f;
        for (int d = lane; d < D; d += 32) {
            dot += q_vec[d] * __bfloat162float(curr_k[d]);
        }
        dot = warp_reduce_sum(dot);

        if (lane == 0) {
            bool prefix_valid = (k_pos >= prefix_len) ||
                (prefix_attention == NULL) || (prefix_attention[k_pos] != 0);
            if (!prefix_valid || (sliding_window > 0 && (int32_t)k_pos < (int32_t)q_pos - (int32_t)sliding_window + 1)) {
                s_scores[ki] = -1e20f;
            } else {
                s_scores[ki] = dot * sm_scale;
            }
        }
    }
    __syncthreads();

    /* 3. Softmax */
    float my_max = -1e20f;
    for (int i = threadIdx.x; i < visible_keys; i += blockDim.x) {
        my_max = fmaxf(my_max, s_scores[i]);
    }
    my_max = block_reduce_max(my_max);
    __shared__ float s_max_val;
    if (threadIdx.x == 0) s_max_val = my_max;
    __syncthreads();

    float my_sum = 0.0f;
    for (int i = threadIdx.x; i < visible_keys; i += blockDim.x) {
        float ex = expf(s_scores[i] - s_max_val);
        s_scores[i] = ex;
        my_sum += ex;
    }
    my_sum = block_reduce_sum(my_sum);
    __shared__ float s_inv_sum;
    if (threadIdx.x == 0) s_inv_sum = (my_sum > 0.0f) ? (1.0f / my_sum) : 0.0f;
    __syncthreads();

    for (int i = threadIdx.x; i < visible_keys; i += blockDim.x) {
        /* Match the attested BF16 probability boundary. */
        s_scores[i] = __bfloat162float(__float2bfloat16(s_scores[i] * s_inv_sum));
    }
    __syncthreads();

    /* 4. Output projection: out = scores * V */
    const __nv_bfloat16 *v_base = v + k_start_offset * D;
    int out_offset = (q_idx * GEMMA4_NUM_ATTN_HEADS + head_idx) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float val = 0.0f;
        for (int ki = 0; ki < visible_keys; ++ki) {
            val += s_scores[ki] * __bfloat162float(v_base[ki * D + d]);
        }
        out[out_offset + d] = __float2bfloat16(val);
    }
}

#endif /* GEMMA4_LISTENER_KERNELS_CUH */
