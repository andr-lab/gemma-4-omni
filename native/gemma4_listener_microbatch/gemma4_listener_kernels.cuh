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

struct gemma4_q4_position_range {
    uint32_t total_k_len;
    uint32_t start_offset;
    bool valid;
};

__host__ __device__ constexpr gemma4_q4_position_range gemma4_q4_range_from_position(
    uint32_t base_position,
    uint32_t q_len,
    uint32_t capacity,
    uint32_t sliding_window
) {
    const uint64_t total = (uint64_t)base_position + (uint64_t)q_len;
    const uint32_t start = sliding_window > 0 && base_position >= sliding_window
        ? base_position - sliding_window + 1u
        : 0u;
    return {
        total <= UINT32_MAX ? (uint32_t)total : UINT32_MAX,
        start,
        q_len > 0 && total <= capacity && start < total,
    };
}

static_assert(gemma4_q4_range_from_position(34, 4, 4096, 0).valid &&
              gemma4_q4_range_from_position(34, 4, 4096, 0).total_k_len == 38 &&
              gemma4_q4_range_from_position(34, 4, 4096, 0).start_offset == 0,
              "Q4 full-attention graph range must include prefix plus current frame");
static_assert(gemma4_q4_range_from_position(511, 4, 4096, 512).start_offset == 0 &&
              gemma4_q4_range_from_position(512, 4, 4096, 512).start_offset == 1,
              "Q4 sliding range must match the legacy window boundary");
static_assert(!gemma4_q4_range_from_position(4093, 4, 4096, 0).valid,
              "Q4 graph range must reject an append beyond fixed cache capacity");

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
 * Fixed canonical prefix mask used when importing a ShapeStable cache.
 * ------------------------------------------------------------------------- */
__global__ void kernel_set_canonical_prefix_mask(
    int64_t *mask,
    int length
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        mask[index] = (index == 0 || index == length - 1) ? 1 : 0;
    }
}

/* -------------------------------------------------------------------------
 * Explicit repeat-KV staging for the masked ATen oracle.
 * ------------------------------------------------------------------------- */
__global__ void kernel_repeat_kv_bf16(
    const __nv_bfloat16 * __restrict__ source,
    __nv_bfloat16 * __restrict__ destination,
    int key_len,
    int head_dim
) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = key_len * head_dim;
    if (element >= total) return;
    const int key = element / head_dim;
    const int dim = element - key * head_dim;
    const __nv_bfloat16 value = source[key * head_dim + dim];
    #pragma unroll
    for (int head = 0; head < GEMMA4_NUM_ATTN_HEADS; ++head) {
        destination[(head * key_len + key) * head_dim + dim] = value;
    }
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

/* Diagnostic-only expansion of the current kernel.  It deliberately keeps
 * the same block reduction and powf path as kernel_rmsnorm_bf16, while
 * exposing its FP32 boundaries for first-difference diagnosis.  Production
 * execution never launches this kernel. */
template <int D>
__global__ void kernel_rmsnorm_bf16_trace(
    const __nv_bfloat16 * __restrict__ x,
    const __nv_bfloat16 * __restrict__ w,
    float * __restrict__ input_fp32,
    float * __restrict__ x_squared,
    float * __restrict__ row_sum,
    float * __restrict__ row_mean,
    float * __restrict__ mean_plus_eps,
    float * __restrict__ pow_neg_half,
    float * __restrict__ normalized,
    float * __restrict__ weighted,
    int num_rows,
    float eps
) {
    int row = blockIdx.x;
    if (row >= num_rows) return;

    const __nv_bfloat16 *row_x = x + row * D;
    float *row_input = input_fp32 + row * D;
    float *row_squared = x_squared + row * D;
    float *row_normalized = normalized + row * D;
    float *row_weighted = weighted + row * D;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float value = __bfloat162float(row_x[i]);
        row_input[i] = value;
        row_squared[i] = value * value;
        sum_sq += row_squared[i];
    }

    __shared__ float s_rsqrt;
    float block_sum = block_reduce_sum(sum_sq);
    if (threadIdx.x == 0) {
        const float mean = block_sum / (float)D;
        const float with_eps = mean + eps;
        const float scale = powf(with_eps, -0.5f);
        row_sum[row] = block_sum;
        row_mean[row] = mean;
        mean_plus_eps[row] = with_eps;
        pow_neg_half[row] = scale;
        s_rsqrt = scale;
    }
    __syncthreads();

    const float scale = s_rsqrt;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        const float normalized_value = row_input[i] * scale;
        const float weight = w ? __bfloat162float(w[i]) : 1.0f;
        row_normalized[i] = normalized_value;
        row_weighted[i] = normalized_value * weight;
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
    const uint32_t * __restrict__ device_position,
    float scaling
) {
    int half_dim = head_dim / 2;
    int pair_idx = threadIdx.x;
    int q_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (pair_idx >= half_dim || q_idx >= q_len || head_idx >= num_heads) return;

    /* Memory layout is [q_len, num_heads, head_dim] */
    int offset = (q_idx * num_heads + head_idx) * head_dim;
    const uint32_t effective_base_position = device_position ? *device_position : base_position;
    float pos = (float)(effective_base_position + q_idx);
    float freq = pos * inv_freq[pair_idx];

    /* modeling_gemma4 computes cos/sin in FP32, casts them to BF16, then
     * applies the two pointwise BF16 multiplies and BF16 add/subtract.  Do
     * not fuse the whole rotation through FP32: that creates a different
     * rounding boundary and breaks strict parity at the first RoPE tap. */
    const __nv_bfloat16 cos_bf16 = __float2bfloat16(cosf(freq) * scaling);
    const __nv_bfloat16 sin_bf16 = __float2bfloat16(sinf(freq) * scaling);
    const __nv_bfloat16 x1 = x[offset + pair_idx];
    const __nv_bfloat16 x2 = x[offset + pair_idx + half_dim];

    const __nv_bfloat16 x1_cos = __hmul(x1, cos_bf16);
    const __nv_bfloat16 x2_sin = __hmul(x2, sin_bf16);
    const __nv_bfloat16 x2_cos = __hmul(x2, cos_bf16);
    const __nv_bfloat16 x1_sin = __hmul(x1, sin_bf16);
    x[offset + pair_idx] = __hsub(x1_cos, x2_sin);
    x[offset + pair_idx + half_dim] = __hadd(x2_cos, x1_sin);
}

/* Shift already-rotated cache keys by a constant position delta during a
 * bounded KV rebase. Cache layout is [token, head_dim] with one KV head. */
__global__ void kernel_rope_rebase_bf16(
    __nv_bfloat16 * __restrict__ x,
    const float * __restrict__ inv_freq,
    int token_count,
    int head_dim,
    int position_delta,
    float scaling
) {
    const int pair_idx = threadIdx.x;
    const int token_idx = blockIdx.x;
    const int half_dim = head_dim / 2;
    if (pair_idx >= half_dim || token_idx >= token_count) return;

    const int offset = token_idx * head_dim;
    const float freq = (float)position_delta * inv_freq[pair_idx];
    const __nv_bfloat16 cos_bf16 = __float2bfloat16(cosf(freq) * scaling);
    const __nv_bfloat16 sin_bf16 = __float2bfloat16(sinf(freq) * scaling);
    const __nv_bfloat16 x1 = x[offset + pair_idx];
    const __nv_bfloat16 x2 = x[offset + pair_idx + half_dim];
    const __nv_bfloat16 x1_cos = __hmul(x1, cos_bf16);
    const __nv_bfloat16 x2_sin = __hmul(x2, sin_bf16);
    const __nv_bfloat16 x2_cos = __hmul(x2, cos_bf16);
    const __nv_bfloat16 x1_sin = __hmul(x1, sin_bf16);
    x[offset + pair_idx] = __hsub(x1_cos, x2_sin);
    x[offset + pair_idx + half_dim] = __hadd(x2_cos, x1_sin);
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

/* Elementwise BF16 scale used by the legacy custom path. */
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

/* Exact canonical BF16-input * FP32-scalar followed by a BF16 store. Keep
 * this separate so the ATen parity bridge can use the canonical scalar
 * promotion without changing the default custom path. */
__global__ void kernel_scale_bf16_fp32_scalar(
    const __nv_bfloat16 * __restrict__ input,
    __nv_bfloat16 * __restrict__ out,
    int total_elements,
    float scalar
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        const float value = __bfloat162float(input[idx]);
        out[idx] = __float2bfloat16(value * scalar);
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
 * Explicit bool mask for the diagnostic ATen SDPA bridge.
 * Layout: compact [q_len, visible_k_len], with absolute K positions derived
 * from k_start_offset.  This is not used by the default native kernel path.
 * ------------------------------------------------------------------------- */
__global__ void kernel_build_explicit_sdpa_mask(
    bool * __restrict__ mask,
    int q_len,
    int visible_k_len,
    int k_start_offset,
    uint32_t base_position,
    uint32_t sliding_window,
    const int64_t * __restrict__ prefix_attention,
    uint32_t prefix_len
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = q_len * visible_k_len;
    if (index >= elements) return;
    const int query_index = index / visible_k_len;
    const int key_index = index % visible_k_len;
    const uint32_t key_position = static_cast<uint32_t>(k_start_offset + key_index);
    const uint32_t query_position = base_position + static_cast<uint32_t>(query_index);
    const bool prefix_valid = key_position >= prefix_len ||
        prefix_attention == NULL || prefix_attention[key_position] != 0;
    const bool causal_valid = key_position <= query_position;
    const bool window_valid = sliding_window == 0 ||
        static_cast<int32_t>(key_position) >=
            static_cast<int32_t>(query_position) - static_cast<int32_t>(sliding_window) + 1;
    mask[index] = prefix_valid && causal_valid && window_valid;
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
    const uint32_t * __restrict__ device_position,
    uint32_t sliding_window,
    const int64_t * __restrict__ prefix_attention,
    uint32_t prefix_len,
    float sm_scale
) {
    int head_idx = blockIdx.x;
    if (device_position) {
        base_position = *device_position;
        const gemma4_q4_position_range range = gemma4_q4_range_from_position(
            base_position, GEMMA4_STREAM_Q_LEN, (uint32_t)score_capacity, sliding_window);
        if (!range.valid) return;
        total_k_len = (int)range.total_k_len;
        k_start_offset = (int)range.start_offset;
    }
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
            /* Keep the probability in FP32. PyTorch SDPA's BF16 input path
             * accumulates softmax and the weighted-V reduction in FP32; a
             * BF16 probability round here is an extra boundary. */
            scores[qi * score_capacity + ki] =
                scores[qi * score_capacity + ki] * inv_s;
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

/* Repeat the four immutable PAD-token PLE rows for a bounded history batch. */
__global__ void kernel_repeat_frame_pad_ple_bf16(
    const __nv_bfloat16 * __restrict__ frame_ple,
    __nv_bfloat16 * __restrict__ repeated_ple,
    int token_count,
    int elements_per_token
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = token_count * elements_per_token;
    if (index >= total) return;
    const int token = index / elements_per_token;
    const int element = index % elements_per_token;
    repeated_ple[index] = frame_ple[(token % GEMMA4_STREAM_Q_LEN) * elements_per_token + element];
}

/* -------------------------------------------------------------------------
 * General Prefill SDPA Kernel (q_len <= 291, num_heads = 8, num_kv_heads = 1)
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
    __shared__ float s_scores[1024];

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
        /* Keep the probability in FP32 for the weighted-V reduction. */
        s_scores[i] = s_scores[i] * s_inv_sum;
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


/* -------------------------------------------------------------------------
 * Opt-in four-query BF16 GEMV for streaming dense projections.
 * One weight pair is reused across all four input rows in FP32 accumulation.
 * Only M=4 calls with contiguous BF16 row-major operands are dispatched here;
 * prefix/history projections remain on cuBLAS.
 * ------------------------------------------------------------------------- */
__global__ void kernel_linear_q4_gemv4_bf16(
    const __nv_bfloat16 * __restrict__ weight, /* [N,K], row-major */
    const __nv_bfloat16 * __restrict__ input,  /* [4,K], row-major */
    __nv_bfloat16 * __restrict__ output,       /* [4,N], row-major */
    int N,
    int K
) {
    constexpr int K_TILE = 128;
    constexpr int PAIRS_PER_TILE = K_TILE / 2;
    __shared__ __nv_bfloat162 input_tile[4][PAIRS_PER_TILE];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int output_col = (int)blockIdx.x * 4 + warp;
    const __nv_bfloat162 *weight_pairs =
        reinterpret_cast<const __nv_bfloat162 *>(weight);
    const __nv_bfloat162 *input_pairs =
        reinterpret_cast<const __nv_bfloat162 *>(input);

    float accum0 = 0.0f;
    float accum1 = 0.0f;
    float accum2 = 0.0f;
    float accum3 = 0.0f;

    for (int k_base = 0; k_base < K; k_base += K_TILE) {
        for (int item = tid; item < 4 * PAIRS_PER_TILE; item += blockDim.x) {
            const int row = item / PAIRS_PER_TILE;
            const int pair = item - row * PAIRS_PER_TILE;
            const int absolute_pair = k_base / 2 + pair;
            input_tile[row][pair] = (2 * absolute_pair < K)
                ? input_pairs[row * (K / 2) + absolute_pair]
                : __float2bfloat162_rn(0.0f);
        }
        __syncthreads();

        if (output_col < N) {
            for (int pair = lane; pair < PAIRS_PER_TILE; pair += 32) {
                const int absolute_pair = k_base / 2 + pair;
                if (2 * absolute_pair >= K) continue;
                const __nv_bfloat162 packed_weight =
                    weight_pairs[output_col * (K / 2) + absolute_pair];
                const float2 w = __bfloat1622float2(packed_weight);
                const float2 x0 = __bfloat1622float2(input_tile[0][pair]);
                const float2 x1 = __bfloat1622float2(input_tile[1][pair]);
                const float2 x2 = __bfloat1622float2(input_tile[2][pair]);
                const float2 x3 = __bfloat1622float2(input_tile[3][pair]);
                accum0 = fmaf(w.x, x0.x, accum0);
                accum0 = fmaf(w.y, x0.y, accum0);
                accum1 = fmaf(w.x, x1.x, accum1);
                accum1 = fmaf(w.y, x1.y, accum1);
                accum2 = fmaf(w.x, x2.x, accum2);
                accum2 = fmaf(w.y, x2.y, accum2);
                accum3 = fmaf(w.x, x3.x, accum3);
                accum3 = fmaf(w.y, x3.y, accum3);
            }
        }
        __syncthreads();
    }

    const float sum0 = warp_reduce_sum(accum0);
    const float sum1 = warp_reduce_sum(accum1);
    const float sum2 = warp_reduce_sum(accum2);
    const float sum3 = warp_reduce_sum(accum3);
    if (lane == 0 && output_col < N) {
        output[output_col] = __float2bfloat16_rn(sum0);
        output[N + output_col] = __float2bfloat16_rn(sum1);
        output[2 * N + output_col] = __float2bfloat16_rn(sum2);
        output[3 * N + output_col] = __float2bfloat16_rn(sum3);
    }
}

#endif /* GEMMA4_LISTENER_KERNELS_CUH */
