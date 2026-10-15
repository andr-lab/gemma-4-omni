/* Exact-shape CUDA primitives for the first Qwen native executor milestone. */
#include "qwen_talker_kernels.cuh"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <math.h>
#include <stdint.h>

namespace {

__global__ void copy_joint_inputs_kernel(
    const __nv_bfloat16 *prefix,
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= QWEN_NATIVE_HIDDEN_SIZE) return;
    hidden[index] = prefix[index];
    hidden[QWEN_NATIVE_HIDDEN_SIZE + index] = audio[index];
}

__global__ void copy_audio_input_kernel(
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < QWEN_NATIVE_HIDDEN_SIZE) hidden[index] = audio[index];
}

__global__ void write_position_ids_kernel(
    int64_t *destination,
    uint32_t prefix_position,
    uint32_t audio_position
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        destination[0] = static_cast<int64_t>(prefix_position);
        destination[1] = static_cast<int64_t>(audio_position);
    }
}

template <int DIM, int BLOCK_X>
__global__ void square_rows_kernel(
    const __nv_bfloat16 *input,
    float *squared,
    uint32_t rows
) {
    const uint32_t row = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (row >= rows) return;
    const __nv_bfloat16 *src = input + static_cast<size_t>(row) * DIM;
    float *dst = squared + static_cast<size_t>(row) * DIM;
    int index = static_cast<int>(lane);
    while (index * 4 + 3 < DIM) {
        const int base = index * 4;
        const float x0 = __bfloat162float(src[base + 0]);
        const float x1 = __bfloat162float(src[base + 1]);
        const float x2 = __bfloat162float(src[base + 2]);
        const float x3 = __bfloat162float(src[base + 3]);
        dst[base + 0] = x0 * x0;
        dst[base + 1] = x1 * x1;
        dst[base + 2] = x2 * x2;
        dst[base + 3] = x3 * x3;
        index += BLOCK_X;
    }
}

template <int DIM, int BLOCK_X>
__global__ void mean_rows_kernel(
    const float *squared,
    float *row_sum,
    float *row_mean,
    uint32_t rows
) {
    const uint32_t row = blockIdx.x;
    if (row >= rows) return;
    const float *src = squared + static_cast<size_t>(row) * DIM;
    float values[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int index = static_cast<int>(threadIdx.x);
    while (index * 4 + 3 < DIM) {
        const int base = index * 4;
        values[0] += src[base + 0];
        values[1] += src[base + 1];
        values[2] += src[base + 2];
        values[3] += src[base + 3];
        index += BLOCK_X;
    }
    values[0] += values[1];
    values[0] += values[2];
    values[0] += values[3];

    __shared__ float shared[BLOCK_X];
    shared[threadIdx.x] = values[0];
    __syncthreads();
    for (int offset = BLOCK_X / 2; offset >= 32; offset >>= 1) {
        if (threadIdx.x < static_cast<uint32_t>(offset)) {
            shared[threadIdx.x] += shared[threadIdx.x + offset];
        }
        __syncthreads();
    }
    float value = shared[threadIdx.x];
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (threadIdx.x == 0) {
        row_sum[row] = value;
        row_mean[row] = value * (1.0f / static_cast<float>(DIM));
    }
}

__global__ void finalize_rmsnorm_kernel(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    const float *row_mean,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    float eps
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(rows) * dim;
    if (index >= elements) return;
    const uint32_t row = index / dim;
    const float normalized_fp32 = __bfloat162float(input[index]) *
        rsqrtf(row_mean[row] + eps);
    /* Qwen3TTSRMSNorm has an observable BF16 boundary before multiplying the
     * BF16 affine weight: weight * hidden_states.to(input_dtype). */
    const __nv_bfloat16 normalized_bf16 = __float2bfloat16(normalized_fp32);
    const float weighted = __bfloat162float(normalized_bf16) *
        __bfloat162float(weight[index % dim]);
    output[index] = __float2bfloat16(weighted);
}

__global__ void pack_heads_kernel(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t rows,
    uint32_t heads,
    uint32_t head_dim
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(rows) * heads * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t head = static_cast<uint32_t>((index / head_dim) % heads);
    const uint32_t row = static_cast<uint32_t>(index / (static_cast<uint64_t>(head_dim) * heads));
    const uint64_t src_index = (static_cast<uint64_t>(row) * heads + head) * head_dim + d;
    const uint64_t dst_index = (static_cast<uint64_t>(head) * rows + row) * head_dim + d;
    destination[dst_index] = source[src_index];
}

__device__ __forceinline__ uint32_t rope_half_modality(uint32_t half_index) {
    /* Qwen's interleaved MRoPE copies modality 1 into 1:60:3 and modality 2
     * into 2:60:3; all remaining half-dim positions use modality 0. */
    if (half_index < 60u && (half_index % 3u) == 1u) return 1u;
    if (half_index < 60u && (half_index % 3u) == 2u) return 2u;
    return 0u;
}

__global__ void rope_interleaved_kernel(
    __nv_bfloat16 *values,
    const __nv_bfloat16 *inv_freq,
    uint32_t heads,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t prefix_position,
    uint32_t audio_position,
    const uint32_t *device_position
) {
    /* Each output element depends on its opposite half.  A block owns one
     * [head,row] and stages the complete original head in shared memory so
     * the in-place transform cannot race with its paired writer. */
    if (head_dim != 128u || threadIdx.x >= head_dim) return;
    const uint32_t head_row = blockIdx.x;
    if (head_row >= heads * rows) return;
    const uint32_t row = head_row % rows;
    const uint32_t head = head_row / rows;
    const uint32_t d = threadIdx.x;
    const uint64_t base = (static_cast<uint64_t>(head) * rows + row) * head_dim;
    __shared__ __nv_bfloat16 original[128];
    original[d] = values[base + d];
    __syncthreads();

    const uint32_t half_index = d < (head_dim / 2u) ? d : d - (head_dim / 2u);
    const uint32_t modality = rope_half_modality(half_index);
    (void)modality; /* all three supplied position streams are equal in TTS. */
    const uint32_t position = device_position
        ? *device_position
        : (row == 0u ? prefix_position : audio_position);
    const float frequency = static_cast<float>(position) *
        __bfloat162float(inv_freq[half_index]);
    const __nv_bfloat16 cos_bf16 = __float2bfloat16(cosf(frequency));
    const __nv_bfloat16 sin_bf16 = __float2bfloat16(sinf(frequency));
    const uint32_t pair = d < head_dim / 2u ? d + head_dim / 2u : d - head_dim / 2u;
    const float x = __bfloat162float(original[d]);
    const float rotated = __bfloat162float(original[pair]);
    const float signed_rotated = d < head_dim / 2u ? -rotated : rotated;
    const __nv_bfloat16 first_product = __float2bfloat16(
        x * __bfloat162float(cos_bf16));
    const __nv_bfloat16 second_product = __float2bfloat16(
        signed_rotated * __bfloat162float(sin_bf16));
    values[base + d] = __float2bfloat16(
        __bfloat162float(first_product) + __bfloat162float(second_product));
}

__global__ void write_joint_kv_kernel(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(heads) * 2u * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t row = static_cast<uint32_t>((index / head_dim) % 2u);
    const uint32_t head = static_cast<uint32_t>(index / (2u * head_dim));
    const uint32_t position = row == 0u ? prefix_position : audio_position;
    const uint64_t src_index = (static_cast<uint64_t>(head) * 2u + row) * head_dim + d;
    const uint64_t dst_index = (static_cast<uint64_t>(head) * max_cache_len + position) * head_dim + d;
    cache[dst_index] = source[src_index];
}

__global__ void write_audio_kv_kernel(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t audio_position,
    const uint32_t *device_position,
    uint32_t *device_error
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(heads) * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t head = static_cast<uint32_t>(index / head_dim);
    const uint32_t position = device_position ? *device_position : audio_position;
    if (position >= max_cache_len) {
        if (device_error && index == 0u) atomicExch(device_error, 1u);
        return;
    }
    const uint64_t destination =
        (static_cast<uint64_t>(head) * max_cache_len + position) * head_dim + d;
    cache[destination] = source[index];
}

__global__ void copy_cache_joint_rows_kernel(
    const __nv_bfloat16 *cache,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(2u) * heads * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t head = static_cast<uint32_t>((index / head_dim) % heads);
    const uint32_t row = static_cast<uint32_t>(index / (static_cast<uint64_t>(heads) * head_dim));
    const uint32_t position = row == 0u ? prefix_position : audio_position;
    const uint64_t source_index =
        (static_cast<uint64_t>(head) * max_cache_len + position) * head_dim + d;
    const uint64_t destination_index =
        (static_cast<uint64_t>(row) * heads + head) * head_dim + d;
    rows[destination_index] = cache[source_index];
}

__global__ void import_cache_prefix_kernel(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t source_max_len,
    uint32_t max_cache_len,
    uint32_t stable_length
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(heads) * stable_length * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t position = static_cast<uint32_t>((index / head_dim) % stable_length);
    const uint32_t head = static_cast<uint32_t>(index / (static_cast<uint64_t>(stable_length) * head_dim));
    const uint64_t src_index = (static_cast<uint64_t>(head) * source_max_len + position) * head_dim + d;
    const uint64_t dst_index = (static_cast<uint64_t>(head) * max_cache_len + position) * head_dim + d;
    destination[dst_index] = source[src_index];
}

__global__ void build_joint_mask_kernel(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    float normalized_prefix_log_gate
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2u * key_len) return;
    const uint32_t row = index / key_len;
    const uint32_t position = index % key_len;
    const bool stable = position < prefix_start;
    const bool visible_prefix = position >= prefix_start && position <= prefix_position;
    const bool audio = position >= audio_start && position <= audio_position;
    const bool prefix_allowed = visible_prefix && (row == 0u || prefix_attention_mode != 1u);
    const bool allowed = stable || prefix_allowed || (row == 1u && audio);
    float value = allowed ? 0.0f : -3.3895313892515355e38f;
    if (row == 1u && visible_prefix && prefix_attention_mode == 2u) {
        value = normalized_prefix_log_gate;
    }
    mask[index] = __float2bfloat16(value);
}

__global__ void build_audio_only_mask_kernel(
    __nv_bfloat16 *mask,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    float normalized_prefix_log_gate
) {
    const uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= key_len) return;
    const bool stable = position < prefix_start;
    const bool visible_prefix =
        position >= prefix_start && position <= visible_prefix_position;
    const bool causal_audio =
        position >= audio_start && position <= audio_position;
    const bool prefix_allowed = visible_prefix && prefix_attention_mode != 1u;
    const bool allowed = stable || prefix_allowed || causal_audio;
    float value = allowed ? 0.0f : -3.3895313892515355e38f;
    if (visible_prefix && prefix_attention_mode == 2u) {
        value = normalized_prefix_log_gate;
    }
    mask[position] = __float2bfloat16(value);
}

__global__ void build_audio_q1_causal_mask_cursor_kernel(
    __nv_bfloat16 *mask,
    uint32_t key_len,
    const uint32_t *device_position
) {
    const uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= key_len) return;
    const uint32_t cursor = *device_position;
    const float value = position <= cursor ? 0.0f : -3.3895313892515355e38f;
    mask[position] = __float2bfloat16(value);
}

__global__ void initialize_audio_q1_cursor_kernel(
    uint32_t *device_position,
    uint32_t *device_error,
    uint32_t initial_position
) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) {
        *device_position = initial_position;
        *device_error = 0u;
    }
}

__global__ void increment_audio_q1_cursor_kernel(uint32_t *device_position) {
    if (blockIdx.x == 0u && threadIdx.x == 0u) ++(*device_position);
}

__global__ void repeat_kv_kernel(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t kv_heads,
    uint32_t repeats,
    uint32_t key_len,
    uint32_t source_stride_len,
    uint32_t head_dim
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(kv_heads) * repeats * key_len * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t position = static_cast<uint32_t>((index / head_dim) % key_len);
    const uint32_t repeat = static_cast<uint32_t>((index / (static_cast<uint64_t>(key_len) * head_dim)) % repeats);
    const uint32_t kv_head = static_cast<uint32_t>(index / (static_cast<uint64_t>(repeats) * key_len * head_dim));
    const uint64_t source_index = (static_cast<uint64_t>(kv_head) * source_stride_len + position) * head_dim + d;
    const uint64_t destination_head = static_cast<uint64_t>(kv_head) * repeats + repeat;
    const uint64_t destination_index = (destination_head * key_len + position) * head_dim + d;
    destination[destination_index] = source[source_index];
}

__device__ __forceinline__ float reduce_max_256(float value, float *shared) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    if (lane == 0u) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8u ? shared[lane] : -1.0e30f;
    if (warp == 0u) {
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
        }
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__device__ __forceinline__ float reduce_sum_256(float value, float *shared) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    if (lane == 0u) shared[warp] = value;
    __syncthreads();
    value = threadIdx.x < 8u ? shared[lane] : 0.0f;
    if (warp == 0u) {
        for (uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
    }
    return __shfl_sync(0xffffffffu, value, 0);
}

__global__ void qwen_attention_softmax_kernel(
    float *scores,
    const __nv_bfloat16 *mask,
    uint32_t query_rows,
    uint32_t key_len,
    uint32_t score_stride
) {
    const uint32_t head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (head >= QWEN_NATIVE_NUM_ATTN_HEADS || row >= query_rows || key_len == 0u ||
        key_len > score_stride) return;

    __shared__ float reduction[8];
    __shared__ float row_max;
    __shared__ float inverse_sum;
    float *row_scores = scores +
        (static_cast<size_t>(head) * query_rows + row) * score_stride;
    for (uint32_t key_index = threadIdx.x; key_index < key_len; key_index += blockDim.x) {
        const float additive_mask = __bfloat162float(mask[
            static_cast<size_t>(row) * key_len + key_index]);
        if (additive_mask < -1.0e20f) row_scores[key_index] = -1.0e30f;
        else row_scores[key_index] += additive_mask;
    }
    __syncthreads();

    float local_max = -1.0e30f;
    for (uint32_t key_index = threadIdx.x; key_index < key_len; key_index += blockDim.x) {
        local_max = fmaxf(local_max, row_scores[key_index]);
    }
    const float maximum = reduce_max_256(local_max, reduction);
    if (threadIdx.x == 0u) row_max = maximum;
    __syncthreads();

    float local_sum = 0.0f;
    for (uint32_t key_index = threadIdx.x; key_index < key_len; key_index += blockDim.x) {
        const float probability = expf(row_scores[key_index] - row_max);
        row_scores[key_index] = probability;
        local_sum += probability;
    }
    const float total = reduce_sum_256(local_sum, reduction);
    if (threadIdx.x == 0u) inverse_sum = total > 0.0f ? 1.0f / total : 0.0f;
    __syncthreads();
    for (uint32_t key_index = threadIdx.x; key_index < key_len; key_index += blockDim.x) {
        row_scores[key_index] *= inverse_sum;
    }
}

__global__ void qwen_attention_output_kernel(
    const float *scores,
    const __nv_bfloat16 *value,
    __nv_bfloat16 *output,
    uint32_t query_rows,
    uint32_t key_len,
    uint32_t score_stride
) {
    const uint32_t head = blockIdx.x;
    const uint32_t row = blockIdx.y;
    if (head >= QWEN_NATIVE_NUM_ATTN_HEADS || row >= query_rows) return;
    const float *row_scores = scores +
        (static_cast<size_t>(head) * query_rows + row) * score_stride;
    const __nv_bfloat16 *value_head = value +
        static_cast<size_t>(head) * key_len * QWEN_NATIVE_HEAD_DIM;
    __nv_bfloat16 *output_row = output +
        (static_cast<size_t>(head) * query_rows + row) * QWEN_NATIVE_HEAD_DIM;
    for (uint32_t d = threadIdx.x; d < QWEN_NATIVE_HEAD_DIM; d += blockDim.x) {
        float result = 0.0f;
        for (uint32_t key_index = 0; key_index < key_len; ++key_index) {
            result += row_scores[key_index] *
                __bfloat162float(value_head[
                    static_cast<size_t>(key_index) * QWEN_NATIVE_HEAD_DIM + d]);
        }
        output_row[d] = __float2bfloat16(result);
    }
}

__global__ void attention_to_rows_kernel(
    const __nv_bfloat16 *attention,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t q_len,
    uint32_t head_dim
) {
    const uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t elements = static_cast<uint64_t>(q_len) * heads * head_dim;
    if (index >= elements) return;
    const uint32_t d = static_cast<uint32_t>(index % head_dim);
    const uint32_t row = static_cast<uint32_t>((index / head_dim) % q_len);
    const uint32_t head = static_cast<uint32_t>(index / (static_cast<uint64_t>(q_len) * head_dim));
    const uint64_t src_index = (static_cast<uint64_t>(head) * q_len + row) * head_dim + d;
    const uint64_t dst_index = (static_cast<uint64_t>(row) * heads + head) * head_dim + d;
    rows[dst_index] = attention[src_index];
}

__global__ void add_rows_kernel(
    const __nv_bfloat16 *left,
    const __nv_bfloat16 *right,
    __nv_bfloat16 *output,
    uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    output[index] = __float2bfloat16(
        __bfloat162float(left[index]) + __bfloat162float(right[index]));
}

__global__ void silu_mul_kernel(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    __nv_bfloat16 *output,
    uint32_t elements
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const float gate_value = __bfloat162float(gate[index]);
    const float up_value = __bfloat162float(up[index]);
    /* torch.nn.functional.silu(BF16) computes the unary SiLU in FP32 and
     * rounds that result to BF16 before the following BF16 product. */
    const __nv_bfloat16 silu_bf16 = __float2bfloat16(
        gate_value / (1.0f + expf(-gate_value)));
    output[index] = __hmul(silu_bf16, up[index]);
}

__global__ void copy_audio_row_kernel(
    const __nv_bfloat16 *hidden,
    __nv_bfloat16 *output,
    uint32_t row,
    uint32_t dim
) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= dim) return;
    output[index] = hidden[static_cast<size_t>(row) * dim + index];
}

int check_launch() {
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

} // namespace

int qwen_write_position_ids(
    int64_t *destination,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
) {
    if (!destination) return -1;
    write_position_ids_kernel<<<1, 1, 0, stream>>>(destination, prefix_position, audio_position);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

int qwen_copy_joint_inputs(
    const __nv_bfloat16 *prefix,
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden,
    cudaStream_t stream
) {
    copy_joint_inputs_kernel<<<(QWEN_NATIVE_HIDDEN_SIZE + 255u) / 256u, 256, 0, stream>>>(prefix, audio, hidden);
    return check_launch();
}

int qwen_copy_audio_input(
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden,
    cudaStream_t stream
) {
    if (!audio || !hidden) return -1;
    copy_audio_input_kernel<<<(QWEN_NATIVE_HIDDEN_SIZE + 255u) / 256u, 256, 0, stream>>>(audio, hidden);
    return check_launch();
}

int qwen_rmsnorm_exact(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    float *squared_workspace,
    float *row_sum_workspace,
    cudaStream_t stream
) {
    if (!input || !weight || !output || !squared_workspace || !row_sum_workspace || rows == 0u) return -1;
    if (dim == QWEN_NATIVE_HIDDEN_SIZE) {
        square_rows_kernel<QWEN_NATIVE_HIDDEN_SIZE, 64><<<rows, 64, 0, stream>>>(input, squared_workspace, rows);
        if (check_launch() != 0) return -3;
        mean_rows_kernel<QWEN_NATIVE_HIDDEN_SIZE, 64><<<rows, 64, 0, stream>>>(squared_workspace, row_sum_workspace, row_sum_workspace + rows, rows);
    } else if (dim == QWEN_NATIVE_HEAD_DIM) {
        square_rows_kernel<QWEN_NATIVE_HEAD_DIM, 32><<<rows, 32, 0, stream>>>(input, squared_workspace, rows);
        if (check_launch() != 0) return -3;
        mean_rows_kernel<QWEN_NATIVE_HEAD_DIM, 32><<<rows, 32, 0, stream>>>(squared_workspace, row_sum_workspace, row_sum_workspace + rows, rows);
    } else {
        return -2;
    }
    if (check_launch() != 0) return -3;
    const uint64_t elements = static_cast<uint64_t>(rows) * dim;
    finalize_rmsnorm_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        input, weight, row_sum_workspace + rows, output, rows, dim, eps);
    return check_launch();
}

int qwen_pack_heads(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t rows,
    uint32_t heads,
    uint32_t head_dim,
    cudaStream_t stream
) {
    const uint64_t elements = static_cast<uint64_t>(rows) * heads * head_dim;
    pack_heads_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(source, destination, rows, heads, head_dim);
    return check_launch();
}

int qwen_apply_rope_interleaved(
    __nv_bfloat16 *q_or_k,
    const __nv_bfloat16 *inv_freq,
    uint32_t heads,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
) {
    rope_interleaved_kernel<<<heads * rows, head_dim, 0, stream>>>(
        q_or_k, inv_freq, heads, rows, head_dim, prefix_position, audio_position, nullptr);
    return check_launch();
}

int qwen_apply_rope_interleaved_cursor(
    __nv_bfloat16 *q_or_k,
    const __nv_bfloat16 *inv_freq,
    uint32_t heads,
    uint32_t rows,
    uint32_t head_dim,
    const uint32_t *device_position,
    cudaStream_t stream
) {
    if (!q_or_k || !inv_freq || !device_position || rows != 1u) return -1;
    rope_interleaved_kernel<<<heads * rows, head_dim, 0, stream>>>(
        q_or_k, inv_freq, heads, rows, head_dim, 0u, 0u, device_position);
    return check_launch();
}

int qwen_write_joint_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
) {
    const uint64_t elements = static_cast<uint64_t>(heads) * 2u * head_dim;
    write_joint_kv_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        source, cache, heads, head_dim, max_cache_len, prefix_position, audio_position);
    return check_launch();
}

int qwen_write_audio_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t audio_position,
    cudaStream_t stream
) {
    if (!source || !cache || heads == 0u || head_dim == 0u || audio_position >= max_cache_len) return -1;
    const uint64_t elements = static_cast<uint64_t>(heads) * head_dim;
    write_audio_kv_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        source, cache, heads, head_dim, max_cache_len, audio_position, nullptr, nullptr);
    return check_launch();
}

int qwen_write_audio_kv_cursor(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    const uint32_t *device_position,
    uint32_t *device_error,
    cudaStream_t stream
) {
    if (!source || !cache || !device_position || !device_error || heads == 0u || head_dim == 0u || max_cache_len == 0u) return -1;
    const uint64_t elements = static_cast<uint64_t>(heads) * head_dim;
    write_audio_kv_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        source, cache, heads, head_dim, max_cache_len, 0u, device_position, device_error);
    return check_launch();
}

int qwen_import_cache_prefix(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t source_max_len,
    uint32_t max_cache_len,
    uint32_t stable_length,
    cudaStream_t stream
) {
    if (stable_length == 0u) return 0;
    const uint64_t elements = static_cast<uint64_t>(heads) * stable_length * head_dim;
    import_cache_prefix_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        source, destination, heads, head_dim, source_max_len, max_cache_len, stable_length);
    return check_launch();
}

int qwen_copy_cache_joint_rows(
    const __nv_bfloat16 *cache,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
) {
    if (!cache || !rows || heads == 0u || head_dim == 0u ||
        prefix_position >= max_cache_len || audio_position >= max_cache_len) return -1;
    const uint64_t elements = static_cast<uint64_t>(2u) * heads * head_dim;
    copy_cache_joint_rows_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        cache, rows, heads, head_dim, max_cache_len, prefix_position, audio_position);
    return check_launch();
}

int qwen_build_joint_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
) {
    if (!mask || key_len == 0u || key_len > max_cache_len || prefix_attention_mode > 2u ||
        !isfinite(prefix_log_gate) ||
        (prefix_attention_mode == 2u && visible_prefix_count == 0u)) return -1;
    const float normalized_gate = prefix_attention_mode == 2u
        ? prefix_log_gate - logf(static_cast<float>(visible_prefix_count))
        : 0.0f;
    build_joint_mask_kernel<<<(2u * key_len + 255u) / 256u, 256, 0, stream>>>(
        mask, max_cache_len, key_len, prefix_start, audio_start, prefix_position, audio_position,
        prefix_attention_mode, normalized_gate);
    return check_launch();
}

int qwen_build_audio_only_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
) {
    if (!mask || key_len == 0u || key_len > max_cache_len || prefix_attention_mode > 2u ||
        visible_prefix_position < prefix_start || visible_prefix_position >= audio_start ||
        audio_position < audio_start || audio_position >= key_len || !isfinite(prefix_log_gate) ||
        (prefix_attention_mode == 2u && visible_prefix_count == 0u)) return -1;
    const float normalized_gate = prefix_attention_mode == 2u
        ? prefix_log_gate - logf(static_cast<float>(visible_prefix_count))
        : 0.0f;
    build_audio_only_mask_kernel<<<(key_len + 255u) / 256u, 256, 0, stream>>>(
        mask, key_len, prefix_start, audio_start, visible_prefix_position, audio_position,
        prefix_attention_mode, normalized_gate);
    return check_launch();
}

int qwen_build_audio_q1_causal_mask_cursor(
    __nv_bfloat16 *mask,
    uint32_t key_len,
    const uint32_t *device_position,
    cudaStream_t stream
) {
    if (!mask || !device_position || key_len == 0u) return -1;
    build_audio_q1_causal_mask_cursor_kernel<<<(key_len + 255u) / 256u, 256, 0, stream>>>(
        mask, key_len, device_position);
    return check_launch();
}

int qwen_initialize_audio_q1_cursor(
    uint32_t *device_position,
    uint32_t *device_error,
    uint32_t initial_position,
    cudaStream_t stream
) {
    if (!device_position || !device_error) return -1;
    initialize_audio_q1_cursor_kernel<<<1, 1, 0, stream>>>(device_position, device_error, initial_position);
    return check_launch();
}

int qwen_increment_audio_q1_cursor(uint32_t *device_position, cudaStream_t stream) {
    if (!device_position) return -1;
    increment_audio_q1_cursor_kernel<<<1, 1, 0, stream>>>(device_position);
    return check_launch();
}

int qwen_build_terminal_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
) {
    return qwen_build_audio_only_mask(
        mask, max_cache_len, key_len, prefix_start, audio_start,
        visible_prefix_position, audio_position, prefix_attention_mode,
        visible_prefix_count, prefix_log_gate, stream);
}

int qwen_repeat_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t kv_heads,
    uint32_t repeats,
    uint32_t key_len,
    uint32_t source_stride_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    if (!source || !destination || kv_heads == 0u || repeats == 0u || key_len == 0u || head_dim == 0u) return -1;
    const uint64_t elements = static_cast<uint64_t>(kv_heads) * repeats * key_len * head_dim;
    repeat_kv_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        source, destination, kv_heads, repeats, key_len, source_stride_len, head_dim);
    return check_launch();
}

int qwen_attention_gqa(
    const __nv_bfloat16 *query,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    const __nv_bfloat16 *mask,
    __nv_bfloat16 *output,
    float scale,
    uint32_t query_rows,
    uint32_t key_len,
    uint32_t score_stride,
    cublasHandle_t blas,
    float *scores,
    cudaStream_t stream
) {
    if (!query || !key || !value || !mask || !output || !scores || !blas ||
        (query_rows != 1u && query_rows != 2u) || key_len == 0u ||
        key_len > score_stride || !isfinite(scale)) return -1;
    if (cublasSetStream(blas, stream) != CUBLAS_STATUS_SUCCESS) return -2;
    const float alpha_qk = scale;
    const float beta = 0.0f;
    const long long query_stride =
        static_cast<long long>(query_rows * QWEN_NATIVE_HEAD_DIM);
    const long long key_stride = static_cast<long long>(key_len * QWEN_NATIVE_HEAD_DIM);
    const long long score_batch_stride = static_cast<long long>(query_rows * score_stride);
    const cublasStatus_t qk_status = cublasGemmStridedBatchedEx(
        blas, CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(key_len), static_cast<int>(query_rows), QWEN_NATIVE_HEAD_DIM,
        &alpha_qk,
        key, CUDA_R_16BF, QWEN_NATIVE_HEAD_DIM, key_stride,
        query, CUDA_R_16BF, QWEN_NATIVE_HEAD_DIM, query_stride,
        &beta,
        scores, CUDA_R_32F, static_cast<int>(score_stride), score_batch_stride,
        QWEN_NATIVE_NUM_ATTN_HEADS, CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (qk_status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[qwen_attention] QK cublas status=%d key_len=%u score_stride=%u\\n",
                static_cast<int>(qk_status), key_len, score_stride);
        return -3;
    }

    dim3 grid(QWEN_NATIVE_NUM_ATTN_HEADS, query_rows, 1u);
    qwen_attention_softmax_kernel<<<grid, 256, 0, stream>>>(
        scores, mask, query_rows, key_len, score_stride);
    if (check_launch() != 0) return -4;

    qwen_attention_output_kernel<<<grid, 256, 0, stream>>>(
        scores, value, output, query_rows, key_len, score_stride);
    return check_launch();
}

int qwen_attention_to_rows(
    const __nv_bfloat16 *attention,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t q_len,
    uint32_t head_dim,
    cudaStream_t stream
) {
    const uint64_t elements = static_cast<uint64_t>(q_len) * heads * head_dim;
    attention_to_rows_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(
        attention, rows, heads, q_len, head_dim);
    return check_launch();
}

int qwen_add_rows(
    const __nv_bfloat16 *left,
    const __nv_bfloat16 *right,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    cudaStream_t stream
) {
    const uint64_t elements = static_cast<uint64_t>(rows) * dim;
    add_rows_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(left, right, output, static_cast<uint32_t>(elements));
    return check_launch();
}

int qwen_silu_mul(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    __nv_bfloat16 *output,
    uint32_t elements,
    cudaStream_t stream
) {
    silu_mul_kernel<<<(elements + 255u) / 256u, 256, 0, stream>>>(gate, up, output, elements);
    return check_launch();
}

int qwen_copy_audio_row(
    const __nv_bfloat16 *hidden,
    __nv_bfloat16 *output,
    uint32_t row,
    uint32_t dim,
    cudaStream_t stream
) {
    copy_audio_row_kernel<<<(dim + 255u) / 256u, 256, 0, stream>>>(hidden, output, row, dim);
    return check_launch();
}
