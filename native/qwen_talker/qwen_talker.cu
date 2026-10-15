/* Native incremental Qwen3-TTS Talker backbone executor. */
#include "qwen_talker.h"
#include "qwen_talker_kernels.cuh"
#include "qwen_talker_aten_attention.h"
#include "qwen_talker_aten_rmsnorm.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <new>
#include <type_traits>

static constexpr size_t QWEN_NATIVE_Q1_CUBLAS_WORKSPACE_BYTES = 32u * 1024u * 1024u;

enum class qwen_execution_mode : uint32_t {
    none = 0u,
    joint_q2 = 1u,
    recurrent_audio_q1 = 2u,
};

struct qwen_talker_ctx {
    qwen_talker_config config{};
    qwen_talker_weights weights{};
    int device = -1;
    cublasHandle_t blas = nullptr;
    void *q1_cublas_workspace = nullptr;
    cudaStream_t q1_graph_stream = nullptr;
    bool q1_graph_stream_bound = false;
    bool q1_graph_capture_active = false;
    bool q1_graph_captured = false;
    uint64_t q1_graph_replays = 0;
    qwen_talker_attention_graph_state *attention_graph = nullptr;
    qwen_talker_rmsnorm_graph_state *rmsnorm_graph = nullptr;
    qwen_talker_rmsnorm_graph_state *rmsnorm_graph_q1 = nullptr;
    unsigned char *arena = nullptr;
    size_t arena_bytes = 0;

    __nv_bfloat16 *hidden_a = nullptr;
    __nv_bfloat16 *hidden_b = nullptr;
    __nv_bfloat16 *norm = nullptr;
    __nv_bfloat16 *rms_graph_input = nullptr;
    __nv_bfloat16 *rms_graph_weight = nullptr;
    __nv_bfloat16 *rms_graph_output = nullptr;
    __nv_bfloat16 *q_linear = nullptr;
    __nv_bfloat16 *k_linear = nullptr;
    __nv_bfloat16 *v_linear = nullptr;
    __nv_bfloat16 *q_heads = nullptr;
    __nv_bfloat16 *k_heads = nullptr;
    __nv_bfloat16 *v_heads = nullptr;
    __nv_bfloat16 *attention_key = nullptr;
    __nv_bfloat16 *attention_value = nullptr;
    __nv_bfloat16 *attention = nullptr;
    __nv_bfloat16 *attention_rows = nullptr;
    __nv_bfloat16 *gate = nullptr;
    __nv_bfloat16 *up = nullptr;
    __nv_bfloat16 *mlp = nullptr;
    __nv_bfloat16 *mask = nullptr;
    uint32_t *q1_device_position = nullptr;
    uint32_t *q1_device_error = nullptr;
    __nv_bfloat16 *trace_prefix_input = nullptr;
    __nv_bfloat16 *trace_audio_input = nullptr;
    __nv_bfloat16 *trace_output_hidden = nullptr;
    int64_t *trace_position_ids = nullptr;
    float *squared = nullptr;
    float *row_stats = nullptr;
    float *debug_input_rms_mean = nullptr;
    __nv_bfloat16 *debug_taps[QWEN_NATIVE_DEBUG_TAP_COUNT]{};
    __nv_bfloat16 *cache_k[QWEN_NATIVE_NUM_LAYERS]{};
    __nv_bfloat16 *cache_v[QWEN_NATIVE_NUM_LAYERS]{};

    bool imported = false;
    qwen_execution_mode mode = qwen_execution_mode::none;
    uint32_t stable_length = 0;
    uint32_t prefix_start = 0;
    uint32_t audio_start = 0;
    uint32_t current_prefix_position = 0;
    uint32_t current_audio_position = 0;
    uint32_t next_prefix_position = 0;
    uint32_t next_audio_position = 0;
    bool position_contract_set = false;
    bool virtual_positions = false;
    uint32_t prefix_attention_mode = 0;
    uint32_t next_prefix_rope_position = 0;
    uint32_t next_audio_rope_position = 0;
    uint32_t audio_q1_limit_position = 0;
    uint64_t joint_steps = 0;
    uint64_t segment_joint_steps = 0;
    uint64_t segment_terminal_queries = 0;
    uint64_t terminal_prefix_cache_appends = 0;
    uint64_t terminal_audio_cache_appends = 0;
    uint32_t terminal_query_position = UINT32_MAX;
    uint32_t terminal_visible_prefix_position = UINT32_MAX;
    bool terminal_query_started = false;
    bool terminal_query_done = false;
    uint64_t trace_completed_steps = 0;
    uint64_t cache_imports = 0;
    cudaStream_t last_stream = nullptr;
    bool has_last_stream = false;
    uint32_t last_cache_layer = UINT32_MAX;
    bool last_cache_all_layers = false;
    bool trace_enabled = false;
    uint64_t trace_generation = 0;
    uint64_t trace_expected_steps = 0;
    uint32_t trace_cache_layers_written = 0;
    uint8_t trace_previous_chain[QWEN_NATIVE_TRACE_CHAIN_BYTES]{};
    uint8_t trace_chain[QWEN_NATIVE_TRACE_CHAIN_BYTES]{};
    int32_t last_error = 0;
};

namespace {

constexpr size_t ALIGNMENT = 256;
constexpr uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotr32(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

struct Sha256 {
    uint32_t state[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };
    uint8_t block[64]{};
    size_t block_size = 0;
    uint64_t bit_length = 0;

    void transform() {
        uint32_t schedule[64]{};
        for (uint32_t i = 0; i < 16u; ++i) {
            schedule[i] = (static_cast<uint32_t>(block[4u * i]) << 24u) |
                (static_cast<uint32_t>(block[4u * i + 1u]) << 16u) |
                (static_cast<uint32_t>(block[4u * i + 2u]) << 8u) |
                static_cast<uint32_t>(block[4u * i + 3u]);
        }
        for (uint32_t i = 16u; i < 64u; ++i) {
            const uint32_t s0 = rotr32(schedule[i - 15u], 7u) ^ rotr32(schedule[i - 15u], 18u) ^ (schedule[i - 15u] >> 3u);
            const uint32_t s1 = rotr32(schedule[i - 2u], 17u) ^ rotr32(schedule[i - 2u], 19u) ^ (schedule[i - 2u] >> 10u);
            schedule[i] = schedule[i - 16u] + s0 + schedule[i - 7u] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (uint32_t i = 0; i < 64u; ++i) {
            const uint32_t s1 = rotr32(e, 6u) ^ rotr32(e, 11u) ^ rotr32(e, 25u);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + choice + SHA256_K[i] + schedule[i];
            const uint32_t s0 = rotr32(a, 2u) ^ rotr32(a, 13u) ^ rotr32(a, 22u);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + majority;
            h = g; g = f; f = e; e = d + temp1;
            d = c; c = b; b = a; a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
        block_size = 0;
    }

    void update(const void *data, size_t size) {
        const auto *bytes = static_cast<const uint8_t *>(data);
        bit_length += static_cast<uint64_t>(size) * 8u;
        while (size != 0u) {
            const size_t take = std::min(size, sizeof(block) - block_size);
            std::memcpy(block + block_size, bytes, take);
            block_size += take;
            bytes += take;
            size -= take;
            if (block_size == sizeof(block)) transform();
        }
    }

    void update_u32(uint32_t value) {
        const uint8_t bytes[4] = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
            static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
        };
        update(bytes, sizeof(bytes));
    }

    void update_u64(uint64_t value) {
        const uint8_t bytes[8] = {
            static_cast<uint8_t>(value), static_cast<uint8_t>(value >> 8u),
            static_cast<uint8_t>(value >> 16u), static_cast<uint8_t>(value >> 24u),
            static_cast<uint8_t>(value >> 32u), static_cast<uint8_t>(value >> 40u),
            static_cast<uint8_t>(value >> 48u), static_cast<uint8_t>(value >> 56u),
        };
        update(bytes, sizeof(bytes));
    }

    void final(uint8_t output[QWEN_NATIVE_TRACE_CHAIN_BYTES]) {
        const uint64_t original_bits = bit_length;
        const uint8_t one = 0x80u;
        update(&one, 1u);
        const uint8_t zero = 0u;
        while (block_size != 56u) update(&zero, 1u);
        for (int index = 7; index >= 0; --index) {
            const uint8_t byte = static_cast<uint8_t>(original_bits >> (8u * static_cast<uint32_t>(index)));
            update(&byte, 1u);
        }
        for (uint32_t index = 0; index < 8u; ++index) {
            output[4u * index] = static_cast<uint8_t>(state[index] >> 24u);
            output[4u * index + 1u] = static_cast<uint8_t>(state[index] >> 16u);
            output[4u * index + 2u] = static_cast<uint8_t>(state[index] >> 8u);
            output[4u * index + 3u] = static_cast<uint8_t>(state[index]);
        }
    }
};

void update_native_trace_chain(
    const qwen_talker_ctx *ctx,
    const uint8_t previous[QWEN_NATIVE_TRACE_CHAIN_BYTES],
    uint8_t output[QWEN_NATIVE_TRACE_CHAIN_BYTES]
) {
    Sha256 digest;
    digest.update(previous, QWEN_NATIVE_TRACE_CHAIN_BYTES);
    static constexpr char DOMAIN[] = "qwen-native-step-chain-v1";
    digest.update(DOMAIN, sizeof(DOMAIN) - 1u);
    digest.update_u64(ctx->trace_generation);
    digest.update_u64(ctx->trace_completed_steps - 1u);
    digest.update_u64(ctx->trace_expected_steps);
    digest.update_u32(ctx->stable_length);
    digest.update_u32(ctx->prefix_start);
    digest.update_u32(ctx->audio_start);
    digest.update_u32(ctx->current_prefix_position);
    digest.update_u32(ctx->current_audio_position);
    digest.update_u32(ctx->trace_cache_layers_written);
    digest.update_u32(static_cast<uint32_t>(ctx->last_error));
    digest.update_u64(ctx->joint_steps);
    digest.update_u64(ctx->cache_imports);
    digest.final(output);
}
constexpr uint32_t MAX_RMS_ROWS = QWEN_NATIVE_NUM_ATTN_HEADS * 2u;

size_t aligned_size(size_t value) {
    return (value + ALIGNMENT - 1u) & ~(ALIGNMENT - 1u);
}

struct ArenaBuilder {
    size_t offset = 0;

    template <typename T>
    T *take(size_t count) {
        offset = aligned_size(offset);
        T *result = reinterpret_cast<T *>(offset);
        offset += sizeof(T) * count;
        return result;
    }
};

template <typename T>
T *bind_arena(unsigned char *base, T *relative) {
    return reinterpret_cast<T *>(base + reinterpret_cast<uintptr_t>(relative));
}

bool is_device_pointer(const void *ptr, int device) {
    if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 1u) != 0u) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) return false;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}

size_t dtype_bytes(uint32_t dtype) {
    if (dtype == QWEN_NATIVE_META_BF16) return sizeof(__nv_bfloat16);
    if (dtype == QWEN_NATIVE_META_FP32) return sizeof(float);
    if (dtype == QWEN_NATIVE_META_INT64) return sizeof(int64_t);
    return 0u;
}

bool valid_meta(
    const void *ptr,
    const qwen_tensor_meta *meta,
    uint32_t dtype,
    uint32_t ndim,
    const uint64_t *dims,
    int device
) {
    if (!ptr || !meta || meta->ptr != ptr || meta->dtype != dtype || meta->ndim != ndim ||
        meta->contiguous != 1u || meta->reserved != 0u || !is_device_pointer(ptr, device)) return false;
    uint64_t running = 1;
    for (int index = static_cast<int>(ndim) - 1; index >= 0; --index) {
        if (meta->dims[index] != dims[index] || meta->strides[index] != running || dims[index] == 0u) return false;
        running *= dims[index];
    }
    for (uint32_t index = ndim; index < QWEN_NATIVE_META_MAX_RANK; ++index) {
        if (meta->dims[index] != 0u || meta->strides[index] != 0u) return false;
    }
    const size_t element_bytes = dtype_bytes(dtype);
    return element_bytes != 0u && running <= UINT64_MAX / element_bytes &&
        meta->capacity_bytes >= running * element_bytes;
}

bool valid_vector_meta(const void *ptr, const qwen_tensor_meta *meta, uint32_t length, int device) {
    const uint64_t dims[] = {length};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 1u, dims, device);
}

bool valid_matrix_meta(const void *ptr, const qwen_tensor_meta *meta, uint32_t rows, uint32_t cols, int device) {
    const uint64_t dims[] = {rows, cols};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 2u, dims, device);
}

bool valid_embedding_meta(const void *ptr, const qwen_tensor_meta *meta, int device) {
    const uint64_t dims[] = {1u, 1u, QWEN_NATIVE_HIDDEN_SIZE};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 3u, dims, device);
}

bool valid_hidden_rows_meta(const void *ptr, const qwen_tensor_meta *meta, int device) {
    const uint64_t dims[] = {1u, 2u, QWEN_NATIVE_HIDDEN_SIZE};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 3u, dims, device);
}

bool valid_trace_mask_meta(const void *ptr, const qwen_tensor_meta *meta, uint32_t max_cache_len, int device) {
    const uint64_t dims[] = {2u, max_cache_len};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 2u, dims, device);
}

bool valid_trace_position_meta(const void *ptr, const qwen_tensor_meta *meta, int device) {
    const uint64_t dims[] = {2u};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_INT64, 1u, dims, device);
}

bool valid_trace_cache_meta(const void *ptr, const qwen_tensor_meta *meta, int device) {
    const uint64_t dims[] = {
        QWEN_NATIVE_NUM_LAYERS, 2u, QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM
    };
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 4u, dims, device);
}

bool valid_cache_rows_meta(const void *ptr, const qwen_tensor_meta *meta, int device) {
    const uint64_t dims[] = {2u, QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 3u, dims, device);
}

bool valid_cache_meta(
    const void *ptr,
    const qwen_tensor_meta *meta,
    uint32_t heads,
    uint32_t min_length,
    int device
) {
    if (!ptr || !meta || meta->ptr != ptr || meta->dtype != QWEN_NATIVE_META_BF16 ||
        meta->ndim != 4u || meta->contiguous != 1u || meta->reserved != 0u ||
        meta->dims[0] != 1u || meta->dims[1] != heads || meta->dims[2] < min_length ||
        meta->dims[3] != QWEN_NATIVE_HEAD_DIM || !is_device_pointer(ptr, device)) return false;
    const uint64_t expected[] = {1u, heads, meta->dims[2], QWEN_NATIVE_HEAD_DIM};
    return valid_meta(ptr, meta, QWEN_NATIVE_META_BF16, 4u, expected, device);
}

qwen_tensor_meta make_meta(
    const void *ptr,
    uint32_t dtype,
    uint32_t ndim,
    const uint64_t *dims
) {
    qwen_tensor_meta meta{};
    meta.ptr = ptr;
    meta.dtype = dtype;
    meta.ndim = ndim;
    meta.contiguous = 1u;
    uint64_t stride = 1;
    for (int index = static_cast<int>(ndim) - 1; index >= 0; --index) {
        meta.dims[index] = dims[index];
        meta.strides[index] = stride;
        stride *= dims[index];
    }
    const size_t element_bytes = dtype_bytes(dtype);
    meta.capacity_bytes = element_bytes != 0u && stride <= UINT64_MAX / element_bytes
        ? stride * element_bytes : 0u;
    return meta;
}

int record(qwen_talker_ctx *ctx, int code) {
    if (code != 0 && ctx) ctx->last_error = code;
    return code;
}

bool stream_matches(const qwen_talker_ctx *ctx, cudaStream_t stream) {
    return !ctx->has_last_stream || ctx->last_stream == stream;
}

void remember_stream(qwen_talker_ctx *ctx, cudaStream_t stream) {
    ctx->last_stream = stream;
    ctx->has_last_stream = true;
}

int check_cuda(qwen_talker_ctx *ctx) {
    const cudaError_t error = cudaPeekAtLastError();
    return error == cudaSuccess ? 0 : record(ctx, -100 - static_cast<int>(error));
}

int check_cublas(qwen_talker_ctx *ctx, cublasStatus_t status) {
    return status == CUBLAS_STATUS_SUCCESS ? 0 : record(ctx, -200 - static_cast<int>(status));
}

int snapshot_rms_mean(
    qwen_talker_ctx *ctx,
    cudaStream_t stream
) {
    if (!stream_matches(ctx, stream)) return record(ctx, -304);
    if (cudaMemcpyAsync(ctx->debug_input_rms_mean, ctx->row_stats + 2u,
                        2u * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return record(ctx, -302);
    }
    return 0;
}

int snapshot(
    qwen_talker_ctx *ctx,
    uint32_t tap_id,
    const __nv_bfloat16 *source,
    uint32_t elements,
    cudaStream_t stream
) {
    if (!stream_matches(ctx, stream)) return record(ctx, -305);
    if (tap_id >= QWEN_NATIVE_DEBUG_TAP_COUNT || elements > 2u * std::max(
            QWEN_NATIVE_HIDDEN_SIZE, QWEN_NATIVE_INTERMEDIATE_SIZE)) return record(ctx, -300);
    if (cudaMemcpyAsync(ctx->debug_taps[tap_id], source,
                        static_cast<size_t>(elements) * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -301);
    return 0;
}

int linear(
    qwen_talker_ctx *ctx,
    const __nv_bfloat16 *input,
    const void *weight,
    __nv_bfloat16 *output,
    int m,
    int k,
    int n,
    cudaStream_t stream
) {
    if (ctx->q1_graph_stream_bound) {
        if (stream != ctx->q1_graph_stream) return -1;
    } else if (check_cublas(ctx, cublasSetStream(ctx->blas, stream)) != 0) {
        return -1;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    /* Row-major C[m,n] = A[m,k] * W[n,k]^T is represented as the
     * column-major product W[k,n]^T * A[k,m]. */
    return check_cublas(ctx, cublasGemmEx(
        ctx->blas,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        weight,
        CUDA_R_16BF,
        k,
        input,
        CUDA_R_16BF,
        k,
        &beta,
        output,
        CUDA_R_16BF,
        n,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

int validate_layer_weights(const qwen_talker_layer_weights &weights, int device) {
    const uint64_t h = QWEN_NATIVE_HIDDEN_SIZE;
    const uint64_t f = QWEN_NATIVE_INTERMEDIATE_SIZE;
    const uint64_t q = QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM;
    const uint64_t kv = QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM;
    if (!is_device_pointer(weights.input_norm_w, device) ||
        !is_device_pointer(weights.post_attention_norm_w, device) ||
        !is_device_pointer(weights.q_proj_w, device) ||
        !is_device_pointer(weights.k_proj_w, device) ||
        !is_device_pointer(weights.v_proj_w, device) ||
        !is_device_pointer(weights.o_proj_w, device) ||
        !is_device_pointer(weights.q_norm_w, device) ||
        !is_device_pointer(weights.k_norm_w, device) ||
        !is_device_pointer(weights.gate_proj_w, device) ||
        !is_device_pointer(weights.up_proj_w, device) ||
        !is_device_pointer(weights.down_proj_w, device)) return -1;
    (void)h; (void)f; (void)q; (void)kv;
    return 0;
}

int hidden_rmsnorm_exact_graph(
    qwen_talker_ctx *ctx,
    const __nv_bfloat16 *input,
    const void *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    cudaStream_t stream
) {
    if (!ctx || (rows != 1u && rows != 2u) || !input || !weight || !output) return -1;
    qwen_talker_rmsnorm_graph_state *graph = rows == 1u
        ? ctx->rmsnorm_graph_q1
        : ctx->rmsnorm_graph;
    if (!graph) return -1;
    if (ctx->q1_graph_capture_active) {
        return qwen_talker_rmsnorm_eager_inline(
            input, weight, output, rows,
            static_cast<double>(QWEN_NATIVE_RMS_NORM_EPS), stream);
    }

    return qwen_talker_rmsnorm_graph_run(
        graph,
        input,
        weight,
        output,
        ctx->row_stats + 2u,
        static_cast<double>(
            QWEN_NATIVE_RMS_NORM_EPS
        ),
        stream
    );
}

bool validate_joint_position_contract(
    qwen_talker_ctx *ctx,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_rope_position,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode
) {
    if (!ctx || prefix_attention_mode > 2u ||
        prefix_position < ctx->prefix_start || prefix_position >= ctx->audio_start ||
        audio_position < ctx->audio_start || audio_position >= ctx->config.max_cache_len) {
        return false;
    }
    const uint32_t prefix_frame = prefix_position - ctx->prefix_start;
    const uint32_t audio_frame = audio_position - ctx->audio_start;
    if (prefix_frame != audio_frame) return false;
    const bool virtual_positions = prefix_attention_mode != 0u;
    const uint32_t expected_prefix_rope = prefix_position;
    const uint32_t expected_audio_rope = virtual_positions
        ? ctx->prefix_start + audio_frame
        : audio_position;
    if (prefix_rope_position != expected_prefix_rope ||
        audio_rope_position != expected_audio_rope) return false;

    if (!ctx->position_contract_set) {
        if (ctx->segment_joint_steps != 0u) return false;
        ctx->position_contract_set = true;
        ctx->virtual_positions = virtual_positions;
        ctx->prefix_attention_mode = prefix_attention_mode;
        ctx->next_prefix_rope_position = prefix_rope_position;
        ctx->next_audio_rope_position = audio_rope_position;
    }
    return ctx->virtual_positions == virtual_positions &&
        ctx->prefix_attention_mode == prefix_attention_mode &&
        ctx->next_prefix_rope_position == prefix_rope_position &&
        ctx->next_audio_rope_position == audio_rope_position;
}

bool validate_audio_position_contract(
    qwen_talker_ctx *ctx,
    uint32_t audio_position,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode
) {
    if (!ctx || !ctx->position_contract_set ||
        prefix_attention_mode != ctx->prefix_attention_mode ||
        audio_position < ctx->audio_start || audio_position >= ctx->config.max_cache_len) {
        return false;
    }
    const uint32_t frame = audio_position - ctx->audio_start;
    const uint32_t expected_rope = ctx->virtual_positions
        ? ctx->prefix_start + frame
        : audio_position;
    return audio_rope_position == expected_rope &&
        audio_rope_position == ctx->next_audio_rope_position;
}

int execute_layer(
    qwen_talker_ctx *ctx,
    uint32_t layer_idx,
    __nv_bfloat16 *hidden_in,
    __nv_bfloat16 *hidden_out,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_rope_position,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    uint32_t rows,
    bool audio_only_query,
    cudaStream_t stream,
    const uint32_t *device_position = nullptr
) {
    if (layer_idx >= QWEN_NATIVE_NUM_LAYERS) return record(ctx, -4);
    const qwen_talker_layer_weights &weights = ctx->weights.layers[layer_idx];
    const uint32_t hidden = QWEN_NATIVE_HIDDEN_SIZE;
    if ((rows != 1u && rows != 2u) || audio_only_query != (rows == 1u) ||
        audio_position >= ctx->config.max_cache_len || prefix_position > audio_position ||
        prefix_rope_position > audio_rope_position || prefix_attention_mode > 2u ||
        !std::isfinite(prefix_log_gate) ||
        (prefix_attention_mode == 2u && visible_prefix_count == 0u)) {
        return record(ctx, -5);
    }
    if (hidden_rmsnorm_exact_graph(
        ctx,
        hidden_in,
        weights.input_norm_w,
        ctx->norm,
        rows,
        stream
    ) != 0) {
        return record(ctx, -10);
    }
    if (snapshot_rms_mean(ctx, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_INPUT_NORM, ctx->norm, rows * hidden, stream) != 0) return -10;
    if (linear(ctx, ctx->norm, weights.q_proj_w, ctx->q_linear, rows, hidden,
               QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -11;
    if (linear(ctx, ctx->norm, weights.k_proj_w, ctx->k_linear, rows, hidden,
               QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -12;
    if (linear(ctx, ctx->norm, weights.v_proj_w, ctx->v_linear, rows, hidden,
               QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -13;
    if (snapshot(ctx, QWEN_NATIVE_TAP_Q_PROJ, ctx->q_linear,
                 rows * QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_K_PROJ, ctx->k_linear,
                 rows * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_V_PROJ, ctx->v_linear,
                 rows * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -13;

    if (qwen_rmsnorm_exact(
            ctx->q_linear, reinterpret_cast<const __nv_bfloat16 *>(weights.q_norm_w),
            ctx->q_linear, QWEN_NATIVE_NUM_ATTN_HEADS * rows, QWEN_NATIVE_HEAD_DIM,
            QWEN_NATIVE_RMS_NORM_EPS, ctx->squared, ctx->row_stats, stream) != 0) return -14;
    if (qwen_rmsnorm_exact(
            ctx->k_linear, reinterpret_cast<const __nv_bfloat16 *>(weights.k_norm_w),
            ctx->k_linear, QWEN_NATIVE_NUM_KV_HEADS * rows, QWEN_NATIVE_HEAD_DIM,
            QWEN_NATIVE_RMS_NORM_EPS, ctx->squared, ctx->row_stats, stream) != 0) return -15;
    if (snapshot(ctx, QWEN_NATIVE_TAP_Q_NORM, ctx->q_linear,
                 rows * QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_K_NORM, ctx->k_linear,
                 rows * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -15;
    if (qwen_pack_heads(ctx->q_linear, ctx->q_heads, rows, QWEN_NATIVE_NUM_ATTN_HEADS,
                        QWEN_NATIVE_HEAD_DIM, stream) != 0) return -16;
    if (qwen_pack_heads(ctx->k_linear, ctx->k_heads, rows, QWEN_NATIVE_NUM_KV_HEADS,
                        QWEN_NATIVE_HEAD_DIM, stream) != 0) return -17;
    if (qwen_pack_heads(ctx->v_linear, ctx->v_heads, rows, QWEN_NATIVE_NUM_KV_HEADS,
                        QWEN_NATIVE_HEAD_DIM, stream) != 0) return -18;
    const uint32_t first_query_position = audio_only_query
        ? audio_rope_position : prefix_rope_position;
    const int q_rope_status = device_position
        ? qwen_apply_rope_interleaved_cursor(
            ctx->q_heads, reinterpret_cast<const __nv_bfloat16 *>(ctx->weights.rotary_inv_freq),
            QWEN_NATIVE_NUM_ATTN_HEADS, rows, QWEN_NATIVE_HEAD_DIM, device_position, stream)
        : qwen_apply_rope_interleaved(
            ctx->q_heads, reinterpret_cast<const __nv_bfloat16 *>(ctx->weights.rotary_inv_freq),
            QWEN_NATIVE_NUM_ATTN_HEADS, rows, QWEN_NATIVE_HEAD_DIM,
            first_query_position, audio_rope_position, stream);
    if (q_rope_status != 0) return -19;
    const int k_rope_status = device_position
        ? qwen_apply_rope_interleaved_cursor(
            ctx->k_heads, reinterpret_cast<const __nv_bfloat16 *>(ctx->weights.rotary_inv_freq),
            QWEN_NATIVE_NUM_KV_HEADS, rows, QWEN_NATIVE_HEAD_DIM, device_position, stream)
        : qwen_apply_rope_interleaved(
            ctx->k_heads, reinterpret_cast<const __nv_bfloat16 *>(ctx->weights.rotary_inv_freq),
            QWEN_NATIVE_NUM_KV_HEADS, rows, QWEN_NATIVE_HEAD_DIM,
            first_query_position, audio_rope_position, stream);
    if (k_rope_status != 0) return -20;
    if (snapshot(ctx, QWEN_NATIVE_TAP_Q_ROPE, ctx->q_heads,
                 QWEN_NATIVE_NUM_ATTN_HEADS * rows * QWEN_NATIVE_HEAD_DIM, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_K_ROPE, ctx->k_heads,
                 QWEN_NATIVE_NUM_KV_HEADS * rows * QWEN_NATIVE_HEAD_DIM, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_V_HEADS, ctx->v_heads,
                 QWEN_NATIVE_NUM_KV_HEADS * rows * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -20;

    if (audio_only_query) {
        const int k_write_status = device_position
            ? qwen_write_audio_kv_cursor(
                ctx->k_heads, ctx->cache_k[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len, device_position,
                ctx->q1_device_error, stream)
            : qwen_write_audio_kv(
                ctx->k_heads, ctx->cache_k[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len, audio_position, stream);
        if (k_write_status != 0) return -21;
        const int v_write_status = device_position
            ? qwen_write_audio_kv_cursor(
                ctx->v_heads, ctx->cache_v[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len, device_position,
                ctx->q1_device_error, stream)
            : qwen_write_audio_kv(
                ctx->v_heads, ctx->cache_v[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len, audio_position, stream);
        if (v_write_status != 0) return -22;
    } else {
        if (qwen_write_joint_kv(
                ctx->k_heads, ctx->cache_k[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
                prefix_position, audio_position, stream) != 0) return -21;
        if (qwen_write_joint_kv(
                ctx->v_heads, ctx->cache_v[layer_idx], QWEN_NATIVE_NUM_KV_HEADS,
                QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
                prefix_position, audio_position, stream) != 0) return -22;
    }
    if (qwen_repeat_kv(
            ctx->cache_k[layer_idx], ctx->attention_key,
            QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_NUM_ATTN_HEADS / QWEN_NATIVE_NUM_KV_HEADS,
            ctx->config.max_cache_len, ctx->config.max_cache_len,
            QWEN_NATIVE_HEAD_DIM, stream) != 0) return -23;
    if (qwen_repeat_kv(
            ctx->cache_v[layer_idx], ctx->attention_value,
            QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_NUM_ATTN_HEADS / QWEN_NATIVE_NUM_KV_HEADS,
            ctx->config.max_cache_len, ctx->config.max_cache_len,
            QWEN_NATIVE_HEAD_DIM, stream) != 0) return -23;
    const int mask_status = audio_only_query
        ? (device_position
            ? qwen_build_audio_q1_causal_mask_cursor(
                ctx->mask, ctx->config.max_cache_len, device_position, stream)
            : qwen_build_audio_only_mask(
                ctx->mask, ctx->config.max_cache_len, ctx->config.max_cache_len,
                ctx->prefix_start, ctx->audio_start, prefix_position, audio_position,
                prefix_attention_mode, visible_prefix_count, prefix_log_gate, stream))
        : qwen_build_joint_mask(
            ctx->mask, ctx->config.max_cache_len, ctx->config.max_cache_len,
            ctx->prefix_start, ctx->audio_start, prefix_position, audio_position,
            prefix_attention_mode, visible_prefix_count, prefix_log_gate, stream);
    if (mask_status != 0) return -23;

    const int attention_status = ctx->q1_graph_capture_active
        ? qwen_talker_attention_gqa_inline(
            ctx->q_heads, ctx->attention_key, ctx->attention_value, ctx->mask,
            ctx->attention, rows, ctx->config.max_cache_len,
            QWEN_NATIVE_ATTENTION_SCALE, stream)
        : qwen_talker_attention_graph_run(
            ctx->attention_graph,
            ctx->q_heads, ctx->attention_key, ctx->attention_value, ctx->mask,
            ctx->attention, rows, ctx->config.max_cache_len,
            QWEN_NATIVE_ATTENTION_SCALE, stream);
    if (attention_status != 0) return record(ctx, -24);
    if (qwen_attention_to_rows(
            ctx->attention, ctx->attention_rows, QWEN_NATIVE_NUM_ATTN_HEADS, rows,
            QWEN_NATIVE_HEAD_DIM, stream) != 0) return -25;
    if (snapshot(ctx, QWEN_NATIVE_TAP_ATTENTION, ctx->attention,
                 QWEN_NATIVE_NUM_ATTN_HEADS * rows * QWEN_NATIVE_HEAD_DIM, stream) != 0) return -25;
    if (linear(ctx, ctx->attention_rows, weights.o_proj_w, ctx->q_linear, rows, hidden, hidden, stream) != 0) return -26;
    if (snapshot(ctx, QWEN_NATIVE_TAP_O_PROJ, ctx->q_linear, rows * hidden, stream) != 0) return -26;
    if (qwen_add_rows(hidden_in, ctx->q_linear, hidden_out, rows, hidden, stream) != 0) return -27;
    if (snapshot(ctx, QWEN_NATIVE_TAP_ATTENTION_RESIDUAL, hidden_out, rows * hidden, stream) != 0) return -27;

    if (hidden_rmsnorm_exact_graph(
        ctx,
        hidden_out,
        weights.post_attention_norm_w,
        ctx->norm,
        rows,
        stream
    ) != 0) {
        return -28;
    }
    if (snapshot(ctx, QWEN_NATIVE_TAP_FF_NORM, ctx->norm, rows * hidden, stream) != 0) return -28;
    if (linear(ctx, ctx->norm, weights.gate_proj_w, ctx->gate, rows, hidden,
               QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0) return -29;
    if (linear(ctx, ctx->norm, weights.up_proj_w, ctx->up, rows, hidden,
               QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0) return -30;
    if (snapshot(ctx, QWEN_NATIVE_TAP_GATE, ctx->gate, rows * QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0 ||
        snapshot(ctx, QWEN_NATIVE_TAP_UP, ctx->up, rows * QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0) return -30;
    if (qwen_silu_mul(
            ctx->gate, ctx->up, ctx->mlp,
            rows * QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0) return -31;
    if (snapshot(ctx, QWEN_NATIVE_TAP_ACTIVATION, ctx->mlp, rows * QWEN_NATIVE_INTERMEDIATE_SIZE, stream) != 0) return -31;
    if (linear(ctx, ctx->mlp, weights.down_proj_w, ctx->q_linear, rows,
               QWEN_NATIVE_INTERMEDIATE_SIZE, hidden, stream) != 0) return -32;
    if (snapshot(ctx, QWEN_NATIVE_TAP_DOWN, ctx->q_linear, rows * hidden, stream) != 0) return -32;
    if (qwen_add_rows(hidden_out, ctx->q_linear, hidden_out, rows, hidden, stream) != 0) return -33;
    if (snapshot(ctx, QWEN_NATIVE_TAP_LAYER_OUTPUT, hidden_out, rows * hidden, stream) != 0) return -33;
    return 0;
}

int validate_creation(
    const qwen_talker_config *config,
    const qwen_talker_weights *weights,
    int device
) {
    if (!config || !weights || !config->weight_meta ||
        config->weight_meta_count != QWEN_NATIVE_WEIGHT_META_COUNT ||
        config->max_cache_len == 0u || config->max_cache_len > QWEN_NATIVE_MAX_POSITION_EMBEDDINGS ||
        config->execute_layers == 0u || config->execute_layers > QWEN_NATIVE_NUM_LAYERS ||
        !valid_vector_meta(weights->final_norm_w, &config->weight_meta[0], QWEN_NATIVE_HIDDEN_SIZE, device) ||
        !valid_vector_meta(weights->rotary_inv_freq, &config->weight_meta[1], QWEN_NATIVE_HEAD_DIM / 2u, device)) return -1;

    const uint64_t h = QWEN_NATIVE_HIDDEN_SIZE;
    const uint64_t f = QWEN_NATIVE_INTERMEDIATE_SIZE;
    const uint64_t q = QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM;
    const uint64_t kv = QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM;
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        const qwen_talker_layer_weights &w = weights->layers[layer];
        const qwen_tensor_meta *m = &config->weight_meta[QWEN_NATIVE_GLOBAL_WEIGHT_META_COUNT + layer * QWEN_NATIVE_LAYER_WEIGHT_META_COUNT];
        if (validate_layer_weights(w, device) != 0 ||
            !valid_vector_meta(w.input_norm_w, &m[0], h, device) ||
            !valid_vector_meta(w.post_attention_norm_w, &m[1], h, device) ||
            !valid_matrix_meta(w.q_proj_w, &m[2], q, h, device) ||
            !valid_matrix_meta(w.k_proj_w, &m[3], kv, h, device) ||
            !valid_matrix_meta(w.v_proj_w, &m[4], kv, h, device) ||
            !valid_matrix_meta(w.o_proj_w, &m[5], h, q, device) ||
            !valid_vector_meta(w.q_norm_w, &m[6], QWEN_NATIVE_HEAD_DIM, device) ||
            !valid_vector_meta(w.k_norm_w, &m[7], QWEN_NATIVE_HEAD_DIM, device) ||
            !valid_matrix_meta(w.gate_proj_w, &m[8], f, h, device) ||
            !valid_matrix_meta(w.up_proj_w, &m[9], f, h, device) ||
            !valid_matrix_meta(w.down_proj_w, &m[10], h, f, device)) return -2;
    }
    return 0;
}

void bind_arena_pointers(qwen_talker_ctx *ctx) {
    ArenaBuilder builder;
    auto bind = [&builder, ctx](auto *&pointer, size_t count) {
        using Type = std::remove_pointer_t<std::remove_reference_t<decltype(pointer)>>;
        Type *relative = builder.take<Type>(count);
        pointer = bind_arena(ctx->arena, relative);
    };
    bind(ctx->hidden_a, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->hidden_b, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->norm, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->rms_graph_input, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->rms_graph_weight, QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->rms_graph_output, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->q_linear, 2u * QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->k_linear, 2u * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->v_linear, 2u * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->q_heads, QWEN_NATIVE_NUM_ATTN_HEADS * 2u * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->k_heads, QWEN_NATIVE_NUM_KV_HEADS * 2u * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->v_heads, QWEN_NATIVE_NUM_KV_HEADS * 2u * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->attention_key, static_cast<size_t>(QWEN_NATIVE_NUM_ATTN_HEADS) * ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->attention_value, static_cast<size_t>(QWEN_NATIVE_NUM_ATTN_HEADS) * ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->attention, QWEN_NATIVE_NUM_ATTN_HEADS * 2u * QWEN_NATIVE_HEAD_DIM);
    bind(ctx->attention_rows, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->gate, 2u * QWEN_NATIVE_INTERMEDIATE_SIZE);
    bind(ctx->up, 2u * QWEN_NATIVE_INTERMEDIATE_SIZE);
    bind(ctx->mlp, 2u * QWEN_NATIVE_INTERMEDIATE_SIZE);
    bind(ctx->mask, 2u * ctx->config.max_cache_len);
    bind(ctx->q1_device_position, 1u);
    bind(ctx->q1_device_error, 1u);
    bind(ctx->trace_prefix_input, QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->trace_audio_input, QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->trace_output_hidden, QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->trace_position_ids, 2u);
    bind(ctx->squared, 2u * QWEN_NATIVE_HIDDEN_SIZE);
    bind(ctx->row_stats, 2u * MAX_RMS_ROWS);
    bind(ctx->debug_input_rms_mean, 2u);
    for (uint32_t tap = 0; tap < QWEN_NATIVE_DEBUG_TAP_COUNT; ++tap) {
        bind(ctx->debug_taps[tap], 2u * std::max(QWEN_NATIVE_HIDDEN_SIZE, QWEN_NATIVE_INTERMEDIATE_SIZE));
    }
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        bind(ctx->cache_k[layer], static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM);
        bind(ctx->cache_v[layer], static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM);
    }
}

} // namespace

extern "C" int qwen_talker_create(
    const qwen_talker_config *config,
    const qwen_talker_weights *weights,
    qwen_talker_ctx **out_ctx
) {
    if (!out_ctx) return -1;
    *out_ctx = nullptr;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return -2;
    if (validate_creation(config, weights, device) != 0) return -3;

    qwen_talker_ctx *ctx = new (std::nothrow) qwen_talker_ctx();
    if (!ctx) return -4;
    ctx->config = *config;
    ctx->weights = *weights;
    ctx->device = device;

    ArenaBuilder builder;
    auto count = [&builder](size_t bytes) { (void)builder.take<unsigned char>(bytes); };
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_NUM_ATTN_HEADS * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_NUM_ATTN_HEADS * 2u * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_NUM_KV_HEADS * 2u * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_NUM_KV_HEADS * 2u * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(static_cast<size_t>(QWEN_NATIVE_NUM_ATTN_HEADS) * config->max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(static_cast<size_t>(QWEN_NATIVE_NUM_ATTN_HEADS) * config->max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_NUM_ATTN_HEADS * 2u * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    count(2u * QWEN_NATIVE_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    count(2u * config->max_cache_len * sizeof(__nv_bfloat16));
    count(sizeof(uint32_t));
    count(sizeof(uint32_t));
    count(QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    count(2u * sizeof(int64_t));
    count(2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(float));
    count(2u * MAX_RMS_ROWS * sizeof(float));
    count(2u * sizeof(float));
    for (uint32_t tap = 0; tap < QWEN_NATIVE_DEBUG_TAP_COUNT; ++tap) {
        count(2u * std::max(QWEN_NATIVE_HIDDEN_SIZE, QWEN_NATIVE_INTERMEDIATE_SIZE) * sizeof(__nv_bfloat16));
    }
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        count(static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * config->max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
        count(static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * config->max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16));
    }
    ctx->arena_bytes = aligned_size(builder.offset);
    if (cudaMalloc(reinterpret_cast<void **>(&ctx->arena), ctx->arena_bytes) != cudaSuccess) {
        delete ctx;
        return -5;
    }
    bind_arena_pointers(ctx);
    if (cudaMalloc(&ctx->q1_cublas_workspace, QWEN_NATIVE_Q1_CUBLAS_WORKSPACE_BYTES) != cudaSuccess) {
        cudaFree(ctx->arena);
        delete ctx;
        return -6;
    }
    if (cublasCreate(&ctx->blas) != CUBLAS_STATUS_SUCCESS ||
        qwen_talker_attention_graph_create(
            &ctx->attention_graph
        ) != 0 ||
        qwen_talker_rmsnorm_graph_create(
            &ctx->rmsnorm_graph,
            2u,
            ctx->rms_graph_input,
            ctx->rms_graph_weight,
            ctx->rms_graph_output
        ) != 0 ||
        qwen_talker_rmsnorm_graph_create(
            &ctx->rmsnorm_graph_q1,
            1u,
            ctx->rms_graph_input,
            ctx->rms_graph_weight,
            ctx->rms_graph_output
        ) != 0) {

        if (ctx->rmsnorm_graph_q1) {
            qwen_talker_rmsnorm_graph_destroy(ctx->rmsnorm_graph_q1);
        }
        if (ctx->rmsnorm_graph) {
            qwen_talker_rmsnorm_graph_destroy(
                ctx->rmsnorm_graph
            );
        }

        if (ctx->attention_graph) {
            qwen_talker_attention_graph_destroy(
                ctx->attention_graph
            );
        }

        if (ctx->blas) {
            cublasDestroy(ctx->blas);
        }

        if (ctx->q1_cublas_workspace) cudaFree(ctx->q1_cublas_workspace);
        cudaFree(ctx->arena);
        delete ctx;
        return -7;
    }
    *out_ctx = ctx;
    return 0;
}

extern "C" int qwen_talker_import_cache_async(
    qwen_talker_ctx *ctx,
    const qwen_talker_imported_cache *cache,
    uint32_t stable_length,
    uint32_t prefix_start,
    uint32_t audio_start,
    cudaStream_t stream
) {
    if (!ctx || !cache || stable_length == 0u || stable_length > ctx->config.max_cache_len ||
        prefix_start != stable_length || audio_start <= prefix_start ||
        audio_start >= ctx->config.max_cache_len) return record(ctx, -40);
    if (!stream_matches(ctx, stream)) return record(ctx, -44);
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        const qwen_talker_imported_kv &source = cache->layers[layer];
        if (!valid_cache_meta(source.k_ptr, source.k_meta, QWEN_NATIVE_NUM_KV_HEADS, stable_length, ctx->device) ||
            !valid_cache_meta(source.v_ptr, source.v_meta, QWEN_NATIVE_NUM_KV_HEADS, stable_length, ctx->device) ||
            source.k_meta->dims[2] != source.v_meta->dims[2]) return record(ctx, -41);
    }
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        const size_t bytes = static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16);
        if (cudaMemsetAsync(ctx->cache_k[layer], 0, bytes, stream) != cudaSuccess ||
            cudaMemsetAsync(ctx->cache_v[layer], 0, bytes, stream) != cudaSuccess) return record(ctx, -42);
        const qwen_talker_imported_kv &source = cache->layers[layer];
        if (qwen_import_cache_prefix(
                reinterpret_cast<const __nv_bfloat16 *>(source.k_ptr), ctx->cache_k[layer],
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM,
                static_cast<uint32_t>(source.k_meta->dims[2]), ctx->config.max_cache_len,
                stable_length, stream) != 0 ||
            qwen_import_cache_prefix(
                reinterpret_cast<const __nv_bfloat16 *>(source.v_ptr), ctx->cache_v[layer],
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM,
                static_cast<uint32_t>(source.v_meta->dims[2]), ctx->config.max_cache_len,
                stable_length, stream) != 0) return record(ctx, -43);
    }
    ctx->imported = true;
    ctx->mode = qwen_execution_mode::joint_q2;
    ctx->stable_length = stable_length;
    ctx->prefix_start = prefix_start;
    ctx->audio_start = audio_start;
    ctx->current_prefix_position = prefix_start - 1u;
    ctx->current_audio_position = audio_start - 1u;
    ctx->next_prefix_position = prefix_start;
    ctx->next_audio_position = audio_start;
    ctx->position_contract_set = false;
    ctx->virtual_positions = false;
    ctx->prefix_attention_mode = 0u;
    ctx->next_prefix_rope_position = prefix_start;
    ctx->next_audio_rope_position = audio_start;
    ctx->segment_joint_steps = 0;
    ctx->segment_terminal_queries = 0;
    ctx->terminal_prefix_cache_appends = 0;
    ctx->terminal_audio_cache_appends = 0;
    ctx->terminal_query_position = UINT32_MAX;
    ctx->terminal_visible_prefix_position = UINT32_MAX;
    ctx->terminal_query_started = false;
    ctx->terminal_query_done = false;
    ctx->last_cache_layer = UINT32_MAX;
    ctx->last_cache_all_layers = false;
    ctx->trace_cache_layers_written = 0;
    ctx->trace_expected_steps = 0;
    ctx->trace_completed_steps = 0;
    std::memset(ctx->trace_previous_chain, 0, sizeof(ctx->trace_previous_chain));
    std::memset(ctx->trace_chain, 0, sizeof(ctx->trace_chain));
    ++ctx->trace_generation;
    remember_stream(ctx, stream);
    ++ctx->cache_imports;
    return 0;
}

extern "C" int qwen_talker_import_audio_cache_async(
    qwen_talker_ctx *ctx,
    const qwen_talker_imported_cache *cache,
    uint32_t prefill_length,
    uint32_t max_audio_frames,
    cudaStream_t stream
) {
    const uint64_t required_capacity = static_cast<uint64_t>(prefill_length) + max_audio_frames;
    if (!ctx || !cache || ctx->trace_enabled || prefill_length == 0u || max_audio_frames == 0u ||
        required_capacity > ctx->config.max_cache_len ||
        required_capacity > QWEN_NATIVE_MAX_POSITION_EMBEDDINGS) return record(ctx, -45);
    if (!stream_matches(ctx, stream)) return record(ctx, -46);
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        const qwen_talker_imported_kv &source = cache->layers[layer];
        if (!valid_cache_meta(source.k_ptr, source.k_meta, QWEN_NATIVE_NUM_KV_HEADS,
                              prefill_length, ctx->device) ||
            !valid_cache_meta(source.v_ptr, source.v_meta, QWEN_NATIVE_NUM_KV_HEADS,
                              prefill_length, ctx->device) ||
            source.k_meta->dims[2] != prefill_length ||
            source.v_meta->dims[2] != prefill_length) return record(ctx, -47);
    }
    ctx->imported = false;
    ctx->mode = qwen_execution_mode::none;
    for (uint32_t layer = 0; layer < QWEN_NATIVE_NUM_LAYERS; ++layer) {
        const size_t bytes = static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) *
            ctx->config.max_cache_len * QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16);
        if (cudaMemsetAsync(ctx->cache_k[layer], 0, bytes, stream) != cudaSuccess ||
            cudaMemsetAsync(ctx->cache_v[layer], 0, bytes, stream) != cudaSuccess) return record(ctx, -48);
        const qwen_talker_imported_kv &source = cache->layers[layer];
        if (qwen_import_cache_prefix(
                reinterpret_cast<const __nv_bfloat16 *>(source.k_ptr), ctx->cache_k[layer],
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, prefill_length,
                ctx->config.max_cache_len, prefill_length, stream) != 0 ||
            qwen_import_cache_prefix(
                reinterpret_cast<const __nv_bfloat16 *>(source.v_ptr), ctx->cache_v[layer],
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, prefill_length,
                ctx->config.max_cache_len, prefill_length, stream) != 0) return record(ctx, -49);
    }
    if (qwen_initialize_audio_q1_cursor(
            ctx->q1_device_position, ctx->q1_device_error, prefill_length, stream) != 0) {
        return record(ctx, -50);
    }
    ctx->imported = true;
    ctx->mode = qwen_execution_mode::recurrent_audio_q1;
    ctx->stable_length = prefill_length;
    ctx->prefix_start = 0u;
    ctx->audio_start = prefill_length;
    ctx->current_prefix_position = prefill_length - 1u;
    ctx->current_audio_position = prefill_length - 1u;
    ctx->next_prefix_position = prefill_length;
    ctx->next_audio_position = prefill_length;
    ctx->position_contract_set = true;
    ctx->virtual_positions = false;
    ctx->prefix_attention_mode = 0u;
    ctx->next_prefix_rope_position = prefill_length;
    ctx->next_audio_rope_position = prefill_length;
    ctx->audio_q1_limit_position = static_cast<uint32_t>(required_capacity);
    ctx->segment_joint_steps = 0u;
    ctx->segment_terminal_queries = 0u;
    ctx->terminal_prefix_cache_appends = 0u;
    ctx->terminal_audio_cache_appends = 0u;
    ctx->terminal_query_position = UINT32_MAX;
    ctx->terminal_visible_prefix_position = UINT32_MAX;
    ctx->terminal_query_started = false;
    ctx->terminal_query_done = false;
    ctx->last_cache_layer = UINT32_MAX;
    ctx->last_cache_all_layers = false;
    ctx->trace_cache_layers_written = 0u;
    ctx->trace_expected_steps = 0u;
    ctx->trace_completed_steps = 0u;
    std::memset(ctx->trace_previous_chain, 0, sizeof(ctx->trace_previous_chain));
    std::memset(ctx->trace_chain, 0, sizeof(ctx->trace_chain));
    ++ctx->trace_generation;
    remember_stream(ctx, stream);
    ++ctx->cache_imports;
    return 0;
}

extern "C" int qwen_talker_debug_execute_layer_async(
    qwen_talker_ctx *ctx,
    uint32_t layer_idx,
    const void *prefix_embed_bf16,
    const qwen_tensor_meta *prefix_meta,
    const void *audio_embed_bf16,
    const qwen_tensor_meta *audio_meta,
    uint32_t prefix_position,
    uint32_t audio_position,
    void *hidden_rows_bf16,
    const qwen_tensor_meta *hidden_rows_meta,
    cudaStream_t stream
) {
    if (!ctx) return -50;
    if (!ctx->imported) return record(ctx, -51);
    if (layer_idx >= QWEN_NATIVE_NUM_LAYERS) return record(ctx, -52);
    if (!valid_embedding_meta(prefix_embed_bf16, prefix_meta, ctx->device)) return record(ctx, -53);
    if (!valid_embedding_meta(audio_embed_bf16, audio_meta, ctx->device)) return record(ctx, -54);
    if (!valid_hidden_rows_meta(hidden_rows_bf16, hidden_rows_meta, ctx->device)) return record(ctx, -55);
    if (prefix_position < ctx->prefix_start || prefix_position >= ctx->audio_start ||
        audio_position < ctx->audio_start || audio_position >= ctx->config.max_cache_len) return record(ctx, -56);
    if (!stream_matches(ctx, stream)) return record(ctx, -54);
    if (qwen_copy_joint_inputs(
            reinterpret_cast<const __nv_bfloat16 *>(prefix_embed_bf16),
            reinterpret_cast<const __nv_bfloat16 *>(audio_embed_bf16), ctx->hidden_a, stream) != 0) return record(ctx, -51);
    const int layer_status = execute_layer(ctx, layer_idx, ctx->hidden_a, ctx->hidden_b,
                                            prefix_position, audio_position,
                                            prefix_position, audio_position,
                                            0u, 0u, -8.0f, 2u, false, stream);
    if (layer_status != 0) return record(ctx, layer_status);
    const size_t bytes = 2u * QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(hidden_rows_bf16, ctx->hidden_b, bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -53);
    ctx->current_prefix_position = prefix_position;
    ctx->current_audio_position = audio_position;
    ctx->last_cache_layer = layer_idx;
    ctx->last_cache_all_layers = false;
    remember_stream(ctx, stream);
    return 0;
}

extern "C" int qwen_talker_debug_copy_layer_cache_async(
    qwen_talker_ctx *ctx,
    uint32_t layer_idx,
    void *key_cache_bf16,
    const qwen_tensor_meta *key_cache_meta,
    void *value_cache_bf16,
    const qwen_tensor_meta *value_cache_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->imported || layer_idx >= QWEN_NATIVE_NUM_LAYERS ||
        !valid_cache_meta(key_cache_bf16, key_cache_meta, QWEN_NATIVE_NUM_KV_HEADS,
                          ctx->config.max_cache_len, ctx->device) ||
        !valid_cache_meta(value_cache_bf16, value_cache_meta, QWEN_NATIVE_NUM_KV_HEADS,
                          ctx->config.max_cache_len, ctx->device) ||
        key_cache_bf16 == value_cache_bf16) return record(ctx, -55);
    if (!stream_matches(ctx, stream)) return record(ctx, -61);
    const size_t bytes = static_cast<size_t>(QWEN_NATIVE_NUM_KV_HEADS) * ctx->config.max_cache_len *
        QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(key_cache_bf16, ctx->cache_k[layer_idx], bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(value_cache_bf16, ctx->cache_v[layer_idx], bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -56);
    return 0;
}

extern "C" int qwen_talker_debug_copy_layer_cache_positions_async(
    qwen_talker_ctx *ctx,
    uint32_t layer_idx,
    uint32_t prefix_position,
    uint32_t audio_position,
    void *key_rows_bf16,
    const qwen_tensor_meta *key_rows_meta,
    void *value_rows_bf16,
    const qwen_tensor_meta *value_rows_meta,
    uint32_t key_elements,
    uint32_t value_elements,
    cudaStream_t stream
) {
    const uint32_t expected_elements = 2u * QWEN_NATIVE_NUM_KV_HEADS * QWEN_NATIVE_HEAD_DIM;
    if (!ctx || !ctx->imported || layer_idx >= QWEN_NATIVE_NUM_LAYERS ||
        prefix_position >= ctx->config.max_cache_len ||
        audio_position >= ctx->config.max_cache_len ||
        key_elements != expected_elements || value_elements != expected_elements ||
        key_rows_bf16 == value_rows_bf16 ||
        !valid_cache_rows_meta(key_rows_bf16, key_rows_meta, ctx->device) ||
        !valid_cache_rows_meta(value_rows_bf16, value_rows_meta, ctx->device)) return record(ctx, -58);
    if (!stream_matches(ctx, stream) ||
        (ctx->current_prefix_position != prefix_position) ||
        (ctx->current_audio_position != audio_position) ||
        (!ctx->last_cache_all_layers && ctx->last_cache_layer != layer_idx)) {
        return record(ctx, -60);
    }
    if (qwen_copy_cache_joint_rows(
            ctx->cache_k[layer_idx], reinterpret_cast<__nv_bfloat16 *>(key_rows_bf16),
            QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
            prefix_position, audio_position, stream) != 0 ||
        qwen_copy_cache_joint_rows(
            ctx->cache_v[layer_idx], reinterpret_cast<__nv_bfloat16 *>(value_rows_bf16),
            QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
            prefix_position, audio_position, stream) != 0) return record(ctx, -59);
    return 0;
}

extern "C" int qwen_talker_debug_copy_tap_async(
    qwen_talker_ctx *ctx,
    uint32_t tap_id,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    uint32_t elements,
    cudaStream_t stream
) {
    if (!ctx || tap_id >= QWEN_NATIVE_DEBUG_TAP_COUNT ||
        !valid_meta(destination_bf16, destination_meta, QWEN_NATIVE_META_BF16,
                    destination_meta ? destination_meta->ndim : 0u,
                    destination_meta ? destination_meta->dims : nullptr, ctx->device) ||
        elements > 2u * std::max(QWEN_NATIVE_HIDDEN_SIZE, QWEN_NATIVE_INTERMEDIATE_SIZE) ||
        destination_meta->capacity_bytes < static_cast<uint64_t>(elements) * sizeof(__nv_bfloat16)) return record(ctx, -57);
    if (cudaMemcpyAsync(destination_bf16, ctx->debug_taps[tap_id],
                        static_cast<size_t>(elements) * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -58);
    return 0;
}

extern "C" int qwen_talker_debug_copy_input_rms_mean_async(
    qwen_talker_ctx *ctx,
    void *destination_fp32,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
) {
    const uint64_t dims[] = {2u};
    if (!ctx || !valid_meta(destination_fp32, destination_meta, QWEN_NATIVE_META_FP32, 1u, dims, ctx->device)) return record(ctx, -59);
    if (cudaMemcpyAsync(destination_fp32, ctx->debug_input_rms_mean,
                        2u * sizeof(float), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -59);
    return 0;
}

extern "C" int qwen_talker_trace_set_enabled(qwen_talker_ctx *ctx, uint32_t enabled) {
    if (!ctx) return -70;
    if (enabled != 0u && ctx->mode == qwen_execution_mode::recurrent_audio_q1) return record(ctx, -70);
    ctx->trace_enabled = enabled != 0u;
    return 0;
}

extern "C" int qwen_talker_trace_set_expected_steps(qwen_talker_ctx *ctx, uint64_t expected_steps) {
    if (!ctx || !ctx->trace_enabled || !ctx->imported || expected_steps == 0u || ctx->trace_completed_steps != 0u) {
        return record(ctx, -76);
    }
    ctx->trace_expected_steps = expected_steps;
    return 0;
}

extern "C" int qwen_talker_trace_copy_effective_mask_async(
    qwen_talker_ctx *ctx,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->trace_enabled || !ctx->imported || !stream_matches(ctx, stream) ||
        !valid_trace_mask_meta(destination_bf16, destination_meta,
                                ctx->config.max_cache_len, ctx->device)) return record(ctx, -71);
    if (cudaMemcpyAsync(
            destination_bf16, ctx->mask,
            static_cast<size_t>(2u) * ctx->config.max_cache_len * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -72);
    return 0;
}

extern "C" int qwen_talker_trace_copy_inputs_async(
    qwen_talker_ctx *ctx,
    void *prefix_destination_bf16,
    const qwen_tensor_meta *prefix_destination_meta,
    void *audio_destination_bf16,
    const qwen_tensor_meta *audio_destination_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->trace_enabled || !stream_matches(ctx, stream) ||
        !valid_embedding_meta(prefix_destination_bf16, prefix_destination_meta, ctx->device) ||
        !valid_embedding_meta(audio_destination_bf16, audio_destination_meta, ctx->device)) {
        return record(ctx, -73);
    }
    const size_t bytes = QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(prefix_destination_bf16, ctx->trace_prefix_input, bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(audio_destination_bf16, ctx->trace_audio_input, bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -74);
    return 0;
}

extern "C" int qwen_talker_trace_copy_position_ids_async(
    qwen_talker_ctx *ctx,
    void *destination_int64,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->trace_enabled || !stream_matches(ctx, stream) ||
        !valid_trace_position_meta(destination_int64, destination_meta, ctx->device)) {
        return record(ctx, -77);
    }
    if (cudaMemcpyAsync(destination_int64, ctx->trace_position_ids,
                        2u * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return record(ctx, -78);
    }
    return 0;
}

extern "C" int qwen_talker_trace_copy_output_async(
    qwen_talker_ctx *ctx,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->trace_enabled || !stream_matches(ctx, stream) ||
        !valid_embedding_meta(destination_bf16, destination_meta, ctx->device)) {
        return record(ctx, -79);
    }
    if (cudaMemcpyAsync(destination_bf16, ctx->trace_output_hidden,
                        QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        return record(ctx, -80);
    }
    return 0;
}

extern "C" int qwen_talker_trace_copy_cache_rows_async(
    qwen_talker_ctx *ctx,
    void *key_destination_bf16,
    const qwen_tensor_meta *key_destination_meta,
    void *value_destination_bf16,
    const qwen_tensor_meta *value_destination_meta,
    cudaStream_t stream
) {
    const size_t row_bytes = static_cast<size_t>(2u) * QWEN_NATIVE_NUM_KV_HEADS *
        QWEN_NATIVE_HEAD_DIM * sizeof(__nv_bfloat16);
    const size_t layer_bytes = row_bytes;
    if (!ctx || !ctx->trace_enabled || !ctx->imported || !ctx->last_cache_all_layers ||
        !valid_trace_cache_meta(key_destination_bf16, key_destination_meta, ctx->device) ||
        !valid_trace_cache_meta(value_destination_bf16, value_destination_meta, ctx->device) ||
        key_destination_bf16 == value_destination_bf16 || ctx->joint_steps == 0u) {
        return record(ctx, -81);
    }
    if (!stream_matches(ctx, stream)) return record(ctx, -82);
    for (uint32_t layer = 0; layer < ctx->config.execute_layers; ++layer) {
        auto *key_row_destination = reinterpret_cast<__nv_bfloat16 *>(key_destination_bf16) +
            static_cast<size_t>(layer) * layer_bytes / sizeof(__nv_bfloat16);
        auto *value_row_destination = reinterpret_cast<__nv_bfloat16 *>(value_destination_bf16) +
            static_cast<size_t>(layer) * layer_bytes / sizeof(__nv_bfloat16);
        if (qwen_copy_cache_joint_rows(
                ctx->cache_k[layer], key_row_destination,
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
                ctx->current_prefix_position, ctx->current_audio_position, stream) != 0 ||
            qwen_copy_cache_joint_rows(
                ctx->cache_v[layer], value_row_destination,
                QWEN_NATIVE_NUM_KV_HEADS, QWEN_NATIVE_HEAD_DIM, ctx->config.max_cache_len,
                ctx->current_prefix_position, ctx->current_audio_position, stream) != 0) {
            return record(ctx, -83);
        }
    }
    return 0;
}

extern "C" int qwen_talker_trace_last_step(
    const qwen_talker_ctx *ctx,
    qwen_talker_trace_step *destination
) {
    if (!ctx || !destination || !ctx->trace_enabled || ctx->trace_completed_steps == 0u) return -75;
    *destination = qwen_talker_trace_step{};
    destination->step_index = ctx->trace_completed_steps - 1u;
    destination->trace_generation = ctx->trace_generation;
    destination->expected_step_count = ctx->trace_expected_steps;
    destination->completed_step_count = ctx->trace_completed_steps;
    destination->stable_length = ctx->stable_length;
    destination->prefix_start = ctx->prefix_start;
    destination->audio_start = ctx->audio_start;
    destination->prefix_position = ctx->current_prefix_position;
    destination->audio_position = ctx->current_audio_position;
    destination->cache_layers_written = ctx->trace_cache_layers_written;
    destination->trace_enabled = ctx->trace_enabled ? 1u : 0u;
    destination->native_complete = (
        ctx->trace_expected_steps != 0u && ctx->trace_completed_steps == ctx->trace_expected_steps
    ) ? 1u : 0u;
    destination->joint_steps = ctx->joint_steps;
    destination->cache_imports = ctx->cache_imports;
    destination->cuda_error_state = ctx->last_error;
    std::memcpy(destination->native_previous_chain, ctx->trace_previous_chain, sizeof(destination->native_previous_chain));
    std::memcpy(destination->native_chain, ctx->trace_chain, sizeof(destination->native_chain));
    return 0;
}

extern "C" uint32_t qwen_talker_trace_abi_version(void) { return 3u; }

extern "C" int qwen_talker_step_joint_async(
    qwen_talker_ctx *ctx,
    const void *prefix_embed_bf16,
    const qwen_tensor_meta *prefix_meta,
    const void *audio_embed_bf16,
    const qwen_tensor_meta *audio_meta,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_rope_position,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    void *last_hidden_bf16,
    const qwen_tensor_meta *last_hidden_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->imported || ctx->mode != qwen_execution_mode::joint_q2 ||
        ctx->config.execute_layers != QWEN_NATIVE_NUM_LAYERS ||
        (ctx->trace_enabled && ctx->trace_expected_steps == 0u) ||
        !valid_embedding_meta(prefix_embed_bf16, prefix_meta, ctx->device) ||
        !valid_embedding_meta(audio_embed_bf16, audio_meta, ctx->device) ||
        !valid_embedding_meta(last_hidden_bf16, last_hidden_meta, ctx->device) ||
        ctx->terminal_query_started || ctx->terminal_query_done ||
        prefix_position < ctx->prefix_start || prefix_position >= ctx->audio_start ||
        audio_position < ctx->audio_start || audio_position >= ctx->config.max_cache_len) return record(ctx, -60);
    if (!stream_matches(ctx, stream) || prefix_position != ctx->next_prefix_position ||
        audio_position != ctx->next_audio_position) return record(ctx, -65);
    if (!validate_joint_position_contract(
            ctx, prefix_position, audio_position, prefix_rope_position,
            audio_rope_position, prefix_attention_mode) ||
        (prefix_attention_mode == 2u &&
         visible_prefix_count != prefix_position - ctx->prefix_start + 1u) ||
        !std::isfinite(prefix_log_gate)) return record(ctx, -66);
    if (ctx->trace_enabled) {
        const size_t input_bytes = QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16);
        if (qwen_write_position_ids(
                ctx->trace_position_ids, prefix_rope_position, audio_rope_position, stream) != 0 ||
            cudaMemcpyAsync(ctx->trace_prefix_input, prefix_embed_bf16, input_bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(ctx->trace_audio_input, audio_embed_bf16, input_bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -61);
    }
    if (qwen_copy_joint_inputs(
            reinterpret_cast<const __nv_bfloat16 *>(prefix_embed_bf16),
            reinterpret_cast<const __nv_bfloat16 *>(audio_embed_bf16), ctx->hidden_a, stream) != 0) return record(ctx, -61);
    __nv_bfloat16 *current = ctx->hidden_a;
    __nv_bfloat16 *next = ctx->hidden_b;
    for (uint32_t layer = 0; layer < ctx->config.execute_layers; ++layer) {
        const int layer_status = execute_layer(ctx, layer, current, next,
                                               prefix_position, audio_position,
                                               prefix_rope_position, audio_rope_position,
                                               prefix_attention_mode, visible_prefix_count,
                                               prefix_log_gate, 2u, false, stream);
        if (layer_status != 0) return record(ctx, layer_status);
        std::swap(current, next);
    }
    if (ctx->config.execute_layers == QWEN_NATIVE_NUM_LAYERS) {
        if (hidden_rmsnorm_exact_graph(
            ctx,
            current,
            ctx->weights.final_norm_w,
            ctx->norm,
            2u,
            stream
        ) != 0) {
            return record(ctx, -63);
        }
        current = ctx->norm;
    }
    if (qwen_copy_audio_row(current, reinterpret_cast<__nv_bfloat16 *>(last_hidden_bf16),
                            1u, QWEN_NATIVE_HIDDEN_SIZE, stream) != 0) return record(ctx, -64);
    if (ctx->trace_enabled && qwen_copy_audio_row(
            current, ctx->trace_output_hidden, 1u, QWEN_NATIVE_HIDDEN_SIZE, stream
        ) != 0) return record(ctx, -64);
    ctx->current_prefix_position = prefix_position;
    ctx->current_audio_position = audio_position;
    ctx->last_cache_layer = UINT32_MAX;
    ctx->last_cache_all_layers = (ctx->config.execute_layers == QWEN_NATIVE_NUM_LAYERS);
    ctx->trace_cache_layers_written = ctx->last_cache_all_layers ? ctx->config.execute_layers : 0u;
    ctx->next_prefix_position = prefix_position + 1u;
    ctx->next_audio_position = audio_position + 1u;
    ctx->next_prefix_rope_position = prefix_rope_position + 1u;
    ctx->next_audio_rope_position = audio_rope_position + 1u;
    remember_stream(ctx, stream);
    ++ctx->joint_steps;
    ++ctx->segment_joint_steps;
    ++ctx->trace_completed_steps;
    if (ctx->trace_enabled) {
        std::memcpy(ctx->trace_previous_chain, ctx->trace_chain, sizeof(ctx->trace_previous_chain));
        update_native_trace_chain(ctx, ctx->trace_previous_chain, ctx->trace_chain);
    }
    return 0;
}

extern "C" int qwen_talker_step_terminal_async(
    qwen_talker_ctx *ctx,
    const void *audio_embed_bf16,
    const qwen_tensor_meta *audio_meta,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    void *last_hidden_bf16,
    const qwen_tensor_meta *last_hidden_meta,
    cudaStream_t stream
) {
    const bool joint_q2_mode = ctx && ctx->mode == qwen_execution_mode::joint_q2;
    const bool audio_q1_suffix_mode = ctx &&
        ctx->mode == qwen_execution_mode::recurrent_audio_q1 &&
        ctx->next_prefix_position == ctx->audio_start;
    const uint32_t expected_prefix_position = audio_q1_suffix_mode
        ? ctx->audio_start - 1u
        : (ctx ? ctx->current_prefix_position : 0u);
    if (!ctx || !ctx->imported ||
        (!joint_q2_mode && !audio_q1_suffix_mode) ||
        ctx->config.execute_layers != QWEN_NATIVE_NUM_LAYERS ||
        ctx->terminal_query_started || ctx->terminal_query_done || ctx->segment_joint_steps == 0u ||
        (ctx->trace_enabled && (ctx->trace_expected_steps == 0u ||
                                ctx->trace_completed_steps != ctx->trace_expected_steps)) ||
        !valid_embedding_meta(audio_embed_bf16, audio_meta, ctx->device) ||
        !valid_embedding_meta(last_hidden_bf16, last_hidden_meta, ctx->device) ||
        visible_prefix_position != expected_prefix_position ||
        visible_prefix_position < ctx->prefix_start ||
        visible_prefix_position >= ctx->audio_start ||
        audio_position != ctx->next_audio_position ||
        audio_position < ctx->audio_start ||
        audio_position >= ctx->config.max_cache_len) return record(ctx, -90);
    if (!stream_matches(ctx, stream)) return record(ctx, -91);
    if (!validate_audio_position_contract(
            ctx, audio_position, audio_rope_position, prefix_attention_mode) ||
        (prefix_attention_mode == 2u && visible_prefix_count == 0u) ||
        !std::isfinite(prefix_log_gate)) return record(ctx, -92);
    ctx->terminal_query_started = true;

    if (ctx->trace_enabled) {
        if (qwen_write_position_ids(
                ctx->trace_position_ids, audio_rope_position, audio_rope_position, stream) != 0 ||
            cudaMemsetAsync(ctx->trace_prefix_input, 0,
                            QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16), stream) != cudaSuccess ||
            cudaMemcpyAsync(ctx->trace_audio_input, audio_embed_bf16,
                            QWEN_NATIVE_HIDDEN_SIZE * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -92);
    }
    if (qwen_copy_audio_input(
            reinterpret_cast<const __nv_bfloat16 *>(audio_embed_bf16), ctx->hidden_a, stream) != 0) {
        return record(ctx, -92);
    }

    __nv_bfloat16 *current = ctx->hidden_a;
    __nv_bfloat16 *next = ctx->hidden_b;
    for (uint32_t layer = 0; layer < ctx->config.execute_layers; ++layer) {
        const int layer_status = execute_layer(
            ctx, layer, current, next, visible_prefix_position, audio_position,
            audio_rope_position, audio_rope_position, prefix_attention_mode,
            visible_prefix_count, prefix_log_gate, 1u, true, stream);
        if (layer_status != 0) return record(ctx, layer_status);
        std::swap(current, next);
    }
    if (hidden_rmsnorm_exact_graph(
            ctx, current, ctx->weights.final_norm_w, ctx->norm, 1u, stream) != 0) {
        return record(ctx, -93);
    }
    if (qwen_copy_audio_row(
            ctx->norm, reinterpret_cast<__nv_bfloat16 *>(last_hidden_bf16),
            0u, QWEN_NATIVE_HIDDEN_SIZE, stream) != 0) return record(ctx, -94);
    if (ctx->trace_enabled && qwen_copy_audio_row(
            ctx->norm, ctx->trace_output_hidden, 0u, QWEN_NATIVE_HIDDEN_SIZE, stream) != 0) {
        return record(ctx, -94);
    }

    ctx->current_audio_position = audio_position;
    ctx->next_audio_position = audio_position + 1u;
    ctx->next_audio_rope_position = audio_rope_position + 1u;
    ctx->terminal_query_position = audio_position;
    ctx->terminal_visible_prefix_position = visible_prefix_position;
    ctx->terminal_query_done = true;
    ++ctx->segment_terminal_queries;
    ctx->terminal_audio_cache_appends = 1u;
    ctx->last_cache_layer = UINT32_MAX;
    ctx->last_cache_all_layers = true;
    ctx->trace_cache_layers_written = ctx->config.execute_layers;
    remember_stream(ctx, stream);
    return 0;
}

extern "C" int qwen_talker_step_audio_async(
    qwen_talker_ctx *ctx,
    const void *audio_embed_bf16,
    const qwen_tensor_meta *audio_meta,
    uint32_t audio_rope_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    void *last_hidden_bf16,
    const qwen_tensor_meta *last_hidden_meta,
    cudaStream_t stream
) {
    if (!ctx || !ctx->imported ||
        ctx->config.execute_layers != QWEN_NATIVE_NUM_LAYERS || ctx->trace_enabled ||
        !valid_embedding_meta(audio_embed_bf16, audio_meta, ctx->device) ||
        !valid_embedding_meta(last_hidden_bf16, last_hidden_meta, ctx->device)) return record(ctx, -95);
    if (!stream_matches(ctx, stream)) return record(ctx, -96);
    if (ctx->q1_graph_capture_active) {
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (!ctx->q1_graph_stream_bound || stream != ctx->q1_graph_stream ||
            prefix_attention_mode != 0u || visible_prefix_count != 0u ||
            cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess ||
            capture_status != cudaStreamCaptureStatusActive) return record(ctx, -96);
    }
    if (ctx->mode == qwen_execution_mode::joint_q2) {
        if (ctx->next_prefix_position != ctx->audio_start || ctx->segment_joint_steps == 0u ||
            ctx->next_audio_position < ctx->audio_start ||
            ctx->next_audio_position >= ctx->config.max_cache_len) return record(ctx, -95);
        ctx->mode = qwen_execution_mode::recurrent_audio_q1;
        ctx->audio_q1_limit_position = ctx->config.max_cache_len;
    }
    if (ctx->mode != qwen_execution_mode::recurrent_audio_q1 ||
        ctx->terminal_query_started || ctx->terminal_query_done ||
        ctx->next_audio_position < ctx->audio_start ||
        ctx->next_audio_position >= ctx->audio_q1_limit_position ||
        ctx->next_audio_position >= ctx->config.max_cache_len) return record(ctx, -95);

    const uint32_t audio_position = ctx->next_audio_position;
    const uint32_t visible_prefix_position = ctx->audio_start - 1u;
    if (!validate_audio_position_contract(
            ctx, audio_position, audio_rope_position, prefix_attention_mode) ||
        (prefix_attention_mode == 2u &&
         visible_prefix_count != ctx->audio_start - ctx->prefix_start) ||
        !std::isfinite(prefix_log_gate)) return record(ctx, -95);
    if (qwen_copy_audio_input(
            reinterpret_cast<const __nv_bfloat16 *>(audio_embed_bf16), ctx->hidden_a, stream) != 0) {
        return record(ctx, -97);
    }

    __nv_bfloat16 *current = ctx->hidden_a;
    __nv_bfloat16 *next = ctx->hidden_b;
    for (uint32_t layer = 0; layer < ctx->config.execute_layers; ++layer) {
        const int layer_status = execute_layer(
            ctx, layer, current, next, visible_prefix_position, audio_position,
            audio_rope_position, audio_rope_position, prefix_attention_mode,
            visible_prefix_count, prefix_log_gate, 1u, true, stream,
            ctx->q1_graph_capture_active ? ctx->q1_device_position : nullptr);
        if (layer_status != 0) return record(ctx, layer_status);
        std::swap(current, next);
    }
    if (hidden_rmsnorm_exact_graph(
            ctx, current, ctx->weights.final_norm_w, ctx->norm, 1u, stream) != 0) {
        return record(ctx, -98);
    }
    if (qwen_copy_audio_row(
            ctx->norm, reinterpret_cast<__nv_bfloat16 *>(last_hidden_bf16),
            0u, QWEN_NATIVE_HIDDEN_SIZE, stream) != 0) return record(ctx, -99);
    if (ctx->q1_graph_capture_active) {
        if (qwen_increment_audio_q1_cursor(ctx->q1_device_position, stream) != 0) return record(ctx, -100);
    } else {
        ctx->current_audio_position = audio_position;
        ctx->next_audio_position = audio_position + 1u;
        ctx->next_audio_rope_position = audio_rope_position + 1u;
        ctx->last_cache_layer = UINT32_MAX;
        ctx->last_cache_all_layers = true;
    }
    remember_stream(ctx, stream);
    return 0;
}

extern "C" int qwen_talker_q1_graph_bind_stream(qwen_talker_ctx *ctx, cudaStream_t stream) {
    if (!ctx || !stream || ctx->trace_enabled || !ctx->blas || !ctx->q1_cublas_workspace) return -1;
    if (ctx->q1_graph_stream_bound && ctx->q1_graph_stream != stream) return -2;
    if (check_cublas(ctx, cublasSetStream(ctx->blas, stream)) != 0 ||
        check_cublas(ctx, cublasSetWorkspace(
            ctx->blas, ctx->q1_cublas_workspace, QWEN_NATIVE_Q1_CUBLAS_WORKSPACE_BYTES)) != 0) return -3;
    ctx->q1_graph_stream = stream;
    ctx->q1_graph_stream_bound = true;
    remember_stream(ctx, stream);
    return 0;
}

extern "C" int qwen_talker_q1_graph_capture_begin(qwen_talker_ctx *ctx) {
    if (!ctx || !ctx->q1_graph_stream_bound || !ctx->imported ||
        ctx->mode != qwen_execution_mode::recurrent_audio_q1 || ctx->trace_enabled ||
        ctx->q1_graph_capture_active || ctx->q1_graph_captured ||
        ctx->next_audio_position >= ctx->audio_q1_limit_position) return -1;
    ctx->q1_graph_capture_active = true;
    return 0;
}

extern "C" int qwen_talker_q1_graph_capture_end(qwen_talker_ctx *ctx, uint32_t success) {
    if (!ctx || !ctx->q1_graph_capture_active || success > 1u) return -1;
    ctx->q1_graph_capture_active = false;
    ctx->q1_graph_captured = success == 1u;
    if (!success) ctx->q1_graph_replays = 0u;
    return success ? 0 : -2;
}

extern "C" int qwen_talker_q1_graph_can_replay(qwen_talker_ctx *ctx) {
    if (!ctx || !ctx->q1_graph_stream_bound || !ctx->q1_graph_captured ||
        ctx->q1_graph_capture_active || !ctx->imported ||
        ctx->mode != qwen_execution_mode::recurrent_audio_q1 || ctx->trace_enabled ||
        ctx->terminal_query_started || ctx->terminal_query_done ||
        ctx->next_audio_position < ctx->audio_start ||
        ctx->next_audio_position >= ctx->audio_q1_limit_position ||
        ctx->next_audio_position >= ctx->config.max_cache_len) return -1;
    return 0;
}

extern "C" int qwen_talker_q1_graph_replay_commit(qwen_talker_ctx *ctx) {
    if (qwen_talker_q1_graph_can_replay(ctx) != 0) return -1;
    ctx->current_audio_position = ctx->next_audio_position;
    ++ctx->next_audio_position;
    ++ctx->next_audio_rope_position;
    ctx->last_cache_layer = UINT32_MAX;
    ctx->last_cache_all_layers = true;
    ++ctx->q1_graph_replays;
    return 0;
}

extern "C" int qwen_talker_q1_graph_invalidate(qwen_talker_ctx *ctx) {
    if (!ctx || ctx->q1_graph_capture_active) return -1;
    ctx->q1_graph_captured = false;
    ctx->q1_graph_replays = 0u;
    return 0;
}

extern "C" int qwen_talker_q1_graph_snapshot_device_state(
    qwen_talker_ctx *ctx, void *device_output_u32x2, cudaStream_t stream
) {
    if (!ctx) return -1;
    if (!device_output_u32x2 || !ctx->q1_device_position || !ctx->q1_device_error ||
        !ctx->q1_graph_stream_bound || !stream_matches(ctx, stream) ||
        !is_device_pointer(device_output_u32x2, ctx->device)) return record(ctx, -3);
    auto *output = reinterpret_cast<uint32_t *>(device_output_u32x2);
    if (cudaMemcpyAsync(output, ctx->q1_device_position, sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(output + 1, ctx->q1_device_error, sizeof(uint32_t),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return record(ctx, -4);
    remember_stream(ctx, stream);
    return 0;
}

extern "C" int32_t qwen_talker_runtime_q1_device_cursor_supported(const qwen_talker_ctx *ctx) {
    return ctx && ctx->q1_device_position ? 1 : 0;
}

extern "C" int32_t qwen_talker_runtime_q1_graph_captured(const qwen_talker_ctx *ctx) {
    return ctx && ctx->q1_graph_captured ? 1 : 0;
}

extern "C" uint64_t qwen_talker_runtime_q1_graph_replays(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->q1_graph_replays : 0u;
}

extern "C" void qwen_talker_destroy(qwen_talker_ctx *ctx) {
    if (!ctx) return;
    if (ctx->has_last_stream) cudaStreamSynchronize(ctx->last_stream);
    if (ctx->rmsnorm_graph_q1) {
        qwen_talker_rmsnorm_graph_destroy(ctx->rmsnorm_graph_q1);
    }
    if (ctx->rmsnorm_graph) {
        qwen_talker_rmsnorm_graph_destroy(ctx->rmsnorm_graph);
    }
    if (ctx->attention_graph) qwen_talker_attention_graph_destroy(ctx->attention_graph);
    if (ctx->blas) cublasDestroy(ctx->blas);
    if (ctx->q1_cublas_workspace) cudaFree(ctx->q1_cublas_workspace);
    if (ctx->arena) cudaFree(ctx->arena);
    delete ctx;
}

extern "C" uint32_t qwen_talker_abi_version(void) { return 4u; }
extern "C" uint64_t qwen_talker_runtime_workspace_bytes(const qwen_talker_ctx *ctx) { return ctx ? ctx->arena_bytes : 0u; }
extern "C" uint64_t qwen_talker_runtime_joint_steps(const qwen_talker_ctx *ctx) { return ctx ? ctx->joint_steps : 0u; }
extern "C" uint64_t qwen_talker_runtime_terminal_query_count(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->segment_terminal_queries : 0u;
}
extern "C" uint64_t qwen_talker_runtime_terminal_prefix_cache_appends(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->terminal_prefix_cache_appends : 0u;
}
extern "C" uint64_t qwen_talker_runtime_terminal_audio_cache_appends(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->terminal_audio_cache_appends : 0u;
}
extern "C" uint32_t qwen_talker_runtime_terminal_query_position(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->terminal_query_position : UINT32_MAX;
}
extern "C" uint32_t qwen_talker_runtime_terminal_visible_prefix_position(const qwen_talker_ctx *ctx) {
    return ctx ? ctx->terminal_visible_prefix_position : UINT32_MAX;
}
extern "C" uint64_t qwen_talker_runtime_cache_imports(const qwen_talker_ctx *ctx) { return ctx ? ctx->cache_imports : 0u; }
extern "C" uint32_t qwen_talker_runtime_execute_layers(const qwen_talker_ctx *ctx) { return ctx ? ctx->config.execute_layers : 0u; }
extern "C" uint32_t qwen_talker_runtime_current_prefix_position(const qwen_talker_ctx *ctx) { return ctx ? ctx->current_prefix_position : 0u; }
extern "C" uint32_t qwen_talker_runtime_current_audio_position(const qwen_talker_ctx *ctx) { return ctx ? ctx->current_audio_position : 0u; }
extern "C" uint32_t qwen_talker_runtime_max_cache_len(const qwen_talker_ctx *ctx) { return ctx ? ctx->config.max_cache_len : 0u; }
extern "C" int32_t qwen_talker_runtime_last_error(const qwen_talker_ctx *ctx) { return ctx ? ctx->last_error : -1; }
extern "C" int32_t qwen_talker_runtime_attention_graph_captured(const qwen_talker_ctx *ctx) {
    return ctx ? qwen_talker_attention_graph_captured(ctx->attention_graph) : 0;
}

extern "C"
int32_t qwen_talker_runtime_rmsnorm_graph_captured(
    const qwen_talker_ctx *ctx
) {
    return ctx
        ? qwen_talker_rmsnorm_graph_captured(
              ctx->rmsnorm_graph
          )
        : 0;
}
