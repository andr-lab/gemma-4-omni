/*
 * gemma4_listener.cu — Core Native Executor for Gemma 4 Listener LM.
 *
 * Single C ABI entrypoint, device-resident preallocated arena, direct cuBLAS GEMMs,
 * and zero allocations during steady-state.
 * Conforms to P4-NATIVE Implementation Plan §6, §7, §8, §9, §17, §18.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "gemma4_listener.h"
#include "gemma4_listener_aten.h"
#include "gemma4_listener_aten_norm.h"
#include "generated_topology.h"
#include "gemma4_listener_kernels.cuh"
#include "gemma4_listener_attention_ordered.cuh"
#include "listener_cuda_create_failure_log.cuh"
#include "sliding_repeat_capacity.h"

/* Internal forward declarations for cache functions */
typedef struct gemma4_cache gemma4_cache;
int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len, bool allocate_device_position);
int gemma4_cache_append_async(gemma4_cache *c, int layer_idx, const __nv_bfloat16 *new_k, const __nv_bfloat16 *new_v, int q_len, cudaStream_t stream);
int gemma4_cache_append_device_position_async(gemma4_cache *c, int layer_idx, const __nv_bfloat16 *new_k, const __nv_bfloat16 *new_v, int q_len, cudaStream_t stream);
int gemma4_cache_set_device_position_async(gemma4_cache *c, uint32_t position, cudaStream_t stream);
int gemma4_cache_advance_device_position_async(gemma4_cache *c, uint32_t amount, cudaStream_t stream);
const uint32_t *gemma4_cache_device_position(const gemma4_cache *c);
int gemma4_cache_import_async(gemma4_cache *c, const gemma4_imported_cache *imported, uint32_t current_position, cudaStream_t stream);
int gemma4_cache_rebase_async(
    gemma4_cache *c, uint32_t prefix_len, uint32_t retained_audio_tokens,
    cudaStream_t stream);
int gemma4_cache_advance_position(gemma4_cache *c, int q_len);
uint32_t gemma4_cache_valid_kv_len(const gemma4_cache *c);
void gemma4_cache_get_ptrs(const gemma4_cache *c, int layer_idx, int q_len, uint32_t base_pos, const __nv_bfloat16 **out_k, const __nv_bfloat16 **out_v, int *out_total_len, int *out_start_offset);
int gemma4_cache_reset(gemma4_cache *c);
void gemma4_cache_destroy(gemma4_cache *c);
const void *gemma4_cache_get_layer_k(const gemma4_cache *c, int layer_idx);
const void *gemma4_cache_get_layer_v(const gemma4_cache *c, int layer_idx);

#define GEMMA4_RMS_WEIGHT_MIRROR_CAPACITY (1 + GEMMA4_NUM_LAYERS * 8)
typedef struct {
    const void *bf16_ptr;
    float *fp32_ptr;
    uint32_t dim;
    bool ready;
} gemma4_rms_weight_mirror;

/* Internal Context Definition */
struct gemma4_listener_ctx {
    gemma4_listener_config config;
    gemma4_listener_weights weights;
    gemma4_cache *cache;
    cublasHandle_t cublas;

    /* Precomputed RoPE frequencies (device) */
    float *d_inv_freq_sliding; /* [128] */
    float *d_inv_freq_full;    /* [256] */

    /* Canonical prefix state copied once at prime time. */
    int64_t *d_prefix_attention; /* [canonical_prefix_len] */
    __nv_bfloat16 *d_prefix_ple; /* [canonical_prefix_len, layers, ple_dim] */
    uint32_t prefix_len;
    bool primed;
    bool poisoned;
    cudaStream_t last_stream;
    bool debug_capture_enabled;
    bool debug_layer_trace_enabled;

    /* Fixed Preallocated Device Scratch Arena */
    __nv_bfloat16 *d_hidden_a;        /* [q_max, 1536] */
    __nv_bfloat16 *d_hidden_b;        /* [q_max, 1536] */
    __nv_bfloat16 *d_norm_buf;        /* [q_max, 1536] */

    __nv_bfloat16 *d_q_buf;           /* [q_max, 8 * 512] */
    __nv_bfloat16 *d_k_buf;           /* [q_max, 1 * 512] */
    __nv_bfloat16 *d_v_buf;           /* [q_max, 1 * 512] */
    __nv_bfloat16 *d_attn_out;        /* [q_max, 8 * 512] */
    bool *d_sdpa_mask;                 /* diagnostic ATen bridge [q_max, max_seq_len] */
    /* Persistent explicit repeat-KV staging for the masked ATen oracle.  The
     * bridge receives contiguous [1, 8, K, D] views, so it does not allocate
     * or materialize K/V through expand+reshape on every layer. */
    __nv_bfloat16 *d_sdpa_k_repeat;    /* compact [8, sliding_capacity, max_sliding_D] */
    __nv_bfloat16 *d_sdpa_v_repeat;    /* compact [8, sliding_capacity, max_sliding_D] */
    /* Full-attention fallback sources 4, 9 and 14 keep one compact repeated
     * cache each. Backend-1 consumers therefore see the same contiguous K/V
     * layout as the literal ATen repeat-KV oracle. */
    __nv_bfloat16 *d_sdpa_full_k_repeat[3]; /* compact [8, max_seq_len, 512] */
    __nv_bfloat16 *d_sdpa_full_v_repeat[3]; /* compact [8, max_seq_len, 512] */
    /* Source 13 backs layer 13 and all later shared sliding layers. Keep one
     * compact contiguous oracle staging copy per frame instead of repeating
     * the same K/V once for every consumer layer. */
    __nv_bfloat16 *d_sdpa_shared_sliding_k_repeat; /* [8, sliding_capacity, max_sliding_D] */
    __nv_bfloat16 *d_sdpa_shared_sliding_v_repeat; /* [8, sliding_capacity, max_sliding_D] */
    gemma4_sliding_staging::Contract sliding_staging;
    gemma4_ordered_workspace ordered_workspace; /* persistent CUTLASS_ORDERED_TILES_V1 storage */
    uint32_t attention_backend;        /* CUSTOM, ATEN_REPEAT_EXACT, ATEN_GQA_EXACT, NATIVE_SPLITK_GQA */
    uint64_t splitk_layer_mask;        /* explicit full-attention layer eligibility */
    uint32_t rmsnorm_backend;           /* CUSTOM_FAST, ATEN_EXACT oracle, NATIVE_EXACT */
    gemma4_aten_rms_workspace rms_workspace; /* persistent ATen oracle scratch */
    gemma4_rms_exact_workspace native_rms_workspace; /* persistent native scratch */
    gemma4_rms_weight_mirror rms_weight_mirrors[GEMMA4_RMS_WEIGHT_MIRROR_CAPACITY];
    uint32_t rms_weight_mirror_count;
    size_t rms_derived_weight_bytes;
    uint64_t ordered_calls_by_layer[GEMMA4_NUM_LAYERS];

    __nv_bfloat16 *d_gate_buf;        /* [q_max, 6144] */
    __nv_bfloat16 *d_up_buf;          /* [q_max, 6144] */
    __nv_bfloat16 *d_mlp_act_buf;     /* [q_max, 6144] */
    __nv_bfloat16 *d_mlp_down_buf;    /* [q_max, 1536] */

    __nv_bfloat16 *d_ple_raw_proj;    /* [q_max, 35 * 256] */
    __nv_bfloat16 *d_projected_ple;   /* [q_max, 35 * 256] */
    __nv_bfloat16 *d_ple_gate_buf;    /* [q_max, 256] */
    __nv_bfloat16 *d_ple_act_buf;     /* [q_max, 256] */
    __nv_bfloat16 *d_ple_proj_out;    /* [q_max, 1536] */

    __nv_bfloat16 *d_output_tensor;   /* [4, 1536] persistent user output */
    int *d_output_health;              /* device flag: non-finite output observed */
    bool owns_output_tensor;

    /* Diagnostic-only tensor taps. Each slot stores one contiguous Q=4
     * tensor; widths are overprovisioned for the selected layer's topology. */
    __nv_bfloat16 *d_debug_hidden_taps; /* 16 * [4, 1536] */
    __nv_bfloat16 *d_debug_q_taps;      /* 4 * [4, 4096] */
    __nv_bfloat16 *d_debug_kv_taps;     /* 5 * [4, 512] */
    __nv_bfloat16 *d_debug_ffn_taps;    /* 4 * [4, 12288] */
    __nv_bfloat16 *d_debug_ple_taps;    /* 3 * [4, 256] */
    __nv_bfloat16 *d_debug_layer_outputs; /* [35, 4, 1536], debug step only */

    /* CUDA Graph handle for steady-state step */
    bool graph_captured;
    bool graph_capture_active;
    cudaGraph_t cuda_graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t graph_stream;
    const void *graph_input_ptr;
    uint64_t state_generation;
    uint64_t graph_generation;
    uint32_t current_position;
};

static void destroy_graph_handles(gemma4_listener_ctx *ctx) {
    if (!ctx) return;
    if (ctx->graph_exec) cudaGraphExecDestroy(ctx->graph_exec);
    if (ctx->cuda_graph) cudaGraphDestroy(ctx->cuda_graph);
    ctx->graph_exec = nullptr;
    ctx->cuda_graph = nullptr;
    ctx->graph_captured = false;
    ctx->graph_input_ptr = nullptr;
    ctx->graph_stream = nullptr;
}

static int invalidate_graph_for_state_change(gemma4_listener_ctx *ctx, cudaStream_t stream) {
    if (!ctx) return -1;
    if (ctx->graph_captured) {
        cudaStream_t fence_stream = ctx->graph_stream ? ctx->graph_stream : stream;
        if (fence_stream && cudaStreamSynchronize(fence_stream) != cudaSuccess) return -2;
        destroy_graph_handles(ctx);
    }
    ++ctx->state_generation;
    return 0;
}

/* Observe completed work without waiting.  A stream query is deliberately
 * non-blocking: cudaErrorNotReady means that the prior transaction is still
 * ordered on the stream and is not itself a failure.  Any other status is a
 * completed asynchronous CUDA fault and poisons the context before more work
 * can be enqueued.  `verify_output_health()` remains the explicit synchronizing
 * health boundary for callers that need a definitive completion result. */
static int observe_async_stream_error(gemma4_listener_ctx *ctx) {
    if (!ctx || !ctx->last_stream) return 0;
    cudaError_t status = cudaStreamQuery(ctx->last_stream);
    if (status == cudaSuccess || status == cudaErrorNotReady) return 0;
    ctx->poisoned = true;
    fprintf(stderr, "[gemma4_listener] asynchronous stream failure: %s\\n", cudaGetErrorString(status));
    return -1;
}

static const uint64_t GEMMA4_SPLITK_STATE_SAFE_HYBRID_MASK =
    (UINT64_C(1) << 19) | (UINT64_C(1) << 24) |
    (UINT64_C(1) << 29) | (UINT64_C(1) << 34);

static const uint64_t GEMMA4_SPLITK_ALL_FULL_MASK =
    (UINT64_C(1) << 4) | (UINT64_C(1) << 9) |
    (UINT64_C(1) << 14) | GEMMA4_SPLITK_STATE_SAFE_HYBRID_MASK;

static int parse_splitk_layer_mask(const char *raw, uint64_t *out_mask) {
    if (!out_mask) return -1;
    const char *value = raw ? raw : "";
    while (*value == ' ' || *value == '\t' || *value == '\r' || *value == '\n') ++value;
    if (*value == '\0' || strcmp(value, "STATE_SAFE_HYBRID") == 0) {
        *out_mask = GEMMA4_SPLITK_STATE_SAFE_HYBRID_MASK;
        return 0;
    }
    if (strcmp(value, "ALL_FULL") == 0) {
        *out_mask = GEMMA4_SPLITK_ALL_FULL_MASK;
        return 0;
    }
    if (strcmp(value, "NONE") == 0) {
        *out_mask = 0;
        return 0;
    }

    char buffer[256];
    const size_t length = strlen(value);
    if (length == 0 || length >= sizeof(buffer)) return -1;
    memcpy(buffer, value, length + 1);
    uint64_t mask = 0;
    char *token = strtok(buffer, ",; \t\r\n");
    if (!token) return -1;
    for (; token; token = strtok(NULL, ",; \t\r\n")) {
        errno = 0;
        char *end = NULL;
        const long layer = strtol(token, &end, 10);
        if (errno == ERANGE || end == token || *end != '\0' ||
            layer < 0 || layer >= GEMMA4_NUM_LAYERS ||
            gemma4_layers[layer].attn_type != GEMMA_ATTN_FULL) return -1;
        const uint64_t bit = UINT64_C(1) << (unsigned)layer;
        if (mask & bit) return -1;
        mask |= bit;
    }
    *out_mask = mask;
    return 0;
}

static bool is_current_device_pointer(const void *ptr, int current_device) {
    if (!ptr) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) return false;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == current_device;
}

static bool validate_tensor_meta(
    const gemma4_tensor_meta *meta,
    const void *raw_ptr,
    int current_device,
    uint32_t dtype,
    uint32_t ndim,
    const uint64_t *dims
) {
    if (!meta || meta->ptr != raw_ptr) return false;
    if (!raw_ptr) {
        if (meta->dtype != 0 || meta->ndim != 0 || meta->contiguous != 0 || meta->reserved != 0) return false;
        for (int i = 0; i < GEMMA4_TENSOR_META_MAX_RANK; ++i) {
            if (meta->dims[i] != 0 || meta->strides[i] != 0) return false;
        }
        return true;
    }
    if (meta->ndim != ndim || ndim > GEMMA4_TENSOR_META_MAX_RANK ||
        meta->dtype != dtype || meta->contiguous != 1 || meta->reserved != 0) {
        return false;
    }
    if (!is_current_device_pointer(raw_ptr, current_device) || ((uintptr_t)raw_ptr & 1u) != 0u) {
        return false;
    }
    uint64_t expected_strides[GEMMA4_TENSOR_META_MAX_RANK] = {0, 0, 0, 0};
    uint64_t running = 1;
    for (int i = (int)ndim - 1; i >= 0; --i) {
        if (!dims[i] || dims[i] > UINT64_MAX / running) return false;
        expected_strides[i] = running;
        running *= dims[i];
    }
    for (int i = 0; i < GEMMA4_TENSOR_META_MAX_RANK; ++i) {
        if (i < (int)ndim) {
            if (meta->dims[i] != dims[i] || meta->strides[i] != expected_strides[i]) return false;
        } else if (meta->dims[i] != 0 || meta->strides[i] != 0) {
            return false;
        }
    }
    return true;
}

static int validate_imported_cache_metadata(
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    int current_device
) {
    if (!imported || current_position == 0) return -1;
    for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
        const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
        const gemma4_imported_kv *source = &imported->layers[layer_idx];
        if (spec->is_kv_shared) {
            if (source->k_ptr || source->k_meta || source->v_ptr || source->v_meta) return -1;
            continue;
        }
        const uint64_t dims[] = {1, 1, current_position, spec->head_dim};
        if (!validate_tensor_meta(
                source->k_meta, source->k_ptr, current_device,
                GEMMA4_META_BF16, 4, dims) ||
            !validate_tensor_meta(
                source->v_meta, source->v_ptr, current_device,
                GEMMA4_META_BF16, 4, dims)) return -1;
    }
    return 0;
}

static bool validate_layer_device_pointers(
    const gemma4_layer_weights *weights,
    const GemmaLayerSpec *spec,
    int current_device
) {
    if (!weights || !spec) return false;
    const void *required[] = {
        weights->input_norm_w, weights->post_attn_norm_w,
        weights->pre_ff_norm_w, weights->post_ff_norm_w,
        weights->q_proj_w, weights->q_norm_w, weights->o_proj_w,
        weights->gate_proj_w, weights->up_proj_w, weights->down_proj_w,
        weights->ple_gate_w, weights->ple_proj_w, weights->ple_norm_w,
    };
    for (const void *ptr : required) {
        if (!is_current_device_pointer(ptr, current_device)) return false;
    }
    if (!spec->is_kv_shared) {
        const void *kv_required[] = {
            weights->k_proj_w, weights->v_proj_w,
            weights->k_norm_w,
        };
        for (const void *ptr : kv_required) {
            if (!is_current_device_pointer(ptr, current_device)) return false;
        }
    }
    return true;
}

/* Device-side output guard. Non-finite values are replaced with zero and
 * recorded in a persistent device flag; host inspection is an explicit
 * non-steady-state health operation. */
static int debug_snapshot_bf16(
    gemma4_listener_ctx *ctx,
    __nv_bfloat16 *base,
    int slot,
    int slot_elements,
    const __nv_bfloat16 *source,
    int elements,
    cudaStream_t stream
) {
    if (!ctx || !ctx->debug_capture_enabled) return 0;
    if (!base || !source || slot < 0 || elements < 0 ||
        elements > slot_elements || slot_elements <= 0) return -1;
    __nv_bfloat16 *destination = base + (size_t)slot * slot_elements;
    return cudaMemcpyAsync(
        destination, source, (size_t)elements * sizeof(__nv_bfloat16),
        cudaMemcpyDeviceToDevice, stream
    ) == cudaSuccess ? 0 : -2;
}

static int guard_output_finite(
    gemma4_listener_ctx *ctx,
    __nv_bfloat16 *output,
    int count,
    cudaStream_t stream
) {
    if (!ctx || !output || !ctx->d_output_health || count <= 0) return -1;
    if (cudaMemsetAsync(ctx->d_output_health, 0, sizeof(int), stream) != cudaSuccess) return -2;
    kernel_finite_guard_bf16<<<(count + 255) / 256, 256, 0, stream>>>(
        output, count, ctx->d_output_health
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

/* Direct cuBLAS GEMM path using the same column-major reinterpretation as
 * PyTorch row-major F.linear: Y[M,N] = X[M,K] * W[N,K]^T. */
static int run_linear_cublas(
    cublasHandle_t handle,
    const void *d_w,
    const void *d_x,
    void *d_y,
    int M,
    int N,
    int K,
    cudaStream_t stream
) {
    if (!handle || !d_w || !d_x || !d_y || M <= 0 || N <= 0 || K <= 0) return -1;
    cublasStatus_t status = cublasSetStream(handle, stream);
    if (status != CUBLAS_STATUS_SUCCESS) return -2;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    status = cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        d_w,
        CUDA_R_16BF,
        K,
        d_x,
        CUDA_R_16BF,
        K,
        &beta,
        d_y,
        CUDA_R_16BF,
        N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    return status == CUBLAS_STATUS_SUCCESS ? 0 : -3;
}

static gemma4_rms_weight_mirror *find_rms_weight_mirror(
    gemma4_listener_ctx *ctx,
    const void *weight
) {
    if (!ctx || !weight) return nullptr;
    for (uint32_t i = 0; i < ctx->rms_weight_mirror_count; ++i) {
        if (ctx->rms_weight_mirrors[i].bf16_ptr == weight) {
            return &ctx->rms_weight_mirrors[i];
        }
    }
    return nullptr;
}

static bool rmsnorm_uses_exact_path(const gemma4_listener_ctx *ctx) {
    return ctx && ctx->rmsnorm_backend != GEMMA4_RMS_CUSTOM_FAST;
}

static int run_rmsnorm_exact_or_custom(
    gemma4_listener_ctx *ctx,
    const __nv_bfloat16 *input,
    const void *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    cudaStream_t stream
) {
    if (!ctx) return -1;
    if (!input || !output || rows == 0) {
        ctx->poisoned = true;
        return -1;
    }

    int status = 0;
    if (ctx->rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT ||
        ctx->rmsnorm_backend == GEMMA4_RMS_NATIVE_EXACT) {
        gemma4_rms_weight_mirror *mirror = find_rms_weight_mirror(ctx, weight);
        const void *cached_weight_fp32 = nullptr;
        if (weight) {
            if (!mirror || mirror->dim != dim) {
                ctx->poisoned = true;
                return -2;
            }
            if (!mirror->ready) {
                if (gemma4_aten_rmsnorm_cache_weight_async(
                        weight, mirror->fp32_ptr, dim, stream) != 0) {
                    ctx->poisoned = true;
                    return -3;
                }
                mirror->ready = true;
            }
            cached_weight_fp32 = mirror->fp32_ptr;
        }
        if (ctx->rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT) {
            status = gemma4_aten_rmsnorm_preallocated_async(
                input, weight, output, rows, dim, eps,
                &ctx->rms_workspace, cached_weight_fp32, stream
            );
        } else {
            status = gemma4_rmsnorm_native_exact_async(
                input, (const float *)cached_weight_fp32, output, rows, dim,
                (float)eps, &ctx->native_rms_workspace, stream
            );
        }
    } else if (ctx->rmsnorm_backend == GEMMA4_RMS_CUSTOM_FAST && dim == 1536) {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
        status = cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
    } else if (ctx->rmsnorm_backend == GEMMA4_RMS_CUSTOM_FAST && dim == 256) {
        kernel_rmsnorm_bf16<256><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
        status = cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
    } else if (ctx->rmsnorm_backend == GEMMA4_RMS_CUSTOM_FAST && dim == 512) {
        kernel_rmsnorm_bf16<512><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
        status = cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
    } else {
        status = -2;
    }

    if (status != 0) ctx->poisoned = true;
    return status;
}

static gemma4_tensor_meta make_bridge_meta(
    const void *ptr,
    uint32_t dtype,
    uint32_t ndim,
    const uint64_t *dims,
    const uint64_t *strides,
    uint32_t contiguous
) {
    gemma4_tensor_meta meta = {};
    meta.ptr = ptr;
    meta.dtype = dtype;
    meta.ndim = ndim;
    meta.contiguous = contiguous;
    for (uint32_t index = 0; index < ndim && index < GEMMA4_TENSOR_META_MAX_RANK; ++index) {
        meta.dims[index] = dims[index];
        meta.strides[index] = strides[index];
    }
    return meta;
}

static int full_repeat_slot_for_layer(int layer_idx) {
    if (layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS) return -1;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    const int source = spec->is_kv_shared ? spec->kv_shared_src_index : layer_idx;
    if (source == 4) return 0;
    if (source == 9) return 1;
    if (source == 14) return 2;
    return -1;
}

static int stage_shared_sliding_source13(
    gemma4_listener_ctx *ctx,
    int q_len,
    uint32_t base_pos,
    cudaStream_t stream
) {
    if (!ctx || !ctx->cache || !ctx->d_sdpa_shared_sliding_k_repeat ||
        !ctx->d_sdpa_shared_sliding_v_repeat || q_len <= 0) return -1;
    if (gemma4_sliding_staging::validate_position_query(
            base_pos, gemma4_cache_valid_kv_len(ctx->cache), q_len, q_len,
            GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN) !=
        gemma4_sliding_staging::Status::kOk) return -2;
    const __nv_bfloat16 *source_k = nullptr;
    const __nv_bfloat16 *source_v = nullptr;
    int total_k_len = 0;
    int k_start_offset = 0;
    gemma4_cache_get_ptrs(
        ctx->cache, 13, q_len, base_pos,
        &source_k, &source_v, &total_k_len, &k_start_offset
    );
    const int visible_k_len = total_k_len - k_start_offset;
    const uint32_t head_dim = gemma4_layers[13].head_dim;
    if (!source_k || !source_v || visible_k_len <= 0 ||
        gemma4_sliding_staging::validate_visible_window(
            visible_k_len, ctx->config.max_seq_len, ctx->sliding_staging) !=
            gemma4_sliding_staging::Status::kOk ||
        gemma4_sliding_staging::validate_head_dim(
            head_dim, ctx->sliding_staging) !=
            gemma4_sliding_staging::Status::kOk ||
        k_start_offset < 0 || k_start_offset > total_k_len) return -2;
    const int repeat_blocks = ((visible_k_len * (int)head_dim) + 255) / 256;
    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
        source_k + (size_t)k_start_offset * head_dim,
        ctx->d_sdpa_shared_sliding_k_repeat,
        visible_k_len,
        (int)head_dim
    );
    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
        source_v + (size_t)k_start_offset * head_dim,
        ctx->d_sdpa_shared_sliding_v_repeat,
        visible_k_len,
        (int)head_dim
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

/* Execute 1 layer of Gemma 4 text decoder */
static int execute_layer(
    gemma4_listener_ctx *ctx,
    int layer_idx,
    int compute_q_len,
    int valid_q_len,
    uint32_t base_pos,
    cudaStream_t stream
) {
    if (!ctx || !ctx->cache || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS ||
        gemma4_sliding_staging::validate_position_query(
            base_pos, gemma4_cache_valid_kv_len(ctx->cache), compute_q_len,
            valid_q_len, GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN) !=
            gemma4_sliding_staging::Status::kOk) return -8;
    int q_len = compute_q_len;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    const gemma4_layer_weights *lw = &ctx->weights.layers[layer_idx];
    const uint32_t *device_position = ctx->graph_capture_active
        ? gemma4_cache_device_position(ctx->cache)
        : nullptr;
    int D = spec->head_dim;
    int hidden_size = GEMMA4_HIDDEN_SIZE;
    int intermediate_size = spec->intermediate_size;

    /* 1. Input RMSNorm: norm(hidden_a) -> norm_buf */
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 0,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;
    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_a, lw->input_norm_w, ctx->d_norm_buf,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->input_norm_eps, stream) != 0) return -8;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 1,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_norm_buf, q_len * hidden_size, stream) != 0) return -7;

    /* 2. Attention: Q projection */
    int q_dim = GEMMA4_NUM_ATTN_HEADS * D;
    if (run_linear_cublas(
        ctx->cublas,
        lw->q_proj_w,
        ctx->d_norm_buf,
        ctx->d_q_buf,
        q_len, q_dim, hidden_size,
        stream
    ) != 0) return -4;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_q_taps, 0,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512,
                             ctx->d_q_buf, q_len * q_dim, stream) != 0) return -7;

    /* Q norm */
    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_q_buf, lw->q_norm_w, ctx->d_q_buf,
            (uint32_t)(GEMMA4_NUM_ATTN_HEADS * q_len), (uint32_t)D,
            (double)spec->q_norm_eps, stream) != 0) return -8;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_q_taps, 1,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512,
                             ctx->d_q_buf, q_len * q_dim, stream) != 0) return -7;

    /* Q RoPE (Layout is [q_len, num_heads, head_dim]) */
    const float *inv_freq = (spec->attn_type == GEMMA_ATTN_FULL) ? ctx->d_inv_freq_full : ctx->d_inv_freq_sliding;
    dim3 rope_grid(q_len, GEMMA4_NUM_ATTN_HEADS);
    kernel_rope_bf16<<<rope_grid, D / 2, 0, stream>>>(
        ctx->d_q_buf,
        inv_freq,
        GEMMA4_NUM_ATTN_HEADS,
        q_len,
        D,
        base_pos,
        device_position,
        spec->attn_scaling
    );
    if (debug_snapshot_bf16(ctx, ctx->d_debug_q_taps, 2,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512,
                             ctx->d_q_buf, q_len * q_dim, stream) != 0) return -7;

    /* K & V computation if not shared */
    if (!spec->is_kv_shared) {
        if (run_linear_cublas(
            ctx->cublas,
            lw->k_proj_w,
            ctx->d_norm_buf,
            ctx->d_k_buf,
            q_len, D, hidden_size,
            stream
        ) != 0) return -4;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 0,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_k_buf, q_len * D, stream) != 0) return -7;
        if (run_linear_cublas(
            ctx->cublas,
            lw->v_proj_w,
            ctx->d_norm_buf,
            ctx->d_v_buf,
            q_len, D, hidden_size,
            stream
        ) != 0) return -4;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 3,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_v_buf, q_len * D, stream) != 0) return -7;

        /* K norm */
        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_k_buf, lw->k_norm_w, ctx->d_k_buf,
                (uint32_t)q_len, (uint32_t)D,
                (double)spec->k_norm_eps, stream) != 0) return -8;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 1,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_k_buf, q_len * D, stream) != 0) return -7;

        /* K RoPE */
        kernel_rope_bf16<<<dim3(q_len, 1), D / 2, 0, stream>>>(
            ctx->d_k_buf,
            inv_freq,
            1,
            q_len,
            D,
            base_pos,
            device_position,
            spec->attn_scaling
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 2,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_k_buf, q_len * D, stream) != 0) return -7;

        /* V norm */
        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_v_buf, lw->v_norm_w, ctx->d_v_buf,
                (uint32_t)q_len, (uint32_t)D,
                (double)spec->v_norm_eps, stream) != 0) return -8;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 4,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_v_buf, q_len * D, stream) != 0) return -7;

        /* Write to KV cache */
        const int append_status = ctx->graph_capture_active
            ? gemma4_cache_append_device_position_async(
                ctx->cache, layer_idx, ctx->d_k_buf, ctx->d_v_buf, valid_q_len, stream)
            : gemma4_cache_append_async(
                ctx->cache, layer_idx, ctx->d_k_buf, ctx->d_v_buf, valid_q_len, stream);
        if (append_status != 0) {
            return -3;
        }
    }

    /* 3. SDPA Attention */
    const __nv_bfloat16 *k_data, *v_data;
    int total_k_len, k_start_offset;
    gemma4_cache_get_ptrs(ctx->cache, layer_idx, valid_q_len, base_pos, &k_data, &v_data, &total_k_len, &k_start_offset);

    float sm_scale = spec->attn_scaling; /* Exactly 1.0 in Gemma 4 */
    size_t smem_bytes = 4 * (size_t)ctx->config.max_seq_len * sizeof(float);

    if (ctx->attention_backend != GEMMA4_ATTN_CUSTOM) {
        /* Backend 3 changes only eligible full attention.  Every other case,
         * including sliding attention and prefix/prefill work, deliberately
         * remains on the backend-1 repeat-KV ATen oracle. */
        const int visible_k_len = total_k_len - k_start_offset;
        const bool use_native_ordered =
            ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA &&
            spec->attn_type == GEMMA_ATTN_FULL &&
            valid_q_len == GEMMA4_STREAM_Q_LEN && D == 512 &&
            k_start_offset == 0 &&
            ((ctx->splitk_layer_mask >> layer_idx) & UINT64_C(1)) != 0 &&
            visible_k_len > GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE;
        if (use_native_ordered) {
            if (!ctx->ordered_workspace.initialized ||
                visible_k_len > ctx->ordered_workspace.max_keys ||
                visible_k_len > GEMMA4_ORDERED_MAX_KEYS) return -8;
            if (gemma4_ordered_attention_launch(
                    ctx->d_q_buf,
                    k_data,
                    v_data,
                    ctx->d_prefix_attention,
                    ctx->prefix_len,
                    (uint32_t)visible_k_len,
                    base_pos,
                    sm_scale,
                    ctx->d_attn_out,
                    stream,
                    &ctx->ordered_workspace) != 0) return -8;
            ctx->ordered_calls_by_layer[layer_idx]++;
        } else {
            if (!ctx->d_sdpa_mask || visible_k_len <= 0) return -8;
            kernel_build_explicit_sdpa_mask<<<
            ((q_len * visible_k_len) + 255) / 256, 256, 0, stream
        >>>(
            ctx->d_sdpa_mask,
            q_len,
            visible_k_len,
            k_start_offset,
            base_pos,
            spec->sliding_window,
            ctx->d_prefix_attention,
            ctx->prefix_len
        );
        if (cudaPeekAtLastError() != cudaSuccess) return -8;

        const uint64_t q_dims[] = {1, GEMMA4_NUM_ATTN_HEADS, (uint64_t)q_len, (uint64_t)D};
        const bool use_gqa =
            ctx->attention_backend == GEMMA4_ATTN_ATEN_GQA_EXACT &&
            spec->attn_type == GEMMA_ATTN_FULL;
        const bool source13 =
            (!spec->is_kv_shared && layer_idx == 13) ||
            (spec->is_kv_shared && spec->kv_shared_src_index == 13);
        const bool use_persistent_shared_sliding =
            !use_gqa && D == 256 && spec->attn_type == GEMMA_ATTN_SLIDING && source13;
        const bool shared_sliding_owner =
            use_persistent_shared_sliding && !spec->is_kv_shared && layer_idx == 13;
        const uint64_t q_strides[] = {
            (uint64_t)GEMMA4_NUM_ATTN_HEADS * (uint64_t)q_len * (uint64_t)D,
            (uint64_t)D,
            (uint64_t)GEMMA4_NUM_ATTN_HEADS * (uint64_t)D,
            1,
        };
        const void *bridge_k = k_data + (size_t)k_start_offset * (size_t)D;
        const void *bridge_v = v_data + (size_t)k_start_offset * (size_t)D;
        const int full_repeat_slot = !use_gqa && D == 512
            ? full_repeat_slot_for_layer(layer_idx) : -1;
        const bool use_persistent_full_repeat = full_repeat_slot >= 0;
        const bool full_repeat_owner =
            use_persistent_full_repeat && !spec->is_kv_shared;
        if (!use_gqa && !use_persistent_full_repeat &&
            (spec->attn_type != GEMMA_ATTN_SLIDING ||
             gemma4_sliding_staging::validate_head_dim(
                 (uint32_t)D, ctx->sliding_staging) !=
                 gemma4_sliding_staging::Status::kOk ||
             gemma4_sliding_staging::validate_visible_window(
                 visible_k_len, ctx->config.max_seq_len, ctx->sliding_staging) !=
                 gemma4_sliding_staging::Status::kOk)) return -8;
        const uint64_t kv_heads = use_gqa
            ? GEMMA4_NUM_KV_HEADS : GEMMA4_NUM_ATTN_HEADS;
        const uint64_t kv_dims[] = {
            1, kv_heads, (uint64_t)visible_k_len, (uint64_t)D
        };
        uint64_t kv_strides[4] = {};
        if (use_gqa) {
            kv_strides[0] = (uint64_t)ctx->config.max_seq_len * (uint64_t)D;
            kv_strides[1] = (uint64_t)ctx->config.max_seq_len * (uint64_t)D;
        } else if (use_persistent_shared_sliding || use_persistent_full_repeat) {
            kv_strides[0] = (uint64_t)GEMMA4_NUM_ATTN_HEADS *
                (uint64_t)visible_k_len * (uint64_t)D;
            kv_strides[1] = (uint64_t)visible_k_len * (uint64_t)D;
        } else {
            kv_strides[0] = (uint64_t)GEMMA4_NUM_ATTN_HEADS *
                (uint64_t)visible_k_len * (uint64_t)D;
            kv_strides[1] = (uint64_t)visible_k_len * (uint64_t)D;
        }
        kv_strides[2] = (uint64_t)D;
        kv_strides[3] = 1;
        if (!use_gqa) {
            if (D > 512 || visible_k_len > (int)ctx->config.max_seq_len) return -8;
            if (use_persistent_shared_sliding) {
                if (!ctx->d_sdpa_shared_sliding_k_repeat ||
                    !ctx->d_sdpa_shared_sliding_v_repeat || visible_k_len <= 0) return -8;
                if (shared_sliding_owner &&
                    stage_shared_sliding_source13(ctx, q_len, base_pos, stream) != 0) return -8;
                bridge_k = ctx->d_sdpa_shared_sliding_k_repeat;
                bridge_v = ctx->d_sdpa_shared_sliding_v_repeat;
            } else if (use_persistent_full_repeat) {
                if (!ctx->d_sdpa_full_k_repeat[full_repeat_slot] ||
                    !ctx->d_sdpa_full_v_repeat[full_repeat_slot] ||
                    k_start_offset != 0 || total_k_len <= 0) return -8;
                if (full_repeat_owner) {
                    const int repeat_blocks = ((visible_k_len * D) + 255) / 256;
                    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
                        static_cast<const __nv_bfloat16 *>(bridge_k),
                        ctx->d_sdpa_full_k_repeat[full_repeat_slot],
                        visible_k_len,
                        D
                    );
                    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
                        static_cast<const __nv_bfloat16 *>(bridge_v),
                        ctx->d_sdpa_full_v_repeat[full_repeat_slot],
                        visible_k_len,
                        D
                    );
                    if (cudaPeekAtLastError() != cudaSuccess) return -8;
                }
                bridge_k = ctx->d_sdpa_full_k_repeat[full_repeat_slot];
                bridge_v = ctx->d_sdpa_full_v_repeat[full_repeat_slot];
            } else {
                if (!ctx->d_sdpa_k_repeat || !ctx->d_sdpa_v_repeat) return -8;
                const int repeat_blocks = ((visible_k_len * D) + 255) / 256;
                kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(bridge_k),
                    ctx->d_sdpa_k_repeat,
                    visible_k_len,
                    D
                );
                kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16 *>(bridge_v),
                    ctx->d_sdpa_v_repeat,
                    visible_k_len,
                    D
                );
                if (cudaPeekAtLastError() != cudaSuccess) return -8;
                bridge_k = ctx->d_sdpa_k_repeat;
                bridge_v = ctx->d_sdpa_v_repeat;
            }
        }
        const uint64_t mask_dims[] = {1, 1, (uint64_t)q_len, (uint64_t)visible_k_len};
        const uint64_t mask_strides[] = {
            (uint64_t)q_len * (uint64_t)visible_k_len,
            (uint64_t)q_len * (uint64_t)visible_k_len,
            (uint64_t)visible_k_len,
            1,
        };
        gemma4_tensor_meta q_meta = make_bridge_meta(
            ctx->d_q_buf, GEMMA4_META_BF16, 4, q_dims, q_strides, 0);
        gemma4_tensor_meta k_meta = make_bridge_meta(
            bridge_k, GEMMA4_META_BF16, 4, kv_dims, kv_strides,
            use_gqa ? 0 : 1);
        gemma4_tensor_meta v_meta = make_bridge_meta(
            bridge_v, GEMMA4_META_BF16, 4, kv_dims, kv_strides,
            use_gqa ? 0 : 1);
        gemma4_tensor_meta mask_meta = make_bridge_meta(
            ctx->d_sdpa_mask, GEMMA4_META_BOOL, 4, mask_dims, mask_strides, 1);
        gemma4_tensor_meta out_meta = make_bridge_meta(
            ctx->d_attn_out, GEMMA4_META_BF16, 4, q_dims, q_strides, 0);
        const int bridge_status = use_gqa
            ? gemma4_listener_debug_attention_gqa(
                ctx->d_q_buf, &q_meta,
                bridge_k, &k_meta,
                bridge_v, &v_meta,
                ctx->d_sdpa_mask, &mask_meta,
                ctx->d_attn_out, &out_meta,
                (double)sm_scale, stream)
            : gemma4_listener_debug_attention(
                ctx->d_q_buf, &q_meta,
                bridge_k, &k_meta,
                bridge_v, &v_meta,
                ctx->d_sdpa_mask, &mask_meta,
                ctx->d_attn_out, &out_meta,
                (double)sm_scale, stream);
        if (bridge_status != 0) return -8;
        }
    } else
    if (valid_q_len == 4) {
        if (D == 256) {
            kernel_stream_sdpa_q4_bf16<256><<<GEMMA4_NUM_ATTN_HEADS, 256, smem_bytes, stream>>>(
                ctx->d_q_buf,
                k_data,
                v_data,
                ctx->d_attn_out,
                total_k_len,
                (int)ctx->config.max_seq_len,
                k_start_offset,
                base_pos,
                device_position,
                spec->sliding_window,
                ctx->d_prefix_attention,
                ctx->prefix_len,
                sm_scale
            );
        } else {
            kernel_stream_sdpa_q4_bf16<512><<<GEMMA4_NUM_ATTN_HEADS, 256, smem_bytes, stream>>>(
                ctx->d_q_buf,
                k_data,
                v_data,
                ctx->d_attn_out,
                total_k_len,
                (int)ctx->config.max_seq_len,
                k_start_offset,
                base_pos,
                device_position,
                spec->sliding_window,
                ctx->d_prefix_attention,
                ctx->prefix_len,
                sm_scale
            );
        }
    } else {
        dim3 prefill_grid(GEMMA4_NUM_ATTN_HEADS, valid_q_len);
        if (D == 256) {
            kernel_prefill_sdpa_bf16<256><<<prefill_grid, 256, 0, stream>>>(
                ctx->d_q_buf,
                k_data,
                v_data,
                ctx->d_attn_out,
                valid_q_len,
                total_k_len,
                k_start_offset,
                base_pos,
                spec->sliding_window,
                ctx->d_prefix_attention,
                ctx->prefix_len,
                sm_scale
            );
        } else {
            kernel_prefill_sdpa_bf16<512><<<prefill_grid, 256, 0, stream>>>(
                ctx->d_q_buf,
                k_data,
                v_data,
                ctx->d_attn_out,
                valid_q_len,
                total_k_len,
                k_start_offset,
                base_pos,
                spec->sliding_window,
                ctx->d_prefix_attention,
                ctx->prefix_len,
                sm_scale
            );
        }
    }
    if (debug_snapshot_bf16(ctx, ctx->d_debug_q_taps, 3,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512,
                             ctx->d_attn_out, q_len * q_dim, stream) != 0) return -7;

    /* Shape-stable canonical prefill uses M=target_length for the O
     * projection.  Its invalid rows are zero-padded and never selected. */
    if (q_len > valid_q_len) {
        if (cudaMemsetAsync(
            ctx->d_attn_out + (size_t)valid_q_len * q_dim,
            0,
            (size_t)(q_len - valid_q_len) * q_dim * sizeof(__nv_bfloat16),
            stream
        ) != cudaSuccess) return -5;
    }

    /* 4. Attention O projection: attn_out -> hidden_b */
    if (run_linear_cublas(
        ctx->cublas,
        lw->o_proj_w,
        ctx->d_attn_out,
        ctx->d_hidden_b,
        q_len, hidden_size, q_dim,
        stream
    ) != 0) return -4;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 2,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_b, q_len * hidden_size, stream) != 0) return -7;

    /* Post-attention layernorm on O projection */
    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_b, lw->post_attn_norm_w, ctx->d_hidden_b,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->post_attn_norm_eps, stream) != 0) return -8;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 3,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_b, q_len * hidden_size, stream) != 0) return -7;

    /* Attention Residual: hidden_a = hidden_a + hidden_b */
    kernel_residual_add_bf16<<<(q_len * hidden_size + 255) / 256, 256, 0, stream>>>(
        ctx->d_hidden_a,
        ctx->d_hidden_b,
        ctx->d_hidden_a,
        q_len * hidden_size
    );
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 4,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;

    /* 5. FFN Block */
    /* Pre-FF norm on hidden_a -> norm_buf.  The exact ATen branch is selected
     * only by explicit opt-in; the default path remains the custom kernel. */
    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_a, lw->pre_ff_norm_w, ctx->d_norm_buf,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->pre_ff_norm_eps, stream) != 0) return -8;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 5,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_norm_buf, q_len * hidden_size, stream) != 0) return -7;

    /* Gate and Up projections */
    if (run_linear_cublas(
        ctx->cublas,
        lw->gate_proj_w,
        ctx->d_norm_buf,
        ctx->d_gate_buf,
        q_len, intermediate_size, hidden_size,
        stream
    ) != 0) return -4;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_ffn_taps, 0,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE,
                             ctx->d_gate_buf, q_len * intermediate_size, stream) != 0) return -7;
    if (run_linear_cublas(
        ctx->cublas,
        lw->up_proj_w,
        ctx->d_norm_buf,
        ctx->d_up_buf,
        q_len, intermediate_size, hidden_size,
        stream
    ) != 0) return -4;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_ffn_taps, 1,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE,
                             ctx->d_up_buf, q_len * intermediate_size, stream) != 0) return -7;

    /* Canonical GeGLU has two BF16 elementwise boundaries:
     * gelu_tanh(gate), then activation * up. */
    int ffn_elems = q_len * intermediate_size;
    kernel_gelu_tanh_bf16<<<(ffn_elems + 255) / 256, 256, 0, stream>>>(
        ctx->d_gate_buf,
        ctx->d_mlp_act_buf,
        ffn_elems
    );
    if (debug_snapshot_bf16(ctx, ctx->d_debug_ffn_taps, 2,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE,
                             ctx->d_mlp_act_buf, ffn_elems, stream) != 0) return -7;
    kernel_mul_bf16<<<(ffn_elems + 255) / 256, 256, 0, stream>>>(
        ctx->d_mlp_act_buf,
        ctx->d_up_buf,
        ctx->d_mlp_act_buf,
        ffn_elems
    );
    if (debug_snapshot_bf16(ctx, ctx->d_debug_ffn_taps, 3,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE,
                             ctx->d_mlp_act_buf, ffn_elems, stream) != 0) return -7;

    /* Down projection: mlp_act_buf -> hidden_b */
    if (run_linear_cublas(
        ctx->cublas,
        lw->down_proj_w,
        ctx->d_mlp_act_buf,
        ctx->d_hidden_b,
        q_len, hidden_size, intermediate_size,
        stream
    ) != 0) return -4;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 7,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_b, q_len * hidden_size, stream) != 0) return -7;

    /* Post-FF norm */
    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_b, lw->post_ff_norm_w, ctx->d_hidden_b,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->post_ff_norm_eps, stream) != 0) return -8;
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 8,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_b, q_len * hidden_size, stream) != 0) return -7;

    /* FF Residual: hidden_a = hidden_a + hidden_b */
    kernel_residual_add_bf16<<<(q_len * hidden_size + 255) / 256, 256, 0, stream>>>(
        ctx->d_hidden_a,
        ctx->d_hidden_b,
        ctx->d_hidden_a,
        q_len * hidden_size
    );
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 9,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;

    /* 6. PLE (Per-Layer Embedding) Block */
    if (GEMMA4_PLE_DIM > 0) {
        /* ple_gate = linear(ple_gate_w, hidden_a) */
        if (run_linear_cublas(
            ctx->cublas,
            lw->ple_gate_w,
            ctx->d_hidden_a,
            ctx->d_ple_gate_buf,
            q_len, GEMMA4_PLE_DIM, hidden_size,
            stream
        ) != 0) return -4;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_ple_taps, 0,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_PLE_DIM,
                                 ctx->d_ple_gate_buf, q_len * GEMMA4_PLE_DIM, stream) != 0) return -7;

        /* ple_act = gelu_tanh(ple_gate) * projected_ple[:, layer_idx, :]
         * with the same two BF16 elementwise boundaries as the MLP. */
        int ple_elems = valid_q_len * GEMMA4_PLE_DIM;
        kernel_gelu_tanh_bf16<<<(ple_elems + 255) / 256, 256, 0, stream>>>(
            ctx->d_ple_gate_buf,
            ctx->d_ple_act_buf,
            ple_elems
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_ple_taps, 1,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_PLE_DIM,
                                 ctx->d_ple_act_buf, ple_elems, stream) != 0) return -7;
        kernel_ple_layer_mult_bf16<<<(ple_elems + 255) / 256, 256, 0, stream>>>(
            ctx->d_ple_act_buf,
            ctx->d_projected_ple,
            ctx->d_ple_act_buf,
            valid_q_len,
            layer_idx
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_ple_taps, 2,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_PLE_DIM,
                                 ctx->d_ple_act_buf, ple_elems, stream) != 0) return -7;

        /* Match the shape-stable projection M-shape without exposing stale
         * invalid rows to the GEMM. */
        if (q_len > valid_q_len) {
            if (cudaMemsetAsync(
                ctx->d_ple_act_buf + (size_t)valid_q_len * GEMMA4_PLE_DIM,
                0,
                (size_t)(q_len - valid_q_len) * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16),
                stream
            ) != cudaSuccess) return -5;
        }

        /* ple_proj = linear(ple_proj_w, ple_act) */
        if (run_linear_cublas(
            ctx->cublas,
            lw->ple_proj_w,
            ctx->d_ple_act_buf,
            ctx->d_ple_proj_out,
            q_len, hidden_size, GEMMA4_PLE_DIM,
            stream
        ) != 0) return -4;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 10,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                                 ctx->d_ple_proj_out, q_len * hidden_size, stream) != 0) return -7;

        /* ple_norm */
        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_ple_proj_out, lw->ple_norm_w, ctx->d_ple_proj_out,
                (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
                (double)spec->post_ple_norm_eps, stream) != 0) return -8;
        if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 11,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                                 ctx->d_ple_proj_out, q_len * hidden_size, stream) != 0) return -7;

        /* Canonical order is two BF16 elementwise operations: residual add,
         * then multiplication by the BF16 layer scalar. */
        kernel_residual_add_bf16<<<(q_len * hidden_size + 255) / 256, 256, 0, stream>>>(
            ctx->d_hidden_a,
            ctx->d_ple_proj_out,
            ctx->d_hidden_a,
            q_len * hidden_size
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 12,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                                 ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;
        kernel_scale_bf16<<<(q_len * hidden_size + 255) / 256, 256, 0, stream>>>(
            ctx->d_hidden_a,
            ctx->d_hidden_a,
            q_len * hidden_size,
            spec->layer_scalar
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 13,
                                 GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                                 ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;
    }
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 14,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;

    if (ctx->debug_layer_trace_enabled && ctx->d_debug_layer_outputs) {
        const size_t hidden_bytes = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);
        __nv_bfloat16 *destination = ctx->d_debug_layer_outputs + (size_t)layer_idx * GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE;
        if (cudaMemcpyAsync(destination, ctx->d_hidden_a, hidden_bytes,
                            cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return -7;
    }
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    return 0;
}

static int normalize_backend_env(
    const char *raw,
    char *normalized,
    size_t capacity
) {
    if (!normalized || capacity == 0) return -1;
    normalized[0] = '\0';
    if (!raw) return 0;
    size_t begin = 0;
    size_t end = strlen(raw);
    while (begin < end && (raw[begin] == ' ' || raw[begin] == '\t' ||
                           raw[begin] == '\r' || raw[begin] == '\n')) ++begin;
    while (end > begin && (raw[end - 1] == ' ' || raw[end - 1] == '\t' ||
                            raw[end - 1] == '\r' || raw[end - 1] == '\n')) --end;
    if (end - begin >= capacity) return -1;
    for (size_t index = begin; index < end; ++index) {
        char value = raw[index];
        if (value >= 'a' && value <= 'z') value = (char)(value - 'a' + 'A');
        normalized[index - begin] = value;
    }
    normalized[end - begin] = '\0';
    return 0;
}

/* API: Create Context */
#define GEMMA4_CREATE_ALLOC(ptr, bytes) \
    do { \
        const size_t requested_bytes = (size_t)(bytes); \
        const cudaError_t allocation_status = \
            cudaMalloc((void **)&(ptr), requested_bytes); \
        if (allocation_status != cudaSuccess) { \
            gemma4_log_create_failure(#ptr, requested_bytes, "cudaMalloc", \
                                      true, allocation_status); \
            gemma4_listener_destroy(ctx); \
            return -4; \
        } \
    } while (0)

static int gemma4_listener_create_impl(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx,
    int32_t requested_attention_backend,
    int32_t requested_rmsnorm_backend
) {
    if (!config || !weights || !out_ctx) return -1;
    const bool explicit_attention_backend = requested_attention_backend >= 0;
    const char *attention_backend_env = getenv("GEMMA4_ATTENTION_BACKEND");
    const char *aten_bridge_env = getenv("GEMMA4_ATEN_SDPA_BRIDGE");
    char normalized_attention_backend[64] = {0};
    if (!explicit_attention_backend && normalize_backend_env(
            attention_backend_env,
            normalized_attention_backend,
            sizeof(normalized_attention_backend)) != 0) {
        fprintf(stderr, "[gemma4_listener] invalid GEMMA4_ATTENTION_BACKEND\n");
        return -1;
    }
    const char *attention_backend_name = normalized_attention_backend[0]
        ? normalized_attention_backend : nullptr;
    uint32_t selected_attention_backend = GEMMA4_ATTN_CUSTOM;
    uint64_t splitk_layer_mask = 0;
    if (explicit_attention_backend) {
        if (requested_attention_backend < GEMMA4_ATTN_CUSTOM ||
            requested_attention_backend > GEMMA4_ATTN_NATIVE_SPLITK_GQA) {
            fprintf(stderr, "[gemma4_listener] invalid explicit attention backend: %d\\n",
                    requested_attention_backend);
            return -1;
        }
        selected_attention_backend = (uint32_t)requested_attention_backend;
        if (selected_attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA) {
            splitk_layer_mask = GEMMA4_SPLITK_STATE_SAFE_HYBRID_MASK;
        }
    } else if (attention_backend_name) {
        if (strcmp(attention_backend_name, "CUSTOM") == 0 ||
            strcmp(attention_backend_name, "0") == 0) {
            selected_attention_backend = GEMMA4_ATTN_CUSTOM;
        } else if (strcmp(attention_backend_name, "ATEN_GQA_EXACT") == 0 ||
            strcmp(attention_backend_name, "2") == 0) {
            selected_attention_backend = GEMMA4_ATTN_ATEN_GQA_EXACT;
        } else if (strcmp(attention_backend_name, "NATIVE_SPLITK_GQA") == 0 ||
                   strcmp(attention_backend_name, "ATEN_SPLITK_GQA") == 0 ||
                   strcmp(attention_backend_name, "3") == 0) {
            selected_attention_backend = GEMMA4_ATTN_NATIVE_SPLITK_GQA;
        } else if (strcmp(attention_backend_name, "ATEN_REPEAT_EXACT") == 0 ||
                   strcmp(attention_backend_name, "ATEN_EXACT") == 0 ||
                   strcmp(attention_backend_name, "1") == 0) {
            selected_attention_backend = GEMMA4_ATTN_ATEN_REPEAT_EXACT;
        } else {
            fprintf(stderr, "[gemma4_listener] unknown attention backend: %s\\n",
                    attention_backend_env);
            return -1;
        }
    } else if (aten_bridge_env && strcmp(aten_bridge_env, "0") != 0) {
        /* Backward-compatible flag: backend 1 is the permanent oracle. */
        selected_attention_backend = GEMMA4_ATTN_ATEN_REPEAT_EXACT;
    }
    if (!explicit_attention_backend && selected_attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA &&
        parse_splitk_layer_mask(getenv("GEMMA4_SPLITK_LAYER_MASK"), &splitk_layer_mask) != 0) {
        fprintf(stderr, "[gemma4_listener] invalid GEMMA4_SPLITK_LAYER_MASK\n");
        return -1;
    }
    if (selected_attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA &&
        splitk_layer_mask != GEMMA4_SPLITK_STATE_SAFE_HYBRID_MASK) {
        fprintf(stderr,
                "[gemma4_listener] CUTLASS_ORDERED_TILES_V1 requires mask 19,24,29,34\n");
        return -1;
    }
    if (config->enable_cuda_graph && selected_attention_backend != GEMMA4_ATTN_CUSTOM) {
        fprintf(stderr,
                "[gemma4_listener] CUDA Graph Q4 requires attention backend CUSTOM\\n");
        return -1;
    }
    const bool use_aten_sdpa = selected_attention_backend != GEMMA4_ATTN_CUSTOM;
    const uint32_t max_allowed_seq_len = use_aten_sdpa
        ? GEMMA4_ATEN_SDPA_MAX_SEQ_LEN
        : GEMMA4_CUSTOM_SDPA_MAX_SEQ_LEN;
    if (config->stream_q_len != GEMMA4_STREAM_Q_LEN ||
        config->max_seq_len < GEMMA4_CANONICAL_PREFIX_LEN ||
        config->max_seq_len > max_allowed_seq_len ||
        !config->frame_pad_ple_ptr || !config->frame_pad_ple_meta ||
        !config->weight_meta || config->weight_meta_count != GEMMA4_WEIGHT_META_COUNT ||
        !weights->final_norm_w || !weights->per_layer_model_proj_w ||
        !weights->per_layer_proj_norm_w ||
        (config->output_tensor_ptr && !config->output_meta) ||
        (!config->output_tensor_ptr && config->output_meta)) {
        return -1;
    }
    int current_device = -1;
    bool device_ok = cudaGetDevice(&current_device) == cudaSuccess;
    const uint64_t output_dims[] = {1, config->stream_q_len, GEMMA4_HIDDEN_SIZE, 0};
    const uint64_t pad_dims[] = {1, config->stream_q_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM};
    bool output_meta_ok = !config->output_tensor_ptr ||
        (device_ok && validate_tensor_meta(
            config->output_meta, config->output_tensor_ptr, current_device,
            GEMMA4_META_BF16, 3, output_dims));
    bool pad_meta_ok = device_ok && validate_tensor_meta(
        config->frame_pad_ple_meta, config->frame_pad_ple_ptr, current_device,
        GEMMA4_META_BF16, 4, pad_dims);
    bool weight_meta_ok = device_ok;
    if (weight_meta_ok) {
        const gemma4_tensor_meta *meta = config->weight_meta;
        const uint64_t global_dims[GEMMA4_GLOBAL_WEIGHT_META_COUNT][4] = {
            {GEMMA4_HIDDEN_SIZE, 0, 0, 0},
            {GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM, GEMMA4_HIDDEN_SIZE, 0, 0},
            {GEMMA4_PLE_DIM, 0, 0, 0},
        };
        const void *global_ptrs[GEMMA4_GLOBAL_WEIGHT_META_COUNT] = {
            weights->final_norm_w, weights->per_layer_model_proj_w, weights->per_layer_proj_norm_w,
        };
        for (int i = 0; i < GEMMA4_GLOBAL_WEIGHT_META_COUNT; ++i) {
            if (!validate_tensor_meta(&meta[i], global_ptrs[i], current_device,
                                       GEMMA4_META_BF16, i == 1 ? 2 : 1, global_dims[i])) {
                fprintf(stderr, "[gemma4_listener] global metadata invalid at %d\n", i);
                weight_meta_ok = false;
                break;
            }
        }
        for (int layer_idx = 0; weight_meta_ok && layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
            const gemma4_layer_weights *lw = &weights->layers[layer_idx];
            const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
            const uint64_t d = spec->head_dim;
            const uint64_t intermediate = spec->intermediate_size;
            const uint64_t layer_dims[GEMMA4_LAYER_WEIGHT_META_COUNT][4] = {
                {GEMMA4_HIDDEN_SIZE, 0, 0, 0}, {GEMMA4_HIDDEN_SIZE, 0, 0, 0},
                {GEMMA4_HIDDEN_SIZE, 0, 0, 0}, {GEMMA4_HIDDEN_SIZE, 0, 0, 0},
                {GEMMA4_NUM_ATTN_HEADS * d, GEMMA4_HIDDEN_SIZE, 0, 0},
                {GEMMA4_NUM_KV_HEADS * d, GEMMA4_HIDDEN_SIZE, 0, 0},
                {GEMMA4_NUM_KV_HEADS * d, GEMMA4_HIDDEN_SIZE, 0, 0},
                {GEMMA4_HIDDEN_SIZE, GEMMA4_NUM_ATTN_HEADS * d, 0, 0},
                {d, 0, 0, 0}, {d, 0, 0, 0}, {d, 0, 0, 0},
                {intermediate, GEMMA4_HIDDEN_SIZE, 0, 0},
                {intermediate, GEMMA4_HIDDEN_SIZE, 0, 0},
                {GEMMA4_HIDDEN_SIZE, intermediate, 0, 0},
                {GEMMA4_PLE_DIM, GEMMA4_HIDDEN_SIZE, 0, 0},
                {GEMMA4_HIDDEN_SIZE, GEMMA4_PLE_DIM, 0, 0},
                {GEMMA4_HIDDEN_SIZE, 0, 0, 0},
            };
            const void *layer_ptrs[GEMMA4_LAYER_WEIGHT_META_COUNT] = {
                lw->input_norm_w, lw->post_attn_norm_w, lw->pre_ff_norm_w, lw->post_ff_norm_w,
                lw->q_proj_w, lw->k_proj_w, lw->v_proj_w, lw->o_proj_w,
                lw->q_norm_w, lw->k_norm_w, lw->v_norm_w,
                lw->gate_proj_w, lw->up_proj_w, lw->down_proj_w,
                lw->ple_gate_w, lw->ple_proj_w, lw->ple_norm_w,
            };
            const size_t base = GEMMA4_GLOBAL_WEIGHT_META_COUNT +
                (size_t)layer_idx * GEMMA4_LAYER_WEIGHT_META_COUNT;
            for (int field = 0; field < GEMMA4_LAYER_WEIGHT_META_COUNT; ++field) {
                const bool optional = (field == 5 || field == 6 || field == 9 || field == 10) && spec->is_kv_shared;
                const bool optional_v_norm = field == 10 && !lw->v_norm_w;
                const uint32_t field_dtype = (optional || optional_v_norm) ? 0 : GEMMA4_META_BF16;
                const uint32_t field_ndim = (optional || optional_v_norm) ? 0 : (field == 0 || field == 1 || field == 2 || field == 3 || field == 8 || field == 9 || field == 10 || field == 16 ? 1 : 2);
                if (!validate_tensor_meta(&meta[base + field], layer_ptrs[field], current_device,
                                           field_dtype, field_ndim, layer_dims[field])) {
                    fprintf(stderr, "[gemma4_listener] layer metadata invalid at layer=%d field=%d ptr=%p meta_ptr=%p ndim=%u dtype=%u\n",
                            layer_idx, field, layer_ptrs[field], meta[base + field].ptr,
                            meta[base + field].ndim, meta[base + field].dtype);
                    fprintf(stderr, "[gemma4_listener] dims=(%llu,%llu,%llu,%llu) strides=(%llu,%llu,%llu,%llu)\n",
                            (unsigned long long)meta[base + field].dims[0],
                            (unsigned long long)meta[base + field].dims[1],
                            (unsigned long long)meta[base + field].dims[2],
                            (unsigned long long)meta[base + field].dims[3],
                            (unsigned long long)meta[base + field].strides[0],
                            (unsigned long long)meta[base + field].strides[1],
                            (unsigned long long)meta[base + field].strides[2],
                            (unsigned long long)meta[base + field].strides[3]);
                    weight_meta_ok = false;
                    break;
                }
            }
        }
    }
    if (!pad_meta_ok) fprintf(stderr, "[gemma4_listener] frame_pad metadata invalid\n");
    if (!output_meta_ok) fprintf(stderr, "[gemma4_listener] output metadata invalid\n");
    if (!weight_meta_ok) fprintf(stderr, "[gemma4_listener] weight metadata invalid\n");
    bool pad_ok = device_ok && is_current_device_pointer(config->frame_pad_ple_ptr, current_device);
    bool output_ok = !config->output_tensor_ptr ||
        (device_ok && is_current_device_pointer(config->output_tensor_ptr, current_device));
    bool final_ok = device_ok && is_current_device_pointer(weights->final_norm_w, current_device);
    bool global_proj_ok = device_ok && is_current_device_pointer(weights->per_layer_model_proj_w, current_device);
    bool global_norm_ok = device_ok && is_current_device_pointer(weights->per_layer_proj_norm_w, current_device);
    if (!device_ok || !pad_ok || !pad_meta_ok || !output_ok || !output_meta_ok ||
        !weight_meta_ok || !final_ok || !global_proj_ok || !global_norm_ok) {
        fprintf(stderr, "[gemma4_listener] device pointer validation failed: device=%d pad=%d output=%d final=%d proj=%d norm=%d\n",
                current_device, pad_ok, output_ok, final_ok, global_proj_ok, global_norm_ok);
        return -1;
    }
    for (int i = 0; i < GEMMA4_NUM_LAYERS; ++i) {
        const gemma4_layer_weights *lw = &weights->layers[i];
        if (!lw->input_norm_w || !lw->post_attn_norm_w || !lw->pre_ff_norm_w ||
            !lw->post_ff_norm_w || !lw->q_proj_w || !lw->q_norm_w ||
            !lw->o_proj_w || !lw->gate_proj_w || !lw->up_proj_w ||
            !lw->down_proj_w || !lw->ple_gate_w || !lw->ple_proj_w ||
            !lw->ple_norm_w) {
            return -1;
        }
        if (!gemma4_layers[i].is_kv_shared &&
            (!lw->k_proj_w || !lw->v_proj_w || !lw->k_norm_w)) {
            return -1;
        }
        if (!validate_layer_device_pointers(lw, &gemma4_layers[i], current_device)) {
            fprintf(stderr, "[gemma4_listener] device pointer validation failed in layer %d\n", i);
            return -1;
        }
    }

    gemma4_listener_ctx *ctx = (gemma4_listener_ctx *)calloc(1, sizeof(gemma4_listener_ctx));
    if (!ctx) return -1;

    ctx->config = *config;
    ctx->weights = *weights;
    if (!gemma4_sliding_staging::derive_contract(
            gemma4_layers, GEMMA4_NUM_LAYERS, config->max_seq_len,
            GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN,
            &ctx->sliding_staging)) {
        fprintf(stderr,
                "[gemma4_listener] invalid generated sliding-staging contract\n");
        gemma4_listener_destroy(ctx);
        return -1;
    }
    ctx->current_position = 0;
    ctx->graph_captured = false;
    ctx->graph_capture_active = false;
    ctx->state_generation = 1;
    ctx->graph_generation = 0;
    ctx->prefix_len = 0;
    ctx->primed = false;
    ctx->poisoned = false;
    ctx->last_stream = 0;
    ctx->debug_capture_enabled = false;
    ctx->attention_backend = selected_attention_backend;
    ctx->splitk_layer_mask = splitk_layer_mask;
    const char *rms_backend_env = getenv("GEMMA4_RMSNORM_BACKEND");
    if (!rms_backend_env) rms_backend_env = getenv("GEMMA4_ATEN_RMSNORM_BACKEND");
    const char *legacy_rms_env = getenv("GEMMA4_ATEN_RMSNORM_BRIDGE");
    ctx->rmsnorm_backend = GEMMA4_RMS_CUSTOM_FAST;
    if (requested_rmsnorm_backend >= 0) {
        if (requested_rmsnorm_backend > GEMMA4_RMS_NATIVE_EXACT) {
            fprintf(stderr, "[gemma4_listener] invalid explicit RMSNorm backend: %d\\n",
                    requested_rmsnorm_backend);
            gemma4_listener_destroy(ctx);
            return -1;
        }
        ctx->rmsnorm_backend = (uint32_t)requested_rmsnorm_backend;
    } else if (rms_backend_env &&
               (strcmp(rms_backend_env, "NATIVE_EXACT") == 0 || strcmp(rms_backend_env, "2") == 0)) {
        ctx->rmsnorm_backend = GEMMA4_RMS_NATIVE_EXACT;
    } else if ((rms_backend_env &&
                (strcmp(rms_backend_env, "ATEN_EXACT") == 0 || strcmp(rms_backend_env, "1") == 0)) ||
               (rms_backend_env == nullptr && legacy_rms_env && strcmp(legacy_rms_env, "0") != 0)) {
        ctx->rmsnorm_backend = GEMMA4_RMS_ATEN_EXACT;
    } else if (rms_backend_env &&
               strcmp(rms_backend_env, "CUSTOM_FAST") != 0 && strcmp(rms_backend_env, "0") != 0) {
        fprintf(stderr, "[gemma4_listener] unknown RMSNorm backend: %s\\n", rms_backend_env);
        gemma4_listener_destroy(ctx);
        return -1;
    }
    if (config->enable_cuda_graph && ctx->rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT) {
        fprintf(stderr,
                "[gemma4_listener] CUDA Graph Q4 requires RMSNorm CUSTOM_FAST or NATIVE_EXACT\\n");
        gemma4_listener_destroy(ctx);
        return -1;
    }

    const bool exact_rms = ctx->rmsnorm_backend != GEMMA4_RMS_CUSTOM_FAST;
    if (exact_rms) {
        const uint32_t max_rows =
            GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN > 4u * GEMMA4_NUM_LAYERS
                ? GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN
                : 4u * GEMMA4_NUM_LAYERS;
        const uint32_t max_dim = GEMMA4_HIDDEN_SIZE;
        const size_t max_elements = (size_t)max_rows * max_dim;
        const size_t elements_bytes = max_elements * sizeof(float);
        const size_t rows_bytes = (size_t)max_rows * sizeof(float);
        const size_t dim_bytes = (size_t)max_dim * sizeof(float);
        if (ctx->rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT) {
            ctx->rms_workspace.max_rows = max_rows;
            ctx->rms_workspace.max_dim = max_dim;
            ctx->rms_workspace.max_elements = max_elements;
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.input_fp32, elements_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.squared_fp32, elements_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.mean_fp32, rows_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.mean_plus_eps_fp32, rows_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.scale_fp32, rows_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.normalized_fp32, elements_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.weighted_fp32, elements_bytes);
            GEMMA4_CREATE_ALLOC(ctx->rms_workspace.weight_fp32, dim_bytes);
        } else {
            ctx->native_rms_workspace.max_rows = max_rows;
            ctx->native_rms_workspace.max_dim = max_dim;
            ctx->native_rms_workspace.max_elements = max_elements;
            GEMMA4_CREATE_ALLOC(ctx->native_rms_workspace.squared_fp32, elements_bytes);
            GEMMA4_CREATE_ALLOC(ctx->native_rms_workspace.mean_fp32, rows_bytes);
            GEMMA4_CREATE_ALLOC(ctx->native_rms_workspace.scale_fp32, rows_bytes);
        }

        auto register_rms_weight_mirror = [&](const void *bf16_ptr, uint32_t dim) -> bool {
            if (!bf16_ptr) return true;
            for (uint32_t i = 0; i < ctx->rms_weight_mirror_count; ++i) {
                if (ctx->rms_weight_mirrors[i].bf16_ptr == bf16_ptr) {
                    if (ctx->rms_weight_mirrors[i].dim == dim) return true;
                    gemma4_log_create_failure(
                        "rms_weight_mirror.duplicate_dimension",
                        (size_t)dim * sizeof(float), "dimension_mismatch",
                        false, cudaSuccess);
                    return false;
                }
            }
            const size_t requested_bytes = (size_t)dim * sizeof(float);
            if (ctx->rms_weight_mirror_count >= GEMMA4_RMS_WEIGHT_MIRROR_CAPACITY) {
                gemma4_log_create_failure(
                    "rms_weight_mirror.capacity", requested_bytes,
                    "mirror_capacity_exceeded", false, cudaSuccess);
                return false;
            }
            gemma4_rms_weight_mirror *mirror =
                &ctx->rms_weight_mirrors[ctx->rms_weight_mirror_count++];
            mirror->bf16_ptr = bf16_ptr;
            mirror->dim = dim;
            mirror->ready = false;
            const cudaError_t allocation_status =
                cudaMalloc((void **)&mirror->fp32_ptr, requested_bytes);
            if (allocation_status != cudaSuccess) {
                gemma4_log_create_failure(
                    "rms_weight_mirror.fp32_ptr", requested_bytes,
                    "cudaMalloc", true, allocation_status);
                return false;
            }
            ctx->rms_derived_weight_bytes += requested_bytes;
            return true;
        };
        if (!register_rms_weight_mirror(ctx->weights.final_norm_w, GEMMA4_HIDDEN_SIZE) ||
            !register_rms_weight_mirror(ctx->weights.per_layer_proj_norm_w, GEMMA4_PLE_DIM)) {
            gemma4_listener_destroy(ctx);
            return -4;
        }
        for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
            const gemma4_layer_weights *lw = &ctx->weights.layers[layer_idx];
            const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
            const uint32_t d = spec->head_dim;
            const void *norm_weights[] = {
                lw->input_norm_w, lw->post_attn_norm_w, lw->pre_ff_norm_w,
                lw->post_ff_norm_w, lw->q_norm_w, lw->k_norm_w,
                lw->v_norm_w, lw->ple_norm_w,
            };
            const uint32_t norm_dims[] = {
                GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE, GEMMA4_HIDDEN_SIZE,
                GEMMA4_HIDDEN_SIZE, d, d, d, GEMMA4_HIDDEN_SIZE,
            };
            for (int index = 0; index < 8; ++index) {
                if (!register_rms_weight_mirror(norm_weights[index], norm_dims[index])) {
                    gemma4_listener_destroy(ctx);
                    return -4;
                }
            }
        }
    }

    /* The handle is persistent; projection calls only switch its stream. */
    cublasStatus_t cb_stat = cublasCreate(&ctx->cublas);
    if (cb_stat != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return -2;
    }

    /* Initialize Cache */
    int cache_err = gemma4_cache_create(
        &ctx->cache, config->max_seq_len, config->enable_cuda_graph);
    if (cache_err != 0) {
        gemma4_listener_destroy(ctx);
        return -3;
    }

    /* The custom stream SDPA kernel keeps four score rows in dynamic shared
     * memory and is deliberately capped at the native custom workspace limit. The exact ATen bridge
     * owns the longer-sequence path and must never configure the custom
     * scalar kernel for its 16384-token certification capacity. */
    if (ctx->attention_backend == GEMMA4_ATTN_CUSTOM) {
        size_t stream_smem_bytes = 4 * (size_t)config->max_seq_len * sizeof(float);
        if (stream_smem_bytes > 48 * 1024) {
            cudaError_t attr256 = cudaFuncSetAttribute(
                (const void *)kernel_stream_sdpa_q4_bf16<256>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)stream_smem_bytes);
            cudaError_t attr512 = cudaFuncSetAttribute(
                (const void *)kernel_stream_sdpa_q4_bf16<512>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)stream_smem_bytes);
            if (attr256 != cudaSuccess || attr512 != cudaSuccess) {
                gemma4_listener_destroy(ctx);
                return -3;
            }
        }
    }

    /* Precompute and upload RoPE frequencies */
    float inv_freq_s[128];
    const float sliding_rope_base = gemma4_layers[0].rope_base;
    for (int i = 0; i < 128; ++i) {
        inv_freq_s[i] = 1.0f / powf(sliding_rope_base, (float)(2 * i) / 256.0f);
    }
    GEMMA4_CREATE_ALLOC(ctx->d_inv_freq_sliding, 128 * sizeof(float));
    const cudaError_t sliding_rope_copy = cudaMemcpy(
        ctx->d_inv_freq_sliding, inv_freq_s, sizeof(inv_freq_s),
        cudaMemcpyHostToDevice);
    if (sliding_rope_copy != cudaSuccess) {
        gemma4_log_create_failure(
            "d_inv_freq_sliding.upload", sizeof(inv_freq_s), "cudaMemcpy",
            true, sliding_rope_copy);
        gemma4_listener_destroy(ctx);
        return -4;
    }

    float inv_freq_f[256];
    /* Gemma 4 full attention uses proportional RoPE with partial_rotary_factor = 0.25 */
    /* Only the first 0.25 * 512 / 2 = 64 frequency pairs are rotated; the remaining 192 are 0.0f (NO-PE) */
    const int rope_angles_f = 64;
    const float full_rope_base = gemma4_layers[4].rope_base;
    for (int i = 0; i < rope_angles_f; ++i) {
        inv_freq_f[i] = 1.0f / powf(full_rope_base, (float)(2 * i) / 512.0f);
    }
    for (int i = rope_angles_f; i < 256; ++i) {
        inv_freq_f[i] = 0.0f;
    }
    GEMMA4_CREATE_ALLOC(ctx->d_inv_freq_full, 256 * sizeof(float));
    const cudaError_t full_rope_copy = cudaMemcpy(
        ctx->d_inv_freq_full, inv_freq_f, sizeof(inv_freq_f),
        cudaMemcpyHostToDevice);
    if (full_rope_copy != cudaSuccess) {
        gemma4_log_create_failure(
            "d_inv_freq_full.upload", sizeof(inv_freq_f), "cudaMemcpy",
            true, full_rope_copy);
        gemma4_listener_destroy(ctx);
        return -4;
    }

    /* Allocate persistent canonical-prefix state.  The prefix mask and PLE
     * are copied during prime so caller-owned tensors may be released safely
     * after the stream-ordered prime has completed. */
    GEMMA4_CREATE_ALLOC(ctx->d_prefix_attention,
                        GEMMA4_CANONICAL_PREFIX_LEN * sizeof(int64_t));
    GEMMA4_CREATE_ALLOC(ctx->d_prefix_ple,
                        (size_t)GEMMA4_CANONICAL_PREFIX_LEN * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));

    /* The shape-stable oracle fixes the prefill GEMM M-shape. */
    size_t q_max = (config->stream_q_len > GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN)
        ? config->stream_q_len : GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN;
    GEMMA4_CREATE_ALLOC(ctx->d_hidden_a, q_max * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_hidden_b, q_max * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_norm_buf, q_max * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));

    GEMMA4_CREATE_ALLOC(ctx->d_q_buf, GEMMA4_NUM_ATTN_HEADS * q_max * 512 * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_k_buf, q_max * 512 * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_v_buf, q_max * 512 * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_attn_out, GEMMA4_NUM_ATTN_HEADS * q_max * 512 * sizeof(__nv_bfloat16));
    if (ctx->attention_backend != GEMMA4_ATTN_CUSTOM) {
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_mask,
                            q_max * (size_t)config->max_seq_len * sizeof(bool));
        const size_t repeat_elements =
            (size_t)GEMMA4_NUM_ATTN_HEADS * config->max_seq_len * 512u;
        for (int slot = 0; slot < 3; ++slot) {
            GEMMA4_CREATE_ALLOC(ctx->d_sdpa_full_k_repeat[slot],
                                repeat_elements * sizeof(__nv_bfloat16));
            GEMMA4_CREATE_ALLOC(ctx->d_sdpa_full_v_repeat[slot],
                                repeat_elements * sizeof(__nv_bfloat16));
        }
        size_t sliding_repeat_bytes = 0;
        if (!gemma4_sliding_staging::repeat_buffer_bytes(
                GEMMA4_NUM_ATTN_HEADS,
                ctx->sliding_staging.capacity,
                ctx->sliding_staging.max_sliding_head_dim,
                sizeof(__nv_bfloat16), &sliding_repeat_bytes) ||
            sliding_repeat_bytes == 0) {
            fprintf(stderr,
                    "[gemma4_listener] invalid sliding repeat-staging allocation shape\n");
            gemma4_listener_destroy(ctx);
            return -1;
        }
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_shared_sliding_k_repeat,
                            sliding_repeat_bytes);
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_shared_sliding_v_repeat,
                            sliding_repeat_bytes);
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_k_repeat, sliding_repeat_bytes);
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_v_repeat, sliding_repeat_bytes);
    }
    if (ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA) {
        if (config->max_seq_len > GEMMA4_ORDERED_MAX_KEYS) {
            gemma4_log_create_failure(
                "ordered_workspace.max_seq_len", 0,
                "max_seq_len_exceeds_ordered_key_limit", false, cudaSuccess);
            gemma4_listener_destroy(ctx);
            return -4;
        }
        gemma4_ordered_create_failure ordered_failure = {};
        const int ordered_status = gemma4_ordered_workspace_create_with_diagnostics(
            &ctx->ordered_workspace, config->max_seq_len, &ordered_failure);
        if (ordered_status != 0) {
            gemma4_log_create_failure(
                ordered_failure.field ? ordered_failure.field : "ordered_workspace.create",
                ordered_failure.requested_bytes,
                ordered_failure.reason ? ordered_failure.reason : "ordered_workspace_create_failed",
                ordered_failure.cuda_error_known != 0,
                static_cast<cudaError_t>(ordered_failure.cuda_error));
            gemma4_listener_destroy(ctx);
            return -4;
        }
    }

    GEMMA4_CREATE_ALLOC(ctx->d_gate_buf, q_max * GEMMA4_MAX_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_up_buf, q_max * GEMMA4_MAX_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_mlp_act_buf, q_max * GEMMA4_MAX_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_mlp_down_buf, q_max * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));

    GEMMA4_CREATE_ALLOC(ctx->d_ple_raw_proj, q_max * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_projected_ple, q_max * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_ple_gate_buf, q_max * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_ple_act_buf, q_max * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_ple_proj_out, q_max * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));

    if (config->output_tensor_ptr) {
        ctx->d_output_tensor = (__nv_bfloat16 *)config->output_tensor_ptr;
        ctx->owns_output_tensor = false;
    } else {
        GEMMA4_CREATE_ALLOC(ctx->d_output_tensor, 4 * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));
        ctx->owns_output_tensor = true;
    }
    GEMMA4_CREATE_ALLOC(ctx->d_output_health, sizeof(int));
    if (cudaMemset(ctx->d_output_health, 0, sizeof(int)) != cudaSuccess) {
        gemma4_listener_destroy(ctx);
        return -2;
    }

    GEMMA4_CREATE_ALLOC(ctx->d_debug_hidden_taps,
                        (size_t)16 * GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_debug_q_taps,
                        (size_t)4 * GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512 * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_debug_kv_taps,
                        (size_t)5 * GEMMA4_STREAM_Q_LEN * 512 * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_debug_ffn_taps,
                        (size_t)4 * GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_debug_ple_taps,
                        (size_t)3 * GEMMA4_STREAM_Q_LEN * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16));
    GEMMA4_CREATE_ALLOC(ctx->d_debug_layer_outputs,
                        (size_t)GEMMA4_NUM_LAYERS * GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16));

    *out_ctx = ctx;
    return 0;
}

int gemma4_listener_create(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx
) {
    return gemma4_listener_create_impl(config, weights, out_ctx, -1, -1);
}

int gemma4_listener_create_with_policy(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx,
    int32_t attention_backend,
    int32_t rmsnorm_backend
) {
    return gemma4_listener_create_impl(
        config, weights, out_ctx, attention_backend, rmsnorm_backend);
}
#undef GEMMA4_CREATE_ALLOC

/* API: Import an exact ShapeStable DynamicCache snapshot at any logical
 * position at or after the canonical prefix.  This is the parity-first path:
 * the prefix arithmetic stays in PyTorch, while the native executor takes
 * ownership of a D2D copy of the 15 non-shared source caches. Shared consumer
 * layers already alias sources 13 and 14 in gemma4_cache. */
int gemma4_listener_import_cache_async(
    gemma4_listener_ctx *ctx,
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    cudaStream_t stream
) {
    if (!ctx || !imported) return -1;
    if (current_position < GEMMA4_CANONICAL_PREFIX_LEN ||
        current_position > ctx->config.max_seq_len) return -2;
    if (ctx->poisoned) return -9;
    if (ctx->primed || ctx->current_position != 0) return -3;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }

    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    if (validate_imported_cache_metadata(imported, current_position, current_device) != 0) {
        fprintf(stderr, "[gemma4_listener] imported cache metadata invalid\\n");
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    if (gemma4_cache_import_async(ctx->cache, imported, current_position, stream) != 0) {
        ctx->poisoned = true;
        return -7;
    }
    kernel_set_canonical_prefix_mask<<<1, 64, 0, stream>>>(
        ctx->d_prefix_attention, GEMMA4_CANONICAL_PREFIX_LEN
    );
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->prefix_len = GEMMA4_CANONICAL_PREFIX_LEN;
    ctx->current_position = current_position;
    if (ctx->config.enable_cuda_graph) ++ctx->state_generation;
    ctx->last_stream = stream;
    ctx->primed = true;
    return 0;
}

/* API: isolated native split-K GQA attention diagnostic. */
int gemma4_listener_debug_attention_splitk(
    gemma4_listener_ctx *ctx,
    const void *query,
    const gemma4_tensor_meta *query_meta,
    const void *key,
    const gemma4_tensor_meta *key_meta,
    const void *value,
    const gemma4_tensor_meta *value_meta,
    const int64_t *prefix_attention,
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t base_position,
    double scale,
    cudaStream_t stream
) {
    if (!ctx || !query || !query_meta || !key || !key_meta || !value || !value_meta ||
        !output || !output_meta || ctx->attention_backend != GEMMA4_ATTN_NATIVE_SPLITK_GQA ||
        !ctx->ordered_workspace.initialized || ctx->poisoned || !std::isfinite(scale)) return -1;
    if (observe_async_stream_error(ctx) != 0 || cudaGetLastError() != cudaSuccess) return -12;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) return -10;
    const uint64_t q_dims[] = {1, GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_ATTN_HEADS, 512};
    const uint64_t out_dims[] = {1, GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_ATTN_HEADS, 512};
    const uint64_t key_dims[] = {1, GEMMA4_NUM_KV_HEADS, key_meta->dims[2], 512};
    const uint64_t value_dims[] = {1, GEMMA4_NUM_KV_HEADS, value_meta->dims[2], 512};
    const uint64_t prefix_dims[] = {1, prefix_len, 0, 0};
    const bool prefix_ok = prefix_len == 0
        ? (prefix_attention == nullptr && prefix_attention_meta == nullptr)
        : prefix_attention != nullptr && prefix_attention_meta != nullptr &&
          validate_tensor_meta(prefix_attention_meta, prefix_attention, current_device,
                               GEMMA4_META_INT64, 2, prefix_dims);
    if (key_meta->dims[2] == 0 ||
        !validate_tensor_meta(query_meta, query, current_device, GEMMA4_META_BF16, 4, q_dims) ||
        !validate_tensor_meta(key_meta, key, current_device, GEMMA4_META_BF16, 4, key_dims) ||
        !validate_tensor_meta(value_meta, value, current_device, GEMMA4_META_BF16, 4, value_dims) ||
        key_meta->dims[2] != value_meta->dims[2] ||
        prefix_len > key_meta->dims[2] || !prefix_ok ||
        !validate_tensor_meta(output_meta, output, current_device, GEMMA4_META_BF16, 4, out_dims)) {
        return -10;
    }
    const int key_len = static_cast<int>(key_meta->dims[2]);
    if (key_len <= GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE ||
        key_len > GEMMA4_ORDERED_MAX_KEYS ||
        base_position > GEMMA4_ATEN_SDPA_MAX_SEQ_LEN - GEMMA4_STREAM_Q_LEN ||
        static_cast<uint64_t>(base_position) + GEMMA4_STREAM_Q_LEN >
            static_cast<uint64_t>(GEMMA4_ATEN_SDPA_MAX_SEQ_LEN)) return -2;
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    if (gemma4_ordered_attention_launch(
            static_cast<const __nv_bfloat16 *>(query),
            static_cast<const __nv_bfloat16 *>(key),
            static_cast<const __nv_bfloat16 *>(value),
            prefix_attention,
            prefix_len,
            (uint32_t)key_len,
            base_position,
            (float)scale,
            static_cast<__nv_bfloat16 *>(output),
            stream,
            &ctx->ordered_workspace) != 0) return -8;
    ctx->last_stream = stream;
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -8;
}

/* API: Execute one layer from exact oracle tensors for first-divergence
 * diagnostics. This intentionally keeps the context unprimed and does not
 * advance its public position; each call imports the canonical source cache
 * before executing the selected layer. */
int gemma4_listener_debug_execute_layer(
    gemma4_listener_ctx *ctx,
    uint32_t layer_idx,
    const void *hidden_input,
    const gemma4_tensor_meta *hidden_input_meta,
    const void *projected_ple,
    const gemma4_tensor_meta *projected_ple_meta,
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    uint32_t query_position,
    cudaStream_t stream
) {
    if (!ctx || !hidden_input || !hidden_input_meta || !projected_ple ||
        !projected_ple_meta || !imported) return -1;
    if (layer_idx >= GEMMA4_NUM_LAYERS ||
        current_position != GEMMA4_CANONICAL_PREFIX_LEN ||
        query_position != current_position || ctx->primed ||
        ctx->current_position != 0 || ctx->poisoned) return -2;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t hidden_dims[] = {1, GEMMA4_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE};
    const uint64_t ple_dims[] = {
        1, GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM
    };
    if (!validate_tensor_meta(
            hidden_input_meta, hidden_input, current_device,
            GEMMA4_META_BF16, 3, hidden_dims) ||
        !validate_tensor_meta(
            projected_ple_meta, projected_ple, current_device,
            GEMMA4_META_BF16, 4, ple_dims) ||
        validate_imported_cache_metadata(imported, current_position, current_device) != 0) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    if (gemma4_cache_import_async(ctx->cache, imported, current_position, stream) != 0) {
        ctx->poisoned = true;
        return -7;
    }
    kernel_set_canonical_prefix_mask<<<1, 64, 0, stream>>>(
        ctx->d_prefix_attention, GEMMA4_CANONICAL_PREFIX_LEN
    );
    const size_t hidden_bytes =
        (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    const size_t ple_bytes = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_LAYERS *
        GEMMA4_PLE_DIM * sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(ctx->d_hidden_a, hidden_input, hidden_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(ctx->d_projected_ple, projected_ple, ple_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ctx->poisoned = true;
        return -7;
    }
    const uint32_t saved_prefix_len = ctx->prefix_len;
    ctx->prefix_len = GEMMA4_CANONICAL_PREFIX_LEN;
    const GemmaLayerSpec *debug_spec = &gemma4_layers[layer_idx];
    const bool debug_uses_shared_sliding_source13 =
        debug_spec->attn_type == GEMMA_ATTN_SLIDING &&
        debug_spec->head_dim == 256 &&
        ((debug_spec->is_kv_shared && debug_spec->kv_shared_src_index == 13) ||
         (!debug_spec->is_kv_shared && layer_idx == 13));
    if (debug_uses_shared_sliding_source13 && debug_spec->is_kv_shared &&
        stage_shared_sliding_source13(ctx, GEMMA4_STREAM_Q_LEN, query_position, stream) != 0) {
        ctx->poisoned = true;
        ctx->prefix_len = saved_prefix_len;
        return -8;
    }
    ctx->debug_capture_enabled = true;
    int layer_status = execute_layer(
        ctx, (int)layer_idx, GEMMA4_STREAM_Q_LEN, GEMMA4_STREAM_Q_LEN,
        query_position, stream
    );
    ctx->debug_capture_enabled = false;
    ctx->prefix_len = saved_prefix_len;
    if (layer_status != 0) {
        ctx->poisoned = true;
        return -3;
    }
    if (cudaMemcpyAsync(ctx->d_output_tensor, ctx->d_hidden_a, hidden_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        guard_output_finite(ctx, ctx->d_output_tensor,
                            GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE, stream) != 0) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->last_stream = stream;
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    return 0;
}

/* API: Prime Debug (execute first num_layers layers during prefix priming) */
int gemma4_listener_prime_debug(
    gemma4_listener_ctx *ctx,
    const void *prefix_embeds,
    const gemma4_tensor_meta *prefix_embeds_meta,
    const void *prefix_ple,
    const gemma4_tensor_meta *prefix_ple_meta,
    const int64_t *prefix_attention,
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    uint32_t num_layers,
    bool apply_final_norm,
    cudaStream_t stream
) {
    if (!ctx || !prefix_embeds || !prefix_embeds_meta || !prefix_ple || !prefix_ple_meta ||
        !prefix_attention || !prefix_attention_meta || prefix_len == 0) return -1;
    if (prefix_len != GEMMA4_CANONICAL_PREFIX_LEN ||
        prefix_len > ctx->config.max_seq_len) return -2;
    if (ctx->primed || ctx->current_position != 0) return -3;
    if (ctx->poisoned) return -9;
    if (observe_async_stream_error(ctx) != 0) return -12;
    cudaError_t prior_error = cudaGetLastError();
    if (prior_error != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    const uint64_t prefix_embeds_dims[] = {1, prefix_len, GEMMA4_HIDDEN_SIZE, 0};
    const uint64_t prefix_ple_dims[] = {1, prefix_len, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM};
    const uint64_t prefix_attention_dims[] = {1, prefix_len, 0, 0};
    const bool prefix_embeds_ok = cudaGetDevice(&current_device) == cudaSuccess &&
        validate_tensor_meta(prefix_embeds_meta, prefix_embeds, current_device,
                             GEMMA4_META_BF16, 3, prefix_embeds_dims);
    const bool prefix_ple_ok = prefix_embeds_ok &&
        validate_tensor_meta(prefix_ple_meta, prefix_ple, current_device,
                             GEMMA4_META_BF16, 4, prefix_ple_dims);
    const bool prefix_attention_ok = prefix_ple_ok &&
        validate_tensor_meta(prefix_attention_meta, prefix_attention, current_device,
                             GEMMA4_META_INT64, 2, prefix_attention_dims);
    if (!prefix_embeds_ok || !prefix_ple_ok || !prefix_attention_ok) {
        fprintf(stderr, "[gemma4_listener] prefix metadata invalid: embeds=%d ple=%d attention=%d\n",
                prefix_embeds_ok, prefix_ple_ok, prefix_attention_ok);
        fprintf(stderr, "[gemma4_listener] attention ptr=%p meta_ptr=%p ndim=%u dtype=%u contiguous=%u dims=(%llu,%llu) strides=(%llu,%llu)\n",
                prefix_attention, prefix_attention_meta->ptr, prefix_attention_meta->ndim,
                prefix_attention_meta->dtype, prefix_attention_meta->contiguous,
                (unsigned long long)prefix_attention_meta->dims[0],
                (unsigned long long)prefix_attention_meta->dims[1],
                (unsigned long long)prefix_attention_meta->strides[0],
                (unsigned long long)prefix_attention_meta->strides[1]);
        ctx->poisoned = true;
        return -10;
    }
    int64_t host_prefix_attention[GEMMA4_CANONICAL_PREFIX_LEN];
    if (cudaMemcpy(
        host_prefix_attention,
        prefix_attention,
        sizeof(host_prefix_attention),
        cudaMemcpyDeviceToHost
    ) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    for (int i = 0; i < GEMMA4_CANONICAL_PREFIX_LEN; ++i) {
        if (host_prefix_attention[i] != 0 && host_prefix_attention[i] != 1) {
            ctx->poisoned = true;
            return -10;
        }
    }
    if (num_layers > GEMMA4_NUM_LAYERS) num_layers = GEMMA4_NUM_LAYERS;

    /* Copy all caller-owned prime inputs into persistent device storage. */
    size_t in_bytes = prefix_len * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    size_t ple_bytes = (size_t)prefix_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(ctx->d_hidden_a, prefix_embeds, in_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ctx->poisoned = true;
        return -5;
    }
    if (cudaMemsetAsync(
        ctx->d_hidden_a + (size_t)prefix_len * GEMMA4_HIDDEN_SIZE,
        0,
        (size_t)(GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN - prefix_len) * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16),
        stream
    ) != cudaSuccess) {
        ctx->poisoned = true;
        return -5;
    }
    if (cudaMemcpyAsync(ctx->d_prefix_ple, prefix_ple, ple_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ctx->poisoned = true;
        return -5;
    }
    if (cudaMemcpyAsync(ctx->d_prefix_attention, prefix_attention,
                        prefix_len * sizeof(int64_t), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ctx->poisoned = true;
        return -5;
    }
    ctx->prefix_len = prefix_len;
    ctx->last_stream = stream;

    /* 1. Project Per-Layer Inputs (PLE) for prefix */
    if (GEMMA4_PLE_DIM > 0 && ctx->weights.per_layer_model_proj_w) {
        int ple_total_dim = GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM; /* 35 * 256 = 8960 */
        if (run_linear_cublas(
            ctx->cublas,
            ctx->weights.per_layer_model_proj_w,
            ctx->d_hidden_a,
            ctx->d_ple_raw_proj,
            GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN, ple_total_dim, GEMMA4_HIDDEN_SIZE,
            stream
        ) != 0) {
            ctx->poisoned = true;
            return -4;
        }

        const __nv_bfloat16 *token_id = ctx->d_prefix_ple;
        kernel_ple_project_and_norm_bf16<<<prefix_len * GEMMA4_NUM_LAYERS, 256, 0, stream>>>(
            ctx->d_ple_raw_proj,
            (const __nv_bfloat16 *)ctx->weights.per_layer_proj_norm_w,
            token_id,
            ctx->d_projected_ple,
            prefix_len * GEMMA4_NUM_LAYERS,
            ctx->weights.per_layer_model_proj_scale,
            1e-6f,
            ctx->weights.per_layer_input_scale
        );
        const int prefix_ple_elements = prefix_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
        kernel_residual_add_bf16<<<(prefix_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_projected_ple,
            ctx->d_prefix_ple,
            ctx->d_projected_ple,
            prefix_ple_elements
        );
        kernel_scale_bf16<<<(prefix_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_projected_ple,
            ctx->d_projected_ple,
            prefix_ple_elements,
            ctx->weights.per_layer_input_scale
        );
    }

    /* 2. Advance requested layers.  Compute on the fixed padded M-shape,
     * but append/return only the valid canonical prefix rows. */
    for (uint32_t l = 0; l < num_layers; ++l) {
        if (execute_layer(ctx, l, GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN,
                          (int)prefix_len, 0, stream) != 0) {
            ctx->poisoned = true;
            return -3;
        }
    }

    /* 3. Output / Final Norm */
    if (apply_final_norm) {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<prefix_len, 256, 0, stream>>>(
            ctx->d_hidden_a,
            (const __nv_bfloat16 *)ctx->weights.final_norm_w,
            ctx->d_hidden_a,
            prefix_len,
            GEMMA4_FINAL_NORM_EPS
        );
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }

    /* Advance cache position */
    if (gemma4_cache_advance_position(ctx->cache, (int)prefix_len) != 0) {
        ctx->poisoned = true;
        return -4;
    }
    ctx->current_position += prefix_len;
    if (ctx->config.enable_cuda_graph) {
        if (gemma4_cache_set_device_position_async(
                ctx->cache, ctx->current_position, stream) != 0) {
            ctx->poisoned = true;
            return -4;
        }
        ++ctx->state_generation;
    }
    ctx->primed = (num_layers == GEMMA4_NUM_LAYERS);
    return 0;
}

/* API: Prime Canonical Prefix */
int gemma4_listener_prime_async(
    gemma4_listener_ctx *ctx,
    const void *prefix_embeds,       /* [1, prefix_len, hidden_size] BF16 */
    const gemma4_tensor_meta *prefix_embeds_meta,
    const void *prefix_ple,          /* [1, prefix_len, num_layers, ple_dim] BF16 */
    const gemma4_tensor_meta *prefix_ple_meta,
    const int64_t *prefix_attention, /* [1, prefix_len], CUDA int64 0/1 mask */
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    cudaStream_t stream
) {
    return gemma4_listener_prime_debug(
        ctx, prefix_embeds, prefix_embeds_meta, prefix_ple, prefix_ple_meta,
        prefix_attention, prefix_attention_meta, prefix_len, GEMMA4_NUM_LAYERS,
        true, stream
    );
}

/* API: Step Debug (execute first num_layers layers) */
static int gemma4_listener_step_preflight(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    cudaStream_t stream
) {
    if (!ctx || !frame_audio_soft || !frame_audio_soft_meta) return -1;
    if (ctx->poisoned) return -9;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    const uint64_t frame_dims[] = {1, GEMMA4_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE, 0};
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        !validate_tensor_meta(frame_audio_soft_meta, frame_audio_soft, current_device,
                              GEMMA4_META_BF16, 3, frame_dims)) {
        ctx->poisoned = true;
        return -10;
    }
    const uint32_t q_len = GEMMA4_STREAM_Q_LEN;
    if (ctx->current_position > ctx->config.max_seq_len - q_len) return -5;
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    return 0;
}

static int gemma4_listener_step_impl(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    uint32_t num_layers,
    bool apply_final_norm,
    bool capture_layer_trace,
    bool prevalidated,
    cudaStream_t stream
) {
    if (!prevalidated) {
        const int validation_status = gemma4_listener_step_preflight(
            ctx, frame_audio_soft, frame_audio_soft_meta, stream);
        if (validation_status != 0) return validation_status;
    }
    if (num_layers > GEMMA4_NUM_LAYERS) num_layers = GEMMA4_NUM_LAYERS;

    uint32_t q_len = GEMMA4_STREAM_Q_LEN;
    if (ctx->current_position > ctx->config.max_seq_len - q_len) return -5;
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    size_t in_bytes = q_len * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);

    /* Copy the dynamic caller input into the persistent hidden arena. */
    if (cudaMemcpyAsync(ctx->d_hidden_a, frame_audio_soft, in_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        ctx->poisoned = true;
        return -7;
    }

    uint32_t base_pos = ctx->current_position;
    ctx->last_stream = stream;

    /* 1. Project Per-Layer Inputs (PLE) for audio frame */
    if (GEMMA4_PLE_DIM > 0 && ctx->weights.per_layer_model_proj_w) {
        int ple_total_dim = GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM; /* 35 * 256 = 8960 */
        if (run_linear_cublas(
            ctx->cublas,
            ctx->weights.per_layer_model_proj_w,
            ctx->d_hidden_a,
            ctx->d_ple_raw_proj,
            q_len, ple_total_dim, GEMMA4_HIDDEN_SIZE,
            stream
        ) != 0) {
            ctx->poisoned = true;
            return -4;
        }

        const __nv_bfloat16 *token_id = (const __nv_bfloat16 *)ctx->config.frame_pad_ple_ptr;
        const int frame_ple_elements = q_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
        if (rmsnorm_uses_exact_path(ctx)) {
            kernel_scale_bf16_fp32_scalar<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
                ctx->d_ple_raw_proj,
                ctx->d_projected_ple,
                frame_ple_elements,
                ctx->weights.per_layer_model_proj_scale
            );
        } else {
            kernel_scale_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
                ctx->d_ple_raw_proj,
                ctx->d_projected_ple,
                frame_ple_elements,
                ctx->weights.per_layer_model_proj_scale
            );
        }
        if (run_rmsnorm_exact_or_custom(
                ctx,
                ctx->d_projected_ple,
                ctx->weights.per_layer_proj_norm_w,
                ctx->d_projected_ple,
                (uint32_t)(q_len * GEMMA4_NUM_LAYERS),
                GEMMA4_PLE_DIM,
                1e-6,
                stream) != 0) return -8;
        kernel_residual_add_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_projected_ple,
            token_id,
            ctx->d_projected_ple,
            frame_ple_elements
        );
        if (rmsnorm_uses_exact_path(ctx)) {
            kernel_scale_bf16_fp32_scalar<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
                ctx->d_projected_ple,
                ctx->d_projected_ple,
                frame_ple_elements,
                ctx->weights.per_layer_input_scale
            );
        } else {
            kernel_scale_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
                ctx->d_projected_ple,
                ctx->d_projected_ple,
                frame_ple_elements,
                ctx->weights.per_layer_input_scale
            );
        }
    }

    /* 2. Execute requested layers.  The layer trace is diagnostic-only and
     * captures one hidden output after each executed layer without adding any
     * work to the production step_async path. */
    ctx->debug_layer_trace_enabled = capture_layer_trace;
    /* Fine-grained debug taps share this flag. Keep them disabled on every
     * production step; step_debug is the only caller requesting capture. */
    ctx->debug_capture_enabled = capture_layer_trace;
    int layer_status = 0;
    for (uint32_t l = 0; l < num_layers; ++l) {
        if (execute_layer(ctx, l, (int)q_len, (int)q_len, base_pos, stream) != 0) {
            layer_status = -3;
            break;
        }
    }
    ctx->debug_layer_trace_enabled = false;
    ctx->debug_capture_enabled = false;
    if (layer_status != 0) {
        ctx->poisoned = true;
        return layer_status;
    }

    /* 3. Output */
    if (apply_final_norm) {
        if (run_rmsnorm_exact_or_custom(
                ctx,
                ctx->d_hidden_a,
                ctx->weights.final_norm_w,
                ctx->d_output_tensor,
                (uint32_t)q_len,
                GEMMA4_HIDDEN_SIZE,
                GEMMA4_FINAL_NORM_EPS,
                stream) != 0) return -8;
    } else {
        if (cudaMemcpyAsync(ctx->d_output_tensor, ctx->d_hidden_a, in_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            ctx->poisoned = true;
            return -7;
        }
    }
    if (guard_output_finite(ctx, ctx->d_output_tensor, (int)(q_len * GEMMA4_HIDDEN_SIZE), stream) != 0) {
        ctx->poisoned = true;
        return -8;
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }

    /* The captured frame owns a device-side cursor increment. The host-side
     * mirror advances only after cudaGraphLaunch succeeds. */
    if (ctx->graph_capture_active) {
        if (gemma4_cache_advance_device_position_async(ctx->cache, q_len, stream) != 0) {
            ctx->poisoned = true;
            return -4;
        }
    } else {
        if (gemma4_cache_advance_position(ctx->cache, (int)q_len) != 0) {
            ctx->poisoned = true;
            return -4;
        }
        ctx->current_position += q_len;
    }
    return 0;
}

/* API: Step Debug (execute first num_layers layers and retain layer trace). */
int gemma4_listener_step_debug(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    uint32_t num_layers,
    bool apply_final_norm,
    cudaStream_t stream
) {
    return gemma4_listener_step_impl(
        ctx, frame_audio_soft, frame_audio_soft_meta, num_layers,
        apply_final_norm, true, false, stream
    );
}

/* Diagnostic API: expose the current custom RMSNorm's intermediate FP32
 * boundaries without changing the production kernel or executor state. */
int gemma4_listener_debug_rmsnorm_trace(
    gemma4_listener_ctx *ctx,
    const void *input,
    const gemma4_tensor_meta *input_meta,
    const void *weight,
    const gemma4_tensor_meta *weight_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t dim,
    double eps,
    float *input_fp32,
    float *x_squared_fp32,
    float *row_sum_fp32,
    float *row_mean_fp32,
    float *mean_plus_eps_fp32,
    float *pow_neg_half_fp32,
    float *normalized_fp32,
    float *weighted_fp32,
    cudaStream_t stream
) {
    if (!ctx || !input || !input_meta || !weight || !weight_meta ||
        !output || !output_meta || !input_fp32 || !x_squared_fp32 ||
        !row_sum_fp32 || !row_mean_fp32 || !mean_plus_eps_fp32 ||
        !pow_neg_half_fp32 || !normalized_fp32 || !weighted_fp32 ||
        rows == 0 || rows > GEMMA4_STREAM_Q_LEN ||
        (dim != 256 && dim != 512 && dim != GEMMA4_HIDDEN_SIZE) ||
        eps != eps || eps < 0.0 || eps > 3.402823466e38) return -1;
    if (ctx->poisoned) return -9;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }

    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t input_dims[] = {1, rows, dim, 0};
    const uint64_t weight_dims[] = {dim, 0, 0, 0};
    const bool metadata_ok =
        validate_tensor_meta(input_meta, input, current_device,
                             GEMMA4_META_BF16, 3, input_dims) &&
        validate_tensor_meta(weight_meta, weight, current_device,
                             GEMMA4_META_BF16, 1, weight_dims) &&
        validate_tensor_meta(output_meta, output, current_device,
                             GEMMA4_META_BF16, 3, input_dims);
    const bool trace_ok =
        is_current_device_pointer(input_fp32, current_device) &&
        is_current_device_pointer(x_squared_fp32, current_device) &&
        is_current_device_pointer(row_sum_fp32, current_device) &&
        is_current_device_pointer(row_mean_fp32, current_device) &&
        is_current_device_pointer(mean_plus_eps_fp32, current_device) &&
        is_current_device_pointer(pow_neg_half_fp32, current_device) &&
        is_current_device_pointer(normalized_fp32, current_device) &&
        is_current_device_pointer(weighted_fp32, current_device);
    if (!metadata_ok || !trace_ok) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;

    const float eps_f = (float)eps;
    if (dim == 256) {
        kernel_rmsnorm_bf16<256><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            (__nv_bfloat16 *)output, (int)rows, eps_f);
        kernel_rmsnorm_bf16_trace<256><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            input_fp32, x_squared_fp32, row_sum_fp32, row_mean_fp32,
            mean_plus_eps_fp32, pow_neg_half_fp32, normalized_fp32,
            weighted_fp32, (int)rows, eps_f);
    } else if (dim == 512) {
        kernel_rmsnorm_bf16<512><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            (__nv_bfloat16 *)output, (int)rows, eps_f);
        kernel_rmsnorm_bf16_trace<512><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            input_fp32, x_squared_fp32, row_sum_fp32, row_mean_fp32,
            mean_plus_eps_fp32, pow_neg_half_fp32, normalized_fp32,
            weighted_fp32, (int)rows, eps_f);
    } else {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            (__nv_bfloat16 *)output, (int)rows, eps_f);
        kernel_rmsnorm_bf16_trace<GEMMA4_HIDDEN_SIZE><<<rows, 256, 0, stream>>>(
            (const __nv_bfloat16 *)input, (const __nv_bfloat16 *)weight,
            input_fp32, x_squared_fp32, row_sum_fp32, row_mean_fp32,
            mean_plus_eps_fp32, pow_neg_half_fp32, normalized_fp32,
            weighted_fp32, (int)rows, eps_f);
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->last_stream = stream;
    return 0;
}

/* Diagnostic staged native-exact RMSNorm. This bypasses the selected runtime
 * backend deliberately: it is a direct gate for the NATIVE_EXACT primitive. */
int gemma4_listener_debug_rmsnorm_native_exact(
    gemma4_listener_ctx *ctx,
    const void *input,
    const gemma4_tensor_meta *input_meta,
    const void *weight_fp32,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t dim,
    double eps,
    float *squared_fp32,
    float *row_sum_fp32,
    float *mean_fp32,
    float *mean_plus_eps_fp32,
    float *scale_fp32,
    float *normalized_fp32,
    float *weighted_fp32,
    cudaStream_t stream
) {
    if (!ctx || !input || !input_meta || !output || !output_meta ||
        !squared_fp32 || !row_sum_fp32 || !mean_fp32 || !mean_plus_eps_fp32 || !scale_fp32 ||
        !normalized_fp32 || !weighted_fp32 || rows == 0 ||
        rows > 4u * GEMMA4_NUM_LAYERS ||
        (dim != 256 && dim != 512 && dim != GEMMA4_HIDDEN_SIZE) ||
        eps != eps || eps < 0.0 || eps > 3.402823466e38 || ctx->poisoned) {
        return -1;
    }
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t input_dims[] = {1, rows, dim, 0};
    if (!validate_tensor_meta(input_meta, input, current_device,
                              GEMMA4_META_BF16, 3, input_dims) ||
        !validate_tensor_meta(output_meta, output, current_device,
                              GEMMA4_META_BF16, 3, input_dims) ||
        (weight_fp32 && !is_current_device_pointer(weight_fp32, current_device))) {
        ctx->poisoned = true;
        return -10;
    }
    const size_t elements = (size_t)rows * dim;
    const bool buffers_ok =
        is_current_device_pointer(squared_fp32, current_device) &&
        is_current_device_pointer(row_sum_fp32, current_device) &&
        is_current_device_pointer(mean_fp32, current_device) &&
        is_current_device_pointer(mean_plus_eps_fp32, current_device) &&
        is_current_device_pointer(scale_fp32, current_device) &&
        is_current_device_pointer(normalized_fp32, current_device) &&
        is_current_device_pointer(weighted_fp32, current_device) &&
        elements <= SIZE_MAX / sizeof(float);
    if (!buffers_ok || (size_t)rows > SIZE_MAX / sizeof(float)) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    const int status = gemma4_rmsnorm_native_exact_trace_async(
        input, static_cast<const float *>(weight_fp32), output, rows, dim,
        (float)eps, squared_fp32, row_sum_fp32, mean_fp32, mean_plus_eps_fp32, scale_fp32,
        normalized_fp32, weighted_fp32, stream);
    if (status != 0) ctx->poisoned = true;
    else ctx->last_stream = stream;
    return status;
}

/* Diagnostic direct FFN primitives. These entrypoints deliberately reuse the
 * production cuBLAS/kernel implementations with caller-provided oracle
 * inputs; they do not alter the listener cache or public position. */
int gemma4_listener_debug_linear(
    gemma4_listener_ctx *ctx,
    const void *input,
    const gemma4_tensor_meta *input_meta,
    const void *weight,
    const gemma4_tensor_meta *weight_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t out_dim,
    uint32_t in_dim,
    cudaStream_t stream
) {
    if (!ctx || !input || !input_meta || !weight || !weight_meta ||
        !output || !output_meta || rows == 0 || rows > GEMMA4_STREAM_Q_LEN ||
        in_dim == 0 || out_dim == 0 || in_dim > GEMMA4_MAX_INTERMEDIATE_SIZE ||
        out_dim > GEMMA4_MAX_INTERMEDIATE_SIZE || ctx->poisoned) return -1;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t input_dims[] = {1, rows, in_dim, 0};
    const uint64_t weight_dims[] = {out_dim, in_dim, 0, 0};
    const uint64_t output_dims[] = {1, rows, out_dim, 0};
    if (!validate_tensor_meta(input_meta, input, current_device,
                              GEMMA4_META_BF16, 3, input_dims) ||
        !validate_tensor_meta(weight_meta, weight, current_device,
                               GEMMA4_META_BF16, 2, weight_dims) ||
        !validate_tensor_meta(output_meta, output, current_device,
                              GEMMA4_META_BF16, 3, output_dims)) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    if (run_linear_cublas(
            ctx->cublas, weight, input, output,
            (int)rows, (int)out_dim, (int)in_dim, stream) != 0) {
        ctx->poisoned = true;
        return -4;
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->last_stream = stream;
    return 0;
}

int gemma4_listener_debug_gelu_tanh(
    gemma4_listener_ctx *ctx,
    const void *input,
    const gemma4_tensor_meta *input_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t dim,
    cudaStream_t stream
) {
    if (!ctx || !input || !input_meta || !output || !output_meta ||
        rows == 0 || rows > GEMMA4_STREAM_Q_LEN || dim == 0 ||
        dim > GEMMA4_MAX_INTERMEDIATE_SIZE || ctx->poisoned) return -1;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t dims[] = {1, rows, dim, 0};
    if (!validate_tensor_meta(input_meta, input, current_device,
                              GEMMA4_META_BF16, 3, dims) ||
        !validate_tensor_meta(output_meta, output, current_device,
                              GEMMA4_META_BF16, 3, dims)) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    const int elements = (int)((size_t)rows * dim);
    kernel_gelu_tanh_bf16<<<(elements + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16 *)input, (__nv_bfloat16 *)output, elements);
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->last_stream = stream;
    return 0;
}

int gemma4_listener_debug_mul(
    gemma4_listener_ctx *ctx,
    const void *lhs,
    const gemma4_tensor_meta *lhs_meta,
    const void *rhs,
    const gemma4_tensor_meta *rhs_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t dim,
    cudaStream_t stream
) {
    if (!ctx || !lhs || !lhs_meta || !rhs || !rhs_meta || !output ||
        !output_meta || rows == 0 || rows > GEMMA4_STREAM_Q_LEN || dim == 0 ||
        dim > GEMMA4_MAX_INTERMEDIATE_SIZE || ctx->poisoned) return -1;
    if (observe_async_stream_error(ctx) != 0) return -12;
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
        ctx->poisoned = true;
        return -10;
    }
    const uint64_t dims[] = {1, rows, dim, 0};
    if (!validate_tensor_meta(lhs_meta, lhs, current_device,
                              GEMMA4_META_BF16, 3, dims) ||
        !validate_tensor_meta(rhs_meta, rhs, current_device,
                              GEMMA4_META_BF16, 3, dims) ||
        !validate_tensor_meta(output_meta, output, current_device,
                              GEMMA4_META_BF16, 3, dims)) {
        ctx->poisoned = true;
        return -10;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    const int elements = (int)((size_t)rows * dim);
    kernel_mul_bf16<<<(elements + 255) / 256, 256, 0, stream>>>(
        (const __nv_bfloat16 *)lhs, (const __nv_bfloat16 *)rhs,
        (__nv_bfloat16 *)output, elements);
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    ctx->last_stream = stream;
    return 0;
}

static int gemma4_listener_graph_step_async(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    cudaStream_t stream
) {
    const int preflight_status = gemma4_listener_step_preflight(
        ctx, frame_audio_soft, frame_audio_soft_meta, stream);
    if (preflight_status != 0) return preflight_status;
    if (!ctx->config.enable_cuda_graph ||
        ctx->attention_backend != GEMMA4_ATTN_CUSTOM ||
        ctx->rmsnorm_backend == GEMMA4_RMS_ATEN_EXACT) return -13;

    if (ctx->graph_captured && ctx->graph_generation != ctx->state_generation) {
        if (invalidate_graph_for_state_change(ctx, stream) != 0) {
            ctx->poisoned = true;
            return -14;
        }
    }
    if (ctx->graph_captured) {
        if (ctx->graph_stream != stream || ctx->graph_input_ptr != frame_audio_soft) return -6;
        if (cudaGraphLaunch(ctx->graph_exec, stream) != cudaSuccess) {
            ctx->poisoned = true;
            return -15;
        }
    } else {
        for (uint32_t index = 0; index < ctx->rms_weight_mirror_count; ++index) {
            gemma4_rms_weight_mirror *mirror = &ctx->rms_weight_mirrors[index];
            if (mirror->ready || !mirror->bf16_ptr) continue;
            if (gemma4_aten_rmsnorm_cache_weight_async(
                    mirror->bf16_ptr, mirror->fp32_ptr, mirror->dim, stream) != 0) {
                ctx->poisoned = true;
                return -16;
            }
            mirror->ready = true;
        }
        if (!gemma4_cache_device_position(ctx->cache) ||
            gemma4_cache_set_device_position_async(
                ctx->cache, ctx->current_position, stream) != 0) {
            ctx->poisoned = true;
            return -17;
        }
        if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            ctx->poisoned = true;
            return -18;
        }
        ctx->graph_capture_active = true;
        const int body_status = gemma4_listener_step_impl(
            ctx, frame_audio_soft, frame_audio_soft_meta,
            GEMMA4_NUM_LAYERS, true, false, true, stream);
        ctx->graph_capture_active = false;

        cudaGraph_t captured_graph = nullptr;
        const cudaError_t end_status = cudaStreamEndCapture(stream, &captured_graph);
        if (body_status != 0 || end_status != cudaSuccess || !captured_graph) {
            if (captured_graph) cudaGraphDestroy(captured_graph);
            ctx->poisoned = true;
            return body_status != 0 ? body_status : -19;
        }
        ctx->cuda_graph = captured_graph;
        if (cudaGraphInstantiate(
                &ctx->graph_exec, ctx->cuda_graph, nullptr, nullptr, 0) != cudaSuccess) {
            destroy_graph_handles(ctx);
            ctx->poisoned = true;
            return -20;
        }
        ctx->graph_captured = true;
        ctx->graph_stream = stream;
        ctx->graph_input_ptr = frame_audio_soft;
        ctx->graph_generation = ctx->state_generation;
        if (cudaGraphLaunch(ctx->graph_exec, stream) != cudaSuccess) {
            destroy_graph_handles(ctx);
            ctx->poisoned = true;
            return -21;
        }
    }

    /* Host counters are fenced to the successfully enqueued replay. Cache
     * writes and the device cursor increment are ordered inside the graph. */
    if (gemma4_cache_advance_position(ctx->cache, GEMMA4_STREAM_Q_LEN) != 0) {
        ctx->poisoned = true;
        return -4;
    }
    ctx->current_position += GEMMA4_STREAM_Q_LEN;
    ctx->last_stream = stream;
    return 0;
}

/* API: Step Asynchronously (Q=4) */
int gemma4_listener_step_async(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    cudaStream_t stream
) {
    if (!ctx || !frame_audio_soft || !frame_audio_soft_meta) return -1;
    if (ctx->poisoned) return -9;
    if (!ctx->primed || ctx->prefix_len == 0) return -4;
    if (ctx->config.enable_cuda_graph) {
        return gemma4_listener_graph_step_async(
            ctx, frame_audio_soft, frame_audio_soft_meta, stream);
    }
    return gemma4_listener_step_impl(
        ctx, frame_audio_soft, frame_audio_soft_meta,
        GEMMA4_NUM_LAYERS, true, false, false, stream);
}

/* Rebase local RoPE positions and retain bounded KV without resetting the
 * canonical prefix or the session's external absolute listener frame IDs. */
int gemma4_listener_rebase_async(
    gemma4_listener_ctx *ctx,
    uint32_t retained_audio_tokens,
    cudaStream_t stream
) {
    if (!ctx) return -1;
    if (ctx->poisoned) return -9;
    if (!ctx->primed || ctx->prefix_len != GEMMA4_CANONICAL_PREFIX_LEN) return -4;
    if (retained_audio_tokens == 0 ||
        retained_audio_tokens > GEMMA4_LISTENER_MAX_ROLLOVER_AUDIO_TOKENS ||
        retained_audio_tokens % GEMMA4_STREAM_Q_LEN != 0 ||
        ctx->current_position <= ctx->prefix_len + retained_audio_tokens) return -5;
    if (observe_async_stream_error(ctx) != 0) return -12;
    if (cudaGetLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -11;
    }
    if (ctx->last_stream != 0 && ctx->last_stream != stream) return -6;
    if (ctx->config.enable_cuda_graph &&
        invalidate_graph_for_state_change(ctx, stream) != 0) {
        ctx->poisoned = true;
        return -14;
    }
    const uint32_t old_position = ctx->current_position;
    const uint32_t source_start = old_position - retained_audio_tokens;
    const int position_delta = (int)ctx->prefix_len - (int)source_start;
    if (gemma4_cache_rebase_async(
            ctx->cache,
            ctx->prefix_len,
            retained_audio_tokens,
            stream) != 0) {
        ctx->poisoned = true;
        return -7;
    }
    for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
        const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
        if (spec->is_kv_shared) continue;
        const float *inv_freq = spec->attn_type == GEMMA_ATTN_FULL
            ? ctx->d_inv_freq_full : ctx->d_inv_freq_sliding;
        const int head_dim = (int)spec->head_dim;
        const void *raw_key = gemma4_cache_get_layer_k(ctx->cache, layer_idx);
        if (!raw_key) {
            ctx->poisoned = true;
            return -8;
        }
        __nv_bfloat16 *recent_keys =
            const_cast<__nv_bfloat16 *>(static_cast<const __nv_bfloat16 *>(raw_key)) +
            (size_t)ctx->prefix_len * head_dim;
        kernel_rope_rebase_bf16<<<
            retained_audio_tokens, head_dim / 2, 0, stream
        >>>(
            recent_keys,
            inv_freq,
            (int)retained_audio_tokens,
            head_dim,
            position_delta,
            spec->attn_scaling);
        if (cudaPeekAtLastError() != cudaSuccess) {
            ctx->poisoned = true;
            return -8;
        }
    }
    ctx->current_position = ctx->prefix_len + retained_audio_tokens;
    ctx->last_stream = stream;
    if (cudaPeekAtLastError() != cudaSuccess) {
        ctx->poisoned = true;
        return -8;
    }
    return 0;
}


/* API: Output Pointer */
const void *gemma4_listener_output_ptr(gemma4_listener_ctx *ctx) {
    return ctx ? (const void *)ctx->d_output_tensor : NULL;
}

int gemma4_listener_output_health(gemma4_listener_ctx *ctx, int *out_nonfinite) {
    if (!ctx || !out_nonfinite || !ctx->d_output_health) return -1;
    if (ctx->poisoned) return -9;
    cudaError_t sync_err = cudaStreamSynchronize(ctx->last_stream);
    if (sync_err != cudaSuccess) {
        ctx->poisoned = true;
        return -2;
    }
    cudaError_t copy_err = cudaMemcpy(
        out_nonfinite, ctx->d_output_health, sizeof(int), cudaMemcpyDeviceToHost
    );
    if (copy_err != cudaSuccess) {
        ctx->poisoned = true;
        return -3;
    }
    if (*out_nonfinite != 0) {
        ctx->poisoned = true;
        return 1;
    }
    return 0;
}

/* API: Debug Pointer */
const void *gemma4_listener_debug_ptr(gemma4_listener_ctx *ctx, const char *name) {
    if (!ctx || !name) return NULL;
    const size_t hidden_slot = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE;
    const size_t q_slot = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS * 512;
    const size_t kv_slot = (size_t)GEMMA4_STREAM_Q_LEN * 512;
    const size_t ffn_slot = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_MAX_INTERMEDIATE_SIZE;
    const size_t ple_slot = (size_t)GEMMA4_STREAM_Q_LEN * GEMMA4_PLE_DIM;
    if (strcmp(name, "tap_input_hidden") == 0) return ctx->d_debug_hidden_taps;
    if (strcmp(name, "tap_input_norm") == 0) return ctx->d_debug_hidden_taps + hidden_slot;
    if (strcmp(name, "tap_o_proj") == 0) return ctx->d_debug_hidden_taps + 2 * hidden_slot;
    if (strcmp(name, "tap_post_attn_norm") == 0) return ctx->d_debug_hidden_taps + 3 * hidden_slot;
    if (strcmp(name, "tap_attn_residual") == 0) return ctx->d_debug_hidden_taps + 4 * hidden_slot;
    if (strcmp(name, "tap_pre_ff_norm") == 0) return ctx->d_debug_hidden_taps + 5 * hidden_slot;
    if (strcmp(name, "tap_down_proj") == 0) return ctx->d_debug_hidden_taps + 7 * hidden_slot;
    if (strcmp(name, "tap_post_ff_norm") == 0) return ctx->d_debug_hidden_taps + 8 * hidden_slot;
    if (strcmp(name, "tap_ff_residual") == 0) return ctx->d_debug_hidden_taps + 9 * hidden_slot;
    if (strcmp(name, "tap_ple_proj") == 0) return ctx->d_debug_hidden_taps + 10 * hidden_slot;
    if (strcmp(name, "tap_ple_norm") == 0) return ctx->d_debug_hidden_taps + 11 * hidden_slot;
    if (strcmp(name, "tap_ple_residual") == 0) return ctx->d_debug_hidden_taps + 12 * hidden_slot;
    if (strcmp(name, "tap_layer_scaled") == 0) return ctx->d_debug_hidden_taps + 13 * hidden_slot;
    if (strcmp(name, "tap_layer_output") == 0) return ctx->d_debug_hidden_taps + 14 * hidden_slot;
    if (strcmp(name, "tap_q_proj") == 0) return ctx->d_debug_q_taps;
    if (strcmp(name, "tap_q_norm") == 0) return ctx->d_debug_q_taps + q_slot;
    if (strcmp(name, "tap_q_rope") == 0) return ctx->d_debug_q_taps + 2 * q_slot;
    if (strcmp(name, "tap_attn_output") == 0) return ctx->d_debug_q_taps + 3 * q_slot;
    if (strcmp(name, "tap_k_proj") == 0) return ctx->d_debug_kv_taps;
    if (strcmp(name, "tap_k_norm") == 0) return ctx->d_debug_kv_taps + kv_slot;
    if (strcmp(name, "tap_k_rope") == 0) return ctx->d_debug_kv_taps + 2 * kv_slot;
    if (strcmp(name, "tap_v_proj") == 0) return ctx->d_debug_kv_taps + 3 * kv_slot;
    if (strcmp(name, "tap_v_norm") == 0) return ctx->d_debug_kv_taps + 4 * kv_slot;
    if (strcmp(name, "tap_gate_proj") == 0) return ctx->d_debug_ffn_taps;
    if (strcmp(name, "tap_up_proj") == 0) return ctx->d_debug_ffn_taps + ffn_slot;
    if (strcmp(name, "tap_mlp_act") == 0) return ctx->d_debug_ffn_taps + 2 * ffn_slot;
    if (strcmp(name, "tap_mlp_gate_up") == 0) return ctx->d_debug_ffn_taps + 3 * ffn_slot;
    if (strcmp(name, "tap_ple_gate") == 0) return ctx->d_debug_ple_taps;
    if (strcmp(name, "tap_ple_act") == 0) return ctx->d_debug_ple_taps + ple_slot;
    if (strcmp(name, "tap_ple_mult") == 0) return ctx->d_debug_ple_taps + 2 * ple_slot;
    if (strcmp(name, "norm_buf") == 0) return ctx->d_norm_buf;
    if (strcmp(name, "q_buf") == 0) return ctx->d_q_buf;
    if (strcmp(name, "k_buf") == 0) return ctx->d_k_buf;
    if (strcmp(name, "v_buf") == 0) return ctx->d_v_buf;
    if (strcmp(name, "attn_out") == 0) return ctx->d_attn_out;
    if (strcmp(name, "hidden_b") == 0) return ctx->d_hidden_b;
    if (strcmp(name, "gate_buf") == 0) return ctx->d_gate_buf;
    if (strcmp(name, "up_buf") == 0) return ctx->d_up_buf;
    if (strcmp(name, "mlp_act_buf") == 0) return ctx->d_mlp_act_buf;
    if (strcmp(name, "mlp_down_buf") == 0) return ctx->d_mlp_down_buf;
    if (strcmp(name, "ple_raw_proj") == 0) return ctx->d_ple_raw_proj;
    if (strcmp(name, "projected_ple") == 0) return ctx->d_projected_ple;
    if (strcmp(name, "ple_gate_buf") == 0) return ctx->d_ple_gate_buf;
    if (strcmp(name, "ple_act_buf") == 0) return ctx->d_ple_act_buf;
    if (strcmp(name, "ple_proj_out") == 0) return ctx->d_ple_proj_out;
    if (strcmp(name, "hidden_a") == 0) return ctx->d_hidden_a;
    if (strcmp(name, "layer_outputs") == 0) return ctx->d_debug_layer_outputs;
    return NULL;
}

const void *gemma4_listener_get_cached_k(const gemma4_listener_ctx *ctx, int layer_idx) {
    if (!ctx) return NULL;
    return gemma4_cache_get_layer_k(ctx->cache, layer_idx);
}

const void *gemma4_listener_get_cached_v(const gemma4_listener_ctx *ctx, int layer_idx) {
    if (!ctx) return NULL;
    return gemma4_cache_get_layer_v(ctx->cache, layer_idx);
}


/* API: Reset */
int gemma4_listener_reset(gemma4_listener_ctx *ctx) {
    if (!ctx) return -1;
    /* Reset is an explicit lifecycle boundary, not a steady-state operation.
     * Drain the last stream before invalidating counters and aliased KV state. */
    cudaError_t sync_err = cudaStreamSynchronize(ctx->last_stream);
    if (sync_err != cudaSuccess) {
        ctx->poisoned = true;
        return -2;
    }
    destroy_graph_handles(ctx);
    ++ctx->state_generation;
    ctx->current_position = 0;
    ctx->prefix_len = 0;
    ctx->primed = false;
    int cache_status = gemma4_cache_reset(ctx->cache);
    if (cache_status != 0) return cache_status;
    if (ctx->config.enable_cuda_graph &&
        gemma4_cache_set_device_position_async(ctx->cache, 0, ctx->last_stream) != 0) {
        ctx->poisoned = true;
        return -3;
    }
    if (ctx->d_output_health && cudaMemset(ctx->d_output_health, 0, sizeof(int)) != cudaSuccess) {
        ctx->poisoned = true;
        return -3;
    }
    ctx->poisoned = false;
    ctx->debug_capture_enabled = false;
    ctx->last_stream = 0;
    return 0;
}

/* API: Destroy */
void gemma4_listener_destroy(gemma4_listener_ctx *ctx) {
    if (!ctx) return;
    /* Cleanup is a lifetime boundary, so drain enqueued work before freeing
     * the borrowed-output arena, scratch buffers, or KV cache. This path is
     * never used by steady-state step execution. */
    if (ctx->last_stream) cudaStreamSynchronize(ctx->last_stream);
    destroy_graph_handles(ctx);
    if (ctx->cache) gemma4_cache_destroy(ctx->cache);
    if (ctx->cublas) cublasDestroy(ctx->cublas);

    if (ctx->d_inv_freq_sliding) cudaFree(ctx->d_inv_freq_sliding);
    if (ctx->d_inv_freq_full) cudaFree(ctx->d_inv_freq_full);
    if (ctx->d_prefix_attention) cudaFree(ctx->d_prefix_attention);
    if (ctx->d_prefix_ple) cudaFree(ctx->d_prefix_ple);

    if (ctx->d_hidden_a) cudaFree(ctx->d_hidden_a);
    if (ctx->d_hidden_b) cudaFree(ctx->d_hidden_b);
    if (ctx->d_norm_buf) cudaFree(ctx->d_norm_buf);

    if (ctx->rms_workspace.input_fp32) cudaFree(ctx->rms_workspace.input_fp32);
    if (ctx->rms_workspace.squared_fp32) cudaFree(ctx->rms_workspace.squared_fp32);
    if (ctx->rms_workspace.mean_fp32) cudaFree(ctx->rms_workspace.mean_fp32);
    if (ctx->rms_workspace.mean_plus_eps_fp32) cudaFree(ctx->rms_workspace.mean_plus_eps_fp32);
    if (ctx->rms_workspace.scale_fp32) cudaFree(ctx->rms_workspace.scale_fp32);
    if (ctx->rms_workspace.normalized_fp32) cudaFree(ctx->rms_workspace.normalized_fp32);
    if (ctx->rms_workspace.weighted_fp32) cudaFree(ctx->rms_workspace.weighted_fp32);
    if (ctx->rms_workspace.weight_fp32) cudaFree(ctx->rms_workspace.weight_fp32);
    if (ctx->native_rms_workspace.squared_fp32) cudaFree(ctx->native_rms_workspace.squared_fp32);
    if (ctx->native_rms_workspace.mean_fp32) cudaFree(ctx->native_rms_workspace.mean_fp32);
    if (ctx->native_rms_workspace.scale_fp32) cudaFree(ctx->native_rms_workspace.scale_fp32);
    for (uint32_t i = 0; i < ctx->rms_weight_mirror_count; ++i) {
        if (ctx->rms_weight_mirrors[i].fp32_ptr) cudaFree(ctx->rms_weight_mirrors[i].fp32_ptr);
    }

    if (ctx->d_q_buf) cudaFree(ctx->d_q_buf);
    if (ctx->d_k_buf) cudaFree(ctx->d_k_buf);
    if (ctx->d_v_buf) cudaFree(ctx->d_v_buf);
    if (ctx->d_attn_out) cudaFree(ctx->d_attn_out);
    if (ctx->d_sdpa_mask) cudaFree(ctx->d_sdpa_mask);
    if (ctx->d_sdpa_k_repeat) cudaFree(ctx->d_sdpa_k_repeat);
    if (ctx->d_sdpa_v_repeat) cudaFree(ctx->d_sdpa_v_repeat);
    if (ctx->d_sdpa_shared_sliding_k_repeat) cudaFree(ctx->d_sdpa_shared_sliding_k_repeat);
    if (ctx->d_sdpa_shared_sliding_v_repeat) cudaFree(ctx->d_sdpa_shared_sliding_v_repeat);
    for (int slot = 0; slot < 3; ++slot) {
        if (ctx->d_sdpa_full_k_repeat[slot]) cudaFree(ctx->d_sdpa_full_k_repeat[slot]);
        if (ctx->d_sdpa_full_v_repeat[slot]) cudaFree(ctx->d_sdpa_full_v_repeat[slot]);
    }
    gemma4_ordered_workspace_destroy(&ctx->ordered_workspace);

    if (ctx->d_gate_buf) cudaFree(ctx->d_gate_buf);
    if (ctx->d_up_buf) cudaFree(ctx->d_up_buf);
    if (ctx->d_mlp_act_buf) cudaFree(ctx->d_mlp_act_buf);
    if (ctx->d_mlp_down_buf) cudaFree(ctx->d_mlp_down_buf);

    if (ctx->d_ple_raw_proj) cudaFree(ctx->d_ple_raw_proj);
    if (ctx->d_projected_ple) cudaFree(ctx->d_projected_ple);
    if (ctx->d_ple_gate_buf) cudaFree(ctx->d_ple_gate_buf);
    if (ctx->d_ple_act_buf) cudaFree(ctx->d_ple_act_buf);
    if (ctx->d_ple_proj_out) cudaFree(ctx->d_ple_proj_out);

    if (ctx->owns_output_tensor && ctx->d_output_tensor) {
        cudaFree(ctx->d_output_tensor);
    }
    if (ctx->d_output_health) cudaFree(ctx->d_output_health);
    if (ctx->d_debug_hidden_taps) cudaFree(ctx->d_debug_hidden_taps);
    if (ctx->d_debug_q_taps) cudaFree(ctx->d_debug_q_taps);
    if (ctx->d_debug_kv_taps) cudaFree(ctx->d_debug_kv_taps);
    if (ctx->d_debug_ffn_taps) cudaFree(ctx->d_debug_ffn_taps);
    if (ctx->d_debug_ple_taps) cudaFree(ctx->d_debug_ple_taps);
    if (ctx->d_debug_layer_outputs) cudaFree(ctx->d_debug_layer_outputs);

    free(ctx);
}

/* Manifest & ABI validation */
const char *gemma4_listener_manifest_sha256(void) {
    return GEMMA4_TOPOLOGY_MANIFEST_SHA256;
}

size_t gemma4_listener_abi_config_size(void) {
    return sizeof(gemma4_listener_config);
}

size_t gemma4_listener_abi_tensor_meta_size(void) {
    return sizeof(gemma4_tensor_meta);
}

size_t gemma4_listener_abi_layer_weights_size(void) {
    return sizeof(gemma4_layer_weights);
}

size_t gemma4_listener_abi_weights_size(void) {
    return sizeof(gemma4_listener_weights);
}

size_t gemma4_listener_abi_imported_kv_size(void) {
    return sizeof(gemma4_imported_kv);
}

size_t gemma4_listener_abi_imported_cache_size(void) {
    return sizeof(gemma4_imported_cache);
}

static size_t copy_abi_offsets(const size_t *offsets, size_t count, size_t *out, size_t capacity) {
    if (out != nullptr && capacity >= count) {
        for (size_t i = 0; i < count; ++i) {
            out[i] = offsets[i];
        }
    }
    return count;
}

size_t gemma4_listener_abi_tensor_meta_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_tensor_meta, ptr),
        offsetof(gemma4_tensor_meta, dtype),
        offsetof(gemma4_tensor_meta, ndim),
        offsetof(gemma4_tensor_meta, contiguous),
        offsetof(gemma4_tensor_meta, reserved),
        offsetof(gemma4_tensor_meta, dims),
        offsetof(gemma4_tensor_meta, strides),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

size_t gemma4_listener_abi_config_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_listener_config, max_seq_len),
        offsetof(gemma4_listener_config, stream_q_len),
        offsetof(gemma4_listener_config, enable_cuda_graph),
        offsetof(gemma4_listener_config, output_tensor_ptr),
        offsetof(gemma4_listener_config, frame_pad_ple_ptr),
        offsetof(gemma4_listener_config, weight_meta),
        offsetof(gemma4_listener_config, weight_meta_count),
        offsetof(gemma4_listener_config, output_meta),
        offsetof(gemma4_listener_config, frame_pad_ple_meta),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

size_t gemma4_listener_abi_layer_weights_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_layer_weights, input_norm_w),
        offsetof(gemma4_layer_weights, post_attn_norm_w),
        offsetof(gemma4_layer_weights, pre_ff_norm_w),
        offsetof(gemma4_layer_weights, post_ff_norm_w),
        offsetof(gemma4_layer_weights, q_proj_w),
        offsetof(gemma4_layer_weights, k_proj_w),
        offsetof(gemma4_layer_weights, v_proj_w),
        offsetof(gemma4_layer_weights, o_proj_w),
        offsetof(gemma4_layer_weights, q_norm_w),
        offsetof(gemma4_layer_weights, k_norm_w),
        offsetof(gemma4_layer_weights, v_norm_w),
        offsetof(gemma4_layer_weights, gate_proj_w),
        offsetof(gemma4_layer_weights, up_proj_w),
        offsetof(gemma4_layer_weights, down_proj_w),
        offsetof(gemma4_layer_weights, ple_gate_w),
        offsetof(gemma4_layer_weights, ple_proj_w),
        offsetof(gemma4_layer_weights, ple_norm_w),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

size_t gemma4_listener_abi_weights_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_listener_weights, final_norm_w),
        offsetof(gemma4_listener_weights, per_layer_model_proj_w),
        offsetof(gemma4_listener_weights, per_layer_proj_norm_w),
        offsetof(gemma4_listener_weights, per_layer_model_proj_scale),
        offsetof(gemma4_listener_weights, per_layer_input_scale),
        offsetof(gemma4_listener_weights, layers),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

size_t gemma4_listener_abi_imported_kv_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_imported_kv, k_ptr),
        offsetof(gemma4_imported_kv, k_meta),
        offsetof(gemma4_imported_kv, v_ptr),
        offsetof(gemma4_imported_kv, v_meta),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

size_t gemma4_listener_abi_imported_cache_offsets(size_t *out_offsets, size_t capacity) {
    static const size_t offsets[] = {
        offsetof(gemma4_imported_cache, layers),
    };
    return copy_abi_offsets(offsets, sizeof(offsets) / sizeof(offsets[0]), out_offsets, capacity);
}

int gemma4_listener_abi_version(void) {
    return GEMMA4_ABI_VERSION;
}

uint32_t gemma4_listener_runtime_attention_backend(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->attention_backend : GEMMA4_ATTN_CUSTOM;
}

uint32_t gemma4_listener_runtime_rmsnorm_backend(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->rmsnorm_backend : GEMMA4_RMS_CUSTOM_FAST;
}

uint32_t gemma4_listener_runtime_max_seq_len(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->config.max_seq_len : 0;
}

uint32_t gemma4_listener_runtime_current_position(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->current_position : 0;
}

uint64_t gemma4_listener_runtime_splitk_layer_mask(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->splitk_layer_mask : 0;
}

uint64_t gemma4_listener_runtime_ordered_calls_by_layer(
    const gemma4_listener_ctx *ctx, uint32_t layer_idx) {
    if (!ctx || layer_idx >= GEMMA4_NUM_LAYERS ||
        ctx->attention_backend != GEMMA4_ATTN_NATIVE_SPLITK_GQA) return 0;
    return ctx->ordered_calls_by_layer[layer_idx];
}

size_t gemma4_listener_runtime_derived_weight_bytes(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->rms_derived_weight_bytes : 0;
}

const char *gemma4_listener_runtime_attention_implementation(
    const gemma4_listener_ctx *ctx) {
    return ctx && ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
        ? GEMMA4_ORDERED_IMPLEMENTATION
        : nullptr;
}

uint32_t gemma4_listener_runtime_ordered_key_tile_size(
    const gemma4_listener_ctx *ctx) {
    return ctx && ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
        ? GEMMA4_ORDERED_KEY_TILE_SIZE
        : 0;
}

uint32_t gemma4_listener_runtime_ordered_min_keys_exclusive(
    const gemma4_listener_ctx *ctx) {
    return ctx && ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
        ? GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE
        : 0;
}

size_t gemma4_listener_runtime_ordered_workspace_bytes(
    const gemma4_listener_ctx *ctx) {
    return ctx && ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA
        ? gemma4_ordered_workspace_bytes(&ctx->ordered_workspace)
        : 0;
}

#define GEMMA4_ORDERED_RUNTIME_COUNTER(name, getter) \
    uint64_t name(const gemma4_listener_ctx *ctx) { \
        return ctx && ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA \
            ? getter(&ctx->ordered_workspace) : 0; \
    }
GEMMA4_ORDERED_RUNTIME_COUNTER(
    gemma4_listener_runtime_ordered_pack_launches,
    gemma4_ordered_pack_launches)
GEMMA4_ORDERED_RUNTIME_COUNTER(
    gemma4_listener_runtime_ordered_stage_a_launches,
    gemma4_ordered_stage_a_launches)
GEMMA4_ORDERED_RUNTIME_COUNTER(
    gemma4_listener_runtime_ordered_stage_b_launches,
    gemma4_ordered_stage_b_launches)
GEMMA4_ORDERED_RUNTIME_COUNTER(
    gemma4_listener_runtime_ordered_stage_c_launches,
    gemma4_ordered_stage_c_launches)
GEMMA4_ORDERED_RUNTIME_COUNTER(
    gemma4_listener_runtime_ordered_stage_d_launches,
    gemma4_ordered_stage_d_launches)
#undef GEMMA4_ORDERED_RUNTIME_COUNTER

uint32_t gemma4_listener_current_position(const gemma4_listener_ctx *ctx) {
    return gemma4_listener_runtime_current_position(ctx);
}
