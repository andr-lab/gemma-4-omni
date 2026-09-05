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
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "gemma4_listener.h"
#include "gemma4_listener_aten.h"
#include "gemma4_listener_aten_norm.h"
#include "generated_topology.h"
#include "gemma4_listener_kernels.cuh"

/* Internal forward declarations for cache functions */
typedef struct gemma4_cache gemma4_cache;
int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len);
int gemma4_cache_append_async(gemma4_cache *c, int layer_idx, const __nv_bfloat16 *new_k, const __nv_bfloat16 *new_v, int q_len, cudaStream_t stream);
int gemma4_cache_import_async(gemma4_cache *c, const gemma4_imported_cache *imported, uint32_t current_position, cudaStream_t stream);
int gemma4_cache_advance_position(gemma4_cache *c, int q_len);
void gemma4_cache_get_ptrs(const gemma4_cache *c, int layer_idx, int q_len, uint32_t base_pos, const __nv_bfloat16 **out_k, const __nv_bfloat16 **out_v, int *out_total_len, int *out_start_offset);
int gemma4_cache_reset(gemma4_cache *c);
void gemma4_cache_destroy(gemma4_cache *c);
const void *gemma4_cache_get_layer_k(const gemma4_cache *c, int layer_idx);
const void *gemma4_cache_get_layer_v(const gemma4_cache *c, int layer_idx);

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

    /* Fixed Preallocated Device Scratch Arena */
    __nv_bfloat16 *d_hidden_a;        /* [q_max, 1536] */
    __nv_bfloat16 *d_hidden_b;        /* [q_max, 1536] */
    __nv_bfloat16 *d_norm_buf;        /* [q_max, 1536] */

    __nv_bfloat16 *d_q_buf;           /* [q_max, 8 * 512] */
    __nv_bfloat16 *d_k_buf;           /* [q_max, 1 * 512] */
    __nv_bfloat16 *d_v_buf;           /* [q_max, 1 * 512] */
    __nv_bfloat16 *d_attn_out;        /* [q_max, 8 * 512] */
    bool *d_sdpa_mask;                 /* diagnostic ATen bridge [q_max, max_seq_len] */
    bool use_aten_sdpa_bridge;         /* explicit opt-in; default native kernel */
    bool use_aten_rmsnorm_bridge;      /* explicit opt-in; pre-FF norm only */

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

    /* CUDA Graph handle for steady-state step */
    bool graph_captured;
    cudaGraph_t cuda_graph;
    cudaGraphExec_t graph_exec;
    uint32_t current_position;
};

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

/* Execute 1 layer of Gemma 4 text decoder */
static int execute_layer(
    gemma4_listener_ctx *ctx,
    int layer_idx,
    int compute_q_len,
    int valid_q_len,
    uint32_t base_pos,
    cudaStream_t stream
) {
    int q_len = compute_q_len;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    const gemma4_layer_weights *lw = &ctx->weights.layers[layer_idx];
    int D = spec->head_dim;
    int hidden_size = GEMMA4_HIDDEN_SIZE;
    int intermediate_size = spec->intermediate_size;

    /* 1. Input RMSNorm: norm(hidden_a) -> norm_buf */
    if (debug_snapshot_bf16(ctx, ctx->d_debug_hidden_taps, 0,
                             GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE,
                             ctx->d_hidden_a, q_len * hidden_size, stream) != 0) return -7;
    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_a,
        (const __nv_bfloat16 *)lw->input_norm_w,
        ctx->d_norm_buf,
        q_len,
        spec->input_norm_eps
    );
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
    if (D == 256) {
        kernel_rmsnorm_bf16<256><<<GEMMA4_NUM_ATTN_HEADS * q_len, 256, 0, stream>>>(
            ctx->d_q_buf,
            (const __nv_bfloat16 *)lw->q_norm_w,
            ctx->d_q_buf,
            GEMMA4_NUM_ATTN_HEADS * q_len,
            spec->q_norm_eps
        );
    } else {
        kernel_rmsnorm_bf16<512><<<GEMMA4_NUM_ATTN_HEADS * q_len, 256, 0, stream>>>(
            ctx->d_q_buf,
            (const __nv_bfloat16 *)lw->q_norm_w,
            ctx->d_q_buf,
            GEMMA4_NUM_ATTN_HEADS * q_len,
            spec->q_norm_eps
        );
    }
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
        if (D == 256) {
            kernel_rmsnorm_bf16<256><<<q_len, 256, 0, stream>>>(
                ctx->d_k_buf,
                (const __nv_bfloat16 *)lw->k_norm_w,
                ctx->d_k_buf,
                q_len,
                spec->k_norm_eps
            );
        } else {
            kernel_rmsnorm_bf16<512><<<q_len, 256, 0, stream>>>(
                ctx->d_k_buf,
                (const __nv_bfloat16 *)lw->k_norm_w,
                ctx->d_k_buf,
                q_len,
                spec->k_norm_eps
            );
        }
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
            spec->attn_scaling
        );
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 2,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_k_buf, q_len * D, stream) != 0) return -7;

        /* V norm */
        if (D == 256) {
            kernel_rmsnorm_bf16<256><<<q_len, 256, 0, stream>>>(
                ctx->d_v_buf,
                (const __nv_bfloat16 *)lw->v_norm_w,
                ctx->d_v_buf,
                q_len,
                spec->v_norm_eps
            );
        } else {
            kernel_rmsnorm_bf16<512><<<q_len, 256, 0, stream>>>(
                ctx->d_v_buf,
                (const __nv_bfloat16 *)lw->v_norm_w,
                ctx->d_v_buf,
                q_len,
                spec->v_norm_eps
            );
        }
        if (debug_snapshot_bf16(ctx, ctx->d_debug_kv_taps, 4,
                                 GEMMA4_STREAM_Q_LEN * 512,
                                 ctx->d_v_buf, q_len * D, stream) != 0) return -7;

        /* Write to KV cache */
        if (gemma4_cache_append_async(
                ctx->cache, layer_idx, ctx->d_k_buf, ctx->d_v_buf, valid_q_len, stream) != 0) {
            return -3;
        }
    }

    /* 3. SDPA Attention */
    const __nv_bfloat16 *k_data, *v_data;
    int total_k_len, k_start_offset;
    gemma4_cache_get_ptrs(ctx->cache, layer_idx, valid_q_len, base_pos, &k_data, &v_data, &total_k_len, &k_start_offset);

    float sm_scale = spec->attn_scaling; /* Exactly 1.0 in Gemma 4 */
    size_t smem_bytes = 4 * (size_t)ctx->config.max_seq_len * sizeof(float);

    if (ctx->use_aten_sdpa_bridge) {
        const int visible_k_len = total_k_len - k_start_offset;
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
        const uint64_t q_strides[] = {
            (uint64_t)GEMMA4_NUM_ATTN_HEADS * (uint64_t)q_len * (uint64_t)D,
            (uint64_t)D,
            (uint64_t)GEMMA4_NUM_ATTN_HEADS * (uint64_t)D,
            1,
        };
        const uint64_t kv_dims[] = {1, GEMMA4_NUM_KV_HEADS, (uint64_t)visible_k_len, (uint64_t)D};
        const uint64_t kv_strides[] = {
            (uint64_t)ctx->config.max_seq_len * (uint64_t)D,
            (uint64_t)ctx->config.max_seq_len * (uint64_t)D,
            (uint64_t)D,
            1,
        };
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
            k_data + (size_t)k_start_offset * (size_t)D,
            GEMMA4_META_BF16, 4, kv_dims, kv_strides, 0);
        gemma4_tensor_meta v_meta = make_bridge_meta(
            v_data + (size_t)k_start_offset * (size_t)D,
            GEMMA4_META_BF16, 4, kv_dims, kv_strides, 0);
        gemma4_tensor_meta mask_meta = make_bridge_meta(
            ctx->d_sdpa_mask, GEMMA4_META_BOOL, 4, mask_dims, mask_strides, 1);
        gemma4_tensor_meta out_meta = make_bridge_meta(
            ctx->d_attn_out, GEMMA4_META_BF16, 4, q_dims, q_strides, 0);
        if (gemma4_listener_debug_attention(
                ctx->d_q_buf, &q_meta,
                k_data + (size_t)k_start_offset * (size_t)D, &k_meta,
                v_data + (size_t)k_start_offset * (size_t)D, &v_meta,
                ctx->d_sdpa_mask, &mask_meta,
                ctx->d_attn_out, &out_meta,
                (double)sm_scale, stream) != 0) {
            return -8;
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
    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_b,
        (const __nv_bfloat16 *)lw->post_attn_norm_w,
        ctx->d_hidden_b,
        q_len,
        spec->post_attn_norm_eps
    );
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
    /* Pre-FF norm on hidden_a -> norm_buf.  The ATen branch is a diagnostic
     * numerical island selected only by explicit opt-in; every other norm
     * remains on the current custom kernel. */
    if (ctx->use_aten_rmsnorm_bridge) {
        if (gemma4_aten_rmsnorm_parity_async(
                ctx->d_hidden_a,
                lw->pre_ff_norm_w,
                ctx->d_norm_buf,
                (uint32_t)q_len,
                GEMMA4_HIDDEN_SIZE,
                (double)spec->pre_ff_norm_eps,
                stream) != 0) return -8;
    } else {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_hidden_a,
            (const __nv_bfloat16 *)lw->pre_ff_norm_w,
            ctx->d_norm_buf,
            q_len,
            spec->pre_ff_norm_eps
        );
    }
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
    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_b,
        (const __nv_bfloat16 *)lw->post_ff_norm_w,
        ctx->d_hidden_b,
        q_len,
        spec->post_ff_norm_eps
    );
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
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_ple_proj_out,
            (const __nv_bfloat16 *)lw->ple_norm_w,
            ctx->d_ple_proj_out,
            q_len,
            spec->post_ple_norm_eps
        );
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

    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    return 0;
}

/* API: Create Context */
#define GEMMA4_CREATE_ALLOC(ptr, bytes) \
    do { \
        if (cudaMalloc((void **)&(ptr), (bytes)) != cudaSuccess) { \
            gemma4_listener_destroy(ctx); \
            return -4; \
        } \
    } while (0)

int gemma4_listener_create(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx
) {
    if (!config || !weights || !out_ctx) return -1;
    if (config->stream_q_len != GEMMA4_STREAM_Q_LEN ||
        config->max_seq_len < GEMMA4_CANONICAL_PREFIX_LEN ||
        config->max_seq_len > GEMMA4_MAX_STREAM_SCORE_KEYS ||
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
    ctx->current_position = 0;
    ctx->graph_captured = false;
    ctx->prefix_len = 0;
    ctx->primed = false;
    ctx->poisoned = false;
    ctx->last_stream = 0;
    ctx->debug_capture_enabled = false;
    const char *aten_bridge_env = getenv("GEMMA4_ATEN_SDPA_BRIDGE");
    ctx->use_aten_sdpa_bridge = aten_bridge_env && strcmp(aten_bridge_env, "0") != 0;
    const char *aten_rmsnorm_env = getenv("GEMMA4_ATEN_RMSNORM_BRIDGE");
    ctx->use_aten_rmsnorm_bridge = aten_rmsnorm_env && strcmp(aten_rmsnorm_env, "0") != 0;

    /* The handle is persistent; projection calls only switch its stream. */
    cublasStatus_t cb_stat = cublasCreate(&ctx->cublas);
    if (cb_stat != CUBLAS_STATUS_SUCCESS) {
        free(ctx);
        return -2;
    }

    /* Initialize Cache */
    int cache_err = gemma4_cache_create(&ctx->cache, config->max_seq_len);
    if (cache_err != 0) {
        gemma4_listener_destroy(ctx);
        return -3;
    }

    /* The stream SDPA kernel keeps four score rows in dynamic shared memory.
     * Opt in to the larger per-block budget needed by max_seq_len > 2048. */
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

    /* Precompute and upload RoPE frequencies */
    float inv_freq_s[128];
    const float sliding_rope_base = gemma4_layers[0].rope_base;
    for (int i = 0; i < 128; ++i) {
        inv_freq_s[i] = 1.0f / powf(sliding_rope_base, (float)(2 * i) / 256.0f);
    }
    GEMMA4_CREATE_ALLOC(ctx->d_inv_freq_sliding, 128 * sizeof(float));
    if (cudaMemcpy(ctx->d_inv_freq_sliding, inv_freq_s, 128 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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
    if (cudaMemcpy(ctx->d_inv_freq_full, inv_freq_f, 256 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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
    if (ctx->use_aten_sdpa_bridge) {
        GEMMA4_CREATE_ALLOC(ctx->d_sdpa_mask,
                            q_max * (size_t)config->max_seq_len * sizeof(bool));
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

    *out_ctx = ctx;
    return 0;
}
#undef GEMMA4_CREATE_ALLOC

/* API: Import an exact canonical ShapeStable DynamicCache.  This is the
 * parity-first path: the prefix arithmetic stays in PyTorch, while the native
 * executor takes ownership of a D2D copy of the 15 non-shared source caches.
 * Shared consumer layers already alias sources 13 and 14 in gemma4_cache. */
int gemma4_listener_import_cache_async(
    gemma4_listener_ctx *ctx,
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    cudaStream_t stream
) {
    if (!ctx || !imported) return -1;
    if (current_position != GEMMA4_CANONICAL_PREFIX_LEN ||
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
    ctx->last_stream = stream;
    ctx->primed = true;
    return 0;
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
int gemma4_listener_step_debug(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    uint32_t num_layers,
    bool apply_final_norm,
    cudaStream_t stream
) {
    if (!ctx || !frame_audio_soft || !frame_audio_soft_meta) return -1;
    if (ctx->poisoned) return -9;
    if (observe_async_stream_error(ctx) != 0) return -12;
    cudaError_t prior_error = cudaGetLastError();
    if (prior_error != cudaSuccess) {
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
        kernel_ple_project_and_norm_bf16<<<q_len * GEMMA4_NUM_LAYERS, 256, 0, stream>>>(
            ctx->d_ple_raw_proj,
            (const __nv_bfloat16 *)ctx->weights.per_layer_proj_norm_w,
            token_id,
            ctx->d_projected_ple,
            q_len * GEMMA4_NUM_LAYERS,
            ctx->weights.per_layer_model_proj_scale,
            1e-6f,
            ctx->weights.per_layer_input_scale
        );
        const int frame_ple_elements = q_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
        kernel_residual_add_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_projected_ple,
            token_id,
            ctx->d_projected_ple,
            frame_ple_elements
        );
        kernel_scale_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_projected_ple,
            ctx->d_projected_ple,
            frame_ple_elements,
            ctx->weights.per_layer_input_scale
        );
    }

    /* 2. Execute requested layers */
    for (uint32_t l = 0; l < num_layers; ++l) {
        if (execute_layer(ctx, l, (int)q_len, (int)q_len, base_pos, stream) != 0) {
            ctx->poisoned = true;
            return -3;
        }
    }

    /* 3. Output */
    if (apply_final_norm) {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_hidden_a,
            (const __nv_bfloat16 *)ctx->weights.final_norm_w,
            ctx->d_output_tensor,
            q_len,
            GEMMA4_FINAL_NORM_EPS
        );
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

    /* Advance cache position */
    if (gemma4_cache_advance_position(ctx->cache, (int)q_len) != 0) {
        ctx->poisoned = true;
        return -4;
    }
    ctx->current_position += q_len;
    return 0;
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
    return gemma4_listener_step_debug(
        ctx, frame_audio_soft, frame_audio_soft_meta, GEMMA4_NUM_LAYERS, true, stream
    );
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
    ctx->current_position = 0;
    ctx->prefix_len = 0;
    ctx->primed = false;
    int cache_status = gemma4_cache_reset(ctx->cache);
    if (cache_status != 0) return cache_status;
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
    if (ctx->cache) gemma4_cache_destroy(ctx->cache);
    if (ctx->cublas) cublasDestroy(ctx->cublas);

    if (ctx->d_inv_freq_sliding) cudaFree(ctx->d_inv_freq_sliding);
    if (ctx->d_inv_freq_full) cudaFree(ctx->d_inv_freq_full);
    if (ctx->d_prefix_attention) cudaFree(ctx->d_prefix_attention);
    if (ctx->d_prefix_ple) cudaFree(ctx->d_prefix_ple);

    if (ctx->d_hidden_a) cudaFree(ctx->d_hidden_a);
    if (ctx->d_hidden_b) cudaFree(ctx->d_hidden_b);
    if (ctx->d_norm_buf) cudaFree(ctx->d_norm_buf);

    if (ctx->d_q_buf) cudaFree(ctx->d_q_buf);
    if (ctx->d_k_buf) cudaFree(ctx->d_k_buf);
    if (ctx->d_v_buf) cudaFree(ctx->d_v_buf);
    if (ctx->d_attn_out) cudaFree(ctx->d_attn_out);
    if (ctx->d_sdpa_mask) cudaFree(ctx->d_sdpa_mask);

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

uint32_t gemma4_listener_current_position(const gemma4_listener_ctx *ctx) {
    return ctx ? ctx->current_position : 0;
}
