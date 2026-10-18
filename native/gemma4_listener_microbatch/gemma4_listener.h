/*
 * gemma4_listener.h — C ABI for DS4-style Gemma 4 Native Listener Executor.
 *
 * Model-specific, device-resident, zero-allocation native GPU execution for Gemma 4 E2B.
 * Conforms to P4-NATIVE Implementation Plan §6 & §7.
 */
#ifndef GEMMA4_LISTENER_H
#define GEMMA4_LISTENER_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <cuda_runtime.h>
#include "generated_topology.h"
#include "gemma4_listener_rms_exact.cuh"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque context handle */
typedef struct gemma4_listener_ctx gemma4_listener_ctx;

#define GEMMA4_CUSTOM_SDPA_MAX_SEQ_LEN 4096
#define GEMMA4_ATEN_SDPA_MAX_SEQ_LEN 16384
#define GEMMA4_LISTENER_MAX_ROLLOVER_AUDIO_TOKENS (64 * GEMMA4_STREAM_Q_LEN)
#define GEMMA4_MAX_STREAM_SCORE_KEYS GEMMA4_ATEN_SDPA_MAX_SEQ_LEN
#define GEMMA4_TENSOR_META_MAX_RANK 4
#define GEMMA4_GLOBAL_WEIGHT_META_COUNT 3
#define GEMMA4_LAYER_WEIGHT_META_COUNT 17
#define GEMMA4_WEIGHT_META_COUNT (GEMMA4_GLOBAL_WEIGHT_META_COUNT + GEMMA4_NUM_LAYERS * GEMMA4_LAYER_WEIGHT_META_COUNT)

typedef enum {
    GEMMA4_META_BF16 = 1,
    GEMMA4_META_INT64 = 2,
    GEMMA4_META_FP32 = 3, /* diagnostic ATen bridge masks only */
    GEMMA4_META_BOOL = 4, /* diagnostic ATen bridge masks only */
} gemma4_tensor_dtype;

/* Host-side descriptor used only for synchronous preflight.  The native
 * executor copies no caller-owned descriptor and never retains this pointer. */
typedef struct {
    const void *ptr;
    uint32_t dtype;
    uint32_t ndim;
    uint32_t contiguous;
    uint32_t reserved;
    uint64_t dims[GEMMA4_TENSOR_META_MAX_RANK];
    uint64_t strides[GEMMA4_TENSOR_META_MAX_RANK];
} gemma4_tensor_meta;

/* Configuration for context creation */
typedef struct {
    uint32_t max_seq_len;         /* custom <=4096; ATen exact <=16384 */
    uint32_t stream_q_len;        /* exactly 2 soft tokens per 80 ms frame */
    bool enable_cuda_graph;       /* opt-in per-session CUSTOM Q4 step graph; default false */
    void *output_tensor_ptr;      /* optional preallocated [1, stream_q_len, hidden_size] BF16 */
    const void *frame_pad_ple_ptr;/* precomputed [1, stream_q_len, num_layers, ple_dim] BF16 token identities */
    const gemma4_tensor_meta *weight_meta; /* exactly GEMMA4_WEIGHT_META_COUNT host descriptors */
    uint32_t weight_meta_count;
    const gemma4_tensor_meta *output_meta; /* descriptor for output_tensor_ptr, or NULL with NULL output */
    const gemma4_tensor_meta *frame_pad_ple_meta;
} gemma4_listener_config;

/* Per-layer weights registered from PyTorch (BF16, device pointers, zero duplication) */
typedef struct {
    const void *input_norm_w;         /* [hidden_size] */
    const void *post_attn_norm_w;     /* [hidden_size] */
    const void *pre_ff_norm_w;        /* [hidden_size] */
    const void *post_ff_norm_w;       /* [hidden_size] */

    const void *q_proj_w;             /* [num_heads * head_dim, hidden_size] */
    const void *k_proj_w;             /* [num_kv_heads * head_dim, hidden_size] or NULL */
    const void *v_proj_w;             /* [num_kv_heads * head_dim, hidden_size] or NULL */
    const void *o_proj_w;             /* [hidden_size, num_heads * head_dim] */

    const void *q_norm_w;             /* [head_dim] */
    const void *k_norm_w;             /* [head_dim] or NULL */
    const void *v_norm_w;             /* [head_dim] or NULL */

    const void *gate_proj_w;          /* [intermediate_size, hidden_size] */
    const void *up_proj_w;            /* [intermediate_size, hidden_size] */
    const void *down_proj_w;          /* [hidden_size, intermediate_size] */

    const void *ple_gate_w;           /* [ple_dim, hidden_size] */
    const void *ple_proj_w;           /* [hidden_size, ple_dim] */
    const void *ple_norm_w;           /* [hidden_size] */
} gemma4_layer_weights;

/* Global model weights */
typedef struct {
    const void *final_norm_w;                 /* [hidden_size] */
    const void *per_layer_model_proj_w;       /* [num_layers * ple_dim, hidden_size] */
    const void *per_layer_proj_norm_w;        /* [ple_dim] */
    float per_layer_model_proj_scale;
    float per_layer_input_scale;

    gemma4_layer_weights layers[GEMMA4_NUM_LAYERS];
} gemma4_listener_weights;

/* Canonical DynamicCache import descriptor. Only non-shared source layers
 * carry pointers; shared consumer layers must be zeroed and alias their
 * attested source layer in the native cache. Descriptors are host-side and
 * are consumed synchronously during API validation; K/V payloads are copied
 * device-to-device on the supplied stream. */
typedef struct {
    const void *k_ptr;
    const gemma4_tensor_meta *k_meta;
    const void *v_ptr;
    const gemma4_tensor_meta *v_meta;
} gemma4_imported_kv;

typedef struct {
    gemma4_imported_kv layers[GEMMA4_NUM_LAYERS];
} gemma4_imported_cache;

/* Context Lifecycle API */
int gemma4_listener_create(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx
);
/* Production/session startup entrypoint. Negative policy values preserve the
 * legacy environment resolution for that individual policy. */
int gemma4_listener_create_with_policy(
    const gemma4_listener_config *config,
    const gemma4_listener_weights *weights,
    gemma4_listener_ctx **out_ctx,
    int32_t attention_backend,
    int32_t rmsnorm_backend
);

/* Asynchronously prime canonical prompt / prefix (34 tokens: BOS + 32 text slots + BOA). */
int gemma4_listener_import_cache_async(
    gemma4_listener_ctx *ctx,
    const gemma4_imported_cache *cache,
    uint32_t current_position,
    cudaStream_t stream
);

/* Import a ShapeStable cache with its actual semantic-prefix boundary and
 * attention mask. The legacy import above keeps the canonical 34-token mask. */
int gemma4_listener_import_cache_with_prefix_async(
    gemma4_listener_ctx *ctx,
    const gemma4_imported_cache *cache,
    uint32_t current_position,
    const int64_t *prefix_attention,
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    cudaStream_t stream
);

/* Execute one selected layer from exact oracle-provided hidden, projected PLE
 * and canonical source KV. This diagnostic entrypoint does not advance the
 * listener position or mark the context primed; callers should use a reset or
 * dedicated context between independent probes. */
int gemma4_listener_debug_execute_layer(
    gemma4_listener_ctx *ctx,
    uint32_t layer_idx,
    const void *hidden_input,
    const gemma4_tensor_meta *hidden_input_meta,
    const void *projected_ple,
    const gemma4_tensor_meta *projected_ple_meta,
    const gemma4_imported_cache *cache,
    uint32_t current_position,
    uint32_t query_position,
    cudaStream_t stream
);

/* Diagnostic native split-K GQA primitive.  It borrows contiguous CUDA BF16
 * Q=[1,4,8,512] in the production q-major layout, K/V=[1,1,K,512], and
 * applies causal masking plus the optional explicit prefix mask. This is the
 * isolated kernel gate; production execution enters the same kernels from
 * execute_layer with the canonical prefix mask. */
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
);

int gemma4_listener_prime_async(
    gemma4_listener_ctx *ctx,
    const void *prefix_embeds,       /* [1, prefix_len, hidden_size] BF16 */
    const gemma4_tensor_meta *prefix_embeds_meta,
    const void *prefix_ple,          /* [1, prefix_len, num_layers, ple_dim] BF16 */
    const gemma4_tensor_meta *prefix_ple_meta,
    const int64_t *prefix_attention, /* [1, prefix_len], CUDA int64 0/1 mask */
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,             /* valid length in [1, GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN] */
    cudaStream_t stream
);

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
);

/* Asynchronously execute one 80 ms audio frame (Q=2 soft tokens).
 * Calls in one primed context must use the same CUDA stream; reset() is the
 * explicit boundary for switching streams.  This avoids implicit sync while
 * keeping ordering fail-closed. */
int gemma4_listener_step_async(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft, /* [1, 4, hidden_size] BF16 (dynamic pointer, copied D2D) */
    const gemma4_tensor_meta *frame_audio_soft_meta,
    cudaStream_t stream
);

/* Append already-observed chronological Q4 audio history in one causal prefill.
 * Requires a freshly primed prefix (cache/current position == prefix_len), a
 * contiguous BF16 [1, audio_tokens, hidden] input, and audio_tokens in 4..256.
 * Only valid audio tokens commit to KV; failure poisons the context. */
int gemma4_listener_append_history_batch_async(
    gemma4_listener_ctx *ctx,
    const void *audio_soft,
    const gemma4_tensor_meta *audio_soft_meta,
    uint32_t audio_tokens,
    cudaStream_t stream
);

/* Compact the append-only KV to canonical prefix + most recent audio tokens.
 * retained_audio_tokens must be a positive multiple of Q and at most 256.
 * Audio K RoPE is shifted to the new local positions; omitted full-attention
 * entries are intentionally truncated. Calls remain stream-affine. */
int gemma4_listener_rebase_async(
    gemma4_listener_ctx *ctx,
    uint32_t retained_audio_tokens,
    cudaStream_t stream
);

/* Debug execution of first num_layers layers (writes ctx->d_hidden_a directly to output_tensor if !apply_final_norm) */
int gemma4_listener_step_debug(
    gemma4_listener_ctx *ctx,
    const void *frame_audio_soft,
    const gemma4_tensor_meta *frame_audio_soft_meta,
    uint32_t num_layers,
    bool apply_final_norm,
    cudaStream_t stream
);

/* Diagnostic-only trace of the current custom RMSNorm kernel.  Input/output
 * are contiguous CUDA BF16 [1, rows, dim], weight is CUDA BF16 [dim].  The
 * FP32 trace buffers are device arrays [rows, dim] or [rows], owned by the
 * caller and used only for the duration of the asynchronous operation. */
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
);

/* Diagnostic staged native-exact RMSNorm. Float buffers are CUDA tensors with
 * shapes [rows, dim] for squared/normalized/weighted and [rows] for the
 * scalar boundaries. */
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
);

/* Diagnostic direct FFN primitive entrypoints.  All tensors are contiguous
 * CUDA BF16 [1, rows, dim] except a linear weight [out_dim, in_dim]. */
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
);

int gemma4_listener_debug_gelu_tanh(
    gemma4_listener_ctx *ctx,
    const void *input,
    const gemma4_tensor_meta *input_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    uint32_t rows,
    uint32_t dim,
    cudaStream_t stream
);

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
);


/* Returns pointer to stable device output tensor [1, 4, hidden_size] BF16 */
const void *gemma4_listener_output_ptr(
    gemma4_listener_ctx *ctx
);

/* Returns pointer to named internal device scratch buffer for debugging */
const void *gemma4_listener_debug_ptr(
    gemma4_listener_ctx *ctx,
    const char *name
);


/* Reset KV cache and state to frame 0 without destroying preallocated memory */
int gemma4_listener_reset(
    gemma4_listener_ctx *ctx
);

/* Destroy context and free preallocated arena */
void gemma4_listener_destroy(
    gemma4_listener_ctx *ctx
);

/* Manifest & ABI validation */
const char *gemma4_listener_manifest_sha256(void);
int gemma4_listener_abi_version(void);
size_t gemma4_listener_abi_config_size(void);
size_t gemma4_listener_abi_tensor_meta_size(void);
size_t gemma4_listener_abi_layer_weights_size(void);
size_t gemma4_listener_abi_weights_size(void);
size_t gemma4_listener_abi_imported_kv_size(void);
size_t gemma4_listener_abi_imported_cache_size(void);
/* Returns the number of field offsets and optionally fills a host array. */
size_t gemma4_listener_abi_config_offsets(size_t *out_offsets, size_t capacity);
size_t gemma4_listener_abi_tensor_meta_offsets(size_t *out_offsets, size_t capacity);
size_t gemma4_listener_abi_layer_weights_offsets(size_t *out_offsets, size_t capacity);
size_t gemma4_listener_abi_weights_offsets(size_t *out_offsets, size_t capacity);
size_t gemma4_listener_abi_imported_kv_offsets(size_t *out_offsets, size_t capacity);
size_t gemma4_listener_abi_imported_cache_offsets(size_t *out_offsets, size_t capacity);
enum {
    GEMMA4_ATTN_CUSTOM = 0,
    GEMMA4_ATTN_ATEN_REPEAT_EXACT = 1,
    GEMMA4_ATTN_ATEN_GQA_EXACT = 2,
    GEMMA4_ATTN_NATIVE_SPLITK_GQA = 3,
    GEMMA4_ATTN_ATEN_EXACT = GEMMA4_ATTN_ATEN_REPEAT_EXACT, /* legacy source name */
};

enum {
    GEMMA4_RMS_CUSTOM_FAST = 0,
    GEMMA4_RMS_ATEN_EXACT = 1, /* golden oracle only */
    GEMMA4_RMS_NATIVE_EXACT = 2,
};

uint32_t gemma4_listener_runtime_attention_backend(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_rmsnorm_backend(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_max_seq_len(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_current_position(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_prefix_len(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_prefix_capacity(void);
uint64_t gemma4_listener_runtime_splitk_layer_mask(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_calls_by_layer(
    const gemma4_listener_ctx *ctx, uint32_t layer_idx);
size_t gemma4_listener_runtime_derived_weight_bytes(const gemma4_listener_ctx *ctx);
const char *gemma4_listener_runtime_attention_implementation(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_ordered_key_tile_size(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_runtime_ordered_min_keys_exclusive(const gemma4_listener_ctx *ctx);
size_t gemma4_listener_runtime_ordered_workspace_bytes(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_pack_launches(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_stage_a_launches(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_stage_b_launches(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_stage_c_launches(const gemma4_listener_ctx *ctx);
uint64_t gemma4_listener_runtime_ordered_stage_d_launches(const gemma4_listener_ctx *ctx);
uint32_t gemma4_listener_current_position(const gemma4_listener_ctx *ctx);
const void *gemma4_listener_get_cached_k(const gemma4_listener_ctx *ctx, int layer_idx);
const void *gemma4_listener_get_cached_v(const gemma4_listener_ctx *ctx, int layer_idx);
/* Explicit lifecycle/diagnostic check; synchronizes last_stream by contract. */
int gemma4_listener_output_health(gemma4_listener_ctx *ctx, int *out_nonfinite);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_LISTENER_H */
