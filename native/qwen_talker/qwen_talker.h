/*
 * qwen_talker.h — native incremental Qwen3-TTS Talker backbone ABI.
 *
 * The executor consumes two already-built BF16 embeddings per speaker frame
 * (prefix, audio), preserves the Q=2 joint traversal, and owns a fixed
 * position-addressable KV arena.  It does not own codec embeddings, sampling,
 * SubTalker, or Code2Wav.
 */
#ifndef QWEN_TALKER_H
#define QWEN_TALKER_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stddef.h>
#include "generated/qwen_native_topology.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen_talker_ctx qwen_talker_ctx;

#define QWEN_NATIVE_META_MAX_RANK 4u
#define QWEN_NATIVE_GLOBAL_WEIGHT_META_COUNT 2u
#define QWEN_NATIVE_LAYER_WEIGHT_META_COUNT 11u
#define QWEN_NATIVE_WEIGHT_META_COUNT \
    (QWEN_NATIVE_GLOBAL_WEIGHT_META_COUNT + QWEN_NATIVE_NUM_LAYERS * QWEN_NATIVE_LAYER_WEIGHT_META_COUNT)
#define QWEN_NATIVE_DEBUG_TAP_COUNT 18u

typedef enum {
    QWEN_NATIVE_TAP_INPUT_NORM = 0,
    QWEN_NATIVE_TAP_Q_PROJ = 1,
    QWEN_NATIVE_TAP_K_PROJ = 2,
    QWEN_NATIVE_TAP_V_PROJ = 3,
    QWEN_NATIVE_TAP_Q_NORM = 4,
    QWEN_NATIVE_TAP_K_NORM = 5,
    QWEN_NATIVE_TAP_Q_ROPE = 6,
    QWEN_NATIVE_TAP_K_ROPE = 7,
    QWEN_NATIVE_TAP_V_HEADS = 8,
    QWEN_NATIVE_TAP_ATTENTION = 9,
    QWEN_NATIVE_TAP_O_PROJ = 10,
    QWEN_NATIVE_TAP_ATTENTION_RESIDUAL = 11,
    QWEN_NATIVE_TAP_FF_NORM = 12,
    QWEN_NATIVE_TAP_GATE = 13,
    QWEN_NATIVE_TAP_UP = 14,
    QWEN_NATIVE_TAP_ACTIVATION = 15,
    QWEN_NATIVE_TAP_DOWN = 16,
    QWEN_NATIVE_TAP_LAYER_OUTPUT = 17,
} qwen_talker_debug_tap;

typedef enum {
    QWEN_NATIVE_META_BF16 = 1,
    QWEN_NATIVE_META_FP32 = 3,
    QWEN_NATIVE_META_INT64 = 4,
} qwen_tensor_dtype;

/* Host-side descriptors are consumed synchronously by the ABI call.  The
 * capacity is part of the fail-closed raw-pointer contract. */
typedef struct {
    const void *ptr;
    uint32_t dtype;
    uint32_t ndim;
    uint32_t contiguous;
    uint32_t reserved;
    uint64_t dims[QWEN_NATIVE_META_MAX_RANK];
    uint64_t strides[QWEN_NATIVE_META_MAX_RANK];
    uint64_t capacity_bytes;
} qwen_tensor_meta;

typedef struct {
    const void *input_norm_w;
    const void *post_attention_norm_w;
    const void *q_proj_w;
    const void *k_proj_w;
    const void *v_proj_w;
    const void *o_proj_w;
    const void *q_norm_w;
    const void *k_norm_w;
    const void *gate_proj_w;
    const void *up_proj_w;
    const void *down_proj_w;
} qwen_talker_layer_weights;

typedef struct {
    const void *final_norm_w;
    const void *rotary_inv_freq;
    qwen_talker_layer_weights layers[QWEN_NATIVE_NUM_LAYERS];
} qwen_talker_weights;

typedef struct {
    uint32_t max_cache_len;
    uint32_t execute_layers; /* 1..QWEN_NATIVE_NUM_LAYERS; diagnostic layer gate may use 1. */
    const qwen_tensor_meta *weight_meta;
    uint32_t weight_meta_count;
} qwen_talker_config;

typedef struct {
    const void *k_ptr;
    const qwen_tensor_meta *k_meta;
    const void *v_ptr;
    const qwen_tensor_meta *v_meta;
} qwen_talker_imported_kv;

typedef struct {
    qwen_talker_imported_kv layers[QWEN_NATIVE_NUM_LAYERS];
} qwen_talker_imported_cache;

/* Native-produced trace metadata.  Tensor payloads are copied through the
 * trace debug APIs below; positions/counters are read from this struct. */
#define QWEN_NATIVE_TRACE_CHAIN_BYTES 32u

typedef struct {
    uint64_t step_index;
    uint64_t trace_generation;
    uint64_t expected_step_count;
    uint64_t completed_step_count;
    uint32_t stable_length;
    uint32_t prefix_start;
    uint32_t audio_start;
    uint32_t prefix_position;
    uint32_t audio_position;
    uint32_t cache_layers_written;
    uint32_t trace_enabled;
    uint32_t native_complete;
    uint32_t reserved;
    uint64_t joint_steps;
    uint64_t cache_imports;
    int32_t cuda_error_state;
    uint8_t native_previous_chain[QWEN_NATIVE_TRACE_CHAIN_BYTES];
    uint8_t native_chain[QWEN_NATIVE_TRACE_CHAIN_BYTES];
} qwen_talker_trace_step;

int qwen_talker_create(
    const qwen_talker_config *config,
    const qwen_talker_weights *weights,
    qwen_talker_ctx **out_ctx
);

int qwen_talker_import_cache_async(
    qwen_talker_ctx *ctx,
    const qwen_talker_imported_cache *cache,
    uint32_t stable_length,
    uint32_t prefix_start,
    uint32_t audio_start,
    cudaStream_t stream
);

/* Import an eager HF DynamicCache prefill as immutable prefix KV for a new
 * audio-only Q1 response. Each source layer is exact BF16 [1, KVH, S, D]. */
int qwen_talker_import_audio_cache_async(
    qwen_talker_ctx *ctx,
    const qwen_talker_imported_cache *cache,
    uint32_t prefill_length,
    uint32_t max_audio_frames,
    cudaStream_t stream
);

/* Full backbone step.  Prefix and audio are [1,1,H] contiguous CUDA BF16.
 * After import, positions must advance contiguously as
 * (prefix_start+t, audio_start+t); the call is fail-closed on gaps/rewrites. */
int qwen_talker_step_joint_async(
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
);

/* One terminal audio-only query after at least one JOINT Q=2 acoustic step.
 * It writes exactly one audio KV row at the next audio position and reuses,
 * but never rewrites, the last visible prefix row. */
int qwen_talker_step_terminal_async(
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
);

/* Recurrent audio-only Q1 step. The native executor assigns the next position
 * beginning at prefill_length and advances it once per successful call. */
int qwen_talker_step_audio_async(
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
);

/* Whole-step plain-Q1 graph control. The graph itself is captured/replayed by
 * torch.cuda.graph; these calls bind the slot stream and keep host state in
 * lockstep with the device cursor without synchronizing the GPU. */
int qwen_talker_q1_graph_bind_stream(qwen_talker_ctx *ctx, cudaStream_t stream);
int qwen_talker_q1_graph_capture_begin(qwen_talker_ctx *ctx);
int qwen_talker_q1_graph_capture_end(qwen_talker_ctx *ctx, uint32_t success);
int qwen_talker_q1_graph_can_replay(qwen_talker_ctx *ctx);
int qwen_talker_q1_graph_replay_commit(qwen_talker_ctx *ctx);
int qwen_talker_q1_graph_invalidate(qwen_talker_ctx *ctx);
int qwen_talker_q1_graph_snapshot_device_state(
    qwen_talker_ctx *ctx, void *device_output_u32x2, cudaStream_t stream);
int32_t qwen_talker_runtime_q1_device_cursor_supported(const qwen_talker_ctx *ctx);
int32_t qwen_talker_runtime_q1_graph_captured(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_q1_graph_replays(const qwen_talker_ctx *ctx);

/* Layer-local gate: executes exactly one decoder layer and returns [1,2,H]
 * before the final model norm.  This is diagnostic and does not replace the
 * full step API. */
int qwen_talker_debug_execute_layer_async(
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
);

int qwen_talker_debug_copy_layer_cache_async(
    qwen_talker_ctx *ctx,
    uint32_t layer_idx,
    void *key_cache_bf16,
    const qwen_tensor_meta *key_cache_meta,
    void *value_cache_bf16,
    const qwen_tensor_meta *value_cache_meta,
    cudaStream_t stream
);

/* Copies only the two rows written by the current JOINT Q=2 step, laid out as
 * [2, num_kv_heads, head_dim].  This is diagnostic; production never copies
 * cache data back to Python. */
int qwen_talker_debug_copy_layer_cache_positions_async(
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
);

int qwen_talker_debug_copy_tap_async(
    qwen_talker_ctx *ctx,
    uint32_t tap_id,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    uint32_t elements,
    cudaStream_t stream
);

int qwen_talker_debug_copy_input_rms_mean_async(
    qwen_talker_ctx *ctx,
    void *destination_fp32,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
);

/* Enable the native-produced trace.  Disabled by default so production
 * serving keeps the existing zero-copy path. */
int qwen_talker_trace_set_enabled(qwen_talker_ctx *ctx, uint32_t enabled);
int qwen_talker_trace_set_expected_steps(qwen_talker_ctx *ctx, uint64_t expected_steps);
int qwen_talker_trace_copy_effective_mask_async(
    qwen_talker_ctx *ctx,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
);
int qwen_talker_trace_copy_inputs_async(
    qwen_talker_ctx *ctx,
    void *prefix_destination_bf16,
    const qwen_tensor_meta *prefix_destination_meta,
    void *audio_destination_bf16,
    const qwen_tensor_meta *audio_destination_meta,
    cudaStream_t stream
);
int qwen_talker_trace_copy_position_ids_async(
    qwen_talker_ctx *ctx,
    void *destination_int64,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
);
int qwen_talker_trace_copy_output_async(
    qwen_talker_ctx *ctx,
    void *destination_bf16,
    const qwen_tensor_meta *destination_meta,
    cudaStream_t stream
);
int qwen_talker_trace_copy_cache_rows_async(
    qwen_talker_ctx *ctx,
    void *key_destination_bf16,
    const qwen_tensor_meta *key_destination_meta,
    void *value_destination_bf16,
    const qwen_tensor_meta *value_destination_meta,
    cudaStream_t stream
);
int qwen_talker_trace_last_step(
    const qwen_talker_ctx *ctx,
    qwen_talker_trace_step *destination
);
uint32_t qwen_talker_trace_abi_version(void);

void qwen_talker_destroy(qwen_talker_ctx *ctx);

uint32_t qwen_talker_abi_version(void);
uint64_t qwen_talker_runtime_workspace_bytes(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_joint_steps(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_terminal_query_count(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_terminal_prefix_cache_appends(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_terminal_audio_cache_appends(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_terminal_query_position(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_terminal_visible_prefix_position(const qwen_talker_ctx *ctx);
uint64_t qwen_talker_runtime_cache_imports(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_execute_layers(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_current_prefix_position(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_current_audio_position(const qwen_talker_ctx *ctx);
uint32_t qwen_talker_runtime_max_cache_len(const qwen_talker_ctx *ctx);
int32_t qwen_talker_runtime_last_error(const qwen_talker_ctx *ctx);
int32_t qwen_talker_runtime_attention_graph_captured(const qwen_talker_ctx *ctx);
int32_t qwen_talker_runtime_rmsnorm_graph_captured(const qwen_talker_ctx *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TALKER_H */
