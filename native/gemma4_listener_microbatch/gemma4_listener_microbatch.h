#ifndef GEMMA4_LISTENER_MICROBATCH_H
#define GEMMA4_LISTENER_MICROBATCH_H

#include <stddef.h>
#include "gemma4_listener.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEMMA4_LISTENER_BATCH_API_VERSION 5u
#define GEMMA4_LISTENER_BATCH_PROBE_API_VERSION 1u
#define GEMMA4_LISTENER_BATCH_TRACE_API_VERSION 1u
#define GEMMA4_BATCH_SUPPORTED_MASK_ANY_POSITIVE_N 0x1u

enum {
    GEMMA4_BATCH_TRACE_GEMM_WRAPPER_NONE = 0u,
    GEMMA4_BATCH_TRACE_GEMM_WRAPPER_LEGACY_CUBLAS = 1u,
    GEMMA4_BATCH_TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM = 2u
};

typedef struct gemma4_listener_batch_ctx gemma4_listener_batch_ctx;

/* Numeric identity is allocated by the Python adapter. Textual session IDs,
 * hashes, and runtime signatures remain in Python and are checked there. */
typedef struct {
    uint64_t native_context_id;
    uint64_t session_generation;
    uint64_t state_epoch_id;
    uint64_t frame_id;
    uint64_t listener_state_version;
    uint64_t sequence_position;
    uint32_t cache_position;
    uint32_t valid_history_tokens;
} gemma4_listener_batch_row;

typedef struct {
    uint32_t struct_size;
    uint32_t api_version;
    uint32_t max_batch_size;       /* positive initial capacity hint; grow with ensure_capacity */
    uint32_t attention_backend;    /* currently requires NATIVE_SPLITK_GQA (3) */
    uint32_t rmsnorm_backend;      /* same IDs as ABI3 */
    int32_t listener_stream_priority; /* primary -1; lower-priority shadow owner 0 */
    uint32_t reserved[2];
} gemma4_listener_batch_config;

typedef struct {
    uint32_t struct_size;
    uint32_t native_abi_version;
    uint32_t batch_api_version;
    uint32_t max_batch_size;       /* currently allocated capacity */
    uint32_t supported_batch_mask; /* GEMMA4_BATCH_SUPPORTED_MASK_ANY_POSITIVE_N */
    uint32_t attention_backend_id;
    uint32_t rmsnorm_backend_id;
    int32_t listener_stream_priority;
    uint32_t max_seq_len;
    uint32_t stream_q_len;
    uint32_t stream_priority_verified;
} gemma4_listener_batch_runtime_info;

typedef struct {
    uint64_t forward_calls;
    uint64_t layers_executed;
    uint64_t projection_gemm_calls;
    uint64_t projection_rows;
    uint64_t session_rows;
    uint64_t output_generation;
    uint32_t batch_api_version;
    uint32_t max_batch_size;
} gemma4_listener_batch_counters;

/* Read-only host state of one private batch slot.  KV payloads are copied by a
 * separate async API on the bound Listener stream. */
typedef struct {
    uint32_t struct_size;
    uint32_t slot;
    uint64_t native_context_id;
    uint64_t session_generation;
    uint64_t state_epoch_id;
    uint64_t expected_frame_id;
    uint64_t expected_state_version;
    uint64_t expected_sequence_position;
    uint32_t prefix_len;
    uint32_t current_position;
    uint32_t valid_kv_len;
    uint32_t max_seq_len;
    uint32_t primed;
    uint32_t reserved[3];
} gemma4_listener_batch_session_state;

/* Proof-only N=2 numerical-boundary capture. The API is opt-in, one-shot, and
 * host-blocking only when the caller explicitly requests the raw arrays. It
 * never changes ABI3 or the ordinary batch forward/counter path. */
typedef struct {
    uint32_t struct_size;
    uint32_t state;
    uint32_t row_count;
    uint32_t rms_dim;
    uint32_t rms_backend_id;
    uint32_t rms_batched_block_x;
    uint32_t rms_batched_block_y;
    uint32_t rms_reference_block_x;
    uint32_t rms_reference_block_y;
    uint32_t q_dim;
    uint32_t q_batched_m;
    uint32_t q_split_m;
    uint32_t q_k;
    uint32_t q_lda;
    uint32_t q_ldb;
    uint32_t q_ldc;
    uint32_t q4_gemv4_opt_in;
    uint32_t q4_gemv4_split0_eligible;
    uint32_t q4_gemv4_split1_eligible;
    uint32_t cublas_math_mode;
    uint32_t cublas_atomics_mode;
    uint32_t cublas_pointer_mode;
    uint32_t cublas_stream_matches_listener;
    uint64_t forward_generation;
    uint64_t rms_weight_device_ptr;
    uint64_t q_weight_device_ptr;
    uint64_t listener_stream_handle;
    float rms_eps;
} gemma4_listener_batch_probe_info;

/* Optional proof-only operator trace. The fixed stage ID inventory lives in
 * gemma4_listener_batch_contract.h. Input/actual/reference arrays are raw
 * contiguous BF16 values. Attention input1/input2 are compact concatenations
 * of the two SID-specific visible K/V windows; their per-SID lengths, starts,
 * prefixes, and cursors are included below. */
typedef struct {
    uint32_t struct_size;
    uint32_t stage_id;
    uint32_t kind;
    uint32_t flags;
    uint32_t layer_index;
    uint32_t q_len;
    uint32_t row_count;
    uint32_t reference_rows_per_sid;
    uint32_t input0_rows;
    uint32_t input0_cols;
    uint32_t input1_rows;
    uint32_t input1_cols;
    uint32_t input2_rows;
    uint32_t input2_cols;
    uint32_t actual_rows;
    uint32_t actual_cols;
    uint32_t reference_rows;
    uint32_t reference_cols;
    uint32_t rms_backend_id;
    uint32_t attention_backend_id;
    uint32_t attention_is_sliding;
    uint32_t sliding_window;
    uint32_t attention_mask_causal;
    uint32_t gemm_m;
    uint32_t gemm_n;
    uint32_t gemm_k;
    uint32_t gemm_lda;
    uint32_t gemm_ldb;
    uint32_t gemm_ldc;
    uint32_t cublas_math_mode;
    uint32_t cublas_atomics_mode;
    uint32_t cublas_pointer_mode;
    uint32_t cublas_stream_matches_listener;
    uint32_t launch_block_x;
    uint32_t launch_block_y;
    uint32_t launch_block_z;
    uint32_t launch_grid_x;
    uint32_t gemm_actual_wrapper;
    uint32_t gemm_reference_wrapper;
    uint32_t q4_gemv4_opt_in;
    uint32_t q4_gemv4_split0_eligible;
    uint32_t q4_gemv4_split1_eligible;
    uint32_t q4_gemv4_split0_selected;
    uint32_t q4_gemv4_split1_selected;
    uint32_t cache_start[2];
    uint32_t visible_len[2];
    uint32_t prefix_len[2];
    uint32_t base_position[2];
    uint64_t input0_elements;
    uint64_t input1_elements;
    uint64_t input2_elements;
    uint64_t actual_elements;
    uint64_t reference_elements;
    uint64_t weight_device_ptr;
    uint64_t forward_generation;
    float epsilon;
    float scalar;
} gemma4_listener_batch_trace_info;

enum {
    GEMMA4_BATCH_TRACE_FLAG_INPUTS_CAPTURED = 1u,
    GEMMA4_BATCH_TRACE_FLAG_ACTUAL_CAPTURED = 2u,
    GEMMA4_BATCH_TRACE_FLAG_REFERENCE_CAPTURED = 4u,
    GEMMA4_BATCH_TRACE_FLAG_REFERENCE_DEFERRED = 8u,
    GEMMA4_BATCH_TRACE_FLAG_INPLACE = 16u,
    GEMMA4_BATCH_TRACE_FLAG_ATTENTION_OUTPUT_ROW0 = 1u << 8,
    GEMMA4_BATCH_TRACE_FLAG_ATTENTION_OUTPUT_ROW1 = 1u << 9
};

uint32_t gemma4_listener_batch_api_version(void);
uint64_t gemma4_listener_batch_capabilities(void);
int gemma4_listener_batch_create(
    const gemma4_listener_config *listener_config,
    const gemma4_listener_weights *weights,
    const gemma4_listener_batch_config *batch_config,
    gemma4_listener_batch_ctx **out_batch);
/* Grow at an owner-thread tick boundary with no outstanding output lease.
 * Existing session slots/KV state remain bound and available after success. */
int gemma4_listener_batch_ensure_capacity(
    gemma4_listener_batch_ctx *batch,
    uint32_t requested_capacity,
    uint32_t *out_capacity);
int gemma4_listener_batch_destroy(gemma4_listener_batch_ctx *batch);
int gemma4_listener_batch_bind_stream(
    gemma4_listener_batch_ctx *batch,
    cudaStream_t listener_stream);
/* Optional startup-only GEMM graphs. Each stable M=4 projection operation
 * has one graph per geometric lane-width bucket (1, 2, 4, ... through the
 * capacity); dispatch launches the smallest bucket covering the active rows.
 * Capacity growth prepares all replacement buckets before commit; a failed
 * growth keeps the old capacity and graph cache usable. */
int gemma4_listener_batch_prepare_gemm_graphs(
    gemma4_listener_batch_ctx *batch,
    uint64_t *out_graph_count);
uint64_t gemma4_listener_batch_gemm_graph_count(
    const gemma4_listener_batch_ctx *batch);
int gemma4_listener_batch_import_cache_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    const gemma4_imported_cache *cache,
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    cudaStream_t listener_stream);
int gemma4_listener_batch_step_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *audio_soft, /* contiguous BF16 [N,2,1536] */
    const gemma4_tensor_meta *audio_soft_meta,
    cudaStream_t listener_stream);
/* Replay previously queued [2,1536] audio soft tokens through native audio
 * PLE/KV after a model turn. Cache advances by Q2 while listener frame,
 * state-version, and sequence cursors remain unchanged because the original
 * ephemeral Q0 tick already accounted for this audio frame. */
int gemma4_listener_batch_step_queued_audio_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *audio_soft,
    const gemma4_tensor_meta *audio_soft_meta,
    cudaStream_t listener_stream);
/* Run a token-text Q2 step using model-produced embeddings and complete
 * projected PLE. Tensors are contiguous BF16 [N,2,1536] and
 * [N,2,35,256] respectively; final norm remains part of the native step. */
int gemma4_listener_batch_step_embedded_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *inputs_embeds,
    const gemma4_tensor_meta *inputs_embeds_meta,
    const void *per_layer_inputs,
    const gemma4_tensor_meta *per_layer_inputs_meta,
    cudaStream_t listener_stream);
/* Commit only the logical prefix of the immediately preceding physical-Q2
 * append. The caller supplies that append's pre-step row identity. Frame,
 * state-version, and sequence cursors stay advanced; only KV visibility and
 * cache position are truncated. Must be called by the same owner before it
 * publishes a replacement cursor. */
int gemma4_listener_batch_truncate_after_step_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *pre_step_row,
    uint32_t logical_q_len,
    cudaStream_t listener_stream);
/* Rewind an idle native session's visible KV tail after a cancelled assistant
 * branch. The row must describe the exact current session cursor and target
 * must remain at or beyond the imported true-chat prefix. */
int gemma4_listener_batch_rewind_session_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *current_row,
    uint32_t target_position,
    cudaStream_t listener_stream);
int gemma4_listener_batch_output_acquire(
    gemma4_listener_batch_ctx *batch,
    uint64_t *out_generation,
    const void **out_tensor, /* borrowed [N,2,1536], valid until release */
    const gemma4_listener_batch_row **out_rows, /* ordered row provenance */
    uint32_t *out_row_count,
    cudaEvent_t *out_completion_event);
int gemma4_listener_batch_output_clone_async(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    uint32_t row_count,
    void *clone_tensor, /* caller-owned contiguous BF16 [N,2,1536] */
    const gemma4_tensor_meta *clone_meta,
    cudaStream_t listener_stream,
    cudaEvent_t *out_clone_event);
int gemma4_listener_batch_output_release(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    cudaEvent_t consumer_done_event,
    cudaStream_t listener_stream);
/* One owner-thread aggregate finite check for the latest released output.
 * Rows must exactly match the native generation's ordered input provenance.
 * Returns 0 for finite output, 1 when the batch-wide guard found nonfinite
 * values, and a negative code for stale/malformed provenance or CUDA failure. */
int gemma4_listener_batch_output_health(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    const gemma4_listener_batch_row *rows,
    const uint32_t *logical_q_lens,
    uint32_t row_count,
    int *out_nonfinite);
/* Private-owner cleanup before any borrowed pointer escaped. Queues a wait on
 * compute completion, or clone completion if a private copy was enqueued. */
int gemma4_listener_batch_output_abandon_async(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    cudaStream_t listener_stream);
int gemma4_listener_batch_release_session_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    cudaStream_t listener_stream);
int gemma4_listener_batch_rollover_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    uint32_t retained_audio_tokens,
    cudaStream_t listener_stream);
int gemma4_listener_batch_get_counters(
    const gemma4_listener_batch_ctx *batch,
    gemma4_listener_batch_counters *out_counters);
int gemma4_listener_batch_get_runtime_info(
    const gemma4_listener_batch_ctx *batch,
    gemma4_listener_batch_runtime_info *out_info);
int gemma4_listener_batch_get_session_state(
    const gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    gemma4_listener_batch_session_state *out_state);
/* Copies one non-shared source layer's valid K/V rows to caller-owned
 * contiguous CUDA BF16 [1,1,current_position,head_dim] tensors. The caller
 * records a CUDA event on listener_stream after all requested layer copies. */
int gemma4_listener_batch_copy_session_kv_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    uint32_t layer_idx,
    void *key_tensor,
    const gemma4_tensor_meta *key_meta,
    void *value_tensor,
    const gemma4_tensor_meta *value_meta,
    cudaStream_t listener_stream);
/* Promote one valid source cache snapshot into a distinct destination owner.
 * The caller first orders source_ready_event before listener_stream. Session
 * generation/epoch/frame/state/sequence must match; native context identity
 * remains destination-owned while its prefix and cache cursor are replaced. */
int gemma4_listener_batch_copy_session_cache_async(
    gemma4_listener_batch_ctx *destination,
    uint32_t destination_slot,
    const gemma4_listener_batch_row *destination_row,
    gemma4_listener_batch_ctx *source,
    uint32_t source_slot,
    const gemma4_listener_batch_row *source_row,
    cudaStream_t listener_stream);

/* Clone one completed private session into an unprimed slot of the same
 * owner/context. Source and destination rows must carry the same logical
 * generation/epoch/frame/state/sequence and exact valid-KV cursor; only the
 * destination native-context identity differs. K/V and semantic-prefix state
 * are copied on listener_stream before the new slot is published. */
int gemma4_listener_batch_clone_session_slot_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t destination_slot,
    const gemma4_listener_batch_row *destination_row,
    uint32_t source_slot,
    const gemma4_listener_batch_row *source_row,
    cudaStream_t listener_stream);

/* Optional capability bit 7 advertises this proof-only API. Every RMS buffer
 * carries all four rows (N=2 times Q=2), contiguous BF16 [4,1536]: X is the
 * captured pre-RMS input, batched is the production N=2 RMS output, and split
 * concatenates two reference RMS calls of [2,1536] each. Every q_proj buffer
 * likewise carries all four rows: input is the captured RMS output [4,1536],
 * actual is the production q_proj capture [4,q_dim], batched is an explicit
 * replay through the same GEMM helper at M=4, and split concatenates two M=2
 * helper calls. These are proof buffers only; reads wait only when explicitly
 * invoked and do not change the ABI3 or ordinary batch forward path. */
int gemma4_listener_batch_probe_arm(gemma4_listener_batch_ctx *batch);
uint32_t gemma4_listener_batch_probe_api_version(void);
int gemma4_listener_batch_probe_read_rms(
    gemma4_listener_batch_ctx *batch,
    void *input_host_u16, size_t input_bytes,
    void *batched_host_u16, size_t batched_bytes,
    void *split_host_u16, size_t split_bytes,
    gemma4_listener_batch_probe_info *out_info);
int gemma4_listener_batch_probe_mark_rms_exact(
    gemma4_listener_batch_ctx *batch,
    uint32_t exact_match);
int gemma4_listener_batch_probe_qproj_async(
    gemma4_listener_batch_ctx *batch,
    cudaStream_t listener_stream);
int gemma4_listener_batch_probe_read_qproj(
    gemma4_listener_batch_ctx *batch,
    void *input_host_u16, size_t input_bytes,
    void *actual_host_u16, size_t actual_bytes,
    void *batched_host_u16, size_t batched_bytes,
    void *split_host_u16, size_t split_bytes,
    gemma4_listener_batch_probe_info *out_info);

/* Optional capability bit 8. The trace is one-shot, N=2, layer-0 plus the
 * audio-processor PLE prelude, and is allocated only after explicit arm. */
uint32_t gemma4_listener_batch_trace_api_version(void);
uint32_t gemma4_listener_batch_trace_stage_count(void);
int gemma4_listener_batch_trace_arm(gemma4_listener_batch_ctx *batch);
int gemma4_listener_batch_trace_get_stage_info(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    gemma4_listener_batch_trace_info *out_info);
int gemma4_listener_batch_trace_read_stage(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    void *input0_host_u16, size_t input0_bytes,
    void *input1_host_u16, size_t input1_bytes,
    void *input2_host_u16, size_t input2_bytes,
    void *actual_host_u16, size_t actual_bytes,
    void *reference_host_u16, size_t reference_bytes);

#ifdef __cplusplus
}
#endif
#endif
