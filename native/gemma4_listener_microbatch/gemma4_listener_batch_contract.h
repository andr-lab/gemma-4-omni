#ifndef GEMMA4_LISTENER_BATCH_CONTRACT_H
#define GEMMA4_LISTENER_BATCH_CONTRACT_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define GEMMA4_BATCH_PROBE_HIDDEN_DIM 1536u
#define GEMMA4_BATCH_PROBE_Q_ROWS 8u
#define GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS 4u
#define GEMMA4_BATCH_TRACE_MAX_BYTES (100u * 1024u * 1024u)
#define GEMMA4_BATCH_TRACE_Q_LEN 4u
#define GEMMA4_BATCH_TRACE_HEADS 8u
#define GEMMA4_BATCH_TRACE_LAYERS 35u
#define GEMMA4_BATCH_TRACE_PLE_DIM 256u

typedef enum {
    GEMMA4_BATCH_TRACE_AUDIO_PLE_GEMM = 0,
    GEMMA4_BATCH_TRACE_AUDIO_PLE_SCALE = 1,
    GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS = 2,
    GEMMA4_BATCH_TRACE_AUDIO_PLE_FRAME_PAD = 3,
    GEMMA4_BATCH_TRACE_AUDIO_PLE_PAD_RESIDUAL = 4,
    GEMMA4_BATCH_TRACE_AUDIO_PLE_INPUT_SCALE = 5,
    GEMMA4_BATCH_TRACE_L0_INPUT_NORM = 6,
    GEMMA4_BATCH_TRACE_L0_Q_PROJ = 7,
    GEMMA4_BATCH_TRACE_L0_Q_NORM = 8,
    GEMMA4_BATCH_TRACE_L0_K_PROJ = 9,
    GEMMA4_BATCH_TRACE_L0_V_PROJ = 10,
    GEMMA4_BATCH_TRACE_L0_K_NORM = 11,
    GEMMA4_BATCH_TRACE_L0_V_NORM = 12,
    GEMMA4_BATCH_TRACE_L0_ATTENTION = 13,
    GEMMA4_BATCH_TRACE_L0_O_PROJ = 14,
    GEMMA4_BATCH_TRACE_L0_POST_ATTN_NORM = 15,
    GEMMA4_BATCH_TRACE_L0_POST_ATTN_RESIDUAL = 16,
    GEMMA4_BATCH_TRACE_L0_PRE_FF_NORM = 17,
    GEMMA4_BATCH_TRACE_L0_GATE_PROJ = 18,
    GEMMA4_BATCH_TRACE_L0_UP_PROJ = 19,
    GEMMA4_BATCH_TRACE_L0_GATE_GELU = 20,
    GEMMA4_BATCH_TRACE_L0_GATE_UP_MUL = 21,
    GEMMA4_BATCH_TRACE_L0_DOWN_PROJ = 22,
    GEMMA4_BATCH_TRACE_L0_POST_FF_NORM = 23,
    GEMMA4_BATCH_TRACE_L0_POST_FF_RESIDUAL = 24,
    GEMMA4_BATCH_TRACE_L0_PLE_GATE_PROJ = 25,
    GEMMA4_BATCH_TRACE_L0_PLE_GELU = 26,
    GEMMA4_BATCH_TRACE_L0_PLE_LAYER_MULT = 27,
    GEMMA4_BATCH_TRACE_L0_PLE_PROJ = 28,
    GEMMA4_BATCH_TRACE_L0_PLE_NORM = 29,
    GEMMA4_BATCH_TRACE_L0_PLE_RESIDUAL = 30,
    GEMMA4_BATCH_TRACE_L0_LAYER_SCALE = 31,
    GEMMA4_BATCH_TRACE_FINAL_LAYER_OUTPUT = 32,
    GEMMA4_BATCH_TRACE_STAGE_COUNT = 33
} gemma4_batch_trace_stage;

typedef enum {
    GEMMA4_BATCH_TRACE_KIND_GEMM = 1,
    GEMMA4_BATCH_TRACE_KIND_RMS = 2,
    GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE = 3,
    GEMMA4_BATCH_TRACE_KIND_ATTENTION = 4
} gemma4_batch_trace_kind;

static inline bool gemma4_batch_trace_rows_valid(uint32_t n) {
    return n == 2u;
}

static inline bool gemma4_batch_trace_mark_row(uint32_t *mask, uint32_t row) {
    if (!mask || row >= 2u || (*mask & (1u << row)) != 0u) return false;
    *mask |= (1u << row);
    return true;
}

static inline bool gemma4_batch_trace_rows_complete(uint32_t mask) {
    return mask == 3u;
}

static inline bool gemma4_batch_trace_reference_layout(
    gemma4_batch_trace_stage stage,
    uint32_t n,
    uint32_t *actual_rows,
    uint32_t *reference_rows,
    uint32_t *dim
) {
    if (!actual_rows || !reference_rows || !dim || !gemma4_batch_trace_rows_valid(n))
        return false;
    switch (stage) {
        case GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS:
            *actual_rows = n * GEMMA4_BATCH_TRACE_Q_LEN * GEMMA4_BATCH_TRACE_LAYERS;
            *reference_rows = GEMMA4_BATCH_TRACE_Q_LEN * GEMMA4_BATCH_TRACE_LAYERS;
            *dim = GEMMA4_BATCH_TRACE_PLE_DIM;
            return true;
        case GEMMA4_BATCH_TRACE_L0_Q_NORM:
            *actual_rows = n * GEMMA4_BATCH_TRACE_Q_LEN * GEMMA4_BATCH_TRACE_HEADS;
            *reference_rows = GEMMA4_BATCH_TRACE_Q_LEN * GEMMA4_BATCH_TRACE_HEADS;
            *dim = GEMMA4_BATCH_TRACE_PLE_DIM;
            return true;
        case GEMMA4_BATCH_TRACE_L0_INPUT_NORM:
        case GEMMA4_BATCH_TRACE_L0_POST_ATTN_NORM:
        case GEMMA4_BATCH_TRACE_L0_PRE_FF_NORM:
        case GEMMA4_BATCH_TRACE_L0_POST_FF_NORM:
        case GEMMA4_BATCH_TRACE_L0_PLE_NORM:
            *actual_rows = n * GEMMA4_BATCH_TRACE_Q_LEN;
            *reference_rows = GEMMA4_BATCH_TRACE_Q_LEN;
            *dim = GEMMA4_BATCH_PROBE_HIDDEN_DIM;
            return true;
        default:
            return false;
    }
}

static inline bool gemma4_batch_trace_storage_allowed(size_t bytes) {
    return bytes > 0 && bytes <= (size_t)GEMMA4_BATCH_TRACE_MAX_BYTES;
}

/* The immutable PAD PLE has exactly four distinct Q rows. The N=2 trace
 * reference reuses those same rows for each SID and the kernel indexes them
 * with token % Q_LEN, so the capture must retain all four source rows. */
static inline uint32_t gemma4_batch_trace_frame_pad_source_row(uint32_t token) {
    return token % GEMMA4_BATCH_TRACE_Q_LEN;
}

static inline bool gemma4_batch_trace_frame_pad_input_valid(
    uint32_t input_rows,
    uint32_t input_cols,
    uint64_t available_input_elements,
    uint32_t reference_rows_per_sid,
    uint32_t output_width
) {
    if (input_rows != GEMMA4_BATCH_TRACE_Q_LEN || !input_cols ||
        input_cols != output_width || !reference_rows_per_sid || !output_width) return false;
    const uint64_t expected_input_elements = (uint64_t)input_rows * input_cols;
    if (available_input_elements != expected_input_elements) return false;
    const uint32_t highest_source_row =
        gemma4_batch_trace_frame_pad_source_row(reference_rows_per_sid - 1u);
    const uint64_t required_input_elements =
        ((uint64_t)highest_source_row + 1u) * output_width;
    return highest_source_row < input_rows &&
        required_input_elements <= available_input_elements;
}

typedef enum {
    GEMMA4_BATCH_PROBE_IDLE = 0,
    GEMMA4_BATCH_PROBE_ARMED = 1,
    GEMMA4_BATCH_PROBE_RMS_CAPTURED = 2,
    GEMMA4_BATCH_PROBE_RMS_EXACT = 3,
    GEMMA4_BATCH_PROBE_RMS_DIFFERENT = 4,
    GEMMA4_BATCH_PROBE_QPROJ_CAPTURED = 5,
    GEMMA4_BATCH_PROBE_DONE = 6,
    GEMMA4_BATCH_PROBE_ERROR = 7
} gemma4_batch_probe_state;

static inline bool gemma4_batch_probe_can_arm(
    gemma4_batch_probe_state state,
    uint32_t max_batch,
    uint32_t rmsnorm_backend
) {
    /* This proof is intentionally restricted to the failing N=2 native-exact
     * path; it must not silently become a production fallback or N=4 pass. */
    return state == GEMMA4_BATCH_PROBE_IDLE && max_batch >= 2 &&
        rmsnorm_backend == 2u;
}

static inline bool gemma4_batch_probe_rows_valid(uint32_t row_count) {
    return row_count == 2u;
}

static inline bool gemma4_batch_probe_mark_rms_captured(
    gemma4_batch_probe_state *state,
    uint32_t n_rows,
    uint32_t hidden_dim
) {
    if (!state || *state != GEMMA4_BATCH_PROBE_ARMED ||
        n_rows != GEMMA4_BATCH_PROBE_Q_ROWS ||
        hidden_dim != GEMMA4_BATCH_PROBE_HIDDEN_DIM) return false;
    *state = GEMMA4_BATCH_PROBE_RMS_CAPTURED;
    return true;
}

static inline bool gemma4_batch_probe_mark_rms_result(
    gemma4_batch_probe_state *state,
    bool exact
) {
    if (!state || *state != GEMMA4_BATCH_PROBE_RMS_CAPTURED) return false;
    *state = exact ? GEMMA4_BATCH_PROBE_RMS_EXACT
                   : GEMMA4_BATCH_PROBE_RMS_DIFFERENT;
    return true;
}

static inline bool gemma4_batch_probe_can_run_qproj(
    gemma4_batch_probe_state state
) {
    return state == GEMMA4_BATCH_PROBE_RMS_EXACT;
}

static inline bool gemma4_batch_probe_mark_qproj_captured(
    gemma4_batch_probe_state *state
) {
    if (!state || *state != GEMMA4_BATCH_PROBE_RMS_EXACT) return false;
    *state = GEMMA4_BATCH_PROBE_QPROJ_CAPTURED;
    return true;
}

static inline bool gemma4_batch_probe_mark_done(
    gemma4_batch_probe_state *state
) {
    if (!state || *state != GEMMA4_BATCH_PROBE_QPROJ_CAPTURED) return false;
    *state = GEMMA4_BATCH_PROBE_DONE;
    return true;
}

static inline size_t gemma4_batch_probe_elements(uint32_t rows, uint32_t cols) {
    if (!rows || !cols || (size_t)rows > SIZE_MAX / (size_t)cols) return 0;
    return (size_t)rows * cols;
}

static inline bool gemma4_batch_probe_rms_launch_shape(
    uint32_t rows,
    uint32_t dim,
    uint32_t *block_x,
    uint32_t *block_y
) {
    if (!block_x || !block_y || dim != GEMMA4_BATCH_PROBE_HIDDEN_DIM ||
        (rows != GEMMA4_BATCH_PROBE_Q_ROWS &&
         rows != GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS)) return false;
    *block_x = rows == GEMMA4_BATCH_PROBE_Q_ROWS ? 64u : 128u;
    *block_y = rows == GEMMA4_BATCH_PROBE_Q_ROWS ? 8u : 4u;
    return true;
}

typedef struct {
    uint32_t prefix_position;
    uint32_t audio_tokens;
    uint32_t compute_tokens;
    uint32_t final_position;
    uint32_t maximum_visible_keys;
} gemma4_history_batch_plan;

static inline bool gemma4_history_batch_make_plan(
    uint32_t prefix_len,
    uint32_t current_position,
    uint32_t valid_kv_len,
    uint32_t audio_tokens,
    uint32_t max_seq_len,
    uint32_t prefill_compute_tokens,
    uint32_t max_audio_tokens,
    uint32_t max_visible_keys,
    gemma4_history_batch_plan *out
) {
    if (!out || prefix_len == 0 || current_position != prefix_len ||
        valid_kv_len != prefix_len || audio_tokens == 0 ||
        audio_tokens > max_audio_tokens || (audio_tokens % 4u) != 0 ||
        prefix_len > prefill_compute_tokens || prefill_compute_tokens < audio_tokens ||
        (uint64_t)prefix_len + (uint64_t)audio_tokens > max_seq_len) return false;
    const uint64_t final_position = (uint64_t)prefix_len + audio_tokens;
    if (final_position > max_visible_keys || final_position > UINT32_MAX) return false;
    out->prefix_position = prefix_len;
    out->audio_tokens = audio_tokens;
    out->compute_tokens = prefill_compute_tokens;
    out->final_position = (uint32_t)final_position;
    out->maximum_visible_keys = (uint32_t)final_position;
    return true;
}

static inline uint32_t gemma4_history_batch_causal_key_limit(
    const gemma4_history_batch_plan *plan,
    uint32_t query_index
) {
    if (!plan || query_index >= plan->audio_tokens) return 0;
    const uint32_t query_end = plan->prefix_position + query_index + 1u;
    return query_end < plan->final_position ? query_end : plan->final_position;
}

static inline uint32_t gemma4_history_batch_sliding_start(
    const gemma4_history_batch_plan *plan,
    uint32_t query_index,
    uint32_t sliding_window
) {
    if (!plan || query_index >= plan->audio_tokens || sliding_window == 0) return 0;
    const uint32_t query_position = plan->prefix_position + query_index;
    return query_position >= sliding_window
        ? query_position - sliding_window + 1u
        : 0u;
}

static inline bool gemma4_history_batch_commit_candidate(
    const gemma4_history_batch_plan *plan,
    uint32_t current_position,
    bool health_ok,
    uint32_t *out_position
) {
    if (!plan || !out_position || current_position != plan->prefix_position) return false;
    *out_position = health_ok ? plan->final_position : current_position;
    return true;
}

static inline uint32_t gemma4_history_batch_ple_source_row(uint32_t query_index) {
    return query_index % 4u;
}

#endif
