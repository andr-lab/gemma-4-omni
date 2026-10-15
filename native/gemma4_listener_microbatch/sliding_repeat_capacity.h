#ifndef GEMMA4_LISTENER_SLIDING_REPEAT_CAPACITY_H
#define GEMMA4_LISTENER_SLIDING_REPEAT_CAPACITY_H

#include <stddef.h>
#include <stdint.h>

#include "generated_topology.h"

namespace gemma4_sliding_staging {

struct Contract {
    uint32_t capacity;
    uint32_t max_sliding_window;
    uint32_t max_sliding_head_dim;
};

enum class Status {
    kOk = 0,
    kInvalidContract,
    kPositionMismatch,
    kInvalidQuery,
    kWindowTooLarge,
    kHeadDimTooLarge,
};

/* Bound the repeated-KV staging buffers from the generated layer topology and
 * the largest supported compute query. For a sliding window, the maximum
 * visible cache rows are window + query - 1; max_seq_len may impose a lower
 * ceiling. */
inline bool derive_contract(
    const GemmaLayerSpec *layers,
    uint32_t layer_count,
    uint32_t max_seq_len,
    uint32_t max_compute_query,
    Contract *out
) {
    if (!layers || layer_count == 0 || max_seq_len == 0 ||
        max_compute_query == 0 || !out) return false;

    uint32_t max_window = 0;
    uint32_t max_head_dim = 0;
    for (uint32_t index = 0; index < layer_count; ++index) {
        const GemmaLayerSpec *spec = &layers[index];
        if (spec->attn_type != GEMMA_ATTN_SLIDING) continue;
        if (spec->sliding_window == 0 || spec->head_dim == 0) return false;
        if (spec->sliding_window > max_window) max_window = spec->sliding_window;
        if (spec->head_dim > max_head_dim) max_head_dim = spec->head_dim;
    }
    if (max_window == 0 || max_head_dim == 0) return false;

    const uint64_t window_bound = (uint64_t)max_window + max_compute_query - 1u;
    const uint32_t capacity = window_bound < max_seq_len
        ? static_cast<uint32_t>(window_bound) : max_seq_len;
    if (capacity == 0) return false;
    out->capacity = capacity;
    out->max_sliding_window = max_window;
    out->max_sliding_head_dim = max_head_dim;
    return true;
}

inline Status validate_position_query(
    uint32_t base_position,
    uint32_t cache_valid_position,
    int compute_query_len,
    int valid_query_len,
    uint32_t max_compute_query
) {
    if (base_position != cache_valid_position) return Status::kPositionMismatch;
    if (compute_query_len <= 0 || valid_query_len <= 0 ||
        valid_query_len > compute_query_len ||
        static_cast<uint32_t>(compute_query_len) > max_compute_query) {
        return Status::kInvalidQuery;
    }
    return Status::kOk;
}

inline Status validate_visible_window(
    int visible_k_len,
    uint32_t max_seq_len,
    const Contract &contract
) {
    if (visible_k_len <= 0 || max_seq_len == 0 || contract.capacity == 0 ||
        static_cast<uint32_t>(visible_k_len) > max_seq_len ||
        static_cast<uint32_t>(visible_k_len) > contract.capacity) {
        return Status::kWindowTooLarge;
    }
    return Status::kOk;
}

inline Status validate_head_dim(uint32_t head_dim, const Contract &contract) {
    if (head_dim == 0 || contract.max_sliding_head_dim == 0 ||
        head_dim > contract.max_sliding_head_dim) {
        return Status::kHeadDimTooLarge;
    }
    return Status::kOk;
}

inline bool repeat_buffer_bytes(
    uint64_t heads,
    uint64_t capacity,
    uint64_t head_dim,
    uint64_t element_bytes,
    size_t *out_bytes
) {
    if (!out_bytes || heads == 0 || capacity == 0 || head_dim == 0 ||
        element_bytes == 0) return false;
    const uint64_t max_size = static_cast<uint64_t>(static_cast<size_t>(-1));
    if (heads > max_size / capacity) return false;
    const uint64_t rows = heads * capacity;
    if (rows > max_size / head_dim) return false;
    const uint64_t elements = rows * head_dim;
    if (elements > max_size / element_bytes) return false;
    *out_bytes = static_cast<size_t>(elements * element_bytes);
    return true;
}

} // namespace gemma4_sliding_staging

#endif // GEMMA4_LISTENER_SLIDING_REPEAT_CAPACITY_H
