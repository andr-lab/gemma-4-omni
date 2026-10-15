#ifndef GEMMA4_LISTENER_MICROBATCH_FULL_REPEAT_CAPACITY_H
#define GEMMA4_LISTENER_MICROBATCH_FULL_REPEAT_CAPACITY_H

#include <cstddef>
#include <cstdint>
#include <limits>

namespace gemma4_full_repeat_staging {

constexpr uint32_t kInitialCapacity = 2048u;

inline bool initial_capacity(uint32_t logical_max, uint32_t *out_capacity) {
    if (!out_capacity || logical_max == 0) return false;
    *out_capacity = logical_max < kInitialCapacity ? logical_max : kInitialCapacity;
    return true;
}

inline bool next_capacity(
    uint32_t current_capacity,
    uint32_t needed_capacity,
    uint32_t logical_max,
    uint32_t *out_capacity
) {
    if (!out_capacity || logical_max == 0 || needed_capacity == 0 ||
        needed_capacity > logical_max || current_capacity > logical_max) return false;
    if (needed_capacity <= current_capacity) {
        *out_capacity = current_capacity;
        return true;
    }

    uint32_t capacity = current_capacity;
    if (capacity == 0 && !initial_capacity(logical_max, &capacity)) return false;
    while (capacity < needed_capacity) {
        if (capacity >= logical_max) return false;
        const uint64_t doubled = (uint64_t)capacity * 2u;
        capacity = doubled > logical_max ? logical_max : static_cast<uint32_t>(doubled);
    }
    *out_capacity = capacity;
    return true;
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
    const uint64_t max_size = static_cast<uint64_t>(std::numeric_limits<size_t>::max());
    if (heads > max_size / capacity) return false;
    const uint64_t rows = heads * capacity;
    if (rows > max_size / head_dim) return false;
    const uint64_t elements = rows * head_dim;
    if (elements > max_size / element_bytes) return false;
    *out_bytes = static_cast<size_t>(elements * element_bytes);
    return true;
}

/* Allocate both replacements before publishing either live pointer. The caller
 * supplies the owner-stream fence; old storage is released only after it passes. */
template <typename Allocate, typename Deallocate, typename Fence>
inline bool replace_pair_transactionally(
    void *&live_key,
    void *&live_value,
    uint32_t &live_capacity,
    uint32_t target_capacity,
    size_t bytes_per_buffer,
    const char *key_label,
    const char *value_label,
    Allocate allocate,
    Deallocate deallocate,
    Fence fence
) {
    const bool has_key = live_key != nullptr;
    const bool has_value = live_value != nullptr;
    if (!key_label || !value_label || bytes_per_buffer == 0 || target_capacity == 0 ||
        target_capacity <= live_capacity || has_key != has_value ||
        (live_capacity == 0) != !has_key) return false;

    void *new_key = nullptr;
    if (!allocate(&new_key, bytes_per_buffer, key_label) || !new_key) {
        if (new_key) deallocate(new_key);
        return false;
    }
    void *new_value = nullptr;
    if (!allocate(&new_value, bytes_per_buffer, value_label) || !new_value) {
        if (new_value) deallocate(new_value);
        deallocate(new_key);
        return false;
    }
    if (has_key && !fence()) {
        deallocate(new_value);
        deallocate(new_key);
        return false;
    }

    void *old_key = live_key;
    void *old_value = live_value;
    live_key = new_key;
    live_value = new_value;
    live_capacity = target_capacity;
    if (old_key) deallocate(old_key);
    if (old_value) deallocate(old_value);
    return true;
}

} // namespace gemma4_full_repeat_staging

#endif // GEMMA4_LISTENER_MICROBATCH_FULL_REPEAT_CAPACITY_H
