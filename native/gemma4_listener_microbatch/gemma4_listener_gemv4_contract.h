#ifndef GEMMA4_LISTENER_GEMV4_CONTRACT_H
#define GEMMA4_LISTENER_GEMV4_CONTRACT_H

#include <cstdint>

/* Host-only eligibility contract shared by the runtime dispatch and CPU tests.
 * The fast path handles exactly four BF16 rows with verified contiguous
 * row-major matrices and pair-aligned pointers; all other cases stay cuBLAS. */
constexpr bool gemma4_q4_gemv4_dispatch_eligible(
    bool enabled,
    bool bf16,
    bool contiguous_row_major,
    int rows,
    int output_cols,
    int reduction_cols,
    uintptr_t weight_ptr,
    uintptr_t input_ptr,
    uintptr_t output_ptr
) {
    return enabled && bf16 && contiguous_row_major && rows == 4 &&
        output_cols > 0 && reduction_cols > 0 && (reduction_cols & 1) == 0 &&
        weight_ptr != 0 && input_ptr != 0 && output_ptr != 0 &&
        (weight_ptr & 3u) == 0 && (input_ptr & 3u) == 0 &&
        (output_ptr & 1u) == 0;
}

constexpr uint64_t gemma4_q4_gemv4_weight_index(int output_row, int k, int K) {
    return (uint64_t)output_row * (uint64_t)K + (uint64_t)k;
}

constexpr uint64_t gemma4_q4_gemv4_output_index(int input_row, int output_col, int N) {
    return (uint64_t)input_row * (uint64_t)N + (uint64_t)output_col;
}

#endif
