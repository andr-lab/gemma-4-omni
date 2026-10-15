#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "listener_create_failure_log.h"
#include "sliding_repeat_capacity.h"

int main() {
    using namespace gemma4_sliding_staging;

    Contract contract{};
    assert(derive_contract(
        gemma4_layers,
        GEMMA4_NUM_LAYERS,
        16384,
        GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN,
        &contract));
    assert(contract.max_sliding_window == 512);
    assert(contract.max_sliding_head_dim == 256);
    assert(contract.capacity == 802);

    Contract clamped{};
    assert(derive_contract(gemma4_layers, GEMMA4_NUM_LAYERS, 600, 291, &clamped));
    assert(clamped.capacity == 600);

    size_t generic_bytes = 0;
    size_t shared_bytes = 0;
    assert(repeat_buffer_bytes(8, contract.capacity, contract.max_sliding_head_dim,
                              sizeof(uint16_t), &generic_bytes));
    assert(repeat_buffer_bytes(8, contract.capacity, contract.max_sliding_head_dim,
                              sizeof(uint16_t), &shared_bytes));
    assert(generic_bytes == 8u * 802u * 256u * 2u);
    assert(shared_bytes == generic_bytes);

    assert(validate_position_query(0, 0, 291, 34, 291) == Status::kOk);
    assert(validate_position_query(512, 512, 4, 4, 291) == Status::kOk);
    assert(validate_position_query(512, 511, 4, 4, 291) == Status::kPositionMismatch);
    assert(validate_position_query(0, 0, 292, 4, 291) == Status::kInvalidQuery);
    assert(validate_position_query(0, 0, 4, 5, 291) == Status::kInvalidQuery);
    assert(validate_position_query(0, 0, 0, 0, 291) == Status::kInvalidQuery);

    assert(validate_visible_window(802, 16384, contract) == Status::kOk);
    assert(validate_visible_window(803, 16384, contract) == Status::kWindowTooLarge);
    assert(validate_visible_window(600, 600, clamped) == Status::kOk);
    assert(validate_visible_window(601, 600, clamped) == Status::kWindowTooLarge);
    assert(validate_head_dim(256, contract) == Status::kOk);
    assert(validate_head_dim(512, contract) == Status::kHeadDimTooLarge);

    size_t overflow_bytes = 0;
    assert(!repeat_buffer_bytes(UINT64_MAX, contract.capacity,
                                contract.max_sliding_head_dim, sizeof(uint16_t),
                                &overflow_bytes));

    /* Inject a failed CUDA memory query as data so this logger contract can be
     * tested on a CPU-only host without calling CUDA runtime APIs. */
    FILE *diagnostic_stream = tmpfile();
    assert(diagnostic_stream != nullptr);
    gemma4_listener_diagnostics::write_context_create_failure(
        diagnostic_stream,
        "cache.layers[13].k_data",
        123456u,
        "cudaMalloc",
        "cudaErrorMemoryAllocation",
        2,
        "out of memory",
        false,
        0,
        0,
        "cudaErrorNoDevice",
        100);
    assert(fflush(diagnostic_stream) == 0);
    assert(fseek(diagnostic_stream, 0, SEEK_SET) == 0);
    char diagnostic[1024] = {};
    const size_t diagnostic_size =
        fread(diagnostic, 1, sizeof(diagnostic) - 1, diagnostic_stream);
    assert(diagnostic_size > 0);
    fclose(diagnostic_stream);
    assert(std::strstr(diagnostic, "field=cache.layers[13].k_data") != nullptr);
    assert(std::strstr(diagnostic, "requested_bytes=123456") != nullptr);
    assert(std::strstr(diagnostic, "cuda_error=cudaErrorMemoryAllocation(2)") != nullptr);
    assert(std::strstr(diagnostic, "cuda_message=out of memory") != nullptr);
    assert(std::strstr(diagnostic, "free_bytes=unknown total_bytes=unknown") != nullptr);
    assert(std::strstr(diagnostic, "memory_query_error=cudaErrorNoDevice(100)") != nullptr);
    return 0;
}
