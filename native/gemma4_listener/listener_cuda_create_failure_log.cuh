#ifndef GEMMA4_LISTENER_CUDA_CREATE_FAILURE_LOG_CUH
#define GEMMA4_LISTENER_CUDA_CREATE_FAILURE_LOG_CUH

#include <stdio.h>
#include <cuda_runtime_api.h>

#include "listener_create_failure_log.h"

/* Failure-only CUDA adapter over the host-testable formatter. Static linkage
 * keeps this diagnostic helper internal to each native translation unit. */
static inline void gemma4_log_create_failure(
    const char *field,
    size_t requested_bytes,
    const char *reason,
    bool cuda_error_known,
    cudaError_t cuda_error
) {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t memory_query = cudaMemGetInfo(&free_bytes, &total_bytes);
    const char *cuda_name = cuda_error_known
        ? cudaGetErrorName(cuda_error) : "not_applicable";
    const char *cuda_message = cuda_error_known
        ? cudaGetErrorString(cuda_error) : "not_applicable";
    const char *memory_query_name = cudaGetErrorName(memory_query);
    gemma4_listener_diagnostics::write_context_create_failure(
        stderr, field, requested_bytes, reason,
        cuda_name, cuda_error_known ? (int)cuda_error : -1,
        cuda_message, memory_query == cudaSuccess,
        free_bytes, total_bytes, memory_query_name, (int)memory_query);
}

#endif // GEMMA4_LISTENER_CUDA_CREATE_FAILURE_LOG_CUH
