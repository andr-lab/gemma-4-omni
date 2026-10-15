#ifndef GEMMA4_LISTENER_CREATE_FAILURE_LOG_H
#define GEMMA4_LISTENER_CREATE_FAILURE_LOG_H

#include <stddef.h>
#include <stdio.h>

namespace gemma4_listener_diagnostics {

/* Format one context-creation failure record. The caller supplies CUDA error
 * strings and an optional memory snapshot so the exact production formatter
 * can be verified by a host-only test without a CUDA device. */
inline void write_context_create_failure(
    FILE *stream,
    const char *field,
    size_t requested_bytes,
    const char *reason,
    const char *cuda_error_name,
    int cuda_error_code,
    const char *cuda_message,
    bool memory_info_available,
    size_t free_bytes,
    size_t total_bytes,
    const char *memory_query_error_name,
    int memory_query_error_code
) {
    if (!stream) return;
    if (memory_info_available) {
        fprintf(stream,
                "[gemma4_listener] context create failure field=%s requested_bytes=%zu reason=%s cuda_error=%s(%d) cuda_message=%s free_bytes=%zu total_bytes=%zu memory_query_error=none\n",
                field ? field : "unknown", requested_bytes,
                reason ? reason : "unknown",
                cuda_error_name ? cuda_error_name : "unknown",
                cuda_error_code,
                cuda_message ? cuda_message : "unknown",
                free_bytes, total_bytes);
    } else {
        fprintf(stream,
                "[gemma4_listener] context create failure field=%s requested_bytes=%zu reason=%s cuda_error=%s(%d) cuda_message=%s free_bytes=unknown total_bytes=unknown memory_query_error=%s(%d)\n",
                field ? field : "unknown", requested_bytes,
                reason ? reason : "unknown",
                cuda_error_name ? cuda_error_name : "unknown",
                cuda_error_code,
                cuda_message ? cuda_message : "unknown",
                memory_query_error_name ? memory_query_error_name : "unknown",
                memory_query_error_code);
    }
}

} // namespace gemma4_listener_diagnostics

#endif // GEMMA4_LISTENER_CREATE_FAILURE_LOG_H
