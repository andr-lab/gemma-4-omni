/* CUDA kernels for the native Qwen3-TTS Talker backbone. */
#ifndef QWEN_TALKER_KERNELS_CUH
#define QWEN_TALKER_KERNELS_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <stdint.h>
#include "generated/qwen_native_topology.h"

int qwen_copy_joint_inputs(
    const __nv_bfloat16 *prefix,
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden,
    cudaStream_t stream
);

int qwen_copy_audio_input(
    const __nv_bfloat16 *audio,
    __nv_bfloat16 *hidden,
    cudaStream_t stream
);

int qwen_write_position_ids(
    int64_t *destination,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
);

int qwen_rmsnorm_exact(
    const __nv_bfloat16 *input,
    const __nv_bfloat16 *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    float eps,
    float *squared_workspace,
    float *row_sum_workspace,
    cudaStream_t stream
);

int qwen_pack_heads(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t rows,
    uint32_t heads,
    uint32_t head_dim,
    cudaStream_t stream
);

int qwen_apply_rope_interleaved(
    __nv_bfloat16 *q_or_k,
    const __nv_bfloat16 *inv_freq,
    uint32_t heads,
    uint32_t rows,
    uint32_t head_dim,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
);

int qwen_apply_rope_interleaved_cursor(
    __nv_bfloat16 *q_or_k,
    const __nv_bfloat16 *inv_freq,
    uint32_t heads,
    uint32_t rows,
    uint32_t head_dim,
    const uint32_t *device_position,
    cudaStream_t stream
);

int qwen_write_joint_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
);

int qwen_write_audio_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t audio_position,
    cudaStream_t stream
);

int qwen_write_audio_kv_cursor(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *cache,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    const uint32_t *device_position,
    uint32_t *device_error,
    cudaStream_t stream
);

int qwen_import_cache_prefix(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t source_max_len,
    uint32_t max_cache_len,
    uint32_t stable_length,
    cudaStream_t stream
);

int qwen_copy_cache_joint_rows(
    const __nv_bfloat16 *cache,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t head_dim,
    uint32_t max_cache_len,
    uint32_t prefix_position,
    uint32_t audio_position,
    cudaStream_t stream
);

int qwen_build_joint_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
);

int qwen_build_terminal_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
);

int qwen_build_audio_only_mask(
    __nv_bfloat16 *mask,
    uint32_t max_cache_len,
    uint32_t key_len,
    uint32_t prefix_start,
    uint32_t audio_start,
    uint32_t visible_prefix_position,
    uint32_t audio_position,
    uint32_t prefix_attention_mode,
    uint32_t visible_prefix_count,
    float prefix_log_gate,
    cudaStream_t stream
);

int qwen_build_audio_q1_causal_mask_cursor(
    __nv_bfloat16 *mask,
    uint32_t key_len,
    const uint32_t *device_position,
    cudaStream_t stream
);

int qwen_initialize_audio_q1_cursor(
    uint32_t *device_position,
    uint32_t *device_error,
    uint32_t initial_position,
    cudaStream_t stream
);

int qwen_increment_audio_q1_cursor(
    uint32_t *device_position,
    cudaStream_t stream
);

int qwen_repeat_kv(
    const __nv_bfloat16 *source,
    __nv_bfloat16 *destination,
    uint32_t kv_heads,
    uint32_t repeats,
    uint32_t key_len,
    uint32_t source_stride_len,
    uint32_t head_dim,
    cudaStream_t stream
);

/* Allocation-free Q=2 GQA attention.  Q/K/V and mask are persistent
 * row-major buffers; scores is persistent [heads, Q, score_stride]. */
int qwen_attention_gqa(
    const __nv_bfloat16 *query,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    const __nv_bfloat16 *mask,
    __nv_bfloat16 *output,
    float scale,
    uint32_t query_rows,
    uint32_t key_len,
    uint32_t score_stride,
    cublasHandle_t blas,
    float *scores,
    cudaStream_t stream
);

int qwen_attention_to_rows(
    const __nv_bfloat16 *attention,
    __nv_bfloat16 *rows,
    uint32_t heads,
    uint32_t q_len,
    uint32_t head_dim,
    cudaStream_t stream
);

int qwen_add_rows(
    const __nv_bfloat16 *left,
    const __nv_bfloat16 *right,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    cudaStream_t stream
);

int qwen_silu_mul(
    const __nv_bfloat16 *gate,
    const __nv_bfloat16 *up,
    __nv_bfloat16 *output,
    uint32_t elements,
    cudaStream_t stream
);

int qwen_copy_audio_row(
    const __nv_bfloat16 *hidden,
    __nv_bfloat16 *output,
    uint32_t row,
    uint32_t dim,
    cudaStream_t stream
);

#endif /* QWEN_TALKER_KERNELS_CUH */
