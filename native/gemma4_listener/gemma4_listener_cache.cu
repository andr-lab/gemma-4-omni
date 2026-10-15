/*
 * gemma4_listener_cache.cu — Fixed-address DynamicCache equivalent for Gemma 4.
 *
 * Implements persistent K/V device buffers for layers 0..14, shared-KV aliasing
 * for layers 15..34 (Layer 13 sliding, Layer 14 global), and sliding-window logic.
 * Conforms strictly to P4-NATIVE Implementation Plan §10 & §11.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "gemma4_listener.h"
#include "generated_topology.h"
#include "listener_cuda_create_failure_log.cuh"

typedef struct {
    __nv_bfloat16 *k_data;   /* [max_seq_len, head_dim] */
    __nv_bfloat16 *v_data;   /* [max_seq_len, head_dim] */
    __nv_bfloat16 *rebase_k_scratch; /* [max rollover tokens, head_dim] */
    __nv_bfloat16 *rebase_v_scratch; /* [max rollover tokens, head_dim] */
    uint32_t head_dim;
    bool is_shared;
    int32_t shared_src;
} gemma4_layer_cache;

typedef struct {
    uint32_t max_seq_len;
    uint32_t current_pos;
    uint32_t valid_kv_len;
    uint32_t *device_position;
    gemma4_layer_cache layers[GEMMA4_NUM_LAYERS];
} gemma4_cache;

__global__ static void gemma4_set_position_kernel(uint32_t *position, uint32_t value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *position = value;
}

__global__ static void gemma4_advance_position_kernel(uint32_t *position, uint32_t amount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *position += amount;
}

__global__ static void gemma4_append_q4_by_position_kernel(
    __nv_bfloat16 *k_dst,
    __nv_bfloat16 *v_dst,
    const __nv_bfloat16 *k_src,
    const __nv_bfloat16 *v_src,
    const uint32_t *position,
    uint32_t head_dim,
    uint32_t max_seq_len
) {
    __shared__ uint32_t base_position;
    if (threadIdx.x == 0) base_position = *position;
    __syncthreads();

    const uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t element_count = GEMMA4_STREAM_Q_LEN * head_dim;
    if (element >= element_count || base_position > max_seq_len - GEMMA4_STREAM_Q_LEN) return;
    const uint32_t row = element / head_dim;
    const uint32_t column = element - row * head_dim;
    const uint32_t destination = (base_position + row) * head_dim + column;
    k_dst[destination] = k_src[element];
    v_dst[destination] = v_src[element];
}

void gemma4_cache_destroy(gemma4_cache *c);

/* Allocate persistent K/V device memory */
int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len, bool allocate_device_position) {
    if (!out_cache || max_seq_len == 0) return -1;

    gemma4_cache *c = (gemma4_cache *)calloc(1, sizeof(gemma4_cache));
    if (!c) return -1;

    c->max_seq_len = max_seq_len;
    c->current_pos = 0;
    c->valid_kv_len = 0;
    if (allocate_device_position) {
        const cudaError_t position_allocation =
            cudaMalloc((void **)&c->device_position, sizeof(uint32_t));
        if (position_allocation != cudaSuccess) {
            gemma4_log_create_failure(
                "cache.device_position", sizeof(uint32_t), "cudaMalloc",
                true, position_allocation);
            gemma4_cache_destroy(c);
            return -2;
        }
        const cudaError_t position_initialization =
            cudaMemset(c->device_position, 0, sizeof(uint32_t));
        if (position_initialization != cudaSuccess) {
            gemma4_log_create_failure(
                "cache.device_position", sizeof(uint32_t), "cudaMemset",
                true, position_initialization);
            gemma4_cache_destroy(c);
            return -2;
        }
    }

    for (int i = 0; i < GEMMA4_NUM_LAYERS; ++i) {
        const GemmaLayerSpec *spec = &gemma4_layers[i];
        c->layers[i].head_dim = spec->head_dim;
        c->layers[i].is_shared = spec->is_kv_shared;
        c->layers[i].shared_src = spec->kv_shared_src_index;

        if (!spec->is_kv_shared) {
            /* Layers 0..14 own independent persistent device buffers */
            size_t bytes = (size_t)max_seq_len * spec->head_dim * sizeof(__nv_bfloat16);
            cudaError_t err1 = cudaMalloc((void **)&c->layers[i].k_data, bytes);
            cudaError_t err2 = cudaMalloc((void **)&c->layers[i].v_data, bytes);
            if (err1 != cudaSuccess || err2 != cudaSuccess) {
                char field[64];
                if (err1 != cudaSuccess) {
                    snprintf(field, sizeof(field), "cache.layers[%d].k_data", i);
                    gemma4_log_create_failure(
                        field, bytes, "cudaMalloc", true, err1);
                }
                if (err2 != cudaSuccess) {
                    snprintf(field, sizeof(field), "cache.layers[%d].v_data", i);
                    gemma4_log_create_failure(
                        field, bytes, "cudaMalloc", true, err2);
                }
                gemma4_cache_destroy(c);
                return -2;
            }
            const size_t rebase_bytes =
                (size_t)GEMMA4_LISTENER_MAX_ROLLOVER_AUDIO_TOKENS *
                spec->head_dim * sizeof(__nv_bfloat16);
            const cudaError_t rebase_k_allocation =
                cudaMalloc((void **)&c->layers[i].rebase_k_scratch, rebase_bytes);
            if (rebase_k_allocation != cudaSuccess) {
                char field[64];
                snprintf(field, sizeof(field), "cache.layers[%d].rebase_k_scratch", i);
                gemma4_log_create_failure(
                    field, rebase_bytes, "cudaMalloc", true,
                    rebase_k_allocation);
                gemma4_cache_destroy(c);
                return -2;
            }
            const cudaError_t rebase_v_allocation =
                cudaMalloc((void **)&c->layers[i].rebase_v_scratch, rebase_bytes);
            if (rebase_v_allocation != cudaSuccess) {
                char field[64];
                snprintf(field, sizeof(field), "cache.layers[%d].rebase_v_scratch", i);
                gemma4_log_create_failure(
                    field, rebase_bytes, "cudaMalloc", true,
                    rebase_v_allocation);
                gemma4_cache_destroy(c);
                return -2;
            }
            const cudaError_t k_initialization =
                cudaMemset(c->layers[i].k_data, 0, bytes);
            if (k_initialization != cudaSuccess) {
                char field[64];
                snprintf(field, sizeof(field), "cache.layers[%d].k_data", i);
                gemma4_log_create_failure(
                    field, bytes, "cudaMemset", true, k_initialization);
                gemma4_cache_destroy(c);
                return -2;
            }
            const cudaError_t v_initialization =
                cudaMemset(c->layers[i].v_data, 0, bytes);
            if (v_initialization != cudaSuccess) {
                char field[64];
                snprintf(field, sizeof(field), "cache.layers[%d].v_data", i);
                gemma4_log_create_failure(
                    field, bytes, "cudaMemset", true, v_initialization);
                gemma4_cache_destroy(c);
                return -2;
            }
        } else {
            /* Layers 15..34 alias Layer 13 (sliding) or Layer 14 (global) */
            int src = spec->kv_shared_src_index;
            c->layers[i].k_data = c->layers[src].k_data;
            c->layers[i].v_data = c->layers[src].v_data;
        }
    }

    *out_cache = c;
    return 0;
}

/* Append Q new tokens to non-shared layer caches */
int gemma4_cache_append_async(
    gemma4_cache *c,
    int layer_idx,
    const __nv_bfloat16 *new_k,
    const __nv_bfloat16 *new_v,
    int q_len,
    cudaStream_t stream
) {
    if (!c || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS ||
        !new_k || !new_v || q_len < 0) return -1;
    if (c->layers[layer_idx].is_shared) return 0; /* shared layers don't write */
    if (c->valid_kv_len > c->max_seq_len ||
        (uint32_t)q_len > c->max_seq_len - c->valid_kv_len) return -2;

    int head_dim = c->layers[layer_idx].head_dim;
    size_t offset_elements = (size_t)c->valid_kv_len * head_dim;
    size_t copy_bytes = (size_t)q_len * head_dim * sizeof(__nv_bfloat16);

    if (cudaMemcpyAsync(
        c->layers[layer_idx].k_data + offset_elements,
        new_k,
        copy_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ) != cudaSuccess) return -3;
    if (cudaMemcpyAsync(
        c->layers[layer_idx].v_data + offset_elements,
        new_v,
        copy_bytes,
        cudaMemcpyDeviceToDevice,
        stream
    ) != cudaSuccess) return -3;
    return 0;
}

/* Fixed-grid append for graph replay. Host counters validate capacity before
 * capture; the graph reads the device cursor to select each frame's KV rows. */
int gemma4_cache_append_device_position_async(
    gemma4_cache *c,
    int layer_idx,
    const __nv_bfloat16 *new_k,
    const __nv_bfloat16 *new_v,
    int q_len,
    cudaStream_t stream
) {
    if (!c || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS ||
        !new_k || !new_v || q_len != GEMMA4_STREAM_Q_LEN || !c->device_position) return -1;
    if (c->layers[layer_idx].is_shared) return 0;
    if (c->valid_kv_len > c->max_seq_len ||
        (uint32_t)q_len > c->max_seq_len - c->valid_kv_len) return -2;

    const uint32_t head_dim = c->layers[layer_idx].head_dim;
    const uint32_t elements = (uint32_t)q_len * head_dim;
    const uint32_t blocks = (elements + 255u) / 256u;
    gemma4_append_q4_by_position_kernel<<<blocks, 256, 0, stream>>>(
        c->layers[layer_idx].k_data,
        c->layers[layer_idx].v_data,
        new_k,
        new_v,
        c->device_position,
        head_dim,
        c->max_seq_len
    );
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

int gemma4_cache_set_device_position_async(
    gemma4_cache *c,
    uint32_t position,
    cudaStream_t stream
) {
    if (!c) return -1;
    if (!c->device_position) return 0;
    if (position > c->max_seq_len) return -2;
    gemma4_set_position_kernel<<<1, 1, 0, stream>>>(c->device_position, position);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

int gemma4_cache_advance_device_position_async(
    gemma4_cache *c,
    uint32_t amount,
    cudaStream_t stream
) {
    if (!c || !c->device_position || amount > c->max_seq_len) return -1;
    gemma4_advance_position_kernel<<<1, 1, 0, stream>>>(c->device_position, amount);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

const uint32_t *gemma4_cache_device_position(const gemma4_cache *c) {
    return c ? c->device_position : NULL;
}

/* Advance position counter after all layers finish current step */
/* Import exact canonical DynamicCache source layers. The caller has already
 * validated descriptors and device ownership; this function only enqueues
 * device-to-device copies and updates host-side logical counters. */
int gemma4_cache_import_async(
    gemma4_cache *c,
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    cudaStream_t stream
) {
    if (!c || !imported || current_position > c->max_seq_len) return -1;
    for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
        const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
        if (spec->is_kv_shared) continue;
        const gemma4_imported_kv *source = &imported->layers[layer_idx];
        const int head_dim = (int)spec->head_dim;
        const size_t bytes = (size_t)current_position * head_dim * sizeof(__nv_bfloat16);
        if (cudaMemcpyAsync(
                c->layers[layer_idx].k_data, source->k_ptr, bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return -2;
        if (cudaMemcpyAsync(
                c->layers[layer_idx].v_data, source->v_ptr, bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return -2;
    }
    c->valid_kv_len = current_position;
    c->current_pos = current_position;
    return gemma4_cache_set_device_position_async(c, current_position, stream);
}

/* Compact the canonical prefix and newest audio KV rows to the front of the
 * fixed cache. The caller shifts retained key RoPE after these stream-ordered
 * copies. This intentionally truncates direct old full-KV visibility. */
int gemma4_cache_rebase_async(
    gemma4_cache *c,
    uint32_t prefix_len,
    uint32_t retained_audio_tokens,
    cudaStream_t stream
) {
    if (!c || prefix_len == 0 || prefix_len > c->max_seq_len ||
        retained_audio_tokens == 0 ||
        retained_audio_tokens > GEMMA4_LISTENER_MAX_ROLLOVER_AUDIO_TOKENS ||
        retained_audio_tokens % GEMMA4_STREAM_Q_LEN != 0 ||
        c->valid_kv_len != c->current_pos ||
        c->valid_kv_len > c->max_seq_len ||
        c->valid_kv_len <= prefix_len + retained_audio_tokens ||
        c->valid_kv_len - prefix_len < retained_audio_tokens) return -1;

    const uint32_t old_position = c->valid_kv_len;
    const uint32_t source_start = old_position - retained_audio_tokens;
    const uint32_t new_position = prefix_len + retained_audio_tokens;
    for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
        const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
        if (spec->is_kv_shared) continue;
        gemma4_layer_cache *layer = &c->layers[layer_idx];
        if (!layer->k_data || !layer->v_data ||
            !layer->rebase_k_scratch || !layer->rebase_v_scratch) return -2;
        const size_t row_bytes = (size_t)layer->head_dim * sizeof(__nv_bfloat16);
        const size_t copy_bytes = (size_t)retained_audio_tokens * row_bytes;
        if (cudaMemcpyAsync(
                layer->rebase_k_scratch,
                layer->k_data + (size_t)source_start * layer->head_dim,
                copy_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(
                layer->rebase_v_scratch,
                layer->v_data + (size_t)source_start * layer->head_dim,
                copy_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(
                layer->k_data + (size_t)prefix_len * layer->head_dim,
                layer->rebase_k_scratch,
                copy_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            cudaMemcpyAsync(
                layer->v_data + (size_t)prefix_len * layer->head_dim,
                layer->rebase_v_scratch,
                copy_bytes, cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return -3;
    }
    c->valid_kv_len = new_position;
    c->current_pos = new_position;
    return gemma4_cache_set_device_position_async(c, new_position, stream);
}

int gemma4_cache_advance_position(gemma4_cache *c, int q_len) {
    if (!c || q_len < 0 || c->valid_kv_len > c->max_seq_len ||
        (uint32_t)q_len > c->max_seq_len - c->valid_kv_len) return -1;
    c->valid_kv_len += (uint32_t)q_len;
    c->current_pos += (uint32_t)q_len;
    return 0;
}

uint32_t gemma4_cache_valid_kv_len(const gemma4_cache *c) {
    return c ? c->valid_kv_len : UINT32_MAX;
}

/* Retrieve read pointers and visible attention window bounds */
void gemma4_cache_get_ptrs(
    const gemma4_cache *c,
    int layer_idx,
    int q_len,
    uint32_t base_pos,
    const __nv_bfloat16 **out_k,
    const __nv_bfloat16 **out_v,
    int *out_total_len,
    int *out_start_offset
) {
    if (!c || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS) return;

    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    int effective_src = spec->is_kv_shared ? spec->kv_shared_src_index : layer_idx;

    *out_k = c->layers[effective_src].k_data;
    *out_v = c->layers[effective_src].v_data;
    int total_len = c->valid_kv_len + q_len;
    *out_total_len = total_len;

    if (spec->sliding_window > 0 && (int)base_pos >= (int)spec->sliding_window) {
        *out_start_offset = (int)base_pos - (int)spec->sliding_window + 1;
    } else {
        *out_start_offset = 0;
    }
}


/* Reset valid positions to 0 */
int gemma4_cache_reset(gemma4_cache *c) {
    if (!c) return -1;
    c->valid_kv_len = 0;
    c->current_pos = 0;
    return 0;
}

/* Free device allocations */
void gemma4_cache_destroy(gemma4_cache *c) {
    if (!c) return;
    for (int i = 0; i < GEMMA4_NUM_LAYERS; ++i) {
        if (!c->layers[i].is_shared) {
            if (c->layers[i].k_data) cudaFree(c->layers[i].k_data);
            if (c->layers[i].v_data) cudaFree(c->layers[i].v_data);
            if (c->layers[i].rebase_k_scratch) cudaFree(c->layers[i].rebase_k_scratch);
            if (c->layers[i].rebase_v_scratch) cudaFree(c->layers[i].rebase_v_scratch);
        }
    }
    if (c->device_position) cudaFree(c->device_position);
    free(c);
}

const void *gemma4_cache_get_layer_k(const gemma4_cache *c, int layer_idx) {
    if (!c || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS) return NULL;
    return (const void *)c->layers[layer_idx].k_data;
}

const void *gemma4_cache_get_layer_v(const gemma4_cache *c, int layer_idx) {
    if (!c || layer_idx < 0 || layer_idx >= GEMMA4_NUM_LAYERS) return NULL;
    return (const void *)c->layers[layer_idx].v_data;
}
