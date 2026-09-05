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

typedef struct {
    __nv_bfloat16 *k_data;   /* [max_seq_len, head_dim] */
    __nv_bfloat16 *v_data;   /* [max_seq_len, head_dim] */
    uint32_t head_dim;
    bool is_shared;
    int32_t shared_src;
} gemma4_layer_cache;

typedef struct {
    uint32_t max_seq_len;
    uint32_t current_pos;
    uint32_t valid_kv_len;
    gemma4_layer_cache layers[GEMMA4_NUM_LAYERS];
} gemma4_cache;

void gemma4_cache_destroy(gemma4_cache *c);

/* Allocate persistent K/V device memory */
int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len) {
    if (!out_cache || max_seq_len == 0) return -1;

    gemma4_cache *c = (gemma4_cache *)calloc(1, sizeof(gemma4_cache));
    if (!c) return -1;

    c->max_seq_len = max_seq_len;
    c->current_pos = 0;
    c->valid_kv_len = 0;

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
                fprintf(stderr, "[gemma4_cache] cudaMalloc failed for layer %d\n", i);
                gemma4_cache_destroy(c);
                return -2;
            }
            if (cudaMemset(c->layers[i].k_data, 0, bytes) != cudaSuccess ||
                cudaMemset(c->layers[i].v_data, 0, bytes) != cudaSuccess) {
                fprintf(stderr, "[gemma4_cache] cudaMemset failed for layer %d\n", i);
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

/* Advance position counter after all layers finish current step */
int gemma4_cache_advance_position(gemma4_cache *c, int q_len) {
    if (!c || q_len < 0 || c->valid_kv_len > c->max_seq_len ||
        (uint32_t)q_len > c->max_seq_len - c->valid_kv_len) return -1;
    c->valid_kv_len += (uint32_t)q_len;
    c->current_pos += (uint32_t)q_len;
    return 0;
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
        }
    }
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
