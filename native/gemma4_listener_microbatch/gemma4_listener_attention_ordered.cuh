/*
 * CUTLASS_ORDERED_TILES_V1: four-stage full-attention implementation.
 *
 * The public ABI intentionally exposes only lifecycle/launch functions.  The
 * workspace is persistent and is created once when backend 3 is selected.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include "generated_topology.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEMMA4_ORDERED_KEY_TILE_SIZE 128
#define GEMMA4_ORDERED_QUERY_ROWS GEMMA4_STREAM_Q_LEN
#define GEMMA4_ORDERED_QUERY_TILE_ROWS 32
#define GEMMA4_ORDERED_HEADS 8
#define GEMMA4_ORDERED_HEAD_DIM 512
#define GEMMA4_ORDERED_MAX_KEYS 16384
#define GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE 2048
#define GEMMA4_ORDERED_IMPLEMENTATION "CUTLASS_ORDERED_TILES_V1"
#define GEMMA4_ORDERED_REDUCTION_ORDER "ASCENDING_KEY_TILE"

/* Opaque-to-CUDA-layout workspace. Every pointer is device-resident. */
typedef struct gemma4_ordered_workspace {
    void *q_head_major;
    void *qk_fragments;
    void *probability_tiles;
    float *s_prime;
    float *out_rescale;
    void *pv_fragments;
    float *output_accum;
    size_t q_head_major_bytes;
    size_t qk_fragment_bytes;
    size_t probability_tile_bytes;
    size_t softmax_state_bytes;
    size_t pv_fragment_bytes;
    size_t output_accum_bytes;
    size_t total_bytes;
    uint32_t max_keys;
    uint32_t max_tiles;
    uint32_t initialized;
    uint64_t pack_launches;
    uint64_t stage_a_launches;
    uint64_t stage_b_launches;
    uint64_t stage_c_launches;
    uint64_t stage_d_launches;
} gemma4_ordered_workspace;

typedef struct gemma4_ordered_create_failure {
    const char *field;
    const char *reason;
    size_t requested_bytes;
    int cuda_error;
    int cuda_error_known;
} gemma4_ordered_create_failure;

int gemma4_ordered_workspace_create(
    gemma4_ordered_workspace *workspace,
    uint32_t max_keys);
int gemma4_ordered_workspace_create_with_diagnostics(
    gemma4_ordered_workspace *workspace,
    uint32_t max_keys,
    gemma4_ordered_create_failure *failure);

void gemma4_ordered_workspace_destroy(gemma4_ordered_workspace *workspace);

size_t gemma4_ordered_workspace_bytes(const gemma4_ordered_workspace *workspace);
uint64_t gemma4_ordered_pack_launches(const gemma4_ordered_workspace *workspace);
uint64_t gemma4_ordered_stage_a_launches(const gemma4_ordered_workspace *workspace);
uint64_t gemma4_ordered_stage_b_launches(const gemma4_ordered_workspace *workspace);
uint64_t gemma4_ordered_stage_c_launches(const gemma4_ordered_workspace *workspace);
uint64_t gemma4_ordered_stage_d_launches(const gemma4_ordered_workspace *workspace);

/*
 * Input Q is the existing native q-major contiguous [Q,H,D] buffer. The
 * launch performs one bitwise pack to persistent head-major [H,Q,D], then
 * launches A(QK), B(softmax), C(PV), and D(output fold) on the same stream.
 */
int gemma4_ordered_attention_launch(
    const __nv_bfloat16 *query_q_major,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    const int64_t *prefix_attention,
    uint32_t prefix_len,
    uint32_t key_len,
    uint32_t base_position,
    float scale,
    __nv_bfloat16 *output,
    cudaStream_t stream,
    gemma4_ordered_workspace *workspace);

#ifdef __cplusplus
}
#endif
