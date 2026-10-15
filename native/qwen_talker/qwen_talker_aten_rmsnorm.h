#ifndef GEMMA4_QWEN_TALKER_ATEN_RMSNORM_H
#define GEMMA4_QWEN_TALKER_ATEN_RMSNORM_H

#include "qwen_talker.h"

#ifdef __cplusplus
struct qwen_talker_rmsnorm_graph_state;
extern "C" {
#else
typedef struct qwen_talker_rmsnorm_graph_state qwen_talker_rmsnorm_graph_state;
#endif

int qwen_talker_rmsnorm_graph_create(
    qwen_talker_rmsnorm_graph_state **out_state,
    uint32_t rows,
    void *staging_input_bf16,
    void *staging_weight_bf16,
    void *staging_output_bf16
);

int qwen_talker_rmsnorm_graph_run(
    qwen_talker_rmsnorm_graph_state *state,
    const void *input_bf16,
    const void *weight_bf16,
    void *output_bf16,
    float *row_mean_output,
    double epsilon,
    cudaStream_t stream
);

void qwen_talker_rmsnorm_graph_destroy(
    qwen_talker_rmsnorm_graph_state *state
);

int qwen_talker_rmsnorm_graph_captured(
    const qwen_talker_rmsnorm_graph_state *state
);

#ifdef __cplusplus
}
#endif

#endif
