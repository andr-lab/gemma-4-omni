/* Exact ATen SDPA bridge used by the initial Qwen native attention path. */
#ifndef QWEN_TALKER_ATEN_ATTENTION_H
#define QWEN_TALKER_ATEN_ATTENTION_H

#include "qwen_talker.h"

#ifdef __cplusplus
struct qwen_talker_attention_graph_state;
#else
typedef struct qwen_talker_attention_graph_state qwen_talker_attention_graph_state;
#endif

#ifdef __cplusplus
extern "C" {
#endif

int qwen_talker_attention_gqa(
    const void *query,
    const qwen_tensor_meta *query_meta,
    const void *key,
    const qwen_tensor_meta *key_meta,
    const void *value,
    const qwen_tensor_meta *value_meta,
    const void *attention_mask,
    const qwen_tensor_meta *attention_mask_meta,
    void *output,
    const qwen_tensor_meta *output_meta,
    double scale,
    cudaStream_t stream
);

/* The exact ATen SDPA oracle is captured once against context-owned fixed
 * buffers. Replays launch only the captured graph and do not enter ATen. */
int qwen_talker_attention_graph_create(
    qwen_talker_attention_graph_state **out_state
);

int qwen_talker_attention_graph_run(
    qwen_talker_attention_graph_state *state,
    const void *query,
    const void *key,
    const void *value,
    const void *attention_mask,
    void *output,
    uint32_t query_len,
    uint32_t max_key_len,
    double scale,
    cudaStream_t stream
);

void qwen_talker_attention_graph_destroy(
    qwen_talker_attention_graph_state *state
);

int qwen_talker_attention_graph_captured(
    const qwen_talker_attention_graph_state *state
);

#ifdef __cplusplus
}
#endif

#endif /* QWEN_TALKER_ATEN_ATTENTION_H */
