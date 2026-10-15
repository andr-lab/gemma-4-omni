/*
 * gemma4_listener_aten.h — diagnostic ATen SDPA parity bridge.
 *
 * This entrypoint is isolation-only.  It borrows caller-owned CUDA storage,
 * executes public ATen scaled_dot_product_attention on the supplied stream,
 * and writes into caller-owned output storage.  It is not part of the native
 * steady-state executor or its zero-allocation contract.
 */
#ifndef GEMMA4_LISTENER_ATEN_H
#define GEMMA4_LISTENER_ATEN_H

#include "gemma4_listener.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Contract for the current Gemma 4 text attention oracle:
 *   query:    BF16 contiguous [B, Hq, Q, D]
 *   key:      BF16 contiguous [B, Hkv, K, D]
 *   value:    BF16 contiguous [B, Hkv, K, D]
 *   attn_mask: BF16, FP32, or BOOL contiguous broadcastable
 *              [B|1, Hq|1, Q, K], containing the explicit Transformers mask
 *              (BOOL means True=allowed, matching SDPA's bool-mask contract)
 *   output:   BF16 contiguous [B, Hq, Q, D]
 *
 * K/V are expanded with the literal Transformers repeat_kv layout
 * [B,Hkv,1,K,D].expand([B,Hkv,Hq/Hkv,K,D]).reshape([B,Hq,K,D]); SDPA is
 * then called with dropout_p=0, is_causal=false, and enable_gqa=false.
 * `scale` is passed as the public ATen optional scale (Gemma uses 1.0).
 * All pointers are borrowed and must remain valid until the supplied stream
 * has completed its work.  The function does not synchronize the stream.
 */
int gemma4_listener_debug_attention(
    const void *query,
    const gemma4_tensor_meta *query_meta,
    const void *key,
    const gemma4_tensor_meta *key_meta,
    const void *value,
    const gemma4_tensor_meta *value_meta,
    const void *attn_mask,
    const gemma4_tensor_meta *attn_mask_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    double scale,
    cudaStream_t stream
);

/* Same explicit-mask contract as gemma4_listener_debug_attention, but passes
 * [B,Hkv,K,D] directly to public ATen SDPA with enable_gqa=true.  Q may be a
 * native strided view and is made contiguous; K/V are borrowed without a
 * bridge-side contiguous/repeat/reshape operation. */
int gemma4_listener_debug_attention_gqa(
    const void *query,
    const gemma4_tensor_meta *query_meta,
    const void *key,
    const gemma4_tensor_meta *key_meta,
    const void *value,
    const gemma4_tensor_meta *value_meta,
    const void *attn_mask,
    const gemma4_tensor_meta *attn_mask_meta,
    void *output,
    const gemma4_tensor_meta *output_meta,
    double scale,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* GEMMA4_LISTENER_ATEN_H */
