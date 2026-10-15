/*
 * gemma4_listener_aten.cpp — public ATen SDPA diagnostic bridge.
 *
 * The bridge intentionally lives outside the native CUDA executor.  It is a
 * controlled oracle-isolation tool, not a production replacement for the
 * native attention kernel.
 */
#include "gemma4_listener_aten.h"

#include <ATen/ops/from_blob.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>

namespace {

bool is_device_pointer(const void *ptr, int device) {
    if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 1u) != 0u) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) return false;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}

bool validate_bridge_meta(
    const void *ptr,
    const gemma4_tensor_meta *meta,
    uint32_t dtype,
    uint32_t ndim,
    int device
) {
    if (!ptr || !meta || meta->ptr != ptr || meta->dtype != dtype ||
        meta->ndim != ndim || meta->contiguous > 1 || meta->reserved != 0 ||
        ndim > GEMMA4_TENSOR_META_MAX_RANK || !is_device_pointer(ptr, device)) {
        return false;
    }
    uint64_t running = 1;
    for (int index = static_cast<int>(ndim) - 1; index >= 0; --index) {
        if (meta->dims[index] == 0 || meta->dims[index] > UINT64_MAX / running ||
            meta->strides[index] == 0 ||
            (meta->contiguous == 1 && meta->strides[index] != running)) {
            return false;
        }
        running *= meta->dims[index];
    }
    for (uint32_t index = ndim; index < GEMMA4_TENSOR_META_MAX_RANK; ++index) {
        if (meta->dims[index] != 0 || meta->strides[index] != 0) return false;
    }
    return true;
}

at::ScalarType scalar_type(uint32_t dtype) {
    switch (dtype) {
        case GEMMA4_META_BF16: return at::kBFloat16;
        case GEMMA4_META_FP32: return at::kFloat;
        case GEMMA4_META_BOOL: return at::kBool;
        default: throw std::invalid_argument("unsupported ATen bridge dtype");
    }
}

at::Tensor borrowed_tensor(
    const void *ptr,
    const gemma4_tensor_meta *meta,
    int device
) {
    int64_t sizes[GEMMA4_TENSOR_META_MAX_RANK] = {};
    int64_t strides[GEMMA4_TENSOR_META_MAX_RANK] = {};
    for (uint32_t index = 0; index < meta->ndim; ++index) {
        if (meta->dims[index] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
            meta->strides[index] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("ATen bridge metadata exceeds int64");
        }
        sizes[index] = static_cast<int64_t>(meta->dims[index]);
        strides[index] = static_cast<int64_t>(meta->strides[index]);
    }
    const auto options = at::TensorOptions()
        .dtype(scalar_type(meta->dtype))
        .device(c10::DeviceType::CUDA, device);
    return at::from_blob(
        const_cast<void *>(ptr),
        at::IntArrayRef(sizes, meta->ndim),
        at::IntArrayRef(strides, meta->ndim),
        options
    );
}

bool broadcast_dimension(uint64_t actual, uint64_t expected) {
    return actual == 1 || actual == expected;
}

} // namespace

extern "C" int gemma4_listener_debug_attention(
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
) {
    try {
        if (!query || !key || !value || !attn_mask || !output || !std::isfinite(scale)) return -1;
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) return -2;
        if (!validate_bridge_meta(query, query_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(key, key_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(value, value_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(output, output_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(
                attn_mask, attn_mask_meta, attn_mask_meta ? attn_mask_meta->dtype : 0, 4, device)) {
            return -1;
        }
        if (attn_mask_meta->dtype != GEMMA4_META_BF16 &&
            attn_mask_meta->dtype != GEMMA4_META_FP32 &&
            attn_mask_meta->dtype != GEMMA4_META_BOOL) return -1;

        const uint64_t batch = query_meta->dims[0];
        const uint64_t query_heads = query_meta->dims[1];
        const uint64_t query_len = query_meta->dims[2];
        const uint64_t head_dim = query_meta->dims[3];
        const uint64_t key_heads = key_meta->dims[1];
        const uint64_t key_len = key_meta->dims[2];
        if (batch == 0 || query_heads == 0 || key_heads == 0 || query_heads % key_heads != 0 ||
            key_meta->dims[0] != batch || value_meta->dims[0] != batch ||
            value_meta->dims[1] != key_heads || key_meta->dims[2] != value_meta->dims[2] ||
            key_meta->dims[3] != head_dim || value_meta->dims[3] != head_dim ||
            output_meta->dims[0] != batch || output_meta->dims[1] != query_heads ||
            output_meta->dims[2] != query_len || output_meta->dims[3] != head_dim ||
            !broadcast_dimension(attn_mask_meta->dims[0], batch) ||
            !broadcast_dimension(attn_mask_meta->dims[1], query_heads) ||
            attn_mask_meta->dims[2] != query_len || attn_mask_meta->dims[3] != key_len) {
            return -1;
        }

        // Make ATen use exactly the caller's CUDA stream.  No stream or device
        // synchronization is performed by this bridge.
        const c10::cuda::CUDAStream external_stream =
            c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);

        // Native scratch/cache layouts are Q-major or max-sequence pitched.
        // Materialize those borrowed views before SDPA so the bridge uses the
        // same contiguous public-ATen dispatch as the isolated oracle.  This
        // allocation is confined to the explicit diagnostic bridge.
        const at::Tensor q = borrowed_tensor(query, query_meta, device).contiguous();
        const at::Tensor k_view = borrowed_tensor(key, key_meta, device);
        const at::Tensor v_view = borrowed_tensor(value, value_meta, device);
        const at::Tensor k = key_heads == query_heads ? k_view : k_view.contiguous();
        const at::Tensor v = key_heads == query_heads ? v_view : v_view.contiguous();
        const at::Tensor mask = borrowed_tensor(attn_mask, attn_mask_meta, device).contiguous();
        const at::Tensor out = borrowed_tensor(output, output_meta, device);

        // Backend-1 callers use this literal five-dimensional
        // expand/reshape, exactly as transformers.integrations.sdpa_attention
        // repeat_kv.  The production listener may instead provide an already
        // explicitly repeated [1, query_heads, K, D] staging view.  Preserve
        // that view (including a position-major stride over a fixed arena) so
        // ATen can consume it without allocating a second K/V copy.
        const int64_t groups = static_cast<int64_t>(query_heads / key_heads);
        const at::Tensor expanded_key = key_heads == query_heads
            ? k
            : k.view({
                static_cast<int64_t>(batch), static_cast<int64_t>(key_heads), 1,
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            }).expand({
                static_cast<int64_t>(batch), static_cast<int64_t>(key_heads), groups,
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            }).reshape({
                static_cast<int64_t>(batch), static_cast<int64_t>(query_heads),
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            });
        const at::Tensor expanded_value = key_heads == query_heads
            ? v
            : v.view({
                static_cast<int64_t>(batch), static_cast<int64_t>(key_heads), 1,
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            }).expand({
                static_cast<int64_t>(batch), static_cast<int64_t>(key_heads), groups,
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            }).reshape({
                static_cast<int64_t>(batch), static_cast<int64_t>(query_heads),
                static_cast<int64_t>(key_len), static_cast<int64_t>(head_dim)
            });

        const at::Tensor result = at::scaled_dot_product_attention(
            q, expanded_key, expanded_value, std::optional<at::Tensor>(mask),
            0.0, false, std::optional<double>(scale), false
        );
        out.copy_(result);
        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[gemma4_listener_aten] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[gemma4_listener_aten] %s\n", error.what());
        return -3;
    }
}

extern "C" int gemma4_listener_debug_attention_gqa(
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
) {
    try {
        if (!query || !key || !value || !attn_mask || !output || !std::isfinite(scale)) return -1;
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) return -2;
        if (!validate_bridge_meta(query, query_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(key, key_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(value, value_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(output, output_meta, GEMMA4_META_BF16, 4, device) ||
            !validate_bridge_meta(
                attn_mask, attn_mask_meta, attn_mask_meta ? attn_mask_meta->dtype : 0, 4, device)) {
            return -1;
        }
        if (attn_mask_meta->dtype != GEMMA4_META_BF16 &&
            attn_mask_meta->dtype != GEMMA4_META_FP32 &&
            attn_mask_meta->dtype != GEMMA4_META_BOOL) return -1;

        const uint64_t batch = query_meta->dims[0];
        const uint64_t query_heads = query_meta->dims[1];
        const uint64_t query_len = query_meta->dims[2];
        const uint64_t head_dim = query_meta->dims[3];
        const uint64_t key_heads = key_meta->dims[1];
        const uint64_t key_len = key_meta->dims[2];
        if (batch == 0 || query_heads != GEMMA4_NUM_ATTN_HEADS ||
            key_heads != GEMMA4_NUM_KV_HEADS || head_dim != 512 ||
            key_meta->dims[0] != batch || value_meta->dims[0] != batch ||
            value_meta->dims[1] != key_heads || key_meta->dims[2] != value_meta->dims[2] ||
            key_meta->dims[3] != head_dim || value_meta->dims[3] != head_dim ||
            output_meta->dims[0] != batch || output_meta->dims[1] != query_heads ||
            output_meta->dims[2] != query_len || output_meta->dims[3] != head_dim ||
            !broadcast_dimension(attn_mask_meta->dims[0], batch) ||
            !broadcast_dimension(attn_mask_meta->dims[1], query_heads) ||
            attn_mask_meta->dims[2] != query_len || attn_mask_meta->dims[3] != key_len) {
            return -1;
        }

        const c10::cuda::CUDAStream external_stream =
            c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);

        // Only Q is normalized to contiguous layout.  K/V and the explicit
        // mask remain borrowed views so GQA's long-K behavior is observable.
        const at::Tensor q = borrowed_tensor(query, query_meta, device).contiguous();
        const at::Tensor k = borrowed_tensor(key, key_meta, device);
        const at::Tensor v = borrowed_tensor(value, value_meta, device);
        const at::Tensor mask = borrowed_tensor(attn_mask, attn_mask_meta, device);
        const at::Tensor out = borrowed_tensor(output, output_meta, device);
        const at::Tensor result = at::scaled_dot_product_attention(
            q, k, v, std::optional<at::Tensor>(mask),
            0.0, false, std::optional<double>(scale), true
        );
        out.copy_(result);
        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[gemma4_listener_aten] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[gemma4_listener_aten] %s\n", error.what());
        return -3;
    }
}
