/* ATen SDPA bridge for Qwen Q=2 GQA attention. */
#include "qwen_talker_aten_attention.h"

#include <ATen/ops/from_blob.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <optional>
#include <stdexcept>

struct qwen_talker_attention_graph_state {
    cudaGraph_t graph[2]{};
    cudaGraphExec_t executable[2]{};
    cudaStream_t stream = nullptr;
    cudaEvent_t input_ready = nullptr;
    cudaEvent_t output_ready = nullptr;
    bool captured[2]{};
};

namespace {

bool device_pointer(const void *ptr, int device) {
    if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 1u) != 0u) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) return false;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}

bool valid_meta(
    const void *ptr,
    const qwen_tensor_meta *meta,
    uint32_t dtype,
    uint32_t ndim,
    int device
) {
    if (!ptr || !meta || meta->ptr != ptr || meta->dtype != dtype || meta->ndim != ndim ||
        meta->contiguous != 1u || meta->reserved != 0u || !device_pointer(ptr, device)) {
        return false;
    }
    uint64_t running = 1;
    for (int index = static_cast<int>(ndim) - 1; index >= 0; --index) {
        if (meta->dims[index] == 0 || meta->dims[index] > UINT64_MAX / running ||
            meta->strides[index] != running) return false;
        running *= meta->dims[index];
    }
    for (uint32_t index = ndim; index < QWEN_NATIVE_META_MAX_RANK; ++index) {
        if (meta->dims[index] != 0 || meta->strides[index] != 0) return false;
    }
    return true;
}

at::Tensor borrowed(
    const void *ptr,
    const qwen_tensor_meta *meta,
    int device
) {
    int64_t sizes[QWEN_NATIVE_META_MAX_RANK] = {};
    int64_t strides[QWEN_NATIVE_META_MAX_RANK] = {};
    for (uint32_t index = 0; index < meta->ndim; ++index) {
        if (meta->dims[index] > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            throw std::invalid_argument("Qwen attention metadata exceeds int64");
        }
        sizes[index] = static_cast<int64_t>(meta->dims[index]);
        strides[index] = static_cast<int64_t>(meta->strides[index]);
    }
    return at::from_blob(
        const_cast<void *>(ptr),
        at::IntArrayRef(sizes, meta->ndim),
        at::IntArrayRef(strides, meta->ndim),
        at::TensorOptions().dtype(at::kBFloat16).device(c10::DeviceType::CUDA, device)
    );
}

qwen_tensor_meta graph_meta(
    const void *ptr,
    uint32_t dtype,
    const uint64_t *dims,
    uint32_t ndim
) {
    qwen_tensor_meta meta{};
    meta.ptr = ptr;
    meta.dtype = dtype;
    meta.ndim = ndim;
    meta.contiguous = 1u;
    uint64_t stride = 1u;
    for (int index = static_cast<int>(ndim) - 1; index >= 0; --index) {
        meta.dims[index] = dims[index];
        meta.strides[index] = stride;
        stride *= dims[index];
    }
    const size_t bytes = dtype == QWEN_NATIVE_META_BF16 ? 2u : sizeof(float);
    meta.capacity_bytes = stride * bytes;
    return meta;
}

} // namespace

extern "C" int qwen_talker_attention_gqa(
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
) {
    try {
        if (!query || !key || !value || !attention_mask || !output || !std::isfinite(scale)) return -1;
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) return -2;
        if (!valid_meta(query, query_meta, QWEN_NATIVE_META_BF16, 4u, device) ||
            !valid_meta(key, key_meta, QWEN_NATIVE_META_BF16, 4u, device) ||
            !valid_meta(value, value_meta, QWEN_NATIVE_META_BF16, 4u, device) ||
            !valid_meta(attention_mask, attention_mask_meta, QWEN_NATIVE_META_BF16, 4u, device) ||
            !valid_meta(output, output_meta, QWEN_NATIVE_META_BF16, 4u, device)) return -1;

        const uint64_t batch = query_meta->dims[0];
        const uint64_t query_heads = query_meta->dims[1];
        const uint64_t query_len = query_meta->dims[2];
        const uint64_t head_dim = query_meta->dims[3];
        const uint64_t key_heads = key_meta->dims[1];
        const uint64_t key_len = key_meta->dims[2];
        if (batch != 1u || query_heads != QWEN_NATIVE_NUM_ATTN_HEADS ||
            (query_len != 1u && query_len != 2u) || head_dim != QWEN_NATIVE_HEAD_DIM ||
            (key_heads != QWEN_NATIVE_NUM_KV_HEADS && key_heads != query_heads) ||
            key_meta->dims[0] != batch || value_meta->dims[0] != batch ||
            value_meta->dims[1] != key_heads || value_meta->dims[2] != key_len ||
            value_meta->dims[3] != head_dim || output_meta->dims[0] != batch ||
            output_meta->dims[1] != query_heads || output_meta->dims[2] != query_len ||
            output_meta->dims[3] != head_dim || attention_mask_meta->dims[0] != batch ||
            attention_mask_meta->dims[1] != 1u || attention_mask_meta->dims[2] != query_len ||
            attention_mask_meta->dims[3] != key_len) return -1;

        const c10::cuda::CUDAStream external_stream =
            c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);
        const at::Tensor q = borrowed(query, query_meta, device);
        const at::Tensor k = borrowed(key, key_meta, device);
        const at::Tensor v = borrowed(value, value_meta, device);
        const at::Tensor mask = borrowed(attention_mask, attention_mask_meta, device);
        const at::Tensor out = borrowed(output, output_meta, device);

        /* This is the public exact GQA contract used by the current Qwen
         * oracle: BF16 Q/K/V, additive BF16 mask, no dropout, explicit scale. */
        const at::Tensor result = at::scaled_dot_product_attention(
            q, k, v, std::optional<at::Tensor>(mask),
            0.0, false, std::optional<double>(scale), key_heads != query_heads
        );
        out.copy_(result);
        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[qwen_talker_attention] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[qwen_talker_attention] %s\n", error.what());
        return -3;
    }
}

extern "C" int qwen_talker_attention_graph_create(
    qwen_talker_attention_graph_state **out_state
) {
    if (!out_state) return -1;
    *out_state = new (std::nothrow) qwen_talker_attention_graph_state();
    if (!*out_state) return -2;
    qwen_talker_attention_graph_state *state = *out_state;
    if (cudaStreamCreateWithFlags(&state->stream, cudaStreamNonBlocking) != cudaSuccess ||
        cudaEventCreateWithFlags(&state->input_ready, cudaEventDisableTiming) != cudaSuccess ||
        cudaEventCreateWithFlags(&state->output_ready, cudaEventDisableTiming) != cudaSuccess) {
        qwen_talker_attention_graph_destroy(state);
        *out_state = nullptr;
        return -3;
    }
    return 0;
}

extern "C" int qwen_talker_attention_graph_run(
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
) {
    if (!state || !query || !key || !value || !attention_mask || !output ||
        (query_len != 1u && query_len != 2u) || max_key_len == 0u ||
        !std::isfinite(scale)) return -1;
    if (!state->stream || !state->input_ready || !state->output_ready) return -2;
    if (cudaEventRecord(state->input_ready, stream) != cudaSuccess ||
        cudaStreamWaitEvent(state->stream, state->input_ready, 0) != cudaSuccess) return -2;

    const uint32_t graph_index = query_len - 1u;
    const uint64_t query_dims[] = {
        1u, QWEN_NATIVE_NUM_ATTN_HEADS, query_len, QWEN_NATIVE_HEAD_DIM
    };
    const uint64_t key_dims[] = {
        1u, QWEN_NATIVE_NUM_ATTN_HEADS, max_key_len, QWEN_NATIVE_HEAD_DIM
    };
    const uint64_t mask_dims[] = {1u, 1u, query_len, max_key_len};
    const qwen_tensor_meta query_meta = graph_meta(
        query, QWEN_NATIVE_META_BF16, query_dims, 4u);
    const qwen_tensor_meta key_meta = graph_meta(
        key, QWEN_NATIVE_META_BF16, key_dims, 4u);
    const qwen_tensor_meta value_meta = graph_meta(
        value, QWEN_NATIVE_META_BF16, key_dims, 4u);
    const qwen_tensor_meta mask_meta = graph_meta(
        attention_mask, QWEN_NATIVE_META_BF16, mask_dims, 4u);
    const qwen_tensor_meta output_meta = graph_meta(
        output, QWEN_NATIVE_META_BF16, query_dims, 4u);

    cudaGraph_t graph = nullptr;
    const cudaError_t begin_status = state->captured[graph_index] ? cudaSuccess :
        cudaStreamBeginCapture(state->stream, cudaStreamCaptureModeThreadLocal);
    if (begin_status != cudaSuccess) {
        std::fprintf(stderr, "[qwen_attention_graph] begin capture failed: %s\\n",
                     cudaGetErrorString(begin_status));
        return -3;
    }
    const int bridge_status = state->captured[graph_index] ? 0 : qwen_talker_attention_gqa(
        query, &query_meta, key, &key_meta, value, &value_meta,
        attention_mask, &mask_meta, output, &output_meta, scale, state->stream);
    const cudaError_t end_status = state->captured[graph_index] ? cudaSuccess :
        cudaStreamEndCapture(state->stream, &graph);
    if (!state->captured[graph_index] && (bridge_status != 0 || end_status != cudaSuccess || !graph)) {
        std::fprintf(stderr, "[qwen_attention_graph] capture failed: bridge=%d end=%s\\n",
                     bridge_status, cudaGetErrorString(end_status));
        if (graph) cudaGraphDestroy(graph);
        return bridge_status != 0 ? -4 : -5;
    }
    if (!state->captured[graph_index]) {
        cudaGraphExec_t executable = nullptr;
        const cudaError_t instantiate_status = cudaGraphInstantiate(
            &executable, graph, nullptr, nullptr, 0);
        if (instantiate_status != cudaSuccess || !executable) {
            std::fprintf(stderr, "[qwen_attention_graph] instantiate failed: %s\\n",
                         cudaGetErrorString(instantiate_status));
            cudaGraphDestroy(graph);
            return -6;
        }
        state->graph[graph_index] = graph;
        state->executable[graph_index] = executable;
        state->captured[graph_index] = true;
    }
    const cudaError_t launch_status = cudaGraphLaunch(state->executable[graph_index], state->stream);
    if (launch_status != cudaSuccess ||
        cudaEventRecord(state->output_ready, state->stream) != cudaSuccess ||
        cudaStreamWaitEvent(stream, state->output_ready, 0) != cudaSuccess) {
        std::fprintf(stderr, "[qwen_attention_graph] launch failed: %s\\n",
                     cudaGetErrorString(launch_status));
        return -7;
    }
    return 0;
}

extern "C" void qwen_talker_attention_graph_destroy(
    qwen_talker_attention_graph_state *state
) {
    if (!state) return;
    if (state->stream) cudaStreamSynchronize(state->stream);
    for (uint32_t index = 0; index < 2u; ++index) {
        if (state->executable[index]) cudaGraphExecDestroy(state->executable[index]);
        if (state->graph[index]) cudaGraphDestroy(state->graph[index]);
    }
    if (state->input_ready) cudaEventDestroy(state->input_ready);
    if (state->output_ready) cudaEventDestroy(state->output_ready);
    if (state->stream) cudaStreamDestroy(state->stream);
    delete state;
}

extern "C" int qwen_talker_attention_graph_captured(
    const qwen_talker_attention_graph_state *state
) {
    return state && (state->captured[0] || state->captured[1]) ? 1 : 0;
}
