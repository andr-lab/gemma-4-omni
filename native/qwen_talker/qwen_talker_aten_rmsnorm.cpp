/*
 * Exact ATen CUDA-graph bridge for Qwen hidden-size RMSNorm.
 *
 * This path exists only for [1,2,2048] hidden RMSNorm calls.
 * Q/K head-dim=128 RMSNorm remains in qwen_talker_kernels.cu.
 */
#include "qwen_talker_aten_rmsnorm.h"

#include <ATen/ATen.h>
#include <ATen/ops/from_blob.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <new>
#include <stdexcept>

struct qwen_talker_rmsnorm_graph_state {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    cudaStream_t capture_stream = nullptr;
    cudaEvent_t input_ready = nullptr;

    int device = -1;

    void *staging_input_bf16 = nullptr;
    void *staging_weight_bf16 = nullptr;
    void *staging_output_bf16 = nullptr;

    /*
     * Borrowed staging tensors.
     */
    at::Tensor input_bf16;
    at::Tensor weight_bf16;
    at::Tensor output_bf16;

    /*
     * Keep every graph-owned ATen result alive for the lifetime of the
     * executable.  The CUDA graph references these stable allocations.
     */
    at::Tensor input_fp32;
    at::Tensor squared_fp32;
    at::Tensor variance_fp32;
    at::Tensor variance_eps_fp32;
    at::Tensor inv_rms_fp32;
    at::Tensor normalized_fp32;
    at::Tensor normalized_bf16;
    at::Tensor result_bf16;

    bool captured = false;
    uint32_t rows = 2u;
    double epsilon = 0.0;
};

namespace {

constexpr size_t BF16_BYTES = 2u;
constexpr size_t WEIGHT_ELEMENTS =
    static_cast<size_t>(QWEN_NATIVE_HIDDEN_SIZE);

bool device_pointer(const void *ptr, int device) {
    if (!ptr) return false;

    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) {
        return false;
    }

    return attributes.type == cudaMemoryTypeDevice &&
           attributes.device == device;
}

at::Tensor borrowed_bf16(
    void *ptr,
    at::IntArrayRef sizes,
    int device
) {
    return at::from_blob(
        ptr,
        sizes,
        at::TensorOptions()
            .dtype(at::kBFloat16)
            .device(c10::DeviceType::CUDA, device)
    );
}

/*
 * IMPORTANT:
 *
 * This deliberately reproduces the Qwen3TTSRMSNorm Python forward expression,
 * not an independently derived RMS formula and not a custom reduction tree.
 *
 * Reference:
 *
 *   input_dtype = hidden_states.dtype
 *   hidden_states = hidden_states.to(torch.float32)
 *   variance = hidden_states.pow(2).mean(-1, keepdim=True)
 *   hidden_states =
 *       hidden_states * torch.rsqrt(variance + variance_epsilon)
 *   return weight * hidden_states.to(input_dtype)
 */
int capture_exact_qwen_rmsnorm_body(
    qwen_talker_rmsnorm_graph_state *state,
    double epsilon
) {
    try {
        const c10::cuda::CUDAStream external =
            c10::cuda::getStreamFromExternal(
                state->capture_stream,
                state->device
            );

        c10::cuda::CUDAStreamGuard guard(external);

        state->input_fp32 =
            state->input_bf16.to(at::kFloat);

        state->squared_fp32 =
            state->input_fp32.pow(2);

        state->variance_fp32 =
            state->squared_fp32.mean({-1}, true);

        state->variance_eps_fp32 = state->variance_fp32 + epsilon;

        state->inv_rms_fp32 =
            state->variance_eps_fp32.rsqrt();

        state->normalized_fp32 =
            state->input_fp32 * state->inv_rms_fp32;

        state->normalized_bf16 =
            state->normalized_fp32.to(at::kBFloat16);

        /*
         * Preserve Qwen operand order:
         *
         *     weight * hidden_states.to(input_dtype)
         */
        state->result_bf16 =
            state->weight_bf16 * state->normalized_bf16;

        state->output_bf16.copy_(state->result_bf16);

        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(
            stderr,
            "[qwen_rmsnorm_graph] ATen capture body failed: %s\n",
            error.what()
        );
        return -1;
    } catch (const std::exception &error) {
        std::fprintf(
            stderr,
            "[qwen_rmsnorm_graph] capture body failed: %s\n",
            error.what()
        );
        return -1;
    }
}

} // namespace

extern "C" int qwen_talker_rmsnorm_graph_create(
    qwen_talker_rmsnorm_graph_state **out_state,
    uint32_t rows,
    void *staging_input_bf16,
    void *staging_weight_bf16,
    void *staging_output_bf16
) {
    if (!out_state) return -1;
    *out_state = nullptr;

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return -2;

    if ((rows != 1u && rows != 2u) ||
        !device_pointer(staging_input_bf16, device) ||
        !device_pointer(staging_weight_bf16, device) ||
        !device_pointer(staging_output_bf16, device)) {
        return -3;
    }

    auto *state =
        new (std::nothrow) qwen_talker_rmsnorm_graph_state();

    if (!state) return -4;

    state->device = device;
    state->rows = rows;
    state->staging_input_bf16 = staging_input_bf16;
    state->staging_weight_bf16 = staging_weight_bf16;
    state->staging_output_bf16 = staging_output_bf16;

    if (cudaStreamCreateWithFlags(
            &state->capture_stream,
            cudaStreamNonBlocking
        ) != cudaSuccess ||
        cudaEventCreateWithFlags(
            &state->input_ready,
            cudaEventDisableTiming
        ) != cudaSuccess) {
        qwen_talker_rmsnorm_graph_destroy(state);
        return -5;
    }

    try {
        const int64_t input_sizes[] = {
            1,
            static_cast<int64_t>(rows),
            static_cast<int64_t>(QWEN_NATIVE_HIDDEN_SIZE)
        };

        const int64_t weight_sizes[] = {
            static_cast<int64_t>(QWEN_NATIVE_HIDDEN_SIZE)
        };

        state->input_bf16 = borrowed_bf16(
            staging_input_bf16,
            at::IntArrayRef(input_sizes, 3),
            device
        );

        state->weight_bf16 = borrowed_bf16(
            staging_weight_bf16,
            at::IntArrayRef(weight_sizes, 1),
            device
        );

        state->output_bf16 = borrowed_bf16(
            staging_output_bf16,
            at::IntArrayRef(input_sizes, 3),
            device
        );
    } catch (const c10::Error &error) {
        std::fprintf(
            stderr,
            "[qwen_rmsnorm_graph] staging tensor creation failed: %s\n",
            error.what()
        );
        qwen_talker_rmsnorm_graph_destroy(state);
        return -6;
    } catch (const std::exception &error) {
        std::fprintf(
            stderr,
            "[qwen_rmsnorm_graph] staging tensor creation failed: %s\n",
            error.what()
        );
        qwen_talker_rmsnorm_graph_destroy(state);
        return -6;
    }

    *out_state = state;
    return 0;
}

extern "C" int qwen_talker_rmsnorm_graph_run(
    qwen_talker_rmsnorm_graph_state *state,
    const void *input_bf16,
    const void *weight_bf16,
    void *output_bf16,
    float *row_mean_output,
    double epsilon,
    cudaStream_t stream
) {
    if (!state ||
        !input_bf16 ||
        !weight_bf16 ||
        !output_bf16 ||
        !row_mean_output ||
        !std::isfinite(epsilon) ||
        epsilon <= 0.0) {
        return -1;
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess ||
        device != state->device) {
        return -2;
    }

    if (!device_pointer(input_bf16, device) ||
        !device_pointer(weight_bf16, device) ||
        !device_pointer(output_bf16, device) ||
        !device_pointer(row_mean_output, device)) {
        return -3;
    }

    if (state->captured && epsilon != state->epsilon) {
        /*
         * The scalar epsilon is captured into the graph.
         * A changed epsilon requires a different graph and therefore fails
         * closed instead of silently replaying stale semantics.
         */
        return -4;
    }

    const size_t input_bytes =
        static_cast<size_t>(state->rows) * QWEN_NATIVE_HIDDEN_SIZE * BF16_BYTES;

    const size_t weight_bytes =
        WEIGHT_ELEMENTS * BF16_BYTES;

    /*
     * Stable pointer staging.  These copies are intentionally outside the
     * graph so one captured graph can be reused for all 28 layers and the
     * final model norm while each call supplies a different weight pointer.
     */
    if (cudaMemcpyAsync(
            state->staging_input_bf16,
            input_bf16,
            input_bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ) != cudaSuccess ||
        cudaMemcpyAsync(
            state->staging_weight_bf16,
            weight_bf16,
            weight_bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ) != cudaSuccess) {
        return -5;
    }

    if (!state->captured) {
        if (cudaEventRecord(
                state->input_ready,
                stream
            ) != cudaSuccess ||
            cudaStreamWaitEvent(
                state->capture_stream,
                state->input_ready,
                0
            ) != cudaSuccess) {
            return -6;
        }

        const cudaError_t begin_status =
            cudaStreamBeginCapture(
                state->capture_stream,
                cudaStreamCaptureModeThreadLocal
            );

        if (begin_status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[qwen_rmsnorm_graph] begin capture failed: %s\n",
                cudaGetErrorString(begin_status)
            );
            return -7;
        }

        const int body_status =
            capture_exact_qwen_rmsnorm_body(
                state,
                epsilon
            );

        cudaGraph_t graph = nullptr;
        const cudaError_t end_status =
            cudaStreamEndCapture(
                state->capture_stream,
                &graph
            );

        if (body_status != 0 ||
            end_status != cudaSuccess ||
            !graph) {
            std::fprintf(
                stderr,
                "[qwen_rmsnorm_graph] capture failed: body=%d end=%s\n",
                body_status,
                cudaGetErrorString(end_status)
            );

            if (graph) cudaGraphDestroy(graph);
            return -8;
        }

        cudaGraphExec_t executable = nullptr;

        const cudaError_t instantiate_status =
            cudaGraphInstantiate(
                &executable,
                graph,
                nullptr,
                nullptr,
                0
            );

        if (instantiate_status != cudaSuccess ||
            !executable) {
            std::fprintf(
                stderr,
                "[qwen_rmsnorm_graph] instantiate failed: %s\n",
                cudaGetErrorString(instantiate_status)
            );

            cudaGraphDestroy(graph);
            return -9;
        }

        state->graph = graph;
        state->executable = executable;
        state->epsilon = epsilon;
        state->captured = true;
    }

    /*
     * Launch on the caller's stream.
     *
     * This preserves strict ordering with:
     *
     *   input D2D copies
     *   graph
     *   output D2D copies
     *
     * and avoids steady-state cross-stream events.
     */
    const cudaError_t launch_status =
        cudaGraphLaunch(
            state->executable,
            stream
        );

    if (launch_status != cudaSuccess) {
        std::fprintf(
            stderr,
            "[qwen_rmsnorm_graph] launch failed: %s\n",
            cudaGetErrorString(launch_status)
        );
        return -10;
    }

    if (cudaMemcpyAsync(
            output_bf16,
            state->staging_output_bf16,
            input_bytes,
            cudaMemcpyDeviceToDevice,
            stream
        ) != cudaSuccess) {
        return -11;
    }

    /*
     * Preserve the existing debug ABI.
     *
     * variance_fp32 has shape [1,2,1], so its first two FP32 values are the
     * two row means formerly stored at ctx->row_stats + 2.
     */
    if (!state->variance_fp32.defined() ||
        state->variance_fp32.numel() != state->rows) {
        return -12;
    }

    if (cudaMemcpyAsync(
            row_mean_output,
            state->variance_fp32.data_ptr<float>(),
            static_cast<size_t>(state->rows) * sizeof(float),
            cudaMemcpyDeviceToDevice,
            stream
        ) != cudaSuccess) {
        return -13;
    }

    return 0;
}

extern "C" void qwen_talker_rmsnorm_graph_destroy(
    qwen_talker_rmsnorm_graph_state *state
) {
    if (!state) return;

    if (state->capture_stream) {
        cudaStreamSynchronize(state->capture_stream);
    }

    if (state->executable) {
        cudaGraphExecDestroy(state->executable);
    }

    if (state->graph) {
        cudaGraphDestroy(state->graph);
    }

    if (state->input_ready) {
        cudaEventDestroy(state->input_ready);
    }

    if (state->capture_stream) {
        cudaStreamDestroy(state->capture_stream);
    }

    delete state;
}

extern "C" int qwen_talker_rmsnorm_graph_captured(
    const qwen_talker_rmsnorm_graph_state *state
) {
    return state && state->captured ? 1 : 0;
}
