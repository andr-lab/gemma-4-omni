/*
 * gemma4_listener_aten_norm.cpp — literal ATen RMSNorm parity oracle.
 *
 * This deliberately uses ordinary ATen tensor expressions instead of a
 * fused/generic RMSNorm operator. The operation order is the ShapeStable
 * Python contract and all work is enqueued on the caller's CUDA stream.
 * ATen's external-storage contract is documented at:
 * https://docs.pytorch.org/cppdocs/api/aten/creation.html
 */
#include "gemma4_listener_aten_norm.h"

#include <ATen/ops/add.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/mul.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <cstdio>
#include <exception>

namespace {

bool is_current_device_pointer(const void *ptr, int device) {
    if (!ptr || (reinterpret_cast<uintptr_t>(ptr) & 1u) != 0u) return false;
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, ptr) != cudaSuccess) return false;
    return attributes.type == cudaMemoryTypeDevice && attributes.device == device;
}

} // namespace

extern "C" int gemma4_aten_rmsnorm_parity_async(
    const void *input,
    const void *weight,
    void *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    cudaStream_t stream
) {
    try {
        if (!input || !weight || !output || rows == 0 || dim == 0 ||
            eps != eps || eps < 0.0 || eps > 3.402823466e38) return -1;

        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            !is_current_device_pointer(input, device) ||
            !is_current_device_pointer(weight, device) ||
            !is_current_device_pointer(output, device)) return -2;

        const auto external_stream = c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);
        const auto options = at::TensorOptions()
            .dtype(at::kBFloat16)
            .device(c10::DeviceType::CUDA, device);
        const int64_t sizes[] = {1, static_cast<int64_t>(rows), static_cast<int64_t>(dim)};
        const int64_t weight_size[] = {static_cast<int64_t>(dim)};
        const at::Tensor input_bf16 = at::from_blob(
            const_cast<void *>(input), at::IntArrayRef(sizes, 3), options);
        const at::Tensor weight_bf16 = at::from_blob(
            const_cast<void *>(weight), at::IntArrayRef(weight_size, 1), options);
        const at::Tensor output_bf16 = at::from_blob(
            output, at::IntArrayRef(sizes, 3), options);

        // Keep this sequence textually aligned with
        // ShapeStableGemmaListenerLM._padded_rmsnorm. In particular, do not
        // replace it with torch.rms_norm or a fused operator.
        const at::Tensor input_fp32 = input_bf16.to(at::kFloat);
        const at::Tensor x_squared = input_fp32.pow(2);
        const at::Tensor mean_squared = x_squared.mean(-1, true);
        const at::Tensor mean_plus_eps = at::add(mean_squared, eps);
        const at::Tensor scale = mean_plus_eps.pow(-0.5);
        const at::Tensor normalized = at::mul(input_fp32, scale);
        const at::Tensor weighted = at::mul(normalized, weight_bf16.to(at::kFloat));
        output_bf16.copy_(weighted.to(at::kBFloat16));
        if (cudaPeekAtLastError() != cudaSuccess) return -4;
        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[gemma4_aten_norm] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[gemma4_aten_norm] %s\n", error.what());
        return -3;
    }
}
