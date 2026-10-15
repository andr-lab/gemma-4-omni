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
#include <ATen/ops/mean.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/pow.h>
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
        if (!input || !output || rows == 0 || dim == 0 ||
            eps != eps || eps < 0.0 || eps > 3.402823466e38) return -1;

        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            !is_current_device_pointer(input, device) ||
            (weight && !is_current_device_pointer(weight, device)) ||
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
        const at::Tensor weight_bf16 = weight
            ? at::from_blob(const_cast<void *>(weight), at::IntArrayRef(weight_size, 1), options)
            : at::Tensor();
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
        const at::Tensor weighted = weight
            ? at::mul(normalized, weight_bf16.to(at::kFloat))
            : normalized;
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

extern "C" int gemma4_aten_rmsnorm_preallocated_async(
    const void *input,
    const void *weight,
    void *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    const gemma4_aten_rms_workspace *workspace,
    const void *cached_weight_fp32,
    cudaStream_t stream
) {
    try {
        if (!input || !output || !workspace || rows == 0 || dim == 0 ||
            eps != eps || eps < 0.0 || eps > 3.402823466e38 ||
            rows > workspace->max_rows || dim > workspace->max_dim ||
            static_cast<size_t>(rows) > SIZE_MAX / static_cast<size_t>(dim) ||
            static_cast<size_t>(rows) * static_cast<size_t>(dim) > workspace->max_elements) {
            return -1;
        }

        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            !is_current_device_pointer(input, device) ||
            (weight && !is_current_device_pointer(weight, device)) ||
            !is_current_device_pointer(output, device)) return -2;
        const void *workspace_ptrs[] = {
            workspace->input_fp32,
            workspace->squared_fp32,
            workspace->mean_fp32,
            workspace->mean_plus_eps_fp32,
            workspace->scale_fp32,
            workspace->normalized_fp32,
            workspace->weighted_fp32,
            workspace->weight_fp32,
        };
        for (const void *ptr : workspace_ptrs) {
            if (!is_current_device_pointer(ptr, device)) return -2;
        }
        if (cached_weight_fp32 && !is_current_device_pointer(cached_weight_fp32, device)) return -2;

        const auto external_stream = c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);
        const auto bf16_options = at::TensorOptions()
            .dtype(at::kBFloat16)
            .device(c10::DeviceType::CUDA, device);
        const auto fp32_options = at::TensorOptions()
            .dtype(at::kFloat)
            .device(c10::DeviceType::CUDA, device);
        const int64_t sizes[] = {1, static_cast<int64_t>(rows), static_cast<int64_t>(dim)};
        const int64_t mean_sizes[] = {1, static_cast<int64_t>(rows), 1};
        const int64_t weight_size[] = {static_cast<int64_t>(dim)};
        const int64_t last_dim[] = {-1};
        const at::Tensor input_bf16 = at::from_blob(
            const_cast<void *>(input), at::IntArrayRef(sizes, 3), bf16_options);
        const at::Tensor weight_bf16 = weight
            ? at::from_blob(const_cast<void *>(weight), at::IntArrayRef(weight_size, 1), bf16_options)
            : at::Tensor();
        const at::Tensor output_bf16 = at::from_blob(
            output, at::IntArrayRef(sizes, 3), bf16_options);
        const at::Tensor input_fp32 = at::from_blob(
            workspace->input_fp32, at::IntArrayRef(sizes, 3), fp32_options);
        const at::Tensor squared = at::from_blob(
            workspace->squared_fp32, at::IntArrayRef(sizes, 3), fp32_options);
        const at::Tensor mean = at::from_blob(
            workspace->mean_fp32, at::IntArrayRef(mean_sizes, 3), fp32_options);
        const at::Tensor mean_plus_eps = at::from_blob(
            workspace->mean_plus_eps_fp32, at::IntArrayRef(mean_sizes, 3), fp32_options);
        const at::Tensor scale = at::from_blob(
            workspace->scale_fp32, at::IntArrayRef(mean_sizes, 3), fp32_options);
        const at::Tensor normalized = at::from_blob(
            workspace->normalized_fp32, at::IntArrayRef(sizes, 3), fp32_options);
        const at::Tensor weighted = at::from_blob(
            workspace->weighted_fp32, at::IntArrayRef(sizes, 3), fp32_options);
        const at::Tensor workspace_weight_fp32 = at::from_blob(
            workspace->weight_fp32, at::IntArrayRef(weight_size, 1), fp32_options);
        const at::Tensor cached_weight = cached_weight_fp32
            ? at::from_blob(const_cast<void *>(cached_weight_fp32), at::IntArrayRef(weight_size, 1), fp32_options)
            : at::Tensor();

        // Keep the exact ShapeStable operation order while directing every
        // result into context-owned external storage.
        input_fp32.copy_(input_bf16);
        at::pow_out(const_cast<at::Tensor &>(squared), input_fp32, 2);
        at::mean_out(
            const_cast<at::Tensor &>(mean), squared,
            at::IntArrayRef(last_dim, 1), true, at::kFloat);
        at::add_out(const_cast<at::Tensor &>(mean_plus_eps), mean, eps);
        at::pow_out(const_cast<at::Tensor &>(scale), mean_plus_eps, -0.5);
        at::mul_out(const_cast<at::Tensor &>(normalized), input_fp32, scale);
        if (weight) {
            const at::Tensor &weight_fp32 = cached_weight_fp32
                ? cached_weight
                : workspace_weight_fp32;
            if (!cached_weight_fp32) {
                workspace_weight_fp32.copy_(weight_bf16);
            }
            at::mul_out(const_cast<at::Tensor &>(weighted), normalized, weight_fp32);
            output_bf16.copy_(weighted);
        } else {
            output_bf16.copy_(normalized);
        }
        if (cudaPeekAtLastError() != cudaSuccess) return -4;
        return 0;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[gemma4_aten_norm_preallocated] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[gemma4_aten_norm_preallocated] %s\n", error.what());
        return -3;
    }
}

extern "C" int gemma4_aten_rmsnorm_cache_weight_async(
    const void *weight_bf16,
    void *weight_fp32,
    uint32_t dim,
    cudaStream_t stream
) {
    try {
        if (!weight_bf16 || !weight_fp32 || dim == 0) return -1;
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            !is_current_device_pointer(weight_bf16, device) ||
            !is_current_device_pointer(weight_fp32, device)) return -2;
        const auto external_stream = c10::cuda::getStreamFromExternal(stream, device);
        c10::cuda::CUDAStreamGuard stream_guard(external_stream);
        const auto bf16_options = at::TensorOptions()
            .dtype(at::kBFloat16)
            .device(c10::DeviceType::CUDA, device);
        const auto fp32_options = at::TensorOptions()
            .dtype(at::kFloat)
            .device(c10::DeviceType::CUDA, device);
        const int64_t sizes[] = {static_cast<int64_t>(dim)};
        const at::Tensor source = at::from_blob(
            const_cast<void *>(weight_bf16), at::IntArrayRef(sizes, 1), bf16_options);
        const at::Tensor destination = at::from_blob(weight_fp32, at::IntArrayRef(sizes, 1), fp32_options);
        destination.copy_(source);
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -4;
    } catch (const c10::Error &error) {
        std::fprintf(stderr, "[gemma4_aten_norm_cache_weight] %s\n", error.what());
        return -3;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "[gemma4_aten_norm_cache_weight] %s\n", error.what());
        return -3;
    }
}
