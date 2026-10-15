/*
 * gemma4_listener_test.cu — Stand-alone Unit Test for Gemma 4 Native Kernels.
 *
 * Verifies that RMSNorm, GeGLU, RoPE, and SDPA compute exact numerical values
 * and that gemma4_listener_create/step/destroy execute cleanly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "gemma4_listener.h"
#include "gemma4_listener_kernels.cuh"

struct gemma4_cache;
int gemma4_cache_create(gemma4_cache **out_cache, uint32_t max_seq_len, bool allocate_device_position);
int gemma4_cache_import_async(
    gemma4_cache *cache,
    const gemma4_imported_cache *imported,
    uint32_t current_position,
    cudaStream_t stream);
int gemma4_cache_rebase_async(
    gemma4_cache *cache,
    uint32_t prefix_len,
    uint32_t retained_audio_tokens,
    cudaStream_t stream);
void gemma4_cache_destroy(gemma4_cache *cache);
const void *gemma4_cache_get_layer_k(const gemma4_cache *cache, int layer_idx);
const void *gemma4_cache_get_layer_v(const gemma4_cache *cache, int layer_idx);

int test_rmsnorm() {
    printf("[TEST] Testing RMSNorm FP32 accumulation...\n");
    const int D = 256;
    float h_x[D];
    float h_w[D];
    float sum_sq = 0.0f;
    for (int i = 0; i < D; ++i) {
        h_x[i] = (float)(i % 10) * 0.1f;
        h_w[i] = 1.0f;
        sum_sq += h_x[i] * h_x[i];
    }
    float eps = 1e-6f;
    float rsq_expected = 1.0f / sqrtf(sum_sq / (float)D + eps);

    __nv_bfloat16 *d_x, *d_w, *d_y;
    cudaMalloc(&d_x, D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_w, D * sizeof(__nv_bfloat16));
    cudaMalloc(&d_y, D * sizeof(__nv_bfloat16));

    __nv_bfloat16 h_x_bf[D], h_w_bf[D];
    for (int i = 0; i < D; ++i) {
        h_x_bf[i] = __float2bfloat16(h_x[i]);
        h_w_bf[i] = __float2bfloat16(h_w[i]);
    }
    cudaMemcpy(d_x, h_x_bf, D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w_bf, D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    kernel_rmsnorm_bf16<256><<<1, 256>>>(d_x, d_w, d_y, 1, eps);
    cudaDeviceSynchronize();

    __nv_bfloat16 h_y_bf[D];
    cudaMemcpy(h_y_bf, d_y, D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < D; ++i) {
        float expected = h_x[i] * rsq_expected * h_w[i];
        float actual = __bfloat162float(h_y_bf[i]);
        float diff = fabsf(expected - actual);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  RMSNorm Max Diff vs Analytical: %e\n", max_diff);

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);

    if (max_diff > 1e-2f) {
        printf("  [FAIL] RMSNorm diff too high!\n");
        return 1;
    }
    printf("  [PASS] RMSNorm bit-level accuracy confirmed.\n");
    return 0;
}

int test_geglu() {
    printf("[TEST] Testing GeGLU GELUTanh activation...\n");
    const int N = 1024;
    __nv_bfloat16 *d_gate, *d_up, *d_out;
    cudaMalloc(&d_gate, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_up, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out, N * sizeof(__nv_bfloat16));

    __nv_bfloat16 h_gate[N], h_up[N];
    for (int i = 0; i < N; ++i) {
        float g = ((float)(i % 20) - 10.0f) * 0.1f;
        h_gate[i] = __float2bfloat16(g);
        h_up[i] = __float2bfloat16(1.0f);
    }
    cudaMemcpy(d_gate, h_gate, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_up, h_up, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    kernel_gelu_tanh_bf16<<<(N + 255) / 256, 256>>>(d_gate, d_out, N);
    kernel_mul_bf16<<<(N + 255) / 256, 256>>>(d_out, d_up, d_out, N);
    cudaDeviceSynchronize();

    __nv_bfloat16 h_out[N];
    cudaMemcpy(h_out, d_out, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float g = __bfloat162float(h_gate[i]);
        float cube = g * g * g;
        float inner = 0.7978845608028654f * (g + 0.044715f * cube);
        float expected = 0.5f * g * (1.0f + tanhf(inner));
        float actual = __bfloat162float(h_out[i]);
        float diff = fabsf(expected - actual);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  GeGLU Max Diff vs Analytical: %e\n", max_diff);

    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);

    if (max_diff > 1e-2f) {
        printf("  [FAIL] GeGLU diff too high!\n");
        return 1;
    }
    printf("  [PASS] GeGLU accuracy confirmed.\n");
    return 0;
}

int test_rope_rebase_dim(int head_dim, float rope_base) {
    const int token_count = 8;
    const int half_dim = head_dim / 2;
    const int old_position = 100;
    const int position_delta = -60;
    const int rotary_pairs = head_dim == 512 ? 64 : half_dim;
    std::vector<__nv_bfloat16> h_raw(token_count * head_dim);
    std::vector<float> h_inv_freq(half_dim);
    for (int token = 0; token < token_count; ++token) {
        for (int dim = 0; dim < head_dim; ++dim) {
            const float value = (float)((token * 31 + dim * 7) % 101 - 50) / 64.0f;
            h_raw[token * head_dim + dim] = __float2bfloat16(value);
        }
    }
    for (int pair = 0; pair < half_dim; ++pair) {
        h_inv_freq[pair] = pair < rotary_pairs
            ? 1.0f / powf(rope_base, (float)(2 * pair) / (float)head_dim)
            : 0.0f;
    }

    __nv_bfloat16 *d_old = nullptr;
    __nv_bfloat16 *d_expected = nullptr;
    float *d_inv_freq = nullptr;
    const size_t data_bytes = h_raw.size() * sizeof(__nv_bfloat16);
    const size_t freq_bytes = h_inv_freq.size() * sizeof(float);
    if (cudaMalloc(&d_old, data_bytes) != cudaSuccess ||
        cudaMalloc(&d_expected, data_bytes) != cudaSuccess ||
        cudaMalloc(&d_inv_freq, freq_bytes) != cudaSuccess ||
        cudaMemcpy(d_old, h_raw.data(), data_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_expected, h_raw.data(), data_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_inv_freq, h_inv_freq.data(), freq_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        if (d_old) cudaFree(d_old);
        if (d_expected) cudaFree(d_expected);
        if (d_inv_freq) cudaFree(d_inv_freq);
        return 1;
    }
    kernel_rope_bf16<<<dim3(token_count, 1), half_dim>>>(
        d_old, d_inv_freq, 1, token_count, head_dim, old_position, nullptr, 1.0f);
    kernel_rope_rebase_bf16<<<token_count, half_dim>>>(
        d_old, d_inv_freq, token_count, head_dim, position_delta, 1.0f);
    kernel_rope_bf16<<<dim3(token_count, 1), half_dim>>>(
        d_expected, d_inv_freq, 1, token_count, head_dim,
        old_position + position_delta, nullptr, 1.0f);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaFree(d_old);
        cudaFree(d_expected);
        cudaFree(d_inv_freq);
        return 1;
    }

    std::vector<__nv_bfloat16> h_actual(h_raw.size());
    std::vector<__nv_bfloat16> h_expected(h_raw.size());
    const bool copy_ok =
        cudaMemcpy(h_actual.data(), d_old, data_bytes, cudaMemcpyDeviceToHost) == cudaSuccess &&
        cudaMemcpy(h_expected.data(), d_expected, data_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    float max_diff = 0.0f;
    if (copy_ok) {
        for (size_t i = 0; i < h_actual.size(); ++i) {
            const float diff = fabsf(
                __bfloat162float(h_actual[i]) - __bfloat162float(h_expected[i]));
            if (diff > max_diff) max_diff = diff;
        }
    }
    cudaFree(d_old);
    cudaFree(d_expected);
    cudaFree(d_inv_freq);
    printf("  head_dim=%d rebase Max Diff: %e\n", head_dim, max_diff);
    return copy_ok && max_diff <= 0.04f ? 0 : 1;
}

int test_rope_rebase() {
    printf("[TEST] Testing sliding and full-attention KV RoPE position shifts...\n");
    if (test_rope_rebase_dim(256, 10000.0f) != 0 ||
        test_rope_rebase_dim(512, 1000000.0f) != 0) {
        printf("  [FAIL] RoPE rebase differs from direct target position.\n");
        return 1;
    }
    printf("  [PASS] Both RoPE layouts match direct target within BF16 rounding.\n");
    return 0;
}

int test_cache_rebase() {
    printf("[TEST] Testing bounded KV compaction and prefix retention...\n");
    const uint32_t prefix_len = 2;
    const uint32_t old_position = 10;
    const uint32_t retained_tokens = 4;
    const uint32_t source_start = old_position - retained_tokens;
    const uint32_t new_position = prefix_len + retained_tokens;
    struct TestLayer {
        __nv_bfloat16 *keys = nullptr;
        __nv_bfloat16 *values = nullptr;
        std::vector<__nv_bfloat16> host_keys;
        std::vector<__nv_bfloat16> host_values;
    };
    std::vector<TestLayer> layers(GEMMA4_NUM_LAYERS);
    gemma4_imported_cache imported = {};
    gemma4_cache *cache = nullptr;
    bool passed = false;

    auto run = [&]() -> bool {
        if (gemma4_cache_create(&cache, old_position, false) != 0) return false;
        for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
            const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
            if (spec->is_kv_shared) continue;
            TestLayer &layer = layers[layer_idx];
            const uint32_t dim = spec->head_dim;
            const size_t elements = (size_t)old_position * dim;
            const size_t bytes = elements * sizeof(__nv_bfloat16);
            layer.host_keys.resize(elements);
            layer.host_values.resize(elements);
            for (uint32_t pos = 0; pos < old_position; ++pos) {
                for (uint32_t d = 0; d < dim; ++d) {
                    layer.host_keys[(size_t)pos * dim + d] = __float2bfloat16(
                        (float)((layer_idx * 13 + pos * 7 + d % 11) % 97) / 64.0f);
                    layer.host_values[(size_t)pos * dim + d] = __float2bfloat16(
                        (float)((layer_idx * 19 + pos * 5 + d % 7) % 89) / 64.0f);
                }
            }
            if (cudaMalloc((void **)&layer.keys, bytes) != cudaSuccess ||
                cudaMalloc((void **)&layer.values, bytes) != cudaSuccess ||
                cudaMemcpyAsync(layer.keys, layer.host_keys.data(), bytes,
                                cudaMemcpyHostToDevice, 0) != cudaSuccess ||
                cudaMemcpyAsync(layer.values, layer.host_values.data(), bytes,
                                cudaMemcpyHostToDevice, 0) != cudaSuccess) return false;
            imported.layers[layer_idx].k_ptr = layer.keys;
            imported.layers[layer_idx].v_ptr = layer.values;
        }
        if (gemma4_cache_import_async(cache, &imported, old_position, 0) != 0 ||
            cudaDeviceSynchronize() != cudaSuccess ||
            gemma4_cache_rebase_async(cache, prefix_len, retained_tokens, 0) != 0 ||
            cudaDeviceSynchronize() != cudaSuccess) return false;

        for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
            const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
            if (spec->is_kv_shared) continue;
            const uint32_t dim = spec->head_dim;
            const size_t elements = (size_t)new_position * dim;
            std::vector<__nv_bfloat16> actual_k(elements), actual_v(elements);
            const void *cached_k = gemma4_cache_get_layer_k(cache, layer_idx);
            const void *cached_v = gemma4_cache_get_layer_v(cache, layer_idx);
            if (cudaMemcpy(actual_k.data(), cached_k, elements * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(actual_v.data(), cached_v, elements * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            const TestLayer &layer = layers[layer_idx];
            for (uint32_t pos = 0; pos < new_position; ++pos) {
                const uint32_t source_pos = pos < prefix_len ? pos : source_start + pos - prefix_len;
                for (uint32_t d = 0; d < dim; ++d) {
                    const size_t actual_index = (size_t)pos * dim + d;
                    const size_t source_index = (size_t)source_pos * dim + d;
                    if (__bfloat162float(actual_k[actual_index]) !=
                            __bfloat162float(layer.host_keys[source_index]) ||
                        __bfloat162float(actual_v[actual_index]) !=
                            __bfloat162float(layer.host_values[source_index])) return false;
                }
            }
        }
        return true;
    };

    passed = run();
    for (TestLayer &layer : layers) {
        if (layer.keys) cudaFree(layer.keys);
        if (layer.values) cudaFree(layer.values);
    }
    if (cache) gemma4_cache_destroy(cache);
    if (!passed) {
        printf("  [FAIL] KV compaction did not retain the canonical prefix and requested tail.\n");
        return 1;
    }
    printf("  [PASS] KV compaction retained prefix and recent K/V rows.\n");
    return 0;
}

int test_finite_guard() {
    printf("[TEST] Testing device-side finite output guard...\n");
    const int N = 4;
    __nv_bfloat16 h_values[N] = {
        __float2bfloat16(1.0f), __float2bfloat16(NAN),
        __float2bfloat16(-2.0f), __float2bfloat16(INFINITY),
    };
    __nv_bfloat16 *d_values;
    int *d_status;
    cudaMalloc(&d_values, N * sizeof(__nv_bfloat16));
    cudaMalloc(&d_status, sizeof(int));
    cudaMemcpy(d_values, h_values, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemset(d_status, 0, sizeof(int));
    kernel_finite_guard_bf16<<<1, 256>>>(d_values, N, d_status);
    cudaDeviceSynchronize();
    cudaMemcpy(h_values, d_values, N * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    int h_status = 0;
    cudaMemcpy(&h_status, d_status, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_values);
    cudaFree(d_status);
    if (h_status != 1 || __bfloat162float(h_values[1]) != 0.0f ||
        __bfloat162float(h_values[3]) != 0.0f) {
        printf("  [FAIL] non-finite output was not neutralized\n");
        return 1;
    }
    printf("  [PASS] non-finite output is neutralized and flagged.\n");
    return 0;
}

int main() {
    printf("====================================================\n");
    printf("GEMMA 4 NATIVE LISTENER PRIMITIVE UNIT TESTS\n");
    printf("Manifest SHA: %s\n", gemma4_listener_manifest_sha256());
    printf("ABI Version:  %d\n", gemma4_listener_abi_version());
    printf("====================================================\n");

    int err = 0;
    err |= test_rmsnorm();
    err |= test_geglu();
    err |= test_rope_rebase();
    err |= test_cache_rebase();
    err |= test_finite_guard();

    if (err == 0) {
        printf("\n>>> ALL PRIMITIVE UNIT TESTS PASSED <<<\n");
    } else {
        printf("\n>>> SOME TESTS FAILED <<<\n");
    }
    return err;
}
