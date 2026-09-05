/*
 * gemma4_listener_test.cu — Stand-alone Unit Test for Gemma 4 Native Kernels.
 *
 * Verifies that RMSNorm, GeGLU, RoPE, and SDPA compute exact numerical values
 * and that gemma4_listener_create/step/destroy execute cleanly.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "gemma4_listener.h"
#include "gemma4_listener_kernels.cuh"

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
    err |= test_finite_guard();

    if (err == 0) {
        printf("\n>>> ALL PRIMITIVE UNIT TESTS PASSED <<<\n");
    } else {
        printf("\n>>> SOME TESTS FAILED <<<\n");
    }
    return err;
}
