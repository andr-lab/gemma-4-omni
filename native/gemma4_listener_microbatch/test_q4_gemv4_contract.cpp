#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "gemma4_listener_gemv4_contract.h"

static void test_dispatch_contract() {
    const uintptr_t w = 0x1000, x = 0x2000, y = 0x3000;
    assert(gemma4_q4_gemv4_dispatch_eligible(true, true, true, 4, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(false, true, true, 4, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, false, true, 4, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, false, 4, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 3, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 291, 17, 8, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 4, 17, 7, w, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 4, 17, 8, 0, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 4, 17, 8, w + 2, x, y));
    assert(!gemma4_q4_gemv4_dispatch_eligible(true, true, true, 4, 17, 8, w, x, y + 1));
}

static void test_row_major_reference_contract() {
    constexpr int M = 4, N = 5, K = 6;
    std::vector<float> w(N * K), x(M * K), y(M * N, 0.0f), expected(M * N, 0.0f);
    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            w[gemma4_q4_gemv4_weight_index(n, k, K)] = float((n + 1) * (k + 2)) / 13.0f;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            x[m * K + k] = float((m + 2) - k) / 7.0f;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += x[m * K + k] * w[gemma4_q4_gemv4_weight_index(n, k, K)];
            expected[gemma4_q4_gemv4_output_index(m, n, N)] = acc;
        }
    }
    // Independent row-major reference expression mirrors F.linear's [N,K] weight layout.
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                y[m * N + n] += x[m * K + k] * w[n * K + k];
    for (int i = 0; i < M * N; ++i) assert(y[i] == expected[i]);
    assert(gemma4_q4_gemv4_output_index(3, 4, N) == 19);
    assert(gemma4_q4_gemv4_weight_index(4, 5, K) == 29);
}

int main() {
    test_dispatch_contract();
    test_row_major_reference_contract();
    std::puts("q4 gemv4 CPU contract: PASS");
}
