/*
 * CUTLASS_ORDERED_TILES_V1 — prescribed four-stage full-attention path.
 *
 * This TU deliberately instantiates the upstream PyTorch CUTLASS building
 * blocks, but never calls the monolithic attention kernel or a dispatcher.
 * A single stream launch sequence is:
 *   pack-Q, A(QK), B(ordered softmax), C(PV), D(output fold).
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemma4_listener_attention_ordered.cuh"
#include "vendor/pytorch/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h"

namespace {

using OrderedAttention = PyTorchMemEffAttention::AttentionKernel<
    cutlass::bfloat16_t,
    cutlass::arch::Sm80,
    true,
    32,
    GEMMA4_ORDERED_KEY_TILE_SIZE,
    GEMMA4_ORDERED_MAX_KEYS,
    false,
    true>;
using Bf16 = cutlass::bfloat16_t;
using MM0 = typename OrderedAttention::MM0;
using MM1 = typename OrderedAttention::MM1;
using FragmentQK = typename MM0::Mma::FragmentC;
using FragmentPV = typename MM1::Mma::FragmentC;

static_assert(OrderedAttention::kQueriesPerBlock == 32);
static_assert(GEMMA4_ORDERED_QUERY_ROWS == GEMMA4_STREAM_Q_LEN);
static_assert(GEMMA4_STREAM_Q_LEN == 2);
static_assert(OrderedAttention::kKeysPerBlock == 128);
static_assert(OrderedAttention::kNumWarpsPerBlock == 4);
static_assert(OrderedAttention::kNumThreads == 128);
static_assert(OrderedAttention::kMaxK == 16384);
static_assert(!OrderedAttention::kKeepOutputInRF);
static_assert(OrderedAttention::kPreloadV);
static_assert(MM0::Mma::WarpCount::kM == 1);
static_assert(MM0::Mma::WarpCount::kN == 4);
static_assert(MM1::Mma::WarpCount::kM * MM1::Mma::WarpCount::kN == 4);
static_assert(sizeof(FragmentQK) % sizeof(float) == 0);
static_assert(sizeof(FragmentPV) % sizeof(float) == 0);

constexpr size_t kMaxTiles =
    (GEMMA4_ORDERED_MAX_KEYS + GEMMA4_ORDERED_KEY_TILE_SIZE - 1) /
    GEMMA4_ORDERED_KEY_TILE_SIZE;
constexpr size_t kProbabilityElementsPerTile =
    GEMMA4_ORDERED_QUERY_TILE_ROWS * GEMMA4_ORDERED_KEY_TILE_SIZE;

__device__ __forceinline__ int ordered_thread_id() {
    return static_cast<int>(threadIdx.x + threadIdx.y * blockDim.x);
}

__device__ __forceinline__ int ordered_lane_id() {
    return static_cast<int>(threadIdx.x);
}

__device__ __forceinline__ int ordered_warp_id() {
    return static_cast<int>(threadIdx.y);
}

__device__ __forceinline__ size_t ordered_qk_offset(
    int head,
    int tile,
    int thread,
    int element) {
    return ((((size_t)head * kMaxTiles + (size_t)tile) *
              OrderedAttention::kNumThreads + (size_t)thread) *
            sizeof(FragmentQK) / sizeof(float) + (size_t)element);
}

__device__ __forceinline__ size_t ordered_pv_offset(
    int head,
    int tile,
    int value_block,
    int thread,
    int element) {
    return (((((size_t)head * kMaxTiles + (size_t)tile) * 4u +
              (size_t)value_block) * OrderedAttention::kNumThreads +
             (size_t)thread) * sizeof(FragmentPV) / sizeof(float) +
            (size_t)element);
}

__device__ __forceinline__ bool ordered_key_visible(
    int key_position,
    uint32_t query_position,
    const int64_t *prefix_attention,
    uint32_t prefix_len) {
    const bool prefix_valid =
        static_cast<uint32_t>(key_position) >= prefix_len ||
        prefix_attention == nullptr || prefix_attention[key_position] != 0;
    return prefix_valid && static_cast<uint32_t>(key_position) <= query_position;
}

__global__ void ordered_pack_q_head_major(
    const __nv_bfloat16 *query_q_major,
    __nv_bfloat16 *query_head_major) {
    const size_t total = (size_t)GEMMA4_ORDERED_QUERY_ROWS *
        GEMMA4_ORDERED_HEADS * GEMMA4_ORDERED_HEAD_DIM;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < total;
         index += (size_t)blockDim.x * gridDim.x) {
        const size_t d = index % GEMMA4_ORDERED_HEAD_DIM;
        const size_t q = (index / GEMMA4_ORDERED_HEAD_DIM) %
            GEMMA4_ORDERED_QUERY_ROWS;
        const size_t h = index /
            (GEMMA4_ORDERED_QUERY_ROWS * GEMMA4_ORDERED_HEAD_DIM);
        const size_t source =
            ((q * GEMMA4_ORDERED_HEADS + h) * GEMMA4_ORDERED_HEAD_DIM) + d;
        query_head_major[index] = query_q_major[source];
    }
}

template <typename AK>
__global__ __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
void ordered_stage_a_qk(
    const Bf16 *query_head_major,
    const Bf16 *key,
    const int64_t *prefix_attention,
    uint32_t prefix_len,
    uint32_t key_len,
    uint32_t base_position,
    float scale,
    float *qk_fragments) {
    extern __shared__ char smem_buffer[];
    typename AK::SharedStorage &shared_storage =
        *((typename AK::SharedStorage *)smem_buffer);

    const int head = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int key_start = tile * AK::kKeysPerBlock;
    if (head >= GEMMA4_ORDERED_HEADS || key_start >= static_cast<int>(key_len)) {
        return;
    }
    const int problem_m = GEMMA4_ORDERED_QUERY_ROWS;
    const int problem_n = min(
        AK::kKeysPerBlock, static_cast<int>(key_len) - key_start);
    const int tid = ordered_thread_id();
    const int lane = ordered_lane_id();
    const int warp = ordered_warp_id();

    cutlass::gemm::GemmCoord tb_tile_offset = {0, 0, 0};
    cutlass::MatrixCoord tb_offset_a{0, 0};
    cutlass::MatrixCoord tb_offset_b{0, 0};
    const Bf16 *q_head = query_head_major +
        ((size_t)head * GEMMA4_ORDERED_QUERY_ROWS * GEMMA4_ORDERED_HEAD_DIM);

    typename MM0::IteratorA iterator_a(
        typename MM0::IteratorA::Params(typename MM0::MmaCore::LayoutA(
            GEMMA4_ORDERED_HEAD_DIM)),
        const_cast<Bf16 *>(q_head),
        {problem_m, GEMMA4_ORDERED_HEAD_DIM},
        tid,
        tb_offset_a);
    typename MM0::IteratorB iterator_b(
        typename MM0::IteratorB::Params(typename MM0::MmaCore::LayoutB(
            GEMMA4_ORDERED_HEAD_DIM)),
        const_cast<Bf16 *>(key + (size_t)key_start * GEMMA4_ORDERED_HEAD_DIM),
        {GEMMA4_ORDERED_HEAD_DIM, problem_n},
        tid,
        tb_offset_b);
    typename MM0::Mma mma(
        shared_storage.mm0, tid, warp, lane);
    FragmentQK accum;
    accum.clear();
    const int gemm_k_iterations =
        (GEMMA4_ORDERED_HEAD_DIM + MM0::Mma::Shape::kK - 1) /
        MM0::Mma::Shape::kK;
    mma(gemm_k_iterations, accum, iterator_a, iterator_b, accum);
    __syncthreads();

    /* A owns the only QK scale. B therefore receives scale=1.0. */
    accum = cutlass::multiplies<FragmentQK>()(scale, accum);

    /* The upstream BiasLoader storage/layout is retained; only its source is
     * synthesized. No arithmetic is performed while constructing BF16 bias. */
    cutlass::TensorRef<Bf16, cutlass::layout::RowMajor> bias_ref(
        shared_storage.after_mm0.bias.data(),
        cutlass::layout::RowMajor(AK::kKeysPerBlock));
    for (int index = tid; index < AK::kQueriesPerBlock * AK::kKeysPerBlock;
         index += AK::kNumThreads) {
        const int row = index / AK::kKeysPerBlock;
        const int col = index % AK::kKeysPerBlock;
        const int key_position = key_start + col;
        const bool visible = col < problem_n && ordered_key_visible(
            key_position,
            base_position + static_cast<uint32_t>(row),
            prefix_attention,
            prefix_len);
        bias_ref.at({row, col}) = Bf16(visible ? 0.0f : -INFINITY);
    }
    __syncthreads();

    typename MM0::Mma::Operator::IteratorC::TensorCoord iterator_c_tile_offset = {
        (tb_tile_offset.m() * MM0::Mma::WarpCount::kM) +
            (warp % MM0::Mma::WarpCount::kM),
        (tb_tile_offset.n() * MM0::Mma::WarpCount::kN) +
            (warp / MM0::Mma::WarpCount::kM)};
    auto lane_offset = MM0::AccumLambdaIterator::get_lane_offset(
        lane, warp, iterator_c_tile_offset);
    MM0::AccumLambdaIterator::iterateRows(
        lane_offset,
        [&](int) {},
        [&](int accum_m, int accum_n, int index) {
            if (accum_m < problem_m && accum_n < problem_n) {
                accum[index] += bias_ref.at({accum_m, accum_n});
            }
        },
        [&](int) {});

    float *destination = qk_fragments +
        ordered_qk_offset(head, tile, tid, 0);
    for (int index = 0; index < static_cast<int>(sizeof(FragmentQK) / sizeof(float));
         ++index) {
        destination[index] = accum[index];
    }
}

template <typename AK>
__global__ __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
void ordered_stage_b_softmax(
    const float *qk_fragments,
    Bf16 *probability_tiles,
    float *s_prime_tiles,
    float *out_rescale_tiles,
    uint32_t key_len) {
    extern __shared__ char smem_buffer[];
    typename AK::SharedStorage &shared_storage =
        *((typename AK::SharedStorage *)smem_buffer);
    auto &m_prime = shared_storage.m_prime;
    auto &s_prime = shared_storage.s_prime;
    auto &mi = shared_storage.mi;
    auto &out_rescale = shared_storage.out_rescale;
    const int head = static_cast<int>(blockIdx.x);
    if (head >= GEMMA4_ORDERED_HEADS) return;
    const int tid = ordered_thread_id();
    const int lane = ordered_lane_id();
    const int warp = ordered_warp_id();
    if (tid < AK::kQueriesPerBlock) {
        s_prime[tid] = 0.0f;
        out_rescale[tid] = 1.0f;
        m_prime[tid] = -INFINITY;
        mi[tid] = -INFINITY;
    }
    __syncthreads();

    for (int tile = 0; tile * AK::kKeysPerBlock < static_cast<int>(key_len);
         ++tile) {
        const int key_start = tile * AK::kKeysPerBlock;
        const int max_col = static_cast<int>(key_len) - key_start;
        FragmentQK accum;
        const float *source = qk_fragments +
            ordered_qk_offset(head, tile, tid, 0);
        for (int index = 0;
             index < static_cast<int>(sizeof(FragmentQK) / sizeof(float));
             ++index) {
            accum[index] = source[index];
        }
        typename MM1::Mma::FragmentC accum_o;
        accum_o.clear();
        typename MM0::Mma::Operator::IteratorC::TensorCoord iterator_c_tile_offset = {
            warp % MM0::Mma::WarpCount::kM,
            warp / MM0::Mma::WarpCount::kM};
        AK::template iterative_softmax<
            typename MM0::Mma::Operator::IteratorC>(
            accum_o,
            accum,
            mi,
            m_prime,
            s_prime,
            out_rescale,
            shared_storage.addition_storage,
            lane,
            tid,
            warp,
            max_col,
            tile == 0,
            iterator_c_tile_offset,
            1.0f);

        const int warp_idx_mn_0 = warp %
            (MM0::Mma::Base::WarpCount::kM * MM0::Mma::Base::WarpCount::kN);
        const auto output_tile_coords = cutlass::MatrixCoord{
            warp_idx_mn_0 % MM0::Mma::Base::WarpCount::kM,
            warp_idx_mn_0 / MM0::Mma::Base::WarpCount::kM};
        MM0::B2bGemm::accumToSmem(
            shared_storage.after_mm0.si,
            accum,
            lane,
            output_tile_coords);
        __syncthreads();

        auto probability_ref = shared_storage.after_mm0.si.accum_ref();
        Bf16 *probability_destination = probability_tiles +
            (((size_t)head * kMaxTiles + (size_t)tile) *
             kProbabilityElementsPerTile);
        for (int index = tid; index < static_cast<int>(kProbabilityElementsPerTile);
             index += AK::kNumThreads) {
            const int row = index / AK::kKeysPerBlock;
            const int col = index % AK::kKeysPerBlock;
            probability_destination[index] = probability_ref.at({row, col});
        }
        if (tid < AK::kQueriesPerBlock) {
            const size_t state_offset =
                ((size_t)head * kMaxTiles + (size_t)tile) *
                AK::kQueriesPerBlock + (size_t)tid;
            s_prime_tiles[state_offset] = s_prime[tid];
            out_rescale_tiles[state_offset] = out_rescale[tid];
        }
        __syncthreads();
    }
}

template <typename AK>
__global__ __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
void ordered_stage_c_pv(
    const Bf16 *probability_tiles,
    const Bf16 *value,
    float *pv_fragments,
    uint32_t key_len) {
    extern __shared__ char smem_buffer[];
    typename AK::SharedStorage &shared_storage =
        *((typename AK::SharedStorage *)smem_buffer);
    const int head = static_cast<int>(blockIdx.x);
    const int tile = static_cast<int>(blockIdx.y);
    const int value_block = static_cast<int>(blockIdx.z);
    const int key_start = tile * AK::kKeysPerBlock;
    if (head >= GEMMA4_ORDERED_HEADS || key_start >= static_cast<int>(key_len) ||
        value_block >= 4) return;
    const int tid = ordered_thread_id();
    const int lane = ordered_lane_id();
    const int warp = ordered_warp_id();
    const int problem_k = min(
        AK::kKeysPerBlock, static_cast<int>(key_len) - key_start);
    auto probability_ref = shared_storage.after_mm0.si.accum_ref();
    const Bf16 *probability_source = probability_tiles +
        (((size_t)head * kMaxTiles + (size_t)tile) *
         kProbabilityElementsPerTile);
    for (int index = tid; index < static_cast<int>(kProbabilityElementsPerTile);
         index += AK::kNumThreads) {
        const int row = index / AK::kKeysPerBlock;
        const int col = index % AK::kKeysPerBlock;
        probability_ref.at({row, col}) = probability_source[index];
    }
    __syncthreads();

    const int block_n = value_block * MM1::Mma::Shape::kN;
    typename MM1::Mma::IteratorB iterator_v(
        typename MM1::IteratorB::Params{MM1::LayoutB(
            GEMMA4_ORDERED_HEAD_DIM)},
        const_cast<Bf16 *>(value + (size_t)key_start * GEMMA4_ORDERED_HEAD_DIM),
        {problem_k, GEMMA4_ORDERED_HEAD_DIM},
        tid,
        cutlass::MatrixCoord{0, block_n});
    MM1::Mma::prologue(
        shared_storage.after_mm0.mm1,
        iterator_v,
        tid,
        problem_k);

    typename MM1::Mma mma_pv(
        shared_storage.after_mm0.si.accum_ref(),
        shared_storage.after_mm0.mm1.operand_B_ref(),
        tid,
        warp,
        lane);
    mma_pv.set_prologue_done(AK::kPreloadV);
    FragmentPV accum;
    accum.clear();
    const int gemm_k_iterations =
        (problem_k + MM1::Mma::Shape::kK - 1) / MM1::Mma::Shape::kK;
    mma_pv(gemm_k_iterations, accum, iterator_v, accum);
    __syncthreads();

    float *destination = pv_fragments +
        ordered_pv_offset(head, tile, value_block, tid, 0);
    for (int index = 0; index < static_cast<int>(sizeof(FragmentPV) / sizeof(float));
         ++index) {
        destination[index] = accum[index];
    }
}

template <typename AK, bool IsFirst, bool IsLast>
__device__ void ordered_fold_one_tile(
    typename AK::SharedStorage &shared_storage,
    const float *pv_fragments,
    const float *s_prime_tiles,
    const float *out_rescale_tiles,
    Bf16 *output,
    float *output_accum,
    int head,
    int tile,
    int value_block) {
    const int tid = ordered_thread_id();
    const int warp = ordered_warp_id();
    const int lane = ordered_lane_id();
    typename MM1::Mma::FragmentC accum_o;
    const float *source = pv_fragments +
        ordered_pv_offset(head, tile, value_block, tid, 0);
    for (int index = 0;
         index < static_cast<int>(sizeof(FragmentPV) / sizeof(float));
         ++index) {
        accum_o[index] = source[index];
    }
    if (tid < AK::kQueriesPerBlock) {
        const size_t state_offset =
            ((size_t)head * kMaxTiles + (size_t)tile) *
            AK::kQueriesPerBlock + (size_t)tid;
        shared_storage.s_prime[tid] = s_prime_tiles[state_offset];
        shared_storage.out_rescale[tid] = out_rescale_tiles[state_offset];
    }
    __syncthreads();

    using DefaultEpilogue = typename MM1::DefaultEpilogue;
    using DefaultOp = typename MM1::DefaultConfig::EpilogueOutputOp;
    using ElementCompute = typename DefaultOp::ElementCompute;
    using EpilogueOutputOp = typename cutlass::epilogue::thread::
        MemoryEfficientAttentionNormalize<
            typename cutlass::platform::conditional<
                IsLast,
                Bf16,
                float>::type,
            float,
            DefaultOp::kCount,
            typename DefaultOp::ElementAccumulator,
            ElementCompute,
            IsFirst,
            IsLast,
            cutlass::Array<ElementCompute, AK::kQueriesPerBlock>>;
    using Epilogue = typename cutlass::epilogue::threadblock::EpiloguePipelined<
        typename DefaultEpilogue::Shape,
        typename MM1::Mma::Operator,
        DefaultEpilogue::kPartitionsK,
        typename cutlass::platform::conditional<
            IsLast,
            typename MM1::OutputTileIterator,
            typename MM1::OutputTileIteratorAccum>::type,
        typename DefaultEpilogue::AccumulatorFragmentIterator,
        typename DefaultEpilogue::WarpTileIterator,
        typename DefaultEpilogue::SharedLoadIterator,
        EpilogueOutputOp,
        typename DefaultEpilogue::Padding,
        DefaultEpilogue::kFragmentsPerIteration,
        true,
        typename MM1::OutputTileIteratorAccum>;

    const int col = value_block * MM1::Mma::Shape::kN;
    typename MM1::OutputTileIteratorAccum source_iter(
        typename MM1::OutputTileIteratorAccum::Params{
            GEMMA4_ORDERED_HEAD_DIM * GEMMA4_ORDERED_HEADS},
        output_accum + (size_t)head * GEMMA4_ORDERED_HEAD_DIM,
        typename MM1::OutputTileIteratorAccum::TensorCoord{
            GEMMA4_ORDERED_QUERY_ROWS, GEMMA4_ORDERED_HEAD_DIM},
        tid,
        {0, col});
    EpilogueOutputOp rescale(shared_storage.s_prime, shared_storage.out_rescale);
    Epilogue epilogue(
        shared_storage.epilogue_shared_storage(), tid, warp, lane);
    if constexpr (IsLast) {
        typename MM1::OutputTileIterator dest_iter(
            typename MM1::OutputTileIterator::Params{
                GEMMA4_ORDERED_HEADS * GEMMA4_ORDERED_HEAD_DIM},
            output + (size_t)head * GEMMA4_ORDERED_HEAD_DIM,
            typename MM1::OutputTileIterator::TensorCoord{
                GEMMA4_ORDERED_QUERY_ROWS, GEMMA4_ORDERED_HEAD_DIM},
            tid,
            {0, col});
        epilogue(rescale, dest_iter, accum_o, source_iter);
    } else {
        typename MM1::OutputTileIteratorAccum dest_iter(
            typename MM1::OutputTileIteratorAccum::Params{
                GEMMA4_ORDERED_HEAD_DIM * GEMMA4_ORDERED_HEADS},
            output_accum + (size_t)head * GEMMA4_ORDERED_HEAD_DIM,
            typename MM1::OutputTileIteratorAccum::TensorCoord{
                GEMMA4_ORDERED_QUERY_ROWS, GEMMA4_ORDERED_HEAD_DIM},
            tid,
            {0, col});
        epilogue(rescale, dest_iter, accum_o, source_iter);
    }
}

template <typename AK>
__global__ __launch_bounds__(AK::kNumThreads, AK::kMinBlocksPerSm)
void ordered_stage_d_fold(
    const float *pv_fragments,
    const float *s_prime_tiles,
    const float *out_rescale_tiles,
    Bf16 *output,
    float *output_accum,
    uint32_t key_len) {
    extern __shared__ char smem_buffer[];
    typename AK::SharedStorage &shared_storage =
        *((typename AK::SharedStorage *)smem_buffer);
    const int head = static_cast<int>(blockIdx.x);
    const int value_block = static_cast<int>(blockIdx.y);
    if (head >= GEMMA4_ORDERED_HEADS || value_block >= 4) return;
    const int tiles =
        (static_cast<int>(key_len) + AK::kKeysPerBlock - 1) /
        AK::kKeysPerBlock;
    for (int tile = 0; tile < tiles; ++tile) {
        if (tile == 0) {
            ordered_fold_one_tile<AK, true, false>(
                shared_storage,
                pv_fragments,
                s_prime_tiles,
                out_rescale_tiles,
                output,
                output_accum,
                head,
                tile,
                value_block);
        } else if (tile + 1 == tiles) {
            ordered_fold_one_tile<AK, false, true>(
                shared_storage,
                pv_fragments,
                s_prime_tiles,
                out_rescale_tiles,
                output,
                output_accum,
                head,
                tile,
                value_block);
        } else {
            ordered_fold_one_tile<AK, false, false>(
                shared_storage,
                pv_fragments,
                s_prime_tiles,
                out_rescale_tiles,
                output,
                output_accum,
                head,
                tile,
                value_block);
        }
        __syncthreads();
    }
}

int set_ordered_dynamic_shared_memory(
    const char **failed_field,
    size_t *failed_bytes,
    cudaError_t *failed_error
) {
    const size_t bytes = sizeof(typename OrderedAttention::SharedStorage);
    if (bytes <= 48u * 1024u) return 0;
    const cudaFuncAttribute attribute = cudaFuncAttributeMaxDynamicSharedMemorySize;
    cudaError_t status = cudaFuncSetAttribute(
        ordered_stage_a_qk<OrderedAttention>, attribute, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        if (failed_field) *failed_field = "ordered_stage_a.dynamic_shared_memory";
        if (failed_bytes) *failed_bytes = bytes;
        if (failed_error) *failed_error = status;
        return -1;
    }
    status = cudaFuncSetAttribute(
        ordered_stage_b_softmax<OrderedAttention>, attribute, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        if (failed_field) *failed_field = "ordered_stage_b.dynamic_shared_memory";
        if (failed_bytes) *failed_bytes = bytes;
        if (failed_error) *failed_error = status;
        return -1;
    }
    status = cudaFuncSetAttribute(
        ordered_stage_c_pv<OrderedAttention>, attribute, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        if (failed_field) *failed_field = "ordered_stage_c.dynamic_shared_memory";
        if (failed_bytes) *failed_bytes = bytes;
        if (failed_error) *failed_error = status;
        return -1;
    }
    status = cudaFuncSetAttribute(
        ordered_stage_d_fold<OrderedAttention>, attribute, static_cast<int>(bytes));
    if (status != cudaSuccess) {
        if (failed_field) *failed_field = "ordered_stage_d.dynamic_shared_memory";
        if (failed_bytes) *failed_bytes = bytes;
        if (failed_error) *failed_error = status;
        return -1;
    }
    return 0;
}

} // namespace

static int gemma4_ordered_workspace_create_impl(
    gemma4_ordered_workspace *workspace,
    uint32_t max_keys,
    gemma4_ordered_create_failure *failure) {
    if (failure) *failure = {};
    if (!workspace || max_keys > GEMMA4_ORDERED_MAX_KEYS) {
        if (failure) {
            failure->field = "ordered_workspace.max_keys";
            failure->reason = "invalid_workspace_or_key_limit";
        }
        return -1;
    }
    *workspace = {};
    workspace->max_keys = max_keys;
    workspace->max_tiles = static_cast<uint32_t>(
        (max_keys + GEMMA4_ORDERED_KEY_TILE_SIZE - 1) /
        GEMMA4_ORDERED_KEY_TILE_SIZE);
    workspace->q_head_major_bytes =
        (size_t)GEMMA4_ORDERED_HEADS * GEMMA4_ORDERED_QUERY_ROWS *
        GEMMA4_ORDERED_HEAD_DIM * sizeof(__nv_bfloat16);
    workspace->qk_fragment_bytes =
        (size_t)GEMMA4_ORDERED_HEADS * kMaxTiles *
        OrderedAttention::kNumThreads * sizeof(FragmentQK);
    workspace->probability_tile_bytes =
        (size_t)GEMMA4_ORDERED_HEADS * kMaxTiles *
        kProbabilityElementsPerTile * sizeof(Bf16);
    workspace->softmax_state_bytes =
        (size_t)GEMMA4_ORDERED_HEADS * kMaxTiles *
        OrderedAttention::kQueriesPerBlock * 2u * sizeof(float);
    workspace->pv_fragment_bytes =
        (size_t)GEMMA4_ORDERED_HEADS * kMaxTiles * 4u *
        OrderedAttention::kNumThreads * sizeof(FragmentPV);
    workspace->output_accum_bytes =
        (size_t)GEMMA4_ORDERED_QUERY_ROWS * GEMMA4_ORDERED_HEADS *
        GEMMA4_ORDERED_HEAD_DIM * sizeof(float);
    workspace->total_bytes = workspace->q_head_major_bytes +
        workspace->qk_fragment_bytes + workspace->probability_tile_bytes +
        workspace->softmax_state_bytes + workspace->pv_fragment_bytes +
        workspace->output_accum_bytes;

    const char *failed_field = nullptr;
    size_t failed_bytes = 0;
    cudaError_t failed_error = cudaSuccess;
    if (set_ordered_dynamic_shared_memory(
            &failed_field, &failed_bytes, &failed_error) != 0) {
        if (failure) {
            failure->field = failed_field;
            failure->reason = "cudaFuncSetAttribute";
            failure->requested_bytes = failed_bytes;
            failure->cuda_error = (int)failed_error;
            failure->cuda_error_known = 1;
        }
        return -2;
    }
#define ORDERED_ALLOC(member, bytes) \
    do { \
        const size_t requested_bytes = (bytes); \
        const cudaError_t allocation_status = cudaMalloc(&workspace->member, requested_bytes); \
        if (allocation_status != cudaSuccess) { \
            if (failure) { \
                failure->field = "ordered_workspace." #member; \
                failure->reason = "cudaMalloc"; \
                failure->requested_bytes = requested_bytes; \
                failure->cuda_error = (int)allocation_status; \
                failure->cuda_error_known = 1; \
            } \
            gemma4_ordered_workspace_destroy(workspace); \
            return -3; \
        } \
    } while (0)
    ORDERED_ALLOC(q_head_major, workspace->q_head_major_bytes);
    ORDERED_ALLOC(qk_fragments, workspace->qk_fragment_bytes);
    ORDERED_ALLOC(probability_tiles, workspace->probability_tile_bytes);
    ORDERED_ALLOC(s_prime, workspace->softmax_state_bytes / 2u);
    ORDERED_ALLOC(out_rescale, workspace->softmax_state_bytes / 2u);
    ORDERED_ALLOC(pv_fragments, workspace->pv_fragment_bytes);
    ORDERED_ALLOC(output_accum, workspace->output_accum_bytes);
#undef ORDERED_ALLOC
    workspace->initialized = 1;
    return 0;
}

extern "C" int gemma4_ordered_workspace_create(
    gemma4_ordered_workspace *workspace,
    uint32_t max_keys) {
    return gemma4_ordered_workspace_create_impl(workspace, max_keys, nullptr);
}

extern "C" int gemma4_ordered_workspace_create_with_diagnostics(
    gemma4_ordered_workspace *workspace,
    uint32_t max_keys,
    gemma4_ordered_create_failure *failure) {
    return gemma4_ordered_workspace_create_impl(workspace, max_keys, failure);
}

extern "C" void gemma4_ordered_workspace_destroy(
    gemma4_ordered_workspace *workspace) {
    if (!workspace) return;
    if (workspace->q_head_major) cudaFree(workspace->q_head_major);
    if (workspace->qk_fragments) cudaFree(workspace->qk_fragments);
    if (workspace->probability_tiles) cudaFree(workspace->probability_tiles);
    if (workspace->s_prime) cudaFree(workspace->s_prime);
    if (workspace->out_rescale) cudaFree(workspace->out_rescale);
    if (workspace->pv_fragments) cudaFree(workspace->pv_fragments);
    if (workspace->output_accum) cudaFree(workspace->output_accum);
    *workspace = {};
}

extern "C" size_t gemma4_ordered_workspace_bytes(
    const gemma4_ordered_workspace *workspace) {
    return workspace ? workspace->total_bytes : 0;
}

#define ORDERED_COUNTER_GETTER(name, field) \
    extern "C" uint64_t name(const gemma4_ordered_workspace *workspace) { \
        return workspace ? workspace->field : 0; \
    }
ORDERED_COUNTER_GETTER(gemma4_ordered_pack_launches, pack_launches)
ORDERED_COUNTER_GETTER(gemma4_ordered_stage_a_launches, stage_a_launches)
ORDERED_COUNTER_GETTER(gemma4_ordered_stage_b_launches, stage_b_launches)
ORDERED_COUNTER_GETTER(gemma4_ordered_stage_c_launches, stage_c_launches)
ORDERED_COUNTER_GETTER(gemma4_ordered_stage_d_launches, stage_d_launches)
#undef ORDERED_COUNTER_GETTER

int gemma4_ordered_attention_launch(
    const __nv_bfloat16 *query_q_major,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    const int64_t *prefix_attention,
    uint32_t prefix_len,
    uint32_t key_len,
    uint32_t base_position,
    float scale,
    __nv_bfloat16 *output,
    cudaStream_t stream,
    gemma4_ordered_workspace *workspace) {
    if (!query_q_major || !key || !value || !output || !workspace ||
        !workspace->initialized || !std::isfinite(scale) ||
        key_len <= GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE ||
        key_len > workspace->max_keys || key_len > GEMMA4_ORDERED_MAX_KEYS ||
        prefix_len > key_len) return -1;

    const int tiles = static_cast<int>(
        (key_len + GEMMA4_ORDERED_KEY_TILE_SIZE - 1) /
        GEMMA4_ORDERED_KEY_TILE_SIZE);
    ordered_pack_q_head_major<<<64, 256, 0, stream>>>(
        query_q_major,
        static_cast<__nv_bfloat16 *>(workspace->q_head_major));
    ordered_stage_a_qk<OrderedAttention><<<
        dim3(GEMMA4_ORDERED_HEADS, tiles),
        dim3(32, 4),
        sizeof(typename OrderedAttention::SharedStorage),
        stream>>>(
        static_cast<const Bf16 *>(workspace->q_head_major),
        reinterpret_cast<const Bf16 *>(key),
        prefix_attention,
        prefix_len,
        key_len,
        base_position,
        scale,
        static_cast<float *>(workspace->qk_fragments));
    ordered_stage_b_softmax<OrderedAttention><<<
        dim3(GEMMA4_ORDERED_HEADS),
        dim3(32, 4),
        sizeof(typename OrderedAttention::SharedStorage),
        stream>>>(
        static_cast<const float *>(workspace->qk_fragments),
        static_cast<Bf16 *>(workspace->probability_tiles),
        workspace->s_prime,
        workspace->out_rescale,
        key_len);
    ordered_stage_c_pv<OrderedAttention><<<
        dim3(GEMMA4_ORDERED_HEADS, tiles, 4),
        dim3(32, 4),
        sizeof(typename OrderedAttention::SharedStorage),
        stream>>>(
        static_cast<const Bf16 *>(workspace->probability_tiles),
        reinterpret_cast<const Bf16 *>(value),
        static_cast<float *>(workspace->pv_fragments),
        key_len);
    ordered_stage_d_fold<OrderedAttention><<<
        dim3(GEMMA4_ORDERED_HEADS, 4),
        dim3(32, 4),
        sizeof(typename OrderedAttention::SharedStorage),
        stream>>>(
        static_cast<const float *>(workspace->pv_fragments),
        workspace->s_prime,
        workspace->out_rescale,
        reinterpret_cast<Bf16 *>(output),
        workspace->output_accum,
        key_len);
    if (cudaPeekAtLastError() != cudaSuccess) return -2;
    ++workspace->pack_launches;
    ++workspace->stage_a_launches;
    ++workspace->stage_b_launches;
    ++workspace->stage_c_launches;
    ++workspace->stage_d_launches;
    return 0;
}
