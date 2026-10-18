/* New ABI surface for the listener's physical N-session forward.  The legacy
 * ABI3 executor above is intentionally left byte-for-byte behaviorally intact.
 * All projection GEMMs below use one row-major concatenated [N*2,K] operand;
 * per-session work is limited to private KV access and attention state. */

#include <new>

struct gemma4_listener_batch_slot {
    gemma4_cache *cache;
    int64_t *prefix_attention;
    uint64_t context_id;
    uint64_t session_generation;
    uint64_t state_epoch_id;
    uint64_t expected_frame_id;
    uint64_t expected_state_version;
    uint64_t expected_sequence_position;
    uint32_t prefix_len;
    uint32_t current_position;
    bool primed;
};

typedef struct {
    cublasHandle_t handle;
    cudaStream_t stream;
    cudaEvent_t done_event;
    void *cublas_workspace;
    size_t cublas_workspace_bytes;
} gemma4_batch_gemm_lane;

static constexpr size_t GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES = 32u * 1024u * 1024u;

enum {
    GEMMA4_BATCH_GEMM_OP_PRELUDE = 0,
    GEMMA4_BATCH_GEMM_OPS_PER_LAYER = 9,
    GEMMA4_BATCH_GEMM_OP_Q = 0,
    GEMMA4_BATCH_GEMM_OP_K = 1,
    GEMMA4_BATCH_GEMM_OP_V = 2,
    GEMMA4_BATCH_GEMM_OP_O = 3,
    GEMMA4_BATCH_GEMM_OP_GATE = 4,
    GEMMA4_BATCH_GEMM_OP_UP = 5,
    GEMMA4_BATCH_GEMM_OP_DOWN = 6,
    GEMMA4_BATCH_GEMM_OP_PLE_GATE = 7,
    GEMMA4_BATCH_GEMM_OP_PLE_PROJ = 8
};
#define GEMMA4_BATCH_GEMM_OP_COUNT \
    (1u + GEMMA4_NUM_LAYERS * GEMMA4_BATCH_GEMM_OPS_PER_LAYER)
#define GEMMA4_BATCH_GEMM_OP_ID(layer, op) \
    (1u + (uint32_t)(layer) * GEMMA4_BATCH_GEMM_OPS_PER_LAYER + (uint32_t)(op))

typedef struct gemma4_batch_gemm_graph {
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    const void *weight;
    const void *input;
    void *output;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lane_width;
    bool ready;
} gemma4_batch_gemm_graph;

typedef enum {
    GEMMA4_BATCH_TRACE_REF_NONE = 0,
    GEMMA4_BATCH_TRACE_REF_GEMM = 1,
    GEMMA4_BATCH_TRACE_REF_RMS = 2,
    GEMMA4_BATCH_TRACE_REF_SCALE = 3,
    GEMMA4_BATCH_TRACE_REF_RESIDUAL = 4,
    GEMMA4_BATCH_TRACE_REF_GELU = 5,
    GEMMA4_BATCH_TRACE_REF_MUL = 6,
    GEMMA4_BATCH_TRACE_REF_FRAME_PAD = 7,
    GEMMA4_BATCH_TRACE_REF_PLE_LAYER = 8
} gemma4_batch_trace_ref_op;

typedef enum {
    GEMMA4_BATCH_TRACE_IDLE = 0,
    GEMMA4_BATCH_TRACE_ARMED = 1,
    GEMMA4_BATCH_TRACE_CAPTURED = 2,
    GEMMA4_BATCH_TRACE_ERROR = 3
} gemma4_batch_trace_state;

typedef struct {
    gemma4_listener_batch_trace_info info;
    size_t offsets[5];
    size_t capacity_bytes[5];
    const void *weight;
    double epsilon;
    float scalar;
    gemma4_batch_trace_ref_op reference_op;
    uint32_t captured_sessions;
    uint32_t attention_input_mask;
    uint32_t attention_output_mask;
    bool inputs_captured;
    bool actual_captured;
    bool reference_captured;
} gemma4_batch_trace_record;

struct gemma4_listener_batch_ctx {
    gemma4_listener_ctx *model;
    uint32_t capacity;
    bool poisoned;
    bool stream_bound;
    bool priority_verified;
    int32_t configured_stream_priority;
    cudaStream_t listener_stream;
    uint64_t output_generation;
    uint64_t completed_output_generation;
    uint32_t completed_output_row_count;
    bool output_lease_active;
    /* Bound to the latest output generation until its logical truncate and
     * output-health check complete. Text appends do not advance audio identity. */
    bool output_embedded_text_input;
    bool output_preserves_listener_cursors;
    cudaEvent_t completion_event;
    cudaEvent_t clone_event;
    bool clone_enqueued;
    gemma4_listener_batch_slot *slots;
    uint32_t *active_slots;
    gemma4_listener_batch_row *output_rows;
    uint32_t output_row_count;
    gemma4_batch_gemm_lane *gemm_lanes;
    cudaEvent_t gemm_ready_event;
    gemma4_batch_gemm_graph *gemm_graphs;
    bool gemm_graphs_enabled;
    uint32_t gemm_graph_bucket_count;
    uint64_t gemm_graph_count;
    __nv_bfloat16 *hidden_a;
    __nv_bfloat16 *hidden_b;
    __nv_bfloat16 *norm;
    __nv_bfloat16 *q;
    __nv_bfloat16 *k;
    __nv_bfloat16 *v;
    __nv_bfloat16 *attn;
    __nv_bfloat16 *gate;
    __nv_bfloat16 *up;
    __nv_bfloat16 *mlp;
    __nv_bfloat16 *ple_raw;
    __nv_bfloat16 *projected_ple;
    __nv_bfloat16 *ple_gate;
    __nv_bfloat16 *ple_act;
    __nv_bfloat16 *ple_out;
    __nv_bfloat16 *output;
    const void **gemm_a_ptrs;
    const void **gemm_b_ptrs;
    void **gemm_c_ptrs;
    /* Allocated only by the explicit proof-only N=2 probe arm call. */
    __nv_bfloat16 *probe_rms_input;
    __nv_bfloat16 *probe_rms_batched;
    __nv_bfloat16 *probe_rms_split;
    __nv_bfloat16 *probe_q_actual;
    __nv_bfloat16 *probe_q_batched;
    __nv_bfloat16 *probe_q_split;
    cudaEvent_t probe_event;
    gemma4_batch_probe_state probe_state;
    gemma4_listener_batch_probe_info probe_info;
    __nv_bfloat16 *trace_arena;
    size_t trace_arena_bytes;
    cudaEvent_t trace_event;
    gemma4_batch_trace_state trace_state;
    gemma4_batch_trace_record trace_records[GEMMA4_BATCH_TRACE_STAGE_COUNT];
    uint32_t trace_active_slots[2];
    uint64_t forward_calls;
    uint64_t layers_executed;
    uint64_t projection_gemm_calls;
    uint64_t projection_rows;
    uint64_t session_rows;
};

__global__ static void gemma4_batch_fill_projection_ptrs(
    const void *weight,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    const void **a_ptrs,
    const void **b_ptrs,
    void **c_ptrs,
    uint32_t session_count,
    uint32_t rows_per_session,
    uint32_t input_width,
    uint32_t output_width
) {
    const uint32_t sid = blockIdx.x * blockDim.x + threadIdx.x;
    if (sid >= session_count) return;
    a_ptrs[sid] = weight;
    b_ptrs[sid] = input + (size_t)sid * rows_per_session * input_width;
    c_ptrs[sid] = output + (size_t)sid * rows_per_session * output_width;
}

static uint32_t gemma4_batch_graph_bucket_count(uint32_t capacity) {
    if (!capacity) return 0;
    uint32_t count = 1;
    uint32_t width = 1;
    while (width < capacity) {
        width = width > capacity / 2u ? capacity : width * 2u;
        ++count;
    }
    return count;
}

static uint32_t gemma4_batch_graph_bucket_width(uint32_t capacity, uint32_t index) {
    if (!capacity || index >= gemma4_batch_graph_bucket_count(capacity)) return 0;
    uint32_t width = 1;
    while (index-- > 0) width = width > capacity / 2u ? capacity : width * 2u;
    return width;
}

/* Run the standard ABI3 per-session GemmEx shape on independent CUDA lanes.
 * Graph mode dispatches the smallest prewarmed geometric width that covers
 * the active rows, avoiding work proportional to the maximum capacity. */
static int gemma4_batch_run_per_session_gemm(
    gemma4_listener_batch_ctx *batch,
    uint32_t operation_id,
    const void *weight,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t rows_per_session,
    uint32_t output_width,
    uint32_t input_width,
    cudaStream_t stream
) {
    if (!batch || !weight || !input || !output || rows == 0 ||
        rows_per_session == 0 || rows % rows_per_session != 0 ||
        !output_width || !input_width) return -1;
    const uint32_t session_count = rows / rows_per_session;
    if (session_count > batch->capacity) return -2;
    if (batch->gemm_graphs_enabled) {
        if (rows_per_session != GEMMA4_STREAM_Q_LEN ||
            operation_id >= GEMMA4_BATCH_GEMM_OP_COUNT || !batch->gemm_graphs ||
            !batch->gemm_graph_bucket_count)
            return -7;
        uint32_t bucket_index = 0;
        uint32_t bucket_width = 0;
        for (; bucket_index < batch->gemm_graph_bucket_count; ++bucket_index) {
            const uint32_t candidate_width = gemma4_batch_graph_bucket_width(
                batch->capacity, bucket_index);
            if (candidate_width >= session_count) {
                bucket_width = candidate_width;
                break;
            }
        }
        if (!bucket_width || bucket_width > batch->capacity) return -7;
        const gemma4_batch_gemm_graph &entry = batch->gemm_graphs[
            (size_t)bucket_index * GEMMA4_BATCH_GEMM_OP_COUNT + operation_id];
        if (!entry.ready || !entry.executable || entry.weight != weight ||
            entry.input != input || entry.output != output ||
            entry.m != rows_per_session || entry.n != output_width ||
            entry.k != input_width || entry.lane_width != bucket_width) return -7;
        return cudaGraphLaunch(entry.executable, stream) == cudaSuccess ? 0 : -5;
    }
    if (!batch->gemm_graphs_enabled && session_count == 1) {
        return run_linear_cublas(batch->model->cublas, weight, input, output,
            (int)rows_per_session, (int)output_width, (int)input_width, stream);
    }
    if (!batch->gemm_lanes || !batch->gemm_ready_event) return -2;
    if (cudaEventRecord(batch->gemm_ready_event, stream) != cudaSuccess) return -3;
    for (uint32_t i = 0; i < session_count; ++i) {
        gemma4_batch_gemm_lane &lane = batch->gemm_lanes[i];
        if (!lane.handle || !lane.stream || !lane.done_event ||
            cudaStreamWaitEvent(lane.stream, batch->gemm_ready_event, 0) != cudaSuccess)
            return -4;
        const size_t input_offset = (size_t)i * rows_per_session * input_width;
        const size_t output_offset = (size_t)i * rows_per_session * output_width;
        if (run_linear_cublas(lane.handle, weight, input + input_offset,
                output + output_offset, (int)rows_per_session,
                (int)output_width, (int)input_width, lane.stream) != 0 ||
            cudaEventRecord(lane.done_event, lane.stream) != cudaSuccess) return -5;
    }
    for (uint32_t i = 0; i < session_count; ++i) {
        if (cudaStreamWaitEvent(stream, batch->gemm_lanes[i].done_event, 0) != cudaSuccess)
            return -6;
    }
    return 0;
}

static const uint64_t GEMMA4_BATCH_CAPABILITIES_V2 =
    UINT64_C(1) | /* cache import for 34/291 token prefixes */
    UINT64_C(2) | /* true N-row shared-weight forward */
    UINT64_C(4) | /* fenced borrowed output lease */
    UINT64_C(8) | /* per-session position rebase */
    UINT64_C(16) | /* safe private-owner output lease abandonment */
    UINT64_C(32) | /* private session-slot retirement */
    UINT64_C(64) | /* read-only per-session K/V and cursor snapshots for parity gates */
    UINT64_C(128) | /* staged one-shot RMS/GEMM numerical diagnosis */
    UINT64_C(256) | /* opt-in all-stage N2 operator trace */
    UINT64_C(512) | /* arbitrary positive row count with safe capacity growth */
    UINT64_C(1024) | /* latest-generation row-bound aggregate output health */
    UINT64_C(2048) | /* opt-in prewarmed geometric-bucket GEMM graphs per operation */
    UINT64_C(4096) | /* owner-ordered cross-context cache promotion */
    UINT64_C(8192) | /* logical Q0/Q1 commit after physical Q2 forward */
    UINT64_C(16384) | /* text embeddings with model-projected token PLE */
    UINT64_C(32768) | /* compare-and-swap rollback of an assistant KV tail */
    UINT64_C(65536) | /* queued Q2 audio replay without listener-cursor advance */
    UINT64_C(131072); /* same-owner private session-slot fork */

static bool gemma4_batch_size_supported(uint32_t n) {
    return n >= 1;
}

static bool gemma4_batch_row_identity_matches(
    const gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    bool require_position
) {
    if (!batch || !row || slot >= batch->capacity) return false;
    const gemma4_listener_batch_slot &state = batch->slots[slot];
    return state.primed && row->native_context_id != 0 &&
        row->native_context_id == state.context_id &&
        row->session_generation == state.session_generation &&
        row->state_epoch_id == state.state_epoch_id &&
        (!require_position || row->cache_position == state.current_position);
}

static void gemma4_batch_poison(gemma4_listener_batch_ctx *batch) {
    if (!batch) return;
    batch->poisoned = true;
    if (batch->model) batch->model->poisoned = true;
}

static int gemma4_batch_bind_stream(
    gemma4_listener_batch_ctx *batch,
    cudaStream_t stream
) {
    if (!batch) return -1;
    if (batch->stream_bound && batch->listener_stream != stream) return -6;
    if (!batch->stream_bound) {
        int actual_priority = 0;
        if (cudaStreamGetPriority(stream, &actual_priority) != cudaSuccess ||
            actual_priority != batch->configured_stream_priority) return -13;
        batch->stream_bound = true;
        batch->listener_stream = stream;
        batch->priority_verified = true;
    }
    if (batch->model->last_stream != 0 && batch->model->last_stream != stream) return -6;
    if (batch->model->last_stream != 0) {
        const cudaError_t query = cudaStreamQuery(batch->model->last_stream);
        if (query != cudaSuccess && query != cudaErrorNotReady) {
            gemma4_batch_poison(batch);
            return -12;
        }
    }
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -11;
    }
    return 0;
}

struct gemma4_batch_workspace {
    __nv_bfloat16 *hidden_a, *hidden_b, *norm, *q, *k, *v, *attn;
    __nv_bfloat16 *gate, *up, *mlp, *ple_raw, *projected_ple;
    __nv_bfloat16 *ple_gate, *ple_act, *ple_out, *output;
    const void **gemm_a_ptrs;
    const void **gemm_b_ptrs;
    void **gemm_c_ptrs;
};

static bool gemma4_batch_mul_size(size_t left, size_t right, size_t *out) {
    if (!out || (left != 0 && right > SIZE_MAX / left)) return false;
    *out = left * right;
    return true;
}

static bool gemma4_batch_product3(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab = 0;
    return gemma4_batch_mul_size(a, b, &ab) && gemma4_batch_mul_size(ab, c, out);
}

static void gemma4_batch_workspace_free(gemma4_batch_workspace *workspace) {
    if (!workspace) return;
    if (workspace->hidden_a) cudaFree(workspace->hidden_a);
    if (workspace->hidden_b) cudaFree(workspace->hidden_b);
    if (workspace->norm) cudaFree(workspace->norm);
    if (workspace->q) cudaFree(workspace->q);
    if (workspace->k) cudaFree(workspace->k);
    if (workspace->v) cudaFree(workspace->v);
    if (workspace->attn) cudaFree(workspace->attn);
    if (workspace->gate) cudaFree(workspace->gate);
    if (workspace->up) cudaFree(workspace->up);
    if (workspace->mlp) cudaFree(workspace->mlp);
    if (workspace->ple_raw) cudaFree(workspace->ple_raw);
    if (workspace->projected_ple) cudaFree(workspace->projected_ple);
    if (workspace->ple_gate) cudaFree(workspace->ple_gate);
    if (workspace->ple_act) cudaFree(workspace->ple_act);
    if (workspace->ple_out) cudaFree(workspace->ple_out);
    if (workspace->output) cudaFree(workspace->output);
    if (workspace->gemm_a_ptrs) cudaFree((void *)workspace->gemm_a_ptrs);
    if (workspace->gemm_b_ptrs) cudaFree((void *)workspace->gemm_b_ptrs);
    if (workspace->gemm_c_ptrs) cudaFree(workspace->gemm_c_ptrs);
    memset(workspace, 0, sizeof(*workspace));
}

static bool gemma4_batch_workspace_alloc_field(
    void **ptr, size_t elements, size_t element_size, const char *name
) {
    size_t bytes = 0;
    if (!ptr || !gemma4_batch_mul_size(elements, element_size, &bytes) || !bytes) return false;
    const cudaError_t status = cudaMalloc(ptr, bytes);
    if (status != cudaSuccess) {
        gemma4_log_create_failure(name, bytes, "cudaMalloc", true, status);
        return false;
    }
    return true;
}

static int gemma4_batch_workspace_allocate(
    uint32_t capacity,
    gemma4_batch_workspace *workspace
) {
    if (!capacity || !workspace) return -1;
    memset(workspace, 0, sizeof(*workspace));
    size_t rows = 0;
    if (!gemma4_batch_mul_size((size_t)capacity, GEMMA4_STREAM_Q_LEN, &rows)) return -20;
#define GEMMA4_WORKSPACE_ALLOC(field, a, b) \
    do { \
        size_t elements = 0; \
        if (!gemma4_batch_product3(rows, (size_t)(a), (size_t)(b), &elements)) { \
            gemma4_batch_workspace_free(workspace); return -20; \
        } \
        if (!gemma4_batch_workspace_alloc_field((void **)&workspace->field, elements, \
                sizeof(__nv_bfloat16), "batch." #field)) { \
            gemma4_batch_workspace_free(workspace); return -4; \
        } \
    } while (0)
    GEMMA4_WORKSPACE_ALLOC(hidden_a, GEMMA4_HIDDEN_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(hidden_b, GEMMA4_HIDDEN_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(norm, GEMMA4_HIDDEN_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(q, GEMMA4_NUM_ATTN_HEADS, 512u);
    GEMMA4_WORKSPACE_ALLOC(k, 512u, 1u);
    GEMMA4_WORKSPACE_ALLOC(v, 512u, 1u);
    GEMMA4_WORKSPACE_ALLOC(attn, GEMMA4_NUM_ATTN_HEADS, 512u);
    GEMMA4_WORKSPACE_ALLOC(gate, GEMMA4_MAX_INTERMEDIATE_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(up, GEMMA4_MAX_INTERMEDIATE_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(mlp, GEMMA4_MAX_INTERMEDIATE_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(ple_raw, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM);
    GEMMA4_WORKSPACE_ALLOC(projected_ple, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM);
    GEMMA4_WORKSPACE_ALLOC(ple_gate, GEMMA4_PLE_DIM, 1u);
    GEMMA4_WORKSPACE_ALLOC(ple_act, GEMMA4_PLE_DIM, 1u);
    GEMMA4_WORKSPACE_ALLOC(ple_out, GEMMA4_HIDDEN_SIZE, 1u);
    GEMMA4_WORKSPACE_ALLOC(output, GEMMA4_HIDDEN_SIZE, 1u);
#undef GEMMA4_WORKSPACE_ALLOC
    size_t pointer_bytes = 0;
    if (!gemma4_batch_mul_size((size_t)capacity, sizeof(void *), &pointer_bytes)) {
        gemma4_batch_workspace_free(workspace);
        return -20;
    }
    if (!gemma4_batch_workspace_alloc_field((void **)&workspace->gemm_a_ptrs,
            pointer_bytes, 1u, "batch.gemm_a_ptrs") ||
        !gemma4_batch_workspace_alloc_field((void **)&workspace->gemm_b_ptrs,
            pointer_bytes, 1u, "batch.gemm_b_ptrs") ||
        !gemma4_batch_workspace_alloc_field((void **)&workspace->gemm_c_ptrs,
            pointer_bytes, 1u, "batch.gemm_c_ptrs")) {
        gemma4_batch_workspace_free(workspace);
        return -4;
    }
    return 0;
}

static gemma4_batch_workspace gemma4_batch_workspace_take(
    const gemma4_listener_batch_ctx *batch
) {
    gemma4_batch_workspace workspace = {};
    workspace.hidden_a = batch->hidden_a; workspace.hidden_b = batch->hidden_b;
    workspace.norm = batch->norm; workspace.q = batch->q; workspace.k = batch->k;
    workspace.v = batch->v; workspace.attn = batch->attn; workspace.gate = batch->gate;
    workspace.up = batch->up; workspace.mlp = batch->mlp; workspace.ple_raw = batch->ple_raw;
    workspace.projected_ple = batch->projected_ple; workspace.ple_gate = batch->ple_gate;
    workspace.ple_act = batch->ple_act; workspace.ple_out = batch->ple_out;
    workspace.output = batch->output; workspace.gemm_a_ptrs = batch->gemm_a_ptrs;
    workspace.gemm_b_ptrs = batch->gemm_b_ptrs; workspace.gemm_c_ptrs = batch->gemm_c_ptrs;
    return workspace;
}

static void gemma4_batch_workspace_install(
    gemma4_listener_batch_ctx *batch,
    gemma4_batch_workspace *workspace
) {
    batch->hidden_a = workspace->hidden_a; batch->hidden_b = workspace->hidden_b;
    batch->norm = workspace->norm; batch->q = workspace->q; batch->k = workspace->k;
    batch->v = workspace->v; batch->attn = workspace->attn; batch->gate = workspace->gate;
    batch->up = workspace->up; batch->mlp = workspace->mlp; batch->ple_raw = workspace->ple_raw;
    batch->projected_ple = workspace->projected_ple; batch->ple_gate = workspace->ple_gate;
    batch->ple_act = workspace->ple_act; batch->ple_out = workspace->ple_out;
    batch->output = workspace->output; batch->gemm_a_ptrs = workspace->gemm_a_ptrs;
    batch->gemm_b_ptrs = workspace->gemm_b_ptrs; batch->gemm_c_ptrs = workspace->gemm_c_ptrs;
    memset(workspace, 0, sizeof(*workspace));
}

static void gemma4_batch_workspace_free_ctx(gemma4_listener_batch_ctx *batch) {
    gemma4_batch_workspace workspace = gemma4_batch_workspace_take(batch);
    gemma4_batch_workspace_free(&workspace);
    gemma4_batch_workspace empty = {};
    gemma4_batch_workspace_install(batch, &empty);
}

static void gemma4_batch_destroy_graph_table(
    gemma4_listener_batch_ctx *batch,
    gemma4_batch_gemm_graph *graphs,
    uint32_t bucket_count
) {
    if (!graphs) return;
    if (batch && batch->listener_stream) cudaStreamSynchronize(batch->listener_stream);
    for (uint32_t i = 0; batch && batch->gemm_lanes && i < batch->capacity; ++i) {
        if (batch->gemm_lanes[i].stream) cudaStreamSynchronize(batch->gemm_lanes[i].stream);
    }
    const size_t graph_entries = (size_t)bucket_count * GEMMA4_BATCH_GEMM_OP_COUNT;
    for (size_t op = 0; op < graph_entries; ++op) {
        gemma4_batch_gemm_graph &entry = graphs[op];
        if (entry.executable) cudaGraphExecDestroy(entry.executable);
        if (entry.graph) cudaGraphDestroy(entry.graph);
    }
    free(graphs);
}

static void gemma4_batch_clear_graphs(gemma4_listener_batch_ctx *batch) {
    if (!batch) return;
    gemma4_batch_destroy_graph_table(
        batch, batch->gemm_graphs, batch->gemm_graph_bucket_count);
    batch->gemm_graphs = nullptr;
    batch->gemm_graphs_enabled = false;
    batch->gemm_graph_bucket_count = 0;
    batch->gemm_graph_count = 0;
}

static int gemma4_batch_capture_failure(
    uint32_t operation_id,
    uint32_t capacity,
    uint32_t bucket_width,
    const char *stage,
    int native_status,
    cudaError_t cuda_status
) {
    fprintf(stderr,
        "[gemma4_listener_batch] GEMM graph capture failed op=%u capacity=%u bucket=%u "
        "stage=%s status=%d cuda_status=%d cuda_error=%s\n",
        operation_id, capacity, bucket_width, stage, native_status, (int)cuda_status,
        cudaGetErrorString(cuda_status));
    return native_status;
}

static int gemma4_batch_capture_gemm_graph(
    gemma4_listener_batch_ctx *batch,
    gemma4_batch_gemm_graph *graphs,
    uint64_t *graph_count,
    uint32_t operation_id,
    uint32_t bucket_width,
    const void *weight,
    const __nv_bfloat16 *input,
    __nv_bfloat16 *output,
    uint32_t n,
    uint32_t k
) {
    if (!batch || !graphs || !graph_count || !batch->gemm_lanes ||
        !batch->gemm_ready_event || !batch->listener_stream ||
        operation_id >= GEMMA4_BATCH_GEMM_OP_COUNT || !weight || !input || !output ||
        !n || !k || !batch->capacity || !bucket_width ||
        bucket_width > batch->capacity) return -1;
    gemma4_batch_gemm_graph &entry = graphs[operation_id];
    if (entry.ready) return -2;
    cudaError_t cuda_status = cudaStreamBeginCapture(
        batch->listener_stream, cudaStreamCaptureModeThreadLocal);
    if (cuda_status != cudaSuccess) {
        return gemma4_batch_capture_failure(
            operation_id, batch->capacity, bucket_width,
            "begin_capture", -3, cuda_status);
    }

    int capture_status = 0;
    const char *capture_stage = nullptr;
    cuda_status = cudaEventRecord(batch->gemm_ready_event, batch->listener_stream);
    if (cuda_status != cudaSuccess) {
        capture_status = -4;
        capture_stage = "record_ready_event";
    }
    for (uint32_t i = 0; capture_status == 0 && i < bucket_width; ++i) {
        gemma4_batch_gemm_lane &lane = batch->gemm_lanes[i];
        const size_t input_offset = (size_t)i * GEMMA4_STREAM_Q_LEN * k;
        const size_t output_offset = (size_t)i * GEMMA4_STREAM_Q_LEN * n;
        if (!lane.handle || !lane.stream || !lane.done_event || !lane.cublas_workspace ||
            lane.cublas_workspace_bytes < GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES) {
            capture_status = -5;
            capture_stage = "lane_resources";
            break;
        }
        cuda_status = cudaStreamWaitEvent(lane.stream, batch->gemm_ready_event, 0);
        if (cuda_status != cudaSuccess) {
            capture_status = -5;
            capture_stage = "lane_wait_ready";
            break;
        }
        if (run_linear_cublas_bound(lane.handle, weight, input + input_offset,
                output + output_offset, GEMMA4_STREAM_Q_LEN, (int)n, (int)k,
                lane.stream) != 0) {
            capture_status = -6;
            capture_stage = "cublas_gemm";
            break;
        }
        cuda_status = cudaEventRecord(lane.done_event, lane.stream);
        if (cuda_status != cudaSuccess) {
            capture_status = -7;
            capture_stage = "record_lane_done";
            break;
        }
    }
    for (uint32_t i = 0; capture_status == 0 && i < bucket_width; ++i) {
        cuda_status = cudaStreamWaitEvent(batch->listener_stream,
            batch->gemm_lanes[i].done_event, 0);
        if (cuda_status != cudaSuccess) {
            capture_status = -8;
            capture_stage = "owner_wait_lane_done";
        }
    }

    cudaGraph_t graph = nullptr;
    const cudaError_t end_status = cudaStreamEndCapture(batch->listener_stream, &graph);
    if (capture_status != 0 || end_status != cudaSuccess || !graph) {
        if (graph) cudaGraphDestroy(graph);
        if (capture_status != 0) {
            return gemma4_batch_capture_failure(operation_id, batch->capacity, bucket_width,
                capture_stage ? capture_stage : "capture", capture_status, cuda_status);
        }
        return gemma4_batch_capture_failure(operation_id, batch->capacity, bucket_width,
            "end_capture", -9, end_status);
    }
    cudaGraphExec_t executable = nullptr;
    cuda_status = cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (cuda_status != cudaSuccess || !executable) {
        cudaGraphDestroy(graph);
        return gemma4_batch_capture_failure(operation_id, batch->capacity, bucket_width,
            "instantiate", -10, cuda_status);
    }
    entry.graph = graph;
    entry.executable = executable;
    entry.weight = weight;
    entry.input = input;
    entry.output = output;
    entry.m = GEMMA4_STREAM_Q_LEN;
    entry.n = n;
    entry.k = k;
    entry.lane_width = bucket_width;
    entry.ready = true;
    (*graph_count)++;
    return 0;
}

static int gemma4_batch_build_gemm_graphs(
    gemma4_listener_batch_ctx *batch,
    gemma4_batch_gemm_graph **out_graphs,
    uint64_t *out_graph_count,
    uint32_t *out_bucket_count
) {
    if (!batch || !out_graphs || !out_graph_count || !out_bucket_count || !batch->stream_bound ||
        !batch->listener_stream || !batch->capacity || !batch->gemm_lanes) return -1;
    *out_graphs = nullptr;
    *out_graph_count = 0;
    *out_bucket_count = 0;
    const uint32_t bucket_count = gemma4_batch_graph_bucket_count(batch->capacity);
    if (!bucket_count || (size_t)bucket_count >
        SIZE_MAX / GEMMA4_BATCH_GEMM_OP_COUNT / sizeof(gemma4_batch_gemm_graph)) return -2;
    const size_t graph_entries = (size_t)bucket_count * GEMMA4_BATCH_GEMM_OP_COUNT;
    gemma4_batch_gemm_graph *graphs = static_cast<gemma4_batch_gemm_graph *>(
        calloc(graph_entries, sizeof(gemma4_batch_gemm_graph)));
    if (!graphs) return -2;
    const size_t hidden_bytes = (size_t)batch->capacity * GEMMA4_STREAM_Q_LEN *
        GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16);
    const size_t ple_act_bytes = (size_t)batch->capacity * GEMMA4_STREAM_Q_LEN *
        GEMMA4_PLE_DIM * sizeof(__nv_bfloat16);
#define GEMMA4_GRAPH_CLEAR(field, bytes) \
    do { \
        const cudaError_t clear_status = cudaMemsetAsync( \
            batch->field, 0, (bytes), batch->listener_stream); \
        if (clear_status != cudaSuccess) { \
            fprintf(stderr, \
                "[gemma4_listener_batch] graph prepare clear failed field=%s capacity=%u bytes=%zu status=%d error=%s\\n", \
                #field, batch->capacity, (size_t)(bytes), (int)clear_status, \
                cudaGetErrorString(clear_status)); \
            gemma4_batch_destroy_graph_table(batch, graphs, bucket_count); \
            return -3; \
        } \
    } while (0)
    GEMMA4_GRAPH_CLEAR(hidden_a, hidden_bytes);
    GEMMA4_GRAPH_CLEAR(norm, hidden_bytes);
    GEMMA4_GRAPH_CLEAR(attn, hidden_bytes);
    GEMMA4_GRAPH_CLEAR(mlp, hidden_bytes);
    GEMMA4_GRAPH_CLEAR(ple_act, ple_act_bytes);
#undef GEMMA4_GRAPH_CLEAR
    const cudaError_t clear_sync_status = cudaStreamSynchronize(batch->listener_stream);
    if (clear_sync_status != cudaSuccess) {
        fprintf(stderr,
            "[gemma4_listener_batch] graph prepare clear synchronization failed capacity=%u status=%d error=%s\\n",
            batch->capacity, (int)clear_sync_status, cudaGetErrorString(clear_sync_status));
        gemma4_batch_destroy_graph_table(batch, graphs, bucket_count);
        return -3;
    }

    gemma4_listener_ctx *ctx = batch->model;
    uint64_t graph_count = 0;
    int status = 0;
    for (uint32_t bucket_index = 0;
         status == 0 && bucket_index < bucket_count; ++bucket_index) {
        const uint32_t bucket_width = gemma4_batch_graph_bucket_width(
            batch->capacity, bucket_index);
        gemma4_batch_gemm_graph *bucket_graphs = graphs +
            (size_t)bucket_index * GEMMA4_BATCH_GEMM_OP_COUNT;
        status = gemma4_batch_capture_gemm_graph(batch, bucket_graphs, &graph_count,
            GEMMA4_BATCH_GEMM_OP_PRELUDE, bucket_width,
            ctx->weights.per_layer_model_proj_w, batch->hidden_a, batch->ple_raw,
            GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM, GEMMA4_HIDDEN_SIZE);
        for (int layer = 0; status == 0 && layer < GEMMA4_NUM_LAYERS; ++layer) {
            const GemmaLayerSpec *spec = &gemma4_layers[layer];
            const gemma4_layer_weights *lw = &ctx->weights.layers[layer];
            const uint32_t base = GEMMA4_BATCH_GEMM_OP_ID(layer, 0);
            const uint32_t d = spec->head_dim;
            const uint32_t qdim = GEMMA4_NUM_ATTN_HEADS * d;
            const uint32_t intermediate = spec->intermediate_size;
            status = gemma4_batch_capture_gemm_graph(batch, bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_Q, bucket_width,
                lw->q_proj_w, batch->norm, batch->q, qdim, GEMMA4_HIDDEN_SIZE);
            if (status == 0 && !spec->is_kv_shared) {
                status = gemma4_batch_capture_gemm_graph(batch, bucket_graphs, &graph_count,
                    base + GEMMA4_BATCH_GEMM_OP_K, bucket_width,
                    lw->k_proj_w, batch->norm, batch->k, d, GEMMA4_HIDDEN_SIZE);
                if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                    bucket_graphs, &graph_count,
                    base + GEMMA4_BATCH_GEMM_OP_V, bucket_width,
                    lw->v_proj_w, batch->norm, batch->v, d, GEMMA4_HIDDEN_SIZE);
            }
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_O, bucket_width,
                lw->o_proj_w, batch->attn, batch->hidden_b,
                GEMMA4_HIDDEN_SIZE, qdim);
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_GATE, bucket_width,
                lw->gate_proj_w, batch->norm, batch->gate,
                intermediate, GEMMA4_HIDDEN_SIZE);
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_UP, bucket_width,
                lw->up_proj_w, batch->norm, batch->up,
                intermediate, GEMMA4_HIDDEN_SIZE);
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_DOWN, bucket_width,
                lw->down_proj_w, batch->mlp, batch->hidden_b,
                GEMMA4_HIDDEN_SIZE, intermediate);
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_PLE_GATE, bucket_width,
                lw->ple_gate_w, batch->hidden_a, batch->ple_gate,
                GEMMA4_PLE_DIM, GEMMA4_HIDDEN_SIZE);
            if (status == 0) status = gemma4_batch_capture_gemm_graph(batch,
                bucket_graphs, &graph_count,
                base + GEMMA4_BATCH_GEMM_OP_PLE_PROJ, bucket_width,
                lw->ple_proj_w, batch->ple_act, batch->ple_out,
                GEMMA4_HIDDEN_SIZE, GEMMA4_PLE_DIM);
        }
    }
    if (status != 0) {
        gemma4_batch_destroy_graph_table(batch, graphs, bucket_count);
        return status;
    }
    *out_graphs = graphs;
    *out_graph_count = graph_count;
    *out_bucket_count = bucket_count;
    return 0;
}

static int gemma4_batch_prepare_gemm_graphs_internal(
    gemma4_listener_batch_ctx *batch
) {
    gemma4_batch_gemm_graph *graphs = nullptr;
    uint64_t graph_count = 0;
    uint32_t bucket_count = 0;
    const int status = gemma4_batch_build_gemm_graphs(
        batch, &graphs, &graph_count, &bucket_count);
    if (status != 0) return status;
    gemma4_batch_clear_graphs(batch);
    batch->gemm_graphs = graphs;
    batch->gemm_graph_count = graph_count;
    batch->gemm_graph_bucket_count = bucket_count;
    batch->gemm_graphs_enabled = true;
    return 0;
}

static void gemma4_batch_destroy_lane_range(
    gemma4_batch_gemm_lane *lanes, uint32_t begin, uint32_t end
) {
    if (!lanes) return;
    for (uint32_t i = begin; i < end; ++i) {
        gemma4_batch_gemm_lane &lane = lanes[i];
        if (lane.stream) cudaStreamSynchronize(lane.stream);
        if (lane.done_event) cudaEventDestroy(lane.done_event);
        if (lane.handle) cublasDestroy(lane.handle);
        if (lane.cublas_workspace) cudaFree(lane.cublas_workspace);
        if (lane.stream) cudaStreamDestroy(lane.stream);
        memset(&lane, 0, sizeof(lane));
    }
}

static int gemma4_batch_create_lane(gemma4_batch_gemm_lane *lane, int32_t priority) {
    if (!lane) return -1;
    memset(lane, 0, sizeof(*lane));
    if (cudaStreamCreateWithPriority(&lane->stream, cudaStreamNonBlocking, priority) != cudaSuccess ||
        cublasCreate(&lane->handle) != CUBLAS_STATUS_SUCCESS ||
        cublasSetStream(lane->handle, lane->stream) != CUBLAS_STATUS_SUCCESS ||
        cudaMalloc(&lane->cublas_workspace, GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES) != cudaSuccess ||
        cublasSetWorkspace(lane->handle, lane->cublas_workspace,
            GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES) != CUBLAS_STATUS_SUCCESS ||
        cudaEventCreateWithFlags(&lane->done_event, cudaEventDisableTiming) != cudaSuccess) {
        gemma4_batch_destroy_lane_range(lane, 0, 1);
        return -4;
    }
    lane->cublas_workspace_bytes = GEMMA4_BATCH_CUBLAS_WORKSPACE_BYTES;
    return 0;
}

static int gemma4_batch_resize_capacity(
    gemma4_listener_batch_ctx *batch, uint32_t target_capacity
) {
    if (!batch || !batch->model || target_capacity <= batch->capacity) return -1;
    const uint32_t old_capacity = batch->capacity;
    if ((size_t)target_capacity > SIZE_MAX / sizeof(gemma4_listener_batch_slot) ||
        (size_t)target_capacity > SIZE_MAX / sizeof(uint32_t) ||
        (size_t)target_capacity > SIZE_MAX / sizeof(gemma4_listener_batch_row) ||
        (size_t)target_capacity > SIZE_MAX / sizeof(gemma4_batch_gemm_lane)) return -20;
    gemma4_listener_batch_slot *new_slots = static_cast<gemma4_listener_batch_slot *>(
        calloc(target_capacity, sizeof(gemma4_listener_batch_slot)));
    uint32_t *new_active = static_cast<uint32_t *>(calloc(target_capacity, sizeof(uint32_t)));
    gemma4_listener_batch_row *new_rows = static_cast<gemma4_listener_batch_row *>(
        calloc(target_capacity, sizeof(gemma4_listener_batch_row)));
    gemma4_batch_gemm_lane *new_lanes = static_cast<gemma4_batch_gemm_lane *>(
        calloc(target_capacity, sizeof(gemma4_batch_gemm_lane)));
    gemma4_batch_workspace new_workspace = {};
    if (!new_slots || !new_active || !new_rows || !new_lanes) {
        free(new_slots); free(new_active); free(new_rows); free(new_lanes);
        return -3;
    }
    if (old_capacity) {
        memcpy(new_slots, batch->slots, (size_t)old_capacity * sizeof(*new_slots));
        memcpy(new_active, batch->active_slots, (size_t)old_capacity * sizeof(*new_active));
        memcpy(new_rows, batch->output_rows, (size_t)old_capacity * sizeof(*new_rows));
        memcpy(new_lanes, batch->gemm_lanes, (size_t)old_capacity * sizeof(*new_lanes));
    } else {
        new_slots[0].cache = batch->model->cache;
        new_slots[0].prefix_attention = batch->model->d_prefix_attention;
    }
    const int workspace_status = gemma4_batch_workspace_allocate(target_capacity, &new_workspace);
    if (workspace_status != 0) {
        free(new_slots); free(new_active); free(new_rows); free(new_lanes);
        return workspace_status;
    }
    const uint32_t first_new_slot = old_capacity ? old_capacity : 1u;
    for (uint32_t i = first_new_slot; i < target_capacity; ++i) {
        if (gemma4_cache_create(&new_slots[i].cache,
                batch->model->config.max_seq_len, true) != 0 ||
            cudaMalloc((void **)&new_slots[i].prefix_attention,
                GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN * sizeof(int64_t)) != cudaSuccess) {
            for (uint32_t j = first_new_slot; j <= i; ++j) {
                if (new_slots[j].cache) gemma4_cache_destroy(new_slots[j].cache);
                if (new_slots[j].prefix_attention) cudaFree(new_slots[j].prefix_attention);
            }
            gemma4_batch_workspace_free(&new_workspace);
            free(new_slots); free(new_active); free(new_rows); free(new_lanes);
            return -4;
        }
    }
    for (uint32_t i = old_capacity; i < target_capacity; ++i) {
        if (gemma4_batch_create_lane(&new_lanes[i], batch->configured_stream_priority) != 0) {
            gemma4_batch_destroy_lane_range(new_lanes, old_capacity, i + 1u);
            for (uint32_t j = first_new_slot; j < target_capacity; ++j) {
                if (new_slots[j].cache) gemma4_cache_destroy(new_slots[j].cache);
                if (new_slots[j].prefix_attention) cudaFree(new_slots[j].prefix_attention);
            }
            gemma4_batch_workspace_free(&new_workspace);
            free(new_slots); free(new_active); free(new_rows); free(new_lanes);
            return -4;
        }
    }
    const bool rebuild_gemm_graphs = batch->gemm_graphs_enabled;
    gemma4_batch_gemm_graph *new_graphs = nullptr;
    uint64_t new_graph_count = 0;
    uint32_t new_graph_bucket_count = 0;
    if (rebuild_gemm_graphs) {
        gemma4_listener_batch_ctx candidate = *batch;
        candidate.capacity = target_capacity;
        candidate.gemm_lanes = new_lanes;
        candidate.gemm_graphs = nullptr;
        candidate.gemm_graphs_enabled = false;
        candidate.gemm_graph_bucket_count = 0;
        candidate.gemm_graph_count = 0;
        gemma4_batch_workspace candidate_workspace = new_workspace;
        gemma4_batch_workspace_install(&candidate, &candidate_workspace);
        const int graph_status = gemma4_batch_build_gemm_graphs(
            &candidate, &new_graphs, &new_graph_count, &new_graph_bucket_count);
        if (graph_status != 0) {
            /* Candidate capture failed; keep the old capacity, slots,
             * workspace and graph cache intact. */
            gemma4_batch_destroy_lane_range(new_lanes, old_capacity, target_capacity);
            for (uint32_t j = first_new_slot; j < target_capacity; ++j) {
                if (new_slots[j].cache) gemma4_cache_destroy(new_slots[j].cache);
                if (new_slots[j].prefix_attention) cudaFree(new_slots[j].prefix_attention);
            }
            gemma4_batch_workspace_free(&new_workspace);
            free(new_slots); free(new_active); free(new_rows); free(new_lanes);
            return -1000 + graph_status;
        }
    }
    gemma4_listener_batch_slot *old_slots = batch->slots;
    uint32_t *old_active = batch->active_slots;
    gemma4_listener_batch_row *old_rows = batch->output_rows;
    gemma4_batch_gemm_lane *old_lanes = batch->gemm_lanes;
    /* All allocations succeeded. Retire the old operation graphs before
     * releasing their captured workspace addresses or transferring lanes. */
    gemma4_batch_clear_graphs(batch);
    gemma4_batch_workspace old_workspace = gemma4_batch_workspace_take(batch);
    batch->slots = new_slots; batch->active_slots = new_active;
    batch->output_rows = new_rows; batch->gemm_lanes = new_lanes;
    gemma4_batch_workspace_install(batch, &new_workspace);
    batch->capacity = target_capacity;
    if (rebuild_gemm_graphs) {
        batch->gemm_graphs = new_graphs;
        batch->gemm_graph_count = new_graph_count;
        batch->gemm_graph_bucket_count = new_graph_bucket_count;
        batch->gemm_graphs_enabled = true;
    }
    free(old_slots); free(old_active); free(old_rows); free(old_lanes);
    gemma4_batch_workspace_free(&old_workspace);
    return 0;
}

static void gemma4_batch_destroy_allocations(gemma4_listener_batch_ctx *batch) {
    if (!batch) return;
    gemma4_batch_clear_graphs(batch);
    if (batch->completion_event) cudaEventDestroy(batch->completion_event);
    if (batch->clone_event) cudaEventDestroy(batch->clone_event);
    if (batch->probe_event) cudaEventDestroy(batch->probe_event);
    if (batch->trace_event) cudaEventDestroy(batch->trace_event);
    if (batch->gemm_ready_event) cudaEventDestroy(batch->gemm_ready_event);
    for (uint32_t i = 1; batch->slots && i < batch->capacity; ++i) {
        if (batch->slots[i].cache) gemma4_cache_destroy(batch->slots[i].cache);
        if (batch->slots[i].prefix_attention) cudaFree(batch->slots[i].prefix_attention);
    }
    gemma4_batch_destroy_lane_range(batch->gemm_lanes, 0, batch->capacity);
    free(batch->slots); free(batch->active_slots); free(batch->output_rows); free(batch->gemm_lanes);
    gemma4_batch_workspace_free_ctx(batch);
    if (batch->probe_rms_input) cudaFree(batch->probe_rms_input);
    if (batch->probe_rms_batched) cudaFree(batch->probe_rms_batched);
    if (batch->probe_rms_split) cudaFree(batch->probe_rms_split);
    if (batch->probe_q_actual) cudaFree(batch->probe_q_actual);
    if (batch->probe_q_batched) cudaFree(batch->probe_q_batched);
    if (batch->probe_q_split) cudaFree(batch->probe_q_split);
    if (batch->trace_arena) cudaFree(batch->trace_arena);
    if (batch->model) gemma4_listener_destroy(batch->model);
    free(batch);
}

uint32_t gemma4_listener_batch_api_version(void) {
    return GEMMA4_LISTENER_BATCH_API_VERSION;
}

uint64_t gemma4_listener_batch_capabilities(void) {
    return GEMMA4_BATCH_CAPABILITIES_V2;
}

int gemma4_listener_batch_create(
    const gemma4_listener_config *listener_config,
    const gemma4_listener_weights *weights,
    const gemma4_listener_batch_config *batch_config,
    gemma4_listener_batch_ctx **out_batch
) {
    if (!listener_config || !weights || !batch_config || !out_batch ||
        batch_config->struct_size != sizeof(gemma4_listener_batch_config) ||
        batch_config->api_version != GEMMA4_LISTENER_BATCH_API_VERSION ||
        batch_config->max_batch_size == 0 ||
        batch_config->attention_backend != GEMMA4_ATTN_NATIVE_SPLITK_GQA ||
        batch_config->rmsnorm_backend != GEMMA4_RMS_NATIVE_EXACT ||
        (batch_config->listener_stream_priority != -1 &&
         batch_config->listener_stream_priority != 0) ||
        batch_config->reserved[0] || batch_config->reserved[1] ||
        listener_config->enable_cuda_graph ||
        listener_config->max_seq_len != GEMMA4_ATEN_SDPA_MAX_SEQ_LEN ||
        listener_config->stream_q_len != GEMMA4_STREAM_Q_LEN) return -1;
    *out_batch = nullptr;
    gemma4_listener_batch_ctx *batch =
        static_cast<gemma4_listener_batch_ctx *>(calloc(1, sizeof(gemma4_listener_batch_ctx)));
    if (!batch) return -2;
    batch->configured_stream_priority = batch_config->listener_stream_priority;

    gemma4_listener_config model_config = *listener_config;
    model_config.enable_cuda_graph = false;
    /* The legacy owner's output is unused; batch output has its own [N,2,H] slot. */
    model_config.output_tensor_ptr = nullptr;
    model_config.output_meta = nullptr;
    const int create_status = gemma4_listener_create_impl(
        &model_config, weights, &batch->model,
        (int32_t)batch_config->attention_backend,
        (int32_t)batch_config->rmsnorm_backend, true);
    if (create_status != 0) {
        free(batch);
        return create_status;
    }
    if (cudaEventCreateWithFlags(&batch->gemm_ready_event, cudaEventDisableTiming) != cudaSuccess) {
        gemma4_batch_destroy_allocations(batch);
        return -4;
    }
    if (cudaEventCreateWithFlags(&batch->completion_event, cudaEventDisableTiming) != cudaSuccess) {
        gemma4_batch_destroy_allocations(batch);
        return -4;
    }
    if (cudaEventCreateWithFlags(&batch->clone_event, cudaEventDisableTiming) != cudaSuccess) {
        gemma4_batch_destroy_allocations(batch);
        return -4;
    }
    const int capacity_status = gemma4_batch_resize_capacity(
        batch, batch_config->max_batch_size);
    if (capacity_status != 0) {
        gemma4_batch_destroy_allocations(batch);
        return capacity_status;
    }
    *out_batch = batch;
    return 0;
}

int gemma4_listener_batch_destroy(gemma4_listener_batch_ctx *batch) {
    if (!batch) return 0;
    if (batch->output_lease_active) return -6;
    if (batch->listener_stream &&
        cudaStreamSynchronize(batch->listener_stream) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    gemma4_batch_destroy_allocations(batch);
    return 0;
}

int gemma4_listener_batch_ensure_capacity(
    gemma4_listener_batch_ctx *batch,
    uint32_t requested_capacity,
    uint32_t *out_capacity
) {
    if (!batch || !out_capacity || requested_capacity == 0 || batch->poisoned) return -1;
    *out_capacity = batch->capacity;
    if (batch->output_lease_active || batch->probe_state == GEMMA4_BATCH_PROBE_ARMED ||
        batch->trace_state == GEMMA4_BATCH_TRACE_ARMED) return -1;
    if (requested_capacity <= batch->capacity) return 0;
    if (!batch->stream_bound || !batch->listener_stream) return -1;

    uint32_t target_capacity = requested_capacity;
    if (batch->capacity && batch->capacity <= UINT32_MAX / 2u) {
        const uint32_t geometric_capacity = batch->capacity * 2u;
        if (target_capacity < geometric_capacity) target_capacity = geometric_capacity;
    }
    if (cudaStreamSynchronize(batch->listener_stream) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    for (uint32_t i = 0; i < batch->capacity; ++i) {
        if (batch->gemm_lanes[i].stream &&
            cudaStreamSynchronize(batch->gemm_lanes[i].stream) != cudaSuccess) {
            gemma4_batch_poison(batch);
            return -2;
        }
    }
    const int resize_status = gemma4_batch_resize_capacity(batch, target_capacity);
    if (resize_status != 0) {
        return resize_status;
    }
    *out_capacity = batch->capacity;
    return 0;
}

uint32_t gemma4_listener_batch_probe_api_version(void) {
    return GEMMA4_LISTENER_BATCH_PROBE_API_VERSION;
}

static void gemma4_batch_probe_free_buffers(gemma4_listener_batch_ctx *batch) {
    if (!batch) return;
    if (batch->probe_rms_input) cudaFree(batch->probe_rms_input);
    if (batch->probe_rms_batched) cudaFree(batch->probe_rms_batched);
    if (batch->probe_rms_split) cudaFree(batch->probe_rms_split);
    if (batch->probe_q_actual) cudaFree(batch->probe_q_actual);
    if (batch->probe_q_batched) cudaFree(batch->probe_q_batched);
    if (batch->probe_q_split) cudaFree(batch->probe_q_split);
    batch->probe_rms_input = nullptr;
    batch->probe_rms_batched = nullptr;
    batch->probe_rms_split = nullptr;
    batch->probe_q_actual = nullptr;
    batch->probe_q_batched = nullptr;
    batch->probe_q_split = nullptr;
    if (batch->probe_event) {
        cudaEventDestroy(batch->probe_event);
        batch->probe_event = nullptr;
    }
}

int gemma4_listener_batch_probe_arm(gemma4_listener_batch_ctx *batch) {
    if (!batch || batch->poisoned || batch->output_lease_active ||
        !batch->stream_bound ||
        !gemma4_batch_probe_can_arm(batch->probe_state, batch->capacity,
            batch->model ? batch->model->rmsnorm_backend : UINT32_MAX)) return -1;
    const size_t rms_elements = gemma4_batch_probe_elements(
        GEMMA4_BATCH_PROBE_Q_ROWS, GEMMA4_BATCH_PROBE_HIDDEN_DIM);
    const size_t q_max_dim = (size_t)GEMMA4_NUM_ATTN_HEADS * 512u;
    const size_t q_elements = gemma4_batch_probe_elements(
        GEMMA4_BATCH_PROBE_Q_ROWS, (uint32_t)q_max_dim);
    if (!rms_elements || !q_elements ||
        cudaEventCreateWithFlags(&batch->probe_event, cudaEventDisableTiming) != cudaSuccess) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -2;
    }
#define GEMMA4_PROBE_ALLOC(field, elements) \
    do { \
        if (cudaMalloc((void **)&batch->field, \
                (elements) * sizeof(__nv_bfloat16)) != cudaSuccess) { \
            gemma4_batch_probe_free_buffers(batch); \
            batch->probe_state = GEMMA4_BATCH_PROBE_IDLE; \
            return -3; \
        } \
    } while (0)
    GEMMA4_PROBE_ALLOC(probe_rms_input, rms_elements);
    GEMMA4_PROBE_ALLOC(probe_rms_batched, rms_elements);
    GEMMA4_PROBE_ALLOC(probe_rms_split, rms_elements);
    GEMMA4_PROBE_ALLOC(probe_q_actual, q_elements);
    GEMMA4_PROBE_ALLOC(probe_q_batched, q_elements);
    GEMMA4_PROBE_ALLOC(probe_q_split, q_elements);
#undef GEMMA4_PROBE_ALLOC
    memset(&batch->probe_info, 0, sizeof(batch->probe_info));
    batch->probe_info.struct_size = sizeof(batch->probe_info);
    batch->probe_info.rms_backend_id = batch->model->rmsnorm_backend;
    batch->probe_info.row_count = GEMMA4_BATCH_PROBE_Q_ROWS;
    batch->probe_state = GEMMA4_BATCH_PROBE_ARMED;
    batch->probe_info.state = (uint32_t)batch->probe_state;
    return 0;
}

static bool gemma4_batch_probe_output_bytes_match(
    const void *ptr,
    size_t supplied_bytes,
    size_t required_bytes
) {
    return ptr && required_bytes != 0 && supplied_bytes == required_bytes;
}

int gemma4_listener_batch_probe_read_rms(
    gemma4_listener_batch_ctx *batch,
    void *input_host_u16,
    size_t input_bytes,
    void *batched_host_u16,
    size_t batched_bytes,
    void *split_host_u16,
    size_t split_bytes,
    gemma4_listener_batch_probe_info *out_info
) {
    if (!batch || batch->poisoned || !out_info ||
        out_info->struct_size != sizeof(gemma4_listener_batch_probe_info) ||
        batch->probe_state != GEMMA4_BATCH_PROBE_RMS_CAPTURED) return -1;
    const size_t required = gemma4_batch_probe_elements(
        GEMMA4_BATCH_PROBE_Q_ROWS, GEMMA4_BATCH_PROBE_HIDDEN_DIM) *
        sizeof(__nv_bfloat16);
    if (!gemma4_batch_probe_output_bytes_match(input_host_u16, input_bytes, required) ||
        !gemma4_batch_probe_output_bytes_match(batched_host_u16, batched_bytes, required) ||
        !gemma4_batch_probe_output_bytes_match(split_host_u16, split_bytes, required)) return -2;
    if (cudaEventSynchronize(batch->completion_event) != cudaSuccess ||
        cudaMemcpy(input_host_u16, batch->probe_rms_input, required, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(batched_host_u16, batch->probe_rms_batched, required, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(split_host_u16, batch->probe_rms_split, required, cudaMemcpyDeviceToHost) != cudaSuccess) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -3;
    }
    batch->probe_info.state = (uint32_t)batch->probe_state;
    *out_info = batch->probe_info;
    return 0;
}

int gemma4_listener_batch_probe_mark_rms_exact(
    gemma4_listener_batch_ctx *batch,
    uint32_t exact_match
) {
    if (!batch || batch->probe_state != GEMMA4_BATCH_PROBE_RMS_CAPTURED ||
        exact_match > 1u ||
        !gemma4_batch_probe_mark_rms_result(&batch->probe_state, exact_match != 0)) return -1;
    batch->probe_info.state = (uint32_t)batch->probe_state;
    return 0;
}

int gemma4_listener_batch_probe_qproj_async(
    gemma4_listener_batch_ctx *batch,
    cudaStream_t listener_stream
) {
    if (!batch || batch->poisoned || batch->output_lease_active ||
        !gemma4_batch_probe_can_run_qproj(batch->probe_state) ||
        !batch->probe_rms_batched || !batch->probe_q_actual ||
        !batch->probe_q_batched || !batch->probe_q_split) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    const GemmaLayerSpec *spec = &gemma4_layers[0];
    const gemma4_layer_weights *lw = &batch->model->weights.layers[0];
    const int qdim = (int)(GEMMA4_NUM_ATTN_HEADS * spec->head_dim);
    const int hidden = GEMMA4_HIDDEN_SIZE;
    if (qdim <= 0 || qdim > (int)(GEMMA4_NUM_ATTN_HEADS * 512u)) return -2;

    cublasMath_t math_mode;
    cublasAtomicsMode_t atomics_mode;
    cublasPointerMode_t pointer_mode;
    cudaStream_t handle_stream = nullptr;
    if (cublasGetMathMode(batch->model->cublas, &math_mode) != CUBLAS_STATUS_SUCCESS ||
        cublasGetAtomicsMode(batch->model->cublas, &atomics_mode) != CUBLAS_STATUS_SUCCESS ||
        cublasGetPointerMode(batch->model->cublas, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
        cublasGetStream(batch->model->cublas, &handle_stream) != CUBLAS_STATUS_SUCCESS) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -3;
    }
    const bool q4_enabled = q4_gemv4_opt_in_enabled();
    const size_t x_stride = (size_t)GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS * hidden;
    const size_t y_stride = (size_t)GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS * qdim;
    const bool q4_split0 = gemma4_q4_gemv4_dispatch_eligible(
        q4_enabled, true, true, GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, qdim, hidden,
        (uintptr_t)lw->q_proj_w, (uintptr_t)batch->probe_rms_batched,
        (uintptr_t)batch->probe_q_split);
    const bool q4_split1 = gemma4_q4_gemv4_dispatch_eligible(
        q4_enabled, true, true, GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, qdim, hidden,
        (uintptr_t)lw->q_proj_w,
        (uintptr_t)(batch->probe_rms_batched + x_stride),
        (uintptr_t)(batch->probe_q_split + y_stride));
    if (run_linear_cublas(batch->model->cublas, lw->q_proj_w,
            batch->probe_rms_batched, batch->probe_q_batched,
            GEMMA4_BATCH_PROBE_Q_ROWS, qdim, hidden, listener_stream) != 0 ||
        run_linear_cublas(batch->model->cublas, lw->q_proj_w,
            batch->probe_rms_batched, batch->probe_q_split,
            GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, qdim, hidden, listener_stream) != 0 ||
        run_linear_cublas(batch->model->cublas, lw->q_proj_w,
            batch->probe_rms_batched + x_stride,
            batch->probe_q_split + y_stride,
            GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, qdim, hidden, listener_stream) != 0 ||
        cudaEventRecord(batch->probe_event, listener_stream) != cudaSuccess) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -4;
    }
    if (!gemma4_batch_probe_mark_qproj_captured(&batch->probe_state)) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -5;
    }
    batch->probe_info.state = (uint32_t)batch->probe_state;
    batch->probe_info.q_dim = (uint32_t)qdim;
    batch->probe_info.q_batched_m = GEMMA4_BATCH_PROBE_Q_ROWS;
    batch->probe_info.q_split_m = GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS;
    batch->probe_info.q_k = GEMMA4_HIDDEN_SIZE;
    batch->probe_info.q_lda = GEMMA4_HIDDEN_SIZE;
    batch->probe_info.q_ldb = GEMMA4_HIDDEN_SIZE;
    batch->probe_info.q_ldc = (uint32_t)qdim;
    batch->probe_info.q4_gemv4_opt_in = q4_enabled ? 1u : 0u;
    batch->probe_info.q4_gemv4_split0_eligible = q4_split0 ? 1u : 0u;
    batch->probe_info.q4_gemv4_split1_eligible = q4_split1 ? 1u : 0u;
    batch->probe_info.cublas_math_mode = (uint32_t)math_mode;
    batch->probe_info.cublas_atomics_mode = (uint32_t)atomics_mode;
    batch->probe_info.cublas_pointer_mode = (uint32_t)pointer_mode;
    batch->probe_info.cublas_stream_matches_listener =
        handle_stream == listener_stream ? 1u : 0u;
    batch->probe_info.q_weight_device_ptr = (uint64_t)(uintptr_t)lw->q_proj_w;
    batch->probe_info.listener_stream_handle = (uint64_t)(uintptr_t)listener_stream;
    return 0;
}

int gemma4_listener_batch_probe_read_qproj(
    gemma4_listener_batch_ctx *batch,
    void *input_host_u16,
    size_t input_bytes,
    void *actual_host_u16,
    size_t actual_bytes,
    void *batched_host_u16,
    size_t batched_bytes,
    void *split_host_u16,
    size_t split_bytes,
    gemma4_listener_batch_probe_info *out_info
) {
    if (!batch || batch->poisoned || !out_info ||
        out_info->struct_size != sizeof(gemma4_listener_batch_probe_info) ||
        batch->probe_state != GEMMA4_BATCH_PROBE_QPROJ_CAPTURED ||
        batch->probe_info.q_dim == 0) return -1;
    const size_t input_required = gemma4_batch_probe_elements(
        GEMMA4_BATCH_PROBE_Q_ROWS, GEMMA4_BATCH_PROBE_HIDDEN_DIM) *
        sizeof(__nv_bfloat16);
    const size_t q_required = gemma4_batch_probe_elements(
        GEMMA4_BATCH_PROBE_Q_ROWS, batch->probe_info.q_dim) *
        sizeof(__nv_bfloat16);
    if (!gemma4_batch_probe_output_bytes_match(input_host_u16, input_bytes, input_required) ||
        !gemma4_batch_probe_output_bytes_match(actual_host_u16, actual_bytes, q_required) ||
        !gemma4_batch_probe_output_bytes_match(batched_host_u16, batched_bytes, q_required) ||
        !gemma4_batch_probe_output_bytes_match(split_host_u16, split_bytes, q_required)) return -2;
    if (cudaEventSynchronize(batch->probe_event) != cudaSuccess ||
        cudaMemcpy(input_host_u16, batch->probe_rms_batched, input_required, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(actual_host_u16, batch->probe_q_actual, q_required, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(batched_host_u16, batch->probe_q_batched, q_required, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(split_host_u16, batch->probe_q_split, q_required, cudaMemcpyDeviceToHost) != cudaSuccess) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -3;
    }
    if (!gemma4_batch_probe_mark_done(&batch->probe_state)) {
        batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
        return -4;
    }
    batch->probe_info.state = (uint32_t)batch->probe_state;
    *out_info = batch->probe_info;
    return 0;
}

static __nv_bfloat16 *gemma4_batch_trace_buffer(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    uint32_t buffer_index
) {
    if (!batch || !batch->trace_arena || stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT ||
        buffer_index >= 5) return nullptr;
    const size_t offset = batch->trace_records[stage_id].offsets[buffer_index];
    return reinterpret_cast<__nv_bfloat16 *>(
        reinterpret_cast<unsigned char *>(batch->trace_arena) + offset);
}

static bool gemma4_batch_trace_set_stage(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    uint32_t kind,
    uint32_t input0_rows,
    uint32_t input0_cols,
    uint32_t input1_rows,
    uint32_t input1_cols,
    uint32_t input2_rows,
    uint32_t input2_cols,
    uint32_t actual_rows,
    uint32_t actual_cols,
    uint32_t capacity_input1_rows,
    uint32_t capacity_input2_rows,
    gemma4_batch_trace_ref_op reference_op,
    const void *weight,
    double epsilon,
    float scalar,
    bool inplace,
    size_t *arena_bytes
) {
    if (!batch || !arena_bytes || stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT ||
        !input0_rows || !input0_cols || !actual_rows || !actual_cols ||
        (kind < GEMMA4_BATCH_TRACE_KIND_GEMM || kind > GEMMA4_BATCH_TRACE_KIND_ATTENTION))
        return false;
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    memset(&record, 0, sizeof(record));
    gemma4_listener_batch_trace_info &info = record.info;
    info.struct_size = sizeof(info);
    info.stage_id = stage_id;
    info.kind = kind;
    info.flags = inplace ? GEMMA4_BATCH_TRACE_FLAG_INPLACE : 0u;
    info.layer_index = stage_id < GEMMA4_BATCH_TRACE_L0_INPUT_NORM ? UINT32_MAX : 0u;
    if (stage_id == GEMMA4_BATCH_TRACE_FINAL_LAYER_OUTPUT)
        info.layer_index = GEMMA4_NUM_LAYERS - 1u;
    info.q_len = GEMMA4_STREAM_Q_LEN;
    info.row_count = 2u;
    info.input0_rows = input0_rows;
    info.input0_cols = input0_cols;
    info.input1_rows = input1_rows;
    info.input1_cols = input1_cols;
    info.input2_rows = input2_rows;
    info.input2_cols = input2_cols;
    info.actual_rows = actual_rows;
    info.actual_cols = actual_cols;
    info.reference_rows = reference_op == GEMMA4_BATCH_TRACE_REF_NONE ? 0u : actual_rows;
    info.reference_cols = reference_op == GEMMA4_BATCH_TRACE_REF_NONE ? 0u : actual_cols;
    info.reference_rows_per_sid = reference_op == GEMMA4_BATCH_TRACE_REF_NONE
        ? 0u : actual_rows / 2u;
    info.rms_backend_id = batch->model->rmsnorm_backend;
    info.attention_backend_id = batch->model->attention_backend;
    info.input0_elements = gemma4_batch_probe_elements(input0_rows, input0_cols);
    info.input1_elements = input1_rows && input1_cols
        ? gemma4_batch_probe_elements(input1_rows, input1_cols) : 0u;
    info.input2_elements = input2_rows && input2_cols
        ? gemma4_batch_probe_elements(input2_rows, input2_cols) : 0u;
    info.actual_elements = gemma4_batch_probe_elements(actual_rows, actual_cols);
    info.reference_elements = reference_op == GEMMA4_BATCH_TRACE_REF_NONE
        ? 0u : gemma4_batch_probe_elements(actual_rows, actual_cols);
    if (!info.input0_elements || !info.actual_elements ||
        (reference_op != GEMMA4_BATCH_TRACE_REF_NONE && !info.reference_elements) ||
        (input1_rows != 0 && !info.input1_elements) ||
        (input2_rows != 0 && !info.input2_elements)) return false;
    if (reference_op == GEMMA4_BATCH_TRACE_REF_NONE) {
        info.flags |= GEMMA4_BATCH_TRACE_FLAG_REFERENCE_DEFERRED;
    }
    record.weight = weight;
    record.epsilon = epsilon;
    record.scalar = scalar;
    record.reference_op = reference_op;
    info.weight_device_ptr = (uint64_t)(uintptr_t)weight;
    info.epsilon = (float)epsilon;
    info.scalar = scalar;

    const uint64_t element_counts[5] = {
        info.input0_elements,
        (uint64_t)capacity_input1_rows * input1_cols,
        (uint64_t)capacity_input2_rows * input2_cols,
        info.actual_elements,
        info.reference_elements,
    };
    for (uint32_t index = 0; index < 5; ++index) {
        if (element_counts[index] > (uint64_t)SIZE_MAX / sizeof(__nv_bfloat16)) return false;
        const size_t bytes = (size_t)element_counts[index] * sizeof(__nv_bfloat16);
        record.offsets[index] = *arena_bytes;
        record.capacity_bytes[index] = bytes;
        if (bytes > (size_t)GEMMA4_BATCH_TRACE_MAX_BYTES - *arena_bytes) return false;
        *arena_bytes += bytes;
    }
    if (kind == GEMMA4_BATCH_TRACE_KIND_GEMM) {
        info.gemm_m = actual_rows;
        info.gemm_n = actual_cols;
        info.gemm_k = input0_cols;
        info.gemm_lda = input0_cols;
        info.gemm_ldb = input0_cols;
        info.gemm_ldc = actual_cols;
        info.gemm_actual_wrapper = GEMMA4_BATCH_TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM;
        if (reference_op == GEMMA4_BATCH_TRACE_REF_GEMM) {
            /* The M=4 references below use the exact legacy GEMV/cuBLAS
             * dispatcher used by the ABI3 listener. */
            info.gemm_reference_wrapper = GEMMA4_BATCH_TRACE_GEMM_WRAPPER_LEGACY_CUBLAS;
            info.q4_gemv4_opt_in = q4_gemv4_opt_in_enabled() ? 1u : 0u;
        }
    }
    if (kind == GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE) {
        info.launch_block_x = 256u;
        info.launch_block_y = 1u;
        info.launch_block_z = 1u;
        info.launch_grid_x = (actual_rows * actual_cols + 255u) / 256u;
    }
    return true;
}

static int gemma4_batch_trace_copy_input(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    uint32_t buffer_index,
    const void *source,
    uint64_t elements,
    cudaStream_t stream
) {
    if (!elements) return source ? -1 : 0;
    if (!batch || !source || stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    const size_t bytes = (size_t)elements * sizeof(__nv_bfloat16);
    if (buffer_index >= 3 || bytes > record.capacity_bytes[buffer_index]) return -2;
    return cudaMemcpyAsync(gemma4_batch_trace_buffer(batch, stage_id, buffer_index),
                           source, bytes, cudaMemcpyDeviceToDevice, stream) == cudaSuccess
        ? 0 : -3;
}

static int gemma4_batch_trace_begin(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    const void *input0,
    const void *input1,
    const void *input2,
    cudaStream_t stream
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED ||
        stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    if (record.inputs_captured ||
        gemma4_batch_trace_copy_input(batch, stage_id, 0, input0,
            record.info.input0_elements, stream) != 0 ||
        gemma4_batch_trace_copy_input(batch, stage_id, 1, input1,
            record.info.input1_elements, stream) != 0 ||
        gemma4_batch_trace_copy_input(batch, stage_id, 2, input2,
            record.info.input2_elements, stream) != 0) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -2;
    }
    record.inputs_captured = true;
    record.info.flags |= GEMMA4_BATCH_TRACE_FLAG_INPUTS_CAPTURED;
    return 0;
}

static int gemma4_batch_trace_reference(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    cudaStream_t stream
) {
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    if (record.reference_op == GEMMA4_BATCH_TRACE_REF_NONE) return 0;
    const uint32_t rows_per_sid = record.info.reference_rows_per_sid;
    const uint32_t cols = record.info.actual_cols;
    const __nv_bfloat16 *input0 = gemma4_batch_trace_buffer(batch, stage_id, 0);
    const __nv_bfloat16 *input1 = gemma4_batch_trace_buffer(batch, stage_id, 1);
    __nv_bfloat16 *reference = gemma4_batch_trace_buffer(batch, stage_id, 4);
    if (!rows_per_sid || !cols || !input0 || !reference) return -1;

    if (record.reference_op == GEMMA4_BATCH_TRACE_REF_GEMM) {
        const uint32_t m = record.info.gemm_m;
        const uint32_t n = record.info.gemm_n;
        const uint32_t k = record.info.gemm_k;
        if (m != 2u * rows_per_sid || !record.weight || !n || !k) return -2;
        const bool q4_enabled = q4_gemv4_opt_in_enabled();
        if (record.info.gemm_actual_wrapper != GEMMA4_BATCH_TRACE_GEMM_WRAPPER_BATCH_ROW_GEMM ||
            record.info.gemm_reference_wrapper != GEMMA4_BATCH_TRACE_GEMM_WRAPPER_LEGACY_CUBLAS ||
            record.info.q4_gemv4_opt_in != (q4_enabled ? 1u : 0u)) return -3;
        for (uint32_t sid = 0; sid < 2; ++sid) {
            const __nv_bfloat16 *split_input = input0 + (size_t)sid * rows_per_sid * k;
            __nv_bfloat16 *split_output = reference + (size_t)sid * rows_per_sid * n;
            const bool q4_eligible = gemma4_q4_gemv4_dispatch_eligible(
                q4_enabled, true, true, (int)rows_per_sid, (int)n, (int)k,
                (uintptr_t)record.weight, (uintptr_t)split_input, (uintptr_t)split_output);
            if (sid == 0) {
                record.info.q4_gemv4_split0_eligible = q4_eligible ? 1u : 0u;
                record.info.q4_gemv4_split0_selected = q4_eligible ? 1u : 0u;
            } else {
                record.info.q4_gemv4_split1_eligible = q4_eligible ? 1u : 0u;
                record.info.q4_gemv4_split1_selected = q4_eligible ? 1u : 0u;
            }
            if (run_linear_cublas(batch->model->cublas, record.weight,
                    split_input, split_output,
                    (int)rows_per_sid, (int)n, (int)k, stream) != 0) return -4;
        }
    } else if (record.reference_op == GEMMA4_BATCH_TRACE_REF_RMS) {
        if (!record.weight) return -4;
        for (uint32_t sid = 0; sid < 2; ++sid) {
            if (run_rmsnorm_exact_or_custom(batch->model,
                    input0 + (size_t)sid * rows_per_sid * cols,
                    record.weight,
                    reference + (size_t)sid * rows_per_sid * cols,
                    rows_per_sid, cols, record.epsilon, stream) != 0) return -5;
        }
    } else {
        for (uint32_t sid = 0; sid < 2; ++sid) {
            const size_t input_offset = (size_t)sid * rows_per_sid * record.info.input0_cols;
            const size_t output_offset = (size_t)sid * rows_per_sid * cols;
            const int element_count = (int)((size_t)rows_per_sid * cols);
            __nv_bfloat16 *out = reference + output_offset;
            switch (record.reference_op) {
                case GEMMA4_BATCH_TRACE_REF_SCALE:
                    if (rmsnorm_uses_exact_path(batch->model)) {
                        kernel_scale_bf16_fp32_scalar<<<(element_count + 255) / 256, 256, 0, stream>>>(
                            input0 + input_offset, out, element_count, record.scalar);
                    } else {
                        kernel_scale_bf16<<<(element_count + 255) / 256, 256, 0, stream>>>(
                            input0 + input_offset, out, element_count, record.scalar);
                    }
                    break;
                case GEMMA4_BATCH_TRACE_REF_RESIDUAL:
                    if (!input1) return -6;
                    kernel_residual_add_bf16<<<(element_count + 255) / 256, 256, 0, stream>>>(
                        input0 + input_offset, input1 + output_offset, out, element_count);
                    break;
                case GEMMA4_BATCH_TRACE_REF_GELU:
                    kernel_gelu_tanh_bf16<<<(element_count + 255) / 256, 256, 0, stream>>>(
                        input0 + input_offset, out, element_count);
                    break;
                case GEMMA4_BATCH_TRACE_REF_MUL:
                    if (!input1) return -7;
                    kernel_mul_bf16<<<(element_count + 255) / 256, 256, 0, stream>>>(
                        input0 + input_offset, input1 + output_offset, out, element_count);
                    break;
                case GEMMA4_BATCH_TRACE_REF_FRAME_PAD: {
                    const uint32_t width = record.info.actual_cols;
                    if (!gemma4_batch_trace_frame_pad_input_valid(
                            record.info.input0_rows, record.info.input0_cols,
                            record.info.input0_elements, rows_per_sid, width)) return -10;
                    const int blocks = (int)(((size_t)rows_per_sid * width + 255u) / 256u);
                    kernel_repeat_frame_pad_ple_bf16<<<blocks, 256, 0, stream>>>(
                        input0, out, rows_per_sid, (int)width);
                    break;
                }
                case GEMMA4_BATCH_TRACE_REF_PLE_LAYER:
                    if (!input1) return -8;
                    kernel_ple_layer_mult_bf16<<<(element_count + 255) / 256, 256, 0, stream>>>(
                        input0 + input_offset,
                        input1 + (size_t)sid * rows_per_sid * record.info.input1_cols,
                        out, (int)rows_per_sid, 0);
                    break;
                default:
                    return -9;
            }
            if (cudaPeekAtLastError() != cudaSuccess) return -10;
        }
    }
    record.reference_captured = true;
    record.info.flags |= GEMMA4_BATCH_TRACE_FLAG_REFERENCE_CAPTURED;
    return 0;
}

static int gemma4_batch_trace_finish(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    const void *actual,
    cudaStream_t stream
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED ||
        stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT || !actual) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    if (!record.inputs_captured || record.actual_captured) return -2;
    const size_t bytes = (size_t)record.info.actual_elements * sizeof(__nv_bfloat16);
    if (bytes > record.capacity_bytes[3] ||
        cudaMemcpyAsync(gemma4_batch_trace_buffer(batch, stage_id, 3), actual, bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -3;
    }
    record.actual_captured = true;
    record.info.flags |= GEMMA4_BATCH_TRACE_FLAG_ACTUAL_CAPTURED;
    if (record.info.kind == GEMMA4_BATCH_TRACE_KIND_GEMM) {
        cublasMath_t math_mode;
        cublasAtomicsMode_t atomics_mode;
        cublasPointerMode_t pointer_mode;
        cudaStream_t handle_stream = nullptr;
        if (cublasGetMathMode(batch->model->cublas, &math_mode) != CUBLAS_STATUS_SUCCESS ||
            cublasGetAtomicsMode(batch->model->cublas, &atomics_mode) != CUBLAS_STATUS_SUCCESS ||
            cublasGetPointerMode(batch->model->cublas, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
            cublasGetStream(batch->model->cublas, &handle_stream) != CUBLAS_STATUS_SUCCESS) {
            batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
            return -4;
        }
        record.info.cublas_math_mode = (uint32_t)math_mode;
        record.info.cublas_atomics_mode = (uint32_t)atomics_mode;
        record.info.cublas_pointer_mode = (uint32_t)pointer_mode;
        record.info.cublas_stream_matches_listener = handle_stream == stream ? 1u : 0u;
    }
    if (record.reference_op != GEMMA4_BATCH_TRACE_REF_NONE &&
        gemma4_batch_trace_reference(batch, stage_id, stream) != 0) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -5;
    }
    return 0;
}

static int gemma4_batch_trace_record_attention_input(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    int layer_idx,
    const __nv_bfloat16 *query,
    const __nv_bfloat16 *key,
    const __nv_bfloat16 *value,
    uint32_t start,
    uint32_t visible,
    uint32_t base_position,
    uint32_t head_dim,
    cudaStream_t stream
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED || layer_idx != 0) return 0;
    gemma4_batch_trace_record &record = batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION];
    uint32_t row = 2u;
    for (uint32_t i = 0; i < 2; ++i) {
        if (batch->trace_active_slots[i] == slot) row = i;
    }
    if (row >= 2 || record.captured_sessions >= 2 || !query || !key || !value || !visible ||
        (record.attention_input_mask & (1u << row)) != 0u ||
        visible != record.info.visible_len[row] || start != record.info.cache_start[row] ||
        base_position != record.info.base_position[row] || head_dim != record.info.input1_cols) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -1;
    }
    size_t kv_row_offset = 0;
    for (uint32_t i = 0; i < row; ++i) kv_row_offset += record.info.visible_len[i];
    const size_t q_elements = (size_t)GEMMA4_STREAM_Q_LEN * record.info.input0_cols;
    const size_t kv_elements = (size_t)visible * head_dim;
    const size_t q_offset = (size_t)row * q_elements;
    __nv_bfloat16 *input0 = gemma4_batch_trace_buffer(batch, GEMMA4_BATCH_TRACE_L0_ATTENTION, 0);
    __nv_bfloat16 *input1 = gemma4_batch_trace_buffer(batch, GEMMA4_BATCH_TRACE_L0_ATTENTION, 1);
    __nv_bfloat16 *input2 = gemma4_batch_trace_buffer(batch, GEMMA4_BATCH_TRACE_L0_ATTENTION, 2);
    const size_t bytes = kv_elements * sizeof(__nv_bfloat16);
    if (q_offset + q_elements > record.capacity_bytes[0] / sizeof(__nv_bfloat16) ||
        (kv_row_offset + visible) * head_dim > record.capacity_bytes[1] / sizeof(__nv_bfloat16) ||
        cudaMemcpyAsync(input0 + q_offset, query, q_elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(input1 + kv_row_offset * head_dim,
                        key + (size_t)start * head_dim, bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(input2 + kv_row_offset * head_dim,
                        value + (size_t)start * head_dim, bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -2;
    }
    ++record.captured_sessions;
    if (!gemma4_batch_trace_mark_row(&record.attention_input_mask, row)) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -3;
    }
    if (gemma4_batch_trace_rows_complete(record.attention_input_mask)) {
        record.inputs_captured = true;
        record.info.flags |= GEMMA4_BATCH_TRACE_FLAG_INPUTS_CAPTURED;
    }
    return 0;
}

static int gemma4_batch_trace_record_attention_output(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const __nv_bfloat16 *actual,
    cudaStream_t stream
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED || !actual) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION];
    uint32_t row = 2u;
    for (uint32_t i = 0; i < 2; ++i) {
        if (batch->trace_active_slots[i] == slot) row = i;
    }
    if (row >= 2 || (record.attention_input_mask & (1u << row)) == 0u ||
        (record.attention_output_mask & (1u << row)) != 0u)
        return -2;
    const size_t session_elements = (size_t)GEMMA4_STREAM_Q_LEN * record.info.actual_cols;
    const size_t offset = (size_t)row * session_elements;
    __nv_bfloat16 *out = gemma4_batch_trace_buffer(batch, GEMMA4_BATCH_TRACE_L0_ATTENTION, 3);
    if (offset + session_elements > record.capacity_bytes[3] / sizeof(__nv_bfloat16) ||
        cudaMemcpyAsync(out + offset, actual, session_elements * sizeof(__nv_bfloat16),
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -3;
    }
    /* Count output rows separately from input-session capture. */
    if (!gemma4_batch_trace_mark_row(&record.attention_output_mask, row)) {
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -4;
    }
    record.info.flags |= row == 0u
        ? GEMMA4_BATCH_TRACE_FLAG_ATTENTION_OUTPUT_ROW0
        : GEMMA4_BATCH_TRACE_FLAG_ATTENTION_OUTPUT_ROW1;
    if (gemma4_batch_trace_rows_complete(record.attention_output_mask)) {
        record.actual_captured = true;
        record.info.flags |= GEMMA4_BATCH_TRACE_FLAG_ACTUAL_CAPTURED;
    }
    return 0;
}

static uint32_t gemma4_batch_trace_visible_keys(
    const gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const GemmaLayerSpec *spec
) {
    if (!batch || slot >= batch->capacity || !spec || !batch->slots[slot].primed)
        return 0;
    const uint32_t base = batch->slots[slot].current_position;
    const uint32_t total = base + GEMMA4_STREAM_Q_LEN;
    const uint32_t start = spec->sliding_window > 0 &&
        base >= (uint32_t)spec->sliding_window
        ? base - (uint32_t)spec->sliding_window + 1u : 0u;
    return total > start ? total - start : 0u;
}

static void gemma4_batch_trace_clear(gemma4_listener_batch_ctx *batch) {
    if (!batch) return;
    if (batch->trace_event) cudaEventDestroy(batch->trace_event);
    if (batch->trace_arena) cudaFree(batch->trace_arena);
    batch->trace_event = nullptr;
    batch->trace_arena = nullptr;
    batch->trace_arena_bytes = 0;
    batch->trace_state = GEMMA4_BATCH_TRACE_IDLE;
    memset(batch->trace_records, 0, sizeof(batch->trace_records));
    memset(batch->trace_active_slots, 0, sizeof(batch->trace_active_slots));
}

static bool gemma4_batch_trace_plan(
    gemma4_listener_batch_ctx *batch,
    size_t *arena_bytes
) {
    if (!batch || !arena_bytes || batch->capacity < 2) return false;
    const GemmaLayerSpec *spec = &gemma4_layers[0];
    if (spec->is_kv_shared || !spec->head_dim || !spec->intermediate_size ||
        spec->intermediate_size > GEMMA4_MAX_INTERMEDIATE_SIZE) return false;
    const uint32_t hidden = GEMMA4_HIDDEN_SIZE;
    const uint32_t qdim = GEMMA4_NUM_ATTN_HEADS * spec->head_dim;
    const uint32_t head_dim = spec->head_dim;
    const uint32_t intermediate = spec->intermediate_size;
    const uint32_t ple_width = GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
    const uint32_t q_rows = 2u * GEMMA4_STREAM_Q_LEN;
    uint32_t largest_visible[2] = {0, 0};
    uint32_t primed_count = 0;
    for (uint32_t slot = 0; slot < batch->capacity; ++slot) {
        const uint32_t visible = gemma4_batch_trace_visible_keys(batch, slot, spec);
        if (!visible) continue;
        ++primed_count;
        if (visible > largest_visible[0]) {
            largest_visible[1] = largest_visible[0];
            largest_visible[0] = visible;
        } else if (visible > largest_visible[1]) {
            largest_visible[1] = visible;
        }
    }
    if (primed_count < 2 || !largest_visible[1]) return false;
    const uint32_t attention_capacity_rows = largest_visible[0] + largest_visible[1];
    size_t bytes = 0;
/* This trace is an actual-only boundary capture. Split-row references used
 * to abort on a supported layer-0 value-norm path and obscured the actual
 * first divergence. Reference arrays stay empty and are explicitly deferred. */
#define GEMMA4_TRACE_SET(id, kind, i0r, i0c, i1r, i1c, i2r, i2c, ar, ac, cap1, cap2, refop, weight, eps, scalar, inplace) \
    do { \
        if (!gemma4_batch_trace_set_stage(batch, id, kind, i0r, i0c, i1r, i1c, \
                i2r, i2c, ar, ac, cap1, cap2, GEMMA4_BATCH_TRACE_REF_NONE, nullptr, \
                eps, scalar, inplace, &bytes)) \
            return false; \
    } while (0)

    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_GEMM, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, ple_width, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, batch->model->weights.per_layer_model_proj_w,
        0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_SCALE, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, ple_width, 0, 0, 0, 0, q_rows, ple_width, 0, 0,
        GEMMA4_BATCH_TRACE_REF_SCALE, nullptr, 0.0,
        batch->model->weights.per_layer_model_proj_scale, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows * GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM, 0, 0, 0, 0,
        q_rows * GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, batch->model->weights.per_layer_proj_norm_w,
        1e-6, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_FRAME_PAD, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        GEMMA4_STREAM_Q_LEN, ple_width, 0, 0, 0, 0, q_rows, ple_width,
        0, 0, GEMMA4_BATCH_TRACE_REF_FRAME_PAD, nullptr, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_PAD_RESIDUAL, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, ple_width, q_rows, ple_width, 0, 0, q_rows, ple_width,
        q_rows, 0, GEMMA4_BATCH_TRACE_REF_RESIDUAL, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_AUDIO_PLE_INPUT_SCALE, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, ple_width, 0, 0, 0, 0, q_rows, ple_width, 0, 0,
        GEMMA4_BATCH_TRACE_REF_SCALE, nullptr, 0.0,
        batch->model->weights.per_layer_input_scale, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_INPUT_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->input_norm_w, spec->input_norm_eps, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_Q_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, qdim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->q_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_Q_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows * GEMMA4_NUM_ATTN_HEADS, head_dim, 0, 0, 0, 0,
        q_rows * GEMMA4_NUM_ATTN_HEADS, head_dim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->q_norm_w, spec->q_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_K_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, head_dim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->k_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_V_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, head_dim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->v_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_K_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, head_dim, 0, 0, 0, 0, q_rows, head_dim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->k_norm_w, spec->k_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_V_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, head_dim, 0, 0, 0, 0, q_rows, head_dim, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->v_norm_w, spec->v_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_ATTENTION, GEMMA4_BATCH_TRACE_KIND_ATTENTION,
        q_rows, qdim, attention_capacity_rows, head_dim,
        attention_capacity_rows, head_dim, q_rows, qdim,
        attention_capacity_rows, attention_capacity_rows,
        GEMMA4_BATCH_TRACE_REF_NONE, nullptr, 0.0, 0.0f, false);
    batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION].info.attention_is_sliding =
        spec->attn_type == GEMMA_ATTN_SLIDING ? 1u : 0u;
    batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION].info.sliding_window =
        spec->sliding_window > 0 ? (uint32_t)spec->sliding_window : 0u;
    batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION].info.attention_mask_causal = 1u;
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_O_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, qdim, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->o_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_POST_ATTN_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->post_attn_norm_w,
        spec->post_attn_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_POST_ATTN_RESIDUAL, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, hidden, q_rows, hidden, 0, 0, q_rows, hidden, q_rows, 0,
        GEMMA4_BATCH_TRACE_REF_RESIDUAL, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PRE_FF_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->pre_ff_norm_w, spec->pre_ff_norm_eps, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_GATE_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, intermediate, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->gate_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_UP_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, intermediate, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->up_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_GATE_GELU, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, intermediate, 0, 0, 0, 0, q_rows, intermediate, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GELU, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_GATE_UP_MUL, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, intermediate, q_rows, intermediate, 0, 0, q_rows, intermediate, q_rows, 0,
        GEMMA4_BATCH_TRACE_REF_MUL, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_DOWN_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, intermediate, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->down_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_POST_FF_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->post_ff_norm_w, spec->post_ff_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_POST_FF_RESIDUAL, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, hidden, q_rows, hidden, 0, 0, q_rows, hidden, q_rows, 0,
        GEMMA4_BATCH_TRACE_REF_RESIDUAL, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_GATE_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, hidden, 0, 0, 0, 0, q_rows, GEMMA4_PLE_DIM, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->ple_gate_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_GELU, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, GEMMA4_PLE_DIM, 0, 0, 0, 0, q_rows, GEMMA4_PLE_DIM, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GELU, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_LAYER_MULT, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, GEMMA4_PLE_DIM, q_rows, ple_width, 0, 0,
        q_rows, GEMMA4_PLE_DIM, q_rows, 0,
        GEMMA4_BATCH_TRACE_REF_PLE_LAYER, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_PROJ, GEMMA4_BATCH_TRACE_KIND_GEMM,
        q_rows, GEMMA4_PLE_DIM, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_GEMM, lw->ple_proj_w, 0.0, 0.0f, false);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_NORM, GEMMA4_BATCH_TRACE_KIND_RMS,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_RMS, lw->ple_norm_w, spec->post_ple_norm_eps, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_PLE_RESIDUAL, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, hidden, q_rows, hidden, 0, 0, q_rows, hidden, q_rows, 0,
        GEMMA4_BATCH_TRACE_REF_RESIDUAL, nullptr, 0.0, 0.0f, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_L0_LAYER_SCALE, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_SCALE, nullptr, 0.0, spec->layer_scalar, true);
    GEMMA4_TRACE_SET(GEMMA4_BATCH_TRACE_FINAL_LAYER_OUTPUT, GEMMA4_BATCH_TRACE_KIND_ELEMENTWISE,
        q_rows, hidden, 0, 0, 0, 0, q_rows, hidden, 0, 0,
        GEMMA4_BATCH_TRACE_REF_NONE, nullptr, 0.0, 0.0f, false);
#undef GEMMA4_TRACE_SET
    *arena_bytes = bytes;
    return bytes > 0 && bytes <= (size_t)GEMMA4_BATCH_TRACE_MAX_BYTES;
}

static bool gemma4_batch_trace_capture_complete(
    const gemma4_listener_batch_ctx *batch
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED) return false;
    for (uint32_t stage_id = 0; stage_id < GEMMA4_BATCH_TRACE_STAGE_COUNT; ++stage_id) {
        const gemma4_batch_trace_record &record = batch->trace_records[stage_id];
        if (!record.inputs_captured || !record.actual_captured) return false;
        if (record.reference_op == GEMMA4_BATCH_TRACE_REF_NONE) {
            if ((record.info.flags & GEMMA4_BATCH_TRACE_FLAG_REFERENCE_DEFERRED) == 0u) return false;
        } else if (!record.reference_captured) {
            return false;
        }
    }
    return true;
}

uint32_t gemma4_listener_batch_trace_api_version(void) {
    return GEMMA4_LISTENER_BATCH_TRACE_API_VERSION;
}

uint32_t gemma4_listener_batch_trace_stage_count(void) {
    return GEMMA4_BATCH_TRACE_STAGE_COUNT;
}

int gemma4_listener_batch_trace_arm(gemma4_listener_batch_ctx *batch) {
    if (!batch || batch->poisoned || batch->output_lease_active ||
        !batch->stream_bound || !batch->priority_verified ||
        batch->trace_state != GEMMA4_BATCH_TRACE_IDLE || batch->capacity < 2) return -1;
    size_t bytes = 0;
    if (!gemma4_batch_trace_plan(batch, &bytes) ||
        !gemma4_batch_trace_storage_allowed(bytes)) {
        gemma4_batch_trace_clear(batch);
        return -2;
    }
    if (cudaEventCreateWithFlags(&batch->trace_event, cudaEventDisableTiming) != cudaSuccess) {
        gemma4_batch_trace_clear(batch);
        return -3;
    }
    if (cudaMalloc((void **)&batch->trace_arena, bytes) != cudaSuccess) {
        gemma4_batch_trace_clear(batch);
        return -4;
    }
    batch->trace_arena_bytes = bytes;
    batch->trace_state = GEMMA4_BATCH_TRACE_ARMED;
    return 0;
}

static int gemma4_batch_trace_update_attention_layout(
    gemma4_listener_batch_ctx *batch,
    uint32_t row_count
) {
    if (!batch || batch->trace_state != GEMMA4_BATCH_TRACE_ARMED || row_count != 2) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[GEMMA4_BATCH_TRACE_L0_ATTENTION];
    const GemmaLayerSpec *spec = &gemma4_layers[0];
    const uint32_t d = spec->head_dim;
    uint64_t rows = 0;
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t slot = batch->active_slots[i];
        if (slot >= batch->capacity || !batch->slots[slot].primed) return -2;
        const uint32_t base = batch->slots[slot].current_position;
        const uint32_t start = spec->sliding_window > 0 &&
            base >= (uint32_t)spec->sliding_window
            ? base - (uint32_t)spec->sliding_window + 1u : 0u;
        const uint32_t visible = base + GEMMA4_STREAM_Q_LEN - start;
        batch->trace_active_slots[i] = slot;
        record.info.cache_start[i] = start;
        record.info.visible_len[i] = visible;
        record.info.prefix_len[i] = batch->slots[slot].prefix_len;
        record.info.base_position[i] = base;
        rows += visible;
    }
    const uint64_t elements = rows * d;
    if (elements * sizeof(__nv_bfloat16) > record.capacity_bytes[1] ||
        elements * sizeof(__nv_bfloat16) > record.capacity_bytes[2] ||
        rows > UINT32_MAX) return -3;
    record.info.input1_rows = (uint32_t)rows;
    record.info.input1_cols = d;
    record.info.input2_rows = (uint32_t)rows;
    record.info.input2_cols = d;
    record.info.input1_elements = elements;
    record.info.input2_elements = elements;
    record.captured_sessions = 0;
    record.attention_input_mask = 0;
    record.attention_output_mask = 0;
    record.inputs_captured = false;
    record.actual_captured = false;
    record.info.flags &= ~(GEMMA4_BATCH_TRACE_FLAG_INPUTS_CAPTURED |
                           GEMMA4_BATCH_TRACE_FLAG_ACTUAL_CAPTURED);
    return 0;
}

int gemma4_listener_batch_trace_get_stage_info(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    gemma4_listener_batch_trace_info *out_info
) {
    if (!batch || !out_info || stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT ||
        out_info->struct_size != sizeof(*out_info) ||
        batch->trace_state != GEMMA4_BATCH_TRACE_CAPTURED) return -1;
    *out_info = batch->trace_records[stage_id].info;
    return 0;
}

static bool gemma4_batch_trace_host_buffer_matches(const void *ptr, size_t bytes, size_t required) {
    return bytes == required && ((required == 0 && ptr == nullptr) ||
                                 (required != 0 && ptr != nullptr));
}

int gemma4_listener_batch_trace_read_stage(
    gemma4_listener_batch_ctx *batch,
    uint32_t stage_id,
    void *input0_host_u16, size_t input0_bytes,
    void *input1_host_u16, size_t input1_bytes,
    void *input2_host_u16, size_t input2_bytes,
    void *actual_host_u16, size_t actual_bytes,
    void *reference_host_u16, size_t reference_bytes
) {
    if (!batch || batch->poisoned || stage_id >= GEMMA4_BATCH_TRACE_STAGE_COUNT ||
        batch->trace_state != GEMMA4_BATCH_TRACE_CAPTURED) return -1;
    gemma4_batch_trace_record &record = batch->trace_records[stage_id];
    size_t required[5];
    const uint64_t elements[5] = {
        record.info.input0_elements, record.info.input1_elements,
        record.info.input2_elements, record.info.actual_elements,
        record.info.reference_elements,
    };
    const void *host_ptrs[5] = {
        input0_host_u16, input1_host_u16, input2_host_u16,
        actual_host_u16, reference_host_u16,
    };
    const size_t sizes[5] = {
        input0_bytes, input1_bytes, input2_bytes, actual_bytes, reference_bytes,
    };
    for (uint32_t i = 0; i < 5; ++i) {
        if (elements[i] > (uint64_t)SIZE_MAX / sizeof(__nv_bfloat16)) return -2;
        required[i] = (size_t)elements[i] * sizeof(__nv_bfloat16);
        if (!gemma4_batch_trace_host_buffer_matches(host_ptrs[i], sizes[i], required[i]))
            return -3;
    }
    if (cudaEventSynchronize(batch->trace_event) != cudaSuccess) return -4;
    for (uint32_t i = 0; i < 5; ++i) {
        if (required[i] && cudaMemcpy(
                const_cast<void *>(host_ptrs[i]),
                gemma4_batch_trace_buffer(batch, stage_id, i),
                required[i], cudaMemcpyDeviceToHost) != cudaSuccess) return -5;
    }
    return 0;
}

int gemma4_listener_batch_bind_stream(
    gemma4_listener_batch_ctx *batch,
    cudaStream_t listener_stream
) {
    if (!batch || batch->poisoned) return -1;
    return gemma4_batch_bind_stream(batch, listener_stream);
}

int gemma4_listener_batch_prepare_gemm_graphs(
    gemma4_listener_batch_ctx *batch,
    uint64_t *out_graph_count
) {
    if (!batch || !out_graph_count || batch->poisoned || !batch->stream_bound ||
        batch->output_lease_active || batch->probe_state == GEMMA4_BATCH_PROBE_ARMED ||
        batch->trace_state == GEMMA4_BATCH_TRACE_ARMED) return -1;
    *out_graph_count = 0;
    for (uint32_t i = 0; i < batch->capacity; ++i) {
        if (batch->slots[i].primed) return -2;
    }
    if (batch->gemm_graphs_enabled) {
        *out_graph_count = batch->gemm_graph_count;
        return 0;
    }
    if (cudaStreamSynchronize(batch->listener_stream) != cudaSuccess) return -3;
    const int status = gemma4_batch_prepare_gemm_graphs_internal(batch);
    if (status != 0) return status;
    *out_graph_count = batch->gemm_graph_count;
    return 0;
}

uint64_t gemma4_listener_batch_gemm_graph_count(
    const gemma4_listener_batch_ctx *batch
) {
    return batch && batch->gemm_graphs_enabled ? batch->gemm_graph_count : 0;
}

int gemma4_listener_batch_import_cache_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    const gemma4_imported_cache *cache,
    const gemma4_tensor_meta *prefix_attention_meta,
    uint32_t prefix_len,
    cudaStream_t listener_stream
) {
    if (!batch || !row || !cache || slot >= batch->capacity || batch->poisoned ||
        batch->slots[slot].primed || row->native_context_id == 0 || !prefix_attention_meta ||
        prefix_len == 0 ||
        prefix_len > GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN ||
        row->cache_position < prefix_len ||
        row->valid_history_tokens != row->cache_position ||
        row->frame_id == UINT64_MAX ||
        row->listener_state_version == UINT64_MAX ||
        row->sequence_position == UINT64_MAX ||
        row->cache_position > batch->model->config.max_seq_len) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess ||
        validate_imported_cache_metadata(cache, row->cache_position, device) != 0) return -10;
    const uint64_t prefix_dims[] = {1, prefix_len, 0, 0};
    if (!validate_tensor_meta(prefix_attention_meta, prefix_attention_meta->ptr, device,
                              GEMMA4_META_INT64, 2, prefix_dims)) return -10;
    for (uint32_t i = 0; i < batch->capacity; ++i) {
        if (i != slot && batch->slots[i].primed &&
            batch->slots[i].context_id == row->native_context_id) return -1;
    }
    if (batch->model->last_stream != 0 && batch->model->last_stream != listener_stream) return -6;
    const int import_status = gemma4_cache_import_async(
        batch->slots[slot].cache, cache, row->cache_position, listener_stream);
    if (import_status != 0 ||
        cudaMemcpyAsync(batch->slots[slot].prefix_attention,
                        prefix_attention_meta->ptr,
                        (size_t)prefix_len * sizeof(int64_t),
                        cudaMemcpyDeviceToDevice, listener_stream) != cudaSuccess ||
        cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -7;
    }
    gemma4_listener_batch_slot &state = batch->slots[slot];
    state.context_id = row->native_context_id;
    state.session_generation = row->session_generation;
    state.state_epoch_id = row->state_epoch_id;
    state.expected_frame_id = row->frame_id;
    state.expected_state_version = row->listener_state_version;
    state.expected_sequence_position = row->sequence_position;
    state.prefix_len = prefix_len;
    state.current_position = row->cache_position;
    state.primed = true;
    batch->model->last_stream = listener_stream;
    return 0;
}

/* One session's attention uses the shared attention workspace but its own
 * query rows, prefix mask and cache.  Projection inputs/outputs are batch-wide. */
static int gemma4_batch_attention_one(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    int layer_idx,
    const __nv_bfloat16 *query,
    __nv_bfloat16 *attention_out,
    uint32_t base_position,
    cudaStream_t stream
) {
    gemma4_listener_ctx *ctx = batch->model;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    const int d = (int)spec->head_dim;
    const int q_dim = GEMMA4_NUM_ATTN_HEADS * d;
    const int64_t *prefix = batch->slots[slot].prefix_attention;
    const uint32_t prefix_len = batch->slots[slot].prefix_len;
    const __nv_bfloat16 *key = nullptr;
    const __nv_bfloat16 *value = nullptr;
    int total_k_len = 0;
    int start = 0;
    gemma4_cache_get_ptrs(batch->slots[slot].cache, layer_idx, GEMMA4_STREAM_Q_LEN,
                          base_position, &key, &value, &total_k_len, &start);
    const int visible = total_k_len - start;
    if (!key || !value || visible <= 0 || visible > (int)ctx->config.max_seq_len) return -8;
    if (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED && layer_idx == 0 &&
        gemma4_batch_trace_record_attention_input(batch, slot, layer_idx, query,
            key, value, (uint32_t)start, (uint32_t)visible, base_position,
            (uint32_t)d, stream) != 0) return -8;

    const bool ordered = ctx->attention_backend == GEMMA4_ATTN_NATIVE_SPLITK_GQA &&
        spec->attn_type == GEMMA_ATTN_FULL && d == 512 && start == 0 &&
        ((ctx->splitk_layer_mask >> layer_idx) & UINT64_C(1)) != 0 &&
        visible > GEMMA4_ORDERED_MIN_KEYS_EXCLUSIVE;
    if (ordered) {
        if (!ctx->ordered_workspace.initialized ||
            visible > (int)ctx->ordered_workspace.max_keys ||
            gemma4_ordered_attention_launch(
                query, key, value, prefix, prefix_len, (uint32_t)visible,
                base_position, spec->attn_scaling, attention_out,
                stream, &ctx->ordered_workspace) != 0) return -8;
        ctx->ordered_calls_by_layer[layer_idx]++;
        if (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED && layer_idx == 0 &&
            gemma4_batch_trace_record_attention_output(batch, slot,
                attention_out, stream) != 0) return -8;
        return cudaPeekAtLastError() == cudaSuccess ? 0 : -8;
    }

    const int full_slot = d == 512 ? full_repeat_slot_for_layer(layer_idx) : -1;
    if (full_slot >= 0 && gemma4_listener_ensure_full_repeat_capacity(
            ctx, full_slot, visible, stream) != 0) return -8;

    kernel_build_explicit_sdpa_mask<<<
        (GEMMA4_STREAM_Q_LEN * visible + 255) / 256, 256, 0, stream
    >>>(ctx->d_sdpa_mask, GEMMA4_STREAM_Q_LEN, visible, start, base_position,
        spec->sliding_window, prefix, prefix_len);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;

    const uint64_t q_dims[] = {1, GEMMA4_NUM_ATTN_HEADS, GEMMA4_STREAM_Q_LEN, (uint64_t)d};
    const uint64_t q_strides[] = {
        (uint64_t)q_dim * GEMMA4_STREAM_Q_LEN,
        (uint64_t)d,
        (uint64_t)q_dim,
        1,
    };
    const uint64_t kv_dims[] = {1, GEMMA4_NUM_ATTN_HEADS, (uint64_t)visible, (uint64_t)d};
    const uint64_t kv_strides[] = {
        (uint64_t)GEMMA4_NUM_ATTN_HEADS * visible * d,
        (uint64_t)visible * d,
        (uint64_t)d,
        1,
    };
    const uint64_t mask_dims[] = {1, 1, GEMMA4_STREAM_Q_LEN, (uint64_t)visible};
    const uint64_t mask_strides[] = {
        (uint64_t)GEMMA4_STREAM_Q_LEN * visible,
        (uint64_t)GEMMA4_STREAM_Q_LEN * visible,
        (uint64_t)visible,
        1,
    };
    const void *bridge_k = key + (size_t)start * d;
    const void *bridge_v = value + (size_t)start * d;
    __nv_bfloat16 *repeat_k = nullptr;
    __nv_bfloat16 *repeat_v = nullptr;
    if (full_slot >= 0) {
        repeat_k = ctx->d_sdpa_full_k_repeat[full_slot];
        repeat_v = ctx->d_sdpa_full_v_repeat[full_slot];
    } else if (d == 256 && spec->attn_type == GEMMA_ATTN_SLIDING &&
               ((!spec->is_kv_shared && layer_idx == 13) ||
                (spec->is_kv_shared && spec->kv_shared_src_index == 13))) {
        repeat_k = ctx->d_sdpa_shared_sliding_k_repeat;
        repeat_v = ctx->d_sdpa_shared_sliding_v_repeat;
    } else {
        repeat_k = ctx->d_sdpa_k_repeat;
        repeat_v = ctx->d_sdpa_v_repeat;
    }
    if (!repeat_k || !repeat_v) return -8;
    const int repeat_blocks = (visible * d + 255) / 256;
    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(bridge_k), repeat_k, visible, d);
    kernel_repeat_kv_bf16<<<repeat_blocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16 *>(bridge_v), repeat_v, visible, d);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    gemma4_tensor_meta q_meta = make_bridge_meta(query, GEMMA4_META_BF16, 4, q_dims, q_strides, 0);
    gemma4_tensor_meta k_meta = make_bridge_meta(repeat_k, GEMMA4_META_BF16, 4, kv_dims, kv_strides, 1);
    gemma4_tensor_meta v_meta = make_bridge_meta(repeat_v, GEMMA4_META_BF16, 4, kv_dims, kv_strides, 1);
    gemma4_tensor_meta mask_meta = make_bridge_meta(
        ctx->d_sdpa_mask, GEMMA4_META_BOOL, 4, mask_dims, mask_strides, 1);
    gemma4_tensor_meta out_meta = make_bridge_meta(
        attention_out, GEMMA4_META_BF16, 4, q_dims, q_strides, 0);
    if (gemma4_listener_debug_attention(
            query, &q_meta, repeat_k, &k_meta, repeat_v, &v_meta,
            ctx->d_sdpa_mask, &mask_meta, attention_out, &out_meta,
            (double)spec->attn_scaling, stream) != 0) return -8;
    if (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED && layer_idx == 0 &&
        gemma4_batch_trace_record_attention_output(batch, slot,
            attention_out, stream) != 0) return -8;
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -8;
}

/* Uses the same BF16 boundaries/order as execute_layer(), except each dense
 * layer projection is a single GEMM over all N*2 rows. */
static int gemma4_batch_execute_layer(
    gemma4_listener_batch_ctx *batch,
    uint32_t n,
    int layer_idx,
    cudaStream_t stream
) {
    gemma4_listener_ctx *ctx = batch->model;
    const int rows = (int)(n * GEMMA4_STREAM_Q_LEN);
    const int hidden = GEMMA4_HIDDEN_SIZE;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    const gemma4_layer_weights *lw = &ctx->weights.layers[layer_idx];
    const int d = (int)spec->head_dim;
    const int qdim = GEMMA4_NUM_ATTN_HEADS * d;
    const int intermediate = (int)spec->intermediate_size;
    const bool capture_trace = batch->trace_state == GEMMA4_BATCH_TRACE_ARMED && layer_idx == 0;
#define GEMMA4_TRACE_BEGIN(id, in0, in1, in2) \
    do { if (capture_trace && gemma4_batch_trace_begin(batch, id, in0, in1, in2, stream) != 0) return -1000 - (int)(id); } while (0)
#define GEMMA4_TRACE_FINISH(id, out) \
    do { if (capture_trace && gemma4_batch_trace_finish(batch, id, out, stream) != 0) return -2000 - (int)(id); } while (0)

    const bool capture_probe_rms = batch->probe_state == GEMMA4_BATCH_PROBE_ARMED &&
        layer_idx == 0;
    if (capture_probe_rms) {
        if (rows != (int)GEMMA4_BATCH_PROBE_Q_ROWS ||
            !batch->probe_rms_input || !batch->probe_rms_batched ||
            !batch->probe_rms_split ||
            cudaMemcpyAsync(batch->probe_rms_input, batch->hidden_a,
                (size_t)rows * hidden * sizeof(__nv_bfloat16),
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
            return -30;
        }
    }
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_INPUT_NORM, batch->hidden_a, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->hidden_a, lw->input_norm_w,
            batch->norm, (uint32_t)rows, hidden, spec->input_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_INPUT_NORM, batch->norm);
    if (capture_probe_rms) {
        const size_t rms_bytes = (size_t)rows * hidden * sizeof(__nv_bfloat16);
        uint32_t actual_bx = 0, actual_by = 0, reference_bx = 0, reference_by = 0;
        if (!gemma4_batch_probe_rms_launch_shape(
                GEMMA4_BATCH_PROBE_Q_ROWS, GEMMA4_BATCH_PROBE_HIDDEN_DIM,
                &actual_bx, &actual_by) ||
            !gemma4_batch_probe_rms_launch_shape(
                GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, GEMMA4_BATCH_PROBE_HIDDEN_DIM,
                &reference_bx, &reference_by) ||
            cudaMemcpyAsync(batch->probe_rms_batched, batch->norm, rms_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess ||
            run_rmsnorm_exact_or_custom(ctx, batch->probe_rms_input,
                lw->input_norm_w, batch->probe_rms_split,
                GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, hidden,
                spec->input_norm_eps, stream) != 0 ||
            run_rmsnorm_exact_or_custom(ctx,
                batch->probe_rms_input +
                    (size_t)GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS * hidden,
                lw->input_norm_w,
                batch->probe_rms_split +
                    (size_t)GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS * hidden,
                GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS, hidden,
                spec->input_norm_eps, stream) != 0 ||
            !gemma4_batch_probe_mark_rms_captured(
                &batch->probe_state, (uint32_t)rows, (uint32_t)hidden)) {
            batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
            return -31;
        }
        batch->probe_info.struct_size = sizeof(batch->probe_info);
        batch->probe_info.state = (uint32_t)batch->probe_state;
        batch->probe_info.row_count = GEMMA4_BATCH_PROBE_Q_ROWS;
        batch->probe_info.rms_dim = GEMMA4_BATCH_PROBE_HIDDEN_DIM;
        batch->probe_info.rms_backend_id = ctx->rmsnorm_backend;
        batch->probe_info.rms_batched_block_x = actual_bx;
        batch->probe_info.rms_batched_block_y = actual_by;
        batch->probe_info.rms_reference_block_x = reference_bx;
        batch->probe_info.rms_reference_block_y = reference_by;
        batch->probe_info.rms_weight_device_ptr =
            (uint64_t)(uintptr_t)lw->input_norm_w;
        batch->probe_info.rms_eps = (float)spec->input_norm_eps;
    }
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_Q_PROJ, batch->norm, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_Q),
            lw->q_proj_w, batch->norm, batch->q,
            (uint32_t)rows, GEMMA4_STREAM_Q_LEN, (uint32_t)qdim,
            (uint32_t)hidden, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_Q_PROJ, batch->q);
    batch->projection_gemm_calls++;
    batch->projection_rows += rows;
    if (capture_probe_rms) {
        const size_t q_bytes = (size_t)rows * qdim * sizeof(__nv_bfloat16);
        if ((uint32_t)qdim > GEMMA4_NUM_ATTN_HEADS * 512u ||
            !batch->probe_q_actual ||
            cudaMemcpyAsync(batch->probe_q_actual, batch->q, q_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
            batch->probe_state = GEMMA4_BATCH_PROBE_ERROR;
            return -32;
        }
        batch->probe_info.q_dim = (uint32_t)qdim;
        batch->probe_info.q_batched_m = GEMMA4_BATCH_PROBE_Q_ROWS;
        batch->probe_info.q_split_m = GEMMA4_BATCH_PROBE_RMS_REFERENCE_ROWS;
        batch->probe_info.q_k = GEMMA4_HIDDEN_SIZE;
        batch->probe_info.q_lda = GEMMA4_HIDDEN_SIZE;
        batch->probe_info.q_ldb = GEMMA4_HIDDEN_SIZE;
        batch->probe_info.q_ldc = (uint32_t)qdim;
        batch->probe_info.q_weight_device_ptr =
            (uint64_t)(uintptr_t)lw->q_proj_w;
    }
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_Q_NORM, batch->q, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->q, lw->q_norm_w, batch->q,
            (uint32_t)(rows * GEMMA4_NUM_ATTN_HEADS), (uint32_t)d,
            spec->q_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_ATTN_HEADS) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_Q_NORM, batch->q);
    const float *inv_freq = spec->attn_type == GEMMA_ATTN_FULL
        ? ctx->d_inv_freq_full : ctx->d_inv_freq_sliding;
    for (uint32_t slot = 0; slot < n; ++slot) {
        __nv_bfloat16 *session_q = batch->q +
            (size_t)slot * GEMMA4_STREAM_Q_LEN * qdim;
        kernel_rope_bf16<<<dim3(GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_ATTN_HEADS), d / 2, 0, stream>>>(
            session_q, inv_freq, GEMMA4_NUM_ATTN_HEADS, GEMMA4_STREAM_Q_LEN, d,
            batch->slots[batch->active_slots[slot]].current_position, nullptr, spec->attn_scaling);
    }
    if (cudaPeekAtLastError() != cudaSuccess) return -8;

    if (!spec->is_kv_shared) {
        GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_K_PROJ, batch->norm, nullptr, nullptr);
        if (gemma4_batch_run_per_session_gemm(batch,
                GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_K),
                lw->k_proj_w, batch->norm, batch->k,
                (uint32_t)rows, GEMMA4_STREAM_Q_LEN, (uint32_t)d,
                (uint32_t)hidden, stream) != 0) return -4;
        GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_K_PROJ, batch->k);
        GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_V_PROJ, batch->norm, nullptr, nullptr);
        if (gemma4_batch_run_per_session_gemm(batch,
                GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_V),
                lw->v_proj_w, batch->norm, batch->v,
                (uint32_t)rows, GEMMA4_STREAM_Q_LEN, (uint32_t)d,
                (uint32_t)hidden, stream) != 0) return -4;
        GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_V_PROJ, batch->v);
        batch->projection_gemm_calls += 2;
        batch->projection_rows += (uint64_t)rows * 2;
        GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_K_NORM, batch->k, nullptr, nullptr);
        if (run_rmsnorm_exact_or_custom(ctx, batch->k, lw->k_norm_w, batch->k,
                (uint32_t)rows, (uint32_t)d, spec->k_norm_eps, stream,
                GEMMA4_STREAM_Q_LEN) != 0) return -8;
        GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_K_NORM, batch->k);
        GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_V_NORM, batch->v, nullptr, nullptr);
        if (run_rmsnorm_exact_or_custom(ctx, batch->v, lw->v_norm_w, batch->v,
                (uint32_t)rows, (uint32_t)d, spec->v_norm_eps, stream,
                GEMMA4_STREAM_Q_LEN) != 0) return -8;
        GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_V_NORM, batch->v);
        for (uint32_t slot = 0; slot < n; ++slot) {
            __nv_bfloat16 *session_k = batch->k + (size_t)slot * GEMMA4_STREAM_Q_LEN * d;
            const uint32_t native_slot = batch->active_slots[slot];
            kernel_rope_bf16<<<dim3(GEMMA4_STREAM_Q_LEN, 1), d / 2, 0, stream>>>(
                session_k, inv_freq, 1, GEMMA4_STREAM_Q_LEN, d,
                batch->slots[native_slot].current_position, nullptr, spec->attn_scaling);
            if (gemma4_cache_append_async(batch->slots[native_slot].cache, layer_idx,
                    session_k, batch->v + (size_t)slot * GEMMA4_STREAM_Q_LEN * d,
                    GEMMA4_STREAM_Q_LEN, stream) != 0) return -3;
        }
        if (cudaPeekAtLastError() != cudaSuccess) return -8;
    }

    for (uint32_t slot = 0; slot < n; ++slot) {
        const __nv_bfloat16 *session_q = batch->q +
            (size_t)slot * GEMMA4_STREAM_Q_LEN * qdim;
        __nv_bfloat16 *session_attn = batch->attn +
            (size_t)slot * GEMMA4_STREAM_Q_LEN * qdim;
        const uint32_t native_slot = batch->active_slots[slot];
        if (gemma4_batch_attention_one(batch, native_slot, layer_idx, session_q,
                session_attn, batch->slots[native_slot].current_position, stream) != 0) return -8;
    }

    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_O_PROJ, batch->attn, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_O),
            lw->o_proj_w, batch->attn,
            batch->hidden_b, (uint32_t)rows, GEMMA4_STREAM_Q_LEN,
            (uint32_t)hidden, (uint32_t)qdim, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_O_PROJ, batch->hidden_b);
    batch->projection_gemm_calls++;
    batch->projection_rows += rows;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_POST_ATTN_NORM, batch->hidden_b, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->hidden_b, lw->post_attn_norm_w,
            batch->hidden_b, (uint32_t)rows, hidden, spec->post_attn_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_POST_ATTN_NORM, batch->hidden_b);
    const int hidden_elems = rows * hidden;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_POST_ATTN_RESIDUAL,
        batch->hidden_a, batch->hidden_b, nullptr);
    kernel_residual_add_bf16<<<(hidden_elems + 255) / 256, 256, 0, stream>>>(
        batch->hidden_a, batch->hidden_b, batch->hidden_a, hidden_elems);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_POST_ATTN_RESIDUAL, batch->hidden_a);

    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PRE_FF_NORM, batch->hidden_a, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->hidden_a, lw->pre_ff_norm_w,
            batch->norm, (uint32_t)rows, hidden, spec->pre_ff_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PRE_FF_NORM, batch->norm);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_GATE_PROJ, batch->norm, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_GATE),
            lw->gate_proj_w, batch->norm,
            batch->gate, (uint32_t)rows, GEMMA4_STREAM_Q_LEN,
            (uint32_t)intermediate, (uint32_t)hidden, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_GATE_PROJ, batch->gate);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_UP_PROJ, batch->norm, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_UP),
            lw->up_proj_w, batch->norm,
            batch->up, (uint32_t)rows, GEMMA4_STREAM_Q_LEN,
            (uint32_t)intermediate, (uint32_t)hidden, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_UP_PROJ, batch->up);
    batch->projection_gemm_calls += 2;
    batch->projection_rows += (uint64_t)rows * 2;
    const int ffn_elems = rows * intermediate;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_GATE_GELU, batch->gate, nullptr, nullptr);
    kernel_gelu_tanh_bf16<<<(ffn_elems + 255) / 256, 256, 0, stream>>>(
        batch->gate, batch->mlp, ffn_elems);
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_GATE_GELU, batch->mlp);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_GATE_UP_MUL, batch->mlp, batch->up, nullptr);
    kernel_mul_bf16<<<(ffn_elems + 255) / 256, 256, 0, stream>>>(
        batch->mlp, batch->up, batch->mlp, ffn_elems);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_GATE_UP_MUL, batch->mlp);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_DOWN_PROJ, batch->mlp, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_DOWN),
            lw->down_proj_w, batch->mlp,
            batch->hidden_b, (uint32_t)rows, GEMMA4_STREAM_Q_LEN, (uint32_t)hidden,
            (uint32_t)intermediate, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_DOWN_PROJ, batch->hidden_b);
    batch->projection_gemm_calls++;
    batch->projection_rows += rows;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_POST_FF_NORM, batch->hidden_b, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->hidden_b, lw->post_ff_norm_w,
            batch->hidden_b, (uint32_t)rows, hidden, spec->post_ff_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_POST_FF_NORM, batch->hidden_b);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_POST_FF_RESIDUAL,
        batch->hidden_a, batch->hidden_b, nullptr);
    kernel_residual_add_bf16<<<(hidden_elems + 255) / 256, 256, 0, stream>>>(
        batch->hidden_a, batch->hidden_b, batch->hidden_a, hidden_elems);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_POST_FF_RESIDUAL, batch->hidden_a);

    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_GATE_PROJ, batch->hidden_a, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_PLE_GATE),
            lw->ple_gate_w, batch->hidden_a,
            batch->ple_gate, (uint32_t)rows, GEMMA4_STREAM_Q_LEN,
            GEMMA4_PLE_DIM, (uint32_t)hidden, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_GATE_PROJ, batch->ple_gate);
    batch->projection_gemm_calls++;
    batch->projection_rows += rows;
    const int ple_elems = rows * GEMMA4_PLE_DIM;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_GELU, batch->ple_gate, nullptr, nullptr);
    kernel_gelu_tanh_bf16<<<(ple_elems + 255) / 256, 256, 0, stream>>>(
        batch->ple_gate, batch->ple_act, ple_elems);
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_GELU, batch->ple_act);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_LAYER_MULT,
        batch->ple_act, batch->projected_ple, nullptr);
    kernel_ple_layer_mult_bf16<<<(ple_elems + 255) / 256, 256, 0, stream>>>(
        batch->ple_act, batch->projected_ple, batch->ple_act, rows, layer_idx);
    if (cudaPeekAtLastError() != cudaSuccess) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_LAYER_MULT, batch->ple_act);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_PROJ, batch->ple_act, nullptr, nullptr);
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_ID(layer_idx, GEMMA4_BATCH_GEMM_OP_PLE_PROJ),
            lw->ple_proj_w, batch->ple_act,
            batch->ple_out, (uint32_t)rows, GEMMA4_STREAM_Q_LEN,
            (uint32_t)hidden, GEMMA4_PLE_DIM, stream) != 0) return -4;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_PROJ, batch->ple_out);
    batch->projection_gemm_calls++;
    batch->projection_rows += rows;
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_NORM, batch->ple_out, nullptr, nullptr);
    if (run_rmsnorm_exact_or_custom(ctx, batch->ple_out, lw->ple_norm_w,
            batch->ple_out, (uint32_t)rows, hidden, spec->post_ple_norm_eps, stream,
            GEMMA4_STREAM_Q_LEN) != 0) return -8;
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_NORM, batch->ple_out);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_PLE_RESIDUAL,
        batch->hidden_a, batch->ple_out, nullptr);
    kernel_residual_add_bf16<<<(hidden_elems + 255) / 256, 256, 0, stream>>>(
        batch->hidden_a, batch->ple_out, batch->hidden_a, hidden_elems);
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_PLE_RESIDUAL, batch->hidden_a);
    GEMMA4_TRACE_BEGIN(GEMMA4_BATCH_TRACE_L0_LAYER_SCALE, batch->hidden_a, nullptr, nullptr);
    kernel_scale_bf16<<<(hidden_elems + 255) / 256, 256, 0, stream>>>(
        batch->hidden_a, batch->hidden_a, hidden_elems, spec->layer_scalar);
    GEMMA4_TRACE_FINISH(GEMMA4_BATCH_TRACE_L0_LAYER_SCALE, batch->hidden_a);
    if (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED &&
        layer_idx == GEMMA4_NUM_LAYERS - 1) {
        if (gemma4_batch_trace_begin(batch, GEMMA4_BATCH_TRACE_FINAL_LAYER_OUTPUT,
                batch->hidden_a, nullptr, nullptr, stream) != 0 ||
            gemma4_batch_trace_finish(batch, GEMMA4_BATCH_TRACE_FINAL_LAYER_OUTPUT,
                batch->hidden_a, stream) != 0) return -1032;
    }
    const int final_status = cudaPeekAtLastError() == cudaSuccess ? 0 : -8;
#undef GEMMA4_TRACE_BEGIN
#undef GEMMA4_TRACE_FINISH
    return final_status;
}

static int gemma4_listener_batch_step_core(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *inputs_embeds,
    const gemma4_tensor_meta *inputs_embeds_meta,
    const void *per_layer_inputs,
    const gemma4_tensor_meta *per_layer_inputs_meta,
    bool embedded_text_input,
    bool preserve_listener_cursors,
    cudaStream_t listener_stream
) {
    if (!batch || !rows || !inputs_embeds || !inputs_embeds_meta || batch->poisoned ||
        !gemma4_batch_size_supported(row_count) || row_count > batch->capacity ||
        batch->output_lease_active || batch->output_generation == UINT64_MAX) return -1;
    if ((batch->probe_state == GEMMA4_BATCH_PROBE_ARMED &&
         !gemma4_batch_probe_rows_valid(row_count)) ||
        (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED &&
         !gemma4_batch_trace_rows_valid(row_count)) ||
        (batch->probe_state != GEMMA4_BATCH_PROBE_IDLE &&
         batch->probe_state != GEMMA4_BATCH_PROBE_ARMED &&
         batch->probe_state != GEMMA4_BATCH_PROBE_DONE)) return -25;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    int device = -1;
    const uint64_t dims[] = {row_count, GEMMA4_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE, 0};
    if (cudaGetDevice(&device) != cudaSuccess ||
        !validate_tensor_meta(inputs_embeds_meta, inputs_embeds, device,
                              GEMMA4_META_BF16, 3, dims)) return -10;
    const uint64_t ple_dims[] = {
        row_count, GEMMA4_STREAM_Q_LEN, GEMMA4_NUM_LAYERS, GEMMA4_PLE_DIM
    };
    if (embedded_text_input) {
        if (!per_layer_inputs || !per_layer_inputs_meta ||
            !validate_tensor_meta(per_layer_inputs_meta, per_layer_inputs, device,
                                  GEMMA4_META_BF16, 4, ple_dims)) return -10;
    } else if (per_layer_inputs || per_layer_inputs_meta) {
        return -1;
    }
    for (uint32_t i = 0; i < row_count; ++i) {
        uint32_t native_slot = batch->capacity;
        for (uint32_t candidate = 0; candidate < batch->capacity; ++candidate) {
            if (batch->slots[candidate].primed &&
                batch->slots[candidate].context_id == rows[i].native_context_id) {
                native_slot = candidate;
                break;
            }
        }
        if (native_slot == batch->capacity ||
            !gemma4_batch_row_identity_matches(batch, native_slot, &rows[i], true) ||
            rows[i].cache_position > batch->model->config.max_seq_len - GEMMA4_STREAM_Q_LEN ||
            rows[i].valid_history_tokens != rows[i].cache_position) return -1;
        for (uint32_t j = 0; j < i; ++j) {
            if (rows[i].native_context_id == rows[j].native_context_id) return -1;
        }
        const gemma4_listener_batch_slot &state = batch->slots[native_slot];
        if (rows[i].frame_id != state.expected_frame_id ||
            rows[i].listener_state_version != state.expected_state_version ||
            rows[i].sequence_position != state.expected_sequence_position ||
            (!preserve_listener_cursors && (rows[i].frame_id == UINT64_MAX ||
             rows[i].listener_state_version == UINT64_MAX ||
             rows[i].sequence_position == UINT64_MAX))) return -1;
        batch->active_slots[i] = native_slot;
    }
    if (batch->trace_state == GEMMA4_BATCH_TRACE_ARMED &&
        gemma4_batch_trace_update_attention_layout(batch, row_count) != 0) return -25;
    if (cudaMemcpyAsync(batch->hidden_a, inputs_embeds,
            (size_t)row_count * GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE * sizeof(__nv_bfloat16),
            cudaMemcpyDeviceToDevice, listener_stream) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -7;
    }
    gemma4_listener_ctx *ctx = batch->model;
    const int rows_n = (int)(row_count * GEMMA4_STREAM_Q_LEN);
    const int ple_width = GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
    const int ple_elements = rows_n * ple_width;
    const bool trace_prelude = batch->trace_state == GEMMA4_BATCH_TRACE_ARMED;
    if (embedded_text_input) {
        /* Text PLE already includes token identity, context projection, RMSNorm,
         * and Gemma's per-layer input scale. Do not apply audio frame padding
         * or a second projection/scale here. */
        if (trace_prelude) return -25;
        const size_t token_ple_bytes = (size_t)row_count * GEMMA4_STREAM_Q_LEN *
            GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM * sizeof(__nv_bfloat16);
        if (cudaMemcpyAsync(batch->projected_ple, per_layer_inputs,
                token_ple_bytes, cudaMemcpyDeviceToDevice, listener_stream) != cudaSuccess) {
            gemma4_batch_poison(batch);
            return -7;
        }
    } else {
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_GEMM, batch->hidden_a, nullptr, nullptr,
            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -301;
    }
    if (gemma4_batch_run_per_session_gemm(batch,
            GEMMA4_BATCH_GEMM_OP_PRELUDE,
            ctx->weights.per_layer_model_proj_w, batch->hidden_a,
            batch->ple_raw, (uint32_t)rows_n, GEMMA4_STREAM_Q_LEN,
            (uint32_t)ple_width, GEMMA4_HIDDEN_SIZE, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -4;
    }
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_GEMM, batch->ple_raw, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -401;
    }
    batch->projection_gemm_calls++;
    batch->projection_rows += rows_n;
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_SCALE, batch->ple_raw, nullptr, nullptr,
            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -302;
    }
    if (rmsnorm_uses_exact_path(ctx)) {
        kernel_scale_bf16_fp32_scalar<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
            batch->ple_raw, batch->projected_ple, ple_elements,
            ctx->weights.per_layer_model_proj_scale);
    } else {
        kernel_scale_bf16<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
            batch->ple_raw, batch->projected_ple, ple_elements,
            ctx->weights.per_layer_model_proj_scale);
    }
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_SCALE, batch->projected_ple, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -402;
    }
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS, batch->projected_ple, nullptr, nullptr,
            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -303;
    }
    if (run_rmsnorm_exact_or_custom(ctx, batch->projected_ple,
            ctx->weights.per_layer_proj_norm_w, batch->projected_ple,
            (uint32_t)(rows_n * GEMMA4_NUM_LAYERS), GEMMA4_PLE_DIM,
            1e-6, listener_stream,
            GEMMA4_STREAM_Q_LEN * GEMMA4_NUM_LAYERS) != 0) {
        gemma4_batch_poison(batch);
        return -8;
    }
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS, batch->projected_ple, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -403;
    }
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_FRAME_PAD, ctx->config.frame_pad_ple_ptr,
            nullptr, nullptr, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -304;
    }
    kernel_repeat_frame_pad_ple_bf16<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
        (const __nv_bfloat16 *)ctx->config.frame_pad_ple_ptr,
        batch->ple_raw, rows_n, ple_width);
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_FRAME_PAD, batch->ple_raw, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -404;
    }
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_PAD_RESIDUAL, batch->projected_ple,
            batch->ple_raw, nullptr, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -305;
    }
    kernel_residual_add_bf16<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
        batch->projected_ple, batch->ple_raw, batch->projected_ple, ple_elements);
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_PAD_RESIDUAL, batch->projected_ple,
            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -405;
    }
    if (trace_prelude && gemma4_batch_trace_begin(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_INPUT_SCALE, batch->projected_ple,
            nullptr, nullptr, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -306;
    }
    if (rmsnorm_uses_exact_path(ctx)) {
        kernel_scale_bf16_fp32_scalar<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
            batch->projected_ple, batch->projected_ple, ple_elements,
            ctx->weights.per_layer_input_scale);
    } else {
        kernel_scale_bf16<<<(ple_elements + 255) / 256, 256, 0, listener_stream>>>(
            batch->projected_ple, batch->projected_ple, ple_elements,
            ctx->weights.per_layer_input_scale);
    }
    if (trace_prelude && gemma4_batch_trace_finish(batch,
            GEMMA4_BATCH_TRACE_AUDIO_PLE_INPUT_SCALE, batch->projected_ple,
            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -406;
    }
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -8;
    }

    for (int layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
        const int layer_status = gemma4_batch_execute_layer(batch, row_count, layer, listener_stream);
        if (layer_status != 0) {
            gemma4_batch_poison(batch);
            return layer_status;
        }
        batch->layers_executed++;
    }
    if (run_rmsnorm_exact_or_custom(ctx, batch->hidden_a, ctx->weights.final_norm_w,
            batch->output, (uint32_t)rows_n, GEMMA4_HIDDEN_SIZE,
            GEMMA4_FINAL_NORM_EPS, listener_stream,
            GEMMA4_STREAM_Q_LEN) != 0 ||
        guard_output_finite(ctx, batch->output, rows_n * GEMMA4_HIDDEN_SIZE,
                            listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -22;
    }
    for (uint32_t i = 0; i < row_count; ++i) {
        const uint32_t native_slot = batch->active_slots[i];
        if (gemma4_cache_advance_position(batch->slots[native_slot].cache,
                                          GEMMA4_STREAM_Q_LEN) != 0 ||
            gemma4_cache_set_device_position_async(batch->slots[native_slot].cache,
                batch->slots[native_slot].current_position + GEMMA4_STREAM_Q_LEN,
                listener_stream) != 0) {
            gemma4_batch_poison(batch);
            return -23;
        }
        batch->slots[native_slot].current_position += GEMMA4_STREAM_Q_LEN;
        if (!preserve_listener_cursors) {
            batch->slots[native_slot].expected_frame_id = rows[i].frame_id + 1;
            batch->slots[native_slot].expected_state_version = rows[i].listener_state_version + 1;
            batch->slots[native_slot].expected_sequence_position = rows[i].sequence_position + 1;
        }
    }
    const bool trace_armed = batch->trace_state == GEMMA4_BATCH_TRACE_ARMED;
    if (trace_armed && !gemma4_batch_trace_capture_complete(batch)) {
        gemma4_batch_poison(batch);
        batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -24;
    }
    if (trace_armed) {
        for (uint32_t stage_id = 0; stage_id < GEMMA4_BATCH_TRACE_STAGE_COUNT; ++stage_id)
            batch->trace_records[stage_id].info.forward_generation = batch->output_generation + 1u;
    }
    if (cudaEventRecord(batch->completion_event, listener_stream) != cudaSuccess ||
        (trace_armed && cudaEventRecord(batch->trace_event, listener_stream) != cudaSuccess) ||
        cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(batch);
        if (trace_armed) batch->trace_state = GEMMA4_BATCH_TRACE_ERROR;
        return -24;
    }
    ctx->last_stream = listener_stream;
    batch->session_rows += row_count;
    batch->forward_calls++;
    for (uint32_t i = 0; i < row_count; ++i) batch->output_rows[i] = rows[i];
    batch->output_row_count = row_count;
    batch->output_embedded_text_input = embedded_text_input;
    batch->output_preserves_listener_cursors = preserve_listener_cursors;
    batch->output_generation++;
    batch->completed_output_generation = 0;
    batch->completed_output_row_count = 0;
    if (trace_armed) batch->trace_state = GEMMA4_BATCH_TRACE_CAPTURED;
    if (batch->probe_state == GEMMA4_BATCH_PROBE_RMS_CAPTURED) {
        batch->probe_info.forward_generation = batch->output_generation;
    }
    batch->output_lease_active = true;
    batch->clone_enqueued = false;
    return 0;
}

int gemma4_listener_batch_step_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *audio_soft,
    const gemma4_tensor_meta *audio_soft_meta,
    cudaStream_t listener_stream
) {
    return gemma4_listener_batch_step_core(
        batch, rows, row_count, audio_soft, audio_soft_meta,
        nullptr, nullptr, false, false, listener_stream);
}

int gemma4_listener_batch_step_queued_audio_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *audio_soft,
    const gemma4_tensor_meta *audio_soft_meta,
    cudaStream_t listener_stream
) {
    return gemma4_listener_batch_step_core(
        batch, rows, row_count, audio_soft, audio_soft_meta,
        nullptr, nullptr, false, true, listener_stream);
}

int gemma4_listener_batch_step_embedded_async(
    gemma4_listener_batch_ctx *batch,
    const gemma4_listener_batch_row *rows,
    uint32_t row_count,
    const void *inputs_embeds,
    const gemma4_tensor_meta *inputs_embeds_meta,
    const void *per_layer_inputs,
    const gemma4_tensor_meta *per_layer_inputs_meta,
    cudaStream_t listener_stream
) {
    return gemma4_listener_batch_step_core(
        batch, rows, row_count, inputs_embeds, inputs_embeds_meta,
        per_layer_inputs, per_layer_inputs_meta, true, true, listener_stream);
}

int gemma4_listener_batch_truncate_after_step_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *pre_step_row,
    uint32_t logical_q_len,
    cudaStream_t listener_stream
) {
    if (!batch || !pre_step_row || batch->poisoned || batch->output_lease_active ||
        slot >= batch->capacity || logical_q_len >= GEMMA4_STREAM_Q_LEN ||
        !batch->stream_bound || listener_stream != batch->listener_stream) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    gemma4_listener_batch_slot &state = batch->slots[slot];
    const bool embedded_text_input = batch->output_embedded_text_input;
    const bool preserve_listener_cursors = batch->output_preserves_listener_cursors;
    if (!state.primed || pre_step_row->native_context_id != state.context_id ||
        pre_step_row->session_generation != state.session_generation ||
        pre_step_row->state_epoch_id != state.state_epoch_id ||
        pre_step_row->cache_position > batch->model->config.max_seq_len - GEMMA4_STREAM_Q_LEN ||
        pre_step_row->valid_history_tokens != pre_step_row->cache_position ||
        (!preserve_listener_cursors && (pre_step_row->frame_id == UINT64_MAX ||
         pre_step_row->listener_state_version == UINT64_MAX ||
         pre_step_row->sequence_position == UINT64_MAX)) ||
        state.current_position != pre_step_row->cache_position + GEMMA4_STREAM_Q_LEN ||
        (embedded_text_input && logical_q_len == 0u) ||
        (embedded_text_input &&
         (state.expected_frame_id != pre_step_row->frame_id ||
          state.expected_state_version != pre_step_row->listener_state_version ||
          state.expected_sequence_position != pre_step_row->sequence_position)) ||
        (!preserve_listener_cursors &&
         (state.expected_frame_id != pre_step_row->frame_id + 1u ||
          state.expected_state_version != pre_step_row->listener_state_version + 1u ||
          state.expected_sequence_position != pre_step_row->sequence_position + 1u)) ||
        (preserve_listener_cursors &&
         (state.expected_frame_id != pre_step_row->frame_id ||
          state.expected_state_version != pre_step_row->listener_state_version ||
          state.expected_sequence_position != pre_step_row->sequence_position)) ||
        state.current_position != gemma4_cache_valid_kv_len(state.cache)) return -2;
    const uint32_t logical_position = pre_step_row->cache_position + logical_q_len;
    if (logical_position < state.prefix_len || logical_position > state.current_position) return -3;
    if (gemma4_cache_truncate_async(state.cache, logical_position, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -4;
    }
    state.current_position = logical_position;
    batch->model->last_stream = listener_stream;
    return 0;
}

int gemma4_listener_batch_rewind_session_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *current_row,
    uint32_t target_position,
    cudaStream_t listener_stream
) {
    if (!batch || !current_row || batch->poisoned || batch->output_lease_active ||
        slot >= batch->capacity || !batch->stream_bound ||
        listener_stream != batch->listener_stream) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    gemma4_listener_batch_slot &state = batch->slots[slot];
    if (!gemma4_batch_row_identity_matches(batch, slot, current_row, true) ||
        current_row->session_generation != state.session_generation ||
        current_row->state_epoch_id != state.state_epoch_id ||
        current_row->frame_id != state.expected_frame_id ||
        current_row->listener_state_version != state.expected_state_version ||
        current_row->sequence_position != state.expected_sequence_position ||
        current_row->valid_history_tokens != current_row->cache_position ||
        target_position < state.prefix_len ||
        target_position > state.current_position ||
        state.current_position != gemma4_cache_valid_kv_len(state.cache)) return -2;
    if (gemma4_cache_truncate_async(state.cache, target_position, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -3;
    }
    state.current_position = target_position;
    batch->model->last_stream = listener_stream;
    return 0;
}

int gemma4_listener_batch_output_acquire(
    gemma4_listener_batch_ctx *batch,
    uint64_t *out_generation,
    const void **out_tensor,
    const gemma4_listener_batch_row **out_rows,
    uint32_t *out_row_count,
    cudaEvent_t *out_completion_event
) {
    if (!batch || !out_generation || !out_tensor || !out_rows || !out_row_count ||
        !out_completion_event ||
        batch->poisoned || !batch->output_lease_active) return -1;
    *out_generation = batch->output_generation;
    *out_tensor = batch->output;
    *out_rows = batch->output_rows;
    *out_row_count = batch->output_row_count;
    *out_completion_event = batch->completion_event;
    return 0;
}

int gemma4_listener_batch_output_clone_async(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    uint32_t row_count,
    void *clone_tensor,
    const gemma4_tensor_meta *clone_meta,
    cudaStream_t listener_stream,
    cudaEvent_t *out_clone_event
) {
    if (!batch || !clone_tensor || !clone_meta || !out_clone_event || batch->poisoned ||
        !batch->output_lease_active || batch->clone_enqueued ||
        generation != batch->output_generation ||
        row_count != batch->output_row_count ||
        !gemma4_batch_size_supported(row_count) || row_count > batch->capacity ||
        listener_stream != batch->listener_stream) return -1;
    int device = -1;
    const uint64_t dims[] = {row_count, GEMMA4_STREAM_Q_LEN, GEMMA4_HIDDEN_SIZE, 0};
    if (cudaGetDevice(&device) != cudaSuccess ||
        !validate_tensor_meta(clone_meta, clone_tensor, device, GEMMA4_META_BF16, 3, dims)) return -10;
    const size_t elements = (size_t)row_count * GEMMA4_STREAM_Q_LEN * GEMMA4_HIDDEN_SIZE;
    if (cudaMemcpyAsync(clone_tensor, batch->output,
            elements * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice,
            listener_stream) != cudaSuccess ||
        cudaEventRecord(batch->clone_event, listener_stream) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    batch->clone_enqueued = true;
    *out_clone_event = batch->clone_event;
    return 0;
}

int gemma4_listener_batch_output_release(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    cudaEvent_t consumer_done_event,
    cudaStream_t listener_stream
) {
    if (!batch || !consumer_done_event || batch->poisoned ||
        !batch->output_lease_active || !batch->clone_enqueued ||
        generation != batch->output_generation || consumer_done_event != batch->clone_event ||
        listener_stream != batch->listener_stream) return -1;
    /* Cross-stream ordering is queued, never host-synchronized.  The caller
     * records consumer_done_event after cloning/consuming the borrowed view. */
    if (cudaStreamWaitEvent(listener_stream, consumer_done_event, 0) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    batch->completed_output_generation = generation;
    batch->completed_output_row_count = batch->output_row_count;
    batch->output_lease_active = false;
    batch->clone_enqueued = false;
    batch->output_row_count = 0;
    return 0;
}

int gemma4_listener_batch_output_health(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    const gemma4_listener_batch_row *rows,
    const uint32_t *logical_q_lens,
    uint32_t row_count,
    int *out_nonfinite
) {
    if (!batch || !rows || !logical_q_lens || !out_nonfinite || batch->poisoned ||
        batch->output_lease_active || !batch->stream_bound ||
        !gemma4_batch_size_supported(row_count) || row_count > batch->capacity ||
        generation == 0 || generation != batch->output_generation ||
        generation != batch->completed_output_generation ||
        row_count != batch->completed_output_row_count) return -1;
    *out_nonfinite = 0;
    for (uint32_t i = 0; i < row_count; ++i) {
        const gemma4_listener_batch_row &expected = batch->output_rows[i];
        const gemma4_listener_batch_row &row = rows[i];
        if (row.native_context_id != expected.native_context_id ||
            row.session_generation != expected.session_generation ||
            row.state_epoch_id != expected.state_epoch_id ||
            row.frame_id != expected.frame_id ||
            row.listener_state_version != expected.listener_state_version ||
            row.sequence_position != expected.sequence_position ||
            row.cache_position != expected.cache_position ||
            row.valid_history_tokens != expected.valid_history_tokens) {
            gemma4_batch_poison(batch);
            return -2;
        }
        uint32_t slot = batch->capacity;
        for (uint32_t candidate = 0; candidate < batch->capacity; ++candidate) {
            if (batch->slots[candidate].primed &&
                batch->slots[candidate].context_id == row.native_context_id) {
                slot = candidate;
                break;
            }
        }
        const uint32_t logical_q_len = logical_q_lens[i];
        if (logical_q_len > GEMMA4_STREAM_Q_LEN || slot == batch->capacity ||
            !gemma4_batch_row_identity_matches(batch, slot, &row, false)) {
            gemma4_batch_poison(batch);
            return -3;
        }
        const gemma4_listener_batch_slot &state = batch->slots[slot];
        const bool embedded_text_input = batch->output_embedded_text_input;
        const bool preserve_listener_cursors = batch->output_preserves_listener_cursors;
        const bool audio_identity_matches =
            state.expected_frame_id == row.frame_id + 1u &&
            state.expected_state_version == row.listener_state_version + 1u &&
            state.expected_sequence_position == row.sequence_position + 1u;
        const bool preserved_identity_matches =
            state.expected_frame_id == row.frame_id &&
            state.expected_state_version == row.listener_state_version &&
            state.expected_sequence_position == row.sequence_position;
        if (row.cache_position > UINT32_MAX - logical_q_len ||
            state.current_position != row.cache_position + logical_q_len ||
            (embedded_text_input && logical_q_len == 0u) ||
            !(preserve_listener_cursors ? preserved_identity_matches : audio_identity_matches) ||
            gemma4_cache_valid_kv_len(state.cache) != state.current_position) {
            gemma4_batch_poison(batch);
            return -4;
        }
    }
    const int status = gemma4_listener_output_health(batch->model, out_nonfinite);
    batch->completed_output_generation = 0;
    batch->completed_output_row_count = 0;
    batch->output_embedded_text_input = false;
    batch->output_preserves_listener_cursors = false;
    if (status != 0) {
        gemma4_batch_poison(batch);
        return status == 1 ? 1 : -5;
    }
    return 0;
}

int gemma4_listener_batch_output_abandon_async(
    gemma4_listener_batch_ctx *batch,
    uint64_t generation,
    cudaStream_t listener_stream
) {
    if (!batch || !batch->output_lease_active ||
        generation != batch->output_generation ||
        listener_stream != batch->listener_stream) return -1;
    /* The private Python owner never exposes the raw borrowed pointer. If a
     * clone was queued, fence it; otherwise fence the completed forward. */
    const cudaEvent_t fence = batch->clone_enqueued
        ? batch->clone_event : batch->completion_event;
    if (!fence || cudaStreamWaitEvent(listener_stream, fence, 0) != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    batch->output_lease_active = false;
    batch->clone_enqueued = false;
    batch->output_row_count = 0;
    batch->completed_output_generation = 0;
    batch->completed_output_row_count = 0;
    return 0;
}

int gemma4_listener_batch_release_session_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    cudaStream_t listener_stream
) {
    if (!batch || !row || batch->poisoned || batch->output_lease_active ||
        !gemma4_batch_row_identity_matches(batch, slot, row, true) ||
        row->valid_history_tokens != row->cache_position ||
        row->frame_id != batch->slots[slot].expected_frame_id ||
        row->listener_state_version != batch->slots[slot].expected_state_version ||
        row->sequence_position != batch->slots[slot].expected_sequence_position) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    gemma4_listener_batch_slot &state = batch->slots[slot];
    if (gemma4_cache_reset(state.cache) != 0 ||
        gemma4_cache_set_device_position_async(state.cache, 0, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -7;
    }
    state.context_id = 0;
    state.session_generation = 0;
    state.state_epoch_id = 0;
    state.expected_frame_id = 0;
    state.expected_state_version = 0;
    state.expected_sequence_position = 0;
    state.prefix_len = 0;
    state.current_position = 0;
    state.primed = false;
    batch->model->last_stream = listener_stream;
    return 0;
}

int gemma4_listener_batch_rollover_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    uint32_t retained_audio_tokens,
    cudaStream_t listener_stream
) {
    if (!batch || !row || batch->poisoned || batch->output_lease_active ||
        !gemma4_batch_row_identity_matches(batch, slot, row, true) ||
        row->valid_history_tokens != row->cache_position ||
        row->frame_id != batch->slots[slot].expected_frame_id ||
        row->listener_state_version != batch->slots[slot].expected_state_version ||
        row->sequence_position != batch->slots[slot].expected_sequence_position ||
        retained_audio_tokens == 0 ||
        retained_audio_tokens > GEMMA4_LISTENER_MAX_ROLLOVER_AUDIO_TOKENS ||
        retained_audio_tokens % GEMMA4_STREAM_Q_LEN != 0 ||
        batch->slots[slot].current_position <=
            batch->slots[slot].prefix_len + retained_audio_tokens) return -1;
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    gemma4_listener_ctx *ctx = batch->model;
    gemma4_listener_batch_slot &state = batch->slots[slot];
    const uint32_t old_position = state.current_position;
    const uint32_t source_start = old_position - retained_audio_tokens;
    const int position_delta = (int)state.prefix_len - (int)source_start;
    if (gemma4_cache_rebase_async(state.cache, state.prefix_len,
            retained_audio_tokens, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -7;
    }
    for (int layer_idx = 0; layer_idx < GEMMA4_NUM_LAYERS; ++layer_idx) {
        const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
        if (spec->is_kv_shared) continue;
        const float *inv_freq = spec->attn_type == GEMMA_ATTN_FULL
            ? ctx->d_inv_freq_full : ctx->d_inv_freq_sliding;
        const int head_dim = (int)spec->head_dim;
        const void *raw_key = gemma4_cache_get_layer_k(state.cache, layer_idx);
        if (!raw_key) {
            gemma4_batch_poison(batch);
            return -8;
        }
        __nv_bfloat16 *recent_keys = const_cast<__nv_bfloat16 *>(
            static_cast<const __nv_bfloat16 *>(raw_key)) +
            (size_t)state.prefix_len * head_dim;
        kernel_rope_rebase_bf16<<<retained_audio_tokens, head_dim / 2, 0, listener_stream>>>(
            recent_keys, inv_freq, (int)retained_audio_tokens, head_dim,
            position_delta, spec->attn_scaling);
        if (cudaPeekAtLastError() != cudaSuccess) {
            gemma4_batch_poison(batch);
            return -8;
        }
    }
    state.current_position = state.prefix_len + retained_audio_tokens;
    if (gemma4_cache_set_device_position_async(
            state.cache, state.current_position, listener_stream) != 0) {
        gemma4_batch_poison(batch);
        return -8;
    }
    ctx->last_stream = listener_stream;
    return 0;
}

static bool gemma4_batch_snapshot_row_matches(
    const gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row
) {
    if (!batch || !row || batch->poisoned || batch->output_lease_active ||
        !batch->stream_bound || slot >= batch->capacity ||
        !gemma4_batch_row_identity_matches(batch, slot, row, true)) return false;
    const gemma4_listener_batch_slot &state = batch->slots[slot];
    return row->valid_history_tokens == row->cache_position &&
        row->frame_id == state.expected_frame_id &&
        row->listener_state_version == state.expected_state_version &&
        row->sequence_position == state.expected_sequence_position &&
        row->cache_position == state.current_position &&
        gemma4_cache_valid_kv_len(state.cache) == state.current_position &&
        state.current_position <= batch->model->config.max_seq_len;
}

int gemma4_listener_batch_get_session_state(
    const gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    gemma4_listener_batch_session_state *out_state
) {
    if (!out_state || out_state->struct_size != sizeof(gemma4_listener_batch_session_state) ||
        !gemma4_batch_snapshot_row_matches(batch, slot, row)) return -1;
    const gemma4_listener_batch_slot &state = batch->slots[slot];
    out_state->slot = slot;
    out_state->native_context_id = state.context_id;
    out_state->session_generation = state.session_generation;
    out_state->state_epoch_id = state.state_epoch_id;
    out_state->expected_frame_id = state.expected_frame_id;
    out_state->expected_state_version = state.expected_state_version;
    out_state->expected_sequence_position = state.expected_sequence_position;
    out_state->prefix_len = state.prefix_len;
    out_state->current_position = state.current_position;
    out_state->valid_kv_len = gemma4_cache_valid_kv_len(state.cache);
    out_state->max_seq_len = batch->model->config.max_seq_len;
    out_state->primed = state.primed ? 1u : 0u;
    out_state->reserved[0] = 0;
    out_state->reserved[1] = 0;
    out_state->reserved[2] = 0;
    return 0;
}

int gemma4_listener_batch_copy_session_kv_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t slot,
    const gemma4_listener_batch_row *row,
    uint32_t layer_idx,
    void *key_tensor,
    const gemma4_tensor_meta *key_meta,
    void *value_tensor,
    const gemma4_tensor_meta *value_meta,
    cudaStream_t listener_stream
) {
    if (!batch || !row || !key_tensor || !key_meta || !value_tensor || !value_meta ||
        !gemma4_batch_snapshot_row_matches(batch, slot, row) ||
        listener_stream != batch->listener_stream || layer_idx >= GEMMA4_NUM_LAYERS) return -1;
    const GemmaLayerSpec *spec = &gemma4_layers[layer_idx];
    if (spec->is_kv_shared || spec->head_dim == 0) return -1;
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return -10;
    const uint64_t kv_dims[] = {1, 1, batch->slots[slot].current_position, spec->head_dim};
    const uint64_t kv_strides[] = {
        (uint64_t)batch->slots[slot].current_position * spec->head_dim,
        (uint64_t)batch->slots[slot].current_position * spec->head_dim,
        spec->head_dim,
        1,
    };
    if (!validate_tensor_meta(key_meta, key_tensor, device, GEMMA4_META_BF16, 4, kv_dims) ||
        !validate_tensor_meta(value_meta, value_tensor, device, GEMMA4_META_BF16, 4, kv_dims) ||
        key_meta->strides[0] != kv_strides[0] || key_meta->strides[1] != kv_strides[1] ||
        key_meta->strides[2] != kv_strides[2] || key_meta->strides[3] != kv_strides[3] ||
        value_meta->strides[0] != kv_strides[0] || value_meta->strides[1] != kv_strides[1] ||
        value_meta->strides[2] != kv_strides[2] || value_meta->strides[3] != kv_strides[3]) return -10;
    const void *key_source = gemma4_cache_get_layer_k(batch->slots[slot].cache, (int)layer_idx);
    const void *value_source = gemma4_cache_get_layer_v(batch->slots[slot].cache, (int)layer_idx);
    if (!key_source || !value_source) return -10;
    const size_t bytes = (size_t)batch->slots[slot].current_position * spec->head_dim *
        sizeof(__nv_bfloat16);
    if (cudaMemcpyAsync(key_tensor, key_source, bytes, cudaMemcpyDeviceToDevice,
                        listener_stream) != cudaSuccess ||
        cudaMemcpyAsync(value_tensor, value_source, bytes, cudaMemcpyDeviceToDevice,
                        listener_stream) != cudaSuccess ||
        cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -2;
    }
    return 0;
}

static bool gemma4_batch_model_weights_match(
    const gemma4_listener_ctx *left,
    const gemma4_listener_ctx *right
) {
    if (!left || !right ||
        left->config.max_seq_len != right->config.max_seq_len ||
        left->config.stream_q_len != right->config.stream_q_len ||
        left->config.frame_pad_ple_ptr != right->config.frame_pad_ple_ptr ||
        left->attention_backend != right->attention_backend ||
        left->rmsnorm_backend != right->rmsnorm_backend ||
        left->weights.final_norm_w != right->weights.final_norm_w ||
        left->weights.per_layer_model_proj_w != right->weights.per_layer_model_proj_w ||
        left->weights.per_layer_proj_norm_w != right->weights.per_layer_proj_norm_w ||
        left->weights.per_layer_model_proj_scale != right->weights.per_layer_model_proj_scale ||
        left->weights.per_layer_input_scale != right->weights.per_layer_input_scale) return false;
    for (uint32_t layer = 0; layer < GEMMA4_NUM_LAYERS; ++layer) {
        if (memcmp(&left->weights.layers[layer], &right->weights.layers[layer],
                   sizeof(gemma4_layer_weights)) != 0) return false;
    }
    return true;
}

int gemma4_listener_batch_clone_session_slot_async(
    gemma4_listener_batch_ctx *batch,
    uint32_t destination_slot,
    const gemma4_listener_batch_row *destination_row,
    uint32_t source_slot,
    const gemma4_listener_batch_row *source_row,
    cudaStream_t listener_stream
) {
    if (!batch || !destination_row || !source_row || batch->poisoned ||
        batch->output_lease_active || !batch->stream_bound ||
        destination_slot >= batch->capacity || source_slot >= batch->capacity ||
        destination_slot == source_slot || !batch->slots[destination_slot].cache ||
        !batch->slots[destination_slot].prefix_attention ||
        batch->slots[destination_slot].primed ||
        destination_row->native_context_id == 0) return -1;
    if (!gemma4_batch_snapshot_row_matches(batch, source_slot, source_row)) return -2;

    const gemma4_listener_batch_slot &source_state = batch->slots[source_slot];
    gemma4_listener_batch_slot &destination_state = batch->slots[destination_slot];
    if (destination_row->native_context_id == source_row->native_context_id ||
        destination_row->session_generation != source_row->session_generation ||
        destination_row->state_epoch_id != source_row->state_epoch_id ||
        destination_row->frame_id != source_row->frame_id ||
        destination_row->listener_state_version != source_row->listener_state_version ||
        destination_row->sequence_position != source_row->sequence_position ||
        destination_row->cache_position != source_row->cache_position ||
        destination_row->valid_history_tokens != source_row->valid_history_tokens ||
        destination_row->valid_history_tokens != destination_row->cache_position ||
        source_state.prefix_len == 0 ||
        source_state.prefix_len > GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN ||
        source_state.current_position < source_state.prefix_len ||
        source_state.current_position > batch->model->config.max_seq_len ||
        gemma4_cache_valid_kv_len(destination_state.cache) != 0 ||
        gemma4_cache_valid_kv_len(source_state.cache) != source_state.current_position ||
        !source_state.prefix_attention) return -2;
    for (uint32_t i = 0; i < batch->capacity; ++i) {
        if (i != destination_slot && batch->slots[i].primed &&
            batch->slots[i].context_id == destination_row->native_context_id) return -2;
    }
    const int stream_status = gemma4_batch_bind_stream(batch, listener_stream);
    if (stream_status != 0) return stream_status;
    if (batch->model->last_stream != 0 && batch->model->last_stream != listener_stream)
        return -6;

    const int copy_status = gemma4_cache_copy_async(
        destination_state.cache, source_state.cache, source_state.current_position,
        listener_stream);
    if (copy_status != 0) {
        if (copy_status == -3 || copy_status == -4) {
            gemma4_batch_poison(batch);
            return -3;
        }
        return -2;
    }
    if (cudaMemcpyAsync(destination_state.prefix_attention, source_state.prefix_attention,
            (size_t)source_state.prefix_len * sizeof(int64_t), cudaMemcpyDeviceToDevice,
            listener_stream) != cudaSuccess || cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(batch);
        return -3;
    }

    destination_state.context_id = destination_row->native_context_id;
    destination_state.session_generation = source_state.session_generation;
    destination_state.state_epoch_id = source_state.state_epoch_id;
    destination_state.expected_frame_id = source_state.expected_frame_id;
    destination_state.expected_state_version = source_state.expected_state_version;
    destination_state.expected_sequence_position = source_state.expected_sequence_position;
    destination_state.prefix_len = source_state.prefix_len;
    destination_state.current_position = source_state.current_position;
    destination_state.primed = true;
    batch->model->last_stream = listener_stream;
    return 0;
}

int gemma4_listener_batch_copy_session_cache_async(
    gemma4_listener_batch_ctx *destination,
    uint32_t destination_slot,
    const gemma4_listener_batch_row *destination_row,
    gemma4_listener_batch_ctx *source,
    uint32_t source_slot,
    const gemma4_listener_batch_row *source_row,
    cudaStream_t listener_stream
) {
    if (!destination || !source || destination == source || !destination_row || !source_row ||
        destination->poisoned || source->poisoned || !destination->stream_bound ||
        !source->stream_bound || destination->output_lease_active || source->output_lease_active ||
        listener_stream != destination->listener_stream) return -1;
    if (destination->configured_stream_priority != -1 ||
        source->configured_stream_priority != 0) return -4;
    if (!gemma4_batch_model_weights_match(destination->model, source->model)) return -5;
    if (!gemma4_batch_snapshot_row_matches(destination, destination_slot, destination_row)) return -6;
    if (!gemma4_batch_snapshot_row_matches(source, source_slot, source_row)) return -7;

    const gemma4_listener_batch_slot &destination_state = destination->slots[destination_slot];
    gemma4_listener_batch_slot &destination_mutable = destination->slots[destination_slot];
    const gemma4_listener_batch_slot &source_state = source->slots[source_slot];
    if (destination_row->native_context_id == source_row->native_context_id ||
        destination_row->session_generation != source_row->session_generation ||
        destination_row->state_epoch_id != source_row->state_epoch_id ||
        destination_row->frame_id != source_row->frame_id ||
        destination_row->listener_state_version != source_row->listener_state_version ||
        destination_row->sequence_position != source_row->sequence_position ||
        destination_state.session_generation != source_state.session_generation ||
        destination_state.state_epoch_id != source_state.state_epoch_id ||
        destination_state.prefix_len == 0 || source_state.prefix_len == 0 ||
        source_state.prefix_len > GEMMA4_CANONICAL_LM_PREFILL_TARGET_LEN ||
        source_state.current_position < source_state.prefix_len ||
        (source_state.current_position - source_state.prefix_len) % GEMMA4_STREAM_Q_LEN != 0 ||
        source_state.current_position > destination->model->config.max_seq_len ||
        !destination_state.prefix_attention || !source_state.prefix_attention) return -2;

    const int copy_status = gemma4_cache_copy_async(
        destination_state.cache, source_state.cache, source_state.current_position,
        listener_stream);
    if (copy_status != 0 ||
        cudaMemcpyAsync(destination_state.prefix_attention, source_state.prefix_attention,
            (size_t)source_state.prefix_len * sizeof(int64_t), cudaMemcpyDeviceToDevice,
            listener_stream) != cudaSuccess || cudaPeekAtLastError() != cudaSuccess) {
        gemma4_batch_poison(destination);
        return -3;
    }
    destination_mutable.prefix_len = source_state.prefix_len;
    destination_mutable.current_position = source_state.current_position;
    destination->model->last_stream = listener_stream;
    return 0;
}

int gemma4_listener_batch_get_counters(
    const gemma4_listener_batch_ctx *batch,
    gemma4_listener_batch_counters *out_counters
) {
    if (!batch || !out_counters) return -1;
    out_counters->forward_calls = batch->forward_calls;
    out_counters->layers_executed = batch->layers_executed;
    out_counters->projection_gemm_calls = batch->projection_gemm_calls;
    out_counters->projection_rows = batch->projection_rows;
    out_counters->session_rows = batch->session_rows;
    out_counters->output_generation = batch->output_generation;
    out_counters->batch_api_version = GEMMA4_LISTENER_BATCH_API_VERSION;
    out_counters->max_batch_size = batch->capacity;
    return 0;
}

int gemma4_listener_batch_get_runtime_info(
    const gemma4_listener_batch_ctx *batch,
    gemma4_listener_batch_runtime_info *out_info
) {
    if (!batch || !out_info ||
        out_info->struct_size != sizeof(gemma4_listener_batch_runtime_info)) return -1;
    out_info->native_abi_version = GEMMA4_ABI_VERSION;
    out_info->batch_api_version = GEMMA4_LISTENER_BATCH_API_VERSION;
    out_info->max_batch_size = batch->capacity;
    out_info->supported_batch_mask = GEMMA4_BATCH_SUPPORTED_MASK_ANY_POSITIVE_N;
    out_info->attention_backend_id = batch->model->attention_backend;
    out_info->rmsnorm_backend_id = batch->model->rmsnorm_backend;
    out_info->listener_stream_priority = batch->configured_stream_priority;
    out_info->max_seq_len = batch->model->config.max_seq_len;
    out_info->stream_q_len = batch->model->config.stream_q_len;
    out_info->stream_priority_verified = batch->priority_verified ? 1u : 0u;
    return 0;
}
