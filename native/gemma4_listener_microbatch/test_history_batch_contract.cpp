#include <cassert>
#include <cstdint>
#include "gemma4_listener_batch_contract.h"

int main() {
    assert(GEMMA4_BATCH_TRACE_STAGE_COUNT == 33);
    assert(gemma4_batch_trace_rows_valid(2));
    assert(!gemma4_batch_trace_rows_valid(1));
    assert(!gemma4_batch_trace_rows_valid(4));
    uint32_t captured_rows = 0;
    assert(gemma4_batch_trace_mark_row(&captured_rows, 0));
    assert(!gemma4_batch_trace_rows_complete(captured_rows));
    assert(!gemma4_batch_trace_mark_row(&captured_rows, 0));
    assert(gemma4_batch_trace_mark_row(&captured_rows, 1));
    assert(gemma4_batch_trace_rows_complete(captured_rows));
    assert(!gemma4_batch_trace_mark_row(&captured_rows, 2));
    uint32_t trace_actual_rows = 0, trace_reference_rows = 0, trace_dim = 0;
    assert(gemma4_batch_trace_reference_layout(GEMMA4_BATCH_TRACE_L0_Q_NORM, 2,
        &trace_actual_rows, &trace_reference_rows, &trace_dim));
    assert(trace_actual_rows == 32 && trace_reference_rows == 16 && trace_dim == 256);
    assert(gemma4_batch_trace_reference_layout(GEMMA4_BATCH_TRACE_AUDIO_PLE_RMS, 2,
        &trace_actual_rows, &trace_reference_rows, &trace_dim));
    assert(trace_actual_rows == 140 && trace_reference_rows == 70 && trace_dim == 256);
    assert(gemma4_batch_trace_reference_layout(GEMMA4_BATCH_TRACE_L0_INPUT_NORM, 2,
        &trace_actual_rows, &trace_reference_rows, &trace_dim));
    assert(trace_actual_rows == 4 && trace_reference_rows == 2 && trace_dim == 1536);
    assert(!gemma4_batch_trace_reference_layout(GEMMA4_BATCH_TRACE_L0_INPUT_NORM, 4,
        &trace_actual_rows, &trace_reference_rows, &trace_dim));
    assert(gemma4_batch_trace_storage_allowed(1));
    assert(gemma4_batch_trace_storage_allowed(GEMMA4_BATCH_TRACE_MAX_BYTES));
    assert(!gemma4_batch_trace_storage_allowed(0));
    assert(!gemma4_batch_trace_storage_allowed((size_t)GEMMA4_BATCH_TRACE_MAX_BYTES + 1u));
    const uint32_t ple_width = 35u * 256u;
    assert(gemma4_batch_trace_frame_pad_input_valid(2, ple_width, 2u * ple_width, 8, ple_width));
    assert(!gemma4_batch_trace_frame_pad_input_valid(1, ple_width, ple_width, 8, ple_width));
    assert(!gemma4_batch_trace_frame_pad_input_valid(2, ple_width, ple_width, 8, ple_width));
    assert(!gemma4_batch_trace_frame_pad_input_valid(2, ple_width, 2u * ple_width, 0, ple_width));
    const uint16_t pad_rows[2] = {11, 23};
    const uint16_t expected_pad_rows[8] = {11, 23, 11, 23, 11, 23, 11, 23};
    for (uint32_t token = 0; token < 8; ++token) {
        assert(pad_rows[gemma4_batch_trace_frame_pad_source_row(token)] == expected_pad_rows[token]);
    }

    // The numerical diagnostic is a strict N=2, native-exact, one-shot
    // state machine.  It cannot be armed for routine production calls, and
    // q_proj comparison is reachable only after an exact RMS result.
    assert(!gemma4_batch_probe_can_arm(GEMMA4_BATCH_PROBE_IDLE, 1, 2));
    assert(gemma4_batch_probe_can_arm(GEMMA4_BATCH_PROBE_IDLE, 2, 2));
    assert(gemma4_batch_probe_can_arm(GEMMA4_BATCH_PROBE_IDLE, 4, 2));
    assert(!gemma4_batch_probe_can_arm(GEMMA4_BATCH_PROBE_IDLE, 2, 0));
    assert(!gemma4_batch_probe_can_arm(GEMMA4_BATCH_PROBE_ARMED, 2, 2));
    assert(gemma4_batch_probe_rows_valid(2));
    assert(!gemma4_batch_probe_rows_valid(1));
    assert(!gemma4_batch_probe_rows_valid(4));
    gemma4_batch_probe_state probe = GEMMA4_BATCH_PROBE_ARMED;
    assert(!gemma4_batch_probe_can_run_qproj(probe));
    assert(!gemma4_batch_probe_mark_rms_captured(&probe, 2, 1536));
    assert(probe == GEMMA4_BATCH_PROBE_ARMED);
    assert(gemma4_batch_probe_mark_rms_captured(&probe, 4, 1536));
    assert(probe == GEMMA4_BATCH_PROBE_RMS_CAPTURED);
    assert(gemma4_batch_probe_mark_rms_result(&probe, true));
    assert(probe == GEMMA4_BATCH_PROBE_RMS_EXACT);
    assert(gemma4_batch_probe_can_run_qproj(probe));
    assert(gemma4_batch_probe_mark_qproj_captured(&probe));
    assert(gemma4_batch_probe_mark_done(&probe));
    assert(probe == GEMMA4_BATCH_PROBE_DONE);
    assert(gemma4_batch_probe_elements(4, 1536) == 4u * 1536u);
    if (SIZE_MAX > UINT32_MAX) {
        assert(gemma4_batch_probe_elements(UINT32_MAX, UINT32_MAX) ==
               (size_t)UINT32_MAX * (size_t)UINT32_MAX);
    } else {
        assert(gemma4_batch_probe_elements(UINT32_MAX, UINT32_MAX) == 0);
    }
    uint32_t block_x = 0, block_y = 0;
    assert(gemma4_batch_probe_rms_launch_shape(4, 1536, &block_x, &block_y));
    assert(block_x == 64 && block_y == 8);
    assert(gemma4_batch_probe_rms_launch_shape(2, 1536, &block_x, &block_y));
    assert(block_x == 128 && block_y == 4);
    assert(!gemma4_batch_probe_rms_launch_shape(4, 512, &block_x, &block_y));
    assert(!gemma4_batch_probe_rms_launch_shape(16, 1536, &block_x, &block_y));

    gemma4_batch_probe_state red_probe = GEMMA4_BATCH_PROBE_ARMED;
    assert(gemma4_batch_probe_mark_rms_captured(&red_probe, 4, 1536));
    assert(gemma4_batch_probe_mark_rms_result(&red_probe, false));
    assert(red_probe == GEMMA4_BATCH_PROBE_RMS_DIFFERENT);
    assert(!gemma4_batch_probe_can_run_qproj(red_probe));

    gemma4_history_batch_plan plan{};
    assert(gemma4_history_batch_make_plan(265, 265, 265, 256, 4096, 291, 256, 1024, &plan));
    assert(plan.compute_tokens == 291);
    assert(plan.final_position == 521);
    assert(plan.maximum_visible_keys == 521);
    assert(gemma4_history_batch_causal_key_limit(&plan, 0) == 266);
    assert(gemma4_history_batch_causal_key_limit(&plan, 3) == 269);
    assert(gemma4_history_batch_causal_key_limit(&plan, 255) == 521);
    assert(gemma4_history_batch_causal_key_limit(&plan, 256) == 0);
    assert(gemma4_history_batch_sliding_start(&plan, 0, 128) == 138);
    assert(gemma4_history_batch_sliding_start(&plan, 255, 128) == 393);
    assert(gemma4_history_batch_ple_source_row(0) == 0);
    assert(gemma4_history_batch_ple_source_row(3) == 3);
    assert(gemma4_history_batch_ple_source_row(4) == 0);
    assert(gemma4_history_batch_ple_source_row(255) == 3);

    // Prefix/cache cursor candidate advances only after successful health.
    uint32_t committed_position = 0;
    assert(gemma4_history_batch_commit_candidate(&plan, 265, false, &committed_position));
    assert(committed_position == 265);
    assert(gemma4_history_batch_commit_candidate(&plan, 265, true, &committed_position));
    assert(committed_position == 521);
    assert(gemma4_history_batch_make_plan(91, 91, 91, 4, 4096, 291, 256, 1024, &plan));
    assert(plan.final_position == 95);
    assert(!gemma4_history_batch_make_plan(91, 92, 91, 4, 4096, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(91, 91, 90, 4, 4096, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(91, 91, 91, 0, 4096, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(91, 91, 91, 6, 4096, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(91, 91, 91, 260, 4096, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(91, 91, 91, 256, 340, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(291, 291, 291, 256, 546, 291, 256, 1024, &plan));
    assert(!gemma4_history_batch_make_plan(291, 291, 291, 256, 4096, 291, 256, 512, &plan));
    return 0;
}
