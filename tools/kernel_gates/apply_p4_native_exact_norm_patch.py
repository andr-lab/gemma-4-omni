#!/usr/bin/env python3
"""Apply the P4-NATIVE parity remediation to gemma4_listener.cu.

This is intentionally a deterministic source transformer rather than a loose
set of instructions.  It promotes the already-proven literal ATen RMSNorm
bridge to every Q=4 decoder RMSNorm boundary and to the frame PLE projection
normalization, while leaving cuBLAS/GELU/mul/residual math untouched.

Run from repository root, inspect the diff, build in the certification Docker
image, then run the parity gates.  The script fails closed if the expected
source patterns do not match exactly.
"""
from __future__ import annotations

from pathlib import Path

PATH = Path("native/gemma4_listener/gemma4_listener.cu")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def main() -> int:
    text = PATH.read_text(encoding="utf-8")

    # 1) The flag now selects exact ATen RMSNorm for every realtime Q=4 norm.
    text = replace_once(
        text,
        "bool use_aten_rmsnorm_bridge;      /* explicit opt-in; pre-FF norm only */",
        "bool use_aten_rmsnorm_bridge;      /* exact ATen RMSNorm for all Q=4 norm boundaries */",
        "rmsnorm flag comment",
    )

    # 2) Add one dispatcher so all call sites share exactly the same semantics.
    anchor = """static gemma4_tensor_meta make_bridge_meta(\n"""
    helper = r'''static int run_rmsnorm_exact_or_custom(
    gemma4_listener_ctx *ctx,
    const __nv_bfloat16 *input,
    const void *weight,
    __nv_bfloat16 *output,
    uint32_t rows,
    uint32_t dim,
    double eps,
    cudaStream_t stream
) {
    if (!ctx || !input || !weight || !output || rows == 0) return -1;
    if (ctx->use_aten_rmsnorm_bridge) {
        return gemma4_aten_rmsnorm_parity_async(
            input, weight, output, rows, dim, eps, stream
        );
    }
    if (dim == 1536) {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
    } else if (dim == 256) {
        kernel_rmsnorm_bf16<256><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
    } else if (dim == 512) {
        kernel_rmsnorm_bf16<512><<<rows, 256, 0, stream>>>(
            input,
            (const __nv_bfloat16 *)weight,
            output,
            (int)rows,
            (float)eps
        );
    } else {
        return -2;
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -3;
}

'''
    if text.count(anchor) != 1:
        raise RuntimeError("dispatcher insertion anchor not unique")
    text = text.replace(anchor, helper + anchor, 1)

    # Input RMSNorm.
    old = r'''    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_a,
        (const __nv_bfloat16 *)lw->input_norm_w,
        ctx->d_norm_buf,
        q_len,
        spec->input_norm_eps
    );
'''
    new = r'''    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_a, lw->input_norm_w, ctx->d_norm_buf,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->input_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "input RMSNorm")

    # Q norm — replace the dimension branch with the shared dispatcher.
    old = r'''    if (D == 256) {
        kernel_rmsnorm_bf16<256><<<GEMMA4_NUM_ATTN_HEADS * q_len, 256, 0, stream>>>(
            ctx->d_q_buf,
            (const __nv_bfloat16 *)lw->q_norm_w,
            ctx->d_q_buf,
            GEMMA4_NUM_ATTN_HEADS * q_len,
            spec->q_norm_eps
        );
    } else {
        kernel_rmsnorm_bf16<512><<<GEMMA4_NUM_ATTN_HEADS * q_len, 256, 0, stream>>>(
            ctx->d_q_buf,
            (const __nv_bfloat16 *)lw->q_norm_w,
            ctx->d_q_buf,
            GEMMA4_NUM_ATTN_HEADS * q_len,
            spec->q_norm_eps
        );
    }
'''
    new = r'''    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_q_buf, lw->q_norm_w, ctx->d_q_buf,
            (uint32_t)(GEMMA4_NUM_ATTN_HEADS * q_len), (uint32_t)D,
            (double)spec->q_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "Q RMSNorm")

    # K norm.
    old = r'''        if (D == 256) {
            kernel_rmsnorm_bf16<256><<<q_len, 256, 0, stream>>>(
                ctx->d_k_buf,
                (const __nv_bfloat16 *)lw->k_norm_w,
                ctx->d_k_buf,
                q_len,
                spec->k_norm_eps
            );
        } else {
            kernel_rmsnorm_bf16<512><<<q_len, 256, 0, stream>>>(
                ctx->d_k_buf,
                (const __nv_bfloat16 *)lw->k_norm_w,
                ctx->d_k_buf,
                q_len,
                spec->k_norm_eps
            );
        }
'''
    new = r'''        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_k_buf, lw->k_norm_w, ctx->d_k_buf,
                (uint32_t)q_len, (uint32_t)D,
                (double)spec->k_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "K RMSNorm")

    # V norm.
    old = r'''        if (D == 256) {
            kernel_rmsnorm_bf16<256><<<q_len, 256, 0, stream>>>(
                ctx->d_v_buf,
                (const __nv_bfloat16 *)lw->v_norm_w,
                ctx->d_v_buf,
                q_len,
                spec->v_norm_eps
            );
        } else {
            kernel_rmsnorm_bf16<512><<<q_len, 256, 0, stream>>>(
                ctx->d_v_buf,
                (const __nv_bfloat16 *)lw->v_norm_w,
                ctx->d_v_buf,
                q_len,
                spec->v_norm_eps
            );
        }
'''
    new = r'''        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_v_buf, lw->v_norm_w, ctx->d_v_buf,
                (uint32_t)q_len, (uint32_t)D,
                (double)spec->v_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "V RMSNorm")

    # Post-attention norm.
    old = r'''    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_b,
        (const __nv_bfloat16 *)lw->post_attn_norm_w,
        ctx->d_hidden_b,
        q_len,
        spec->post_attn_norm_eps
    );
'''
    new = r'''    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_b, lw->post_attn_norm_w, ctx->d_hidden_b,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->post_attn_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "post-attention RMSNorm")

    # Pre-FF norm: simplify the existing one-off conditional to the dispatcher.
    old = r'''    if (ctx->use_aten_rmsnorm_bridge) {
        if (gemma4_aten_rmsnorm_parity_async(
                ctx->d_hidden_a,
                lw->pre_ff_norm_w,
                ctx->d_norm_buf,
                (uint32_t)q_len,
                GEMMA4_HIDDEN_SIZE,
                (double)spec->pre_ff_norm_eps,
                stream) != 0) return -8;
    } else {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_hidden_a,
            (const __nv_bfloat16 *)lw->pre_ff_norm_w,
            ctx->d_norm_buf,
            q_len,
            spec->pre_ff_norm_eps
        );
    }
'''
    new = r'''    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_a, lw->pre_ff_norm_w, ctx->d_norm_buf,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->pre_ff_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "pre-FF RMSNorm")

    # Post-FF norm.
    old = r'''    kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
        ctx->d_hidden_b,
        (const __nv_bfloat16 *)lw->post_ff_norm_w,
        ctx->d_hidden_b,
        q_len,
        spec->post_ff_norm_eps
    );
'''
    new = r'''    if (run_rmsnorm_exact_or_custom(
            ctx, ctx->d_hidden_b, lw->post_ff_norm_w, ctx->d_hidden_b,
            (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
            (double)spec->post_ff_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "post-FF RMSNorm")

    # Per-layer PLE output norm.
    old = r'''        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_ple_proj_out,
            (const __nv_bfloat16 *)lw->ple_norm_w,
            ctx->d_ple_proj_out,
            q_len,
            spec->post_ple_norm_eps
        );
'''
    new = r'''        if (run_rmsnorm_exact_or_custom(
                ctx, ctx->d_ple_proj_out, lw->ple_norm_w, ctx->d_ple_proj_out,
                (uint32_t)q_len, GEMMA4_HIDDEN_SIZE,
                (double)spec->post_ple_norm_eps, stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "PLE RMSNorm")

    # Frame PLE global projection normalization.  The canonical ordering is:
    # BF16 linear -> BF16 projection_scale -> literal RMSNorm(dim=256)
    # -> BF16 token identity add -> BF16 input_scale.
    old = r'''        const __nv_bfloat16 *token_id = (const __nv_bfloat16 *)ctx->config.frame_pad_ple_ptr;
        kernel_ple_project_and_norm_bf16<<<q_len * GEMMA4_NUM_LAYERS, 256, 0, stream>>>(
            ctx->d_ple_raw_proj,
            (const __nv_bfloat16 *)ctx->weights.per_layer_proj_norm_w,
            token_id,
            ctx->d_projected_ple,
            q_len * GEMMA4_NUM_LAYERS,
            ctx->weights.per_layer_model_proj_scale,
            1e-6f,
            ctx->weights.per_layer_input_scale
        );
        const int frame_ple_elements = q_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
'''
    new = r'''        const __nv_bfloat16 *token_id = (const __nv_bfloat16 *)ctx->config.frame_pad_ple_ptr;
        const int frame_ple_elements = q_len * GEMMA4_NUM_LAYERS * GEMMA4_PLE_DIM;
        kernel_scale_bf16<<<(frame_ple_elements + 255) / 256, 256, 0, stream>>>(
            ctx->d_ple_raw_proj,
            ctx->d_projected_ple,
            frame_ple_elements,
            ctx->weights.per_layer_model_proj_scale
        );
        if (run_rmsnorm_exact_or_custom(
                ctx,
                ctx->d_projected_ple,
                ctx->weights.per_layer_proj_norm_w,
                ctx->d_projected_ple,
                (uint32_t)(q_len * GEMMA4_NUM_LAYERS),
                GEMMA4_PLE_DIM,
                1e-6,
                stream) != 0) return -8;
'''
    text = replace_once(text, old, new, "frame projected-PLE RMSNorm")

    # Final Q=4 norm.  There are two final-norm call sites: native prefix and
    # streaming step.  Replace only the streaming one whose row count is q_len.
    old = r'''    if (apply_final_norm) {
        kernel_rmsnorm_bf16<GEMMA4_HIDDEN_SIZE><<<q_len, 256, 0, stream>>>(
            ctx->d_hidden_a,
            (const __nv_bfloat16 *)ctx->weights.final_norm_w,
            ctx->d_output_tensor,
            q_len,
            GEMMA4_FINAL_NORM_EPS
        );
    } else {
'''
    new = r'''    if (apply_final_norm) {
        if (run_rmsnorm_exact_or_custom(
                ctx,
                ctx->d_hidden_a,
                ctx->weights.final_norm_w,
                ctx->d_output_tensor,
                (uint32_t)q_len,
                GEMMA4_HIDDEN_SIZE,
                GEMMA4_FINAL_NORM_EPS,
                stream) != 0) return -8;
    } else {
'''
    text = replace_once(text, old, new, "stream final RMSNorm")

    PATH.write_text(text, encoding="utf-8")
    print(f"patched {PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
