#!/usr/bin/env python3
"""P4 Gemma Listener CUDA Graph / StaticCache Precision Tensor Bisect.

Investigates exact numerical equivalence between DynamicCache and StaticCache /
CUDA Graph starting from PREFILL (tokens 0-35) and Frame 0, inspecting:
1. inputs_embeds
2. position_ids
3. RoPE (cos, sin)
4. attention_mask
5. cache_position
6. K cache / V cache
7. shared_kv_states
8. attention output
9. MLP output
10. layer residual
11. final norm
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from duplex_kernel.listener import GemmaListener
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import (
    create_masks_for_generate,
    apply_rotary_pos_emb,
)


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 78)
    print("P4 GEMMA LISTENER TENSOR BISECT: DynamicCache vs StaticCache")
    print("=" * 78, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model

    backend = GemmaListener.from_thinker(th).backend
    ple_frame = backend._frame_pad_per_layer_inputs(lm)

    ss_dyn = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    cache_dyn = DynamicCache(config=lm.config)

    ss_stat = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    max_cache_len = 384
    cache_stat = StaticCache(
        config=lm.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )

    # Prefill (35 tokens)
    rng = np.random.default_rng(2026)
    stable_embeds = torch.tensor(
        0.05 * rng.standard_normal((1, 35, 1536)), device=device, dtype=dtype
    )
    pos_ids = torch.arange(35, device=device, dtype=torch.long).view(1, -1)
    att = torch.ones((1, 35), device=device, dtype=torch.long)

    m_dyn = create_masks_for_generate(
        lm.config, stable_embeds, att, past_key_values=cache_dyn, position_ids=pos_ids
    )
    m_stat = create_masks_for_generate(
        lm.config, stable_embeds, att, past_key_values=cache_stat, position_ids=pos_ids
    )
    stable_ple = lm.get_per_layer_inputs(
        torch.full((1, 35), 2, device=device, dtype=torch.long), stable_embeds
    )

    print("\n--- PREFILL INSPECTION (35 tokens) ---")
    print(f"Position IDs: {pos_ids.tolist()}")
    for k in m_dyn:
        print(f"Mask [{k}]: dyn shape={m_dyn[k].shape}, stat shape={m_stat[k].shape}")
        if isinstance(m_dyn[k], torch.Tensor) and isinstance(m_stat[k], torch.Tensor):
            val_len = m_dyn[k].shape[-1]
            diff_m = (m_dyn[k].float() - m_stat[k][..., :val_len].float()).abs().max().item()
            print(f"  valid prefix diff: {diff_m:.8e}")
            print(f"  stat tail min={m_stat[k][..., val_len:].min().item()}, max={m_stat[k][..., val_len:].max().item()}")

    # Layer-by-layer inspection during PREFILL
    with torch.inference_mode():
        h_d = stable_embeds.clone()
        h_s = stable_embeds.clone()
        ple_proj_d = ss_dyn._project_per_layer_inputs(h_d, stable_ple)
        ple_proj_s = ss_stat._project_per_layer_inputs(h_s, stable_ple)
        ple_diff = (ple_proj_d.float() - ple_proj_s.float()).abs().max().item()
        print(f"PLE projection diff: {ple_diff:.8e}")

        pe_d = {lt: lm.rotary_emb(h_d, pos_ids, lt) for lt in lm.unique_layer_types}
        pe_s = {lt: lm.rotary_emb(h_s, pos_ids, lt) for lt in lm.unique_layer_types}

        for l_idx, layer in enumerate(lm.layers):
            layer_type = layer.layer_type
            mask_d = m_dyn.get(layer_type, None)
            mask_s = m_stat.get(layer_type, None)
            pe = pe_d[layer_type]

            hn_d = ss_dyn._padded_rmsnorm(layer.input_layernorm, h_d)
            hn_s = ss_stat._padded_rmsnorm(layer.input_layernorm, h_s)
            norm_diff = (hn_d.float() - hn_s.float()).abs().max().item()

            att_out_d = ss_dyn._attention(
                layer.self_attn,
                hn_d,
                pe,
                mask_d,
                pos_ids,
                getattr(cache_dyn, "shared_layers", {}),
                cache_dyn,
            )
            att_out_s = ss_stat._attention(
                layer.self_attn,
                hn_s,
                pe,
                mask_s,
                pos_ids,
                getattr(cache_stat, "shared_layers", {}),
                cache_stat,
            )
            att_diff = (att_out_d.float() - att_out_s.float()).abs().max().item()

            if not layer.self_attn.is_kv_shared_layer:
                kd = cache_dyn.layers[layer.self_attn.layer_idx].keys
                if hasattr(cache_stat, "layers"):
                    ks = cache_stat.layers[layer.self_attn.layer_idx].keys[..., : kd.shape[-2], :]
                else:
                    ks = cache_stat.key_cache[layer.self_attn.layer_idx][..., : kd.shape[-2], :]
                k_diff = (kd.float() - ks.float()).abs().max().item()
            else:
                k_diff = 0.0

            post_norm_d = ss_dyn._padded_rmsnorm(layer.post_attention_layernorm, att_out_d + hn_d)
            post_norm_s = ss_stat._padded_rmsnorm(layer.post_attention_layernorm, att_out_s + hn_s)

            mlp_d = ss_dyn._mlp(layer.mlp, post_norm_d)
            mlp_s = ss_stat._mlp(layer.mlp, post_norm_s)
            mlp_diff = (mlp_d.float() - mlp_s.float()).abs().max().item()

            h_d = h_d + att_out_d + mlp_d
            h_s = h_s + att_out_s + mlp_s
            h_diff = (h_d.float() - h_s.float()).abs().max().item()

            print(
                f"  [Prefill Layer {l_idx:2d} ({layer_type:18s})] norm={norm_diff:.2e} | k_cache={k_diff:.2e} | att={att_diff:.2e} | mlp={mlp_diff:.2e} | resid={h_diff:.2e}"
            )
            if att_diff > 1e-4:
                print(f"    --> PREFILL ATTENTION DIVERGED AT LAYER {l_idx}!")
                print(f"    --> att_out_d min/max: ({att_out_d.min().item():.3f}, {att_out_d.max().item():.3f})")
                print(f"    --> att_out_s min/max: ({att_out_s.min().item():.3f}, {att_out_s.max().item():.3f})")
                print(f"    --> mask_d min/max: ({mask_d.min().item()}, {mask_d.max().item()})")
                print(f"    --> mask_s min/max: ({mask_s.min().item()}, {mask_s.max().item()})")
                print(f"    --> mask_d shape: {mask_d.shape}, mask_s shape: {mask_s.shape}")
                print(f"    --> Check SDPA forward call parameters...")
                break


if __name__ == "__main__":
    main()
