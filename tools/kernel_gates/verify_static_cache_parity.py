#!/usr/bin/env python3
"""Verify numerical equivalence between DynamicCache and StaticCache on Gemma Listener."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.listener import GemmaListener, GemmaListenerState
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers import StaticCache, DynamicCache
from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate


def main():
    device = "cuda:0"
    dtype = torch.bfloat16
    print("=" * 70)
    print("[PARITY AUDIT] DynamicCache vs StaticCache in ShapeStableGemmaListenerLM")
    print("=" * 70, flush=True)

    th = load_thinker("/workspace", device=device, dtype=dtype)
    lm = th.model.model.language_model
    backend = GemmaListener.from_thinker(th).backend

    # 1. Initialize DynamicCache path (baseline)
    ss_dyn = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    cache_dyn = DynamicCache(config=lm.config)

    # 2. Initialize StaticCache path
    ss_stat = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)
    cache_stat = StaticCache(
        config=lm.config,
        max_batch_size=1,
        max_cache_len=384,
        device=device,
        dtype=dtype,
    )

    # 3. Run Prefill (35 text tokens) on both
    text_ids = tuple(backend.text_token_ids)
    input_ids, attention_mask = backend._build_multimodal_inputs(text_ids, 1024, device=device)
    stable_ids = input_ids[:, :35]
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
    pos_ids = torch.arange(35, device=device, dtype=torch.long).view(1, -1)
    stable_att = attention_mask[:, :35].to(device=device, dtype=torch.long)

    mask_dyn = create_masks_for_generate(lm.config, stable_embeds, stable_att, past_key_values=cache_dyn, position_ids=pos_ids)
    mask_stat = create_masks_for_generate(lm.config, stable_embeds, stable_att, past_key_values=cache_stat, position_ids=pos_ids)

    with torch.inference_mode():
        out_pre_dyn = ss_dyn.forward(
            inputs_embeds=stable_embeds,
            per_layer_inputs=stable_ple,
            attention_mask=mask_dyn,
            position_ids=pos_ids,
            past_key_values=cache_dyn,
            shared_kv_states=None,
        )
        out_pre_stat = ss_stat.forward(
            inputs_embeds=stable_embeds,
            per_layer_inputs=stable_ple,
            attention_mask=mask_stat,
            position_ids=pos_ids,
            past_key_values=cache_stat,
            shared_kv_states=None,
        )

    diff_pre = (out_pre_dyn.last_hidden_state.float() - out_pre_stat.last_hidden_state.float()).abs().max().item()
    print(f"Prefill (35 tokens) max diff: {diff_pre:.8e}")

    # 4. Run 30 streaming frames (4 tokens each)
    print("\nRunning 30 streaming frames comparing DynamicCache vs StaticCache...")
    ple_frame = backend._frame_pad_per_layer_inputs(lm)
    rng = np.random.default_rng(2026)

    maes = []
    max_diffs = []
    cosines = []

    for f in range(30):
        # 4 random audio soft tokens
        frame_embeds = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
        start_pos = 35 + f * 4
        cur_positions = torch.arange(start_pos, start_pos + 4, device=device, dtype=torch.long).view(1, -1)
        cur_att = torch.ones((1, start_pos + 4), device=device, dtype=torch.long)

        cur_mask_dyn = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_dyn, position_ids=cur_positions)
        cur_mask_stat = create_masks_for_generate(lm.config, frame_embeds, cur_att, past_key_values=cache_stat, position_ids=cur_positions)

        with torch.inference_mode():
            out_d = ss_dyn.forward(
                inputs_embeds=frame_embeds,
                per_layer_inputs=ple_frame,
                attention_mask=cur_mask_dyn,
                position_ids=cur_positions,
                past_key_values=cache_dyn,
                shared_kv_states=getattr(out_pre_dyn.past_key_values, "shared_layers", None),
            )
            out_s = ss_stat.forward(
                inputs_embeds=frame_embeds,
                per_layer_inputs=ple_frame,
                attention_mask=cur_mask_stat,
                position_ids=cur_positions,
                past_key_values=cache_stat,
                shared_kv_states=getattr(out_pre_stat.past_key_values, "shared_layers", None),
            )

        hd = out_d.last_hidden_state[:, 0, :].float()
        hs = out_s.last_hidden_state[:, 0, :].float()
        diff = (hd - hs).abs()
        mae = diff.mean().item()
        m_diff = diff.max().item()
        cos = F.cosine_similarity(hd.view(1, -1), hs.view(1, -1)).item()

        maes.append(mae)
        max_diffs.append(m_diff)
        cosines.append(cos)

        if mae > 1e-6 or f < 5 or f % 10 == 0:
            print(f"  Frame {f:2d}: MAE={mae:.8e} | MaxDiff={m_diff:.8e} | Cosine={cos:.8f}")

    print(f"\n[SUMMARY over 30 frames]:")
    print(f"  Mean MAE:   {np.mean(maes):.8e}")
    print(f"  Max Diff:   {np.max(max_diffs):.8e}")
    print(f"  Min Cosine: {np.min(cosines):.8f}")


if __name__ == "__main__":
    main()
