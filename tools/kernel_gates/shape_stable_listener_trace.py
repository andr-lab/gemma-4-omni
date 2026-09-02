#!/usr/bin/env python3
"""One-prefix diagnostic: functional listener versus the prior isolated patch."""
from __future__ import annotations

import os
import sys
from collections import UserDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PIPELINE_ENV", "local3090")

from duplex_kernel.gemma_representation import _audio_soft_tokens
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from tools.kernel_gates.p4_candidate_validation import _validation_dataset
from tools.kernel_gates.p4_precision_ablations import _candidate


def stats(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    delta = (a.float() - b.float()).abs()
    cosine = torch.nn.functional.cosine_similarity(a.float().flatten()[None], b.float().flatten()[None]).item()
    return float(delta.mean()), float(cosine)


def main() -> None:
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    ds, _, _ = _validation_dataset(Path("/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"))
    thinker = load_thinker("/workspace", device="cuda", dtype=torch.bfloat16)
    model = thinker.model.model
    lm = model.language_model
    item = ds[0]
    batch = {key: item[key].unsqueeze(0).cuda() for key in ("input_ids", "attention_mask", "input_features", "input_features_mask")}
    audio_pos = (batch["input_ids"][0] == int(model.config.audio_token_id)).nonzero().flatten().tolist()
    audio_start = int(audio_pos[0])
    audio = _audio_soft_tokens(model, input_features=batch["input_features"], input_features_mask=batch["input_features_mask"])
    ids = batch["input_ids"][:, :audio_start]
    attn = batch["attention_mask"][:, :audio_start].long()
    embeds = lm.embed_tokens(ids)
    ple = lm.get_per_layer_inputs(ids, embeds)
    positions = torch.arange(audio_start, device="cuda", dtype=torch.long).view(1, -1)
    masks = create_masks_for_generate(model.config, embeds, attn, past_key_values=None, position_ids=positions)

    functional_trace: dict[int, torch.Tensor] = {}
    runtime = ShapeStableGemmaListenerLM(lm, target_length=291)
    functional_ple = runtime._project_per_layer_inputs(embeds, ple)
    functional_norm = runtime._padded_rmsnorm(lm.layers[0].input_layernorm, embeds)
    functional_q = runtime._padded_linear(lm.layers[0].self_attn.q_proj, functional_norm)
    from transformers.cache_utils import DynamicCache
    layer0 = lm.layers[0]
    layer0_positions = lm.rotary_emb(embeds, positions, lm.config.layer_types[0])
    functional_attention = runtime._attention(
        layer0.self_attn,
        functional_norm,
        layer0_positions,
        masks[lm.config.layer_types[0]],
        positions,
        UserDict(),
        DynamicCache(config=lm.config),
    )
    functional = runtime.forward(
        inputs_embeds=embeds,
        per_layer_inputs=ple,
        attention_mask=masks,
        position_ids=positions,
        past_key_values=None,
        shared_kv_states=UserDict(),
        trace=functional_trace,
    ).last_hidden_state

    patched_trace: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(lambda _m, _a, output, index=index: patched_trace.__setitem__(index, output.detach()))
        for index, layer in enumerate(lm.layers)
    ]
    lm._p4_padding_length = 291
    try:
        with _candidate(lm, "shape_padded_all_ops_kv"):
            patched_ple = lm.project_per_layer_inputs(embeds, ple)
            patched_norm = lm.layers[0].input_layernorm(embeds)
            patched_q = lm.layers[0].self_attn.q_proj(patched_norm)
            print("ple", stats(functional_ple, patched_ple))
            print("input_norm", stats(functional_norm, patched_norm))
            print("qproj", stats(functional_q, patched_q))
            patched_attention, _ = layer0.self_attn(
                hidden_states=patched_norm,
                position_embeddings=layer0_positions,
                attention_mask=masks[lm.config.layer_types[0]],
                position_ids=positions,
                past_key_values=DynamicCache(config=lm.config),
            )
            print("attention", stats(functional_attention, patched_attention))
            patched = lm(
                inputs_embeds=embeds,
                per_layer_inputs=ple,
                attention_mask=masks,
                position_ids=positions,
                past_key_values=None,
                use_cache=True,
                return_dict=True,
            ).last_hidden_state
    finally:
        del lm._p4_padding_length
        for handle in handles:
            handle.remove()
    print("final", stats(functional, patched))
    for index in range(len(lm.layers)):
        print(index, stats(functional_trace[index], patched_trace[index]))


if __name__ == "__main__":
    main()
