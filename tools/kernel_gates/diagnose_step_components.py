#!/usr/bin/env python3
"""Compare native and PyTorch components at selected full-step layers."""
from __future__ import annotations

import sys
from collections import UserDict

if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import numpy as np
import torch

from duplex_kernel.gemma_representation import AUDIO_START_AFTER_TEXT
from duplex_kernel.listener import GemmaAudioTowerBackend
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.native_listener import NativeGemmaListenerLM
from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
from transformers.cache_utils import DynamicCache
from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, create_masks_for_generate


def metrics(reference: torch.Tensor, actual: torch.Tensor) -> tuple[float, float, float]:
    diff = (reference.float() - actual.float()).abs()
    a = reference.float().flatten()
    b = actual.float().flatten()
    cosine = torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return float(diff.mean()), float(cosine), float(diff.max())


def main() -> None:
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    thinker = load_thinker("/workspace", device=device, dtype=dtype)
    lm = thinker.model.model.language_model
    backend = GemmaAudioTowerBackend(thinker, device=str(device))
    native = NativeGemmaListenerLM(lm, max_seq_len=2048, stream_q_len=4)
    shape_stable = ShapeStableGemmaListenerLM(lm, target_length=291, stream_length=4)

    input_ids, attention_mask = backend._build_multimodal_inputs((), 1024, device=device)
    prefix_ids = input_ids[:, :AUDIO_START_AFTER_TEXT]
    prefix_attention = attention_mask[:, :AUDIO_START_AFTER_TEXT].to(device=device, dtype=torch.long).contiguous()
    with torch.inference_mode():
        prefix_embeds = lm.embed_tokens(prefix_ids)
        prefix_ple = lm.get_per_layer_inputs(prefix_ids, prefix_embeds)

    # Build the exact reference prefix cache and shared-KV state.
    positions_prefix = torch.arange(AUDIO_START_AFTER_TEXT, device=device, dtype=torch.long).unsqueeze(0)
    masks_prefix = create_masks_for_generate(
        lm.config, prefix_embeds, prefix_attention, past_key_values=None, position_ids=positions_prefix
    )
    past = DynamicCache(config=lm.config)
    shared = UserDict()
    projected_prefix = shape_stable._project_per_layer_inputs(prefix_embeds, prefix_ple)
    rotary_prefix = {
        layer_type: lm.rotary_emb(prefix_embeds, positions_prefix, layer_type)
        for layer_type in lm.unique_layer_types
    }
    reference_hidden = prefix_embeds.clone()
    for layer_index, layer in enumerate(lm.layers):
        reference_hidden = shape_stable._decoder_layer(
            layer,
            reference_hidden,
            projected_prefix[:, :, layer_index, :],
            rotary_prefix[lm.config.layer_types[layer_index]],
            masks_prefix[lm.config.layer_types[layer_index]],
            positions_prefix,
            shared,
            past,
        )

    rng = np.random.default_rng(2026)
    audio = torch.tensor(0.05 * rng.standard_normal((1, 4, 1536)), device=device, dtype=dtype)
    positions = torch.arange(AUDIO_START_AFTER_TEXT, AUDIO_START_AFTER_TEXT + 4, device=device).unsqueeze(0)
    step_attention = torch.ones((1, AUDIO_START_AFTER_TEXT + 4), device=device, dtype=torch.long)
    step_attention[:, :AUDIO_START_AFTER_TEXT] = prefix_attention
    masks_step = create_masks_for_generate(
        lm.config, audio, step_attention, past_key_values=past, position_ids=positions
    )
    pad_ids = torch.zeros((1, 4), device=device, dtype=torch.long)
    frame_ple = lm.get_per_layer_inputs(pad_ids, audio)
    projected_step = shape_stable._project_per_layer_inputs(audio, frame_ple)
    rotary_step = {
        layer_type: lm.rotary_emb(audio, positions, layer_type)
        for layer_type in lm.unique_layer_types
    }

    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    capture: dict[int, dict[str, torch.Tensor]] = {}
    hidden = audio.clone()
    for layer_index, layer in enumerate(lm.layers):
        current: dict[str, torch.Tensor] = {"input": hidden.detach().clone()}
        attention = layer.self_attn
        residual = hidden
        normed = shape_stable._padded_rmsnorm(layer.input_layernorm, hidden)
        hidden_shape = (*normed.shape[:-1], -1, attention.head_dim)
        cos, sin = rotary_step[lm.config.layer_types[layer_index]]
        q = shape_stable._padded_rmsnorm(
            attention.q_norm,
            shape_stable._padded_linear(attention.q_proj, normed).view(hidden_shape),
        )
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        current["q"] = q.detach().clone().permute(0, 2, 1, 3).reshape(4, 8, attention.head_dim)

        if attention.is_kv_shared_layer:
            key_states, value_states = past.shared_layers[attention.kv_shared_layer_index]
            key_states = key_states.to(device)
            value_states = value_states.to(device)
        else:
            key_states = shape_stable._padded_rmsnorm(
                attention.k_norm,
                shape_stable._padded_linear(attention.k_proj, normed).view(hidden_shape),
            )
            value_states = shape_stable._padded_rmsnorm(
                attention.v_norm,
                shape_stable._padded_linear(attention.v_proj, normed).view(hidden_shape),
            )
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
            value_states = value_states.transpose(1, 2)
        if not attention.is_kv_shared_layer:
            key_states, value_states = past.update(key_states, value_states, attention.layer_idx)
        if attention.store_full_length_kv:
            if not hasattr(past, "shared_layers"):
                past.shared_layers = {}
            past.shared_layers[attention.layer_idx] = (key_states, value_states)
        key_length = int(key_states.shape[-2])
        key_padded = shape_stable._workspace.padded(key_states, axis=-2, slot="attention_key")
        value_padded = shape_stable._workspace.padded(value_states, axis=-2, slot="attention_value")
        mask_padded = shape_stable._workspace.padded_mask(
            masks_step[lm.config.layer_types[layer_index]], key_length=key_length
        )
        attention_output, _ = sdpa_attention_forward(
            attention,
            q,
            key_padded,
            value_padded,
            mask_padded,
            dropout=0.0,
            scaling=attention.scaling,
            sliding_window=attention.sliding_window,
            position_ids=positions,
        )
        attention_output = attention_output.reshape(1, 4, -1).contiguous()
        current["attn_out"] = attention_output.reshape(4, 8, attention.head_dim).detach().clone()
        hidden = residual + shape_stable._padded_rmsnorm(
            layer.post_attention_layernorm,
            shape_stable._padded_linear(attention.o_proj, attention_output),
        )

        residual = hidden
        ff_norm = shape_stable._padded_rmsnorm(layer.pre_feedforward_layernorm, hidden)
        gate = shape_stable._padded_linear(layer.mlp.gate_proj, ff_norm)
        up = shape_stable._padded_linear(layer.mlp.up_proj, ff_norm)
        act = layer.mlp.act_fn(gate) * up
        down = shape_stable._padded_linear(layer.mlp.down_proj, act)
        current["gate"] = gate.detach().clone().reshape(4, -1)
        current["up"] = up.detach().clone().reshape(4, -1)
        current["act"] = act.detach().clone().reshape(4, -1)
        hidden = residual + shape_stable._padded_rmsnorm(layer.post_feedforward_layernorm, down)

        residual = hidden
        ple_gate = shape_stable._padded_linear(layer.per_layer_input_gate, hidden)
        ple_act = layer.act_fn(ple_gate) * projected_step[:, :, layer_index, :]
        ple_proj = shape_stable._padded_linear(layer.per_layer_projection, ple_act)
        ple_norm = shape_stable._padded_rmsnorm(layer.post_per_layer_input_norm, ple_proj)
        current["ple_gate"] = ple_gate.detach().clone().reshape(4, 256)
        current["ple_act"] = ple_act.detach().clone().reshape(4, 256)
        current["ple_proj"] = ple_norm.detach().clone().reshape(4, 1536)
        hidden = (residual + ple_norm) * layer.layer_scalar
        current["output"] = hidden.detach().clone()
        capture[layer_index] = current

    print("=== STEP COMPONENT PARITY ===")
    for target in (14, 15, 19, 24, 34):
        native.reset()
        native.prime_async(prefix_embeds, prefix_ple, prefix_attention)
        torch.cuda.synchronize()
        native_hidden = native.step_debug(audio, num_layers=target + 1, apply_final_norm=False)
        torch.cuda.synchronize()
        ref = capture[target]
        native_values = {
            "q": native.copy_debug_tensor("q_buf", (4, 8, ref["q"].shape[-1])),
            "attn_out": native.copy_debug_tensor("attn_out", (4, 8, ref["attn_out"].shape[-1])),
            "gate": native.copy_debug_tensor("gate_buf", (4, ref["gate"].shape[-1])),
            "up": native.copy_debug_tensor("up_buf", (4, ref["up"].shape[-1])),
            "act": native.copy_debug_tensor("mlp_act_buf", (4, ref["act"].shape[-1])),
            "ple_gate": native.copy_debug_tensor("ple_gate_buf", (4, 256)),
            "ple_act": native.copy_debug_tensor("ple_act_buf", (4, 256)),
            "ple_proj": native.copy_debug_tensor("ple_proj_out", (4, 1536)),
            "hidden": native.copy_debug_tensor("hidden_a", (4, 1536)),
        }
        print(f"LAYER {target}")
        for name, actual in native_values.items():
            reference = ref["output" if name == "hidden" else name]
            mean, cosine, maximum = metrics(reference, actual)
            print(f"  {name:9s} mae={mean:.4e} cos={cosine:.6f} max={maximum:.4f}")


if __name__ == "__main__":
    main()
