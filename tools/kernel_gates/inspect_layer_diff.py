#!/usr/bin/env python3
"""Confronta layer per layer tra _qwen_logits e FuturePrefixQwen per frame 0."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import CausalConv1D, DeepCausalTransformerProjectorV9
from tools.kernel_gates.p4_checks import _qwen_logits, _stack_cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    talker_dtype = torch.float32

    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker_model = talker.talker
    talker_model.to(dtype=dtype)
    model = talker_model.model
    model.to(dtype=dtype)

    sd = torch.load("/workspace/checkpoints/v9_causal_l0_unknown.pt", map_location="cpu", weights_only=True)
    proj = DeepCausalTransformerProjectorV9().to(device=device, dtype=dtype).eval()
    proj.load_state_dict(sd.get("proj", sd), strict=False)

    causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3).to(device=device, dtype=dtype).eval()
    if "causal_conv" in sd:
        causal_conv.conv.load_state_dict(sd["causal_conv"])
    else:
        hsd = torch.load("/workspace/_archive_ckpts/gemma4_duplex_sota/custom_duplex_heads.pt", map_location="cpu", weights_only=True)
        causal_conv.conv.load_state_dict(hsd["causal_conv"], strict=False)

    raw, anchors, codes, _ = _stack_cache(Path("/workspace/cache_ars_runtime_aligned_v1.pt"))
    seq_raw = raw[0:1].to(device=device, dtype=dtype)
    seq_codes = codes[0:1].to(device=device)
    if seq_codes.dim() == 2:
        seq_codes = seq_codes.unsqueeze(-1)
    if seq_codes.shape[-1] == 1:
        pad_cbs = torch.zeros((*seq_codes.shape[:-1], 15), dtype=seq_codes.dtype, device=seq_codes.device)
        seq_codes = torch.cat((seq_codes, pad_cbs), dim=-1)

    with torch.no_grad():
        h_tx = seq_raw[:, 0, :]
        c = causal_conv(seq_raw)
        pfx = proj(c, h_tx)

    # In Ref A, construct inputs and masks exactly
    batch_size = 1
    prefix_frames = pfx.shape[1]
    icl, text, bos, _ = talker.build_prefill_parts("Verifica causale del kernel.")
    embeddings = torch.cat((icl, text, pfx, bos), dim=1)
    prefix_start = icl.shape[1] + text.shape[1]
    prefix_end = prefix_start + prefix_frames
    audio_inputs = torch.cat([talker_model.get_input_embeddings()(seq_codes[:, :-1, :1])], dim=-2).sum(dim=-2)
    # wait, aggregate_code_embeddings:
    from tools.kernel_gates.p4_checks import _aggregate_code_embeddings
    audio_inputs = _aggregate_code_embeddings(talker_model, seq_codes[:, :-1].to(device=pfx.device))
    embeddings = torch.cat((embeddings, audio_inputs), dim=1)
    length = embeddings.shape[1]
    mask_value = torch.finfo(embeddings.dtype).min
    attention_mask = torch.zeros((batch_size, 1, length, length), device=pfx.device, dtype=embeddings.dtype)
    causal = torch.triu(torch.ones((length, length), device=pfx.device, dtype=torch.bool), diagonal=1)
    attention_mask[:, 0].masked_fill_(causal, mask_value)
    for audio_index in range(prefix_frames):
        query = prefix_end + audio_index
        visible_until = min(prefix_frames - 1, audio_index)
        attention_mask[:, 0, query, prefix_start + visible_until + 1:prefix_end] = mask_value
    position_ids = torch.arange(length, device=pfx.device, dtype=torch.long).view(1, 1, length).expand(3, batch_size, length)

    # Hook layers in model
    ref_layer_outs = {}
    hooks = []
    for idx, layer in enumerate(model.layers):
        def make_hook(i):
            def hook_fn(m, inp, out):
                # out[0] is hidden
                ref_layer_outs[i] = out[0].detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    with torch.inference_mode():
        model(inputs_embeds=embeddings, attention_mask=attention_mask, position_ids=position_ids, use_cache=False)

    for h in hooks: h.remove()

    # Now run Candidate B and capture layer outputs for query BOS
    run_layer_outs = {}
    hooks = []
    for idx, layer in enumerate(model.layers):
        def make_hook(i):
            def hook_fn(m, inp, out):
                run_layer_outs[i] = out[0].detach()
            return hook_fn
        hooks.append(layer.register_forward_hook(make_hook(idx)))

    speaker = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=64, device=device, dtype=dtype, max_audio_frames=64)
    speaker.begin("Verifica causale del kernel.")
    speaker.push_v9_frame(0, pfx[:, 0, :])
    step0 = speaker.step_frame()

    for h in hooks: h.remove()

    print("=== LAYER-BY-LAYER MAE AT POSITION BOS (query=80) ===")
    for idx in range(len(model.layers)):
        ref_h = ref_layer_outs[idx][:, 80, :].float()
        run_h = run_layer_outs[idx][:, 0, :].float()
        mae = (ref_h - run_h).abs().mean().item()
        cos = F.cosine_similarity(ref_h, run_h).item()
        max_d = (ref_h - run_h).abs().max().item()
        print(f"  Layer {idx:02d}: MAE={mae:.6f}, MaxDiff={max_d:.6f}, Cosine={cos:.7f}")


if __name__ == "__main__":
    main()
