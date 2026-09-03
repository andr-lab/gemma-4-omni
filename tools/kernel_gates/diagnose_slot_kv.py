#!/usr/bin/env python3
"""Diagnostica dettagliata della divergenza tra _qwen_logits e FuturePrefixQwen slot-KV."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen, _SlotKVCache
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import CausalConv1D, DeepCausalTransformerProjectorV9
from tools.kernel_gates.p4_checks import _qwen_logits, _stack_cache


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    talker = load_talker("/workspace", device=device, load_code2wav=False)
    talker_model = talker.talker
    model = talker_model.model

    # Checkpoint
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

    # Reference A
    with torch.no_grad():
        ref_logits = _qwen_logits(talker, pfx, seq_codes, causal_prefix=True, prefix_lookahead=0)

    # Candidate B
    speaker = FuturePrefixQwen(
        talker,
        lookahead_frames=0,
        max_prefix_frames=64,
        device=device,
        dtype=dtype,
        max_audio_frames=64,
    )
    speaker.begin("Verifica causale del kernel.")
    speaker.push_v9_frame(0, pfx[:, 0, :])
    step0 = speaker.step_frame()
    run_logits_0 = step0.logits[:, :1, :]

    diff0 = (ref_logits[0, 0].float() - run_logits_0[0, 0].float()).abs()
    print("FRAME 0 COMPARISON:")
    print(f"  MAE: {diff0.mean().item():.6f}")
    print(f"  Max Diff: {diff0.max().item():.6f}")
    print(f"  Cosine: {F.cosine_similarity(ref_logits[0, 0].unsqueeze(0).float(), run_logits_0[0, 0].unsqueeze(0).float()).item():.6f}")
    print(f"  Ref Top 5: {ref_logits[0, 0].topk(5).indices.tolist()}")
    print(f"  Run Top 5: {run_logits_0[0, 0].topk(5).indices.tolist()}")

    # Now let's inspect what query at frame 0 attended to in both cases!
    # In Ref A:
    # icl (len L_icl), text (len L_txt), prefix[0] (pos L_icl+L_txt+0), bos (pos L_icl+L_txt+64)
    # Total tokens before query: L_icl + L_txt + 1 (prefix[0])
    # The prefix slots 1..63 are MASKED with -inf!
    # But in Ref A, the sequence has length L_icl + L_txt + 64 + 1 + 63 = N!
    # Rotary embeddings:
    # In Ref A, position_ids for bos is L_icl + L_txt + 64!
    # In Candidate B, what is position_ids for bos?!
    pos_b = speaker._position_ids(torch.tensor([speaker._audio_start]), 1)
    print(f"  Ref A bos position index: {talker.build_prefill_parts('Verifica causale del kernel.')[0].shape[1] + talker.build_prefill_parts('Verifica causale del kernel.')[1].shape[1] + 64}")
    print(f"  Cand B audio_start: {speaker._audio_start}")
    print(f"  Cand B pos_b: {pos_b}")

    # Check rotary embedding table and cache length:
    # Does Qwen's attention in Transformers compute attention over the entire max_cache_len (e.g. 256)?
    # In Candidate B:
    # key_states in attention has length max_cache_len = 256!
    # And attention_mask has shape (1, 1, 1, 256)!
    # positions > query have -inf!
    # What value is in -inf?
    # torch.finfo(dtype).min is -65504 in bfloat16!
    # Let's check if softmax(-65504) has numerical precision leakage when summed over 256 keys vs 130 keys!
    # In bfloat16, exp(-65504) is 0.0.
    # What about float32?
    # Let's test what happens in float32!
    print("Testing same in float32...")
    diff_fp32 = (ref_logits.float() - run_logits_0.float())

    return 0


if __name__ == "__main__":
    main()
