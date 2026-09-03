#!/usr/bin/env python3
"""Test numerico di equivalenza tra:
Method A (baseline 2-pass): _cache_prefix_through(t) + _step_frame_slot_kv(t) [2 forward pass 28-layer]
Method B (joint 1-pass): un singolo forward pass 2-token con positions=[p_pfx, p_aud] e mask causale [1 forward pass 28-layer]
"""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from duplex_kernel.future_prefix import FuturePrefixQwen, _aggregate_committed_codes
from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.v9 import load_v9_conditioning

device = "cuda"
dtype = torch.float32  # test in float32 per isolare la matematica dalla precisione

print("[TEST JOINT] Caricamento Talker...", flush=True)
talker = load_talker("/workspace", device=device, load_code2wav=False)
talker.talker.to(dtype=dtype)
projector, causal_conv = load_v9_conditioning(device=device, dtype=dtype)

model = getattr(getattr(talker.talker, "talker", talker.talker), "model", talker.talker)
codec_head = getattr(talker.talker, "codec_head", None) or talker.talker.model.codec_head

turn_len = 16
speaker_a = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=turn_len, device=device, dtype=dtype, max_audio_frames=turn_len)
speaker_b = FuturePrefixQwen(talker, lookahead_frames=0, max_prefix_frames=turn_len, device=device, dtype=dtype, max_audio_frames=turn_len)

speaker_a.begin("Verifica causale del kernel.")
speaker_b.begin("Verifica causale del kernel.")

dummy_pfx = torch.randn((turn_len, 2048), device=device, dtype=dtype)
dummy_codes = torch.randint(0, 2048, (16,), device=device, dtype=torch.long)

max_diffs = []
top1_matches = []

for t in range(turn_len):
    h_pfx = dummy_pfx[t:t+1]
    
    # ----------------------------------------------------
    # Method A: Standard 2-pass
    # ----------------------------------------------------
    speaker_a.push_v9_frame(t, h_pfx)
    out_a = speaker_a.step_frame()
    speaker_a.commit_codes(dummy_codes)

    # ----------------------------------------------------
    # Method B: Joint 1-pass
    # ----------------------------------------------------
    speaker_b.push_v9_frame(t, h_pfx)
    frame_id = t
    pos_pfx = speaker_b._prefix_start + frame_id
    pos_aud = speaker_b._audio_start + frame_id
    
    # Embedding: Query 0 = h_pfx, Query 1 = audio_emb
    if frame_id == 0:
        aud_emb = speaker_b._bos
    else:
        aud_emb = _aggregate_committed_codes(
            getattr(talker.talker, "talker", talker.talker),
            speaker_b._committed_codes[-1:],
            device=device,
            dtype=dtype,
        )
    joint_embeds = torch.cat([h_pfx.unsqueeze(0), aud_emb], dim=1) # [1, 2, 2048]
    joint_positions = torch.tensor([pos_pfx, pos_aud], device=device, dtype=torch.long)
    
    # Mask per joint queries [1, 1, 2, max_cache_len]
    joint_mask = speaker_b._slot_mask(joint_positions, visible_prefix_end=frame_id)
    # Query 0 (prefix) non deve vedere Query 1 (audio)
    joint_mask[0, 0, 0, pos_aud] = torch.finfo(dtype).min

    with torch.inference_mode():
        joint_out = model(
            inputs_embeds=joint_embeds,
            attention_mask=joint_mask,
            position_ids=speaker_b._position_ids(joint_positions, 1),
            cache_position=joint_positions,
            past_key_values=speaker_b._slot_cache,
            use_cache=True,
        )
        hidden_b = joint_out.last_hidden_state[:, 1:, :] # prendi token audio
        logits_b = codec_head(hidden_b)
    
    speaker_b.commit_codes(dummy_codes)
    speaker_b._current_frame_id += 1
    speaker_b._cached_prefix_frames = frame_id + 1

    diff = (out_a.logits - logits_b).abs().max().item()
    top1_a = out_a.logits.argmax(-1).item()
    top1_b = logits_b.argmax(-1).item()
    top1_match = (top1_a == top1_b)

    max_diffs.append(diff)
    top1_matches.append(top1_match)
    print(f"Frame {t:02d}: Max Logit Diff = {diff:.6e}, Top-1 match = {top1_match}")

print("\n" + "=" * 60)
print(f"Overall Max Diff: {max(max_diffs):.6e}")
print(f"Top-1 Agreement: {sum(top1_matches)}/{len(top1_matches)} ({sum(top1_matches)/len(top1_matches)*100:.2f}%)")
if max(max_diffs) < 1e-4 and all(top1_matches):
    print("SUCCESS: Joint 1-pass e' 100% matematicamente equivalente al 2-pass!")
else:
    print("FAIL: discrepanza numerica rilevata!")
