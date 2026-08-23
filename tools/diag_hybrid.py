#!/usr/bin/env python3
"""diag_hybrid.py — Diagnostica: cb0 GT + livelli 1..15 predetti dal sub-talker
condizionato su h_t del modello. Se l'audio ibrido è corretto, h_t è sano e il
collo di bottiglia è solo il decoding di cb0."""
import os, sys, json

sys.path.insert(0, "/workspace")
os.environ.setdefault("PIPELINE_ENV", "local3090")

import torch
import numpy as np
import soundfile as sf
import kaggle_pipeline as KP

def main():
    duplex_model, tokenizer = KP.load_gemma4_duplex_system()
    KP.load_gemma4_duplex_checkpoint(duplex_model, "/workspace/checkpoints/task11i_cv26_v2recipe/final")
    duplex_model.eval()

    prefix = "/workspace/data/train_tokens_cv26_it_40.8h_31916s.parquet"
    user_mel, agent_mel, agent_codes, tbw = KP.load_tokens_from_parquet(prefix)
    t_agent = agent_codes.shape[1]
    seg_ends = json.load(open(prefix.replace(".parquet", "_hyperparams.json"))).get("SEGMENT_ENDS", [])
    val_start = int(0.90 * t_agent)
    for e in seg_ends:
        if e >= val_start:
            val_start = e
            break
    n_val_w = (t_agent - val_start) // 64
    val_ds = KP.FullDuplexDataset(
        user_mel[val_start * 8:], agent_mel[val_start * 8:],
        agent_codes[:, val_start:], seq_len=64,
        text_by_window=tbw[len(tbw) - n_val_w:] if tbw else None,
    )

    device = next(duplex_model.base_model.parameters()).device
    # Allinea esplicitamente i moduli trainabili al device del backbone
    duplex_model.ht_adapter.to(device)
    duplex_model.causal_conv.to(device)
    if hasattr(duplex_model, "semantic_head"):
        duplex_model.semantic_head.to(device)
    decoder = KP.Code2WavDecoder()
    from faster_whisper import WhisperModel
    asr = WhisperModel("small", device="cuda", compute_type="float16")

    os.makedirs("/workspace/samples_tf", exist_ok=True)
    fpf = KP.FEATURES_PER_AGENT_FRAME if KP.SELF_LISTENING else KP.AGENT_FEATURES_PER_FRAME
    afs = 2 + KP.TEXT_BLOCK_LEN

    with torch.inference_mode():
        for i in [1, 3]:
            item = val_ds[i]
            n_real = int((item["audio_targets"][:, 0] != KP.PAD_TARGET_ID).sum().item())
            inp = {
                "input_ids": item["input_ids"].unsqueeze(0).to(device),
                "attention_mask": item["attention_mask"].unsqueeze(0).to(device),
                "input_features": item["input_features"].unsqueeze(0).to(device),
                "input_features_mask": item["input_features_mask"].unsqueeze(0).to(device),
            }
            go = duplex_model.base_model(**inp, output_hidden_states=True, return_dict=True)
            lh = go.hidden_states[-1]
            T = item["audio_targets"].shape[0]
            ah = lh[:, afs + fpf - 1 : afs + fpf * T : fpf, :].float()
            ad = duplex_model._apply_causal_conv(ah)
            h_t = duplex_model.ht_adapter(ad)
            h_t = h_t.to(next(duplex_model.subtalker.parameters()).dtype)

            gt_codes = item["audio_targets"].unsqueeze(0).to(next(duplex_model.subtalker.parameters()).device)
            sub_logits = duplex_model.subtalker.forward_sub_talker_finetune(h_t, gt_codes)
            pred_rest = torch.argmax(sub_logits, dim=-1)
            hybrid = gt_codes.clone()
            hybrid[:, :, 1:] = pred_rest

            codes_np = hybrid[0][:n_real].cpu().numpy().astype(np.int64)
            wav = decoder.decode(torch.from_numpy(codes_np))
            out_f = f"/workspace/samples_tf/hybrid_{i:03d}.wav"
            sf.write(out_f, wav, 24000)

            # testo atteso (solo per riferimento stampato DOPO la trascrizione cieca)
            segs, _ = asr.transcribe(out_f, language="it", beam_size=5)
            blind = " ".join(s.text.strip() for s in segs)
            w_idx = val_start // 64 + i
            expected = ""
            if tbw and w_idx < len(tbw) and tbw[w_idx]:
                try:
                    expected = tokenizer.decode(tbw[w_idx])[:120]
                except Exception:
                    pass
            print(f"[TF] hybrid_{i:03d} blind_asr: {blind} | atteso: {expected}", flush=True)

if __name__ == "__main__":
    main()
