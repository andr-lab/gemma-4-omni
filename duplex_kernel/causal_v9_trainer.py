# -*- coding: utf-8 -*-
"""duplex_kernel/causal_v9_trainer.py — Causal V9 Training Engine through Frozen Qwen (P4D-R).

Questo modulo implementa il training differenziabile di V9/projector attraverso
il modello Qwen3-TTS congelato con visibilita' causale rigorosa L=0.

Proprieta' verificate:
1. Qwen Talker 1.7B e' 100% frozen (requires_grad=False).
2. I gradienti fluiscono da L_primary (CrossEntropy su CB0) attraverso Qwen in h_pfx e nei pesi V9.
3. La maschera 4D impone L=0: l'audio query al frame t puo' vedere unicamente
   il condizionamento V9 [0..t] e la storia audio [0..t-1].
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from duplex_kernel.models.v9 import DeepCausalTransformerProjectorV9, CausalConv1D


def aggregate_code_embeddings(talker_model: Any, codes: torch.Tensor) -> torch.Tensor:
    """Somma gli embedding CB0..CB15 per i token audio, esattamente come nella ricetta Qwen."""
    if codes.dim() == 2:
        codes = codes.unsqueeze(-1)
    parts = [talker_model.get_input_embeddings()(codes[..., :1])]
    code_predictor = getattr(talker_model, "code_predictor", None)
    if code_predictor is not None:
        sub_embs = code_predictor.get_input_embeddings()
        max_cbs = min(codes.shape[-1], len(sub_embs) + 1, 16)
        for codebook in range(1, max_cbs):
            parts.append(sub_embs[codebook - 1](codes[..., codebook:codebook + 1]))
    return torch.cat(parts, dim=-2).sum(dim=-2)


def build_differentiable_qwen_mask(
    batch_size: int,
    prefix_start: int,
    prefix_frames: int,
    total_length: int,
    *,
    lookahead_frames: int = 0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Costruisce la maschera 4D causale che impone il lookahead L per ciascun frame audio."""
    mask_value = torch.finfo(dtype).min
    prefix_end = prefix_start + prefix_frames
    
    mask = torch.zeros((batch_size, 1, total_length, total_length), device=device, dtype=dtype)
    causal_triu = torch.triu(torch.ones((total_length, total_length), device=device, dtype=torch.bool), diagonal=1)
    mask[:, 0].masked_fill_(causal_triu, mask_value)
    
    # Mascheramento future prefix per ciascuna audio query t
    for audio_index in range(prefix_frames):
        query_pos = prefix_end + audio_index
        visible_until = min(prefix_frames - 1, audio_index + int(lookahead_frames))
        # Maschera gli slot di prefisso > visible_until
        future_start = prefix_start + visible_until + 1
        if future_start < prefix_end:
            mask[:, 0, query_pos, future_start:prefix_end] = mask_value

    return mask


class DifferentiableCausalQwenV9(nn.Module):
    """Wrapper differenziabile per l'ottimizzazione causale di V9 attraverso Qwen frozen."""

    def __init__(
        self,
        projector: DeepCausalTransformerProjectorV9,
        causal_conv: CausalConv1D,
        talker: Any,
        *,
        train_conv: bool = False,
    ) -> None:
        super().__init__()
        self.projector = projector
        self.causal_conv = causal_conv
        self.talker = talker

        # Assicura che Qwen sia congelato
        for p in self.talker.talker.parameters():
            p.requires_grad = False

        # Configura la trainabilita' di V9
        for p in self.projector.parameters():
            p.requires_grad = True

        for p in self.causal_conv.parameters():
            p.requires_grad = bool(train_conv)

    def forward(
        self,
        h_raw: torch.Tensor,
        h_tx: torch.Tensor,
        codes: torch.Tensor,
        *,
        lookahead_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Esegue il forward pass differenziabile.
        
        Args:
            h_raw: (B, T, 1536) hidden Gemma pre-causal_conv.
            h_tx: (B, 1536) causal anchor (first_raw_frame).
            codes: (B, T, 16) ground-truth RVQ codes.
            lookahead_frames: lookahead consentito (default 0 per produzione causale).
            
        Returns:
            logits: (B, T, 2048) logit non normalizzati su CB0.
            h_pfx: (B, T, 2048) rappresentazione proiettata da V9.
        """
        device = h_raw.device
        dtype = h_raw.dtype
        B, T, _ = h_raw.shape

        # 1. Forward V9 (differenziabile rispetto ai pesi di V9)
        h_conv = self.causal_conv(h_raw)
        h_pfx = self.projector(h_conv, h_tx)  # (B, T, 2048)

        # 2. Parti prefisso Qwen (ICL + prompt testo + BOS)
        talker_model = self.talker.talker
        icl, text, bos, _ = self.talker.build_prefill_parts("Verifica causale del kernel.")
        icl = icl.expand(B, -1, -1).to(device=device, dtype=dtype)
        text = text.expand(B, -1, -1).to(device=device, dtype=dtype)
        bos = bos.expand(B, -1, -1).to(device=device, dtype=dtype)

        prefix_start = icl.shape[1] + text.shape[1]
        prefix_end = prefix_start + T

        # 3. Embedding audio passato: frame 0..T-2 predicono i frame 1..T-1
        audio_inputs = aggregate_code_embeddings(talker_model, codes[:, :-1].to(device=device)).to(dtype=dtype)

        # Sequenza completa: [ICL | text | h_pfx (T) | BOS (predice frame 0) | audio_inputs (T-1, predicono frame 1..T-1)]
        embeddings = torch.cat((icl, text, h_pfx, bos, audio_inputs), dim=1)
        total_length = embeddings.shape[1]

        # 4. Maschera 4D causale con lookahead L
        attention_mask = build_differentiable_qwen_mask(
            batch_size=B,
            prefix_start=prefix_start,
            prefix_frames=T,
            total_length=total_length,
            lookahead_frames=lookahead_frames,
            device=device,
            dtype=dtype,
        )

        position_ids = (
            torch.arange(total_length, device=device, dtype=torch.long)
            .view(1, 1, total_length)
            .expand(3, B, total_length)
        )

        # 5. Forward attraverso Qwen frozen
        # Nota: Qwen parameters hanno requires_grad=False, ma embeddings contiene h_pfx
        # che traccia il gradiente verso self.projector!
        model = talker_model.model
        hidden = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        # Le predizioni CB0 iniziano a prefix_end (il BOS predice frame 0)
        audio_hidden = hidden[:, prefix_end:prefix_end + T]
        logits = talker_model.codec_head(audio_hidden)

        return logits, h_pfx


def compute_causal_v9_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    teacher_logits: Optional[torch.Tensor] = None,
    teacher_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Calcola la loss di training: CrossEntropy su CB0 + distillazione opzionale."""
    target_cb0 = (targets[:, :, 0] if targets.dim() == 3 else targets).contiguous()
    B, T, V = logits.shape
    
    ce_loss = F.cross_entropy(logits.view(-1, V), target_cb0.view(-1))
    total_loss = ce_loss
    metrics = {"ce_loss": float(ce_loss.item())}

    if teacher_logits is not None and teacher_weight > 0.0:
        # Distillazione KL dai logit del teacher full-context
        student_log_probs = F.log_softmax(logits.float(), dim=-1)
        teacher_probs = F.softmax(teacher_logits.float(), dim=-1)
        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        total_loss = total_loss + teacher_weight * kl_div
        metrics["kl_loss"] = float(kl_div.item())

    metrics["total_loss"] = float(total_loss.item())

    # Top-1 e Top-5 accuracy diagnostica
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc1 = (preds == target_cb0).float().mean().item()
        top5_preds = logits.topk(5, dim=-1).indices
        acc5 = top5_preds.eq(target_cb0.unsqueeze(-1)).any(dim=-1).float().mean().item()
        metrics["top1_acc"] = acc1
        metrics["top5_acc"] = acc5

    return total_loss, metrics
