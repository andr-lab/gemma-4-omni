# -*- coding: utf-8 -*-
"""Differentiable causal V9 training through frozen Qwen.

Corrective P4D-R v2 contract:

    exact sample text
    + exact sample Gemma h_raw
    + fixed first_raw_frame causal anchor
    + production speaker x-vector
        -> causal V9 L=0
        -> frozen Qwen
        -> CB0 teacher forcing
        -> codec EOS

Only V9 projector parameters are trainable by default.
"""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from duplex_kernel.models.v9 import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
)
from duplex_kernel.qwen_conditioning import (
    QWEN_AUDIO_CONDITIONING_RECIPE,
    aggregate_qwen_code_embeddings,
    build_qwen_audio_conditioning,
    build_qwen_audio_conditioning_sequence,
)


ACOUSTIC_CB0_VOCAB = 2048
PAD_TARGET_ID = -100

# Compatibility export for existing P4D helpers.  The implementation is now
# shared with eager, slot-KV/native, and reference paths.
aggregate_code_embeddings = aggregate_qwen_code_embeddings


def build_differentiable_qwen_mask(
    batch_size: int,
    prefix_start: int,
    prefix_frames: int,
    total_length: int,
    *,
    lookahead_frames: int = 0,
    acoustic_frames: int | None = None,
    terminal_queries: int = 1,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Causal mask with separate prefix, acoustic and terminal budgets.

    ``prefix_frames`` is the conditioning capacity.  ``acoustic_frames`` is
    the number of real RVQ queries in this item; the terminal query is an
    additional decoder position and is not an acoustic frame.
    """
    if acoustic_frames is None:
        acoustic_frames = prefix_frames
    if acoustic_frames < 0 or terminal_queries < 0:
        raise ValueError("acoustic_frames e terminal_queries devono essere non negativi")
    if lookahead_frames < 0:
        raise ValueError("lookahead negativo")

    mask_value = torch.finfo(dtype).min
    prefix_end = prefix_start + prefix_frames

    mask = torch.zeros(
        (batch_size, 1, total_length, total_length),
        device=device,
        dtype=dtype,
    )

    causal_triu = torch.triu(
        torch.ones(
            (total_length, total_length),
            device=device,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    mask[:, 0].masked_fill_(causal_triu, mask_value)

    # Query at prefix_end + t predicts an acoustic frame for t < acoustic_frames;
    # the final terminal query is t == acoustic_frames.  Both remain causal.
    for audio_index in range(int(acoustic_frames) + int(terminal_queries)):
        query_pos = prefix_end + audio_index
        if query_pos >= total_length:
            raise ValueError("total_length insufficiente per le query acustiche/terminali")
        # Only annotated acoustic frames may expose a conditioning slot.  In
        # particular, the extra terminal query after N frames sees at most
        # prefix slot N-1, never the padding slot N.
        visible_until = min(
            prefix_frames - 1,
            audio_index + int(lookahead_frames),
            int(acoustic_frames) - 1,
        )
        future_start = prefix_start + visible_until + 1

        if future_start < prefix_end:
            mask[:, 0, query_pos, future_start:prefix_end] = mask_value

    return mask


def codec_eos_token_id(talker: Any) -> int:
    value = getattr(talker.tcfg, "codec_eos_token_id", None)

    if value is None:
        value = getattr(talker.tcfg, "codec_eos_id", None)

    if value is None:
        value = 2150

    eos = int(value)
    vocab = int(talker.talker.codec_head.out_features)

    if eos < ACOUSTIC_CB0_VOCAB or eos >= vocab:
        raise RuntimeError(
            f"codec EOS deve essere fuori dal vocabolario acustico: {eos}/{vocab}"
        )

    return eos


class DifferentiableCausalQwenV9(nn.Module):
    """Causal V9 optimized through frozen production Qwen semantics."""

    audio_conditioning_recipe = QWEN_AUDIO_CONDITIONING_RECIPE

    def __init__(
        self,
        projector: DeepCausalTransformerProjectorV9,
        causal_conv: CausalConv1D,
        talker: Any,
        *,
        train_conv: bool = False,
        voice_conditioning: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.projector = projector
        self.causal_conv = causal_conv

        # Plain wrapper, not an nn.Module child.
        object.__setattr__(self, "talker_wrapper", talker)

        self.voice_conditioning = talker.resolve_voice_conditioning(
            voice_conditioning
        )

        for parameter in talker.talker.parameters():
            parameter.requires_grad = False

        for parameter in self.projector.parameters():
            parameter.requires_grad = True

        for parameter in self.causal_conv.parameters():
            parameter.requires_grad = bool(train_conv)

        self._train_conv = bool(train_conv)

    @property
    def codec_eos_id(self) -> int:
        return codec_eos_token_id(self.talker_wrapper)

    def _forward_qwen_one(
        self,
        *,
        h_pfx: torch.Tensor,
        codes: torch.Tensor,
        text: str,
        lookahead_frames: int,
        trailing_text_hidden: torch.Tensor | None = None,
        valid_frame_count: int | None = None,
        voice_conditioning: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """One semantic sample because text-prefill lengths are variable."""
        if h_pfx.dim() != 3 or h_pfx.shape[0] != 1:
            raise ValueError("h_pfx sample deve essere 1,T,D")

        if codes.dim() != 3 or codes.shape[0] != 1:
            raise ValueError("codes sample deve essere 1,T,C")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("sample text vuoto")

        wrapper = self.talker_wrapper
        talker_model = wrapper.talker
        device = h_pfx.device
        dtype = h_pfx.dtype
        prefix_capacity = int(h_pfx.shape[1])
        if int(codes.shape[1]) > prefix_capacity:
            raise ValueError("codes eccedono la prefix capacity")
        acoustic_frames = prefix_capacity if valid_frame_count is None else int(valid_frame_count)
        if acoustic_frames < 0 or acoustic_frames > int(codes.shape[1]):
            raise ValueError("valid_frame_count non coerente con codes")

        icl, text_part, bos, tts_pad_embed = wrapper.build_prefill_parts(
            text.strip(),
            voice_conditioning=self.voice_conditioning if voice_conditioning is None else voice_conditioning,
        )

        icl = icl.to(device=device, dtype=dtype)
        text_part = text_part.to(device=device, dtype=dtype)
        bos = bos.to(device=device, dtype=dtype)
        tts_pad_embed = tts_pad_embed.to(device=device, dtype=dtype)

        prefix_start = int(icl.shape[1] + text_part.shape[1])
        prefix_end = prefix_start + prefix_capacity

        # BOS predicts the first acoustic frame.  Each valid GT RVQ frame then
        # advances one query, and one additional query is reserved for EOS.
        code_inputs = codes[:, :acoustic_frames].to(device=device, dtype=torch.long)
        if acoustic_frames:
            if trailing_text_hidden is None:
                audio_inputs = build_qwen_audio_conditioning(
                    talker_model,
                    code_inputs,
                    tts_pad_embed=tts_pad_embed,
                )
            else:
                audio_inputs = build_qwen_audio_conditioning_sequence(
                    talker_model,
                    code_inputs,
                    tts_pad_embed=tts_pad_embed,
                    trailing_text_hidden=trailing_text_hidden.to(device=device, dtype=dtype),
                )
            audio_inputs = audio_inputs.to(dtype=dtype)
        else:
            audio_inputs = torch.zeros(
                (1, 0, bos.shape[-1]), device=device, dtype=dtype
            )

        embeddings = torch.cat(
            (icl, text_part, h_pfx, bos, audio_inputs),
            dim=1,
        )
        total_length = int(embeddings.shape[1])

        attention_mask = build_differentiable_qwen_mask(
            batch_size=1,
            prefix_start=prefix_start,
            prefix_frames=prefix_capacity,
            total_length=total_length,
            lookahead_frames=lookahead_frames,
            acoustic_frames=acoustic_frames,
            terminal_queries=1,
            device=device,
            dtype=dtype,
        )

        position_ids = (
            torch.arange(total_length, device=device, dtype=torch.long)
            .view(1, 1, total_length)
            .expand(3, 1, total_length)
        )

        hidden = talker_model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        ).last_hidden_state

        audio_hidden = hidden[:, prefix_end : prefix_end + acoustic_frames + 1]
        if int(audio_hidden.shape[1]) != acoustic_frames + 1:
            raise RuntimeError(
                "Qwen training output non contiene N_acoustic+N_terminal prediction slots"
            )

        return talker_model.codec_head(audio_hidden)

    def forward(
        self,
        h_raw: torch.Tensor,
        h_tx: torch.Tensor,
        codes: torch.Tensor,
        texts: Sequence[str],
        *,
        lookahead_frames: int = 0,
        trailing_text_hidden: Sequence[torch.Tensor | None] | None = None,
        valid_frame_mask: torch.Tensor | None = None,
        voice_conditioning_by_sample: Sequence[dict[str, Any] | None] | None = None,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Differentiable causal forward using each sample's exact text."""
        if h_raw.dim() != 3:
            raise ValueError("h_raw deve essere B,T,1536")

        if codes.dim() != 3:
            raise ValueError("codes deve essere B,T,C")

        batch_size, frames, raw_dim = h_raw.shape

        if raw_dim != 1536:
            raise ValueError(f"h_raw dim={raw_dim}")

        if codes.shape[0] != batch_size or codes.shape[1] != frames:
            raise ValueError("codes/h_raw mismatch")

        if len(texts) != batch_size:
            raise ValueError("texts batch mismatch")
        if trailing_text_hidden is not None and len(trailing_text_hidden) != batch_size:
            raise ValueError("trailing_text_hidden batch mismatch")
        if voice_conditioning_by_sample is not None and len(voice_conditioning_by_sample) != batch_size:
            raise ValueError("voice_conditioning_by_sample batch mismatch")
        if valid_frame_mask is None:
            valid_frame_mask = torch.ones(
                (batch_size, frames), dtype=torch.bool, device=h_raw.device
            )
        if valid_frame_mask.shape != (batch_size, frames):
            raise ValueError("valid_frame_mask deve avere shape B,T")
        valid_frame_mask = valid_frame_mask.to(device=h_raw.device, dtype=torch.bool)
        for row in valid_frame_mask:
            count = int(row.sum().item())
            if count and not bool(row[:count].all()) or bool(row[count:].any()):
                raise ValueError("valid_frame_mask deve avere frame validi contigui in testa")

        if h_tx.dim() != 2 or h_tx.shape != (batch_size, 1536):
            raise ValueError("h_tx deve essere B,1536")

        h_conv = self.causal_conv(h_raw)
        h_pfx = self.projector(h_conv, h_tx)

        logits: list[torch.Tensor] = []
        for index in range(batch_size):
            logits.append(
                self._forward_qwen_one(
                    h_pfx=h_pfx[index : index + 1],
                    codes=codes[index : index + 1],
                    text=texts[index],
                    lookahead_frames=lookahead_frames,
                    trailing_text_hidden=(
                        None
                        if trailing_text_hidden is None
                        else trailing_text_hidden[index]
                    ),
                    valid_frame_count=int(valid_frame_mask[index].sum().item()),
                    voice_conditioning=(
                        None
                        if voice_conditioning_by_sample is None
                        else voice_conditioning_by_sample[index]
                    ),
                )
            )

        return logits, h_pfx


def _production_allowed_logits(
    logits: torch.Tensor,
    *,
    eos_token_id: int,
    acoustic_vocab_size: int,
) -> torch.Tensor:
    """Exact logical support of production CB0 sampling.

    Production allows codec ids 0..2047 plus codec EOS. Other special codec
    ids are excluded from the loss support as they are from runtime sampling.
    """
    vocab = int(logits.shape[-1])

    if acoustic_vocab_size <= 0 or acoustic_vocab_size > vocab:
        raise ValueError("acoustic vocab invalido")

    if eos_token_id < 0 or eos_token_id >= vocab:
        raise ValueError("EOS fuori vocab")

    if eos_token_id < acoustic_vocab_size:
        raise ValueError("EOS collide con acoustic CB0 vocab")

    return torch.cat(
        (
            logits[..., :acoustic_vocab_size],
            logits[..., eos_token_id : eos_token_id + 1],
        ),
        dim=-1,
    )


def sample_causal_training_history(
    logits_by_sample: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    eos_token_id: int,
    sample_probability: float,
    valid_frame_mask: torch.Tensor | None = None,
    acoustic_vocab_size: int = ACOUSTIC_CB0_VOCAB,
    top_k: int = 20,
    temperature: float = 0.6,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Replace a fraction of teacher-forced CB0 history with sampled CB0.

    This is a scheduled-sampling exposure augmentation, not a full rollout:
    sampled CB0 codes are drawn from each gold-history query, while residual
    codebooks remain the reference RVQ codes. Sampled EOS is not an audio code
    and therefore leaves that position teacher-forced. PyTorch 2.12 documents
    ``torch.multinomial`` over row-wise probabilities with an optional
    ``torch.Generator``: https://docs.pytorch.org/docs/2.12/generated/torch.multinomial.html
    """
    if targets.dim() != 3 or int(targets.shape[-1]) != 16:
        raise ValueError("targets must have shape (B,T,16)")
    if targets.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("targets must use an integer dtype")
    if len(logits_by_sample) != int(targets.shape[0]):
        raise ValueError("logits batch mismatch")
    if isinstance(sample_probability, bool) or not isinstance(sample_probability, (int, float)):
        raise ValueError("sample_probability must be numeric")
    probability = float(sample_probability)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("sample_probability must be finite and between zero and one")
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if isinstance(temperature, bool) or not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")

    batch_size, frames, _ = targets.shape
    if valid_frame_mask is None:
        valid_frame_mask = targets[:, :, 0].ne(PAD_TARGET_ID)
    if tuple(valid_frame_mask.shape) != (batch_size, frames):
        raise ValueError("valid_frame_mask must have shape (B,T)")
    valid_frame_mask = valid_frame_mask.to(device=targets.device, dtype=torch.bool)

    mixed_targets = targets.detach().clone()
    valid_total = 0
    selected_total = 0
    replaced_total = 0
    changed_total = 0
    eos_fallback_total = 0
    for index in range(batch_size):
        mask = valid_frame_mask[index]
        valid_count = int(mask.sum().item())
        if (valid_count and not bool(mask[:valid_count].all())) or bool(mask[valid_count:].any()):
            raise ValueError("valid_frame_mask must be a contiguous true prefix")
        if valid_count <= 0:
            raise ValueError("each scheduled-sampling sample needs a valid frame")
        raw_logits = logits_by_sample[index]
        if raw_logits.dim() != 3 or int(raw_logits.shape[0]) != 1:
            raise ValueError("per-sample logits must have shape (1,N_queries,V)")
        if raw_logits.device != targets.device:
            raise ValueError("per-sample logits and targets must be on the same device")
        if int(raw_logits.shape[1]) != valid_count + 1:
            raise ValueError("per-sample logits must contain N acoustic + 1 terminal queries")
        frame_targets = targets[index, :valid_count, 0].long()
        if bool((frame_targets < 0).any()) or bool((frame_targets >= acoustic_vocab_size).any()):
            raise ValueError("valid CB0 targets must be within the acoustic vocabulary")
        allowed = _production_allowed_logits(
            raw_logits,
            eos_token_id=eos_token_id,
            acoustic_vocab_size=acoustic_vocab_size,
        )[0, :valid_count].detach().float()
        valid_total += valid_count
        if probability == 0.0:
            continue

        sample_mask = torch.rand(
            valid_count,
            device=targets.device,
            generator=generator,
        ).lt(probability)
        top_values, top_classes = torch.topk(
            allowed,
            k=min(top_k, int(allowed.shape[-1])),
            dim=-1,
        )
        probabilities = torch.softmax(top_values / float(temperature), dim=-1)
        sampled_rank = torch.multinomial(probabilities, num_samples=1, generator=generator)
        sampled_classes = top_classes.gather(-1, sampled_rank).squeeze(-1)
        eos_sampled = sampled_classes.eq(int(acoustic_vocab_size))
        replace_mask = sample_mask & ~eos_sampled
        chosen_codes = torch.where(replace_mask, sampled_classes, frame_targets)
        mixed_targets[index, :valid_count, 0] = chosen_codes.to(dtype=mixed_targets.dtype)
        selected_total += int(sample_mask.sum().item())
        eos_fallback_total += int((sample_mask & eos_sampled).sum().item())
        replaced_total += int(replace_mask.sum().item())
        changed_total += int((replace_mask & sampled_classes.ne(frame_targets)).sum().item())

    return mixed_targets, {
        "history_rollin_probability": probability,
        "history_rollin_valid_frame_count": float(valid_total),
        "history_rollin_selected_frame_count": float(selected_total),
        "history_rollin_replaced_cb0_count": float(replaced_total),
        "history_rollin_changed_cb0_count": float(changed_total),
        "history_rollin_eos_fallback_count": float(eos_fallback_total),
    }


def compute_causal_v9_loss(
    logits_by_sample: Sequence[torch.Tensor],
    targets: torch.Tensor,
    *,
    eos_token_id: int,
    acoustic_vocab_size: int = ACOUSTIC_CB0_VOCAB,
    valid_frame_mask: torch.Tensor | None = None,
    terminal_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Masked CB0 + applicable natural-EOS objective.

    The terminal query is at ``N_valid_acoustic_frames``.  It is an additional
    query, not frame ``T-1`` and not a label for every fixed-size window.
    """
    if targets.dim() != 3:
        raise ValueError("targets deve essere B,T,C")
    batch_size, frames, _ = targets.shape
    if len(logits_by_sample) != batch_size:
        raise ValueError("logits batch mismatch")

    if valid_frame_mask is None:
        valid_frame_mask = targets[:, :, 0].ne(PAD_TARGET_ID)
    if tuple(valid_frame_mask.shape) != (batch_size, frames):
        raise ValueError("valid_frame_mask deve avere shape B,T")
    valid_frame_mask = valid_frame_mask.to(device=targets.device, dtype=torch.bool)
    if terminal_mask is None:
        # Fail closed: a fixed-size window is not an utterance boundary.
        terminal_mask = torch.zeros(batch_size, dtype=torch.bool, device=targets.device)
    if tuple(terminal_mask.shape) != (batch_size,):
        raise ValueError("terminal_mask deve avere shape B")
    terminal_mask = terminal_mask.to(device=targets.device, dtype=torch.bool)

    losses: list[torch.Tensor] = []
    frame_losses: list[torch.Tensor] = []
    eos_losses: list[torch.Tensor] = []
    top1_values: list[torch.Tensor] = []
    top5_values: list[torch.Tensor] = []
    eos_top1_values: list[torch.Tensor] = []
    eos_ranks: list[torch.Tensor] = []
    valid_frame_total = 0
    terminal_total = 0
    eos_class = int(acoustic_vocab_size)

    for index in range(batch_size):
        raw_logits = logits_by_sample[index]
        if raw_logits.dim() != 3 or raw_logits.shape[0] != 1:
            raise ValueError("logits sample shape deve essere 1,N_queries,V")
        mask = valid_frame_mask[index]
        valid_count = int(mask.sum().item())
        if valid_count and not bool(mask[:valid_count].all()) or bool(mask[valid_count:].any()):
            raise ValueError("valid_frame_mask deve avere frame validi contigui in testa")
        required_queries = valid_count + 1
        if int(raw_logits.shape[1]) != required_queries:
            raise ValueError(
                "logits sample deve contenere N_acoustic + 1 query terminale: "
                f"got={raw_logits.shape[1]} expected={required_queries}"
            )

        frame_targets = targets[index, :valid_count, 0].long().contiguous()
        if valid_count and (bool((frame_targets < 0).any()) or bool((frame_targets >= acoustic_vocab_size).any())):
            low = int(frame_targets.min().item())
            high = int(frame_targets.max().item())
            raise RuntimeError(
                "P4D_R_CB0_TARGET_CONTRACT_BLOCKER: "
                f"range={low}..{high}, production range=0..{acoustic_vocab_size - 1}"
            )

        allowed = _production_allowed_logits(
            raw_logits, eos_token_id=eos_token_id, acoustic_vocab_size=acoustic_vocab_size
        )
        sample_terms: list[torch.Tensor] = []
        if valid_count:
            frame_logits = allowed[0, :valid_count].float()
            frame_loss = F.cross_entropy(frame_logits, frame_targets)
            sample_terms.append(frame_loss)
            frame_losses.append(frame_loss)
            top1_values.append(frame_logits.argmax(dim=-1).eq(frame_targets).float().mean())
            top5_values.append(
                frame_logits.topk(k=min(5, frame_logits.shape[-1]), dim=-1)
                .indices.eq(frame_targets.unsqueeze(-1)).any(dim=-1).float().mean()
            )
            valid_frame_total += valid_count

        if bool(terminal_mask[index].item()):
            terminal_logits = allowed[0, valid_count : valid_count + 1].float()
            eos_target = torch.tensor([eos_class], device=allowed.device, dtype=torch.long)
            eos_loss = F.cross_entropy(terminal_logits, eos_target)
            sample_terms.append(eos_loss)
            eos_losses.append(eos_loss)
            final_logits = terminal_logits[0]
            eos_top1_values.append((final_logits.argmax() == eos_class).float())
            eos_logit = final_logits[eos_class]
            eos_ranks.append((final_logits > eos_logit).sum().float() + 1)
            terminal_total += 1

        if sample_terms:
            losses.append(torch.stack(sample_terms).mean())

    if not losses:
        anchor = logits_by_sample[0].sum() * 0.0
        total_loss = anchor
    else:
        total_loss = torch.stack(losses).mean()

    def _mean_or_zero(values: list[torch.Tensor]) -> float:
        return 0.0 if not values else float(torch.stack(values).mean().detach().item())

    metrics = {
        "total_loss": float(total_loss.detach().item()),
        "cb0_ce_loss": _mean_or_zero(frame_losses),
        "eos_ce_loss": _mean_or_zero(eos_losses),
        "top1_acc": _mean_or_zero(top1_values),
        "top5_acc": _mean_or_zero(top5_values),
        "eos_top1_acc": _mean_or_zero(eos_top1_values),
        "eos_mean_rank": _mean_or_zero(eos_ranks),
        "valid_frame_count": float(valid_frame_total),
        "terminal_label_count": float(terminal_total),
    }
    return total_loss, metrics


__all__ = [
    "ACOUSTIC_CB0_VOCAB",
    "DifferentiableCausalQwenV9",
    "aggregate_code_embeddings",
    "build_differentiable_qwen_mask",
    "codec_eos_token_id",
    "compute_causal_v9_loss",
    "sample_causal_training_history",
]
