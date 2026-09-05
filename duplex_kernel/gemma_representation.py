"""Canonical Gemma multimodal representation used by V9.

The contract in this module is deliberately small and shared by the offline
cache extractor and the realtime listener: run the multimodal language model,
then select the first audio feature of an 80 ms frame from the final LM layer.
The selected tensor is before the V9 causal convolution.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


# The released duplex training manifest uses a 32-token left-padded text
# block.  Keep this value centralized; the incremental canonical helper also
# derives it from a supplied sequence so legacy 48-token fixtures remain
# diagnosable rather than silently shifting the audio positions.
TEXT_BLOCK_LEN = 32
AUDIO_FEATURES_PER_FRAME = 4
AUDIO_START_AFTER_TEXT = 2 + TEXT_BLOCK_LEN  # BOA is position 1+TEXT_BLOCK_LEN
# The listener's canonical shape-stable prefill workspace is part of the
# numerical contract: changing its GEMM M-shape changes BF16 rounding.
CANONICAL_LM_PREFILL_TARGET_LEN = 291
CANONICAL_RAW_STAGE = "pre_causal_conv"
MEL_BINS = 128
MEL_PER_FRAME = 8


def interleave_duplex_mel(
    user_mel: np.ndarray,
    assistant_mel: np.ndarray,
    *,
    mel_per_frame: int = MEL_PER_FRAME,
) -> np.ndarray:
    """Interleave two mel streams in the canonical *frame-block* order.

    The Gemma/Qwen training contract is ``[u[t,0:8], a[t,0:8]]`` for every
    80-ms frame, not row-wise ``[u[t,0], a[t,0], ...]``.  Keeping this
    operation in the shared representation module prevents the realtime
    listener and the offline cache builder from silently feeding different
    spectro-temporal neighborhoods to Gemma's 2-D subsampler.
    """
    user = np.asarray(user_mel, dtype=np.float32)
    assistant = np.asarray(assistant_mel, dtype=np.float32)
    if user.ndim != 2 or assistant.ndim != 2:
        raise ValueError("user_mel e assistant_mel devono avere shape (rows, bins)")
    if user.shape != assistant.shape:
        raise ValueError(f"mel user/assistant non allineati: {user.shape} != {assistant.shape}")
    if user.shape[1] != MEL_BINS:
        raise ValueError(f"mel deve avere {MEL_BINS} bande, got {user.shape[1]}")
    if mel_per_frame <= 0 or user.shape[0] % mel_per_frame:
        raise ValueError("rows mel deve essere un multiplo positivo di mel_per_frame")
    user_frames = user.reshape(-1, mel_per_frame, MEL_BINS)
    assistant_frames = assistant.reshape(-1, mel_per_frame, MEL_BINS)
    return np.stack((user_frames, assistant_frames), axis=1).reshape(-1, MEL_BINS)


def build_audio_placeholder_inputs(
    *,
    batch_size: int,
    audio_feature_rows: int,
    device: torch.device | str,
    audio_token_id: int = 258881,
    bos_token_id: int = 2,
    boa_token_id: int = 256000,
    eoa_token_id: int = 258883,
    pad_token_id: int = 0,
    text_token_ids: tuple[int, ...] | list[int] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact ``[BOS|text|BOA|audio placeholders|EOA]`` contract."""
    if audio_feature_rows <= 0 or audio_feature_rows % AUDIO_FEATURES_PER_FRAME:
        raise ValueError("audio_feature_rows deve essere un multiplo positivo di 4")
    device = torch.device(device)
    audio_tokens = audio_feature_rows // AUDIO_FEATURES_PER_FRAME
    length = 3 + TEXT_BLOCK_LEN + audio_tokens
    input_ids = torch.full((batch_size, length), int(audio_token_id), dtype=torch.long, device=device)
    input_ids[:, 0] = int(bos_token_id)
    input_ids[:, 1:1 + TEXT_BLOCK_LEN] = int(pad_token_id)
    input_ids[:, 1 + TEXT_BLOCK_LEN] = int(boa_token_id)
    input_ids[:, -1] = int(eoa_token_id)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, 1:1 + TEXT_BLOCK_LEN] = 0
    values = tuple(int(value) for value in text_token_ids[-TEXT_BLOCK_LEN:])
    if values:
        start = TEXT_BLOCK_LEN - len(values)
        input_ids[:, 1 + start:1 + TEXT_BLOCK_LEN] = torch.tensor(values, dtype=torch.long, device=device)
        attention_mask[:, 1 + start:1 + TEXT_BLOCK_LEN] = 1
    return input_ids, attention_mask


def _resolve_gemma_model(model: Any) -> Any:
    """Unwrap Transformers/PEFT wrappers to the native ``Gemma4Model``."""
    pending = [model]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if callable(getattr(candidate, "get_audio_features", None)) and hasattr(
            candidate, "language_model"
        ):
            return candidate
        for name in ("model", "base_model"):
            child = getattr(candidate, name, None)
            if child is not None and child is not candidate:
                pending.append(child)
        getter = getattr(candidate, "get_base_model", None)
        if callable(getter):
            child = getter()
            if child is not candidate:
                pending.append(child)
    raise TypeError("modello fornito non contiene un Gemma4Model multimodale")


def _audio_soft_tokens(
    model: Any,
    *,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> torch.Tensor:
    """Project native audio-tower states into LM soft-token space."""
    model = _resolve_gemma_model(model)
    audio_output = model.get_audio_features(
        input_features=input_features,
        input_features_mask=input_features_mask,
        return_dict=True,
    )
    pooled = getattr(audio_output, "pooler_output", None)
    valid_mask = getattr(audio_output, "attention_mask", None)
    if pooled is None or valid_mask is None:
        raise RuntimeError("Gemma audio tower non ha restituito pooler_output/mask")
    valid_mask = valid_mask.to(pooled.device)
    rows = [pooled[index][valid_mask[index]] for index in range(pooled.shape[0])]
    if not rows or any(value.ndim != 2 or value.shape[-1] != 1536 for value in rows):
        raise RuntimeError("Gemma audio soft-token output non valido")
    if len({int(value.shape[0]) for value in rows}) != 1:
        raise RuntimeError("batch Gemma con numero di audio feature non uniforme")
    return torch.stack(rows, dim=0)


def _lm_has_shared_kv(language_model: Any) -> bool:
    return any(
        bool(getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False))
        for layer in getattr(language_model, "layers", ())
    )


@torch.inference_mode()
def canonical_incremental_lm_hidden(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the canonical Gemma LM state machine one audio frame at a time.

    This is intentionally the same cache transition used by
    :class:`GemmaAudioTowerBackend`: stable ``[BOS|text|BOA]`` tokens are
    prefixed once, then four audio soft tokens are appended per 80-ms frame.
    The final EOA token is evaluated only to return a complete hidden tensor;
    it cannot influence any earlier causal audio position.  This function is
    the runtime candidate; :func:`canonical_full_lm_hidden` remains the
    independent full-forward oracle used to certify it.
    """
    model = _resolve_gemma_model(model)
    lm = model.language_model
    lm_dtype = lm.embed_tokens.weight.dtype
    audio_soft = _audio_soft_tokens(
        model, input_features=input_features, input_features_mask=input_features_mask
    )
    batch_size, audio_rows, hidden_size = audio_soft.shape
    if hidden_size != 1536 or audio_rows <= 0 or audio_rows % AUDIO_FEATURES_PER_FRAME:
        raise RuntimeError(f"Gemma audio soft-token shape inattesa: {tuple(audio_soft.shape)}")
    # ``input_features`` are mel rows, while the audio tower returns one
    # 1536-dim soft token per output position.  The placeholder sequence has
    # exactly one audio token per returned soft token.
    if input_ids.ndim != 2 or input_ids.shape[0] != batch_size:
        raise ValueError(f"sequenza Gemma non allineata: ids={tuple(input_ids.shape)}")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask Gemma deve avere la stessa shape di input_ids")

    text_block_len = int(input_ids.shape[1] - 3 - audio_rows)
    if text_block_len <= 0:
        raise ValueError("sequenza Gemma priva di un blocco testo valido")
    audio_start = 2 + text_block_len
    stable_ids = input_ids[:, :audio_start]
    stable_attention = attention_mask[:, :audio_start].to(
        device=input_ids.device, dtype=torch.long
    )
    stable_embeds = lm.embed_tokens(stable_ids)
    stable_ple = lm.get_per_layer_inputs(stable_ids, stable_embeds)
    shared_enabled = _lm_has_shared_kv(lm)
    past = None
    shared = None
    hidden_chunks: list[torch.Tensor] = []
    frame_rows = AUDIO_FEATURES_PER_FRAME

    # Prime the canonical prefix as its own transaction. Concatenating the
    # first frame would change the GEMM M-shape from the attested prefill
    # contract and would not represent the production listener state machine.
    from duplex_kernel.shape_stable_listener import ShapeStableGemmaListenerLM
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    shape_stable = ShapeStableGemmaListenerLM(
        lm,
        target_length=CANONICAL_LM_PREFILL_TARGET_LEN,
        stream_length=AUDIO_FEATURES_PER_FRAME,
    )
    prefix_positions = torch.arange(audio_start, device=input_ids.device, dtype=torch.long)
    prefix_position_ids = prefix_positions.view(1, -1).expand(batch_size, -1)
    prefix_masks = create_masks_for_generate(
        lm.config,
        stable_embeds,
        stable_attention,
        past_key_values=None,
        position_ids=prefix_position_ids,
    )
    prefix_output = shape_stable.forward(
        inputs_embeds=stable_embeds,
        per_layer_inputs=stable_ple,
        attention_mask=prefix_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        shared_kv_states=None,
    )
    past = prefix_output.past_key_values
    shared = prefix_output.shared_kv_states
    if shared_enabled and shared is None:
        raise RuntimeError("Gemma LM non ha restituito shared_kv_states")
    hidden_chunks.append(prefix_output.last_hidden_state)

    for frame_index in range(audio_rows // frame_rows):
        start = frame_index * frame_rows
        frame_audio = audio_soft[:, start:start + frame_rows, :].to(dtype=lm_dtype)
        # Gemma4Model.forward replaces multimodal ids with PAD before both
        # main embedding construction and PLE construction, then scatters the
        # audio tower output only into the main embedding tensor.
        pad_id = int(getattr(model.config.text_config, "pad_token_id", 0))
        pad_ids = torch.full(
            (batch_size, frame_rows), pad_id, dtype=torch.long, device=input_ids.device
        )
        pad_embeds = lm.embed_tokens(pad_ids)
        frame_ple = lm.get_per_layer_inputs(pad_ids, pad_embeds)
        embeds = frame_audio
        per_layer = frame_ple
        start_position = audio_start + start
        positions = torch.arange(
            start_position, start_position + frame_rows,
            device=input_ids.device, dtype=torch.long,
        )
        chunk_attention = torch.ones(
            (batch_size, start_position + frame_rows),
            device=input_ids.device, dtype=torch.long,
        )
        chunk_attention[:, :audio_start] = stable_attention
        position_ids = positions.view(1, -1).expand(batch_size, -1)
        # Gemma4Model.forward constructs this mapping before entering the LM;
        # pass the same native 4-D sliding/full masks instead of asking a
        # standalone LM call to infer a possibly different mask.
        mask_mapping = create_masks_for_generate(
            model.config,
            embeds,
            chunk_attention,
            past_key_values=past,
            position_ids=position_ids,
        )
        output = shape_stable.forward(
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            attention_mask=mask_mapping,
            position_ids=position_ids,
            past_key_values=past,
            shared_kv_states=shared,
        )
        next_shared = output.shared_kv_states
        if shared_enabled and next_shared is None:
            raise RuntimeError("Gemma LM non ha restituito shared_kv_states")
        past = output.past_key_values
        shared = next_shared
        # Prefix and every audio frame remain separate cache transactions;
        # each chunk contains only the rows produced by that transaction.
        hidden_chunks.append(output.last_hidden_state)

    # EOA is causal-after-all-audio and is returned only for shape-compatible
    # cache artifacts; it never changes the selected audio hidden states.
    eoa_ids = input_ids[:, -1:]
    eoa_embeds = lm.embed_tokens(eoa_ids)
    eoa_ple = lm.get_per_layer_inputs(eoa_ids, eoa_embeds)
    eoa_position = audio_start + audio_rows
    eoa_attention = torch.ones(
        (batch_size, eoa_position + 1), device=input_ids.device, dtype=torch.long
    )
    eoa_attention[:, :audio_start] = stable_attention
    eoa_position_ids = torch.full(
        (batch_size, 1), eoa_position, device=input_ids.device, dtype=torch.long
    )
    eoa_masks = create_masks_for_generate(
        lm.config,
        eoa_embeds,
        eoa_attention,
        past_key_values=past,
        position_ids=eoa_position_ids,
    )
    eoa_output = shape_stable.forward(
        inputs_embeds=eoa_embeds,
        per_layer_inputs=eoa_ple,
        attention_mask=eoa_masks,
        position_ids=eoa_position_ids,
        past_key_values=past,
        shared_kv_states=shared,
    )
    hidden_chunks.append(eoa_output.last_hidden_state[:, :1, :])
    return torch.cat(hidden_chunks, dim=1), audio_soft


@torch.inference_mode()
def canonical_full_lm_hidden(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the independent, native ``Gemma4Model.forward`` construction.

    ``Gemma4Model.forward`` first changes multimodal placeholder ids to PAD
    for the text/PLE path, then scatters the audio-tower projection into the
    *main* embeddings and finally builds the hybrid 4-D
    ``sliding_attention``/``full_attention`` masks.  Reproducing those steps
    here is intentional: the incremental cache path is a runtime candidate,
    while this full forward is the representation oracle.
    """
    from transformers.models.gemma4.modeling_gemma4 import create_masks_for_generate

    model = _resolve_gemma_model(model)
    lm = model.language_model
    audio_soft = _audio_soft_tokens(
        model, input_features=input_features, input_features_mask=input_features_mask
    )
    batch_size, audio_rows, hidden_size = audio_soft.shape
    if hidden_size != 1536 or audio_rows <= 0 or audio_rows % AUDIO_FEATURES_PER_FRAME:
        raise RuntimeError(f"Gemma audio soft-token shape inattesa: {tuple(audio_soft.shape)}")
    if input_ids.ndim != 2 or input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids e attention_mask Gemma devono essere matrici allineate")
    if input_ids.shape[0] != batch_size:
        raise ValueError("batch input_ids/audio features Gemma non allineato")

    config = model.config
    audio_token_id = int(getattr(config, "audio_token_id", 258881))
    pad_token_id = int(getattr(getattr(config, "text_config", None), "pad_token_id", 0))
    multimodal_mask = input_ids.eq(audio_token_id)
    expected_audio_tokens = int(multimodal_mask.sum().item())
    if expected_audio_tokens != batch_size * audio_rows:
        raise ValueError(
            f"placeholder audio Gemma non allineati: ids={expected_audio_tokens}, "
            f"features={batch_size * audio_rows}"
        )

    # This is the same ordering as Gemma4Model.forward, including PAD PLE at
    # multimodal slots.  ``masked_scatter`` is used rather than assignment to
    # preserve the official flattened batch/slot ordering.
    llm_input_ids = input_ids.clone()
    llm_input_ids[multimodal_mask] = pad_token_id
    inputs_embeds = lm.embed_tokens(llm_input_ids)
    per_layer_inputs = lm.get_per_layer_inputs(llm_input_ids, inputs_embeds)
    expanded_audio_mask = multimodal_mask.unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(
        expanded_audio_mask, audio_soft.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    )

    position_ids = torch.arange(
        input_ids.shape[1], device=inputs_embeds.device, dtype=torch.long
    ).unsqueeze(0)
    mask_mapping = create_masks_for_generate(
        config,
        inputs_embeds,
        attention_mask,
        past_key_values=None,
        position_ids=position_ids,
    )
    output = lm(
        inputs_embeds=inputs_embeds,
        per_layer_inputs=per_layer_inputs,
        attention_mask=mask_mapping,
        position_ids=position_ids,
        use_cache=True,
        return_dict=True,
    )
    return output.last_hidden_state, audio_soft


@torch.inference_mode()
def canonical_lm_hidden(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_features: torch.Tensor,
    input_features_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the canonical final-LM hidden and native audio soft tokens.

    This public helper is the full-forward oracle.  The runtime listener uses
    :func:`canonical_incremental_lm_hidden`'s transition logic separately and
    must be checked against this oracle rather than defining parity by
    comparing two copies of the same incremental code.
    """
    return canonical_full_lm_hidden(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_features=input_features,
        input_features_mask=input_features_mask,
    )


def select_audio_frame_hidden(
    hidden: torch.Tensor,
    *,
    frame_index: int,
    audio_start: int = AUDIO_START_AFTER_TEXT,
    features_per_frame: int = AUDIO_FEATURES_PER_FRAME,
) -> torch.Tensor:
    """Select canonical ``gemma_hidden_t`` from the first feature of a frame."""
    if hidden.ndim != 3:
        raise ValueError(f"hidden deve avere shape (B,L,H), got {tuple(hidden.shape)}")
    if frame_index < 0:
        raise ValueError("frame_index non può essere negativo")
    position = int(audio_start) + int(frame_index) * int(features_per_frame)
    selected = hidden[:, position, :]
    if selected.shape[-1] != 1536:
        raise ValueError(f"hidden Gemma V9 deve avere ultima dimensione 1536, got {selected.shape[-1]}")
    return selected


__all__ = [
    "AUDIO_FEATURES_PER_FRAME",
    "AUDIO_START_AFTER_TEXT",
    "CANONICAL_LM_PREFILL_TARGET_LEN",
    "CANONICAL_RAW_STAGE",
    "MEL_BINS",
    "MEL_PER_FRAME",
    "TEXT_BLOCK_LEN",
    "build_audio_placeholder_inputs",
    "interleave_duplex_mel",
    "canonical_incremental_lm_hidden",
    "canonical_full_lm_hidden",
    "canonical_lm_hidden",
    "select_audio_frame_hidden",
]
