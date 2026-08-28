# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/processors.py — Processori inter-stadio (firme REALI del framework).

Pattern di riferimento (SDD):
- vllm_omni/model_executor/stage_input_processors/qwen3_omni.py
  (thinker2talker_token_only: sync_process_input_func orchestrator-side;
   thinker2talker_full_payload: custom_process_next_stage_input_func connector)
- vllm_omni/model_executor/stage_input_processors/nemotron_voicechat.py

Flusso:
1. Sync processor (stage 1, orchestrator): placeholder prompt + additional_information
   (text da sintetizzare = risposta del thinker, voce, lingua). I tensori bulk NON
   passano da qui.
2. Full-payload / async-chunk processor (stage 0, connector): trasporta gli hidden
   states dei token GENERATI dal thinker (chiave "v9_thinker_hidden") al talker.
3. Il talker Gemma4OmniV9TalkerForConditionalGeneration calcola il prefisso V9
   al primo prefill e lo inietta nel prompt: [ICL_text | h_pfx | BOS].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from vllm_omni_plugin.models.talker_v9 import V9_EXTRA_FRAMES, V9_HIDDEN_KEY

# Default voce/lingua per il talker (overridabili dal prompt originale).
DEFAULT_LANGUAGE = "Italian"


def _ensure_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return [x]


def _extract_voice_and_language(prompt: Any, index: int = 0) -> tuple[str | None, str]:
    """Estrae voce/lingua dal prompt originale (formato difensivo)."""
    voice: str | None = None
    language: str = DEFAULT_LANGUAGE
    if prompt is None:
        return voice, language
    if isinstance(prompt, list):
        prompt = prompt[index] if index < len(prompt) else None
    if prompt is None:
        return voice, language
    # Campi tipici: multi_modal_data / additional_information / dict puro.
    for container_name in ("multi_modal_data", "additional_information"):
        container = getattr(prompt, container_name, None)
        if isinstance(container, dict):
            v = container.get("voice") or container.get("speaker")
            if isinstance(v, str) and v.strip():
                voice = v.strip()
            lang = container.get("language")
            if isinstance(lang, str) and lang.strip():
                language = lang.strip()
    if isinstance(prompt, dict):
        v = prompt.get("voice") or prompt.get("speaker")
        if isinstance(v, str) and v.strip():
            voice = v.strip()
        lang = prompt.get("language")
        if isinstance(lang, str) and lang.strip():
            language = lang.strip()
    return voice, language


# ---------------------------------------------------------------------------
# Stima lunghezza placeholder (stesso codice del serving ufficiale)
# ---------------------------------------------------------------------------
_ESTIMATE_CACHE: dict[str, Any] = {}


def _estimate_components() -> tuple[Any, Any]:
    """(tokenizer, talker_config) del repo Qwen3-TTS, caricati una sola volta."""
    if "tok" not in _ESTIMATE_CACHE:
        from transformers import AutoTokenizer

        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig

        from vllm_omni_plugin.checkpoint import resolve_qwen3_tts_root

        repo = resolve_qwen3_tts_root("", None, None)
        _ESTIMATE_CACHE["tok"] = AutoTokenizer.from_pretrained(repo, use_fast=True)
        _ESTIMATE_CACHE["cfg"] = Qwen3TTSConfig.from_pretrained(repo).talker_config
    return _ESTIMATE_CACHE["tok"], _ESTIMATE_CACHE["cfg"]


def _estimate_qwen3tts_prompt_len(additional_information: dict[str, Any], task_type: str = "CustomVoice") -> int:
    """Stima ufficiale della lunghezza del prompt nativo (senza prefisso V9)."""
    try:
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
            Qwen3TTSPromptEmbedsBuilder,
        )

        tok, talker_config = _estimate_components()
        return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
            additional_information=additional_information,
            task_type=task_type,
            tokenize_prompt=lambda t: tok(t, padding=False)["input_ids"],
            codec_language_id=getattr(talker_config, "codec_language_id", None),
            spk_is_dialect=getattr(talker_config, "spk_is_dialect", None),
        )
    except Exception as exc:
        # Fallback conservativo: SOVRASTIMA (il modello padda con tts_pad_embed).
        text = (additional_information.get("text") or [""])[0]
        estimate = max(64, int(len(str(text)) * 6)) + 32
        try:
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "[V9] Stima prompt fallita (%s): fallback %d.", exc, estimate
            )
        except Exception:
            pass
        return estimate


# ---------------------------------------------------------------------------
# 1. Sync processor (stage 1 talker) — firma orchestrator
# ---------------------------------------------------------------------------
def thinker2talker_v9_token_only(
    source_outputs: list[Any],
    prompt: Any = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
) -> list[Any]:
    """Costruisce gli input dello stage 1 (talker) dagli output finiti del thinker.

    Placeholder [1]*ph_len (riserva KV) + additional_information con il testo
    della risposta. Il prefisso V9 (16 frame) + BOS richiedono V9_EXTRA_FRAMES
    posizioni extra rispetto alla stima nativa.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    talker_inputs: list[OmniTokensPrompt] = []
    for i, thinker_output in enumerate(source_outputs):
        if not getattr(thinker_output, "finished", False):
            continue
        output = thinker_output.outputs[0]
        reply_text = (
            getattr(output, "cumulative_text", None)
            or getattr(output, "text", None)
            or ""
        ).strip()
        if not reply_text:
            continue

        voice, language = _extract_voice_and_language(prompt, index=i)
        additional_information: dict[str, Any] = {
            "text": [reply_text],
            # VoiceDesign: unico task che replica il prefill del sistema
            # addestrato (interactive_gemma4_omni_live.py): tag think + lingua,
            # NESSUN speaker embedding (il modello 12Hz Base non ha spk_id
            # CustomVoice registrati).
            "task_type": ["VoiceDesign"],
            "language": [language],
            "non_streaming_mode": [True],
        }
        if voice:
            additional_information["speaker"] = [voice]

        ph_len = _estimate_qwen3tts_prompt_len(additional_information) + V9_EXTRA_FRAMES
        talker_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1] * ph_len,
                additional_information=additional_information,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return talker_inputs


# ---------------------------------------------------------------------------
# 2. Full-payload processor (stage 0 -> connector) — firma connector
# ---------------------------------------------------------------------------
def thinker2talker_v9_full_payload(
    transfer_manager: Any,
    pooling_output: dict[str, Any],
    request: Any,
) -> dict[str, Any] | None:
    """Pacchetta gli hidden states dei token generati dal thinker per il talker."""
    rid = getattr(request, "request_id", None)
    if not isinstance(pooling_output, Mapping):
        return None
    hidden = pooling_output.get("hidden")
    prompt_len = len(_ensure_list(getattr(request, "prompt_token_ids", None)))
    if not isinstance(hidden, torch.Tensor) or hidden.numel() == 0:
        # Nessun hidden state: il talker andrà in modalità text-only (fail-soft).
        return {"hidden_states": {}, "meta": {"finished": torch.tensor(True, dtype=torch.bool)}}
    generated = hidden[prompt_len:].contiguous() if hidden.shape[0] > prompt_len else hidden.contiguous()
    return {
        "hidden_states": {V9_HIDDEN_KEY: generated.detach().cpu()},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


# ---------------------------------------------------------------------------
# 3. Async-chunk processor (stage 0 -> connector streaming) — firma connector
# ---------------------------------------------------------------------------
# In modalità async_chunk il connector è l'UNICA fonte di input dello stage 1
# (orchestrator.py: _forward_to_next_stage viene saltato per gli stage che
# ricevono chunk): qui dobbiamo consegnare TUTTO — testo da sintetizzare,
# placeholder ids e hidden states V9.

_TALKER_TOKENIZER_CACHE: dict[str, Any] = {}


def _gemma_tokenizer():
    """Tokenizer Gemma (lazy, una sola volta per processo)."""
    if "tok" not in _TALKER_TOKENIZER_CACHE:
        import glob as _glob
        from transformers import AutoTokenizer

        snaps = sorted(
            _glob.glob(
                "/workspace/.hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/*"
            )
        )
        if not snaps:
            raise FileNotFoundError("Thinker Gemma non trovato per detokenizzazione.")
        _TALKER_TOKENIZER_CACHE["tok"] = AutoTokenizer.from_pretrained(snaps[-1], use_fast=True)
    return _TALKER_TOKENIZER_CACHE["tok"]


def thinker2talker_v9_async_chunk(
    transfer_manager: Any,
    multimodal_output: dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
):
    """Accumula hidden states del thinker per chunk; emette il payload alla fine.

    Il sender del connector esige un OmniPayloadStruct (forbid_unknown_fields):
    - ids.output  = token ids GENERATI dal thinker (vocab Gemma): il talker V9
      li detokenizza worker-side (tokenizer Gemma lazy) per la text conditioning.
    - hidden_states.output = hidden last-layer dei token generati (V9).
    - meta.replace_streaming_prompt + next_stage_prompt_len = placeholder KV.
    """
    from vllm_omni.data_entry_keys import HiddenStatesStruct, IdsStruct, MetaStruct, OmniPayloadStruct

    rid = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
    finished = bool(is_finished or request.is_finished())
    if isinstance(multimodal_output, Mapping):
        hs = multimodal_output.get("hidden_states")
        if isinstance(hs, torch.Tensor) and hs.numel() > 0:
            buf = transfer_manager.request_payload.setdefault(rid, [])
            buf.append(hs.detach().cpu())
    if not finished:
        return None

    chunks = transfer_manager.request_payload.pop(rid, [])
    if not chunks:
        return None
    full = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
    prompt_len = len(_ensure_list(getattr(request, "prompt_token_ids", None)))
    generated = full[prompt_len:].contiguous() if full.shape[0] > prompt_len else full.contiguous()
    gen_ids = _ensure_list(getattr(request, "all_token_ids", None))[prompt_len:]
    if not gen_ids:
        return None

    # Detokenizzazione (tokenizer Gemma lazy, una volta per processo) per la
    # stima ESATTA del placeholder e per la text conditioning worker-side.
    reply_text = ""
    try:
        reply_text = _gemma_tokenizer().decode(gen_ids, skip_special_tokens=True).strip()
    except Exception as exc:  # fail-soft
        try:
            from vllm.logger import init_logger

            init_logger(__name__).warning("[V9] Detokenizzazione fallita (%s).", exc)
        except Exception:
            pass
    if not reply_text:
        return None

    # Lunghezza placeholder del talker: stima ufficiale + V9 (16+1).
    try:
        ph_len = _estimate_qwen3tts_prompt_len(
            {"text": [reply_text], "task_type": ["VoiceDesign"], "language": [DEFAULT_LANGUAGE]}
        ) + V9_EXTRA_FRAMES
    except Exception:
        ph_len = int(len(gen_ids) * 1.6) + 64 + V9_EXTRA_FRAMES

    return OmniPayloadStruct(
        ids=IdsStruct(all=[1] * ph_len, prompt=[1] * ph_len, output=gen_ids),
        hidden_states=HiddenStatesStruct(output=generated),
        meta=MetaStruct(replace_streaming_prompt=True, next_stage_prompt_len=ph_len),
    )
