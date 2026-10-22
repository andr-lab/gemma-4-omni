# -*- coding: utf-8 -*-
"""
duplex_kernel/models/thinker.py — Loader Thinker Gemma 4 E2B (bf16 nativo).

Niente Unsloth, niente LoRA, niente bitsandbytes: i safetensors del checkpoint
sono BF16 puri (verificato: embed_tokens 262144×1536 BF16). Carichiamo il
modello con transformers nativo in bfloat16, attiviamo la raccolta hidden
states per il V9, e gestiamo il forward in modo ottimizzato.

Config base: unsloth/gemma-4-e2b-it-unsloth-bnb-4bit (pesi BF16 nativi,
model_type=gemma4, hidden_size=1536). Il modello è istruito (it) — niente
adattatori addestrati necessari per il serving conversazionale.
"""

from __future__ import annotations

import glob
import logging
import numpy as np
import os
import time
from numbers import Integral
from typing import Any, Callable

import torch

logger = logging.getLogger("duplex_kernel.thinker")


def normalize_eos_token_ids(value: Any) -> set[int]:
    """Normalize Transformers' scalar or sequence EOS configuration."""
    if isinstance(value, Integral):
        return {int(value)}
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if item is not None}
    if value is None:
        return set()
    raise TypeError(f"eos_token_id deve essere int o sequenza di int, ricevuto {type(value).__name__}")


def resolve_thinker_path(workspace: str = "/workspace") -> str:
    # Il checkpoint ufficiale BF16 include lo stato completo dell'audio tower.
    # I mirror BNB 4-bit di Unsloth omettono i quant_state dell'audio tower e
    # non sono eseguibili dal percorso multimodale di Transformers 5.
    official = os.path.join(workspace, ".hf_cache/models--google--gemma-4-E2B-it/full")
    if os.path.isfile(os.path.join(official, "model.safetensors")):
        return official
    snaps = sorted(
        glob.glob(os.path.join(workspace, ".hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/*"))
    )
    if snaps:
        return snaps[-1]
    snaps = sorted(glob.glob(os.path.join(workspace, ".hf_cache/models--google--gemma-4-E2B-it/snapshots/*")))
    if snaps:
        return snaps[-1]
    raise FileNotFoundError("Thinker Gemma 4 E2B non trovato in .hf_cache")


class ThinkerModel:
    """Wrapper del thinker: forward con hidden states, generate, V9 hook."""

    def __init__(self, model: Any, tokenizer: Any, processor: Any | None = None, *, device: str = "cuda") -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.device = device
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int, temperature: float, top_p: float) -> torch.Tensor:
        return self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.pad_token_id,
        )

    @torch.inference_mode()
    def transcribe_audio(
        self,
        pcm: np.ndarray,
        *,
        max_new_tokens: int = 64,
        decode_step: Callable[[Callable[[], Any]], Any] | None = None,
        prefill_step: Callable[[Callable[[], Any]], Any] | None = None,
        past_key_values: Any | None = None,
        decode_forward: Callable[[torch.Tensor, Any, torch.Tensor], Any] | None = None,
        static_cache_max_len: int | None = None,
        stop_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """ASR nativo E2B: PCM -> mel Gemma -> prefill multimodale + decode KV.

        Il decode è esplicito: non ripassa input_features a ``generate`` a ogni
        token, evitando il percorso multimodale incompatibile con alcuni loader
        4-bit di Transformers 5. ``decode_step`` può serializzare ogni singolo
        forward autoregressivo senza trattenere il lock durante l'audio-prefill.
        """
        if self.processor is None:
            raise RuntimeError("processor audio Gemma non caricato")
        def cancellation_result(
            *,
            feature_prepare_ms: float,
            audio_prefill_ms: float = 0.0,
            decode_ms: float = 0.0,
            decode_steps: int = 0,
        ) -> dict[str, Any]:
            timings = {
                "feature_prepare_ms": round(feature_prepare_ms, 3),
                "audio_prefill_ms": round(audio_prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "decode_steps": int(decode_steps),
                "terminal_token_id": None,
                "terminal_reason": "cancelled",
            }
            return {
                "text": "",
                "gen_ids": [],
                "cancelled": True,
                "cache_info": {},
                "phase_timings_ms": timings,
                "terminal_token_id": None,
                "terminal_reason": "cancelled",
            }

        if stop_check is not None and stop_check():
            return cancellation_result(feature_prepare_ms=0.0)
        feature_prepare_start = time.perf_counter()
        pcm = np.asarray(pcm, dtype=np.float32)
        audio_marker = (
            getattr(self.processor, "audio_token", None)
            or getattr(self.tokenizer, "audio_token", None)
            or "<|audio|>"
        )
        from duplex_kernel.asr_prompt import asr_prompt_content

        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": asr_prompt_content("Italian", audio_marker)}],
            tokenize=False, add_generation_prompt=True,
        )
        feats = self.processor(
            text=[chat],
            audio=[pcm],
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        ids = feats["input_ids"].to(self.device)
        features = feats["input_features"].to(self.device)
        feature_mask = feats["input_features_mask"].to(self.device)
        attn = feats.get("attention_mask", torch.ones_like(ids)).to(self.device)
        eoa_id = int(self.tokenizer.convert_tokens_to_ids(getattr(self.processor, "eoa_token", "<audio|>")))
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        feature_prepare_ms = (time.perf_counter() - feature_prepare_start) * 1000.0
        if stop_check is not None and stop_check():
            return cancellation_result(feature_prepare_ms=feature_prepare_ms)

        audio_prefill_start = time.perf_counter()
        if static_cache_max_len is not None:
            if past_key_values is None:
                raise ValueError("static_cache_max_len richiede un Cache ASR esplicita")
            if int(ids.shape[-1]) + int(max_new_tokens) > int(static_cache_max_len):
                raise ValueError(
                    "prompt ASR + max_new_tokens supera la capacità della cache statica "
                    f"({int(ids.shape[-1])} + {int(max_new_tokens)} > {int(static_cache_max_len)})"
                )
        prefill_kwargs = {
            "input_ids": ids,
            "attention_mask": attn,
            "input_features": features,
            "input_features_mask": feature_mask,
            "use_cache": True,
            "logits_to_keep": 1,
            "return_dict": True,
        }
        if past_key_values is not None:
            prefill_kwargs["past_key_values"] = past_key_values
        audio_prefill = lambda: self.model(**prefill_kwargs)
        out = prefill_step(audio_prefill) if prefill_step is not None else audio_prefill()
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        audio_prefill_ms = (time.perf_counter() - audio_prefill_start) * 1000.0
        if out is None or (stop_check is not None and stop_check()):
            return cancellation_result(
                feature_prepare_ms=feature_prepare_ms,
                audio_prefill_ms=audio_prefill_ms,
            )
        past = out.past_key_values
        if past is None:
            raise RuntimeError("prefill audio Gemma non ha restituito una cache KV")
        prefill_cache_length = int(past.get_seq_length()) if callable(getattr(past, "get_seq_length", None)) else None
        if static_cache_max_len is not None:
            if prefill_cache_length is None:
                raise RuntimeError("StaticCache ASR non espone get_seq_length()")
            if prefill_cache_length != int(attn.shape[-1]):
                raise RuntimeError(
                    "StaticCache ASR e attention_mask non sono allineate dopo il prefill "
                    f"(cache={prefill_cache_length}, mask={int(attn.shape[-1])})"
                )
            if prefill_cache_length + int(max_new_tokens) > int(static_cache_max_len):
                raise ValueError(
                    "prefill ASR + max_new_tokens supera la capacità della cache statica "
                    f"({prefill_cache_length} + {int(max_new_tokens)} > {int(static_cache_max_len)})"
                )
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated: list[int] = []
        generation_eos = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        model_eos = getattr(getattr(self.model, "config", None), "eos_token_id", None)
        if generation_eos is not None:
            model_stop_ids = normalize_eos_token_ids(generation_eos)
        elif model_eos is not None:
            model_stop_ids = normalize_eos_token_ids(model_eos)
        else:
            model_stop_ids = normalize_eos_token_ids(getattr(self.tokenizer, "eos_token_id", None))
        stop_ids = model_stop_ids | {eoa_id}
        terminal_token_id: int | None = None
        terminal_reason = "budget_exhausted"
        decode_ms = 0.0
        for _ in range(max_new_tokens):
            if stop_check is not None and stop_check():
                terminal_reason = "cancelled"
                break
            value = int(token.item())
            if value in stop_ids:
                terminal_token_id = value
                terminal_reason = "audio_end" if value == eoa_id else "eos"
                break
            generated.append(value)
            attn = torch.cat([attn, torch.ones((1, 1), device=self.device, dtype=attn.dtype)], dim=1)

            def forward_decode_step() -> Any:
                if decode_forward is not None:
                    return decode_forward(token, past, attn)
                step_output = self.model(
                    input_ids=token,
                    attention_mask=attn,
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                if str(self.device).startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                return step_output

            decode_start = time.perf_counter()
            out = forward_decode_step() if decode_step is None else decode_step(forward_decode_step)
            decode_ms += (time.perf_counter() - decode_start) * 1000.0
            if out is None or (stop_check is not None and stop_check()):
                terminal_reason = "cancelled"
                break
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        if terminal_reason == "cancelled":
            return cancellation_result(
                feature_prepare_ms=feature_prepare_ms,
                audio_prefill_ms=audio_prefill_ms,
                decode_ms=decode_ms,
                decode_steps=len(generated),
            )
        return {
            "text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "gen_ids": generated,
            "cache_info": {
                "cache_type": type(past).__name__,
                "prefill_sequence_length": prefill_cache_length,
                "capacity_tokens": static_cache_max_len,
                "compiled_q1_decode": decode_forward is not None,
            },
            "phase_timings_ms": {
                "feature_prepare_ms": round(feature_prepare_ms, 3),
                "audio_prefill_ms": round(audio_prefill_ms, 3),
                "decode_ms": round(decode_ms, 3),
                "decode_steps": len(generated),
                "terminal_token_id": terminal_token_id,
                "terminal_reason": terminal_reason,
            },
            "terminal_token_id": terminal_token_id,
            "terminal_reason": terminal_reason,
        }

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Ultimo layer hidden per la sequenza completa (prompt + generati)."""
        out = self.model(input_ids, output_hidden_states=True, return_dict=True)
        return out.hidden_states[-1]  # (1, L, 1536)


def load_thinker(
    workspace: str = "/workspace",
    *,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_seq_len: int = 8192,
) -> ThinkerModel:
    """Carica il thinker in bf16 nativo (senza unsloth/LoRA/bnb)."""
    import transformers

    path = resolve_thinker_path(workspace)
    logger.info("[thinker] caricamento da %s (bf16, no unsloth)", path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=dtype,
        device_map="cuda" if (str(device).startswith("cuda") or (isinstance(device, torch.device) and device.type == "cuda")) else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    # Preserve the exact source path for native listener checkpoint
    # attestation; Transformers does not reliably retain `_name_or_path` on
    # the nested Gemma language model after loading.
    setattr(model, "_gemma4_checkpoint_path", os.path.abspath(path))
    nested_model = getattr(model, "model", None)
    if nested_model is not None:
        setattr(nested_model, "_gemma4_checkpoint_path", os.path.abspath(path))
        language_model = getattr(nested_model, "language_model", None)
        if language_model is not None:
            setattr(language_model, "_gemma4_checkpoint_path", os.path.abspath(path))
    for p in model.parameters():
        p.requires_grad = False
    logger.info("[thinker] caricato (%.2f GB)", sum(p.numel() for p in model.parameters()) * 2 / 1e9)
    try:
        processor = transformers.AutoProcessor.from_pretrained(path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("[thinker] processor audio non disponibile: %s", exc)
        processor = None
    return ThinkerModel(model, tokenizer, processor, device=device)


__all__ = ["ThinkerModel", "load_thinker", "resolve_thinker_path"]
