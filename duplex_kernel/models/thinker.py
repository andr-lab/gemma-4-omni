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
from typing import Any

import torch

logger = logging.getLogger("duplex_kernel.thinker")


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

    @torch.no_grad()
    def transcribe_audio(self, pcm: np.ndarray, *, max_new_tokens: int = 64) -> dict[str, Any]:
        """ASR nativo E2B: PCM -> mel Gemma -> prefill multimodale + decode KV.

        Il decode è esplicito: non ripassa input_features a ``generate`` a ogni
        token, evitando il percorso multimodale incompatibile con alcuni loader
        4-bit di Transformers 5.
        """
        if self.processor is None:
            raise RuntimeError("processor audio Gemma non caricato")
        pcm = np.asarray(pcm, dtype=np.float32)
        audio_marker = getattr(self.processor, "audio_token", "<|audio|>")
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": "Trascrivi fedelmente il parlato nell'audio."}, {"type": "audio"}]}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt = chat.replace("<|audio|>", audio_marker)
        feats = self.processor(text=[prompt], audio=[pcm], return_tensors="pt", padding=True)
        ids = feats["input_ids"].to(self.device)
        features = feats["input_features"].to(self.device)
        feature_mask = feats["input_features_mask"].to(self.device)
        attn = feats.get("attention_mask", torch.ones_like(ids)).to(self.device)
        eoa_id = int(self.tokenizer.convert_tokens_to_ids(getattr(self.processor, "eoa_token", "<audio|>")))
        out = self.model(input_ids=ids, attention_mask=attn, input_features=features,
                         input_features_mask=feature_mask, use_cache=True, return_dict=True)
        past = out.past_key_values
        token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated: list[int] = []
        stop_ids = {self.tokenizer.eos_token_id, eoa_id}
        for _ in range(max_new_tokens):
            value = int(token.item())
            if value in stop_ids:
                break
            generated.append(value)
            attn = torch.cat([attn, torch.ones((1, 1), device=self.device, dtype=attn.dtype)], dim=1)
            out = self.model(input_ids=token, attention_mask=attn, past_key_values=past, use_cache=True, return_dict=True)
            past = out.past_key_values
            token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        return {"text": self.tokenizer.decode(generated, skip_special_tokens=True).strip(), "gen_ids": generated}

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
