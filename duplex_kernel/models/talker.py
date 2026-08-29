# -*- coding: utf-8 -*-
"""
duplex_kernel/models/talker.py — Loader Talker Qwen3-TTS + Code2Wav.

Talker: qwen_tts.core.models.Qwen3TTSForConditionalGeneration (pesi HF locali).
Code2Wav: Qwen3TTSTokenizerV2Model (vocoder 24kHz).

Autocontenuto: shim torchaudio/ROPE necessari per l'import di qwen_tts in
transformers 5.x (stesso pattern dei benchmark di riferimento), niente
dipendenze dalla pipeline di training.
"""

from __future__ import annotations

import glob
import logging
import os
import sys
import types
import importlib.machinery
from typing import Any

import torch

logger = logging.getLogger("duplex_kernel.talker")


def _install_shims(workspace: str) -> None:
    """Shim torchaudio + ROPE per l'import di qwen_tts in transformers 5.x."""
    try:
        import torchaudio  # noqa: F401
    except Exception:
        _m = types.ModuleType("torchaudio")
        _m.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)
        sys.modules["torchaudio"] = _m
        sys.modules["torchaudio._extension"] = types.ModuleType("torchaudio._extension")
        sys.modules["torchaudio.compliance"] = types.ModuleType("torchaudio.compliance")
        sys.modules["torchaudio.compliance.kaldi"] = types.ModuleType("torchaudio.compliance.kaldi")
    sys.modules["sox"] = types.ModuleType("sox")

    import transformers.modeling_rope_utils as _ropeu

    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        _head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        _dim = int(_head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, _dim, 2, dtype=torch.int64).float().to(device) / _dim))
        return inv_freq, 1.0

    _ropeu.ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    _ropeu.ROPE_INIT_FUNCTIONS["interleaved"] = _default_rope_init
    sys.path.insert(0, workspace)
    sys.path.insert(0, os.path.join(workspace, "qwen3tts_src"))


class TalkerModel:
    """Talker Qwen3-TTS + Code2Wav (pesi HF locali)."""

    def __init__(self, talker: Any, proc: Any, code2wav: Any, cfg: Any, tcfg: Any, *, device: str) -> None:
        self.talker = talker
        self.proc = proc
        self.code2wav = code2wav
        self.cfg = cfg
        self.tcfg = tcfg
        self.device = device

    def build_prefill_parts(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Restituisce prefisso stabile, parte dinamica, BOS e TTS-PAD."""
        t = self.talker
        cfg, tcfg = self.cfg, self.tcfg
        raw_ids = self.proc(text=text, return_tensors="pt")["input_ids"].to(self.device)
        bos_e, eos_e, pad_e = t.text_projection(
            t.get_text_embeddings()(
                torch.tensor(
                    [[cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id]],
                    device=self.device,
                    dtype=raw_ids.dtype,
                )
            )
        ).chunk(3, dim=1)
        codec_prefill = [[tcfg.codec_think_id, tcfg.codec_think_bos_id, 2070, tcfg.codec_think_eos_id]]
        c0 = t.get_input_embeddings()(torch.tensor(codec_prefill, device=self.device, dtype=raw_ids.dtype))
        c1 = t.get_input_embeddings()(torch.tensor([[tcfg.codec_pad_id, tcfg.codec_bos_id]], device=self.device, dtype=raw_ids.dtype))
        codec_emb = torch.cat([c0, c1], dim=1)
        role = self.proc(text="<|im_start|>assistant\n", return_tensors="pt")["input_ids"].to(self.device)
        role_emb = t.text_projection(t.get_text_embeddings()(role))
        icl_core = torch.cat((pad_e.expand(-1, codec_emb.shape[1] - 2, -1), bos_e), dim=1) + codec_emb[:, :-1]
        icl_part = torch.cat((role_emb, icl_core), dim=1)
        text_emb = t.text_projection(t.get_text_embeddings()(raw_ids))
        cpad = t.get_input_embeddings()(
            torch.tensor([[tcfg.codec_pad_id] * (raw_ids.shape[1] + 1)], device=self.device, dtype=raw_ids.dtype)
        )
        text_part = torch.cat([text_emb, eos_e], dim=1) + cpad
        bos_part = pad_e + t.get_input_embeddings()(
            torch.tensor([[tcfg.codec_bos_id]], device=self.device, dtype=raw_ids.dtype)
        )
        return icl_part, text_part, bos_part, pad_e

    def build_prefill_embeddings(self, text: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compatibilità: embedding completo del prefill ufficiale."""
        icl_part, text_part, bos_part, pad_e = self.build_prefill_parts(text)
        return torch.cat([icl_part, text_part], dim=1), bos_part, pad_e

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """RVQ codes → waveform 24kHz."""
        return self.code2wav.decode(codes).audio_values[0].detach().cpu()


def load_talker(workspace: str = "/workspace", *, device: str = "cuda") -> TalkerModel:
    _install_shims(workspace)
    import qwen_tts.core.models as _M
    import safetensors.torch as st
    from transformers import AutoTokenizer
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

    snaps = sorted(glob.glob(os.path.join(workspace, ".hf_cache/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/*")))
    if not snaps:
        raise FileNotFoundError("Talker Qwen3-TTS non trovato")
    snap = snaps[-1]
    _cfg = _M.Qwen3TTSConfig.from_pretrained(snap)
    _tc = _cfg.talker_config
    if not hasattr(_tc, "pad_token_id"):
        object.__setattr__(_tc, "pad_token_id", None)
    teacher = _M.Qwen3TTSForConditionalGeneration(_cfg).to(torch.bfloat16)
    _sd = st.load_file(os.path.join(snap, "model.safetensors"), device="cpu")
    teacher.load_state_dict(_sd, strict=False)
    del _sd
    torch.cuda.empty_cache()
    teacher = teacher.to(device=device, dtype=torch.bfloat16).eval()
    tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
    proc = _M.Qwen3TTSProcessor(tokenizer=tok)
    snaps_tok = sorted(
        glob.glob(os.path.join(workspace, ".hf_cache/models--Qwen--Qwen3-TTS-Tokenizer-12Hz/snapshots/*"))
    )
    if not snaps_tok:
        raise FileNotFoundError("Code2Wav (TokenizerV2) non trovato")
    code2wav = Qwen3TTSTokenizerV2Model.from_pretrained(snaps_tok[-1]).to(device).eval()
    logger.info("[talker] Qwen3-TTS + Code2Wav caricati")
    return TalkerModel(teacher.talker, proc, code2wav, teacher.config, _cfg.talker_config, device=device)


__all__ = ["TalkerModel", "load_talker"]
