# -*- coding: utf-8 -*-
"""
duplex_kernel/engine.py — PipelineEngine (Thinker → V9 → Talker → Code2Wav).

Kernel narrow autocontenuto (ADR-029, ispirato a DwarfStar): carica i modelli
addestrati direttamente (bf16, niente unsloth/LoRA/bnb per il thinker) e
orchestra la pipeline in un processo singolo con KV persistente per sessione.

Flusso (ADR-022/023/024):
1. Thinker (Gemma 4 E2B bf16) genera testo + hidden states (1536).
2. Hidden → V9: h_tx (media) + h_raw_style (16 frame) → causal_conv → h_pfx
   (1, 16, 2048). h_pfx è anche h_t per la turnhead (media sui frame).
3. Talker Qwen3-TTS: prefill [ICL | h_pfx | BOS] (prefix KV persistente),
   loop autoregressivo CB0 (top_k=20, T=0.6) + sub-talker (top_k=20, T=0.8).
4. Code2Wav: RVQ → PCM 24kHz, trim + peak norm.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from duplex_kernel.models.talker import load_talker
from duplex_kernel.models.thinker import load_thinker
from duplex_kernel.models.v9 import load_v9_conditioning

logger = logging.getLogger("duplex_kernel.engine")


class PipelineEngine:
    """Engine a 3 stadi: Thinker + V9 + Talker(Qwen3-TTS) + Code2Wav."""

    def __init__(self, *, device: str = "cuda", dtype: torch.dtype = torch.bfloat16) -> None:
        self.device = device
        self.dtype = dtype
        self.thinker = None
        self._thinker_wrapper = None
        self.tokenizer = None
        self.talker = None
        self.v9 = None
        self.causal_conv = None
        self.subtalker_graph: Any | None = None
        self.residual_sampler: Any | None = None
        self._mtp_ready = False  # MTP parallelo: disattivato (fallback generate HF bit-exact)
        self.loaded = False
        self._stop_event = False
        torch.set_float32_matmul_precision("high")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # ------------------------------------------------------------------
    def load(self, *, workspace: str = "/workspace") -> bool:
        try:
            t0 = time.time()
            th = load_thinker(workspace, device=self.device, dtype=self.dtype)
            self._thinker_wrapper = th
            self.thinker = th.model
            self.tokenizer = th.tokenizer
            logger.info("[engine] thinker in %.1fs", time.time() - t0)

            self.talker = load_talker(workspace, device=self.device)
            logger.info("[engine] talker+code2wav in %.1fs", time.time() - t0)

            self.v9, self.causal_conv = load_v9_conditioning(device=self.device, dtype=self.dtype)
            logger.info("[engine] V9 in %.1fs", time.time() - t0)

            # CUDA graph sub-talker (ADR-026): prefill + 14 step autoregressivi
            # catturati in CUDA graph statici, sampling esterno bit-exact con RNG
            # globale (warper T->top_k->top_p come transformers). Fail-soft: eager.
            from duplex_kernel.subtalker_graph import SubtalkerGraph

            cp = self.talker.talker.code_predictor
            self.subtalker_graph = SubtalkerGraph(cp, talker_embed=self.talker.talker.get_input_embeddings()).capture()
            logger.info("[engine] subtalker graph: %s", "catturato" if self.subtalker_graph.captured else "eager")

            self.loaded = True
            logger.info("[engine] LOAD COMPLETO in %.1fs", time.time() - t0)
            return True
        except Exception as exc:
            logger.exception("Engine load fallito: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Thinker
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def think(self, prompt_token_ids: list[int], *, max_new_tokens: int = 64,
              temperature: float = 0.7, top_p: float = 0.9) -> dict[str, Any]:
        input_ids = torch.tensor([prompt_token_ids], device=self.device)
        out = self.thinker.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p
        )
        gen_ids = out[0, input_ids.shape[1]:].cpu().tolist()
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if not gen_ids:
            raise RuntimeError("Thinker ha prodotto una risposta vuota")
        # hidden states: il wrapper ThinkerModel espone hidden_states(input_ids)
        if isinstance(self.thinker, torch.nn.Module):
            with torch.no_grad():
                full = self.thinker(out, output_hidden_states=True, return_dict=True)
                hidden = full.hidden_states[-1]
        else:
            hidden = self.thinker.hidden_states(out)
        return {"text": text, "gen_ids": gen_ids, "hidden": hidden, "prompt_len": int(input_ids.shape[1])}

    def transcribe_audio(self, pcm: np.ndarray, *, max_new_tokens: int = 64) -> dict[str, Any]:
        if self._thinker_wrapper is None:
            raise RuntimeError("engine non caricato")
        return self._thinker_wrapper.transcribe_audio(pcm, max_new_tokens=max_new_tokens)

    # ------------------------------------------------------------------
    # V9 → h_pfx (16×2048) — h_t per la turnhead = media di h_pfx
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def v9_prefix(self, hidden: torch.Tensor, prompt_len: int) -> torch.Tensor:
        lh = hidden  # (1, L, 1536)
        gen = lh[:, prompt_len:, :]
        h_tx = gen.mean(dim=1, keepdim=True).to(self.dtype)
        T_pfx = 16
        seq_trans = gen.transpose(1, 2)
        h_raw_style = F.interpolate(seq_trans, size=T_pfx, mode="linear", align_corners=False).transpose(1, 2)
        cc = self.causal_conv(h_raw_style.to(self.dtype))
        return self.v9(cc, h_tx)  # (1, 16, 2048)

    # ------------------------------------------------------------------
    # Talker
    # ------------------------------------------------------------------
    def _cb0_sample(self, logits: torch.Tensor, temperature: float = 0.6) -> int:
        # logits può essere (B, vocab) o (B, T, vocab): prendi l'ultimo passo.
        lg = logits[..., -1, :].clone() if logits.dim() >= 3 else logits[0].clone()
        lg = lg[0] if lg.dim() >= 2 else lg
        lg[2048:] = -float("inf")
        topk_vals, topk_idx = torch.topk(lg, k=20, dim=-1)
        filtered = torch.full_like(lg, -float("inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)
        probs = F.softmax(filtered / temperature, dim=-1)
        return int(torch.multinomial(probs, 1).item())

    @torch.no_grad()
    def stop_generation(self) -> None:
        """Richiede l'arresto al prossimo frame; sicuro da chiamare dal WS."""
        self._stop_event = True

    @torch.inference_mode()
    def talk(self, text: str, h_pfx: torch.Tensor, *, n_words: int,
             voice_conditioning: dict[str, Any] | None = None,
             stop_check: Any | None = None) -> np.ndarray:
        self._stop_event = False
        if voice_conditioning is not None:
            # Il builder nativo può consumare questo payload; lo manteniamo sul
            # wrapper per non duplicare il protocollo Qwen nel loop caldo.
            self.talker.voice_conditioning = voice_conditioning
        t = self.talker.talker
        tcfg = self.talker.tcfg
        T_gen = max(48, min(140, int(n_words * 4.8) + 12))
        icl_part, bos_part, tts_pad_embed = self.talker.build_prefill_embeddings(text)
        pfx = torch.cat([icl_part, h_pfx.to(self.dtype), bos_part], dim=1)
        out_pfx = t.model(inputs_embeds=pfx, use_cache=True)
        past_kv = out_pfx.past_key_values
        last_h = out_pfx.last_hidden_state[:, -1:, :]
        codes: list[list[int]] = []
        for _ in range(T_gen):
            if self._stop_event or (stop_check is not None and stop_check()):
                break
            lg = t.codec_head(last_h)[:, -1, :]
            c0 = self._cb0_sample(lg)
            # Sub-talker: CUDA graph (ADR-026, 15 residui in ~12ms) oppure
            # fallback eager bit-exact (stesso warper order di transformers).
            c0_embed = t.get_input_embeddings()(torch.tensor([[c0]], device=self.device))
            resid = self.subtalker_graph.generate(last_h, c0, c0_embed=c0_embed)
            codes.append([c0] + resid)
            if len(codes) >= T_gen:
                break
            cur = torch.tensor(codes[-1], device=self.device)
            agg = t.get_input_embeddings()(cur[0:1].unsqueeze(0))
            for g in range(1, 16):
                agg = agg + t.code_predictor.get_input_embeddings()[g - 1](cur[g:g + 1].unsqueeze(0))
            nxt = agg.transpose(0, 1) + tts_pad_embed
            out_step = t.model(inputs_embeds=nxt, past_key_values=past_kv, use_cache=True)
            past_kv = out_step.past_key_values
            last_h = out_step.last_hidden_state[:, -1:, :]

        if not codes:
            return np.zeros(0, dtype=np.float32)
        t_gen = torch.clamp(torch.tensor(codes, device=self.device, dtype=torch.long).unsqueeze(0), 0, 2047)
        wav = self.talker.decode(t_gen).numpy().astype(np.float32)
        return self._postprocess(wav)

    @staticmethod
    def _postprocess(wav: np.ndarray) -> np.ndarray:
        """Trim coda + fade + peak norm (ricetta benchmark_real_stream_turnhead).

        Cerca l'ultimo chunk con energia significativa scorrendo dall'indietro
        e tiene +3 chunk (240ms) dopo la voce; se l'audio è tutto rumore
        (energia costante alta) il loop non trova silenzio e mantiene tutto.
        """
        sr = 24000
        chunk = int(0.080 * sr)
        energies = [np.sqrt(np.mean(wav[i:i + chunk] ** 2)) for i in range(0, len(wav), chunk)]
        end = len(energies)
        # Soglia adattiva: media dei chunk bassi (fondo) o 0.015
        floor = max(0.015, float(np.percentile(energies, 15)) * 0.8)
        for i in range(len(energies) - 1, 8, -1):
            if energies[i] > floor:
                end = min(len(energies), i + 3)
                break
        wav = wav[: end * chunk].copy()
        if len(wav) > 1200:
            fade = int(0.030 * sr)
            wav[-fade:] *= np.linspace(1.0, 0.0, fade)
        p = float(np.percentile(np.abs(wav), 99.7))
        return np.clip(wav / (p * 1.05), -1.0, 1.0) if p > 1e-6 else wav

    def unload(self) -> None:
        self.loaded = False
        self.subtalker_graph = None
        self.thinker = self._thinker_wrapper = self.talker = self.v9 = self.causal_conv = None
        torch.cuda.empty_cache()


__all__ = ["PipelineEngine"]
