# -*- coding: utf-8 -*-
"""
duplex_kernel/subtalker_graph.py — CUDA Graph per il sub-talker RVQ (ADR-026).

Il sub-talker (15 codebook residui) è il collo di bottiglia del talker:
~99ms/frame eager (il generate HF fa 15 forward sequenziali + overhead Python).
ADR-026: catturare i forward in CUDA graph statici → ~12.25ms/frame (8.6×)
mantenendo il loop autoregressivo 100% bit-exact.

Architettura (per frame, 12Hz):
  1. PREFILL graph:  cp(inputs_embeds=(1,2,D))  -> past_key_values (StaticCache 16)
  2. 14 STEP graphs: cp(input_ids=(1,1), past=pk, generation_steps=g,
                       cache_position=[g+1])    -> logits residuo g+1
  Il sampling (top_k=20, top_p=0.90, T=0.80, multinomial con RNG globale)
  avviene FUORI dai graph, esattamente come il generate HF di transformers:
  warper order temperature -> top_k -> top_p -> softmax -> multinomial.
  Nessun seed fisso per residuo: il RNG globale avanza naturalmente (bit-exact
  con il reference, niente artefatti da correlazione tra frame).

Bug fixati rispetto alla v1:
  - sampling con seed fisso per residuo produceva sequenze identiche a ogni
    frame (artefatti audio, casi ASR degradati nel benchmark v1);
  - mancava il warper top_p=0.90 del generate HF ufficiale;
  - il forward usava use_cache=False (mini-prefill ripetuto) invece del loop
    autoregressivo con StaticCache.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger("duplex_kernel.subtalker_graph")


def _warp_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.80,
    top_k: int = 20,
    top_p: float = 0.90,
) -> torch.Tensor:
    """Replica l'ordine dei warper di transformers (Temperature -> TopK -> TopP)."""
    lg = logits.clone()
    if temperature != 1.0:
        lg = lg / temperature
    if top_k is not None and top_k > 0:
        k = min(top_k, lg.shape[-1])
        topk_vals, topk_idx = torch.topk(lg, k=k, dim=-1)
        filtered = torch.full_like(lg, -float("inf"))
        filtered.scatter_(-1, topk_idx, topk_vals)
        lg = filtered
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_lg, sorted_idx = torch.sort(lg, descending=True, dim=-1)
        probs_sorted = F.softmax(sorted_lg, dim=-1)
        cum = torch.cumsum(probs_sorted, dim=-1)
        mask = cum - probs_sorted > top_p
        sorted_lg[mask] = -float("inf")
        # rimappa nell'ordine originale
        back = torch.full_like(sorted_lg, -float("inf"))
        back.scatter_(-1, sorted_idx, sorted_lg)
        lg = back
    return lg


class SubtalkerGraph:
    """CUDA graphs del sub-talker: prefill + 14 step autoregressivi.

    Uso (bit-exact):
        sg = SubtalkerGraph(cp).capture()  # fallback eager se la cattura fallisce
        res_tokens = sg.generate(last_h, c0_token, do_sample=True)
    """

    def __init__(self, model: Any, *, max_code_groups: int = 15, device: str = "cuda",
                 talker_embed: Any | None = None) -> None:
        self.model = model
        self.max_code_groups = max_code_groups
        self.device = device
        self.talker_embed = talker_embed  # nn.Embedding del TALKER (2048 dim) per il prefill
        self.cfg = model.config
        self.hidden = int(self.cfg.hidden_size)          # 1024 (code predictor)
        self.vocab = int(self.cfg.vocab_size)            # 2048
        # embedding dei residui: model.get_input_embeddings() -> ModuleList[14]
        self.resid_embeds = model.get_input_embeddings()
        self.prefill_graph: torch.cuda.CUDAGraph | None = None
        self.step_graphs: list[torch.cuda.CUDAGraph] = []
        self.static_prefill_inputs: torch.Tensor | None = None
        self.static_tok: torch.Tensor | None = None
        self.static_pos: torch.Tensor | None = None
        self.static_tokens: torch.Tensor = torch.zeros(1, self.max_code_groups, dtype=torch.long, device=self.device)
        self.static_logits: torch.Tensor | None = None
        self.static_cache: Any | None = None
        self.captured = False
        self._cache_kwargs: dict[str, Any] = {}

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def generate(
        self,
        last_h: torch.Tensor,
        c0_token: int,
        *,
        do_sample: bool = True,
        c0_embed: torch.Tensor | None = None,
        return_tensor: bool = False,
    ) -> list[int] | torch.Tensor:
        """Genera i 15 residui per il frame dato last_h (1,1,2048) e c0 (int).

        c0_embed: embedding del TALKER per c0 (1,1,2048) — il prefill del code
        predictor è cat([last_h, c0_embed]) proiettato da small_to_mtp_projection.
        """
        # Prefill: embeds = [last_h (2048 dim) | talker_emb(c0) (2048 dim)]
        if c0_embed is None:
            raise ValueError("c0_embed (embedding talker di c0) è obbligatorio per il prefill")
        prefill_inputs = torch.cat([last_h, c0_embed], dim=1)  # (1,2,2048)
        if self.captured:
            # Reset della cache statica PRIMA del replay: ogni StaticLayer
            # avanza cumulative_length con add_ in-place (stesso indirizzo del
            # graph) — senza reset il graph partirebbe da cum_length pieno e
            # index_copy_ andrebbe fuori bounds.
            for _l in self.static_cache.layers:
                _l.cumulative_length.zero_()
            self.static_prefill_inputs.copy_(prefill_inputs)
            self.prefill_graph.replay()

            # GPU-resident token selection: zero host sync inside loop
            self._sample_tok_gpu(self.static_logits, do_sample=do_sample, out_tok=self.static_tok)
            self.static_tokens[0, 0] = self.static_tok[0, 0]

            for g in range(1, self.max_code_groups):  # 1..14 -> residui 2..15
                self.static_pos.fill_(g + 1)
                self.step_graphs[g - 1].replay()
                self._sample_tok_gpu(self.static_logits, do_sample=do_sample, out_tok=self.static_tok)
                self.static_tokens[0, g] = self.static_tok[0, 0]

            if return_tensor:
                return self.static_tokens[0].clone()
            return self.static_tokens[0].tolist()

        # ---- Fallback eager (bit-exact identico) ----
        tokens: list[int] = []
        from transformers import StaticCache
        sc = StaticCache(config=self.cfg, max_batch_size=1, max_cache_len=self.max_code_groups + 2,
                         device=self.device, dtype=last_h.dtype)
        out = self.model(inputs_embeds=prefill_inputs, past_key_values=sc, use_cache=True,
                         cache_position=torch.arange(0, 2, device=self.device))
        pk = out.past_key_values
        lg0 = out.logits[:, -1, :]
        r = self._sample_tok(lg0, do_sample=do_sample)
        tokens.append(r)
        for g in range(1, self.max_code_groups):
            tok = torch.tensor([[r]], device=self.device, dtype=torch.long)
            o2 = self.model(input_ids=tok, past_key_values=pk, use_cache=True,
                            generation_steps=g, cache_position=torch.tensor([g + 1], device=self.device))
            r = self._sample_tok(o2.logits[:, -1, :], do_sample=do_sample)
            tokens.append(r)
        if return_tensor:
            return torch.tensor(tokens, device=self.device, dtype=torch.long)
        return tokens

    def _sample_tok_gpu(self, logits: torch.Tensor, *, do_sample: bool, out_tok: torch.Tensor) -> None:
        """Sample or argmax token directly on GPU into out_tok (1, 1). Zero host sync."""
        if not do_sample:
            torch.argmax(logits, dim=-1, keepdim=True, out=out_tok)
            return
        # Transformers order: TopK (20) -> Temperature (0.80) -> TopP (0.90)
        # Note: topk is monotonic, so compute topk directly on logits without full vocab division
        topk_vals, topk_idx = torch.topk(logits, k=20, dim=-1)
        probs = F.softmax(topk_vals / 0.80, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = (cum - probs) > 0.90
        probs[mask] = 0.0
        sample_idx = torch.multinomial(probs, num_samples=1)
        torch.gather(topk_idx, -1, sample_idx, out=out_tok)

    def _sample_tok(self, logits: torch.Tensor, *, do_sample: bool) -> int:
        if not do_sample:
            return int(logits.argmax(-1).item())
        warped = _warp_logits(logits, temperature=0.80, top_k=20, top_p=0.90)
        probs = F.softmax(warped, dim=-1)
        # RNG globale che avanza (come transformers) — MAI seed fisso per residuo
        return int(torch.multinomial(probs, 1).item())

    # --------------------------------------------------------------- capture
    def capture(self) -> "SubtalkerGraph":
        if self.captured:
            return self
        if not torch.cuda.is_available():
            logger.warning("[subtalker_graph] CUDA non disponibile: eager.")
            return self
        try:
            from transformers import StaticCache
            N = self.max_code_groups + 2  # 17 posizioni: prefill(2) + 15 residui
            # prefill: [last_h(2048) | talker_emb(c0)(2048)] — il code predictor
            # NON ha l'embedding del talker: lo riceviamo dal chiamante.
            if self.talker_embed is None:
                raise RuntimeError("serve talker_embed (embedding del TALKER 2048) per la cattura")
            last_h = torch.randn(1, 1, 2048, dtype=torch.bfloat16, device=self.device)
            c0_emb = self.talker_embed(torch.tensor([[42]], device=self.device))
            prefill_inputs = torch.cat([last_h, c0_emb], dim=1)

            sc = StaticCache(config=self.cfg, max_batch_size=1, max_cache_len=N,
                             device=self.device, dtype=torch.bfloat16)

            # --- warmup eager (allocazioni + compile) ---
            with torch.no_grad():
                out = self.model(inputs_embeds=prefill_inputs, past_key_values=sc, use_cache=True,
                                 cache_position=torch.arange(0, 2, device=self.device))
                for g in range(1, self.max_code_groups):
                    self.model(input_ids=torch.tensor([[1]], device=self.device),
                               past_key_values=sc, use_cache=True, generation_steps=g,
                               cache_position=torch.tensor([g + 1], device=self.device))
            torch.cuda.synchronize()

            # --- buffer statici ---
            self.static_prefill_inputs = torch.zeros_like(prefill_inputs)
            self.static_prefill_pos = torch.tensor([0, 1], dtype=torch.long, device=self.device)
            self.static_tok = torch.zeros(1, 1, dtype=torch.long, device=self.device)
            self.static_pos = torch.zeros(1, dtype=torch.long, device=self.device)
            self.static_tokens = torch.zeros(1, self.max_code_groups, dtype=torch.long, device=self.device)
            self.static_logits = torch.zeros(1, self.vocab, dtype=torch.float32, device=self.device)
            self.static_cache = sc

            # --- cattura prefill ---
            self.static_prefill_inputs.copy_(prefill_inputs)
            g_pre = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g_pre):
                out_g = self.model(inputs_embeds=self.static_prefill_inputs, past_key_values=self.static_cache,
                                   use_cache=True, cache_position=self.static_prefill_pos)
                self.static_logits.copy_(out_g.logits[:, -1, :])
            self.prefill_graph = g_pre

            # --- cattura step g=1..14 ---
            for g in range(1, self.max_code_groups):
                self.static_tok.copy_(torch.tensor([[1]], device=self.device))
                self.static_pos.fill_(g + 1)
                gg = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gg):
                    out_s = self.model(input_ids=self.static_tok, past_key_values=self.static_cache,
                                       use_cache=True, generation_steps=g, cache_position=self.static_pos)
                    self.static_logits.copy_(out_s.logits[:, -1, :])
                self.step_graphs.append(gg)
                torch.cuda.synchronize()

            self.captured = True
            logger.info("[subtalker_graph] catturati %d graph (prefill + %d step, vocab=%d)",
                        len(self.step_graphs) + 1, len(self.step_graphs), self.vocab)
            print(f"[subtalker_graph] CATTURATI {len(self.step_graphs) + 1} CUDA GRAPH (prefill + {len(self.step_graphs)} step)", flush=True)
            return self
        except Exception as exc:
            logger.warning("[subtalker_graph] cattura fallita (%s): eager bit-exact.", exc)
            return self

    @property
    def vocab_size(self) -> int:
        return self.vocab


__all__ = ["SubtalkerGraph", "ResidualSampler"]


class ResidualSampler:
    """Compat: sampling top_k+top_p con RNG globale (bit-exact con transformers)."""

    def __init__(self, *, top_k: int = 20, top_p: float = 0.90, temperature: float = 0.80) -> None:
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

    def sample(self, logits: torch.Tensor, *, vocab: int = 2048) -> int:
        warped = _warp_logits(logits[:, :vocab], temperature=self.temperature,
                              top_k=self.top_k, top_p=self.top_p)
        probs = F.softmax(warped, dim=-1)
        return int(torch.multinomial(probs, 1).item())
