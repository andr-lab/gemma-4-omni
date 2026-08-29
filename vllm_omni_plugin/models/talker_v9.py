# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/models/talker_v9.py — Gemma4OmniV9TalkerForConditionalGeneration.

Talker Qwen3-TTS esteso con Prefix KV-Cache Conditioning V9 (ADR-022/023/024):
- Eredita TUTTA la meccanica nativa del talker vLLM-Omni (prefill da testo,
  MTP fast-path per i codebook residui, logits masking, streaming async_chunk).
- Al primo prefill, se il payload inter-stadio contiene gli hidden states dei
  token generati dal thinker (chiave "v9_thinker_hidden"), calcola il prefisso
  V9 (16 frame, projector 211.83M + causal conv addestrata) e lo APPENDE al
  prompt nativo: [ICL_text | h_pfx | BOS] — ricetta esatta di
  tools/interactive_gemma4_omni_live.py.

La causal conv (7.08M, Task 11i SOTA) NON è opzionale: i feature di training
del projector (extract_ars_cache_v2.py) erano già post-conv, quindi va
applicata per matchare la distribuzione d'addestramento.

Caricamento pesi: il tracker di vLLM (DefaultModelLoader.track_weights_loading)
esige che ogni parametro del modello provenga dal checkpoint dello stage. I
moduli V9 vengono quindi caricati DENTRO load_weights() (dai checkpoint
addestrati in /workspace) e i loro nomi aggiunti al set di pesi consumati.
"""

from __future__ import annotations

import os
from typing import Any, Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.logger import init_logger

from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
    _has_tts_text_conditioning,
)

from vllm_omni_plugin.models.v9_projector import (
    CausalConv1D,
    DeepCausalTransformerProjectorV9,
    compute_v9_prefix,
    load_v9_conditioning,
)

logger = init_logger(__name__)

# Chiave payload inter-stadio prodotta da thinker2talker_v9_full_payload.
V9_HIDDEN_KEY = "v9_thinker_hidden"
# Frame del prefisso V9 (ground truth: T_pfx_len = 16).
V9_PREFIX_FRAMES = 16
# Frames extra rispetto alla STIMA nativa del placeholder:
#   stima     = role + codec_prefix + (assistant-6)          [senza eos, senza bos]
#   nativo    = stima + 2                                    [eos + bos]
#   V9 finale = nativo - 1 (strip bos nativo) + 16 (pfx) + 1 (bos) = stima + 18
# Se il placeholder è corto anche di 1 frame, il BOS viene troncato e il
# decode parte da uno stato rotto (silenzio + niente EOS).
V9_EXTRA_FRAMES = V9_PREFIX_FRAMES + 2


class _V9PromptEmbedsBuilderWrapper:
    """Wrapper del prompt builder nativo che appende [h_pfx | BOS] al prefill.

    Delega ogni chiamata al builder originale; intercetta solo
    ``build_prompt_embeds`` quando il payload contiene gli hidden del thinker.
    """

    def __init__(self, owner: "Gemma4OmniV9TalkerForConditionalGeneration", delegate: Any):
        object.__setattr__(self, "_owner", owner)
        object.__setattr__(self, "_delegate", delegate)

    def _has_v9(self, info_dict: dict[str, Any]) -> torch.Tensor | None:
        h = info_dict.get(V9_HIDDEN_KEY)
        if isinstance(h, torch.Tensor):
            return h
        hs = info_dict.get("hidden_states")
        if isinstance(hs, dict):
            h2 = hs.get(V9_HIDDEN_KEY)
            if isinstance(h2, torch.Tensor):
                return h2
            # Forma canone del connector async_chunk: HiddenStatesStruct.output.
            h3 = hs.get("output")
            if isinstance(h3, torch.Tensor) and h3.numel() > 0:
                return h3
        return None

    def build_prompt_embeds(self, *, task_type: str, info_dict: dict[str, Any]):
        owner: Gemma4OmniV9TalkerForConditionalGeneration = self._owner
        out = self._delegate.build_prompt_embeds(task_type=task_type, info_dict=info_dict)
        talker_prompt, trailing_text_hidden, ref_code_len, ref_code = out

        v9_hidden = self._has_v9(info_dict)
        if v9_hidden is None or not owner.v9_ready:
            return out
        if os.environ.get("GEMMA4_OMNI_DISABLE_V9"):
            return out

        try:
            # Device/dtype di riferimento = modello NATIVO del talker (sempre
            # corretto a runtime; i moduli V9 possono ancora essere su CPU/meta
            # al primo prefill).
            v9_dev = next(owner.model.parameters()).device
            v9_dtype = next(owner.model.parameters()).dtype
            h_pfx = compute_v9_prefix(
                v9_hidden,
                owner.v9,
                owner.causal_conv,
                device=v9_dev,
                dtype=v9_dtype,
                n_prefix_frames=V9_PREFIX_FRAMES,
            )  # (1, 16, 2048)
        except Exception as exc:  # fail-soft: mai bloccare il serving
            logger.error("[V9] Calcolo prefisso fallito (%s): talker in modalità text-only.", exc)
            return out

        # BOS: tts_pad_embed + embed(codec_bos_id) — ground truth bos_part.
        with torch.no_grad():
            bos_embed = owner._tts_pad_embed.to(device=h_pfx.device, dtype=h_pfx.dtype).reshape(1, -1)
            bos_ids = torch.tensor(
                [[int(owner.talker_config.codec_bos_id)]], device=h_pfx.device, dtype=torch.long
            )
            # embed_input_ids ritorna (1,1,H): schiaccia a (1,H) PRIMA della somma
            # per evitare il broadcast che genera tensori 3D.
            bos_embed = (
                bos_embed + owner.embed_input_ids(bos_ids).to(dtype=h_pfx.dtype).reshape(1, -1)
            ).reshape(1, -1)

            # Nel path VoiceDesign non-streaming il prompt nativo TERMINA già con
            # il frame BOS: lo rimuoviamo per replicare la struttura addestrata
            # [ICL | h_pfx | BOS] (il BOS va DOPO il prefisso V9, non prima).
            task_type = (info_dict.get("task_type") or [""])[0]
            if task_type == "VoiceDesign" and talker_prompt.shape[0] >= 1:
                talker_prompt = talker_prompt[:-1]

            # Il builder nativo lavora in [L, H]: schiaccia il batch del prefisso.
            h_pfx = h_pfx.squeeze(0) if h_pfx.dim() == 3 else h_pfx
            talker_prompt = torch.cat([talker_prompt, h_pfx, bos_embed], dim=0).contiguous()
        return talker_prompt, trailing_text_hidden, ref_code_len, ref_code

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


class Gemma4OmniV9TalkerForConditionalGeneration(Qwen3TTSTalkerForConditionalGeneration):
    """Talker Qwen3-TTS con Prefix KV-Cache Conditioning V9."""

    v9: nn.Module | None
    causal_conv: nn.Module | None
    v9_ready: bool = False
    _gemma_tok_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # ARIA Streaming Pacing (ADR-023): 4.8 frame/parola + 12 di overhead.
    # Il budget di generazione acustica per-request viene calcolato dal
    # testo della risposta (worker-side, dove il testo è noto a runtime) e
    # applicato in compute_logits forzando il token EOS del talker (2150)
    # quando il contatore dei frame generati raggiunge il budget.
    # Reference: tools/interactive_gemma4_omni_live.py (T_gen_frames).
    # ------------------------------------------------------------------
    ARIA_FRAMES_PER_WORD = 4.8
    ARIA_OVERHEAD_FRAMES = 12
    ARIA_MIN_FRAMES = 48
    ARIA_MAX_FRAMES = 140

    # Token EOS del Talker Qwen3-TTS (stop_token_ids nel pipeline config).
    TALKER_EOS_TOKEN_ID = 2150

    def _aria_budget_for_text(self, text: str) -> int:
        """Budget di frame acustici ARIA per un testo di risposta.

        T_gen = max(48, min(140, n_words*4.8 + 12)) — stessa formula del
        reference locale (interactive_gemma4_omni_live.py). Il tetto a 140
        frame (11.2s) evita code di silenzio/rumore per risposte lunghe;
        il pavimento a 48 garantisce frasi brevi complete.
        """
        n_words = max(1, len(str(text).split()))
        budget = int(n_words * self.ARIA_FRAMES_PER_WORD) + self.ARIA_OVERHEAD_FRAMES
        return max(self.ARIA_MIN_FRAMES, min(self.ARIA_MAX_FRAMES, budget))

    def _aria_budget_for_info(self, info_dict: dict[str, Any]) -> int | None:
        """Budget ARIA dal testo presente nell'info payload del talker.

        Il testo arriva da additional_information.text (flusso sync) o dai
        token ids Gemma ids.output (flusso async_chunk). Ritorna None se
        nessun testo è recuperabile (fail-soft: nessun pacing forzato).
        """
        text = None
        additional = info_dict.get("additional_information")
        if isinstance(additional, dict):
            raw = additional.get("text")
            if isinstance(raw, list) and raw and isinstance(raw[0], str):
                text = raw[0].strip()
            elif isinstance(raw, str) and raw.strip():
                text = raw.strip()
        if not text:
            text = self._recover_text_from_info(info_dict)
        if not text:
            return None
        return self._aria_budget_for_text(text)

    def _gemma_tokenizer(self):
        """Tokenizer Gemma per recuperare il testo della risposta dai token ids.

        Nel flusso async_chunk il payload connector non può trasportare stringhe
        (OmniPayloadStruct forbid_unknown_fields): il testo arriva come ids
        Gemma in ``ids.output`` e viene detokenizzato qui, worker-side, una
        sola volta per processo.
        """
        if "tok" not in self._gemma_tok_cache:
            import glob

            from transformers import AutoTokenizer

            snaps = sorted(
                glob.glob(
                    "/workspace/.hf_cache/models--unsloth--gemma-4-e2b-it-unsloth-bnb-4bit/snapshots/*"
                )
            )
            if not snaps:
                raise FileNotFoundError("Thinker Gemma non trovato per detokenizzazione.")
            self._gemma_tok_cache["tok"] = AutoTokenizer.from_pretrained(snaps[-1], use_fast=True)
        return self._gemma_tok_cache["tok"]

    def _recover_text_from_info(self, info_dict: dict[str, Any]) -> str | None:
        """Recupera il testo della risposta dagli ids Gemma presenti nel payload."""
        ids = info_dict.get("ids")
        if not isinstance(ids, dict):
            return None
        out_ids = ids.get("output")
        if not isinstance(out_ids, (list, tuple)) or not out_ids:
            return None
        try:
            text = self._gemma_tokenizer().decode(list(out_ids), skip_special_tokens=True).strip()
            return text or None
        except Exception as exc:
            logger.warning("[V9] Recupero testo dai token ids fallito (%s).", exc)
            return None

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info_dict: Any):
        import os as _os

        if _os.environ.get("DEBUG_REGISTRY"):
            try:
                _plen = info_dict.get("_omni_prompt_len")
                _nct = info_dict.get("_omni_num_computed_tokens")
                _isp = info_dict.get("_omni_is_prefill")
                print(
                    f"[V9-talker] preprocess span={int(input_ids.shape[0])} "
                    f"prompt_len={_plen} computed={_nct} is_prefill={_isp}",
                    flush=True,
                )
            except Exception:
                pass
        merged = dict(info_dict)
        additional_information = merged.get("additional_information")
        if isinstance(additional_information, dict):
            for k, v in additional_information.items():
                merged.setdefault(k, v)
        # Il task Base implica codec_streaming=True nel preprocess nativo
        # (streaming del codec RVQ), incompatibile con il nostro deploy
        # full-payload: forziamo False per un decode one-shot coerente.
        meta = merged.get("meta")
        if isinstance(meta, dict):
            meta = dict(meta)
            meta["codec_streaming"] = False
            merged["meta"] = meta
        elif meta is None:
            merged["meta"] = {"codec_streaming": False}
        # Nel flusso async il payload connector (OmniPayloadStruct) non può
        # trasportare task_type/language: il default nativo (CustomVoice)
        # richiederebbe uno speaker non disponibile. Forziamo VoiceDesign
        # (replica del prefill addestrato) quando il task non è specificato.
        if not merged.get("task_type"):
            merged.setdefault("task_type", ["VoiceDesign"])
            merged.setdefault("language", ["Italian"])
            if isinstance(additional_information, dict):
                additional_information = dict(additional_information)
                additional_information.setdefault("task_type", ["VoiceDesign"])
                additional_information.setdefault("language", ["Italian"])
                merged["additional_information"] = additional_information
        if not _has_tts_text_conditioning(merged, merged.get("hidden_states")):
            text = self._recover_text_from_info(merged)
            text = self._recover_text_from_info(merged)
            if text:
                if isinstance(additional_information, dict):
                    additional_information = dict(additional_information)
                    additional_information.setdefault("text", [text])
                    merged["additional_information"] = additional_information
                merged.setdefault("text", [text])
            else:
                # Nessuna via di recupero: lascia fallire il check nativo con
                # un messaggio chiaro.
                logger.error(
                    "[V9] Text conditioning assente e nessun ids.output nel payload; "
                    "il talker non può sintetizzare."
                )
        # ARIA pacing: registra il budget per-request DAL TEXTO ricostruito
        # qui (il flusso async arriva con request_id nei req_infos).
        try:
            rid = merged.get("request_id") or merged.get("_omni_request_id")
            if rid:
                budget = self._aria_budget_for_info(merged)
                if budget is not None:
                    self._req_aria_budget[str(rid)] = budget
                    if _os.environ.get("DEBUG_REGISTRY"):
                        print(f"[V9-ARIA] rid={rid} budget={budget}", flush=True)
        except Exception:  # fail-soft: il pacing non deve mai rompere il talker
            pass
        return super().preprocess(input_ids, input_embeds, **merged)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        # Budget ARIA per-request: request_id -> frame di generazione massimi.
        # Popolato in preprocess_batch (testo noto worker-side), consumato in
        # compute_logits (forcing EOS quando il contatore raggiunge il budget).
        self._req_aria_budget: dict[str, int] = {}

        # Profili voce di default (voci registrate come custom_voice_dir):
        # il loader nativo gira DENTRO super().__init__, quindi impostiamo il
        # campo config e ricarichiamo qui per le voci della nostra dir.
        voices_dir = os.environ.get("GEMMA4_OMNI_CUSTOM_VOICE_DIR")
        if voices_dir and os.path.isdir(voices_dir):
            try:
                self.config.custom_voice_dir = voices_dir
                self._load_custom_voice_profiles()
            except Exception as exc:  # fail-soft
                logger.warning("[V9] Caricamento profili voce fallito (%s).", exc)

        # I moduli V9 vengono costruiti qui ma i pesi caricati in load_weights()
        # (il tracker vLLM esige che ogni parametro provenga dal flusso di load).
        self.v9 = DeepCausalTransformerProjectorV9()
        self.causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
        self.v9_ready = True
        self._v9_ckpt = os.environ.get("GEMMA4_OMNI_V9_CKPT")
        self._heads_ckpt = os.environ.get("GEMMA4_OMNI_HEADS_CKPT")

        # Sostituisce il prompt builder con il wrapper che inietta il prefisso.
        self._prompt_builder = _V9PromptEmbedsBuilderWrapper(self, self._prompt_builder)

    # ------------------------------------------------------------------
    # ARIA pacing: override di preprocess_batch (budget) e compute_logits
    # (forcing EOS). Il talker genera un token RVQ per frame (12.5 Hz): il
    # contatore len(output_token_ids[i]) è il numero di frame già emessi.
    # ------------------------------------------------------------------
    def preprocess_batch(
        self,
        *,
        req_ids: list[str],
        model_intermediate_buffer: dict[str, dict[str, Any]],
        device: torch.device,
    ) -> None:
        """Calcola il budget ARIA per-request dal testo della risposta.

        Fallback su richiesta per-request: il testo arriva via
        additional_information.text (flusso sync) o recuperato dai token ids
        Gemma ids.output (flusso async_chunk). Il budget è il numero MASSIMO
        di frame acustici che il talker può emettere; oltre, compute_logits
        forza il token EOS (2150).
        """
        for req_id in req_ids:
            info_dict = model_intermediate_buffer.get(req_id)
            if isinstance(info_dict, dict):
                budget = self._aria_budget_for_info(info_dict)
                if budget is not None:
                    self._req_aria_budget[req_id] = budget
        return super().preprocess_batch(
            req_ids=req_ids,
            model_intermediate_buffer=model_intermediate_buffer,
            device=device,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor | Any,
        sampling_metadata: Any = None,
    ) -> torch.Tensor | None:
        """Forza EOS quando il frame corrente supera il budget ARIA.

        Il pacing è per-request: ogni risposta ha un budget proporzionale al
        numero di parole del testo (4.8 frame/parola + 12). Senza questo, le
        risposte brevi con EOS ritardato generano code di silenzio/rumore
        (NEXT_STEP §3.2). Fail-soft: se il budget non è noto, nessun forcing.
        """
        logits = super().compute_logits(hidden_states, sampling_metadata)
        if logits is None or not self._req_aria_budget:
            return logits
        output_token_ids = getattr(sampling_metadata, "output_token_ids", None)
        if output_token_ids is None or len(output_token_ids) != logits.shape[0]:
            return logits
        batch_req_ids = self._batch_req_ids
        if len(batch_req_ids) != logits.shape[0]:
            return logits
        eos_id = self.TALKER_EOS_TOKEN_ID
        force = False
        import os as _os
        for i, rid in enumerate(batch_req_ids):
            budget = self._req_aria_budget.get(rid)
            if budget is None:
                continue
            generated = len(output_token_ids[i])
            if _os.environ.get("DEBUG_REGISTRY") and generated in (1, 5, 6, 47, 48, 49):
                logger.warning("[V9-ARIA] rid=%s generated=%d budget=%s out_token_sample=%s",
                               rid, generated, budget, list(output_token_ids[i][-3:]))
            # >=: al frame budget-1 non forziamo ancora (lasciamo l'ultimo
            # frame naturale); al frame budget EOS è forzato.
            if generated >= budget:
                logits[i, eos_id] = float("inf")
                force = True
        if force:
            logger.info("[V9-ARIA] pacing EOS forzato su %d request (budget attivi=%d).",
                        sum(1 for i in range(len(batch_req_ids))
                            if self._req_aria_budget.get(batch_req_ids[i]) is not None
                            and len(output_token_ids[i]) >= self._req_aria_budget[batch_req_ids[i]]),
                        len(self._req_aria_budget))
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Carica i pesi nativi del talker + i moduli V9 dai checkpoint addestrati."""
        loaded = set(super().load_weights(weights))
        try:
            v9, causal_conv = load_v9_conditioning(v9_ckpt=self._v9_ckpt, heads_ckpt=self._heads_ckpt)
            self.v9.load_state_dict(v9.state_dict(), assign=True)
            self.causal_conv.load_state_dict(causal_conv.state_dict(), assign=True)
            self.v9_ready = True
            logger.info("[V9] Pesi projector V9 + causal conv caricati dai checkpoint addestrati.")
        except Exception as exc:  # fail-soft: text-only fallback
            self.v9_ready = False
            logger.error("[V9] Caricamento checkpoint V9 fallito (%s): text-only fallback.", exc)

        # Dichiara i parametri V9 come "consumati" per il tracker dei pesi vLLM.
        # I nomi devono combaciare con model.named_parameters() (prefix "v9." e
        # "causal_conv.").
        for attr, mod in (("v9", self.v9), ("causal_conv", self.causal_conv)):
            for name, _ in mod.named_parameters():
                loaded.add(f"{attr}.{name}")
        return loaded

    @property
    def v9_extra_frames(self) -> int:
        """Frames extra che il placeholder del talker deve riservare (16+1)."""
        return V9_EXTRA_FRAMES if self.v9_ready else 0
