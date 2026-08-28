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
        # Text conditioning mancante? Nel flusso async_chunk il testo viaggia
        # come token ids Gemma (ids.output) e va ricostruito qui.
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
        return super().preprocess(input_ids, input_embeds, **merged)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

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
