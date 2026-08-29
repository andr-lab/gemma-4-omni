# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/duplex_extension.py — Gemma4OmniDuplexRuntimeExtension.

Extension duplex della pipeline Gemma-4-Omni (porting L2: flusso continuo).

Protocollo (SDD):
- vllm_omni/experimental/fullduplex/engine/contracts.py (DuplexRuntimeExtension)
- vllm_omni/experimental/fullduplex/minicpmo45/runtime.py (riferimento continuo)

Livello 2 (data plane audio-native):
- plan_append: il turno audio viene committato al fallback chat (STT upstream
  via encoder audio nativo Gemma4 — niente Whisper). Il prompt dello stage 0
  trasporta l'audio via history (audio_url), quindi il piano non richiede un
  prompt token speciale.
- decide_output: TurnDetectorHead (ADR-027) ascolta l'audio utente (u_mel 12.5Hz)
  e lo stato del THINKER (h_t 2048) → governa listen/speak/barge-in. Su
  BARGE_IN durante l'emissione → DIRECT_RESPONSE con flush; su YIELD → stop.
"""

from __future__ import annotations

from typing import Any, Mapping

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)

from vllm_omni_plugin.turnhead_runtime import Gemma4OmniTurnHeadRuntime

_turnhead: Gemma4OmniTurnHeadRuntime | None = None
_turnhead_init = False


def _get_turnhead() -> Gemma4OmniTurnHeadRuntime | None:
    """Istanza lazy condivisa della turnhead (fail-soft)."""
    global _turnhead, _turnhead_init
    if not _turnhead_init:
        _turnhead_init = True
        try:
            rt = Gemma4OmniTurnHeadRuntime(device="cuda")
            if rt.load():
                _turnhead = rt
        except Exception as exc:  # fail-soft
            from vllm.logger import init_logger

            init_logger(__name__).warning("[TurnHead] init fallita (%s): policy duplex disattivata.", exc)
            _turnhead = None
    return _turnhead


class Gemma4OmniDuplexRuntimeExtension:
    """Policy duplex del modello Gemma-4-Omni (flusso continuo L2)."""

    def configure_sampling_params(
        self,
        *,
        runtime_config: dict[str, Any],
        defaults: tuple[object, ...],
    ) -> tuple[object, ...]:
        return defaults

    def plan_append(
        self,
        *,
        request_id: str,
        fence: Any,
        session_config: dict[str, Any],
        runtime_config: dict[str, Any],
        seq: int,
        turn_seq: int,
        mode: DuplexInputMode,
        payload: object,
        final: bool,
        sampling_params: object,
    ) -> DuplexAppendPlan:
        # Il fallback chat committa il turno (audio → history → STT upstream
        # Thinker nativo); il prompt dello stage 0 viene costruito dal chat
        # service, quindi il piano non richiede token speciali.
        return DuplexAppendPlan(prompt={})

    def decide_output(
        self,
        *,
        stage_id: int,
        final_stage_id: int,
        segment_finished: bool,
        segment_token_ids: tuple[int, ...],
        segment_output_metadata: dict[str, Any],
        output: object,
    ) -> DuplexOutputDecision | None:
        # Policy turnhead opzionale: il flusso standard (fallback chat con
        # auto_response) gestisce già la risposta continua; la turnhead qui
        # può in futuro forzare listen/speak/barge-in sul data plane nativo.
        # Per ora il DIRECT_RESPONSE resta delegato al framework.
        return None
