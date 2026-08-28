# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/duplex_extension.py — Gemma4OmniDuplexRuntimeExtension.

Extension duplex minima per la pipeline Gemma-4-Omni (livello 1: fallback chat).

Protocollo (SDD):
- vllm_omni/experimental/fullduplex/engine/contracts.py (DuplexRuntimeExtension)
- vllm_omni/experimental/fullduplex/nemotron_voicechat/runtime.py (riferimento)

Livello 1: nessuna policy custom (plan_append costruisce il prompt del turno
dal payload; decide_output delega al flusso standard). Il TurnDetectorHead
(ADR-027) come policy nativa richiede il data plane audio (livello 2).
"""

from __future__ import annotations

from typing import Any, Mapping

from vllm_omni.experimental.fullduplex.engine.contracts import (
    DuplexAppendPlan,
    DuplexInputMode,
    DuplexOutputAction,
    DuplexOutputDecision,
)


class Gemma4OmniDuplexRuntimeExtension:
    """Policy duplex del modello Gemma-4-Omni (turn-based, chat fallback)."""

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
        # Il fallback chat non richiede un piano speciale: il turno viene
        # committato e la risposta parte via chat completion.
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
        # Nessuna policy custom: il flusso standard (fallback chat) gestisce
        # la risposta. (Il TurnDetectorHead verrà integrato qui al livello 2.)
        return None
