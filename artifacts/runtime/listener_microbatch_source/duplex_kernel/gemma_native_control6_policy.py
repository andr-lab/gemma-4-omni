"""Native six-row Gemma policy for fresh per-frame listener hidden states."""
from __future__ import annotations

import time
import math
from typing import Any

import torch

from training.duplex_qlora.native_control6 import (
    ACTIONS,
    ACTION_TO_TOKEN_ID,
    HIDDEN_WIDTH,
    NativeControl6LoRA,
    TOKEN_ID_TO_STRING,
)
from duplex_kernel.duplex_control6 import DuplexControl6Policy
from duplex_kernel.gemma_native_control6_batch_policy import NativeControl6RawScoreSnapshot
from duplex_kernel.turn_policy import ControlAction, PolicyDecision, PolicyInput


class GemmaNativeControl6Policy:
    """Score the six trained Gemma output rows from the current listener hidden.

    This policy intentionally has no V9, audio-query, or MLP fallback. A
    missing/stale hidden is a runtime contract error and the scheduler can
    fail closed instead of guessing an action.
    """

    policy_id = "gemma_native_control6"

    def __init__(
        self,
        adapter: NativeControl6LoRA,
        *,
        checkpoint_path: str,
        trace_scores: bool = False,
        fused_readback: bool = False,
    ) -> None:
        if not isinstance(adapter, NativeControl6LoRA):
            raise TypeError("gemma_native_control6 requires NativeControl6LoRA")
        if not checkpoint_path:
            raise ValueError("gemma_native_control6 requires a loaded adapter checkpoint")
        self.adapter = adapter.eval()
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_path = str(checkpoint_path)
        self.trace_scores = bool(trace_scores)
        self.fused_readback = bool(fused_readback)

    def step(self, state: PolicyInput) -> PolicyDecision:
        hidden = state.native_control_hidden
        source_frame = state.native_control_hidden_frame_id
        available_at_ns = state.native_control_hidden_at_ns
        legal = tuple(state.legal_actions or ())
        unknown = set(legal) - set(ACTIONS)
        if unknown:
            raise ValueError(f"native Control6 received unknown legal actions: {sorted(unknown)}")
        if not legal:
            raise RuntimeError("native Control6 requires a non-empty legal-action mask")
        if hidden is None:
            if state.native_control_no_input_tick:
                if ControlAction.SPEAK.value in legal:
                    raise RuntimeError(
                        "native Control6 requires a fresh listener hidden for a pending user turn"
                    )
                action = ControlAction.CONTINUE if state.speaker_active else ControlAction.LISTEN
                if action.value not in legal:
                    raise RuntimeError(f"native Control6 no-input safety action {action.value} is not legal")
                return PolicyDecision(
                    action=action, frame_id=int(state.frame_id), confidence=1.0,
                    reason="native_control_exact_zero_no_input_tick",
                    generation_id=int(state.generation_id),
                    source="scheduler_verified_exact_zero_user_and_self_pcm",
                    session=str(state.session_id),
                    diagnostics={"policy_id": self.policy_id, "selected_action": action.value,
                                 "no_input_tick": True, "response_lora_applied": False},
                )
            raise RuntimeError("native Control6 requires fresh Gemma listener hidden state")
        if isinstance(source_frame, bool) or not isinstance(source_frame, int):
            raise RuntimeError("native Control6 requires a listener hidden frame watermark")
        if source_frame != int(state.frame_id):
            raise RuntimeError(
                "native Control6 refuses stale listener hidden: "
                f"source_frame={source_frame} policy_frame={state.frame_id}"
            )
        if available_at_ns is not None and (
            isinstance(available_at_ns, bool)
            or not isinstance(available_at_ns, int)
            or available_at_ns <= 0
        ):
            raise RuntimeError("native Control6 hidden availability watermark is invalid")
        if available_at_ns is not None and available_at_ns > time.monotonic_ns():
            raise RuntimeError("native Control6 hidden availability watermark is in the future")
        if not isinstance(hidden, torch.Tensor) or hidden.shape[-1:] != (HIDDEN_WIDTH,):
            raise RuntimeError(
                f"native Control6 requires final-normalized Gemma hidden width {HIDDEN_WIDTH}"
            )
        if hidden.numel() != HIDDEN_WIDTH:
            raise RuntimeError(
                f"native Control6 expects exactly one listener frame hidden, got {tuple(hidden.shape)}"
            )

        parameter = self.adapter.lora_A
        native_hidden = hidden.reshape(1, HIDDEN_WIDTH).to(
            device=parameter.device,
            dtype=self.adapter.base_rows.dtype,
        )
        raw_score_snapshot = state.native_control6_raw_scores
        safe_fallback_action = (
            ControlAction.CONTINUE.value
            if state.speaker_active and ControlAction.CONTINUE.value in legal
            else ControlAction.LISTEN.value
            if ControlAction.LISTEN.value in legal
            else ControlAction.CONTINUE.value
            if ControlAction.CONTINUE.value in legal
            else None
        )
        safe_fallback_index = (
            ACTIONS.index(safe_fallback_action) if safe_fallback_action is not None else None
        )
        if raw_score_snapshot is not None:
            if not isinstance(raw_score_snapshot, NativeControl6RawScoreSnapshot):
                raise TypeError("native Control6 raw scores require a typed row snapshot")
            listener_generation = state.native_listener_generation
            if (
                raw_score_snapshot.session_id != state.session_id
                or isinstance(listener_generation, bool)
                or not isinstance(listener_generation, int)
                or raw_score_snapshot.session_generation != listener_generation
                or raw_score_snapshot.frame_id != state.frame_id
                or raw_score_snapshot.checkpoint_path != self.checkpoint_path
                or raw_score_snapshot.query_index != 0
            ):
                raise RuntimeError("native Control6 raw score provenance does not match policy row")
        with torch.inference_mode():
            logits = (
                torch.tensor(
                    raw_score_snapshot.raw_logits,
                    dtype=torch.float32,
                    device=parameter.device,
                )
                if raw_score_snapshot is not None
                else self.adapter(native_hidden).reshape(-1).float()
            )
            if tuple(logits.shape) != (len(ACTIONS),):
                raise RuntimeError("native Control6 produced invalid six-way logits")
            if not self.fused_readback and not bool(torch.isfinite(logits).all()):
                raise RuntimeError("native Control6 produced invalid six-way logits")
            finite_flag = torch.isfinite(logits).all() if self.fused_readback else None
            raw_index_tensor = logits.argmax()
            legal_mask = torch.tensor(
                [action in legal for action in ACTIONS],
                dtype=torch.bool,
                device=logits.device,
            )
            selected_logits = logits.masked_fill(~legal_mask, -torch.inf)
            raw_is_legal_tensor = legal_mask[raw_index_tensor]
            fallback_index_tensor = (
                torch.tensor(safe_fallback_index, dtype=torch.long, device=logits.device)
                if safe_fallback_index is not None
                else selected_logits.argmax()
            )
            selected_index_tensor = torch.where(
                raw_is_legal_tensor,
                raw_index_tensor,
                fallback_index_tensor,
            )
            legal_probabilities = torch.softmax(selected_logits, dim=0)
            if self.fused_readback:
                probability = torch.gather(
                    legal_probabilities,
                    0,
                    selected_index_tensor.reshape(1),
                ).reshape(())
                packed_scores = torch.stack((
                    finite_flag.to(dtype=torch.float32),  # type: ignore[union-attr]
                    raw_index_tensor.to(dtype=torch.float32),
                    selected_index_tensor.to(dtype=torch.float32),
                    probability.to(dtype=torch.float32),
                    raw_is_legal_tensor.to(dtype=torch.float32),
                ))
                packet = packed_scores.detach().cpu().tolist()
                if not isinstance(packet, list) or len(packet) != 5:
                    raise RuntimeError("native Control6 produced invalid fused six-way readback")
                try:
                    finite_value, raw_value, selected_value, confidence, raw_is_legal_value = (
                        float(value) for value in packet
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "native Control6 produced invalid fused six-way readback"
                    ) from exc
                if finite_value != 1.0:
                    raise RuntimeError("native Control6 produced invalid six-way logits")
                if (
                    not all(math.isfinite(value) for value in (raw_value, selected_value, confidence))
                    or raw_value != int(raw_value)
                    or selected_value != int(selected_value)
                    or not 0 <= int(raw_value) < len(ACTIONS)
                    or not 0 <= int(selected_value) < len(ACTIONS)
                    or not 0.0 <= confidence <= 1.0
                    or raw_is_legal_value not in {0.0, 1.0}
                ):
                    raise RuntimeError("native Control6 produced invalid fused six-way readback")
                raw_index = int(raw_value)
                selected_index = int(selected_value)
                if raw_is_legal_value == 0.0 and safe_fallback_index is None:
                    raise RuntimeError(
                        "native Control6 raw action is illegal and no safe non-vocal fallback is legal"
                    )
                fallback_reason = (
                    None
                    if raw_is_legal_value == 1.0
                    else "raw_action_illegal_nonvocal_fallback"
                )
            else:
                raw_index = int(raw_index_tensor.item())
                raw_action_value = ACTIONS[raw_index]
                if raw_action_value in legal:
                    selected_index = raw_index
                    fallback_reason = None
                else:
                    if safe_fallback_index is None:
                        raise RuntimeError(
                            "native Control6 raw action is illegal and no safe non-vocal fallback is legal"
                        )
                    selected_index = safe_fallback_index
                    fallback_reason = "raw_action_illegal_nonvocal_fallback"
                probability = legal_probabilities[selected_index]
                confidence = float(probability.item())

        action = ControlAction(ACTIONS[selected_index])
        raw_action = ACTIONS[raw_index]
        raw_token_id = ACTION_TO_TOKEN_ID[raw_action]
        selected_token_id = ACTION_TO_TOKEN_ID[action.value]
        diagnostics: dict[str, Any] = {
            "policy_id": self.policy_id,
            "checkpoint_path": self.checkpoint_path,
            "base_action": raw_action,
            "native_control6_raw_row_index": raw_index,
            "native_control6_raw_token_id": raw_token_id,
            "native_control6_raw_token": TOKEN_ID_TO_STRING[raw_token_id],
            "native_control6_selected_row_index": selected_index,
            "native_control6_selected_token_id": selected_token_id,
            "native_control6_selected_token": TOKEN_ID_TO_STRING[selected_token_id],
            "native_action": action.value,
            "native_action_probability": confidence,
            "selected_action_probability": confidence,
            "selected_action": action.value,
            "policy_fallback_reason": fallback_reason,
            "legal_actions": list(legal),
            "listener_hidden_frame_id": int(source_frame),
            "listener_hidden_at_ns": int(available_at_ns) if available_at_ns is not None else None,
            "listener_hidden_age_frames": 0,
            "hidden_contract": "gemma4_listener_final_norm_frame_v1",
            "response_lora_applied": False,
        }
        if raw_score_snapshot is not None:
            diagnostics.update({
                "native_control6_batched_raw_scores": True,
                "native_control6_batch_manifest_id": raw_score_snapshot.manifest_id,
                "native_control6_native_output_generation": raw_score_snapshot.output_generation,
                "native_control6_listener_state_version": raw_score_snapshot.listener_state_version,
                "native_control6_query_index": raw_score_snapshot.query_index,
            })
        if self.trace_scores:
            base_logits = torch.nn.functional.linear(
                native_hidden, self.adapter.base_rows
            ).reshape(-1).float()
            diagnostics.update({
                "native_action_order": list(ACTIONS),
                "native_action_base_logits": base_logits.detach().cpu().tolist(),
                "native_action_logits": logits.detach().cpu().tolist(),
                "native_action_probabilities": legal_probabilities.detach().cpu().tolist(),
            })
        return PolicyDecision(
            action=action,
            frame_id=int(state.frame_id),
            confidence=confidence,
            reason="gemma_native_control6",
            generation_id=int(state.generation_id),
            listener_frame_id=int(source_frame),
            source="gemma_lm_head_selected_rows+rank16_delta:ListenerOutput.gemma_hidden_t",
            session=str(state.session_id),
            diagnostics=diagnostics,
        )

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        return None


__all__ = ["GemmaNativeControl6Policy"]

CONTROL6_POLICY_TYPES = (DuplexControl6Policy, GemmaNativeControl6Policy)


def is_controller6_policy(policy: Any) -> bool:
    """True only for the legacy MLP and native-row six-action policies."""
    return isinstance(policy, CONTROL6_POLICY_TYPES)
