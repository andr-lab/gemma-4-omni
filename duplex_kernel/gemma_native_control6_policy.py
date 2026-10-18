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
from duplex_kernel.native_control6_phase import NATIVE_CONTROL6_PHASE_T_WIDTH
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
        self._hysteresis_session_id: str | None = None
        self._hysteresis_speaker_active: bool | None = None
        self._hysteresis_frame_id: int | None = None
        self._hysteresis_candidate: str | None = None
        self._hysteresis_count = 0
        self._last_microbatch_tick_id: tuple[str, int, int] | None = None
        self._last_microbatch_listener_tick: tuple[str, int, int] | None = None

    def step(self, state: PolicyInput) -> PolicyDecision:
        return self._step_impl(state, precomputed_logits=None)

    def step_with_logits(
        self,
        state: PolicyInput,
        logits: torch.Tensor | None,
        *,
        tick_id: tuple[str, int, int],
    ) -> PolicyDecision:
        """Apply this session's normal mask/hysteresis to one batched row.

        ``logits=None`` is reserved for the existing exact-zero no-input safety
        path, which does not run the learned head. A provided tensor must be
        the result of the shared adapter forward for this tick.
        """
        if not isinstance(tick_id, tuple) or len(tick_id) != 3:
            raise ValueError("microbatch policy tick ID must be (session_id, generation_id, frame_id)")
        expected = (state.session_id, state.generation_id, state.frame_id)
        if tick_id != expected:
            raise RuntimeError("microbatch policy tick ID differs from its state")
        last = self._last_microbatch_tick_id
        if last == tick_id:
            raise RuntimeError("duplicate microbatch policy tick would apply hysteresis twice")
        if (
            last is not None
            and last[0] == tick_id[0]
            and last[1] == tick_id[1]
            and tick_id[2] <= last[2]
        ):
            raise RuntimeError("stale microbatch policy tick cannot update hysteresis")
        listener_generation = state.native_listener_generation
        if listener_generation is None:
            listener_generation = state.generation_id
        if (
            isinstance(listener_generation, bool)
            or not isinstance(listener_generation, int)
            or listener_generation < 0
        ):
            raise ValueError("microbatch policy listener generation is invalid")
        listener_tick = (state.session_id, listener_generation, state.frame_id)
        last_listener_tick = self._last_microbatch_listener_tick
        if (
            last_listener_tick is not None
            and last_listener_tick[:2] == listener_tick[:2]
            and listener_tick[2] <= last_listener_tick[2]
        ):
            raise RuntimeError("stale listener frame cannot update a newer policy turn")
        if logits is None and not state.native_control_no_input_tick:
            raise ValueError("precomputed Control6 logits are required for an input tick")
        if logits is not None and state.native_control_no_input_tick:
            raise ValueError("exact-zero no-input safety tick must not score listener logits")
        decision = self._step_impl(state, precomputed_logits=logits)
        self._last_microbatch_tick_id = tick_id
        self._last_microbatch_listener_tick = listener_tick
        return decision

    def _step_impl(
        self,
        state: PolicyInput,
        *,
        precomputed_logits: torch.Tensor | None,
    ) -> PolicyDecision:
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
            self.reset()
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
            self.reset()
            raise RuntimeError("native Control6 requires a listener hidden frame watermark")
        if source_frame != int(state.frame_id):
            self.reset()
            raise RuntimeError(
                "native Control6 refuses stale listener hidden: "
                f"source_frame={source_frame} policy_frame={state.frame_id}"
            )
        if available_at_ns is not None and (
            isinstance(available_at_ns, bool)
            or not isinstance(available_at_ns, int)
            or available_at_ns <= 0
        ):
            self.reset()
            raise RuntimeError("native Control6 hidden availability watermark is invalid")
        if available_at_ns is not None and available_at_ns > time.monotonic_ns():
            self.reset()
            raise RuntimeError("native Control6 hidden availability watermark is in the future")
        if not isinstance(hidden, torch.Tensor) or hidden.shape[-1:] != (HIDDEN_WIDTH,):
            self.reset()
            raise RuntimeError(
                f"native Control6 requires final-normalized Gemma hidden width {HIDDEN_WIDTH}"
            )
        if hidden.numel() != HIDDEN_WIDTH:
            self.reset()
            raise RuntimeError(
                f"native Control6 expects exactly one listener frame hidden, got {tuple(hidden.shape)}"
            )

        parameter = self.adapter.lora_A
        native_hidden = hidden.reshape(1, HIDDEN_WIDTH).to(
            device=parameter.device,
            dtype=self.adapter.base_rows.dtype,
        )
        raw_score_snapshot = state.native_control6_raw_scores
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
                self.reset()
                raise RuntimeError("native Control6 raw score provenance does not match policy row")
        with torch.inference_mode():
            phase_t = state.native_control_phase_t
            phase_tensor = (
                torch.zeros((1, NATIVE_CONTROL6_PHASE_T_WIDTH), device=parameter.device)
                if phase_t is None
                else torch.as_tensor(
                    phase_t, device=parameter.device, dtype=torch.float32
                ).reshape(1, -1)
            )
            if tuple(phase_tensor.shape) != (1, NATIVE_CONTROL6_PHASE_T_WIDTH):
                raise RuntimeError("native Control6 requires exactly four phase_t values")
            # The raw batch snapshot was scored without phase_t. Keep it for
            # diagnostics and legacy phase-less callers, but never let it
            # override the current phase-conditioned policy score. The
            # microbatch path supplies logits from its sole mixed-source
            # adapter forward and continues through this same policy logic.
            use_phase_less_snapshot = (
                precomputed_logits is None
                and raw_score_snapshot is not None
                and phase_t is None
            )
            if precomputed_logits is not None:
                logits = precomputed_logits.detach().to(
                    device=parameter.device, dtype=torch.float32
                ).reshape(-1)
            elif use_phase_less_snapshot:
                logits = torch.tensor(
                    raw_score_snapshot.raw_logits,
                    dtype=torch.float32,
                    device=parameter.device,
                )
            else:
                logits = self.adapter(native_hidden, phase_t=phase_tensor).reshape(-1).float()
            if tuple(logits.shape) != (len(ACTIONS),):
                raise RuntimeError("native Control6 produced invalid six-way logits")
            if not self.fused_readback and not bool(torch.isfinite(logits).all()):
                raise RuntimeError("native Control6 produced invalid six-way logits")
            finite_flag = torch.isfinite(logits).all() if self.fused_readback else None
            raw_index_tensor = logits.argmax()
            raw_probabilities = torch.softmax(logits, dim=0)
            if state.speaker_active:
                effective_members = {
                    "HOLD": ("LISTEN", "CONTINUE"),
                    "SPEAK": (),
                    "YIELD": ("YIELD", "REPLAN"),
                    "BACKCHANNEL": (),
                }
                effective_actions = (
                    ControlAction.CONTINUE.value,
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                    ControlAction.BACKCHANNEL.value,
                )
            else:
                effective_members = {
                    "HOLD": ("LISTEN", "CONTINUE"),
                    "SPEAK": ("SPEAK", "REPLAN"),
                    "YIELD": (),
                    "BACKCHANNEL": ("BACKCHANNEL",),
                }
                effective_actions = (
                    ControlAction.LISTEN.value,
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                    ControlAction.BACKCHANNEL.value,
                )
            effective_order = ("HOLD", "SPEAK", "YIELD", "BACKCHANNEL")
            action_indices = {action: index for index, action in enumerate(ACTIONS)}
            zero_probability = torch.zeros((), dtype=raw_probabilities.dtype, device=logits.device)
            # Concatenate the four bucket scalars without using ``stack``;
            # stack is reserved for the fused host readback packet below.
            effective_probabilities = torch.cat(tuple(
                sum(
                    (raw_probabilities[action_indices[action]] for action in effective_members[name]),
                    zero_probability,
                ).reshape(1)
                for name in effective_order
            ))
            effective_legal_mask = torch.tensor(
                [action in legal for action in effective_actions],
                dtype=torch.bool,
                device=logits.device,
            )
            legal_effective_probabilities = effective_probabilities.masked_fill(
                ~effective_legal_mask, 0.0
            )
            legal_effective_mass_tensor = legal_effective_probabilities.sum()
            if not self.fused_readback:
                if not bool(torch.isfinite(raw_probabilities).all()):
                    raise RuntimeError("native Control6 produced invalid six-way probabilities")
                legal_mass_value = float(legal_effective_mass_tensor.item())
                if not math.isfinite(legal_mass_value) or legal_mass_value < 0.0:
                    raise RuntimeError("native Control6 has invalid legal physical action mass")
            normalized_legal_effective_probabilities = legal_effective_probabilities / (
                legal_effective_mass_tensor.clamp_min(torch.finfo(logits.dtype).tiny)
            )
            effective_index_tensor = normalized_legal_effective_probabilities.argmax()
            selected_effective_probability_tensor = torch.gather(
                legal_effective_probabilities,
                0,
                effective_index_tensor.reshape(1),
            ).reshape(())
            confidence_tensor = torch.gather(
                normalized_legal_effective_probabilities,
                0,
                effective_index_tensor.reshape(1),
            ).reshape(())
            if self.fused_readback:
                packed_scores = torch.stack((
                    finite_flag.to(dtype=torch.float32),  # type: ignore[union-attr]
                    raw_index_tensor.to(dtype=torch.float32),
                    effective_index_tensor.to(dtype=torch.float32),
                    legal_effective_mass_tensor.to(dtype=torch.float32),
                    confidence_tensor.to(dtype=torch.float32),
                    selected_effective_probability_tensor.to(dtype=torch.float32),
                    torch.isfinite(raw_probabilities).all().to(dtype=torch.float32),
                ))
                packet = packed_scores.detach().cpu().tolist()
                if not isinstance(packet, list) or len(packet) != 7:
                    raise RuntimeError("native Control6 produced invalid fused physical-action readback")
                try:
                    (
                        finite_value,
                        raw_value,
                        effective_value,
                        legal_mass,
                        confidence,
                        selected_effective_probability,
                        probabilities_finite_value,
                    ) = (
                        float(value) for value in packet
                    )
                except (TypeError, ValueError, OverflowError) as exc:
                    raise RuntimeError(
                        "native Control6 produced invalid fused physical-action readback"
                    ) from exc
                if finite_value != 1.0 or probabilities_finite_value != 1.0:
                    raise RuntimeError("native Control6 produced invalid six-way logits")
                if (
                    not all(math.isfinite(value) for value in (
                        raw_value, effective_value, legal_mass, confidence,
                        selected_effective_probability,
                    ))
                    or raw_value != int(raw_value)
                    or effective_value != int(effective_value)
                    or not 0 <= int(raw_value) < len(ACTIONS)
                    or not 0 <= int(effective_value) < len(effective_order)
                    or legal_mass < 0.0
                    or legal_mass > 1.000001
                    or not 0.0 <= confidence <= 1.0
                    or not 0.0 <= selected_effective_probability <= legal_mass
                ):
                    raise RuntimeError("native Control6 produced invalid fused physical-action readback")
                raw_index = int(raw_value)
                selected_effective_index = int(effective_value)
                retained_effective_mass = legal_mass
                selected_effective_probability = selected_effective_probability
            else:
                raw_index = int(raw_index_tensor.item())
                selected_effective_index = int(effective_index_tensor.item())
                retained_effective_mass = legal_mass_value
                confidence = float(confidence_tensor.item())
                selected_effective_probability = float(selected_effective_probability_tensor.item())

            candidate_action_value = effective_actions[selected_effective_index]
            zero_legal_mass_hold = retained_effective_mass == 0.0
            if zero_legal_mass_hold:
                hold_action = (
                    ControlAction.CONTINUE.value
                    if state.speaker_active else ControlAction.LISTEN.value
                )
                if hold_action not in legal:
                    raise RuntimeError(
                        "native Control6 has no legal physical probability and its hold action is illegal"
                    )
                candidate_action_value = hold_action
            action_value, hysteresis_count, hysteresis_pending = self._apply_hysteresis(
                state,
                candidate_action_value,
            )
            action = ControlAction(action_value)
            selected_index = action_indices[candidate_action_value]
            fallback_reason = (
                "two_distinct_consecutive_fresh_ticks_required"
                if hysteresis_pending
                else "zero_legal_physical_mass_safe_hold" if zero_legal_mass_hold else None
            )
        raw_action = ACTIONS[raw_index]
        raw_token_id = ACTION_TO_TOKEN_ID[raw_action]
        selected_token_id = ACTION_TO_TOKEN_ID[candidate_action_value]
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
            "native_action": candidate_action_value,
            "native_action_probability": confidence,
            "selected_action_probability": confidence,
            "selected_action": candidate_action_value,
            "native_control6_effective_candidate": candidate_action_value,
            "native_control6_effective_order": list(effective_order),
            "native_control6_effective_actions": list(effective_actions),
            "native_control6_effective_members": {
                name: list(effective_members[name]) for name in effective_order
            },
            "native_control6_legal_effective_mass": retained_effective_mass,
            "native_control6_selected_effective_probability": selected_effective_probability,
            "native_control6_hysteresis_count": hysteresis_count,
            "native_control6_hysteresis_pending": hysteresis_pending,
            "native_control6_hysteresis_applied_action": action.value,
            "native_control6_zero_legal_mass_safe_hold": zero_legal_mass_hold,
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
                "native_control6_batched_raw_scores": bool(use_phase_less_snapshot),
                "native_control6_batched_raw_scores_diagnostic_only": not use_phase_less_snapshot,
                "native_control6_batch_manifest_id": raw_score_snapshot.manifest_id,
                "native_control6_native_output_generation": raw_score_snapshot.output_generation,
                "native_control6_listener_state_version": raw_score_snapshot.listener_state_version,
                "native_control6_query_index": raw_score_snapshot.query_index,
            })
        if self.trace_scores:
            base_logits = torch.nn.functional.linear(
                native_hidden, self.adapter.base_rows
            ).reshape(-1).float()
            legal_row_mask = torch.tensor(
                [
                    any(
                        action in effective_members[bucket]
                        and effective_actions[bucket_index] in legal
                        for bucket_index, bucket in enumerate(effective_order)
                    )
                    for action in ACTIONS
                ],
                dtype=torch.bool,
                device=logits.device,
            )
            legal_row_probabilities = torch.softmax(
                logits.masked_fill(~legal_row_mask, -torch.inf), dim=0
            )
            diagnostics.update({
                "native_action_order": list(ACTIONS),
                "native_action_base_logits": base_logits.detach().cpu().tolist(),
                "native_action_logits": logits.detach().cpu().tolist(),
                "native_action_probabilities": legal_row_probabilities.detach().cpu().tolist(),
                "native_action_raw_probabilities": raw_probabilities.detach().cpu().tolist(),
                "native_action_effective_order": list(effective_order),
                "native_action_effective_probabilities": {
                    name: float(effective_probabilities[index].detach().cpu())
                    for index, name in enumerate(effective_order)
                },
                "native_action_legal_effective_probabilities": {
                    name: float(legal_effective_probabilities[index].detach().cpu())
                    for index, name in enumerate(effective_order)
                },
            })
        return PolicyDecision(
            action=action,
            frame_id=int(state.frame_id),
            confidence=confidence,
            reason=("gemma_native_control6_hysteresis_hold" if hysteresis_pending else "gemma_native_control6"),
            generation_id=int(state.generation_id),
            listener_frame_id=int(source_frame),
            source="gemma_lm_head_selected_rows+rank16_delta:ListenerOutput.gemma_hidden_t",
            session=str(state.session_id),
            diagnostics=diagnostics,
        )

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    def reset(self) -> None:
        self._hysteresis_session_id = None
        self._hysteresis_speaker_active = None
        self._hysteresis_frame_id = None
        self._hysteresis_candidate = None
        self._hysteresis_count = 0

    def _apply_hysteresis(self, state: PolicyInput, candidate: str) -> tuple[str, int, bool]:
        """Require two fresh adjacent listener frames for SPEAK or YIELD."""
        session_id = str(state.session_id)
        speaker_active = bool(state.speaker_active)
        frame_id = int(state.frame_id)
        if (
            self._hysteresis_session_id != session_id
            or self._hysteresis_speaker_active != speaker_active
        ):
            self.reset()
            self._hysteresis_session_id = session_id
            self._hysteresis_speaker_active = speaker_active

        if candidate not in {ControlAction.SPEAK.value, ControlAction.YIELD.value}:
            self._hysteresis_frame_id = frame_id
            self._hysteresis_candidate = None
            self._hysteresis_count = 0
            return candidate, 0, False

        last_frame = self._hysteresis_frame_id
        if last_frame == frame_id:
            if self._hysteresis_candidate != candidate:
                self._hysteresis_candidate = None
                self._hysteresis_count = 0
            count = self._hysteresis_count
        elif last_frame is None or frame_id != last_frame + 1:
            count = 1
            self._hysteresis_candidate = candidate
        elif self._hysteresis_candidate == candidate:
            count = min(2, self._hysteresis_count + 1)
        else:
            count = 1
            self._hysteresis_candidate = candidate

        self._hysteresis_frame_id = frame_id
        self._hysteresis_count = count
        if count >= 2:
            return candidate, count, False
        hold_action = (
            ControlAction.CONTINUE.value if speaker_active else ControlAction.LISTEN.value
        )
        if hold_action not in tuple(state.legal_actions or ()):
            raise RuntimeError(
                f"native Control6 hysteresis hold action {hold_action} is not legal"
            )
        return hold_action, count, True


__all__ = ["GemmaNativeControl6Policy", "legal_control_actions"]

CONTROL6_POLICY_TYPES = (DuplexControl6Policy, GemmaNativeControl6Policy)


def is_controller6_policy(policy: Any) -> bool:
    """True only for the legacy MLP and native-row six-action policies."""
    return isinstance(policy, CONTROL6_POLICY_TYPES)


def legal_control_actions(
    *,
    speaker_active: bool,
    user_turn_pending: bool,
    peer_turn_active: bool,
) -> tuple[str, ...]:
    """Return native Control6's legal physical actions for this tick."""
    if speaker_active:
        return (ControlAction.CONTINUE.value, ControlAction.YIELD.value)
    if user_turn_pending or peer_turn_active:
        return (
            ControlAction.LISTEN.value,
            ControlAction.BACKCHANNEL.value,
            ControlAction.SPEAK.value,
        )
    return (ControlAction.LISTEN.value,)
