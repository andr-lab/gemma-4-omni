"""Native six-row Gemma policy for fresh per-frame listener hidden states."""
from __future__ import annotations

import time
import math
import numbers
from typing import Any

import numpy as np
import torch

from duplex_kernel.acoustic_phase_features import EPSILON, TICK_SAMPLES
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


def native_control6_effective_members(
    speaker_active: bool,
    user_activity_detected: bool,
) -> dict[str, tuple[str, ...]]:
    """Map each raw Control6 row into the physical action buckets for this context."""
    if speaker_active:
        if user_activity_detected:
            return {
                "HOLD": ("CONTINUE",),
                "SPEAK": (),
                "YIELD": ("LISTEN", "YIELD", "REPLAN"),
                "BACKCHANNEL": (),
            }
        return {
            "HOLD": ("LISTEN", "CONTINUE"),
            "SPEAK": (),
            "YIELD": ("YIELD", "REPLAN"),
            "BACKCHANNEL": (),
        }
    return {
        "HOLD": ("LISTEN", "CONTINUE"),
        "SPEAK": ("SPEAK", "REPLAN"),
        "YIELD": (),
        "BACKCHANNEL": ("BACKCHANNEL",),
    }


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
        high_confidence_single_tick_threshold: float = 0.85,
        minimum_transition_confidence: float = 0.70,
        early_speak_confidence_threshold: float = 0.80,
        hysteresis_required_ticks: int = 2,
        speech_rms_floor: float = 0.001,
        phase_profile: str = "legacy",
    ) -> None:
        if not isinstance(adapter, NativeControl6LoRA):
            raise TypeError("gemma_native_control6 requires NativeControl6LoRA")
        if not checkpoint_path:
            raise ValueError("gemma_native_control6 requires a loaded adapter checkpoint")
        if (
            isinstance(high_confidence_single_tick_threshold, bool)
            or not isinstance(high_confidence_single_tick_threshold, numbers.Real)
            or not math.isfinite(float(high_confidence_single_tick_threshold))
            or not 0.0 <= float(high_confidence_single_tick_threshold) <= 1.0
        ):
            raise ValueError("high_confidence_single_tick_threshold must be finite in [0, 1]")
        if (
            isinstance(minimum_transition_confidence, bool)
            or not isinstance(minimum_transition_confidence, numbers.Real)
            or not math.isfinite(float(minimum_transition_confidence))
            or not 0.0 <= float(minimum_transition_confidence)
            or float(minimum_transition_confidence)
            > float(high_confidence_single_tick_threshold)
        ):
            raise ValueError(
                "minimum_transition_confidence must be finite and between 0 and "
                "high_confidence_single_tick_threshold"
            )
        if (
            isinstance(early_speak_confidence_threshold, bool)
            or not isinstance(early_speak_confidence_threshold, numbers.Real)
            or not math.isfinite(float(early_speak_confidence_threshold))
            or not (
                float(early_speak_confidence_threshold) <= 0.0
                or float(minimum_transition_confidence)
                <= float(early_speak_confidence_threshold)
                <= float(high_confidence_single_tick_threshold)
            )
        ):
            raise ValueError(
                "early_speak_confidence_threshold must be finite and between "
                "minimum_transition_confidence and high_confidence_single_tick_threshold, or <= 0 to disable"
            )
        if (
            isinstance(hysteresis_required_ticks, bool)
            or not isinstance(hysteresis_required_ticks, int)
            or hysteresis_required_ticks < 1
        ):
            raise ValueError("hysteresis_required_ticks must be a positive integer")
        if (
            isinstance(speech_rms_floor, bool)
            or not isinstance(speech_rms_floor, numbers.Real)
            or not math.isfinite(float(speech_rms_floor))
            or not 0.0 < float(speech_rms_floor) <= 1.0
        ):
            raise ValueError("speech_rms_floor must be finite in (0, 1]")
        if phase_profile not in {"legacy", "native1540"}:
            raise ValueError("phase_profile must be 'legacy' or 'native1540'")
        self.adapter = adapter.eval()
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_path = str(checkpoint_path)
        self.trace_scores = bool(trace_scores)
        self.fused_readback = bool(fused_readback)
        self.high_confidence_single_tick_threshold = float(
            high_confidence_single_tick_threshold
        )
        self.minimum_transition_confidence = float(minimum_transition_confidence)
        self.early_speak_confidence_threshold = float(early_speak_confidence_threshold)
        self.hysteresis_required_ticks = hysteresis_required_ticks
        self.speech_rms_floor = float(speech_rms_floor)
        self.phase_profile = phase_profile
        self._hysteresis_session_id: str | None = None
        self._hysteresis_speaker_active: bool | None = None
        self._hysteresis_frame_id: int | None = None
        self._hysteresis_candidate: str | None = None
        self._hysteresis_count = 0
        self._endpoint_speak_session_id: str | None = None
        self._endpoint_speak_turn_id: int | None = None
        self._endpoint_speak_frame_id: int | None = None
        self._endpoint_speak_evidence_count = 0
        self._endpoint_speak_quiet_count = 0
        self._endpoint_speak_pending = False
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
                    listener_frame_id=int(state.frame_id),
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
            current_user_rms = self._current_user_rms(state.user_pcm)
            phase_t = state.native_control_phase_t
            ratio_activity_valid, user_assistant_log_ratio = (
                self._native1540_user_ratio_activity(
                    state,
                    current_user_rms=current_user_rms,
                    phase_t=phase_t,
                )
            )
            ratio_activity_detected = bool(
                ratio_activity_valid and user_assistant_log_ratio > -0.15
            )
            user_activity_detected = bool(
                state.peer_turn_active
                or current_user_rms >= self.speech_rms_floor
                or ratio_activity_detected
            )
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
            effective_members = native_control6_effective_members(
                speaker_active=bool(state.speaker_active),
                user_activity_detected=user_activity_detected,
            )
            if state.speaker_active:
                effective_actions = (
                    ControlAction.CONTINUE.value,
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                    ControlAction.BACKCHANNEL.value,
                )
            else:
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
            speak_row_is_legal = (
                ControlAction.SPEAK.value in legal
                and ControlAction.SPEAK.value in effective_members[ControlAction.SPEAK.value]
            )
            legal_speak_member_probability = (
                raw_probabilities[action_indices[ControlAction.SPEAK.value]]
                if speak_row_is_legal
                else zero_probability
            )
            early_speak_confidence_tensor = torch.where(
                legal_effective_mass_tensor > 0,
                legal_speak_member_probability
                / legal_effective_mass_tensor.clamp_min(torch.finfo(logits.dtype).tiny),
                zero_probability,
            )
            effective_index_tensor = normalized_legal_effective_probabilities.argmax()
            selected_effective_pre_normalization_probability_tensor = torch.gather(
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
                    selected_effective_pre_normalization_probability_tensor.to(dtype=torch.float32),
                    early_speak_confidence_tensor.to(dtype=torch.float32),
                    torch.isfinite(raw_probabilities).all().to(dtype=torch.float32),
                ))
                packet = packed_scores.detach().cpu().tolist()
                if not isinstance(packet, list) or len(packet) != 8:
                    raise RuntimeError("native Control6 produced invalid fused physical-action readback")
                try:
                    (
                        finite_value,
                        raw_value,
                        effective_value,
                        legal_mass,
                        confidence,
                        selected_effective_pre_normalization_probability,
                        readback_early_speak_confidence,
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
                        selected_effective_pre_normalization_probability,
                        readback_early_speak_confidence,
                    ))
                    or raw_value != int(raw_value)
                    or effective_value != int(effective_value)
                    or not 0 <= int(raw_value) < len(ACTIONS)
                    or not 0 <= int(effective_value) < len(effective_order)
                    or legal_mass < 0.0
                    or legal_mass > 1.000001
                    or not 0.0 <= confidence <= 1.0
                    or not 0.0 <= readback_early_speak_confidence <= 1.0
                    or not 0.0 <= selected_effective_pre_normalization_probability <= legal_mass
                ):
                    raise RuntimeError("native Control6 produced invalid fused physical-action readback")
                raw_index = int(raw_value)
                selected_effective_index = int(effective_value)
                retained_effective_mass = legal_mass
                selected_effective_probability = confidence
            else:
                raw_index = int(raw_index_tensor.item())
                selected_effective_index = int(effective_index_tensor.item())
                retained_effective_mass = legal_mass_value
                confidence = float(confidence_tensor.item())
                selected_effective_probability = confidence
                selected_effective_pre_normalization_probability = float(
                    selected_effective_pre_normalization_probability_tensor.item()
                )

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
            hysteresis_action_value, hysteresis_count, hysteresis_pending = self._apply_hysteresis(
                state,
                candidate_action_value,
                selected_effective_probability,
            )
            current_user_voiced = current_user_rms >= self.speech_rms_floor
            early_speak_activity_active = bool(
                state.peer_turn_active or current_user_voiced
            )
            quiet_peer_endpoint_window = bool(
                state.peer_turn_active and not current_user_voiced
            )
            early_speak_confidence: float | None = None
            early_speak_confidence_source = None
            early_speak_gate_active = bool(
                state.peer_turn_active or current_user_voiced
            )
            if early_speak_gate_active or self.trace_scores:
                early_speak_confidence_source = "legal_normalized_raw_speak_member"
                early_speak_confidence = (
                    readback_early_speak_confidence
                    if self.fused_readback
                    else float(early_speak_confidence_tensor.item())
                )
            early_speak_gate_confidence = (
                early_speak_confidence
                if early_speak_gate_active else None
            )
            early_speak_gated = (
                candidate_action_value == ControlAction.SPEAK.value
                and early_speak_gate_confidence is not None
                and early_speak_gate_confidence < self.early_speak_confidence_threshold
            )
            action_value = hysteresis_action_value
            if early_speak_gated:
                hold_action = (
                    ControlAction.CONTINUE.value
                    if state.speaker_active else ControlAction.LISTEN.value
                )
                if hold_action not in legal:
                    raise RuntimeError(
                        f"native Control6 early SPEAK hold action {hold_action} is not legal"
                    )
                action_value = hold_action
            (
                endpoint_speak_pending,
                endpoint_speak_evidence_count,
                endpoint_speak_quiet_count,
                endpoint_speak_ready,
            ) = self._update_endpoint_speak_memory(
                state,
                current_user_voiced=current_user_voiced,
                physical_speak_candidate=(
                    candidate_action_value == ControlAction.SPEAK.value
                ),
                physical_speak_probability=selected_effective_probability,
            )
            endpoint_speak_released = bool(
                endpoint_speak_ready
                and ControlAction.SPEAK.value in legal
                and not state.speaker_active
            )
            if endpoint_speak_released:
                action_value = ControlAction.SPEAK.value
                endpoint_speak_pending = False
                self._clear_endpoint_speak_memory()
            transition_below_minimum = (
                candidate_action_value in {
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                }
                and selected_effective_probability < self.minimum_transition_confidence
            )
            action = ControlAction(action_value)
            selected_index = action_indices[candidate_action_value]
            early_speak_fallback_reason = (
                "native_control6_early_speak_peer_quiet_hold"
                if quiet_peer_endpoint_window
                else "native_control6_early_speak_user_active_hold"
            )
            fallback_reason = (
                "native_control6_endpoint_speak_memory_release"
                if endpoint_speak_released
                else early_speak_fallback_reason
                if early_speak_gated
                else "native_control6_transition_confidence_below_minimum"
                if transition_below_minimum
                else "two_distinct_consecutive_fresh_ticks_required"
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
            "native_control6_selected_effective_pre_normalization_probability": (
                selected_effective_pre_normalization_probability
            ),
            "native_control6_hysteresis_count": hysteresis_count,
            "native_control6_hysteresis_pending": hysteresis_pending,
            "native_control6_hysteresis_applied_action": hysteresis_action_value,
            "native_control6_decision_action": action.value,
            "native_control6_early_speak_gated": early_speak_gated,
            "native_control6_early_speak_confidence": early_speak_confidence,
            "native_control6_early_speak_activity_active": early_speak_activity_active,
            "native_control6_early_speak_confidence_source": early_speak_confidence_source,
            "native_control6_early_speak_confidence_threshold": (
                self.early_speak_confidence_threshold
            ),
            "native_control6_endpoint_speak_pending": endpoint_speak_pending,
            "native_control6_endpoint_speak_evidence_count": endpoint_speak_evidence_count,
            "native_control6_endpoint_speak_quiet_count": endpoint_speak_quiet_count,
            "native_control6_endpoint_speak_ready": endpoint_speak_ready,
            "native_control6_endpoint_speak_released": endpoint_speak_released,
            "native_control6_endpoint_speak_turn_onset_frame_id": (
                state.user_turn_onset_frame_id
            ),
            "native_control6_hysteresis_required_ticks": (
                1
                if candidate_action_value in {
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                }
                and selected_effective_probability >= self.high_confidence_single_tick_threshold
                else self.hysteresis_required_ticks
            ),
            "native_control6_high_confidence_single_tick_threshold": self.high_confidence_single_tick_threshold,
            "native_control6_minimum_transition_confidence": self.minimum_transition_confidence,
            "native_control6_transition_confidence_eligible": not (
                candidate_action_value in {
                    ControlAction.SPEAK.value,
                    ControlAction.YIELD.value,
                }
                and selected_effective_probability < self.minimum_transition_confidence
            ),
            "native_control6_user_rms": current_user_rms,
            "native_control6_ratio_activity_valid": ratio_activity_valid,
            "native_control6_user_assistant_log_ratio": (
                user_assistant_log_ratio if ratio_activity_valid else None
            ),
            "native_control6_ratio_activity_detected": ratio_activity_detected,
            "native_control6_peer_turn_active": state.peer_turn_active,
            "native_control6_user_activity_detected": user_activity_detected,
            "native_control6_speech_rms_floor": self.speech_rms_floor,
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
            reason=(
                "gemma_native_control6_early_speak_peer_quiet_hold"
                if early_speak_gated and quiet_peer_endpoint_window
                else "gemma_native_control6_early_speak_user_active_hold"
                if early_speak_gated
                else "gemma_native_control6_hysteresis_hold"
                if hysteresis_pending
                else "gemma_native_control6"
            ),
            generation_id=int(state.generation_id),
            listener_frame_id=int(source_frame),
            source="gemma_lm_head_selected_rows+rank16_delta:ListenerOutput.gemma_hidden_t",
            session=str(state.session_id),
            diagnostics=diagnostics,
        )

    def update_hidden(self, hidden: Any | None) -> None:
        del hidden

    @staticmethod
    def _current_user_rms(user_pcm: Any | None) -> float:
        if user_pcm is None:
            return 0.0
        try:
            pcm = np.asarray(user_pcm, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("native Control6 current user PCM must be numeric") from exc
        if pcm.ndim != 1 or not pcm.size or not np.isfinite(pcm).all():
            raise ValueError("native Control6 current user PCM must be a finite mono frame")
        values = pcm.astype(np.float64, copy=False)
        return float(np.sqrt(np.mean(values * values, dtype=np.float64)))

    def _native1540_user_ratio_activity(
        self,
        state: PolicyInput,
        *,
        current_user_rms: float,
        phase_t: Any | None,
    ) -> tuple[bool, float]:
        """Use the existing ratio only with exact native1540 t-1 PCM provenance."""
        if self.phase_profile != "native1540" or phase_t is None:
            return False, 0.0
        assistant_pcm = state.native_control_assistant_pcm_prior
        assistant_frame_id = state.native_control_assistant_prior_frame_id
        if (
            assistant_pcm is None
            or isinstance(assistant_frame_id, bool)
            or not isinstance(assistant_frame_id, int)
            or assistant_frame_id != int(state.frame_id) - 1
            or current_user_rms <= EPSILON
        ):
            return False, 0.0
        try:
            assistant_values = np.asarray(assistant_pcm, dtype=np.float32)
            if (
                assistant_values.shape != (TICK_SAMPLES,)
                or not np.isfinite(assistant_values).all()
            ):
                return False, 0.0
            assistant_rms = self._current_user_rms(assistant_values)
            if assistant_rms < self.speech_rms_floor:
                return False, 0.0
            if isinstance(phase_t, torch.Tensor):
                phase_values = phase_t.detach().to(device="cpu", dtype=torch.float32).numpy()
            else:
                phase_values = np.asarray(phase_t, dtype=np.float32)
            phase_values = np.asarray(phase_values, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return False, 0.0
        if (
            phase_values.shape != (NATIVE_CONTROL6_PHASE_T_WIDTH,)
            or not np.isfinite(phase_values).all()
        ):
            return False, 0.0
        return True, float(phase_values[3])

    def reset(self) -> None:
        self._hysteresis_session_id = None
        self._hysteresis_speaker_active = None
        self._hysteresis_frame_id = None
        self._hysteresis_candidate = None
        self._hysteresis_count = 0
        self._clear_endpoint_speak_memory()

    def _clear_endpoint_speak_memory(self) -> None:
        self._endpoint_speak_session_id = None
        self._endpoint_speak_turn_id = None
        self._endpoint_speak_frame_id = None
        self._endpoint_speak_evidence_count = 0
        self._endpoint_speak_quiet_count = 0
        self._endpoint_speak_pending = False

    def _update_endpoint_speak_memory(
        self,
        state: PolicyInput,
        *,
        current_user_voiced: bool,
        physical_speak_candidate: bool,
        physical_speak_probability: float,
    ) -> tuple[bool, int, int, bool]:
        """Remember strong voiced SPEAK intent until a same-turn endpoint."""
        turn_id = state.user_turn_onset_frame_id
        if (
            self.early_speak_confidence_threshold <= 0.0
            or turn_id is None
            or state.speaker_active
        ):
            self._clear_endpoint_speak_memory()
            return False, 0, 0, False

        session_id = str(state.session_id)
        frame_id = int(state.frame_id)
        if (
            self._endpoint_speak_session_id != session_id
            or self._endpoint_speak_turn_id != int(turn_id)
        ):
            self._clear_endpoint_speak_memory()
            self._endpoint_speak_session_id = session_id
            self._endpoint_speak_turn_id = int(turn_id)

        last_frame_id = self._endpoint_speak_frame_id
        if last_frame_id == frame_id:
            ready = (
                self._endpoint_speak_pending
                and self._endpoint_speak_quiet_count >= 2
            )
            return (
                self._endpoint_speak_pending,
                self._endpoint_speak_evidence_count,
                self._endpoint_speak_quiet_count,
                ready,
            )
        if last_frame_id is not None and frame_id != last_frame_id + 1:
            self._endpoint_speak_evidence_count = 0
            self._endpoint_speak_quiet_count = 0

        if current_user_voiced:
            self._endpoint_speak_quiet_count = 0
            if not self._endpoint_speak_pending:
                qualifies = (
                    physical_speak_candidate
                    and physical_speak_probability
                    >= self.early_speak_confidence_threshold
                )
                self._endpoint_speak_evidence_count = (
                    self._endpoint_speak_evidence_count + 1 if qualifies else 0
                )
                if self._endpoint_speak_evidence_count >= 2:
                    self._endpoint_speak_pending = True
        elif self._endpoint_speak_pending:
            self._endpoint_speak_quiet_count += 1
        else:
            self._endpoint_speak_evidence_count = 0
            self._endpoint_speak_quiet_count = 0

        self._endpoint_speak_frame_id = frame_id
        ready = (
            self._endpoint_speak_pending
            and self._endpoint_speak_quiet_count >= 2
        )
        return (
            self._endpoint_speak_pending,
            self._endpoint_speak_evidence_count,
            self._endpoint_speak_quiet_count,
            ready,
        )

    def _apply_hysteresis(
        self, state: PolicyInput, candidate: str, effective_probability: float
    ) -> tuple[str, int, bool]:
        """Use one fresh frame only for high-confidence physical SPEAK/YIELD."""
        required_ticks = self.hysteresis_required_ticks
        if (
            candidate in {ControlAction.SPEAK.value, ControlAction.YIELD.value}
            and effective_probability >= self.high_confidence_single_tick_threshold
        ):
            required_ticks = 1
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

        if effective_probability < self.minimum_transition_confidence:
            hold_action = (
                ControlAction.CONTINUE.value if speaker_active else ControlAction.LISTEN.value
            )
            if hold_action not in tuple(state.legal_actions or ()):
                raise RuntimeError(
                    f"native Control6 hysteresis hold action {hold_action} is not legal"
                )
            self._hysteresis_frame_id = frame_id
            self._hysteresis_candidate = None
            self._hysteresis_count = 0
            return hold_action, 0, False

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
            count = min(required_ticks, self._hysteresis_count + 1)
        else:
            count = 1
            self._hysteresis_candidate = candidate

        self._hysteresis_frame_id = frame_id
        self._hysteresis_count = count
        if count >= required_ticks:
            return candidate, count, False
        hold_action = (
            ControlAction.CONTINUE.value if speaker_active else ControlAction.LISTEN.value
        )
        if hold_action not in tuple(state.legal_actions or ()):
            raise RuntimeError(
                f"native Control6 hysteresis hold action {hold_action} is not legal"
            )
        return hold_action, count, True


__all__ = [
    "GemmaNativeControl6Policy",
    "legal_control_actions",
    "native_control6_effective_members",
]

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
