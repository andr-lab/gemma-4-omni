# -*- coding: utf-8 -*-
"""
duplex_kernel/turnhead.py — TurnDetectorHead + StreamingDuplexController nel kernel.

La turndynamics ascolta l'audio utente (u_mel 12.5Hz + Δu_mel) e lo stato del
THINKER e governa il flusso: YIELD (auto-commit), BARGE_IN (flush), backchannel.

Feature (ADR-027, benchmark_real_stream_turnhead.py — versione più recente):
    z_t = [h_t(2048) ∥ u_mel(128) ∥ Δu_mel(128)] ∈ R^2304

    h_t(2048) = stato del THINKER proiettato a 2048. L'ht_adapter legacy è in
    archive: il proiettore moderno è il V9 (ADR-024) che produce h_pfx
    (1, 16, 2048) dagli hidden del thinker. h_t = media degli h_pfx frame
    (o ultimo frame) — nessuna dipendenza da componenti legacy.

Checkpoint: turn_dynamics_v1_final.pt (action_head.* → in_dim 2304).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("duplex_kernel.turnhead")

MEL_FRAMES_PER_TURN = 8  # 8 frame mel (10ms) per frame di turno (12.5Hz)
MEL_BINS = 128
MEL_FRAME = 320
MEL_HOP = 160
TURNHEAD_HIDDEN_DIM = 2048
TURNHEAD_INPUT_DIM = TURNHEAD_HIDDEN_DIM + 2 * MEL_BINS
_MEL_FILTERS: np.ndarray | None = None


def _mel_filters() -> np.ndarray:
    """Bank HTK 128-bin identica al Gemma4AudioFeatureExtractor."""
    global _MEL_FILTERS
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS
    hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel_to_hz = lambda m: 700.0 * (np.power(10.0, m / 2595.0) - 1.0)
    freqs = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(8000.0), MEL_BINS + 2))
    fft_freqs = np.linspace(0.0, 8000.0, 257)
    slopes = freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / np.diff(freqs)[:-1]
    up = slopes[:, 2:] / np.diff(freqs)[1:]
    _MEL_FILTERS = np.maximum(0.0, np.minimum(down, up)).astype(np.float32)
    return _MEL_FILTERS


def _gemma_mel(frame: np.ndarray) -> np.ndarray:
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(MEL_FRAME) / MEL_FRAME)
    magnitude = np.abs(np.fft.rfft(frame * window, n=512))
    return np.log(magnitude @ _mel_filters() + 1e-3).astype(np.float32)


class TurnDetectorHead(nn.Module):
    """Classificatore causale a 5 classi (SPEAK/YIELD/BARGE_IN/USER_BC/EMIT_BC).

    Architettura: causal Conv1D (k=3, left-pad) + Pre-LN Deep Residual MLP
    (512→512→128) + classifier 5. Input z_t (B, T, in_dim).
    """

    def __init__(self, in_dim: int = TURNHEAD_INPUT_DIM, hidden_dim: int = 512, mid_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.in_dim = int(in_dim)
        self.causal_conv = nn.Conv1d(self.in_dim, self.in_dim, kernel_size=3, padding=0, bias=False)
        with torch.no_grad():
            self.causal_conv.weight.zero_()
            self.causal_conv.weight[:, :, -1] = torch.eye(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, mid_dim)
        self.ln3 = nn.LayerNorm(mid_dim)
        self.classifier = nn.Linear(mid_dim, num_classes)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, T, D) o (B, D)
        squeeze = z.dim() == 2
        if squeeze:
            z = z.unsqueeze(1)
        zt = z.transpose(1, 2)
        zp = F.pad(zt, (2, 0))
        zc = self.causal_conv(zp).transpose(1, 2)
        x = self.act(self.ln1(self.fc1(zc)))
        x = x + self.act(self.ln2(self.fc2(x)))
        x = self.act(self.ln3(self.fc3(x)))
        logits = self.classifier(x)
        return logits.squeeze(1) if squeeze else logits


ACTION_NAMES = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]


class StreamingDuplexController:
    """FSM del duplex (ADR-027): barge-in flush <10ms, yield stop, backchannel."""

    def __init__(self, barge_in_frames: int = 1, yield_frames: int = 2, bc_cooldown: int = 5):
        self.barge_in_thresh = barge_in_frames
        self.yield_thresh = yield_frames
        self.bc_cooldown_max = bc_cooldown
        self.current_state = "IDLE"  # IDLE | LISTENING | SPEAKING
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0

    def step(self, action: int, confidence: float = 1.0) -> dict[str, Any]:
        del confidence
        event = {
            "flush_audio_buffer": False,
            "stop_synthesis": False,
            "trigger_backchannel": False,
            "reset_kv_cache": False,
            "state_changed": False,
            "state": self.current_state,
        }
        if self.bc_cooldown > 0:
            self.bc_cooldown -= 1
        if self.current_state == "SPEAKING":
            if action == 2:  # BARGE_IN
                self.barge_in_counter += 1
                if self.barge_in_counter >= self.barge_in_thresh:
                    self.current_state = "LISTENING"
                    self.barge_in_counter = 0
                    event.update(flush_audio_buffer=True, stop_synthesis=True, state_changed=True)
            else:
                self.barge_in_counter = 0
            if action == 1:  # YIELD
                self.yield_counter += 1
                if self.yield_counter >= self.yield_thresh:
                    self.current_state = "LISTENING"
                    self.yield_counter = 0
                    event.update(stop_synthesis=True, state_changed=True)
            else:
                self.yield_counter = 0
        else:  # LISTENING / IDLE
            self.barge_in_counter = 0
            if action == 0:  # SPEAK
                self.current_state = "SPEAKING"
                event["state_changed"] = True
            elif action == 4 and self.bc_cooldown == 0:  # EMIT_BACKCHANNEL
                event["trigger_backchannel"] = True
                self.bc_cooldown = self.bc_cooldown_max
        event["state"] = self.current_state
        return event

    def reset(self) -> None:
        self.current_state = "IDLE"
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0


class TurnHeadRuntime:
    """Runtime turnhead: carica il checkpoint, classifica i frame, governa la FSM.

    h_t(2048) arriva dal V9 (h_pfx): set_thinker_hidden riceve h_pfx
    (1, 16, 2048) e ne usa la media come h_t del frame corrente.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.head: TurnDetectorHead | None = None
        self.controller = StreamingDuplexController()
        self._loaded = False
        self.load_error: str | None = None
        self._mel_buffer: np.ndarray = np.empty((0, MEL_BINS), dtype=np.float32)
        self._pcm_buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._pcm_buffer_start = 0
        self._pcm_samples_seen = 0
        self._next_mel_frame = 0
        self._h_t_last: torch.Tensor | None = None
        self._feature_history: list[torch.Tensor] = []

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def has_thinker_hidden(self) -> bool:
        return self._h_t_last is not None

    # ------------------------------------------------------------------
    def load(self, turnhead_ckpt: str) -> bool:
        """Load only a complete v1 2304D action head; reject partial weights."""
        self.head = None
        self._loaded = False
        self.load_error = None
        try:
            payload = torch.load(turnhead_ckpt, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict):
                raise ValueError("checkpoint deve contenere un mapping")
            in_dim = payload.get("in_dim")
            if isinstance(in_dim, bool) or not isinstance(in_dim, int) or in_dim != TURNHEAD_INPUT_DIM:
                raise ValueError(
                    f"checkpoint TurnDynamics v1 incompatibile: in_dim={in_dim!r}, "
                    f"atteso {TURNHEAD_INPUT_DIM} (=2048+128+128)"
                )
            state = payload.get("model_state_dict")
            if not isinstance(state, dict):
                raise ValueError("checkpoint privo di model_state_dict")
            action_state = {
                key[len("action_head."):]: value
                for key, value in state.items()
                if isinstance(key, str) and key.startswith("action_head.")
            }
            if not action_state:
                raise ValueError("checkpoint privo di pesi action_head.*")

            candidate = TurnDetectorHead(in_dim=in_dim)
            candidate.load_state_dict(action_state, strict=True)
            candidate = candidate.to(self.device).eval()
            self.head = candidate
            self._loaded = True
            logger.info("[turnhead] head caricata (in_dim=%d) da %s", in_dim, turnhead_ckpt)
            return True
        except Exception as exc:
            self.head = None
            self._loaded = False
            self.load_error = str(exc)
            logger.error("[turnhead] caricamento fallito: %s", exc)
            return False

    # ------------------------------------------------------------------
    def feed_user_mel(self, mel_chunk: np.ndarray) -> None:
        """Accumulate Gemma log-mel rows, retaining only the latest two turns."""
        mel = np.asarray(mel_chunk, dtype=np.float32)
        if mel.size == 0:
            return
        if mel.ndim != 2 or mel.shape[-1] != MEL_BINS:
            raise ValueError(f"mel_chunk deve avere shape (N,{MEL_BINS}), got {mel.shape}")
        self._mel_buffer = np.concatenate((self._mel_buffer, mel), axis=0)
        if len(self._mel_buffer) > MEL_FRAMES_PER_TURN * 2:
            self._mel_buffer = self._mel_buffer[-MEL_FRAMES_PER_TURN * 2:]

    def feed_pcm(self, pcm_chunk: np.ndarray, *, finalize_frame: bool = False) -> int:
        """Extract causal Gemma log-mel rows from incremental PCM.

        The Gemma frontend's 321-sample unfold needs one sentinel sample beyond
        the 320-sample window. ``finalize_frame`` supplies the same zero-valued
        sentinel at the end of a complete 80ms policy tick, so all eight mel
        rows are available without waiting for the next turn tick. PCM storage
        is a rolling window, not a session-length accumulator.
        """
        if not isinstance(finalize_frame, bool):
            raise TypeError("finalize_frame deve essere bool")
        pcm = np.asarray(pcm_chunk, dtype=np.float32).reshape(-1)
        if not pcm.size:
            return 0
        if not np.isfinite(pcm).all():
            raise ValueError("pcm_chunk contiene valori non finiti")
        if self._pcm_buffer_start + self._pcm_buffer.size != self._pcm_samples_seen:
            raise RuntimeError("PCM rolling buffer ha un offset assoluto incoerente")

        self._pcm_buffer = np.concatenate((self._pcm_buffer, pcm))
        self._pcm_samples_seen += int(pcm.size)
        buffer_end = self._pcm_samples_seen
        rows = []
        while True:
            required_end = self._next_mel_frame * MEL_HOP + MEL_HOP
            ready = buffer_end > required_end or (finalize_frame and buffer_end == required_end)
            if not ready:
                break
            start = self._next_mel_frame * MEL_HOP - MEL_HOP
            frame = np.zeros(MEL_FRAME, dtype=np.float32)
            src_start = max(0, start)
            src_end = min(buffer_end, start + MEL_FRAME)
            if src_start < self._pcm_buffer_start:
                raise RuntimeError("PCM rolling buffer ha scartato un campione ancora necessario")
            if src_end > src_start:
                local_start = src_start - self._pcm_buffer_start
                dst_start = src_start - start
                dst_end = dst_start + (src_end - src_start)
                frame[dst_start:dst_end] = self._pcm_buffer[local_start:local_start + (src_end - src_start)]
            rows.append(_gemma_mel(frame))
            self._next_mel_frame += 1

        keep_from = max(0, self._next_mel_frame * MEL_HOP - MEL_HOP)
        discard = min(max(0, keep_from - self._pcm_buffer_start), self._pcm_buffer.size)
        if discard:
            self._pcm_buffer = self._pcm_buffer[discard:]
            self._pcm_buffer_start += int(discard)
        if rows:
            self.feed_user_mel(np.asarray(rows, dtype=np.float32))
        return len(rows)

    def set_thinker_hidden(self, h_pfx: Any | None) -> bool:
        """Store only a shape-compatible V9 h_t (2048D); clear stale features otherwise."""
        self._h_t_last = None
        if h_pfx is None:
            return False
        try:
            hp = torch.as_tensor(h_pfx, device=self.device)
        except (TypeError, ValueError, RuntimeError) as exc:
            logger.debug("[turnhead] hidden ignorato: %s", exc)
            return False
        if hp.dim() == 1 and tuple(hp.shape) == (TURNHEAD_HIDDEN_DIM,):
            hp = hp.reshape(1, 1, TURNHEAD_HIDDEN_DIM)
        elif hp.dim() == 2 and hp.shape[-1] == TURNHEAD_HIDDEN_DIM:
            hp = hp.unsqueeze(1) if hp.shape[0] == 1 else hp.unsqueeze(0)
        if (
            hp.dim() != 3
            or hp.shape[0] != 1
            or hp.shape[1] < 1
            or hp.shape[-1] != TURNHEAD_HIDDEN_DIM
        ):
            logger.debug("[turnhead] hidden ignorato: atteso (1,T,%d), got %s", TURNHEAD_HIDDEN_DIM, tuple(hp.shape))
            return False
        with torch.no_grad():
            self._h_t_last = hp.to(dtype=torch.float32).mean(dim=1)  # (1, 2048)
        return True

    def decide(self) -> dict[str, Any]:
        """Classifica il frame corrente e fa avanzare la FSM. Ritorna l'evento."""
        if not self._loaded or self.head is None:
            return {
                "pred": None,
                "pred_name": "UNAVAILABLE",
                "probs": [],
                "event": {"flush_audio_buffer": False, "stop_synthesis": False, "trigger_backchannel": False},
                "active": False,
            }
        with torch.no_grad():
            if len(self._mel_buffer) >= MEL_FRAMES_PER_TURN:
                frames = self._mel_buffer[-MEL_FRAMES_PER_TURN:].reshape(1, MEL_FRAMES_PER_TURN, MEL_BINS)
                e_user = frames.mean(axis=1)  # (1,128)
                d_user = np.zeros_like(e_user)
                if len(self._mel_buffer) >= MEL_FRAMES_PER_TURN * 2:
                    prev = self._mel_buffer[-MEL_FRAMES_PER_TURN * 2:-MEL_FRAMES_PER_TURN].reshape(
                        1, MEL_FRAMES_PER_TURN, MEL_BINS
                    ).mean(axis=1)
                    d_user = e_user - prev
                extra = torch.from_numpy(np.concatenate([e_user, d_user], axis=-1).astype(np.float32)).to(self.device)
            else:
                extra = torch.zeros(1, 256, device=self.device)
            if self._h_t_last is not None:
                z = torch.cat([self._h_t_last, extra], dim=-1)  # (1, 2304)
            else:
                z = torch.cat([
                    torch.zeros(1, TURNHEAD_HIDDEN_DIM, device=self.device), extra
                ], dim=-1)
            if z.shape[-1] != self.head.in_dim:
                raise RuntimeError(
                    f"feature TurnHead incompatibili: z={z.shape[-1]}, head={self.head.in_dim}"
                )
            # The trained causal Conv1D has a three-tick receptive field.
            # Supplying a 2D tensor here silently resets its left context on
            # every call, so classify the newest frame with the last 3 real
            # feature vectors and retain only that bounded causal history.
            feature_history = (self._feature_history + [z.detach()])[-3:]
            sequence = torch.stack(feature_history, dim=1)  # (1, T<=3, 2304)
            logits = self.head(sequence)
            if (
                logits.ndim != 3
                or logits.shape[0] != 1
                or logits.shape[1] != sequence.shape[1]
                or logits.shape[2] != len(ACTION_NAMES)
            ):
                raise RuntimeError(f"logits TurnHead inattesi: {tuple(logits.shape)}")
            logits = logits[:, -1, :]
            self._feature_history = feature_history
            probs = torch.softmax(logits, dim=-1)[0]
            pred = int(torch.argmax(probs).item())
            event = self.controller.step(pred, confidence=float(probs[pred].item()))
        return {
            "pred": pred,
            "pred_name": ACTION_NAMES[pred],
            "probs": probs.detach().cpu().numpy().tolist(),
            "event": event,
            "active": True,
        }

    def reset(self) -> None:
        self.controller.reset()
        self._mel_buffer = np.empty((0, MEL_BINS), dtype=np.float32)
        self._pcm_buffer = np.empty(0, dtype=np.float32)
        self._pcm_buffer_start = 0
        self._pcm_samples_seen = 0
        self._next_mel_frame = 0
        self._h_t_last = None
        self._feature_history.clear()


__all__ = ["ACTION_NAMES", "StreamingDuplexController", "TurnDetectorHead", "TurnHeadRuntime"]
