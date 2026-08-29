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


class TurnDetectorHead(nn.Module):
    """Classificatore causale a 5 classi (SPEAK/YIELD/BARGE_IN/USER_BC/EMIT_BC).

    Architettura: causal Conv1D (k=3, left-pad) + Pre-LN Deep Residual MLP
    (512→512→128) + classifier 5. Input z_t (B, T, in_dim).
    """

    def __init__(self, in_dim: int = 2304, hidden_dim: int = 512, mid_dim: int = 128, num_classes: int = 5):
        super().__init__()
        self.causal_conv = nn.Conv1d(in_dim, in_dim, kernel_size=3, padding=0, bias=False)
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
        self._mel_buffer: np.ndarray = np.empty((0, MEL_BINS), dtype=np.float32)
        self._h_t_last: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def load(self, turnhead_ckpt: str) -> bool:
        try:
            sd = torch.load(turnhead_ckpt, map_location="cpu", weights_only=False)
            in_dim = int(sd.get("in_dim", 2304))
            self.head = TurnDetectorHead(in_dim=in_dim).to(self.device).eval()
            extracted = {}
            st = sd.get("model_state_dict", sd)
            for k, v in st.items():
                if k.startswith("action_head."):
                    extracted[k.replace("action_head.", "")] = v
            self.head.load_state_dict(extracted, strict=False)
            self.head.eval()
            logger.info("[turnhead] head caricata (in_dim=%d) da %s", in_dim, turnhead_ckpt)
            self._loaded = True
            return True
        except Exception as exc:
            logger.error("[turnhead] caricamento fallito: %s", exc)
            return False

    # ------------------------------------------------------------------
    def feed_user_mel(self, mel_chunk: np.ndarray) -> None:
        """Accumula mel utente (N,128). Buffer rolling per il frame corrente."""
        self._mel_buffer = np.concatenate([self._mel_buffer, np.asarray(mel_chunk, dtype=np.float32)])
        if len(self._mel_buffer) > MEL_FRAMES_PER_TURN * 2:
            self._mel_buffer = self._mel_buffer[-MEL_FRAMES_PER_TURN * 2:]

    def set_thinker_hidden(self, h_pfx: torch.Tensor) -> None:
        """h_t dal V9: h_pfx (1, 16, 2048) → media sui frame → (1, 2048)."""
        if h_pfx is not None:
            with torch.no_grad():
                hp = h_pfx.to(self.device)
                if hp.dim() == 2:
                    hp = hp.unsqueeze(0)
                self._h_t_last = hp.mean(dim=1).float()  # (1, 2048)

    def decide(self) -> dict[str, Any]:
        """Classifica il frame corrente e fa avanzare la FSM. Ritorna l'evento."""
        if not self._loaded or self.head is None:
            return {
                "pred": 0,
                "pred_name": "SPEAK",
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
                z = torch.cat([torch.zeros(1, 2048, device=self.device), extra], dim=-1)
            logits = self.head(z)
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
        self._h_t_last = None


__all__ = ["ACTION_NAMES", "StreamingDuplexController", "TurnDetectorHead", "TurnHeadRuntime"]
