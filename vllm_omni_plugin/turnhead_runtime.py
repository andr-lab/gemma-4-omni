# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/turnhead_runtime.py — TurnDetectorHead + StreamingDuplexController
per il data plane full-duplex L2.

Architettura (ADR-027, benchmark_real_stream_turnhead.py):
    z_t = [h_t(2048) ∥ u_mel(128) ∥ d_u_mel(128)] ∈ R^2304
        h_t     = hidden del THINKER (Gemma 4 E2B, 1536) sui frame agente,
                  proiettati a 2048 dal TemporalResidualAdapter (ht_adapter)
                  + causal_conv addestrata (NO talker: la semantica del turno
                  vive nello stato del thinker, ADR-027).
        u_mel   = mel utente 12.5Hz (media su 8 frame mel 128-bin)
        d_u_mel = delta temporale del mel utente
    → TurnDetectorHead (in_dim=2304, causal Conv1D k=3 + Pre-LN Deep Residual
      MLP 512→512→128) → 5 classi [SPEAK, YIELD, BARGE_IN, USER_BC, EMIT_BC]
    → StreamingDuplexController (FSM: barge-in flush <10ms, zero-reset KV,
      yield stop, backchannel cooldown)

Checkpoint:
    - turn_dynamics_v1_final.pt  (action_head.* → TurnDetectorHead, in_dim 2304)
    - ht_adapter_temporal2_1000.pt (adapter.* → TemporalResidualAdapter 1536→2048)
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.logger import init_logger

from vllm_omni_plugin.models.v9_projector import CausalConv1D
from vllm_omni_plugin.turn_detector_head import (
    ACTION_NAMES,
    TurnDetectorConfig,
    TurnDetectorHead,
    StreamingDuplexController,
)

logger = init_logger(__name__)

# Frame mel del Thinker Gemma4 (bit-pari col Gemma4AudioFeatureExtractor):
# 12.5 Hz → 8 frame mel per frame di turno.
MEL_FRAMES_PER_TURN = 8
MEL_BINS = 128


class TemporalResidualAdapter(nn.Module):
    """Proiezione hidden thinker 1536 → 2048 (TemporalResidualAdapter).

    Ricetta training (train_turnhead_v5_final.py): ResidualAdapter con
    causal_conv 1D (k=3) + skip_proj + MLP + LayerNorm, seguita dal modulo
    temporale. Qui carichiamo SOLO i pesi ``adapter.*`` (la proiezione) —
    il modulo temporale serve al training, non alla classificazione frame.
    """

    def __init__(self) -> None:
        super().__init__()
        self.causal_conv = CausalConv1D(hidden_size=1536, kernel_size=3)
        self.skip_proj = nn.Linear(1536, 2048, bias=True)
        self.mlp = nn.Sequential(
            nn.Linear(1536, 3072),
            nn.SiLU(),
            nn.Linear(3072, 2048),
        )
        self.norm = nn.LayerNorm(2048)

    def forward(self, h_thinker: torch.Tensor) -> torch.Tensor:
        # h_thinker: (B, T, 1536) hidden del thinker sui frame agente.
        h = self.causal_conv(h_thinker)
        return self.norm(self.skip_proj(h) + self.mlp(h))


class Gemma4OmniTurnHeadRuntime:
    """Runtime L2: turn detector + controller per il data plane duplex."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._loaded = False
        self.head: TurnDetectorHead | None = None
        self.adapter: TemporalResidualAdapter | None = None
        self.controller = StreamingDuplexController(
            barge_in_consecutive_frames=1,
            yield_consecutive_frames=2,
            emit_bc_cooldown_frames=5,
            maintain_kv_cache_on_barge_in=True,
        )

    # ------------------------------------------------------------------
    # Caricamento checkpoint (fail-soft: senza i pesi la turnhead è disattiva)
    # ------------------------------------------------------------------
    def load(self, *, checkpoints_dir: str | None = None) -> bool:
        if self._loaded:
            return True
        ckpt_dir = checkpoints_dir or os.environ.get("GEMMA4_OMNI_CKPT_DIR", "/workspace/checkpoints")
        turn_ckpt = os.path.join(ckpt_dir, "turn_dynamics_v1_final.pt")
        if not os.path.isfile(turn_ckpt):
            logger.warning("[TurnHead] checkpoint turnhead non trovato: %s", turn_ckpt)
            return False
        adapter_candidates = [
            os.path.join(ckpt_dir, "ht_adapter_temporal2_1000.pt"),
            "/workspace/_archive_ckpts/temporal2/ht_adapter_temporal2_1000.pt",
            "/workspace/_archive_ckpts/temporal2/ht_adapter_stage2r2_final.pt",
        ]
        adapter_ckpt = next((p for p in adapter_candidates if os.path.isfile(p)), None)
        try:
            sd_turn = torch.load(turn_ckpt, map_location="cpu", weights_only=False)
            in_dim = int(sd_turn.get("in_dim", 2304))
            self.head = TurnDetectorHead(TurnDetectorConfig(in_dim=in_dim)).to(self.device).eval()
            extracted: dict[str, torch.Tensor] = {}
            st_dict = sd_turn.get("model_state_dict", sd_turn)
            for k, v in st_dict.items():
                if k.startswith("action_head."):
                    extracted[k.replace("action_head.", "")] = v
            self.head.load_state_dict(extracted, strict=False)
            self.head.eval()
            logger.info("[TurnHead] TurnDetectorHead caricata (in_dim=%d) da %s", in_dim, turn_ckpt)

            if adapter_ckpt:
                sd_adapt = torch.load(adapter_ckpt, map_location="cpu", weights_only=False)
                self.adapter = TemporalResidualAdapter().to(self.device).eval()
                adapt_weights: dict[str, torch.Tensor] = {}
                for k, v in sd_adapt.items():
                    if k.startswith("adapter."):
                        adapt_weights[k.replace("adapter.", "")] = v
                missing, unexpected = self.adapter.load_state_dict(adapt_weights, strict=False)
                logger.info(
                    "[TurnHead] ht_adapter caricato da %s (missing=%d, unexpected=%d)",
                    adapter_ckpt,
                    len(missing),
                    len(unexpected),
                )
            else:
                logger.warning("[TurnHead] ht_adapter non trovato: la turnhead userà h_t=0 (mel-only).")
            self._loaded = True
            return True
        except Exception as exc:  # fail-soft
            logger.error("[TurnHead] caricamento fallito (%s): turnhead disattivata.", exc)
            self.head = None
            self.adapter = None
            return False

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    def user_mel_features(self, user_mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """u_mel (128) + d_u_mel (128) a 12.5Hz dal mel utente grezzo.

        Args:
            user_mel: (N*8, 128) mel 128-bin del Thinker (10ms/frame, 16kHz).
        Returns:
            (e_user (128,), d_e_user (128,)) — feature per l'ultimo frame di turno.
        """
        n = max(1, len(user_mel) // MEL_FRAMES_PER_TURN)
        frames = user_mel[: n * MEL_FRAMES_PER_TURN].reshape(n, MEL_FRAMES_PER_TURN, MEL_BINS)
        e_user = frames.mean(axis=1)  # (n, 128)
        d_e_user = np.zeros_like(e_user)
        d_e_user[1:] = e_user[1:] - e_user[:-1]
        return e_user[-1], d_e_user[-1]

    def _h_t_for_frame(self, hidden_thinker: torch.Tensor) -> torch.Tensor:
        """Proietta gli hidden del thinker (1536) a 2048 via ht_adapter.

        Args:
            hidden_thinker: (1, T, 1536) hidden_states[-1] del Thinker sui
                            frame agente.
        Returns:
            (1, 2048) h_t dell'ultimo frame.
        """
        if self.adapter is None:
            return torch.zeros(1, 2048, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            h = hidden_thinker.to(self.device)
            h = self.adapter(h)  # (1, T, 2048)
            return h[:, -1, :].float()  # (1, 2048)

    # ------------------------------------------------------------------
    # Decisione frame-sincrona
    # ------------------------------------------------------------------
    def decide(
        self,
        *,
        hidden_thinker: torch.Tensor | None,
        user_mel: np.ndarray | None,
    ) -> dict[str, Any]:
        """Classifica il frame corrente e fa avanzare la FSM.

        Returns:
            dict con predizione, probabilità e azione del controller:
            {pred, pred_name, probs, event} — event contiene i flag
            (flush_audio_buffer, stop_synthesis, trigger_backchannel, ...).
        """
        if not self._loaded or self.head is None:
            return {
                "pred": 0,
                "pred_name": "SPEAK",
                "probs": None,
                "event": {"flush_audio_buffer": False, "stop_synthesis": False, "trigger_backchannel": False},
                "active": False,
            }
        with torch.no_grad():
            if hidden_thinker is not None:
                h_t = self._h_t_for_frame(hidden_thinker)  # (1, 2048)
            else:
                h_t = torch.zeros(1, 2048, device=self.device, dtype=torch.float32)
            if user_mel is not None and len(user_mel):
                e_user, d_e_user = self.user_mel_features(np.asarray(user_mel, dtype=np.float32))
                z_extra = np.concatenate([e_user, d_e_user]).astype(np.float32)  # (256,)
                z = torch.cat(
                    [h_t, torch.from_numpy(z_extra).unsqueeze(0).to(self.device)],
                    dim=-1,
                )  # (1, 2304)
            else:
                z = torch.cat([h_t, torch.zeros(1, 256, device=self.device)], dim=-1)
            logits = self.head(z)  # (1, 5)
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


__all__ = [
    "MEL_FRAMES_PER_TURN",
    "MEL_BINS",
    "Gemma4OmniTurnHeadRuntime",
    "TemporalResidualAdapter",
]
