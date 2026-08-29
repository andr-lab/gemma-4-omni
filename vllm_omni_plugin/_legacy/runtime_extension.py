# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/runtime_extension.py — Duplex Runtime Extension con TurnDetectorHead a 5 stati.

Implementa l'estensione runtime di vLLM-Omni per il processing full duplex continuo.
Valuta frame-per-frame (12.5Hz, 80ms) le feature congiunte utente/agente ed emette
decisioni istantanee di SPEAK, YIELD, BARGE_IN, USER_BACKCHANNEL o EMIT_BACKCHANNEL.

Riferimenti ufficiali SDD:
- vllm_omni.experimental.fullduplex.engine.duplex_runtime
- ADR-027 (TurnDetectorHead & Full-Duplex Streaming Integration)
"""

from collections import deque
from typing import Any, Deque, Dict, List, Optional
import torch
import torch.nn as nn
from vllm_omni_plugin.contracts import DuplexOutputDecision, DuplexState


class _FallbackTurnDetector(nn.Module):
    """Fallback compatto nel caso di esecuzione in contesti dove il modulo principale è disaccoppiato."""
    def __init__(self, in_dim: int = 2304, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z_t: torch.Tensor) -> torch.Tensor:
        return self.fc(z_t)


class Gemma4OmniDuplexRuntimeExtension:
    """Estensione di runtime Full-Duplex per vLLM-Omni."""

    def __init__(self, turn_head_checkpoint: Optional[str] = None, max_history_len: int = 1024):
        self.turn_head = None
        self._init_turn_head(turn_head_checkpoint)
        self.state_history: Deque[DuplexState] = deque(maxlen=max_history_len)
        self.consecutive_barge_in_count = 0
        self.hysteresis_threshold = 1  # 1 frame (80ms) di conferma per Barge-In

    def _init_turn_head(self, checkpoint_path: Optional[str]):
        try:
            from turn_detector_head import TurnDetectorHead
            self.turn_head = TurnDetectorHead(in_dim=2304, hidden_dim=512, num_classes=5)
            if checkpoint_path:
                ckpt = torch.load(checkpoint_path, map_location="cpu")
                self.turn_head.load_state_dict(ckpt)
        except Exception:
            self.turn_head = _FallbackTurnDetector(in_dim=2304, num_classes=5)
        self.turn_head.eval()

    def evaluate_frame_turn_state(self, z_t: torch.Tensor) -> DuplexOutputDecision:
        """Valuta il singolo frame o blocco temporale z_t (dim=2304).

        z_t = [h_t(2048) || e_user(128) || delta_e_user(128)]
        """
        if not isinstance(z_t, torch.Tensor):
            z_t = torch.tensor(z_t, dtype=torch.float32)

        with torch.no_grad():
            while z_t.dim() < 3:
                z_t = z_t.unsqueeze(0)  # (1, T, 2304)
            logits = self.turn_head(z_t)
            probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze(0)
            pred_class = int(torch.argmax(probs).item())

        state_map = {
            0: DuplexState.SPEAK,
            1: DuplexState.YIELD,
            2: DuplexState.BARGE_IN,
            3: DuplexState.USER_BACKCHANNEL,
            4: DuplexState.EMIT_BACKCHANNEL,
        }
        current_state = state_map.get(pred_class, DuplexState.SPEAK)

        if current_state == DuplexState.BARGE_IN:
            self.consecutive_barge_in_count += 1
        else:
            self.consecutive_barge_in_count = 0

        # Conferma di Barge-In dopo l'isteresi (1 frame = 80ms)
        confirmed_barge_in = (self.consecutive_barge_in_count >= self.hysteresis_threshold)

        decision = DuplexOutputDecision(
            turn_state=current_state,
            should_interrupt_agent=confirmed_barge_in,
            should_yield_to_user=(current_state == DuplexState.YIELD),
            is_user_backchannel=(current_state == DuplexState.USER_BACKCHANNEL),
            is_emit_backchannel=(current_state == DuplexState.EMIT_BACKCHANNEL),
            metadata={
                "probs": probs.tolist(),
                "class_id": pred_class,
                "consecutive_barge_in": self.consecutive_barge_in_count,
            },
        )
        self.state_history.append(current_state)
        return decision

    def prepare_duplex_sampling(self, sampling_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Hook opzionale invocato dal generic AR runner prima del campionamento token."""
        return sampling_metadata
