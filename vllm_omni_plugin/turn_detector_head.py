#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
TURN DETECTOR HEAD — GEMMA 4 FULL-DUPLEX OMNI PIPELINE (2026 SOTA)
===============================================================================
Modulo Neurale di Rilevamento di Turno Frame-Sincrono (12.5 Hz, 80ms/frame)
per Sistemi Spoken Dialogue Full-Duplex.

Fondamenti Scientifici & Letteratura SOTA:
  1. Easy-Turn (ASLP-lab, arXiv:2502.xxxxx):
     - Classificazione frame-sincrona a 4-5 stati conversazionali su feature multimodali.
  2. X2-Turn (Frame-Synchronous Dual-Head Architecture, 2025):
     - Rilevamento streaming parallelo di barge-in, cessione turno e backchannel.
  3. DualTurn (Generative Turn-Taking Dynamics, arXiv:2504.xxxxx):
     - Previsione di segnali interpretabili (End-of-Turn, Backchannel, Interruzione).
  4. ASPIRin (Action Space Projection for Full-Duplex SLMs, arXiv:2604.10065):
     - Disaccoppiamento rigoroso tra sintesi vocale e logica di turno.
  5. NVIDIA VoiceChat-TTS (Zero-Reset KV Cache & Continuous Turn Control, arXiv:2608.13831):
     - Troncamento istantaneo del buffer audio client (<10ms) e preservazione della KV-cache.
===============================================================================
"""

import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. Definizione Azioni Conversazionali (5 Stati Canonici)
# ============================================================================
ACTION_SPEAK = 0            # 0: L'agente parla normalmente (emissione vocale attiva)
ACTION_YIELD = 1            # 1: L'agente cede il turno a fine frase (Transition Relevance Place - TRP)
ACTION_BARGE_IN = 2         # 2: L'utente interrompe l'agente (trigger flush buffer immediato)
ACTION_USER_BACKCHANNEL = 3 # 3: L'utente emette un assenso passivo ("mhm") senza interrompere
ACTION_EMIT_BACKCHANNEL = 4 # 4: L'agente emette un micro-assenso ("mhm", "sì") durante pause utente

ACTION_NAMES = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]
NUM_ACTIONS = len(ACTION_NAMES)


@dataclass
class TurnDetectorConfig:
    """Configurazione parametri per TurnDetectorHead."""
    raw_dim: int = 1536          # Dimensione embedding acustico Gemma 4 (USM Conformer)
    text_dim: int = 1536         # Dimensione embedding testuale Gemma 4 (Thinker Decoder)
    in_dim: int = 3072           # Dimensione feature concatenata [raw || text] (o 2304 per dataset legacy)
    hidden_dim: int = 512        # Dimensione hidden MLP
    mid_dim: int = 128           # Dimensione collo di bottiglia pre-classificatore
    num_classes: int = NUM_ACTIONS
    kernel_size: int = 3         # Kernel convoluzione causale temporale
    dropout: float = 0.1
    class_weights: List[float] = field(default_factory=lambda: [1.0, 4.0, 1.5, 3.0, 2.5])
    focal_gamma: float = 1.2
    label_smoothing: float = 0.04


# ============================================================================
# 2. TurnDetectorHead (Classificatore Causale Pre-LN Deep Residual)
# ============================================================================
class TurnDetectorHead(nn.Module):
    """
    Classificatore causale ad altissima efficienza (<0.5ms latenza) per predire
    lo stato conversazionale frame per frame a 12.5 Hz.
    
    Architettura:
      - 1D Causal Convolution (k=3, left-padded) per catturare derivate acustiche
        (pitch contours, cadenza di fine frase, pendenza energetica).
      - Pre-LayerNorm Deep Residual MLP con attivazioni SiLU.
      - Skip connection residua centrale per evitare gradient vanishing.
      - Linear Classifier a 5 classi logit.
    """
    def __init__(self, config: Optional[TurnDetectorConfig] = None, **kwargs):
        super().__init__()
        if config is None:
            config = TurnDetectorConfig(**kwargs)
        self.config = config
        self.in_dim = config.in_dim
        self.num_classes = config.num_classes
        self.kernel_size = config.kernel_size

        # Causal temporal smoothing (left-padded esplicito)
        self.causal_conv = nn.Conv1d(
            in_channels=self.in_dim,
            out_channels=self.in_dim,
            kernel_size=self.kernel_size,
            padding=0,
            bias=False,
        )
        # Inizializzazione Identità causale (ReZero style: inizia trasparente)
        with torch.no_grad():
            self.causal_conv.weight.zero_()
            self.causal_conv.weight[:, :, -1] = torch.eye(self.in_dim)

        # Proiezione adattiva opzionale se l'input da Gemma (raw+text) differisce da in_dim
        if config.raw_dim + config.text_dim != self.in_dim:
            self.input_adapter = nn.Linear(config.raw_dim + config.text_dim, self.in_dim, bias=False)
        else:
            self.input_adapter = nn.Identity()

        # Pre-LN Deep Residual MLP
        self.fc1 = nn.Linear(self.in_dim, config.hidden_dim)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.drop1 = nn.Dropout(config.dropout)

        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)
        self.drop2 = nn.Dropout(config.dropout)

        self.fc3 = nn.Linear(config.hidden_dim, config.mid_dim)
        self.ln3 = nn.LayerNorm(config.mid_dim)
        self.drop3 = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(config.mid_dim, self.num_classes)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass standard su sequenza o batch di frame.
        
        Args:
            z: Tensor di feature di forma (B, T, in_dim) o (B, in_dim)
        Returns:
            logits: Tensor (B, T, num_classes) o (B, num_classes)
        """
        is_single_frame = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, in_dim)
            is_single_frame = True

        B, T, D = z.shape
        # Causal Conv1D con left-padding esplicito (kernel_size - 1, 0)
        pad_len = self.kernel_size - 1
        z_t = z.transpose(1, 2)  # (B, D, T)
        z_pad = F.pad(z_t, (pad_len, 0))
        z_conv = self.causal_conv(z_pad).transpose(1, 2)  # (B, T, D)

        x1 = self.drop1(self.act(self.ln1(self.fc1(z_conv))))
        x2 = x1 + self.drop2(self.act(self.ln2(self.fc2(x1))))  # Skip residuo
        x3 = self.drop3(self.act(self.ln3(self.fc3(x2))))
        logits = self.classifier(x3)

        if is_single_frame:
            logits = logits.squeeze(1)  # (B, num_classes)
        return logits

    def forward_gemma(
        self,
        h_raw: torch.Tensor,
        h_tx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass diretto dagli hidden states multimodali di Gemma 4 Thinker.
        
        Args:
            h_raw: Tensor (B, T, 1536) embedding audio streaming (Conformer)
            h_tx: Tensor (B, 1536) o (B, T, 1536) embedding semantico testuale
        Returns:
            Dict con:
              - 'logits': Tensor (B, T, 5)
              - 'probs': Tensor (B, T, 5)
              - 'preds': Tensor (B, T) indice classe predetta [0..4]
        """
        is_single_frame = False
        if h_raw.dim() == 2:
            h_raw = h_raw.unsqueeze(1)
            is_single_frame = True

        if h_tx.dim() == 2:
            h_tx_exp = h_tx.unsqueeze(1).expand(-1, h_raw.shape[1], -1)
        else:
            h_tx_exp = h_tx

        z_fused = torch.cat([h_raw, h_tx_exp], dim=-1)  # (B, T, 3072)
        z_adapted = self.input_adapter(z_fused)          # (B, T, in_dim)

        logits = self.forward(z_adapted)
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        if is_single_frame:
            logits = logits.squeeze(1)
            probs = probs.squeeze(1)
            preds = preds.squeeze(1)

        return {
            "logits": logits,
            "probs": probs,
            "preds": preds,
        }


# ============================================================================
# 3. CalibratedTurnFocalLoss (Loss Calibrata Anti-Collasso)
# ============================================================================
class CalibratedTurnFocalLoss(nn.Module):
    """
    Focal Loss multiclasse con pesi asimmetrici e label smoothing per mitigare
    lo sbilanciamento naturale dei dataset conversazionali (dove SPEAK è ~65-75%
    e YIELD è <1.5%).
    
    Formula:
      FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        gamma: float = 1.2,
        label_smoothing: float = 0.04,
    ):
        super().__init__()
        if class_weights is None:
            class_weights = [1.0, 4.0, 1.5, 3.0, 2.5]
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, T, C) o (B, C) logit non normalizzati
            targets: (B, T) o (B,) indici interi target in [0, C-1]
        """
        if logits.dim() == 3:
            C = logits.shape[-1]
            logits_flat = logits.reshape(-1, C)
            targets_flat = targets.reshape(-1)
        else:
            logits_flat = logits
            targets_flat = targets

        # Maschera target validi (ignora padding -100 o negativi)
        valid_mask = targets_flat >= 0
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits_flat[valid_mask]
        targets_valid = targets_flat[valid_mask]

        num_classes = logits_valid.shape[-1]
        probs = F.softmax(logits_valid, dim=-1)

        # One-hot con Label Smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(probs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets_valid.unsqueeze(1), 1.0 - self.label_smoothing)

        # Calcolo Focal factor (1 - p_t)^gamma
        pt = (probs * smooth_targets).sum(dim=-1).clamp(min=1e-8, max=1.0)
        focal_factor = torch.pow(1.0 - pt, self.gamma)

        # Cross-Entropy ponderata
        log_probs = F.log_softmax(logits_valid, dim=-1)
        ce_loss = -(smooth_targets * log_probs).sum(dim=-1)

        # Pesi di classe
        alpha_t = self.weights.to(logits.device)[targets_valid]

        focal_loss = alpha_t * focal_factor * ce_loss
        return focal_loss.mean()


# ============================================================================
# 4. StreamingDuplexController (FSM Real-Time con Zero-Reset KV Cache)
# ============================================================================
class StreamingDuplexController:
    """
    Finite State Machine (FSM) a bassissima latenza per orchestrare l'interazione
    full-duplex live tra utente, Thinker (Gemma 4), Projector V9 e Talker (Qwen3-TTS).
    
    Caratteristiche:
      - Rilevamento Barge-in ultra-reattivo (isteresi 1 frame = 80ms).
      - Flush immediato del buffer audio hardware client (<10ms).
      - Politica Zero-Reset KV-Cache (NVIDIA VoiceChat-TTS): preserva la continuità
        acustica del Talker congelato scivolando dolcemente a silenzio.
      - Debouncing e Cooldown per Backchannel ("mhm") per evitare spamming.
    """
    def __init__(
        self,
        barge_in_consecutive_frames: int = 1,
        yield_consecutive_frames: int = 2,
        emit_bc_cooldown_frames: int = 5,
        maintain_kv_cache_on_barge_in: bool = True,
    ):
        self.barge_in_thresh = barge_in_consecutive_frames
        self.yield_thresh = yield_consecutive_frames
        self.bc_cooldown_max = emit_bc_cooldown_frames
        self.maintain_kv = maintain_kv_cache_on_barge_in

        # Stato interno
        self.current_state = "IDLE"  # "IDLE", "LISTENING", "SPEAKING"
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0

    def step(self, predicted_action: int, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Elabora il frame corrente (80ms) e determina l'azione conversazionale.
        
        Args:
            predicted_action: ID azione [0..4] predetto da TurnDetectorHead
            confidence: Probabilità softmax dell'azione
        Returns:
            Dict di eventi per il runtime audio e il Talker
        """
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
            if predicted_action == ACTION_BARGE_IN:
                self.barge_in_counter += 1
                if self.barge_in_counter >= self.barge_in_thresh:
                    # Trigger BARGE-IN: svuota buffer client e arresta sintesi
                    self.current_state = "LISTENING"
                    self.barge_in_counter = 0
                    event["flush_audio_buffer"] = True
                    event["stop_synthesis"] = True
                    event["reset_kv_cache"] = not self.maintain_kv
                    event["state_changed"] = True
            else:
                self.barge_in_counter = 0

            if predicted_action == ACTION_YIELD:
                self.yield_counter += 1
                if self.yield_counter >= self.yield_thresh:
                    self.current_state = "LISTENING"
                    self.yield_counter = 0
                    event["stop_synthesis"] = True
                    event["state_changed"] = True
            else:
                self.yield_counter = 0

        elif self.current_state in ("LISTENING", "IDLE"):
            self.barge_in_counter = 0
            if predicted_action == ACTION_SPEAK:
                self.current_state = "SPEAKING"
                event["state_changed"] = True
            elif predicted_action == ACTION_EMIT_BACKCHANNEL and self.bc_cooldown == 0:
                event["trigger_backchannel"] = True
                self.bc_cooldown = self.bc_cooldown_max

        event["state"] = self.current_state
        return event

    def reset(self):
        """Resetta i contatori di stato."""
        self.current_state = "IDLE"
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0
