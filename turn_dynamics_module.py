#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# TURN DYNAMICS MODULE — GEMMA 4 FULL-DUPLEX OMNI PIPELINE
# Architettura:
#   1. TurnActionHead: Classificatore neurale causale Pre-LN a 5 azioni conversazionali
#      (SPEAK, LISTEN, BARGE_IN, YIELD, EMIT_BACKCHANNEL)
#   2. LearnableStateEmbeddings: Vettori continui latenti nello spazio del Talker (2048D)
#      (e_silence, e_yield, e_bc)
#   3. GatedStateModulator: Fusione continua differenziabile tra h_speech (V7 frozen)
#      e i vettori di stato (ASPIRin & VoiceChat-TTS pattern)
#   4. StreamingDuplexController: FSM a bassa latenza (<80ms) per inferenza streaming
#      (Barge-in buffer flush, Turn Yielding, Backchannel trigger)
# ============================================================================
import os
import sys
import math
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. Definizione Azioni Conversazionali (5 Stati Canonici)
# ============================================================================
ACTION_SPEAK = 0            # 0: L'agente parla normalmente (gate aperto su V7)
ACTION_YIELD = 1            # 1: L'agente cede il turno (fine risposta, transizione a silenzio)
ACTION_BARGE_IN = 2         # 2: L'utente interrompe l'agente (trigger flush buffer immediato)
ACTION_USER_BACKCHANNEL = 3 # 3: L'utente emette un feedback passivo (non interrompe)
ACTION_EMIT_BACKCHANNEL = 4 # 4: L'agente emette un micro-feedback ("mhm", "sì")

ACTION_NAMES = ["SPEAK", "YIELD", "BARGE_IN", "USER_BACKCHANNEL", "EMIT_BACKCHANNEL"]
NUM_ACTIONS = len(ACTION_NAMES)

# Costanti Token Speciali Qwen3-TTS (Verificate su snapshot Qwen3-TTS 12Hz & Report Tecnico 2601.15621)
QWEN3_TTS_PAD_ID = 151671       # CFG.tts_pad_token_id (vocabolario testuale)
QWEN3_TTS_BOS_ID = 151672       # CFG.tts_bos_token_id
QWEN3_TTS_EOS_ID = 151673       # CFG.tts_eos_token_id
QWEN3_CODEC_PAD_ID = 2148       # TCFG.codec_pad_id (embedding acustici Talker)
QWEN3_CODEC_BOS_ID = 2149       # TCFG.codec_bos_id
QWEN3_CODEC_EOS_ID = 2150       # TCFG.codec_eos_token_id
QWEN3_CHATML_IM_START = 151644  # <|im_start|>
QWEN3_CHATML_IM_END = 151645    # <|im_end|>


# ============================================================================
# 2. TurnActionHead (Classificatore Neurale di Turno)
# ============================================================================
class TurnActionHead(nn.Module):
    """
    Classificatore causale Pre-LN Deep Residual per predire a ogni frame (12.5 Hz)
    l'azione conversazionale ottima a partire dagli hidden states di Gemma 4 congelata.
    
    Input:
      z_t = [h_raw (1536) || h_tx (1536)] in R^3072
      oppure con feature acustiche estese:
      z_t = [h_raw (1536) || h_tx (1536) || e_user (128) || delta_e_user (128)] in R^3328
    """
    def __init__(
        self,
        in_dim: int = 3072,
        hidden_dim: int = 1024,
        mid_dim: int = 256,
        num_classes: int = NUM_ACTIONS,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes

        # Causal temporal smoothing (k=3, left-padded)
        self.causal_conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=3,
            padding=0,
            bias=False
        )
        # Inizializzazione identità (ReZero style)
        with torch.no_grad():
            self.causal_conv.weight.zero_()
            self.causal_conv.weight[:, :, -1] = torch.eye(in_dim)

        # Deep Pre-LN Residual MLP
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.drop2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim, mid_dim)
        self.ln3 = nn.LayerNorm(mid_dim)
        self.drop3 = nn.Dropout(dropout)

        self.classifier = nn.Linear(mid_dim, num_classes)
        self.act = nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Tensor (B, T, in_dim) o (B, in_dim)
        Returns:
            logits: Tensor (B, T, num_classes) o (B, num_classes)
        """
        is_single_frame = False
        if z.dim() == 2:
            z = z.unsqueeze(1)  # (B, 1, in_dim)
            is_single_frame = True

        B, T, D = z.shape
        # Causal Conv1D con left-padding esplicito (2, 0)
        z_t = z.transpose(1, 2)  # (B, D, T)
        z_pad = F.pad(z_t, (2, 0))
        z_conv = self.causal_conv(z_pad).transpose(1, 2)  # (B, T, D)

        x1 = self.drop1(self.act(self.ln1(self.fc1(z_conv))))
        x2 = x1 + self.drop2(self.act(self.ln2(self.fc2(x1))))  # Residual skip
        x3 = self.drop3(self.act(self.ln3(self.fc3(x2))))
        logits = self.classifier(x3)

        if is_single_frame:
            logits = logits.squeeze(1)  # (B, num_classes)
        return logits


# ============================================================================
# 3. LearnableStateEmbeddings (Vettori di Stato Speciali nel Talker)
# ============================================================================
class LearnableStateEmbeddings(nn.Module):
    """
    Vettori di stato continui addestrabili nello spazio latente del Talker (2048D).
    Rappresentano gli stati di silenzio, cessione di turno, backchannel e transizione
    di interruzione fluida (VoiceChat-TTS NVIDIA pattern).
    """
    def __init__(self, out_dim: int = 2048):
        super().__init__()
        self.out_dim = out_dim

        # Vettore silenzio / ascolto quiescente
        self.e_silence = nn.Parameter(torch.zeros(out_dim))
        # Vettore cessione di turno (forza EOS / abbattimento energia)
        self.e_yield = nn.Parameter(torch.zeros(out_dim))
        # Vettore ancoraggio backchannel acustico ("mhm", "sì")
        self.e_bc = nn.Parameter(torch.zeros(out_dim))
        # Vettore di transizione interruzione fluida (<INT> VoiceChat-TTS)
        self.e_interrupt = nn.Parameter(torch.zeros(out_dim))

        # Inizializzazione calibrata (piccola varianza attorno a zero)
        nn.init.normal_(self.e_silence, mean=0.0, std=0.02)
        nn.init.normal_(self.e_yield, mean=0.0, std=0.02)
        nn.init.normal_(self.e_bc, mean=0.0, std=0.02)
        nn.init.normal_(self.e_interrupt, mean=0.0, std=0.02)

    def forward(self) -> Dict[str, torch.Tensor]:
        return {
            "silence": self.e_silence,
            "yield": self.e_yield,
            "bc": self.e_bc,
            "interrupt": self.e_interrupt,
        }


# ============================================================================
# 4. GatedStateModulator (ASPIRin & VoiceChat-TTS Pattern)
# ============================================================================
class GatedStateModulator(nn.Module):
    """
    Esegue la modulazione continua e differenziabile tra il vettore di sintesi
    vocale pura h_speech (emesso dal Projector V7 congelato) e i vettori di stato.
    """
    def __init__(self, out_dim: int = 2048, gamma: float = 0.25):
        super().__init__()
        self.out_norm = nn.LayerNorm(out_dim)
        self.gamma = gamma

    def forward(
        self,
        h_speech: torch.Tensor,
        action_logits: torch.Tensor,
        state_embeds: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_speech: (B, T, 2048) da Projector V7 congelato
            action_logits: (B, T, 5) da TurnActionHead
            state_embeds: dizionario con e_silence, e_yield, e_bc
        Returns:
            h_cond: (B, T, 2048) vettore modulato pronto per il loop Qwen3-TTS
            gate: (B, T, 1) valore del gate di parlato [0.0 = silenzio, 1.0 = voce]
        """
        probs = F.softmax(action_logits, dim=-1)  # (B, T, 5)
        p_yield = probs[..., ACTION_YIELD:ACTION_YIELD+1]
        p_bc = probs[..., ACTION_EMIT_BACKCHANNEL:ACTION_EMIT_BACKCHANNEL+1]
        p_int = probs[..., ACTION_BARGE_IN:ACTION_BARGE_IN+1]

        # Gate di parlato continuo: sigma(logit_speak - max(logit_yield, logit_barge_in))
        silence_competitors = torch.maximum(
            action_logits[..., ACTION_YIELD:ACTION_YIELD+1],
            action_logits[..., ACTION_BARGE_IN:ACTION_BARGE_IN+1]
        )
        logit_diff = action_logits[..., ACTION_SPEAK:ACTION_SPEAK+1] - silence_competitors
        gate = torch.sigmoid(logit_diff)  # (B, T, 1)

        e_silence = state_embeds["silence"].to(h_speech.dtype)
        e_yield = state_embeds["yield"].to(h_speech.dtype)
        e_bc = state_embeds["bc"].to(h_speech.dtype)
        e_int = state_embeds.get("interrupt", e_silence).to(h_speech.dtype)

        # Fusione VoiceChat-TTS & ASPIRin:
        # h_fused = (gate * h_speech) + ((1 - gate) * e_silence) + (p_yield * e_yield) + (p_bc * e_bc) + (p_int * e_int)
        h_fused = (
            (gate * h_speech)
            + ((1.0 - gate) * e_silence)
            + (p_yield * e_yield)
            + (p_bc * e_bc)
            + (p_int * e_int)
        )
        h_cond = (self.gamma) * self.out_norm(h_fused)

        return h_cond, gate


# ============================================================================
# 5. Modulo Unificato: TurnDynamicsModule
# ============================================================================
class TurnDynamicsModule(nn.Module):
    """
    Modulo completo di Dinamiche Conversazionali Full-Duplex.
    Mantiene il Projector V7 congelato e addestra solo la logica di turno.
    """
    def __init__(
        self,
        in_dim: int = 3072,
        out_dim: int = 2048,
        hidden_dim: int = 1024,
        mid_dim: int = 256,
        num_classes: int = NUM_ACTIONS,
        gamma: float = 0.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_head = TurnActionHead(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            mid_dim=mid_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.state_embeds = LearnableStateEmbeddings(out_dim=out_dim)
        self.modulator = GatedStateModulator(out_dim=out_dim, gamma=gamma)

    def forward(
        self,
        h_raw: torch.Tensor,
        h_tx: torch.Tensor,
        frozen_v7_projector: Optional[nn.Module] = None,
        extra_acoustic_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            h_raw: (B, T, 1536) audio hidden da Gemma 4
            h_tx: (B, 1536) o (B, T, 1536) text hidden da Gemma 4
            frozen_v7_projector: Istanza del TextAnchoredProjectionV7 congelato
            extra_acoustic_feats: Opzionale (B, T, 256) per e_user e delta_e_user
        Returns:
            Dizionario con:
              - 'h_cond': (B, T, 2048)
              - 'action_logits': (B, T, 5)
              - 'gate': (B, T, 1)
              - 'action_preds': (B, T) argmax delle azioni
        """
        if h_tx.dim() == 2:
            h_tx_expanded = h_tx.unsqueeze(1).expand(-1, h_raw.shape[1], -1)
        else:
            h_tx_expanded = h_tx

        if extra_acoustic_feats is not None:
            z = torch.cat([h_raw, h_tx_expanded, extra_acoustic_feats], dim=-1)
        else:
            z = torch.cat([h_raw, h_tx_expanded], dim=-1)

        action_logits = self.action_head(z)
        embeds = self.state_embeds()

        if frozen_v7_projector is not None:
            with torch.no_grad():
                h_speech = frozen_v7_projector(h_raw, h_tx)
        else:
            # Fallback a zero se il projector non è passato
            h_speech = torch.zeros(h_raw.shape[0], h_raw.shape[1], self.state_embeds.out_dim,
                                   device=h_raw.device, dtype=h_raw.dtype)

        h_cond, gate = self.modulator(h_speech, action_logits, embeds)
        action_preds = action_logits.argmax(dim=-1)

        return {
            "h_cond": h_cond,
            "action_logits": action_logits,
            "gate": gate,
            "action_preds": action_preds,
        }


# ============================================================================
# 6. Loss Function: Calibrated Focal Loss per Dinamiche di Turno
# ============================================================================
class CalibratedTurnFocalLoss(nn.Module):
    """
    Focal Loss multi-classe pesata con Label Smoothing per prevenire il collasso
    su SPEAK e massimizzare la reattività di YIELD e BARGE_IN (ADR-020).
    """
    def __init__(
        self,
        class_weights: Optional[List[float]] = None,
        gamma: float = 1.2,
        label_smoothing: float = 0.04,
    ):
        super().__init__()
        if class_weights is None:
            # SPEAK: 1.0, LISTEN: 1.2, BARGE_IN: 4.0, YIELD: 3.0, EMIT_BC: 2.5
            class_weights = [1.0, 1.2, 4.0, 3.0, 2.5]
        self.register_buffer("weights", torch.tensor(class_weights, dtype=torch.float32))
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B * T, num_classes) o (B, T, num_classes)
            targets: (B * T,) o (B, T)
        """
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)

        # Filtra target non validi / ignorati (-100)
        valid_mask = (targets >= 0) & (targets < logits.shape[-1])
        if not valid_mask.any():
            return torch.zeros((), device=logits.device, dtype=logits.dtype)

        logits = logits[valid_mask]
        targets = targets[valid_mask]

        num_classes = logits.shape[-1]
        probs = F.softmax(logits, dim=-1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Focal term: (1 - p_t)^gamma
        focal_factor = (1.0 - p_t) ** self.gamma

        # Pesi di classe
        w_t = self.weights[targets]

        # Cross Entropy con Label Smoothing
        log_probs = F.log_softmax(logits, dim=-1)
        smooth_loss = -log_probs.mean(dim=-1)
        nll_loss = F.nll_loss(log_probs, targets, reduction='none')
        ce_loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        loss = focal_factor * w_t * ce_loss
        return loss.mean()


# ============================================================================
# 7. StreamingDuplexController (Runtime Engine per Inferenza a Bassa Latenza)
# ============================================================================
class StreamingDuplexController:
    """
    Controller a stati finiti (FSM) per gestire lo streaming real-time a 12.5 Hz.
    Esegue il debouncing delle decisioni e azzera la latenza su Barge-in e Yield.
    Implementa la gestione 'Zero-Reset KV-Cache' di NVIDIA VoiceChat-TTS.
    """
    def __init__(
        self,
        barge_in_consecutive_frames: int = 1,      # 80ms di persistenza per trigger
        yield_consecutive_frames: int = 2,         # 160ms di persistenza
        emit_bc_cooldown_frames: int = 30,         # ~2.4s minimo tra un backchannel e l'altro
        maintain_kv_cache_on_barge_in: bool = True, # VoiceChat-TTS: preserva contesto acustico
    ):
        self.barge_in_thresh = barge_in_consecutive_frames
        self.yield_thresh = yield_consecutive_frames
        self.bc_cooldown_max = emit_bc_cooldown_frames
        self.maintain_kv = maintain_kv_cache_on_barge_in

        self.current_state = "IDLE"  # IDLE, SPEAKING, LISTENING
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0

    def step(self, predicted_action: int) -> Dict[str, Any]:
        """
        Esegue un passo della FSM al frame corrente t.
        Returns:
            Dizionario con le istruzioni per il runtime:
              - 'flush_audio_buffer': bool (True se l'agente deve zittire l'altoparlante client all'istante)
              - 'reset_kv_cache': bool (False se usa la transizione continua VoiceChat-TTS)
              - 'inject_interrupt_state': bool (True se iniettare e_interrupt / e_silence)
              - 'stop_synthesis': bool
              - 'trigger_backchannel': bool
              - 'state': str
        """
        event = {
            "flush_audio_buffer": False,
            "reset_kv_cache": False,
            "inject_interrupt_state": False,
            "stop_synthesis": False,
            "trigger_backchannel": False,
            "state": self.current_state,
        }

        if self.bc_cooldown > 0:
            self.bc_cooldown -= 1

        if self.current_state == "SPEAKING":
            if predicted_action == ACTION_BARGE_IN:
                self.barge_in_counter += 1
                if self.barge_in_counter >= self.barge_in_thresh:
                    # Trigger BARGE-IN immediato: svuota buffer client e transiziona a silenzio
                    self.current_state = "LISTENING"
                    self.barge_in_counter = 0
                    event["flush_audio_buffer"] = True
                    event["stop_synthesis"] = True
                    event["inject_interrupt_state"] = True
                    event["reset_kv_cache"] = not self.maintain_kv
            else:
                self.barge_in_counter = 0

            if predicted_action == ACTION_YIELD:
                self.yield_counter += 1
                if self.yield_counter >= self.yield_thresh:
                    self.current_state = "LISTENING"
                    self.yield_counter = 0
                    event["stop_synthesis"] = True
            else:
                self.yield_counter = 0

        elif self.current_state in ("LISTENING", "IDLE"):
            self.barge_in_counter = 0
            if predicted_action == ACTION_SPEAK:
                self.current_state = "SPEAKING"
            elif predicted_action == ACTION_EMIT_BACKCHANNEL and self.bc_cooldown == 0:
                event["trigger_backchannel"] = True
                self.bc_cooldown = self.bc_cooldown_max

        event["state"] = self.current_state
        return event

    def reset(self):
        self.current_state = "IDLE"
        self.barge_in_counter = 0
        self.yield_counter = 0
        self.bc_cooldown = 0


# ============================================================================
# 8. OneTokenLookaheadTextBuffer (NVIDIA VoiceChat-TTS 2-Subword Window)
# ============================================================================
class OneTokenLookaheadTextBuffer:
    """
    Buffer di lookahead a 1 token (finestra minima di 2 subword) per la sintesi vocale.
    Previene esitazioni fonetiche e intonazione scorretta all'avvio della frase,
    introducendo solo ~30-50ms di ritardo totalmente impercettibile.
    """
    def __init__(self, min_tokens_before_emit: int = 2):
        self.min_tokens = min_tokens_before_emit
        self.token_queue: List[Any] = []
        self.is_finalized = False

    def push(self, token_or_repr: Any):
        """Aggiunge un token al buffer di lookahead."""
        self.token_queue.append(token_or_repr)

    def finalize(self):
        """Segnala la fine della generazione testuale (EOS)."""
        self.is_finalized = True

    def ready_to_emit(self) -> bool:
        """True se il buffer contiene abbastanza contesto o è arrivato l'EOS."""
        return len(self.token_queue) >= self.min_tokens or self.is_finalized

    def pop_current(self) -> Optional[Any]:
        """Emette il token pronto per la proiezione acustica."""
        if self.ready_to_emit() and len(self.token_queue) > 0:
            return self.token_queue.pop(0)
        return None

    def clear(self):
        """Svuota il buffer (es. in caso di barge-in)."""
        self.token_queue.clear()
        self.is_finalized = False
