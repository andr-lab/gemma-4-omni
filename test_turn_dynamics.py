#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit Test & Integration Verification for TurnDynamicsModule, VoiceChat-TTS Lookahead,
and TextAnchoredProjectionV7.
Verifica:
  1. Istanziazione e forward pass di TurnActionHead, StateEmbeddings (incluso e_interrupt) e GatedStateModulator
  2. Integrazione con TextAnchoredProjectionV7 (congelato) e backprop Focal Loss + Coupling
  3. Gating & State Embeddings Modulation (transizione SPEAK -> LISTEN con e_silence ed e_interrupt)
  4. Funzionamento della FSM di StreamingDuplexController (Zero-Reset KV Cache, Barge-in, Yield, Backchannel)
  5. Funzionamento del OneTokenLookaheadTextBuffer (finestra minima 2 subword)
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from turn_dynamics_module import (
    TurnDynamicsModule,
    TurnActionHead,
    LearnableStateEmbeddings,
    GatedStateModulator,
    CalibratedTurnFocalLoss,
    StreamingDuplexController,
    OneTokenLookaheadTextBuffer,
    ACTION_SPEAK,
    ACTION_YIELD,
    ACTION_BARGE_IN,
    ACTION_USER_BACKCHANNEL,
    ACTION_EMIT_BACKCHANNEL,
)

# Mock del TextAnchoredProjectionV7
class MockTextAnchoredProjectionV7(nn.Module):
    def __init__(self, in_dim=1536, out_dim=2048, gamma=0.25):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.gamma = gamma

    def forward(self, h_raw, h_tx):
        if h_tx.dim() == 2:
            h_tx = h_tx.unsqueeze(1)
        h = self.proj(h_raw) + self.proj(h_tx)
        return self.gamma * self.norm(h)


def test_turn_dynamics_pipeline():
    print("=== [TEST 1/5] Forward Pass TurnDynamicsModule (con e_interrupt) ===")
    B, T = 2, 64
    dim_in = 1536
    dim_out = 2048

    h_raw = torch.randn(B, T, dim_in)
    h_tx = torch.randn(B, dim_in)

    v7_proj = MockTextAnchoredProjectionV7(in_dim=dim_in, out_dim=dim_out, gamma=0.25)
    for p in v7_proj.parameters():
        p.requires_grad = False  # Congelato

    dynamics_mod = TurnDynamicsModule(
        in_dim=dim_in * 2,
        out_dim=dim_out,
        hidden_dim=512,
        mid_dim=128,
        gamma=0.25,
    )

    out = dynamics_mod(h_raw, h_tx, frozen_v7_projector=v7_proj)

    h_cond = out["h_cond"]
    logits = out["action_logits"]
    gate = out["gate"]
    preds = out["action_preds"]

    assert h_cond.shape == (B, T, dim_out), f"Shape mismatch: {h_cond.shape} vs {(B, T, dim_out)}"
    assert logits.shape == (B, T, 5), f"Logits shape mismatch: {logits.shape}"
    assert gate.shape == (B, T, 1), f"Gate shape mismatch: {gate.shape}"
    assert preds.shape == (B, T), f"Preds shape mismatch: {preds.shape}"
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), "Gate values out of [0, 1] range!"
    print(f"  OK: Forward pass completato con h_cond shape {h_cond.shape}, gate medio {gate.mean().item():.3f}")

    print("\n=== [TEST 2/5] Backward Pass (Focal Loss + Coupling) ===")
    criterion = CalibratedTurnFocalLoss(class_weights=[1.0, 4.0, 1.5, 3.0, 2.5], gamma=1.2)
    targets = torch.randint(0, 5, (B, T))

    loss_action = criterion(logits, targets)
    loss_talker_coupling = h_cond.sum() * 0.01
    loss_total = loss_action + loss_talker_coupling

    assert loss_total.item() > 0.0, "Loss deve essere strettamente positiva"
    loss_total.backward()

    # Verifica che i gradienti si propaghino su TUTTI i parametri di TurnDynamicsModule (incluso e_interrupt)
    for name, param in dynamics_mod.named_parameters():
        assert param.grad is not None, f"Gradiente mancante per {name} in TurnDynamicsModule"
    # Verifica che V7 rimanga 100% isolato senza gradienti
    for name, param in v7_proj.named_parameters():
        assert param.grad is None, f"Il V7 congelato non deve ricevere gradienti ({name})"
    print(f"  OK: Backward pass completato. Loss_action={loss_action.item():.4f}, Loss_total={loss_total.item():.4f}. V7 isolato al 100%.")

    print("\n=== [TEST 3/5] Gating & State Embeddings Modulation (VoiceChat-TTS & ASPIRin) ===")
    # Forza logit di SPEAK altissimo -> gate deve essere ~1.0
    mock_logits_speak = torch.zeros(1, 1, 5)
    mock_logits_speak[0, 0, ACTION_SPEAK] = 10.0
    mock_logits_speak[0, 0, ACTION_YIELD] = -10.0

    mock_h_speech = torch.ones(1, 1, dim_out)
    embeds = dynamics_mod.state_embeds()
    h_fused_speak, gate_speak = dynamics_mod.modulator(mock_h_speech, mock_logits_speak, embeds)
    assert gate_speak.item() > 0.99, f"Gate speak atteso > 0.99, ottenuto {gate_speak.item()}"

    # Forza logit di YIELD altissimo -> gate deve essere ~0.0
    mock_logits_yield = torch.zeros(1, 1, 5)
    mock_logits_yield[0, 0, ACTION_SPEAK] = -10.0
    mock_logits_yield[0, 0, ACTION_YIELD] = 10.0

    h_fused_yield, gate_yield = dynamics_mod.modulator(mock_h_speech, mock_logits_yield, embeds)
    assert gate_yield.item() < 0.01, f"Gate yield atteso < 0.01, ottenuto {gate_yield.item()}"
    print("  OK: GatedStateModulator transisce perfettamente tra SPEAK (gate=1.0) e YIELD/SILENZIO (gate=0.0)")

    print("\n=== [TEST 4/5] StreamingDuplexController FSM & Zero-Reset KV Cache ===")
    controller = StreamingDuplexController(
        barge_in_consecutive_frames=1,
        yield_consecutive_frames=2,
        emit_bc_cooldown_frames=5,
        maintain_kv_cache_on_barge_in=True,
    )

    # 1. Start speaking
    ev1 = controller.step(ACTION_SPEAK)
    assert controller.current_state == "SPEAKING"
    assert not ev1["flush_audio_buffer"]

    # 2. Barge-in test (L'utente interrompe l'agente)
    ev2 = controller.step(ACTION_BARGE_IN)
    assert controller.current_state == "LISTENING"
    assert ev2["flush_audio_buffer"], "Barge-in deve richiedere FLUSH IMMEDIATO dell'audio"
    assert ev2["stop_synthesis"], "Barge-in deve fermare la sintesi"
    assert not ev2["reset_kv_cache"], "VoiceChat-TTS: la KV cache NON deve essere azzerata su barge-in"
    assert ev2["inject_interrupt_state"], "Deve iniettare e_interrupt"

    # 3. Backchannel trigger in ascolto
    ev3 = controller.step(ACTION_EMIT_BACKCHANNEL)
    assert ev3["trigger_backchannel"], "Deve emettere backchannel"

    # 4. Backchannel cooldown verification
    ev4 = controller.step(ACTION_EMIT_BACKCHANNEL)
    assert not ev4["trigger_backchannel"], "Backchannel cooldown deve bloccare spam ravvicinato"

    # 5. Yield test (L'agente cede il turno)
    controller.current_state = "SPEAKING"
    ev5_1 = controller.step(ACTION_YIELD)
    ev5_2 = controller.step(ACTION_YIELD)
    assert controller.current_state == "LISTENING"
    assert ev5_2["stop_synthesis"]

    print("  OK: StreamingDuplexController ha superato tutti i test di transizione di stato, flush e zero-reset KV cache.")

    print("\n=== [TEST 5/5] OneTokenLookaheadTextBuffer (NVIDIA 2-Subword Window) ===")
    buf = OneTokenLookaheadTextBuffer(min_tokens_before_emit=2)

    # Inserimento 1 token -> non deve emettere subito (aspetta lookahead)
    buf.push("token_1")
    assert not buf.ready_to_emit()
    assert buf.pop_current() is None

    # Inserimento 2 token -> raggiunge finestra minima e sblocca il primo token
    buf.push("token_2")
    assert buf.ready_to_emit()
    t1 = buf.pop_current()
    assert t1 == "token_1", f"Atteso token_1, ottenuto {t1}"

    # Finalize (EOS) -> sblocca anche l'ultimo token residuo
    buf.finalize()
    assert buf.ready_to_emit()
    t2 = buf.pop_current()
    assert t2 == "token_2", f"Atteso token_2, ottenuto {t2}"
    assert buf.pop_current() is None

    print("  OK: OneTokenLookaheadTextBuffer gestisce correttamente la finestra di lookahead a 2 subword.")
    print("\n>>> TUTTI I 5 TEST DI VALIDAZIONE SONO PASSATI CON SUCCESSO AL 100%! <<<")


if __name__ == "__main__":
    test_turn_dynamics_pipeline()
