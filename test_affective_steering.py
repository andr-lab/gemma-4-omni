"""
test_affective_steering.py — Unit & Integration Test Suite for Affective Steering Vectors.

Tests ADR-015:
1. AffectiveSteeringBank: vector registration, checkpoint save/load, unit normalization, mixed vector composition.
2. apply_norm_preserving_steering: strict Euclidean norm preservation across dimensions and alpha scaling.
3. AffectiveSteeringHook: forward hook activation modulation on Transformer mock layers.
4. GemmaToneExtractor: tone classification probability and continuous latent tone projection.
5. SteeredTalkerController: coordination of ingress steering and deep layer hooks with Mock TextAnchoredProjectionV7.

Usage:
  python test_affective_steering.py
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from affective_steering_module import (
    AffectiveSteeringBank,
    AffectiveSteeringHook,
    GemmaToneExtractor,
    SteeredTalkerController,
    apply_norm_preserving_steering,
)
from tools.extract_affective_steering_vectors import build_synthetic_affective_bank


class MockAttentionLayer(nn.Module):
    """Mock of Qwen3-TTS Talker Transformer Self-Attention Layer with Residual & LayerNorm."""
    def __init__(self, hidden_dim=2048):
        super().__init__()
        self.self_attn = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        nn.init.eye_(self.self_attn.weight)
        nn.init.zeros_(self.self_attn.bias)

    def forward(self, x):
        return self.norm(x + self.self_attn(x))


class MockTalkerModel(nn.Module):
    """Mock of Qwen3-TTS Talker with 28 Transformer layers."""
    def __init__(self, num_layers=28, hidden_dim=2048):
        super().__init__()
        self.layers = nn.ModuleList([MockAttentionLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestAffectiveSteeringModule(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 2048
        self.bank = build_synthetic_affective_bank(dim=self.dim, target_layers=[12, 14, 16])

    def test_bank_registration_and_normalization(self):
        """Verify registered vectors have exactly unit L2 norm."""
        for tone in AffectiveSteeringBank.SUPPORTED_TONES:
            if tone == "neutral":
                continue
            v = self.bank.get_vector(tone)
            self.assertIsNotNone(v, f"Vector for tone '{tone}' should exist")
            norm = torch.norm(v, p=2).item()
            self.assertAlmostEqual(norm, 1.0, places=5, msg=f"Vector '{tone}' must have unit norm")

    def test_bank_checkpoint_save_and_load(self):
        """Verify bank persistence to disk and reload consistency."""
        ckpt_path = "checkpoints/test_steering_bank.pt"
        self.bank.save_checkpoint(ckpt_path)
        self.assertTrue(os.path.exists(ckpt_path))

        new_bank = AffectiveSteeringBank(dim=self.dim)
        new_bank.load_checkpoint(ckpt_path)

        for tone in ["empathetic", "calm", "enthusiastic"]:
            v_orig = self.bank.get_vector(tone)
            v_new = new_bank.get_vector(tone)
            self.assertIsNotNone(v_new)
            cos_sim = F.cosine_similarity(v_orig.unsqueeze(0), v_new.unsqueeze(0)).item()
            self.assertAlmostEqual(cos_sim, 1.0, places=5)

        # Cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)

    def test_mixed_vector_composition(self):
        """Verify linear composition of multiple emotional directions."""
        weights = {"calm": 0.7, "empathetic": 0.3}
        v_mix = self.bank.compose_mixed_vector(weights)
        self.assertEqual(v_mix.shape, (self.dim,))
        mix_norm = torch.norm(v_mix, p=2).item()
        self.assertAlmostEqual(mix_norm, 1.0, places=5, msg="Mixed vector must be unit normalized")

        # Test empty weights returns zero
        v_empty = self.bank.compose_mixed_vector({})
        self.assertEqual(torch.norm(v_empty).item(), 0.0)

    def test_strict_euclidean_norm_preservation(self):
        """
        Verify that apply_norm_preserving_steering maintains the EXACT original activation norm
        across various shapes and intensity values alpha in [0.0, 3.5].
        """
        v_steer = self.bank.get_vector("enthusiastic")

        # 3D Tensor: (Batch=2, Time=64, Dim=2048)
        h_3d = torch.randn(2, 64, self.dim, dtype=torch.float32) * 5.0
        orig_norms = torch.norm(h_3d, p=2, dim=-1)

        for alpha in [0.0, 0.5, 1.0, 2.0, 3.5]:
            h_steered = apply_norm_preserving_steering(h_3d, v_steer, alpha=alpha)
            new_norms = torch.norm(h_steered, p=2, dim=-1)

            diff = torch.abs(orig_norms - new_norms).max().item()
            self.assertLess(diff, 1e-4, f"Norm drifted by {diff} at alpha={alpha}")

            if alpha > 0.0:
                # Assert direction has shifted (cosine similarity < 1.0)
                cos_sim = F.cosine_similarity(h_3d.view(-1, self.dim), h_steered.view(-1, self.dim), dim=-1).mean().item()
                self.assertLess(cos_sim, 0.9999, "Steering should alter activation direction")

    def test_affective_steering_hook_execution(self):
        """Verify forward hook dynamically steers intermediate transformer layers."""
        mock_talker = MockTalkerModel(num_layers=28, hidden_dim=self.dim)
        controller = SteeredTalkerController(
            bank=self.bank,
            talker_model=mock_talker,
            target_layers=[12, 14, 16],
        )
        self.assertEqual(len(controller.hooks), 3)

        x_in = torch.randn(1, 16, self.dim)
        
        # 1. Baseline Neutral (alpha=0 or tone=neutral -> no shift)
        controller.set_affective_tone("neutral", alpha=0.0)
        out_neutral = mock_talker(x_in.clone())

        # 2. Steered with Enthusiasm (alpha=2.0)
        controller.set_affective_tone("enthusiastic", alpha=2.0)
        out_steered = mock_talker(x_in.clone())

        # Check norms are identical but content is modified
        self.assertAlmostEqual(
            torch.norm(out_neutral).item(),
            torch.norm(out_steered).item(),
            places=3,
            msg="Output norm must be strictly preserved"
        )
        cos_diff = F.cosine_similarity(out_neutral.view(-1, self.dim), out_steered.view(-1, self.dim)).mean().item()
        self.assertLess(cos_diff, 0.995, "Steered output must differ from neutral baseline")

        # Cleanup
        controller.cleanup()
        self.assertEqual(len(controller.hooks), 0)

    def test_gemma_tone_extractor(self):
        """Verify tone classifier and continuous latent projection."""
        extractor = GemmaToneExtractor(in_dim=1536, out_dim=2048, num_tones=8)
        h_gemma = torch.randn(4, 1536)

        # Softmax probability distribution
        weights = extractor.predict_tone_weights(h_gemma)
        self.assertEqual(weights.shape, (4, 8))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-5))

        # Continuous projection
        v_cont = extractor.extract_continuous_steering(h_gemma)
        self.assertEqual(v_cont.shape, (4, 2048))
        norms = torch.norm(v_cont, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones(4), atol=1e-5))

    def test_ingress_condition_steering(self):
        """Verify ingress modulation of h_cond before entering Talker loop."""
        controller = SteeredTalkerController(bank=self.bank)
        h_cond = torch.randn(1, 64, 2048) * 3.0
        orig_norm = torch.norm(h_cond, p=2, dim=-1)

        # Ingress steering with Calma
        controller.set_affective_tone("calm", alpha=1.5)
        h_cond_steered = controller.steer_ingress_condition(h_cond)

        new_norm = torch.norm(h_cond_steered, p=2, dim=-1)
        self.assertTrue(torch.allclose(orig_norm, new_norm, atol=1e-4))

        # Mixed ingress steering
        controller.set_mixed_affective_tone({"calm": 0.6, "empathetic": 0.4}, alpha=1.5)
        h_cond_mixed = controller.steer_ingress_condition(h_cond)
        mixed_norm = torch.norm(h_cond_mixed, p=2, dim=-1)
        self.assertTrue(torch.allclose(orig_norm, mixed_norm, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
