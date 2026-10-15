from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from duplex_kernel.acoustic_phase_features import ACOUSTIC_PHASE_FEATURE_CONTRACT
from training.duplex_qlora.control import ControlHead, load_control_head
from training.duplex_qlora.features import listener_feature_contract


class AcousticPhaseControlHeadTests(unittest.TestCase):
    def test_legacy_contract_and_head_remain_1536(self) -> None:
        head = ControlHead(architecture="mlp_256_gelu_dropout_0.1")
        self.assertEqual(head.input_dim, 1536)
        self.assertEqual(listener_feature_contract(1536)["input_dim"], 1536)
        self.assertEqual(head(torch.zeros(2, 1536)).shape, (2, 6))

    def test_1540_head_roundtrips_with_explicit_acoustic_phase_contract(self) -> None:
        head = ControlHead(architecture="mlp_256_gelu_dropout_0.1", input_dim=1540)
        self.assertEqual(head(torch.zeros(2, 1540)).shape, (2, 6))
        self.assertEqual(
            listener_feature_contract(1540)["acoustic_phase"],
            ACOUSTIC_PHASE_FEATURE_CONTRACT,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head.pt"
            head.save(path)
            loaded = load_control_head(path)
        self.assertEqual(loaded.input_dim, 1540)
        self.assertEqual(loaded(torch.zeros(1, 1540)).shape, (1, 6))

    def test_1540_head_rejects_legacy_width(self) -> None:
        head = ControlHead(architecture="mlp_256_gelu_dropout_0.1", input_dim=1540)
        with self.assertRaises(ValueError):
            head(torch.zeros(1, 1536))

    def test_native_phase_fusion_keeps_phase_branch_separate_and_trainable(self) -> None:
        head = ControlHead(architecture="native_phase_fusion_mlp", input_dim=1540)
        self.assertEqual(tuple(head.native_norm.normalized_shape), (1536,))
        self.assertEqual(head.native_projection.in_features, 1536)
        self.assertEqual(head.phase_projection.in_features, 4)
        self.assertEqual(head.classifier.in_features, 288)

        observed_phase = []
        hook = head.phase_projection.register_forward_pre_hook(
            lambda _module, inputs: observed_phase.append(inputs[0].detach().clone())
        )
        torch.manual_seed(7)
        features = torch.zeros(2, 1540)
        features[:, :1536] = torch.randn(2, 1536)
        features[:, 1536:] = torch.tensor([-6.0, -3.0, -4.0, 0.5])
        logits = head(features)
        hook.remove()
        self.assertEqual(logits.shape, (2, 6))
        torch.testing.assert_close(
            observed_phase[0],
            torch.tensor([[-1.0, -0.5, -4.0 / 6.0, 0.5]]).expand(2, -1),
        )

        logits.square().sum().backward()
        self.assertIsNotNone(head.native_projection.weight.grad)
        self.assertGreater(float(head.native_projection.weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(head.phase_projection.weight.grad)
        self.assertGreater(float(head.phase_projection.weight.grad.abs().sum()), 0.0)

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "head_f.pt"
            head.save(path)
            loaded = load_control_head(path)
        self.assertEqual(loaded.architecture, "native_phase_fusion_mlp")
        self.assertEqual(loaded(torch.zeros(1, 1540)).shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
