"""
affective_steering_module.py — Affective Prosody & Emotional Steering Vectors for Gemma4-Omni.

Implements ADR-015:
- AffectiveSteeringBank: Manages normalized emotional direction vectors across tones/emotions.
- AffectiveSteeringHook: PyTorch hook for Talker intermediate attention layers with Euclidean norm preservation.
- GemmaToneExtractor: Extracts or classifies emotional tone vectors from Gemma 4 latent activations.
- SteeredTalkerController: High-level wrapper for inference-time steering and mixed-emotion composition.
"""

import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Affective Steering Vector Bank
# ---------------------------------------------------------------------------
class AffectiveSteeringBank(nn.Module):
    """
    Manages emotional steering vectors for ingress conditioning (h_cond, 2048D)
    and layer-wise Talker intermediate hooks.
    """

    SUPPORTED_TONES = [
        "neutral",
        "empathetic",    # Empatico / Accogliente
        "calm",          # Calmo / Rassicurante
        "enthusiastic",  # Entusiasta / Dinamico
        "curious",       # Curioso / Incuriosito
        "assertive",     # Autorevole / Sicuro
        "joyful",        # Allegro / Gioioso
        "sad",           # Malinconico / Triste
    ]

    def __init__(self, dim: int = 2048, default_alpha: float = 1.2):
        super().__init__()
        self.dim = dim
        self.default_alpha = default_alpha
        # Dictionary of registered unit vectors: name -> Tensor (dim,)
        self.vectors = nn.ParameterDict()
        # Optional layer-wise vectors: f"layer_{l}_{tone}" -> Tensor (dim,)
        self.layer_vectors = nn.ParameterDict()

    def register_vector(self, name: str, vector: torch.Tensor, layer_idx: Optional[int] = None):
        """Register a new steering vector, automatically normalizing to unit L2 norm."""
        vector = vector.detach().squeeze()
        if vector.numel() != self.dim:
            raise ValueError(f"Vector dim {vector.numel()} does not match bank dim {self.dim}")
        
        # Unit L2 normalization
        norm = torch.norm(vector, p=2)
        if norm > 1e-8:
            unit_vec = vector / norm
        else:
            unit_vec = vector

        param = nn.Parameter(unit_vec.to(dtype=torch.float32), requires_grad=False)
        if layer_idx is None:
            self.vectors[name] = param
        else:
            self.layer_vectors[f"layer_{layer_idx}_{name}"] = param

    def get_vector(self, name: str, layer_idx: Optional[int] = None) -> Optional[torch.Tensor]:
        """Retrieve unit steering vector by tone name and optional layer index."""
        if layer_idx is not None:
            key = f"layer_{layer_idx}_{name}"
            if key in self.layer_vectors:
                return self.layer_vectors[key]
        if name in self.vectors:
            return self.vectors[name]
        return None

    def compose_mixed_vector(
        self,
        weights: Dict[str, float],
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Linearly compose multiple tone vectors:
        v_mix = sum(w_i * v_i) / ||sum(w_i * v_i)||
        """
        if not weights:
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros(self.dim, device=dev, dtype=dtype)

        # Normalize weights
        total_w = sum(weights.values())
        if total_w <= 1e-8:
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros(self.dim, device=dev, dtype=dtype)

        norm_weights = {k: v / total_w for k, v in weights.items()}
        accum = None

        for tone, w in norm_weights.items():
            vec = self.get_vector(tone, layer_idx=layer_idx)
            if vec is None:
                continue
            if accum is None:
                accum = w * vec.to(dtype=torch.float32)
            else:
                accum = accum + w * vec.to(dtype=torch.float32)

        if accum is None or torch.norm(accum, p=2) <= 1e-8:
            dev = device if device is not None else torch.device("cpu")
            return torch.zeros(self.dim, device=dev, dtype=dtype)

        unit_mix = accum / torch.norm(accum, p=2)
        dev = device if device is not None else unit_mix.device
        return unit_mix.to(device=dev, dtype=dtype)

    def save_checkpoint(self, path: str):
        """Save bank vectors and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "dim": self.dim,
            "default_alpha": self.default_alpha,
            "vectors": {k: v.cpu() for k, v in self.vectors.items()},
            "layer_vectors": {k: v.cpu() for k, v in self.layer_vectors.items()},
            "supported_tones": self.SUPPORTED_TONES,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str, map_location="cpu"):
        """Load bank vectors from disk."""
        payload = torch.load(path, map_location=map_location)
        self.dim = payload.get("dim", self.dim)
        self.default_alpha = payload.get("default_alpha", self.default_alpha)
        for k, v in payload.get("vectors", {}).items():
            self.register_vector(k, v)
        for k, v in payload.get("layer_vectors", {}).items():
            # key format: layer_{idx}_{name}
            parts = k.split("_")
            if len(parts) >= 3 and parts[0] == "layer":
                l_idx = int(parts[1])
                t_name = "_".join(parts[2:])
                self.register_vector(t_name, v, layer_idx=l_idx)


# ---------------------------------------------------------------------------
# 2. Ingress & Hook Euclidean Norm-Preserving Steering
# ---------------------------------------------------------------------------
def apply_norm_preserving_steering(
    h: torch.Tensor,
    v_steer: torch.Tensor,
    alpha: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Applies activation addition with strict Euclidean norm preservation:
    h_shifted = h + alpha * (||h||_2 / sqrt(D)) * v_steer  (or relative fraction)
    h_steered = (||h||_2 / ||h_shifted||_2) * h_shifted

    Args:
        h: Activation tensor of shape (..., D).
        v_steer: Unit steering direction of shape (D,) or (..., D).
        alpha: Steering intensity gain (>= 0.0).
        eps: Small epsilon for numerical stability.

    Returns:
        Steered activation tensor with exactly the original Euclidean norm.
    """
    if alpha == 0.0 or v_steer is None or torch.norm(v_steer) <= eps:
        return h

    # Match dimensions for broadcasting
    if v_steer.dim() == 1:
        v_steer = v_steer.view(*([1] * (h.dim() - 1)), -1)

    v_steer = v_steer.to(device=h.device, dtype=h.dtype)
    orig_norm = torch.norm(h, p=2, dim=-1, keepdim=True)
    # Scale shift proportionally to total activation norm so alpha represents relative vector strength
    h_shifted = h + alpha * orig_norm * v_steer
    shifted_norm = torch.norm(h_shifted, p=2, dim=-1, keepdim=True) + eps

    # Scale back to original norm
    scale = orig_norm / shifted_norm
    return h_shifted * scale


# ---------------------------------------------------------------------------
# 3. Intermediate Layer Forward Hook for Talker Attention
# ---------------------------------------------------------------------------
class AffectiveSteeringHook:
    """
    PyTorch forward hook attached to intermediate Transformer layers (e.g. layers 12-16)
    of the Qwen3-TTS Talker model to steer attention activations during streaming generation.
    """

    def __init__(
        self,
        module: nn.Module,
        bank: AffectiveSteeringBank,
        layer_idx: int,
        tone: str = "neutral",
        alpha: float = 1.0,
    ):
        self.module = module
        self.bank = bank
        self.layer_idx = layer_idx
        self.tone = tone
        self.alpha = alpha
        self.enabled = True
        self.weights: Optional[Dict[str, float]] = None
        self.handle = module.register_forward_hook(self.hook_fn)

    def set_tone(self, tone: str, alpha: float = 1.0):
        self.tone = tone
        self.alpha = alpha
        self.weights = None
        self.enabled = (tone != "neutral" and alpha > 0.0)

    def set_mixed_tone(self, weights: Dict[str, float], alpha: float = 1.0):
        self.weights = weights
        self.alpha = alpha
        self.enabled = (alpha > 0.0 and len(weights) > 0)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def hook_fn(self, module, input_tensor, output_tensor):
        if not self.enabled:
            return output_tensor

        # Handle tuple output from attention layers (output, past_kv, etc.)
        is_tuple = isinstance(output_tensor, tuple)
        hidden = output_tensor[0] if is_tuple else output_tensor

        # Retrieve or compose steering vector
        if self.weights is not None:
            v_steer = self.bank.compose_mixed_vector(
                self.weights,
                layer_idx=self.layer_idx,
                device=hidden.device,
                dtype=hidden.dtype,
            )
        else:
            v_steer = self.bank.get_vector(self.tone, layer_idx=self.layer_idx)
            if v_steer is None:
                # Fallback to global bank vector
                v_steer = self.bank.get_vector(self.tone)

        if v_steer is None:
            return output_tensor

        steered_h = apply_norm_preserving_steering(
            hidden,
            v_steer.to(device=hidden.device, dtype=hidden.dtype),
            alpha=self.alpha,
        )

        if is_tuple:
            return (steered_h,) + output_tensor[1:]
        return steered_h


# ---------------------------------------------------------------------------
# 4. Gemma 4 Tone Extractor (Latent & Semantic Classification)
# ---------------------------------------------------------------------------
class GemmaToneExtractor(nn.Module):
    """
    Extracts affective tone embeddings and mixing weights from Gemma 4 latent states.
    Maps h_gemma (1536D) to tone classification logits or direct 2048D steering offset.
    """

    def __init__(self, in_dim: int = 1536, out_dim: int = 2048, num_tones: int = 8):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_tones = num_tones

        # 1. Tone classification head (predicts soft weights w_i across tones)
        self.tone_classifier = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_tones),
        )

        # 2. Continuous latent tone projection (h_gemma 1536 -> v_steer 2048)
        self.latent_tone_proj = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, out_dim),
        )

    def predict_tone_weights(self, h_gemma: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Predicts softmax probability distribution over supported emotional tones.
        h_gemma: Tensor of shape (..., 1536)
        Returns: Softmax weights (..., num_tones)
        """
        logits = self.tone_classifier(h_gemma)
        return F.softmax(logits / max(temperature, 1e-4), dim=-1)

    def extract_continuous_steering(self, h_gemma: torch.Tensor) -> torch.Tensor:
        """
        Extracts continuous dynamic steering vector in 2048D space from Gemma 4.
        Returns: Unit-normalized steering vector (..., 2048)
        """
        v_raw = self.latent_tone_proj(h_gemma)
        norm = torch.norm(v_raw, p=2, dim=-1, keepdim=True)
        # Avoid division by zero
        unit_v = torch.where(norm > 1e-8, v_raw / norm, torch.zeros_like(v_raw))
        return unit_v


# ---------------------------------------------------------------------------
# 5. Steered Talker Controller
# ---------------------------------------------------------------------------
class SteeredTalkerController:
    """
    Coordinates high-level affective conditioning during inference:
    - Ingress modulation of h_cond from TextAnchoredProjectionV7
    - Deep layer hooking on Talker intermediate attention blocks (e.g. layers 12-16)
    - Dynamic mixing and intensity scaling (alpha)
    """

    def __init__(
        self,
        bank: AffectiveSteeringBank,
        talker_model: Optional[nn.Module] = None,
        target_layers: Optional[List[int]] = None,
    ):
        self.bank = bank
        self.talker_model = talker_model
        self.target_layers = target_layers or [12, 14, 16]
        self.hooks: List[AffectiveSteeringHook] = []
        self.current_tone = "neutral"
        self.current_alpha = 1.0
        self.mixed_weights: Optional[Dict[str, float]] = None

        if self.talker_model is not None:
            self._attach_hooks()

    def _attach_hooks(self):
        """Attach hooks to the specified Transformer layers in Qwen3-TTS Talker."""
        if not hasattr(self.talker_model, "layers") and not hasattr(self.talker_model, "model"):
            return

        layers = self.talker_model.layers if hasattr(self.talker_model, "layers") else self.talker_model.model.layers
        for l_idx in self.target_layers:
            if l_idx < len(layers):
                target_mod = layers[l_idx].self_attn if hasattr(layers[l_idx], "self_attn") else layers[l_idx]
                hook = AffectiveSteeringHook(
                    module=target_mod,
                    bank=self.bank,
                    layer_idx=l_idx,
                    tone=self.current_tone,
                    alpha=self.current_alpha,
                )
                self.hooks.append(hook)

    def set_affective_tone(self, tone: str, alpha: float = 1.2):
        """Set a single categorical affective tone with intensity alpha."""
        self.current_tone = tone
        self.current_alpha = alpha
        self.mixed_weights = None
        for h in self.hooks:
            h.set_tone(tone, alpha=alpha)

    def set_mixed_affective_tone(self, weights: Dict[str, float], alpha: float = 1.2):
        """Set composable mixed affective tones (e.g. {'calm': 0.6, 'empathetic': 0.4})."""
        self.mixed_weights = weights
        self.current_alpha = alpha
        self.current_tone = "mixed"
        for h in self.hooks:
            h.set_mixed_tone(weights, alpha=alpha)

    def steer_ingress_condition(self, h_cond: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        """
        Modulate the ingress condition tensor h_cond (2048D) before entering the Talker loop.
        Applies norm-preserving steering.
        """
        use_alpha = alpha if alpha is not None else self.current_alpha
        if use_alpha <= 0.0 or self.current_tone == "neutral":
            return h_cond

        if self.mixed_weights is not None:
            v_steer = self.bank.compose_mixed_vector(
                self.mixed_weights,
                device=h_cond.device,
                dtype=h_cond.dtype,
            )
        else:
            v = self.bank.get_vector(self.current_tone)
            v_steer = v.to(device=h_cond.device, dtype=h_cond.dtype) if v is not None else None

        if v_steer is None:
            return h_cond

        return apply_norm_preserving_steering(h_cond, v_steer, alpha=use_alpha)

    def cleanup(self):
        """Remove all active hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
