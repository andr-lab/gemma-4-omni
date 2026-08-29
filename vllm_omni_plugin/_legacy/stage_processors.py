# -*- coding: utf-8 -*-
"""
vllm_omni_plugin/stage_processors.py — Processori di Input e Handoff Inter-Stadio.

Gestisce la trasformazione dei tensori latenti tra:
1. Stage 0 (Thinker) -> Stage 1 (Talker): Proiezione tramite Projector V9 (ADR-022 / Prefix KV-Cache).
2. Stage 1 (Talker) -> Stage 2 (Code2Wav): Routing dei chunk di indici RVQ (16 quantizzatori) al vocoder.

Riferimenti ufficiali SDD:
- vllm_omni.model_executor.stage_input_processors
- ADR-022 (Prefix KV-Cache Conditioning vs Iniezione Additiva)
- ADR-023 (ARIA - Adaptive Rate Interleaved Alignment)
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


class _LightweightProjectorV9(nn.Module):
    """Proiettore causale compatto per fallback o test isolati in assenza di checkpoint completo."""
    def __init__(self, in_dim: int = 1536, out_dim: int = 2048, n_prefix_frames: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_prefix_frames = n_prefix_frames
        self.proj = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, h_thinker: torch.Tensor) -> torch.Tensor:
        # h_thinker: (B, T, in_dim)
        B, T, _ = h_thinker.shape
        proj_h = self.norm(self.proj(h_thinker))
        if T >= self.n_prefix_frames:
            # Campionamento equidistribuito o slice finale dei frame di pensiero
            idx = torch.linspace(0, T - 1, self.n_prefix_frames, device=h_thinker.device).round().long()
            prefix = proj_h[:, idx, :]
        else:
            # Left-padding se la sequenza è inferiore a 16 token
            pad_len = self.n_prefix_frames - T
            pad = torch.zeros(B, pad_len, self.out_dim, dtype=proj_h.dtype, device=proj_h.device)
            prefix = torch.cat([pad, proj_h], dim=1)
        return prefix


# Dizionario di caching per il modulo proiettore parametrizzato per dimensioni
_PROJECTOR_V9_CACHE: Dict[Tuple[int, int, int], nn.Module] = {}


def get_or_create_projector_v9(in_dim: int = 1536, out_dim: int = 2048, n_prefix_frames: int = 16) -> nn.Module:
    key = (in_dim, out_dim, n_prefix_frames)
    if key not in _PROJECTOR_V9_CACHE:
        _PROJECTOR_V9_CACHE[key] = _LightweightProjectorV9(in_dim, out_dim, n_prefix_frames)
    return _PROJECTOR_V9_CACHE[key]


def thinker2talker_v9_prefix(
    stage0_output: Dict[str, Any],
    n_prefix_frames: int = 16,
    talker_dim: int = 2048,
    projector_module: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Handoff da Stage 0 (Thinker) a Stage 1 (Talker).

    Estrae gli stati nascosti emessi da Gemma 4 e calcola il blocco di condizionamento
    Prefix KV-Cache a 16 frame per il Talker Qwen3-TTS.
    """
    h_thinker = stage0_output.get("hidden_states")
    if h_thinker is None:
        raise ValueError("stage0_output deve contenere la chiave 'hidden_states'.")

    if not isinstance(h_thinker, torch.Tensor):
        h_thinker = torch.tensor(h_thinker, dtype=torch.bfloat16)

    device = h_thinker.device
    dtype = h_thinker.dtype
    in_dim = h_thinker.shape[-1]

    if projector_module is None:
        projector_module = get_or_create_projector_v9(in_dim, talker_dim, n_prefix_frames).to(device=device, dtype=dtype)

    with torch.no_grad():
        prefix_kv = projector_module(h_thinker)

    return {
        "prefix_kv": prefix_kv,
        "talker_prompt_embeds": prefix_kv,
        "text": stage0_output.get("text", []),
        "stage0_output_type": "prefix_kv_conditioned",
    }


def talker2code2wav_async_chunk(talker_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handoff asincrono streaming (Async Chunk) da Stage 1 (Talker) a Stage 2 (Code2Wav)."""
    codes = talker_payload.get("codes")
    if codes is None:
        raise ValueError("talker_payload deve contenere la chiave 'codes'.")

    return {
        "codes": codes,
        "is_last_chunk": bool(talker_payload.get("is_last_chunk", False)),
        "chunk_index": talker_payload.get("chunk_index", 0),
        "streaming_mode": "async_chunk",
    }


def talker2code2wav_full_payload(talker_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handoff sincrono completo da Stage 1 (Talker) a Stage 2 (Code2Wav)."""
    codes = talker_payload.get("codes")
    if codes is None:
        raise ValueError("talker_payload deve contenere la chiave 'codes'.")

    return {
        "codes": codes,
        "is_last_chunk": True,
        "streaming_mode": "full_payload",
    }
