# -*- coding: utf-8 -*-
"""
duplex_kernel/kv_cache.py — Gestione KV cache per il duplex continuo.

Focus critico (ADR-029 §KV): in full-duplex il contesto cresce senza limite.
Meccanismi implementati:

1. **Thinker — rolling context window**: contesto = [system | ultimi K turni |
   audio chunk corrente]. A fine turno evict del più vecchio. Il prefill del
   thinker resta limitato (non cresce con la durata della sessione).
   Sfrutta la sliding window nativa 4:1 di Gemma 4 E2B.

2. **Talker — prefix KV persistente + troncamento a fine turno**: il prefisso
   [ICL_text | h_pfx | BOS] è stabile per sessione → la sua KV si calcola una
   volta e si riusa. Durante il turno si appendono i codici RVQ; a fine turno
   la KV viene troncata ai token del prefisso (zero-reset VoiceChat-TTS), così
   il turno successivo riparte dal prefisso con costo ~0.

3. **KV quantizzata (opzionale)**: bf16 → FP8 sul thinker per raddoppiare la
   finestra utile a parità di VRAM.

Le strutture sono indipendenti dal framework: lavorano su past_key_values
PyTorch (tuple di tuple (K,V) per layer).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger("duplex_kernel.kv")

# Formato past_key_values HuggingFace: tuple[layer][2] = (K, V), shape
# (batch, n_kv_heads, seq_len, head_dim).


@dataclass
class TalkerPrefixKV:
    """KV persistente del prefisso stabile del talker.

    Calcolata una sola volta per sessione (primo turno): copre i token
    [ICL_text | h_pfx | BOS]. I turni successivi ripartono da qui.
    """

    kv: Any | None = None
    last_hidden: torch.Tensor | None = None
    seq_len: int = 0
    dtype: str = "bfloat16"
    device: str = "cuda"

    def is_ready(self) -> bool:
        return self.kv is not None

    def store(self, past_key_values: Any) -> None:
        """Salva la KV del prefisso (dopo il prefill di [ICL|h_pfx|BOS])."""
        self.kv = copy.deepcopy(past_key_values)
        self.seq_len = _kv_seq_len(past_key_values)
        logger.debug("[kv] prefix KV memorizzata (seq_len=%d)", self.seq_len)

    def clear(self) -> None:
        self.kv = None
        self.last_hidden = None
        self.seq_len = 0


@dataclass
class TurnKVState:
    """KV del talker per il turno corrente.

    Inizia come clone del prefix KV; durante il decode i frame RVQ vengono
    appesi. A fine turno si tronca ai token del prefisso (zero-reset).
    """

    kv: Any | None = None
    frames_generated: int = 0
    prefix_len: int = 0
    device: str = "cuda"

    def start_from_prefix(self, prefix: TalkerPrefixKV) -> None:
        """Avvia il turno dal Cache nativo del prefisso."""
        if prefix.is_ready():
            # Questo stato è diagnostico/ownership; il decode usa il past
            # locale del motore per evitare una copia GPU per ogni frame.
            self.kv = prefix.kv
            self.prefix_len = prefix.seq_len
        else:
            self.kv = None
            self.prefix_len = 0
        self.frames_generated = 0

    def append(self, past_key_values: Any) -> None:
        """Aggiorna la Cache nativa col passo corrente."""
        self.kv = past_key_values

    def truncate_to_prefix(self) -> None:
        """Tronca la KV ai token del prefisso (zero-reset su barge-in/fine turno)."""
        if self.kv is None or self.prefix_len <= 0:
            return
        self.kv = _slice_kv(self.kv, self.prefix_len)
        self.frames_generated = 0

    def clear(self) -> None:
        self.kv = None
        self.frames_generated = 0


@dataclass
class ThinkerContext:
    """Rolling context window del thinker.

    Mantiene gli ultimi K turni di testo (token ids) + il chunk audio corrente.
    Il prefill del thinker usa SOLO questo contesto: non cresce con la durata
    della sessione. La storia oltre K turni può essere riassunta (testo)
    e iniettata come system-followup.
    """

    system_prompt: str = ""
    turns: list[list[int]] = field(default_factory=list)  # token ids per turno
    message_turns: list[dict[str, Any]] = field(default_factory=list)
    max_turns: int = 8
    # Riassunto opzionale della storia oltre la finestra (testo compresso).
    summary: str = ""
    max_context_tokens: int = 8192

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Aggiorna la finestra conversazionale usata dal prompt runtime."""
        self.message_turns = [dict(m) for m in messages[-self.max_turns * 2:]]

    def messages(self) -> list[dict[str, Any]]:
        return [dict(m) for m in self.message_turns]

    def add_turn(self, token_ids: list[int]) -> None:
        """Aggiunge un turno completo (user o assistant) alla finestra."""
        if token_ids:
            self.turns.append(list(token_ids))
        # Evict del turno più vecchio oltre la finestra.
        while len(self.turns) > self.max_turns:
            dropped = self.turns.pop(0)
            logger.debug("[kv] thinker evict turno (%d token)", len(dropped))
        self._trim_to_budget()

    def _trim_to_budget(self) -> None:
        """Se il contesto supera il budget, evict dai turni più vecchi."""
        total = sum(len(t) for t in self.turns)
        if total <= self.max_context_tokens:
            return
        while self.turns and total > self.max_context_tokens:
            dropped = self.turns.pop(0)
            total -= len(dropped)
            logger.debug("[kv] thinker trim: -%d token (totale %d)", len(dropped), total)

    def flatten(self, tokenizer: Any) -> list[int]:
        """Costruisce la sequenza completa [system | summary | turns]."""
        ids: list[int] = []
        if self.system_prompt:
            ids.extend(tokenizer.encode(self.system_prompt, add_special_tokens=False))
        if self.summary:
            ids.extend(tokenizer.encode(self.summary, add_special_tokens=False))
        for turn in self.turns:
            ids.extend(turn)
        return ids

    def clear_turns(self) -> None:
        self.turns.clear()
        self.message_turns.clear()


# ---------------------------------------------------------------------------
# Utils su past_key_values
# ---------------------------------------------------------------------------
def _clone_kv(past_key_values: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Clona la KV (tuple di tuple (K,V)) senza condividere memoria."""
    if past_key_values is None:
        return ()
    if hasattr(past_key_values, "past_key_values"):  # wrapper
        past_key_values = past_key_values.past_key_values
    if isinstance(past_key_values, tuple) and past_key_values and isinstance(past_key_values[0], torch.Tensor):
        # formato piattizzato (2, n_layer, ...)
        return past_key_values
    out = []
    for layer in past_key_values:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            out.append((layer[0].detach().clone(), layer[1].detach().clone()))
    return tuple(out)


def _kv_seq_len(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            pass
    for layer in past_key_values:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2 and layer[0] is not None:
            return int(layer[0].shape[2])
    return 0


def _slice_kv(
    past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    keep_len: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Tronca cache legacy o Transformers 5 ai primi keep_len token."""
    if hasattr(past_key_values, "crop"):
        past_key_values.crop(keep_len)
        return past_key_values
    out = []
    for k, v in past_key_values:
        out.append((k[:, :, :keep_len, :].contiguous(), v[:, :, :keep_len, :].contiguous()))
    return tuple(out)


def kv_memory_bytes(past_key_values: Any) -> int:
    """Stima la memoria KV in byte (diagnostica)."""
    total = 0
    for layer in past_key_values:
        if isinstance(layer, (list, tuple)) and len(layer) >= 2:
            for t in layer[:2]:
                if t is not None:
                    total += t.numel() * t.element_size()
    return total


__all__ = [
    "TalkerPrefixKV",
    "TurnKVState",
    "ThinkerContext",
    "kv_memory_bytes",
]
